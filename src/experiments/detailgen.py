"""Stage 3: detail model. Given the 165 base strokes (+ the prompt), generate the 196 detail strokes (extract_v3 data).

One bidirectional diffusion transformer over all 361 slots (SetDiT from strokegen): base slots are fed CLEAN and marked
"given", detail slots are noised; the loss is only on detail slots. At sampling time the base slots are clamped to the
given strokes every step (completion, like Collaborative Neural Painting). Works on top of either arm: any base works.

  python detailgen.py --path out/final --text-json out/final/caps.json --out out/detail --minutes 90
Eval: on held-out items, generated detail vs no detail vs fitted detail, all on the TRUE base, PSNR against the real
image at 256 px (cached jpg); sheet: real | base | base+generated detail | base+fitted detail.  --chain <B ckpt> also
renders prompt -> B base -> + detail.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import batched
import detail_level as dl
import strokegen as sg
from extract_v3 import safe_name

D = sg.D


def to_vec(S, keep, anchors):  # like strokegen.to_vec but widths down to 0.004 (detail strokes are thinner)
    v = sg.to_vec(S, keep, anchors).to(S.device)
    v[..., 7] = S[..., 6].clamp(0.004, 0.6).log()
    return v


def vec_to_strokes(v, anchors):
    s = sg.vec_to_strokes(v, anchors)
    s[..., 6] = v[..., 7].exp().clamp(0.004, 0.6)
    return s


class GroupNorm:
    """Per-channel stats computed separately for base and detail slots (detail strokes are smaller)."""

    def __init__(s, v, keep, nb):
        m, s.nb = keep[..., None].float(), nb
        stats = []
        for sl in (slice(0, nb), slice(nb, None)):
            vv, mm = v[:, sl, 1:], m[:, sl]
            mu = (vv * mm).sum((0, 1)) / mm.sum()
            sd = (((vv - mu) ** 2 * mm).sum((0, 1)) / mm.sum()).sqrt().clamp_min(1e-3)
            stats.append((mu, sd))
        N = v.shape[1]
        s.mean = torch.stack([stats[0][0] if i < nb else stats[1][0] for i in range(N)])
        s.std = torch.stack([stats[0][1] if i < nb else stats[1][1] for i in range(N)])

    def enc(s, v, keep):
        o = v.clone()
        o[..., 1:] = (v[..., 1:] - s.mean.to(v.device)) / s.std.to(v.device) * keep[..., None].float()
        return o

    def dec(s, z):
        o = z.clone()
        o[..., 1:] = z[..., 1:] * s.std.to(z.device) + s.mean.to(z.device)
        return o


class DetailDiT(sg.SetDiT):
    def __init__(s, N, nb, d, L, H, text_dim):
        super().__init__(N, 1, d, L, H, text_dim)
        s.nb = nb
        s.given = nn.Parameter(torch.zeros(d))

    def forward(s, x, t, y):
        c = s.tm(sg.temb(t)) + s.cls(y)
        h = s.inp(x) + s.slot
        h = torch.cat([h[:, :s.nb] + s.given, h[:, s.nb:]], 1)
        for b in s.blocks:
            h = b(h, c)
        sh, sc = s.fada(F.silu(c))[:, None].chunk(2, -1)
        return s.out(s.fn(h) * (1 + sc) + sh)


@torch.no_grad()
def sample_detail(model, base_z, y, N, nb, steps=40, w=1.5):
    """base_z (B, nb, D) normalised base slots -> full (B, N, D) with generated detail slots."""
    B, dev = len(y), y.device
    null = model.cls.null_like(y)
    x = torch.randn(B, N, D, device=dev)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    for i in range(steps):
        x[:, :nb] = base_z  # base is given, never noised
        t = ts[i].expand(B)
        ab, abn = sg.ab_fn(ts[i]), sg.ab_fn(ts[i + 1])
        v = model(x, t, null) + w * (model(x, t, y) - model(x, t, null)) if w != 1 else model(x, t, y)
        a, s = ab.sqrt(), (1 - ab).sqrt()
        x0 = (a * x - s * v).clamp(-6, 6)
        eps = s * x + a * v
        x = abn.sqrt() * x0 + (1 - abn).sqrt() * eps
    x0[:, :nb] = base_z
    return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--text-json", required=True)
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=90)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--chain", default=None, help="arm-B ckpt: also sheet prompt -> B base -> + detail")
    ap.add_argument("--prompts-file", default=None)
    ap.add_argument("--eval-only", action="store_true", help="load <out>/ckpt.pt and only run the evaluation / chain sheets")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    S, keep, labels, ids, anchors = sg.load_data(a.path, "v2", slots=10 ** 6)  # all 361 slots
    caps = json.load(open(a.text_json))
    sel = [j for j, i in enumerate(ids) if i in caps]
    S, keep, ids = S[sel], keep[sel], [ids[j] for j in sel]
    N, nb = S.shape[1], 165
    v = to_vec(S, keep, anchors)
    norm = GroupNorm(v, keep, nb)
    Z = norm.enc(v, keep)
    perm = torch.randperm(len(Z))
    nv = max(min(64, len(Z) // 5), len(Z) // 50)
    vi, ti = perm[:nv], perm[nv:]
    print(f"{len(Z)} items x {N} slots ({nb} base); detail kept {keep[:, nb:].float().mean():.2f}", flush=True)

    uniq = sorted({c for i in ids for c in caps[i]})
    if a.text_cache and Path(a.text_cache).exists() and torch.load(a.text_cache)["strings"] == uniq:
        Tall = torch.load(a.text_cache)["emb"]
    else:
        Tall = sg.embed_text(uniq, dev)
        if a.text_cache:
            torch.save({"strings": uniq, "emb": Tall}, a.text_cache)
    mu, sdv = Tall.mean(0), Tall.std(0) + 1e-6
    Tall = ((Tall - mu) / sdv).to(dev)
    pos = {c: k for k, c in enumerate(uniq)}
    maxk = max(len(caps[i]) for i in ids)
    cap_idx = torch.full((len(ids), maxk), -1, dtype=torch.long)
    for r, i in enumerate(ids):
        cap_idx[r, :len(caps[i])] = torch.tensor([pos[c] for c in caps[i]])
    cap_n = (cap_idx >= 0).sum(1).to(dev)
    cap_idx = cap_idx.to(dev)
    Zg, Kg, Ag = Z.to(dev), keep.to(dev), anchors.to(dev)

    def cond_for(idx):
        j = (torch.rand(len(idx), device=dev) * cap_n[idx]).long().clamp_max(cap_idx.shape[1] - 1)
        return Tall[cap_idx[idx, j]]

    model = DetailDiT(N, nb, a.d, a.layers, a.heads, Tall.shape[1]).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"detail model {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.99), weight_decay=0.01)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
    dmask = torch.zeros(N, device=dev)
    dmask[nb:] = 1

    def loss_fn(idx, train=True):
        z, kp, y = Zg[idx], Kg[idx], cond_for(idx)
        if train:
            y = sg.drop_cond(y, 0.1, 1)
        t = torch.rand(len(z), device=dev)
        ab = sg.ab_fn(t)[:, None, None]
        eps = torch.randn_like(z)
        xt = ab.sqrt() * z + (1 - ab).sqrt() * eps
        xt[:, :nb] = z[:, :nb]  # base given clean
        tgt = ab.sqrt() * eps - (1 - ab).sqrt() * z
        w = torch.cat([torch.ones_like(z[..., :1]), kp[..., None].float().expand(-1, -1, D - 1)], -1) * dmask[None, :, None]
        with amp:
            pred = model(xt, t, y).float()
        return ((pred - tgt) ** 2 * w).sum() / w.sum()

    t0, step, total, best, best_state, bad = time.perf_counter(), 0, a.max_steps, 1e9, None, 0
    hist = []
    if a.eval_only:
        ema.load_state_dict(torch.load(out / "ckpt.pt", map_location=dev)["ema"])
        best_state = ema.state_dict()
    while not a.eval_only:
        lr = a.lr * min(1, (step + 1) / 200)
        if total:
            lr *= 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, step / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        loss = loss_fn(ti[torch.randint(0, len(ti), (a.batch,))].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            dec = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dec)
        step += 1
        if step == 50 and total is None:
            total = int(a.minutes * 60 / ((time.perf_counter() - t0) / 50))
            print(f"-> {total} steps", flush=True)
        if step % 250 == 0 or (total and step >= total):
            model.eval()
            with torch.no_grad():
                torch.manual_seed(1)
                vl = float(np.mean([loss_fn(vi[j:j + a.batch].to(dev), False).item() for j in range(0, len(vi), a.batch)]))
            torch.manual_seed(step)
            hist.append({"step": step, "train": round(loss.item(), 4), "val": round(vl, 4), "min": round((time.perf_counter() - t0) / 60, 1)})
            print(json.dumps(hist[-1]), flush=True)
            if vl < best:
                best, bad, best_state = vl, 0, {k: x.detach().clone() for k, x in ema.state_dict().items()}
            else:
                bad += 1
            if a.patience and bad >= a.patience:
                print("early stop", flush=True)
                break
        if total and step >= total:
            break
    ema.load_state_dict(best_state)
    if not a.eval_only:
        torch.save({"ema": ema.state_dict(), "N": N, "nb": nb, "norm": (norm.mean, norm.std), "text_norm": (mu, sdv), "anchors": anchors,
                    "args": vars(a)}, out / "ckpt.pt")

    # ---- eval on held-out items: true base, PSNR vs the real picture at 256 px
    H, g = 256, 14
    ev = vi[:a.n_eval].tolist()
    load_real = lambda i: torch.from_numpy(np.asarray(Image.open(Path(a.path) / "img" / f"{safe_name(ids[i])}.jpg").convert("RGB")
                                                   .resize((H, H), Image.LANCZOS)).copy()).permute(2, 0, 1).reshape(3, -1).float() / 255
    real = torch.stack([load_real(i) for i in ev]).to(dev)
    y = cond_for(torch.tensor(ev, device=dev))
    zg = sample_detail(ema, Zg[ev, :nb], y, N, nb)
    st_gen = vec_to_strokes(norm.dec(zg), Ag)
    k_counts = Kg[ti[:2000], nb:].sum(1)
    kv = norm.dec(zg)[:, nb:, 0]
    for r in range(len(ev)):  # number of detail strokes ON from the data distribution, most confident slots win
        k = int(k_counts[torch.randint(len(k_counts), (1,))])
        on = torch.zeros(N - nb, device=dev)
        on[kv[r].topk(k).indices] = 1
        st_gen[r, nb:, 10] = on
    st_fit = vec_to_strokes(norm.dec(Zg[ev]), Ag)
    with torch.no_grad():
        base_hi = sg.render_samples(st_fit[:, :nb], H)
        gen_hi = torch.cat([dl.render_detail(st_gen[j:j + 8, nb:], base_hi[j:j + 8], g, H) for j in range(0, len(ev), 8)])
        fit_hi = torch.cat([dl.render_detail(st_fit[j:j + 8, nb:], base_hi[j:j + 8], g, H) for j in range(0, len(ev), 8)])
    ps = lambda x: batched.psnr(x, real).mean().item()
    met = {"n": len(ev), "psnr_base_only": ps(base_hi), "psnr_base_plus_generated_detail": ps(gen_hi),
           "psnr_base_plus_fitted_detail (ceiling)": ps(fit_hi), "best_val": best, "hist": hist}
    print(json.dumps({k: v for k, v in met.items() if k != "hist"}), flush=True)
    T = H
    tile = lambda x: Image.fromarray((x.view(3, T, T).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    n_show = min(10, len(ev))
    sheet = Image.new("RGB", (T * 4, T * n_show + 16), "white")
    dr = ImageDraw.Draw(sheet)
    for c, hd in enumerate(["real", "base (165)", "base + GENERATED detail", "base + fitted detail"]):
        dr.text((c * T + 4, 2), hd, fill=(0, 0, 0))
    for r in range(n_show):
        for c, im in enumerate([real[r], base_hi[r], gen_hi[r], fit_hi[r]]):
            sheet.paste(tile(im), (c * T, 16 + r * T))
    sheet.save(out / "detail_eval.png")

    if a.chain and a.prompts_file:  # prompt -> arm-B base -> + generated detail
        ck = torch.load(a.chain, map_location=dev)
        bn = sg.Norm.__new__(sg.Norm)
        bn.mean, bn.std = ck["norm"]
        bm = sg.SetDiT(ck["N"], len(ck["classes"]), ck["args"]["d"], ck["args"]["layers"], ck["args"]["heads"], ck["text_dim"]).to(dev)
        bm.load_state_dict(ck["ema"])
        bm.eval()
        prompts = [l.strip() for l in open(a.prompts_file) if l.strip()]
        pe = sg.embed_text(prompts, dev).to(dev)
        tmu, tsd = ck["text_norm"]
        per = 3
        yb = ((pe - tmu.to(dev)) / tsd.to(dev)).repeat_interleave(per, 0)
        zb = sg.sample_B(bm, yb, ck["N"], 50, 2.0)
        sb = sg.vec_to_strokes(bn.dec(zb), Ag[:nb])
        kb = Kg[ti[:2000], :nb].sum(1)
        kvb = bn.dec(zb)[..., 0]
        for r in range(len(zb)):
            k = int(kb[torch.randint(len(kb), (1,))])
            on = torch.zeros(nb, device=dev)
            on[kvb[r].topk(k).indices] = 1
            sb[r, :, 10] = on
        vb = to_vec(sb, sb[..., 10] > 0, Ag[:nb])  # B's base strokes, re-encoded with the detail model's normaliser
        zbase = norm.enc(torch.cat([vb, torch.zeros(len(vb), N - nb, D, device=dev)], 1), torch.cat([sb[..., 10] > 0, torch.zeros(len(vb), N - nb, dtype=torch.bool, device=dev)], 1))[:, :nb]
        yd = ((pe - mu.to(dev)) / sdv.to(dev)).repeat_interleave(per, 0)
        zd = sample_detail(ema, zbase, yd, N, nb)
        sd_ = vec_to_strokes(norm.dec(zd), Ag)
        kvd = norm.dec(zd)[:, nb:, 0]
        for r in range(len(zd)):
            k = int(k_counts[torch.randint(len(k_counts), (1,))])
            on = torch.zeros(N - nb, device=dev)
            on[kvd[r].topk(k).indices] = 1
            sd_[r, nb:, 10] = on
        with torch.no_grad():
            bh = sg.render_samples(sb, H)
            fh = torch.cat([dl.render_detail(sd_[j:j + 8, nb:], bh[j:j + 8], g, H) for j in range(0, len(zd), 8)])
        sheet = Image.new("RGB", (T * per * 2, T * len(prompts)), "white")
        dr = ImageDraw.Draw(sheet)
        for r, p in enumerate(prompts):
            for q in range(per):
                sheet.paste(tile(bh[r * per + q]), (2 * q * T, r * T))
                sheet.paste(tile(fh[r * per + q]), ((2 * q + 1) * T, r * T))
            dr.text((3, r * T + 2), p[:60] + "   (pairs: base | + detail)", fill=(255, 0, 0))
        sheet.save(out / "chain_prompts.png")
        import open_clip  # does the detail make generated pictures match their prompt better?

        cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        cm = cm.to(dev).eval()
        MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
        STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)
        with torch.no_grad():
            enc = lambda x: F.normalize(cm.encode_image((F.interpolate(x.view(-1, 3, H, H), size=224, mode="bilinear") - MEAN) / STD).float(), dim=-1)
            te = pe.repeat_interleave(per, 0)
            met["chain_clip_prompt_match"] = {"base": (enc(bh) * te).sum(1).mean().item(), "base_plus_detail": (enc(fh) * te).sum(1).mean().item()}
        print("chain CLIP prompt match", met["chain_clip_prompt_match"], flush=True)
    (out / "metrics.json").write_text(json.dumps(met, indent=1))
    print("DETAIL_DONE", flush=True)


if __name__ == "__main__":
    main()
