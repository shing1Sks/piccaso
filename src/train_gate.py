"""Learnability gate: can a tiny GPT learn to emit stroke sequences? (GPU)

Strokes (160 per image, 10 numbers each) are quantized into tokens: 6 positions (128 bins on [0,1]), width (16 log bins),
r/g/b (16 bins each); alpha is always 1. The model is conditioned on one class token (QuickDraw class, or 'emoji').
Direct approach: 1600 tokens per image, field-masked softmax, horizontal-flip augmentation (exact on stroke params).

  python train_gate.py --data out/s0 --out out/gate --minutes 12
Writes out/gate/{metrics.json, samples.png, ckpt.pt}.
"""
import argparse
import glob
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

NPOS, NW, NC = 128, 16, 16
W_LO, W_HI = 0.008, 0.6
V = NPOS + NW + NC  # token vocabulary; class tokens are V + class_index
FIELD_RANGE = [(0, NPOS)] * 6 + [(NPOS, NPOS + NW)] + [(NPOS + NW, V)] * 3


def quantize(S):
    """S (N,K,11) float -> tokens (N,K*10) long."""
    pos = (S[..., 0:6].clamp(0, 1) * (NPOS - 1)).round()
    u = ((S[..., 6].clamp(W_LO, W_HI).log() - math.log(W_LO)) / (math.log(W_HI) - math.log(W_LO)))
    w = (u * (NW - 1)).round() + NPOS
    col = (S[..., 7:10].clamp(0, 1) * (NC - 1)).round() + NPOS + NW
    return torch.cat([pos, w[..., None], col], -1).long().flatten(1)


def dequantize(tok):
    """tokens (N,K*10) -> strokes (N,K,11) float, alpha = 1."""
    t = tok.view(tok.shape[0], -1, 10).float()
    pos = t[..., 0:6] / (NPOS - 1)
    w = torch.exp(math.log(W_LO) + (t[..., 6] - NPOS) / (NW - 1) * (math.log(W_HI) - math.log(W_LO)))
    col = (t[..., 7:10] - NPOS - NW) / (NC - 1)
    return torch.cat([pos, w[..., None], col, torch.ones_like(w[..., None])], -1)


STAGE_SPLITS = [(0, 16), (16, 64), (64, 160)]  # fast-recipe stage boundaries (prefix property)


def canon(S, sort=True, band=8):
    """Reversing a quadratic Bezier (p0<->p2) draws the identical curve, so put the top-left endpoint first (free).
    sort=True also orders the strokes inside each stage by (row band of start point, x): NOT render-exact, diagnostic."""
    S = S.clone()
    swap = (S[..., 0] * 1000 + S[..., 1]) > (S[..., 4] * 1000 + S[..., 5])
    a = S[..., 0:2].clone()
    S[..., 0:2] = torch.where(swap[..., None], S[..., 4:6], S[..., 0:2])
    S[..., 4:6] = torch.where(swap[..., None], a, S[..., 4:6])
    if not sort:
        return S
    out = []
    for lo, hi in STAGE_SPLITS:
        if lo >= S.shape[1]:
            break
        blk = S[:, lo:hi]
        idx = ((blk[..., 1] * band).floor() * 10 + blk[..., 0]).argsort(1)
        out.append(torch.gather(blk, 1, idx[..., None].expand(-1, -1, 11)))
    return torch.cat(out, 1)


def flip(S):
    S = S.clone()
    S[..., [0, 2, 4]] = 1 - S[..., [0, 2, 4]]
    return S


class Block(nn.Module):
    def __init__(s, d, H, p=0.0):
        super().__init__()
        s.p = p
        s.H, s.ln1, s.ln2 = H, nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.H, D // s.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + F.dropout(s.proj(a.transpose(1, 2).reshape(B, T, D)), s.p, s.training)
        return x + F.dropout(s.mlp(s.ln2(x)), s.p, s.training)


class GPT(nn.Module):
    def __init__(s, n_cond, K=160, d=256, L=6, H=8, p=0.0):
        super().__init__()
        s.tok = nn.Embedding(V + n_cond, d)
        s.stroke, s.field = nn.Embedding(K, d), nn.Embedding(10, d)  # structured positions: which stroke, which field
        s.blocks = nn.ModuleList(Block(d, H, p) for _ in range(L))
        s.lnf, s.head = nn.LayerNorm(d), nn.Linear(d, V)
        mask = torch.full((10, V), -1e4)
        for f, (a, b) in enumerate(FIELD_RANGE):
            mask[f, a:b] = 0
        s.register_buffer("mask", mask)

    def forward(s, x):
        T = x.shape[1]
        t = torch.arange(T, device=x.device)  # embeddings describe the position being PREDICTED
        h = s.tok(x) + s.stroke(t // 10) + s.field(t % 10)
        for b in s.blocks:
            h = b(h)
        return s.head(s.lnf(h)).float() + s.mask[t % 10]


def load_shards(path):
    S, labels, ids, P = [], [], [], []
    for f in sorted(glob.glob(f"{path}/shard_*.pt")):
        sh = torch.load(f)
        ok = sh["ok"]
        S.append(sh["strokes"][ok].float())
        P.append(sh["stage_psnr"][ok])
        labels += [l for l, o in zip(sh["labels"], ok) if o]
        ids += [i for i, o in zip(sh["ids"], ok) if o]
    labels = ["emoji" if l.startswith("emoji") else l for l in labels]
    return torch.cat(S), torch.cat(P), labels, ids


@torch.no_grad()
def render(S, side, chunk=2):
    return torch.cat([batched.render(S[i:i + chunk], side, side) for i in range(0, len(S), chunk)])


def to_pil(x, side):
    a = x.reshape(3, side, side).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((a * 255).astype(np.uint8))


@torch.no_grad()
def sample(model, conds, T, temp, topk):
    x = (conds + V)[:, None]
    for _ in range(T):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x)[:, -1] / temp
        if topk:
            lg[lg < lg.topk(topk, -1).values[:, -1:]] = -1e4
        x = torch.cat([x, torch.multinomial(lg.softmax(-1), 1)], 1)
    return x[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--n-per-class", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--strokes", type=int, default=160)
    ap.add_argument("--canon", choices=["none", "dir", "sort"], default="none")
    ap.add_argument("--dropout", type=float, default=0.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    S, P, labels, ids = load_shards(a.data)
    if a.canon != "none":
        S = canon(S, sort=a.canon == "sort")
    S = S[:, :a.strokes]  # prefix = coarse-to-fine, so fewer strokes is a valid shorter sequence
    classes = sorted(set(labels))
    cid = torch.tensor([classes.index(l) for l in labels])
    N, K = len(S), S.shape[1]
    T = K * 10
    perm = torch.randperm(N)
    n_val = min(max(64, N // 20), N // 4)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    tok = torch.stack([quantize(S), quantize(flip(S))], 1).to(dev)  # (N,2,T)
    cid = cid.to(dev)
    print(f"{N} sequences ({len(tr_i)} train / {n_val} val), {len(classes)} conditions, T={T}", flush=True)

    # quantization cost: render the continuous strokes vs their dequantized versions
    q_psnr = batched.psnr(render(S[val_i[:32]].to(dev), 128), render(dequantize(tok[val_i[:32], 0]), 128)).mean().item()
    print(f"quantization PSNR (continuous vs dequantized strokes, 128 px): {q_psnr:.2f} dB", flush=True)

    model = GPT(len(classes), K, a.d, a.layers, 8, a.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params {n_params / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)

    def batch_xy(idx, aug):
        f = torch.randint(0, 2, (len(idx),), device=dev) if aug else torch.zeros(len(idx), dtype=torch.long, device=dev)
        y = tok[idx, f]
        return torch.cat([(cid[idx] + V)[:, None], y[:, :-1]], 1), y

    def val_loss(shuffle_cond=False):
        model.eval()
        tot, per = 0.0, torch.zeros(10)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, n_val, 32):
                idx = val_i[i:i + 32].to(dev)
                x, y = batch_xy(idx, False)
                if shuffle_cond:
                    x = x.clone()
                    x[:, 0] = x[:, 0][torch.randperm(len(idx), device=dev)]
                ce = F.cross_entropy(model(x).transpose(1, 2), y, reduction="none")  # (B,T)
                tot += ce.mean().item() * len(idx)
                per += ce.view(len(idx), -1, 10).mean((0, 1)).cpu() * len(idx)
        model.train()
        return tot / n_val, (per / n_val).tolist()

    t0, step, total = time.perf_counter(), 0, a.max_steps
    hist = []
    model.train()
    while True:
        lr = a.lr * min(1, (step + 1) / 100)
        if total:
            lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, step / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = tr_i[torch.randint(0, len(tr_i), (a.batch,))].to(dev)
        x, y = batch_xy(idx, True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).transpose(1, 2), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step == 30 and total is None:  # now we know the step time: fix the schedule to the time budget
            torch.cuda.synchronize()
            total = int(a.minutes * 60 / ((time.perf_counter() - t0) / 30))
            print(f"~{(time.perf_counter() - t0) / 30:.3f}s/step -> {total} steps", flush=True)
        if step % 100 == 0 or (total and step >= total):
            vl, _ = val_loss()
            hist.append({"step": step, "train": round(loss.item(), 4), "val": round(vl, 4), "min": round((time.perf_counter() - t0) / 60, 1)})
            print(json.dumps(hist[-1]), flush=True)
        if total and step >= total:
            break

    vl, per_field = val_loss()
    vl_shuf, _ = val_loss(shuffle_cond=True)
    print(f"val nll/token {vl:.4f}; with shuffled class {vl_shuf:.4f} (gap x{T} = {(vl_shuf - vl) * T:.1f} nats/sequence)", flush=True)
    torch.save({"model": model.state_dict(), "classes": classes}, out / "ckpt.pt")

    # samples: one row per condition: [fitted training example | n generated samples]
    model.eval()
    t_s = time.perf_counter()
    show = [c for c in classes if c != "emoji"][:16] + (["emoji"] if "emoji" in classes else [])
    conds = torch.tensor([classes.index(c) for c in show for _ in range(a.n_per_class)], device=dev)
    gen = sample(model, conds, T, a.temp, a.topk)
    print(f"sampling took {time.perf_counter() - t_s:.0f}s", flush=True)
    gen_strokes = dequantize(gen)
    imgs = render(gen_strokes, 160)
    ref = []
    for c in show:
        j = next(i for i in range(N) if labels[i] == c)
        ref.append(S[j])
    ref_imgs = render(torch.stack(ref).to(dev), 160)
    Tl = 160
    sheet = Image.new("RGB", (Tl * (1 + a.n_per_class), Tl * len(show)), "white")
    d = ImageDraw.Draw(sheet)
    for r, c in enumerate(show):
        sheet.paste(to_pil(ref_imgs[r], Tl), (0, r * Tl))
        d.text((3, r * Tl + 2), f"{c} (train fit)", fill=(255, 0, 0))
        for k in range(a.n_per_class):
            sheet.paste(to_pil(imgs[r * a.n_per_class + k], Tl), (Tl * (1 + k), r * Tl))
    sheet.save(out / "samples.png")
    torch.save({"conds": show, "tokens": gen.cpu()}, out / "samples.pt")
    metrics = {"n_seq": N, "classes": classes, "params_M": round(n_params / 1e6, 2), "steps": step, "val_nll": vl,
               "val_nll_shuffled_class": vl_shuf, "quant_psnr_db": q_psnr, "per_field_val_nll": per_field, "hist": hist,
               "train_psnr_by_stage_mean": P.mean(0).tolist()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
