"""Memorisation test (CPU): can each arm reproduce TRAINING pictures when told exactly which picture to draw?

Every one of K training items gets its own id as the condition (no text encoder in the way). If the loss rewards the
right thing, training to a low loss should make sample(id i) look like picture i.  Same models/losses as strokegen.py.

  python memtest.py B out/mem/B.json --k 64 --steps 1500
  python memtest.py A out/mem/A.json --k 64 --steps 1500
  python memtest.py probe out/mem/probe.json          # no training: what does the loss count vs what the eye sees?
"""
import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

import strokegen as sg

ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["A", "B", "probe"])
ap.add_argument("out")
ap.add_argument("--k", type=int, default=64)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--d", type=int, default=384)
ap.add_argument("--layers", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(0)

S, keep, labels, ids, anchors = sg.load_data("out/mixed/v2", "v2")
if torch.cuda.is_available():  # everything created from here on lives on the GPU
    S, keep, anchors = S.cuda(), keep.cuda(), anchors.cuda()
    torch.set_default_device("cuda")
src = [i.split("_")[0] for i in ids]
norm = sg.Norm(sg.to_vec(S, keep, anchors), keep)  # same normalisation as the pilot (whole set)
Zall = norm.enc(sg.to_vec(S, keep, anchors), keep)
by = {}
for j, s_ in enumerate(src):
    by.setdefault(s_, []).append(j)
rng = random.Random(0)
pick = []
while len(pick) < a.k:  # round-robin over sources -> a mixed set
    for s_ in sorted(by):
        if by[s_] and len(pick) < a.k:
            pick.append(by[s_].pop(rng.randrange(len(by[s_]))))
pick = torch.tensor(pick)
Z, K = Zall[pick], keep[pick]
N, D = Z.shape[1], sg.D
tgt_img = sg.render_samples(sg.vec_to_strokes(norm.dec(Z), anchors))  # (k,3,HW) the fitted pictures
lev_of = torch.cat([torch.full((g * g,), li) for li, (g, *_) in enumerate([(4,), (7,), (10,)])])


def psnr(x, y):
    return (-10 * torch.log10(((x - y) ** 2).mean(tuple(range(1, x.dim()))).clamp_min(1e-10)))


def weights(kp):
    return torch.cat([torch.ones_like(kp[..., None]).float(), kp[..., None].float().expand(-1, -1, D - 1)], -1)


res = {"k": a.k, "sources": [src[i] for i in pick.tolist()]}
if a.arm == "probe":
    # 1) equal parameter-space damage, very different picture damage: perturb only one level / one channel group
    w = weights(K)
    groups = {"coarse 16 slots": lev_of == 0, "mid 49 slots": lev_of == 1, "fine 100 slots": lev_of == 2}
    chans = {"position (start,p1,p2)": list(range(1, 7)), "width": [7], "colour": [8, 9, 10]}
    out = []
    for sig in (0.3,):
        for gname, gm in groups.items():
            for cname, ch in chans.items():
                torch.manual_seed(1)
                nz = torch.randn_like(Z) * sig
                mask = torch.zeros_like(Z)
                mask[:, gm.nonzero()[:, 0][:, None], torch.tensor(ch)[None]] = 1
                Zp = Z + nz * mask
                loss = ((Zp - Z) ** 2 * w).mean().item()  # exactly the loss term these errors would cost
                im = sg.render_samples(sg.vec_to_strokes(norm.dec(Zp), anchors))
                out.append({"slots": gname, "params": cname, "param_loss": round(loss, 5),
                            "psnr_vs_target": round(psnr(im, tgt_img).mean().item(), 2)})
                print(out[-1], flush=True)
    res["equal_noise_different_damage"] = out
    # 2) what a "wrong but plausible" answer costs: another picture from the set, and the dataset-average strokes
    other = Z[torch.roll(torch.arange(a.k), 1)]
    res["loss_if_answer_is_a_different_picture"] = ((other - Z) ** 2 * w).mean().item()
    res["psnr_different_picture"] = psnr(sg.render_samples(sg.vec_to_strokes(norm.dec(other), anchors)), tgt_img).mean().item()
    mean_z = Zall.mean(0, keepdim=True).expand_as(Z)
    res["loss_if_answer_is_the_average"] = ((mean_z - Z) ** 2 * w).mean().item()
    res["psnr_average"] = psnr(sg.render_samples(sg.vec_to_strokes(norm.dec(mean_z), anchors)), tgt_img).mean().item()
    print({k: v for k, v in res.items() if k.startswith(("loss", "psnr"))}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    raise SystemExit

y_all = torch.arange(a.k)
model = sg.SetDiT(N, a.k, a.d, a.layers, 6) if a.arm == "B" else sg.ARFlow(N, a.k, a.d, a.layers, 6, True, 0.0)
ema = __import__("copy").deepcopy(model).eval().requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), 3e-4, betas=(0.9, 0.99), weight_decay=0.01)
Ag = anchors


def loss_fn(idx, t_fixed=None, per_slot=False):
    z, kp, y = Z[idx], K[idx], y_all[idx]
    w = weights(kp)
    if a.arm == "B":
        y = sg.drop_cond(y, 0.1, a.k) if t_fixed is None else y
        t = torch.rand(len(z)) if t_fixed is None else torch.full((len(z),), t_fixed)
        ab = sg.ab_fn(t)[:, None, None]
        eps = torch.randn_like(z)
        xt = ab.sqrt() * z + (1 - ab).sqrt() * eps
        tgt = ab.sqrt() * eps - (1 - ab).sqrt() * z
        e = (model(xt, t, y) - tgt) ** 2 * w
    else:
        with torch.no_grad():
            canv = sg.prefix_canvases(sg.vec_to_strokes(norm.dec(z), Ag), sg.CSIDE)
        prev = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], 1)
        y = sg.drop_cond(y, 0.1, a.k) if t_fixed is None else y
        cemb = model.canvas(canv.flatten(0, 1)).view(len(z), N, -1)
        h = model.trunk(prev, cemb, y)
        t = torch.rand(len(z), N, 2) if t_fixed is None else torch.full((len(z), N, 2), t_fixed)
        eps = torch.randn(len(z), N, 2, D)
        xt = (1 - t[..., None]) * eps + t[..., None] * z[:, :, None]
        e = ((model.head(h[:, :, None].expand(-1, -1, 2, -1), xt, t) - (z[:, :, None] - eps)) ** 2 * w[:, :, None]).mean(2)
    return e.mean((0, 2)) if per_slot else e.mean()


hist, t0 = [], time.time()
for step in range(1, a.steps + 1):
    lr = 3e-4 * min(1, step / 100)
    for g in opt.param_groups:
        g["lr"] = lr
    model.train()
    loss = loss_fn(torch.randint(0, a.k, (a.batch,)))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        dec = min(0.999, (1 + step) / (10 + step))
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - dec)
    if step % 100 == 0:
        hist.append({"step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)})
        print(json.dumps(hist[-1]), flush=True)
res["hist"] = hist
model.load_state_dict(ema.state_dict())
model.eval()

with torch.no_grad():
    # where does the remaining loss sit? (by noise level / flow time, and by slot level)
    torch.manual_seed(2)
    ts = [0.05, 0.25, 0.5, 0.75, 0.95]
    ev = torch.arange(min(a.k, 128))
    res["loss_by_t"] = {str(t): round(loss_fn(ev, t).item(), 4) for t in ts}
    ps = torch.stack([loss_fn(ev, None, True) for _ in range(4)]).mean(0)
    res["loss_by_level"] = {n: round(ps[lev_of == li].mean().item(), 4) for li, n in enumerate(["coarse", "mid", "fine"])}
    print("loss by t", res["loss_by_t"], "by level", res["loss_by_level"], flush=True)
    out = {}
    for cfg in ((1.0, 2.0) if a.arm == "B" else (1.0,)):
        torch.manual_seed(3)
        zs = torch.cat([sg.sample_B(model, y_all[j:j + 256], N, 50, cfg) if a.arm == "B" else sg.sample_A(model, y_all[j:j + 256], N, Ag, norm, 12, cfg)
                        for j in range(0, a.k, 256)])
        st = sg.vec_to_strokes(norm.dec(zs), Ag)
        im = sg.render_samples(st)
        p = psnr(im, tgt_img)
        small = lambda x: F.avg_pool2d(x.view(-1, 3, 128, 128), 4).flatten(1)
        dist = torch.cdist(small(im), small(tgt_img))
        ident = (dist.argmin(1) == y_all).float().mean().item()  # is sample i closest to picture i among all k?
        pz = ((zs - Z) ** 2 * weights(K)).mean().item()
        out[str(cfg)] = {"psnr_mean": round(p.mean().item(), 2), "identify_top1": ident, "chance": 1 / a.k,
                         "param_mse_vs_target": round(pz, 4), "slots_on": st[..., 10].sum(1).mean().item(),
                         "target_slots_on": K.float().sum(1).mean().item()}
        print(cfg, out[str(cfg)], flush=True)
        torch.save({"gen": im[:48].half().cpu(), "tgt": tgt_img[:48].half().cpu(), "src": res["sources"][:48]}, a.out.replace(".json", f"_cfg{cfg}.pt"))
    res["samples"] = out
json.dump(res, open(a.out, "w"), indent=1)
print("MEM_DONE", flush=True)
