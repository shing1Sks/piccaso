"""Quality/speed sweeps for the batched decomposer on a stratified subset of the pilot images (GPU).

python sweep.py steps,segs,constraints,res,budget [--per-source 4]
Appends one JSON line per config to out/sweep/results.jsonl. Quality is scored twice:
  fit_psnr : at the fitting resolution (per stage end)
  ref_psnr : strokes re-rendered at 256 px vs the 256 px original (resolution-independent comparison)
"""
import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import torch

import batched
from extract import load_square

REF = 256
BASE_STAGES = [(16, 0.30), (48, 0.15), (96, 0.07), (96, 0.035)]  # 256 strokes
LONG_STAGES = BASE_STAGES + [(96, 0.02), (96, 0.012)]  # 448 strokes, prefix = shorter budgets

CLEAN = {"width": (0.008, 0.6), "pos": (0.0, 1.0), "alpha": (1.0, 1.0)}
CONFIGS = {
    "steps": {
        "steps150_lr.01": dict(steps=150, lr=0.01),
        "steps75_lr.02": dict(steps=75, lr=0.02),
        "steps50_lr.03": dict(steps=50, lr=0.03),
        "steps30_lr.05": dict(steps=30, lr=0.05),
        "steps20_lr.08": dict(steps=20, lr=0.08),
    },
    "segs": {
        "segs8": dict(segs=8),
        "segs6": dict(segs=6),
        "segs4": dict(segs=4),
        "segs3": dict(segs=3),
    },
    "constraints": {
        "baseline": dict(),
        "minw.008": dict(clamp={"width": (0.008, 0.6)}),
        "onCanvas": dict(clamp={"pos": (0.0, 1.0)}),
        "minw+onCanvas": dict(clamp={"width": (0.008, 0.6), "pos": (0.0, 1.0)}),
        "alpha1": dict(clamp={"alpha": (1.0, 1.0)}),
        "clean": dict(clamp={"width": (0.008, 0.6), "pos": (0.0, 1.0), "alpha": (1.0, 1.0)}),
    },
    "res": {f"fit{r}": dict(size=r) for r in (64, 96, 128, 192, 256)},
    "recipe": {
        "K4_matched": dict(segs=4),
        "K6_matched": dict(segs=6),
        "r160_K6_clean": dict(stages=BASE_STAGES[:3], segs=6, clamp=CLEAN),
        "r256_K6_clean": dict(segs=6, clamp=CLEAN),
        "r256_K6_clean_sched": dict(segs=6, clamp=CLEAN, steps=[150, 150, 100, 60]),
        "r256_K6_clean_refine256x20": dict(segs=6, clamp=CLEAN, refine=dict(size=256, steps=20, lr=0.003)),
        "r256_K6_clean_refine256x40": dict(segs=6, clamp=CLEAN, refine=dict(size=256, steps=40, lr=0.003)),
    },
    "final": {
        "final_fast": dict(stages=BASE_STAGES[:3], segs=4, clamp=CLEAN, steps=[150, 150, 100]),
        "final_quality": dict(segs=4, clamp=CLEAN, steps=[150, 150, 100, 60], refine=dict(size=256, steps=40, lr=0.003)),
    },
    "budget": {"long448@128": dict(stages=LONG_STAGES), "long448@192": dict(stages=LONG_STAGES, size=192)},
}


def pick_subset(manifest, per_source):
    seen, rows = {}, []
    for r in manifest:
        seen[r["source"]] = seen.get(r["source"], 0) + 1
        if seen[r["source"]] <= per_source:
            rows.append(r)
    return rows


@torch.no_grad()
def render_ref(strokes, side=REF, chunk=2, mask_fn=None):
    return torch.cat([batched.render(strokes[i:i + chunk], side, side, mask_fn=mask_fn)
                      for i in range(0, len(strokes), chunk)])


def usage(s):
    p = s.reshape(-1, 11)
    off = ((p[:, :6] < 0) | (p[:, :6] > 1)).any(1).float().mean().item()
    return {"width_lt_1px@128": round((p[:, 6] < 1 / 128).float().mean().item(), 3),
            "ctrl_off_canvas": round(off, 3), "alpha_eq_1": round((p[:, 10] >= 0.9999).float().mean().item(), 3)}


def run(name, group, cfg, rows, lpips_fn, out_path, batch):
    size = cfg.get("size", 128)
    stages = cfg.get("stages", BASE_STAGES)
    steps, lr = cfg.get("steps", 150), cfg.get("lr", 0.01)
    K = cfg.get("segs", 8)

    def masks(s, g, H):
        return batched.stroke_masks(s, g, H, K=K)

    mask_fn = torch.compile(masks)
    ref = torch.stack([load_square(r["path"], REF) for r in rows]).cuda()
    tgt = torch.stack([load_square(r["path"], size) for r in rows]).cuda()
    strokes, fit_ps, secs = [], [], 0.0
    torch.manual_seed(0)
    for i in range(0, len(rows), batch):
        t = tgt[i:i + batch]
        batched.decompose(t, size, size, stages, steps=2, mask_fn=mask_fn, clamp=cfg.get("clamp"))  # warm-up/compile
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s, _, ps = batched.decompose(t, size, size, stages, steps=steps, lr=lr, mask_fn=mask_fn, clamp=cfg.get("clamp"))
        torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        strokes.append(s)
        fit_ps.append(ps)
    S, fit_ps = torch.cat(strokes), torch.cat(fit_ps)
    if cfg.get("refine"):  # joint fine-tune of all strokes at a higher resolution, starting from the low-res fit
        rf = cfg["refine"]
        t0 = time.perf_counter()
        outs = []
        for i in range(0, len(S), 8):
            s = S[i:i + 8].clone().requires_grad_(True)
            opt = torch.optim.Adam([s], lr=rf["lr"])
            tg = ref[i:i + 8]
            for _ in range(rf["steps"]):
                out = batched.render(s, rf["size"], rf["size"], mask_fn=mask_fn)
                loss = ((out - tg).pow(2).mean(dim=(1, 2)) + 0.5 * (out - tg).abs().mean(dim=(1, 2))).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                batched.clamp_(s, **{**batched.DEFAULT_CLAMP, **(cfg.get("clamp") or {})})
            outs.append(s.detach())
        torch.cuda.synchronize()
        secs += time.perf_counter() - t0
        S = torch.cat(outs)
    cum = np.cumsum([n for n, _ in stages]).tolist()
    ref_ps = [batched.psnr(render_ref(S[:, :k], mask_fn=masks), ref).mean().item() for k in cum]
    final = render_ref(S, mask_fn=masks).reshape(-1, 3, REF, REF)
    row = {
        "group": group, "name": name, "fit_size": size, "n_images": len(rows), "strokes": cum[-1],
        "steps": steps, "lr": lr, "sec_per_img": round(secs / len(rows), 3), "imgs_per_hour": int(3600 * len(rows) / secs),
        "stage_ends": cum, "fit_psnr": [round(v, 2) for v in fit_ps.mean(0).tolist()],
        "ref256_psnr": [round(v, 2) for v in ref_ps],
        "ref256_lpips": round(lpips_fn(final * 2 - 1, ref.reshape(-1, 3, REF, REF) * 2 - 1).mean().item(), 4) if lpips_fn else None,
        **usage(S),
    }
    torch.save({"strokes": S.cpu(), "ids": [r["id"] for r in rows], "sources": [r["source"] for r in rows],
                "labels": [r["label"] for r in rows], "size": size, "stages": stages, "cfg": {k: str(v) for k, v in cfg.items()}},
               out_path.parent / f"strokes_{name}.pt")
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups")
    ap.add_argument("--per-source", type=int, default=4)
    ap.add_argument("--only", default=None, help="comma-separated config names")
    a = ap.parse_args()
    rows = pick_subset([json.loads(l) for l in open("pilot/manifest.jsonl")], a.per_source)
    out = Path("out/sweep")
    out.mkdir(parents=True, exist_ok=True)
    try:
        import lpips

        lp = lpips.LPIPS(net="alex", verbose=False).cuda()
        lpips_fn = lambda x, y: torch.cat([lp(x[i:i + 4], y[i:i + 4]).flatten() for i in range(0, len(x), 4)])
    except Exception as e:
        print("lpips unavailable:", e)
        lpips_fn = None
    print(f"device={torch.cuda.get_device_name(0)} images={len(rows)}", flush=True)
    for group in a.groups.split(","):
        for name, cfg in CONFIGS[group].items():
            if a.only and name not in a.only.split(","):
                continue
            batch = 8 if cfg.get("size", 128) >= 192 else len(rows)
            run(name, group, cfg, rows, lpips_fn, out / "results.jsonl", batch)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
