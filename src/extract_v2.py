"""Extractor v2: anchored, ordered, pruned strokes (research/08). Runs on a GPU pod; CPU works for smoke tests.

Differences from extract_prod.py (v1):
  - slots, not random placement: level l has a g x g grid (4x4, 7x7, 10x10 = 165 slots); slot k starts at the
    error-weighted centroid of its own cell, coloured with the error-weighted target colour of that cell, with a fixed
    per-cell orientation. Same image -> same strokes; similar images -> similar slots.
  - order is the slot order (level, then row-major cell) during the fit itself, so it is render-exact.
  - p0/p2 swapped at the end so p0 is the top-left endpoint (identical render).
  - keep bit: a slot whose removal changes the final image by < --prune mean abs value is switched off (alpha 0).

  python extract_v2.py --out out/v2 --classes cat,house,sun,fish,tree,car --per-class 2500 --emoji-limit 1000
Shard: strokes (N,165,11) fp16 (alpha 0 = off), keep (N,165) bool, stage_psnr (N,3), psnr_pruned (N,), ids, labels, sources.
"""
import argparse
import concurrent.futures as cf
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

import batched
from extract_prod import CLEAN, SIZE, emoji_items, load_item, parse_shards, quickdraw_items, to_tensor

LEVELS = [(4, 0.30, 150), (7, 0.15, 150), (10, 0.07, 100)]  # (grid side, initial width, Adam steps)
SMOKE_LEVELS = [(2, 0.30, 3), (3, 0.15, 3), (4, 0.07, 3)]


def init_anchored(target, canvas, g, width, H):
    """One stroke per grid cell, row-major. Deterministic: no random numbers."""
    B, dev = target.shape[0], target.device
    pos = batched.grid(H, H, dev)  # (HW, 2) as (x, y)
    cell = (pos * g).floor().clamp(max=g - 1)
    cid = (cell[:, 1] * g + cell[:, 0]).long()  # row-major cell id per pixel
    w = (target - canvas).abs().sum(1) + 1e-6  # (B, HW) error weights
    idx = cid[None].expand(B, -1)
    den = torch.zeros(B, g * g, device=dev).scatter_add_(1, idx, w)
    cen = torch.zeros(B, g * g, 2, device=dev).scatter_add_(1, idx[..., None].expand(-1, -1, 2), w[..., None] * pos)
    col = torch.zeros(B, g * g, 3, device=dev).scatter_add_(
        1, idx[..., None].expand(-1, -1, 3), w[..., None] * target.transpose(1, 2))
    c = cen / den[..., None]
    rows, cols = torch.div(torch.arange(g * g, device=dev), g, rounding_mode="floor"), torch.arange(g * g, device=dev) % g
    ang = ((rows + cols) % 4).float() * (math.pi / 4)  # 0/45/90/135 degree pattern
    d = torch.stack([ang.cos(), ang.sin()], -1)[None] * width * 0.75
    s = torch.empty(B, g * g, batched.N_PARAMS, device=dev)
    s[..., 0:2], s[..., 2:4], s[..., 4:6] = c - d, c, c + d
    s[..., 6] = width
    s[..., 7:10] = col / den[..., None]
    s[..., 10] = 1.0
    return s


def decompose_v2(target, H, levels, mask_fn, lr=0.01):
    B, dev = target.shape[0], target.device
    frozen = torch.empty(B, 0, batched.N_PARAMS, device=dev)
    canvas = torch.ones(B, 3, H * H, device=dev)
    ps = []
    for g, width, steps in levels:
        s = init_anchored(target, canvas, g, width, H).requires_grad_(True)
        batched.clamp_(s, **{**batched.DEFAULT_CLAMP, **CLEAN})
        opt = torch.optim.Adam([s], lr=lr)
        for _ in range(steps):
            out = batched.render(s, H, H, base=canvas, mask_fn=mask_fn)
            diff = out - target
            loss = (diff.pow(2).mean(dim=(1, 2)) + 0.5 * diff.abs().mean(dim=(1, 2))).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            batched.clamp_(s, **{**batched.DEFAULT_CLAMP, **CLEAN})
        frozen = torch.cat([frozen, s.detach()], 1)
        with torch.no_grad():
            canvas = batched.render(s.detach(), H, H, base=canvas, mask_fn=mask_fn)
        ps.append(batched.psnr(canvas, target))
    return frozen, torch.stack(ps, 1)


def canon_direction(S):
    """Reversing a quadratic Bezier (p0 <-> p2) draws the identical curve: put the top-left endpoint first."""
    swap = (S[..., 0] * 1000 + S[..., 1]) > (S[..., 4] * 1000 + S[..., 5])
    a = S[..., 0:2].clone()
    S[..., 0:2] = torch.where(swap[..., None], S[..., 4:6], S[..., 0:2])
    S[..., 4:6] = torch.where(swap[..., None], a, S[..., 4:6])
    return S


@torch.no_grad()
def removal_effect(S, H, mask_fn):
    """Exact mean |change| of the final image if stroke i alone were removed: a_i * |c_i - canvas_before_i| * after_i."""
    B, N, _ = S.shape
    g = batched.grid(H, H, S.device)
    am = S[..., 10:11] * (mask_fn or batched.stroke_masks)(S.reshape(-1, batched.N_PARAMS), g, H).view(B, N, -1)
    rev = torch.cumprod((1 - am).flip(1), 1).flip(1)
    after = torch.cat([rev[:, 1:], torch.ones_like(rev[:, :1])], 1)
    canvas = torch.ones(B, 3, H * H, device=S.device)
    eff = []
    for i in range(N):
        a, c = am[:, i][:, None], S[:, i, 7:10, None]
        eff.append((a * (c - canvas)).abs().mul(after[:, i][:, None]).sum(1).mean(1))
        canvas = canvas * (1 - a) + c * a
    return torch.stack(eff, 1)  # (B, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="cat,house,sun,fish,tree,car")
    ap.add_argument("--classes-file", default=None, help="text file, one QuickDraw category per line (overrides --classes)")
    ap.add_argument("--per-class", type=int, default=2500)
    ap.add_argument("--qd-skip", type=int, default=0)
    ap.add_argument("--emoji-limit", type=int, default=1000)
    ap.add_argument("--shard-size", type=int, default=400)
    ap.add_argument("--shards", default="all")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--prune", type=float, default=1e-4)
    ap.add_argument("--manifest", default=None, help="jsonl rows {id,label|human_text,source,path|url}: use these items instead of QuickDraw/emoji")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items_file = out / "items.json"
    if items_file.exists():
        items = json.loads(items_file.read_text())
    else:
        cls_list = [l.strip() for l in open(a.classes_file) if l.strip()] if a.classes_file else [c for c in a.classes.split(",") if c]
        items = quickdraw_items(cls_list, a.per_class, a.qd_skip, out) if a.per_class else []
        items += emoji_items(a.emoji_limit, out) if a.emoji_limit else []
        if a.manifest:  # photo / art rows (url or local path); label = first human text
            items += [{**r, "label": r.get("label") or (r.get("texts") or [r.get("human_text", "")])[0]} for r in map(json.loads, open(a.manifest))]
        random.Random(0).shuffle(items)
        items_file.write_text(json.dumps(items))
    n_shards = (len(items) + a.shard_size - 1) // a.shard_size
    todo = parse_shards(a.shards, n_shards)
    print(f"{len(items)} items, {n_shards} shards; this run: {todo}", flush=True)
    if a.list_only:
        return

    levels = SMOKE_LEVELS if a.smoke else LEVELS
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def masks(s, g, H):
        return batched.stroke_masks(s, g, H, K=4)

    mask_fn = masks if a.no_compile else torch.compile(masks)
    pool = cf.ThreadPoolExecutor(16)
    for k in todo:
        path = out / f"shard_{k:05d}.pt"
        if path.exists():
            continue
        chunk = items[k * a.shard_size:(k + 1) * a.shard_size]
        t0 = time.perf_counter()
        def safe_load(it):
            try:
                return load_item(it)
            except Exception as e:
                print("load failed:", it["id"], type(e).__name__, str(e)[:80], flush=True)
                return None

        loaded = list(pool.map(safe_load, chunk))
        load_ok = torch.tensor([im is not None for im in loaded])
        tgt_all = torch.stack([to_tensor(im if im is not None else Image.new("RGB", (SIZE, SIZE), "white")) for im in loaded])
        t_load = time.perf_counter() - t0
        S, P, PP, KEEP, t_fit = [], [], [], [], 0.0
        for i in range(0, len(chunk), a.batch):
            tgt = tgt_all[i:i + a.batch]
            n = len(tgt)
            if n < a.batch:  # constant shapes for the compiled mask function
                tgt = torch.cat([tgt, tgt[:1].expand(a.batch - n, -1, -1)])
            tgt = tgt.to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            s, ps = decompose_v2(tgt, SIZE, levels, mask_fn)
            s = canon_direction(s)
            keep = removal_effect(s, SIZE, mask_fn) >= a.prune
            s[..., 10] = keep.float()  # alpha 0 = slot off
            with torch.no_grad():
                pp = batched.psnr(batched.render(s, SIZE, SIZE, mask_fn=mask_fn), tgt)
            if dev == "cuda":
                torch.cuda.synchronize()
            t_fit += time.perf_counter() - t1
            S.append(s[:n].cpu()); P.append(ps[:n].cpu()); PP.append(pp[:n].cpu()); KEEP.append(keep[:n].cpu())
        S, P, PP, KEEP = torch.cat(S), torch.cat(P), torch.cat(PP), torch.cat(KEEP)
        ok = torch.isfinite(S).flatten(1).all(1) & load_ok
        shard = {"strokes": canon_direction(S.half().float()).half(),  # re-canonicalize after fp16 rounding (ties)
                  "keep": KEEP, "stage_psnr": P, "psnr_pruned": PP, "ok": ok,
                 "ids": [c["id"] for c in chunk], "labels": [c["label"] for c in chunk], "sources": [c["source"] for c in chunk],
                 "size": SIZE, "levels": levels, "version": "v2"}
        torch.save(shard, str(path) + ".tmp")
        Path(str(path) + ".tmp").replace(path)
        rec = {"shard": k, "n": len(chunk), "fit_sec": round(t_fit, 1), "load_sec": round(t_load, 1),
               "imgs_per_hr_fit": int(3600 * len(chunk) / t_fit), "psnr": round(P[:, -1].mean().item(), 2),
               "psnr_pruned": round(PP.mean().item(), 2), "keep_frac": round(KEEP.float().mean().item(), 3),
               "bad": int((~ok).sum()), "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    print("DONE_ALL", flush=True)


if __name__ == "__main__":
    main()
