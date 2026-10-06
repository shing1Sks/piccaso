"""Extractor v3 = v2 base (165 anchored slots fitted at 128 px, unchanged) + a detail level (14x14 = 196 small strokes
fitted at 256 px on top, detail_level.py) + a cached copy of every source image for captioning / CLIP.

The first 165 slots are exactly a v2 picture, so base-only models keep working; the detail slots are optional.

  python extract_v3.py --out out/full --manifest out/full/manifest.jsonl --shards 0-9
  python extract_v3.py --out out/d0 --manifest m.jsonl --smoke --no-compile      # CPU plumbing test
Shard: strokes (N,361,11) fp16, keep (N,361), anchors (361,2), n_base 165, stage_psnr (N,3) @128, psnr_base128 (pruned),
psnr_base256, psnr_full256, ok, ids, labels, sources, levels.  Images: <out>/img/<id>.jpg (384 px, q90).
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

import batched
import detail_level as dl
from extract_prod import load_item, parse_shards, to_tensor
from extract_v2 import LEVELS, SMOKE_LEVELS, canon_direction, decompose_v2, removal_effect

LO, HI, CACHE = 128, 256, 384
DETAIL_G, DETAIL_STEPS = 14, 100


def anchors_v3(levels, g):
    base = [((c + .5) / s, (r + .5) / s) for s, *_ in levels for r in range(s) for c in range(s)]
    det = [((k % g + .5) / g, (k // g + .5) / g) for k in dl.detail_order(g)]
    return torch.tensor(base + det)


def safe_name(i):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", i)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True, help="jsonl rows: {id, source, label?, texts?, url|path|svg_url|drawing|code, box?}")
    ap.add_argument("--shard-size", type=int, default=400)
    ap.add_argument("--shards", default="all")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--prune", type=float, default=1e-4)
    ap.add_argument("--detail-steps", type=int, default=DETAIL_STEPS)
    ap.add_argument("--no-detail", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--threads", type=int, default=16, help="image download/load threads")
    ap.add_argument("--detail-g", type=int, default=DETAIL_G, help="detail grid side (14 -> 196 detail slots)")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    items_file = out / "items.json"
    if items_file.exists():
        items = json.loads(items_file.read_text())
    else:
        items = [json.loads(l) for l in open(a.manifest) if l.strip()]
        for r in items:
            r.setdefault("label", (r.get("texts") or [r["id"]])[0])
        random.Random(0).shuffle(items)  # every shard is a mixed sample
        items_file.write_text(json.dumps(items))
    n_shards = (len(items) + a.shard_size - 1) // a.shard_size
    todo = parse_shards(a.shards, n_shards)
    print(f"{len(items)} items, {n_shards} shards; this run: {todo}", flush=True)

    levels = SMOKE_LEVELS if a.smoke else LEVELS
    g, dsteps = (4, 3) if a.smoke else (a.detail_g, a.detail_steps)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mask_fn = (lambda s, gr, H: batched.stroke_masks(s, gr, H, K=4))
    if not a.no_compile:
        mask_fn = torch.compile(mask_fn)
    anchors = anchors_v3(levels, g)
    n_base = sum(s * s for s, *_ in levels)

    def safe_load(it):
        try:
            im = load_item(it, CACHE)
            im.save(out / "img" / f"{safe_name(it['id'])}.jpg", quality=90)
            return im
        except Exception as e:
            print("load failed:", it["id"], type(e).__name__, str(e)[:80], flush=True)
            return None

    pool, pre = cf.ThreadPoolExecutor(a.threads), cf.ThreadPoolExecutor(1)
    todo = [k for k in todo if not (out / f"shard_{k:05d}.pt").exists()]
    fetch = lambda k: list(pool.map(safe_load, items[k * a.shard_size:(k + 1) * a.shard_size]))
    nxt = pre.submit(fetch, todo[0]) if todo else None
    for q, k in enumerate(todo):
        path = out / f"shard_{k:05d}.pt"
        chunk = items[k * a.shard_size:(k + 1) * a.shard_size]
        t0 = time.perf_counter()
        loaded = nxt.result()  # images were downloaded while the previous shard was being fitted
        nxt = pre.submit(fetch, todo[q + 1]) if q + 1 < len(todo) else None
        n_try = len(chunk)
        chunk = [c for c, im in zip(chunk, loaded) if im is not None]  # dead links never reach the GPU
        loaded = [im for im in loaded if im is not None]
        if not chunk:
            continue
        load_ok = torch.tensor([im is not None for im in loaded])
        blank = Image.new("RGB", (CACHE, CACHE), "white")
        lo = torch.stack([to_tensor((im or blank).resize((LO, LO), Image.LANCZOS)) for im in loaded])
        hi = torch.stack([to_tensor((im or blank).resize((HI, HI), Image.LANCZOS)) for im in loaded])
        t_load = time.perf_counter() - t0
        out_s, out_k, P, PB, PB2, PF = [], [], [], [], [], []
        t_base = t_det = 0.0
        for i in range(0, len(chunk), a.batch):
            tl, th = lo[i:i + a.batch], hi[i:i + a.batch]
            n = len(tl)
            if n < a.batch:  # constant shapes for the compiled mask function
                tl = torch.cat([tl, tl[:1].expand(a.batch - n, -1, -1)])
                th = torch.cat([th, th[:1].expand(a.batch - n, -1, -1)])
            tl, th = tl.to(dev), th.to(dev)
            sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
            sync()
            t1 = time.perf_counter()
            s, ps = decompose_v2(tl, LO, levels, mask_fn)  # base: identical to extractor v2
            s = canon_direction(s)
            keep = removal_effect(s, LO, mask_fn) >= a.prune
            s[..., 10] = keep.float()
            with torch.no_grad():
                pb = batched.psnr(batched.render(s, LO, LO, mask_fn=mask_fn), tl)
                base_hi = batched.render(s, HI, HI, mask_fn=mask_fn)  # vector strokes re-rendered at 256
                pb2 = batched.psnr(base_hi, th)
            sync()
            t2 = time.perf_counter()
            if a.no_detail:
                sd = torch.zeros(len(tl), g * g, batched.N_PARAMS, device=dev)
                kd = torch.zeros(len(tl), g * g, dtype=torch.bool, device=dev)
                pf = pb2
            else:
                sd, _ = dl.fit_detail(th, base_hi, g, HI, steps=dsteps, compiled=not a.no_compile)
                sd = canon_direction(sd)
                kd = dl.detail_keep(sd, base_hi, th, g, HI, a.prune)
                sd[..., 10] = kd.float()
                with torch.no_grad():
                    pf = batched.psnr(dl.render_detail(sd, base_hi, g, HI), th)
            sync()
            t3 = time.perf_counter()
            t_base += t2 - t1
            t_det += t3 - t2
            out_s.append(torch.cat([s, sd], 1)[:n].cpu())
            out_k.append(torch.cat([keep, kd], 1)[:n].cpu())
            P.append(ps[:n].cpu()); PB.append(pb[:n].cpu()); PB2.append(pb2[:n].cpu()); PF.append(pf[:n].cpu())
        S, KEEP = torch.cat(out_s), torch.cat(out_k)
        P, PB, PB2, PF = map(torch.cat, (P, PB, PB2, PF))
        ok = torch.isfinite(S).flatten(1).all(1) & load_ok
        shard = {"strokes": canon_direction(S.half().float()).half(), "keep": KEEP, "anchors": anchors, "n_base": n_base,
                 "stage_psnr": P, "psnr_pruned": PB, "psnr_base256": PB2, "psnr_full256": PF, "ok": ok,
                 "ids": [c["id"] for c in chunk], "labels": [c["label"] for c in chunk], "sources": [c["source"] for c in chunk],
                 "size": LO, "detail": {"g": g, "size": HI, "steps": dsteps, "order": dl.detail_order(g)},
                 "levels": levels, "version": "v3"}
        torch.save(shard, str(path) + ".tmp")
        Path(str(path) + ".tmp").replace(path)
        fit = t_base + t_det
        rec = {"shard": k, "n": len(chunk), "tried": n_try, "base_sec": round(t_base, 1), "detail_sec": round(t_det, 1), "load_sec": round(t_load, 1),
               "imgs_per_hr_fit": int(3600 * len(chunk) / fit), "psnr_base128": round(PB[ok].mean().item(), 2),
               "psnr_base256": round(PB2[ok].mean().item(), 2), "psnr_full256": round(PF[ok].mean().item(), 2),
               "keep_base": round(KEEP[:, :n_base].float().mean().item(), 3), "keep_detail": round(KEEP[:, n_base:].float().mean().item(), 3),
               "bad": int((~ok).sum()), "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    print("DONE_ALL", flush=True)


if __name__ == "__main__":
    main()
