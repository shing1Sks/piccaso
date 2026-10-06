"""Image -> stroke dataset extraction (batched, GPU). Writes one shard plus a timing/quality log.

Run from spike/:
  python extract.py pilot/manifest.jsonl out/pilot_128 --size 128 --batch 20
Shard (torch.save): strokes (N, S, 11) float32, stage_psnr (N, stages), ids, labels, sources, size, stages.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import batched

STAGES_DEFAULT = [(16, 0.30), (48, 0.15), (96, 0.07), (96, 0.035)]  # big brushes first, 256 strokes total


def load_square(path, size):
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    im = im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).reshape(3, -1).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--stages", type=str, default=None, help='JSON, e.g. "[[16,0.3],[48,0.15]]"')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compile", action="store_true", help="torch.compile the mask function")
    ap.add_argument("--checkpoint", action="store_true", help="recompute masks in backward to save memory")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stages = [tuple(x) for x in json.loads(a.stages)] if a.stages else STAGES_DEFAULT
    rows = [json.loads(l) for l in open(a.manifest)][: a.limit]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mask_fn = torch.compile(batched.stroke_masks) if a.compile else None

    torch.manual_seed(0)
    all_strokes, all_psnr, log = [], [], []
    t_start = time.perf_counter()
    for i in range(0, len(rows), a.batch):
        chunk = rows[i : i + a.batch]
        target = torch.stack([load_square(r["path"], a.size) for r in chunk]).to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        s, canvas, ps = batched.decompose(target, a.size, a.size, stages, steps=a.steps, mask_fn=mask_fn,
                                          use_checkpoint=a.checkpoint)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else 0.0
        all_strokes.append(s.cpu())
        all_psnr.append(ps.cpu())
        log.append({"first": i, "n": len(chunk), "sec": round(dt, 2), "sec_per_img": round(dt / len(chunk), 3),
                    "peak_gb": round(peak, 2), "final_psnr_mean": round(ps[:, -1].mean().item(), 2)})
        print(json.dumps(log[-1]), flush=True)

    total = time.perf_counter() - t_start
    shard = {
        "strokes": torch.cat(all_strokes), "stage_psnr": torch.cat(all_psnr),
        "ids": [r["id"] for r in rows], "labels": [r["label"] for r in rows], "sources": [r["source"] for r in rows],
        "size": a.size, "stages": stages, "steps": a.steps,
    }
    torch.save(shard, out / "shard.pt")
    summary = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "n_images": len(rows),
               "size": a.size, "batch": a.batch, "steps": a.steps, "stages": stages, "compile": a.compile,
               "checkpoint": a.checkpoint, "total_sec": round(total, 1),
               "sec_per_img": round(total / len(rows), 3), "imgs_per_hour": int(3600 * len(rows) / total),
               "batches": log}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", json.dumps({k: v for k, v in summary.items() if k != "batches"}))


if __name__ == "__main__":
    main()
