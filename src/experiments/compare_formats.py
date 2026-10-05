"""Equal-budget comparison on the same images: Paint Transformer stamps vs the Bezier decomposer.

Needs out/pt_gpu_smoke/samples_<size>.png (rows of [target | PT canvas]) and out/<stem>_<size>_s150_strokes.npy.
Run from spike/: python compare_formats.py <size>
"""
import json
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image

from run_spike import load
from stroke import render

SIZE = int(sys.argv[1])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
names = ["astronaut", "chelsea", "coffee", "rocket"]  # same order pt_bench_gpu.py saved them (sorted)
sheet = np.asarray(Image.open(f"out/pt_gpu_smoke/samples_{SIZE}.png").convert("RGB"))
metric = lpips.LPIPS(net="alex", verbose=False).to(DEV)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().div(255).to(DEV)


def psnr(a, b):
    return (10 * torch.log10(1 / (a - b).pow(2).mean())).item()


rows = []
for i, name in enumerate(names):
    img = next(Path("images").glob(name + ".*"))
    target = load(img, SIZE).reshape(1, 3, SIZE, SIZE)
    pt = to_t(sheet[i * SIZE:(i + 1) * SIZE, SIZE:2 * SIZE])
    s = torch.tensor(np.load(f"out/{name}_{SIZE}_s150_strokes.npy"), device=DEV)
    with torch.no_grad():
        mine = render(s, SIZE, SIZE).reshape(1, 3, SIZE, SIZE).clamp(0, 1)
        row = {
            "image": name,
            "bezier_strokes": len(s),
            "bezier_psnr": round(psnr(mine, target), 2),
            "bezier_lpips": round(metric(mine * 2 - 1, target * 2 - 1).item(), 4),
            "pt_psnr": round(psnr(pt, target), 2),
            "pt_lpips": round(metric(pt * 2 - 1, target * 2 - 1).item(), 4),
        }
    rows.append(row)
    print(json.dumps(row), flush=True)

mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if k not in ("image", "bezier_strokes")}
print("MEAN", json.dumps(mean))
