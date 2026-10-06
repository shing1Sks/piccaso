"""Which stroke field loses the most quality when quantized? Uses strokes saved by run_spike.py."""
import numpy as np, torch
from pathlib import Path
from run_spike import load
from stroke import render, psnr, quantize

BIG = 10**6  # effectively no quantization
configs = {
    "none": dict(bins_pos=BIG, bins_w=BIG, bins_c=BIG, bins_a=BIG),
    "default(64/16/16/4)": dict(),
    "pos 64 only": dict(bins_pos=64, bins_w=BIG, bins_c=BIG, bins_a=BIG),
    "pos 128 only": dict(bins_pos=128, bins_w=BIG, bins_c=BIG, bins_a=BIG),
    "width 16 only": dict(bins_pos=BIG, bins_w=16, bins_c=BIG, bins_a=BIG),
    "color 16 only": dict(bins_pos=BIG, bins_w=BIG, bins_c=16, bins_a=BIG),
    "color 32 only": dict(bins_pos=BIG, bins_w=BIG, bins_c=32, bins_a=BIG),
    "alpha 4 only": dict(bins_pos=BIG, bins_w=BIG, bins_c=BIG, bins_a=4),
    "alpha 8 only": dict(bins_pos=BIG, bins_w=BIG, bins_c=BIG, bins_a=8),
    "128/32/32/8": dict(bins_pos=128, bins_w=32, bins_c=32, bins_a=8),
}
rows = {k: [] for k in configs}
for npy in sorted(Path("out").glob("*_64_s150_strokes.npy")):
    s = torch.tensor(np.load(npy))
    img = next(Path("images").glob(npy.name.split("_64")[0] + ".*"))
    target = load(img, 64)
    for k, cfg in configs.items():
        with torch.no_grad():
            rows[k].append(psnr(render(quantize(s, **cfg), 64, 64), target))
for k, v in rows.items():
    print(f"{k:22s} mean PSNR {np.mean(v):6.2f}   per-image {[round(x,2) for x in v]}")
