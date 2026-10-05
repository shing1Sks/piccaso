"""Render contact sheets from an extraction shard (CPU is fine). Strokes are re-rendered, not upscaled.

python make_sheet.py out/pilot_128/shard.pt out/pilot_128
Writes <out>/sheet_all.png (target | reconstruction, 5 pairs per row) and <out>/sheet_progression.png
(target | 16 | 64 | 160 | 256 strokes | quantized | 2x-res re-render for two images per source).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import batched
from extract import load_square

shard = torch.load(sys.argv[1])
out = Path(sys.argv[2])
S, size = shard["strokes"], shard["size"]
manifest = {r["id"]: r for r in map(__import__("json").loads, open("pilot/manifest.jsonl"))}
n_stage = np.cumsum([n for n, _ in shard["stages"]])  # e.g. 16, 64, 160, 256


def to_pil(x, side):
    a = x.reshape(3, side, side).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((a * 255).astype(np.uint8))


@torch.no_grad()
def rend(strokes, side):
    return batched.render(strokes[None], side, side)[0]


# sheet 1: every image, target | reconstruction
T = size
pairs_per_row = 5
rows = (len(S) + pairs_per_row - 1) // pairs_per_row
sheet = Image.new("RGB", (2 * T * pairs_per_row, T * rows), "white")
for i in range(len(S)):
    tgt = to_pil(load_square(manifest[shard["ids"][i]]["path"], size), size)
    rec = to_pil(rend(S[i], size), size)
    x, y = (i % pairs_per_row) * 2 * T, (i // pairs_per_row) * T
    sheet.paste(tgt, (x, y))
    sheet.paste(rec, (x + T, y))
sheet.save(out / "sheet_all.png")

# sheet 2: progression for the first two images of each source
chosen, seen = [], {}
for i, src in enumerate(shard["sources"]):
    seen[src] = seen.get(src, 0) + 1
    if seen[src] <= 2:
        chosen.append(i)
cols = 1 + len(n_stage) + 2
prog = Image.new("RGB", (T * cols, T * len(chosen)), "white")
q = batched.quantize(S)
for r, i in enumerate(chosen):
    tiles = [to_pil(load_square(manifest[shard["ids"][i]]["path"], size), size)]
    tiles += [to_pil(rend(S[i, :k], size), size) for k in n_stage]
    tiles.append(to_pil(rend(q[i], size), size))
    tiles.append(to_pil(rend(S[i], size * 2), size * 2).resize((T, T), Image.LANCZOS))  # same strokes at 2x res
    for c, t in enumerate(tiles):
        prog.paste(t, (c * T, r * T))
d = ImageDraw.Draw(prog)
for c, name in enumerate(["target"] + [f"{k} strokes" for k in n_stage] + ["quantized", "2x res"]):
    d.text((c * T + 3, 2), name, fill=(255, 0, 0))
prog.save(out / "sheet_progression.png")
print("wrote", out / "sheet_all.png", sheet.size, out / "sheet_progression.png", prog.size)
