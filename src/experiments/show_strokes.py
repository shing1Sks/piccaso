"""Show what the model would actually have to emit for one image (CPU).

python show_strokes.py out/final_quality/shard.pt <image index> out/final_quality/strokes_view.png
Row 1: target, then the canvas after 1, 2, 4, 8, 16, 32 strokes (the painting order the model would learn).
Row 2: the first 12 strokes, each drawn ALONE on white (what one token-group means).
Row 3: strokes 100-111 alone (late, small, detail strokes).
Also prints the raw 11 numbers of the first strokes and the stroke-size statistics per stage.
"""
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

import batched
from extract import load_square

shard = torch.load(sys.argv[1])
i = int(sys.argv[2])
S, size = shard["strokes"][i], shard["size"]
manifest = {r["id"]: r for r in map(json.loads, open("pilot/manifest.jsonl"))}
T = 160


def pil(x, side):
    a = x.reshape(3, side, side).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((a * 255).astype(np.uint8)).resize((T, T), Image.LANCZOS)


@torch.no_grad()
def rend(s):
    return batched.render(s[None], size, size)[0]


def label(img, text):
    ImageDraw.Draw(img).text((3, 2), text, fill=(255, 0, 0))
    return img


tgt = pil(load_square(manifest[shard["ids"][i]]["path"], size), size)
row1 = [label(tgt, "target")] + [label(pil(rend(S[:k]), size), f"first {k}") for k in (1, 2, 4, 8, 16, 32, 64)]
row2 = [label(pil(rend(S[k:k + 1]), size), f"stroke #{k}") for k in range(8)]
row3 = [label(pil(rend(S[k:k + 1]), size), f"stroke #{k}") for k in range(120, 128)]
row4 = [label(pil(rend(S[k:k + 1]), size), f"stroke #{k}") for k in range(len(S) - 8, len(S))]
rows = [row1, row2, row3, row4]
sheet = Image.new("RGB", (T * 8, T * len(rows)), "white")
for r, row in enumerate(rows):
    for c, im in enumerate(row):
        sheet.paste(im, (c * T, r * T))
sheet.save(sys.argv[3])

print("image:", shard["ids"][i], shard["sources"][i], shard["labels"][i], "n_strokes:", len(S))
names = "x0 y0 x1 y1 x2 y2 width r g b alpha".split()
print("   ".join(f"{n:>6s}" for n in names))
for k in (0, 1, 2, 100, len(S) - 1):
    print(f"#{k:<3d}", "  ".join(f"{v:6.3f}" for v in S[k].tolist()))
cum = np.cumsum([n for n, _ in shard["stages"]]).tolist()
lo = 0
for hi in cum:
    w = S[lo:hi, 6] * 128
    print(f"strokes {lo:3d}-{hi:3d}: median width {w.median():5.1f}px of 128 (= {w.median()/128*100:4.1f}% of canvas)")
    lo = hi
