"""Peek at extracted PixelProse data: real picture | 361-stroke render, with the caption (research/17 stage 1 check).

  python pp_peek.py out/pp/w0 out/pp/peek_w0.png [n=12]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import batched
import detail_level as dl
from extract_v3 import safe_name

d, outp = Path(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
dev = "cuda" if torch.cuda.is_available() else "cpu"
texts = {r["id"]: r["texts"] for r in json.load(open(d / "items.json", encoding="utf-8"))}
sh = torch.load(sorted(glob.glob(str(d / "shard_*.pt")))[-1], weights_only=False)
ok = [k for k, o in enumerate(sh["ok"].tolist()) if o][:n]
st = sh["strokes"][ok].float().to(dev)
with torch.no_grad():
    base = torch.cat([batched.render(st[q:q + 8, :sh["n_base"]], 256, 256) for q in range(0, len(st), 8)])
    full = torch.cat([dl.render_detail(st[q:q + 8, sh["n_base"]:], base[q:q + 8], sh["detail"]["g"], 256) for q in range(0, len(st), 8)])
T = 256
sheet = Image.new("RGB", (T * 4, (T + 44) * ((len(ok) + 1) // 2)), "white")
dr = ImageDraw.Draw(sheet)
for j, k in enumerate(ok):
    i = sh["ids"][k]
    real = Image.open(d / "img" / f"{safe_name(i)}.jpg").resize((T, T))
    fit = Image.fromarray((full[j].view(3, T, T).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    x, y = (j % 2) * 2 * T, (j // 2) * (T + 44)
    sheet.paste(real, (x, y))
    sheet.paste(fit, (x + T, y))
    c = texts[i][0]
    for q in range(3):
        dr.text((x + 2, y + T + 2 + 13 * q), c[q * 80:(q + 1) * 80], fill=(0, 0, 0))
sheet.save(outp)
print("PEEK_SAVED", outp)
