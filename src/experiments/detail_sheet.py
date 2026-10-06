"""Before/after sheet for the detail layer: original | base 165 strokes | base + detail | 2x zoom of the centre (base, detail).
python detail_sheet.py out/demo3 out/demo3/detail_sheet.png"""
import glob
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

import batched
import detail_level as dl
from extract_v3 import safe_name

run, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
skip_src = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else set()
T = 256
rows = []
for f in sorted(glob.glob(f"{run}/shard_*.pt")):
    sh = torch.load(f)
    S, nb, g = sh["strokes"].float(), sh["n_base"], sh["detail"]["g"]
    sel = [j for j in range(len(sh["ids"])) if sh["sources"][j] not in skip_src and sh["ok"][j]][:max(0, limit - len(rows))]
    if not sel:
        continue
    with torch.no_grad():
        base = batched.render(S[sel, :nb], T, T)
        full = dl.render_detail(S[sel, nb:], base, g, T)
    for q, j in enumerate(sel):
        i = sh["ids"][j]
        orig = Image.open(f"{run}/img/{safe_name(i)}.jpg").convert("RGB").resize((T, T), Image.LANCZOS)
        rows.append((i, orig, base[q], full[q], sh["psnr_base256"][j].item(), sh["psnr_full256"][j].item(),
                     int(sh["keep"][j, :nb].sum()), int(sh["keep"][j, nb:].sum())))
tile = lambda x: Image.fromarray((x.view(3, T, T).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
zoom = lambda im: im.crop((T // 4, T // 4, 3 * T // 4, 3 * T // 4)).resize((T, T), Image.NEAREST)
W = 5
sheet = Image.new("RGB", (T * W, T * len(rows) + 18), "white")
dr = ImageDraw.Draw(sheet)
for c, h in enumerate(["original (256 px)", "base: 165 strokes", "base + detail layer", "zoom: base", "zoom: base + detail"]):
    dr.text((c * T + 4, 3), h, fill=(0, 0, 0))
for r, (i, orig, b, fu, pb, pf, kb, kd) in enumerate(rows):
    bi, fi = tile(b), tile(fu)
    for c, im in enumerate([orig, bi, fi, zoom(bi), zoom(fi)]):
        sheet.paste(im, (c * T, 18 + r * T))
    dr.text((T + 4, 18 + r * T + 4), f"{pb:.1f} dB, {kb} strokes", fill=(255, 0, 0))
    dr.text((2 * T + 4, 18 + r * T + 4), f"{pf:.1f} dB, {kb}+{kd} strokes", fill=(255, 0, 0))
sheet.save(out)
print("saved", out)
for i, *_, pb, pf, kb, kd in rows:
    print(f"{i:28s} base {pb:.2f} dB -> +detail {pf:.2f} dB  (+{pf - pb:.2f}), strokes {kb} + {kd}")
