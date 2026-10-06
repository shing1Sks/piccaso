"""Sheet for the memorisation test: rows = training pictures, columns = fitted target | arm B sample | arm A sample.
python mem_sheet.py out/mem/sheet64.png out/mem/B64_cfg1.0.pt out/mem/A64_cfg1.0.pt"""
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

out, files = sys.argv[1], sys.argv[2:]
runs = [torch.load(f, map_location="cpu") for f in files]
n, T = 12, 128
sheet = Image.new("RGB", (T * (1 + len(runs)), T * n + 16), "white")
dr = ImageDraw.Draw(sheet)
for c, name in enumerate(["target (fitted)"] + [f.split("/")[-1].replace(".pt", "") for f in files]):
    dr.text((c * T + 3, 2), name, fill=(200, 0, 0))
tile = lambda x: Image.fromarray((x.float().view(3, T, T).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))
for r in range(n):
    sheet.paste(tile(runs[0]["tgt"][r]), (0, 16 + r * T))
    for c, run in enumerate(runs):
        sheet.paste(tile(run["gen"][r]), ((c + 1) * T, 16 + r * T))
    dr.text((3, 16 + r * T + 2), runs[0]["src"][r], fill=(0, 0, 255))
sheet.save(out)
print("saved", out)
