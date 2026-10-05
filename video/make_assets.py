"""Real assets for the Piccaso journey video: Piccaso-0.1 paintings as SVG (stroke order = paint order) and one real photo with
its fitted 361 strokes. Everything shown in the video is produced by the actual model and pipeline.

  cd video && PP_DATA=path/to/ppfinal python make_assets.py   (CPU is fine, ~1.5 min per painting)
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import paint  # noqa: E402
import strokegen as sg  # noqa: E402

A = HERE / "a"
A.mkdir(exist_ok=True)
LONGCLIP = os.environ.get("LONGCLIP")  # optional local longclip-B.pt; downloaded when unset
DATA = Path(os.environ.get("PP_DATA", "ppfinal"))  # the local stroke dataset (caps.json, items.json, shard_*.pt)

prompts = [
    ("lighthouse", "a lighthouse on a cliff at sunset, oil painting", 3),
    ("ship", "a sailing ship on a stormy sea, oil painting", 1),
    ("city", "a city street at night with glowing lights", 2),
    ("sunflowers", "a field of sunflowers under a blue sky", 4),
    ("fruit", "a bowl of oranges on a wooden table", 5),
    ("room", "a cozy living room with a big window and plants", 6),
    ("portrait", "an oil painting portrait of an old man with a beard", 7),
    ("bus", "a red bus on a city street", 8),
]
cfg, m, norm, aux = paint.load(str(HERE.parent / "hf_export"), "cpu", LONGCLIP)
meta = []
for name, prompt, seed in prompts:
    out = A / f"paint_{name}.svg"
    if out.exists():
        meta.append({"name": name, "prompt": prompt, "svg": out.name})
        continue
    st = paint.paint([prompt], cfg, m, norm, aux, "cpu", 25, 3.0, seed)
    out.write_text(paint.to_svg(st[0].cpu(), 512))
    meta.append({"name": name, "prompt": prompt, "svg": out.name})
    print("painted", name, flush=True)

# one real PixelProse photo + its fitted strokes (the training target), found by caption
caps = json.load(open(DATA / "caps.json", encoding="utf-8"))
items = {r["id"]: r for r in json.load(open(DATA / "items.json"))}
want = ["tunnel of cherry blossom", "lighthouse on", "sunset over the"]
pick = None
for w in want:
    for i, cs in caps.items():
        if w in cs[0].lower() and len(cs[0]) < 400:
            pick = i
            break
    if pick:
        break
import glob

for f in sorted(glob.glob(str(DATA / "shard_*.pt"))):
    sh = torch.load(f, weights_only=False)
    if pick in sh["ids"]:
        k = sh["ids"].index(pick)
        S = sh["strokes"][k].float()
        break
(A / "fitted_photo.svg").write_text(paint.to_svg(S, 512))
req = urllib.request.Request(items[pick]["url"], headers={"User-Agent": "Mozilla/5.0"})
(A / "fitted_photo_real.jpg").write_bytes(urllib.request.urlopen(req, timeout=20).read())
meta.append({"name": "fitted", "caption": caps[pick][0], "id": pick})
json.dump(meta, open(A / "assets.json", "w"), indent=1)
print("ASSETS_DONE", pick, caps[pick][0][:120])
