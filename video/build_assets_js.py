"""Pack the real stroke SVGs into assets.js (window.PAINTINGS) so the film can animate every stroke in paint order.

  python build_assets_js.py   -> assets.js   (stroke = [x0,y0,x1,y1,x2,y2,width] in 0..512 plus "rgb(...)")
"""
import json
import re
from pathlib import Path

A = Path(__file__).resolve().parent / "a"
P = re.compile(r'<path d="M([\d.-]+) ([\d.-]+) Q([\d.-]+) ([\d.-]+) ([\d.-]+) ([\d.-]+)" stroke="(rgb\([^)]*\))" stroke-width="([\d.]+)"')
out = {}
for f in sorted(A.glob("*.svg")):
    strokes = [[*map(float, m.group(1, 2, 3, 4, 5, 6)), float(m.group(8)), m.group(7)] for m in P.finditer(f.read_text())]
    out[f.stem] = strokes
    print(f.stem, len(strokes))
meta = json.load(open(A / "assets.json")) if (A / "assets.json").exists() else []
from PIL import Image  # 28x28 colour grid of the same real photo, for the "pixels" side of the opening shot

im = Image.open(A / "fitted_photo_real.jpg").convert("RGB")
s_ = min(im.size)
im = im.crop(((im.width - s_) // 2, (im.height - s_) // 2, (im.width - s_) // 2 + s_, (im.height - s_) // 2 + s_)).resize((28, 28), Image.BOX)
pixels = ["#%02x%02x%02x" % im.getpixel((x, y)) for y in range(28) for x in range(28)]
(Path(__file__).resolve().parent / "assets.js").write_text("window.PAINTINGS = " + json.dumps(out) + ";\nwindow.ASSET_META = " + json.dumps(meta) + ";\nwindow.PIXELS = " + json.dumps(pixels) + ";\n")
print("assets.js written")
