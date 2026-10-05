# The research film

A ~106 s "research ad": the whole Piccaso journey as one animated page, rendered frame by frame to MP4.
Every painting on screen is real. The `paint_*.svg` files were painted by Piccaso-0.1. `fitted_photo.svg` is a real
PixelProse photo after the stroke-fitting pipeline. The strips in `a/` are cropped from the notebook figures.

| File | What |
|---|---|
| `index.html` | the film: one paused GSAP timeline, 13 scenes, exposes `window.seek(t)`, `window.DURATION`, `window.ready` |
| `make_assets.py` | paints the SVGs with the released model and fits one real photo (CPU is fine) |
| `build_assets_js.py` | packs the SVG strokes into `assets.js` so the page can draw them stroke by stroke |
| `render.cjs` | seeks the timeline frame by frame and pipes screenshots into ffmpeg |

```bash
open index.html                      # plays in the browser
npm i playwright-core                # plus ffmpeg on PATH and Chrome or Edge
PW_CORE=./node_modules/playwright-core node render.cjs --size 1920x1080 --channel msedge --stills 7.4,24.3,104
PW_CORE=./node_modules/playwright-core node render.cjs --size 1920x1080 --channel msedge --out piccaso.mp4
```

Two gotchas we hit, fixed in `index.html`:
- `window.seek` must return nothing. Returning the GSAP timeline makes Playwright serialise it, and that crashes the tab.
- Animate `stroke-dashoffset` through `attr`, not CSS. GSAP rounds CSS px values, so a `pathLength=1` stroke snaps on instead of being drawn.
