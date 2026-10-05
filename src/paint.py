"""Paint a picture from a prompt with Piccaso-0.1 (361 brush strokes) -> PNG and SVG.

  python paint.py "a red bus on a city street at dusk" --n 4 --out paintings
  python paint.py "a lighthouse on a cliff, oil painting" --model path/to/local/dir --steps 25 --cfg 3
Downloads the model from Hugging Face (shing-dev/Piccaso-0.1) and the Long-CLIP-B text encoder (BeichenZhang/LongCLIP-B,
~600 MB) on first use. Runs on CPU (slow: roughly 30-60 s per painting on a laptop) or CUDA.
Each painting is 361 quadratic Bezier strokes; the SVG is the real vector output, the PNG a 512 px render of it.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import batched
import detail_level as dl
import strokegen as sg


def load(model_path, dev, longclip=None):
    from safetensors.torch import load_file

    p = Path(model_path)
    if not p.exists():
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(model_path, allow_patterns=["config.json", "model.safetensors"]))
    cfg = json.load(open(p / "config.json"))
    t = load_file(str(p / "model.safetensors"))
    from huggingface_hub import hf_hub_download

    sg.TEXT_ENC.update(kind=cfg["text_encoder"], ckpt=longclip or hf_hub_download(cfg["longclip_repo"], cfg["longclip_file"]))
    sg.WMIN = cfg["wmin"]
    m = sg.SetDiT(cfg["slots"], cfg["n_classes"], cfg["d"], cfg["layers"], cfg["heads"], cfg["text_dim"],
                  cfg["selfcond"], cfg["canvas_feedback"], cfg["xattn"]).to(dev)
    m.load_state_dict({k[6:]: v for k, v in t.items() if k.startswith("model.")})
    m.eval()
    norm = sg.Norm.__new__(sg.Norm)
    norm.mean, norm.std = t["norm_mean"], t["norm_std"]
    return cfg, m, norm, {k: t[k].to(dev) for k in ("anchors", "text_mu", "text_sd", "keep_pool")}


@torch.no_grad()
def paint(prompts, cfg, m, norm, aux, dev, steps=25, w=3.0, seed=0):
    torch.manual_seed(seed)
    y = (sg.embed_text(prompts, dev).to(dev) - aux["text_mu"]) / aux["text_sd"]
    te = sg.get_text_enc(dev)
    ctx, cm = te(te.tokenize(prompts))
    z = sg.sample_B(m, y, cfg["slots"], steps, w, ctx=ctx, ctx_mask=cm)
    st = sg.vec_to_strokes(norm.dec(z), aux["anchors"])
    return sg.calibrate(st, norm.dec(z)[..., 0], aux["keep_pool"])  # stroke count copied from a real painting


@torch.no_grad()
def render(st, H=512):
    base = batched.render(st[:, :165], H, H)
    return dl.render_detail(st[:, 165:], base, 14, H).view(-1, 3, H, H)


def to_svg(s, size=512):
    """One stroke per <path>: quadratic Bezier, round caps, painted in slot order (coarse to fine), like the renderer."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
           f'<rect width="{size}" height="{size}" fill="white"/>']
    for x0, y0, x1, y1, x2, y2, wd, r, g, b, on in s.tolist():
        if on < 0.5:
            continue
        P = lambda v: f"{v * size:.1f}"
        col = f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        out.append(f'<path d="M{P(x0)} {P(y0)} Q{P(x1)} {P(y1)} {P(x2)} {P(y2)}" stroke="{col}" stroke-width="{wd * size:.2f}" '
                   f'stroke-linecap="round" fill="none"/>')
    return "\n".join(out + ["</svg>"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="shing-dev/Piccaso-0.1")
    ap.add_argument("--out", default="paintings")
    ap.add_argument("--longclip", default=None, help="local longclip-B.pt (otherwise downloaded from Hugging Face)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, m, norm, aux = load(a.model, dev, a.longclip)
    t0 = time.time()
    st = paint([a.prompt] * a.n, cfg, m, norm, aux, dev, a.steps, a.cfg, a.seed)
    ims = render(st)
    secs = (time.time() - t0) / a.n
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", a.prompt.lower()).strip("-")[:60]
    for k in range(a.n):
        Image.fromarray((ims[k].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / f"{slug}-{k}.png")
        (out / f"{slug}-{k}.svg").write_text(to_svg(st[k].cpu()))
    print(f"saved {a.n} painting(s) to {out}/  ({secs:.1f} s per painting on {dev})")
