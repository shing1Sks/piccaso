"""Final evaluation of Piccaso-0.1 from an eval pack (make_eval_pack.py): seen vs unseen vs a different source.

  python eval_final.py ckpt.pt --pack eval_pack.pt --out out/final_eval [--steps 25 --cfg 3]
For each set: CLIP ViT-B/32 retrieval among the set's captions (top-1, gap) for the real picture, the fitted 361 strokes
(the ceiling of the stroke format; PixelProse sets only) and the generated painting. Plus seconds per picture.
Writes results.json and sheet_<set>.png (real | fitted strokes | two generated samples).
"""
import argparse
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import batched
import detail_level as dl
import strokegen as sg

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--pack", required=True)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--bs", type=int, default=50)
ap.add_argument("--out", default="out/final_eval")
ap.add_argument("--longclip", default="out/models/longclip-B.pt")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ca = ck["args"]
sg.TEXT_ENC.update(kind=ca["text_encoder"], ckpt=a.longclip)
sg.WMIN = ck["wmin"]
N = ck["N"]
model = sg.SetDiT(N, len(ck["classes"]), ca["d"], ca["layers"], ca["heads"], ck["text_dim"], ca["selfcond"], ca["canvas_fb"], ca["xattn"]).to(dev)
model.load_state_dict(ck["ema"])
model.eval()
norm = sg.Norm.__new__(sg.Norm)
norm.mean, norm.std = ck["norm"]
A = ck["anchors"].to(dev)
mu, sd = [t.to(dev) for t in ck["text_norm"]]
te = sg.get_text_enc(dev)
pack = torch.load(a.pack, weights_only=False)

import open_clip

clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip = clip.to(dev).eval()
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def ivec(img):
    out_ = []
    for j in range(0, len(img), 64):
        x = F.interpolate(img[j:j + 64].to(dev), size=224, mode="bilinear", align_corners=False)
        out_.append(F.normalize(clip.encode_image((x - MEAN) / STD).float(), dim=-1))
    return torch.cat(out_)


@torch.no_grad()
def render361(st):
    st = st.to(dev).float()
    base = torch.cat([batched.render(st[q:q + 16, :165], 256, 256) for q in range(0, len(st), 16)])
    full = torch.cat([dl.render_detail(st[q:q + 16, 165:], base[q:q + 16], 14, 256) for q in range(0, len(st), 16)])
    return full.view(-1, 3, 256, 256)


def score(ie, tv):
    sim = ie @ tv.T
    d = sim.diag()
    return {"top1": round((sim.argmax(1) == torch.arange(len(tv), device=sim.device)).float().mean().item(), 4),
            "gap": round((d.mean() - (sim.sum() - d.sum()) / (sim.numel() - len(tv))).item(), 4)}


timing = []


def generate(caps, seed=0):
    torch.manual_seed(seed)
    y = (sg.embed_text(caps, dev).to(dev) - mu) / sd
    ctx, cm = te(te.tokenize(caps))
    outs = []
    for j in range(0, len(caps), a.bs):
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        zs = sg.sample_B(model, y[j:j + a.bs], N, a.steps, a.cfg, ctx=ctx[j:j + a.bs], ctx_mask=cm[j:j + a.bs])
        if dev == "cuda":
            torch.cuda.synchronize()
        timing.append((time.time() - t0) / len(zs))
        outs.append(sg.vec_to_strokes(norm.dec(zs), A))
    return render361(torch.cat(outs))


def fetch(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=15).read())).convert("RGB")
    except Exception:
        return None


def to_pil(t):
    return Image.fromarray((t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


def sheet(name, caps, reals, fitted, gens, k=8):
    T, cols = 256, (2 if fitted is not None else 1) + len(gens)
    im = Image.new("RGB", (T * cols, (T + 44) * k), "white")
    dr = ImageDraw.Draw(im)
    for r in range(k):
        tiles = [reals[r]] + ([to_pil(fitted[r])] if fitted is not None else []) + [to_pil(g[r]) for g in gens]
        for c, t in enumerate(tiles):
            im.paste(t.resize((T, T)), (c * T, r * (T + 44)))
        for q in range(3):
            dr.text((4, r * (T + 44) + T + 2 + 13 * q), caps[r][q * 120:(q + 1) * 120], fill=(0, 0, 0))
    im.save(out / f"sheet_{name}.png")


res = {"ckpt": a.ckpt, "steps": a.steps, "cfg": a.cfg, "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
       "n_train_items": pack["n_train"], "n_heldout_items": pack["n_heldout"], "sets": {}}
for name in ("seen", "unseen"):
    p = pack[name]
    tv = sg.score_text(p["captions"], dev).to(dev)
    real = F.normalize(p["img_oai"].float().to(dev), dim=-1)
    fit = render361(p["strokes"])
    gen = generate(p["captions"])
    res["sets"][name] = {"n": len(p["captions"]), "real_picture": score(real, tv), "fitted_strokes": score(ivec(fit), tv),
                         "generated": score(ivec(gen), tv)}
    print(name, json.dumps(res["sets"][name]), flush=True)
    reals = [fetch(u) or Image.new("RGB", (256, 256), "lightgray") for u in p["urls"][:8]]
    sheet(name, p["captions"], reals, fit[:8], [gen[:8], generate(p["captions"][:8], seed=1)])
p = pack["docci"]
tv = sg.score_text(p["captions"], dev).to(dev)
imgs = p["images"].permute(0, 3, 1, 2).float() / 255
gen = generate(p["captions"])
res["sets"]["docci"] = {"n": len(p["captions"]), "real_picture": score(ivec(imgs), tv), "generated": score(ivec(gen), tv)}
print("docci", json.dumps(res["sets"]["docci"]), flush=True)
sheet("docci", p["captions"], [to_pil(x) for x in imgs[:8]], None, [gen[:8], generate(p["captions"][:8], seed=1)])
res["seconds_per_picture"] = round(float(np.median(timing)), 4)
res["batch"] = a.bs
json.dump(res, open(out / "results.json", "w"), indent=1)
print(json.dumps(res, indent=1), "\nEVAL_FINAL_DONE", flush=True)
