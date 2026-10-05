"""Diagnose a trained arm-B checkpoint (research/14): where does it lose, and do sampler settings matter?

  python diag_eval.py out/B58/ckpt.pt --path out/final --out out/B58/diag.json
Same held-out split as strokegen.py (seed 0). Writes <out> (json) and <out>.png (per-source sheet: fitted | generated).
  global:     CLIP image-text retrieval on the first --n held-out captions (exactly strokegen's --clip-eval) for each
              denoising-step count (cfg 2) and each guidance scale (50 steps), plus the calibrated keep-bit variant.
  per source: retrieval among --per-source held-out captions of the SAME source, for the real picture (CLIP vector of the
              photo), the fitted strokes (ceiling of the stroke format) and generated strokes.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import strokegen as sg

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--path", default="out/final")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--steps", default="25,50,100,250")
ap.add_argument("--cfgs", default="1,1.5,3,4")
ap.add_argument("--per-source", type=int, default=150)
ap.add_argument("--bs", type=int, default=125)
ap.add_argument("--all", action="store_true", help="test-only data (e.g. DOCCI): every item is held out")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
ICONS = {"twemoji", "noto", "openmoji", "fluent"}

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
N = ck["N"]
sg.WMIN = ck.get("wmin", 0.008)
torch.manual_seed(0)
S, keep, labels, ids, anchors = sg.load_data(a.path, "v2", N)
caps = json.load(open(f"{a.path}/caps.json"))
sel = [j for j, i in enumerate(ids) if i in caps]
S, keep, ids = S[sel], keep[sel], [ids[j] for j in sel]
norm = sg.Norm.__new__(sg.Norm)
norm.mean, norm.std = ck["norm"][0].cpu(), ck["norm"][1].cpu()
Z = norm.enc(sg.to_vec(S, keep, anchors), keep)
perm = torch.randperm(len(Z))
vi = perm.tolist() if a.all else perm[:max(8, len(Z) // 20)].tolist()
src_of = {}
for f in sorted(glob.glob(f"{a.path}/shard_*.pt")):
    sh = torch.load(f, weights_only=False)
    src_of.update({i: s for i, s in zip(sh["ids"], sh["sources"])})
group = lambda i: "icons" if src_of[i] in ICONS else src_of[i]

ca = ck["args"]
sg.TEXT_ENC.update(kind=ca.get("text_encoder", "clip"), ckpt=ca.get("longclip_ckpt", "out/models/longclip-B.pt"))
model = sg.SetDiT(N, len(ck["classes"]), ca["d"], ca["layers"], ca["heads"], ck["text_dim"], ca.get("selfcond", False),
                  ca.get("canvas_fb", False), ca.get("xattn", False)).to(dev)
model.load_state_dict(ck["ema"])
model.eval()
mu, sd = ck["text_norm"]
cv = torch.load(f"{a.path}/clipvec.pt", weights_only=False)
cv_at = {i: k for k, i in enumerate(cv["ids"])}
IMG_KEY = "img_oai" if "img_oai" in cv else "img"  # real-picture vectors in the scorer's (OpenAI CLIP) space
Ag, Kg = anchors.to(dev), keep.to(dev)
fb = sg.render_fb_fn(norm, Ag) if model.use_canvas else None
ctok = sg.ClipTokens(dev) if model.xattn else None

import open_clip

cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
cm = cm.to(dev).eval()
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def img_vecs(st):
    out = []
    for j in range(0, len(st), 32):
        im = sg.render_samples(st[j:j + 32], 224).view(-1, 3, 224, 224)
        out.append(F.normalize(cm.encode_image((im - MEAN) / STD).float(), dim=-1))
    return torch.cat(out)


def score(ie, te):
    sim = ie @ te.T
    diag = sim.diag()
    return {"top1": round((sim.argmax(1) == torch.arange(len(te), device=dev)).float().mean().item(), 4),
            "gap": round((diag.mean() - (sim.sum() - diag.sum()) / (sim.numel() - len(te))).item(), 4)}


def generate(cond, steps, w, calib=False, seed=1234, ctx=None):
    torch.manual_seed(seed)
    c, m = ctx if ctx is not None else (None, None)
    zs = torch.cat([sg.sample_B(model, cond[j:j + a.bs], N, steps, w, ctx=None if c is None else c[j:j + a.bs],
                                ctx_mask=None if m is None else m[j:j + a.bs], fb=fb) for j in range(0, len(cond), a.bs)])
    st = sg.vec_to_strokes(norm.dec(zs), Ag)
    return sg.calibrate(st, norm.dec(zs)[..., 0], Kg) if calib else st


def setup(items):
    cs = [caps[ids[i]][0] for i in items]
    te = sg.score_text(cs, dev).to(dev)  # scorer: OpenAI CLIP, whatever encoder the model was conditioned on
    cond = ((sg.embed_text(cs, dev) if sg.TEXT_ENC["kind"] != "clip" else te.cpu()).to(dev) - mu.to(dev)) / sd.to(dev)
    fit = sg.vec_to_strokes(norm.dec(Z[items].to(dev)), Ag)
    real = F.normalize(torch.stack([cv[IMG_KEY][cv_at[ids[i]]].float() for i in items]).to(dev), dim=-1)
    return cs, te, cond, fit, real


def ctx_of(cs):
    return ctok(ctok.tokenize(cs)) if ctok is not None else None


res = {"ckpt": a.ckpt, "N": N, "global": {}, "per_source": {}}
g_items = vi[:a.n]
cs, te, cond, fit, real = setup(g_items)
gctx = ctx_of(cs)
res["global"]["n"] = len(g_items)
res["global"]["real_picture"] = score(real, te)
res["global"]["fitted_strokes"] = score(img_vecs(fit), te)
for st_n in [int(x) for x in a.steps.split(",")]:
    res["global"][f"steps{st_n}_cfg2"] = score(img_vecs(generate(cond, st_n, 2.0, ctx=gctx)), te)
    print(st_n, res["global"][f"steps{st_n}_cfg2"], flush=True)
for w in [float(x) for x in a.cfgs.split(",")]:
    res["global"][f"steps50_cfg{w:g}"] = score(img_vecs(generate(cond, 50, w, ctx=gctx)), te)
    print("cfg", w, res["global"][f"steps50_cfg{w:g}"], flush=True)
res["global"]["steps50_cfg2_calibrated"] = score(img_vecs(generate(cond, 50, 2.0, calib=True, ctx=gctx)), te)
print(json.dumps(res["global"]), flush=True)

groups = sorted({group(ids[i]) for i in vi})
T, show = 224, 3
sheet = Image.new("RGB", (T * 2 * show, T * len(groups)), "white")
dr = ImageDraw.Draw(sheet)
for r, gname in enumerate(groups):
    items = [i for i in vi if group(ids[i]) == gname][:a.per_source]
    if len(items) < 8:
        continue
    cs, te, cond, fit, real = setup(items)
    gen = generate(cond, 50, 2.0, ctx=ctx_of(cs))
    fv, gv = img_vecs(fit), img_vecs(gen)
    e = {"n": len(items), "chance_top1": round(1 / len(items), 4), "real_picture": score(real, te), "fitted_strokes": score(fv, te),
         "generated": score(gv, te)}
    e["gap_generated_over_fitted"] = round(e["generated"]["gap"] / max(e["fitted_strokes"]["gap"], 1e-6), 3)
    res["per_source"][gname] = e
    print(gname, json.dumps(e), flush=True)
    ims = sg.render_samples(torch.cat([fit[:show], gen[:show]]), T)
    for q in range(show):
        for k, im in enumerate((ims[q], ims[show + q])):
            arr = (im.view(3, T, T).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            sheet.paste(Image.fromarray(arr), ((2 * q + k) * T, r * T))
        dr.text(((2 * q) * T + 3, r * T + 2), f"{gname}: {cs[q][:40]}", fill=(255, 0, 0))
        dr.text(((2 * q) * T + 3, r * T + 14), "fitted | generated", fill=(255, 0, 0))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
sheet.save(str(Path(a.out).with_suffix(".png")))
json.dump(res, open(a.out, "w"), indent=1)
print("DIAG_DONE", flush=True)
