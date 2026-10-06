"""CLIP pass (GPU): merge captions, score them, filter, dedupe, and save image vectors for reference-picture conditioning.

  python clip_pass.py out/full out/full/florence.json [--limit 2000] [--min-cap 0.22] [--min-trusted 0.15] [--min-render 0.18] [--dup 0.96]
Inputs: <run>/shard_*.pt (extract_v3), <run>/items.json (texts from the manifest), <run>/img/<id>.jpg, Florence json.
Outputs in <run>/:
  caps.json      {id: [captions]}  only kept items, only kept captions (training text)
  clipvec.pt     {ids, img (N,512) fp16, render_full (N,512), render_base (N,512)}  unit-norm CLIP ViT-B/32 image vectors
  clip_report.json  per-source counts + score percentiles + dropped examples (for eyeballing thresholds)
Rules: a caption is kept if CLIP(image, caption) >= min-cap; captions we trust (templates, COCO humans, Florence) only need
min-trusted. An item is dropped if no caption is kept, if its STROKE render no longer matches its best caption
(CLIP(render_full, caption) < min-render: the fit lost the content), or if it nearly duplicates an earlier item (cos >= dup).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import batched
import detail_level as dl
from extract_v3 import safe_name

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("florence")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--min-cap", type=float, default=0.22)
ap.add_argument("--min-trusted", type=float, default=0.15)
ap.add_argument("--min-render", type=float, default=0.18)
ap.add_argument("--dup", type=float, default=0.97)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
import open_clip

model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(dev).eval()
tok = open_clip.get_tokenizer("ViT-B-32")
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)

items = {r["id"]: r for r in json.load(open(f"{a.run}/items.json"))}
flor = json.load(open(a.florence)) if Path(a.florence).exists() else {}
TRUSTED_SRC = {"quickdraw", "twemoji", "noto", "openmoji", "fluent", "coco"}
DEDUP_SRC = {"coco", "coco_obj", "coco_head", "wikiart", "cleveland", "met", "pd12m"}  # CLIP calls different doodles/icons "duplicates"
import re
strip = lambda c: re.sub(r"^(The|This) (image|picture|photo|painting) (shows|depicts|features|is)\s*", "", c).strip()


@torch.no_grad()
def enc_img(x):  # x (B,3,224,224) in [0,1]
    return F.normalize(model.encode_image((x - MEAN) / STD).float(), dim=-1)


@torch.no_grad()
def enc_txt(strings, bs=512):
    out = []
    for i in range(0, len(strings), bs):
        out.append(F.normalize(model.encode_text(tok(strings[i:i + bs]).to(dev)).float(), dim=-1))
    return torch.cat(out) if out else torch.zeros(0, 512, device=dev)


ids, V_img, V_full, V_base, cap_lists, trusted = [], [], [], [], [], []
for f in sorted(glob.glob(f"{a.run}/shard_*.pt")):
    sh = torch.load(f)
    S, ok, nb = sh["strokes"].float(), sh["ok"], sh["n_base"]
    g = sh["detail"]["g"]
    for j0 in range(0, len(S), 64):
        sl = slice(j0, j0 + 64)
        sel = [j for j in range(j0, min(j0 + 64, len(S))) if ok[j]]
        if not sel:
            continue
        st = S[sel].to(dev)
        with torch.no_grad():
            base = batched.render(st[:, :nb], 224, 224)
            full = dl.render_detail(st[:, nb:], base, g, 224)
        imgs = []
        for j in sel:
            p = Path(a.run) / "img" / f"{safe_name(sh['ids'][j])}.jpg"
            im = Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)
            imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255)
        V_img.append(enc_img(torch.stack(imgs).to(dev)).half().cpu())
        V_full.append(enc_img(full.view(-1, 3, 224, 224)).half().cpu())
        V_base.append(enc_img(base.view(-1, 3, 224, 224)).half().cpu())
        for j in sel:
            i = sh["ids"][j]
            it = items[i]
            caps = [c for c in (it.get("texts") or [it["label"]]) if c]
            tr = [it["source"] in TRUSTED_SRC] * len(caps)
            for t in ("short", "detailed"):
                c = (flor.get(i) or {}).get(t)
                if c:
                    c = strip(c)
                    caps.append(c[:1].upper() + c[1:])
                    tr.append(True)
            ids.append(i)
            cap_lists.append(caps)
            trusted.append(tr)
    if a.limit and len(ids) >= a.limit:
        break
V_img, V_full, V_base = torch.cat(V_img), torch.cat(V_full), torch.cat(V_base)
uniq = sorted({c for cs in cap_lists for c in cs})
pos = {c: k for k, c in enumerate(uniq)}
T = enc_txt(uniq).cpu()
src = [items[i]["source"] for i in ids]

# --- scores, filters
keep_caps, drop_reason, scores = {}, {}, []
for n, (i, caps, tr) in enumerate(zip(ids, cap_lists, trusted)):
    tv = T[[pos[c] for c in caps]]
    si = (V_img[n].float() @ tv.T).tolist()
    sr = (V_full[n].float() @ tv.T).tolist()
    kept = [c for c, s, t in zip(caps, si, tr) if s >= (a.min_trusted if t else a.min_cap)]
    scores.append({"id": i, "src": src[n], "img_best": max(si), "render_best": max(sr)})
    if not kept:
        drop_reason[i] = "no caption matches the image"
    elif max(sr) < a.min_render:
        drop_reason[i] = "stroke version lost the content"
    else:
        keep_caps[i] = kept
# near-duplicates (image vectors), within kept items, first one wins
kid = [n for n, i in enumerate(ids) if i in keep_caps and src[n] in DEDUP_SRC]
X = V_img[kid].float().to(dev)
for s0 in range(0, len(kid), 2048):
    sim = X[s0:s0 + 2048] @ X.T
    for r in range(sim.shape[0]):
        n = s0 + r
        if ids[kid[n]] not in keep_caps:
            continue
        hit = (sim[r, :n] >= a.dup).nonzero()
        if len(hit):
            drop_reason[ids[kid[n]]] = f"near-duplicate of {ids[kid[int(hit[0])]]}"
            keep_caps.pop(ids[kid[n]], None)

json.dump(keep_caps, open(f"{a.run}/caps.json", "w"))
torch.save({"ids": ids, "img": V_img, "render_full": V_full, "render_base": V_base}, f"{a.run}/clipvec.pt")
rep = {"thresholds": vars(a), "n_scored": len(ids), "n_kept": len(keep_caps), "by_source": {}}
for s in sorted(set(src)):
    sc = [x for x in scores if x["src"] == s]
    ib, rb = np.array([x["img_best"] for x in sc]), np.array([x["render_best"] for x in sc])
    dr = [x["id"] for x in sc if x["id"] in drop_reason]
    rep["by_source"][s] = {"n": len(sc), "kept": len(sc) - len(dr), "img_best_p10_50_90": np.percentile(ib, [10, 50, 90]).round(3).tolist(),
                           "render_best_p10_50_90": np.percentile(rb, [10, 50, 90]).round(3).tolist(),
                           "dropped_examples": [(d, drop_reason[d], cap_lists[ids.index(d)][:2]) for d in dr[:6]]}
json.dump(rep, open(f"{a.run}/clip_report.json", "w"), indent=1)
print(json.dumps({k: (v["n"], v["kept"], v["img_best_p10_50_90"], v["render_best_p10_50_90"]) for k, v in rep["by_source"].items()}), flush=True)
print("CLIP_DONE scored", len(ids), "kept", len(keep_caps), flush=True)
