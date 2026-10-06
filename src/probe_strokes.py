"""Data probe, GPU step (research/15): how much of each caption survives the stroke format?

  python probe_strokes.py make  pp oi_ln docci          -> out/probe/manifest.jsonl (300 items per source, local paths)
  python extract_v3.py --out out/probe/v3 --manifest out/probe/manifest.jsonl --shard-size 300
  python probe_strokes.py score pp oi_ln docci          -> out/probe/strokes.json
Score: within-source CLIP retrieval (n=150) of the caption from (a) the real picture, (b) the 165 base strokes, (c) all 361
strokes; full caption and first sentence. Same protocol as diag_eval per-source rows, so the numbers compare to COCO/WikiArt.
"""
import glob
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

mode, srcs = sys.argv[1], sys.argv[2:]
if mode == "make":
    random.seed(0)
    with open("out/probe/manifest.jsonl", "w", encoding="utf-8") as f:
        for s in srcs:
            rows = [json.loads(l) for l in open(f"out/probe/{s}/rows.jsonl", encoding="utf-8")]
            rows = [r for r in rows if Path(f"out/probe/{s}/img/{r['id']}.jpg").exists()]
            random.shuffle(rows)
            for r in rows[:300]:
                f.write(json.dumps({"id": f"{s}:{r['id']}", "source": s, "label": s, "path": f"out/probe/{s}/img/{r['id']}.jpg",
                                    "caption": r["caption"]}) + "\n")
    print("MANIFEST_DONE")
    sys.exit()

import open_clip

import batched
import detail_level as dl
import strokegen as sg

dev = "cuda"
cm, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
cm = cm.to(dev).eval()
tok = open_clip.get_tokenizer("ViT-B-32")
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)
caps = {json.loads(l)["id"]: json.loads(l)["caption"] for l in open("out/probe/manifest.jsonl", encoding="utf-8")}


def clean(c):
    c = re.sub(r"^\s*(this image displays|the image shows|in this image(,)? (we can see|there is|there are|i can see))\s*:?\s*", "", c, flags=re.I)
    return c[0].upper() + c[1:] if c else c


def first_sentence(c):
    m = re.match(r"(.+?[.!?])(\s|$)", c)
    return m.group(1) if m else c


@torch.no_grad()
def ivec(img):  # (B,3,H,W) in [0,1]
    img = F.interpolate(img, size=224, mode="bilinear", align_corners=False)
    return F.normalize(cm.encode_image((img - MEAN) / STD).float(), dim=-1)


@torch.no_grad()
def tvec(cs):
    return F.normalize(cm.encode_text(tok(cs).to(dev)).float(), dim=-1)


def retr(ie, te):
    sim = ie @ te.T
    d = sim.diag()
    return {"top1": round((sim.argmax(1) == torch.arange(len(te), device=dev)).float().mean().item(), 3),
            "gap": round((d.mean() - (sim.sum() - d.sum()) / (sim.numel() - len(te))).item(), 4)}


S, ids, psnr = [], [], []
for f in sorted(glob.glob("out/probe/v3/shard_*.pt")):
    sh = torch.load(f, weights_only=False)
    ok = sh["ok"]
    S.append(sh["strokes"][ok].float())
    ids += [i for i, o in zip(sh["ids"], ok) if o]
    psnr += [float(p) for p, o in zip(sh["psnr_full256"], ok) if o] if "psnr_full256" in sh else []
S = torch.cat(S).to(dev)
res = {}
for s in srcs:
    idx = [k for k, i in enumerate(ids) if i.startswith(s + ":")][:150]
    if len(idx) < 20:
        continue
    st = S[idx]
    with torch.no_grad():
        base = torch.cat([batched.render(st[j:j + 8, :165], 224, 224) for j in range(0, len(st), 8)]).view(-1, 3, 224, 224)
        b256 = torch.cat([batched.render(st[j:j + 8, :165], 256, 256) for j in range(0, len(st), 8)])
        full = torch.cat([dl.render_detail(st[j:j + 8, 165:], b256[j:j + 8], 14, 256) for j in range(0, len(st), 8)]).view(-1, 3, 256, 256)
    real = torch.stack([torch.from_numpy(__import__("numpy").asarray(Image.open(f"out/probe/{s}/img/{ids[k].split(':', 1)[1]}.jpg")
                                                                    .convert("RGB").resize((224, 224)))).permute(2, 0, 1).float() / 255
                        for k in idx]).to(dev)
    cfull = [clean(caps[ids[k]]) for k in idx]
    cshort = [first_sentence(c) for c in cfull]
    e = {"n": len(idx)}
    for cname, cs in (("full_caption", cfull), ("first_sentence", cshort)):
        te = tvec(cs)
        e[cname] = {"real": retr(ivec(real), te), "base165": retr(ivec(base), te), "full361": retr(ivec(full), te)}
    sidx = [k for k, i in enumerate(ids) if i.startswith(s + ":")]
    if psnr:
        e["psnr_full256_mean"] = round(sum(psnr[k] for k in sidx) / len(sidx), 2)
    res[s] = e
    print(s, json.dumps(e), flush=True)
json.dump(res, open("out/probe/strokes.json", "w"), indent=1)
print("PROBE_STROKES_DONE")
