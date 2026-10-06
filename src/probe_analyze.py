"""Data probe analysis (research/15), CPU is fine: are these captions/images a good fit?

  python probe_analyze.py pp oi_ln docci      -> out/probe/analysis.json + out/probe/sheet_<src>.png
Per source: caption length (words, CLIP tokens, share cut at 77 tokens), CLIP image-caption similarity and within-source
retrieval top-1 / gap (n=150, same protocol as diag_eval's 'real_picture' row) for the full caption (truncated by CLIP)
and for a short version (first sentence), concept counts, and a contact sheet.
"""
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import open_clip

srcs = sys.argv[1:]
cm, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
cm.eval()
tok = open_clip.get_tokenizer("ViT-B-32")
WORDS = ["dog", "cat", "bird", "horse", "car", "bus", "food", "flower", "tree", "house|building", "man|woman|person|people", "sky",
         "water|sea|lake|river", "mountain", "street|road", "room|kitchen|bedroom", "painting|drawing|illustration", "text|logo|words"]


def clean(c):
    c = re.sub(r"^\s*(this image displays|the image shows|in this image(,)? (we can see|there is|there are|i can see))\s*:?\s*", "",
               c, flags=re.I)
    return c[0].upper() + c[1:] if c else c


def first_sentence(c):
    m = re.match(r"(.+?[.!?])(\s|$)", c)
    return m.group(1) if m else c


@torch.no_grad()
def enc_txt(cs):
    return torch.cat([F.normalize(cm.encode_text(tok(cs[i:i + 64])).float(), dim=-1) for i in range(0, len(cs), 64)])


@torch.no_grad()
def enc_img(paths):
    out = []
    for i in range(0, len(paths), 32):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 32]])
        out.append(F.normalize(cm.encode_image(x).float(), dim=-1))
    return torch.cat(out)


def retr(ie, te):
    sim = ie @ te.T
    d = sim.diag()
    return {"top1": round((sim.argmax(1) == torch.arange(len(te))).float().mean().item(), 3),
            "gap": round((d.mean() - (sim.sum() - d.sum()) / (sim.numel() - len(te))).item(), 4), "mean_sim": round(d.mean().item(), 4)}


res = {}
random.seed(0)
for s in srcs:
    rows = [json.loads(l) for l in open(f"out/probe/{s}/rows.jsonl", encoding="utf-8")]
    rows = [r for r in rows if (Path(f"out/probe/{s}/img") / f"{r['id']}.jpg").exists()]
    random.shuffle(rows)
    caps = [clean(r["caption"]) for r in rows]
    ntok = [int((tok([c])[0] > 0).sum()) for c in caps]
    words = [len(c.split()) for c in caps]
    sub = rows[:150]
    paths = [f"out/probe/{s}/img/{r['id']}.jpg" for r in sub]
    ie = enc_img(paths)
    full = [clean(r["caption"]) for r in sub]
    short = [first_sentence(c) for c in full]
    e = {"n_rows": len(rows), "words_median": sorted(words)[len(words) // 2], "words_p90": sorted(words)[int(len(words) * .9)],
         "share_over_77_tokens": round(sum(t >= 77 for t in ntok) / len(ntok), 3),
         "retrieval_full_caption": retr(ie, enc_txt(full)), "retrieval_first_sentence": retr(ie, enc_txt(short)),
         "concepts_per_1000": {w: round(1000 * sum(bool(re.search(r"\b(" + w + r")s?\b", c.lower())) for c in caps) / len(caps)) for w in WORDS},
         "examples": [full[i][:400] for i in range(4)]}
    res[s] = e
    print(s, json.dumps({k: v for k, v in e.items() if k != "examples"}), flush=True)
    T, cols = 192, 6  # contact sheet: 4 rows x 6 images with the first 90 characters of the caption under each
    sheet = Image.new("RGB", (T * cols, (T + 30) * 4), "white")
    dr = ImageDraw.Draw(sheet)
    for k in range(24):
        r = rows[k]
        im = Image.open(f"out/probe/{s}/img/{r['id']}.jpg").resize((T, T))
        x, y = (k % cols) * T, (k // cols) * (T + 30)
        sheet.paste(im, (x, y))
        c = clean(r["caption"])
        dr.text((x + 2, y + T + 1), c[:45], fill=(0, 0, 0))
        dr.text((x + 2, y + T + 14), c[45:90], fill=(0, 0, 0))
    sheet.save(f"out/probe/sheet_{s}.png")
json.dump(res, open("out/probe/analysis.json", "w"), indent=1)
print("ANALYZE_DONE")
