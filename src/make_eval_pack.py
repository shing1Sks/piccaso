"""Build a small, self-contained evaluation pack from the local dataset (so the final evaluation can run anywhere).

  python make_eval_pack.py --n 200 --out out/eval_pack.pt
Picks, with the exact training split (seed 0, 5% held out):
  seen   : n random training items      unseen : the first n held-out items
and n DOCCI images (long human captions, never trained on). Stores strokes, keep bits, first caption, OpenAI-CLIP vector
of the real picture and the source URL (PixelProse), or the 256 px image itself (DOCCI).
"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

import strokegen as sg

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="out/ppfinal_local/ppfinal")
ap.add_argument("--docci", default="out/probe/docci")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default="out/eval_pack.pt")
a = ap.parse_args()

torch.manual_seed(0)
S, keep, labels, ids, anchors = sg.load_data(a.data, "v2", 361)
caps = json.load(open(f"{a.data}/caps.json", encoding="utf-8"))
sel = [j for j, i in enumerate(ids) if i in caps]
S, keep, ids = S[sel], keep[sel], [ids[j] for j in sel]
perm = torch.randperm(len(ids))  # identical to strokegen.py / diag_eval.py
nv = max(8, len(ids) // 20)
held, train = perm[:nv].tolist(), perm[nv:].tolist()
cv = torch.load(f"{a.data}/clipvec.pt", weights_only=False)
cv_at = {i: k for k, i in enumerate(cv["ids"])}
urls = {r["id"]: r.get("url") for r in json.load(open(f"{a.data}/items.json"))}
rng = np.random.RandomState(1)
pick = {"seen": [train[k] for k in rng.choice(len(train), a.n, replace=False)], "unseen": held[:a.n]}
pack = {"n_train": len(train), "n_heldout": len(held)}
for name, items in pick.items():
    pack[name] = {"ids": [ids[i] for i in items], "strokes": S[items].half(), "captions": [caps[ids[i]][0] for i in items],
                  "img_oai": torch.stack([cv["img_oai"][cv_at[ids[i]]] for i in items]), "urls": [urls.get(ids[i]) for i in items]}
rows = [json.loads(l) for l in open(f"{a.docci}/rows.jsonl", encoding="utf-8")][:a.n]
pack["docci"] = {"ids": [r["id"] for r in rows], "captions": [r["caption"] for r in rows],
                 "images": torch.stack([torch.from_numpy(np.asarray(Image.open(f"{a.docci}/img/{r['id']}.jpg").convert("RGB").resize((256, 256))))
                                        for r in rows])}
torch.save(pack, a.out)
print({k: (len(v["ids"]) if isinstance(v, dict) else v) for k, v in pack.items()}, "PACK_DONE")
