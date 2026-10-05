"""Merge stage-1 outputs from several boxes into one training directory (laptop or a training box; CPU is fine).

  python merge_full.py out/final out/box1/full out/box2/full out/box3/full out/box4/full out/box4/wk
Each input dir has shard_*.pt, caps.json, clipvec.pt, items.json (clip_pass.py outputs). Output dir gets every shard
(renamed shard_<dir#><orig#>.pt so names never collide), one caps.json, one clipvec.pt, items.json, and a report.
Cross-box near-duplicates (photo/art sources only, CLIP image cos >= --dup) are removed from caps.json, so training skips them.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("dirs", nargs="+")
ap.add_argument("--dup", type=float, default=0.97)
a = ap.parse_args()
DEDUP_SRC = {"coco", "coco_obj", "coco_head", "wikiart", "cleveland", "met", "pd12m"}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
caps, items, ids, img, rf, rb = {}, {}, [], [], [], []
for d_i, d in enumerate(map(Path, a.dirs)):
    for f in sorted(d.glob("shard_*.pt")):
        shutil.copy(f, out / f"shard_{d_i}{f.stem.split('_')[1]}.pt")
    c = json.load(open(d / "caps.json"))
    caps.update(c)
    items.update({r["id"]: r for r in json.load(open(d / "items.json")) if r["id"] in c})
    cv = torch.load(d / "clipvec.pt")
    ids += cv["ids"]
    img.append(cv["img"]); rf.append(cv["render_full"]); rb.append(cv["render_base"])
    print(d, "shards", len(list(d.glob("shard_*.pt"))), "kept items", len(c), flush=True)
img, rf, rb = torch.cat(img), torch.cat(rf), torch.cat(rb)
src = [items[i]["source"] if i in items else "?" for i in ids]
sel = [n for n, i in enumerate(ids) if i in caps and src[n] in DEDUP_SRC]
X = img[sel].float()
dropped = 0
for s0 in range(0, len(sel), 4096):
    sim = X[s0:s0 + 4096] @ X.T
    for r in range(sim.shape[0]):
        n = s0 + r
        i = ids[sel[n]]
        if i in caps and len((sim[r, :n] >= a.dup).nonzero()):
            caps.pop(i)
            dropped += 1
json.dump(caps, open(out / "caps.json", "w"))
json.dump(list(items.values()), open(out / "items.json", "w"))
torch.save({"ids": ids, "img": img, "render_full": rf, "render_base": rb}, out / "clipvec.pt")
by = {}
for i in caps:
    s = items[i]["source"] if i in items else "?"
    by[s] = by.get(s, 0) + 1
rep = {"items_trainable": len(caps), "cross_box_duplicates_dropped": dropped, "by_source": by}
json.dump(rep, open(out / "merge_report.json", "w"), indent=1)
print(json.dumps(rep), flush=True)
