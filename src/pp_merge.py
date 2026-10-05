"""Merge PixelProse worker outputs into one training directory + the "survives strokes" filter (research/17 stage 1).

  python pp_merge.py out/ppfinal out/pp/w0 out/pp/w1 out/pp/w2 out/pp/w3 --drop 0.25
Each worker dir: shard_*.pt (extract_v3), items.json (rows with texts), score.pt (pp_score). Output: shard_<w><k>.pt
(symlinked), caps.json {id: [captions]} for KEPT items only (strokegen trains only on ids in caps.json), clipvec.pt
{ids, img (Long-CLIP), img_oai (OpenAI CLIP), survive}, items.json, merge_report.json, survive_sheet.png (kept vs dropped).
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from extract_v3 import safe_name

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("dirs", nargs="+")
ap.add_argument("--drop", type=float, default=0.25, help="drop this fraction with the lowest survives-strokes score")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ids, img, img_oai, surv, agree, items = [], [], [], [], [], {}
psnr = []
for w, d in enumerate(map(Path, a.dirs)):
    for f in sorted(d.glob("shard_*.pt")):
        dst = out / f"shard_{w}{f.stem.split('_')[1]}.pt"
        if not dst.exists():
            try:
                os.symlink(f.resolve(), dst)
            except OSError:
                import shutil

                shutil.copy(f, dst)
    sc = torch.load(d / "score.pt", weights_only=False)
    ids += sc["ids"]
    img.append(sc["img"]); img_oai.append(sc["img_oai"]); surv.append(sc["survive"]); agree.append(sc["agree_real"])
    items.update({r["id"]: {**r, "dir": str(d)} for r in json.load(open(d / "items.json", encoding="utf-8"))})
    for l in open(d / "log.jsonl"):
        psnr.append(json.loads(l)["psnr_full256"])
img, img_oai, surv, agree = torch.cat(img), torch.cat(img_oai), torch.cat(surv), torch.cat(agree)
thr = surv.quantile(a.drop).item()
kept = surv >= thr
caps = {i: items[i]["texts"] for i, k in zip(ids, kept.tolist()) if k}
json.dump(caps, open(out / "caps.json", "w", encoding="utf-8"))
torch.save({"ids": ids, "img": img, "img_oai": img_oai, "survive": surv, "agree_real": agree}, out / "clipvec.pt")
json.dump([{"id": i, "source": "pixelprose", "url": items[i].get("url")} for i in ids], open(out / "items.json", "w"))
rep = {"extracted": len(ids), "kept": len(caps), "survive_threshold": round(thr, 4), "survive_mean_kept": round(surv[kept].mean().item(), 4),
       "survive_mean_dropped": round(surv[~kept].mean().item(), 4), "agree_real_mean": round(agree.mean().item(), 4),
       "psnr_full256_mean_of_shards": round(float(np.mean(psnr)), 2), "shards": len(list(out.glob("shard_*.pt")))}
json.dump(rep, open(out / "merge_report.json", "w"), indent=1)
print(json.dumps(rep), flush=True)

# sheet: 12 kept (top) and 12 dropped (bottom) real pictures with captions, to check the filter by eye
random.seed(0)
order = surv.argsort()
pick = [int(x) for x in random.sample(order[int(len(order) * .5):].tolist(), 12)] + [int(x) for x in random.sample(order[:int(len(order) * a.drop)].tolist(), 12)]
T = 192
sheet = Image.new("RGB", (T * 6, (T + 30) * 4), "white")
dr = ImageDraw.Draw(sheet)
for k, n in enumerate(pick):
    i = ids[n]
    im = Image.open(Path(items[i]["dir"]) / "img" / f"{safe_name(i)}.jpg").resize((T, T))
    x, y = (k % 6) * T, (k // 6) * (T + 30)
    sheet.paste(im, (x, y))
    c = items[i]["texts"][0]
    dr.text((x + 2, y + T + 1), f"{'KEEP' if k < 12 else 'DROP'} {surv[n]:.3f} {c[:30]}", fill=(0, 128, 0) if k < 12 else (200, 0, 0))
    dr.text((x + 2, y + T + 14), c[30:72], fill=(0, 0, 0))
sheet.save(out / "survive_sheet.png")
print("MERGE_DONE", flush=True)
