"""Data probe (research/15): pull a sample of a rich-caption dataset to the laptop, cached like extract_v3 (384 px jpg).

  python probe_fetch.py docci  --n 1500     -> out/probe/docci/{img/*.jpg, rows.jsonl}
  python probe_fetch.py oi_ln  --n 1500     (Open Images validation + Localized Narratives captions)
  python probe_fetch.py pp     --n 1500 --parquet out/probe/pp/cc12m_00.parquet   (PixelProse, URLs -> download)
rows.jsonl: {"id", "caption", "source", ...extra}; summary.json: tried / ok / failed and timings.
"""
import argparse
import io
import json
import random
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src", choices=["docci", "oi_ln", "pp"])
ap.add_argument("--n", type=int, default=1500)
ap.add_argument("--parquet", default=None)
ap.add_argument("--min-aesthetic", type=float, default=5.0)
a = ap.parse_args()
out = Path("out/probe") / a.src
(out / "img").mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research data probe)"}
SIDE = 384


def save(im, key):
    im = im.convert("RGB")
    w, h = im.size
    s = min(w, h)  # centre square crop, same framing as the training cache
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((SIDE, SIDE), Image.LANCZOS)
    im.save(out / "img" / f"{key}.jpg", quality=92)


def fetch(url, key):
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code != 200 or len(r.content) < 2000:
            return f"http {r.status_code}"
        im = Image.open(io.BytesIO(r.content))
        if min(im.size) < 128:
            return "too small"
        save(im, key)
        return "ok"
    except Exception as e:
        return type(e).__name__


t0 = time.time()
rows, stat = [], {}
if a.src == "docci":
    desc = {}
    r = requests.get("https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines", timeout=60)
    for l in r.text.splitlines():
        d = json.loads(l)
        desc[d["example_id"]] = d
    r = requests.get("https://storage.googleapis.com/docci/data/docci_images.tar.gz", stream=True, timeout=60)
    with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:  # stream; stop after n images
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".jpg"):
                continue
            key = Path(m.name).stem
            if key not in desc or not desc[key]["example_id"].startswith(("train", "test", "qual")):
                continue
            save(Image.open(io.BytesIO(tf.extractfile(m).read())), key)
            rows.append({"id": key, "caption": desc[key]["description"], "source": "docci", "split": desc[key]["split"]})
            if len(rows) >= a.n:
                break
    r.close()
    stat = {"tried": len(rows), "ok": len(rows)}
elif a.src == "oi_ln":
    r = requests.get("https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_localized_narratives.jsonl",
                     stream=True, timeout=60)
    cand, seen = [], set()
    for line in r.iter_lines():
        d = json.loads(line)
        if d["image_id"] in seen:
            continue
        seen.add(d["image_id"])
        cand.append({"id": d["image_id"], "caption": d["caption"], "source": "oi_ln"})
        if len(cand) >= int(a.n * 1.15):
            break
    r.close()
    with ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda c: fetch(f"https://open-images-dataset.s3.amazonaws.com/validation/{c['id']}.jpg", c["id"]), cand))
    rows = [c for c, s in zip(cand, res) if s == "ok"][:a.n]
    stat = {"tried": len(cand), "ok": sum(s == "ok" for s in res), "fail_reasons": {s: res.count(s) for s in set(res) if s != "ok"}}
else:
    import pyarrow.parquet as pq

    t = pq.read_table(a.parquet)
    print("columns:", t.column_names, "rows:", t.num_rows, flush=True)
    df = t.to_pandas()
    cols = set(df.columns)
    keep = df
    if "aesthetic_score" in cols:
        keep = keep[keep["aesthetic_score"] >= a.min_aesthetic]
    if "watermark_class_id" in cols:
        keep = keep[keep["watermark_class_id"] == 1]  # 0 watermark, 1 clean, 2 text overlay
    print(f"after filters: {len(keep)} of {len(df)}", flush=True)
    cand = keep.sample(n=min(len(keep), int(a.n * 1.6)), random_state=0)
    cand = [{"id": f"pp_{r['uid'] if 'uid' in cols else i}", "url": r["url"], "caption": r["vlm_caption"], "source": "pixelprose",
             "original_caption": r.get("original_caption"), "aesthetic": float(r["aesthetic_score"]) if "aesthetic_score" in cols else None}
            for i, r in cand.iterrows()]
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(lambda c: fetch(c["url"], c["id"]), cand))
    rows = [c for c, s in zip(cand, res) if s == "ok"][:a.n]
    stat = {"tried": len(cand), "ok": sum(s == "ok" for s in res), "fail_reasons": {s: res.count(s) for s in set(res) if s != "ok"},
            "rows_in_parquet": len(df), "after_filters": len(keep)}
stat["seconds"] = round(time.time() - t0)
with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
    for r_ in rows:
        f.write(json.dumps(r_) + "\n")
json.dump(stat, open(out / "summary.json", "w"), indent=1)
print(json.dumps(stat), "PROBE_FETCH_DONE", flush=True)
