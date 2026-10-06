"""WikiArt from the huggan/wikiart parquet files (no viewer API, no rate limit).  Runs on the GPU box.

  python fetch_wikiart.py out/full/wikiart.jsonl --files 0,1,2,3,4,5,17,22,27,32,37,42,47,52 [--per-file 1132]
Writes images to out/full/wikiart_src/<row>.jpg (longest side 512) and manifest rows {id, source, path, texts, license}.
Files 0-11 are mostly named Impressionist artists, 17-58 mixed styles, 59-71 have no labels: pick a spread.
"""
import argparse
import io
import json
import os
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--files", default="0,1,2,3,4,5,17,22,27,32,37,42,47,52")
ap.add_argument("--per-file", type=int, default=100000)
a = ap.parse_args()
R = "https://huggingface.co/datasets/huggan/wikiart/resolve/main"
F = json.load(urllib.request.urlopen(f"{R}/dataset_infos.json"))["huggan--wikiart"]["features"]
A, G, S = (F[k]["names"] for k in ("artist", "genre", "style"))
src = Path("out/full/wikiart_src")
src.mkdir(parents=True, exist_ok=True)
pretty = lambda x: x.replace("_", " ").replace("-", " ").strip()
rows = []
for fi in map(int, a.files.split(",")):
    local = f"/root/wk_{fi}.parquet"
    if not os.path.exists(local):
        urllib.request.urlretrieve(f"{R}/data/train-{fi:05d}-of-00072.parquet", local)
    k = 0
    for b in pq.ParquetFile(local).iter_batches(batch_size=64):
        for j, r in enumerate(b.to_pylist()):
            if k >= a.per_file:
                break
            rid = f"wikiart_{fi}_{k}"
            k += 1
            try:
                im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            except Exception:
                continue
            im.thumbnail((512, 512))
            p = src / f"{rid}.jpg"
            im.save(p, quality=92)
            artist, genre, style = A[r["artist"]], G[r["genre"]], S[r["style"]]
            g = "" if genre.startswith("Unknown") else pretty(genre).replace(" painting", "")
            who = "" if artist.startswith("Unknown") else pretty(artist).title()
            t1 = f"{g + ' ' if g else ''}painting, {pretty(style)}" + (f", by {who}" if who else "")
            t2 = f"a {pretty(style)} {g or 'painting'}"
            rows.append({"id": rid, "source": "wikiart", "path": str(p), "texts": [t1, t2], "label": t1, "license": "WikiArt (research)"})
    os.remove(local)  # ~500 MB each; keep the disk small
    print(fi, "->", len(rows), flush=True)
Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print("WIKIART_DONE", len(rows))
