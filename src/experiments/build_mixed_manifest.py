"""Build the mixed-data manifest (URL based; nothing big is downloaded here).  Run on the laptop, upload the jsonl.

python build_mixed_manifest.py out/mixed/manifest.jsonl --coco 1500 --wikiart 1000 --cleveland 600 --met 600 --pd12m 1000
Row: {id, source, url, texts: [human/metadata texts...], license}.  QuickDraw + emoji are produced by extract_v2 itself.
Filters: Cleveland keeps Painting/Drawing/Print/Textile-free types and drops manuscripts/text pages; Met uses European Paintings
(dept 11) + Drawings and Prints (dept 9); PD12M rows come with a synthetic caption.
"""
import argparse
import io
import json
import random
import time
import urllib.request
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--coco", type=int, default=1500)
ap.add_argument("--wikiart", type=int, default=1000)
ap.add_argument("--cleveland", type=int, default=600)
ap.add_argument("--met", type=int, default=600)
ap.add_argument("--pd12m", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
UA = {"User-Agent": "Mozilla/5.0 (stroke-research manifest builder)"}
rng = random.Random(a.seed)
rows = []


def get(u, t=120):
    for k in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t).read()
        except Exception:
            if k == 3:
                raise
            time.sleep(2 * (k + 1))


def coco(n):
    z = zipfile.ZipFile(io.BytesIO(get("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 900)))
    caps = json.loads(z.read("annotations/captions_train2017.json"))
    by = {}
    for x in caps["annotations"]:
        by.setdefault(x["image_id"], []).append(x["caption"].strip())
    ids = sorted(by)
    rng.shuffle(ids)
    for i in ids[:n]:
        rows.append({"id": f"coco_{i}", "source": "coco", "url": f"http://images.cocodataset.org/train2017/{i:012d}.jpg", "texts": by[i][:5],
                     "license": "COCO/Flickr (research)"})


def wikiart(n):
    off = list(range(0, 11000, 1))
    rng.shuffle(off)
    got = 0
    for o in off:
        if got >= n:
            break
        try:  # 100 rows per request keeps the request count low
            if got % 100 == 0 or got == 0:
                base = (o // 100) * 100
                meta = json.loads(get(f"https://datasets-server.huggingface.co/rows?dataset=huggan/wikiart&config=default&split=train&offset={base}&length=100"))
                names = {f["name"]: f["type"].get("names") for f in meta["features"] if isinstance(f["type"], dict)}
                batch = meta["rows"]
            for r in batch:
                row = r["row"]
                pick = lambda k: names[k][row[k]] if names.get(k) else row.get(k)
                txt = f"{pick('genre')} painting, {pick('style')}, by {pick('artist')}".replace("_", " ")
                rows.append({"id": f"wikiart_{r['row_idx']}", "source": "wikiart", "url": row["image"]["src"], "texts": [txt], "license": "WikiArt (research)"})
                got += 1
                if got >= n:
                    break
        except Exception as e:
            print("wikiart skip", type(e).__name__)
            time.sleep(1)


def cleveland(n):
    bad = ("Text", "Manuscript", "Book", "Textile", "Coin", "Medal", "Costume", "Photograph")
    skip = rng.randrange(0, 20000)
    d = json.loads(get(f"https://openaccess-api.clevelandart.org/api/artworks/?limit={n * 5}&has_image=1&cc0=1&skip={skip}&fields=id,title,type,creators,images,department"))
    got = 0
    for x in d["data"]:
        if x["type"] not in ("Painting", "Drawing", "Print", "Sculpture") or any(b in (x["title"] or "") for b in bad):
            continue
        who = ", ".join(c["description"] for c in x.get("creators", [])[:1])
        rows.append({"id": f"cleveland_{x['id']}", "source": "cleveland", "url": x["images"]["web"]["url"], "texts": [f"{x['title']} ({x['type'].lower()}) {who}".strip()],
                     "license": "Cleveland Museum of Art CC0"})
        got += 1
        if got >= n:
            break


def met(n):
    base = "https://collectionapi.metmuseum.org/public/collection/v1"
    ids = []
    for dept in (11, 9):
        ids += json.loads(get(f"{base}/objects?departmentIds={dept}"))["objectIDs"]
    rng.shuffle(ids)
    got = 0
    for oid in ids:
        if got >= n:
            break
        time.sleep(0.4)
        try:
            o = json.loads(get(f"{base}/objects/{oid}"))
        except Exception:
            continue
        if not (o.get("isPublicDomain") and o.get("primaryImageSmall")):
            continue
        rows.append({"id": f"met_{oid}", "source": "met", "url": o["primaryImageSmall"],
                     "texts": [f"{o['title']} ({o.get('objectName') or 'artwork'}) {o.get('artistDisplayName') or ''}".strip()], "license": "Met Open Access CC0"})
        got += 1


def pd12m(n):
    start = rng.randrange(0, 5_000_000)
    got = 0
    while got < n:
        meta = json.loads(get(f"https://datasets-server.huggingface.co/rows?dataset=Spawning/PD12M&config=default&split=train&offset={start}&length=100"))
        for r in meta["rows"]:
            row = r["row"]
            if row["mime_type"] != "image/jpeg" or int(row["width"]) < 256 or int(row["height"]) < 256:
                continue
            rows.append({"id": f"pd12m_{row['id']}", "source": "pd12m", "url": row["url"], "texts": [row["caption"]], "license": "PD12M CC0 (" + str(row.get("source")) + ")"})
            got += 1
            if got >= n:
                break
        start += 100_000  # jump so the sample is spread out, not one museum


for name, n in (("coco", a.coco), ("wikiart", a.wikiart), ("cleveland", a.cleveland), ("met", a.met), ("pd12m", a.pd12m)):
    n0 = len(rows)
    if n:
        try:
            globals()[name](n)
        except Exception as e:
            print(f"{name} FAILED after {len(rows) - n0}: {type(e).__name__}: {str(e)[:150]}")
    print(f"{name}: {len(rows) - n0} rows", flush=True)
rng.shuffle(rows)
open(a.out, "w").write("\n".join(json.dumps(r) for r in rows) + "\n")
print("total", len(rows), "->", a.out)
