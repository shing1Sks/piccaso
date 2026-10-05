"""Manifest for the full run (research/12). Runs on the laptop (metadata only; images are fetched by the extractor).

  python build_full_manifest.py out/full/manifest_web.jsonl            # everything except WikiArt
  python build_full_manifest.py out/full/m_test.jsonl --scale 0.01     # 1% dry run
WikiArt comes from parquet files on the GPU box (fetch_wikiart.py) and is appended there.
Row: {id, source, texts: [...], license, + one of url | svg_url | drawing | path, box?}.  'source' values:
quickdraw, twemoji, noto, openmoji, fluent, coco, coco_obj, coco_head, cleveland, met, pd12m (+ wikiart on the box).
"""
import argparse
import io
import json
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only", default=None, help="comma list of sections to (re)build; others are read from <out>.<section>.jsonl")
a = ap.parse_args()
N = {"quickdraw": 72, "coco": 15000, "coco_obj": 7000, "coco_head": 3000, "cleveland": 16000, "met": 0, "pd12m": 10000}
UA = {"User-Agent": "Mozilla/5.0 (stroke-research manifest builder)"}
rng = random.Random(a.seed)
CACHE = Path("out/full/cache")
CACHE.mkdir(parents=True, exist_ok=True)


def get(u, t=120, tries=5):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def cached(name, url, t=900):
    f = CACHE / name
    if not f.exists():
        f.write_bytes(get(url, t))
    return f.read_bytes()


def n_of(k):
    return max(1, int(N[k] * a.scale))


def clean(t):
    t = unicodedata.normalize("NFKC", t or "").replace("�", "-")
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    return re.sub(r"\b(\w+) \1\b", r"\1", t)  # "painting painting" -> "painting"


# ---------------------------------------------------------------- sections
def quickdraw():
    cats = [l.strip() for l in open("quickdraw_categories.txt") if l.strip()]
    per, rows = n_of("quickdraw"), []
    for c in cats:
        resp = urllib.request.urlopen(urllib.request.Request(
            f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{urllib.parse.quote(c)}.ndjson", headers=UA), timeout=120)
        got = 0
        for line in resp:  # streamed: only the head of each file is read
            r = json.loads(line)
            if r.get("recognized") and 2 <= len(r["drawing"]) <= 15:
                rows.append({"id": f"qd_{c}_{r['key_id']}", "source": "quickdraw", "drawing": r["drawing"], "label": c,
                             "texts": [t.format(c=c.lower()) for t in ("a drawing of a {c}", "a sketch of a {c}", "a doodle of a {c}",
                                                                      "a simple line drawing of a {c}", "a {c}")], "license": "CC-BY 4.0"})
                got += 1
                if got >= per:
                    break
        resp.close()
    return rows


JUNK = ("skin tone", "fitzpatrick", "regional indicator", "tag ", "variation selector", "zero width")


def emoji_name(cp_hex):
    try:
        nm = unicodedata.name(chr(int(cp_hex, 16))).lower()
    except (ValueError, KeyError):
        return None
    return None if any(j in nm for j in JUNK) else nm


def icon_texts(name, extra, style):
    t = [f"an emoji of {name}", f"a {name} emoji", f"{name}, {style} icon"]
    return t + [f"{x}, emoji" for x in extra[:2]]


def icons():
    rows = []
    tree = json.loads(get("https://api.github.com/repos/jdecked/twemoji/git/trees/main?recursive=1"))["tree"]
    for t in tree:
        p = t["path"]
        if p.startswith("assets/svg/") and p.endswith(".svg") and "-" not in p.split("/")[-1]:
            code = p.split("/")[-1][:-4]
            nm = emoji_name(code)
            if nm:
                rows.append({"id": f"twemoji_{code}", "source": "twemoji", "svg_url": f"https://raw.githubusercontent.com/jdecked/twemoji/main/{p}",
                             "label": nm, "texts": icon_texts(nm, [], "flat"), "license": "CC-BY 4.0"})
    tree = json.loads(get("https://api.github.com/repos/googlefonts/noto-emoji/git/trees/main?recursive=1"))["tree"]
    for t in tree:
        m = re.fullmatch(r"2D/svg/emoji_u([0-9a-f]+)\.svg", t["path"])
        if m and (nm := emoji_name(m.group(1))):
            rows.append({"id": f"noto_{m.group(1)}", "source": "noto", "svg_url": f"https://raw.githubusercontent.com/googlefonts/noto-emoji/main/{t['path']}",
                         "label": nm, "texts": icon_texts(nm, [], "glossy"), "license": "Apache-2.0"})
    for e in json.loads(get("https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/data/openmoji.json")):
        hx = e["hexcode"]
        if "-" in hx or e.get("skintone") or e.get("group") in ("extras-openmoji", "extras-unicode", "component", "flags")                 or any(j in e["annotation"] for j in JUNK):
            continue
        tags = [x.strip() for x in (e.get("tags") or "").split(",") if x.strip()]
        rows.append({"id": f"openmoji_{hx}", "source": "openmoji", "svg_url": f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/svg/{hx}.svg",
                     "label": e["annotation"], "texts": icon_texts(e["annotation"], tags, "outlined"), "license": "CC BY-SA 4.0"})
    tree = json.loads(get("https://api.github.com/repos/microsoft/fluentui-emoji/git/trees/main?recursive=1"))["tree"]
    for t in tree:
        m = re.fullmatch(r"assets/([^/]+)/Flat/([^/]+)_flat\.svg", t["path"])
        if m:
            nm = m.group(1).lower()
            rows.append({"id": f"fluent_{m.group(2)}", "source": "fluent", "label": nm,
                         "svg_url": "https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/" + urllib.parse.quote(t["path"]),
                         "texts": icon_texts(nm, [], "flat 3d-style"), "license": "MIT"})
    return rows


def coco():
    z = zipfile.ZipFile(io.BytesIO(cached("coco_ann.zip", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip")))
    caps = json.loads(z.read("annotations/captions_train2017.json"))
    by = {}
    for x in caps["annotations"]:
        by.setdefault(x["image_id"], []).append(clean(x["caption"]))
    inst = json.loads(z.read("annotations/instances_train2017.json"))
    cat = {c["id"]: c["name"] for c in inst["categories"]}
    img = {i["id"]: i for i in inst["images"]}
    url = lambda i: f"http://images.cocodataset.org/train2017/{i:012d}.jpg"
    ids = sorted(by)
    rng.shuffle(ids)
    full_ids = ids[:n_of("coco")]
    rows = [{"id": f"coco_{i}", "source": "coco", "url": url(i), "texts": by[i][:5], "license": "COCO (research)"} for i in full_ids]
    # object crops: the largest non-crowd object of an image not used above, big enough to survive 384 px
    used = set(full_ids)
    best = {}
    for an in inst["annotations"]:
        if an["iscrowd"] or an["image_id"] in used:
            continue
        x, y, w, h = an["bbox"]
        if min(w, h) < 96 or an["area"] < 0.08 * img[an["image_id"]]["width"] * img[an["image_id"]]["height"]:
            continue
        if an["category_id"] == 1:
            continue  # people handled by head crops / full images
        if an["image_id"] not in best or an["area"] > best[an["image_id"]]["area"]:
            best[an["image_id"]] = an
    obj = list(best.values())
    rng.shuffle(obj)
    for an in obj[:n_of("coco_obj")]:
        x, y, w, h = an["bbox"]
        c = cat[an["category_id"]]
        rows.append({"id": f"cocoobj_{an['id']}", "source": "coco_obj", "url": url(an["image_id"]), "box": [x, y, x + w, y + h], "label": c,
                     "texts": [f"a photo of a {c}", f"a {c}"], "license": "COCO (research)"})
        used.add(an["image_id"])
    # head / face crops from person keypoints (nose, eyes, ears)
    kp = json.loads(z.read("annotations/person_keypoints_train2017.json"))
    heads = []
    for an in kp["annotations"]:
        if an["iscrowd"] or an["image_id"] in used:
            continue
        k = an["keypoints"]
        pts = [(k[3 * j], k[3 * j + 1]) for j in range(5) if k[3 * j + 2] > 0]
        if len(pts) < 4:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        span = max(max(xs) - min(xs), 1)
        side = 1.9 * span  # ear-to-ear (or eye span) -> the head with a little hair and neck
        if side < 110:
            continue
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2 + 0.15 * side
        heads.append({"id": f"cocohead_{an['id']}", "source": "coco_head", "url": url(an["image_id"]),
                      "box": [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], "margin": 0.05, "label": "portrait",
                      "texts": ["a portrait photo of a person", "a close-up of a person's face"], "license": "COCO (research)"})
        used.add(an["image_id"])
    rng.shuffle(heads)
    return rows + heads[:n_of("coco_head")]


def cleveland():
    n, rows, skip, seen = n_of("cleveland"), [], 0, set()
    while len(rows) < n and skip < 80000:
        d = json.loads(get(f"https://openaccess-api.clevelandart.org/api/artworks/?limit=1000&skip={skip}&has_image=1&cc0=1"
                           f"&fields=id,title,type,creators,images,culture,creation_date"))
        if not d["data"]:
            break
        skip += 1000
        for x in d["data"]:
            if x["type"] not in ("Painting", "Drawing", "Print") or x["id"] in seen:
                continue
            if any(b in (x["title"] or "") for b in ("Text", "Manuscript", "Folio", "Page from", "Leaf from")):
                continue
            im = (x.get("images") or {}).get("web") or {}
            if not im.get("url"):
                continue
            seen.add(x["id"])
            who = ", ".join(clean(c["description"]).split("(")[0].strip() for c in (x.get("creators") or [])[:1])
            t = clean(f"{x['title']}, {x['type'].lower()}" + (f" by {who}" if who else ""))
            rows.append({"id": f"cleveland_{x['id']}", "source": "cleveland", "url": im["url"], "texts": [t], "license": "CC0"})
        time.sleep(0.5)
    rng.shuffle(rows)
    return rows[:n]


def met():
    import concurrent.futures as cf

    base = "https://collectionapi.metmuseum.org/public/collection/v1"
    ids = []
    for dept in (11, 9, 21):  # European Paintings, Drawings and Prints, Modern and Contemporary Art
        ids += json.loads(get(f"{base}/objects?departmentIds={dept}"))["objectIDs"]
    rng.shuffle(ids)
    n = n_of("met")

    def one(oid):  # the API allows 80 requests/s; 8 threads with small pauses stay far below that
        time.sleep(0.1)
        try:
            o = json.loads(get(f"{base}/objects/{oid}", 60, 2))
        except Exception:
            return None
        if not (o.get("isPublicDomain") and o.get("primaryImageSmall")):
            return None
        who = o.get("artistDisplayName") or ""
        t = clean(f"{o['title']}, {(o.get('objectName') or 'artwork').lower()}" + (f" by {who}" if who else ""))
        return {"id": f"met_{oid}", "source": "met", "url": o["primaryImageSmall"], "texts": [t], "license": "CC0"}

    rows = []
    with cf.ThreadPoolExecutor(8) as ex:
        for k in range(0, len(ids), 400):
            rows += [r for r in ex.map(one, ids[k:k + 400]) if r]
            print("  met", len(rows), flush=True)
            if len(rows) >= n:
                break
    return rows[:n]


def pd12m():
    n, rows, start = n_of("pd12m"), [], rng.randrange(0, 2_000_000)
    while len(rows) < n:
        try:
            meta = json.loads(get(f"https://datasets-server.huggingface.co/rows?dataset=Spawning/PD12M&config=default&split=train&offset={start}&length=100"))
        except Exception:
            time.sleep(10)
            start += 7_000
            continue
        for r in meta["rows"]:
            row = r["row"]
            if row["mime_type"] != "image/jpeg" or int(row["width"]) < 384 or int(row["height"]) < 384:
                continue
            cap = clean(re.sub(r"^(The|This) (image|picture|photo|photograph) (shows|depicts|features|is)\s*", "", row["caption"]))
            cap = cap[:1].upper() + cap[1:]
            first = re.split(r"(?<=[.!?])\s", cap)[0]
            rows.append({"id": f"pd12m_{row['id']}", "source": "pd12m", "url": row["url"], "texts": [first, cap[:300]], "license": "CC0 (PD12M)"})
            if len(rows) >= n:
                break
        start += 11_000  # spread across the 12M rows, not one collection
        time.sleep(1.0)
    return rows


SECTIONS = {"quickdraw": quickdraw, "icons": icons, "coco": coco, "cleveland": cleveland, "pd12m": pd12m}  # Met API blocks bulk use (stage 0)
only = set(a.only.split(",")) if a.only else set(SECTIONS)
allrows = []
for name, fn in SECTIONS.items():
    part = Path(f"{a.out}.{name}.jsonl")
    if name in only or not part.exists():
        t0 = time.time()
        try:
            rows = fn()
        except Exception as e:
            print(f"{name} FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        part.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"{name}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    rows = [json.loads(l) for l in open(part) if l.strip()]
    allrows += rows
seen, uniq = set(), []
for r in allrows:
    if r["id"] not in seen:
        seen.add(r["id"])
        uniq.append(r)
Path(a.out).write_text("\n".join(json.dumps(r) for r in uniq) + "\n")
by = {}
for r in uniq:
    by[r["source"]] = by.get(r["source"], 0) + 1
print("TOTAL", len(uniq), by)
