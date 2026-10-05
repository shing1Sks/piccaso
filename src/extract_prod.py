"""Production Stage-0 extractor: the 'fast' recipe, sharded and resumable. Meant to run on a GPU pod.

Items come from raw sources (QuickDraw ndjson, Twemoji SVG, or any URL manifest) and are rendered on the pod, so
nothing large is uploaded from the laptop. Items are listed in a fixed shuffled order; shard k = items[k*S:(k+1)*S],
so any subset of shards is a mixed sample, parallel pods can take disjoint shard ranges, and a rerun skips finished shards.

  python extract_prod.py --out out/s0 --classes cat,apple --per-class 300 --emoji-limit 800 --shard-size 400
  python extract_prod.py --out out/s0 --shards 0-3          # only these shards (for parallel pods)
Output: <out>/items.json, <out>/shard_00000.pt ..., <out>/log.jsonl
Shard (fp16 strokes, 11 columns like batched.py, alpha is always 1): strokes (N,160,11), stage_psnr (N,3), ids, labels, sources.
"""
import argparse
import concurrent.futures as cf
import io
import json
import random
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import batched

SIZE = 128
UA = {"User-Agent": "Mozilla/5.0 (stroke-research)"}
CLEAN = {"width": (0.008, 0.6), "pos": (0.0, 1.0), "alpha": (1.0, 1.0)}
RECIPES = {"fast": dict(stages=[(16, 0.30), (48, 0.15), (96, 0.07)], steps=[150, 150, 100], segs=4),
           "smoke": dict(stages=[(4, 0.30), (6, 0.15), (10, 0.07)], steps=[3, 3, 3], segs=4)}  # plumbing test only
DEFAULT_CLASSES = ("cat,apple,car,house,flower,sun,tree,fish,bicycle,airplane,bird,cup,"
                   "clock,face,umbrella,guitar").split(",")


def get(url, tries=4, timeout=60):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1 + 2 * k)


def center_square(im, size=SIZE):
    im = im.convert("RGB")
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)


# ---------- item lists (cheap: no image work) ----------
def quickdraw_items(classes, per_class, skip, cache):
    items = []
    for cls in classes:
        f = cache / f"qd_{cls}_{skip}_{per_class}.json"
        if f.exists():
            recs = json.loads(f.read_text())
        else:
            recs, seen = [], 0
            resp = urllib.request.urlopen(urllib.request.Request(
                f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{urllib.parse.quote(cls)}.ndjson", headers=UA), timeout=60)
            for line in resp:  # streamed; only the head of the file is read
                r = json.loads(line)
                if not r.get("recognized") or not 2 <= len(r["drawing"]) <= 15:
                    continue
                seen += 1
                if seen <= skip:
                    continue
                recs.append({"key": r["key_id"], "drawing": r["drawing"]})
                if len(recs) >= per_class:
                    break
            resp.close()
            f.write_text(json.dumps(recs))
        items += [{"id": f"qd_{cls}_{r['key']}", "label": cls, "source": "quickdraw", "drawing": r["drawing"]} for r in recs]
    return items


def emoji_items(limit, cache):
    f = cache / "twemoji_svgs.json"
    if f.exists():
        codes = json.loads(f.read_text())
    else:
        tree = json.loads(get("https://api.github.com/repos/jdecked/twemoji/git/trees/main?recursive=1"))["tree"]
        codes = sorted(t["path"].split("/")[-1][:-4] for t in tree
                       if t["path"].startswith("assets/svg/") and t["path"].endswith(".svg"))
        f.write_text(json.dumps(codes))
    items = []
    for code in codes:
        if "-" in code:  # keep single-codepoint emoji (no skin-tone / ZWJ sequences)
            continue
        try:
            name = unicodedata.name(chr(int(code, 16))).lower()
        except (ValueError, KeyError):
            continue
        items.append({"id": f"emoji_{code}", "label": f"emoji {name}", "source": "emoji", "code": code})
    return items[:limit]


# ---------- per-item image loading (runs in a thread pool) ----------
def fresh_url(it):
    """WikiArt links from the HF viewer are signed and expire after ~1 h: fetch a fresh one at load time."""
    if it.get("source") == "wikiart" and "row_idx" not in it:
        idx = int(it["id"].split("_")[1])
        meta = json.loads(get(f"https://datasets-server.huggingface.co/rows?dataset=huggan/wikiart&config=default&split=train&offset={idx}&length=1"))
        return meta["rows"][0]["row"]["image"]["src"]
    return it["url"]


def svg_to_image(svg):
    import resvg_py

    rgba = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, width=512, height=512)))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def crop_box(im, box, margin=0.15):
    """box = [x0, y0, x1, y1] in pixels: a square around it (plus margin), shifted to stay inside the image."""
    x0, y0, x1, y1 = box
    side = min(max(x1 - x0, y1 - y0) * (1 + 2 * margin), im.width, im.height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    l = min(max(cx - side / 2, 0), im.width - side)
    t = min(max(cy - side / 2, 0), im.height - side)
    return im.crop((int(l), int(t), int(l + side), int(t + side)))


def load_item(it, size=SIZE):
    if it["source"] == "quickdraw":
        S = 512  # draw large, then downsample for antialiasing
        im = Image.new("RGB", (S, S), "white")
        dr = ImageDraw.Draw(im)
        for xs, ys in it["drawing"]:
            pts = [(x * S / 256, y * S / 256) for x, y in zip(xs, ys)]
            dr.line(pts, fill="black", width=16, joint="curve")
            if len(pts) == 1:
                dr.ellipse([pts[0][0] - 8, pts[0][1] - 8, pts[0][0] + 8, pts[0][1] + 8], fill="black")
        return im.resize((size, size), Image.LANCZOS)
    if it["source"] == "emoji" and "svg_url" not in it:
        svg = get(f"https://raw.githubusercontent.com/jdecked/twemoji/main/assets/svg/{it['code']}.svg").decode()
        return center_square(svg_to_image(svg), size)
    if "svg_url" in it:  # any SVG icon (Noto, OpenMoji, Fluent)
        return center_square(svg_to_image(get(it["svg_url"]).decode()), size)
    im = (Image.open(it["path"]) if "path" in it  # local file or URL; web-scale manifests set tries/timeout to fail fast on dead links
          else Image.open(io.BytesIO(get(fresh_url(it), tries=it.get("tries", 4), timeout=it.get("timeout", 60)))))
    im = im.convert("RGB")
    if it.get("box"):
        im = crop_box(im, it["box"], it.get("margin", 0.15))
    return center_square(im, size)


def to_tensor(im):
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).reshape(3, -1).float() / 255


def parse_shards(spec, n_shards):
    if spec == "all":
        return list(range(n_shards))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return [k for k in out if k < n_shards]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--qd-skip", type=int, default=0)
    ap.add_argument("--emoji-limit", type=int, default=1200)
    ap.add_argument("--manifest", default=None, help="jsonl of {id,label,source,url} to add (diverse photo/art sets)")
    ap.add_argument("--shard-size", type=int, default=400)
    ap.add_argument("--shards", default="all")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--recipe", default="fast")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items_file = out / "items.json"
    if items_file.exists():
        items = json.loads(items_file.read_text())
    else:
        items = []
        if a.per_class > 0:
            items += quickdraw_items([c for c in a.classes.split(",") if c], a.per_class, a.qd_skip, out)
        if a.emoji_limit > 0:
            items += emoji_items(a.emoji_limit, out)
        if a.manifest:
            items += [json.loads(l) for l in open(a.manifest) if l.strip()]
        random.Random(0).shuffle(items)
        items_file.write_text(json.dumps(items))
    n_shards = (len(items) + a.shard_size - 1) // a.shard_size
    todo = parse_shards(a.shards, n_shards)
    print(f"{len(items)} items, {n_shards} shards of {a.shard_size}; this run: shards {todo}", flush=True)
    if a.list_only:
        return

    rc = RECIPES[a.recipe]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda" or a.no_compile, "no GPU visible"

    def masks(s, g, H):
        return batched.stroke_masks(s, g, H, K=rc["segs"])

    mask_fn = None if a.no_compile else torch.compile(masks)
    pool = cf.ThreadPoolExecutor(16)
    warmed = False
    for k in todo:
        path = out / f"shard_{k:05d}.pt"
        if path.exists():
            continue
        chunk = items[k * a.shard_size:(k + 1) * a.shard_size]
        t_load = time.perf_counter()
        imgs = list(pool.map(load_item, chunk))
        tgt_all = torch.stack([to_tensor(im) for im in imgs])
        t_load = time.perf_counter() - t_load
        torch.manual_seed(1000 + k)
        S, P, t_fit = [], [], 0.0
        for i in range(0, len(chunk), a.batch):
            tgt = tgt_all[i:i + a.batch]
            n = len(tgt)
            if n < a.batch:  # pad the last batch so compiled shapes stay constant; extras are dropped
                tgt = torch.cat([tgt, tgt[:1].expand(a.batch - n, -1, -1)])
            tgt = tgt.to(dev)
            if not warmed:  # compile outside the timed region
                batched.decompose(tgt, SIZE, SIZE, rc["stages"], steps=2, mask_fn=mask_fn, clamp=CLEAN)
                warmed = True
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s, _, ps = batched.decompose(tgt, SIZE, SIZE, rc["stages"], steps=rc["steps"], mask_fn=mask_fn, clamp=CLEAN)
            if dev == "cuda":
                torch.cuda.synchronize()
            t_fit += time.perf_counter() - t0
            S.append(s[:n].cpu())
            P.append(ps[:n].cpu())
        S, P = torch.cat(S), torch.cat(P)
        ok = torch.isfinite(S).flatten(1).all(1)
        shard = {"strokes": S.half(), "stage_psnr": P, "ids": [c["id"] for c in chunk], "labels": [c["label"] for c in chunk],
                 "sources": [c["source"] for c in chunk], "ok": ok, "size": SIZE, "recipe": a.recipe, "seed": 1000 + k}
        torch.save(shard, str(path) + ".tmp")
        Path(str(path) + ".tmp").replace(path)  # atomic: a half-written shard never looks finished
        rec = {"shard": k, "n": len(chunk), "fit_sec": round(t_fit, 1), "load_sec": round(t_load, 1),
               "imgs_per_hr_fit": int(3600 * len(chunk) / t_fit), "psnr_mean": round(P[:, -1].mean().item(), 2),
               "bad": int((~ok).sum()), "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "t": time.time()}
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    print("DONE_ALL", flush=True)


if __name__ == "__main__":
    main()
