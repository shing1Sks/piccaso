"""PixelProse -> extraction manifests (research/17 stage 1).

  python pp_manifest.py --parquets out/probe/pp/cc12m_00.parquet out/probe/pp/cc12m_01.parquet --n 420000 --workers 4 --out out/pp
Filters: watermark_class_id == 1 (clean; 0 = watermark, 2 = text overlay), aesthetic >= 5, original short side >= 256,
low toxicity / sexual content. Captions: Gemini's 'This image displays:' prefix stripped; each item gets the full caption,
its first sentence and its first two sentences (training samples one of them, so short prompts work too).
Writes manifest_<w>.jsonl (one per GPU worker; rows: id, source, label, url, texts, tries, timeout) + manifest_report.json.
"""
import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

PREFIX = re.compile(r"^\s*(this image (displays|shows|depicts|features|is)|the image (shows|displays|depicts|features))\s*:?\s*", re.I)


def clean(c):
    c = re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()
    c = PREFIX.sub("", c)
    return c[:1].upper() + c[1:]


def sentences(c):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", c) if x.strip()]


def caption_set(raw):
    full = clean(raw)
    ss = sentences(full)
    out = [full]
    for k in (1, 2):
        if len(ss) > k - 1:
            v = " ".join(ss[:k])
            if v not in out and len(v.split()) >= 3:
                out.append(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=420000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="out/pp")
    ap.add_argument("--min-aesthetic", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["uid", "url", "vlm_caption", "watermark_class_id", "aesthetic_score", "original_width", "original_height", "width", "height",
            "toxicity", "sexual_explicit", "obscene"]
    per_file = a.n // len(a.parquets) + 1
    rows, rep = [], {"files": {}}
    for f in a.parquets:
        df = pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names]).to_pandas()
        n0 = len(df)
        side = df[["original_width", "original_height"]].min(axis=1) if "original_width" in df else df[["width", "height"]].min(axis=1)
        keep = ((df["watermark_class_id"] == 1) & (df["aesthetic_score"] >= a.min_aesthetic) & (side >= 256)
                & (df["toxicity"].fillna(0) < 0.1) & (df["sexual_explicit"].fillna(0) < 0.1) & (df["obscene"].fillna(0) < 0.1)
                & df["vlm_caption"].notna() & df["url"].notna())
        df = df[keep]
        df = df.sample(n=min(per_file, len(df)), random_state=a.seed)
        rep["files"][f] = {"rows": n0, "pass_filters": int(keep.sum()), "taken": len(df)}
        for r in df.itertuples(index=False):
            caps = caption_set(r.vlm_caption)
            if len(caps[0].split()) < 5:
                continue
            rows.append({"id": f"pp_{r.uid}", "source": "pixelprose", "label": caps[-1][:200], "url": r.url, "texts": caps,
                         "aesthetic": round(float(r.aesthetic_score), 2), "tries": 2, "timeout": 12})
        print(f, rep["files"][f], flush=True)
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    rows = uniq[:a.n]
    for w in range(a.workers):
        with open(out / f"manifest_{w}.jsonl", "w", encoding="utf-8") as fo:
            for r in rows[w::a.workers]:
                fo.write(json.dumps(r) + "\n")
    words = sorted(len(r["texts"][0].split()) for r in rows)
    rep.update(total=len(rows), workers=a.workers, caption_words_median=words[len(words) // 2], caption_words_p90=words[int(len(words) * .9)],
               captions_per_item=round(sum(len(r["texts"]) for r in rows) / len(rows), 2))
    json.dump(rep, open(out / "manifest_report.json", "w"), indent=1)
    print(json.dumps(rep), "MANIFEST_DONE", flush=True)
