"""Batched Florence-2-base captions on the extractor's cached images (out/<run>/img/<id>.jpg).  GPU.

  python caption_v2.py out/full out/full/florence.json --tasks short,detailed [--part 0/2] [--limit 200]
Writes {id: {"short": str, "detailed": str}}; resumable. Sources whose own text is already good get only a short caption
or none (see SKIP / SHORT_ONLY). Prints img/s per task so the dry run measures throughput.
"""
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import torch
from PIL import Image

from extract_v3 import safe_name

SKIP = {"quickdraw", "twemoji", "noto", "openmoji", "fluent", "coco"}  # templates / 5 human captions are better
SHORT_ONLY = {"pd12m"}  # ships a long synthetic caption already
TASKS = {"short": "<CAPTION>", "detailed": "<DETAILED_CAPTION>"}

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("out")
ap.add_argument("--tasks", default="short,detailed")
ap.add_argument("--part", default="0/1")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
i0, n = map(int, a.part.split("/"))
items = json.load(open(f"{a.run}/items.json"))
have = {p.stem for p in (Path(a.run) / "img").glob("*.jpg")}  # only items this box extracted
items = [r for k, r in enumerate(items) if k % n == i0 and r["source"] not in SKIP and safe_name(r["id"]) in have]
out = Path(a.out)
done = json.loads(out.read_text()) if out.exists() else {}
if a.limit:
    items = items[:a.limit]

try:  # Florence's remote code imports flash_attn even when unused; skip that import check
    from unittest.mock import patch

    from transformers.dynamic_module_utils import get_imports

    def _no_flash(f):
        return [x for x in get_imports(f) if x != "flash_attn"]
    _p = patch("transformers.dynamic_module_utils.get_imports", _no_flash)
    _p.start()
except Exception:
    pass
from transformers import AutoModelForCausalLM, AutoProcessor

dev, dt = "cuda", torch.float16
name = "microsoft/Florence-2-base"
proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
try:
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dt, trust_remote_code=True).to(dev).eval()
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt, trust_remote_code=True).to(dev).eval()


def load(r):
    p = Path(a.run) / "img" / f"{safe_name(r['id'])}.jpg"
    try:
        return r, Image.open(p).convert("RGB")
    except Exception:
        return r, None


for task in a.tasks.split(","):
    todo = [r for r in items if not (done.get(r["id"]) or {}).get(task) and not (task == "detailed" and r["source"] in SHORT_ONLY)]
    t0, k = time.time(), 0
    with cf.ThreadPoolExecutor(8) as ex:
        loaded = ex.map(load, todo)
        while True:
            batch = [x for _, x in zip(range(a.batch), loaded)]
            if not batch:
                break
            batch = [(r, im) for r, im in batch if im is not None]
            if not batch:
                continue
            inp = proc(text=[TASKS[task]] * len(batch), images=[im for _, im in batch], return_tensors="pt").to(dev, dt)
            with torch.no_grad():
                g = model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                   max_new_tokens=40 if task == "short" else 96, num_beams=3)
            for (r, _), txt in zip(batch, proc.batch_decode(g, skip_special_tokens=True)):
                done.setdefault(r["id"], {})[task] = txt.strip()
            k += len(batch)
            if k % (a.batch * 20) < a.batch:
                out.write_text(json.dumps(done))
                print(f"{task}: {k}/{len(todo)}, {k / (time.time() - t0):.1f} img/s", flush=True)
    out.write_text(json.dumps(done))
    print(f"{task} DONE {k} in {time.time() - t0:.0f}s ({k / max(time.time() - t0, 1e-6):.1f} img/s)", flush=True)
print("CAPTION_DONE", flush=True)
