"""Time pretrained Paint Transformer (9M params) on CPU and count the strokes it emits.

Run from anywhere: python spike/bench_painttransformer.py
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PT = HERE.parent / "third_party" / "PaintTransformer" / "inference"
os.chdir(PT)  # brush textures are loaded by relative path
sys.path.insert(0, str(PT))

import torch  # noqa: E402
import inference  # noqa: E402

torch.set_num_threads(8)
count = {"n": 0}
_orig = inference.param2img_parallel


def counting(param, decision, meta_brushes, cur_canvas):
    count["n"] += int(decision.sum())
    return _orig(param, decision, meta_brushes, cur_canvas)


inference.param2img_parallel = counting

out = HERE / "out" / "painttransformer"
for size in (64, 128, 256):
    for img in sorted((HERE / "images").glob("*")):
        count["n"] = 0
        t0 = time.time()
        inference.main(str(img), "model.pth", str(out / str(size)) + "/", resize_h=size, resize_w=size)
        print(f"{img.name:16s} {size}px  {time.time() - t0:5.2f}s  strokes={count['n']}", flush=True)
