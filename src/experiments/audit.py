"""Audit an extraction shard: quality per source/stage, quantization cost, stroke usage. CPU.

python audit.py out/pilot_128/shard.pt
"""
import sys
from collections import defaultdict

import numpy as np
import torch

import batched
from extract import load_square

import json

sh = torch.load(sys.argv[1])
S, size = sh["strokes"], sh["size"]
manifest = {r["id"]: r for r in map(json.loads, open("pilot/manifest.jsonl"))}
cum = np.cumsum([n for n, _ in sh["stages"]])
srcs = sorted(set(sh["sources"]))
np.set_printoptions(precision=2, suppress=True)

print(f"{len(S)} images, {size}px, {S.shape[1]} strokes/img, stages {list(cum)}")
print("\nPSNR (dB) after each stage, by source")
print(f"{'source':10s} " + " ".join(f"{k:>6d}" for k in cum) + "   quantized(64/16/16/4)  quantized(128/32/32/8)")

# quantization cost: re-render the quantized strokes against the target
targets = torch.stack([load_square(manifest[i]["path"], size) for i in sh["ids"]])
with torch.no_grad():
    q1 = torch.cat([batched.render(batched.quantize(S[i:i + 1]), size, size) for i in range(len(S))])
    q2 = torch.cat([batched.render(batched.quantize(S[i:i + 1], 128, 32, 32, 8), size, size) for i in range(len(S))])
pq1, pq2 = batched.psnr(q1, targets).numpy(), batched.psnr(q2, targets).numpy()
for s in srcs:
    m = np.array([x == s for x in sh["sources"]])
    print(f"{s:10s} " + " ".join(f"{sh['stage_psnr'][m][:, j].mean():6.2f}" for j in range(len(cum)))
          + f"   {pq1[m].mean():10.2f}              {pq2[m].mean():10.2f}")
print(f"{'ALL':10s} " + " ".join(f"{sh['stage_psnr'][:, j].mean():6.2f}" for j in range(len(cum)))
      + f"   {pq1.mean():10.2f}              {pq2.mean():10.2f}")

print("\nStroke usage (all images)")
p = S.reshape(-1, 11)
cx = p[:, [0, 2, 4]].mean(1), p[:, [1, 3, 5]].mean(1)
print(f"  centers outside canvas        : {((cx[0] < 0) | (cx[0] > 1) | (cx[1] < 0) | (cx[1] > 1)).float().mean():.1%}")
print(f"  any control point off-canvas  : {((p[:, :6] < 0) | (p[:, :6] > 1)).any(1).float().mean():.1%}")
print(f"  width at min clamp (0.004)    : {(p[:, 6] <= 0.0041).float().mean():.1%}")
print(f"  width at max clamp (0.6)      : {(p[:, 6] >= 0.599).float().mean():.1%}")
print(f"  alpha at min clamp (0.2)      : {(p[:, 10] <= 0.2001).float().mean():.1%}")
print(f"  alpha at max clamp (1.0)      : {(p[:, 10] >= 0.9999).float().mean():.1%}")
length = ((p[:, 4:6] - p[:, 0:2]).norm(dim=1))
print(f"  chord length  p0->p2 (mean)   : {length.mean():.3f}   (<0.01: {(length < 0.01).float().mean():.1%})")
print("  by stage: mean width / mean alpha / mean chord")
a = 0
for (n, _), b in zip(sh["stages"], cum):
    seg = S[:, a:b].reshape(-1, 11)
    L = (seg[:, 4:6] - seg[:, 0:2]).norm(dim=1)
    print(f"    strokes {a:3d}-{b - 1:3d}: width {seg[:, 6].mean():.3f}  alpha {seg[:, 10].mean():.2f}  chord {L.mean():.3f}")
    a = b

# how many strokes matter? remove the last k strokes and see what is lost
print("\nWhat the last stage buys (PSNR gain of adding stage j, mean over images)")
sp = sh["stage_psnr"]
for j in range(len(cum)):
    prev = sp[:, j - 1].mean() if j else torch.tensor(batched.psnr(torch.ones_like(targets), targets).mean())
    print(f"    stage {j} (+{sh['stages'][j][0]:3d} strokes): +{(sp[:, j].mean() - prev):.2f} dB")
