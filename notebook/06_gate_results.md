# 06 - Stage-0 extraction benchmark + learnability gate (2026-10-03)

## What was run
- `spike/extract_prod.py`: fast recipe (160 strokes, fit at 128 px, 4 segments, compile, clean constraints), sharded and resumable, data fetched on the pod.
  5,800 images = 4,800 QuickDraw (16 classes x 300) + 1,000 single-codepoint Twemoji emoji, 15 shards of 400.
- `spike/train_gate.py`: 4.87M-param GPT, 1,600 tokens per image (6 position tokens in 128 bins, width 16 log bins, rgb 16 bins each),
  one class token, field-masked softmax, flip augmentation, 5,510 train / 290 val sequences.
- Hardware: A40 secure ($0.49/hr). Both the benchmark and the gate used the same pod. Money spent today: $0.50 (RunPod billing API).

## Extraction benchmark (A40, real Stage-0 data)
- **5,060-5,070 img/hr** fit-only (284 s per 400-image shard), 71 min for 5,800 images. Slower than the 5,825 img/hr measured on the mixed pilot (-13%): another reason to treat all other-GPU numbers as +-15%.
- Fit quality at 128 px, PSNR at 16 / 64 / 160 strokes: **emoji 21.9 / 27.3 / 30.4 dB, QuickDraw 14.6 / 19.5 / 22.1 dB**.
  Sketch PSNR is low because thin black lines on white are punished hard; the renders look right (see `spike/out/gate_a40/samples.png` left column).
- Quantization (128 position bins, 16 width, 16 per colour) renders at 30.1 dB PSNR against the continuous strokes at 128 px (the 128-px fit itself is only 22-30 dB), so tokenization is not the bottleneck.
- Zero failed/NaN fits in 5,800.
- RTX 4090 community host: never started (see 05); 4090 speed on this recipe is still the 1.63x ratio, not re-measured.

## Learnability gate: NOT passed in this form
Three runs, same data, 6-12 min of training each:

| run | val NLL/token | start-point x0 / y0 NLL | other geometry (x1..y2) | class gap | samples |
|---|---|---|---|---|---|
| A: original order | 2.915 (best 2.79 at step 3.7k, then overfits) | 4.69 / 4.47 (uniform = 4.85) | 3.7-3.9 | 122 nats/seq | scribbles |
| B: fix curve direction only, dropout 0.1 | 2.730 | 4.61 / 4.46 | 3.4-3.8 | 63 | scribbles |
| C: direction + sort strokes within each stage by position, dropout 0.1 | **2.441** | **3.30 / 2.75** | 3.4-3.8 | 109 | scribbles, slightly more spatial structure |

Interpretation:
- The model **does use the class** (shuffling the class token costs 60-120 nats per sequence) and learns colours (near-black/white) easily.
- It **cannot predict where strokes go** under the raw extraction order: stroke order inside a stage is random (new strokes are placed by sampling the residual error) and each curve can be written in either direction, so the start point of the next stroke is nearly uniform. Sorting strokes by position fixes that (x0/y0 NLL 4.7/4.5 -> 3.3/2.75).
- **The remaining stroke geometry is still near-uniform given the start point** (x1..y2 about 3.4-3.8 nats = ~35 plausible bins out of 128): absolute-coordinate tokens waste the structure that a stroke's control and end points lie close to its start.
- 5,510 sequences x 5M params is tiny: run A overfit after ~4k steps (train 2.6 vs val 2.9). This is a data-starved, compute-starved first probe, **not a verdict on feasibility**.
- Direction canonicalization is free (swapping p0/p2 renders pixel-identically, 100 dB). Sorting strokes by position after extraction is NOT free: 18.5 dB vs the original render for sketches, 23.9 dB for emoji, because later strokes paint over earlier ones. The sorted order has to be built into the extraction (sort each stage's new strokes at initialisation, or refine after sorting), then re-extracted.

## Recommendation (do NOT launch the 100k extraction in the current format)
Extraction format changes would make a 100k run in the current format partly wasted. Cheaper next probe (about $0.6 total, same 5.8k scale):
1. Re-extract with spatial order built in (sort new strokes at init by row band then x; canonical direction), so the sequence is raster-like and render-exact.
2. Tokenize p1 and p2 as offsets from p0 (small ranges, finer bins) instead of absolute coordinates; keep p0 absolute.
3. Retrain with dropout + early stop at the validation minimum; also train a version with a larger dataset slice (the 100k run only after this passes).
4. If sequences still look like noise: change the architecture (per-stroke token with a mixture/continuous head, hierarchical coarse-to-fine, or a conditioning image/latent prefix) as the user expected might be needed.

Files: `spike/out/gate_a40/` (shards, samples, metrics, logs), `gate_dir/`, `gate_sort/` (sample sheets + metrics).
