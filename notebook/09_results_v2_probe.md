# 09 - Results: extractor v2 + two generator arms (2026-10-03)

Setup (research/08): 16,000 images = 6 QuickDraw classes x 2,500 + 1,000 emoji (one "emoji" class), 128 px fits, 165 anchored slots
(4x4 + 7x7 + 10x10 grids), keep bit, canonical curve direction. Native control: the same 15,000 QuickDraw drawings as the human's own
pen strokes fitted with Beziers (<=64 slots, avg 17 strokes). Hardware: Vast.ai RTX 4090s (~$0.41/hr). Evaluation: a small CNN trained on
real images (val acc 0.96-0.98) classifies renders of generated samples (chance = 1/classes).

## Extraction (extractor v2)
- 16,000 images, 0 failed fits, mean fit PSNR 21.9 dB (21.7 after pruning); sketches complete by level 2 (65 strokes); keep fraction 57% (94.7 of 165 slots on).
- Speed: 8,200 img/hr per RTX 4090 on Vast (A40 was 5,065: 1.62x, matching the estimate). 40 shards done on 3 GPUs in ~50 min.

## Generator results (classifier accuracy on generated samples)
| Run | Data | Params | Steps | Acc generated | Notes |
|---|---|---|---|---|---|
| **B** set diffusion | native pen strokes | 21.9M | 28.5k | **0.97** | clean, varied, recognizable (car .96 cat .88 fish .96 house 1.0 sun 1.0 tree 1.0) |
| **B** set diffusion | extractor v2 fits | 21.9M | 27.5k (no render loss) | **0.97** | recognizable but messier; 149 slots on vs 95 in data |
| B + render loss | extractor v2 fits | 21.9M | 12.5k | 0.96 | render loss gave no gain, 2.2x slower steps |
| **A** stroke-by-stroke + canvas feedback | native pen strokes | 16.0M | 44k | 0.80 | overfit hard (train 0.02, val 0.50); scribbly cats/trees |
| **A** stroke-by-stroke + canvas feedback | extractor v2 fits | 16.0M | 24k | **0.40** | val loss rose from 0.50 (step 5k) to 1.29 (24k): sampled the overfit end; scribbles |

Chance is 0.17 (native, 6 classes) or 0.14 (v2, 7 classes). Emoji scores 1.0 for every arm and is not informative: all 1,000 emoji share one label.

## Takeaways
- The old gate failure was the representation. One continuous vector per stroke + diffusion works at ~22M params with 15-16K training images.
- Arm B (set diffusion) clearly beats arm A as configured. Arm A was NOT given a fair chance: its best validation loss was at ~5k steps but the 24-44k step EMA model was sampled. Rerun with early stopping/regularisation before concluding that stroke-by-stroke cannot work.
- Render loss: no measurable benefit here.
- Extractor v2 fits are harder to generate than human strokes (97% vs 97% by the classifier, but visibly messier: samples keep ~150 of 165 slots on vs 95 in training data; six house samples look nearly identical, i.e. low diversity at CFG 2).
- Emoji (one label for 1,000 different icons) only learns a palette: text/CLIP conditioning is needed there.
- Unmeasured: CPU inference speed, text prompts, anything beyond 6 sketch classes.

## Cost of this phase
Vast credit 10.00 -> 7.76 (4 instances, ~$2.24 incl. idle/boot); RunPod ~$0.50 earlier the same day.
Files: spike/out/{nativeA,nativeB,runA,runB,runBnr}/{samples.png,metrics.json,train.log,ckpt.pt}; spike/out/v2/ (40 shards).
