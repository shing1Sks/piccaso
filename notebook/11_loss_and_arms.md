# 11. Why B beats A, and is the loss sound? (2026-10-04)

The user asked two things after the D7 mixed pilot (B-text CLIP top-1 0.042, A-text 0.000, both "overfit" by ~5k steps):
1. In simple terms, why does set diffusion (B) do better than stroke-by-stroke (A), when both learn "strokes for a prompt"?
2. Is it really data/generalisation, or is the loss wrong? If it is overfitting, the model should at least reproduce its training pictures.

Code: `spike/memtest.py` (memorisation test + loss probe), `spike/mem_sheet.py`. Outputs: `spike/out/mem/`. GPU: one Vast RTX 3090 ($0.35/h), runtime image.

## Test 1: loss probe (no training): what does the loss charge vs what the eye sees?

Gaussian noise (sigma 0.3 in normalised units) on ONE group of slots and ONE group of numbers, 64 mixed training pictures.
"param loss" = exactly the training-loss term that error would cost; PSNR = rendered damage (higher = less visible).

| slots | numbers | param loss | PSNR vs target |
|---|---|---|---|
| coarse 16 | position | 0.0043 | 26.8 |
| coarse 16 | width | 0.0007 | 28.0 |
| coarse 16 | colour | 0.0021 | 31.3 |
| mid 49 | position | 0.0122 | 22.4 |
| mid 49 | width | 0.0021 | 26.7 |
| mid 49 | colour | 0.0059 | 27.4 |
| fine 100 | position | 0.0231 | 21.3 |
| fine 100 | width | 0.0039 | 31.8 |
| fine 100 | colour | 0.0116 | 27.7 |

Also: answering with a *different real picture* costs loss 1.25 (PSNR 8.6); answering with the *dataset-average strokes* costs 0.66 (PSNR 10.9).

Reading: the loss is per-number MSE, every slot equal. Per unit of loss, an error in the 16 coarse slots (big strokes that set
the composition) damages the picture ~3-5x more than the same loss spent in the 100 fine slots (coarse width: loss 0.0007 -> 28 dB;
fine width: 5.5x the loss -> 31.8 dB). So the loss over-weights detail and under-weights layout. It is *not wrong* (zero loss = exact
picture) but it is *mis-prioritised*. Second, plain MSE prefers the blurry average over a plausible different picture (0.66 < 1.25) -
the classic regression trap; diffusion/flow training avoids it at sampling time, which is why both arms use a noise-based loss.

## Test 2: memorisation: give each training picture its own id; can the model redraw it?

Removes the text encoder and generalisation from the question. Same models and losses as the pilot.

| run | pictures | steps | final train loss | PSNR sample vs target | identified (nearest of K) |
|---|---|---|---|---|---|
| B (cfg 1) | 64 | 3000 | ~0.08 | 26.2 | 64/64 |
| B (cfg 2) | 64 | 3000 | ~0.08 | 27.8 | 64/64 |
| A (cfg 1) | 64 | 3000 | ~0.06 | 31.1 | 64/64 |
| B (cfg 1) | 1024 | 6000 | ~0.11 | 21.7 | 1018/1024 (99.4%) |
| B (cfg 2) | 1024 | 6000 | ~0.11 | 23.9 | 99.3% |
| A (cfg 1) | 1024 | 6000 | ~0.13 | 21.4 | 946/1024 (92.4%) |

Sheets: `spike/out/mem/sheet64.png`, `sheet1024.png` (target | B | A). At 64 both redraw photos, paintings, emoji and sketches almost
exactly. At 1024 (still training when stopped) both are right; A's errors visibly snowball inside a picture (emoji Q interior,
cow nose, sketch edges) while B's are spread evenly as slight softness - the exposure-bias signature.

Loss by noise level after memorising (64): B 0.106 at t=0.05 (almost clean: the v-target is ~ the added noise, unrecoverable and
harmless) and ~0.001-0.006 elsewhere; A 0.19 at flow-time 0.95 (same thing on its side) and ~0.002 elsewhere. Residual loss is highest
on coarse slots in both (B 0.089 coarse / 0.050 fine; A 0.052 / 0.031) - the slots that matter most for the picture.

## Conclusions

1. **The loss rewards the right thing.** Driven low, it yields the exact picture, for both arms. The pilot failure is not a broken loss.
2. **The pilot never fitted its training set.** Best-val checkpoints had train loss ~0.43 (B) / ~0.70 (A) vs ~0.06-0.08 when memorised (same loss, incl. 10% prompt dropout).
   "Overfitting" there = val rises long before train is fitted: the model starts learning caption -> specific picture shortcuts
   (each item's captions act like an id) instead of caption -> general content. That is a data/conditioning problem.
3. **A can memorise better than B** (31 vs 28 dB) - so A's weak prompt-following is not capacity or loss. It is (a) exposure bias
   (trained on perfect previous strokes, sampled on its own) and (b) weak use of the prompt: given the true previous strokes and the
   canvas, the next stroke is predictable *without* the caption, so the caption gets little gradient. In B, at high noise the caption is
   the only information, so it must be used.
4. **Loss improvement worth testing (cheap):** weight slots by visual importance (level or stroke area), so layout errors cost more
   than detail errors. Also consider a small render/perceptual term at low noise only.


Cost of this round: ~$0.33 Vast (two dead 4090 hosts stuck on image pull ~ $0.01; one RTX 3090 with the *runtime* image, ready in 60 s).
Lesson: use `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (cached on more hosts) unless nvcc is needed.
