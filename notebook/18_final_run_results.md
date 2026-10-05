# 18. Final run results: PixelProse + Long-CLIP 100M stroke painter (2026-10-05)

Spend: $42.75 of the $45.57 credit (cap $43); credit left $2.82. One 8x RTX 5090 box (US, $3.74/h), destroyed 12:36 and verified.
Results on the laptop: `spike/out/final_run/` (final checkpoint `s3/main/ckpt.pt`, 102M params, 408 MB fp32; sheets; all scores),
dataset `spike/out/ppfinal_local/` (232,134 training items: strokes, captions, Long-CLIP + OpenAI-CLIP vectors; DOCCI test set).

## Data (stage 1)
PixelProse cc12m_00 + cc12m_01, filters clean (watermark class 1) + aesthetic >= 5 + short side >= 256 + low toxicity:
418,981 URLs -> 309,512 converted (~74% links alive) -> 232,134 kept by the survives-strokes filter (bottom 25% of
Long-CLIP caption-vs-361-stroke-render similarity dropped). Mean stroke PSNR @256 = 23.45 dB. Captions: Gemini, median 79
words; each item trains on full caption / first sentence / first two sentences. DOCCI 1,500 converted as a test set.
Finer-detail probe (dry run, 732 images): 565 slots (20x20 detail) vs 361: +1.05 dB PSNR, caption match 0.2987 vs 0.2901 (+3%) -> stay at 361.

## Model
Set diffusion over 361 slots, d 640, 11 layers, 10 heads, cross-attention to Long-CLIP-B per-token features (248 tokens) +
pooled vector, self-conditioning, 25% reference-image conditioning, slot weights 3/1.5/1/1.

## Scaling check (stage 2): same data/settings, equal steps, 1,000 held-out PixelProse captions + StrokeBench
| model | step | CLIP gap cfg3 (diag) | objects top-1 / top-5 | colour | style |
|---|---|---|---|---|---|
| 25M | 2,667 / 5,334 / 8,000 | 0.0373 / 0.0440 / 0.0464 | 13% / 37% (8k) | 94% | 41% |
| 100M | 2,667 / 5,334 / 8,000 | 0.0412 / 0.0489 / 0.0523 | 21% / 45% (8k) | 94% | 40% |
| 250M (lr 2e-4, grad-accum 2) | 2,667 / 5,334 | 0.0421 (+2.2% vs 100M) / - | 8% / 34% ; 17.5% / 40.6% | 96% | 40% |
Verdict: 250M is NOT clearly better at equal steps (+2% gap at 2.7k, equal StrokeBench at 5.3k) and costs ~2.5x per step
(1.0 s/step on 4 GPUs vs 0.4 on 2). At this data size 100M is the right size. (250M 5.3k CLIP gap not scored: cut for budget.)

## Final 100M (stage 3): continued from the 100M 8k checkpoint, 7,000 more steps at batch 1,024 (lr 5e-4, cosine)
(~7.2M more training pictures; 8 GPUs; 0.39 s/step)
| model | CLIP top-1 (1,000 held-out PP captions, chance 0.001) | CLIP gap (cfg2, 50 steps) |
|---|---|---|
| 25M @ 8k | 0.034 | 0.0388 |
| 100M @ 8k | 0.040 | 0.0436 |
| **final 100M** | **0.112 (112x chance)** | **0.0595 (+36%)** |
| fitted 361 strokes (ceiling) | 0.725 | 0.1450 |
Old best models on the same held-out PixelProse captions (diag, A1 22M old data): gap 0.0416 (cfg2, 25 steps) / 0.0501 (cfg3).
Sheets: `final_run/s3/main/prompts.png`, peeks at 2.5k / 5k. Visible: grinning emoji with eyes and mouths, birds on branches,
airplanes in the sky, ships in engravings, figures skateboarding, living rooms with windows and couches, city at night,
red buses, portraits with faces. Final StrokeBench / DOCCI scores were not completed (cut at the budget deadline).

## Lessons (operational)
- Killing a parent script by name also kills its forked subshells (same cmdline) -> post-step jobs never ran; ~1 h of 8 GPUs idle.
- Fork after CUDA init deadlocks a ProcessPool (use spawn); `pkill -f pattern` inside ssh matches its own command.
- Two jobs on one GPU -> OOM; long real captions need more memory than dry-run captions (250M needed grad accumulation).
- DDP with tiny per-GPU batches is overhead-bound: batch 32/GPU ran at the same s/step as 128/GPU -> use large per-GPU batches.
- Rank-0-only validation over 11.6k items stalled 7 GPUs -> validate on a fixed 1,024 subset.
- New HARD RULE (memory gpu-audit-rule): audit every rented GPU every 10 minutes, act on idle immediately.

## Next (user to decide)
More steps on this 100M (curve still rising), more PixelProse data, few-step distillation for local speed, the 512 px
finishing-layer measurement (~15k images) and a cascade finisher model, opacity/softness stroke parameters.
