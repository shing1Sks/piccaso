# 14. Where does B lose? Experiments 1-4 (2026-10-04, 4x RTX 5090, $5.24)

All numbers: CLIP image-text retrieval on the same 500 held-out captions (seed-0 split), 50 DDIM steps; "gap" = own caption
similarity minus others. Per-source scores use 150 held-out captions of that source. Code: `spike/diag_eval.py`,
`spike/exp14_report.py`; strokegen gained `--train-frac`, 361-slot support (detail width 0.004, per-level keep calibration,
256 px sheets). Results: `spike/out/exp14/<box>/<run>/` (diag.json, diag.png, prompts.png, references.png, metrics.json, ckpt.pt).

## Free check first: how much does the stroke format keep? (CLIP image retrieval of the fitted strokes vs the real picture)
Icons 95-99% and QuickDraw 79% with 165 strokes; photos/art 5-33%. With the 196 fitted detail strokes: faces 33% -> 83%,
COCO 32% -> 71%. The format is the ceiling for photos, not for emoji/doodles.

## Results
| run | params | data | slots | gap cfg2 | gap cfg3 | gap cfg4 | top-1 cfg2 / cfg4 |
|---|---|---|---|---|---|---|---|
| old B-58M (constant LR, batch 128, early stop) | 58.3M | 100% | 165 | 0.0242 | 0.0286 | 0.0309 | 0.032 / 0.070 |
| **new B-58M (cosine LR, batch 256, 70k steps)** | 58.3M | 100% | 165 | **0.0270** | **0.0322** | **0.0348** | 0.050 / **0.084** |
| B-22M new recipe | 22.2M | 25% | 165 | 0.0190 | 0.0233 | | 0.024 |
| B-22M new recipe | 22.2M | 50% | 165 | 0.0227 | 0.0265 | | 0.042 |
| B-22M new recipe | 22.2M | 100% | 165 | 0.0237 | 0.0275 | | 0.038 |
| **B-22M joint base+detail** | 22.3M | 100% | **361** | **0.0253** | **0.0302** | | 0.042 |
| (old B-22M, for reference) | 22.2M | 100% | 165 | 0.0218 (strokegen eval) | | | 0.042 |

## Answers
1. **Denoising steps do not matter**: old B-58M gap 0.0241 / 0.0242 / 0.0235 / 0.0242 at 25 / 50 / 100 / 250 steps (new: 0.0266 / 0.0270 / 0.0259 / 0.0262). 25 steps is enough -> 2x faster sampling for free.
2. **Guidance was too low**: cfg 2 -> 4 = +28-29% gap on both B-58Ms. Use cfg 3-4 (check diversity on sheets).
3. **Training recipe mattered**: same model + data, cosine LR + batch 256: +12% gap (0.0242 -> 0.0270); B-22M +9% (0.0218 -> 0.0237), now equal to the old 58M.
4. **Data is flattening at this size**: 25 -> 50% +19% gap, 50 -> 100% +4%. 25% overfits (val rises after ~16k steps). 300k images would buy little for 22M/165 slots; not the next lever.
5. **Joint 361-slot generation works** (the separate detail model did not): +7% (cfg2) / +10% (cfg3) overall; per source vs 165 slots: faces +64%, Cleveland +40%, WikiArt +15%, COCO/objects/PD12M +6%, icons 0, QuickDraw -7%. Gains land on photos/art, exactly where the format lost. Sheets at 256 px show faces with eyes/nose/glasses.
6. **The model gap is everywhere**: generated/fitted gap ratio 0.35-0.67 per source for the best model, emoji included (0.48), so model quality (not only the format) still limits every domain.

## Recommended next step (not run)
B-58M x 361 slots x new recipe (cosine, batch 256), sampled at cfg 3-4 with 25 steps. Each lever was measured separately
(recipe +12%, cfg +29%, joint detail +7-10%); they should mostly stack. ~3.5 h on one 5090, ~$2.
After that: richer text conditioning (CLIP token features + cross-attention) for multi-object prompts; domain tag / MoE only after.

## Incidents
- vastai `destroy instance` asks for confirmation; without `-y` it silently did nothing: bx1 idled ~35 min (~$0.30). Fixed in collect.sh (`-y` + verify the instance list).
- A manual collect raced the watcher's collect of bx4 and truncated `B22_f50/ckpt.pt` (lost; metrics/sheets intact). Only the watcher collects.
