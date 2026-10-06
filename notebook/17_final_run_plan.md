# 17. Final run plan: PixelProse base + ~100M stroke painter (agreed 2026-10-05)

**Sign-off amendments (user, 2026-10-05):** doodle/emoji add-ons are NOT part of this run; ~300k images (more later);
361 slots (only measure the 761-slot ceiling in the dry run); no mirror flips (captions carry left/right); 250M only if a
fair check shows >= 10% better CLIP gap + better StrokeBench at equal steps with a non-shrinking lead (scored at 5k/10k/15k
on 1,000 held-out captions + DOCCI); watch for idle GPUs; spend cap $43 of the $45.57 credit.

Budget: Vast credit $45.57 -> spend cap **$43** (keep ~$2 buffer). Rules: small/dry runs before every full run; peek sheets
during runs; only the watcher collects/destroys boxes (`destroy -y` + verify); stop and report if a gate fails or the cap would be hit.

## Decisions (from the discussion)
- Base model trains on **PixelProse only** (clean, detailed captions). DOCCI = held-out human-caption test set. WikiArt/Cleveland dropped.
- Doodles/emoji = small **LoRA add-ons** on the frozen base (QuickDraw/icon strokes already extracted), so the base is not polluted.
- **"Survives strokes" filter**: keep only pairs whose 361-stroke render still matches the caption (CLIP), drop the worst ~25%.
- Text encoder **Long-CLIP-B** (248 tokens, same CLIP space -> reference pictures keep working); train on full caption +
  first sentence(s) so short prompts work. Fallback if it will not load: CLIP-B per-token + first sentences.
- Model: B (set diffusion), 361 slots, **self-conditioning + cross-attention**, cosine LR, batch 256; sample cfg 3-4, 25 steps.
  Render feedback dropped (no gain at 1.9x cost). Arm A retired (code/data/checkpoints kept for the paper).
- Size: **~100M main**. 250M only if the scaling check says it is clearly better and the budget allows.
- New scorecard **StrokeBench** (~200 fixed prompts: objects, colour+object, two objects, style; CLIP multiple-choice checks)
  + CLIP gap on DOCCI/PixelProse held-out + prompt/reference sheets.

## Stages, gates, cost
| stage | where | what | gate to continue | est. cost | est. wall |
|---|---|---|---|---|---|
| 0 local prep | laptop | PixelProse manifest builder (2 cc12m files, filters: clean, aesthetic>=5, ~420k URLs for ~300k alive), caption cleaner, survives-strokes scorer, Long-CLIP loader, StrokeBench, multi-GPU (DDP) training, LoRA add-on, periodic peek sheets; CPU smoke tests | all smoke tests pass | $0 | ~4 h |
| 0b dry run | 1-2x 5090 | 2k PixelProse end to end (download -> strokes -> scores), 300-step training 25M/100M/250M (speed), DDP on 2 GPUs | link survival >= 60%, stroke PSNR ~23+, survives-strokes score as in the probe, s/step measured | ~$1 | ~1 h |
| 1 data | 4x 5090 | ~300k PixelProse -> strokes + scores + Long-CLIP image vectors; DOCCI 1.5k test; merge on hub box | peek sheet after ~1 h; >= 250k usable pairs | ~$17 | ~8-9 h |
| 2 scaling check | 3x 5090 | 25M / 100M / 250M, same short budget (~15k steps), StrokeBench + CLIP gap | pick size: 250M only if >= 10% better than 100M at equal steps and budget fits | ~$5 | ~3 h |
| 3 main run | 4x 5090 (DDP) | ~100M, ~100k steps, peek sheets every 20k steps | beats 22M A1 on StrokeBench and CLIP gap | ~$10 (100M) / ~$20 (250M, fewer steps) | ~6 h |
| 4 add-ons + eval | 1x 5090 | LoRA doodles + emoji; final StrokeBench, sheets, write-up (research/18) | - | ~$2 | ~2 h |
| buffer / overhead | | box setup, uploads, idle | | ~$3 | |
| **total** | | | | **~$38 (100M)**, up to ~$43 with a shortened 250M | **~24 h** |

Known risks: dead links / slow downloads (download on the boxes, parallel threads), Long-CLIP checkpoint loading, DDP bugs
(fallback: single GPU per run, longer wall time), 300k may still be thin for rare concepts (more PixelProse later is ~$5/100k).
