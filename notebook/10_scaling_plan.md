# 10 - Scaling plan: more images + classes + captions, text-conditioned, both arms (2026-10-03)

Principle (user): never run a full job on a new design without small dry runs that show each piece of the process makes sense.

## Dry-run log (what each cheap run showed)
| # | Question | Run | Result | Decision |
|---|---|---|---|---|
| D1 | Why does arm B over-stroke and "all houses look the same"? | CPU diagnostics on saved weights (free) | Keep decisions are 99.9% correct when denoising real data but generation from noise switches ~147/165 slots on (data: 95). Diversity is fine (17.1-17.9 vs real 17.8). | Default sampler: DDIM, CFG 2, top-k keep calibration (k from the real per-class distribution): slots 96, acc 0.91 vs 0.93 |
| D2 | Can arm A (stroke-by-stroke) be fixed? | A + early stop/dropout/prev-noise/CFG, with and without canvas feedback (16k imgs) | 0.36 with canvas, 0.51 without, 0.43 with heavier regularisation. Val best at ~8-12k steps then overfits. | A is data-starved at 16k images; keep it, re-test at 100k; canvas feedback hurts at small scale |
| D3 | Which captioner? | SmolVLM-256M/500M, Florence-2-base on 24 photo/painting/museum images | Florence short caption best (accurate, 0.2-0.4 s/img); SmolVLM-500M sometimes degenerates into repetition; 256M too vague | Florence-2-base `<CAPTION>`; PD12M already has captions; keep human metadata too |
| D4 | Does extractor v2 handle photos/art at 128 px / 165 slots? | 72 mixed images | PSNR 23.9 dB (COCO), 25.7 (WikiArt), 27.1 (Cleveland); 94% of slots kept; manuscript pages blur out | OK; filter Cleveland to Painting/Drawing/Print/Sculpture, drop text folios |
| D5 | Text interface: CLIP vector instead of class id | B on the 16k sketch set, templated captions | raw CLIP vec 0.57 (in-dist) / 0.62 (held-out phrasing); standardised per-dim 0.68 / 0.70; class ids give 0.91-0.97 | Use standardised embeddings; held-out phrasing generalises as well as trained phrasing; gap vs class ids to be re-tested on diverse captions (near-identical template captions have near-identical CLIP vectors) |
| D6 | Are the data sources downloadable at scale? | COCO train, WikiArt (HF), Cleveland, Met (by department), PD12M (URLs + captions) | all reachable; Met needs ~0.4 s pauses between objects; AIC blocked | Use them |

## Full-run target (~100k images) and cost, after the mixed pilot (D7) passes
| Source | Images | Text | License |
|---|---|---|---|
| QuickDraw, 345 classes x ~100 | 34.5k | templates ("a drawing of a X" + variants) | CC-BY 4.0 |
| Emoji (Twemoji) | ~1.4k | emoji name templates | CC-BY 4.0 |
| COCO train | 25k | 5 human captions + Florence | research |
| WikiArt | 15k | style/genre/artist + Florence | research |
| Cleveland (filtered) | ~10k | title/type/artist + Florence | CC0 |
| Met (Paintings, Drawings & Prints) | ~8k | title/artist + Florence | CC0 |
| PD12M subset (photos, museum scans) | ~10k | its own synthetic caption | CC0 |
Extraction ~12.7 GPU-h (~$5.2 at $0.41/h on Vast 4090); Florence captioning of ~33k images ~2 GPU-h (~$0.9); training both arms text-conditioned ~1-2 GPU-h each (~$1-2). Total ~$8-9; Vast credit is ~$7 now.

## D7 (in progress): mixed pilot, ~7k images
345 QuickDraw classes x 6, 500 emoji, ~4.3k photo/art rows with captions. Extract on 2 GPUs, caption with Florence, train B-text and A-text (20 min each), score with CLIP image-text retrieval (generated vs fitted upper bound vs chance) and free-text prompt sheets.
Gate to the full run: B-text retrieval clearly above chance and visibly prompt-following sheets; extraction stable; captions sane.

## D7 result: mixed pilot (6,833 items; 6,554 usable; 13.7k distinct captions; 20 min per arm)
Data: 345 QuickDraw classes x6, 500 emoji, COCO 1,500, WikiArt 722 usable (278 failed: HF viewer rate limit), PD12M 1,000, Met 599, Cleveland 163.
Extraction: 8,144 img/hr/GPU, mean fit 24-30 dB by source (quickdraw 21.5, coco 23.8, pd12m 24.6, wikiart 24.9, cleveland 25.2, met 25.3, emoji 29.9).
Captions: Florence-2 6 img/s; 1,368 kept (395 failed for WikiArt, same rate limit).

| Arm | steps / best val | CLIP retrieval top-1 (chance 0.010, fitted-stroke ceiling 0.281) | CLIP gap (ceiling 0.053) | Prompt sheet |
|---|---|---|---|---|
| **B** set diffusion, text | best val 0.450 at step 4.2k; val then rose to 0.98 by step 27k (strong overfit); best EMA used | **0.042 (4x chance)** | **0.0102** | follows scene/colour/style: green fields + sky for "landscape", dark with blue/orange lights for "city street at night", grey monochrome for "engraving", warm figure for "portrait", boxy roof outline for "sketch of a house", red-orange discs for emoji. No recognisable objects (pizza, skateboard, bird). |
| **A** stroke-by-stroke, text | early stop at 5k, best val 0.713 | 0.000 | 0.0040 | right style family (black-on-white sketches, colourful emoji blobs, earthy photo/painting textures); no content |

Reading: at 6.5k images B learns text -> scene/style but not objects, A learns style only. Both overfit a lot (val minimum at 4-5k steps). Not a refutation: the dataset is ~15x smaller than the full plan and has 13.7k distinct captions over 6.5k images.
Lessons for the full run: (1) WikiArt must come from the parquet files (image bytes), not the throttled viewer API; (2) early-stop on validation for both arms (B had no patience set); (3) add dropout / more data before scaling the model; (4) keep CLIP retrieval + prompt sheets as the gate metric; (5) cost: Vast credit now $6.17.
