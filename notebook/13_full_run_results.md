# 13. Full run results (2026-10-04, run autonomously while the user was away)

**Summary (finished 08:45, 2026-10-04; total spend $9.23 of the $20 ceiling; all GPU instances destroyed)**

| Model (96k images, text + reference-image conditioning) | Params | CLIP top-1 (500 captions, chance 0.002) | CLIP gap | Verdict |
|---|---|---|---|---|
| Pilot B (6.5k images, for reference) | 22M | 0.042 of 96 (4x chance) | 0.0102 | scene/style only |
| A, stroke by stroke | 16M | 0.006 (3x) | 0.0108 | now follows scenes (pilot: style only); no clean objects |
| B, set diffusion | 22M | 0.042 (21x) | 0.0218 | **first recognisable objects** |
| **B-large, set diffusion** | **58M** | **0.046 (23x)** | **0.0245** | **best: cleanest objects** |
| fitted strokes (ceiling of this metric) | - | 0.156 | 0.0516 | |
| Detail model (196 strokes on top of the base) | 22M | prompt match 0.2393 -> 0.2410 on generated pictures | | small gain on generated pictures; cannot recover a real picture's own detail |

What the best model draws (`spike/out/full_B58/prompts.png`): smiley emoji with eyes and open mouths, house outlines with roof
and door, airplanes with wings, apples with stems, pizza on plates, sunset over the sea, red bus, bowl of fruit, faces for
"portrait". Reference pictures work too (`references.png`). Not yet: fine object detail, people doing things, clean scenes
with several objects. Bigger model helped (+12% gap); more data and longer training are the obvious next levers.
Artifacts (all local): data `spike/out/final/` (243 shards, captions, CLIP vectors; 1.2 GB) + `spike/out/final_img.tar`
(96.7k cached source images, 3.9 GB); checkpoints `spike/out/full_{A,B22,B58,detail}/ckpt.pt`.

## Stage 0: dry runs (all gates passed)
| Check | Result | Decision |
|---|---|---|
| GPU benchmark on extractor v3 (real 1k mixed set) | RTX 5090 **9,230 img/h** (with detail-fit compile + image prefetch); 5070 Ti 4,060 (batch 20, 16 GB OOM at 40); 3090 2,870 | 4x 5090 for stage 1 ($5.1 per 100k vs ~$4.5 on cheap cards, but 1/3 the GPU-hours) |
| Detail layer gain (base 165 @256 -> +196 detail) | +1.3 to +2.9 dB on every source (photos +1.6, faces +2.4, icons +2.3-2.9, WikiArt +1.6); Met prints +0.75 | keep |
| Florence-2 batched | 27.6 img/s short, 17.5 img/s detailed; accurate content captions (titles like "portrait painting, Rococo" -> "a painting of a man in a blue coat") | keep; strip "The image shows" |
| CLIP filter | caption + stroke-render checks drop little (every item has at least one good caption); 28 of 4,492 captions dropped | keep thresholds 0.22 / 0.15 / 0.18 |
| CLIP near-duplicate filter | BUG: called different QuickDraw doodles / emoji "duplicates" (dumbbell = rifle = syringe), dropped 12% of QuickDraw | dedupe only photo/art sources, cos >= 0.97 |
| Training dry run (pilot data, 15 min each, 300 held-out captions) | B baseline CLIP top-1 0.013, gap 0.0119; **B + slot weights 3,1.5,1: top-1 0.023, gap 0.0122** | adopt weighted loss |
| Met API | blocks bulk/parallel requests (18 rows kept of 1,200 tried) | dropped Met; Cleveland 12k -> 16k (15,676 available) |

## Stage 1: data (4x RTX 5090, $0.469/h each)
Manifest: 80,857 web rows (QuickDraw 24,840; icons 5,341; COCO 15,000 + object crops 7,000 + head crops 3,000; Cleveland 15,676; PD12M 10,000)
+ WikiArt from 14 parquet files (~15.8k).  Shards: box1 0-60, box2 61-121, box3 122-181, box4 182-202 + WikiArt.
Throughput ~9,150 img/h per box, no load failures in the first hour. Peek sheet: `spike/out/peek1/sheet.png`.
Networking lesson: Vast's ssh proxy only accepts account keys; direct box-to-box works box1->box2 and box3/4->box1 but not
box1->box3/4 (time out) - merge uses box1 as hub with pushes from 3/4.

### Stage 1 result (finished 07:45, ~3h15m wall clock)
| Step | Result |
|---|---|
| Extraction | 243 shards; ~9,150 img/h per 5090 sustained; 0 image load failures; 0 errors |
| WikiArt | 14 parquet files -> 15,840 images |
| Florence captions | 38.9 img/s short, 18.7 img/s detailed (batched 32) |
| CLIP filter | kept 99.7% per box (drops mostly Cleveland/PD12M pairs whose text and image disagree); dedupe now photo/art only |
| Merge (box1 hub) | **96,133 trainable images**, 381 cross-box near-duplicates removed |
| By source | QuickDraw 24,840 · WikiArt 15,812 · Cleveland 15,187 · COCO 14,987 · PD12M 9,983 · COCO objects 6,989 · COCO heads 2,994 · icons 5,341 (Openmoji 1,395, Noto 1,337, Twemoji 1,324, Fluent 1,285) |
| Spend so far | $7.43 of the $25.83 credit (incl. stage 0 and the detail demo) |

## Stage 2 + 3 (launched ~07:35, 4x 5090 in parallel)
| Box | Model | Params | Settings |
|---|---|---|---|
| box1 | B (set diffusion) | 22.2M | text + 25% image-vector conditioning, slot weights 3/1.5/1, 150 min cap, patience 15 evals |
| box2 | A (stroke by stroke + canvas) | 16.4M | same conditioning; dropout 0.1, prev-noise 0.1, wd 0.05, CFG 1.5 |
| box3 | B-large | 58.3M | as box1, d 512, 12 layers |
| box4 | detail model (196 detail strokes given the 165 base) | 22.3M | text conditioning, 120 min cap |

### Arm A result (early stop at step 50k, 22.6 min; best val 0.743)
| Metric (500 held-out captions) | Pilot A (6.5k imgs) | **Full A (96k imgs)** | Pilot B | fitted-stroke ceiling |
|---|---|---|---|---|
| CLIP top-1 (chance 0.002 here, 0.010 pilot) | 0.000 | **0.006 (3x chance)** | 0.042 (n=96) | 0.156 |
| CLIP gap (own caption vs others) | 0.0040 | **0.0108** | 0.0102 | 0.0516 |
Sheets: `spike/out/full_A/prompts.png`, `references.png`. A now follows prompts at scene level (was style only): yellow emoji
disc with features, house/cat/airplane sketches, grey engravings, warm portraits, sunset horizon, red/yellow watermelon emoji.
Reference-picture conditioning works: a blue glass icon -> blue glass-like shapes, pencil portraits -> grey sketch portraits,
a red "no bicycles" sign -> red round signs, doodles -> doodles. Still no clean object shapes.

### Stage 3: detail model (22.3M; early stop at step 64,750, 32 min; best val 0.538)
Held-out items, TRUE base strokes, compared with the real picture at 256 px (128 items; `spike/out/full_detail/sweep.json`, `detail_eval.png`):
| | PSNR | CLIP image similarity to the real picture |
|---|---|---|
| base 165 only | 22.90 | 0.773 |
| base + generated detail, best setting (w 3, keep 25% of detail strokes) | 22.21 | 0.771 |
| base + generated detail, all strokes (w 1.5) | 21.45 | 0.768 |
| base + fitted detail (ceiling) | 24.50 | 0.833 |
Reading: the generated detail is plausible texture in the right regions (face lines, sign interiors) but not the actual detail of
that picture - the model only sees base strokes + caption, which do not contain where the eyes/spokes are. Against the real
image it never beats the base. A detail model needs the information that is missing: e.g. condition on the CLIP image vector of
the reference (reference -> detail), or a higher-resolution base. Chain test (prompt -> B base -> + detail) below.

### Arm B-22M result (early stop at step 77k, 33 min; best val 0.478) - the best model so far
| Metric (500 held-out captions) | Pilot B (6.5k) | **Full B-22M (96k)** | Full A | ceiling |
|---|---|---|---|---|
| CLIP top-1 (chance 0.002) | 0.042 (n=96, 4x chance) | **0.042 (n=500, 21x chance)** | 0.006 | 0.156 |
| CLIP gap | 0.0102 | **0.0218** | 0.0108 | 0.0516 |
Sheets: `spike/out/full_B22/prompts.png`, `references.png`. **Objects appear for the first time**: smiley emoji with eyes and
mouths, house sketches with roof and door, cat heads with ears, round red apples (painted and line-drawn), red watermelon
emoji with dark rind, faces with eyes/nose for "close-up portrait", sunset horizon over a dark sea, red vehicles for "red bus",
an orange pile in a bowl for "bowl of fruit". References: glass icon -> glass/cup shapes, "no bicycles" sign -> red rings with
a drawing inside, sketches -> sketches.

### Chain: prompt -> B-22M base -> + generated detail (`spike/out/full_detail/chain_prompts.png`)
CLIP prompt match (20 prompts x 3): base 0.2393 -> base + detail **0.2410** (+0.002, small but positive). Visually subtle at
256 px. Generated detail helps generated pictures a little; it cannot recover a specific real picture's detail (see stage 3).


### B-large (58.3M) result (early stop at step 61k, 59 min; best val 0.4747)
CLIP top-1 0.046, gap 0.0245 (B-22M: 0.042 / 0.0218). Sheet `spike/out/full_B58/prompts.png` is the cleanest of all runs:
clear smiley faces, house outlines, winged airplanes, apples with stems (painted and line-drawn), pizza on plates, sunset over
the sea, red bus. Bigger model = better at this data size; still under 100M params.

## Cost and timeline
| Phase | Wall clock | Spend |
|---|---|---|
| detail demo + stage 0 (benchmarks, pipeline, dry runs) | 02:45-04:20 | ~$1.5 |
| stage 1 data (4x 5090) | 04:25-07:35 | ~$5.9 |
| stage 2+3 training + evals (4x 5090, boxes destroyed as each finished) | 07:35-08:45 | ~$1.8 |
| **Total** | **~6 h** | **$9.23** (credit left $16.60) |

## Next steps (for the user to choose)
1. Use B-large as the main model; try a longer/larger run (more steps at lower LR, or ~90M params) - cheap (~$1-2).
2. More data per concept is still the main lever for objects (e.g. 300k images ~ $20).
3. Detail model: condition it on the reference image vector, or generate detail jointly with the base (one 361-slot model).
4. CPU inference speed (deferred by the user): B-large sampling = 50 steps x 2 (CFG) passes of a 58M model over 165 tokens.
