# 16. What we have learned so far (compiled 2026-10-04)

Goal: a 10-100M model that paints a picture as quadratic Bezier strokes for a fixed renderer, from a text prompt (and
optionally a reference picture), small enough to run locally.

## Progress in one line per stage
| stage (doc) | what we did | what we learned |
|---|---|---|
| format (04) | benchmarked stroke formats/fitting | 11-number Bezier with constraints; ~160 strokes is the knee; fit at 128 px then refine |
| gate (06, 07) | tiny GPT over per-number tokens, random stroke order | scribbles: order + per-number tokens are unlearnable; literature (CNP) says diffusion over one vector per stroke |
| v2 probe (08, 09) | anchored slots (165 on 4x4/7x7/10x10 grids, keep bit, fixed direction); arms A (stroke by stroke) and B (set diffusion) | a fixed slot layout makes strokes learnable; B 97% class accuracy, A weaker |
| pilot (10) | 6.5k mixed images, CLIP text vector | B follows scene/style but no objects; both overfit at 6.5k |
| loss check (11) | memorisation test | the loss is sound (both arms redraw training pictures); the problem was data/generalisation; weight coarse slots more |
| full run (13) | 96k images, captions, detail layer | B-58M draws first objects (CLIP gap 0.0245); A far behind; separate detail model failed |
| diagnosis (14) | steps, guidance, recipe, data scaling, 361 joint slots | steps irrelevant; guidance 4 +29%; LR bug fix +12%; data 50->100% only +4%; joint base+detail +7-10% (faces +64%) |
| phase A (15) | self-cond, render feedback, cross-attention; PixelProse/DOCCI/OI probe | self-cond +8% free; feedback ~0 extra at 1.9x cost; cross-attn +5% on short captions; PixelProse captions survive strokes best |

CLIP gap (generated drawing vs its own caption, 500 held-out captions): pilot 0.0102 -> full B-22M 0.0218 -> B-58M 0.0245
-> new recipe 0.0270 -> 22M + 361 slots + self-cond 0.0273 (cfg2) / 0.0349 (cfg4). Top-1 retrieval ~0.07 = 35x chance.
Ceiling of the format (fitted strokes): 0.052 (165) / 0.068 (361). Real photos: ~0.10.

## Cross-cutting lessons
1. Representation decides learnability. Anchored slots + canonical direction turned noise into drawings; 361 slots raised the ceiling for photos.
2. Diffusion over the whole stroke set beats stroke-by-stroke for this task (sees everything at once, revises every stroke at every step).
3. Measure the ceiling first. Fitted-stroke scores split every problem into "format can't show it" vs "model didn't learn it".
4. Cheap things gave the biggest wins: guidance scale (+29%), training recipe (+12%), self-conditioning (+8%), joint detail (+7-10%). Model size helped (+14% for 22->58M); raw data volume least (+4% for 2x).
5. Data: caption-picture agreement beats volume. 35% of our data had partly content-free captions (WikiArt/Cleveland titles, generic face captions). Captions must describe what strokes can show: main objects, colours, layout, style.
6. The model learns in order: palette/style -> layout -> big simple objects -> (not yet) small objects, parts, relations, identity.
7. Validation loss is a training-health signal, not a quality score (many valid drawings per caption -> high floor). Judge by CLIP scores + sheets.
8. Small runs first caught real bugs every time (dedupe merging doodles, inverted watermark filter, missing LR decay, Vast destroy without -y).

## Spend
~$20.2 on Vast in total (full run $9.2, diagnosis $5.2, phase A + probe ~$5.1, earlier probes ~$0.7) + ~$1.2 RunPod earlier.
