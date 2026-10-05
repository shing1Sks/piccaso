# 12. Full run plan: ~100k images, detail layer, clean captions, both arms (2026-10-04)

Status: code ready and CPU-tested; full metadata manifest building on the laptop (free); **nothing launched on GPUs**.
Budget from the user: ~$20 new + ~$5.84 Vast credit. Planned spend ~$13-16, rest is reserve.

## 1. What changes vs the pilot (and why)
| Change | Why (evidence) | Code |
|---|---|---|
| ~100k images (pilot 6.5k) | pilot overfit before fitting train (research/11): too few pictures per concept | build_full_manifest.py |
| **Detail layer**: +196 small strokes on a 14x14 grid fitted at 256 px on top of the 165 base strokes | 128 px / 10x10 finest grid loses faces/texture; base stays a complete v2 picture | detail_level.py, extract_v3.py |
| Fast windowed renderer for the detail layer | naive 256 px render = ~5x the whole base fit; windowed is 16x faster, exact (100 dB vs full render) | detail_level.py |
| Detail strokes kept only if visible AND they reduce error | smoke test: unchecked detail made pictures worse (16.65 -> 16.16 dB); with the rule 16.84 | detail_level.detail_keep |
| COCO object crops (7k) + head/face crops (3k) | objects/faces were tiny parts of cluttered scenes | build_full_manifest.coco |
| 5.3k more icons (Twemoji + Noto + OpenMoji + Fluent, junk filtered) | only ~1.4k clean coloured single objects before | build_full_manifest.icons |
| WikiArt from parquet (14 files) | viewer API rate limit lost 28% in the pilot | fetch_wikiart.py |
| Florence short + detailed captions, batched | museum titles / WikiArt metadata say little about content | caption_v2.py |
| CLIP filter: caption-image match, stroke-render-caption match, near-duplicates | drop pairs a model cannot learn from (LAION used 0.28 on raw alt-text; we use 0.22 for untrusted text, calibrated in stage 0) | clip_pass.py |
| Text cleanup (encoding, "painting painting", "The image shows...") | bugs seen in pilot data | build_full_manifest.clean |
| Importance-weighted slot loss (`--slot-weight`) | equal loss on coarse slots hurts the picture 3-5x more (research/11 probe) | strokegen.py |
| Reference-picture conditioning (`--img-cond`, CLIP image vectors) | prompt + reference images; captionless pictures still teach | strokegen.py, clip_pass.py |

## 2. Data mix (target; the CLIP filter will drop some %)
| Source | Images | Text | License |
|---|---|---|---|
| QuickDraw (345 classes x 72) | 24.8k | 5 templates | CC-BY 4.0 |
| Icons: Twemoji / Noto / OpenMoji / Fluent | ~5.3k | names + tags + style word | CC-BY / Apache / CC BY-SA / MIT |
| COCO full scenes | 15k | 5 human captions | research |
| COCO object crops | 7k | template + Florence short + detailed | research |
| COCO head crops | 3k | template + Florence | research |
| WikiArt (parquet files 0-5,17,22,...,52) | ~15.8k | style/genre/artist + Florence | research |
| Cleveland (Painting/Drawing/Print, CC0) | 12k | title/type/artist + Florence | CC0 |
| Met (paintings, drawings & prints, modern) | 4k | title/artist + Florence | CC0 |
| PD12M | 10k | own caption (cleaned, first sentence + long) + Florence short | CC0 |
| **Total** | **~97k** | | |
Verified at 1% scale (6.2k rows incl. all icons): every source loads; crops checked visually (`spike/out/full/crop_check2.png`).

## 3. GPU choice (live Vast prices 2026-10-04, single GPU, reliability > 0.995, CUDA >= 12.8)
Only the 4090 is measured on this workload (v2 base: 8,144 img/h). Others are scaled from fp32 TFLOPS^0.4 x bandwidth^0.6
(that formula reproduces the measured A40/4090 ratio). v3 assumed 1.6x slower than v2 (detail layer) - **stage 0 measures both**.

| GPU | rel. speed | v3 img/h (est.) | $/h min (median) | reliable offers | $ per 100k (min / median) | GPU-h per 100k |
|---|---|---|---|---|---|---|
| RTX 5090 | 1.55 | ~7,900 | 0.469 (0.642) | 23 | 5.9 / 8.1 | 12.6 |
| RTX 4090 (measured base) | 1.00 | ~5,100 | 0.347 (0.476) | 22 | 6.8 / 9.4 | 19.6 |
| RTX 5070 Ti | 0.72 | ~3,700 | 0.161 (0.262) | 6 | 4.4 / 7.1 | 27.2 |
| RTX 3090 | 0.68 | ~3,500 | 0.161 (0.255) | 14 | 4.6 / 7.3 | 28.8 |
| RTX 5080 | 0.83 | ~4,200 | 0.308 (0.322) | 7 | 7.3 / 7.6 | 23.6 |
| A100 SXM4 | 0.86 | ~4,400 | 0.667 | 3 | 15.3 | 22.9 |
Reading: per image, cheap 3090 / 5070 Ti hosts and 5090s are all ~$5-7 per 100k; datacenter cards are 2-3x worse.
5090s have the most supply and need the fewest hours (4 x 5090 ~ 3.2 h wall clock). Plan: benchmark 5090 vs 3090 vs 5070 Ti
for 10 min each in stage 0, then rent 4 of the best $/img. Use the `-runtime` image (boots in ~1 min; the devel image hung on
two hosts) + apt gcc for torch.compile (`/tmp/rp/new_v3.sh`).

## 4. Stages, gates and cost
| Stage | What | Gate to continue | Est. cost |
|---|---|---|---|
| **0 dry runs** | (a) GPU benchmark x3 types on v3; (b) v3 extraction of the 1% manifest (~1k mixed): speed, dB base vs base+detail, sheets of faces/paintings; (c) WikiArt one parquet file; (d) Florence batched on those ~1k: img/s, read captions; (e) CLIP pass: score distributions + dropped examples, set thresholds; (f) B pilot-data runs: slot-weight vs none, img-cond on/off (20 min each); A one run | user looks at sheets + numbers | ~$1.0 |
| **1 data** | 4 GPUs: WikiArt fetch, extraction (each box a shard range), Florence on its own images, CLIP pass; merge on laptop | per-source dB + kept % as in stage 0 | ~$7-9 |
| **2 train** | B and A, text + image conditioning, weighted loss, early stop; eval: CLIP retrieval vs fitted ceiling, prompt sheets, reference-image sheets | B beats pilot clearly (0.042 top-1) | ~$2.5 |
| 3 optional | detail model: adds the 196 detail strokes on top of base | if money left | ~$1.5 |
| overhead | boots, dead hosts, idle while checking | | ~$1.5 |
| **Total** | | | **~$13-16** |

## 5. Risks / open questions answered in stage 0
- Detail layer speed on GPU (16 sequential groups + keep loop): if v3 is >2.5x slower than v2, shrink detail to 50 steps or 12x12.
- Florence batching speed unknown (estimate tens img/s short).
- CLIP thresholds: 0.22 / 0.15 / 0.18 are first guesses; tune from the report's dropped examples.
- The Met API is slow (~2.6 s per kept object), hence 4k Met / 12k Cleveland; the build runs on the laptop for free.
- Licenses: COCO and WikiArt are research-only; a release model would train on the CC0/permissive subset.

## 6. Detail-layer demo (RTX 4090, 2026-10-04, ~$0.05) - `spike/out/demo3/detail_sheet.png`
| image | base 165 @256 | + detail (196 slots) | gain |
|---|---|---|---|
| COCO head crop (beard, sunglasses) | 24.5 dB | 27.2 dB | +2.7 |
| COCO head crop (laughing) | 25.8 | 27.3 | +1.5 |
| COCO skateboarder | 24.5 | 26.5 | +2.0 |
| Fluent icon (bandage) | 31.8 | 35.7 | +3.9 (the small dots appear) |
| WikiArt pen drawing (windmill) | 20.0 | 20.8 | +0.8 (hatching is finer than any stroke) |
| Cleveland manuscript page | 22.9 | 23.5 | +0.6 (text: hopeless; filtered out in the full run) |
Visible: mouths/teeth, eyes, beard texture, glasses shape, icon dots. Not fixed: 1-px pen hatching, text.
Speed at batch 40 WITHOUT torch.compile: base 115 s + detail 41 s per 120 images. With compile the base is ~2.2x faster, so v3 is
~1.8x the cost of v2 (~4,600 img/h on a 4090) - close to the 1.6x assumed; detail fit can also be compiled (stage 0).

## 7. Final end-to-end runbook (agreed version; costs use the measured v3 speed)
GPU: RTX 5090 on Vast (~$0.47-0.55/h reliable; est. ~7,100 v3 img/h = 1.55x the measured ~4,600 on a 4090), runtime image.
| Stage | Boxes | Wall time | Est. cost | Gate / stop rule |
|---|---|---|---|---|
| 0a GPU benchmark (5090 / 3090 / 5070 Ti, 10 min each on v3) | 3 x 1 | 20 min | $0.15 | pick best $/img |
| 0b 1k-image v3 + 1 WikiArt file + Florence + CLIP pass | 1 | 30 min | $0.25 | detail >= +1 dB on photos; captions sane; drop rate 5-20% with sensible drops |
| 0c training dry runs on pilot data: B base vs slot-weight vs img-cond, A once | 2 | 25 min | $0.45 | weighted not worse than baseline |
| -- user review of stage 0 | | | | user go |
| 1 data: WikiArt fetch, v3 extraction (shard ranges), Florence, CLIP pass, merge | 4 x 5090 | ~4-5 h | $7.7-8.8 | per-shard monitoring; pause + ask if projected > $10 |
| 2 train B and A (text + image cond, weighted loss, early stop) + eval | 2 x 5090 | ~3-4 h | $2.6 | target CLIP top-1 >= 2x pilot (0.042); prompt sheets |
| 3 detail model (adds 196 strokes on top of base) | 1 | ~2 h | $0.9 | detail sheets |
| Total | | ~1.5 days incl. reviews | ~$12-13.5 of $25.78 | ~$12 reserve |
