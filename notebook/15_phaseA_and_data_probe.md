# 15. Phase A (model ideas) + rich-caption data probe (2026-10-04 evening)

Spend this phase ~$5.1 (credit 10.71 -> 5.59 incl. setup); project total ~$20.2 (user lifted the $20 ceiling mid-phase).
All Vast instances destroyed. Results: `spike/out/exp14/bxA{1,2,3}/`, probe: `spike/out/probe/` (analysis.json, strokes.json, sheet_*.png).

## Phase A: three ideas on B-22M, 361 slots, cosine LR, batch 256, 60k steps (same as the exp14 baseline)
Code: strokegen `--selfcond`, `--canvas-fb` (64 px render of the current estimate -> small CNN -> 64 extra tokens; base strokes
only, implies selfcond), `--xattn` (cross-attention to CLIP ViT-B/32 per-token features, computed on the fly from cached token ids);
`ClipTokens`, `render_fb_fn`, `sample_B(..., ctx, ctx_mask, fb)`; diag_eval reads the flags from the checkpoint.

| run | params | gap cfg2 | gap cfg3 | gap cfg4 | top-1 cfg2 / cfg3 | train time |
|---|---|---|---|---|---|---|
| baseline B22_361 (exp14) | 22.3M | 0.0253 | 0.0302 | - | 0.042 / 0.058 | ~2.0 h |
| **A1 self-conditioning** | 22.3M | 0.0273 (+8%) | 0.0324 (+7%) | 0.0349 | 0.054 / 0.062 | ~2.3 h |
| A2 self-cond + sees its own render | 22.6M | 0.0284 (+12%) | 0.0325 (+8%) | 0.0351 | 0.052 / 0.076 | ~4.3 h |
| A3 cross-attention to caption tokens | 27.3M | 0.0265 (+5%) | 0.0314 (+4%) | 0.0343 | 0.050 / 0.064 | ~3.0 h |
(B-58M new recipe 165 slots, for scale: 0.0270 / 0.0322 / 0.0348.)

Reading:
- Self-conditioning is a free, consistent win: +7-8%, every source improves (COCO +17%, COCO objects +28%). A 22M model now equals the 58M one.
- The render feedback adds little on top of self-conditioning: +4% at cfg2, equal at cfg3/4, per-source mixed, sheets look alike; costs ~1.9x training time and slower sampling. Not worth it in this form (64 px base-only render). Possible later: higher-res render, feedback only at late steps.
- Cross-attention: +4-5% with today's SHORT captions (expected - the pooled vector already holds "a red apple"). Its real test is long captions (PixelProse/DOCCI); keep for that run.

## Data probe: PixelProse, DOCCI, Open Images + Localized Narratives (local download, GPU only for stroke fitting)
| | PixelProse (cc12m_00, aesthetic>=5, watermark class 1 = clean) | DOCCI | OI + LN |
|---|---|---|---|
| links alive | 73% | hosted (100%) | 100% |
| caption words median / >77 CLIP tokens | 78 / 70% | 114 / 95% | 23 / 4% |
| caption retrieval top-1 from real photo (n=150) | 0.95-0.96 | 0.89-0.93 | 0.54-0.57 |
| **from 361 fitted strokes** (165) | **0.75** (0.57) | 0.54 (0.45) | 0.41 (0.30) |
| gap from 361 strokes | **0.124** | 0.093 | 0.068 |
| stroke PSNR @256 | 23.5 | 23.2 | 24.7 |
Current sources from 361 strokes: COCO 0.61, icons 0.61, PD12M 0.49, COCO objects 0.29, QuickDraw 0.19, Cleveland 0.08, WikiArt 0.04.
PixelProse captions survive strokes better than any current source; ~566k of 907k rows per parquet pass the filters, x73% alive.
Gotcha: PixelProse watermark_class_id 0 = watermark, 1 = clean, 2 = text overlay (first sample used the wrong class).

## Recommended next run (for the user to decide)
~100M B, 361 slots, self-conditioning, cross-attention (+ longer-context text encoder), new recipe, cfg 3-4, 25 steps;
data v2 = PixelProse-heavy realistic photos + recaptioned paintings + less doodle/emoji weight; WikiArt/Cleveland recaptioned or dropped.
