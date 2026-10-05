# 04 — Extraction pipeline: methods, stroke budget, resolution, hardware, cost

Date: 2026-10-03. Everything below tagged **[measured]** was run in this project on the pilot set (spike/PILOT_AUDIT.md, spike/out/sweep_*).
**[research]** comes from papers or pricing pages read by me or a research agent; **(U)** means unverified. Costs are USD.
Measurement caveat: 16 images (4 per source: emoji, QuickDraw, COCO, WikiArt) for the sweeps, 40 images for the final recipes. Small samples, no error bars. Treat differences under ~0.3 dB as noise.

## TL;DR

1. **Stroke format:** 11-param Bézier, but the fit should be constrained: minimum width 0.008 (1 px at 128), control points kept on the canvas, alpha fixed to 1.0. This costs ~0.1 dB and removes the 39% hairline strokes, 30% off-canvas strokes and 32% saturated alphas found in the audit. Strokes become 10 numbers.
2. **Fit resolution:** fit at **128 px**, then jointly fine-tune all strokes at **256 px for 40 steps**. That beat a full fit at 256 px (26.2 vs 25.1 dB PSNR, 0.272 vs 0.291 LPIPS) at 1/7 of the cost. Fitting at 64 px does not transfer upward (20.2 dB at 256 px).
3. **Stroke budget:** 160 is the knee (most of the quality), 256 is the safe default for photos and paintings. Beyond 256 gains are small (+0.35 dB for 352).
4. **Speed:** 4 curve segments instead of 8, plus `torch.compile`, plus a lighter schedule for the fine stages gives **5,825 images/hour on an A40 (160 strokes)**, 3.9x the original pilot. The quality recipe (256 strokes + 256 px refine) runs at 1,350/hour.
5. **Hardware:** consumer cards (RTX 4090/5090) win on price per image; A100/H100/serverless do not. Measured: 4090 = 1.63x an A40 at 128 px.
6. **Cost per 100k images at 128 px:** about $8 (fast recipe, A40), $36 (quality recipe, A40). On a community 4090 at $0.34/hr, estimated $3.6 and $15.5.

## 1. Measured results

### 1.1 Cleaning the targets (A40, 128 px fit, 256 strokes, quality scored by re-rendering at 256 px vs the 256 px original)

| Config | img/hr | ref PSNR | ref LPIPS | strokes < 1 px | off-canvas | alpha = 1 |
|---|---|---|---|---|---|---|
| baseline | 1913 | 24.10 | 0.302 | 39% | 30% | 32% |
| min width 0.008 | 1956 | 24.09 | 0.305 | 0% | 30% | 33% |
| keep on canvas | 1965 | 24.18 | 0.299 | 40% | 0% | 32% |
| both | 1970 | 24.12 | 0.307 | 0% | 0% | 32% |
| alpha fixed 1 | 1948 | 23.92 | 0.298 | 39% | 29% | 100% |
| all three ("clean") | 1891 | 24.01 | 0.305 | 0% | 0% | 100% |

### 1.2 Speed levers

| Lever | img/hr | ref PSNR | ref LPIPS | Verdict |
|---|---|---|---|---|
| 150 steps/stage (baseline) | 1917 | 24.10 | 0.302 | |
| 75 steps, lr 0.02 | 3771 | 23.62 | 0.316 | costs 0.5 dB |
| 50 steps, lr 0.03 | 5347 | 22.86 | 0.339 | costs 1.2 dB |
| 30 steps, lr 0.05 | 8885 | 21.01 | 0.393 | too lossy |
| 8 curve segments (baseline) | 1917 | 24.10 | 0.302 | |
| 6 segments, same K at eval | 2561 | 24.11 | 0.301 | free, +34% |
| 4 segments, same K at eval | 3099 | 24.02 | 0.300 | free, +62% |
| step schedule 150/150/100/60 (clean, K6) | 3723 | 23.69 | 0.316 | −0.3 dB for +50% vs clean K6 |
| batch size 4–16 vs 1, no compile | 645 vs 475 | | | small gain |
| `torch.compile` of the distance field | 2100 vs 645 | | | 3.3x (A40, 128 px) |

(An early segments row used 8-segment rendering at eval time for K=4/3 fits and showed a spurious 0.2–0.7 dB loss. The "matched" rows above are the valid ones. The K=3 row was also slow, 1514 img/hr, cause not investigated.)

### 1.3 Fit resolution (A40, 256 strokes, ref = 256 px)

| Fit size | img/hr | ref PSNR | ref LPIPS |
|---|---|---|---|
| 64 | 2497 | 20.17 | 0.350 |
| 96 | 1127 | 22.64 | 0.314 |
| 128 | 1917 | 24.10 | 0.302 |
| 192 | 264 | 25.11 | 0.292 |
| 256 | 147 | 25.11 | 0.291 |
| 128 fit + joint refine at 256, 20 steps (clean, K6) | 1016 | 25.92 | 0.277 |
| 128 fit + joint refine at 256, 40 steps (clean, K6) | 1100 | **26.17** | **0.272** |

Caveat on refinement: it breaks the coarse-to-fine prefix property a little. Prefix PSNR at 16/64 strokes drops from 18.1/21.3 to 17.2/20.8, while the 160-stroke prefix improves (23.2 to 23.9). If the generator relies on level-wise training, test a prefix-aware refine loss.

### 1.4 Stroke budget (fit at 128, 448 strokes, prefix = shorter budget, ref PSNR at 256 px)

| Strokes | 16 | 64 | 160 | 256 | 352 |
|---|---|---|---|---|---|
| fit at 128 | 18.4 | 21.4 | 23.3 | 24.1 | 24.5 |
| fit at 192 | 18.2 | 22.1 | 24.3 | 25.2 | 25.7 |

Stage gains at 128 px (fit-resolution PSNR): +12.0 (first 16), +3.7 (next 48), +2.7 (next 96), +1.3 (next 96).

### 1.5 Final candidate recipes (A40, all 40 pilot images; sheets in spike/out/final_*/)

| Recipe | Settings | img/hr | ref PSNR | ref LPIPS |
|---|---|---|---|---|
| **fast** | 160 strokes, 4 segments, clean, steps 150/150/100, no refine | **5825** | 23.63 | 0.316 |
| **quality** | 256 strokes, 4 segments, clean, steps 150/150/100/60, refine at 256 for 40 steps | 1350 | **26.60** | **0.255** |

Visually: both are faithful on emoji and sketches. On photos the fast recipe is noticeably more abstract (loses small people and text). The quality recipe is sharper everywhere.

### 1.6 GPUs (same sweep script)

| GPU | tier / price | 128 px baseline | clean | 256 px fit |
|---|---|---|---|---|
| A40 48 GB | secure $0.49 | 1913 | 1891 | 147 |
| RTX 4090 24 GB | secure $0.74 | 3113 | 3211 | 672 |

The 4090 is 1.63x the A40 at 128 px, matching the research estimate (1.45–1.79x). At 256 px it is 4.6x. I did not investigate why (a large L2 cache is my guess, unverified).
Price per image: the secure 4090 (about 4,200 img/$) is no better than the A40 (about 3,900 img/$) on the baseline. It only wins clearly at community/marketplace prices.

## 2. Research findings

### 2.1 Faster fitting methods (my own search, primary-source fetches)
- **Bézier Splatting (arXiv 2503.16424):** renders open curves by sampling 2D Gaussians along them (x scale from neighbor spacing, y scale = stroke width, one opacity and an RGB per curve, alpha blending with depth order). Reports 19.6x faster forward and 149.2x faster backward than DiffVG at 2,040x1,344 with 2,048 curves on an A100, and 3 min 24 s vs 46 min 36 s for a full 2,048-curve fit. The comparison is against DiffVG, not against a compiled dense renderer like mine, so the gain over my current code is unknown. Code location not confirmed (project page: xiliu8006.github.io/Bezier_splatting_project).
- **Amortized initialization works for Gaussian image fitting:** Instant-GaussianImage (arXiv 2506.23479) reports a 2 s network-initialized fit beating a 20 s random-initialized one (over 10x on 2 of 3 images, 5x on the third, A100). Fast 2DGS (arXiv 2512.12774) reaches 40 dB in 2 s of fine-tuning vs 12.7–30.6 s for optimization-only methods on Kodak at 512x512. Both needed substantial training (Fast 2DGS: about 19 GPU-hours; Instant-GI trains on pseudo-labels from 50k-iteration fits). Neither is for stroke primitives, so transfer to our format is untested.
- **Feed-forward painters** (Paint Transformer, AttentionPainter, MambaPainter) are 100x faster but, at our sizes, clearly worse in my measurement (LPIPS 0.37 vs 0.15 at equal budget on 4 images).
- Not covered, because the research agent for this was cancelled by the interrupt: Hertzmann/Im2Oil-style heuristic init, superpixel init, per-stroke early stopping.

### 2.2 Budget, resolution, precision (research agent; read through a summarizing fetch tool, not the paper text)
- Published stroke counts: 200–400 (Learning to Paint, SNP, CNP, PaintCopilot), up to 4000 (AttentionPainter), 16–40 for abstraction (SPIRAL++, CLIPasso, SwiftSketch). CNP, the closest precedent, uses 400 strokes and a continuous 8-param diffusion model with 10 classes. No paper reports generative quality against stroke count (a real gap, and my experiment 3 below addresses it).
- Resolution: SNP and Learning to Paint fit at 128 px and render at higher resolution; nobody reports the loss from fitting low and rendering high (I measured it above).
- Precision: IconShop 100x100 grid with merged xy tokens, OmniSVG 200x200, DeepSVG 8-bit; continuous heads (CNP, PaintCopilot, MAR: 3.50 vs 8.79 FID for diffusion head vs discrete CE, on image latents). Most stroke generators drop alpha.
- Cleaning: Intelli-Paint (60–80% fewer strokes via L1 importance), SuperSVG visibility flag, a 2026 stroke-planning paper (arXiv 2604.02752, 30–50% fewer strokes by re-seeding invisible strokes). No evidence either way on whether pruning helps a generator learn.
- Ordering: keep stage order (coarse to fine); sort spatially within a stage. Low confidence.

### 2.3 Hardware and price (research agent; RunPod numbers from the live catalog, others from aggregator sites and unverified)
- Workload is FP32 elementwise plus memory traffic, not tensor-core work, so consumer cards (4090, 5090) are the best value; A100/H100/H200/B200 cost 2–4x more per image; MI300X is a porting risk; serverless is 1.5–2.5x the pod price.
- Stopped pods can come back with zero GPUs (this happened to me twice). Terminate instead and keep results off the pod.
- Cheaper marketplaces (estimates, (U)): Vast.ai 4090 about $0.32–0.43/hr on demand, Salad 4090 $0.16–0.33, TensorDock 4090 $0.16–0.37. Interruptible instances need sharded, resumable jobs.
- Free pilots: Kaggle (about 30 GPU-hours/week, T4s) could cover small tests (about 30k images/week estimated, (U)).
- Availability in this account's region changed hourly: community 4090/5090 and 4000 Ada were out of stock for most of today; the secure 5090 was out of stock too.

## 3. Cost model (measured rates; other GPUs scaled by the measured 1.63x for 4090, price from the catalog)

| Job | A40 secure $0.49 (measured) | 4090 secure $0.74 (est.) | 4090 community $0.34 (est., when in stock) |
|---|---|---|---|
| fast recipe rate | 5,825 img/hr | ~9,500 | ~9,500 |
| 100k images | $8.4 | $7.8 | $3.6 |
| 1M images | $84 | $78 | $36 |
| quality recipe rate | 1,350 img/hr | ~2,200 | ~2,200 |
| 100k images | $36 | $34 | $15.5 |
| 1M images | $363 | $336 | $155 |

Add compile warm-up (about 1 minute per new input shape) and idle time. These are fitting costs only; the GPU time to train the generator is separate.

## 4. Recommendation

**Recipe by data type:**
- Flat or high-contrast sources (emoji, sketches, QuickDraw): the **fast** recipe is enough (emoji and sketches look identical at 160 strokes).
- Photos and paintings: the **quality** recipe, or fast plus a 256 px refine step if sharpness matters.
- Always store the strokes (resolution independent). Keep the constraints (min width 0.008, on canvas, alpha 1.0).

**Tokenization implication:** with alpha fixed there are 10 numbers per stroke; widths are at least 1 px so the width range is narrower. Use at least 128 position bins and 32/32 for width/color, or a continuous head.

## 5. What to do next, in order, when credits are added

1. **Stage 0 extraction** with the fast recipe: ~10k emoji plus ~20 QuickDraw classes. About 2–4 GPU-hours on an A40 (about $1–2).
2. **Compound the speedups** (about $1): refine from a coarser fit (64 → 128 → 256), 4-segment refine, and a prefix-aware refine loss to fix the prefix degradation. Expect a further 1.5–2x.
3. **Amortized initializer** (about $3–5): generate 5–10k quality fits, train a small image-to-strokes network, then use it to initialize and fine-tune for 30–50 steps. Literature suggests 5–10x on Gaussians; untested for strokes.
4. **Check the rasterizer ceiling:** a tile-local or Bézier-splatting-style kernel. Biggest possible win (the dense per-pixel distance is the remaining waste), highest effort.
5. **Generator experiments** (about 2–3 GPU-hours, from the budget research): token head (discrete 64/16/16/4 vs 128/32/32/8 vs continuous), truncated-prefix evaluation (64/160/256 strokes), within-stage order (random vs spatial).
6. **Scale**: community 4090/5090 or Vast/Salad with sharded, resumable jobs; keep results off the pod.

## 6. Risks and open questions
- Small samples (16 and 40 images, no seeds repeated).
- The quality numbers are PSNR/LPIPS against the original; none of this measures whether a generator trained on these strokes learns well. That is the real test.
- The refine step needs the 256 px source image for every training image.
- Billing: the billing API lags (it showed only $0.32 for the day when I checked). My estimate for this round, from pod run times, is about $1.0–1.1 (A40 about 1.4 h at $0.49, secure 4090 about 0.45 h at $0.74), so roughly $1.4 of the ~$2.5 should remain. Check the console balance, since these are estimates.
- An unexplained standard-storage charge of about $0.10–0.12/day appeared on the account before any of my pods existed (no network volumes are listed).
