# 07 - How the stroke generator should be fed, trained and built (research after the failed gate, 2026-10-03)

## 1. What the gate model actually did (and where it departed from 00_DESIGN.md)

- **Input:** one class token, then all stroke tokens generated so far (up to 1,600: 160 strokes x 10 numbers). The model never sees an image, a prompt beyond the class, or the canvas.
- **How earlier strokes matter:** only through attention over their *tokens*. Nothing is rendered during generation, so the model has to "imagine" the canvas from numbers.
- **Loss:** cross-entropy on each token against the one decomposition the optimizer happened to produce (teacher forcing). Bin 63 vs a true 64 counts as wrong as bin 0. There is no image-space signal, and an equally good but different stroke is punished.
- **Departures from the design doc (the gate was a crude first probe):**

| Design doc (00_DESIGN.md) | Gate as run |
|---|---|
| one sequence position per stroke, a small per-stroke head for the fields | one position per number (1,600 positions) |
| spatial sort of strokes within each level | random order (the extractor places strokes by sampling the error map) |
| optional low-res render of the canvas as input | no canvas |
| S0a on QuickDraw **native pen strokes** (no conversion) | QuickDraw re-fitted into 160 optimizer strokes (~30x more strokes than the human drew) |
| ~70K sketches/class is what Sketch-RNN used | 300 per class |

## 2. Evidence from the closest work

| Work | What it feeds the model | Output / loss | Data | Result relevant to us |
|---|---|---|---|---|
| **Collaborative Neural Painting** (CNP, arXiv 2312.01800, CVIU 2025): closest precedent | Class (adaLN-Zero) + optional context strokes; strokes are **continuous 8-dim vectors, one position per stroke**, 400 strokes in 4 coarse-to-fine levels | **Diffusion** (eps-pred, MSE on noise, ~70 steps for 0.56 s) | ~101K decomposed paintings, **~10K per class**, 10 classes (SD images -> Stylized Neural Painting) | No-context FID **30** vs **320 (BERT, per-parameter tokens), 336 (MaskGIT), 444 (MSE regression)**. Authors: the discrete baselines "fail completely ... We impute this to the independent conversion of each stroke parameter to a different token, leading to an input sequence of length Lx8". MDT-S = 6 layers x 576 (~24M params, inferred from width/depth), B = 8 x 768, L = 12 x 768. Hungarian matching used only for the stroke-L1 *metric*. |
| **PaintCopilot** (arXiv 2605.20941, 2026) | **Current canvas** (SD-VAE latent -> adaLN) + window of the last 50 strokes | Next stroke (8 params) by a causal transformer with a **flow-matching** head "to model the full distribution over plausible strokes rather than regressing to a single mean" | 3,000 portrait paintings, 300-400 strokes each (synthetic SBR) | Canvas-in-the-loop AR works for continuation; color is the hardest field. |
| **Learning to Paint** (ICCV 2019) | Agent sees **canvas + target + step number** | 5 Bezier strokes (13 params) per step; reward = drop in WGAN distance; model-based DDPG through a **differentiable neural renderer** | MNIST to ImageNet, 5-400 strokes | Differentiable renderer gave 20x lower L2 than model-free RL. Image-to-painting, not generation. |
| **SwiftSketch** (SIGGRAPH 2025, from 02) | Image features via cross-attention | **Diffusion over 32 strokes in parallel**, importance-ordered slots, loss = L1 on points + **LPIPS on the rendered image** | 35K synthetic pairs | <1 s per sketch |
| **IconShop** (from 02) | Text prefix + tokens | **AR over quantized coordinate tokens works** (FID 4.65) | **300K icons**, xy merged into one token | Per-coordinate AR is not doomed, but it was trained on ~50x our data, with merged xy and native (file) order |
| **Sketch-RNN** (from 02) | latent / class | GMM over pen offsets (relative coordinates) | **70K sketches per class** | Works per class; mode-averaging on hard classes |
| **DIST2Loss** (ICLR 2026, "Teaching Metric Distance to Discrete AR LMs") | n/a | Replaces one-hot targets with distance-weighted soft targets for numeric/coordinate tokens | n/a | Tighter boxes / better VQ generation: the fix for "every wrong bin is equally wrong" if we keep tokens |

## 3. Answers to the user's questions

1. **How is it fed?** Class token + previous stroke tokens only. No image, no canvas.
2. **Is the canvas in the loop?** No. Earlier strokes affect later ones only symbolically. Canvas feedback is what painting *agents* use (Learning to Paint, Paint Transformer, PaintCopilot, SVGBuilder's partial-render encoder). It is not required for set diffusion (CNP has none and works), but for stroke-by-stroke AR it is the natural fix.
3. **Loss / "what defines a correct slope"?** Today "correct" = exactly the bins of the extractor's stroke; no notion of near-miss, and no notion of "the picture looks right". Better options, in order of evidence: (a) continuous outputs with a diffusion/flow loss (CNP, PaintCopilot, SwiftSketch), which models the spread of valid strokes instead of the mean; (b) an image-space loss through our differentiable renderer (SwiftSketch's LPIPS term; Learning to Paint; RLRF), which accepts *any* stroke set that paints the right picture; (c) distance-aware soft targets (DIST2Loss) if we stay with tokens.
4. **Too few parameters?** Not the first-order problem. 4.9M overfit 5.5K sequences within ~4K steps. CNP's smallest is ~24M on ~100K sequences (10K per class); Sketch-RNN used 70K per class. Data per class (300) is the binding constraint at this point; ~10-25M params is the right range once data is ~10K+ per class.
5. **Are we expressing the strokes right?** No:
   - per-number tokens (CNP's diagnosed failure);
   - random within-stage order and reversible curves (fixed in part by sorting: start-point NLL 4.7 -> 3.3);
   - absolute coordinates for points that sit right next to the start point;
   - **inconsistent decompositions**: random initialization means two similar images get unrelated stroke sets, and sketches become 160 overlapping fragments plus white "eraser" strokes instead of ~10 pen strokes.
   The prompt side (one class token) is fine for class-conditioning; text comes later via a frozen CLIP embedding through adaLN (00_DESIGN 5.5).
6. **Other architectures:**
   - **Diffusion / flow over the whole stroke set** (CNP, SwiftSketch, ChiroDiff, StrokeFusion): one continuous vector per stroke, all strokes denoised together in ~20-70 steps. Strongest evidence for exactly this task.
   - **AR, one position per stroke, continuous flow/GMM head, canvas fed back** (PaintCopilot, MAR-style head): keeps true stroke-by-stroke painting; more steps at inference.
   - Masked/parallel discrete (BERT/MaskGIT): failed in CNP with per-parameter tokens.
   - VAE (Sketch-RNN, DeepSVG): older; mode-averaging.
   - Coarse-to-fine levels (VAR-style): compatible with both of the first two (generate level by level).
   - Two-stage (tiny image generator, then a feed-forward painter): would work but abandons "the model thinks in strokes".

## 4. Recommended next probe (~$2-3 of GPU)

Arm A (main bet), CNP-style stroke-set diffusion:
- one slot per stroke, 10 continuous numbers each, level embedding (16/48/96), class via adaLN-Zero, v- or eps-prediction, 50 DDIM steps;
- ~10-25M params;
- direction-canonicalized strokes, spatial order built into the extraction (sort each stage's new strokes at initialization, deterministic grid init instead of random sampling, so decompositions are consistent);
- optional auxiliary render loss later.

Data: fewer classes, many more examples each: e.g. 6 QuickDraw classes x 2,500 = 15K plus the 1K emoji (~3 h on an A40 = ~$1.5, or ~$0.6 on a 4090 / Vast).

Control arm (free data, isolates "model" from "decomposition"): the same diffusion model on **native QuickDraw pen strokes** (no extraction, 70K per class available). If native strokes work and fitted strokes don't, the decomposition is the problem; if neither works, the model is.

Pass bar: a QuickDraw classifier (or eyeballing, first) recognizes most samples of each class.

Sources: CNP https://arxiv.org/abs/2312.01800 (Table 2 and the baseline paragraph, read from the HTML version) · PaintCopilot https://arxiv.org/abs/2605.20941 · Learning to Paint https://arxiv.org/abs/1903.04411 · DIST2Loss https://proceedings.iclr.cc/paper_files/paper/2026/hash/13b45b44e26c353c64cba9529bf4724f-Abstract-Conference.html · Pix2Seq https://arxiv.org/abs/2109.10852 · others as cited in 02_prior_work_sequence_models.md.
