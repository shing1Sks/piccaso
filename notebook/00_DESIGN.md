# Brushwork: a tiny model that paints instead of denoising

*Design doc, 2026-10-03. Status: proposal for review, not built. Supporting research is in `01_prior_work_painting.md`, `02_prior_work_sequence_models.md`, `03_datasets.md`. Measured feasibility numbers are in `spike/`.*

---

## 1. The product, concretely

You type "a lighthouse on a cliff at sunset" on a laptop with no GPU. On a blank canvas, about 16 big, loose strokes appear first: the sky gradient, a dark cliff mass, the sea. Then about 48 medium strokes add the lighthouse shape and the sun, and then a few hundred small strokes add detail. This takes a few seconds and you watch it happen. The result is a painterly, impressionistic image, not a photo.

Because the output is a **list of brush strokes rather than pixels**, it re-renders sharp at any resolution, every stroke can be edited or recolored, and the painting process is the animation. The whole model is a few tens of MB.

## 2. Is this worth doing? (pressure test)

**Where I push back on the original framing.** "Tiny local image generation" is not new on its own merit. Distilled diffusion models already run on phones; Snap's SnapGen (2024, ~0.4B params) is one example (not re-verified here). If the pitch is only "smaller than diffusion", this loses. The size win is real but not unique.

**The new angle is the representation.** Strokes give you four properties that pixel generators cannot give cheaply:
1. **Process.** Generation *is* a painting timelapse. Diffusion has to fake this with extra models such as ProcessPainter or Paints-Undo.
2. **Editability.** Every stroke is a vector object, so you can recolor, delete, or regenerate one level.
3. **Resolution independence.** One 3 KB stroke list renders at 64 px or 4K (shown in the spike).
4. **Tiny output space.** 256 strokes × 11 numbers ≈ 2.8K numbers per image, quantized to ~3 KB. That is why a 20–60M model is plausible.

**What exists.** Every piece exists in isolation, but the combination doesn't. Section 3 has the details.
- Image→strokes painters work and are small. Paint Transformer is 9.07M params (I counted the released weights).
- Prompt→vector generators exist (IconShop, StrokeNUWA, OmniSVG), but they are icons/SVG, monochrome, or billion-parameter LLMs.
- Prompt→sketch models exist for doodles (Sketch-RNN, SketchFlow, SwiftSketch).
- **The closest precedent is Collaborative Neural Painting (CVIU 2025, arXiv 2312.01800).**
  - What it does: a class-conditional *diffusion* transformer that works directly in brush-stroke space. It generates 400 strokes of 8 params each in about 0.56 s on an A100. Its data came from Stable Diffusion images, cut out and decomposed into strokes.
  - Its limits: only **10 animal classes** on blank backgrounds, no text input (left as "future work"), GPU-oriented, and the dataset is unreleased.
  - What it proves: generating directly in stroke space works.

**So the gap is narrower than "nobody has done this", and that's fine.** Nobody has shown stroke generation that is all three of these:
- **broad / open-vocabulary**: hundreds of classes or free text through a frozen CLIP space;
- **tiny and CPU/browser-native**: under ~50M trainable params, seconds on a laptop;
- **painting-as-you-watch**: one stroke at a time, in painter order.

Our contribution is to take CNP's existence proof from 10 classes to open vocabulary while shrinking it. It's an honest research delta, not a moonshot. CNP's released checkpoint is our baseline.

**Honest risks.**
- Quality ceiling: impressionistic only. You accepted this.
- No training data exists in this form. We must synthesize it (§6), and the synthesis method defines the style.
- Nobody has proven that broad coverage works at this size. Evidence from CLIP-conditioned GANs (LAFITE, GALIP at ~75–80M trainable) suggests it can work if we lean on a frozen CLIP text space.

**Verdict:** worth doing as a research project, with a clear and cheap kill-test at Stage 0/1 (§9).

## 3. What exists (summary)

| Bucket | Best examples | What we take from it |
|---|---|---|
| Image → strokes (reconstruction) | Learning to Paint (RL, 2019), **Paint Transformer** (feed-forward set prediction, 2021, Apache-2.0, 9M), Stylized Neural Painting (optimization, 2021), MambaPainter / AttentionPainter (2024–25, faster, GPU) | Stroke parameterizations. **A feed-forward painter trained only on synthetic strokes, so no dataset and no domain bias** (Paint Transformer). Coarse-to-fine brush sizes. |
| Prompt → strokes (generation) | **Collaborative Neural Painting** (CVIU 2025, 10 classes), AttentionPainter's stroke diffusion (per domain), PaintCopilot (2026, portraits), SPIRAL++ | Stroke-space generation works. Their data recipe is: generate images, decompose them, train on the strokes. |
| Prompt → vector via optimization | CLIPDraw, VectorFusion, DiffSketcher, SVGDreamer | High quality but minutes per image with a 1B-param teacher. Useful only as an offline *data generator*. |
| Prompt → vector sequence models | IconShop, StrokeNUWA, OmniSVG, StarVector, Grimoire, SVGBuilder | Quantized coordinates with a **merged-xy token**. Prefix text conditioning. Keep sequences ≤512. AR beat a BERT-style parallel baseline (FID 4.65 vs 35.1). |
| Sketch generators | Sketch-RNN, Sketchformer, ChiroDiff, SwiftSketch, SketchFlow (2026) | Discrete tokens beat continuous regression on complex classes. Importance-ordered fixed slots. CLIP-latent conditioning for zero-shot labels. |
| Small raster generators | LlamaGen-B (111M), VAR, MAR, MaskGIT, LAFITE, GALIP, DF-GAN (19M) | Next-scale (coarse→fine) prediction is fast. A tiny head can model continuous values. **A frozen CLIP gives sub-100M models broad semantics.** |

The full tables, with links, licenses and verification flags, are in files 01–03.

## 4. Feasibility spike: what I measured on your machine

The code is in `spike/` (`stroke.py` is ~120 lines). It has a fixed analytic renderer (quadratic Bézier, soft edge, alpha "over" compositing; zero learned params) and a coarse-to-fine optimizer that fits 16 → 64 → 160 → 256 strokes to an image. Everything ran on CPU only (PyTorch 2.8 CPU, 12 threads).

Test set: 4 standard test images (astronaut, cat, coffee, rocket) at 64 px. Four images is a sanity check, not a benchmark.

**A. Quality vs. stroke count (optimizer, 150 steps per stage). PSNR in dB, higher is better:**

| Strokes | astronaut | cat | coffee | rocket | mean |
|---|---|---|---|---|---|
| 16 | 15.3 | 21.8 | 18.5 | 24.9 | 20.1 |
| 64 | 18.8 | 24.9 | 23.1 | 29.4 | 24.1 |
| 160 | 21.6 | 28.3 | 26.0 | 33.1 | 27.3 |
| 256 | 22.9 | 29.9 | 27.1 | 35.0 | **28.7** |

At 256 strokes every image is clearly recognizable; see `spike/out/*_64_s150.png`. The tiles go: target | 16 | 64 | 160 | 256 strokes | quantized | the same strokes re-rendered at 256 px. The 256 px re-render looks the most "painted", which is the style we want.

**B. Converter speed and quality on your CPU:**

| Converter | sec / image | strokes | mean PSNR | Verdict |
|---|---|---|---|---|
| Optimizer, 150 steps/stage | 270–360 s | 256 | 28.7 | Best quality, far too slow |
| Optimizer, 30 steps/stage | 45–56 s | 256 | 26.4 | Still clearly recognizable. Usable as a **teacher** on a GPU with batching |
| **Paint Transformer** (pretrained, 9.07M) | 1.0–1.5 s (64 px), 2.2–2.6 s (128 px) | ~110 (64 px), ~360 (128 px) | 16.8 (64 px), 18.6 (128 px) | **Fast but unusable at low res.** The cat is unrecognizable at 128 px (`spike/out/painttransformer_128_sheet.png`). It was built for ≥512 px with 32 px patches. |

**C. Tokenizer loss (quantizing the fitted strokes, mean PSNR, unquantized = 28.72):**

| Quantization | PSNR | Loss |
|---|---|---|
| position 64 bins only | 27.25 | −1.47 |
| position 128 bins only | 28.37 | −0.35 |
| width 16 (log) only | 28.05 | −0.67 |
| colour 16/ch only | 28.30 | −0.42 |
| colour 32/ch only | 28.61 | −0.11 |
| alpha 4 only | 28.20 | −0.52 |
| alpha 8 only | 28.62 | −0.10 |
| **all: pos 128 / width 32 / colour 32 / alpha 8** | **28.05** | **−0.67** |

**What the spike decided:**
1. **The 11-number Bézier stroke is expressive enough.** 64–256 strokes is the right budget for 64 px.
2. **Position needs sub-pixel resolution.** Small strokes are ~2 px wide, so position gets 128 bins, not 64. Everything else tolerates coarse bins.
3. **Off-the-shelf Paint Transformer can't be our converter** at the resolutions a tiny model targets. We need our own low-res Stroke Encoder (§5.3), distilled from the optimizer. This is the biggest design consequence of the spike.
4. **CPU conversion caps out around a few thousand images.** At 50 s/image, 2K images is ~1 day on your laptop. Anything larger needs a GPU for the teacher, then the feed-forward encoder.

## 5. Architecture

The design deliberately mirrors latent image models (VQGAN + transformer), with one twist: **the decoder is a fixed renderer with zero parameters, and the latent space is human-readable brush strokes.**

```
                 TRAINING-DATA SIDE                               GENERATION SIDE
 image ──► [A] Stroke Encoder ──► strokes (levels L0..L3)  ──►  [B] Stroke Generator ◄── prompt
           (feed-forward painter,       │                        (tiny transformer,       │
            trained via renderer)        ▼                         one slot per stroke)   [C] frozen text encoder
                                 [R] Fixed Renderer  ◄─────────── strokes ─────────────────┘   (or precomputed class emb.)
                                 (analytic, 0 params, any resolution)
```

### 5.1 The stroke: 11 numbers

| Field | Values | Why |
|---|---|---|
| p0, p1, p2 | 3 control points (x, y) of a quadratic Bézier, in canvas units | Curved strokes with one control point. Quadratic is enough for brush marks (Learning to Paint uses the same family). Cubic adds 2 numbers for little gain. |
| w | width | log-scale; small strokes need finer resolution |
| r, g, b | colour | |
| a | opacity | allows glazing/layering, which gives the painted look |

The spike confirms this family is expressive enough: 256 strokes give a recognizable 64 px portrait/scene.

**Alternative considered: Paint Transformer's 8-param textured oil-brush stamp** (x, y, h, w, θ, rgb). It is the format CNP generated in. It was attractive only because we could reuse PT as the converter, and the spike killed that. Bézier stays because:
- curves capture contours with fewer strokes;
- it maps 1:1 to SVG/Canvas, so the output is a real vector file.

Stamped brush texture and two-colour strokes (SNP's head/tail RGB) are later render-time and format upgrades, not v1.

### 5.2 [R] Renderer (fixed, zero params)

- **Training:** a soft, differentiable rasterizer computes distance-to-curve → sigmoid coverage → alpha compositing (exactly `spike/stroke.py`). It is used to train the Stroke Encoder and, optionally, as an auxiliary loss for the generator.
- **Inference:** the same maths with hard anti-aliased edges, or plain SVG/Canvas2D `quadraticCurveTo` with round caps. It runs in a browser at any resolution. Note: the spike's high-res re-render shows that strokes fitted with soft edges look thinner and sharper when re-rendered, so train and render with the same edge softness *in pixels*.

### 5.3 [A] Stroke Encoder (the "tokenizer"), image → strokes

**Why it must exist:** the spike measured two options. The optimizer is good but takes 50–300 s/image. Pretrained Paint Transformer is fast at ~1 s/image but unrecognizable at 64–128 px. Neither converts ~1M images. A feed-forward encoder trained *for our resolution and budget* does it in milliseconds on a GPU.

- **Design:** a small CNN or ViT (~5–15M) over a 64–128 px image. It outputs **fixed slots per level**: L0 = 16, L1 = 48, L2 = 96, L3 = 96, giving 256 strokes. Each slot has a keep bit. Each level is conditioned on the render of the previous levels (Paint Transformer's residual trick: predict strokes for what is *still wrong*).
- **Training, in three phases:**
  1. Synthetic pretraining: random stroke paintings with known ground truth. No dataset needed (PT's trick).
  2. **Distillation:** about 20–50K images converted by the optimizer (30 steps/stage, batched on GPU) become supervised targets. This gives good stroke *placement and ordering*.
  3. Real images with only the render loss (L1 + perceptual) to close the gap.

  Optionally polish encoder outputs with ~10 optimizer steps.
- **Pass bar:** within ~2 dB of the 30-step optimizer at 256 strokes. That is ≥ ~24 dB mean on the spike set, versus PT's 16.8.
- **Ordering is solved here.** Level = z-order (big first). Within a level, sort slots by a spatial key (Morton order of the centroid). DeepSVG found canonical ordering beats Hungarian matching. The generator then sees a consistent order.
- **Style lever:** the encoder's brush-size schedule and stroke budget define the art style of the whole system. This one knob changes everything downstream.

### 5.4 [B] Stroke Generator (the actual model that ships)

- **Backbone:** a decoder-only transformer, about 8–12 layers, d = 384–512, **≈20–40M params**.
- **One position per stroke, not one per number.** Per-number tokens would be 11 × 256 ≈ 2.8K positions, too slow on CPU. Per stroke, it is 256 positions plus a few level/condition tokens.
- **Tokenizer (from spike C):** 11 fields per stroke, each a small categorical.
  - x, y of each control point: 128 bins.
  - width: 32 bins, log-scale.
  - r, g, b: 32 bins each.
  - alpha: 8 bins.

  Measured round-trip cost: −0.67 dB.
- **Input embedding of a stroke:** the sum of its 11 field embeddings plus level and slot-position embeddings. Merged-xy tokens (the IconShop trick) exist to shorten sequences, and we already have one position per stroke, so separate x and y keeps vocabularies tiny.
- **Output head:** a tiny sequential per-stroke head (~1–2M params). It predicts x0, y0, x1, y1, x2, y2, w, r, g, b, a in turn, each a ≤128-way categorical conditioned on the trunk state *and* the fields already sampled (RQ-Transformer / MAR-head style). Fields of one stroke are strongly correlated (where it is decides its colour), so independent heads would produce incoherent strokes. The 11 head steps per stroke are cheap compared with one trunk step.
- **Conditioning:** the class or text embedding enters via adaLN in every block, plus 2 prefix tokens. Use classifier-free guidance with 10% condition dropout.
- **Decoding, two modes:**
  - **v1, per-stroke autoregressive with a KV cache.** This is the proven path (IconShop, Grimoire). 256 sequential steps on a 30M int8 model is roughly 1–3 s on a laptop CPU (estimate). It paints *in order*, so the timelapse falls out for free.
  - **v2, level-parallel** (VAR × MaskGIT). Each level is decoded in parallel in 4–8 masked steps, giving ~16–32 passes in total. It is faster. It is also a **hypothesis no paper has tested for strokes**, and IconShop's naive parallel baseline failed, so it gets built only after v1 works.

### 5.5 [C] Text encoder

- **Class-conditional stages:** a learned class embedding table, with no encoder at all.
- **Text stage:** a **frozen MobileCLIP-S0 text tower (~42M)** or the CLIP ViT-B/32 text tower (63M). Pooled embedding only. Train with image CLIP embeddings ↔ text CLIP embeddings interchangeably (the LAFITE trick). That means training on *uncaptioned* images is possible, and captions become optional.

### 5.6 Size and speed budget (target)

| Part | Params | Ships? |
|---|---|---|
| Stroke Generator | 20–40M | yes (int8 ≈ 20–40 MB) |
| Text encoder | 42–63M frozen | yes for free text; no for class/preset prompts |
| Stroke Encoder | 5–15M | no, training-data tool only |
| Renderer | 0 | yes (~100 lines, or the browser canvas) |

## 6. Data: what we use and why

The datasets survey (`03_datasets.md`) found **no public dataset of (caption → coloured brush strokes)**. Every 2024–2026 painting-process paper synthesized its own. So the data strategy *is* the Stroke Encoder: convert captioned images.

| Stage | Data | License | Why |
|---|---|---|---|
| 0: CPU proof | **QuickDraw**: 10–20 classes, native strokes (CC BY 4.0). **Emoji**: Twemoji (CC BY) + Noto (Apache) + Fluent (MIT), ~10K coloured, named SVGs | clean | QuickDraw tests stroke grammar and class conditioning without any conversion. Emoji test *colour* with semantic names. On CPU, convert a **~2K subset** with ≤64 strokes (≈1 day at spike speeds). Do the full 10K on a GPU. |
| 1: class-cond., broad | **ImageNet-64** (100 → 1000 classes) + **WikiArt** (27 styles, 11 genres) | non-commercial | Standard FID benchmark at low res. Paintings give the target style. License-clean swap: Art Institute of Chicago CC0 + PD12M's museum part. |
| 2: text-cond. | **PD12M / PD3M** (permissive, PD/CC0 images, S3-hosted, no link rot) + **DiffusionDB** (CC0, often already painterly) + **COCO captions** | clean (COCO images mixed) | The best license-clean caption sources. ~1M images converted at 128 px is ≈ 3–6 GB of strokes. |
| eval / finetune | **FS-COCO** (10K human caption → sketch strokes) | non-commercial | The only human caption-to-stroke data; use it to check ordering and recognizability. |
| optional distill | "impressionist painting of {caption}" from an open T2I model whose license allows training on outputs | check per model | The images are *already* paintings, so stroke conversion is near-lossless and the style is consistent. Likely the biggest quality lever. |

**Avoid:** Noun Project (the API forbids ML training), LAION (link rot, copyright, safety baggage), and FLUX-dev outputs (license-restricted).

## 7. Evaluation

- **Recognizability:** an off-the-shelf classifier's accuracy on rendered outputs (QuickDraw classifier at Stage 0, ImageNet classifier at 64 px at Stage 1). Cheap and interpretable.
- **Distribution:** FID/KID and **CMMD** at 64 px, measured against **both** the real images and the Stroke Encoder's reconstructions. The reconstructions are our true ceiling.
- **Text alignment:** CLIPScore on COCO-val and PartiPrompts/DrawBench subsets. Also GenEval-style colour/object checks, which strokes with explicit RGB should handle well.
- **Efficiency:** quality vs. stroke count; seconds per image on CPU.

## 8. Compute reality

You have no GPU (CPU-only torch).
- **Measured:** the optimizer converter runs at 45–56 s/image (30 steps, 256 strokes, 64 px) on your CPU. So the laptop can convert ~1.5–2K images/day.
- **Stage 0** runs on your laptop: QuickDraw (no conversion) + a ~2K emoji subset, and a ~5–10M model trained in hours.
- **Stage 1+** needs a GPU for the Stroke Encoder and the generator. Free Colab/Kaggle T4s work for Stage 1 at 64 px. Stage 2 wants a rented 24–48 GB GPU for days. I haven't priced this, so treat it as an order-of-magnitude guess.
- **Inference stays CPU/browser**, which is the point.

## 9. Roadmap, each step with a pass/fail check

| Step | Build | Pass if | Kill/redirect if |
|---|---|---|---|
| S0a | Tokenizer + generator v1 on QuickDraw (10 classes, monochrome, fixed width) | QuickDraw classifier accuracy on samples ≥ ~70% | Can't learn doodles: architecture bug, fix before scaling |
| S0b | ~2K emoji → strokes via the optimizer (CPU) → train colour generator | Samples are recognizable emoji-like icons by name | |
| S1a | Batched optimizer teacher on ~20–50K images (GPU), then distill the Stroke Encoder | ≥ ~24 dB mean on the spike set at 256 strokes (optimizer-30: 26.4, Paint Transformer: 16.8), at ms/image | Quality way below the optimizer: convert fewer images with the optimizer and shrink Stage 1 |
| S1b | Class-conditional generator on converted ImageNet-64 (100 classes) | Classifier top-5 on samples clearly above chance; FID vs reconstructions tracked | **The key kill-test.** If 100 classes aren't recognizable at ~40M, the "broad coverage" premise is in trouble; narrow the domain (e.g. landscapes/portraits) |
| S1c | Baseline: run Collaborative Neural Painting's released checkpoint on its 10 classes and compare with our generator at equal stroke budget | Comparable recognizability at a fraction of CNP's size and on CPU | If CNP is far better, revisit AR vs. diffusion-over-strokes |
| S2 | CLIP-conditioned generator on PD12M subset + distilled paintings | CLIPScore competitive with LAFITE/DF-GAN-class models at 64–128 px | |
| S3 | Level-parallel decoding (v2), int8 export, browser demo that paints live | Same quality, ≥3× faster | Keep v1 |

## 10. Open decisions for you

1. **Licence posture.** Research-only (unlocks WikiArt, ImageNet, MMSVG) or commercial-clean from day one (PD12M, AIC, emoji, QuickDraw)? This changes Stage 1 data. My take: use research-only for the kill-test, commercial-clean for anything you ship.
2. **Style target.** Loose impressionist (fewer, bigger strokes, which suits a tiny model) or detailed painterly (more strokes, more compute)? My take: loose. It plays to the strength.
3. **GPU access** for Stage 1+: Colab/Kaggle free tier, or rented?
4. **Name.** "Brushwork" is a placeholder.
