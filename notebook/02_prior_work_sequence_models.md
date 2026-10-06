# 02 — Prior work: generative models over strokes / vector primitives / discrete tokens, and lessons from small image generators

Survey date: 2026-10-03. Scope: models that *emit primitives* (sketch strokes, SVG paths, brush strokes) and the small or efficient raster-token generators whose tricks carry over to a 10–100M-parameter stroke generator that runs on CPU.

**How this was verified.** Numbers marked as verified were checked against the arXiv/CVF/NeurIPS text or the official GitHub README during this survey. Items marked **(unverified)** come from memory or a secondary source and were not confirmed. FID values are not comparable across rows: the papers use different feature extractors (IconShop computes FID on CLIP features, ChiroDiff uses a QuickDraw-trained Inception) and different datasets.

---

## 1. Master table: stroke, sketch and vector generators

| Work | Year / venue | Link | Representation and tokenization | Seq length | Model size | Conditioning | Data | Reported quality | Code / license |
|---|---|---|---|---|---|---|---|---|---|
| **Sketch-RNN** (Ha & Eck) | 2017 arXiv / ICLR 2018 | [1704.03477](https://arxiv.org/abs/1704.03477) | Continuous **stroke-5** points (dx, dy, p1, p2, p3); offsets are modelled by a **bivariate GMM with M=20**, pen state by a 3-way softmax. One step per pen point. | Up to a few hundred points (Nmax per class) | Enc bi-LSTM 512, dec HyperLSTM 2048, z=128 (verified). Total params not stated; a few M **(unverified)** | Latent z (VAE) or unconditional; trained per class or on small class mixes. A 75-class model was also trained. | QuickDraw: 70K train sketches per class | Good for single classes. On complex classes the paper says outputs become "smoother, more circular… an averaging of many sketches", which is mode-averaging. | magenta (TF), Apache-2.0 **(unverified)** |
| **Sketchformer** | CVPR 2020 | [2002.10381](https://arxiv.org/abs/2002.10381) | Three variants were compared: continuous stroke-5; **Tok-Dict**, a k-means dictionary of K=1000 relative pen moves plus 4 special tokens; and **Tok-Grid**, a 100×100 absolute grid (one token per cell). | Median 30–75 strokes after RDP simplification | Small: 4 MHA blocks, FFN dim 512 | Class supervision (embedding work, not text-to-sketch) | QD-2.5M, **all 345 classes** | Tok-Dict was best on classification and on reconstruction and interpolation of long, complex sketches, ahead of continuous stroke-5 and Sketch-RNN. | Adobe / USP. Code exists **(unverified license)** |
| **CoSE** | NeurIPS 2020 | [paper](https://proceedings.neurips.cc/paper/2020/hash/723e8f97fde15f7a8d5ff8d558ea3f16-Abstract.html) | Each stroke is an ordered point sequence encoded to a fixed latent. The drawing is an **unordered set** of stroke embeddings. A relational model predicts the next stroke's position and embedding (GMM). | Strokes per drawing | n/a | Prior strokes (autocompletion) | DiDi diagrams | Built to be **stroke-order invariant**. | [eth-ait.github.io/cose](https://eth-ait.github.io/cose) |
| **DeepSVG** | NeurIPS 2020 | [2007.11301](https://arxiv.org/abs/2007.11301) | Hierarchical: a set of N_P paths, each with N_C commands. Command = type plus 6 args **quantized to 8 bits (256 bins, +1 for unused)**, embedded and summed. | N_P × N_C fixed slots (e.g. 8×30 **(unverified)**) | Transformer VAE, d_E=256; the checkpoint is 41 MB, so about 10M params **(inferred)** | Latent z; class label for fonts | SVG-Icons8: 100K icons, 56 categories | Feed-forward (non-AR) decoding beat one-stage AR. **Ordered assignment (lexicographic by path start) beat Hungarian matching.** | [alexandre01/deepsvg](https://github.com/alexandre01/deepsvg), MIT |
| **Im2Vec** | CVPR 2021 | [2102.02798](https://arxiv.org/abs/2102.02798) | Each path is a deformed unit circle: k cubic Bézier segments sampled on the circle, then a 1D circular CNN. A bi-LSTM emits T path codes. **Trained only on raster images** through DiffVG plus differentiable compositing. | T paths fixed at the dataset maximum | Small CNN/RNN VAE **(size unverified)** | Latent z | Emoji, fonts, icons, MNIST | Better reconstructions than SVG-VAE and DeepSVG without vector supervision. Reconstructions need only 7–8 segments per path with adaptive sampling. | [project page](http://geometry.cs.ucl.ac.uk/projects/2021/Im2Vec/) **(license unverified)** |
| **Paint Transformer** | ICCV 2021 | [2108.03798](https://arxiv.org/abs/2108.03798) | Strokes are rotated rectangles (x, y, w, h, θ, RGB). **DETR-style set prediction** with N learnable queries and a keep/drop bit per stroke, matched to targets with the Hungarian algorithm. Runs over **K coarse-to-fine scales** with patches, each scale conditioned on the current canvas. | N strokes per patch per scale | Small CNN + transformer **(size unverified)** | Target image (image-to-painting) | Self-generated from random synthetic strokes, no dataset needed | Near-real-time 512² paintings, faster than optimization or RL. Stroke order is left to an image-level loss. | Huage001/PaintTransformer (Paddle/PyTorch) **(unverified)** |
| **IconShop** | SIGGRAPH Asia 2023 (TOG) | [2304.14400](https://arxiv.org/abs/2304.14400) | Paths flattened with `<BOP>`. M/L/C command tokens. Coordinates on a **100×100 grid, each (x, y) merged into one token (x·W+y)**, which "roughly halves" the sequence. Vocab 10,007. Extra x and y embeddings are added to every coordinate token. FIM-style `<Mask>` lets it do infilling. | **512 icon tokens + 50 text tokens** | 12-layer causal decoder. Params not stated **(unverified)** | **Prefix**: frozen word embeddings from a small pretrained BERT (Turc et al.), concatenated before the SVG tokens. Trained with 60% keywords, 30% ChatGPT sentences, 10% blank. | FIGR-8-SVG, monochrome: 1.1M after the ≤512 filter, 300K used | Text-guided FID (CLIP features) **4.65** vs DeepSVG+GAN 12.01 and BERT-style masked 35.10. CLIP score 25.74. About 1.38 s per icon on an A100. | [kingnobro/IconShop](https://github.com/kingnobro/IconShop) **(license not stated)** |
| **ChiroDiff** | ICLR 2023 | [2304.03785](https://arxiv.org/abs/2304.03785) | Continuous point sequence with pen bit, modelled as **velocities**. **Non-autoregressive DDPM** over the whole sequence, T=1000, with DDIM available. | Fixed resampled length | Tiny bi-GRU: 3 layers, D=128 for QuickDraw (<1M params **(inferred)**) | Unconditional or class; conditional sampling | VMNIST, KanjiVG, QuickDraw (7 classes) | On par or better than AR baselines on FID. Supports healing and abstraction. | [ayandas.me/chirodiff](https://ayandas.me/chirodiff) |
| **SketchKnitter** | ICLR 2023 | [OpenReview](https://openreview.net/forum?id=4eJ43EN2g6l) | Diffusion over **stroke-point locations plus pen states**. Generation is treated as reverse deformation of scattered points. An auxiliary RNN adds a recognizability signal. | Fixed number of points | UNet backbone (size not stated) | Unconditional; conditional refinement | QuickDraw, **10 classes** | Competitive with S=30 shortcut sampling steps instead of 1000. | [XDUWQ/SketchKnitter](https://github.com/XDUWQ/SketchKnitter), MIT |
| **DiffSketcher / VectorFusion / SVGDreamer** (optimization family) | NeurIPS 2023 / CVPR 2023 / CVPR 2024 | [2306.14685](https://arxiv.org/abs/2306.14685) | Bézier strokes optimized per prompt with SDS from Stable Diffusion | n/a | Uses SD (~1B) at test time | Text via SD | none | High quality, but minutes per image. **Not deployable on-device**, though usable as a **data generator**. | DiffSketcher has a project page |
| **StarVector** | arXiv 2023 / CVPR 2025 | [2312.11556](https://arxiv.org/abs/2312.11556) | Raw SVG code tokens (code-LLM tokenizer). An image encoder feeds visual tokens into a StarCoder LM. | Up to 8K (1B) and 16K (8B) context | **1B and 8B** | Image (vectorization) or text | SVG-Stack, 2.1M | DinoScore 0.926 (1B) / 0.966 (8B) on SVG-Stack. Strong vectorizer. | [joanrod/star-vector](https://github.com/joanrod/star-vector), Apache-2.0 |
| **StrokeNUWA** | ICML 2024 | [2401.17093](https://arxiv.org/abs/2401.17093) | **VQ-Stroke**: residual VQ, **codebook 4096, dim 512, 2 residual levels**. Command sequences are downsampled 2–4×, compressing SVG code to **6.9%** of its length. An LLM emits the stroke tokens. | Max 512 | **Flan-T5 3B**: encoder frozen, decoder fine-tuned | Text (T5 encoder) | FIGR-8-SVG, ~740K | Claimed FID 6.51 and CLIPScore 17.99, better than IconShop. About 19 s per SVG, "94× faster" than optimization. | Microsoft. Code not found **(unverified)** |
| **Vector Grimoire (GRIMOIRE)** | arXiv 2024 / ICML 2025 | [2410.05991](https://arxiv.org/abs/2410.05991) | **Visual Shape Quantizer**: a raster patch is encoded (ResNet-18, 15.4M) to **FSQ codes, L=[7,5,5,5,5] = 4,375 codes**, and decoded (0.8M) to Bézier segments plus **stroke width and color**. FIGR-8 uses 4 segments and 2 codes per shape. Position is one token on a 256×256 grid. Sequence: `text, <BOS>, (pos, code)…`. | Context 512 | 12–16 decoder blocks, 8 heads (params not stated) | **Prefix**: pretrained BERT encoder | MNIST, fonts (~2M), FIGR-8 subset (427K, 75 classes) | Beats Im2Vec on reconstruction. Better CLIPScore than vector-supervised baselines on fonts and FIGR-8. **Trained from raster supervision only.** | CC BY-NC-ND 4.0 per paper; code "upon acceptance" **(status unverified)** |
| **SVGBuilder** | arXiv 2024 / AAAI 2025 **(venue unverified)** | [2412.10488](https://arxiv.org/abs/2412.10488) | **Component library**: each element is **7 tokens** (BOS, component-id, offset-x, offset-y, scale, R, G, B), placed in a 100×100 box | Short | GPT-2-based decoder | CLIP text, plus CLIP image of the partial render | ColorSVG-100K: 100K colored SVGs, 500 categories | FID 15.93. 1.36 s per SVG. | [svgbuilder.github.io](https://svgbuilder.github.io) |
| **LLM4SVG** | CVPR 2025 | [2412.11102](https://arxiv.org/abs/2412.11102) | SVG text plus **learnable semantic tokens** for tags and attributes. Also models rendering order to reduce occlusion errors. | LLM context | Fine-tunes GPT2-XL (1.5B), Phi-2, Llama-3.2, Qwen2.5-VL, Gemma-3 | Text / image | SVGX-SFT: 250K pretraining + 1M SFT (580K instruction pairs per the paper) | Better than raw-text LLM SVG generation | [ximinng/LLM4SVG](https://github.com/ximinng/LLM4SVG), MIT. Weights "pending" as of 2025-04 |
| **SVGFusion** | arXiv Dec 2024 | [2412.10437](https://arxiv.org/abs/2412.10437) | Each command is a **10-dim continuous vector**, max 1024 commands, with RGB and opacity. A **VP-VAE** (vector + DINOv2 pixel features) produces a continuous latent, and a **latent DiT** (VS-DiT) generates it. Uses rendering-order sequence modelling. | 1024 commands mapped to a latent | VS-DiT S/B/L: 12×384 up to 24×1024, **0.16–0.76B** | **CLIP text, cross-attention** | SVGX: 240K emoji and icons, 128² canvas | Claimed FID 4.64 and CLIPScore 0.399 (L). 36 s per SVG with 20 steps. | [ximinng/SVGFusion](https://github.com/ximinng/SVGFusion), MIT. **Code not yet released** |
| **Chat2SVG** | CVPR 2025 | [2411.16602](https://arxiv.org/abs/2411.16602) | An LLM writes a template SVG of primitives. SDXL + ControlNet adds detail. Then latent-path and point optimization. | n/a | Commercial LLM (Claude per setup docs) + SDXL | Text | none (training-free) | High quality and editable. **Not on-device.** | [kingnobro/Chat2SVG](https://github.com/kingnobro/Chat2SVG), license not stated |
| **SketchAgent** | CVPR 2025 | [2411.17673](https://arxiv.org/abs/2411.17673) | Training-free. A multimodal LLM writes strokes in a **string "sketch language" on a numbered grid**. Each stroke is a Bézier (start, end, ≥2 waypoints, t-values). | LLM text | Off-the-shelf frontier LLM | Text / chat | none | Broad concepts, but needs a big LLM. Shows that **grid-cell coordinate tokens are enough for recognizable sketches**. | [sketch-agent.csail.mit.edu](https://sketch-agent.csail.mit.edu/) |
| **NeuralSVG** | arXiv Jan 2025 / ICCV 2025 | [2501.03992](https://arxiv.org/abs/2501.03992) | Implicit: a small MLP maps a **shape index** to 4 cubic Béziers (12 points) plus fill. **Nested-dropout-style regularization makes shapes importance-ordered** (layered). | n/a | Tiny MLP, trained per prompt with SDS | Text via SDS | none | Per-prompt optimization, so not amortized. The ordering trick is useful. | Project page (Cohen-Or group) |
| **OmniSVG** | arXiv Apr 2025 / NeurIPS 2025 | [2504.06263](https://arxiv.org/abs/2504.06263) | **Parameterized SVG tokens**: commands {M, L, C, A, Z, F}, each point **merged into one token (x·w+y)**, fill color as special hex tokens, `<SOP>` and `<EOS>` markers | Icons ~4–6K tokens. Characters average ~28K. | Qwen2.5-VL backbone: 3B, and v1.1 at 4B and 8B | Text / image / character reference | MMSVG-2M: 1.1M icons, 0.5M illustrations, 0.4M characters | Text-to-SVG icons: FID ~130 (paper's metric), CLIP 0.276. About 4–18 s per 256–1024 tokens on GPU, 17–26 GB VRAM. | [OmniSVG/OmniSVG](https://github.com/OmniSVG/OmniSVG): code Apache-2.0, data CC BY-NC-SA 4.0 |
| **RLRF** | NeurIPS 2025 | [2505.20793](https://arxiv.org/abs/2505.20793) | RL on SVG-emitting VLMs, with rewards computed from **rendered output** compared to the target | n/a | StarVector / Qwen-VL scale | Image / text | SVG-Stack etc. | Clearly better than SFT alone. Shows that **render-in-the-loop rewards help sequence models of primitives.** | (unverified) |
| **SwiftSketch** | SIGGRAPH 2025 | [2502.08642](https://arxiv.org/abs/2502.08642) | **Fixed 32 strokes**, each a cubic Bézier (4 control points). **DDPM over all strokes in parallel** (50 steps). **Strokes are importance-ordered** (contours and salient regions first). L1 on points plus LPIPS on the rasterized output. One-step refinement net. CFG with 10% drop. | 32 stroke slots | Transformer decoder, 8 layers of self- and cross-attention **(params not stated)** | Image, via CLIP-ResNet layer-4 features and **cross-attention** | **Synthetic** pairs made by ControlSketch (SDS + depth ControlNet): 35K over 100 categories | **Under 1 s per sketch.** Generalizes to unseen categories. | [swiftsketch.github.io](https://swiftsketch.github.io) |
| **StrokeFusion** | arXiv Mar 2025 (v4 Nov 2025) | [2503.23752](https://arxiv.org/abs/2503.23752) | Each stroke is encoded jointly with its UDF map into a **stroke feature vector**. **Stroke-level latent sequence diffusion** jointly adjusts position, scale and trajectory. | Set of stroke latents | not stated | Unconditional / class | QuickDraw | Claims SOTA over sketch baselines | "upon publication" |
| **HiVG** | arXiv Apr 2026 | [2604.05072](https://arxiv.org/abs/2604.05072) | **Hierarchical SVG tokenizer**: atomic tokens (structure, command, coordinate, attribute) are merged into learned **segment tokens**. Coordinates are quantized on a 784² canvas, relative after the first point. | 2.7× shorter than raw-string tokenization | Qwen2.5-VL-3B | Text / image | 2.45M SVGs (SVG-Stack + SVGX + MMSVG) | "2.7× fewer training tokens for comparable quality" | GitHub mentioned **(unverified)** |
| **SketchFlow** | arXiv Aug 2026 / SIGGRAPH Asia 2026 | [2608.21659](https://arxiv.org/abs/2608.21659) | A flow-matching model maps a GMM prior (noised category embeddings) **in CLIP latent space** to sketch features. A hybrid 1D-UNet + transformer diffusion decoder produces stroke trajectories. | n/a | not stated | **CLIP latent**, giving zero-shot text beyond the 345 classes | QuickDraw, 345 classes | Zero-shot unseen labels and semantic modifiers | [doudin404/SketchFlow](https://github.com/doudin404/SketchFlow) |
| **Draw This First** | arXiv Aug 2026 | [2608.12064](https://arxiv.org/abs/2608.12064) | Predicts a **2D field that defines drawing order**, then vectorizes and sorts polylines. Uses a pretrained latent flow-matching image prior. | n/a | not stated | Text / image | not stated | Ordered vector sketches with user-controllable order | (unverified) |

Other related work, cited but not deeply checked: SkexGen (CAD, 2022), PolyGen (2020, canonical vertex sorting), VecFusion (fonts, CVPR 2024), LottieGPT ([2604.11792](https://arxiv.org/abs/2604.11792), 2026, tokenizes vector animation), SVG-Score ([2609.03806](https://arxiv.org/abs/2609.03806), 2026, human-aligned evaluation for text-to-SVG), and Learning to Paint (Huang et al., ICCV 2019, RL, coarse-to-fine grid). These are listed from search results and **not verified in detail**.

---

## 2. Small and efficient raster generators: sizes and tricks that carry over

| Model | Year / venue | Link | Tokens | Sizes → quality | Trick relevant to us | Code / license |
|---|---|---|---|---|---|---|
| **MaskGIT** | CVPR 2022 | [2202.04200](https://arxiv.org/abs/2202.04200) | VQGAN with 1024 codes, 16×16 = 256 tokens | 227M (24 layers, 768 dim): ImageNet-256 FID **6.18 in 8 steps**. The same-size AR VQGAN gets 15.78 with 256 steps. | Bidirectional masked prediction with a confidence-based unmasking schedule gives ~30× fewer steps than AR. | google-research/maskgit, Apache-2.0 **(unverified)** |
| **Parti** | TMLR 2022 | [2206.10789](https://arxiv.org/abs/2206.10789) | ViT-VQGAN, 1024 tokens (32×32), 8192 codes **(unverified)** | Encoder-decoder. COCO zero-shot FID: **350M 14.10**, 750M 10.71, 3B 8.10, 20B 7.23. Parti-350M has 12 encoder + 12 decoder layers, d=1024. | Text goes through an encoder-decoder (cross-attention). Quality scales smoothly, and 350M is already usable on COCO. | Not released |
| **LlamaGen** | arXiv Jun 2024 | [2406.06525](https://arxiv.org/abs/2406.06525) | VQ 16×16 (codebook 16384 **(unverified)**) | **B = 111M: FID 5.46** (ImageNet-256, CFG). Up to 3.1B: FID 2.18. | Plain Llama with a class token as prefix. A 111M AR model is already decent on 1000 classes. | [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen), MIT **(unverified)** |
| **VAR** | NeurIPS 2024 (best paper) | [2404.02905](https://arxiv.org/abs/2404.02905) | Multi-scale VQ, shared codebook V=4096, 10 scales | **d16 = 310M: FID 3.30**. d20 600M: 2.57. d24 1.0B: 2.09. d30 2.0B: 1.92. **10 steps.** | Next-scale (coarse-to-fine) prediction, ~20× faster than AR, with power-law scaling. This is the model for "stroke levels". | [FoundationVision/VAR](https://github.com/FoundationVision/VAR). arXiv page lists CC0 for the paper; code license MIT **(unverified)** |
| **MAR** | NeurIPS 2024 | [2406.11838](https://arxiv.org/abs/2406.11838) | Continuous KL-16 tokens. A **small MLP diffusion head** models each token. | **MAR-B 208M: FID 2.31**. L 479M: 1.78. H 943M: 1.55. Head width ablation: **2M params → 2.45**, 6M → 2.11, 21M → 1.97, 45M → 1.91. Time per image is nearly unchanged across head sizes. | See §3 Q1/Q2. Diffusion loss beats CE in every setting. Random order beats raster order. Bidirectional beats causal. | [LTH14/mar](https://github.com/LTH14/mar), MIT **(unverified)** |
| **GIVT** | ECCV 2024 | [2312.02116](https://arxiv.org/abs/2312.02116) | Continuous β-VAE latents (d=16). Output is a **GMM with k=16** (factorized). | Causal B 86M: ~5.67 **(approximate, from summary)**. Default 304M: 3.35. L 1.67B: 2.59. GIVT-MaskGIT reaches ~4.5 in 16 steps. | FID improves as k grows from 1 and plateaus at 16. Variance scaling t∈[0.9, 1.0] is needed. Distribution-based CFG. | google-research/big_vision, Apache-2.0 |
| **TiTok** | NeurIPS 2024 | [2406.07550](https://arxiv.org/abs/2406.07550) | **1D tokenizer: 32 tokens per 256² image** | gFID 1.97 (256) with MaskGIT. 410× faster than DiT at 512. | A very compact *semantic* token set is enough for an image. That argues for ~32–128 "big strokes" as the coarse level. | bytedance/1d-tokenizer **(unverified)** |
| **DiT** (small sizes) | ICCV 2023 | [2212.09748](https://arxiv.org/abs/2212.09748) | Latent patches | **DiT-S/2 33M: FID 68.4**. DiT-B/2 130M: 43.5 (400K steps, no CFG). | Reference point: tiny *pixel-latent* diffusion is poor. adaLN-Zero is a cheap way to inject the condition. | facebookresearch/DiT, CC-BY-NC |
| **DF-GAN** | CVPR 2022 | [2008.05865](https://arxiv.org/abs/2008.05865) | GAN | **19M params, COCO FID 19.32** (trained on COCO) | A tiny text-to-image model with real COCO coverage, at low quality | (code public) |
| **LAFITE** | CVPR 2022 | [2111.13792](https://arxiv.org/abs/2111.13792) | StyleGAN2 + **frozen CLIP** | **75M trainable**, COCO FID **8.12** (supervised). CC3M→COCO zero-shot: 26.94. | A frozen CLIP embedding gives broad semantics cheaply. | (code public) |
| **GALIP** | CVPR 2023 | [2301.12959](https://arxiv.org/abs/2301.12959) | GAN + frozen CLIP-ViT-B/32 | **0.08B trainable + 0.24B frozen CLIP**: COCO zero-shot FID 12.54 (CC12M) | Same lesson. Comparable to LDM (1.45B, 12.63) at about 1/18 the size. | (code public) |

**Text encoders and their costs**
- CLIP ViT-B/32 text tower: 63M parameters, 12 layers, 512 wide (from the CLIP paper).
- MobileCLIP-S0: **text 42.4M + image 11.4M**, about 1.6 ms text latency on an iPhone ([2311.17049](https://arxiv.org/abs/2311.17049)).
- T5-small encoder: ~35M **(unverified)**.
- SigLIP-B text tower: ~110M **(unverified)**.
- Small BERT from Turc et al. (IconShop, Grimoire): ~4–30M depending on variant **(unverified)**.

---

## 3. Answers to the five questions

### Q1. Discrete quantized tokens or continuous regression (MDN / GMM / small diffusion head) for stroke parameters?

**Evidence**
- **For discrete tokens.**
  - Sketchformer: dictionary tokens (K=1000) beat continuous stroke-5 for reconstructing complex sketches and for 345-class classification.
  - Sketch-RNN (GMM, M=20) shows mode-averaging ("smoother, more circular… averaging") on harder classes.
  - DeepSVG (8-bit), IconShop (100×100 grid with a merged xy token), OmniSVG and HiVG (quantized coordinates) all use categorical cross-entropy successfully.
  - Categorical outputs handle multimodality for free. Temperature, top-k and CFG are trivial with them.
- **For continuous outputs.**
  - MAR Table 1, same backbone, CE+VQ vs diffusion loss+KL tokens: AR raster 19.58 vs 19.23; MAR causal 16.22 vs 13.07; MAR bidirectional **8.75 vs 3.43** (all without CFG). The gap is largest for parallel prediction.
  - **The MAR head can be tiny**: a 2M-param MLP head already reaches FID 2.45 with CFG.
  - GIVT: a GMM output works but needs k≈16 mixtures and variance scaling.
  - Diffusion over continuous stroke points works and is fast in parallel: SwiftSketch (<1 s), ChiroDiff (<1M-param GRU), SketchKnitter (30 steps).
- **Caveat.** MAR's advantage comes partly from the better continuous tokenizer (KL-16 vs VQ-16). Our parameters are not learned latents. They are low-dimensional, semantically meaningful scalars (x, y, width, RGB, alpha), and their quantization error is visible only as small jitter. Photorealism is not the goal, so a 64–128-bin grid is already below perceptual tolerance for a painterly look.

**Recommendation for a 10–100M model: a hybrid. This is my synthesis, not something a paper tested.**
1. **One transformer position per stroke** (or per stroke "group"). Do not use one position per scalar: per-scalar tokenization (13 scalars × N strokes) would blow up the sequence length, which is the main CPU cost.
2. Inside that position, decode the stroke's fields with a **tiny per-stroke head**: categorical for coarse fields, plus an optional continuous residual. Analogues are the MAR MLP head, the GIVT GMM head, and the RQ-Transformer "depth" decoder (known from CVPR 2022, not re-verified here).
   - Coarse fields: position bins on a 32–64 grid, merged xy as in IconShop and OmniSVG; color as a palette index of 64–256 entries; width bin; alpha bin.
   - Optional continuous residual: sub-bin offsets and control-point deltas via a 3-layer MLP diffusion or flow head of 1–5M params, or a small GMM with k≈4–8.
3. Start with **pure categorical heads**. They are the simplest, robust and CFG-friendly. Add the continuous residual head only if quantization artifacts show.

### Q2. Autoregressive, set-based (all strokes in parallel), or coarse-to-fine levels?

**Evidence**
- **Pure masked or non-AR without an ordering prior can fail.** IconShop's BERT-style baseline got FID 35.1 vs 4.65 for AR and drew only basic shapes. The paper blames variable length: `<EOS>` appeared at several positions.
- **Set prediction works when the slot count is fixed and the ordering or matching is sensible.**
  - DeepSVG: feed-forward decoding beat one-stage AR, and ordered beat Hungarian.
  - Paint Transformer: DETR set with a keep bit.
  - SwiftSketch: fixed 32 slots, importance-ordered, diffusion in parallel.
- **Parallel masked generation beats raster AR on images.** MAR random-order bidirectional with multi-token steps: FID 3.50 vs 19.23 for raster AR. MaskGIT: 8 steps vs 256.
- **Coarse-to-fine is strong and fast.** VAR (10 steps, 310M → FID 3.30). Paint Transformer's K scales. Classic painterly rendering (big brushes first). NeuralSVG's importance ordering and SwiftSketch's importance ordering are the stroke analogue.

**Recommendation: VAR-style levels × MaskGIT/MAR-style parallel decoding within each level.**
- Fixed stroke budgets per level, e.g. L0 = 16 large strokes (composition and color blocking), L1 = 48, L2 = 128, L3 = 256 small detail strokes.
- Each level is decoded in parallel with a few masked or confidence-ordered steps (4–8). Unused slots use a keep/drop bit (Paint Transformer), which avoids the `<EOS>` problem.
- Condition each level on the previous levels' strokes. Optionally also condition on a **cheap low-resolution render of the canvas so far** (e.g. 32² or 64² through a tiny CNN). The fixed rasterizer then gives feedback for free, as in Paint Transformer and SVGBuilder's partial-render encoder.
- Total sequential network calls: about 4 levels × 4–8 steps = 16–32, compared with ~450+ for per-stroke AR or thousands for per-token AR.
- This combination is not demonstrated in the literature for text-to-strokes, so it is a **hypothesis to test**.
- Fallback: plain per-stroke AR (one position per stroke, ~200–500 positions) with a KV cache. It is the simplest and the most proven path (IconShop, Grimoire).

### Q3. Sequence-length limits for tiny transformers on CPU, and expected tokens/sec

**What tiny vector models used:** IconShop 512 + 50, StrokeNUWA 512, Grimoire 512, DeepSVG a fixed ~8×30 grid. Big LLM-based SVG models need 4K–28K tokens and are not viable on a phone.

**Measured decode speeds:**
- MobileLLM-125M: ~50 tok/s on iPhone.
- llama2.c stories15M: ~110 tok/s on an M1 Air (fp32, early single-thread build).
- SmolLM2-135M: ~46–60 tok/s on phone/CPU (third-party benchmark, **unverified**).

These are with 32K–49K vocabularies. A stroke vocabulary of a few thousand makes the output layer much cheaper.

**My estimate (unverified):** a 30–50M int8 decoder with KV cache runs at roughly **100–400 tok/s** on a modern laptop CPU and **50–200 tok/s** on a phone. Decoding is memory-bandwidth-bound: one weight read per token is about 30–50 MB.

**Practical budgets:**
- Sequential AR: keep it to **≤ 512 steps**. That is 2–10 s on a phone. Fine for "paint as you watch", poor for instant output.
- Per-scalar tokenization (~13 tokens per stroke × 256 strokes ≈ 3.3K tokens): about 15–60 s, and attention cost grows too. Avoid it.
- Parallel or level-wise decoding: one bidirectional pass over 256–512 positions for a 30M model is ~15–30 GFLOP. That is roughly **0.1–0.5 s per pass** on a phone CPU, so 16–32 passes take about 2–10 s, or less with a smaller model or fewer steps. **(estimate)**
- CFG doubles the cost unless the conditional and unconditional passes are batched.

### Q4. How others handle permutation ambiguity of primitive order

1. **Natural or recorded order.** Sketch-RNN, Sketchformer, ChiroDiff, SketchKnitter (QuickDraw time order). IconShop, StrokeNUWA and OmniSVG use file order. Cheap, but noisy, and SVG file order is arbitrary.
2. **Canonical sorting.** DeepSVG sorts paths lexicographically by start point. **This beat Hungarian matching**, because it "breaks symmetries and reduces competition between predicted paths". PolyGen sorts vertices (z-y-x). SVGBuilder uses component ordering.
3. **Hungarian / set matching.** DETR, Paint Transformer, DeepSVG (as the alternative). It works but is weaker or slower to train.
4. **Order-free supervision through rendering.** Im2Vec and Grimoire (raster loss via DiffVG); Paint Transformer (pixel loss "regulates order"); SwiftSketch (LPIPS term); RLRF (render reward).
5. **Importance or coarse-to-fine ordering.** SwiftSketch (contours and salient first); NeuralSVG (nested dropout); "Draw This First" (2026, predicted order field); classic painterly big-to-small.
6. **Rendering or occlusion-aware order.** SVGFusion and LLM4SVG model the z-order and creation logic.
7. **Order-invariant set encoding.** CoSE.
8. **Random-order training.** MAR learns any order, which acts as data augmentation over permutations.

**For alpha-composited brush strokes, order is semantically real** (occlusion), so it cannot be fully discarded. Recommended approach:
- Level (size, coarse→fine) defines the z-order between levels.
- Within a level, use a canonical spatial sort (e.g. by stroke centroid in Morton or raster order) so slots have stable identities.
- Add a rendering loss or reward as an auxiliary signal, since the rasterizer is fixed and can be made differentiable for training.

### Q5. Evidence about the minimum model size for broad (ImageNet/COCO-level) coverage

- **Class-conditional ImageNet (pixel-token models, a much harder target than strokes):**
  - LlamaGen-B 111M: FID 5.46.
  - GIVT-B 86M: ~5.7 (approx.).
  - MaskGIT 227M: 6.18.
  - MAR-B 208M: 2.31.
  - VAR-d16 310M: 3.30.
  - DiT-S/2 33M: 68.4 (no CFG), i.e. poor.

  So for **pixel-level** ImageNet, ~100M is roughly where AR and masked models become good, and ~30M latent diffusion is clearly insufficient.
- **COCO text-to-image:**
  - DF-GAN 19M: FID 19.3 (recognizable, low quality).
  - LAFITE 75M trainable + frozen CLIP: 8.12.
  - GALIP 80M trainable + frozen CLIP: 12.5 zero-shot.
  - Parti-350M: 14.1 zero-shot.

  **Leaning on a frozen CLIP text/image space is how sub-100M models get broad semantic coverage.**
- **Strokes:** no paper shows a <100M stroke generator with ImageNet- or COCO-level breadth.
  - The closest are QuickDraw 345-class models. Sketchformer is a tiny 4-layer encoder, not a generator. SketchFlow (2026) uses CLIP latent space for zero-shot beyond 345 classes.
  - Icon models (IconShop: 12 layers, size unstated) cover thousands of keywords in a narrow style.
- **Inference (unverified):** strokes carry far less entropy than pixels. Recognizability needs ~50–300 strokes, which is about 1–2 orders of magnitude fewer "decisions" than 256 VQ tokens × 14 bits. A **20–60M generator plus a frozen ~40–60M text encoder** (MobileCLIP-S0 text 42M, or precomputed embeddings for a fixed class list) is plausible for *recognizable* broad coverage.
- **The binding constraint is training data, not parameters.** No large, captioned, colored stroke dataset with broad categories exists. SwiftSketch and Grimoire show two ways around this:
  1. Synthesize stroke decompositions from images, using SDS or ControlNet optimization or a painter network.
  2. Supervise only through the rasterizer.

---

## 4. Cheap text conditioning: what is used and what I would use

**What prior work uses**
- **Prefix tokens**: IconShop and Grimoire (frozen BERT embeddings, ~50 tokens); LlamaGen (class token); MAR (class embedding buffer).
- **Cross-attention**: SVGFusion (CLIP), SwiftSketch (CLIP-ResNet features), Parti (encoder-decoder).
- **adaLN**: DiT (class plus timestep).
- **Joint CLIP embedding space**: LAFITE, GALIP, SketchFlow. This gives zero-shot generalization to prompts never seen with strokes.

**Cheapest option for us**
- Inject a **single pooled CLIP/MobileCLIP text embedding** via adaLN (or as 1–4 prefix tokens).
- For a class prompt, **precompute** the embedding, so the text encoder does not need to ship.
- Use **CFG** with ~10% condition dropout (SwiftSketch, IconShop's 10% blank text).
- Cross-attention over 77 text tokens adds per-layer cost and helps mainly with compositional prompts. Defer it.

---

## 5. Main takeaways for the design
1. **Keep the sequence short**: one position per stroke, never one per scalar. Budget ≤ 512 positions in total.
2. **Use categorical heads per field** (merged-xy grid, palette color, width and alpha bins). Add a tiny MAR-style continuous residual head only if needed. MAR shows even a 2M head works.
3. **Generate coarse-to-fine in stroke levels, parallel within each level**, conditioned on a low-res render of the canvas so far. Fixed slots with a keep bit avoid the variable-length `<EOS>` failure seen in IconShop's BERT baseline.
4. **Fix the order canonically** (level → spatial sort). DeepSVG found ordered better than Hungarian. Add render-space loss or reward from the fixed rasterizer.
5. **Get semantic breadth from frozen CLIP-space conditioning** and **synthetic stroke data from images**. Model size (20–60M) is probably not the bottleneck. Data is.

## Sources (verified during this survey unless noted)
Sketch-RNN [1704.03477](https://arxiv.org/abs/1704.03477) · Sketchformer [2002.10381](https://arxiv.org/abs/2002.10381) · CoSE [NeurIPS'20](https://proceedings.neurips.cc/paper/2020/hash/723e8f97fde15f7a8d5ff8d558ea3f16-Abstract.html) · DeepSVG [2007.11301](https://arxiv.org/abs/2007.11301), [code](https://github.com/alexandre01/deepsvg) · Im2Vec [2102.02798](https://arxiv.org/abs/2102.02798) · Paint Transformer [2108.03798](https://arxiv.org/abs/2108.03798) · IconShop [2304.14400](https://arxiv.org/abs/2304.14400) · ChiroDiff [2304.03785](https://arxiv.org/abs/2304.03785) · SketchKnitter [OpenReview](https://openreview.net/forum?id=4eJ43EN2g6l), [code](https://github.com/XDUWQ/SketchKnitter) · DiffSketcher [2306.14685](https://arxiv.org/abs/2306.14685) · StarVector [2312.11556](https://arxiv.org/abs/2312.11556), [code](https://github.com/joanrod/star-vector) · StrokeNUWA [2401.17093](https://arxiv.org/abs/2401.17093) · Grimoire [2410.05991](https://arxiv.org/abs/2410.05991) · SVGBuilder [2412.10488](https://arxiv.org/abs/2412.10488) · LLM4SVG [2412.11102](https://arxiv.org/abs/2412.11102), [code](https://github.com/ximinng/LLM4SVG) · SVGFusion [2412.10437](https://arxiv.org/abs/2412.10437) · Chat2SVG [2411.16602](https://arxiv.org/abs/2411.16602) · SketchAgent [2411.17673](https://arxiv.org/abs/2411.17673) · NeuralSVG [2501.03992](https://arxiv.org/abs/2501.03992) · OmniSVG [2504.06263](https://arxiv.org/abs/2504.06263), [code](https://github.com/OmniSVG/OmniSVG) · RLRF [2505.20793](https://arxiv.org/abs/2505.20793) · SwiftSketch [2502.08642](https://arxiv.org/abs/2502.08642) · StrokeFusion [2503.23752](https://arxiv.org/abs/2503.23752) · HiVG [2604.05072](https://arxiv.org/abs/2604.05072) · SketchFlow [2608.21659](https://arxiv.org/abs/2608.21659) · Draw This First [2608.12064](https://arxiv.org/abs/2608.12064) · MaskGIT [2202.04200](https://arxiv.org/abs/2202.04200) · Parti [2206.10789](https://arxiv.org/abs/2206.10789) · LlamaGen [2406.06525](https://arxiv.org/abs/2406.06525) · VAR [2404.02905](https://arxiv.org/abs/2404.02905) · MAR [2406.11838](https://arxiv.org/abs/2406.11838) · GIVT [2312.02116](https://arxiv.org/abs/2312.02116) · TiTok [2406.07550](https://arxiv.org/abs/2406.07550) · DiT [2212.09748](https://arxiv.org/abs/2212.09748) · DF-GAN [2008.05865](https://arxiv.org/abs/2008.05865) · LAFITE [2111.13792](https://arxiv.org/abs/2111.13792) · GALIP [2301.12959](https://arxiv.org/abs/2301.12959) · MobileCLIP [2311.17049](https://arxiv.org/abs/2311.17049) · MobileLLM [2402.14905](https://arxiv.org/abs/2402.14905) · llama2.c [repo](https://github.com/karpathy/llama2.c)
