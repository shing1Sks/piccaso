# 03 — Dataset Survey for a Tiny Text/Class-Conditioned Brush-Stroke Painter

*Survey date: 2026-10-03. Target: a 10–100M-param model that emits a sequence of parametric strokes (curve + width + RGB + alpha), conditioned on a class label or text. The look we want is painted or impressionistic, not photoreal.*

## Verification legend

| Tag | Meaning |
|---|---|
| **[V]** | Checked this session against the official page, HF dataset card, or paper |
| **[K]** | Well-known fact that I did **not** re-check this session. Confirm before you rely on it |
| **[U]** | Unverified, unclear, or not stated by the source |

**Relevance** means relevance to *this* project: **H** = directly trainable or high value; **M** = useful with conversion or as an auxiliary; **L** = marginal.

**The core gap.** I found **no public dataset of (caption → colored, alpha-blended brush-stroke sequence)** pairs. The closest are:
- monochrome human stroke data (QuickDraw, TU-Berlin, FS-COCO);
- colored *vector* data (emoji, MMSVG). These are filled paths, not strokes.
- painting-process *videos*. The large ones are mostly unreleased.

So the plan has to be: (a) pretrain stroke "grammar" on native vector data, and (b) produce paired stroke data by running a painting decomposer over captioned images.

---

## 1. Native stroke / sketch data

| Dataset | Size | Content | Labels / captions | Resolution / format | License (commercial?) | Download | Relevance |
|---|---|---|---|---|---|---|---|
| **Google Quick, Draw!** | 50M drawings, 345 categories [V]. The Sketch-RNN `.npz` version has 75K per class (70K/2.5K/2.5K) [V] | Doodles drawn in <20 s, black strokes | Class label, "recognized" flag, country [V] | Raw timestamped vectors; simplified vectors at 256×256 (RDP); 28×28 `.npy` bitmaps [V] | **CC BY 4.0** — commercial OK with attribution [V] | `gs://quickdraw_dataset` (gsutil); github.com/googlecreativelab/quickdraw-dataset; HF `google/quickdraw` [V] | **H**. The biggest stroke-order dataset there is, and ideal for Stage 0 and for stroke-grammar pretraining. No color or width |
| **TU-Berlin** (Eitz et al. 2012) | 20,000 sketches, 250 classes, 80 per class [V] | Non-expert object sketches (AMT, 1,350 participants) [V] | Class | SVG with cubic Béziers, stroke order preserved [V] | **CC BY 4.0** [V] | cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/ [V] | **M**. Small, but already Bézier-native, which matches our curve parameterization |
| **Sketchy** (Sangkloy et al. 2016) | 75,471 sketches of 12,500 photos, 125 classes [V] | Sketch–photo pairs | Class plus paired photo | SVG (`sketches-06-04.7z`) [V] | [U] | sketchy.eye.gatech.edu (gave a **TLS certificate error** when fetched) and via SketchKit [V] | **M**. Paired photos are useful for testing image→stroke conversion |
| **QMUL-Shoe-V2 / Chair-V2** | Shoe-V2: 6,730 sketches / 2,000 photos. Chair-V2: 1,275+725 sketches / 300+100 photos [V] | Fine-grained sketch–photo pairs | Instance pairing | Stroke coordinates available [V] | [U] | TIB registry entries; SketchX (Surrey) pages [V/U] | **L**. Narrow domain |
| **SketchyScene** | 7,265 real scenes, plus 21,795 and 217,950 synthesized [V] | Scene sketches, 45 categories [V] | Semantic and instance segmentation [V] | **Raster**, with `.mat` annotations [V] | **CC BY-NC-SA 3.0 + "996 ICU" clause** — non-commercial [V] | github.com/SketchyScene/SketchyScene. Sizes: 750 MB / 1.6 GB / 16.1 GB [V] | **L**. Raster only, so no stroke sequence |
| **FS-COCO** (ECCV 2022) | 10,000 scene sketches by 100 non-experts [V] | Freehand *scene* sketches of COCO images | **Text caption per sketch** plus the paired COCO photo [V] | Vector, with per-point space-time info [V] | **CC BY-NC 4.0** — non-commercial [V] | cvssp.org/data/fscoco/fscoco.tar.gz, plus a Google Drive mirror [V] | **H** (for eval and finetuning). The only human *caption→stroke-sequence* scene data. Too small for pretraining |
| **Creative Birds / Creative Creatures** (DoodlerGAN) | "10k" sketches each [V; exact counts not given] | Creative, imaginative sketches | **Part annotations and free-form text captions** [V] | Processed JSON at 64×64 [V] | Code is MIT; data license [U] | Google Drive folder linked from github.com/facebookresearch/DoodlerGAN [V] | **M**. Captioned strokes, small |
| **ControlSketch** (SwiftSketch, SIGGRAPH 2025) | 35,000 image–sketch pairs, 15 categories [V via secondary source] | *Synthetic*: SDXL images → SDS-optimized vector sketches | Class plus paired image | Vector (Bézier) | [U] | arXiv 2502.08642. Release status [U] | **M**. Shows the "synthetic image → optimized strokes" pipeline we plan to use |
| **ControlSketch-Part** (arXiv 2603.19500, 2026) | [U] | VLM part-annotated vector sketches | Caption, part descriptions, path→part assignment [V] | Vector | [U] | arXiv 2603.19500 [V] | **M**. Text-to-sketch benchmark, worth tracking |
| **Tracing-vs-Freehand** | [U] | Vector sketches | — | JSON stroke attributes, *including color* [V via SketchKit] | [U] | Via SketchKit [V] | **L** |
| **IAM-OnDB** (handwriting) | 221 writers, 13,049 text lines, 86,272 words [V] | Whiteboard handwriting | Transcription | Online pen trajectories in XML [V] | **Non-commercial research; registration required** [V] | fki.tic.heia-fr.ch/databases [V] | **L**. Only useful for a stroke-dynamics sanity check |

**Colored stroke sketches.** I found **no sizeable public dataset of colored, natively stroked sketches**. "TexSketch" (a procedural colored-sketch generator) and vector-colorization papers turned up in search, but I found no released dataset [U].

**Tooling.** SketchKit (HKUST-GZ) offers unified loaders for 13 sketch datasets: QuickDraw, TU-Berlin, Sketchy, FS-COCO, CreativeSketch, ControlSketch and others [V]. It is worth using to avoid writing parsers.

---

## 2. Vector / SVG data

All of these are **filled-path vector art, not brush strokes**. Two ways to use them: convert fills into "fat strokes" or region primitives, or rasterize them and run the painting decomposer. Their value is **color + semantics + clean structure**.

| Dataset | Size | Content | Labels / captions | Format | License | Download | Relevance |
|---|---|---|---|---|---|---|---|
| **SVG-Stack** (StarVector) | 2,283,875 rows; 3.01 GB parquet [V] | SVGs scraped from GitHub (The Stack) | None in this version; filename + SVG code [V] | SVG code, 26 B–35 KB [V] | Card does not state one. Per-file licenses are inherited from GitHub [U] | HF `starvector/svg-stack` [V] | **M**. Huge and diverse. Licensing is murky and quality uneven |
| **Text2SVG-Stack** (StarVector) | 2,175,419 rows; 3.34 GB [V] | Same SVGs | **Captions from BLIP2, CogVLM, LLaVA** [V] | SVG + text | Not stated [U] | HF `starvector/text2svg-stack` [V] | **M**. Text-conditioned vector pairs |
| **svg-emoji** (StarVector) | 10,043 rows (8,710 / 667 / 668); 16.9 MB [V] | OpenMoji + Noto + Twemoji SVGs [V] | Filename only (no captions in the card) [V] | SVG code | Not stated on the card. Inherits the source licenses [U] | HF `starvector/svg-emoji` [V] | **H** for Stage 0. Tiny, colored, semantic |
| **MMSVG-Icon** (OmniSVG) | 904,011 samples (v2.0); 18.7 GB [V] | Icons | **Description, keywords, detailed description** [V] | SVG normalized to 200×200 (picosvg) + PNG at 448×448 [V] | **CC BY-NC-SA 4.0** [V] | HF `OmniSVG/MMSVG-Icon` [V] | **H** (research). Large, colored, captioned. Non-commercial |
| **MMSVG-Illustration** | 255,412 samples (v2.0) [V] | Illustrations | Same caption fields [V] | Same | **CC BY-NC-SA 4.0** [V] | HF `OmniSVG/MMSVG-Illustration` [V] | **H** (research). Closer to "pictures" than icons are |
| **MMSVG-2M (full) / MMSVG-Character** | 2M total, per the paper (55% icons, 25% illustrations, 10% characters, 10% anime) [V] | — | — | — | — | Character subset "planned"; I found no release as of this survey [V/U] | — |
| **SVGX-SFT-1M** (+ SVGX-Core-250k) | >1M instruction samples; 10.6 GB [V] | Text→SVG, SVG→text, image→SVG dialogs | Instructions and captions | SVG text | **CC BY-NC 4.0** [V] | HF `xingxm/SVGX-SFT-1M` [V] | **M** |
| **DeepSVG SVG-Icons8** | 100,000 icons. Metadata CSV 9 MB, tensors zip 3 GB [V] | Icons8 icons | Category metadata [V] | **Preprocessed tensors only**; the raw SVGs need a paid Icons8 plan [V] | Proprietary source [V] | github.com/alexandre01/deepsvg [V] | **L**. Monochrome, license-restricted |
| **FIGR-8** | 1,548,256 images, 17,375 classes, min 8 per class [V] | Noun Project pictograms | Class name, artist, license per image [V] | PNG 192×192 grayscale [V] | Repo is MIT, but **most images are CC with non-commercial reproduction limits**, attribution required [V] | github.com/marcdemers/FIGR-8, Google Drive, Academic Torrents [V] | **L–M**. Huge class vocabulary, monochrome |
| **FIGR-8-SVG / IconShop data** | ~1.5M monochrome icons; ~1.1M after filtering; IconShop trained on 300K [V] | Icons + ChatGPT-expanded descriptions [V] | Keywords → text | SVG [V] | [U] | Google Drive via github.com/kingnobro/IconShop [V] | **M**. Text→vector sequence, the closest analog to our setup. Monochrome |
| **Noun Project (API)** | "Nearly 10 million" icons [V] | Icons | Tags | SVG/PNG | **API terms prohibit AI/ML training without prior approval** [V] | api.thenounproject.com | **L — do not use** without written permission |
| **Iconify** | 200+ open-source icon sets [V] | Icons (mostly monochrome; some sets colored, e.g. Fluent Emoji, Noto) | Icon names | IconifyJSON (SVG bodies); `npm i @iconify/json` [V] | **Per-set licenses**, some need attribution [V] | github.com/iconify/icon-sets [V] | **M**. A convenient aggregator. Filter by license |
| **OpenMoji** | 4,147 emoji (v15.0) [V via secondary source] | Colored emoji, uniform outlined style | `openmoji.json` with names, tags, groups [V] | SVG + PNG [V] | Graphics **CC BY-SA 4.0** (share-alike) [V] | github.com/hfg-gmuend/openmoji [V] | **H**. Colored and semantically tagged. Share-alike matters if we redistribute derived data |
| **Twemoji** (jdecked fork) | Unicode/Emoji 17.0 RGI set [V]. Count ~3.7K [U] | Colored flat emoji | Codepoint → Unicode name (via CLDR) | SVG [V] | Graphics **CC BY 4.0**, code MIT [V] | github.com/jdecked/twemoji (maintained by Discord contributors) [V] | **H**. Clean flat color shapes, commercial OK with attribution |
| **Noto Emoji** | Full Unicode set. Count [U] | Colored emoji | Codepoint names | SVG in repo [V] | Images/tools **Apache 2.0**, fonts OFL [V] | github.com/googlefonts/noto-emoji [V] | **H**. Most permissive of the emoji sets |
| **Microsoft Fluent Emoji** | 1,538 originally. Iconify lists 3,126 (color) / 3,145 (flat) / 1,595 (high-contrast) variants [V via secondary source] | Color, flat, high-contrast SVG + 3D PNG [V] | Per-emoji metadata JSON [V] | SVG | **MIT** [V] | github.com/microsoft/fluentui-emoji [V] | **H**. MIT. The "Flat" style is close to a paint-by-region look |

**Emoji note.** Pooled, OpenMoji + Twemoji + Noto + Fluent come to roughly 10–15K colored, named SVGs (a rough estimate; the StarVector svg-emoji pool of three sets is 10K [V]). They are small, have semantic names that can serve as captions, are colored, and are **mostly commercially clean** (Twemoji CC BY, Noto Apache, Fluent MIT; OpenMoji is share-alike). That makes them an ideal Stage-0 *colored* target set.

---

## 3. Painting-process / brushstroke data

| Dataset | Size | Content | Released? | License | Relevance |
|---|---|---|---|---|---|
| **ProcessPainter** synthetic sequences (arXiv 2406.06062) | 30,000 synthetic sequences, 8 frames each [V] | 10K aesthetic **DiffusionDB** images rendered by **Learning-to-Paint, Stylized Neural Painting, Paint Transformer**, plus a SAM + Depth-Anything foreground→background ordering [V] | Paper doesn't say. A TIB registry entry exists [U] | [U] | **M**. Shows that synthetic SBR data is the field-standard workaround. Reproducible ourselves |
| ProcessPainter artist sequences | 95 sequences from 3 artists (impasto portraits, landscape sketches, line-art coloring) [V] | Real process frames | Source and release not stated [U] | [U] | L |
| **Inverse Painting** (Chen et al., SIGGRAPH Asia 2024 / TOG) | 294 acrylic-landscape time-lapse videos, ~9 min average. 265 train / 29 val; 7,261 training frame pairs [V] | Bob-Ross-like acrylic landscapes [V] | Code released. **Data release not stated** [U] | [U] | **M**. The style matches our target, but we'd have to rebuild the dataset |
| **Loomis Painter** (arXiv 2511.17344) | 737 YouTube videos: acrylic 81, oil 151, pencil 298, Loomis portraits 207 [V] | Real painting processes | **Data NOT released** (licensing). Code/config to rebuild only [V] | — | M. A recipe for a YouTube scrape; legally grey |
| **Paints-UNDO** (lllyasviel) | Not disclosed [V] | Model of drawing behavior. Code is Apache-2.0 [V] | **Training data not released** [V] | — | L for data. As a *teacher*, it can generate process frames for any image |
| **PaintCopilot** (arXiv 2605.20941) | 3,000 portraits with stroke-level supervision [V] | Portrait strokes | Not stated [U] | [U] | L–M |
| **Bob Ross / Joy of Painting** | 403 paintings; per-painting palette (18 named colors + hex) [V] | CSV: title, colors, YouTube and image links [V] | github.com/jwilber/Bob_Ross_Paintings; `BobRossColors` R pkg [V] | Images are © Bob Ross Inc. [K]. The CSV is community data | **L–M**. A palette prior plus 403 style references, not training scale |
| Real captured brushstrokes | — | Search found **no public dataset of physically captured individual brushstrokes**. One extraction paper explicitly notes the absence of such training data [V] | — | — | Gap |
| Digital painting PSD layers | — | **No public large-scale PSD-layer dataset found** [U] | — | — | Gap |

**Takeaway.** Real stroke-level painting data with color is essentially unavailable at scale. Every 2024–2026 process paper (ProcessPainter, Inverse Painting, Loomis Painter, PaintCopilot) either synthesizes its data with stroke-based renderers or scrapes videos it cannot redistribute. **We should synthesize too.**

---

## 4. Art images (style source / conversion targets)

| Dataset | Size | Labels | Resolution | License | Download | Relevance |
|---|---|---|---|---|---|---|
| **WikiArt (HF `huggan/wikiart`)** | Card: **81,444** artworks [V]. Repo 33.7 GB. *The HF viewer reports only 11,320 rows / 5.27 GB* — likely a partial parquet conversion [V] | Artist (129), genre (11), style (27) [V] | Varied, often high-res | **Non-commercial research only**, per WikiArt terms [V] | HF `huggan/wikiart`. Original archive on archive.org [V] | **H** (research). The natural *class-conditional* painting set (style/genre labels). Not license-clean |
| **Best Artworks of All Time** (Kaggle) | 50 artists; 2.29 GB [V]. Image count [U] | Artist + Wikipedia info [V] | Full + resized [V] | **CC BY-NC-SA 3.0** [V] | Kaggle (scraped from artchallenge.ru) [V] | L. Covered by WikiArt |
| **Art Institute of Chicago** open access | 53,438 CC0 images [V via secondary source]; >132K works described [V] | Rich metadata (title, artist, medium, date) | IIIF, high-res | **CC0** [V] | artic.edu API + IIIF [V] | **M–H**. License-clean painting set. Titles work as weak captions |
| **The Met** Open Access | >492,000 PD images [V] | Metadata CSV (CC0) [V] | High-res | **CC0** [V] | github.com/metmuseum/openaccess (CSV only; images fetched separately) [V] | M. Much of it is objects, not paintings. Filter by department |
| **Rijksmuseum** | 600,000+ PD images, 800K+ records [V] | Metadata | IIIF | PD/CC0, commercial OK [V] | data.rijksmuseum.nl (API key, OAI-PMH, LOD dumps) [V] | M |
| **Smithsonian Open Access** | 5.1M+ 2D/3D items [V] | Metadata | Varied | **CC0** [V] | AWS S3 `smithsonian-open-access` (no account needed) [V] | L–M. Few paintings relative to its size |
| **LAION-Aesthetics / LAION-Art** | Aesthetics ~120M; LAION-Art 7.2M [V] | Alt-text, aesthetic score | URL lists | Metadata only; images are copyrighted web content. Re-LAION-5B (Aug 2024) is the safety-cleaned successor [V] | laion.ai [V] | **L**. Link rot, copyright and safety baggage; PD12M is cleaner |
| **JourneyDB** | 4,429,295 Midjourney images; ~3.29 TB [V] | Prompt, caption, VQA [V] | High-res | **Custom terms, gated** [V]. Midjourney ToS applies [K] | HF `JourneyDB/JourneyDB` (gated) [V] | M. Very "painterly", text-rich. Heavy, and the license is restrictive |

---

## 5. General image + caption (to convert into strokes)

| Dataset | Size | Captions / labels | Resolution | License | Download | Relevance |
|---|---|---|---|---|---|---|
| **PD12M** (Spawning) | 12,400,094 pairs. Metadata parquet 2.34 GB [V] | Synthetic recaptions (model not named on the card) [V] | 256 px to 82,700 px wide [V] | Dataset **CDLA-Permissive-2.0**; images **PD/CC0** [V] | HF `Spawning/PD12M` (metadata) + AWS S3 `pd12m` (images) [V]. **PD3M** 3.3M subset [V] | **H**. The best license-clean text→image source. Art-heavy because of museum sources. Images are *hosted*, so no link rot |
| **Recap-DataComp-1B** | ~1.3B images recaptioned (card lists 1.88B rows incl. configs; 941M in main split); 527 GB metadata [V] | LLaVA-1.5-LLaMA3-8B recaptions + originals [V] | URL-based | Metadata **CC BY 4.0**; images are web content [V] | HF `UCSC-VLAA/Recap-DataComp-1B` [V] | M. Great captions, but URL rot and copyright. Only for a subset |
| **DataComp / CommonPool** | 12.8B pool; DataComp-1B [V] | Alt-text | URL-based | Metadata **CC BY 4.0**; images are web content [V] | HF `mlfoundations/datacomp_*` [V] | L. Overkill |
| **CC3M / CC12M** | ~3.3M [K] / ~12M [V] | Alt-text, hypernymed | Webdataset mirrors resized to ≤512 short edge [V] | "Freely used for any purpose" for the annotations; images are web content [V] | Google Research; HF `pixparse/cc3m-wds`, `pixparse/cc12m-wds` [V] | M. The wds mirrors avoid link rot. Captions are weak |
| **COCO Captions** | ~118K train + 5K val (2017), 5 captions per image [K] | Human captions | ~640 px [K] | Annotations CC BY 4.0; images Flickr with mixed licenses [K] | cocodataset.org [K] | **M–H**. Gold-standard human captions, and pairs with FS-COCO |
| **ImageNet-1k / downsampled** | 1.28M, 1000 classes [K]. 64×64: 11.73 GiB download / 10.80 GiB [V] | Class | 32/64 px variants | **Non-commercial research/education only** [V] | TFDS `imagenet_resized/64x64`; HF e.g. `sradc/imagenet_resized_64x64` (gated by ImageNet terms) [V] | **H** for Stage 1. Standard class-conditional FID benchmark at low res |
| **CIFAR-10/100** | 60K images, 32×32, 10/100 classes [K] | Class | 32 px | No formal license; widely used [K] | cs.toronto.edu/~kriz [K] | M. Fast. 32 px is too small for "strokes" to read as strokes |
| **CelebA / FFHQ** | CelebA ~202K [K]; FFHQ 70K [K] | Attributes (CelebA) | FFHQ 1024 px | CelebA **non-commercial research** [V]; FFHQ **CC BY-NC-SA 4.0** [V] | Official sites / HF mirrors [K] | M. Portraits are the classic stroke-painting demo (Learning-to-Paint used CelebA [K]). Single domain |
| **DiffusionDB** | 2M subset (1.6 TB) / 14M Large (6.5 TB); 1.8M unique prompts [V] | Real user prompts + hyperparameters [V] | ~512 px [V] | **CC0 1.0** [V] | HF `poloclub/diffusiondb` (subsets downloadable in 1K-image parts) [V] | **H**. CC0, prompt-paired, often already "painterly". ProcessPainter used it as SBR source [V]. Prompts are noisy (artist names, keyword soup) |
| **text-to-image-2M** | ~2M at 512² + 10K at 1024²; 423 GB [V] | Qwen2-VL-enhanced captions [V] | 512 / 1024 | Card says **MIT**, but it contains FLUX-dev and DALL·E 3 outputs whose upstream terms may apply [V/U] | HF `jackyhate/text-to-image-2M` [V] | M |
| **FLUX.2-dev synthetic 2M** (Kempner) | 2,282,665 images, 512², ~865 GB [V] | Captions from text-to-image-2M [V] | 512 | **"Other" — research; must comply with the FLUX.2-dev license** [V] | HF `KempnerInstituteAI/flux.2-dev-synthetic-2M` [V] | M. Ready-made distillation data. License-restricted |
| **FLUX-Reason-6M** | 6M images, 20M bilingual captions [V via search] | Reasoning-oriented | — | [U] | arXiv 2509.09680 [V] | L |

---

## 6. Evaluation

| What | How | Datasets / tools | Notes |
|---|---|---|---|
| **Distribution fidelity (class-cond.)** | FID / KID at 64 px (and 32 px) | ImageNet-64 val, CIFAR-10, WikiArt-by-style | Use `clean-fid` [K] so resizing is consistent. Report FID against **both** the real images and the *decomposer reconstructions*. The latter is the realistic ceiling for a stroke model |
| **Better-than-FID** | **CMMD** (CLIP features + MMD, CVPR 2024) [V] | Same sets | Unbiased and sample-efficient [V]. Good for small eval sets and painterly outputs, where Inception features are a poor fit |
| **Text alignment** | CLIPScore; VQA-based scores | COCO-val captions; **PartiPrompts** (1,600) [V]; **DrawBench** (~200, 11 categories) [V]; **GenEval** (553, compositional) [V]; GenEval 2 notes benchmark drift [V] | At 10–100M params, PartiPrompts/DrawBench subsets plus CLIPScore are realistic. GenEval object/color/position checks suit stroke paintings well |
| **Color fidelity** | Color-attribute accuracy | **GenColorBench** (44K prompts, 400+ colors). Release was "upon acceptance" as of Oct 2025 [V; current status U] | Relevant because our strokes carry explicit RGB |
| **Recognizability** | Accuracy of a pretrained classifier on rendered outputs | QuickDraw-345 classifier (sketch stage); ImageNet classifier at 64 px; WikiArt style classifier | Cheap and interpretable. Standard in sketch generation [K] |
| **Reconstruction (decomposer quality)** | L2 / LPIPS / SSIM vs. the target image at N strokes | Held-out PD12M / COCO images | Measures conversion quality *before* training the generator |
| **Stroke efficiency** | Strokes per image vs. quality curve | Any | Key for a tiny autoregressive model, since sequence length drives cost |
| **Human eval** | 2AFC preference ("more painterly", "matches caption") | ~100–200 prompts | DrawBench-style A/B protocol [V] |
| **Human stroke realism** | FS-COCO caption→sketch | FS-COCO test split | Only human caption–stroke pairs available (non-commercial) |

---

## 7. Practical issues (flagged)

- **Conversion compute.** Image→stroke conversion is the main cost.
  - Feed-forward decomposers (Paint Transformer, Learning-to-Paint) are amortized and fast.
  - Per-image optimization (Stylized Neural Painting, DiffVG/SDS-style) is far slower.
  - **Benchmark throughput on ~1K images before choosing dataset scale.** I did not verify per-image timings this session. Plan for **GPU** past ~10^4 images.
- **Disk sizes**:
  - DiffusionDB 2M: 1.6 TB.
  - JourneyDB: 3.29 TB.
  - FLUX.2 synth: ~865 GB.
  - text-to-image-2M: 423 GB.
  - WikiArt: 33.7 GB.
  - ImageNet-64: ~11 GB.
  - Download only subsets, and downscale to 128–256 px on ingest.
- **Link rot.** URL-list datasets (LAION, DataComp, CC3M/12M originals, Recap-DataComp) decay. Prefer the hosted ones: PD12M on S3, the pixparse wds mirrors, DiffusionDB.
- **Broken or odd links:**
  - The Sketchy site (`sketchy.eye.gatech.edu`) gave a TLS certificate mismatch.
  - HF `google/quickdraw` is a loading *script* (83 kB). Newer `datasets` releases dropped script support [K], so use `gsutil` on `gs://quickdraw_dataset` instead.
  - The HF WikiArt viewer row count (11,320) disagrees with the card (81,444).
- **Licensing traps:**
  - Noun Project forbids ML training.
  - SketchyScene has an unusual "996 ICU" clause.
  - WikiArt, ImageNet, CelebA, FFHQ, FS-COCO, MMSVG and SVGX are **non-commercial**.
  - FLUX-dev outputs carry model-license terms.
  - OpenMoji is share-alike.
- **Gated access.** JourneyDB and ImageNet (HF) require accepting terms. IAM-OnDB requires registration.

---

## 8. Recommended staged data plan

### Stage 0 — tiny proof (CPU-friendly, < 1 GB)
Goal: show the model learns stroke grammar and class conditioning on a laptop.
- **QuickDraw**, 10–20 classes × 5–10K drawings from the simplified ndjson or Sketch-RNN `.npz`. That's ~100–200K sequences, roughly tens to a few hundred MB (estimate). CC BY 4.0. Native strokes, no conversion needed. Add constant width/color/alpha as dummy channels so the tokenizer is final from day one.
- **Colored emoji**: Twemoji + Noto + Fluent-Flat (+ OpenMoji if share-alike is OK). That's ~10K SVGs (StarVector `svg-emoji` is 16.9 MB [V]). Convert fills into a few fat colored strokes, either geometrically or with a *small* decomposer run at 64 px (CPU-feasible at this count). Emoji names serve as class/text labels.
- Eval: QuickDraw-classifier accuracy, plus visual inspection.

### Stage 1 — class-conditional, broad (single GPU; ~10–50 GB raw, much smaller as strokes)
- **ImageNet-64** (1.28M images, 1000 classes; 10.8 GiB; non-commercial), or a 100-class subset to start. Convert with a feed-forward decomposer at 64 px (fixed budget, e.g. 64–256 strokes). Standard FID/CMMD comparisons.
- **WikiArt** (81K; style/genre labels; non-commercial) gives the painterly *style* conditioning. Convert at 64–128 px.
- License-clean alternative: **AIC CC0 (~53K) + PD12M's museum subset**, keeping only paintings, using museum metadata as weak labels.
- Optional: QuickDraw-345 full-scale pretraining of the stroke decoder.
- Eval: FID/CMMD at 64 px vs. both originals and decomposer reconstructions; classifier accuracy.

### Stage 2 — text-conditional (multi-GPU-days of conversion; ~0.2–1 TB raw, subsetted)
- **Primary: PD12M / PD3M** (CDLA-Permissive-2.0, PD/CC0 images, S3-hosted). Start with ~1M images at 128–256 px, converted to strokes.
- **DiffusionDB 2M subset** (CC0; prompt-paired; often painterly). Clean up the prompts, e.g. recaption or strip artist names.
- **COCO Captions** (~118K) for a human-caption finetune and eval, paired with **FS-COCO** (10K human caption→stroke scenes; non-commercial) to measure/finetune human-like stroke ordering.
- **Distillation (c)**: generate "impressionist painting of {caption}" images with an open-weights T2I model whose license allows training on its outputs (check before use; FLUX-dev-family outputs carry restrictions [V]), then decompose. This produces images that *already* look like stroke paintings, which makes conversion lossless-ish and the style consistent.
- Optional, research-only: **MMSVG-Icon/Illustration** (1.16M captioned colored SVGs, CC BY-NC-SA) as a vector text→shape auxiliary.
- Eval: CLIPScore + CMMD on COCO-val / PartiPrompts subset; GenEval-style color/object checks; small human 2AFC study.

**Ballpark sizes (estimates, not measured):** stroke sequences are compact. 256 strokes × ~12 floats at fp16 ≈ 6 KB/image, so 1M converted images ≈ 6 GB of stroke data. The raw-image downloads and the conversion compute dominate.

---

## Sources (primary)
- QuickDraw: https://github.com/googlecreativelab/quickdraw-dataset · https://huggingface.co/datasets/google/quickdraw
- FS-COCO: https://github.com/pinakinathc/fscoco
- SketchyScene: https://github.com/SketchyScene/SketchyScene
- DoodlerGAN / Creative Sketch: https://github.com/facebookresearch/DoodlerGAN · https://songweige.github.io/projects/creative_sketech_generation/home.html
- TU-Berlin: https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
- SketchKit: https://cislab.hkust-gz.edu.cn/projects/sketchkit/docs/manual/2_dataset.html
- SwiftSketch: https://arxiv.org/abs/2502.08642 · ControlSketch-Part: https://arxiv.org/html/2603.19500
- IAM-OnDB: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database
- SVG-Stack / Text2SVG-Stack / svg-emoji: https://huggingface.co/datasets/starvector/svg-stack · https://huggingface.co/datasets/starvector/text2svg-stack · https://huggingface.co/datasets/starvector/svg-emoji
- MMSVG: https://huggingface.co/datasets/OmniSVG/MMSVG-Icon · https://huggingface.co/datasets/OmniSVG/MMSVG-Illustration · https://arxiv.org/html/2504.06263v2
- SVGX: https://huggingface.co/datasets/xingxm/SVGX-SFT-1M
- DeepSVG: https://github.com/alexandre01/deepsvg · FIGR-8: https://github.com/marcdemers/FIGR-8 · IconShop: https://github.com/kingnobro/IconShop
- Noun Project API prohibited uses: https://help.thenounproject.com/hc/en-us/articles/47964341999131-Prohibited-Use-Cases-for-Noun-Project-API
- Iconify: https://github.com/iconify/icon-sets
- OpenMoji: https://github.com/hfg-gmuend/openmoji · Twemoji: https://github.com/jdecked/twemoji · Noto: https://github.com/googlefonts/noto-emoji · Fluent: https://github.com/microsoft/fluentui-emoji
- ProcessPainter: https://arxiv.org/html/2406.06062v2 · Inverse Painting: https://arxiv.org/html/2409.20556v1 · Loomis Painter: https://arxiv.org/html/2511.17344v2 · PaintCopilot: https://arxiv.org/abs/2605.20941 · Paints-UNDO: https://github.com/lllyasviel/Paints-UNDO
- Bob Ross colors: https://github.com/frankiethull/BobRossColors
- WikiArt: https://huggingface.co/datasets/huggan/wikiart · Best Artworks: https://hyper.ai/en/datasets/31867
- AIC: https://www.artic.edu/open-access/open-access-images · Met: https://github.com/metmuseum/openaccess · Rijksmuseum: https://data.rijksmuseum.nl/docs/api · Smithsonian: https://registry.opendata.aws/smithsonian-open-access
- LAION-Aesthetics: https://laion.ai/blog/laion-aesthetics/ · JourneyDB: https://huggingface.co/datasets/JourneyDB/JourneyDB
- PD12M: https://huggingface.co/datasets/Spawning/PD12M · https://arxiv.org/abs/2410.23144
- Recap-DataComp-1B: https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B · DataComp: https://arxiv.org/pdf/2304.14108
- CC12M: https://huggingface.co/datasets/conceptual_12m · ImageNet-64: https://www.tensorflow.org/datasets/catalog/imagenet_resized
- DiffusionDB: https://huggingface.co/datasets/poloclub/diffusiondb
- text-to-image-2M: https://huggingface.co/datasets/jackyhate/text-to-image-2M · FLUX.2 synth: https://huggingface.co/datasets/KempnerInstituteAI/flux.2-dev-synthetic-2M
- CMMD: https://arxiv.org/abs/2401.09603 · GenColorBench: https://arxiv.org/abs/2510.20586 · PartiPrompts: https://arxiv.org/pdf/2206.10789 · DrawBench: https://arxiv.org/pdf/2205.11487
