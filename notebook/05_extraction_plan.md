# 05 - Stage-0 stroke extraction: hardware, cost, wait time, plan (2026-10-03)

Recipe = `final_fast` from 04: fit at 128 px, 160 strokes (16/48/96), 4 curve segments, torch.compile,
constraints (min width 0.008, on canvas, alpha 1.0), steps 150/150/100, no 256 px refine.
Measured on A40: **5,825 img/hr** (23.6 dB / LPIPS 0.316 at 256 on the mixed 40-image pilot; emoji and sketches are much better than the mean).

## Live RunPod prices and stock (read from the API, 2026-10-03)

| Card | VRAM | $/hr community | $/hr secure | Stock now (1 GPU) |
|---|---|---|---|---|
| RTX 4090 | 24 | **0.34** | 0.74 | **High** |
| RTX 5090 | 32 | 0.69 | 0.99 | none |
| RTX 3090 | 24 | 0.22 | 0.50 | Low |
| RTX 4080 / 4080S / 5080 / 4070 Ti | 16/16/16/12 | 0.27 / 0.28 / 0.39 / 0.19 | - | none |
| RTX A5000 / A4000 | 24 / 16 | 0.16 / 0.17 | 0.27 / 0.25 | none |
| A40 | 48 | 0.35 | 0.49 | Medium |
| RTX PRO 4500 Blackwell | 32 | - | **0.72** (the capacity API showed 0.34, which is a different tier with no stock) | High |
| L40S / RTX 6000 Ada | 48 | 0.79 / 0.74 | 1.09 / 0.84 | Medium / Low |
| A100 / H100 | 80 | 1.19-1.39 / 1.99-2.69 | 1.59 / 2.89-3.49 | not considered (no tensor-core use) |

4-GPU and 8-GPU 4090 pods: **unavailable** right now, so parallelism = several single-GPU pods.
Other marketplaces (web search, unverified, prices move): Vast.ai 4090 about $0.29-0.32/hr on demand (mixed host reliability), 5090 about $0.34-0.40; Salad about $0.16 for a 4090 (container platform, different workflow; not pursued).

## Throughput and cost estimates (100k images, fast recipe)

Only A40 (5,825) and the 4090-vs-A40 ratio (1.63x at 128 px, secure 4090) are measured. Everything else is a proxy:
speed = 0.91 x sqrt(FP32-TFLOPS ratio x memory-bandwidth ratio) vs A40, calibrated to the measured 4090. **Verify with a 10-minute benchmark before trusting.**
Note: `get-capacity` `pricePerHr` is the lowest price across tiers, not the tier you can rent; use `list-gpu-types` (community vs secure) or the `cost` returned by create-pod. A40 has no community stock: the real price is secure $0.49.
TFLOPS/bandwidth for 4090, 4080, 3090, 5090, PRO 4500 came from web search; the rest are from datasheet memory.

| Card @ price | img/hr | hours for 100k | **$ per 100k** | $ for 1.28M |
|---|---|---|---|---|
| 4090 community @0.34 | ~9,500 (measured ratio) | 10.5 | **~3.6** | ~46 |
| 3090 community @0.22 | ~6,000 | 16.6 | ~3.7 | ~47 |
| A5000 community @0.16 | ~4,800 | 20.8 | ~3.3 | ~43 |
| 5090 community @0.69 | ~14,300 | 7.0 | ~4.8 | ~62 |
| PRO 4500 secure @0.72 | ~6,600 | 15.0 | ~10.9 | ~140 |
| 5080 community @0.39 | ~7,600 | 13.1 | ~5.1 | ~65 |
| A40 secure @0.49 | 5,825 (measured) | 17.2 | 8.4 | ~108 |
| 4090 secure @0.74 | ~9,500 | 10.5 | 7.8 | ~100 |
| RTX 6000 Ada / L40S community | ~9,700 / ~9,300 | ~10.5 | ~7.6 / ~8.5 | ~97 / ~109 |

Conclusion: community **4090** is the pick (cheapest among cards that are in stock, and the only one with a measured speedup).
5090 is faster per card (shorter wait) but costs about 35% more per image; use it only when time matters and it is in stock.
Add ~25% for setup, compile warm-up, failed shards and download: realistic 100k = **$4.5-5**, 1.28M = **~$55-60**.

## Wait time

- One batch of 40 images: ~15 s on a 4090 (~25 s on an A40). A 2,000-image shard: ~13 min on a 4090.
- 100k images on one 4090: **~10.5 h**. On 4 pods in parallel: ~2.7 h. On 8 pods: ~1.4 h. Cost is about the same (each extra pod adds ~15 min of setup, ~$0.09).
- Plus ~15 min per pod for start-up, code upload, CUDA check, compile.

## Dataset size: 100k vs 1.28M

- 100k x 160 strokes x ~10 tokens = ~160M tokens. Plenty for a 10-50M-param model to learn Stage 0 (emoji + QuickDraw classes); this is a proof-of-concept size.
- 1.28M (ImageNet scale) only matters for the open-domain / text-conditioned phases (P2/P3), and there the caption data is a bigger bottleneck than strokes. Do not spend on it before P1 shows the strokes are learnable.
- Free multipliers on stroke data: horizontal flip (exact on the stroke parameters), small shift/scale, and several re-fits of one image with different seeds. (Flip is unsafe for text with left/right.)
- Storage: 100k x 160 x 10 values in fp16 = ~320 MB total. Trivial.

## Plan and budget (target $15-20)

1. Benchmark, ~$0.3: one 4090 community pod, check CUDA, run the fast recipe on 400 pilot-like images; confirm the 4090 throughput. If a 5090 / PRO 4500 / 3090 is in stock, benchmark it too for 5 min each.
2. Gate, ~$0.5-1: extract ~5-10k Stage-0 images; train a tiny class-conditional transformer on them (cheap card). Render samples. If they do not look like the classes, fix tokenization/ordering before spending more.
3. Full Stage 0, ~$4.5-5: ~100k images (3.7k Twemoji emoji + ~96k QuickDraw across chosen classes), sharded, run as several parallel 4090 pods.
4. P1 training runs, ~$2-4.
5. Leftover (~$5-10): second dataset (WikiArt/COCO subset) or the 256 px quality refine on a selected subset.

Operational rules: pods download raw data from the source (QuickDraw ndjson, Twemoji) instead of receiving 100k images from the laptop; 2,000-image shards, each saved with its seed and per-image fit PSNR; pull shards as they finish; verify `torch.cuda.is_available()` before work; terminate (never stop) pods when done; poll with short ssh calls.

## Account findings

- Billing API 2026-09-29..10-03: **$1.16 total** = $0.70 pod GPU + $0.02 pod disk + $0.45 storage. The balance itself is not exposed by the API; check the console.
- The storage charge ($0.10-0.12/day, ~$3.5/month) belongs to network volume `9hwsd2izo9`. `list-network-volumes` returns empty and `get-network-volume` returns 404 for that id, so it cannot be seen or deleted through the API. It was not created by my pods (it predates them). Check Console -> Storage; delete it there or contact support.

## Risks

- 4090 community hosts are heterogeneous (a broken-CUDA host happened once, pods that were stopped could not restart). Shard and verify.
- Throughput for every card except the A40 is a proxy until benchmarked; weak host CPUs could slow a GPU-light-launch workload.
- Stock moves fast (RTX 3090 showed "Low" and, a few minutes later in the same session, "unavailable"; 4090 showed High for 1 GPU but none for 4 or 8 GPUs).
- Learnability of the strokes is still untested (see gate in step 2).

## Vast.ai (live offers read via `vastai search offers`, which works without an API key; 2026-10-03)

Filters: 1 GPU, rentable, reliability > 0.97, >= 30 GB disk, >= 200 Mbps down. All returned hosts were `verified`.
Bandwidth is metered at roughly $0.001-0.017/GB (irrelevant for us, <2 GB of traffic); storage is $0.13-0.87/GB-month.

| GPU | offers | min $/hr | median $/hr | est. img/hr | est. $ per 100k at min price |
|---|---|---|---|---|---|
| RTX 3080 (10 GB) | 18 | 0.108 | 0.145 | ~4,900 | ~2.2 (10 GB VRAM may limit batch) |
| RTX 3090 | 34 | 0.134 | 0.268 | ~6,000 | ~2.2 |
| RTX 4070S Ti | 10 | 0.161 | 0.208 | ~5,700 | ~2.8 |
| RTX 5080 | 12 | 0.219 | 0.329 | ~7,650 | ~2.9 |
| RTX 4080 / 4080S | 7 / 7 | 0.221 / 0.242 | 0.254 / 0.296 | ~6,150 | ~3.6-3.9 |
| RTX 5090 | 48 | 0.406 | 0.602 | ~14,300 | ~2.9 |
| RTX 4090 | 57 | 0.348 | 0.469 | ~9,500 | ~3.7 |
| RTX A5000 / A6000 | 8 / 9 | 0.188 / 0.401 | 0.237 / 0.402 | ~4,800 / ~5,700 | ~3.9 / ~7 |
| L40S / RTX 6000 Ada | 9 / 11 | 0.535 / 0.509 | 0.80 / 0.63 | ~9,300 / ~9,700 | ~5.8 / ~5.2 |

Throughput is the same spec-based proxy as above (unmeasured). Takeaways:
- Vast is 10-40% cheaper per image than RunPod community, mostly because RunPod has no 5090 / 3090 stock; at the *same* card the 4090 prices are about equal ($0.34 vs $0.35).
- Best $/image on paper: 3090 / 3080 (~$2.2 per 100k, ~17-20 h per pod) and 5090 / 5080 (~$2.9, 7-13 h). Absolute differences are ~$1 per 100k, so convenience matters more than price at this scale.
- Needs a Vast account, an API key, prepaid credit, a registered SSH key *before* launching, and a different transfer path (`vastai copy` / direct ssh). Host quality varies (some offers have only 4-6 effective CPU cores).
- Vast's official agent docs: https://docs.vast.ai/guides/get-started/agents ; CLI skill source: github.com/vast-ai/vast-cli (`vastai/SKILL.md`, documentation + `allowed-tools: Bash(vastai:*)`, no install scripts). The `vastai` CLI (pip) was installed into `.vastlib/` in the project folder, not globally.

## Diverse-data sources verified reachable (2026-10-03, from the laptop; pods have general internet access)

| Source | Size | License | Status |
|---|---|---|---|
| QuickDraw (345 classes, ndjson streamed from GCS) | 50M drawings | CC-BY 4.0 | works, used in the gate |
| Twemoji SVG (GitHub jdecked/twemoji) | ~3.7k (1.4k single-codepoint) | CC-BY 4.0 | works, used in the gate |
| COCO train2017 images + 5 captions each | 118k | research / Flickr licenses | works (images.cocodataset.org; annotations zip reachable) |
| WikiArt (HF huggan/wikiart parquet via datasets-server /parquet) | 81k | research only | listing works; parquet bytes need a pod-side reader |
| Met Museum open access | 100k+ images | CC0 | API + image download work (earlier 410 was transient) |
| Cleveland Museum of Art open access | ~42k with images | CC0 | API + image download work |
| Art Institute of Chicago | 133k | CC0 | image server returns 403 even with referer: dropped |
| ImageNet-1k | 1.28M | gated | needs the user's HF token + terms acceptance |

Plan for the diverse 100k (after the gate passes): ~40k QuickDraw across many classes, ~3.7k emoji, ~20k COCO (with captions: the text-conditioning data for P3), ~15k WikiArt, ~10k Met, ~10k Cleveland. `extract_prod.py --manifest` already accepts `{id,label,source,url}` rows for URL-addressable sources.
