# 08 - New approach after the failed gate: stroke-set diffusion (proposal, 2026-10-03)

Status: proposal for the user's review. Evidence in 06 (gate) and 07 (research).

## Pipeline

```
image --[extractor v2: anchored, ordered, pruned]--> 165 stroke slots (keep, geometry, width, colour)
                                                          | training data
class --> [Stroke-set diffusion transformer, ~14M] <-----+
            all 165 slots denoised together, 25-50 steps
                         |
                    165 strokes --> fixed renderer --> painting (replayed big->small, in slot order)
```

## 1. Extraction v2 (the data)
- **Anchored slots.** Three levels on fixed grids: 16 (4x4), 49 (7x7), 100 (10x10) = 165 slots. Each slot starts at its grid cell (cell centre, plus the cell's dominant colour), not at a randomly sampled error pixel, and its position is stored relative to the cell centre. Same image -> same strokes; similar images -> similar slots.
- **Order baked in.** Slots are painted in level order, then row-major cell order, during the fit itself, so the stored order is the order the fit used: render-exact, nothing to sort afterwards.
- **Canonical direction.** After each optimizer step, swap p0/p2 so p0 is the top-left endpoint (render-identical, verified 100 dB).
- **Keep bit.** After the fit, a slot whose removal costs under a small PSNR threshold is marked keep = 0 (not drawn). A 5-line doodle uses few slots instead of 160 fragments.
- **Check before use:** fit PSNR within ~1 dB of extractor v1, and a "consistency" check (a flipped/shifted image gives correspondingly flipped/shifted slots).
- **Data:** 6 QuickDraw classes x 2,500 + 1,000 emoji (~16K images). A40 ~3.2 h (~$1.6) or a 4090 ~2 h (~$0.6).

## 2. Representation (one vector per slot, continuous)
`[keep, dx0, dy0 (start, relative to the slot's cell centre), dx1, dy1, dx2, dy2 (control and end, relative to the start), log width, r, g, b]` = 11 numbers, each normalized to about [-1, 1]. No quantization. Alpha stays 1.

## 3. Model: stroke-set diffusion transformer (CNP / DiT style)
- 165 tokens. Token = Linear(noisy 11-vector) + embedding of (level, cell row, cell col).
- **Bidirectional** attention: every stroke attends to every other stroke at every denoising step.
- Class + diffusion time through **adaLN-Zero**. 10% class dropout for classifier-free guidance.
- Size: 8 layers x 384 wide, ~14M params (CNP-S is ~24M).
- Loss: (a) v-prediction MSE on the 11 numbers (masked geometry when keep = 0); (b) an auxiliary **render loss**: render the model's predicted clean strokes at 64 px with our differentiable renderer and compare (L1) with the render of the true strokes, weighted toward low noise levels.
- Sampling: DDIM 25-50 steps, all slots at once, CFG ~2-4.
- Speed estimate (unmeasured): ~4.6 GFLOP per pass, x2 for CFG, x30 steps = ~280 GFLOP -> ~1-3 s on a laptop CPU.

## 4. Control arm: native QuickDraw strokes
Each human pen stroke fitted with 1-3 quadratic Beziers by least squares -> ~10-40 strokes in the human's own order, padded with keep = 0 slots. Same model. Separates "the model can't learn strokes" from "our decomposition is hard to learn".

## 5. Evaluation
- Recognizability: a small CNN trained on QuickDraw bitmaps (the dataset ships 28x28 numpy bitmaps) classifies rendered samples; report accuracy per class. Plus eyeballing sheets.
- Diversity: sample spread per class; nearest-training-neighbour check against memorization.
- Pass: most samples recognizable in the control arm *and* the main arm.

## 6. Fault -> fix map
| Gate fault | Fix |
|---|---|
| 1 number = 1 token (1,600 positions): CNP's diagnosed failure | 1 slot per stroke, 165 positions, continuous vector |
| CE treats a 1-bin miss like a 100-bin miss | continuous MSE (error grows with distance) + render loss |
| one arbitrary decomposition is "the" answer | diffusion models a distribution; render loss accepts equivalent strokes; anchored extraction removes most arbitrariness |
| random stroke order | slot = grid cell; order fixed during the fit (render-exact) |
| reversible curves | canonical p0 (free) |
| absolute coordinates | start relative to cell, control/end relative to start |
| no canvas, strokes known only as numbers | all strokes see each other at every step; render loss gives a pixel signal; (v2 option: feed a 32 px render of the current estimate) |
| inconsistent decompositions (random init) | deterministic anchored init |
| 160 fragments for a 5-line doodle | keep bit + control arm with human strokes |
| 300 per class, overfit after ~4K steps | 2,500 per class, early stop on val, flips, CFG dropout |
| tiny model | ~14M once data is fixed |
| temperature/top-k sampling of high-entropy tokens | DDIM + CFG |
| judged by eye only | QuickDraw classifier accuracy + nearest-neighbour check |

## 7. Concerns about this architecture, and answers
1. **It no longer decides stroke-by-stroke while watching the canvas.** True: all strokes are decided jointly (arguably better: a stroke can be placed knowing every other stroke). The painting animation is unchanged, because the output is still ordered big->small and spatially. If live co-painting (user adds a stroke, model continues) matters, CNP shows the same diffusion model does it by masking (context strokes kept clean). Fallback arm: per-stroke AR with a flow head and a rendered-canvas input (PaintCopilot style).
2. **Fixed slot count.** Handled by the keep bit; levels can be added later.
3. **CPU speed with 25-50 steps.** Each step is parallel over 165 slots, so it is cheaper than 1,600 sequential token steps. Distillation to 4-8 steps exists if needed.
4. **Small data.** CNP used ~10K per class; we start at 2,500. If the main arm is close but blurry, scale data before changing design.
5. **Anchored extraction may waste slots on empty background / cost quality.** Measured before training (pass bar: within ~1 dB of v1). Fallback: deterministic error-driven placement plus sort.
6. **Text later.** adaLN takes a frozen CLIP text embedding in place of the class embedding (as LAFITE/GALIP do for GANs).
7. **Novelty.** The core model follows CNP closely; our delta stays: much smaller, CPU, broader vocabulary, Bezier strokes.

## 7b. Cost
Extraction ~$0.6-1.6, two training arms ~1-1.5 GPU-hours (~$0.5-0.8), classifier training minutes. Total ~$2-3.
