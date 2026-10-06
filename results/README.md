# Final evaluation (Piccaso-0.1)

- `final_eval.json`: caption retrieval (CLIP ViT-B/32) among 200 captions for seen (training), unseen (held-out PixelProse) and
  DOCCI captions; real photo vs fitted 361 strokes vs generated painting (25 DDIM steps, guidance 3). Produced by `src/eval_final.py`
  on an RTX 5090 from an evaluation pack built with `src/make_eval_pack.py`.
- `strokebench.json`: StrokeBench, 200 fixed prompts x 4 samples (`src/strokebench.py`).

Sample sheets are in `../figures/` (`eval_seen.jpg`, `eval_unseen.jpg`, `eval_docci.jpg`, `strokebench.jpg`).
