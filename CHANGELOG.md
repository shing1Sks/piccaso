# Changelog / decision log

## 0.1 (2026-10-06): first public research preview

- **Model:** Piccaso-0.1, 102M set diffusion transformer over 361 brush strokes; Long-CLIP-B text (pooled + cross-attention),
  self-conditioning, 25% reference-image conditioning. Trained on 232,134 PixelProse images, ~9.2M picture-views.
- **Why set diffusion, not stroke-by-stroke:** it won at every scale (notebook 09, 11, 13).
- **Why 361 slots:** joint base + detail generation gave +7-10% (faces +64%); a finer 565-slot layer only +3% caption match (notebook 14, 18).
- **Why PixelProse only:** caption quality beat volume; WikiArt/Cleveland titles were noise (notebook 14, 15).
- **Why 100M, not 250M:** +2% at equal steps for 2.5x the cost at this data size (notebook 18).
- **Dropped for now:** render feedback (no gain at 1.9x cost), separate detail model, stroke-by-stroke arm (kept in code).
- **Not released:** the derived stroke dataset (image rights); pipeline and manifests are in `src/`.
