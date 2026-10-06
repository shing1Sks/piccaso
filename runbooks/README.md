# Runbooks

The shell scripts that drove the final runs on one 8x RTX 5090 machine (notebook 17-18). Kept for reference: they assume the box
layout `/root/spike` with code from `src/` and data under `out/`.

- `dry0b.sh`: stage 0b dry run (2k PixelProse rows end to end, 565-slot ceiling probe, speed tests, eval plumbing)
- `stage1.sh`: stage 1, ~419k PixelProse URLs -> 361 strokes on 8 workers, scoring, merge
- `s2.sh`: stage 2, the 25M / 100M / 250M scaling check with shared evaluation loops
- `evalloop.sh`: scores every new checkpoint (StrokeBench + CLIP retrieval on PixelProse and DOCCI)

The final 100M continuation was launched with `torchrun --nproc_per_node 8 strokegen.py ... --init <8k ckpt> --batch 1024
--lr 5e-4 --max-steps 7000` (see notebook 18).
