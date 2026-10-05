# Lab notebook

The raw, chronological record of the project, written while it happened. Nothing here was cleaned up after the fact: plans include
ideas that turned out wrong, and results include the failures. Paths like `spike/foo.py` refer to the original working directory;
the code now lives in `../src/` (and `../src/experiments/` for earlier scripts). Costs are real rental costs at the time.

| # | Note | What happened |
|---|---|---|
| 00 | [Design](00_DESIGN.md) | The original idea ("Brushwork"), the pressure test, what exists already |
| 01 | [Prior work: painting](01_prior_work_painting.md) | Stroke-based rendering, learned painters, CNP |
| 02 | [Prior work: sequence models](02_prior_work_sequence_models.md) | How to generate strokes |
| 03 | [Datasets](03_datasets.md) | What data could teach painting |
| 04 | [Pipeline decisions](04_pipeline_decisions.md) | Stroke format, budget (160 is the knee), fitting recipe, hardware |
| 05 | [Extraction plan](05_extraction_plan.md) | First extraction run plan |
| 06 | [Gate results](06_gate_results.md) | A GPT over per-number stroke tokens: scribbles. Why. |
| 07 | [Generator research](07_generator_research.md) | Literature says: diffusion over one vector per stroke |
| 08 | [New approach](08_new_approach.md) | Anchored slots + two arms: stroke-by-stroke (A) vs set diffusion (B) |
| 09 | [v2 probe results](09_results_v2_probe.md) | B 97% class accuracy; the representation was the problem |
| 10 | [Scaling plan](10_scaling_plan.md) | Text conditioning, captioners, mixed pilot |
| 11 | [Loss and arms](11_loss_and_arms.md) | Memorisation test: the loss is sound; why B beats A |
| 12 | [Full run plan](12_full_run_plan.md) | 96k mixed images, detail layer, captions |
| 13 | [Full run results](13_full_run_results.md) | First recognisable objects (B-58M); the detail model fails |
| 14 | [Where B loses](14_where_b_loses.md) | Steps don't matter, guidance does, an LR bug, data flattening, 361 joint slots |
| 15 | [Phase A + data probe](15_phaseA_and_data_probe.md) | Self-conditioning, render feedback, cross-attention; PixelProse wins |
| 16 | [Learnings](16_learnings.md) | Everything so far, compiled |
| 17 | [Final run plan](17_final_run_plan.md) | PixelProse-only base, Long-CLIP, 100M vs 250M check |
| 18 | [Final run results](18_final_run_results.md) | Piccaso-0.1 |
