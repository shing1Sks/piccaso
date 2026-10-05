# Earlier experiments

Scripts from the earlier phases, kept as they were run so every result in the notebook can be traced back.
Run them from `src/` with `src` on the path, e.g. `PYTHONPATH=. python experiments/memtest.py ...`.

| Script | Notebook | What it tested |
|---|---|---|
| `stroke.py`, `compare_formats.py`, `sweep.py`, `bench_painttransformer.py`, `quant_ablation.py`, `audit.py` | 04 | stroke format, budget, fitting recipe, Paint Transformer baseline |
| `extract.py` | 04-05 | first extractor (unanchored strokes) |
| `native_qd.py` | 08-09 | QuickDraw pen strokes as a control |
| `keephist.py`, `show_strokes.py`, `make_sheet.py` | 09-10 | keep-bit statistics, stroke sheets |
| `build_mixed_manifest.py`, `build_full_manifest.py`, `fetch_wikiart.py`, `merge_full.py` | 10-13 | the mixed 96k dataset |
| `caption_v2.py`, `clip_pass.py` | 12-13 | Florence-2 captions, CLIP filters |
| `memtest.py`, `mem_sheet.py` | 11 | memorisation test: is the loss sound? |
| `detailgen.py`, `detail_sweep.py`, `detail_sheet.py` | 13 | the separate detail model (dropped) |
| `exp14_report.py` | 14 | where B loses |

The failed per-number GPT generator lives in `../train_gate.py` (its transformer block is still used by the stroke-by-stroke arm).
