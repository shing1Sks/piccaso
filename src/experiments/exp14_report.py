"""Collect experiments 1-4 (research/14) from out/exp14/<box>/<run>/{diag.json,metrics.json} into markdown tables."""
import glob
import json
from pathlib import Path

runs = {}
for f in glob.glob("out/exp14/*/*/diag.json"):
    r = Path(f).parent
    m = json.load(open(r / "metrics.json")) if (r / "metrics.json").exists() else {}
    runs[r.name] = {"diag": json.load(open(f)), "metrics": m}

print("## Global (first 500 held-out captions)")
print("| run | params | steps | best val | top-1 | gap | gap / fitted | fitted top-1 | fitted gap |")
print("|---|---|---|---|---|---|---|---|---|")
for k, v in sorted(runs.items()):
    g, m = v["diag"]["global"], v["metrics"]
    gen = g.get("steps50_cfg2") or g.get("steps50_cfg2.0")
    if gen is None:
        gen = next(x for n, x in g.items() if n.startswith("steps50"))
    fs = g["fitted_strokes"]
    print(f"| {k} | {m.get('params_M', '')} | {m.get('steps', '')} | {round(m['best_val'], 4) if 'best_val' in m else ''} | "
          f"{gen['top1']} | {gen['gap']} | {gen['gap'] / fs['gap']:.2f} | {fs['top1']} | {fs['gap']} |")

print("\n## Sampler settings (global)")
for k, v in sorted(runs.items()):
    g = v["diag"]["global"]
    print(k, {n: (x["top1"], x["gap"]) for n, x in g.items() if n.startswith("steps")})

print("\n## Per source: generated gap / fitted gap (top-1 generated | fitted | real)")
srcs = sorted({s for v in runs.values() for s in v["diag"]["per_source"]})
print("| run | " + " | ".join(srcs) + " |")
print("|---|" + "---|" * len(srcs))
for k, v in sorted(runs.items()):
    ps = v["diag"]["per_source"]
    cells = [f"{ps[s]['gap_generated_over_fitted']:.2f} ({ps[s]['generated']['top1']:.2f} / {ps[s]['fitted_strokes']['top1']:.2f} / "
             f"{ps[s]['real_picture']['top1']:.2f})" if s in ps else "" for s in srcs]
    print(f"| {k} | " + " | ".join(cells) + " |")
