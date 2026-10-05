"""Stage-2 scaling-check table (research/17): StrokeBench + CLIP gap (held-out PixelProse, DOCCI) per model and checkpoint.

  python s2_report.py out/s2 [out/base]  -> prints a markdown table + the decision numbers (100M vs 250M)
"""
import glob
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def gaps(d):
    if not d:
        return "", ""
    g = d["global"]
    return g.get("steps25_cfg2", {}).get("gap", ""), g.get("steps50_cfg3", {}).get("gap", "")


rows = []
for run in sorted(root.glob("m*")):
    for c in sorted(glob.glob(str(run / "ckpt_*.pt"))) + [str(run / "ckpt.pt")]:
        tag = c[:-3]
        m = re.search(r"ckpt_(\d+)", c)
        step = int(m.group(1)) if m else 8000
        sb, dg, dc = load(tag + ".sb.json"), load(tag + ".diag.json"), load(tag + ".docci.json")
        if not (sb or dg):
            continue
        rows.append((run.name, step, sb, gaps(dg), gaps(dc)))
print("| model | step | objects top-1 | top-5 | colour | colour obj | two (both) | style | PP gap cfg2 | PP gap cfg3 | DOCCI gap cfg2 | DOCCI gap cfg3 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, step, sb, pg, dg in rows:
    s = sb or {}
    print(f"| {name} | {step} | {s.get('objects', '')} | {s.get('objects_top5', '')} | {s.get('colour', '')} | {s.get('colour_obj', '')} | "
          f"{s.get('two_both', '')} | {s.get('style', '')} | {pg[0]} | {pg[1]} | {dg[0]} | {dg[1]} |")
if base:
    for b in sorted(base.glob("*")):
        sb, dg = load(b / "strokebench.json"), load(b / "pp.diag.json")
        s = sb or {}
        print(f"| baseline {b.name} | - | {s.get('objects', '')} | {s.get('objects_top5', '')} | {s.get('colour', '')} | {s.get('colour_obj', '')} | "
              f"{s.get('two_both', '')} | {s.get('style', '')} | {gaps(dg)[0]} | {gaps(dg)[1]} | | |")
# decision numbers: 250M vs 100M at each common step (CLIP gap on held-out PixelProse at cfg3, StrokeBench objects top-5)
by = {(n, st): (sb, pg) for n, st, sb, pg, _ in rows}
for st in sorted({st for _, st, _, _, _ in rows}):
    a, b = by.get(("m100", st)), by.get(("m250", st))
    if a and b and a[1][1] and b[1][1]:
        print(f"step {st}: PP gap cfg3 100M {a[1][1]} vs 250M {b[1][1]} -> {100 * (b[1][1] / a[1][1] - 1):+.1f}%; "
              f"objects top-5 {a[0].get('objects_top5') if a[0] else ''} vs {b[0].get('objects_top5') if b[0] else ''}")
