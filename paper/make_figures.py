"""Figures for the Piccaso-0.1 report. Every number here is copied from the notebook (notebook/09-18) or results/*.json.

  python make_figures.py   -> ../figures/*.png
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)
INK, ACCENT, ACCENT2, MUTED, GRID = "#1d1d1f", "#d9480f", "#1c7ed6", "#868e96", "#e9ecef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 160})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. the journey: CLIP gap of the best model at each milestone (two test sets: mixed captions, then PixelProse captions)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [5, 2]})
m1 = [("pilot\n6.5k imgs", 0.0102), ("full run\nB-22M", 0.0218), ("B-58M", 0.0245), ("recipe\nfix", 0.0270), ("361 slots +\nself-cond (22M)", 0.0273)]
a1.plot(range(len(m1)), [v for _, v in m1], "-o", color=ACCENT, lw=2)
for i, (k, v) in enumerate(m1):
    a1.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
a1.set_xticks(range(len(m1)), [k for k, _ in m1], fontsize=8)
a1.set_ylabel("CLIP gap (guidance 2)")
a1.set_title("Mixed data (96k images, 500 held-out captions)", fontsize=10)
a1.set_ylim(0, 0.035)
m2 = [("100M @ 8k\n2.0M seen", 0.0436), ("final 100M\n9.2M seen", 0.0595)]
a2.bar(range(2), [v for _, v in m2], color=[ACCENT2, ACCENT], width=0.55)
for i, (k, v) in enumerate(m2):
    a2.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8.5)
a2.set_xticks(range(2), [k for k, _ in m2], fontsize=8)
a2.set_title("PixelProse (1,000 held-out)", fontsize=10)
a2.set_ylim(0, 0.07)
save(fig, "fig_journey.png")

# 2. learning curve of the 100M on PixelProse + projection (power law, exponent ~0.21 per measured segments)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
seen = [0.68, 1.37, 2.05]
ax.plot(seen, [0.0412, 0.0489, 0.0523], "-o", color=ACCENT2, label="100M scaling check (guidance 3)")
ax.plot([2.05, 9.2], [0.0436, 0.0595], "-o", color=ACCENT, label="100M (guidance 2): 8k steps -> final")
import numpy as np

x = np.array([9.2, 20, 30, 51, 61])
for lo, hi in [(0.15, 0.25)]:
    ax.fill_between(x, 0.0595 * (x / 9.2) ** lo, 0.0595 * (x / 9.2) ** hi, color=ACCENT, alpha=0.12, label="projection with fresh data (exp. 0.15-0.25)")
ax.axhline(0.145, color=MUTED, ls="--", lw=1)
ax.text(0.7, 0.139, "ceiling: fitted strokes of the real photo (0.145)", color=MUTED, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("training pictures seen (millions, log)")
ax.set_ylabel("CLIP gap")
ax.legend(fontsize=7.5, loc="lower right")
ax.set_ylim(0, 0.16)
save(fig, "fig_learning_curve.png")

# 3. what each lever was worth (relative change of the CLIP gap, each measured in isolation)
lev = [("guidance 2 -> 4", 29), ("model 22M -> 58M", 14), ("LR decay + batch 256", 12), ("joint base+detail (361)", 9),
       ("self-conditioning", 8), ("per-word cross-attention", 5), ("2x data (old mix)", 4), ("250M vs 100M (equal steps)", 2),
       ("render feedback on top of self-cond", 2), ("denoising steps 25 -> 250", 0)]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.barh(range(len(lev))[::-1], [v for _, v in lev], color=[ACCENT if v >= 8 else ACCENT2 if v >= 4 else MUTED for _, v in lev])
for i, (k, v) in enumerate(lev):
    ax.text(v + 0.5, len(lev) - 1 - i, f"+{v}%", va="center", fontsize=8.5)
ax.set_yticks(range(len(lev))[::-1], [k for k, _ in lev], fontsize=8.5)
ax.set_xlabel("change in CLIP gap (%)")
ax.set_xlim(0, 35)
ax.grid(axis="y", visible=False)
save(fig, "fig_levers.png")

# 4. the stroke ceiling: can CLIP still find the caption from the strokes alone? (top-1 among 150 of the same source)
src = [("PixelProse", 0.95, 0.57, 0.75), ("DOCCI", 0.89, 0.45, 0.54), ("COCO", 0.78, None, 0.61), ("Icons", 0.63, None, 0.61),
       ("PD12M", 0.83, None, 0.49), ("OpenImages+LN", 0.54, 0.30, 0.41), ("COCO objects", 0.35, None, 0.29), ("QuickDraw", 0.25, None, 0.19),
       ("Cleveland", 0.37, None, 0.08), ("WikiArt", 0.29, None, 0.04)]
fig, ax = plt.subplots(figsize=(7.4, 3.4))
xs = np.arange(len(src))
ax.bar(xs - 0.27, [s[1] for s in src], 0.27, color=GRID, edgecolor=MUTED, label="real photo")
ax.bar(xs, [s[2] or 0 for s in src], 0.27, color=ACCENT2, alpha=0.55, label="165 strokes")
ax.bar(xs + 0.27, [s[3] for s in src], 0.27, color=ACCENT, label="361 strokes")
ax.set_xticks(xs, [s[0] for s in src], rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("caption retrieval top-1")
ax.legend(fontsize=8, ncol=3, loc="upper right")
ax.set_ylim(0, 1.05)
save(fig, "fig_ceiling.png")

# 5. scaling check: same data, settings and steps
fig, ax = plt.subplots(figsize=(5.4, 3.3))
steps = [2667, 5334, 8000]
ax.plot(steps, [0.0373, 0.0440, 0.0464], "-o", color=MUTED, label="25M")
ax.plot(steps, [0.0412, 0.0489, 0.0523], "-o", color=ACCENT2, label="100M")
ax.plot([2667], [0.0421], "s", color=ACCENT, ms=8, label="250M (+2% at equal steps)")
ax.set_xlabel("training steps (batch 256)")
ax.set_ylabel("CLIP gap (guidance 3)")
ax.legend(fontsize=8)
save(fig, "fig_scaling.png")

# 6. seen vs unseen vs a different source (final model), if the evaluation results are present
rp = Path(__file__).resolve().parent.parent / "results" / "final_eval.json"
if rp.exists():
    r = json.load(open(rp))["sets"]
    names = [("seen", "seen (training)"), ("unseen", "unseen (held-out)"), ("docci", "DOCCI (other source)")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(3)
    ax.bar(xs - 0.27, [r[k]["real_picture"]["top1"] for k, _ in names], 0.27, color=GRID, edgecolor=MUTED, label="real photo")
    ax.bar(xs, [r[k].get("fitted_strokes", {}).get("top1", 0) for k, _ in names], 0.27, color=ACCENT2, alpha=0.6, label="fitted 361 strokes")
    ax.bar(xs + 0.27, [r[k]["generated"]["top1"] for k, _ in names], 0.27, color=ACCENT, label="Piccaso-0.1 painting")
    for i, (k, _) in enumerate(names):
        v = r[k]["generated"]["top1"]
        ax.text(i + 0.27, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5)
    ax.text(2, 0.02, "n/a", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(xs, [n for _, n in names])
    ax.set_ylabel(f"caption retrieval top-1 (n={r['seen']['n']})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3)
    save(fig, "fig_seen_unseen.png")
print("figures ->", OUT)
