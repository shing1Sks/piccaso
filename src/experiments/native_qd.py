"""Control-arm data: QuickDraw's own pen strokes -> quadratic Beziers in the human drawing order (CPU, no optimizer).

Each pen stroke (polyline) is split into the fewest equal-arc-length pieces (1..3) whose least-squares quadratic Bezier
fits within --tol (in 0..255 drawing units); endpoints are kept, the control point is solved in closed form.
Output uses the same 11-number stroke format as the extractors (black, width = the raster line width, alpha 1),
padded to --slots with keep = 0.

  python native_qd.py out/v2/items.json out/native/native.pt
"""
import argparse
import json

import numpy as np
import torch

import batched

LINE_W = 3.5 / 128  # soft renderer width that matches the raster ink amount (16/512 over-inks by ~15%)


def fit_quad(P):
    """P (m,2) points -> (A, C, D, max_err) with A=P[0], D=P[-1], chord-length parameters, least-squares C."""
    A, D = P[0], P[-1]
    if len(P) <= 2:
        return A, (A + D) / 2, D, 0.0
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = d / max(d[-1], 1e-9)
    w = 2 * (1 - t) * t
    r = P - ((1 - t) ** 2)[:, None] * A - (t ** 2)[:, None] * D
    C = (w[:, None] * r).sum(0) / max((w ** 2).sum(), 1e-9)
    B = ((1 - t) ** 2)[:, None] * A + w[:, None] * C + (t ** 2)[:, None] * D
    return A, C, D, float(np.linalg.norm(B - P, axis=1).max())


def pen_stroke_to_beziers(xs, ys, tol, max_pieces=8):
    P = np.stack([xs, ys], 1).astype(np.float64)
    if len(P) == 1:
        return [(P[0], P[0], P[0])]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    for k in range(1, max_pieces + 1):
        cuts = [0] + [int(np.searchsorted(d, d[-1] * j / k)) for j in range(1, k)] + [len(P) - 1]
        cuts = sorted(set(cuts))
        pieces = [fit_quad(P[a:b + 1]) for a, b in zip(cuts[:-1], cuts[1:])]
        if max(p[3] for p in pieces) <= tol or k == max_pieces:
            return [(A, C, D) for A, C, D, _ in pieces]


def drawing_to_slots(drawing, slots, tol):
    out = []
    for xs, ys in drawing:
        for A, C, D in pen_stroke_to_beziers(xs, ys, tol):
            out.append(np.r_[A / 256, C / 256, D / 256, LINE_W, 0, 0, 0, 1.0])
    out = out[:slots]
    S = np.zeros((slots, batched.N_PARAMS), np.float32)
    S[:len(out)] = np.array(out, np.float32)
    keep = np.zeros(slots, bool)
    keep[:len(out)] = True
    return S, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1.5)
    ap.add_argument("--check", type=int, default=64, help="images to compare against the raster render")
    a = ap.parse_args()
    items = [it for it in json.load(open(a.items)) if it["source"] == "quickdraw"]
    S, K = zip(*(drawing_to_slots(it["drawing"], a.slots, a.tol) for it in items))
    S, K = torch.from_numpy(np.stack(S)), torch.from_numpy(np.stack(K))
    n = K.sum(1).float()
    print(f"{len(items)} drawings; Beziers per drawing mean {n.mean():.1f}, max {int(n.max())}, "
          f"truncated at {a.slots}: {int((n >= a.slots).sum())}")
    if a.check:
        from extract_prod import SIZE, load_item, to_tensor

        tgt = torch.stack([to_tensor(load_item(it)) for it in items[:a.check]])
        with torch.no_grad():
            rec = torch.cat([batched.render(S[i:i + 8], SIZE, SIZE) for i in range(0, a.check, 8)])
        print(f"render check vs raster ({a.check} drawings, {SIZE} px): PSNR {batched.psnr(rec, tgt).mean():.2f} dB")
    torch.save({"strokes": S.half(), "keep": K, "ids": [it["id"] for it in items], "labels": [it["label"] for it in items],
                "sources": ["quickdraw"] * len(items), "size": 128, "version": "native"}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
