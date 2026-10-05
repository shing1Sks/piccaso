"""Detail level for extractor v3: a g x g grid of small strokes fitted at high resolution ON TOP of the base strokes.

Why a special renderer: a plain render costs (strokes x all pixels); at 256 px with 196 strokes that is ~5x the whole base fit.
Here every detail stroke is kept inside a box around its own cell (control points within +-`reach` cells of the cell
centre, width <= one cell), so its paint can only land in a small window. Only window pixels are computed.
Order: cells are visited in M*M = 16 groups (row % 4, col % 4), row-major inside a group. Windows of one group never overlap,
so a whole group is composited at once and the result equals painting the strokes one by one in that order.
Slot order is therefore group-major; it is fixed and identical for every image (the model sees the same slot->cell map).

  detail_order(g) -> list of cell ids in paint order
  fit_detail(target_hi, base_hi, g, steps) -> strokes (B, g*g, 11) in paint order, canvas (B,3,HW)
"""
import math

import torch

import batched

REACH = 1.0  # control points may move this many cells away from the cell centre
SOFT = 0.75  # same soft edge as batched.stroke_masks (pixels)
MARGIN = 5  # pixels of soft edge kept inside a window (coverage there < 0.2%)
M = 4  # cells of one group are M apart; windows (2*(REACH+0.5) cells + 2*MARGIN px) must fit in M cells


def detail_order(g, m=M):
    return [r * g + c for gr in range(m) for gc in range(m) for r in range(gr, g, m) for c in range(gc, g, m)]


def _windows(g, H, dev):
    """Per cell: flat pixel indices of its window in a padded canvas, and the window pixel centres in [0,1] coords."""
    cell = H / g
    half = math.ceil(cell * (REACH + 0.5) + MARGIN)  # reach + half the max width + soft edge
    assert 2 * half <= M * cell + 1, "windows of one group would overlap"
    P = 2 * half
    pad = half + 1
    Hp = H + 2 * pad
    idx, ctr = [], []
    oy, ox = torch.meshgrid(torch.arange(P, device=dev), torch.arange(P, device=dev), indexing="ij")
    for k in range(g * g):
        r, c = divmod(k, g)
        cy, cx = (r + 0.5) * cell, (c + 0.5) * cell
        y0, x0 = int(math.floor(cy)) - half, int(math.floor(cx)) - half
        yy, xx = y0 + oy, x0 + ox  # unpadded pixel coords (may be outside the image)
        idx.append(((yy + pad) * Hp + (xx + pad)).reshape(-1))
        ctr.append(torch.stack([(xx + 0.5) / H, (yy + 0.5) / H], -1).reshape(-1, 2).float())
    return torch.stack(idx), torch.stack(ctr), pad, Hp  # (g*g, P*P), (g*g, P*P, 2)


def _pad(img, H, pad, Hp):
    B = img.shape[0]
    out = torch.ones(B, 3, Hp, Hp, device=img.device, dtype=img.dtype)
    out[:, :, pad:pad + H, pad:pad + H] = img.view(B, 3, H, H)
    return out.view(B, 3, -1)


def _crop(imgp, H, pad, Hp):
    B = imgp.shape[0]
    return imgp.view(B, 3, Hp, Hp)[:, :, pad:pad + H, pad:pad + H].reshape(B, 3, H * H)


def local_masks(s, ctr, H, K=4):
    """s (B, n, 11), ctr (n, P, 2) -> coverage (B, n, P) on each stroke's own window."""
    B, n, _ = s.shape
    t = torch.linspace(0, 1, K + 1, device=s.device)[None, None, :, None]
    p0, p1, p2 = s[..., None, 0:2], s[..., None, 2:4], s[..., None, 4:6]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2  # (B, n, K+1, 2)
    a, ab = pts[:, :, :-1], pts[:, :, 1:] - pts[:, :, :-1]
    ab2 = (ab * ab).sum(-1) + 1e-8
    best = None
    for k in range(K):
        ag = ctr[None] - a[:, :, k, None]  # (B, n, P, 2)
        u = ((ag * ab[:, :, k, None]).sum(-1) / ab2[:, :, k, None]).clamp(0, 1)
        d = ag - u[..., None] * ab[:, :, k, None]
        d2 = (d * d).sum(-1)
        best = d2 if best is None else torch.minimum(best, d2)
    return torch.sigmoid((s[..., 6:7] / 2 - (best + 1e-12).sqrt()) * H / SOFT)


_lm_compiled = None


def render_detail(s, base, g, H, win=None, compiled=False):
    """s (B, g*g, 11) in paint order (detail_order), base (B, 3, H*H) -> canvas (B, 3, H*H). Differentiable."""
    idx, ctr, pad, Hp = win or _windows(g, H, s.device)
    order = detail_order(g)
    cv = _pad(base, H, pad, Hp)
    m = M
    per = [len(range(gr, g, m)) * len(range(gc, g, m)) for gr in range(m) for gc in range(m)]
    j = 0
    for n in per:
        cells = torch.tensor(order[j:j + n], device=s.device)
        sg = s[:, j:j + n]
        lm = _lm_compiled if compiled and _lm_compiled is not None else local_masks
        am = sg[..., 10:11] * lm(sg, ctr[cells], H)  # (B, n, P)
        ix = idx[cells].reshape(-1)  # windows of one group are disjoint
        loc = cv[:, :, ix].view(cv.shape[0], 3, n, -1)
        new = loc * (1 - am[:, None]) + sg[..., 7:10].permute(0, 2, 1)[..., None] * am[:, None]
        cv = cv.index_copy(2, ix, new.reshape(cv.shape[0], 3, -1))
        j += n
    return _crop(cv, H, pad, Hp)


def clamp_detail(s, g, order_t):
    """Keep each stroke inside its box: control points within REACH cells of its cell centre, width <= one cell."""
    with torch.no_grad():
        r, c = order_t // g, order_t % g
        cen = torch.stack([(c + 0.5) / g, (r + 0.5) / g], -1)  # (n, 2) x, y
        lo, hi = cen - REACH / g, cen + REACH / g
        for j in (0, 2, 4):
            s[..., j:j + 2] = torch.maximum(torch.minimum(s[..., j:j + 2], hi), lo)
        s[..., 6].clamp_(0.004, 1.0 / g)
        s[..., 7:10].clamp_(0, 1)
        s[..., 10] = 1.0


def init_detail(target, canvas, g, H):
    """Like extract_v2.init_anchored (error-weighted centroid + colour per cell), returned in paint order."""
    from extract_v2 import init_anchored

    s = init_anchored(target, canvas, g, 0.6 / g, H)
    return s[:, detail_order(g)]


def fit_detail(target, base, g, H, steps=100, lr=0.004, compiled=False):
    """target, base (B, 3, H*H) at the high resolution. Returns strokes in paint order and the final canvas."""
    dev = target.device
    order_t = torch.tensor(detail_order(g), device=dev)
    win = _windows(g, H, dev)
    s = init_detail(target, base, g, H)
    clamp_detail(s, g, order_t)
    s.requires_grad_(True)
    opt = torch.optim.Adam([s], lr=lr)
    global _lm_compiled
    if compiled and _lm_compiled is None:
        _lm_compiled = torch.compile(local_masks, dynamic=False)
    for _ in range(steps):
        out = render_detail(s, base, g, H, win, compiled)
        diff = out - target
        loss = (diff.pow(2).mean(dim=(1, 2)) + 0.5 * diff.abs().mean(dim=(1, 2))).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        clamp_detail(s, g, order_t)
    with torch.no_grad():
        canvas = render_detail(s.detach(), base, g, H, win)
    return s.detach(), canvas


@torch.no_grad()
def detail_keep(s, base, target, g, H, prune):
    """Keep a detail stroke only if it is visible (removal changes the picture by >= prune, extract_v2 units) AND it helps:
    removing it would make the error against the real image larger. Greedy, one pass in reverse paint order."""
    win = _windows(g, H, s.device)
    s = s.clone()
    err = lambda img: (img - target).pow(2).mean((1, 2))
    full = render_detail(s, base, g, H, win)
    e_full = err(full)
    keep = torch.ones(s.shape[:2], dtype=torch.bool, device=s.device)
    for i in reversed(range(s.shape[1])):
        a = s[:, i, 10].clone()
        s[:, i, 10] = 0
        wo = render_detail(s, base, g, H, win)
        vis = (wo - full).abs().sum(1).mean(1) >= prune
        k = vis & (err(wo) > e_full)
        keep[:, i] = k
        s[:, i, 10] = torch.where(k, a, torch.zeros_like(a))  # dropped strokes stay off
        full = torch.where(k[:, None, None], full, wo)
        e_full = torch.where(k, e_full, err(wo))
    return keep
