"""Batched Bezier-stroke decomposer: fits strokes to many images at once.

Same stroke format and the same coarse-to-fine algorithm as stroke.py (11 numbers per stroke:
p0 p1 p2 control points, width, r g b, alpha), but every tensor carries a leading image dimension B.
Each image keeps its own strokes and its own loss. Adam is per-parameter, so images do not interact.

Memory notes: the per-stroke distance field is the big tensor. It is computed one curve segment at a time
and, with checkpoint=True, recomputed in the backward pass instead of stored.
"""
import torch
from torch.utils.checkpoint import checkpoint

N_PARAMS = 11


def grid(H, W, device):
    ys, xs = torch.meshgrid(
        (torch.arange(H, device=device) + 0.5) / H, (torch.arange(W, device=device) + 0.5) / W, indexing="ij"
    )
    return torch.stack([xs, ys], -1).reshape(-1, 2)  # (HW, 2)


def _min_dist2(pts, g):
    """pts (M, K+1, 2) polyline, g (HW, 2) -> squared distance from every pixel to the polyline, (M, HW)."""
    a, b = pts[:, :-1], pts[:, 1:]
    ab = b - a
    ab2 = (ab * ab).sum(-1) + 1e-8
    best = None
    for k in range(a.shape[1]):
        ak, abk = a[:, k, None], ab[:, k, None]  # (M, 1, 2)
        ag = g[None] - ak  # (M, HW, 2)
        u = ((ag * abk).sum(-1) / ab2[:, k, None]).clamp(0, 1)
        d = ag - u[..., None] * abk
        d2 = (d * d).sum(-1)
        best = d2 if best is None else torch.minimum(best, d2)
    return best


def stroke_masks(s, g, H, K=8, soft=0.75):
    """s (M, 11) -> soft coverage (M, HW)."""
    t = torch.linspace(0, 1, K + 1, device=s.device)[None, :, None]
    p0, p1, p2 = s[:, None, 0:2], s[:, None, 2:4], s[:, None, 4:6]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    d = (_min_dist2(pts, g) + 1e-12).sqrt()
    return torch.sigmoid((s[:, 6:7] / 2 - d) * H / soft)


def composite(s, masks, base):
    """s (B, N, 11), masks (B, N, HW), base (B, 3, HW). Paint strokes in order, closed-form 'over' compositing."""
    am = s[:, :, 10:11] * masks
    rev = torch.cumprod((1 - am).flip(1), 1).flip(1)  # rev[:, i] = prod_{j>=i} (1 - am_j)
    after = torch.cat([rev[:, 1:], torch.ones_like(rev[:, :1])], 1)  # strokes painted after i
    return torch.einsum("bnc,bnp->bcp", s[:, :, 7:10], am * after) + base * rev[:, 0:1]


def render(s, H, W, base=None, mask_fn=None, use_checkpoint=False):
    """s (B, N, 11) -> (B, 3, HW). base defaults to a white canvas."""
    B, N, _ = s.shape
    if base is None:
        base = torch.ones(B, 3, H * W, device=s.device)
    if N == 0:
        return base
    g = grid(H, W, s.device)
    fn = mask_fn or stroke_masks
    flat = s.reshape(B * N, N_PARAMS)
    if use_checkpoint:
        masks = checkpoint(fn, flat, g, H, use_reentrant=False)
    else:
        masks = fn(flat, g, H)
    return composite(s, masks.reshape(B, N, -1), base)


DEFAULT_CLAMP = {"pos": (-0.1, 1.1), "width": (0.004, 0.6), "alpha": (0.2, 1.0)}


def clamp_(s, pos=(-0.1, 1.1), width=(0.004, 0.6), alpha=(0.2, 1.0)):
    with torch.no_grad():
        s[..., 0:6].clamp_(*pos)
        s[..., 6].clamp_(*width)
        s[..., 7:10].clamp_(0, 1)
        s[..., 10].clamp_(*alpha)


def init_strokes(n, target, canvas, H, W, width):
    """Place n new strokes per image where that image's error is largest, colored from its target."""
    B = target.shape[0]
    dev = target.device
    err = (target - canvas).abs().sum(1) + 1e-6  # (B, HW)
    idx = torch.multinomial(err, n, replacement=err.shape[1] < n)  # (B, n)
    c = grid(H, W, dev)[idx]  # (B, n, 2)
    ang = torch.rand(B, n, device=dev) * 3.1416
    dirv = torch.stack([ang.cos(), ang.sin()], -1) * width * (0.5 + torch.rand(B, n, 1, device=dev))
    s = torch.empty(B, n, N_PARAMS, device=dev)
    s[..., 0:2] = c - dirv
    s[..., 2:4] = c + 0.1 * width * torch.randn(B, n, 2, device=dev)
    s[..., 4:6] = c + dirv
    s[..., 6] = width
    s[..., 7:10] = torch.gather(target, 2, idx[:, None, :].expand(-1, 3, -1)).transpose(1, 2)
    s[..., 10] = 0.9
    return s


def psnr(a, b):
    """Per-image PSNR, (B,)."""
    return 10 * torch.log10(1 / (a - b).pow(2).mean(dim=(1, 2)).clamp_min(1e-10))


def decompose(target, H, W, stages, steps=150, lr=0.01, mask_fn=None, use_checkpoint=False, clamp=None):
    """target (B, 3, HW) in [0, 1]. stages: [(n_strokes, init_width), ...] big brushes first.

    clamp: optional dict with keys pos/width/alpha (see DEFAULT_CLAMP). `steps` may be an int or one int per stage.
    Returns strokes (B, sum(n), 11), final canvas (B, 3, HW), and psnr_by_stage (B, len(stages)).
    """
    clamp = {**DEFAULT_CLAMP, **(clamp or {})}
    steps = [steps] * len(stages) if isinstance(steps, int) else list(steps)
    B = target.shape[0]
    dev = target.device
    frozen = torch.empty(B, 0, N_PARAMS, device=dev)
    canvas = torch.ones(B, 3, H * W, device=dev)
    stage_psnr = []
    for (n, width), n_steps in zip(stages, steps):
        s = init_strokes(n, target, canvas, H, W, width).requires_grad_(True)
        clamp_(s, **clamp)  # start inside the allowed ranges
        opt = torch.optim.Adam([s], lr=lr)
        for _ in range(n_steps):
            out = render(s, H, W, base=canvas, mask_fn=mask_fn, use_checkpoint=use_checkpoint)
            diff = out - target
            loss = (diff.pow(2).mean(dim=(1, 2)) + 0.5 * diff.abs().mean(dim=(1, 2))).sum()  # per-image, summed
            opt.zero_grad()
            loss.backward()
            opt.step()
            clamp_(s, **clamp)
        frozen = torch.cat([frozen, s.detach()], 1)
        with torch.no_grad():
            canvas = render(s.detach(), H, W, base=canvas, mask_fn=mask_fn)
        stage_psnr.append(psnr(canvas, target))
    return frozen, canvas, torch.stack(stage_psnr, 1)


def quantize(s, bins_pos=64, bins_w=16, bins_c=16, bins_a=4):
    """Snap every field to a bin center, to simulate the discrete tokenizer. Works on any leading shape."""
    q = s.clone()

    def snap(x, lo, hi, n):
        x = ((x - lo) / (hi - lo)).clamp(0, 1)
        return lo + (torch.round(x * (n - 1)) / (n - 1)) * (hi - lo)

    lo_w, hi_w = torch.log(torch.tensor(0.004)), torch.log(torch.tensor(0.6))
    q[..., 0:6] = snap(s[..., 0:6], -0.1, 1.1, bins_pos)
    q[..., 6] = torch.exp(snap(torch.log(s[..., 6]), lo_w.to(s.device), hi_w.to(s.device), bins_w))
    q[..., 7:10] = snap(s[..., 7:10], 0, 1, bins_c)
    q[..., 10] = snap(s[..., 10], 0.2, 1.0, bins_a)
    return q
