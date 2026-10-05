"""Feasibility spike: analytic stroke renderer + coarse-to-fine image->strokes decomposer (CPU).

Stroke = 11 numbers, all in [0,1] canvas units:
  p0(x,y) p1(x,y) p2(x,y)  quadratic Bezier control points
  w                        brush width
  r g b                    color
  a                        opacity
The renderer is fixed math (no learned weights). It is differentiable so we can fit strokes to an image.
"""
import torch

N_PARAMS = 11


def _grid(H, W):
    ys, xs = torch.meshgrid(
        (torch.arange(H) + 0.5) / H, (torch.arange(W) + 0.5) / W, indexing="ij"
    )
    return torch.stack([xs, ys], -1).reshape(-1, 2)  # (HW, 2)


def stroke_masks(s, H, W, K=8, soft=0.75):
    """Soft coverage of each stroke at each pixel. s: (N, 11) -> (N, HW)."""
    t = torch.linspace(0, 1, K + 1)[None, :, None]
    p0, p1, p2 = s[:, None, 0:2], s[:, None, 2:4], s[:, None, 4:6]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2  # (N, K+1, 2)
    a, b = pts[:, :-1], pts[:, 1:]  # segments (N, K, 2)
    g = _grid(H, W)[None, None]  # (1, 1, HW, 2)
    ab = (b - a)[:, :, None]  # (N, K, 1, 2)
    ag = g - a[:, :, None]
    u = ((ag * ab).sum(-1) / (ab.pow(2).sum(-1) + 1e-8)).clamp(0, 1)
    d = (ag - u[..., None] * ab).norm(dim=-1).amin(1)  # (N, HW) distance to curve
    w = s[:, 6:7]
    return torch.sigmoid((w / 2 - d) * H / soft)


def composite(s, masks, H, W, base):
    """Paint strokes in order over `base` (3, HW). Closed form 'over' compositing."""
    am = s[:, 10:11] * masks  # (N, HW) effective alpha
    keep = 1 - am
    # T_i = prod_{j>i} keep_j  (how much of stroke i survives later strokes)
    rev = torch.cumprod(keep.flip(0), 0).flip(0)
    after = torch.cat([rev[1:], torch.ones_like(rev[:1])], 0)
    color = s[:, 7:10]  # (N, 3)
    out = torch.einsum("nc,np->cp", color, am * after) + base * rev[0:1]
    return out


def render(s, H, W, base=None):
    if base is None:
        base = torch.ones(3, H * W)
    if len(s) == 0:
        return base
    return composite(s, stroke_masks(s, H, W), H, W, base)


def clamp_(s):
    with torch.no_grad():
        s[:, 0:6].clamp_(-0.1, 1.1)
        s[:, 6].clamp_(0.004, 0.6)
        s[:, 7:10].clamp_(0, 1)
        s[:, 10].clamp_(0.2, 1.0)


def init_strokes(n, target, canvas, H, W, width):
    """Place new strokes where the error is largest, colored from the target."""
    err = (target - canvas).abs().sum(0) + 1e-6  # (HW,)
    idx = torch.multinomial(err, n, replacement=len(err) < n)
    g = _grid(H, W)
    c = g[idx]
    ang = torch.rand(n) * 3.1416
    dirv = torch.stack([ang.cos(), ang.sin()], -1) * width * (0.5 + torch.rand(n, 1))
    s = torch.empty(n, N_PARAMS)
    s[:, 0:2] = c - dirv
    s[:, 2:4] = c + 0.1 * width * torch.randn(n, 2)
    s[:, 4:6] = c + dirv
    s[:, 6] = width
    s[:, 7:10] = target[:, idx].T
    s[:, 10] = 0.9
    return s


def decompose(target, H, W, stages, steps=150, lr=0.01, log=None):
    """target: (3, HW) in [0,1]. stages: list of (n_strokes, init_width).
    Coarse-to-fine: each stage adds strokes on top of the frozen previous ones."""
    frozen = torch.empty(0, N_PARAMS)
    canvas = torch.ones(3, H * W)
    for n, width in stages:
        s = init_strokes(n, target, canvas, H, W, width).requires_grad_(True)
        opt = torch.optim.Adam([s], lr=lr)
        for _ in range(steps):
            out = render(s, H, W, base=canvas)
            loss = (out - target).pow(2).mean() + 0.5 * (out - target).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            clamp_(s)
        frozen = torch.cat([frozen, s.detach()])
        with torch.no_grad():
            canvas = render(s.detach(), H, W, base=canvas)
        if log is not None:
            log.append((len(frozen), canvas.clone()))
    return frozen, canvas


def psnr(a, b):
    return (10 * torch.log10(1 / (a - b).pow(2).mean())).item()


def quantize(s, bins_pos=64, bins_w=16, bins_c=16, bins_a=4):
    """Simulate a discrete tokenizer: snap each parameter to a bin center."""
    q = s.clone()

    def snap(x, lo, hi, n):
        x = ((x - lo) / (hi - lo)).clamp(0, 1)
        return lo + (torch.round(x * (n - 1)) / (n - 1)) * (hi - lo)

    q[:, 0:6] = snap(s[:, 0:6], -0.1, 1.1, bins_pos)
    # width on a log scale: small widths need finer resolution
    lw = snap(torch.log(s[:, 6]), torch.log(torch.tensor(0.004)), torch.log(torch.tensor(0.6)), bins_w)
    q[:, 6] = torch.exp(lw)
    q[:, 7:10] = snap(s[:, 7:10], 0, 1, bins_c)
    q[:, 10] = snap(s[:, 10], 0.2, 1.0, bins_a)
    return q
