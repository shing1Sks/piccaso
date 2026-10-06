"""Stroke generators on extractor-v2 data (research/08): two arms, same data, same renderer, same evaluation.

  Arm B ("set"): DiT-style diffusion over all stroke slots at once (CNP-style). Bidirectional, adaLN class conditioning,
                 v-prediction + optional render loss, DDIM + classifier-free guidance.
  Arm A ("ar"):  stroke-by-stroke. A causal transformer takes the previous stroke, the slot, the class and a rendered
                 picture of the canvas so far; a small flow-matching MLP head outputs the whole next stroke (11 continuous
                 numbers at once). After every stroke the canvas is re-rendered and fed back.
Data kinds: v2 (anchored slots from extract_v2.py) or native (QuickDraw pen strokes -> Beziers, native_qd.py).

  python strokegen.py --arm B --data v2 --path out/v2 --out out/runB --minutes 25
  python strokegen.py --arm A --data native --path out/native/native.pt --items out/v2/items.json --out out/runAn --minutes 25
Writes <out>/metrics.json, samples.png, ckpt.pt.
"""
import argparse
import contextlib
import copy
import glob
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import batched
from train_gate import Block

D = 11  # keep(+-1), start-rel-anchor(2), p1-rel-start(2), p2-rel-start(2), log width, rgb(3)
CSIDE = 32  # side of the canvas pictures arm A sees
WMIN = 0.008  # smallest stroke width; 0.004 when the 196 detail slots are used (main() sets it)
N_BASE = 165


# ------------------------------------------------------------------ data
def anchors_for(levels):
    return torch.tensor([((c + .5) / g, (r + .5) / g) for g, *_ in levels for r in range(g) for c in range(g)])


def load_data(path, kind, slots=None):
    if kind == "native":
        d = torch.load(path)
        S, keep, labels, ids = d["strokes"].float(), d["keep"], d["labels"], d["ids"]
        anchors = torch.full((S.shape[1], 2), 0.5)
    else:
        Ss, Ks, labels, ids, lev, anc = [], [], [], [], None, None
        for f in sorted(glob.glob(f"{path}/shard_*.pt")):
            sh = torch.load(f)
            ok = sh["ok"]
            lev, anc = sh["levels"], sh.get("anchors")  # v3 shards carry their own anchors (base + detail slots)
            Ss.append(sh["strokes"][ok].float())
            Ks.append(sh["keep"][ok])
            labels += [l for l, o in zip(sh["labels"], ok) if o]
            ids += [i for i, o in zip(sh["ids"], ok) if o]
        S, keep = torch.cat(Ss), torch.cat(Ks)
        anchors = anc if anc is not None else anchors_for(lev)
        n = slots or (sh.get("n_base") or S.shape[1])
        S, keep, anchors = S[:, :n], keep[:, :n], anchors[:n]
    labels = ["emoji" if l.startswith("emoji") else l for l in labels]
    return S, keep, labels, ids, anchors


def to_vec(S, keep, anchors):
    v = torch.zeros(*S.shape[:2], D)
    v[..., 0] = keep.float() * 2 - 1
    v[..., 1:3] = S[..., 0:2] - anchors
    v[..., 3:5] = S[..., 2:4] - S[..., 0:2]
    v[..., 5:7] = S[..., 4:6] - S[..., 0:2]
    v[..., 7] = S[..., 6].clamp(WMIN, 0.6).log()
    v[..., 8:11] = S[..., 7:10]
    return v


class Norm:
    def __init__(s, v, keep):
        m = keep[..., None].float()
        n = m.sum()
        s.mean = (v[..., 1:] * m).sum((0, 1)) / n
        s.std = (((v[..., 1:] - s.mean) ** 2 * m).sum((0, 1)) / n).sqrt().clamp_min(1e-3)

    def enc(s, v, keep):  # geometry of switched-off slots is set to the channel mean (0 after normalising)
        o = v.clone()
        o[..., 1:] = (v[..., 1:] - s.mean) / s.std * keep[..., None].float()
        return o

    def dec(s, z):
        o = z.clone()
        o[..., 1:] = z[..., 1:] * s.std.to(z.device) + s.mean.to(z.device)
        return o


def vec_to_strokes(v, anchors, soft=False):
    p0 = anchors + v[..., 1:3]
    a = ((v[..., 0:1] + 1) / 2).clamp(0, 1) if soft else (v[..., 0:1] > 0).float()
    return torch.cat([p0, p0 + v[..., 3:5], p0 + v[..., 5:7], v[..., 7:8].exp().clamp(WMIN, 0.6),
                      v[..., 8:11].clamp(0, 1), a], -1)


def stroke_update(canvas, S1, side):
    """canvas (B,3,HW), S1 (B,11) -> canvas after painting that stroke."""
    g = batched.grid(side, side, canvas.device)
    a = S1[:, 10:11] * batched.stroke_masks(S1, g, side)
    return canvas * (1 - a[:, None]) + S1[:, 7:10, None] * a[:, None]


def prefix_canvases(S, side):
    """S (B,N,11) -> the canvas BEFORE each stroke, (B,N,3,side,side)."""
    B, N, _ = S.shape
    g = batched.grid(side, side, S.device)
    a = S[..., 10:11] * batched.stroke_masks(S.reshape(-1, D), g, side).view(B, N, -1)
    canvas, outs = torch.ones(B, 3, side * side, device=S.device), []
    for i in range(N):
        outs.append(canvas)
        canvas = canvas * (1 - a[:, i][:, None]) + S[:, i, 7:10, None] * a[:, i][:, None]
    return torch.stack(outs, 1).view(B, N, 3, side, side)


def temb(t, dim=256):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[..., None].float() * 1000 * f
    return torch.cat([a.sin(), a.cos()], -1)


# ------------------------------------------------------------------ conditioning: class id or text embedding
class Cond(nn.Module):
    """Class ids -> embedding table (id n_cls = null).  Text mode: y is a float (B, text_dim) CLIP embedding; zeros = null."""

    def __init__(s, n_cls, d, text_dim=0):
        super().__init__()
        s.text_dim = text_dim
        if text_dim:
            s.proj = nn.Sequential(nn.Linear(text_dim, d), nn.SiLU(), nn.Linear(d, d))
            s.null = nn.Parameter(torch.zeros(d))
        else:
            s.emb = nn.Embedding(n_cls + 1, d)

    def forward(s, y):
        if s.text_dim:
            return torch.where((y.abs().sum(-1) == 0)[:, None], s.null.expand(len(y), -1), s.proj(y))
        return s.emb(y)

    def null_like(s, y):
        return torch.zeros_like(y) if s.text_dim else torch.full_like(y, s.emb.num_embeddings - 1)


def drop_cond(y, p, n_cls):
    m = torch.rand(len(y), device=y.device) < p
    return y * (~m)[:, None].to(y.dtype) if y.dtype.is_floating_point else torch.where(m, torch.full_like(y, n_cls), y)


TEXT_ENC = {"kind": "clip", "ckpt": "out/models/longclip-B.pt"}  # conditioning text encoder; main()/diag set it from args
_TE = {}


def _lc_tokenize(strings):  # module level so a process pool can run it
    from longclip import longclip

    return longclip.tokenize(strings, truncate=True).int()


class TextEnc:
    """Frozen text tower: per-token features (B, L, 512) projected like the pooled vector, + mask; the pooled (EOT) vector.
    kind 'clip' = OpenAI CLIP ViT-B/32 (77 tokens), 'longclip' = Long-CLIP-B (248 tokens, same image-text space)."""

    def __init__(s, dev, kind="clip", ckpt=None):
        s.kind, s.dev = kind, dev
        if kind == "clip":
            import open_clip

            s.m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
            s._tok = open_clip.get_tokenizer("ViT-B-32")
        else:
            from longclip import longclip

            s.m, _ = longclip.load(ckpt, device="cpu")
        s.m = s.m.float().to(dev).eval().requires_grad_(False)

    def tokenize(s, strings, chunk=2000):
        if s.kind == "clip":
            return torch.cat([s._tok(strings[i:i + chunk]) for i in range(0, len(strings), chunk)]).int()
        parts = [strings[i:i + chunk] for i in range(0, len(strings), chunk)]
        if len(parts) > 4:  # the Long-CLIP BPE tokenizer is pure python: spread big lists over processes
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # spawn, not fork: forking a process that already holds CUDA / thread pools deadlocks
            with ProcessPoolExecutor(min(16, os.cpu_count() or 4), mp_context=mp.get_context("spawn")) as ex:
                return torch.cat(list(ex.map(_lc_tokenize, parts)))
        return torch.cat([_lc_tokenize(p_) for p_ in parts])

    @torch.no_grad()
    def __call__(s, tok):
        tok = tok.to(s.dev).long()
        L = int(tok.argmax(-1).max()) + 1  # causal text tower: positions after the last EOT never matter -> run only up to it
        tok = tok[:, :L]
        m = s.m
        x = m.token_embedding(tok)
        if s.kind == "clip":
            x = x + m.positional_embedding[:L]
        else:
            x = x + (m.positional_embedding * m.mask1.to(x.device) + m.positional_embedding_res * m.mask2.to(x.device))[:L]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=s.dev == "cuda"):
            if s.kind == "clip":
                x = m.transformer(x, attn_mask=m.attn_mask[:L, :L])
            else:
                x = x.permute(1, 0, 2)
                am = m.transformer.resblocks[0].attn_mask[:L, :L].to(device=x.device, dtype=x.dtype)
                for r in m.transformer.resblocks:
                    h = r.ln_1(x)
                    x = x + r.attn(h, h, h, need_weights=False, attn_mask=am)[0]
                    x = x + r.mlp(r.ln_2(x))
                x = x.permute(1, 0, 2)
        x = m.ln_final(x.float()) @ m.text_projection
        mask = torch.arange(tok.shape[1], device=tok.device)[None] <= tok.argmax(-1)[:, None]
        return x, mask

    @torch.no_grad()
    def pooled(s, tok, bs=256):
        out = []
        for i in range(0, len(tok), bs):
            x, _ = s(tok[i:i + bs])
            t_ = tok[i:i + bs].to(s.dev).long()
            out.append(F.normalize(x[torch.arange(len(t_), device=s.dev), t_.argmax(-1)], dim=-1).cpu())
        return torch.cat(out)


def get_text_enc(dev, kind=None):
    kind = kind or TEXT_ENC["kind"]
    if kind not in _TE:
        _TE[kind] = TextEnc(dev, kind, TEXT_ENC["ckpt"])
    return _TE[kind]


def embed_text(strings, dev, bs=256):
    """Pooled, normalised caption vectors of the CONDITIONING text encoder."""
    te = get_text_enc(dev)
    return te.pooled(te.tokenize(list(strings)), bs)


def score_text(strings, dev, bs=256):
    """Pooled, normalised OpenAI CLIP ViT-B/32 vectors: the fixed scorer for every evaluation (independent of the conditioning)."""
    te = get_text_enc(dev, "clip")
    return te.pooled(te.tokenize(list(strings)), bs)


ClipTokens = lambda dev: get_text_enc(dev)  # backwards compatible name


def render_fb_fn(norm, anchors, chunk=32):
    """Clean estimate in normalised space -> (B,3,64,64) render with soft keep bits (what the feedback encoder sees)."""
    def fb(z0):
        st = vec_to_strokes(norm.dec(z0.float()), anchors, soft=True)[:, :N_BASE]  # detail strokes are sub-pixel at 64 px
        return torch.cat([batched.render(st[i:i + chunk], FB_SIDE, FB_SIDE) for i in range(0, len(st), chunk)]).view(-1, 3, FB_SIDE, FB_SIDE)
    return fb


# ------------------------------------------------------------------ arm B: set diffusion
class AdaBlock(nn.Module):
    def __init__(s, d, H, xattn=False):
        super().__init__()
        s.H = H
        s.n1, s.n2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(s.ada.weight), nn.init.zeros_(s.ada.bias)
        if xattn:  # cross-attention to the caption's token features (PixArt order: self-attn, cross-attn, MLP)
            s.n3, s.q2, s.kv2, s.o2 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
            nn.init.zeros_(s.o2.weight), nn.init.zeros_(s.o2.bias)

    def forward(s, x, c, kv=None, kv_mask=None):
        sh1, sc1, g1, sh2, sc2, g2 = s.ada(F.silu(c))[:, None].chunk(6, -1)
        B, N, d = x.shape
        q, k, v = s.qkv(s.n1(x) * (1 + sc1) + sh1).view(B, N, 3, s.H, d // s.H).permute(2, 0, 3, 1, 4)
        x = x + g1 * s.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, d))
        if kv is not None:
            M = kv.shape[1]
            q = s.q2(s.n3(x)).view(B, N, s.H, d // s.H).transpose(1, 2)
            k, v = s.kv2(kv).view(B, M, 2, s.H, d // s.H).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=kv_mask[:, None, None, :])
            x = x + s.o2(o.transpose(1, 2).reshape(B, N, d))
        return x + g2 * s.mlp(s.n2(x) * (1 + sc2) + sh2)


FB_SIDE = 64  # side of the rendered canvas the feedback encoder sees


class CanvasTok(nn.Module):
    """64 px render of the current clean-picture estimate -> 8x8 = 64 tokens."""

    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, d, 3, 2, 1))
        s.pos = nn.Parameter(torch.randn((FB_SIDE // 8) ** 2, d) * .02)

    def forward(s, img):
        return s.net(img * 2 - 1).flatten(2).transpose(1, 2) + s.pos


class SetDiT(nn.Module):
    def __init__(s, N, n_cls, d=384, L=8, H=6, text_dim=0, selfcond=False, canvas=False, xattn=False, ctx_dim=512):
        super().__init__()
        s.selfcond, s.use_canvas, s.xattn = selfcond or canvas, canvas, xattn
        s.inp, s.slot = nn.Linear(D * (2 if s.selfcond else 1), d), nn.Parameter(torch.randn(N, d) * .02)
        s.cls, s.tm = Cond(n_cls, d, text_dim), nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList(AdaBlock(d, H, xattn) for _ in range(L))
        s.fn, s.fada, s.out = nn.LayerNorm(d, elementwise_affine=False), nn.Linear(d, 2 * d), nn.Linear(d, D)
        nn.init.zeros_(s.out.weight), nn.init.zeros_(s.out.bias)
        if canvas:
            s.cnn = CanvasTok(d)
        if xattn:
            s.ctx_in, s.ctx_null = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, d)), nn.Parameter(torch.zeros(1, 1, d))

    def forward(s, x, t, y, sc=None, canvas=None, ctx=None, ctx_mask=None):
        """sc: previous clean estimate (self-conditioning); canvas: (B,3,64,64) render of it; ctx/ctx_mask: caption token features."""
        c = s.tm(temb(t)) + s.cls(y)
        if s.selfcond:
            x = torch.cat([x, sc if sc is not None else torch.zeros_like(x)], -1)
        h = s.inp(x) + s.slot
        n = h.shape[1]
        if s.use_canvas:
            h = torch.cat([h, s.cnn(canvas if canvas is not None else torch.ones(len(x), 3, FB_SIDE, FB_SIDE, device=x.device))], 1)
        kv = kv_mask = None
        if s.xattn:
            kv = torch.cat([s.ctx_null.expand(len(x), -1, -1), s.ctx_in(ctx)], 1)
            kv_mask = torch.cat([torch.ones(len(x), 1, dtype=torch.bool, device=x.device), ctx_mask.bool()], 1)
        for b in s.blocks:
            h = b(h, c, kv, kv_mask)
        h = h[:, :n]
        sh, sc_ = s.fada(F.silu(c))[:, None].chunk(2, -1)
        return s.out(s.fn(h) * (1 + sc_) + sh)


def ab_fn(t):  # cosine schedule, t=0 clean .. t=1 pure noise
    return torch.cos((t + .008) / 1.008 * math.pi / 2) ** 2


@torch.no_grad()
def sample_B(model, y, N, steps=50, w=2.0, eta=0.0, ctx=None, ctx_mask=None, fb=None):
    """ctx/ctx_mask: caption token features (cross-attention models); fb: clean estimate (B,N,D) -> (B,3,64,64) render."""
    B, dev = len(y), y.device
    null = model.cls.null_like(y)
    x = torch.randn(B, N, D, device=dev)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    sc = torch.zeros_like(x) if getattr(model, "selfcond", False) else None
    canv = None
    null_mask = torch.zeros_like(ctx_mask) if ctx_mask is not None else None
    for i in range(steps):
        t = ts[i].expand(B)
        ab, abn = ab_fn(ts[i]), ab_fn(ts[i + 1])
        if getattr(model, "use_canvas", False) and i > 0:
            canv = fb(sc)
        kc = dict(sc=sc, canvas=canv, ctx=ctx, ctx_mask=ctx_mask)
        ku = dict(sc=sc, canvas=canv, ctx=ctx, ctx_mask=null_mask)
        v = model(x, t, null, **ku) + w * (model(x, t, y, **kc) - model(x, t, null, **ku)) if w != 1 else model(x, t, y, **kc)
        a, s = ab.sqrt(), (1 - ab).sqrt()
        x0 = (a * x - s * v).clamp(-6, 6)
        if sc is not None:
            sc = x0
        eps = s * x + a * v
        sig = eta * ((1 - abn) / (1 - ab).clamp_min(1e-8)).sqrt() * (1 - ab / abn.clamp_min(1e-8)).clamp_min(0).sqrt() if i < steps - 1 else 0 * ab
        x = abn.sqrt() * x0 + (1 - abn - sig ** 2).clamp_min(0).sqrt() * eps + sig * torch.randn_like(x)
    return x0


# ------------------------------------------------------------------ arm A: stroke-by-stroke with canvas feedback
class CanvasEnc(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                              nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(128 * (CSIDE // 8) ** 2, d))

    def forward(s, x):
        return s.net(x)


class ARFlow(nn.Module):
    def __init__(s, N, n_cls, d=384, L=8, H=6, canvas=True, dropout=0.0, text_dim=0):
        super().__init__()
        s.use_canvas = canvas
        s.inp, s.slot, s.cls = nn.Linear(D, d), nn.Embedding(N, d), Cond(n_cls, d, text_dim)
        s.start = nn.Parameter(torch.zeros(d))
        s.canvas = CanvasEnc(d)
        s.blocks, s.ln = nn.ModuleList(Block(d, H, dropout) for _ in range(L)), nn.LayerNorm(d)
        s.hin, s.hc, s.ht = nn.Linear(D, 512), nn.Linear(d, 512), nn.Linear(256, 512)
        s.hnet = nn.Sequential(nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, D))
        nn.init.zeros_(s.hnet[-1].weight), nn.init.zeros_(s.hnet[-1].bias)

    def trunk(s, prev, cemb, y):  # prev (B,n,D) previous strokes (zeros first), cemb (B,n,d) canvas embeddings
        B, n, _ = prev.shape
        h = s.inp(prev) + s.slot(torch.arange(n, device=prev.device))[None] + s.cls(y)[:, None]
        h = h + torch.cat([s.start[None, None], torch.zeros(1, n - 1, h.shape[-1], device=h.device)], 1)
        if s.use_canvas:
            h = h + cemb
        for b in s.blocks:
            h = b(h)
        return s.ln(h)

    def head(s, h, xt, t):
        return s.hnet(s.hin(xt) + s.hc(h) + s.ht(temb(t)))


@torch.no_grad()
def sample_A(model, y, N, anchors, norm, steps=12, cfg=1.0):
    B, dev, d = len(y), y.device, model.start.shape[0]
    null = model.cls.null_like(y)
    z, cemb = torch.zeros(B, N, D, device=dev), torch.zeros(B, N, d, device=dev)
    cur = torch.ones(B, 3, CSIDE * CSIDE, device=dev)
    for i in range(N):
        if model.use_canvas:
            cemb[:, i] = model.canvas(cur.view(B, 3, CSIDE, CSIDE))
        prev = torch.cat([torch.zeros(B, 1, D, device=dev), z[:, :i]], 1)
        h = model.trunk(prev, cemb[:, :i + 1], y)[:, i]
        hu = model.trunk(prev, cemb[:, :i + 1], null)[:, i] if cfg != 1 else None
        x = torch.randn(B, D, device=dev)
        for k in range(steps):
            tt = torch.full((B,), k / steps, device=dev)
            v = model.head(h, x, tt)
            if hu is not None:
                vu = model.head(hu, x, tt)
                v = vu + cfg * (v - vu)
            x = x + v / steps
        z[:, i] = x.clamp(-6, 6)
        S1 = vec_to_strokes(norm.dec(z[:, i:i + 1]), anchors[i].to(dev))[:, 0]
        cur = stroke_update(cur, S1, CSIDE)
    return z


# ------------------------------------------------------------------ evaluation: a small classifier on real images
class Cls(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1),
                              nn.GELU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
                              nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 16, n))

    def forward(s, x):
        return s.net(x)


def train_classifier(items_file, classes, dev, epochs=6):
    from concurrent.futures import ThreadPoolExecutor

    from extract_prod import load_item, to_tensor

    items = [it for it in json.load(open(items_file))]
    for it in items:
        it["cls"] = "emoji" if it["source"] == "emoji" else it["label"]
    items = [it for it in items if it["cls"] in classes]
    with ThreadPoolExecutor(16) as ex:
        X = torch.stack([F.avg_pool2d(to_tensor(im).view(3, 128, 128), 4) for im in ex.map(load_item, items)])
    Y = torch.tensor([classes.index(it["cls"]) for it in items])
    perm = torch.randperm(len(X))
    nv = max(1, len(X) // 10)
    vi, ti = perm[:nv], perm[nv:]
    net = Cls(len(classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    for ep in range(epochs):
        net.train()
        for j in range(0, len(ti), 128):
            b = ti[torch.randperm(len(ti))[:128]]
            loss = F.cross_entropy(net(X[b].to(dev)), Y[b].to(dev))
            opt.zero_grad(), loss.backward(), opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(X[vi].to(dev)).argmax(1).cpu() == Y[vi]).float().mean().item()
    return net, acc


def calibrate(st, kv, keep_pool):
    """Keep-bit calibration: copy the stroke count of a random real picture (per segment: base, detail), most confident slots win."""
    N = st.shape[1]
    segs = [(0, N)] if N <= N_BASE else [(0, N_BASE), (N_BASE, N)]
    for r in range(len(st)):
        real = keep_pool[torch.randint(len(keep_pool), (1,))][0]
        on = torch.zeros(N, device=st.device)
        for s0, s1 in segs:
            k = int(real[s0:s1].sum())
            if k:
                on[s0 + kv[r, s0:s1].topk(k).indices] = 1
        st[r, :, 10] = on
    return st


@torch.no_grad()
def render_samples(S, side=128, chunk=4):
    return torch.cat([batched.render(S[i:i + chunk], side, side) for i in range(0, len(S), chunk)])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--data", choices=["v2", "native"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--items", default=None, help="items.json for the classifier (default <path>/items.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--render-loss", type=float, default=0.0)
    ap.add_argument("--no-canvas", action="store_true")
    ap.add_argument("--n-sample", type=int, default=24, help="samples per class")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--prev-noise", type=float, default=0.0, help="arm A: gaussian noise on the previous-stroke input (exposure bias)")
    ap.add_argument("--cdrop", type=float, default=0.1, help="class dropout for guidance")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many evals (200 steps each) without a new best val")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--cfg-A", type=float, default=1.0)
    ap.add_argument("--text-json", default=None, help="json {item id: [captions]} -> text-conditioned mode (CLIP ViT-B/32 embeddings)")
    ap.add_argument("--text-cache", default=None, help="cache file for caption embeddings")
    ap.add_argument("--group-labels", action="store_true", help="classes = data sources (mixed data); skips the class classifier")
    ap.add_argument("--prompts-file", default=None, help="text mode: one free-text prompt per line -> qualitative sheet")
    ap.add_argument("--clip-eval", type=int, default=0, help="text mode: CLIP image-text retrieval on this many held-out items")
    ap.add_argument("--text-norm", action="store_true", help="standardise caption embeddings per dimension (CLIP vectors of similar captions are near-identical)")
    ap.add_argument("--slots", type=int, default=None, help="v3 data: use the first N slots (default: the 165 base slots)")
    ap.add_argument("--slot-weight", default=None, help="loss weight per base level, e.g. 3,1.5,1 (coarse,mid,fine; normalised to mean 1)")
    ap.add_argument("--img-cond", type=float, default=0.0, help="text mode: probability of conditioning on the item's CLIP image vector instead of a caption")
    ap.add_argument("--clipvec", default=None, help="clipvec.pt from clip_pass.py (needed for --img-cond)")
    ap.add_argument("--selfcond", action="store_true", help="arm B: self-conditioning on the previous clean estimate")
    ap.add_argument("--canvas-fb", action="store_true", help="arm B: also see a 64 px render of the current estimate (implies --selfcond)")
    ap.add_argument("--xattn", action="store_true", help="arm B, text mode: cross-attention to CLIP token features of the caption")
    ap.add_argument("--text-encoder", choices=["clip", "longclip"], default="clip", help="conditioning text encoder")
    ap.add_argument("--longclip-ckpt", default="out/models/longclip-B.pt")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=1024, help="held-out items used for the periodic validation loss")
    ap.add_argument("--init", default=None, help="start from this checkpoint's (EMA) weights; optimizer + LR schedule start fresh")
    ap.add_argument("--accum", type=int, default=1, help="split each GPU's batch into this many micro-batches (same update, less memory)")
    ap.add_argument("--cache-only", action="store_true", help="build the caption caches (pooled vectors + token ids) and exit")
    ap.add_argument("--save-every", type=int, default=0, help="also save an EMA checkpoint (+ a peek sheet) every N steps")
    ap.add_argument("--peek-prompts", type=int, default=12, help="prompts (from --prompts-file) on each peek sheet")
    ap.add_argument("--train-frac", type=float, default=1.0, help="train on this fraction of the training split (held-out split unchanged)")
    ap.add_argument("--no-calibrate", action="store_true", help="keep every slot with keep>0 instead of the top-k with k from the real per-class distribution")
    a = ap.parse_args()
    TEXT_ENC.update(kind=a.text_encoder, ckpt=a.longclip_ckpt)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp, rank = world > 1, 0
    if ddp:  # torchrun: one process per GPU, gradients averaged; rank 0 evaluates and saves
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(minutes=90))
        rank = dist.get_rank()
        assert a.max_steps and not a.patience, "multi-GPU runs need --max-steps and no early stopping"
        if rank:
            sys.stdout = open(os.devnull, "w")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    S, keep, labels, ids, anchors = load_data(a.path, a.data, a.slots)
    if a.text_json:  # items removed by the caption / CLIP filter are not trained on
        _caps = json.load(open(a.text_json))
        sel = [j for j, i in enumerate(ids) if i in _caps]
        if len(sel) < len(ids):
            print(f"text filter: {len(ids) - len(sel)} items without kept captions dropped", flush=True)
            S, keep, labels, ids = S[sel], keep[sel], [labels[j] for j in sel], [ids[j] for j in sel]
    if a.group_labels:
        src_of = {}
        for f in sorted(glob.glob(f'{a.path}/shard_*.pt')):
            sh_ = torch.load(f)
            src_of.update({i: sr for i, sr, o in zip(sh_['ids'], sh_['sources'], sh_['ok']) if o})
        labels = [src_of[i] for i in ids]
    classes = sorted(set(labels))
    y_all = torch.tensor([classes.index(l) for l in labels])
    N = S.shape[1]
    global WMIN
    if N > N_BASE:
        WMIN = 0.004
    v_ = to_vec(S, keep, anchors)
    norm = Norm(v_, keep)
    Z = norm.enc(v_, keep)
    del v_, S  # large datasets: keep host memory low (one copy per GPU process)
    perm = torch.randperm(len(Z))
    nv = max(8, len(Z) // 20)
    vi, ti = perm[:nv], perm[nv:]
    if a.train_frac < 1:
        ti = ti[:int(len(ti) * a.train_frac)]
        print(f"training on {len(ti)} items ({a.train_frac:.0%} of the training split)", flush=True)
    print(f"{len(Z)} sequences x {N} slots, classes {classes}, kept fraction {keep.float().mean():.2f}", flush=True)
    Zg, Kg, Yg, Ag = Z.to(dev), keep.to(dev), y_all.to(dev), anchors.to(dev)
    slot_w = None
    if a.slot_weight:  # level sizes 16, 49, 100 (+196 detail); weights normalised so the mean slot weight is 1
        ws = [float(x) for x in a.slot_weight.split(",")]
        sizes = [16, 49, 100, 196]
        per = torch.cat([torch.full((n_,), ws[min(k, len(ws) - 1)]) for k, n_ in enumerate(sizes)])[:N]
        slot_w = (per / per.mean()).to(dev)
        print("slot weights per level:", sorted(set(round(x, 3) for x in slot_w.tolist()), reverse=True), flush=True)
    B = (a.batch or (128 if a.arm == "B" else 48)) // world  # per-GPU batch; the global batch stays --batch
    n_cls = len(classes)
    text_mode = a.text_json is not None
    if text_mode:
        caps = json.load(open(a.text_json))
        uniq = sorted({c for i in ids for c in caps[i]})
        tpl_eval = {"in-dist": ["a drawing of a {}", "a doodle of a {}"], "held-out": ["a child's drawing of a {}", "a black and white sketch showing a {}"]}
        fmt = lambda t, c: ("an emoji icon" if t.startswith("a drawing") or t.startswith("a child") else "a colorful emoji") if c == "emoji" else t.format(c)
        eval_strings = [fmt(t, c) for ts in tpl_eval.values() for t in ts for c in classes]
        allc = uniq + eval_strings
        if ddp and rank:
            dist.barrier()
        if a.text_cache and Path(a.text_cache).exists():
            cache = torch.load(a.text_cache)
            Tall = cache["emb"] if cache["strings"] == allc and cache.get("kind", "clip") == a.text_encoder else None
        else:
            Tall = None
        tok_all = None
        if Tall is None:
            te_ = get_text_enc(dev)
            tok_all = te_.tokenize(allc)  # tokenised once: pooled vectors now, token ids for cross-attention below
            Tall = te_.pooled(tok_all)
            if a.text_cache:
                torch.save({"strings": allc, "emb": Tall, "kind": a.text_encoder}, a.text_cache)
        tn = None
        if a.text_norm:
            mu, sd = Tall[:len(uniq)].mean(0), Tall[:len(uniq)].std(0) + 1e-6
            Tall = (Tall - mu) / sd
            tn = (mu, sd)
        pos = {c: j for j, c in enumerate(allc)}
        maxk = max(len(caps[i]) for i in ids)
        cap_idx = torch.full((len(ids), maxk), -1, dtype=torch.long)
        for r, i in enumerate(ids):
            cap_idx[r, :len(caps[i])] = torch.tensor([pos[c] for c in caps[i]])
        cap_n = (cap_idx >= 0).sum(1)
        Tall, cap_idx, cap_n = Tall.to(dev), cap_idx.to(dev), cap_n.to(dev)
        print(f"text mode: {len(uniq)} unique captions, embedding dim {Tall.shape[1]}", flush=True)
        ctok, Tok = None, None
        if a.xattn:  # per-token features are computed on the fly from cached token ids (all features would not fit in memory)
            ctok = ClipTokens(dev)
            tok_cache = Path(a.text_cache).with_name(Path(a.text_cache).stem + "_tok.pt") if a.text_cache else None
            tc = torch.load(tok_cache) if tok_cache and tok_cache.exists() else None
            if tc is not None and tc["n"] == len(allc) and tc.get("kind", "clip") == a.text_encoder:
                Tok = tc["tok"]
            else:
                Tok = tok_all if tok_all is not None else ctok.tokenize(allc)
                if tok_cache:
                    torch.save({"n": len(allc), "tok": Tok, "kind": a.text_encoder}, tok_cache)
            Tok = Tok.to(dev)
        if ddp and not rank:
            dist.barrier()
        if a.cache_only:
            print("CACHE_DONE", flush=True)
            return
    td = Tall.shape[1] if text_mode else 0

    Iall, img_stats = None, None
    if text_mode and a.img_cond > 0:  # reference-picture conditioning: the item's own CLIP image vector, standardised
        cv = torch.load(a.clipvec)
        at = {i: k for k, i in enumerate(cv["ids"])}
        Iall = torch.stack([cv["img"][at[i]].float() for i in ids])
        imu, isd = Iall.mean(0), Iall.std(0) + 1e-6
        Iall = ((Iall - imu) / isd).to(dev)
        img_stats = (imu, isd)
        print(f"image conditioning on {a.img_cond:.0%} of steps", flush=True)

    def cond_for(idx, with_ctx=False):
        """Returns the pooled condition vector; with_ctx: also caption token features + mask (masked out for image conditioning)."""
        if not text_mode:
            return Yg[idx]
        j = (torch.rand(len(idx), device=dev) * cap_n[idx]).long().clamp_max(cap_idx.shape[1] - 1)
        ci = cap_idx[idx, j]
        c = Tall[ci]
        use = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        if Iall is not None:
            use = torch.rand(len(idx), device=dev) < a.img_cond
            c = torch.where(use[:, None], Iall[idx], c)
        if not with_ctx:
            return c
        ctx, m = ctok(Tok[ci])
        return c, ctx, m & ~use[:, None]

    def ctx_for_strings(strings):
        if not (text_mode and a.xattn):
            return None, None
        return ctok(ctok.tokenize(strings))

    model = (SetDiT(N, n_cls, a.d, a.layers, a.heads, td, a.selfcond, a.canvas_fb, a.xattn and text_mode) if a.arm == "B"
             else ARFlow(N, n_cls, a.d, a.layers, a.heads, not a.no_canvas, a.dropout, td)).to(dev)
    if a.init:  # continue training an earlier run: same architecture, its EMA weights as the starting point
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["ema"])
        print(f"initialised from {a.init}", flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    net = DDP(model, device_ids=[torch.cuda.current_device()]) if ddp else model
    torch.manual_seed(1000 + rank)  # every GPU draws its own batches
    n_par = sum(p.numel() for p in model.parameters())
    print(f"arm {a.arm}, {n_par / 1e6:.1f}M params, batch {B}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.99), weight_decay=a.wd)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda")

    fb = render_fb_fn(norm, Ag) if a.canvas_fb else None

    def loss_fn(idx, train=True):
        ctx = cmask = None
        if a.arm == "B" and a.xattn and text_mode:
            z, kp, (y, ctx, cmask) = Zg[idx], Kg[idx], cond_for(idx, with_ctx=True)
        else:
            z, kp, y = Zg[idx], Kg[idx], cond_for(idx)
        w = torch.cat([torch.ones_like(z[..., :1]), kp[..., None].float().expand(-1, -1, D - 1)], -1)
        if slot_w is not None:
            w = w * slot_w[None, :, None]
        if a.arm == "B":
            if train:
                keep_c = torch.rand(len(z), device=dev) >= 0.1  # classifier-free guidance dropout (pooled + tokens together)
                y = y * keep_c[:, None].to(y.dtype) if y.dtype.is_floating_point else torch.where(keep_c, y, torch.full_like(y, n_cls))
                if cmask is not None:
                    cmask = cmask & keep_c[:, None]
            t = torch.rand(len(z), device=dev)
            ab = ab_fn(t)[:, None, None]
            eps = torch.randn_like(z)
            xt = ab.sqrt() * z + (1 - ab).sqrt() * eps
            tgt = ab.sqrt() * eps - (1 - ab).sqrt() * z
            kw = dict(ctx=ctx, ctx_mask=cmask)
            if model.selfcond:  # half the batch (all of it at validation) gets the model's own first-pass estimate
                sc = torch.zeros_like(z)
                canv = torch.ones(len(z), 3, FB_SIDE, FB_SIDE, device=dev) if model.use_canvas else None
                use = torch.rand(len(z), device=dev) < 0.5 if train else torch.ones(len(z), dtype=torch.bool, device=dev)
                if use.any():
                    u = use.nonzero()[:, 0]
                    with torch.no_grad(), amp:
                        v0 = model(xt[u], t[u], y[u], ctx=None if ctx is None else ctx[u], ctx_mask=None if cmask is None else cmask[u]).float()
                    x0e = (ab[u].sqrt() * xt[u] - (1 - ab[u]).sqrt() * v0).clamp(-6, 6)
                    sc[u] = x0e
                    if canv is not None:
                        with torch.no_grad():
                            canv[u] = fb(x0e)
                kw.update(sc=sc, canvas=canv)
            with amp:
                pred = (net if train else model)(xt, t, y, **kw).float()
            loss = ((pred - tgt) ** 2 * w).mean()
            if a.render_loss > 0 and train:
                x0 = ab.sqrt() * xt - (1 - ab).sqrt() * pred
                k = min(16, len(z))
                p_s = vec_to_strokes(norm.dec(x0[:k]), Ag, soft=True)
                g_s = vec_to_strokes(norm.dec(z[:k]), Ag)
                with torch.no_grad():
                    tgt_img = batched.render(g_s, 64, 64)
                rl = (batched.render(p_s, 64, 64) - tgt_img).abs().mean((1, 2))
                loss = loss + a.render_loss * (rl * ab[:k, 0, 0]).mean()
            return loss
        with torch.no_grad():
            gs = vec_to_strokes(norm.dec(z), Ag)
            canv = prefix_canvases(gs, CSIDE) if model.use_canvas else None
        prev = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], 1)
        if train:
            y = drop_cond(y, a.cdrop, n_cls)
            if a.prev_noise > 0:
                prev = prev + a.prev_noise * torch.randn_like(prev)
        with amp:
            cemb = model.canvas(canv.flatten(0, 1)).view(len(z), N, -1) if canv is not None else None
            h = model.trunk(prev, cemb, y)
            R = 2
            t = torch.rand(len(z), N, R, device=dev)
            eps = torch.randn(len(z), N, R, D, device=dev)
            xt = (1 - t[..., None]) * eps + t[..., None] * z[:, :, None]
            pred = model.head(h[:, :, None].expand(-1, -1, R, -1), xt, t).float()
        return (((pred - (z[:, :, None] - eps)) ** 2) * w[:, :, None]).mean()

    def save_ckpt(path, state=None):
        torch.save({"ema": state if state is not None else ema.state_dict(), "classes": classes, "arm": a.arm, "N": N,
                    "norm": (norm.mean, norm.std), "text_dim": td, "text_norm": tn if text_mode else None, "img_norm": img_stats,
                    "anchors": anchors, "wmin": WMIN, "args": vars(a)}, path)

    def peek(step_):  # small prompt sheet from the current EMA weights (25 steps, cfg 3)
        if not (text_mode and a.prompts_file):
            return
        prompts = [l.strip() for l in open(a.prompts_file) if l.strip()][:a.peek_prompts]
        per_, T_ = 4, (256 if N > N_BASE else 128)
        pe_ = embed_text(prompts, dev).to(dev)
        if tn is not None:
            pe_ = (pe_ - tn[0].to(dev)) / tn[1].to(dev)
        pc, pm = ctx_for_strings(prompts)
        rep_ = lambda x: None if x is None else x.repeat_interleave(per_, dim=0)
        torch.manual_seed(7)
        zs = sample_B(ema, pe_.repeat_interleave(per_, dim=0), N, 25, 3.0, ctx=rep_(pc), ctx_mask=rep_(pm), fb=fb)
        st = calibrate(vec_to_strokes(norm.dec(zs), Ag), norm.dec(zs)[..., 0], Kg)
        ims = render_samples(st, T_)
        sh = Image.new("RGB", (T_ * per_, T_ * len(prompts)), "white")
        dr = ImageDraw.Draw(sh)
        for r_, pr_ in enumerate(prompts):
            for q in range(per_):
                arr = (ims[r_ * per_ + q].view(3, T_, T_).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                sh.paste(Image.fromarray(arr), (q * T_, r_ * T_))
            dr.text((3, r_ * T_ + 2), f"step {step_}: {pr_[:50]}", fill=(255, 0, 0))
        sh.save(out / f"peek_{step_:06d}.png")
        torch.manual_seed(step_ * 10 + rank)

    t0, step, total = time.perf_counter(), 0, a.max_steps
    hist = []
    best, best_state, bad = 1e9, None, 0
    while True:
        lr = a.lr * min(1, (step + 1) / a.warmup)
        if total:
            lr *= 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, step / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        opt.zero_grad(set_to_none=True)
        idx_all = ti[torch.randint(0, len(ti), (B,))].to(dev)
        loss_acc = 0.0
        for q, mb in enumerate(idx_all.chunk(a.accum)):
            sync_ctx = net.no_sync() if ddp and q < a.accum - 1 else contextlib.nullcontext()
            with sync_ctx:
                lq = loss_fn(mb) * len(mb) / len(idx_all)
                lq.backward()
            loss_acc += lq.detach()
        loss = loss_acc
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            dec = min(0.999, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dec)
        step += 1
        if step == 50 and total is None:
            if dev == "cuda":
                torch.cuda.synchronize()
            total = int(a.minutes * 60 / ((time.perf_counter() - t0) / 50))
            print(f"~{(time.perf_counter() - t0) / 50:.3f}s/step -> {total} steps", flush=True)
        if a.save_every and step % a.save_every == 0 and rank == 0 and not (total and step >= total):
            save_ckpt(out / f"ckpt_{step:06d}.pt")
            if a.arm == "B":
                peek(step)
            print(json.dumps({"saved": step, "min": round((time.perf_counter() - t0) / 60, 2)}), flush=True)
        if rank == 0 and (step % 200 == 0 or (total and step >= total)):
            model.eval()
            with torch.no_grad():
                torch.manual_seed(1)
                vsub, vb = vi[:a.val_n], max(B, 128)  # fixed held-out subset, big batches: rank 0 must not stall the other GPUs
                vl = np.mean([loss_fn(vsub[j:j + vb].to(dev), train=False).item() for j in range(0, len(vsub), vb)])
            torch.manual_seed(step * 10 + rank)
            hist.append({"step": step, "train": round(loss.item(), 4), "val": round(float(vl), 4),
                         "min": round((time.perf_counter() - t0) / 60, 2)})
            print(json.dumps(hist[-1]), flush=True)
            if vl < best:
                best, bad = float(vl), 0
                best_state = {k: v.detach().clone() for k, v in ema.state_dict().items()}
            else:
                bad += 1
            if a.patience and bad >= a.patience:
                print(f"early stop at step {step} (best val {best:.4f})", flush=True)
                break
        if total and step >= total:
            break
    if ddp:
        if rank:
            dist.destroy_process_group()
            return
    if best_state is not None:
        ema.load_state_dict(best_state)
    save_ckpt(out / "ckpt.pt")

    # ---- sampling + evaluation: one prompt set in class mode; in text mode an in-distribution and a held-out phrasing set
    items_file = a.items or f"{a.path}/items.json"
    if a.group_labels:
        cls_net, cls_val = None, float('nan')
    else:
        cls_net, cls_val = train_classifier(items_file, classes, dev, 1 if a.smoke else 6)
    with torch.no_grad():
      if cls_net is None:
        fit_acc = float('nan')
      else:
        gi = torch.stack([render_samples(vec_to_strokes(norm.dec(Zg[i:i + 1]), Ag))[0] for i in ti[:200].tolist()])
        fit_acc = (cls_net(F.avg_pool2d(gi.view(-1, 3, 128, 128), 4)).argmax(1) == Yg[ti[:200]]).float().mean().item()
    print(f"classifier val acc on real images {cls_val:.2f}; on fitted training strokes {fit_acc:.2f}", flush=True)
    on_counts = [keep[y_all == c].sum(1) for c in range(n_cls)]
    on_counts_all = keep.sum(1)
    ns, T, per = a.n_sample, (256 if N > N_BASE else 128), 6
    metrics = {"arm": a.arm, "data": a.data, "text_mode": text_mode, "params_M": round(n_par / 1e6, 2), "steps": step, "best_val": best,
               "classes": classes, "hist": hist, "cls_val_acc_real": cls_val, "cls_acc_fitted_train": fit_acc, "chance": 1 / n_cls,
               "kept_slots_data": keep.float().sum(1).mean().item(), "sets": {}}
    sets = [("class", None, 1)]
    if text_mode:
        sets = [(k, [fmt(t, c) for t in ts for c in classes], len(ts)) for k, ts in tpl_eval.items()]
    if a.group_labels:
        sets = []
    for set_name, strings, ntpl in sets:
        t_s = time.perf_counter()
        if strings is None:
            ys = torch.arange(n_cls, device=dev).repeat_interleave(ns)
            cond = ys
        else:  # strings are template-major: for each template, each class; every string gets ns // ntpl samples
            each = max(1, ns // ntpl)
            ys = torch.arange(n_cls, device=dev).repeat(ntpl).repeat_interleave(each)
            cond = Tall[torch.tensor([pos[x] for x in strings], device=dev)].repeat_interleave(each, dim=0)
        bs = 56 if a.arm == "A" else 112
        zs = torch.cat([sample_B(ema, cond[j:j + bs], N, 25 if a.smoke else 50, a.cfg) if a.arm == "B"
                        else sample_A(ema, cond[j:j + bs], N, Ag, norm, cfg=a.cfg_A) for j in range(0, len(cond), bs)])
        strokes = vec_to_strokes(norm.dec(zs), Ag)
        if not a.no_calibrate:  # sampler bias fix: number of strokes ON ~ real per-class distribution, most confident slots win
            kv = norm.dec(zs)[..., 0]
            for r in range(len(zs)):
                pool = on_counts[int(ys[r])]
                k = int(pool[torch.randint(len(pool), (1,))])
                on = torch.zeros(N, device=dev)
                on[kv[r].topk(k).indices] = 1
                strokes[r, :, 10] = on
        imgs = render_samples(strokes)
        with torch.no_grad():
            pr = cls_net(F.avg_pool2d(imgs.view(-1, 3, 128, 128), 4)).argmax(1)
        acc = (pr == ys).float().mean().item()
        pc = [(pr[ys == c] == c).float().mean().item() for c in range(n_cls)]
        print(f"[{set_name}] {len(ys)} drawings in {time.perf_counter() - t_s:.0f}s; slots on {strokes[..., 10].sum(1).mean():.1f}; "
              f"classifier accuracy on GENERATED {acc:.2f} (chance {1 / n_cls:.2f}); per class "
              f"{ {c: round(v, 2) for c, v in zip(classes, pc)} }", flush=True)
        metrics["sets"][set_name] = {"acc": acc, "per_class": dict(zip(classes, pc)), "slots_on": strokes[..., 10].sum(1).mean().item()}
        sheet = Image.new("RGB", (T * (per + 1), T * n_cls), "white")
        dr = ImageDraw.Draw(sheet)
        for c in range(n_cls):
            jj = next((i for i in ti.tolist() if y_all[i] == c), 0)
            tiles = [render_samples(vec_to_strokes(norm.dec(Zg[jj:jj + 1]), Ag))[0]] + [imgs[(ys == c).nonzero()[k, 0]] for k in range(min(per, int((ys == c).sum())))]
            for q, im in enumerate(tiles):
                arr = (im.view(3, T, T).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                sheet.paste(Image.fromarray(arr), (q * T, c * T))
            dr.text((3, c * T + 2), f"{classes[c]}: fitted example | samples ({set_name}) ->", fill=(255, 0, 0))
        sheet.save(out / ("samples.png" if set_name == "class" else f"samples_{set_name}.png"))
    if text_mode and a.prompts_file:  # qualitative: free-text prompts -> 6 samples each
        prompts = [l.strip() for l in open(a.prompts_file) if l.strip()]
        pe = embed_text(prompts, dev).to(dev)
        if tn is not None:
            pe = (pe - tn[0].to(dev)) / tn[1].to(dev)
        cond = pe.repeat_interleave(per, dim=0)
        bs = 56 if a.arm == "A" else 112
        pctx, pmask = ctx_for_strings(prompts)
        if pctx is not None:
            pctx, pmask = pctx.repeat_interleave(per, dim=0), pmask.repeat_interleave(per, dim=0)
        zs = torch.cat([sample_B(ema, cond[j:j + bs], N, 25 if a.smoke else 50, a.cfg, ctx=None if pctx is None else pctx[j:j + bs],
                                 ctx_mask=None if pmask is None else pmask[j:j + bs], fb=fb) if a.arm == "B"
                        else sample_A(ema, cond[j:j + bs], N, Ag, norm, cfg=a.cfg_A) for j in range(0, len(cond), bs)])
        st_ = vec_to_strokes(norm.dec(zs), Ag)
        if not a.no_calibrate:
            st_ = calibrate(st_, norm.dec(zs)[..., 0], Kg)
        ims = render_samples(st_, T)
        sh = Image.new("RGB", (T * per, T * len(prompts)), "white")
        dr2 = ImageDraw.Draw(sh)
        for r_, pr_ in enumerate(prompts):
            for q in range(per):
                arr = (ims[r_ * per + q].view(3, T, T).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                sh.paste(Image.fromarray(arr), (q * T, r_ * T))
            dr2.text((3, r_ * T + 2), pr_[:60], fill=(255, 0, 0))
        sh.save(out / "prompts.png")
    if Iall is not None:  # reference-picture sheet: condition ONLY on a held-out picture's CLIP image vector
        nref, per_r = min(10, len(vi)), 5
        ref_idx = vi[:nref].to(dev)
        cond = Iall[ref_idx].repeat_interleave(per_r, dim=0)
        bs = 56 if a.arm == "A" else 112
        rctx = rmask = None
        if text_mode and a.xattn:  # image-vector conditioning: no caption tokens (all masked)
            rctx, rmask = ctx_for_strings([""] * len(cond))
            rmask = torch.zeros_like(rmask)
        zs = torch.cat([sample_B(ema, cond[j:j + bs], N, 50, a.cfg, ctx=None if rctx is None else rctx[j:j + bs],
                                 ctx_mask=None if rmask is None else rmask[j:j + bs], fb=fb) if a.arm == "B"
                        else sample_A(ema, cond[j:j + bs], N, Ag, norm, cfg=a.cfg_A) for j in range(0, len(cond), bs)])
        st_ = vec_to_strokes(norm.dec(zs), Ag)
        if not a.no_calibrate:
            st_ = calibrate(st_, norm.dec(zs)[..., 0], Kg)
        ims = render_samples(st_, T)
        refs = render_samples(vec_to_strokes(norm.dec(Zg[ref_idx]), Ag), T)
        sh = Image.new("RGB", (T * (per_r + 1), T * nref), "white")
        dr3 = ImageDraw.Draw(sh)
        for r_ in range(nref):
            tiles = [refs[r_]] + [ims[r_ * per_r + q] for q in range(per_r)]
            for q, im in enumerate(tiles):
                arr = (im.view(3, T, T).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                sh.paste(Image.fromarray(arr), (q * T, r_ * T))
            dr3.text((3, r_ * T + 2), "reference (its strokes) | samples from its CLIP image vector", fill=(255, 0, 0))
        sh.save(out / "references.png")
    if text_mode and a.clip_eval:  # CLIP image-text retrieval: does a generated drawing match ITS caption better than the other captions?
        import open_clip

        cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        cm = cm.to(dev).eval()
        sel = [i for i in vi.tolist()][:a.clip_eval]
        sel_caps = [caps[ids[i]][0] for i in sel]
        te = score_text(sel_caps, dev).to(dev)
        cond = Tall[torch.tensor([pos[c] for c in sel_caps], device=dev)]
        bs = 56 if a.arm == "A" else 112
        ectx, emask = ctx_for_strings(sel_caps)
        zs = torch.cat([sample_B(ema, cond[j:j + bs], N, 25 if a.smoke else 50, a.cfg, ctx=None if ectx is None else ectx[j:j + bs],
                                 ctx_mask=None if emask is None else emask[j:j + bs], fb=fb) if a.arm == "B"
                        else sample_A(ema, cond[j:j + bs], N, Ag, norm, cfg=a.cfg_A) for j in range(0, len(cond), bs)])
        gen_st = vec_to_strokes(norm.dec(zs), Ag)
        fit_st = vec_to_strokes(norm.dec(Zg[torch.tensor(sel)]), Ag)
        mean = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
        std = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)

        def retr(st):
            im = render_samples(st, 224).view(-1, 3, 224, 224)
            with torch.no_grad():
                ie = F.normalize(cm.encode_image((im - mean) / std).float(), dim=-1)
            sim = ie @ te.T
            diag = sim.diag()
            return {"top1": (sim.argmax(1) == torch.arange(len(sel), device=dev)).float().mean().item(),
                    "gap": (diag.mean() - (sim.sum() - diag.sum()) / (sim.numel() - len(sel))).item()}
        metrics["clip_retrieval"] = {"n": len(sel), "chance_top1": 1 / len(sel), "generated": retr(gen_st), "fitted_upper_bound": retr(fit_st)}
        print("CLIP retrieval:", json.dumps(metrics["clip_retrieval"]), flush=True)
    accs = [v["acc"] for v in metrics["sets"].values()]
    metrics["cls_acc_generated"] = float(np.mean(accs)) if accs else float("nan")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("RUN_DONE", flush=True)
    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
