"""Sweep the detail model's sampler: guidance w x fraction of detail strokes kept. Same held-out split as detailgen.py.
python detail_sweep.py out/final out/detail   -> out/detail/sweep.json
Metrics on 128 held-out items, true base strokes: PSNR vs the real image @256, and CLIP image-image cosine to the real image."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import batched
import detail_level as dl
import detailgen as dg
import strokegen as sg
from extract_v3 import safe_name

path, run = sys.argv[1], sys.argv[2]
dev = "cuda"
torch.manual_seed(0)
S, keep, labels, ids, anchors = sg.load_data(path, "v2", slots=10 ** 6)
caps = json.load(open(f"{path}/caps.json"))
sel = [j for j, i in enumerate(ids) if i in caps]
S, keep, ids = S[sel], keep[sel], [ids[j] for j in sel]
N, nb = S.shape[1], 165
ck = torch.load(f"{run}/ckpt.pt", map_location=dev)
norm = dg.GroupNorm.__new__(dg.GroupNorm)
norm.mean, norm.std = ck["norm"][0].cpu(), ck["norm"][1].cpu()
Z = norm.enc(dg.to_vec(S, keep, anchors), keep)
perm = torch.randperm(len(Z))
nv = max(min(64, len(Z) // 5), len(Z) // 50)
vi, ti = perm[:nv], perm[nv:]
ev = vi[:128].tolist()
a = ck["args"]
model = dg.DetailDiT(N, nb, a["d"], a["layers"], a["heads"], 512).to(dev)
model.load_state_dict(ck["ema"])
model.eval()
mu, sdv = ck["text_norm"]
y = ((sg.embed_text([caps[ids[i]][0] for i in ev], dev).to(dev) - mu.to(dev)) / sdv.to(dev))
Zg, Ag = Z.to(dev), anchors.to(dev)
H, g = 256, 14
real = torch.stack([torch.from_numpy(np.asarray(Image.open(Path(path) / "img" / f"{safe_name(ids[i])}.jpg").convert("RGB").resize((H, H), Image.LANCZOS)).copy())
                    .permute(2, 0, 1).reshape(3, -1).float() / 255 for i in ev]).to(dev)
import open_clip

cm, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
cm = cm.to(dev).eval()
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)


@torch.no_grad()
def cvec(x):
    x = F.interpolate(x.view(-1, 3, H, H), size=224, mode="bilinear", align_corners=False)
    return F.normalize(cm.encode_image((x - MEAN) / STD).float(), dim=-1)


rv = cvec(real)
st_fit = dg.vec_to_strokes(norm.dec(Zg[ev]), Ag)
with torch.no_grad():
    base_hi = sg.render_samples(st_fit[:, :nb], H)
    fit_hi = torch.cat([dl.render_detail(st_fit[j:j + 16, nb:], base_hi[j:j + 16], g, H) for j in range(0, len(ev), 16)])
score = lambda im: {"psnr": round(batched.psnr(im, real).mean().item(), 3), "clip_to_real": round((cvec(im) * rv).sum(1).mean().item(), 4)}
res = {"base_only": score(base_hi), "fitted_detail": score(fit_hi), "generated": {}}
k_counts = keep[ti[:2000], nb:].sum(1)
for w in (1.0, 1.5, 3.0):
    torch.manual_seed(5)
    zg = dg.sample_detail(model, Zg[ev, :nb], y, N, nb, steps=40, w=w)
    kv = norm.dec(zg)[:, nb:, 0]
    for frac in (0.25, 0.5, 1.0):
        st = dg.vec_to_strokes(norm.dec(zg), Ag)
        for r in range(len(ev)):
            k = int(int(k_counts[torch.randint(len(k_counts), (1,))]) * frac)
            on = torch.zeros(N - nb, device=dev)
            if k:
                on[kv[r].topk(k).indices] = 1
            st[r, nb:, 10] = on
        with torch.no_grad():
            gen = torch.cat([dl.render_detail(st[j:j + 16, nb:], base_hi[j:j + 16], g, H) for j in range(0, len(ev), 16)])
        res["generated"][f"w{w}_keep{frac}"] = score(gen)
        print(w, frac, res["generated"][f"w{w}_keep{frac}"], flush=True)
print(json.dumps(res, indent=1))
json.dump(res, open(f"{run}/sweep.json", "w"), indent=1)
print("SWEEP_DONE")
