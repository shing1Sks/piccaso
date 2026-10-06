"""Score extracted PixelProse items (research/17 stage 1), GPU.

  python pp_score.py out/pp/w0            -> out/pp/w0/score.pt
Per item: Long-CLIP image vector of the real picture (reference-picture conditioning), OpenAI CLIP image vector of the real
picture (the evaluation scorer's space), and the "survives strokes" score = Long-CLIP cosine between the 361-stroke render
and the full caption (agree_real = the same for the real picture, for reference).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import batched
import detail_level as dl
import strokegen as sg
from extract_v3 import safe_name

d = Path(sys.argv[1])
dev = "cuda" if torch.cuda.is_available() else "cpu"
sg.TEXT_ENC.update(kind="longclip", ckpt=sys.argv[2] if len(sys.argv) > 2 else "out/models/longclip-B.pt")
lc = sg.get_text_enc(dev)  # Long-CLIP (text + image towers)
oai = sg.get_text_enc(dev, "clip")  # OpenAI CLIP ViT-B/32
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)
texts = {r["id"]: r["texts"] for r in json.load(open(d / "items.json", encoding="utf-8"))}


@torch.no_grad()
def ivec(m, img):  # img (B,3,H,W) in [0,1]
    x = (F.interpolate(img, size=224, mode="bicubic", align_corners=False).clamp(0, 1) - MEAN) / STD
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        return F.normalize(m.encode_image(x).float(), dim=-1)


ids, I_lc, I_oai, surv, agree = [], [], [], [], []
for f in sorted(glob.glob(str(d / "shard_*.pt"))):
    sh = torch.load(f, weights_only=False)
    ok = sh["ok"]
    S = sh["strokes"][ok].float()
    sid = [i for i, o in zip(sh["ids"], ok) if o]
    for j in range(0, len(sid), 100):
        st = S[j:j + 100].to(dev)
        nb, g = sh["n_base"], sh["detail"]["g"]
        with torch.no_grad():  # render in small chunks: 256 px masks of 165 strokes are large
            base = torch.cat([batched.render(st[q:q + 16, :nb], 256, 256) for q in range(0, len(st), 16)])
            full = torch.cat([dl.render_detail(st[q:q + 16, nb:], base[q:q + 16], g, 256) for q in range(0, len(st), 16)]).view(-1, 3, 256, 256)
        real = torch.stack([torch.from_numpy(np.asarray(Image.open(d / "img" / f"{safe_name(i)}.jpg").convert("RGB"))).permute(2, 0, 1)
                            for i in sid[j:j + 100]]).float().div(255).to(dev)
        tv = lc.pooled(lc.tokenize([texts[i][0] for i in sid[j:j + 100]])).to(dev)
        r_lc, f_lc = ivec(lc.m, real), ivec(lc.m, full)
        ids += sid[j:j + 100]
        I_lc.append(r_lc.half().cpu())
        I_oai.append(ivec(oai.m, real).half().cpu())
        surv.append((f_lc * tv).sum(1).cpu())
        agree.append((r_lc * tv).sum(1).cpu())
    print(f, len(ids), flush=True)
res = {"ids": ids, "img": torch.cat(I_lc), "img_oai": torch.cat(I_oai), "survive": torch.cat(surv), "agree_real": torch.cat(agree)}
torch.save(res, d / "score.pt")
print(json.dumps({"n": len(ids), "survive_mean": round(res["survive"].mean().item(), 4),
                  "survive_p25": round(res["survive"].quantile(.25).item(), 4), "agree_real_mean": round(res["agree_real"].mean().item(), 4)}),
      "SCORE_DONE", flush=True)
