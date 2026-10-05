"""Export a training checkpoint to the Hugging Face layout used by paint.py: model.safetensors + config.json.

  python export_hf.py ckpt.pt --data ../data/ppfinal --out hf_export
Stores the EMA weights, the stroke normalisation, slot anchors, caption-vector standardisation and a pool of real
(base, detail) stroke counts (paint.py copies the stroke count of a random real painting). No pickle in the release.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import strokegen as sg

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--data", required=True, help="training data dir (for the stroke-count pool)")
ap.add_argument("--out", default="hf_export")
ap.add_argument("--pool", type=int, default=20000)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ca = ck["args"]
_, keep, _, _, _ = sg.load_data(a.data, "v2", ck["N"])
g = torch.Generator().manual_seed(0)
pool = keep[torch.randperm(len(keep), generator=g)[:a.pool]]
t = {f"model.{k}": v.contiguous() for k, v in ck["ema"].items()}
t.update(norm_mean=ck["norm"][0], norm_std=ck["norm"][1], anchors=ck["anchors"].float(), text_mu=ck["text_norm"][0],
         text_sd=ck["text_norm"][1], keep_pool=pool.contiguous())
if ck.get("img_norm"):
    t.update(img_mu=ck["img_norm"][0], img_sd=ck["img_norm"][1])
save_file(t, str(out / "model.safetensors"))
cfg = {"architecture": "SetDiT (set diffusion transformer over brush strokes)", "slots": ck["N"], "n_base": 165, "detail_grid": 14,
       "d": ca["d"], "layers": ca["layers"], "heads": ca["heads"], "text_dim": ck["text_dim"], "n_classes": len(ck["classes"]),
       "selfcond": ca["selfcond"], "xattn": ca["xattn"], "canvas_feedback": ca["canvas_fb"], "wmin": ck["wmin"],
       "text_encoder": ca["text_encoder"], "longclip_repo": "BeichenZhang/LongCLIP-B", "longclip_file": "longclip-B.pt",
       "prediction": "v (cosine schedule)", "sampler": "DDIM + classifier-free guidance", "default_steps": 25, "default_cfg": 3.0,
       "stroke_params": ["x0", "y0", "x1", "y1", "x2", "y2", "width", "r", "g", "b", "on"],
       "params": sum(v.numel() for k, v in t.items() if k.startswith("model."))}
json.dump(cfg, open(out / "config.json", "w"), indent=1)
print(json.dumps(cfg, indent=1), "\nEXPORTED", out)
