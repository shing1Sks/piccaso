"""StrokeBench (research/17): a fixed, data-free prompt benchmark for any arm-B checkpoint, scored with OpenAI CLIP ViT-B/32.

  python strokebench.py out/X/ckpt.pt --out out/X/strokebench.json [--per 4 --steps 25 --cfg 3]
Prompt groups (fixed lists, seed 0):
  objects  (80)  "a photo of a {obj}"            -> is {obj} CLIP's top-1 / top-5 among the 80 COCO object names?
  colour   (40)  "a photo of a {colour} {obj}"   -> right object (top-5 of 80) and right colour (top-1 of 10, object fixed)
  two      (40)  "a photo of a {o1} and a {o2}"  -> both objects in the top-5 of 80 (and: at least one)
  style    (40)  "{style} of {subject}"          -> right style (top-1 of 5, subject fixed)
Writes json (scores + chance levels) and <out>.png (one sample per prompt, first 120 prompts).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import strokegen as sg

OBJ = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
       "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
       "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
       "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
       "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
       "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
       "scissors", "teddy bear", "hair drier", "toothbrush"]
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_OBJ = ["car", "bus", "umbrella", "chair", "cup", "backpack", "bicycle", "vase", "bench", "kite", "couch", "boat", "truck", "teddy bear"]
STYLES = ["a photo", "an oil painting", "a watercolor painting", "a pencil sketch", "a cartoon illustration"]
SUBJECTS = ["a house", "a cat", "a mountain landscape", "a woman", "a bowl of fruit", "a sailing ship", "a tree", "a car"]
art = lambda w: ("an " if w[0] in "aeiou" else "a ") + w


def prompts():
    rng = random.Random(0)
    P = [("objects", f"a photo of {art(o)}", {"obj": o}) for o in OBJ]
    for _ in range(40):
        c, o = rng.choice(COLORS), rng.choice(COLOR_OBJ)
        P.append(("colour", f"a photo of {art(c)} {o}", {"obj": o, "colour": c}))
    for _ in range(40):
        o1, o2 = rng.sample(OBJ[1:], 2)
        P.append(("two", f"a photo of {art(o1)} and {art(o2)}", {"o1": o1, "o2": o2}))
    for st in STYLES:
        for sub in SUBJECTS:
            P.append(("style", f"{st} of {sub}", {"style": st, "subject": sub}))
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]
    sg.TEXT_ENC.update(kind=ca.get("text_encoder", "clip"), ckpt=ca.get("longclip_ckpt", "out/models/longclip-B.pt"))
    sg.WMIN = ck.get("wmin", 0.008)
    N = ck["N"]
    model = sg.SetDiT(N, len(ck["classes"]), ca["d"], ca["layers"], ca["heads"], ck["text_dim"], ca.get("selfcond", False),
                      ca.get("canvas_fb", False), ca.get("xattn", False)).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    norm = sg.Norm.__new__(sg.Norm)
    norm.mean, norm.std = ck["norm"][0].cpu(), ck["norm"][1].cpu()
    Ag = ck["anchors"].to(dev)
    fb = sg.render_fb_fn(norm, Ag) if model.use_canvas else None
    mu, sd = ck["text_norm"]
    P = prompts()
    strs = [p[1] for p in P]
    y = ((sg.embed_text(strs, dev).to(dev) - mu.to(dev)) / sd.to(dev)).repeat_interleave(a.per, dim=0)
    ctx = cm = None
    if model.xattn:
        te = sg.get_text_enc(dev)
        ctx, cm = te(te.tokenize(strs))
        ctx, cm = ctx.repeat_interleave(a.per, dim=0), cm.repeat_interleave(a.per, dim=0)
    torch.manual_seed(0)
    zs = torch.cat([sg.sample_B(model, y[j:j + a.bs], N, a.steps, a.cfg, ctx=None if ctx is None else ctx[j:j + a.bs],
                                ctx_mask=None if cm is None else cm[j:j + a.bs], fb=fb) for j in range(0, len(y), a.bs)])
    st = sg.vec_to_strokes(norm.dec(zs), Ag)
    import open_clip

    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip = clip.to(dev).eval()
    MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev).view(1, 3, 1, 1)
    ims, iv = [], []
    with torch.no_grad():
        for j in range(0, len(st), 32):
            im = sg.render_samples(st[j:j + 32], 224).view(-1, 3, 224, 224)
            ims.append(im[::a.per].cpu())
            iv.append(F.normalize(clip.encode_image((im - MEAN) / STD).float(), dim=-1))
    iv = torch.cat(iv)
    T = lambda xs: sg.score_text(xs, dev).to(dev)
    obj_t = T([f"a photo of {art(o)}" for o in OBJ])
    sim_obj = iv @ obj_t.T  # (prompts*per, 80)
    res = {g: [] for g in ("objects", "objects_top5", "colour_obj", "colour", "two_both", "two_any", "style")}
    for k, (g, _, m) in enumerate(P):
        rows = slice(k * a.per, (k + 1) * a.per)
        rank = sim_obj[rows].argsort(1, descending=True)
        top1, top5 = rank[:, 0], rank[:, :5]
        if g == "objects":
            oi = OBJ.index(m["obj"])
            res["objects"] += (top1 == oi).float().tolist()
            res["objects_top5"] += (top5 == oi).any(1).float().tolist()
        elif g == "colour":
            oi = OBJ.index(m["obj"])
            res["colour_obj"] += (top5 == oi).any(1).float().tolist()
            ct = T([f"a photo of {art(c)} {m['obj']}" for c in COLORS])
            res["colour"] += ((iv[rows] @ ct.T).argmax(1) == COLORS.index(m["colour"])).float().tolist()
        elif g == "two":
            a1, a2 = OBJ.index(m["o1"]), OBJ.index(m["o2"])
            h1, h2 = (top5 == a1).any(1), (top5 == a2).any(1)
            res["two_both"] += (h1 & h2).float().tolist()
            res["two_any"] += (h1 | h2).float().tolist()
        else:
            stt = T([f"{s_} of {m['subject']}" for s_ in STYLES])
            res["style"] += ((iv[rows] @ stt.T).argmax(1) == STYLES.index(m["style"])).float().tolist()
    summary = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    summary["chance"] = {"objects": round(1 / 80, 4), "objects_top5": round(5 / 80, 4), "colour": 0.1, "style": 0.2, "two_both": round((5 / 80) ** 2, 4)}
    summary.update(ckpt=a.ckpt, per=a.per, steps=a.steps, cfg=a.cfg, n_prompts=len(P))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps(summary), flush=True)
    ims = torch.cat(ims)  # one sample per prompt
    S_, cols = 160, 10
    n_show = min(120, len(P))
    sheet = Image.new("RGB", (S_ * cols, (S_ + 14) * ((n_show + cols - 1) // cols)), "white")
    dr = ImageDraw.Draw(sheet)
    for k in range(n_show):
        arr = (ims[k].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        x, y_ = (k % cols) * S_, (k // cols) * (S_ + 14)
        sheet.paste(Image.fromarray(arr).resize((S_, S_)), (x, y_))
        dr.text((x + 2, y_ + S_ + 1), P[k][1].replace("a photo of ", "")[:26], fill=(0, 0, 0))
    sheet.save(str(Path(a.out).with_suffix(".png")))
    print("STROKEBENCH_DONE", flush=True)
