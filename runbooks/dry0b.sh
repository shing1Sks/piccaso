#!/bin/bash
# research/17 stage 0b on the 8x5090 box: data pipeline on 2k PixelProse rows, 761-slot ceiling probe, speed tests, eval plumbing
cd /root/spike; mkdir -p out/dry
head -n 1000 out/pp/manifest_0.jsonl > out/dry/m0.jsonl; head -n 1000 out/pp/manifest_1.jsonl > out/dry/m1.jsonl
for w in 0 1; do (CUDA_VISIBLE_DEVICES=$w python -u extract_v3.py --out out/dry/w$w --manifest out/dry/m$w.jsonl --shard-size 400 --threads 48 > out/dry/x$w.log 2>&1 && CUDA_VISIBLE_DEVICES=$w python -u pp_score.py out/dry/w$w >> out/dry/x$w.log 2>&1) & done
(CUDA_VISIBLE_DEVICES=2 python -u extract_v3.py --out out/dry/g20 --manifest out/dry/m0.jsonl --shard-size 400 --threads 48 --detail-g 20 > out/dry/x2.log 2>&1 && CUDA_VISIBLE_DEVICES=2 python -u pp_score.py out/dry/g20 >> out/dry/x2.log 2>&1) &
wait
python -u pp_merge.py out/dry/final out/dry/w0 out/dry/w1 --drop 0.25 > out/dry/merge.log 2>&1; cp out/final/prompts.txt out/dry/final/
python - <<'PY' > out/dry/ceiling761.json
import torch, json
a, b = torch.load("out/dry/w0/score.pt", weights_only=False), torch.load("out/dry/g20/score.pt", weights_only=False)
ib = {i: k for k, i in enumerate(b["ids"])}
common = [k for k, i in enumerate(a["ids"]) if i in ib]
sa = a["survive"][common]; sb = b["survive"][[ib[a["ids"][k]] for k in common]]
pa = [json.loads(l)["psnr_full256"] for l in open("out/dry/w0/log.jsonl")]; pb = [json.loads(l)["psnr_full256"] for l in open("out/dry/g20/log.jsonl")]
print(json.dumps({"n": len(common), "survive_361": round(sa.mean().item(), 4), "survive_565_g20": round(sb.mean().item(), 4),
                  "psnr_361": round(sum(pa) / len(pa), 2), "psnr_565_g20": round(sum(pb) / len(pb), 2)}))
PY
C="--arm B --data v2 --path out/dry/final --text-json out/dry/final/caps.json --text-cache out/dry/final/tc.pt --text-norm --group-labels --prompts-file out/dry/final/prompts.txt --img-cond 0.25 --clipvec out/dry/final/clipvec.pt --batch 256 --patience 0 --slots 361 --slot-weight 3,1.5,1,1 --text-encoder longclip --xattn --selfcond --warmup 100 --max-steps 300 --clip-eval 100 --save-every 200 --peek-prompts 4"
CUDA_VISIBLE_DEVICES=0 python -u strokegen.py $C --cache-only --out out/dry/cache > out/dry/cache.log 2>&1  # caption caches, built once
CUDA_VISIBLE_DEVICES=0 python -u strokegen.py $C --d 384 --layers 8 --heads 6 --out out/dry/m25 > out/dry/s25.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u strokegen.py $C --d 640 --layers 11 --heads 10 --out out/dry/m100 > out/dry/s100.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u strokegen.py $C --d 896 --layers 14 --heads 14 --lr 2e-4 --out out/dry/m250 > out/dry/s250.log 2>&1 &
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --standalone --nproc_per_node 4 strokegen.py $C --d 640 --layers 11 --heads 10 --out out/dry/m100ddp > out/dry/s100ddp.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python -u strokebench.py out/dry/m100/ckpt.pt --out out/dry/m100/strokebench.json --per 1 > out/dry/sb.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u diag_eval.py out/dry/m100/ckpt.pt --path out/dry/final --out out/dry/m100/diag.json --n 100 --steps 25 --cfgs 3 --per-source 30 > out/dry/diag.log 2>&1 &
wait
echo DRY_DONE
