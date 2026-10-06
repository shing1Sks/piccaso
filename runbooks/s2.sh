#!/bin/bash
# research/17 stage 2: fair scaling check 25M / 100M / 250M, same data + settings + 10k steps, checkpoints at 3.3k/6.7k/10k
cd /root/spike; mkdir -p out/s2
C="--arm B --data v2 --path out/ppfinal --text-json out/ppfinal/caps.json --text-cache out/ppfinal/tc.pt --text-norm --group-labels --prompts-file out/ppfinal/prompts.txt --img-cond 0.25 --clipvec out/ppfinal/clipvec.pt --batch 256 --patience 0 --slots 361 --slot-weight 3,1.5,1,1 --text-encoder longclip --xattn --selfcond --warmup 500 --max-steps 10000 --save-every 3334 --clip-eval 1000 --peek-prompts 8"
CUDA_VISIBLE_DEVICES=0 python -u strokegen.py $C --cache-only --out out/s2/cache > out/s2/cache.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -u strokegen.py $C --d 384 --layers 8 --heads 6 --out out/s2/m25 > out/s2/m25.log 2>&1 &
CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes 1 --nproc_per_node 2 --rdzv-backend c10d --rdzv-endpoint localhost:29521 strokegen.py $C --d 640 --layers 11 --heads 10 --out out/s2/m100 > out/s2/m100.log 2>&1 &
CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nnodes 1 --nproc_per_node 4 --rdzv-backend c10d --rdzv-endpoint localhost:29522 strokegen.py $C --d 896 --layers 14 --heads 14 --lr 2e-4 --out out/s2/m250 > out/s2/m250.log 2>&1 &
(CUDA_VISIBLE_DEVICES=7 python -u strokebench.py out/base/A1/ckpt.pt --out out/base/A1/strokebench.json --per 2 > out/s2/base_A1.log 2>&1
 CUDA_VISIBLE_DEVICES=7 python -u strokebench.py out/base/B58/ckpt.pt --out out/base/B58/strokebench.json --per 2 > out/s2/base_B58.log 2>&1
 CUDA_VISIBLE_DEVICES=7 python -u diag_eval.py out/base/A1/ckpt.pt --path out/ppfinal --out out/base/A1/pp.diag.json --n 1000 --steps 25 --cfgs 3 --per-source 0 > out/s2/base_A1_diag.log 2>&1
 CUDA_VISIBLE_DEVICES=7 bash evalloop.sh "out/s2/m*" out/s2/STOP > out/s2/evalloop7.log 2>&1) &
# the 25M run finishes first: its GPU then joins the evaluation
(while [ ! -f out/s2/m25/metrics.json ]; do sleep 60; done; CUDA_VISIBLE_DEVICES=0 bash evalloop.sh "out/s2/m*" out/s2/STOP > out/s2/evalloop0.log 2>&1) &
while [ ! -f out/s2/m100/metrics.json ] || [ ! -f out/s2/m250/metrics.json ] || [ ! -f out/s2/m25/metrics.json ]; do sleep 60; done
touch out/s2/STOP
wait
echo S2_DONE
