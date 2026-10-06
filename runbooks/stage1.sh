#!/bin/bash
# research/17 stage 1 on the 8x5090 box: ~419k PixelProse URLs -> 361-slot strokes + scores (8 workers), DOCCI test set, merge
cd /root/spike; mkdir -p out/pp
for w in 0 1 2 3 4 5 6 7; do
  (CUDA_VISIBLE_DEVICES=$w python -u extract_v3.py --out out/pp/w$w --manifest out/pp/manifest_$w.jsonl --shard-size 400 --threads 48 --batch 64 > out/pp/x$w.log 2>&1 \
   && CUDA_VISIBLE_DEVICES=$w python -u pp_score.py out/pp/w$w >> out/pp/x$w.log 2>&1; echo "W${w}_DONE" >> out/pp/x$w.log) &
done
(CUDA_VISIBLE_DEVICES=7 python -u extract_v3.py --out out/pp/docci --manifest out/pp/manifest_docci.jsonl --shard-size 400 --threads 16 > out/pp/xdocci.log 2>&1 \
 && CUDA_VISIBLE_DEVICES=7 python -u pp_score.py out/pp/docci >> out/pp/xdocci.log 2>&1 && python -u pp_merge.py out/ppdocci out/pp/docci --drop 0.0 >> out/pp/xdocci.log 2>&1) &
wait
python -u pp_merge.py out/ppfinal out/pp/w0 out/pp/w1 out/pp/w2 out/pp/w3 out/pp/w4 out/pp/w5 out/pp/w6 out/pp/w7 --drop 0.25 > out/pp/merge.log 2>&1
cp out/final/prompts.txt out/ppfinal/; cp out/final/prompts.txt out/ppdocci/
echo STAGE1_DONE
