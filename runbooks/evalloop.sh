#!/bin/bash
# usage: CUDA_VISIBLE_DEVICES=k bash evalloop.sh "<glob of run dirs>" <stop-flag-file>   (several loops may share the work: mkdir locks)
# every new ckpt_*.pt / ckpt.pt -> StrokeBench + CLIP retrieval on 1,000 held-out PixelProse captions + DOCCI (all held out)
cd /root/spike
pat=$1; stop=$2
while true; do
  pending=0
  for c in $pat/ckpt_*.pt $pat/ckpt.pt; do
    [ -f "$c" ] || continue
    tag=${c%.pt}
    [ -f "$tag.evaldone" ] && continue
    pending=1
    mkdir "$tag.lock" 2>/dev/null || continue
    sleep 20  # let the writer finish
    python -u strokebench.py $c --out $tag.sb.json --per 2 > $tag.sb.log 2>&1
    python -u diag_eval.py $c --path out/ppfinal --out $tag.diag.json --n 1000 --steps 25 --cfgs 3 --per-source 0 > $tag.diag.log 2>&1
    python -u diag_eval.py $c --path out/ppdocci --all --out $tag.docci.json --n 1000 --steps 25 --cfgs 3 --per-source 0 > $tag.docci.log 2>&1
    touch $tag.evaldone; echo "EVAL_DONE $c $(date +%H:%M)" | tee -a out/evalloop.log
  done
  [ -f "$stop" ] && [ $pending = 0 ] && break
  sleep 60
done
echo EVALLOOP_DONE | tee -a out/evalloop.log
