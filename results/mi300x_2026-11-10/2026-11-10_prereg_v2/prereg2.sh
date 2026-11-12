#!/bin/bash
# Hardware deadline 04:35 UTC. Restart the two long dose runs with no placebos so the dose arms run first;
# the floor comes from stage 4's K=8 forks. The 1800 main run (K=16) and the repair runs keep going as launched.
trap 'echo PREREG2_EXIT' EXIT
set -u
R=/shared-docker/optimizer-autopsy-prereg; OUT=/shared-docker/runs/2026-11-10_prereg_v2; DATA=/shared-docker/data/proxy
CFG=results/mi300x_2026-10-27/t1_pilot/config.yaml
D="docker exec -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -w $R rocm210"
A="--config $CFG --kind corrupt_batch --seed 0 --horizons 1,20,100,200 --data-dir $DATA"
echo "prereg2 start $(date -u +%T) code $(git -C $R rev-parse --short HEAD)"
mkdir -p $OUT/_stopped_for_deadline
for n in main_inj160 main_inj1000 beta0.8; do mv $OUT/$n $OUT/$n.log $OUT/_stopped_for_deadline/ 2>/dev/null; done
run() { name=$1; shift; ( $D python3 -m research.autopsy.mve "$@" --out $OUT/$name >> $OUT/$name.log 2>&1; echo "$name rc=$? $(date -u +%T)" ) & }
active() { pgrep -f "^python3 -m research.autopsy.mve" | wc -l; }
report() { $D python3 -m research.autopsy.report $OUT --out $OUT/report >> $OUT/report.log 2>&1; echo "report rc=$? $(date -u +%T)"; }
( for t in 04:15 04:24 04:31; do until [ "$(date -u +%H:%M)" \> "$t" ]; do sleep 15; done; report; done; echo REPORTS_DONE ) &
run main_inj1000 $A --inject-step 1000 --taus 0 --c2-taus 0 --dts "" --plant-coefs 1,3,10,30 --dose-taus 0
run main_inj160 $A --inject-step 160 --taus 0 --c2-taus 0 --dts "" --plant-coefs 1,3,10,30 --dose-taus 0
sleep 90
for b1 in 0.8 0.95; do
  until [ "$(active)" -le 4 ]; do sleep 15; done
  run beta$b1 $A --inject-step 160 --betas $b1,0.95 --taus 0 --c2-taus 0 --dts "" --plant-coefs 10 --dose-taus 0
  echo "launched beta$b1 $(date -u +%T)"; sleep 60
done
wait
echo "PREREG2_DONE $(date -u +%T)"
