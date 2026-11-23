#!/bin/bash
cd /shared-docker/optimizer-autopsy
for k in corrupt_batch repeat_ngram lr_bump; do
  docker exec -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -w /shared-docker/optimizer-autopsy rocm python3 -m research.autopsy.mve --config /shared-docker/runs/t1_pilot/config.yaml --kind $k --inject-step 160 --seed 0 --data-dir /shared-docker/data/proxy --out /shared-docker/runs/t1_pilot/$k --skip-preflight > /shared-docker/runs/t1_pilot/$k.log 2>&1 &
done
wait; date > /shared-docker/runs/t1_pilot/DONE
