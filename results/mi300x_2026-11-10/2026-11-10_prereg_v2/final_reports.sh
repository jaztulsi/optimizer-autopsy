#!/bin/bash
# Timed reports with the stage-4 w floors lent to the runs that have no placebos of their own.
R=/shared-docker/optimizer-autopsy-prereg; OUT=/shared-docker/runs/2026-11-10_prereg_v2; S=/shared-docker/runs/2026-11-10_stages/stage4
D="docker exec -w $R rocm210"
for t in 04:14 04:24 04:31; do
  until [ "$(date -u +%H:%M)" \> "$t" ]; do sleep 10; done
  $D python3 -m research.autopsy.report $OUT $S/inj160_w $S/inj1000_w --out $OUT/report_final --unseal > $OUT/report_final.log 2>&1
  echo "report_final rc=$? $(date -u +%T)"
done
echo FINAL_REPORTS_DONE
