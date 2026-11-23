#!/bin/bash
# prereg-v3 (docs/prereg_v3.md) on Exea Reserve's MI300X: exea.md's START and RESUME both run this file.
# Safe to rerun every day: a finished job leaves a .done marker, and mve resumes an unfinished one branch by branch
# from its own --out. The job list copies prereg_v3.md's "Run" section as drafted on 2026-11-21: change both together
# before the tag. Extension A was removed (open decision 2, decided 2026-11-22).
set -u
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # before any python imports torch (CLAUDE.md §1)
CK=${EXEA_CHECKPOINT_DIR:?run this from exea.md}
PY=$CK/venv/bin/python DATA=$CK/data/proxy OUT=$CK/prereg_v3 P=5  # P: concurrent mve processes, as in v2
CFG=results/mi300x_2026-10-27/t1_pilot/config.yaml
A="--config $CFG --kind corrupt_batch"
DOSE="--taus 0 --c2-taus 0 --dts= --dose-taus 0 --placebo 8 --placebo-channels w"
mkdir -p "$OUT"
exec >> "$OUT/queue.log" 2>&1
echo "=== session ${EXEA_SESSION_COUNT:-?} start $(date -u +%FT%TZ) code $(git rev-parse --short HEAD) $(git describe --tags --always)"

g0() {  # PLAN_V8 G0 on day 1. Every later day adds one fork that must match day 1's bit for bit (same numerics)
  local d=$OUT/g0 x="$PY -m research.gpu.xproc_gate"
  if [ ! -e "$d/PASS" ]; then
    rm -rf "$d"
    $x prime --out "$d" || return 1
    $x fork --out "$d" --tag a & $x fork --out "$d" --tag b & wait  # two concurrent
    $x fork --out "$d" --tag c                                       # one solo
  else
    rm -f "$d"/fork_s*.json  # day 1's a, b, c vs today's fork only; an older one may be half-written by a 13:00 kill
    $x fork --out "$d" --tag "s${EXEA_SESSION_COUNT:-0}" || return 1
  fi
  rm -f "$d/summary.json"  # else a crashed compare would leave day 1's PASS behind
  $x compare --out "$d" > "$d/compare.log" 2>&1  # exits 0 on FAIL too: read its summary
  grep -q '"PASS": true' "$d/summary.json" && [ "$(ls "$d"/fork_*.json | wc -l)" -ge 3 ] && touch "$d/PASS"
}

run() {  # one mve instance
  local name=$1; shift
  [ -e "$OUT/$name/.done" ] && return
  echo "$name start $(date -u +%T)"
  if "$PY" -m research.autopsy.mve "$@" --data-dir "$DATA" --out "$OUT/$name" >> "$OUT/$name.log" 2>&1; then
    touch "$OUT/$name/.done"
    # ponytail: a done job's snapshots are only resume state (prime_trunk rebuilds them bit for bit); dropping them
    # keeps the 13:00 disk snapshot small.
    rm -f "$OUT/$name"/*.safetensors
    echo "$name done $(date -u +%T)"
  else
    echo "$name FAILED rc=$? $(date -u +%T): retried next session, see $name.log"
  fi
}

pool() {  # run "name args..." lines in order, P at a time
  local name args
  while read -r name args; do
    while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do wait -n; done
    run "$name" $args &  # unquoted on purpose: args is a flag list with no spaces inside a value
  done
  wait
}

missing() { while read -r name _; do [ -e "$OUT/$name/.done" ] || echo "$name"; done; }

e1() {  # core: phases 1000 and 1800, dose arms + floor, and the remedy battery at lag 3
  for inj in 1000 1800; do
    echo "s$1_inj$inj $A --seed $1 --inject-step $inj --horizons 1,20,100,200 --plant-coefs 1,10,30 $DOSE"
    echo "s$1_repair_inj$inj $A --seed $1 --inject-step $inj --horizons 1,20,100,200,999 --battery repair --lag 3"
  done
}

e2() {  # beta2 arms: (beta2, inject step, doses, horizons) rows of prereg_v3.md's E2 table
  local b2 inj coefs hor
  while read -r b2 inj coefs hor; do
    echo "s$1_b2_${b2}_inj$inj $A --seed $1 --betas 0.9,$b2 --inject-step $inj --horizons $hor --plant-coefs $coefs $DOSE"
  done <<'ROWS'
0.95 1980 1,10,30 1,10
0.99 1900 1,22.4,67.1 10,50
0.999 1000 1,70.7,212 100,500
0.99 1000 1,22.4,67.1 100,500
ROWS
  echo "s$1_b2_0.999_repair $A --seed $1 --betas 0.9,0.999 --inject-step 1000 --horizons 100,500 --battery repair --lag 3"
}

b1() {  # beta1 arms at step 1800
  for b in 0.8 0.95; do
    echo "s$1_b1_$b $A --seed $1 --betas $b,0.95 --inject-step 1800 --horizons 1,20,100 --plant-coefs 1,10,30 $DOSE"
  done
}

e3() {  # floor curve, seed 1. K = 8 on the w channel only: my reading of the budget's 48,000 fork-steps
  for inj in 160 500 1000 1400 1800 1950; do
    echo "s1_chaos_inj$inj --config $CFG --battery chaos --seed 1 --inject-step $inj --horizons 1,20,100,200 --placebo 8 --placebo-channels w"
  done
}

g0 || { echo "G0 FAILED ($OUT/g0/compare.log): forks differ on this box, so nothing else runs"; exit 1; }

# Cost rule (prereg_v3.md, Rules): seed 1 of E1 runs alone first and its GPU time decides 7 seeds or 5, before seed 2.
if [ ! -e "$OUT/e1_seeds" ]; then
  # ponytail: wall time in whole minutes, summed over sessions; the GPU runs nothing else meanwhile
  ( while sleep 60 && kill -0 $$ 2>/dev/null; do echo 60 >> "$OUT/seed1_wall_s"; done ) &
  e1 1 | pool
  kill $! 2>/dev/null
  if [ -n "$(e1 1 | missing)" ]; then echo "seed 1 of E1 not finished: nothing else starts this session"; exit 1; fi
  t=$(awk '{s += $1} END {print s + 0}' "$OUT/seed1_wall_s" 2>/dev/null)
  n=7; [ $((7 * t)) -gt 21600 ] && n=5
  echo "$n" > "$OUT/e1_seeds"
  echo "Seed 1 of E1 took $t s of GPU wall time, so 7 seeds would take $((7 * t)) s against the rule's 21600 s (6 h). E1 runs seeds 1-$n.$([ "$n" = 5 ] && echo ' The primary test for P2, P3, P4(b) and P5 becomes the one-sided paired t-test.')" \
    | tee "$OUT/cost_rule.txt"
fi
n=$(cat "$OUT/e1_seeds")

# Order of runs (Rules): E1, E2, beta1 arms, E3
plan() {
  for s in $(seq 2 "$n"); do e1 "$s"; done
  for s in 1 2 3; do e2 "$s"; done
  for s in 1 2 3; do b1 "$s"; done
  e3
}
plan | pool

"$PY" -m research.autopsy.report "$OUT" --out "$OUT/report" > "$OUT/report.log" 2>&1; echo "report rc=$?"
left=$( { e1 1; plan; } | missing)
if [ -z "$left" ]; then
  touch "$OUT/ALL_DONE"; echo "ALL_DONE $(date -u +%FT%TZ): cancel any remaining Exea bookings"
else
  echo "not done: $(echo $left)"
fi
