#!/usr/bin/env bash
# Instrument checks for the baselines' harness changes (PR #20). GPU-only: run on the MI300X, from the
# repo root, on the baselines/rollback-skip checkout:
#
#     bash research/kaggle/baseline_gpu_session.sh [OUT_DIR]
#
# Runs, logging everything to OUT_DIR (default runs/baselines_gpu_<UTC time>):
#   1. python -m research.harness.fork      -- in-process fork gate, must print max|Δ|=0 AND exit 0
#   2. cross-process gate (xproc_gate): prime, forks a + b at once, fork c solo, compare -- all bitwise
#   3. research.kaggle.baseline_gpu_checks  -- data_offset + Brollback rebuild at grad_accum = 40
# Instrument checks only: no repair battery (that waits for the prereg-v2 tag).
# Every step runs even if an earlier one fails; each exit code goes to OUT_DIR/STATUS. Exits 1 if any failed.

set -u
OUT="${1:-runs/baselines_gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
STATUS="$OUT/STATUS"
fail=0

record() {  # record NAME RC
    echo "$1 rc=$2" | tee -a "$STATUS"
    [ "$2" -eq 0 ] || fail=1
}

date -u | tee "$STATUS"
git log -1 --format='commit %h %s' | tee -a "$STATUS"
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" 2>&1 | tee -a "$STATUS"

# The cross-process gate lives on plan-v8 (PR #18). It only uses modules this branch has, so run it HERE,
# against this branch's fork.py / trunk.py -- which is the code under test. Not committed.
if [ ! -f research/kaggle/xproc_gate.py ]; then
    git fetch -q origin plan-v8 && git show origin/plan-v8:research/kaggle/xproc_gate.py > research/kaggle/xproc_gate.py
fi

# 1. In-process fork gate.
python -m research.harness.fork > "$OUT/fork_selfcheck.log" 2>&1
record fork_selfcheck $?
tail -2 "$OUT/fork_selfcheck.log"

# 2. Cross-process + co-tenancy gate.
X="$OUT/xproc"
python -m research.kaggle.xproc_gate prime --out "$X" > "$OUT/xproc_prime.log" 2>&1
record xproc_prime $?
python -m research.kaggle.xproc_gate fork --out "$X" --tag a > "$OUT/xproc_fork_a.log" 2>&1 &
pa=$!
python -m research.kaggle.xproc_gate fork --out "$X" --tag b > "$OUT/xproc_fork_b.log" 2>&1 &
pb=$!
wait $pa; record xproc_fork_a_concurrent $?
wait $pb; record xproc_fork_b_concurrent $?
python -m research.kaggle.xproc_gate fork --out "$X" --tag c > "$OUT/xproc_fork_c.log" 2>&1
record xproc_fork_c_solo $?
python -m research.kaggle.xproc_gate compare --out "$X" > "$OUT/xproc_compare.log" 2>&1
record xproc_compare $?
tail -1 "$OUT/xproc_compare.log"

# 3. The baselines' own bitwise checks on the GPU.
python -m research.kaggle.baseline_gpu_checks --out "$OUT/bgc" > "$OUT/baseline_gpu_checks.log" 2>&1
record baseline_gpu_checks $?
tail -3 "$OUT/baseline_gpu_checks.log"

date -u | tee -a "$STATUS"
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASS -> post $OUT"; else echo "SOME CHECKS FAILED -> post $OUT"; fi | tee -a "$STATUS"
exit "$fail"
