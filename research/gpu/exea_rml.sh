#!/bin/bash
# The RML block's first GPU runs on Exea Reserve's MI300X (PLAN_V8 §11.2 ledger: kill test 0.2 h, X4 0.5 h).
#   kill test, part 1 (docs/novelty/rml3_method.md §8.2): Appeal's read-only hook on B0, width 1, forks at steps
#     1000 and 1800, repeat_ngram and hard_rows at 25% of rows, K = 10. Parts 2 and 3 are not built.
#   X4 echo checks (docs/novelty/method.md §6.2): 60-step forks at z in {2, 6, 10, 30}, at the same two phases.
# Both are exploratory: no tagged pre-registration covers them. About 18,000 fork-steps with priming [estimate].
#
# Not wired into exea.md, and nothing runs it until Jasraj gives the go-ahead for that GPU session. exea.md runs
# prereg-v3 from the prereg-v3 tag, which has neither battery, so this file runs from its own copy of the main commit
# that has it. The venv is prereg-v3's: research/requirements.txt and research/harness/ are the same at the tag and on
# main (checked 2026-11-23). To run it, add these lines at the end of START and of RESUME, with <sha> that commit:
#   S=$EXEA_CHECKPOINT_DIR/rml_src
#   [ -e "$S/COMMIT" ] || { git fetch --quiet origin <sha> && mkdir -p "$S" && git archive FETCH_HEAD | tar -x -C "$S" \
#     && echo <sha> > "$S/COMMIT"; }
#   (cd "$S" && bash research/gpu/exea_rml.sh)
# and to SAVE the same tar line for "$EXEA_CHECKPOINT_DIR/rml" into results/exea_rml/.
# It starts only once prereg-v3's G0 has passed and its seed 1 has been timed (rml3_method §8.2), so it never runs
# inside seed 1's cost-rule clock. Placed after exea_prereg_v3.sh, it runs when that script exits for the session.
set -u
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # before any python imports torch (CLAUDE.md §1)
CK=${EXEA_CHECKPOINT_DIR:?run this from exea.md}
PY=$CK/venv/bin/python DATA=$CK/data/proxy OUT=$CK/rml V3=$CK/prereg_v3 P=5
CFG=results/mi300x_2026-10-27/t1_pilot/config.yaml
A="--config $CFG --seed 1"
mkdir -p "$OUT"
exec >> "$OUT/queue.log" 2>&1
code=$(cat COMMIT 2>/dev/null || git rev-parse --short HEAD)
echo "=== session ${EXEA_SESSION_COUNT:-?} start $(date -u +%FT%TZ) code $code"
[ -e "$V3/g0/PASS" ] || { echo "prereg-v3 G0 has not passed on this box: nothing runs"; exit 1; }
[ -e "$V3/e1_seeds" ] || { echo "prereg-v3 seed 1 is not timed yet: nothing runs"; exit 1; }

# ponytail: run, pool and missing are exea_prereg_v3.sh's, copied; that file is pinned by the prereg-v3 tag
run() {  # one mve instance
  local name=$1; shift
  [ -e "$OUT/$name/.done" ] && return
  echo "$name start $(date -u +%T)"
  if "$PY" -m research.autopsy.mve "$@" --data-dir "$DATA" --out "$OUT/$name" >> "$OUT/$name.log" 2>&1; then
    touch "$OUT/$name/.done"
    rm -f "$OUT/$name"/*.safetensors  # resume state only
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

plan() {
  for inj in 1000 1800; do
    for k in repeat_ngram hard_rows; do
      echo "kill1_${k}_inj$inj $A --kind $k --battery appeal --appeal-k 10 --width 1 --inject-step $inj"
    done
    echo "x4_inj$inj $A --battery echo --echo-z 2,6,10,30 --echo-steps 60 --horizons 60 --inject-step $inj"
  done
}
plan | pool

left=$(plan | missing)
if [ -z "$left" ]; then
  touch "$OUT/ALL_DONE"; echo "ALL_DONE $(date -u +%FT%TZ)"
else
  echo "not done: $(echo $left)"
fi
