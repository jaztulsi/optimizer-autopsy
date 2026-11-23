---
exea_version: 1
# This repo is public: any address written here is published. Fill it in before booking.
owner_email: TODO
github_username: jaztulsi
# The runner injects the token as $EXEA_GITHUB_PAT at run time; it is entered once on the Exea dashboard.
# NEVER write the real token in this file.
github_pat: ${EXEA_GITHUB_PAT}
# Snapshotted at 13:00 PST and restored at 10:00 the next day. The venv, the data shard and every run output live here.
checkpoint_dir: ./checkpoints
---

# exea.md

Runs prereg-v3 (`docs/prereg_v3.md`) from the `prereg-v3` tag on Exea Reserve's AMD MI300X, through
`research/gpu/exea_prereg_v3.sh`.

## SETUP

Checks out the `prereg-v3` tag; with no tag, nothing runs. On the first day it builds a venv and the proxy data shard
inside the checkpoint dir. It stops if torch isn't the pinned ROCm build or can't see the GPU, because `mve` would
otherwise fall back to the CPU.

```bash
set -e
git fetch --quiet origin tag prereg-v3 --no-tags
git checkout --quiet --detach prereg-v3
CK="$EXEA_CHECKPOINT_DIR"
# numpy 2.0.2 has wheels only up to Python 3.12 and scipy 1.16.3 needs 3.11+, so use 3.12 or 3.11 when the box has one
PYB=$(command -v python3.12 || command -v python3.11 || command -v python3)
[ -x "$CK/venv/bin/python" ] || "$PYB" -m venv "$CK/venv"
"$CK/venv/bin/pip" install --quiet torch==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1
"$CK/venv/bin/pip" install --quiet -r research/requirements.txt
"$CK/venv/bin/python" -c "import torch; assert torch.version.hip and torch.cuda.is_available(), 'no ROCm GPU'"
[ -e "$CK/data/proxy/train.bin" ] || "$CK/venv/bin/python" -m research.data.prepare proxy --out-root "$CK/data"
```

## START

First, once: G0's RML-2 check (PLAN_V8 §11.2), which is not part of prereg-v3. A plain fork and a fork with the
probation hook installed inert (`xproc_gate --probation-inert`) replay the same spikes from one snapshot in separate
processes and must match bit for bit (`g0_inert/summary.json`). It runs before the runner, so it stays out of seed 1's
timing, and a FAIL does not stop prereg-v3. It reruns next session only if it crashed before `compare`. This block
runs tagged code: the `prereg-v3` tag already has the flag.

Then the runner: G0, seed 1 of E1 on its own (the cost rule), then the rest in the order prereg-v3 gives. Each `mve`
instance saves after every branch.

```bash
G="${EXEA_CHECKPOINT_DIR:?}/prereg_v3/g0_inert" X="$EXEA_CHECKPOINT_DIR/venv/bin/python -m research.gpu.xproc_gate"
if [ ! -e "$G/.done" ]; then
  rm -rf "$G" && mkdir -p "$G"
  { $X prime --out "$G" && $X fork --out "$G" --tag a && $X fork --out "$G" --tag p --probation-inert &&
    $X compare --out "$G"; } > "$G/log.txt" 2>&1
  [ -e "$G/summary.json" ] && touch "$G/.done"
fi
bash research/gpu/exea_prereg_v3.sh
```

## RESUME

The same command. Finished instances are skipped, and an unfinished one carries on from its last saved branch.

```bash
bash research/gpu/exea_prereg_v3.sh
```

## STOP

First stops the runner, so that nothing new starts, then its python processes. The brackets in the patterns keep
`pkill` from matching this block's own shell.

```bash
kill -TERM "$EXEA_JOB_PID" 2>/dev/null || true
pkill -TERM -f 'exea_prereg_v[3]' || true
pkill -TERM -f 'python.*research[.]' || true
for i in $(seq 90); do pgrep -f 'python.*research[.]' > /dev/null || break; sleep 1; done
pkill -KILL -f 'python.*research[.]' || true
```

## SAVE

Copies the results, without snapshots or the gate's token file, into `results/exea_prereg_v3/`. It commits them on top of the tag and pushes a
branch for each session, `exea/prereg-v3-s<N>`; each branch holds everything up to that session. Merge the last one
through a PR.

```bash
R=results/exea_prereg_v3
mkdir -p "$R"
tar -C "$EXEA_CHECKPOINT_DIR/prereg_v3" --exclude='*.safetensors' --exclude='*.bin' --exclude='*.tmp' -cf - . | tar -C "$R" -xf -
git config user.name > /dev/null || git config user.name "$EXEA_GITHUB_USERNAME"
git config user.email > /dev/null || git config user.email "$EXEA_GITHUB_USERNAME@users.noreply.github.com"
git add "$R"
git commit --quiet -m "exea: prereg-v3 results after session $EXEA_SESSION_COUNT ($EXEA_JOB_ID)" || echo "nothing to commit"
git push --quiet origin "HEAD:refs/heads/exea/prereg-v3-s$EXEA_SESSION_COUNT"
```
