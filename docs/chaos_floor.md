# Chaos-floor benchmark (implemented 2026-10-29, not yet run)

Implements the Lacuna proposal "chaos floor benchmark" (run 24f9727a) on the existing harness. It measures how
far training outcomes move from numerical noise alone: placebo forks with a 1-ulp nudge and no spike. That floor,
σ_chaos, is the unit that spike-damage and repair effects are reported in (d = Δ / σ_chaos).

## Design

| Piece | As implemented |
|---|---|
| Model / data | Proxy GPT (≈ 11M params, 9.7M of them the tied token embedding; ≈ 1.3M in the blocks), TinyStories, `max_steps` 2000, seed 0, constant LR. Config: `results/mi300x_2026-10-27/t1_pilot/config.yaml` |
| Checkpoints (phases) | Fork steps 160 (early; the pilot's spike step), 1000 (plateau, 50%), 1800 (late, 90%) |
| Arms | `w`: every weight ±1 ulp. `g`: every final gradient ±1 ulp. `opt`: every Adam moment m and v ±1 ulp (a v of exactly 0 is kept at 0 so √v stays real). Signs come from a seeded coin per element |
| Forks | K = 111 per (phase, arm) → 999 placebo forks. Plus 8 **data-order forks** per phase (the remaining batches replayed in a seeded new order: the seed-level spread to compare against) |
| Determinism check | `Pl_0`, a fork with no nudge, must equal B\* exactly at every horizon (`zero_pert_exact` in the report) |
| Horizons | 1, 20, 100 and 200 steps after the fork (≈ 0%, 1%, 5% and 10% further tokens), plus end of schedule |
| Metric | σ over forks of Δ = val(fork) − val(B\*), in nats/token, per (arm, phase, horizon), on held-out slots disjoint from the probe batches |

## Run (MI300X only, after the user's go-ahead)

```bash
# inside the pinned container; CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before Python starts
CFG=results/mi300x_2026-10-27/t1_pilot/config.yaml
mkdir -p results/chaos_floor
for inj in 160 1000 1800; do
  for ch in w g opt; do
    extra=""; [ "$ch" = w ] && extra="--data-forks 8"
    python -m research.autopsy.mve --config $CFG --battery chaos --seed 0 --inject-step $inj \
      --placebo 111 --placebo-channels $ch $extra --horizons 1,20,100,200 \
      --out results/chaos_floor/inj${inj}_$ch > results/chaos_floor/inj${inj}_$ch.log 2>&1 &
  done
done; wait
python -m research.autopsy.report results/chaos_floor results/mi300x_2026-10-27/t1_pilot \
  --out results/chaos_floor/report
```

- **One process per (phase, arm).** Each instance writes its own JSON, so concurrent processes never share a file.
  Every process also re-runs B\* and `Pl_0`. The B\* values of the three processes at the same phase must match
  bitwise, which is a free cross-process check.
- **Resume.** A rerun skips finished forks.
- **Cost** [A, from the measured ≈ 0.012 GPU-h per 1,840-step proxy fork]: ≈ 1,030 forks of 1,840 / 1,000 / 200
  steps ≈ 7 MI300X-h, plus the held-out evals.

## Outputs (`results/chaos_floor/report/`)

- **`report.md` / `report.json`:** σ_chaos per arm and horizon for each instance (`sigma_chaos`), the data-order
  spread (`data`), the zero-perturbation check, and the effect unit used (`sigma_unit`: own or borrowed).
- **`chaos_forks.jsonl`:** the release table, with one row per fork per horizon: instance, inject_step, seed,
  betas, arm (`w` / `g` / `opt` / `data` / `zero`), fork, horizon, val_nats, D.
- **`sealed.json` → `_d_end`:** spike damage and the C-2 arms in chaos units. An instance without its own
  placebos (for example the pilot) borrows the w-channel floor of the chaos run forked from the same trunk (same
  cfg, same inject step). So the pilot's `corrupt_batch` instance at step 160 and β = (0.9, 0.95) gets
  d = Δ / σ_chaos. For the pilot's β1 = 0.8 / 0.95 instances, add chaos runs with `--betas 0.8,0.95` /
  `0.95,0.95` at step 160.

## Proposal's prediction and checks (recorded as given)

- **Prediction:** σ ≤ 0.005 nats/token at the key horizons, so the ~0.01-nat pilot effect would reach d ≥ 2.
  Expected σ ≈ 0.003 ± 0.001.
- **Determinism check:** `Pl_0` must replay B\* exactly. Any nonzero value means the harness is broken; stop.
- **Other kill conditions** in the proposal: σ independent of phase and horizon; d < 2 for the RML effects.

**Prior work:**
- Summers & Dinneen 2021 ([arXiv 2103.04514](https://arxiv.org/abs/2103.04514)): one-bit changes to the
  *initial* parameters lead to very different models. The proposal's text says mid-training; the paper's change is
  at initialization.
- Frankle et al. 2020 (linear mode connectivity);
- Altıntaş et al. 2025 ("The Butterfly Effect").

## Not implemented (ponytail)

- **Data-order forks at `grad_accum` > 1** (the 124M config). They need PR #20's `ctx["data_offset"]`; the proxy
  uses `grad_accum` = 1. The code raises before any GPU time is spent.
- **Hugging Face upload of `chaos_forks.jsonl`.** One manual
  `huggingface-cli upload <repo> results/chaos_floor/report --repo-type dataset` after the run.
