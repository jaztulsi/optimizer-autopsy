# Step-timer results — measured GPU seconds/step (Exea Labs compute justification)

These are the *measured* per-step throughput numbers that every GPU-hour estimate in `PLAN_V6.md`
ultimately depends on ("measured seconds-per-step at proxy scale and at the larger scale"). They come
from `research/kaggle/step_timer.py` — the timed unit is exactly what a fork branch repeats:
`forward + backward + grad-clip + AdamW opt.step`, 10 warmup steps (unmeasured), 100 timed steps with
`torch.cuda.synchronize()` around each, reported as mean/std/steps-per-sec.

Each scale is a **separate measurement** — they are not merged into one number.

---

## 1. Proxy scale (~11M params) — MEASURED

- Config: `research/experiments/proxy/config.yaml` (n_layer=3, n_head=6, n_embd=192, block=256, vocab=50304, fp32)
- Params (non-embedding): **10,986,816**
- GPU: **Tesla P100-PCIE-16GB**, torch 2.7.1+cu126
- Source run: Kaggle kernel `jastulsi/optimizer-autopsy-steptime` (P100, 2026-08-29)

| batch | mean s/step | std s/step | steps/sec | note |
|------:|:-----------:|:----------:|:---------:|------|
| 16    | **0.061908** | 0.000184  | 16.15     | MEASURED (batch the P100 fits; the config's target 64 OOMs at 16 GB) |
| 32    | **0.135351** | 0.003946  | 7.39      | MEASURED (same GPU/session) |
| 64    | ≈ **0.25–0.30** | —      | ≈ 3.3–4.0 | EXTRAPOLATED (see below) — the config's real target batch |

The measured `0.0617 s/step` figure = batch-16 mean (0.061908), P100, fp32.

**Scaling (measured 2 points):** 0.135351 / 0.061908 = **2.19× for a 2× batch** — mildly super-linear
(ideal linear would be 2.00×), i.e. throughput does not stay perfectly proportional to batch.

**Batch-64 extrapolation (the config's target):** two doublings from batch 16.
- Linear lower bound: 0.061908 × 4 = **0.248 s/step**
- Compounded super-linear (2.19×²  = 4.78×): 0.061908 × 4.78 = **0.296 s/step**

→ stated as **≈ 0.25–0.30 s/step (≈ 4–5× the batch-16 number)** — a *range*, not a flat 4×, because the
measured scaling is super-linear. Labeled EXTRAPOLATED, not measured (batch-64 fp32 OOMs a free 16 GB P100).

### Proxy number A — fork-replay attribution battery

Formula (from `step_timer.py` / `PLAN_V6.md` §6 back-solve):

```
GPU-hours = recipes × (branches/site) × (seeds/branch) × (fork_length steps) × (sec/step) ÷ 3600
```

Default scenario (recipes=4, branches=7, seeds=3, fork_length=200) → 4·7·3·200 = **16,800 fork steps**.

- At the **measured** batch-16 rate: 16,800 × 0.061908 ÷ 3600 = **0.29 GPU-hours** (MEASURED input)
- At the batch-64 **extrapolated** rate (0.25–0.30): 16,800 × (0.25–0.30) ÷ 3600 = **1.17–1.40 GPU-hours**

### Proxy number B — trunk-training-to-spike

Formula:

```
GPU-hours = (steps to train the trunk until the spike window) × (sec/step) ÷ 3600
```

Proxy trunk reaches its (induced) spike window fast: injections land at the ~160-step plateau and
`research/kaggle/spike_run.py` runs `STEPS=200`, so steps-to-spike ≈ **200**.

- Per single trunk-to-spike run, measured batch-16 rate: 200 × 0.061908 ÷ 3600 = **≈ 0.0034 GPU-hours** (~12 s)
- Whole proxy calibration set (spike_run = 4 recipes × 2 seeds = 8 trunk runs → 1,600 steps):
  1,600 × 0.061908 ÷ 3600 = **≈ 0.028 GPU-hours** (~100 s)

Both proxy numbers confirm proxy-scale work is effectively free on a free P100 — the compute request is
driven by the larger scale below, not the proxy.

---

## 2. Larger scale (124M "robustness check", `research/experiments/llm124m/config.yaml`) — MEASURED

Separate measurement from the proxy — **not** merged into one number.

- Config: `research/experiments/llm124m/config.yaml` (n_layer=12, n_head=12, n_embd=768, block=1024, vocab=50304, fp32 timing)
- Params (non-embedding): **123,587,328** (≈ 11.25× the proxy's 10,986,816 — matches the plan's 124M vs 11M)
- GPU: **Tesla P100-PCIE-16GB**, torch 2.7.1+cu126
- Source run: Kaggle kernel `jastulsi/optimizer-autopsy-steptime-124m` (P100, 2026-08-29)
- Same timed unit as the proxy: `forward + backward + grad-clip + AdamW opt.step`, 10 warmup, 100 timed, `cuda.synchronize()` per step.

**Batch that fit:** probed 64 → 32 → 16 → 8; **batch=8 is the largest power-of-2 that fits 16 GB**.
64/32/16 all `torch.OutOfMemoryError` (batch-16 needed a 3.07 GiB alloc with only 2.85 GiB free — a near miss).

| batch | mean s/step | std s/step | steps/sec | note |
|------:|:-----------:|:----------:|:---------:|------|
| 8     | **1.080201** | 0.003229  | 0.93      | MEASURED — largest batch that fits the P100 |
| 4     | **0.557128** | 0.001390  | 1.79      | MEASURED — scaling point (same GPU/session) |

**Scaling (measured 2 points):** 1.080201 / 0.557128 = **1.94× for a 2× batch** — mildly *sub*-linear
(ideal linear 2.00×). Note this is the **opposite** of the proxy's super-linear 2.19×: at 124M/block-1024
the step is already compute-bound, so larger batches get a small economy of scale rather than a penalty.

### Headline: real per-OPTIMIZER-step cost (as training actually runs) — ≈ 64–68 s/step (UPPER BOUND)

Unlike the proxy (grad_accum=1, so one timed step = one optimizer update), the 124M config runs
**grad_accum=40**: 40 micro-batches of size 12 accumulate before one real optimizer update. So the number
that matches how training/forks actually advance is **per optimizer step**, not per micro-step.

- **Real per-optimizer-step (upper bound): 40 × (micro-step at batch=12) = 40 × ~1.6–1.7 s = ≈ 64–68 s/step.**

Labeled **UPPER BOUND**, *not* the same class of number as the proxy's directly-measured 0.0617 s: the
micro-step figure includes one full clip+opt.step, so ×40 counts that overhead 40× instead of once. The
overshoot is small (clip+opt.step is a few % of a micro-step, so the true value is only ~2–3% below this),
but it is an extrapolation-of-an-extrapolation, so treat ≈ 64–68 s as a ceiling.

**Inputs behind the headline (labeled, not the headline itself):**
- Measured micro-steps: batch-8 = 1.080201 s, batch-4 = 0.557128 s (table above).
- Extrapolated micro-step at the config's target micro-batch (batch_size=12): two-point linear fit
  `t ≈ 0.0341 + 0.13077·batch` → **1.60 s**; ×1.5-from-batch-8 = 1.62; per-sequence-from-batch-4 = 1.67.
  → **≈ 1.6 s/micro-step (range 1.6–1.7)**, EXTRAPOLATED. Sub-linear scaling makes the linear fit a mild
  over-estimate (upper-ish), unlike the proxy's super-linear lower-bound case.

### Fork-step definition — RESOLVED from the code (not inferred from docs)

**A fork/trunk "step" is one OPTIMIZER step, and each optimizer step runs grad_accum micro-batches.**
Proof in `research/harness/trunk.py::train_forward`:
- the step counter is the OUTER loop `for step in range(start_step, start_step + steps)` (trunk.py:75);
- the INNER loop `for micro in range(grad_accum)` (trunk.py:83) does the micro-batches, indexed
  `gstep = step * grad_accum + micro` (trunk.py:85);
- exactly ONE `opt.step()` (trunk.py:115) and ONE loss appended (trunk.py:119) per outer iteration.

`run_fork` forwards `steps` unchanged (fork.py:67-70) and `fork_determinism_gate` asserts
`len(fingerprints) == steps` with one fingerprint per optimizer step (fork.py:116). So BUILD_PLAN's
"500–2000 steps" = **500–2000 optimizer steps**, i.e. 500–2000 × grad_accum(40) micro-batches.

### Real GPU-hour range (single answer, optimizer-step basis)

Using ≈ 64–68 s/optimizer-step (upper bound) and BUILD_PLAN.md's **500–2000 steps/run**:

- **(a) one trunk-to-spike run** (500–2000 optimizer steps):
  500 × 64 … 2000 × 68 ÷ 3600 = **≈ 9 – 38 GPU-hours**.
- **(b) one 8-branch short-fork battery at one site** (Task 22 DoD; 8 branches × 500–2000 steps):
  8 × 500 × 64 … 8 × 2000 × 68 ÷ 3600 = **≈ 71 – 302 GPU-hours**.

Kept separate from the proxy's §1 numbers — different model scale, do not merge.

> **⚠ This contradicts BUILD_PLAN.md line 29's "~1–3 h each" estimate by 1–2 orders of magnitude.**
> The measurement is not ambiguous: forks advance in optimizer steps at grad_accum=40 (proven above),
> so a single 8-branch battery is ~71–302 GPU-h, not 1–3 h. The plan's estimate is simply an
> under-budget (the 124M config is marked "TODO: fill in real values"). The only way back to ~1–3 h is a
> *design change* — run 124M forks at a much smaller grad_accum — which is not what the shared cfg does today.

### Correctness caveat — grad_accum=40 determinism now VERIFIED on GPU (was the open risk)

The 124M number above assumes forks replay deterministically at grad_accum=40. Before this, that had
**never** been checked — the bitwise Δ==0 gate (Tasks 4–7, `results/task7_fork_gate.json`) and both
`_selfcheck`s that exercise it ran only at **grad_accum=1** (`fork.py:263`; proxy config), while the
grad_accum>1 branch (`trunk.py:82-113`) adds a manual accumulation buffer (`acc.add_` / `div_`) absent
from that proven path.

**Verified 2026-08-29 (Kaggle P100, kernel `jastulsi/optimizer-autopsy-gate-124m`):** ran
`fork_determinism_gate` VERBATIM on the 124M config at its real grad_accum=40
(n_layer=12, n_head=12, n_embd=768, block=1024, batch=8, warmup=5, gate=10 steps) →
**noop-vs-noop max|Δ| = 0.0 on cuda** (PASS). So the accumulation path is bitwise-deterministic at
grad_accum=40, and the costed hours above rest on a verified-replayable fork, not an assumption.

Still open (not a determinism problem): the 124M **driver** is a stub
(`research/experiments/llm124m/run.py:14` → `raise SystemExit("TODO")`) and `attribution.attribute()` /
`necessity_sufficiency()` are TODOs (`attribution.py:17-18`) — the path is costed and determinism-proven,
but not yet built.

### Open question to resolve before this number is used (flagged, not resolved)

BUILD_PLAN.md **Task 22** frames the 124M check as running **"on Kaggle within free limits"**, with an
explicit **weekly Kaggle total vs the 30 h cap**, disconnect-survival checkpointing every ~1000 steps, and
"across sessions if needed" — i.e. it was written for **free-tier Kaggle spread over multiple weeks**, not
for the AMD MI300X grant (PLAN_V6). **Is that still the intent?** It determines what these new numbers
affect:
- If 124M stays **free-tier Kaggle**: the 71–302 GPU-h/battery (optimizer-step reading) is a **timeline**
  problem (2–10+ weeks at 30 h/week for one battery), not a line in the grant ask.
- If 124M now runs **on the AMD grant**: it's a real **budget** line — ~71–302 GPU-h for a *single*
  8-branch battery is a large fraction of the 500 GPU-h total, so the number of 124M sites/recipes has to
  be scoped deliberately (and the grad_accum=40 determinism caveat above resolved) before it's committed.
