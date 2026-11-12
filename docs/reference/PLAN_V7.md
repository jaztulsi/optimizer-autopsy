# OPTIMIZER AUTOPSY — PLAN V7
### Current source of truth · consolidated 2026-10-07 · V6 + every decision locked since the V6.1 audit

> **Read this first, every session.** V7 is the plan the team works from. It does not repeat V6's long-form
> material — the math (V6 §3), the C1 rationale (V6 §2), Path B (V6 §4), the pre-registration protocol
> (V6 §8), the cut order (V6 §6) and the honest-scoring note (V6 §10) all still hold. **Where V7 and V6
> disagree, V7 wins.** If a question is not answered here or in V6, ask before inventing an answer.
> Deep research and the new ideas behind §6–§7: `research/spikes/RESEARCH.md`.

---

## 1 · Core values (non-negotiable)

1. **No causal claim without replay.** Do-nothing vs do-nothing must read exactly zero — or, on AMD only,
   pass the pre-registered tolerance in §4.4. Lose this and every result is noise.
2. **Clean counterfactual + held-out loss.** `B*` keeps the data stream aligned (never "skip to the next
   batch"); every effect is `Δ = val_loss(branch) − val_loss(B*)` on held-out data, paired by seed.
3. **Pre-register before you look.** Thresholds, sample sizes, sites and decision rules are written down
   before the data that would test them exists. Never move a goalpost after seeing a result.
4. **Verify your own work — try to break it first.** A CPU check is never reported as GPU-verified. Report
   misses, negative results and limitations plainly.
5. **Compute discipline.** No training on laptops. **No Kaggle runs now** — all GPU work waits for the Exea
   Labs AMD MI300X allocation. Code is written and CPU-checked ahead of time so GPU hours go to science.
6. **Honest scoring.** No inflation (V6 §10). The realistic ceiling is a strong TMLR-shaped paper, a NeurIPS
   D&B/workshop candidate; top-tier acceptance is not assumed.
7. **Free tools and respected licenses.** SPAM has no license → reimplement from scratch (testing against
   its code is fine, copying is not). AdaGC → reimplement (PaddlePaddle). ZClip is Apache-2.0 (wrappable).
8. **Load-bearing code is guarded.** `research/harness/determinism.py` and `preflight.py` are not modified
   without flagging; non-harness roles also leave `harness/fork.py` alone.
9. **Team hygiene.** Everyone commits under their own identity, through PRs. No AI attribution in commits.
   Dev-only packages go in the root `requirements-dev.txt`, never `research/requirements.txt`.
10. **Small verifiable steps.** Every unit of work ships with a check that fails if the logic breaks.

---

## 2 · The project in one paragraph

LLM pretraining loss spikes poison Adam's optimizer state (`m`, `v`), not just the weights, and the damage
may persist after the weights recover. **C1** (built): a bitwise-deterministic fork-and-replay harness —
freeze a training moment, relaunch identical branches that differ only by a named intervention. **C2**
(unbuilt): a localizer that finds which directions of `m`/`v` are poisoned (gradient-directional SNR +
curvature of the preconditioned Hessian `H~ = D^-1/2 H D^-1/2`, `D = diag(√v̂+ε)`). **C3** (unbuilt): a
rank-limited repair that fixes only those directions and, via C1, must beat both doing nothing and a full
reset. Closest prior work (2026): Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam*
(ICML 2026) explains spikes through `v` decoupling from `g²` but neither localizes nor repairs; SPAM resets
moments globally and periodically. Positioning detail: `research/spikes/RESEARCH.md` §1.

---

## 3 · Verified state on 2026-10-07

| Piece | State | Evidence |
|---|---|---|
| C1 determinism + fork `Δ==0` gate | ✅ CUDA-verified (T4 proxy; P100 124M at grad_accum=40); **AMD untested** | `results/task4_*.json`, `task7_fork_gate.json`, `research/kaggle/step_timer_results.md` |
| Spike recipes | `lr_bump` predictive (lead 2) + `corrupt_batch` onset (lead 0) — both **provisional** (V6 regrade of old T4 runs, no fresh prospective run). `precision` ratio 1.08 (no spike); `tiny_eps` inert at proxy scale | `results/task8_spike_detector.json`, `task8_v6_policy_regrade.json` |
| Measured speed (P100, never AMD) | proxy 0.0619 s/step (batch 16); 124M 1.08 s/step (batch 8); ≈64–68 s per 124M optimizer step at grad_accum=40 (upper bound) | `step_timer_results.md` |
| Baselines | skip, clip, full m+v reset, plus Eddie's SPAM/ZClip/AdaGC, skip-step (oracle + detector-timed), v-only reset, `v ← v̄` (Bvbar) — **merged PR #14, 2026-10-12**, CI-tested, never run on GPU | `research/baselines/` |
| Kill-test (Task 9) | built + logic-tested, never run on real data | `research/harness/fork.py`, `research/tests/test_kill_test.py` |
| Localizer (C2), repair (C3), theory | ⬜ unbuilt | `research/localizer/`, `repair/`, `theory/` |

---

## 4 · Locked decisions (since V6.1)

### 4.1 Compute
- **Ask now: 15 MI300X GPU-hours** (drafted 2026-10-07; the verbal agreement with Avneh on 2026-09-01 was
  10 h for the preprint). Avneh said on 2026-10-07 the GPU should arrive around the end of that week.
- **Later: ~85 more (~100 total)** for 124M + 350M work, requested after the preprint with measured MI300X
  speeds attached.
- Why 15: proxy science (all baselines, our variants, 2 extra optimizers, natural-spike search) =
  0.66–2.25M steps × 0.062 s ÷ (10–25× MI300X speedup) ≈ 0.6–4.5 h; ROCm bring-up 3–5 h; setup 1 h; 124M
  smoke 0.5 h; +30% reruns → 7–14 h.
- Market value (Oct 2026 median $2.90/GPU-h, range $1.71–6.00): 15 h ≈ $44; 100 h ≈ $290. Free to us.
- **First GPU job on AMD: `step_timer.py`** — every MI300X number above is an estimate until then.

### 4.2 Spike design (per intervention) — pre-registered
- **30 instances per intervention** = 3 spike types × 2 sites × 5 seeds, each paired with its own `B*`.
  n = 10 per type; the minimum detectable effect is reported, not hidden.
- **Types:** `lr_bump` (predictive), `corrupt_batch` (onset, never called early warning), **natural**
  (from the stress search, §6). `repeat_ngram` and `v_inflate` are additional candidates — reported if
  they qualify, never swapped in after the fact.
- **Sites:** steps **160 and 480** on a **600-step** proxy trunk (both after the initial descent). May be
  revised only before any battery data exist.
- **Seeds:** 0–4. Seed 0 tunes detector thresholds; seeds 1–4 are held out. Note: seeds change the
  initialization only — `get_batch` is a pure function of the step, so data order is identical across seeds.
- **Natural instances:** the first qualifying natural spike of each stress run, runs taken in preset order
  (`natural.STRESS`) then seed order, until 10 are collected (no cherry-picking; later entries are extra
  localizer data, never substitutes). Qualification rule: `research/spikes/natural.py::find_spikes` defaults.
- If hours run short: drop to 3 seeds and say so in the paper (V6 cut order still applies).

### 4.3 Branch menu (every spike instance)
`B0` no fix · `B*` clean · **skip-step** (skip the optimizer update — new; the current `skip.py` swaps in the
clean batch, which does nothing for `lr_bump`) · `Bs` skip+clip · **v-only reset** (new) · `Bg` full m+v
reset · ZClip · AdaGC · SPAM · our repair (3 localizer variants) · `Br` random-subspace control.
- **ZClip/AdaGC warm-up (approved in principle, 2026-10-07):** shadow mode — watch only pre-fork steps,
  never the spike; change nothing in training; **feed each method's *clipped* norm into its running average,
  exactly as the real method does** (ZClip code: "Update EMA with the effective norm"; AdaGC paper: clipped
  norms, λ_rel=1.04, β=0.99, global clipping for the first 100 steps); mirror each method's own warm-up;
  save the state with the snapshot; identical treatment for both. Remaining limitation (stated in the
  paper): real AdaGC's frequent clipping would also have bent the pre-spike trajectory.

### 4.4 ROCm tolerance — pre-registered
Aim for exact zero. If ROCm cannot reach it, a difference counts as real only if it exceeds **both**
1e-3 nats of held-out validation loss **and** 5× the largest noop-vs-noop drift measured over 5 pairs.
Claimed effects must be ≥ 0.01 nats.

### 4.5 Natural spikes
Proxy-scale stress search for the preprint (§6). A 124M natural spike is phase 2. Public LLM360 K2-V2
training logs and the K2-Spike-1/2 paired checkpoints are used for detector/weight-delta validation only —
a 65B model is never forked.

### 4.6 Team
Spike engineer: jaztulsi. Baselines: Eddie (Eddiegah). Repo access: Eddiegah, Orange-neon, tanishp3280 (write);
avyankvanamali-star, aryakumarn (invited). Other roles: fill in — `CONTRIBUTORS.md` updated 2026-10-12;
Avyank's invite still unaccepted.

---

## 5 · Open issues (owner in brackets)

- **Kill-test fork point [harness owner].** `make_branches` is documented as forking from the "pre-spike
  (t0)" snapshot, but a detector `t0` is always ≥ `inject_step`, so a t0 snapshot already contains the
  injection and the branches cannot replay it. Pass the K2 manifest's `fork_snapshot` (step ≤
  `inject_step`) instead — no harness code change needed, only the right input.
- **Baselines PR #14 [Eddie] — merged 2026-10-12** (both must-fixes in; Bskipstep_det + Bvbar added).
  ZClip/AdaGC observers now wired into `capture.run_one`: fork/ring snapshots carry
  `meta["baseline_priming"]`, ready for `make_baseline_branches(..., **meta["baseline_priming"])`.
  Bvbar takes `load_file(entry.vbar_uri)`. Untested until the AMD GPU.
- **Localization output contract [C2/C3 owners]** — see `CLAUDE.md` §7.2.

---

## 6 · Spike-engineer workplan

**Written ahead of the GPU — not yet executed anywhere.** Tests are in `research/tests/test_spike_pipeline.py`
and run in GitHub CI once pushed (never on a personal machine); the GPU paths are checked by
`capture --selfcheck` on AMD day 1:
- `spikes/tune_detector.py` — `OnlineDetector`: the streaming twin of `trigger_mask` (one implementation)
  that also reports the trigger statistic at `t0`.
- `spikes/induce.py` — two new recipes: `repeat_ngram` (OLMo-documented natural cause class: repeated junk
  n-grams in real batches) and `v_inflate` (planted `v` poison on named tensors — exact ground truth for C2).
- `spikes/capture.py` — the capture driver: spike source → ring buffer of snapshots → detector → saves the
  `t0` snapshot, the fork snapshot (step ≤ `inject_step`), matched normal snapshots, the pre-spike `v̄`,
  per-layer telemetry, and one K2 manifest entry per qualifying trigger. GPU-only; one process per run.
- `spikes/natural.py` — pre-registered stress presets, offline natural-spike finder, cross-seed alignment
  test (data-driven vs dynamics-driven spikes), and the natural-search driver.
- `spikes/k2.py` — manifest save/load/validate/build.

**AMD day-1 queue (spike role), in order:**
1. `python -m research.spikes.capture --selfcheck` (seconds) — first real exercise of the driver.
2. `python -m research.kaggle.spike_run` — fresh prospective confirmation of the recipes (+ `repeat_ngram`).
3. `python -m research.kaggle.corrupt_batch_probe` — corrupt_batch lead question, pre-declared rule.
4. `python -m research.spikes.natural --search` — stress grid → natural spike population.
5. `python -m research.spikes.capture --battery` — the 30 instances → `results/k2_manifest.json`.

---

## 7 · Research directions adopted (detail + math in `research/spikes/RESEARCH.md`)

1. **Optimizer-state patching** — transplant `m`/`v` for a subset `S` from `B*` into `B0` at the same step;
   necessity (patch all but `S`) and sufficiency (patch only `S`). The optimizer-state analogue of activation
   patching; gives C2 an oracle target on induced spikes.
2. **Planted-poison benchmark** — `v_inflate` with exact ground truth, plus a falsifiable recovery-time
   prediction: half speed regained after `k½ = ln(κ/3)/(−ln β2)` steps (≈158 for κ=1e4, β2=0.95).
3. **Seedable natural spikes** — literature-backed stress presets give a natural-spike *population*, not
   one case study; cross-seed alignment separates data-driven from optimizer-driven spikes.
4. **Mechanism-grounded detection** — per-layer update-RMS (Wortsman 2023), `v̄`-drift, and the stability
   margin `η·λ_grad(Ĥ)/2` (Bai et al. 2026), logged as telemetry now and evaluated offline later.
5. **Distribution-free false-positive control** — conformal run-level thresholds and CUSUM/e-detectors in
   place of grid-swept z-scores.
6. **Minimal stabilizing repair (C3)** — the smallest change to `v` on the localized subspace that brings
   preconditioned sharpness back under the stability threshold, instead of a reset.

---

## 8 · Change log

- **V7 (2026-10-07):** consolidates the 2026-09-01 → 2026-10-07 decisions: 15 h compute ask (≈100 total),
  30-instance spike design with sites/seeds, expanded branch menu, ROCm tolerance, natural-spike strategy,
  ZClip/AdaGC shadow warm-up rule, team commit rule, the kill-test fork-point issue, and the spike-engineer
  build + research directions. V6 stays as the long-form reference.
