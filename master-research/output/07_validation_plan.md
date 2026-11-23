# Phase 5 · Validation plan (2026-10-22)

*Every number below is a **proposal** for team sign-off. It must be copied into a PLAN_V7 §4.7 "Decision
thresholds" section and committed **before** any AMD data exists (core value 3; improvement A3).*

---

## 1 · Pre-registration table

| ID | Endpoint (exact) | Expected | Refuting result | Test |
|---|---|---|---|---|
| H1 | (m,v) main-effect share of Δ(end) at t1 = s1+10, per type | ≥ 0.3 for `corrupt_batch` and `repeat_ngram` | 95% CI upper bound < 0.1 | paired bootstrap over instances; mixed model |
| H2 | share(τ) for τ ∈ {1,10,30,60,150} | decays; fit a·β2^τ + b with a > 0 | a ≤ 0 (CI) | NLS fit with bootstrap CI |
| H3 | Δ(B\* with B0's v), κ dose ∈ {1, 10², 10⁴} | Δ > τ in ≥ 80% of instances; monotone in κ | < 50%, or Spearman ρ(κ, Δ) ≤ 0 | binomial + Spearman |
| H4 | spike incidence and H1 share, QK-Norm on vs off | incidence ↓ (descriptive) | — | reported |
| H5a | \|S\*\| from group testing | ≤ 3 tensors or ≤ 5% of rows in ≥ 70% of instances | > 6 tensors median | descriptive with CI |
| H5b | AUROC(localizer, S\*) and AUROC(localizer, planted + decoys) | ≥ 0.85 / ≥ 0.9; magnitude-only + 0.10 on decoys | < 0.7, or Δ AUROC vs magnitude-only CI ∋ 0 | DeLong / bootstrap |
| H6 | undo recovery as a fraction of oracle-patch recovery | ≥ 0.7 on `corrupt_batch` at lag ≤ 5; < 0.2 on `lr_bump` | either inequality reversed | paired bootstrap |
| **H7 (PRIMARY)** | G(best ours) − G(best non-oracle baseline, chosen on seed 0) at end of schedule | non-inferior at margin 0.005 nats; G ≥ 0.5 with ≤ 10% of coordinates edited | TOST fails | TOST, then a superiority t-test; Holm for H1–H6 and H8–H9 |
| H8 | H1 share at the mid rung and 124M | > 0, CI excludes 0 (mid); same sign at 124M (n ≥ 3, descriptive) | sign flip at the mid rung | as H1 |
| H9 | lower-tail NIS CUSUM lead time at conformal run-level FP ≤ 0.05 | median lead ≥ 1 step on β2 = 0.999 presets | lead ≤ 0 | Wilcoxon on held-out seeds |

**Fixed in advance:**
- **τ (kill-test margin):** 0.01 nats, the V7 §4.4 claimed-effect floor.
- **τ_f (oracle fire threshold):** max(0.005 nats, 5 × the placebo spread).
- **n_min:** 10 instances per type for the primary contrast; locked after the 4-instance SD pilot.
- **Horizons:** {50, 150, 500, end}.
- **Primary horizon:** end of schedule.
- **Seeds:** seed 0 tunes; seeds 1–9 are held out.
- **Natural-spike B\*:** excluded from causal contrasts (option b).
- **`Bs` clip:** 0.5 × the trunk's `grad_clip` (fixes A1). This is a proposal; Marin used 1.0 → 0.2.

---

## 2 · Minimal viable experiment (≤ 1 day of MI300X, week 2)

**Goal:** before any localizer or repair code, test whether the headline exists (H1/H2), check the harness
(planted sanity), and try the cheapest new repair (H6).

**Setup:**
- **Model and schedule:** the current proxy config with one addition, a warmup+cosine schedule applied as
  a `pre_step`. `lr_bump` must be changed to *multiply* the scheduled LR: today it sets an absolute
  `base × factor`, which would overwrite any schedule [V: induce.py:52-60].
- **Trunk:** 1,500 steps. Fork at step 160.
- **Instances:**
  - `corrupt_batch` (width 5)
  - `lr_bump` (10×, width 5)
  - `v_inflate` (κ = 10⁴ on 2 tensors) plus 2 decoys
  - 4 held-out seeds each, so 12 instances
- **Per instance:**
  - B0 and B\* to t1 ∈ {s1+1, s1+10, s1+60}
  - 6 new hybrid forks per t1, to horizon 300 steps
  - 3 undo forks
  - 5 placebo replicates of B0 and B\* at one t1
- **Forks per instance:** about 18 + 3 + 10 = 31, so about 370 forks in total.
- **Cost:** about 0.1–0.3 GPU-min per fork [A, Method Hunter cost model], so ≈ **1–2 GPU-h**, plus the
  determinism gate.

**Outputs:**
- the damage-migration table per type;
- the v_inflate sanity result;
- undo vs patch;
- the placebo spread;
- the hybrid-violation rate;
- one JSON per instance, stamped with SHA, cfg, argv and versions.

---

## 3 · Go / no-go gates

| Gate | When | Criterion | If it fails |
|---|---|---|---|
| **G0** determinism | week 1 | noop-vs-noop Δ = 0 across two processes through a spike window (or the §4.4 tolerance, measured *in* spike windows) | ROCm checklist (improvement A6). If still failing: tolerance-only claims, and BACKUP 2 becomes the lead |
| **G1** harness sanity | MVE | `v_inflate`: the v main effect ≥ 0.8 of Δ at t1 = s1+1; hybrid violations ≈ 0 for planted cells | **stop**: harness or hybrid bug |
| **G2** headline | MVE | H1 share CI excludes 0 for ≥ 1 data-driven type at t1 = s1+10 | **pivot** to the negative-result framing (§2 insurance) and keep going |
| **G3** placebo | MVE | placebo spread ≤ ½ of the median \|Δ_B0\| | raise K; lengthen the horizon; report only effects above the placebo |
| **G4** localizer | week 5 | AUROC vs S\* ≥ 0.7 on held-out seeds | switch the lead localizer to λ-decomposition (BACKUP 1) |
| **G5** repair | week 7 | H7 non-inferiority holds on the proxy | report a measurement-led paper; repairs become analysis |
| **G6** scale | week 8 | mid-rung H1 sign matches the proxy | narrow the claim to small scale and say so |

---

## 4 · Compute and time budget

Estimates are [A] until `step_timer` runs; GPU-h are MI300X hours.

| Block | GPU-h | Person-h |
|---|---|---|
| G0 bring-up + `step_timer` + cross-process gate | 3–5 | 8 |
| MVE | 1–2 | 20 |
| Realistic proxy + mid-rung trunks, QK-Norm variant, stress re-search | 6–10 | 15 |
| Localization (oracle + AtP + NIS + λ) on proxy + mid | 3–5 | 45 |
| Repairs + strong baselines, full battery on proxy + mid | 6–10 | 40 |
| 124M primary contrast, n ≥ 3 (n = 5 if hours allow) | 25–45 | 15 |
| 350M one-instance check (optional) | 15–25 | 8 |
| Detector side product (CPU on telemetry) | 0–1 | 15 |
| **Total** | **~60–100** | **~165** |

This fits the ~100 h total in V7 §4.1, which means **asking Avneh for the follow-on hours in week 1** with the
`step_timer` numbers attached. The user confirmed GPU access isn't the constraint.

---

## 5 · Reviewer simulation (re-scoring the revised plan)

**Round 0** is the Skeptic's review of the original plan (`02_skeptic_review.md`): weak reject, 3/6.
**Round 1** scores the plan in `06_methodology.md`. Each reviewer gives a 1–10 score with its top remaining
objections.

### R1 — Methods

- **Score:** 4 → **7**.
- **Resolved:**
  - F2: the thesis is now the primary endpoint (H1/H2).
  - F3: signed, causal ground truth; support-limited v.
  - M5: strong baselines plus best-baseline as the primary contrast.
  - M7: data-order seeds, mixed model, TOST, pilot SD.
  - M8: placebo replicates.
  - Hybrid feasibility.
- **Remaining objections:**
  1. The τ_f threshold for group testing interacts with the placebo spread. Could the oracle be unstable?
     Revision: report oracle stability across 2 placebo seeds, and use SuSiE credible sets when unstable.
  2. Cross-world hybrids cut at t1 assume the Markov property. It holds exactly only if the data stream and
     RNG after t1 are identical. Revision: assert that `rng_state` is identical across both branches at t1
     (dropout 0 makes it trivial), and log it.
  3. Non-inferiority margin 0.005 nats against a placebo spread that is still unknown. Revision: G3 makes
     the margin ≥ 2 × the placebo spread.

### R2 — Novelty

- **Score:** 6 → **7**.
- **Resolved:**
  - M6: the instrument is claimed, not the determinism.
  - Xu 2026, Liu & Li, Chimera, Xie 2026, Megatron and OPEN-1B are cited and contrasted in a
    related-work table.
  - Bai's predictor and per-block v-decay serve as the baseline localizer.
- **Remaining objections:**
  1. "Cross-world patching of optimizer state" is Xu's method. Revision: the claimed contribution is the
     *finding* (where spike damage lives and how fast it migrates) plus the *signed* localization and the
     undo log. Patching is described as an instrument, with Xu cited first.
  2. Is the undo log trivial once stated? Revision: show where it fails (H6, the asymmetry with `lr_bump`).
     That failure is the insight: data-driven damage sits in state, LR-driven damage sits in weights.
  3. Read Xu and Xie in full text before writing (they are abstract-only today).

### R3 — Significance and writing

- **Score:** 3 → **6**.
- **Resolved:**
  - F1, partly: warmup+cosine, ≥ 20 TPP of non-embedding parameters, a mid rung, 124M n ≥ 3, an optional
    350M check.
  - M4: QK-Norm ablation.
  - M11: end-of-schedule endpoint.
  - m15: three contributions, with detection and the benchmark moved to the appendix.
- **Remaining objections:**
  1. The largest *full-battery* scale is still about 30M. Revision: make the 124M primary contrast n = 5
     if hours allow, and add OPEN-1B offline localization if its m/v are real (verify in week 1).
  2. Practitioner value: will anyone use the undo log? Revision: measure overhead (memory per window
     step) and recovery per GPU-second vs rollback (idea 28). Release a ~100-line wrapper (idea 21) as
     an artifact.
  3. The title must not overclaim the cause. Draft title: *"Where Does a Loss Spike Live? Causal
     Autopsy and Surgical Repair of Adam's State"*.

### Devil's advocate (strongest counter-argument)

- **The argument:** "If β2 = 0.95 forgets in about 100 steps, then by the end of schedule *any* state
  effect is mediated through weights. Your end-of-schedule Δ shares will be ≈ 0 by construction, and the
  paper argues itself out of its own thesis."
- **Rebuttal:** that is exactly H2's prediction, and it is informative either way. The share at the
  **cut time** is measured on the end-of-schedule outcome, so "state at t1 caused end damage" is a valid
  causal claim even when the state itself has since forgotten. Damage done *through* the state in the
  window persists in the weights. The undo log intervenes inside the window, which is why it can help.
- **Ruling:** 4/5. It survives, provided the paper words "carried at t1" precisely and shows the
  natural-decay curve.

### Round 1 verdict

| Score | Before | After |
|---|---|---|
| Soundness | 4 | 7 |
| Significance | 3 | 6 |
| Novelty | 6 | 7 |
| Clarity | 5 | 7 |
| ICML form (overall, of 6) | 3/6 | about **4–5/6** |

The 4–5/6 is borderline to weak accept, conditional on G0–G2 passing. The remaining ceiling is scale.
