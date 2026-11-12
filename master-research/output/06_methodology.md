# Phase 4 · Methodology upgrade — PRIMARY direction (+ Backup-2 detector side product)

> **v2 corrections (2026-10-22):**
> - **H2 is not identifiable at (0.9, 0.95):** the √β1 weight/momentum rate sits within 2.7% of v's β2 rate, so add the β1 × β2 cross (PLAN_V8 §6.3).
> - **§3.2 λ\* must use Bai's Ĥ (2/η).** Raw D⁻¹H has 38/η (Cohen 2207.14484).
> - **Matched 16,384 tokens per step** at every rung (PLAN_V8 §6.6).

*2026-10-22 · target ICML 2027 · scale plan (user deferred to us): realistic proxy (all experiments) → ~30M
mid rung (key branches) → 124M (primary contrast, n ≥ 3) → 350M one-instance check if hours allow. Everything
here is a plan; nothing has run. [V] verified in repo/source · [I] inferred · [A] assumed.*

---

## 1 · Formal problem statement

**Notation.**
- **Trajectory.** A training trajectory has state Θ_t = (w_t, m_t, v_t, k_t), where k_t is Adam's step
  counter. AdamW with (β1, β2, ε) runs on a fixed data stream with `get_batch` a pure function of the step
  [V].
- **Spike event.** E occupies a window W = [s0, s1).
- **B0.** The trajectory with E.
- **B\*.** The *data-aligned* counterfactual: same init, same stream, E removed. For induced spikes this
  means the hook omitted. For natural spikes, see §7 (open item A4).
- **Cut time.** Both branches are cut at t1 ≥ s1; they read identical data after t1.

**Damage.**
Δ_X(h) = L_val(X, t1+h) − L_val(B\*, t1+h), on a fixed held-out token set. The **primary horizon** is
end-of-schedule.

**Cross-world hybrids.**
Θ^{abc}_{t1} = (w^a, m^b, v^c, k) with a, b, c ∈ {0, \*}. The state at t1 is Markov for the future, so the
2³ hybrids decompose Δ by mediator. Möbius main effects and interactions are standard multi-mediator
estimands (Pearl 2001; VanderWeele 2014; see `02_adjacent_fields.md` T2). Bitwise determinism makes every
cross-world cell *runnable*, not just identifiable.

**Signed poison.**
- Per coordinate, σ_i = sign(log v_i^0 − log v_i^\*).
  - **Over-preconditioned** (σ = +): learning stalls on i.
  - **Under-preconditioned** (σ = −): Bai et al.'s instability mechanism.
- The **causal ground truth** is the set S\* whose patching recovers Δ (sufficiency) while patching its
  complement does not (necessity). It is found by group testing (§3.2).
- The localizer is an unsupervised predictor of S\*. This fixes the sign / basis / circularity objection
  (Skeptic F3).
- For v, edits are **support-limited** (coordinates, rows, heads, tensors), because v is diagonal. Only m
  admits rotated (rank-k) edits.

**Localization output contract** (fixes CLAUDE.md §7.2's open obligation):
```
Localization = {
  "groups":   {group_id: {"tensor": name, "rows": slice|None, "score": float, "sign": +1|-1,
                          "channel": "m"|"v"|"m+v"}},
  "selected": [group_id, ...],            # S, ordered by score
  "method":   str, "ref": "B*"|"vbar"|"fresh_g2", "t": int, "cost_gpu_s": float
}
```

**Repair operator.** R(Θ_t, Localization) → Θ'_t, applied once at fork start through the `intervention`
slot, or at a step through `pre_step` / `pre_opt_step` [V: `run_fork` slots, fork.py:29-83].

---

## 2 · Hypotheses and falsifiable predictions

All of these are pre-registered in `07_validation_plan.md` with refutation criteria. The numbers are
proposals for team sign-off.

| ID | Hypothesis | Prediction (proxy unless noted) | Refuted if |
|---|---|---|---|
| **H1** carrier | Part of a spike's lasting damage is carried by optimizer state | At t1 = s1+10, the (m, v) main-effect share of Δ(end) is ≥ 0.3 for `corrupt_batch` and `repeat_ngram` | The 95% CI upper bound is < 0.1 on held-out seeds |
| **H2** migration | The state-carried share decays with time since the spike, at a rate set by β2 | share(τ) ≈ a·β2^τ + b; ln2/(−ln β2) ≈ 13.5 steps at β2 = 0.95 | The share is flat or rising over τ ∈ {1, 10, 30, 60, 150} |
| **H3** transplant | Poisoned state alone causes damage on clean weights | B\* with B0's v at t1 has Δ > τ in ≥ 80% of instances, rising monotonically in κ (dose-response) | < 50% of instances, or non-monotone |
| **H4** QK-Norm | QK-Norm reduces spike incidence, and the state-carried share of surviving spikes is reported | Incidence drops; we report the share either way | (descriptive; no refutation; it sets the claim's scope) |
| **H5** localization | S\* is sparse, and a cheap signed predictor finds it | \|S\*\| ≤ 3 tensors or ≤ 5% of rows. Signed AtP-state + NIS reaches AUROC ≥ 0.85 vs S\* and ≥ 0.9 on the decoy benchmark, and beats the magnitude-only control by ≥ 0.10 AUROC on decoys | AUROC vs S\* < 0.7, or no gain over magnitude-only |
| **H6** undo asymmetry | The exact undo log fixes data-driven spikes, not LR-driven ones | ≥ 70% of oracle-patch recovery on `corrupt_batch` at lag ≤ 5; < 20% on `lr_bump` | Either inequality reverses |
| **H7** repair vs baselines | A selective repair is at least as good as the best non-oracle baseline while editing little | The best of {undo, certified restoration} is non-inferior (margin 0.005 nats) to the best non-oracle baseline picked on seed 0, closes ≥ 50% of gap G, and edits ≤ 10% of coordinates | Non-inferiority fails, or it edits > 25% |
| **H8** scale | H1's sign holds up the scale ladder | (m, v) share > 0 with its CI excluding 0 at the mid rung, and same sign at 124M (n ≥ 3, descriptive) | Sign flips at the mid rung |
| **H9** detector (side) | The lower tail of Adam's innovations gives lead time on decay-stage spikes | Lower-tail NIS CUSUM median lead ≥ 1 step at conformal run-level FP ≤ 0.05 on β2 = 0.999 presets | No lead at matched FP |

**Pivot insurance:** if H1 is refuted, the paper reports "spike damage is weight-carried beyond ~k steps",
with H2's curve and H6's in-window remedy. It does not drop the result.

---

## 3 · Algorithm

### 3.1 Pipeline diagram
```mermaid
graph LR
  T[trunk: warmup+cosine<br/>ring snapshots + telemetry] --> F[fork point ≤ s0]
  F --> B0[B0: spike on]
  F --> BS[B*: spike off, data-aligned]
  B0 --> C{cut t1}
  BS --> C
  C --> H[2³ hybrids w/m/v<br/>+ feasibility check<br/>+ placebo replicates]
  H --> D[damage-migration curve<br/>Möbius effects]
  C --> O[oracle S* via<br/>adaptive group testing]
  C --> L[signed localizer:<br/>AtP-state + NIS sign]
  O --> SC[score L vs S*<br/>AUROC, repair-AUC]
  L --> R[repair: undo log /<br/>certified restoration]
  R --> E[Δ(h), gap G, edit %,<br/>cost vs baselines]
```

### 3.2 Pseudocode (the new components only; harness untouched)
```
# --- measurement -------------------------------------------------------------
def factorial(snapB0_t1, snapBs_t1):                  # both captured at the same t1, same step/adam_step
    for a,b,c in {0,*}^3:
        Θ = hybrid(w=snap[a].w, m=snap[b].m, v=snap[c].v, k=snap0.k)
        viol[abc] = frac(|Θ.m| > C(β1,β2)·sqrt(Θ.v) + tol)  # C = (1-β1)/sqrt(1-β2) / sqrt(1-β1²/β2) ≈ 1.165
        Δ[abc](h) = val_loss(run_fork(Θ, horizon)) - val_loss(B*)
    return möbius(Δ), viol

# --- causal oracle (noising direction = OR model) -----------------------------
def oracle(groups, snap0, snapS):                       # Hwang / binary splitting; ddmin fallback
    def fires(T): return Δ(run_fork(patch(snapS, from_=snap0, groups=T))) > τ_f   # τ_f ≥ 5×placebo spread
    return group_test(groups, fires)                    # ≤ ⌈log2 C(G,s)⌉ + s - 1 forks

# --- signed localizer ---------------------------------------------------------
def localize(snap0, ref):                               # ref = v̄ (deployable) or B* (analysis)
    g_val = grad of L_val at w' = adamw_step(snap0)     # one backward, "ghost update"
    score_v[i] = g_val[i] · ∂w'/∂log v_i · (log ref.v_i - log snap0.v_i)   # AtP in log-v
    score_m[i] = g_val[i] · ∂w'/∂m_i · (ref.m_i - snap0.m_i)
    sign[i]    = sign(log snap0.v_i - log ref.v_i)     # cross-checked against two-sided NIS tail
    return group_and_rank(score_v, score_m, sign)

# --- repairs ------------------------------------------------------------------
def undo(snap_t1, grads_W, vbar):                       # exact direct-path removal; w untouched
    m -= Σ_{s∈W} β1^{t1-s}(1-β1) g_s
    v -= Σ_{s∈W} β2^{t1-s}(1-β2) (g_s² - vbar)          # replace the spike's g² by the clean expectation
    v = max(v, 0)

def certified_restore(snap, S, target, λ*):           # signed, support-limited
    for α in bisect([0,1]):                             # largest α with λ_grad(H̃(α)) ≤ λ* = (1-δ)·2/η·(bc)
        v_S(α) = v_S^(1-α) ⊙ target_S^α                 # target = v̄ or fresh mean g² (K held-out batches)
    return snap with v_S(α*)                            # certificate: logged λ_grad before/after
```

---

## 4 · Alternatives considered

| Alternative | Decision | Why |
|---|---|---|
| Exhaustive per-tensor patching (A7) | Kept, at proxy only | 42 forks per instance is fine at 11M and too costly at 124M |
| Non-adaptive compressed sensing (A9) | Kept as fallback | Use it when group testing's OR model breaks (interactions) |
| Shapley over groups (A10) | Rejected | Fork cost |
| Distributed alignment search (A16) | Rejected | 40 p-h, plus interpretability-illusion risk |
| Learned probe (A15) | Rejected | Circular from planted to natural |
| Lanczos-only localizer (A4) | Kept as a variant | Fills V7's "3 localizer variants": AtP+NIS, λ-decomposition, magnitude control |
| RESEARCH §5 uniform minimal stabilizing scaling | Subsumed | It is the α-path special case of certified restoration, and covers only one sign |
| LMI minimal diagonal repair (T4) | Stretch goal | Covers one sign only; needs a solver |
| m-projection off H̃ eigenvectors (B10) | Ablation only | m forgets within ~10 steps |
| Rollback-and-skip as *our* method | Rejected | It shifts the data order, so it isn't B\*. Kept as a baseline |

---

## 5 · Baselines

**Strong and recent, each tuned on seed 0 only:**
- rollback-and-skip (PaLM: back ~100 steps, skip M batches; labelled data-shifting)
- **Adam-MR** (m + v + t reset)
- **bias-correct Bg**
- v-only reset; `Bvbar`
- OLMo-core skip (6σ / 128); `Bskipstep_det` (OPEN-1B rule)
- ZClip; AdaGC; SPAM
- **GradientStabilizer** (MIT)
- **StableAdamW** (reimplemented)

**Simple baselines that are easy to beat by accident:**
- B0 (no fix)
- clip-tighter `Bs` (clip below the trunk clip; fixes improvement A1)
- **`Br`**: a random edit matched in support *and* energy (Xie 2026's control)
- **magnitude-only localizer** (‖log v − log v̄‖) with the same repair
- Spectra-style energy-only damping
- **step-counter-only reset** (Adam-Rel control)

**Primary contrast:** best of our repairs vs best non-oracle baseline (chosen on seed 0), on gap closed
G = (Δ_B0 − Δ_X)/Δ_B0 at end of schedule.

## 6 · Ablations (one per claimed contribution)
- **Localizer:** AtP-only / NIS-only / both / λ-decomposition / magnitude-only.
- **Reference:** B\* vs v̄ vs fresh g².
- **Repair target:** v̄ vs fresh g²; with and without step-counter reset.
- **Support:** tensor / row / head.
- **Undo window W:** exact vs detector-timed vs ±2 steps.
- **Horizon** h ∈ {50, 150, 500, end}.
- **Architecture:** QK-Norm on / off.
- **Optimizer:** Muon (2-D) + AdamW (rest) control.
- **Embedding:** tied vs untied (wte artifact, Skeptic m14).

## 7 · Splits, leakage and contamination controls
- **Δ evaluation:** ≥ 1M held-out tokens from `val` batches **disjoint from every localizer probe batch**.
  `k2.PROBE` uses val batches 0–3 [V: audit], so Δ uses val batches ≥ 1000. The set is fixed once and
  identical for every branch.
- **Seed 0 is for tuning only** (detector thresholds, τ_f, baseline hyperparameters, primary-baseline
  choice). Seeds 1–9 are held out.
- **Data-order seeds:** a permutation key on `get_batch`'s step index. B\* alignment holds within a key. This
  breaks the "2 data contexts × 5 inits" pseudo-replication (Skeptic M7).
- **Natural-spike B\*:** pre-register one of two options.
  - (a) Aligned clean replacement of the offending examples found by data-side attribution (A17).
  - (b) Exclude natural spikes from the causal contrasts and use them only for localization and detection.
  - Recommended: (b) for ICML, (a) as an exploratory appendix.
- **Decoys and planted targets** are drawn from a seed not used for anything else.

## 8 · Metrics
- **Damage:** Δ(h) curves and end-of-schedule Δ; gap closed G; % of oracle recovery; recovery time
  (Kaplan–Meier).
- **Localization:** AUROC, AUPRC and P@k vs S\* and vs planted truth (with decoys); repair-top-k AUC.
- **Edit size and cost:** fraction of coordinates edited, plus Pareto curves; cost in GPU-seconds and
  recomputed-step equivalents vs rollback.
- **Validity checks:** hybrid feasibility violation rate; placebo spread.
- **Detector (side):** run-level false-alarm rate (calibration), lead time, recall.
- Everything is reported **per spike type and per scale**, never pooled.

## 9 · Statistics
- **Unit:** the spike instance. Data-order seed × init seed is the cluster.
- **Model:** Δ ~ branch + site + (1 | seed) mixed model, alongside paired bootstrap CIs (scipy, already pinned).
- **Primary test:** one pre-registered primary contrast: TOST non-inferiority (margin 0.005 nats), then
  superiority. Holm across the H1–H9 secondary tests; BH for exploratory ablations.
- **Power:** a pilot of 4 instances estimates the SD of paired Δ, then n is locked. At an SD of 0.007 nats
  and 80% power, n ≈ 10 per type detects 0.01 nats (Skeptic M7 arithmetic, unverified in software) [I].
- **Placebo:** K = 5 replicates per branch with a 1-ulp perturbation. A repair effect must exceed the
  placebo spread.

## 10 · Reproducibility
- **Stamps:** every JSON records git SHA, cfg, argv, and the torch / ROCm / hipBLASLt / AOTriton versions
  (improvement B4).
- **Environment:** one pinned ROCm image or wheel.
- **Determinism:** noop-vs-noop is checked *across processes through each spike window* (M12).
- **Artifacts:** snapshots go to the HF Hub; configs live in the repo; code is released with the paper
  (SpikeState).

## 11 · Threats to validity and mitigations

| Type | Threat | Mitigation |
|---|---|---|
| Internal | Non-bitwise ROCm drift | Cross-process gate through spike windows; the tolerance path in V7 §4.4 |
| Internal | Off-manifold hybrids | \|m\| ≤ C√v violation counts; off-manifold control (B\* state at ±5-step offset) |
| Internal | Chaotic sensitivity near the edge of stability | Placebo replicates |
| Construct | "Poison" is circular | Causal oracle S\*; signed definition; H̃ computed from both v̂_post and v̄ |
| Construct | Held-out loss ≠ downstream quality | State it. At 11M, downstream tasks are meaningless |
| External | Scale | Realistic proxy, mid rung, 124M n ≥ 3, optional 350M, OPEN-1B offline if its m/v exist |
| External | Induced ≠ natural spikes | Classify every instance's failure mode; report per type; natural spikes for localization |
| External | No QK-Norm | Ablation H4 |
| External | AdamW-only | Muon control |
| Statistical | Pseudo-replication; multiplicity | Data-order seeds; mixed model; Holm/BH |

## 12 · Ethics and limitations
- No human data, and no dual-use concern beyond general training efficiency. Compute is ~15–100 MI300X-h
  (disclosed).
- **Limitations to state plainly:**
  - small scale;
  - spikes are mostly induced;
  - AdamW-centric;
  - held-out loss is the only quality metric;
  - the claim is "damage *carried* in optimizer state", not "optimizer state is the root cause" (Marin,
    Tang 2026, Li et al. 2026).
