# 11 · Skeptical review + creative collaboration — Optimizer Autopsy plan (2026-10-22, v1)

> **v2 (2026-10-22):** §2's "genuinely new" item 3 (the exact undo log) is **PARTIAL**: SWIFT owns one-step Adam inversion. The wave-2 extensions are now in `PLAN_V8.md` Appendix A, and Marin is a counter-case (`12_factcheck_citations.md` C24).

**Basis:** PLAN_V7 and RESEARCH.md, plus the code audit (`improvement.md`), the Phase 2 agents (lineage,
skeptic, verifier, methods, adjacent fields), the novelty engine (`04`, `05`) and the analysis in `09`.

**Limits assumed:**
- ICML 2027, deadline about 22–28 Jan 2027 (unofficial; the CFP isn't out);
- about 13 weeks;
- about 60–100 MI300X-h, user-confirmed available;
- a small team with a bus factor of 1 on the spike role.

**Status:** nothing has run on AMD yet, so every result below is a prediction, not a finding.

**Labels:**
- **[V]** verified in code or a primary source
- **[I]** my inference
- **[UNV]** unconfirmed

**v1.** The wave-2 research (finance/filtering transfers, security, μP/scale theory, radical ideas) is still
running and will be folded into §3 when it lands.

---

## 1 · Hidden gaps, ranked by how much each could invalidate the results

| # | Gap | Type | Why it could invalidate | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | **The kill-test is rigged before any data.** `Bs` = skip + clip, with clip = 1.0 = the trunk's clip, so the clip half is a no-op. For `lr_bump`, skip is a no-op, so Bs ≡ B0. For `corrupt_batch`, Bs ≡ B\*. | missing control | Gate B decides whether C2/C3 get built at all. As written it returns PROCEED on `lr_bump` for structural reasons | [V] fork.py:196, config.yaml:18, test_kill_test.py:128 | Pre-register a `Bs` clip below the trunk clip, or make Gate B "best cheap fix" (skip-step, Adam-MR, rollback) |
| 2 | **Training loss used where held-out loss is required.** `_summarize` returns the last training loss on one batch of 16. Nothing computes branch val loss | weak evidence | Every Δ in the battery would measure fit to the training stream, not damage | [V] fork.py:306-310; eval.py is a stub | `val_loss` on ≥ 1M held-out tokens, at a pre-registered horizon |
| 3 | **The central thesis is never directly tested.** No w-vs-(m, v) decomposition. At β2 = 0.95, v forgets a spike in ~100–200 steps | untested claim | If lasting damage is weight-carried, "repair the optimizer state" has no target | [I] `09` §1; [V] Skeptic F2 | 2³ cross-world factorial at several cut times, as the primary endpoint |
| 4 | **"Poisoned direction" is not well defined.** v can be wrong in either direction (too large → stall; decayed → instability, Bai). v is diagonal, so "rank-k repair of v" can't be stored. The H̃ basis is built from the poisoned v | unstated assumption | Localization metrics have no ground truth; the repair could have the wrong sign (`v ← v̄` may re-create Bai's trigger) | [V] Bai 2026; [I] Skeptic F3 | Signed poison; causal ground truth (the patching oracle); "support-limited" wording for v |
| 5 | **AMD determinism is unproven.** On ROCm, `use_deterministic_algorithms` doesn't reach hipBLASLt; in torch 2.10 the AOTriton attention backward ignores `deterministic`; the §4.4 tolerance would be calibrated on calm steps | internal validity | Every causal Δ assumes noop-vs-noop = 0 | [V] torch 2.10 source; [V/UNV] ROCm checklist | Cross-process gate *through spike windows*; math-SDPA in forks or torch ≥ 2.13; tolerance measured in spike windows |
| 6 | **The proxy is unlike LLM pretraining.** 88% of parameters are embedding; < 1 token per parameter; constant LR; sites in the first few hundred steps | external validity | "A 3-layer toy at step 160" | [V] config.yaml (11M, 3L/192, constant 3e-4) | Warmup+cosine trunk at ≥ 20 tokens per non-embedding param; a mid rung; 124M primary contrast n ≥ 3 |
| 7 | **Single-realization Δ near the edge of stability.** Bitwise replay removes noise, but any edit can swing Δ through chaotic divergence | confounder | A repair "effect" could be sensitivity, not causation | [I] Skeptic M8 | Placebo replicates (1-ulp perturbations); effects must exceed the placebo spread |
| 8 | **Strawman baselines.** `Bg` keeps the step counter, so it mis-scales its own post-reset updates by ~4.5×. There's no rollback-and-skip (production practice) | missing baseline | "Beats full reset" says nothing | [V] reset.py; PaLM rollback | Adam-MR (t reset), bias-correct Bg, rollback-and-skip, OLMo/OPEN-1B skip rules, GradientStabilizer, StableAdamW; primary = best non-oracle baseline |
| 9 | **Architecture confound.** Marin 32B traced spikes to the attention stack and QK-Norm removed them; our nanoGPT has no QK-Norm | confounder | Proxy spikes could be attention-logit blow-ups with state damage only a symptom | [V] Marin retro; [V] nanogpt.py | QK-Norm ablation; claim "damage *carried* in state", not "root cause" |
| 10 | **Pseudo-replication and power.** Seeds change only the init (identical data order); 2 sites share a seed's trunk; seed 0 tunes. The SD of paired Δ is unknown | weak evidence | n = 10 may behave like ~5–7; TOST may be badly underpowered | [V] V7 §4.2; [I] `09` §5 | Data-order seeds; mixed model; pilot SD → n rule n = max(10, ⌈8.6(σ̂/δ)²⌉) |
| 11 | **Natural spikes have no B\*.** "Same run with the hook omitted" doesn't exist for them | missing control | 10 of 30 instances can't give a Δ | [V] k2.py:6 | Exclude them from causal contrasts; use them for localization and detection (pre-registered) |
| 12 | **Planted benchmark too easy.** `v_inflate` *is* a v-drift, so a magnitude score wins by construction | weak evidence | C2 "accuracy" would be inflated | [I] Method Hunter | Causally inert decoys + m-only plants + κ doses |
| 13 | **Off-manifold hybrids.** m from B\* with v from B0 can violate the \|m\| ≤ 1.165√v invariant every real Adam state satisfies | unstated assumption | Patch results could reflect impossible states | [I] `09` §2.4 | Count violations on every hybrid; off-manifold control |
| 14 | **Horizon dependence.** Δ inside the state-memory window can vanish by end of schedule; with constant LR there's no decay phase | weak evidence | The headline could depend on an arbitrary horizon | [I] Skeptic M11 | Δ(h) curves; primary = end of schedule |
| 15 | **Eval leakage.** The localizer probe uses val batches 0–3, the same pool Δ would use | leakage | Localizer tuned on the data it's scored on | [V] audit (k2.PROBE) | Δ on val batches ≥ 1000, disjoint |
| 16 | **Code gotcha: `lr_bump` overwrites any LR schedule.** It sets lr = base × factor or base, an absolute value, every step | implementation bug (latent) | The realistic-proxy fix (#6) would silently become constant-LR | [V] induce.py:52-60 | Multiply the scheduled LR instead |
| 17 | **Embedding-row artifact.** Rare-token rows have tiny, stale v and huge preconditioned curvature regardless of any spike | confounder | Localizers rank wte rows first "by default" | [I] Skeptic m14 | Report embedding vs blocks separately; normalise per parameter; untied variant |
| 18 | **Tied weights.** wte = lm_head, so row edits also change output logits | unstated assumption | Row-level repair has a side effect on outputs | [V] nanogpt.py:97-107 | State it; run an untied ablation |
| 19 | **Results not traceable.** No git SHA, cfg or versions in `results/*.json`; the 124M Δ = 0 claim exists only as prose, from torch 2.7.1 vs the pinned 2.10 | reproducibility | Prior "verified" claims can't be audited | [V] audit | Stamp every artifact |
| 20 | **Localization cost vs rollback.** HVP-heavy localization may cost more than just rolling back ~100 steps | significance | "Why not roll back?" | [I] Skeptic M10 | Recovery per GPU-second Pareto; a no-HVP localizer variant |

---

## 2 · Prior work: what's already done (don't repeat it) vs what's genuinely new

| Idea in our plan | Already done by | Status for us | Sources |
|---|---|---|---|
| Spike mechanism: v decouples from g², preconditioned curvature crosses 2/η; λ_grad predictor | Bai et al., ICML 2026 | **Not ours.** Adopt their Ĥ; use their predictor as a baseline localizer | [arXiv 2506.04805](https://arxiv.org/abs/2506.04805) · [PMLR](https://proceedings.mlr.press/v306/bai26o.html) |
| Stale second moment causes spikes (per-tensor RMS) | Wortsman et al. 2023 | **Not ours.** Cite §3.4 (28/30) and App. D (14/15, ViT-Huge) | [2304.13013](https://arxiv.org/abs/2304.13013) |
| Bitwise-deterministic replay including optimizer state | NVIDIA Megatron Core (Oct 2026); OPEN-1B (Sep 2026); Unlearning-at-Scale (2025) | **Not ours.** Claim the *counterfactual instrument*, not determinism | [Megatron blog](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/) · [2609.17380](https://arxiv.org/abs/2609.17380) · [2508.12220](https://arxiv.org/abs/2508.12220) |
| Paired control/shock forks with a shared future; channel-level (θ/m/v) injection | Liu & Li 2026 | **Partly theirs** (design). Ours: real spikes, sub-channel granularity, bitwise gate | [2608.19762](https://arxiv.org/abs/2608.19762) |
| Optimizer-state "surgery" / patching between runs | Xu 2026 (subliminal transfer; m carries ~90%, v ~0%); Chimera interventions (grokking) | **The method is theirs.** Ours: applied to spikes, the v channel, the cut-time sweep, row granularity | [2608.20442](https://arxiv.org/abs/2608.20442) · [2607.06628](https://arxiv.org/abs/2607.06628) |
| Global or periodic moment resets | SPAM (ICLR 2025); Asadi 2023 (v-only reset in RL); Topollai 2026 (reset timing theory) | **Not ours.** We do selective, post-spike, signed resets | [2501.06842](https://arxiv.org/abs/2501.06842) · [2306.17833](https://arxiv.org/abs/2306.17833) · [2603.16731](https://arxiv.org/abs/2603.16731) |
| Rank-k edits of updates/momentum with an energy-matched control | Spectra 2026 (top-1.5% singular-value suppression); Xie et al. 2026 (QK top-3 projection) | **Form not ours.** We edit stored m/v post-spike; our `Br` must be energy+support-matched | [2602.11185](https://arxiv.org/abs/2602.11185) · [2608.02091](https://arxiv.org/abs/2608.02091) |
| Online preconditioned-curvature tracking | Ramasinghe et al. 2026 | **Not ours.** Reuse as tooling | [2606.16768](https://arxiv.org/abs/2606.16768) |
| Gradient-side spike mitigation | ZClip, AdaGC (ICML 2026), GradientStabilizer (ICML 2026), StableAdamW | Baselines | [2504.02507](https://arxiv.org/abs/2504.02507) · [2502.11034](https://arxiv.org/abs/2502.11034) · [2502.17055](https://arxiv.org/abs/2502.17055) |
| Skip-step / rollback | PaLM; OLMo-core; OPEN-1B; Megatron rerun | Baselines | [2204.02311](https://arxiv.org/abs/2204.02311) · [OLMo-core](https://olmo-core.readthedocs.io/en/latest/_modules/olmo_core/optim/skip_step_optimizer.html) |
| Natural v-poisoning by hardware faults | Altenbernd et al. 2026 (SDC drives v → ∞; lists momentum resets as future work) | Evidence for us; a group adjacent to us | [2604.00726](https://arxiv.org/abs/2604.00726) |
| Non-v spike mechanisms | Marin 32B (attention / QK-Norm); Tang 2026 (superquadratic); Li, Zhou & Xu 2026 (weight-norm criticality) | Counter-evidence to address | [Marin](https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/) · [2609.18314](https://arxiv.org/abs/2609.18314) · [2607.21005](https://arxiv.org/abs/2607.21005) |

**Genuinely new (bounded by targeted searches, 2026-10-19/22, not a systematic review):**
1. A causal **decomposition of spike damage across w/m/v over time** (damage-migration curve), with the
   β2-memory prediction.
2. **Signed, support-limited** localization *inside* stored m/v, scored against a causal oracle.
3. The **exact undo log**: a state-only retroactive skip.
4. **Poisoned-state transplant with dose-response.**
5. The **anti-windup invariance** derivation that explains why clipping only softens spikes.
6. A **curated** spike-snapshot set with moments and oracle maps. OPEN-1B's bucket is the closest
   [UNV: whether it holds m/v].

---

## 3 · Raising the novelty: 5 extensions

### E1 · β2 as a causal instrument, a "migration law"
- **What:** run the damage-migration curve at β2 ∈ {0.95, 0.99, 0.999}.
- **Prediction** (`09` §1.4): the state-carried share decays at ≈ −ln β2 per step, i.e. half-lives of
  13.5 / 69 / 693 steps.
- **Why new:** nobody has measured how spike damage moves from optimizer state into weights. Matching a slope
  predicted by first principles across three β2 values is strong mechanistic evidence that is hard to fake.
  Xu 2026 measured one allocation at one lag, for a different phenomenon.
- **Test:** 3 × (8 cells × 5 cut times) forks per instance at proxy (about 1–3 GPU-h). Fit
  share(τ) = a·e^{−rτ} + b and test r vs −ln β2 with bootstrap CIs.

### E2 · Exact undo log + anti-windup theory
- **What:** subtract the spike window's direct contribution from m and v, weights untouched.
- **Theory** (`09` §2.3): a spike-dominated coordinate drifts a **fixed ≈ 5.84·η**, independent of spike
  size, because update/η = 0.447·0.9234^k. That is why clipping only softens spikes.
- **Why new:** no retroactive subtraction from Adam's EMAs found (2 searches); SPAM only zeroes or resets.
  The invariance gives sharp, β-only predictions.
- **Test:**
  - undo-m vs undo-v vs undo-both vs oracle patch, at lags 1–30;
  - predicted ordering: undo-v-alone hurts, undo-m helps, both comes closest;
  - predicted benefit half-life with lag: 8.7 steps;
  - clip invariance at clip 1.0 vs 0.2.

### E3 · Poisoned-state transplant and dose-response ("noising")
- **What:** put B0's (m, v) into B\* (clean weights, clean data) at varying κ.
- **Why new:** nothing found that shows poisoned optimizer state *alone* causes damage on clean weights. It is
  the most direct "state is a carrier" evidence.
- **Prediction:** a monotone dose-response that **plateaus** above the dominance threshold (E2).
- **Test:** κ ∈ {1, 10², 10⁴}, binomial "Δ > τ" rate plus Spearman ρ.

### E4 · Signed localization scored against a causal oracle
- **What:**
  - **Oracle:** adaptive group testing (delta debugging) over tensor/row/head groups.
  - **Cheap predictor:** AtP-state (one backward, log-v), with the **sign** from a two-sided innovation
    test (g²/v_prior lower and upper tails).
- **Why new:** a group-testing oracle for optimizer state was not found; AtP over (m, v) for spikes was not
  found (Xu's costate is adjacent); lower-tail innovation catches Bai's decay stage, which grad-norm and loss
  detectors can't see.
- **Test:**
  - AUROC vs the oracle and vs decoy-augmented plants;
  - gain over a magnitude-only control;
  - repair-top-k curves.

### E5 · SpikeState: a curated spike set with optimizer state and oracle damage maps
- **What:** induced + natural spike instances with (w, m, v, v̄) snapshots, clean twins and oracle maps,
  released on the HF Hub with Croissant metadata.
- **Why new:** no curated public set with moments was found. OPEN-1B's bucket is uncurated, and its events
  were skipped or clipped.
- **Use:** lets others benchmark localizers *without forks*, which raises the paper's floor.
- **Test:** a leaderboard protocol (oracle-agreement AUROC + repair AUC) on held-out instances.

*(After wave 2: candidates include μP scale-collapse of migration curves, jump-robust variance estimators
from finance, covariance localization from data assimilation, and "optimizer-state poisoning attacks" as
a security framing. They'll be added once kill-searched.)*

---

## 4 · Rigor upgrades within the limits

**Experiments**
- The MVE (week 2, ≤ 1 day) tests the headline before any localizer code.
- A cross-process determinism gate through spike windows.
- A realistic proxy plus a mid rung; 124M primary contrast with n ≥ 3 (4 factorial cells, h = 300);
  optional 350M.

**Metrics**
- Shapley state share φ_state/D(111), always with the noising and denoising shares and interactions.
- Gap closed G vs the best non-oracle baseline.
- % of oracle recovery; edit fraction; recovery time (Kaplan–Meier).
- AUROC/AUPRC/P@k vs oracle and planted truth; repair-AUC.
- Cost per GPU-second vs rollback; hybrid-violation rate; placebo spread.

**Ablations**
- **Localizer:** AtP-only / NIS-only / both / λ-decomposition / magnitude-only.
- **Reference:** B\* / v̄ / fresh g².
- **Repair:**
  - target: v̄ vs g²;
  - step-counter reset on vs off;
  - support: tensor / row / head;
  - undo window: exact / detector / ±2.
- **Horizon** h.
- **Architecture:** QK-Norm on / off; tied vs untied embedding.
- **Optimizer:** Muon (2-D) + AdamW (rest).

**Statistical checks**
- One pre-registered primary contrast (TOST non-inferiority → superiority).
- Holm across the secondary tests; BH for exploratory ablations.
- Mixed model Δ ~ branch + site + (1 | seed).
- Data-order seeds.
- A pilot SD that sets n.
- Oracle stability (Jaccard across placebo seeds, SuSiE fallback).
- Never pool spike types.

**Pre-registration**
- Commit the §4.7 table (`09` §9) before any AMD data: τ = 0.01, margin, n rule, horizons, seeds, Δ eval set,
  `Bs` clip, natural-spike rule, β2 instrument.

---

## 5 · The top 3 changes with the biggest payoff, and the order to do them

1. **Fix the instrument and pre-register (week 1, days; prerequisite, so it goes first).**
   - Held-out val-loss Δ (gap 2).
   - A non-degenerate cheap fix / best-cheap-fix Gate B (gap 1).
   - `lr_bump` multiplying the schedule (gap 16).
   - Δ eval disjoint from probe batches (gap 15).
   - Stamps (gap 19).
   - The §4.7 thresholds committed.
   - The spike pipeline pushed through a PR.

   Run this alongside the AMD bring-up and the cross-process determinism gate (gap 5). *Payoff:* without
   it, no number we produce is valid.

2. **Make the w/m/v damage-migration factorial the primary result, and run it as the week-2 MVE.** Include
   placebo replicates, hybrid feasibility, and the undo-log arms (E1–E3).
   - *Payoff:* it turns the thesis from assumed into tested, defines the localizer's ground truth, answers
     Marin, and gives a publishable result either way.
   - *Cost:* about 1–2 GPU-h.
   - *Go/no-go:* G1 harness sanity, G2 headline.

3. **Fix external validity (weeks 3–8).**
   - A warmup+cosine proxy at ≥ 20 tokens per non-embedding param, plus a mid rung, plus the QK-Norm
     ablation.
   - The full battery with the strong baselines.
   - The 124M primary contrast (n ≥ 3).

   *Payoff:* it turns "a 3-layer toy at step 160" into a scale ladder; it is the remaining ceiling on the
   score.

**Why this order.** (1) is cheap and gates the validity of everything else. (2) is the cheapest experiment
that can change the paper's direction (pivot insurance), so it must come before the expensive scale work.
(3) is where most GPU-hours go, so it should only run once (2) has said what to measure.

**Honest bottom line.**
- The core idea survives the prior-work check, but narrower than the original plan claimed.
- As planned, it would likely be rejected (Skeptic: 3/6), mainly for gaps 1–4 and 6.
- With the three changes above, a realistic outcome is borderline to weak accept (4–5/6), conditional on the
  AMD determinism gate passing.
- The biggest single scientific risk is gap 3: the state-carried share could be small at β2 = 0.95. E1 turns
  that risk into a result.
