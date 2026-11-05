# 02 · Adjacent-field transfers for C1/C2/C3 (Phase 2, Adjacent-Field agent)

*Written 2026-10-22. Target: ICML 2027. Nothing here has been run. Every prediction below is a candidate for
pre-registration (PLAN_V7 core value 3); none is a result.*

**Labels.** **[V]** = verified (read on the source page, abstract or arXiv/OpenAlex metadata on 2026-10-22).
**[V◐]** = from a search snippet or secondary summary only, not the source text. **[I]** = inferred (my
derivation or mapping, not from a source). **[A]** = assumed (cost assumption). **UNVERIFIED** = could not confirm.

**Cost units [A].** One *fork* = one `run_fork` of L = 200 steps from a snapshot, plus one held-out eval. L is
my assumption: the eval horizon is not pre-registered yet (improvement.md A3). The measured proxy speed is
0.0619 s/step on a P100 at batch 16 (PLAN_V7 §3), so one fork is about 12 s on a P100. The MI300X speed has not
been measured. One *HVP* is about 2–3× a fwd+bwd under `SDPBackend.MATH` (assumed, see A7). Proxy = 3 layers ×
192 wide, vocab 50304, about 11M parameters, **≈21 named tensors** (3×6 per block + `wte`, `wpe`, `ln_f`;
`lm_head` tied), `grad_accum = 1` (from `experiments/proxy/config.yaml` [V]; the tensor count is [I]).

---

## 1 · Ranked table

Kill-search verdict: **NEW** = no prior application to optimizer state or loss spikes found. **PARTIAL** = the
mechanism has been applied nearby (cited). **DONE** = already done for our use; adopt it and cite it.

| # | Transfer | Field | Mapping onto our objects | Falsifiable prediction (short) | Cost @11M | Verdict | Value |
|---|---|---|---|---|---|---|---|
| T1 | **Exact compensation ("undo log")** of the spike window's contribution to m, v | Databases (ARIES compensation records), machine unlearning (Amnesiac ML) | m, v are linear in logged g, g². Subtract `(1−β1)β1^{t−s}g_s` and `(1−β2)β2^{t−s}g_s²` for s ∈ spike window, with g_s recovered exactly from consecutive ring-buffer snapshots. Weights are left as they are. | For `corrupt_batch`, undo ≈ patching (m,v) from B* (within τ) at lag ≤ 5. The gap grows with lag. For `lr_bump`, undo recovers < 20% | 3 forks/instance, 0 HVP | **NEW** | **5** |
| T2 | **Mediation / 2³ factorial decomposition** over (w, m, v), with a sweep over cut times | Causal inference (Pearl NDE/NIE, VanderWeele 4-way) | Unit = instance×seed. Every cross-world hybrid Θ(a_w,a_m,a_v), a ∈ {B0, B*}, is runnable. Main effects + interactions as a function of the cut time t1 give a "damage migration curve" | (m,v) share > τ at t1−s ≈ 60 steps (≈3 e-folds of v), or the thesis fails. `v_inflate`: v share ≈ 100% (harness sanity) | 6 forks per (instance, t1) | **PARTIAL** (Xu 2026 state surgery) | **5** |
| T3 | **Delta debugging + adaptive group testing** for the oracle poisoned set | Software engineering (Zeller ddmin/dd), information theory (Hwang splitting) | Configuration = per-tensor state source ∈ {B*, B0} over ≈21 tensors. The noising direction gives an OR-model test. dd isolates a 1-minimal failure-inducing difference | `v_inflate` with k planted tensors is found exactly in ≤ ⌈log2 C(21,k)⌉+k−1 forks (5/9/13 for k=1/2/3) vs 21. Natural oracle sets have ≤ 3 tensors | 5–15 forks vs ~20 | **NEW** | **4** |
| T4 | **LMI minimal stabilizing repair** (optimal diagonal preconditioning) | Numerical optimization / convex optimization | λ_max(D^{-1/2}HD^{-1/2}) ≤ c ⇔ H ⪯ cD. With top-k Ritz pairs this becomes a k×k LMI that is *linear* in x = 1/(√v̂+ε). Minimizing ‖x−x⁰‖ subject to it outputs both *where* and *how much* | For an equal stability margin it edits ≤ ⅓ the v mass of uniform-S scaling (RESEARCH §5) with Δ ≤ uniform. If it is infeasible on S, the localizer missed | 30–40 HVP + CPU solve + 1 fork | **NEW** | **4** |
| T5 | **Two-sided innovation test + covariance matching** on per-tensor g²/v̂_{t−1} | Control (Kalman innovation tests, Mehra adaptive filtering) | NIS_S = mean g²/v̂_prior. Upper tail = stale v (Wortsman, SPAM). **Lower tail = v decoupled above g²** (Bai's decay stage, `v_inflate`, post-`corrupt_batch` over-damping). Repair a_S² = robust mean NIS | `v_inflate`: lower-tail NIS ranks the targets with AUROC ≥ 0.95, while grad-norm/loss detectors have recall ≈ 0. β2 = 0.999 presets: lower-tail CUSUM leads the RMS > 2.3 rule by ≥ 1 step | 0 forks (telemetry) | **PARTIAL** | **4** |
| T6 | **Trajectory attribution patching (AtP-k)** over optimizer state | Interpretability (attribution patching, AtP*) | Closed-form ∂L_val/∂m, ∂L_val/∂(log v) through k frozen-gradient AdamW steps, using B0's logged trajectory. One val gradient scores all tensors | Spearman ρ ≥ 0.7 vs exact patching (T3) for k ≤ 20. It fails in linear-v coordinates at κ ≥ 100 but holds in log-v | 1 val-grad per (instance,k), 0 HVP | **PARTIAL** (Xu 2026 costate; Ding 2026) | **4** |
| T7 | **Patching hygiene**: denoising vs noising, the subspace "illusion", resample ablation | Interpretability (Zhang & Nanda; Makelov et al.; causal scrubbing) | Report sufficiency (B*→B0) and necessity (B0→B*). Rank-r subspace patches of m need a random-subspace and a third-state control. Matched-normal states give on-distribution patches (helps A4) | Random rank-r m-subspace patches recover ≤ 10% of the top-r H~ subspace's recovery. If > 50%, the H~ claim fails | ≈0 extra (same forks as T3) | **PARTIAL** | **4** |
| T8 | **Parity residuals + fault signature matrix** | Control (analytical redundancy / FDI), reliability (ABFT, SDC) | r_v = v_t − β2v_{t−1} − (1−β2)g_t² ≡ 0 for honest Adam. Invariant \|m_i\| ≤ C√v_i (C ≈ 1.165 at 0.9/0.95). The residual pattern isolates the fault class | r_v = 0 (to fp32 rounding) on all data/LR/natural spikes and ≠ 0 exactly at `v_inflate`/bit-flip steps. Signatures separate the 3 types with ≥ 90% accuracy | 0 forks | **NEW** | **3** |
| T9 | **Anti-windup lens**: Adam's post-spike drift is invariant to the clip threshold | Control (integrator windup) | On spike-dominated coordinates, m/√v ≈ ((1−β1)/√(1−β2))(β1/√β2)^k = 0.447·0.923^k, independent of the gradient scale | Post-spike update RMS on spiked coordinates stays within ±10% between clip 1.0 and 0.2 while dominance holds. This informs the A1 `Bs` clip | 0 extra (uses `Bs` at 2 clips) | **NEW** | **3** |
| T10 | **Synthetic control** for natural-spike B* (A4) | Econometrics (Abadie et al.) | Donor pool = same-preset seeds without a spike near t_s. The shared data stream is a "common shock". Pre-period fit, then post gap, with placebo p-values | Usable only if pre-period RMSPE < τ/2. Pre-register: otherwise drop natural spikes from Gate B | 0 forks + CPU | **NEW** | **3** |
| T11 | **Multi-stream sparse change detection** (Xie–Siegmund mixture) | Statistics / signal processing | 21 per-tensor streams (log NIS, v̄-drift). A mixture GLR with prior fraction p0 detects sparse changes, and its per-stream terms localize them | Sparse `v_inflate`: detects in ≤ 5 steps with the targets carrying the largest GLR terms. Dense `corrupt_batch`: no better than the global detector | CPU only | **PARTIAL** | **3** |
| T12 | **Rotated-basis moment editing** (SOAP / preconditioner refresh) | Numerical optimization | A direction edit of m is exact. A direction edit of diagonal v is ill-posed unless axis-aligned. Options: coordinate sets, diagonal weights u², or an in-subspace k×k second moment for k steps | Rank ≤ 5 m-direction edits recover as much as tensor-level m patches. Diagonal-weighted v-direction edits recover < 50% of T4 | 1–2 forks + shared Lanczos | **PARTIAL** | **3** |
| T13 | **Robust v̄ and robust poison score** (MAD/trimmed) | Robust statistics (Huber, MAD) | Trimmed v̄ that excludes flagged steps; z = (log v − median)/(1.4826·MAD) over matched-normal snapshots | Gaussian-z FP at \|z\|>3 is ≥ 3× nominal on held-out matched normals, while MAD-z stays within 2×. No loss of `v_inflate` AUROC | CPU only | **PARTIAL** (K2-V2 robust z for loss) | **3** |
| T14 | **Warm-started / thick-restart Krylov** + principal angles between B0 and B* | Numerical linear algebra | Reuse Ritz vectors across steps and branches. Principal angles between the top-k H~ subspaces of B0 and B* show which directions rotated | Warm start reaches cold top-k within 1% at ≤ 40% of the HVPs. ≤ 3 of the top-10 angles exceed 30° post-spike | 30–40 HVP cold, ~12–15 warm | **PARTIAL** | **3** |
| T15 | **HVP-free directional curvature** for S_t telemetry | Numerical optimization (secant / finite difference) | Symmetric FD of the probe loss along Adam's actual update u, normalized by uᵀDu. Needs no double backward, so it runs under flash SDPA | Within 5% of the HVP Rayleigh quotient, with the same lead-time verdict | 2 fwd/measured step | **DONE** as a tool (Kalra 2026; Kim & Lee 2026) | **3** |
| T16 | **Micro-batch robust aggregation** + free SNR replicates | Distributed systems (Byzantine-robust aggregation) | `grad_hook` micro-gradients give per-coordinate variance for C2's SNR (free at 124M, grad_accum 40). A trimmed mean over micro-batches is a prevention baseline | Robust aggregation stops `repeat_ngram` only when < 50% of micro-batches are hit, and fails on whole-batch `corrupt_batch` | proxy: per-sample grads; 124M: free | **NEW** (as baseline) | **2** |
| T17 | **Plasticity transfers**: β1=β2 Adam control, unit-level utility resets, Fisher (EWC) | RL / continual learning | β1=β2 as a control optimizer. Row/column granularity for C2/C3. v ≈ diag empirical Fisher as a repair regularizer (with Kunstner's caveat) | Whether β1=β2 is non-spiky depends on the C of Tang 2026's boundary 1−β2 = C(1−β1). Testable on stress presets | ~0.05 P100-h per stress run | **PARTIAL** | **2** |
| T18 | **Spectrum-based fault localization (Ochiai)** across the spike population | Software engineering (SBFL; DeepFault) | "Coverage" = tensor flagged by telemetry; "failing" = instance with Δ(B0) > τ. Gives a population prior for ordering T3's search | Ochiai top-3 contains the oracle set in ≥ 70% of held-out instances | 0 forks | **PARTIAL** | **2** |
| T19 | **Transactional step rejection / recompute-on-detect** | Numerical ODE step control, reliability | = `Bskipstep_det`, already built | — | — | **DONE** (OLMo-core skip; Altenbernd 2026 recompute) | **1** |

**The five strongest: T1, T2, T3, T4, T5.** T6 and T7 are close behind and reuse T3's forks.

**Two design insights the transfers surface, independent of any single row [I]:**
1. **v-poison has two signs, and they need opposite repairs.** *Over-damping* (v too large: `v_inflate`,
   post-`corrupt_batch`) stalls learning; the repair shrinks v (`Bvbar`, T5 covariance matching). *Under-damping*
   (v decayed below g², Bai's mechanism) destabilizes; the repair grows v (T4, RESEARCH §5). RESEARCH §5's
   "minimal stabilizing repair" covers only the second. The C2 output contract should carry a sign per localized
   group, and T5's lower and upper tails give that sign for free.
2. **Patched hybrids can be infeasible states.** A (m from B*, v from B0) hybrid can violate the invariant
   |m_i| ≤ C√v_i (T8), which every honest Adam state satisfies. Count violations on every hybrid before
   interpreting a patch result. Stewart 2026 (arXiv 2605.17590) makes the same warning for L-BFGS memory:
   parameter-only or memory-only corrections give internally inconsistent states.

---

## 2 · One section per transfer

### T1 · Exact compensation of the spike's contribution to (m, v): an "undo log" for optimizer state
- **Source mechanism.** ARIES (Mohan et al., ACM TODS 17(1):94–162, Mar 1992, doi
  [10.1145/128765.128770](https://doi.org/10.1145/128765.128770)) logs every update and undoes a past action
  with compensation records while keeping later history [V◐]. Amnesiac ML (Graves et al., arXiv
  [2010.10981](https://arxiv.org/abs/2010.10981), 2020-10-21) stores per-batch parameter updates and subtracts
  them to unlearn [V]. That works on **weights**, where it is only approximate, because later updates depend on
  the earlier ones.
- **Mapping [I].** AdamW's moments are linear recurrences with no self-feedback: m_t = β1m_{t−1}+(1−β1)g_t and
  v_t = β2v_{t−1}+(1−β2)g_t², where g_t is the post-clip gradient. The direct contribution of a spike window W to
  the state at t1 is exactly c_m = (1−β1)Σ_{s∈W}β1^{t1−s}g_s and c_v = (1−β2)Σ_{s∈W}β2^{t1−s}g_s². Compensation:
  m ← m − c_m (a retroactive momentum skip), and v ← v − c_v + (1−β2)Σβ2^{t1−s}·v̄ (replace the spike's g² with
  the clean expectation). **No new logging:** (1−β2)g_s² = v_s − β2v_{s−1} and (1−β1)g_s = m_s − β1m_{s−1} come
  from consecutive snapshots in capture's ring buffer. In fp32 the subtraction is accurate on spike steps, where
  g² dominates. Non-negativity holds exactly: v_{t1} − c_v ≥ β2^{(·)}v_{s0−1} ≥ 0. Weights and adam_step are left
  unchanged. This is a *state-only retroactive skip*, distinct from every branch in V7 §4.3: `Bskipstep` skips
  the weight update too and needs to know the spike in advance, `Bvbar` discards post-spike history, and `Bg`
  zeros everything. It also yields a **path-specific effect**: Δ(B0) − Δ(undo) is the direct path
  (spike gradient → state), and Δ(undo) − Δ(patch_mv from B*) is the indirect path (later gradients differ
  because the weights moved).
- **Prediction.** (i) `corrupt_batch`: |Δ(undo_mv) − Δ(patch_mv)| < τ at lag t1−s ≤ 5, and the gap grows
  monotonically with lag. (ii) `lr_bump` (no abnormal g², the damage enters through the weights): undo recovers
  < 20% of Δ(B0). (iii) `v_inflate`: undo is a no-op (no gradient caused it), and T8's parity residual flags it
  instead. If (i) fails at lag ≤ 5, either the moments are not where the `corrupt_batch` damage lives or the
  reconstruction is wrong (unit-test the reconstruction on CPU first).
- **Cost.** 3 forks per instance (m-only, v-only, both), 0 HVPs, ≈36 s P100 [A]. Needs ring-buffer snapshots
  over W.
- **Kill-search.** Queries: "unlearning Adam optimizer state moments remove contribution of a training step
  exact subtraction exponential moving average"; "retroactively remove spike gradient from Adam moments after
  detection…"; "Amnesiac machine learning…". Closest work: SPAM (arXiv
  [2501.06842](https://arxiv.org/abs/2501.06842)) found that zeroing spiked gradients helps, then uses
  reset+clip, with no retroactive subtraction [V◐]; Amnesiac ML (weights only); Stewart 2026,
  [2605.17590](https://arxiv.org/abs/2605.17590) (counterfactual *L-BFGS* state) [V]; DeltaGrad (Wu et al.,
  [2006.14755](https://arxiv.org/abs/2006.14755), replay-based) [V]. **NEW** as of 2026-10-22, from a targeted
  search, not a systematic review.

### T2 · Mediation analysis as a 2³ factorial over (w, m, v), with a cut-time sweep
- **Source mechanism.** Natural direct and indirect effects (Pearl, UAI 2001, arXiv
  [1301.2300](https://arxiv.org/abs/1301.2300)) [V]. The 4-way decomposition into controlled direct effect,
  reference interaction, mediated interaction and pure indirect effect (VanderWeele, *Epidemiology*
  25(5):749–761, 2014, doi [10.1097/EDE.0000000000000121](https://doi.org/10.1097/EDE.0000000000000121)) [V].
  Mediation applied to neurons: Vig et al. 2020, [2004.12265](https://arxiv.org/abs/2004.12265) [V].
- **Mapping [I].** Cut at t1 after the spike window. The state at t1 is Markov: after t1 both branches read the
  same data, so the spike has no direct path that bypasses (w, m, v). The question is therefore a multi-mediator
  decomposition. Deterministic replay makes every cross-world hybrid Θ(a_w, a_m, a_v) directly runnable.
  Epidemiology can only *identify* these under sequential ignorability; we *measure* them (as Vig et al. already
  noted for neurons, so this point alone is not new). Run all 8 configurations (B0 and B* already exist, so 6 new
  forks) and report the Möbius/Shapley main effects and interactions. Sweeping t1 ∈ {s+1, s+10, s+30, s+60}
  gives a **damage-migration curve**: the share of Δ carried by (m, v) vs w as the time since the spike grows.
  This is RESEARCH §4.2's "thesis test" with an estimand vocabulary reviewers already know.
- **Prediction.** (i) `v_inflate` at t1 = inject+1: the v main effect ≈ 100% of Δ and w ≈ 0. This is a harness
  sanity check; a miss means a bug. (ii) The thesis ("damage persists in state after weights recover") predicts
  a (m,v) share > τ at t1 − s ≈ 60 steps (≈3 e-folds of v at β2 = 0.95). It is **falsified** if that share is < τ
  on held-out seeds. (iii) Contrast with Xu 2026, where v carried −0.5% and m 89.9% (one cell): if our v share is
  also ≈0 on `corrupt_batch`/natural spikes, the v-poisoning story (and C3's v-repair) loses its basis. Report
  this whichever way it goes.
- **Cost.** 6 forks per (instance, t1). With 4 cut times that is 24 per instance; at 30 instances, ≈720 forks ≈
  2.5 P100-h [A]. Cheaper variant: main effects only at one t1, 6 forks.
- **Kill-search.** Queries: "causal mediation analysis neural network training dynamics natural direct indirect
  effect optimizer"; "'optimizer state' patching transplant Adam moments between runs causal intervention";
  OpenAlex "optimizer state transplant Adam moments causal". **Found and verified:** Xu 2026, *Stored in
  Optimizer State, Valued by Later Training* (arXiv [2608.20442](https://arxiv.org/abs/2608.20442), 2026-08-20) [V].
  It does block "state surgery" over (w, m, v), uses a discrete-adjoint "costate", and reports a lag-24
  allocation (m 89.9%, w 10.6%, v −0.5%) for subliminal trait transfer. It has no spikes and no repair.
  Also: Liu & Li 2026, [2608.19762](https://arxiv.org/abs/2608.19762) (shocks into {θ, m, v}) [V]; *Chimera
  interventions* in grokking, [2607.06628](https://arxiv.org/abs/2607.06628) (moment transplant as a control)
  [V◐]; *Training Memory* survey, [2601.21624](https://arxiv.org/abs/2601.21624) (carry/reset primitives) [V].
  **PARTIAL.** ⚠ This is a **novelty threat to RESEARCH.md §0 idea #1** ("optimizer-state patching … analogue
  of activation patching"). Xu 2026 must be cited and contrasted: spikes vs subliminal transfer, a sweep over
  cut times, localization within tensors, and repair.

### T3 · Delta debugging + adaptive group testing for the oracle localization
- **Source mechanism.** Zeller & Hildebrandt, *Simplifying and Isolating Failure-Inducing Input*, IEEE TSE
  28(2):183–200, Feb 2002 ([PDF](https://www.st.cs.uni-saarland.de/papers/tse2002/tse2002.pdf)) [V◐]. Two
  algorithms: `ddmin` minimizes a failing input, and `dd` isolates a 1-minimal difference between a passing and
  a failing configuration. Hwang's generalized binary splitting uses ≤ ⌈log2 C(n,d)⌉ + d − 1 tests (per secondary
  sources; the 1972 JASA details are **UNVERIFIED**). Survey: Aldridge, Johnson & Scarlett,
  [1902.06002](https://arxiv.org/abs/1902.06002) [V]. Noisy, non-adaptive variant: SuSiE credible sets (Wang et al.,
  JRSS-B 82:1273–1300, 2020, doi [10.1111/rssb.12388](https://doi.org/10.1111/rssb.12388)) [V◐].
- **Mapping [I].** A configuration is a bitstring over ≈21 tensors (state source B* or B0 per tensor).
  denoise(S) ≡ noise(Sᶜ), so one family of hybrid forks serves both readings. Group testing needs an **OR
  model**: read the *noising* predicate "transplanting B0's T into B* yields Δ > τ_f". This fires if T contains
  any sufficient poisoned tensor. The *denoising* predicate is AND-like (it recovers only if S covers every
  necessary tensor), and that case is what `dd` handles. For parallel GPU scheduling, a non-adaptive Bernoulli
  design plus a sparse regression (SuSiE) runs all forks in one batch and returns credible sets when
  interactions or noise break the OR model.
- **Prediction.** (i) `v_inflate` with k planted tensors: the exact set is recovered in ≤ 5/9/13 forks for
  k = 1/2/3 (vs 21 single-tensor forks), with no false positives if τ_f ≥ 5× the noop drift. (ii) Sparsity
  hypothesis for natural and `corrupt_batch` spikes: the 1-minimal set has ≤ 3 tensors. It is falsified if `dd`
  returns > 6, or if the OR model breaks (T1∪T2 damages while neither alone does), which is itself a reportable
  interaction.
- **Cost.** 5–15 forks per instance vs ~20 for RESEARCH §4.2's greedy search, so a 2–4× saving. 0 HVPs.
- **Kill-search.** Queries: "delta debugging minimal failure-inducing deep learning training bug localization
  group testing neural network"; "'group testing' adaptive localization faulty parameters neural network model
  debugging few evaluations"; "localize which layer causes loss spike bisection parameter groups…". Closest:
  DelBugV (delta debugging of NN *verifiers*, [2305.18558](https://arxiv.org/abs/2305.18558)) [V◐]; Qiu et al.
  2025 ([2510.04212](https://arxiv.org/abs/2510.04212)), manual per-layer toggling to find a low-precision flash
  attention failure [V◐]; neural group testing for inference speed ([2011.10704](https://arxiv.org/abs/2011.10704))
  [V]. **NEW** for optimizer-state localization.

### T4 · Minimal stabilizing repair as an LMI (optimal diagonal preconditioning)
- **Source mechanism.** Qu, Gao, Hinder, Ye & Zhou, *Optimal Diagonal Preconditioning*, *Operations Research*
  73(3):1479–1495, 2025 (arXiv [2209.00809](https://arxiv.org/abs/2209.00809)). They pose diagonal scaling as a
  quasiconvex problem and reduce the one-sided case to a dual SDP [V◐]. The Schur-complement LMI is a standard
  fact.
- **Mapping [I].** For D ≻ 0, λ_max(D^{-1/2}HD^{-1/2}) ≤ c ⇔ H ⪯ cD. The threshold c is 38/η at β1 = 0.9 in our
  D convention (Cohen et al. [2207.14484](https://arxiv.org/abs/2207.14484)), times (1−δ). With top-k Ritz pairs
  H_k = UΛUᵀ from Lanczos, the condition becomes the k×k LMI Λ^{1/2}Uᵀdiag(x)UΛ^{1/2} ⪯ cI, which is **linear in
  x_i = 1/(√v̂_i+ε)**. Repair: min Σw_i(x_i−x_i⁰)² subject to the LMI and x ≤ x⁰ (only grow v). Only coordinates
  with eigenvector mass enter the constraint, so the solution is sparse and gives localization and magnitude
  together. RESEARCH §5's uniform `a` on S with a gradient-direction Rayleigh quotient is the special case k = 1
  with x_S scaled uniformly. A projected first-order solver in torch is enough, so no new dependency is needed.
  This addresses **under-damping only** (see insight 1).
- **Prediction.** (i) At an equal post-repair margin, the LMI repair changes ≤ ⅓ of the v mass that uniform-S
  scaling changes, with Δ(LMI) ≤ Δ(uniform) paired over seeds. (ii) If it is infeasible when restricted to C2's S,
  the localizer missed the set (a built-in diagnostic). (iii) A fresh Lanczos on the next step gives λ_max ≤ c.
  It is falsified if (i) reverses.
- **Cost.** One Lanczos (k = 10, ~30–40 HVPs) + a CPU solve + 1 fork per variant.
- **Kill-search.** Queries: "'optimal diagonal preconditioning' semidefinite programming…"; "rescale Adam
  second moment restore stability preconditioned sharpness below 2/eta intervention spike recovery". Found:
  Bai et al. (mechanism and natural self-recovery, no rescaling intervention, [2506.04805](https://arxiv.org/abs/2506.04805))
  [V◐]; Tang et al. 2026 ([2609.18314](https://arxiv.org/abs/2609.18314), a β1/β2 phase diagram, no repair) [V].
  **NEW** as an application.

### T5 · Two-sided innovation test + covariance-matching repair
- **Source mechanism.** In a healthy Kalman filter the normalized innovations are white with unit scale, and
  their statistics can re-estimate mis-specified noise covariances (Mehra, IEEE TAC 15(2):175–184, Apr 1970)
  [V◐]. Adam as Bayesian filtering: Aitchison, NeurIPS 2020 ([1807.07540](https://arxiv.org/abs/1807.07540)) [V].
- **Mapping [I].** Per tensor, NIS_S,t = mean_{i∈S} g_{i,t}²/v̂_{i,t−1}, using the **prior** v, because the
  posterior v_t bounds the ratio at 1/(1−β2), the bound RESEARCH §3.1 notes. The repo already logs a posterior
  version (update RMS); the upper tail is what Wortsman's RMS > 2.3 and SPAM's g² > θv use. The transfer adds the
  **lower tail as a calibrated sequential test**: persistent NIS ≪ 1 means v is decoupled above g². That is
  Bai's preconditioner-decay stage (a natural precursor), the planted `v_inflate` (NIS ≈ 1/κ), and the
  over-damped aftermath of `corrupt_batch`. Covariance matching gives a v̄-free repair: a_S² = trimmed mean of
  NIS_S over a window, then v_S ← a_S²v_S. It works in both directions (insight 1).
- **Prediction.** (i) `v_inflate`: the lower-tail NIS ranks the planted tensors first with AUROC ≥ 0.95 at zero
  forks, while loss and grad-norm detectors have recall ≈ 0 (no gradient anomaly). (ii) β2 = 0.999 stress
  presets: a lower-tail CUSUM on log NIS alarms before the upper-tail rule by a median ≥ 1 step at conformal
  run-level FP ≤ 0.05. This is exactly RESEARCH §6's adoption criterion; it fails if there is no lead.
  (iii) Covariance-matched rescaling recovers ≥ `Bvbar`'s recovery on `v_inflate`.
- **Cost.** 0 forks for detection; 1 fork per repair variant.
- **Kill-search.** Queries: "Kalman filter interpretation of Adam optimizer innovation…"; OpenAlex "normalized
  innovation Kalman test Adam optimizer second moment outlier"; "early warning loss spike ratio squared gradient
  to second moment low g^2/v preconditioner decay…". Closest: Wortsman 2023 ([2304.13013](https://arxiv.org/abs/2304.13013),
  upper tail; "for lower β2, RMS_t does not deviate far from 1") [V◐]; AdaBelief ([2010.07468](https://arxiv.org/abs/2010.07468),
  (g−m)² inside the optimizer) [V]; TIC (ICML 2026, [page](https://icml.cc/virtual/2026/80057)) lists an
  "Adam-mismatch" baseline whose definition I could not read (**UNVERIFIED** overlap). **PARTIAL.**

### T6 · Trajectory attribution patching (AtP-k) over optimizer state
- **Source mechanism.** Attribution patching is a first-order Taylor approximation of activation patching
  (Nanda, [write-up](https://neelnanda.io/attribution-patching); Syed et al.
  [2310.10348](https://arxiv.org/abs/2310.10348)) [V]. AtP* identifies its failure modes (saturation,
  cancellation) and fixes them (Kramár et al., [2403.00745](https://arxiv.org/abs/2403.00745), 2024-03-01) [V].
- **Mapping [I].** The one-step closed form for AdamW with w' = w(1−ηλ) − η·m̂/(√v̂+ε) is
  ∂L/∂m = −(ηβ1/bc1)·∇L ⊙ 1/(√v̂+ε) and ∂L/∂v = (ηβ2/bc2)·∇L ⊙ m̂/(2√v̂(√v̂+ε)²), where ∇L = ∇L_val(w').
  k-step "frozen-gradient" version: δm_j = β1^jδm and δv_j = β2^jδv propagate through B0's **logged** (m_j, v_j)
  trajectory into δw_{t+k}, contracted with one val gradient at t+k. One gradient scores every tensor and
  coordinate. Use log-v coordinates (∂L/∂log v = v·∂L/∂v), because the poison is multiplicative.
- **Prediction.** (i) Spearman ρ ≥ 0.7 between AtP-k group scores and T3's exact patch recoveries over the 21
  tensors, for k ≤ 20. (ii) In linear-v coordinates AtP fails for κ ≥ 100 (ρ < 0.3) but holds in log-v
  (ρ ≥ 0.7). (iii) It fails near λ > 2/η, the analogue of AtP*'s saturation case; report this as a limit.
- **Cost.** 1 val-batch gradient per (instance, k) + CPU arithmetic; 0 HVPs; validated against T3's forks.
- **Kill-search.** Query: "attribution patching gradient with respect to optimizer state Adam moments linear
  approximation training intervention". Found: Xu 2026's full discrete-adjoint costate
  ([2608.20442](https://arxiv.org/abs/2608.20442)) [V]; Ding et al. 2026, *In-Run Data Shapley for Adam*
  ([2602.00329](https://arxiv.org/abs/2602.00329), a fixed-state linearization for *data* attribution) [V];
  Bae et al. 2024, unrolled differentiation for TDA ([2405.12186](https://arxiv.org/abs/2405.12186)) [V].
  **PARTIAL.** I found no closed-form AtP over (m, v) used for spike localization.

### T7 · Patching hygiene from interpretability
- **Source mechanism.** Denoising and noising patches answer different questions and can disagree (Zhang &
  Nanda, [2309.16042](https://arxiv.org/abs/2309.16042); Heimersheim & Nanda, [2404.15255](https://arxiv.org/abs/2404.15255))
  [V]. Subspace patching can work by activating a *dormant parallel pathway* (Makelov, Lange & Nanda, ICLR
  2024, [2311.17030](https://arxiv.org/abs/2311.17030)) [V]. Causal scrubbing tests hypotheses with
  resampling ablations (Redwood Research, Dec 2022,
  [post](https://lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing);
  author list UNVERIFIED) [V◐]. Causal tracing: Meng et al. ([2202.05262](https://arxiv.org/abs/2202.05262)) [V].
- **Mapping [I].** (a) RESEARCH §4.2's "necessity = patching Sᶜ recovers ≈0" is the same fork as noising S.
  Report it as such, alongside sufficiency. (b) Any rank-r edit of m along top-r H~ directions needs two
  controls: a random rank-r subspace (`Br`), and the same subspace patched from a *third* clean state (another
  seed's matched-normal snapshot). Without them a "recovery" can come from cancellation rather than from
  removing poison. (c) Resample ablation from matched-normal snapshots gives an on-distribution patch. It needs
  no B*, which helps A4. Check every hybrid for T8 invariant violations.
- **Prediction.** (i) For `v_inflate`, the denoise and noise rankings agree; for natural spikes they disagree on
  ≥ 1 of the top-3 tensors (redundancy). (ii) Random rank-r m patches recover ≤ 10% of the top-r H~ subspace's
  recovery. If they recover > 50%, the curvature-localization claim fails.
- **Cost.** About 0 extra (the same configurations as T3); +1 fork per tested set for the resample control.
- **Kill-search.** Same queries as T2/T6. **PARTIAL**: standard in interpretability, not applied to optimizer
  state. Xu 2026 does block surgery; I did not see these controls in the part I read (first 100k chars),
  **UNVERIFIED** for the rest.

### T8 · Analytical-redundancy parity residuals + fault signature matrix
- **Source mechanism.** FDI splits into residual generation and decision. Parity relations give residuals that
  are zero under nominal operation (Chow & Willsky, July 1984,
  [NTRS record](https://ntrs.nasa.gov/citations/19840056958); IEEE TAC venue per search, DOI **UNVERIFIED**)
  [V◐]. ABFT exploits algebraic invariants to catch silent errors (Huang & Abraham, IEEE TC C-33(6):518–528,
  1984) [V◐]. SDC in LLM training: Ma et al. 2025 ([2502.12340](https://arxiv.org/abs/2502.12340)), Altenbernd
  et al. 2026 ([2604.00726](https://arxiv.org/abs/2604.00726)), LLM-PRISM 2026 ([2604.10390](https://arxiv.org/abs/2604.10390)),
  Dixit et al. 2021 ([2102.11245](https://arxiv.org/abs/2102.11245)) [V].
- **Mapping [I].** Residuals: r_v = v_t − β2v_{t−1} − (1−β2)g_t² and r_m = m_t − β1m_{t−1} − (1−β1)g_t. For honest
  Adam they are zero to fp32 rounding regardless of the data. They need the post-clip g_t, read through a
  pre-optimizer-step hook. **Invariant** (Cauchy–Schwarz, zero init, β1² < β2):
  m_i² ≤ [(1−β1)²/((1−β2)(1−β1²/β2))]·v_i, so |m_i| ≤ 1.165√v_i at (0.9, 0.95). This is derived here and needs a
  CPU unit test. Fault signature matrix: rows = {`corrupt_batch`, `repeat_ngram`, `lr_bump`, `v_inflate`,
  v bit-flip, natural}; columns = {r_v ≠ 0, r_m ≠ 0, NIS upper, NIS lower (T5), loss z, grad-norm z, update RMS}.
  Distinct rows are isolable classes. A v exponent bit-flip is a cheap second planted-poison type for the
  benchmark.
- **Prediction.** (i) |r_v|/v ≤ 1e-6 at every step of every data, LR and natural spike, and ≠ 0 exactly at the
  `v_inflate`/bit-flip step, so state corruption is isolated from data faults 100% of the time. (ii)
  Nearest-signature classification of the 3 pre-registered types on held-out seeds reaches ≥ 90% accuracy. A
  shared signature is reported as a limitation.
- **Cost.** 0 forks; one extra tensor op per step.
- **Kill-search.** Queries: "'fault detection and isolation' residual neural network training optimizer state
  loss spike"; "'algorithm-based fault tolerance' deep learning training optimizer checksum silent error
  detection Adam". Found: ABFT for attention ([2507.16676](https://arxiv.org/abs/2507.16676)) [V]; Altenbernd's
  detector of harmful updates + step recompute [V]; AutoTrainer's symptom classes (ICSE 2021,
  [PDF](https://people.cs.umass.edu/~shiqingma/papers/2021/Zhang2021ICSE.pdf)) [V◐]. One snippet says ABFT does
  not readily extend to nonlinear operations [V◐]; the Adam *moment* recurrences are linear, so the gap is real.
  **NEW** (small).

### T9 · Anti-windup lens: post-spike drift does not depend on the clip threshold
- **Source mechanism.** An integrator behind a saturating actuator winds up. The fixes are back-calculation
  (feed the saturation excess back) and conditional integration (Åström & Rundqwist, ACC 1989, pp. 1693–1698,
  [summary](https://en.wikipedia.org/wiki/Integral_windup)) [V◐].
- **Mapping [I].** m is a leaky integrator of g and v of g². On coordinate i where the spike dominates both,
  the normalized update after k steps is m_i/√v_i ≈ ((1−β1)/√(1−β2))·(β1/√β2)^k·sign(g_s,i) = 0.447·0.923^k at
  (0.9, 0.95), with an e-fold of ≈12.5 steps. **It is independent of the spike's magnitude.** Global grad
  clipping rescales g, so it leaves this drift unchanged for as long as dominance holds:
  (1−β2)(c·ĝ_i)² ≫ β2v_i. Conditional integration = don't integrate a flagged g (SPAM's reset and skip-step
  are crude versions); back-calculation = T1.
- **Prediction.** In `Bs` at clip 1.0 vs 0.2, the per-tensor Adam update RMS on spiked coordinates over
  s+1…s+20 agrees within ±10% while dominance holds, and decays at 0.923 ± 0.02 per step. It is falsified if the
  drift scales with c. **Use for A1:** a tighter `Bs` clip should change Δ mainly through the spike step itself.
  Pre-register `Bs` with that expectation.
- **Cost.** 0 extra if `Bs` runs at two clips (the A1 fix); telemetry only.
- **Kill-search.** Queries: "anti-windup momentum optimizer deep learning integrator windup PID optimizer";
  OpenAlex "anti-windup momentum stochastic gradient neural network training"; "Adam update clipping momentum
  windup overshoot after gradient spike…". No hits apply windup to Adam. **NEW** as a lens. The scale-invariance
  fact itself is folklore [I].

### T10 · Synthetic control for the natural-spike counterfactual (resolves A4 either way)
- **Source mechanism.** A treated unit's counterfactual is a convex combination of donors fitted on the
  pre-period, with in-space and in-time placebo inference (Abadie, Diamond & Hainmueller, JASA
  105(490):493–505, 2010, doi [10.1198/jasa.2009.ap08746](https://doi.org/10.1198/jasa.2009.ap08746)) [V].
- **Mapping [I].** Treated = a stress run with its first qualifying natural spike at t_s. Donors = same-preset
  seeds with no qualifying spike in [t_s−Δ, t_s+H]. Because `get_batch` is a pure function of the step, all
  donors see the same data: that is the common-shock structure synthetic control relies on. Fit on the
  pre-period val-loss trajectory, take the post gap as the effect, and compute a placebo permutation p-value. If
  the cross-seed alignment test says the spike is data-caused, the donors spike too and SC is unusable; use a
  data-aligned B* instead.
- **Prediction / pre-registered rule.** SC is admissible only if the pre-period RMSPE < τ/2 (τ = 0.01 nats
  proposed). Otherwise natural spikes are dropped from Gate B. The rule settles A4 before any data exist.
- **Cost.** 0 forks; needs held-out evals along the stress runs; CPU fit. Caveat: 5 seeds per preset is a tiny
  donor pool (SC studies usually have tens). Feasibility is **UNVERIFIED**.
- **Kill-search.** Query: "'synthetic control' neural network training runs counterfactual seeds…". No
  application to training runs found. **NEW.**

### T11 · Multi-stream sparse change detection
- **Source mechanism.** A mixture GLR over parallel streams, with a prior fraction p0 of affected streams,
  analytic ARL and detection-delay approximations (Xie & Siegmund, Ann. Statist. 41(2):670–692, 2013,
  [1207.2386](https://arxiv.org/abs/1207.2386)) [V].
- **Mapping [I].** Streams = 21 per-tensor standardized signals (log NIS from T5, v̄-drift). The mixture
  statistic detects sparse changes; its per-stream GLR terms say which tensors changed, so it detects and
  coarsely localizes in one pass. It complements RESEARCH §3.3's e-detectors (they combine signals but do not
  localize). Calibrate the threshold with the conformal run-level rule.
- **Prediction.** `v_inflate` on 1–2 of 21 tensors with p0 = 0.1: alarm within ≤ 5 steps, with the true tensors
  carrying the largest GLR terms, while global detectors never fire. Dense `corrupt_batch`: delay within ±1 step
  of the global detector.
- **Cost.** CPU only.
- **Kill-search.** Queries: "change point detection loss spikes LLM pretraining CUSUM Bayesian online
  changepoint…"; "per-layer sequential change detection neural network training telemetry multiple streams…".
  Found: per-layer controllers (Baveja et al. 2025, [2509.19698](https://arxiv.org/abs/2509.19698)) [V◐],
  practitioner per-layer monitors, ZClip z-scores. Nothing multi-stream on optimizer telemetry. **PARTIAL.**

### T12 · Rotated-basis editing of moments (a C3 contract constraint)
- **Source mechanism.** SOAP runs Adam in the periodically refreshed eigenbasis of Shampoo's preconditioner
  (Vyas et al., [2409.11321](https://arxiv.org/abs/2409.11321), 2024-09-17) [V]. *How* SOAP carries m/v across a
  basis change was not checked (**UNVERIFIED**). Spectra 2026 ([2602.11185](https://arxiv.org/abs/2602.11185))
  flattens the top momentum directions [V].
- **Mapping [I].** "Rank-limited repair of directions" is well-posed for m (m ← m − (uᵀm)u + (uᵀm*)u) but not
  for Adam's **diagonal** v unless u is axis-aligned. Three options: (a) coordinate-structured v edits (T3/T4);
  (b) diagonal weights u_i² (lossy); (c) for k steps, a SOAP-like update in which the subspace U uses a k×k clean
  second moment Σ_U (from B* or v̄) and the complement keeps D. The C2/C3 output contract (CLAUDE.md §7.2) has to
  choose one.
- **Prediction.** For m-poison, rank ≤ 5 direction edits of m recover as much as tensor-level m patches. For
  v-poison, option (b) recovers < 50% of T4's repair, so v-repair must be coordinate-structured.
- **Cost.** 1–2 forks per variant + Lanczos shared with C2.
- **Kill-search.** Covered by the T4/T14 queries. **PARTIAL** (SOAP/Spectra are training methods, not repair).

### T13 · Robust v̄ and robust poison score
- **Source mechanism.** M-estimation under contamination (Huber, Ann. Math. Statist. 35(1):73–101, 1964, doi
  [10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732)) [V]; MAD scaling. K2-V2 already flags loss
  spikes with a rolling-median/MAD z-score ([2512.06201](https://arxiv.org/abs/2512.06201)) [V◐].
- **Mapping [I].** (a) Make v̄ a trimmed running mean that excludes detector-flagged steps, so earlier
  sub-threshold spikes don't contaminate the reference. (b) Per coordinate or tensor, poison
  z = (log v − med(log v_normals))/(1.4826·MAD(log v_normals)) over capture's matched-normal snapshots.
- **Prediction.** On held-out matched-normal steps, the Gaussian z on log v exceeds the nominal |z| > 3 FP rate
  (0.27%) by ≥ 3×, while MAD-z stays within 2×. `v_inflate` AUROC is unchanged (≥ 0.95): robustness costs no
  power.
- **Cost.** CPU only.
- **Kill-search.** Query: "robust second moment estimate Adam median Huber winsorized squared gradient
  heavy-tailed optimizer variant". Found robust optimizers (AdaTerm-like, heavy-tail analyses) and K2-V2's
  robust loss z. **PARTIAL.**

### T14 · Warm-started thick-restart Krylov + principal angles between B0 and B*
- **Source mechanism.** Thick-restart Lanczos keeps chosen Ritz vectors across restarts at low cost (Wu &
  Simon, SIAM J. Matrix Anal. Appl. 22(2):602–616, doi
  [10.1137/S0895479898334605](https://doi.org/10.1137/S0895479898334605)) [V◐]. Warm-started power iteration on
  P^{-1/2}HP^{-1/2}, about 5 HVPs per step (Ramasinghe et al., [2606.16768](https://arxiv.org/abs/2606.16768)) [V].
  The top eigenspace is small (Gur-Ari et al., [1812.04754](https://arxiv.org/abs/1812.04754)) [V].
- **Mapping [I].** Cold top-k at the fork snapshot, then warm-start every later step and **both branches** from
  the stored Ritz vectors. B0 and B* share the pre-spike snapshot, so their top-k H~ subspaces start identical.
  Principal angles between them after the spike window measure which curvature directions rotated: a
  curvature-based localization signal next to Spectra's energy-only control.
- **Prediction.** (i) The warm start matches the cold top-k eigenvalues within 1% using ≤ 40% of the HVPs.
  (ii) Post-spike, ≤ 3 of the top-10 principal angles exceed 30°, and the coordinates they span overlap T3's
  oracle set with precision ≥ 0.5. It is falsified if the rotation is spread over all k.
- **Cost.** 30–40 HVPs cold, ~12–15 warm per snapshot.
- **Kill-search.** Query: "warm-started Lanczos tracking top Hessian eigenvectors across training steps thick
  restart…". Found Ramasinghe (top-1 only) and BSFA ([2510.25244](https://arxiv.org/abs/2510.25244), periodic
  Lanczos refresh) [V]. No B0-vs-B* principal-angle localization. **PARTIAL.**

### T15 · HVP-free directional curvature for the S_t telemetry
- **Source mechanism.** "Critical sharpness" uses fewer than 10 forward passes given the update direction, and
  reproduces progressive sharpening and EoS in OLMo-2 up to 7B (Kalra et al., [2601.16979](https://arxiv.org/abs/2601.16979),
  2026-01-23) [V]. Kim & Lee, ICML 2026 ([page](https://icml.cc/virtual/2026/72389)), give forward-pass
  finite-difference estimators along the update that absorb the optimizer state into the update direction,
  AdamW included [V◐; author names from a snippet]. Secant curvature estimates: Malitsky & Mishchenko, ICML
  2020 ([1910.09529](https://arxiv.org/abs/1910.09529)) [V].
- **Mapping [I].** Along Adam's actual update u = m̂/(√v̂+ε), estimate the preconditioned directional curvature
  as [L(w+hu) − 2L(w) + L(w−hu)]/(h²·uᵀDu) on the probe batch. Two forward passes, no double backward, so it
  runs under flash SDPA. That sidesteps A7 for *telemetry*; C2's eigensolve still needs HVPs.
- **Prediction.** Within 5% of the math-kernel HVP Rayleigh quotient, with h set by a pre-registered rule, and
  the same lead-time verdict for S_t on held-out seeds.
- **Cost.** 2 forward passes per measured step.
- **Kill-search.** Query: "sharpness estimation without Hessian-vector products gradient difference secant
  edge of stability monitor cheap". **DONE** as a tool: adopt it and cite both. Using it as a spike lead-time
  test with the preconditioned threshold is not done.

### T16 · Micro-batch robust aggregation and free SNR replicates
- **Source mechanism.** Coordinate-wise median and trimmed-mean aggregation (Yin et al., ICML 2018,
  [1803.01498](https://arxiv.org/abs/1803.01498)) [V]. Centered clipping with momentum (Karimireddy, He &
  Jaggi, ICML 2021, [2012.10333](https://arxiv.org/abs/2012.10333)) [V].
- **Mapping [I].** (a) C2's gradient-directional SNR needs per-coordinate gradient variance. The `grad_hook`
  micro-gradients are independent replicates: free at 124M (grad_accum = 40), but at the proxy (grad_accum = 1)
  they need split-batch or per-sample gradients. (b) As a prevention baseline: trim across micro-batches before
  Adam.
- **Prediction.** It prevents `repeat_ngram` spikes only when the offending documents occupy < 50% of
  micro-batches (the breakdown point), and fails on whole-batch `corrupt_batch`. Test with corrupted fraction
  ∈ {1/8, 1/2, 1}.
- **Cost.** Proxy: per-sample grads cost extra memory [A]. 124M: free.
- **Kill-search.** Query: "median of micro-batch gradients robust aggregation prevents loss spikes LLM training
  outlier batch". Only clip-based methods found (SPAM, ZClip, AdaGC). **NEW** as a baseline. Value is low: it
  prevents spikes but does not localize.

### T17 · Plasticity / continual-learning transfers
- **Sources.** Asadi et al. ([2306.17833](https://arxiv.org/abs/2306.17833)); Ellis et al. Adam-Rel
  ([2412.17113](https://arxiv.org/abs/2412.17113); includes an Adam β1=β2 baseline per a secondary summary
  [V◐]); Dohare et al., CBP with Adam ([2306.13812](https://arxiv.org/abs/2306.13812); the step count of
  reinitialized units is reset [V◐], moment handling **UNVERIFIED**); ReDo ([2302.12902](https://arxiv.org/abs/2302.12902)
  [V]; a community implementation says the Adam step-count reset is "crucial" [V◐]); CPR 2026
  ([2607.24996](https://arxiv.org/abs/2607.24996)) [V]; EWC ([1612.00796](https://arxiv.org/abs/1612.00796)) [V];
  Kunstner et al. ([1905.12558](https://arxiv.org/abs/1905.12558)) [V]; Tang et al. 2026, a spiky/non-spiky
  boundary 1−β2 = C(1−β1) ([2609.18314](https://arxiv.org/abs/2609.18314)) [V]. The original "policy collapse /
  β1=β2" Dohare paper: title and venue **UNVERIFIED**.
- **Mapping [I].** (a) Adam with β1 = β2 as a control optimizer (a cfg change). (b) Row/column (unit)
  granularity for C2/C3, with utility scores in the style of CBP or ReDo. (c) v ≈ diag empirical Fisher as an
  EWC-style regularizer for choosing among repairs. Kunstner's caveat applies: it does not measure curvature.
- **Prediction.** Whether β1 = β2 = 0.95 suppresses stress-preset spikes depends on the side of Tang's boundary
  (C = 1 at β1 = β2), so it is testable once C is read from the paper.
- **Kill-search.** **PARTIAL**: plasticity resets are routine in RL; unit-level selective m/v repair for spikes
  is not.

### T18 · Spectrum-based fault localization across the spike population
- **Source.** DeepFault (Eniser et al., [1902.05974](https://arxiv.org/abs/1902.05974)) applies SBFL to DNN
  neurons [V].
- **Mapping / prediction [I].** Ochiai suspiciousness per tensor across ≥ 30 instances plus matched normals.
  The top-3 tensors contain T3's oracle set in ≥ 70% of held-out instances; otherwise there is no population
  regularity. It orders T3's search. 0 forks. **PARTIAL.**

### T19 · Transactional step rejection / recompute-on-detect
- Equivalent to `Bskipstep_det`. Recompute-on-detect for SDC is in Altenbernd 2026
  ([2604.00726](https://arxiv.org/abs/2604.00726)) [V]. **DONE.**

---

## 3 · Novelty and citation notes for the team (found during kill-searches)

1. **Optimizer-state patching is not unclaimed.** Xu 2026 ([2608.20442](https://arxiv.org/abs/2608.20442)) does
   (w, m, v) block surgery with a costate allocation. Chimera interventions ([2607.06628](https://arxiv.org/abs/2607.06628))
   and Exact Network Surgery ([2607.16568](https://arxiv.org/abs/2607.16568)) [V◐] transplant or discard Adam
   moments as controls. None of them concern loss spikes, localization within tensors, or repair. RESEARCH.md
   §0/§4.2 and improvement.md §2.3 do not cite them yet.
2. **New detection neighbours not in RESEARCH.md:** TIC (ICML 2026, ghost AdamW updates + a hazard model at a
   fixed false-alert budget, [page](https://icml.cc/virtual/2026/80057)) [V]; Tang 2026 β1/β2 phase diagram
   ([2609.18314](https://arxiv.org/abs/2609.18314)) [V]; Kalra 2026 critical sharpness and Kim & Lee ICML 2026
   (T15).
3. **Title mismatch.** RESEARCH.md cites arXiv 2606.28116 as *Mechanism-Driven Monitors for Preemptive
   Detection…*. The arXiv API returns *Mechanism-Aligned Observability for Training Instability of Large
   Language Models* (Ruixuan Huang, 2026-06-26) [V]. It may have been retitled; check which version we cite.
4. **Hybrid-state validity.** Stewart 2026 ([2605.17590](https://arxiv.org/abs/2605.17590)) warns that
   parameter-only or memory-only corrections give internally inconsistent optimizer states. T8's |m| ≤ C√v
   check makes this concrete for every patch.

---

## 4 · Search log (all run 2026-10-22)

Semantic Scholar API: **HTTP 429 twice** (no key), so no S2 results were used. OpenAlex worked but is
keyword-noisy. WebSearch/WebFetch did the rest. Retrieved text was treated as data. Two pages contained
requests to create API keys with the user's email; those were ignored.

| # | Engine | Query | Relevant hits → verdict |
|---|---|---|---|
| 1 | WebSearch | "fault detection and isolation" residual neural network training optimizer state loss spike | FDI-for-plants only → T8 NEW |
| 2 | WebSearch | Kalman filter interpretation of Adam optimizer innovation Bayesian filtering unifies adaptive optimization | Aitchison 2020, KOALA → framework only (T5 PARTIAL) |
| 3 | WebSearch | anti-windup momentum optimizer deep learning integrator windup PID optimizer | none for Adam → T9 NEW |
| 4 | WebSearch | attribution patching linear approximation activation patching AtP* Kramar 2024 | AtP* source |
| 5 | WebSearch | silent data corruption LLM training loss spike optimizer state bit flip study | Altenbernd 2026, LLM-PRISM, Ma 2025; no optimizer-state SDC |
| 6 | WebSearch | "Understanding Silent Data Corruption in LLM Training" arXiv | 2502.12340 confirmed |
| 7 | WebSearch | LLM-PRISM silent data corruption permanent GPU faults… | 2604.10390 confirmed |
| 8 | WebSearch | delta debugging minimal failure-inducing deep learning training bug localization group testing neural network | DelBugV only → T3 NEW |
| 9 | WebSearch | causal mediation analysis neural network training dynamics natural direct indirect effect optimizer | **Xu 2026 (2608.20442)** → T2 PARTIAL |
| 10 | WebSearch | Amnesiac machine learning remove batch parameter updates unlearning Graves 2021 | weights-only undo; MSA (checkpoints, no optimizer state) |
| 11 | WebFetch | arxiv.org/abs + /html/2608.20442 | surgery over (w,m,v), costate, lag-24 allocation; no spikes |
| 12 | WebFetch | arxiv.org/abs/2506.20941; 2604.00726 | no Adam-state undo; recompute last step |
| 13 | WebSearch | "optimizer state" patching transplant Adam moments between runs causal intervention | 2607.06628, 2601.21624, 2607.16568 → T2/T7 PARTIAL |
| 14 | WebSearch | ReDo dormant neuron recycling reset Adam moments… | moment handling not stated |
| 15 | WebFetch | github.com/timoklein/redo | step-count reset "crucial" |
| 16 | WebSearch | continual backpropagation Dohare Nature 2024… Adam beta1 equal beta2… | β1=β2 only via secondary sources |
| 17 | WebSearch | "Learning from History for Byzantine Robust Optimization"… | source for T16 |
| 18 | S2 API | loss spike Adam second moment localize repair | 429 |
| 19 | OpenAlex | optimizer state transplant Adam moments causal | Xu 2026, Chimera 2026 |
| 20 | OpenAlex | loss spike optimizer state localization repair (≥2023) | nothing relevant |
| 21 | OpenAlex | anti-windup momentum stochastic gradient neural network training | nothing relevant |
| 22 | OpenAlex | normalized innovation Kalman test Adam optimizer second moment outlier | nothing relevant |
| 23 | WebSearch | Dohare "Overcoming policy collapse…" Adam beta1 beta2 equal non-stationary | title not confirmed; CPR 2026 |
| 24 | WebSearch | AutoTrainer… DeepDiagnosis DeepLocalize numerical errors | symptom classes, restart-based repair |
| 25 | WebSearch | stochastic quasi-Newton Powell damping skip curvature pair… | Powell damping ≈ SPAM's g² clip (no row: DONE-equivalent) |
| 26 | WebSearch | "interpretability illusion" subspace activation patching Makelov… | 2311.17030 |
| 27 | WebSearch | "optimal diagonal preconditioning" semidefinite programming Qu Ye | 2209.00809 / OR 2025 → T4 source |
| 28 | WebSearch | "synthetic control" neural network training runs counterfactual seeds… | none → T10 NEW |
| 29 | WebSearch | change point detection loss spikes LLM pretraining CUSUM Bayesian online changepoint… | ZClip/SPAM/AdaGC/2606.28116 only |
| 30 | WebSearch | Xie Siegmund "sequential multi-sensor change-point detection"… | 1207.2386 |
| 31 | WebSearch | "algorithm-based fault tolerance" deep learning training optimizer checksum… Adam | ABFT matmul/attention only → T8 NEW |
| 32 | WebSearch | Malitsky Mishchenko "Adaptive gradient descent without descent"… | 1910.09529 |
| 33 | arXiv API | 38 IDs (two batches) + 13 more | titles/dates verified (§5) |
| 34 | WebSearch | VanderWeele 2014 4-way decomposition doi | 10.1097/EDE.0000000000000121 |
| 35 | WebSearch | Pearl 2001 "Direct and Indirect Effects"… | 1301.2300 |
| 36 | WebSearch | Zeller Hildebrandt 2002… | TSE 28(2):183–200 |
| 37 | WebSearch | Abadie Diamond Hainmueller 2010… | 10.1198/jasa.2009.ap08746 |
| 38 | WebSearch | Chow Willsky 1984 analytical redundancy… (×2) | NTRS record; DOI unconfirmed |
| 39 | WebSearch | Huang Abraham 1984 ABFT; Åström Rundqwist 1989; Mehra 1970; ARIES; Hwang 1972 | bibliographic checks (Hwang JASA details unconfirmed) |
| 40 | WebSearch | unlearning Adam optimizer state moments remove contribution of a training step exact subtraction… | Stewart 2026 (L-BFGS) → T1 NEW |
| 41 | WebFetch | arxiv 2607.24996 | CPR; Adam state not stated |
| 42 | WebSearch | localize which layer causes loss spike bisection parameter groups… | Qiu 2025 manual toggling; 2510.21770 |
| 43 | WebSearch | sharpness estimation without Hessian-vector products gradient difference secant… | **Kalra 2026, Kim & Lee 2026** → T15 DONE |
| 44 | WebSearch | median of micro-batch gradients robust aggregation prevents loss spikes… | clip-based only; K2-V2 MAD → T16 NEW, T13 PARTIAL |
| 45 | WebFetch | icml.cc/virtual/2026/80057 (TIC); arxiv 2601.16979; 2605.17590; icml.cc/virtual/2026/72389 | detection neighbours; hybrid-state warning |
| 46 | WebSearch | continual backprop … Adam optimizer state reset … Dohare code | timestep reset only |
| 47 | WebSearch | warm-started Lanczos tracking top Hessian eigenvectors… | Ramasinghe, BSFA → T14 PARTIAL |
| 48 | WebSearch | Kunstner "Limitations of the empirical Fisher…" | 1905.12558 |
| 49 | WebSearch | Adam update clipping momentum windup overshoot after gradient spike StableAdamW… | none → T9 NEW |
| 50 | WebSearch | robust second moment estimate Adam median Huber winsorized… | AdaTerm-like only → T13 PARTIAL |
| 51 | WebSearch | rescale Adam second moment restore stability preconditioned sharpness below 2/eta… | Bai; Tang 2026 → T4 NEW |
| 52 | WebSearch | per-layer sequential change detection neural network training telemetry multiple streams… | 2509.19698 → T11 PARTIAL |
| 53 | WebSearch | "group testing" adaptive localization faulty parameters neural network model debugging… | none → T3 NEW |
| 54 | WebFetch | arxiv 2609.18314 | β1/β2 boundary 1−β2 = C(1−β1) |
| 55 | WebSearch | "policy collapse" Dohare … β1 = β2 … openreview 2023 | not confirmed |
| 56 | WebSearch | attribution patching gradient with respect to optimizer state Adam moments… | **Ding 2026 (2602.00329)** → T6 PARTIAL |
| 57 | WebSearch | early warning loss spike ratio squared gradient to second moment low g^2/v… | Bai, Wortsman upper-tail → T5 PARTIAL |
| 58 | WebSearch | retroactively remove spike gradient from Adam moments after detection… | SPAM zeroing; no subtraction → T1 NEW |
| 59 | WebSearch | Wu Simon thick-restart Lanczos; Huber 1964 doi; Chow Willsky TAC doi | bibliographic checks |
| 60 | S2 API | optimizer state surgery Adam moments transplant | 429 |

Gaps: S2 never returned. Google Scholar and IEEE Xplore were not queried directly. The kill-searches are
targeted, not systematic. Redo T1/T3/T4/T8/T9/T10 (the NEW rows) with S2 (with a key) before submission.

---

## 5 · Sources and references (arXiv titles and dates checked against the arXiv API on 2026-10-22 unless marked ◐)

- Xu 2026, *Stored in Optimizer State, Valued by Later Training* — https://arxiv.org/abs/2608.20442 (2026-08-20)
- Liu & Li 2026, *Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW* — https://arxiv.org/abs/2608.19762 (2026-08-20)
- Khanh 2026, *Cross-Trajectory Chimera Interventions…Grokking* — https://arxiv.org/abs/2607.06628 (2026-07-07)
- Sevetlidis & Pavlidis 2026, *Training Memory in Deep Neural Networks* — https://arxiv.org/abs/2601.21624 (2026-01-29)
- Khemais 2026, *Exact Network Surgery* — https://arxiv.org/abs/2607.16568 (2026-07-18)
- Stewart 2026, *Form and Function: Machine Unlearning as a Problem of Misaligned States* — https://arxiv.org/abs/2605.17590 (2026-05-17)
- Graves et al., *Amnesiac Machine Learning* — https://arxiv.org/abs/2010.10981 (2020-10-21)
- Wu et al., *DeltaGrad* — https://arxiv.org/abs/2006.14755 (2020-06-26)
- Mohan et al., ARIES, ACM TODS 1992 — https://doi.org/10.1145/128765.128770 ◐
- Pearl, *Direct and Indirect Effects*, UAI 2001 — https://arxiv.org/abs/1301.2300
- VanderWeele, *A unification of mediation and interaction: a 4-way decomposition*, Epidemiology 2014 — https://doi.org/10.1097/EDE.0000000000000121 · https://pmc.ncbi.nlm.nih.gov/articles/PMC4220271
- Vig et al., *Causal Mediation Analysis for Interpreting Neural NLP* — https://arxiv.org/abs/2004.12265 (2020-04-26)
- Zeller & Hildebrandt, *Simplifying and Isolating Failure-Inducing Input*, TSE 2002 — https://www.st.cs.uni-saarland.de/papers/tse2002/tse2002.pdf ◐
- Aldridge, Johnson & Scarlett, *Group Testing: An Information Theory Perspective* — https://arxiv.org/abs/1902.06002 (2019-02-15)
- Baldassini et al., *The Capacity of Adaptive Group Testing* — https://arxiv.org/abs/1301.7023 (2013-01-29)
- Wang et al., SuSiE, JRSS-B 2020 — https://doi.org/10.1111/rssb.12388 ◐ · https://www.biorxiv.org/content/10.1101/501114.full.pdf
- Elsaleh & Katz, DelBugV — https://arxiv.org/abs/2305.18558 ◐
- Qu et al., *Optimal Diagonal Preconditioning* — https://arxiv.org/abs/2209.00809 (2022-09-02) · Operations Research 2025 ◐ https://pubsonline.informs.org/doi/fpi/10.1287/opre.2022.0592
- Cohen et al., *Adaptive Gradient Methods at the Edge of Stability* — https://arxiv.org/abs/2207.14484
- Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam* — https://arxiv.org/abs/2506.04805 · https://proceedings.mlr.press/v306/bai26o.html
- Tang et al. 2026, *Beyond Quadratic Loss: The Stability Phase Diagram of Adam* — https://arxiv.org/abs/2609.18314 (2026-09-16)
- Mehra 1970, IEEE TAC 15(2):175–184 ◐ — catalogue record https://swov.nl/en/publicatie/identification-variances-and-adaptive-kalman-filtering
- Aitchison, *Bayesian filtering unifies adaptive and non-adaptive neural network optimization methods* — https://arxiv.org/abs/1807.07540 · NeurIPS 2020 https://papers.neurips.cc/paper_files/paper/2020/file/d33174c464c877fb03e77efdab4ae804-Paper.pdf
- Zhuang et al., AdaBelief — https://arxiv.org/abs/2010.07468
- Wortsman et al., *Stable and low-precision training…* — https://arxiv.org/abs/2304.13013
- Huang et al., SPAM — https://arxiv.org/abs/2501.06842
- Nanda, *Attribution Patching* — https://neelnanda.io/attribution-patching ◐
- Syed et al., *Attribution Patching Outperforms Automated Circuit Discovery* — https://arxiv.org/abs/2310.10348
- Kramár et al., AtP* — https://arxiv.org/abs/2403.00745
- Ding et al. 2026, *In-Run Data Shapley for Adam Optimizer* — https://arxiv.org/abs/2602.00329 (2026-01-30)
- Bae et al., *Training Data Attribution via Approximate Unrolled Differentiation* — https://arxiv.org/abs/2405.12186
- Zhang & Nanda, *Towards Best Practices of Activation Patching* — https://arxiv.org/abs/2309.16042
- Heimersheim & Nanda, *How to use and interpret activation patching* — https://arxiv.org/abs/2404.15255
- Makelov, Lange & Nanda, *Is This the Subspace You Are Looking for?* — https://arxiv.org/abs/2311.17030
- Redwood Research, *Causal Scrubbing* (2022) — https://lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing ◐
- Meng et al., ROME — https://arxiv.org/abs/2202.05262
- Chow & Willsky 1984, analytical redundancy — https://ntrs.nasa.gov/citations/19840056958 ◐
- Huang & Abraham 1984, ABFT, IEEE TC C-33(6) ◐ — cited via https://arxiv.org/abs/0806.3121
- Titopoulos et al., *Custom ABFT for Attention Layers* — https://arxiv.org/abs/2507.16676
- Dixit et al., *Silent Data Corruptions at Scale* — https://arxiv.org/abs/2102.11245
- Ma et al., *Understanding Silent Data Corruption in LLM Training* — https://arxiv.org/abs/2502.12340
- Altenbernd, Wiesner & Kao, *Exploring Silent Data Corruption…* — https://arxiv.org/abs/2604.00726 (2026-04-01)
- Tyagi et al., LLM-PRISM — https://arxiv.org/abs/2604.10390 (2026-04-12)
- Zhang et al., AutoTrainer, ICSE 2021 — https://people.cs.umass.edu/~shiqingma/papers/2021/Zhang2021ICSE.pdf ◐
- Åström & Rundqwist 1989, *Integrator Windup and How to Avoid It* ◐ — https://en.wikipedia.org/wiki/Integral_windup
- Abadie, Diamond & Hainmueller 2010 — https://doi.org/10.1198/jasa.2009.ap08746 · https://dspace.mit.edu/handle/1721.1/59447
- Xie & Siegmund 2013 — https://arxiv.org/abs/1207.2386 · https://projecteuclid.org/euclid.aos/1366980561
- Shin, Ramdas & Rinaldo, E-detectors — https://arxiv.org/abs/2203.03532
- Adams & MacKay, BOCPD — https://arxiv.org/abs/0710.3742
- Baveja et al., *A Unified Noise-Curvature View of Loss of Trainability* — https://arxiv.org/abs/2509.19698
- Vyas et al., SOAP — https://arxiv.org/abs/2409.11321
- Huang et al., Spectra — https://arxiv.org/abs/2602.11185
- Huber 1964 — https://doi.org/10.1214/aoms/1177703732
- K2-V2 — https://arxiv.org/abs/2512.06201
- Wu & Simon, thick-restart Lanczos — https://doi.org/10.1137/S0895479898334605 ◐
- Ramasinghe et al., *Taming Curvature* — https://arxiv.org/abs/2606.16768
- Gur-Ari et al., *Gradient Descent Happens in a Tiny Subspace* — https://arxiv.org/abs/1812.04754
- Zhou et al., BSFA — https://arxiv.org/abs/2510.25244
- Kalra et al. 2026, *A Scalable Measure of Loss Landscape Curvature…* — https://arxiv.org/abs/2601.16979 (2026-01-23)
- Kim & Lee, ICML 2026, *A Simple and Efficient Measure of Loss Landscape Curvature* — https://icml.cc/virtual/2026/72389 ◐
- Malitsky & Mishchenko — https://arxiv.org/abs/1910.09529
- Kim, TIC, ICML 2026 — https://icml.cc/virtual/2026/80057
- Huang 2026 (arXiv 2606.28116), *Mechanism-Aligned Observability for Training Instability of LLMs* — https://arxiv.org/abs/2606.28116
- Yin et al., Byzantine-robust distributed learning — https://arxiv.org/abs/1803.01498
- Karimireddy, He & Jaggi — https://arxiv.org/abs/2012.10333
- Asadi et al. — https://arxiv.org/abs/2306.17833 · Ellis et al., Adam-Rel — https://arxiv.org/abs/2412.17113
- Dohare et al., *Maintaining Plasticity in Deep Continual Learning* — https://arxiv.org/abs/2306.13812
- Sokar et al., ReDo — https://arxiv.org/abs/2302.12902 · community code https://github.com/timoklein/redo
- McCutcheon et al., CPR — https://arxiv.org/abs/2607.24996
- Kirkpatrick et al., EWC — https://arxiv.org/abs/1612.00796
- Kunstner et al. — https://arxiv.org/abs/1905.12558
- Eniser et al., DeepFault — https://arxiv.org/abs/1902.05974
- Qiu et al., *Why Low-Precision Transformer Training Fails* — https://arxiv.org/abs/2510.04212
- Liang et al., *Neural Group Testing* — https://arxiv.org/abs/2011.10704
