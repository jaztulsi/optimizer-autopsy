# 02 · Method catalogue: localizing and repairing poisoned Adam state

*Method Hunter, Phase 2 · written 2026-10-22 · target ICML 2027 (deadline ~late Jan 2027) · nothing here has been run.*

**Labels.** **[V]** = verified this session against the primary page or abstract (arXiv API, proceedings page, or full
text where noted). **[I]** = inferred by me from verified material or from repo code I read. **[A]** = assumed
(mostly cost numbers, see §0). **UNVERIFIED** = I could not confirm it. Novelty statements ("no hit") cover only the
targeted searches in §6. They are not a systematic review.

**Repo facts used (read this session) [V].** `run_fork` exposes `intervention` (once, at fork start), `pre_step`
(top of every step) and `pre_opt_step` (on the final gradient just before `opt.step()`)
(`research/harness/fork.py:29-83`). Proxy nanoGPT has **21 named tensors**: wte, wpe, ln_f, and 3 layers × {ln_1,
c_attn, attn.c_proj, ln_2, c_fc, mlp.c_proj}. The 124M model has **75**. wte is tied to lm_head
(`research/model/nanogpt.py:97-107`). At proxy size wte is 50304×192 ≈ 9.66M of ≈11M parameters, about 88% of all
state coordinates [I]. So row-level granularity in wte matters.

---

## 0 · Cost model (all [A] unless marked)

| Quantity | 11M proxy | 124M | Basis |
|---|---|---|---|
| Optimizer step on MI300X | ≈0.02 s (batch 64) | ≈5 s (grad_accum 40) | P100 measured/extrapolated 0.25–0.30 s and 64–68 s (`step_timer_results.md`) [V], ÷ V7 §4.1's assumed 10–25× MI300X speedup [A] |
| **One fork `F`** (incl. val eval) | k=200 steps → **≈0.1 GPU-min** | k=50 steps → **≈4 GPU-min** (k=200 → ≈17 min) | [A] |
| One HVP (math SDPA, one probe batch) | ≈0.05 s | ≈0.3–0.6 s | ≈2–4× one fwd+bwd for double-backward [I]. Math attention is required (improvement.md A7) |
| Snapshot (w, m, v) fp32 | ≈130 MB | ≈1.5–2 GB | params × 12 bytes [I] |

**Consequence [I]:** at 124M, a fork costs about 40× a proxy fork. Any localizer that needs G forks per instance is
proxy-only unless G ≲ 20 or the search is adaptive (A8). HVP- and gradient-based scores are nearly free at both
scales.

---

## A · Localization: which directions of (m, v) are poisoned

| ID | Method family | One-line mechanism | Closest citation (link, date) | Cost / instance: forks · HVPs · GPU-min 11M / 124M | Effort (p-h) | In loss-spike lit? | Expected payoff |
|---|---|---|---|---|---|---|---|
| A1 | **Per-tensor telemetry ranking** (update-RMS ρ, max g²/v̂, v̄-drift ‖v−v̄‖/‖v̄‖) | Rank groups at t0 by stale-v statistics that `capture.tensor_signals` already logs | Wortsman et al. [2304.13013](https://arxiv.org/abs/2304.13013) (2023-04-25) [V]; SPAM [2501.06842](https://arxiv.org/abs/2501.06842) (2025-01-12) [V] | 0 · 0 · ≈0 / ≈0 | 2 | **Yes, as detection.** Wortsman traces spikes to the patch-embedding layer (qualitative, layer-level) [V Q23] | **Mandatory control.** It wins `v_inflate` trivially, because the plant *is* a v-drift (see C4 decoys) |
| A2 | **Gradient SNR (GSNR)** per coordinate/group | E[g]²/Var[g] from micro-batch grads (via `grad_hook`) or m̂²/(v̂−m̂²). Low-SNR coordinates are where v tracks noise | Liu et al. GSNR [2001.07384](https://arxiv.org/abs/2001.07384) (2020-01-21) [V]; Balles & Hennig [1705.07774](https://arxiv.org/abs/1705.07774) (2017-05-22) [V] | 0 · 0 · <0.1 / ≈0 (the 40 micro-batches of a 124M step come free) | 4 | No hit (Q1, Q23) [I] | Medium. Says where v is noise-driven, not whether that state is *harmful*. Note: the repo's `snr.py` stub defines `abs(m)/(√v+ε)`, which is not GSNR (improvement.md C7) |
| A3 | **Gradient-direction curvature, decomposed by group** | λ_grad = gᵀH̃g/‖g‖² split exactly as λ_S = g_S·(H̃g)_S/‖g‖². The groups with the largest λ_S push the step past 2/η | Bai et al. ICML 2026 [2506.04805](https://arxiv.org/abs/2506.04805) (2025-06-05; [PMLR v306](https://proceedings.mlr.press/v306/bai26o.html)) [V], global predictor only | 0 · 1–4 · <0.1 / ≈0.05 | 8 (HVP via `torch.func` under `sdpa_kernel(MATH)` + bitwise check) | **Global version yes** (Bai). Per-group decomposition: no hit (Q1, Q10) [I] | **High.** The same numbers set C3's repair magnitude (B8/B9). Caveat: cross-terms make some λ_S negative |
| A4 | **Lanczos top-k of H̃ + per-group eigenvector mass** | k=10–20 eigenpairs of D^-½HD^-½. Score each group by the fraction of ‖u_i‖² it carries, weighted by λ_i/(2/η) | Cohen et al. [2207.14484](https://arxiv.org/abs/2207.14484) (2022-07-29) [V]; Ghorbani et al. [1901.10159](https://arxiv.org/abs/1901.10159) (2019-01-29) [V]; **module-mass of the top Hessian eigenvector**: Li, Zhou, Xu [2607.21005](https://arxiv.org/abs/2607.21005) (2026-07-23) [V, full text §6.2]; online power iteration: Ramasinghe et al. [2606.16768](https://arxiv.org/abs/2606.16768) (2026-06-15) [V] | 0 · 40–100 per probe batch, 1–4 batches · 0.1–0.3 / 0.5–4 | +8 over A3 (curvlinops 3.0.1 already pinned in `requirements-dev.txt`) | **Partial.** 2607.21005 decomposes the top eigenvector by module (plain H, synthetic 3x→x model) and then turns off weight decay on MLP only (187M AdamW). It does not edit m/v [V] | **High credibility, real risk.** Gradient/top-eigenspace alignment can be spurious (Song et al. [2405.16002](https://arxiv.org/abs/2405.16002), 2024-05-25 [V]). Tang, Chen & Liu argue spikes come from *superquadratic* geometry beyond the Hessian ([2609.18314](https://arxiv.org/abs/2609.18314), 2026-09-16 [V]). Must be validated causally (A7/A8) |
| A5 | **Hutchinson diagonal / per-group trace of H̃** | z ⊙ H̃z with Rademacher z gives a coordinate-resolution curvature map; Hutch++ reduces variance | AdaHessian [2006.00719](https://arxiv.org/abs/2006.00719) (2020-06-01) [V]; Sophia [2305.14342](https://arxiv.org/abs/2305.14342) (2023-05-23) [V]; Hutch++ [2010.09649](https://arxiv.org/abs/2010.09649) (2020-10-19) [V] | 0 · 10–30 · <0.1 / 0.1–0.5 | +3 over A3 | No hit [I] | Medium. Gives the only row/coordinate-level curvature; noisy at 30 probes |
| A6 | **Fresh-gradient Fisher mismatch** r = v̂ / Ê[g²] | K zero-update gradients at the fork state estimate the empirical-Fisher diagonal that v is *supposed* to track. r≫1 means inflated v (stalled learning); r≪1 means decayed v (unstable) | Kunstner et al. [1905.12558](https://arxiv.org/abs/1905.12558) (2019-05-29) [V] (empirical-Fisher caveats); decoupling concept: Bai 2026 [V] | 0 · 0 · <0.1 / 0.2–1. Uses **held-out calibration batches** so the training stream stays aligned [I] | 3 | Aggregate decoupling in Bai; per-tensor RMS in Wortsman. Per-coordinate ratio as a *localizer*: no hit [I] | **Medium-high.** The one cheap score that detects **both poison polarities** (see §4 insight 2) |
| A7 | **Exhaustive optimizer-state patching** (oracle) | For each group S, fork from Θ⁰ with Θ⁰_S ← Θ*_S (m and v separately) and measure the recovered Δ. This is the optimizer-state analogue of activation patching | Vig et al. [2004.12265](https://arxiv.org/abs/2004.12265) (2020-04-26) [V]; Meng et al. ROME [2202.05262](https://arxiv.org/abs/2202.05262) (2022-02-10) [V]; Heimersheim & Nanda [2404.15255](https://arxiv.org/abs/2404.15255) (2024-04-23) [V] | 2G forks: proxy G=21 → 42 F ≈ **4 GPU-min**; 124M at G=12 per-layer groups → 24 F ≈ **1.6 GPU-h** | 8 (patch builder over snapshot dicts + driver; no harness change, RESEARCH §4.2) | **No.** Closest: Liu & Li inject shocks into whole channels {θ, m, v} in a small-perturbation regime, not spikes ([2608.19762](https://arxiv.org/abs/2608.19762), 2026-08-20 [V]) | **Very high.** It is the oracle C2 is scored against and the thesis test (C3) |
| A8 | **Adaptive group testing / delta debugging over groups** | Patch half the candidate groups per fork and recurse on halves that carry the effect (binary splitting; ddmin when effects interact) | Zeller & Hildebrandt, TSE 2002 [doi:10.1109/32.988498](https://doi.org/10.1109/32.988498) [V]; Cleve & Zeller [cs/0012009](https://arxiv.org/abs/cs/0012009) (2000-12-14) [V]; Aldridge et al. survey [1902.06002](https://arxiv.org/abs/1902.06002) (2019-02-15) [V]; group testing for Shapley: Jia et al. [1902.10275](https://arxiv.org/abs/1902.10275) (2019-02-27) [V] | ≈ s·⌈log₂G⌉+s forks. G=256 fine groups, s=2 → ≈18 F ≈ **2 GPU-min / 1.2 GPU-h** | +6 over A7 | **No hit** (Q11) [I] | **High.** Makes **row/head-level causal** localization affordable at 124M. Assumes roughly monotone, additive effects; test with ddmin fallback |
| A9 | **Non-adaptive compressed sensing over groups** | N random patch masks, all independent, so they batch in parallel. Regress Δ on the masks (Lasso, or COMP/DD decoding) to get support, effect sizes and pairwise interactions | Datamodels [2202.00622](https://arxiv.org/abs/2202.00622) (2022-02-01) [V]; RISE [1806.07421](https://arxiv.org/abs/1806.07421) (2018-06-19) [V]; Aldridge 2019 [V] | N≈30–60 F ≈ 3–6 GPU-min / 2–4 GPU-h | +6 | No hit [I] | Medium-high. Noise-robust and yields interactions. Costs more than A8 when s is small |
| A10 | **Shapley values over tensor groups** | Axiomatic credit with interactions; truncated-bandit sampling for top-k | Neuron Shapley [2002.09815](https://arxiv.org/abs/2002.09815) (arXiv Feb 2020; NeurIPS 2020) [V]; KernelSHAP [1705.07874](https://arxiv.org/abs/1705.07874) (2017-05-22) [V] | Exact over 8 coarse groups = 256 F ≈ 26 GPU-min proxy. KernelSHAP over 21 groups ≈ 300–500 F. 124M only for ≤5 groups (32 F ≈ 2 GPU-h) | 8 | No hit [I] | Low-medium. Too costly at fine grain. The 2-player {w, state} case is exact with 4 forks; that is C3 |
| A11 | **Attribution patching for optimizer state ("AtP-state")** | Linearize the patch: score_S = ⟨∇_{Θ_S} L_val(t+h), Θ*_S − Θ⁰_S⟩, using log-v parametrization. **h=1** "ghost update": one val gradient through one AdamW step. **h=k**: reverse-mode through k unrolled steps (an HVP per step). Reference state Θ* = B* (analysis) or v̄ / fresh g² (deployable) | Nanda, [Attribution Patching blog](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) (Feb 4; year 2023 [I]) [V]; Syed et al. [2310.10348](https://arxiv.org/abs/2310.10348) (2023-10-16) [V]; AtP* [2403.00745](https://arxiv.org/abs/2403.00745) (2024-03-01) [V]; unrolled: Maclaurin et al. [1502.03492](https://arxiv.org/abs/1502.03492) (2015-02-11) [V], SOURCE [2405.12186](https://arxiv.org/abs/2405.12186) (2024-05-20) [V]; linear response through m, v: Liu & Li 2026 [V] | h=1: 0 F · 1 HVP · ≈0 / <0.1. h=k: ≈3–5 F of compute, memory ≈ k state copies (proxy k=50 ≈ 6 GB; 124M needs k≤20 or rematerialization) · ≈0.5 / 10–30 | 6 (h=1); +20–30 (h=k) | **No** (Q8). The nearest use is TIC (ICML 2026), which applies "ghost" AdamW updates from a frozen state for *detection* in feature space, without localization or repair ([ICML page](https://icml.cc/virtual/2026/80057)) [V] | **High.** One backward replaces G forks. Publishable as "AtP for optimizer state", scored against fork patching the way Syed et al. scored AtP against ACDC. Risk: Adam's 1/√v nonlinearity gives AtP false negatives (the AtP* failure modes) |
| A12 | **Spectral / factor analysis of log(v/v̄) and of m per matrix** | SVD of the reshaped matrices, or Adafactor-style row/col factor residuals. Exposes row/column-structured poison, e.g. the wte rows of repeated-n-gram tokens | Adafactor [1804.04235](https://arxiv.org/abs/1804.04235) (2018-04-11) [V]; SlimAdam [2503.01843](https://arxiv.org/abs/2503.01843) (2025-03-03) [V]; Spectra [2602.11185](https://arxiv.org/abs/2602.11185) (2026-01-30) [V]; row-level v excess: Biswas, NeurIPS 2026 OPT workshop ([page](https://neurips.cc/virtual/2026/161497)) [V abstract] | 0 · 0 · seconds on the GPU host | 4 | Spectra analyses gradient spectra (an optimizer, not localization). Biswas edits row-level v excess (not spikes) [V] | Medium. Unsupervised and fine-grained, but magnitude-only, so it needs causal confirmation |
| A13 | **Per-tensor change-point "first mover"** | Run a per-group CUSUM / BOCPD / e-detector on telemetry. The earliest-alarming group is the candidate culprit (temporal precedence) | BOCPD [0710.3742](https://arxiv.org/abs/0710.3742) (2007-10-19) [V]; e-detectors [2203.03532](https://arxiv.org/abs/2203.03532) (2022-03-07) [V] | 0 · 0 · CPU on `telemetry.npz` | 4 | Global detectors yes; per-layer precedence only qualitatively (Wortsman, RMS 1–8 its before) [V] | Medium. Useful for natural spikes; correlational |
| A14 | **Spectrum-based fault localization across the spike population** | Ochiai/Tarantula association between "group anomalous" and "instance spiked" over many instances and normal steps | Abreu, Zoeteweij, van Gemund, TAIC-PART 2007 ([record](https://research.tudelft.nl/en/publications/on-the-accuracy-of-spectrum-based-fault-localization/)) [V]; DNN analogs: DeepLocalize [2103.03376](https://arxiv.org/abs/2103.03376) (2021-03-04) [V], AUTOTRAINER ICSE 2021 ([page](https://2021.icse-conferences.org/details/icse-2021-papers/81/AUTOTRAINER-An-Automatic-DNN-Training-Problem-Detection-and-Repair-System)) [V] | free | 3 | No. The DNN fault-localization tools target code/hyperparameter bugs and repair by changing config, not optimizer state [V] | Low-medium. A cheap population prior; correlational |
| A15 | **Learned probe** | Classifier over A1–A6 features, trained on planted poisons, tested on held-out recipe types | TIC hazard model (ICML 2026, detection) [V] | features free; training trivial | 10 | Detection-only learned models exist (TIC) [V] | **Low.** Planted→natural transfer is the very question; reviewers will call it circular |
| A16 | **Distributed alignment search over state** | Learn a rank-r rotation R so that interchanging Rᵀm (or Rᵀ log v) from B* into B0 maximizes recovery. Gradient flows through k unrolled steps | Geiger et al. DAS [2303.02536](https://arxiv.org/abs/2303.02536) (2023-03-05) [V]; illusion caveat: Makelov et al. [2311.17030](https://arxiv.org/abs/2311.17030) (2023-11-28) [V] | 100–500 unrolled backward passes ≈ 30–120 GPU-min proxy; 124M infeasible | 40 | No [I] | High novelty, high risk. The only true non-axis-aligned "direction" finder for m; AdamW's diagonal v can't take a rotated edit |
| A17 | **Data-side attribution of the spike batch** | Per-example gradient alignment (TracIn-style) shows which examples wrote the poison and into which wte rows | TracIn [2002.08484](https://arxiv.org/abs/2002.08484) (2020-02-19) [V]; Grosse et al. [2308.03296](https://arxiv.org/abs/2308.03296) (2023-08-07) [V]; OLMo 2 traced spikes to repeated-n-gram batches, without a gradient-attribution method [I] ([2501.00656](https://arxiv.org/abs/2501.00656), 2024-12-31 [V]) | per-example grads of one batch: <0.1 / ≈1 | 6 | No hit (Q15) [I] | Medium. **Gives natural spikes a data-aligned B\*** (replace only the offending examples), which closes improvement.md A4. It also links the data cause to state rows |

---

## B · Repair: what to do to the localized set S

Every repair costs **1 fork per variant per instance** (≈0.1 GPU-min proxy / ≈4 GPU-min 124M), plus any HVPs listed.

| ID | Repair | Mechanism | Selective? | Hook slot | Closest citation | Extra cost | Effort (p-h) | In loss-spike lit? | Payoff / prediction |
|---|---|---|---|---|---|---|---|---|---|
| B1 | Full m+v(+t) reset (`Bg`) | zero all moments | Global | `intervention` / `pre_step` | SPAM 2501.06842 [V]; Asadi et al. [2306.17833](https://arxiv.org/abs/2306.17833) (2023-06-30) [V] | — | 0 (built) | **Yes** (SPAM, periodic) | Required baseline; predicted ~1/√(1−β2) ≈ 4.5× overshoot (RESEARCH §4.1) |
| B2 | v-only reset | zero v | Global | same | Asadi 2023 [V] | — | 0 (built) | RL only | baseline |
| B3 | Step-counter-only reset | reset t (bias correction) | Global | `intervention` | Adam-Rel, Ellis et al. [2412.17113](https://arxiv.org/abs/2412.17113) (2024-12-22) [V] | — | 1 | No [I] | **Control:** shows repair gains aren't a bias-correction artifact |
| B4 | **Selective zero-reset of (m_S, v_S, t_S)** | masked `zero_moments` (cf. `baselines/reset.py`). torch AdamW keeps a per-parameter `step` [I, check pinned torch] | Tensor/row | `intervention` / `pre_step(at_step)` | Biswas 2026 (clearing row-level v excess ≈ full reset, non-spike) [V abstract] | — | 3 | No per-group post-spike reset found (Q9, Q22) [I] | Simplest selective repair. Zero v_S gives oversized first steps unless t_S is reset too (RAdam variance issue, [1908.03265](https://arxiv.org/abs/1908.03265) [V]) |
| B5 | **Selective v_S ← v̄_S** | restore the pre-spike reference on S only | Tensor/row | `intervention` | Global `Bvbar` is built (PR #14) | — | 2 | No [I] | High on post-spike **inflation** (`v_inflate` prediction: full recovery). On decay-polarity spikes v̄ may itself be the unstable state |
| B6 | **v_S ← mean g² from fresh micro-batches** | re-estimate from current gradients (K micro-batches via `grad_hook`) | Tensor/row or global | `pre_opt_step` at fork step | GI-Adam: Kalra & Barkeshli [2406.09405](https://arxiv.org/abs/2406.09405) (2024-06-13) [V]; Abuduweili & Liu [2412.02153](https://arxiv.org/abs/2412.02153) (2024-12-03) [V] | — | 4 | Initialization-time only, not post-spike [I] | Medium. Tests whether v̄'s history matters (improvement.md §2.3). Single-sample g² is too noisy, so average K samples |
| B7 | Zero-LR re-warm on S | K steps update m_S, v_S with lr_S = 0, then resume | Tensor | `pre_step` (per-group lr) | Ma & Yarats [1910.04209](https://arxiv.org/abs/1910.04209) (2019-10-09) [V]; Kosson et al. [2410.23922](https://arxiv.org/abs/2410.23922) (2024-10-31) [V] | +K steps | 4 | No [I] | Medium. **Alignment caveat:** the re-warm must use the aligned training steps (S frozen) or held-out calibration batches, never extra training batches (V7 core value 2) |
| B8 | **Minimal stabilizing v-scaling** (RESEARCH §5) | v_S ← a²v_S, a = λ_S/(λ*−λ_Sᶜ), λ* = (1−δ)2/η | Tensor/row | `intervention` / `pre_step` | ours; Bai recommends global β2 changes [V] | 2–4 HVP | 6 (on A3) | No [I] | High for **onset/decay polarity** (lr_bump, natural β2=0.999). **Wrong polarity for inflation:** a ≥ 1 makes an already-too-large v larger. Pair it with B9 |
| B9 | **Stability-constrained restoration** (new) | v_S(α) = v_S^(1−α) ⊙ T_S^α toward a target T (v̄ or fresh mean g²). Pick the largest α ∈ [0,1] with λ_grad(H̃(α)) ≤ λ* by 1-D bisection | Tensor/row | `intervention` | ours; generalizes B5 (α=1) and B8 | 6–8 HVP | 8 | No [I] | **High.** One operator for both polarities, with a falsifiable stability certificate. Gives the "how much" for C3 |
| B10 | **Project m off poisoned directions** | In preconditioned coordinates m̃ = D^-½m, set m̃ ← (I−UUᵀ)m̃ for localized top-k H̃ eigvecs U, then map back | **Rank-k, non-axis-aligned** | `intervention` | OGD, Farajtabar et al. [1910.07104](https://arxiv.org/abs/1910.07104) (2019-10-15) [V]; Song et al. 2405.16002: removing the dominant-subspace part of updates costs little [V]; Spectra flattens top directions every step, as an optimizer [V] | Lanczos from A4 | 6 | No as a one-shot repair [I] | Medium-high novelty. m decays in ~10 steps (β1=0.9), so the effect may be small; a clean negative result is still reportable |
| B11 | Selective ε_S increase (local damping) | caps 1/(√v+ε) on S for k steps | Tensor | needs S in its own param group in *all* branches, or a custom step [I] | Bai: lower β2 [V]; "raise ε" appears only in a secondary summary, **UNVERIFIED** in the primary text (Q19) | — | 4 | Global only | Medium; a one-knob comparison |
| B12 | Transient per-tensor update clipping (trust region) | clip RMS(Δw_S) ≤ c for k steps | Tensor | the update is formed inside `opt.step()`, so it needs a post-step correction or custom step [I] | StableAdamW (2304.13013) [V]; Adafactor update clipping [V]; LAMB [1904.00962](https://arxiv.org/abs/1904.00962) [V]; AGC [2102.06171](https://arxiv.org/abs/2102.06171) [V] | — | 4–6 | **Yes, global and always-on** (StableAdamW) | Medium. The selective, transient version is new |
| B13 | Transient β2_S / lr_S lowering | faster re-adaptation on S | Tensor | `pre_step` | Bai (global β2) [V]; Biswas: lowering β2 works worse than clearing the excess [V abstract] | — | 3 | Global only | Low-medium control |
| B14 | Rollback-and-replay with data skip | restart ~100 steps earlier, skip ~200–500 batches | Global | new trunk segment | PaLM [2204.02311](https://arxiv.org/abs/2204.02311) (2022-04-05) [V via Q24]. PaLM also found that replaying the same batches from an earlier state gave no spike: the batch × state interaction [V] | ≈ (span+k)·F | 4 | **Yes** (production practice) | Baseline only. It changes data order, so it is not B\*; report the confound. PaLM's interaction finding supports the thesis |
| B15 | Weight-only selective rollback w_S ← w_S(fork) or w*_S | tests the weight channel | Tensor | `intervention` | Lookahead [1907.08610](https://arxiv.org/abs/1907.08610) [V] (analog) | — | 2 | No [I] | Needed for C3 mediation; not a deployable repair |
| B16 | **Low-rank / factor correction of log v** | subtract the top-r singular (or row/col factor) components of log(v/v̄) on a matrix | Row/col | `intervention` | Adafactor [V]; SlimAdam [V] | — | 6 | No [I] | Medium. Pairs with A12/A8 for row-structured poison |
| B17 | Oracle patch Θ_S ← Θ*_S | transplant from B\* | Tensor/row | `intervention` | — | — | shared with A7 | — | **Ceiling.** Report every repair as a "% of oracle recovery" |

---

## C · Evaluation and causal attribution

| ID | Protocol | What it establishes | Citation | Cost | Effort (p-h) | In loss-spike lit? | Payoff |
|---|---|---|---|---|---|---|---|
| C1 | Held-out Δ vs data-aligned B\*, paired by seed; paired bootstrap; stratified bootstrap / IQM for n=4–5 seeds | the effect size itself | Koehn, EMNLP 2004 ([W04-3250](https://aclanthology.org/W04-3250/)) [V]; rliable [2108.13264](https://arxiv.org/abs/2108.13264) [V]; Bouthillier et al. [2103.03098](https://arxiv.org/abs/2103.03098) [V] | val eval per fork | 6–10 (improvement.md A2) | Spike papers report loss curves, not paired counterfactual Δ [I] | Mandatory |
| C2 | State-patching **sufficiency / necessity / minimality** | S carries the damage (suff ≈ 1), Sᶜ doesn't (nec ≈ 0), and no proper subset suffices | Zhang & Nanda [2309.16042](https://arxiv.org/abs/2309.16042) [V]; faithfulness/completeness/minimality: Wang et al. [2211.00593](https://arxiv.org/abs/2211.00593) [V] | 2 F per candidate S | 4 | No | High |
| C3 | **Weights-vs-state mediation** (2×2 factorial: w ∈ {B0, B\*} × (m, v) ∈ {B0, B\*}, at 2–3 fork times) | direct (weights) vs state-mediated damage + interaction: **the thesis test** | Pearl [1301.2300](https://arxiv.org/abs/1301.2300) (UAI 2001) [V]; Imai, Keele & Tingley 2010 [doi:10.1037/a0020761](https://doi.org/10.1037/a0020761) [V]; factorial identification of optimizer interventions: Z. Li [2607.07206](https://arxiv.org/abs/2607.07206) (2026-07-08) [V] | +2 F per fork time | 4 | Liu & Li (channel shocks, no spikes) [V] | **Very high** |
| C4 | **Planted ground truth with decoys**: P@k, R@k, AUROC/AUPRC vs `recipe.params["targets"]` | localizer accuracy on exact truth. **Add decoys**: large ‖v−v̄‖ that is causally inert (e.g. inflate v on near-zero-gradient wte rows or on ln weights), plus m-only plants, so magnitude scores can't win by construction | InterpBench [2407.14494](https://arxiv.org/abs/2407.14494) [V]; Tracr [2301.05062](https://arxiv.org/abs/2301.05062) [V] | ≈0 (plants into snapshots) | 4 | No | **High.** Blocks the "benchmark too easy" critique |
| C5 | **Repair-top-k (deletion/insertion) curves** | Repair groups in score order and plot recovery vs k against a random order. Area under the curve scores a localizer *for repair* | ROAR [1806.10758](https://arxiv.org/abs/1806.10758) [V]; RISE [V] | 5–8 F per localizer | 3 | No | High |
| C6 | Randomization / sanity tests | Scores on a noop (no-spike) fork, or with shuffled state, must not reproduce the ranking | Adebayo et al. [1810.03292](https://arxiv.org/abs/1810.03292) [V] | 1 localizer run | 2 | No | Medium; cheap reviewer-proofing |
| C7 | Recovery time with censoring | steps until val loss is within τ of B\*; Kaplan–Meier + log-rank across branches; compare with `v_inflate` k½ ≈ 158 | Kaplan & Meier 1958 [doi:10.1080/01621459.1958.10501452](https://doi.org/10.1080/01621459.1958.10501452) [V] | dense val evals (cheap at proxy) | 3 | No [I] | Medium-high. Selective and global repairs differ most in their dynamics |
| C8 | Equivalence / non-inferiority (TOST) | "repair ≡ B\* within ±0.01 nats"; "selective non-inferior to Bg" | Schuirmann 1987 [doi:10.1007/BF01068419](https://doi.org/10.1007/BF01068419) [V] | none | 2 | No | **High.** With n=10, it turns "not significant" into a positive claim |
| C9 | Multiplicity (BH-FDR over branches × types) | controls false discoveries | Benjamini & Hochberg 1995 [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) [V] | none | 1 | — | Required |
| C10 | **Edit-size Pareto** | recovery vs fraction of state edited (coordinates or rank), with magnitude-matched random edits (`Br`) | ours (minimality analog) | covered by C5 | 2 | No | High. It makes "rank-limited" a measured property |

---

## 3 · The three strongest localizer + repair combinations

### #1 — "Patch, then linearize": state patching → AtP-state → stability-constrained restoration *(recommended spine)*
**Pitch.** We bring activation patching to optimizer state. Deterministic forks let us transplant Adam's m/v between
a spiked run and its data-aligned clean twin, which shows where the damage lives and that it outlasts weight damage.
A one-backward linearization (attribution patching for optimizer state) recovers the fork-measured map at ~1/G of the
cost. A stability-certified partial restoration of only those coordinates then matches the clean run on held-out loss
while editing a small, measured fraction of the state.

- **Pieces:** A7 (proxy) + A8 (124M) oracle · A11 (h=1, then h=k) · B9 (with B4/B5 as simple variants) · C2, C3,
  C4-with-decoys, C5, C8, C10.
- **Minimal experiment.** Proxy, site 160, held-out seeds 1–4.
  - **Instances:** `lr_bump`, `corrupt_batch`, and `v_inflate` (κ=1e4 on 2 of 21 tensors, plus 2 decoys).
  - **Per instance:** 42 patch forks, 4 mediation forks, AtP scores, and 6 repair branches (B0, B\*, Bg, v-only, B9,
    `Br`). That is ≈55 F ≈ 5.5 GPU-min, so 12 instances ≈ **1.1 GPU-h**.
  - **Pre-registered success:** Spearman ρ(AtP, patching) ≥ 0.7; AUROC vs planted truth ≥ 0.9 with decoys present;
    B9 TOST-equivalent to B\* and non-inferior to Bg while editing ≤10% of coordinates; a state-mediated share > 0
    with a CI excluding 0.
  - **Scale check:** one 124M instance with A8 (≈18 F ≈ 1.2 GPU-h) to test whether the AtP rank correlation
    transfers.
- **Effort:** ≈40 p-h (h=1 path), +25 p-h for h=k.

### #2 — "One quantity localizes and repairs": decomposed Rayleigh quotient → minimal stabilizing repair *(theory-tied)*
**Pitch.** Bai et al. showed that Adam spikes when the gradient-direction preconditioned curvature crosses 2/η. We
decompose that Rayleigh quotient over optimizer-state groups. The decomposition names the culprit coordinates and also
gives the minimal v-rescaling that restores stability. That makes a falsifiable certificate: the repair should fail
exactly when the undamaged bulk λ_Sᶜ is already above threshold.

- **Pieces:** A3 + A4 · B8 → B9 (B10 m-projection as an ablation) · C4, C5, C7.
- **Minimal experiment.** Proxy, predictive spikes: `lr_bump` plus natural stress presets with β2=0.999 and no clip.
  - **At detector t0** (and at t0−k, for lead time): compute λ_S (4 HVPs) and Lanczos k=10.
  - **Repair branches:** B8/B9 vs Bg, v-only, a global ε bump, and a global β2 drop.
  - **Pre-registered prediction:** sign(λ* − λ_Sᶜ) classifies repair success with AUROC ≥ 0.8. On 2 seeds, the top
    λ_S groups must agree with the A7 oracle.
  - **Cost:** ≈10 F + ≈100 HVP ≈ 1.5 GPU-min per instance, so 20 instances ≈ **0.5 GPU-h**. At 124M, ≈1 GPU-h per
    instance.
- **Risks to pre-empt:**
  - Superquadratic geometry ([2609.18314](https://arxiv.org/abs/2609.18314)).
  - Spurious eigenspace alignment ([2405.16002](https://arxiv.org/abs/2405.16002)).
  - Bai's Ĥ has a (1−β1)/(1+β1) scale factor that our H̃ lacks (improvement.md §2.3).
  - Non-normality from the [H, D] commutator ([2605.23476](https://arxiv.org/abs/2605.23476), 2026-05-22 [V]).
- **Effort:** ≈35 p-h.

### #3 — "Bisect the state": fork group testing at row/head granularity → row-level restoration *(scales to 124M)*
**Pitch.** Fork bisection needs only O(s log G) forks to localize causally, so per-row and per-head optimizer-state
diagnosis becomes affordable at 124M. On data-caused spikes (repeated n-grams, the OLMo 2 cause class), we expect the
poison to concentrate in the embedding rows of the offending tokens. Restoring just those rows' v should recover
held-out loss, while global resets pay the ~4.5× overshoot.

- **Pieces:** A8 with split order seeded by A12 (row SVD) and A17 (data attribution) · B5-row / B16 · C4, C5, C7, C10.
- **Minimal experiment.** `repeat_ngram` at proxy (seeds 1–4) plus 1–2 instances at 124M.
  - **Groups:** G≈256 (wte rows in token buckets, per-head c_attn slices, MLP blocks).
  - **Forks to reach the oracle set:** compare exhaustive vs A9 random masks vs A8 binary splitting.
  - **Ground truth:** precision and recall against the rows A17 identifies.
  - **Repair branches:** B5-row, B16, Bg, and `Br`-row.
  - **Cost:** ≈25 F ≈ 2.5 GPU-min per proxy instance; ≈20 F ≈ **1.3 GPU-h per 124M instance**.
- **Risks:**
  - Interactions break additivity (fall back to ddmin).
  - wte is tied to lm_head, so "row" edits also touch output logits.
- **Effort:** ≈32 p-h.

**How they fit together [I].** V7 §4.3 already budgets "our repair (3 localizer variants)". The natural three are
AtP-state (#1), λ-decomposition (#2) and a magnitude/energy control (A1/A12, or the Spectra-style control from
improvement.md). Use #1 as the paper spine and #3 as the 124M path. Total GPU estimate: ≈2 GPU-h proxy + ≈6–10 GPU-h
124M, inside the 15 → 100 h envelope.

---

## 4 · Cross-cutting insights the planner should act on

1. **`v_inflate` alone is a too-easy benchmark [I].** The plant is a v-drift, so A1/A12 recover it perfectly. Add
   causally inert decoys and m-only plants (C4). Otherwise a reviewer can say the planted test only rewards magnitude.
2. **Two poison polarities [I].**
   - *Decayed* v (Bai's onset mechanism) gives instability; the fix is to damp, by raising v.
   - *Inflated* v (after the spike; `v_inflate`; Biswas's "protection") gives stalled learning; the fix is to
     restore, by lowering v.
   - B8 handles only the first. B9 handles both, and A6 tells them apart.
3. **"Directions" for v must be coordinate sets, rows or columns [I].** AdamW's v is diagonal, so a rotated subspace
   can only be edited in m (B10/A16). Word the C2/C3 claim precisely.
4. **At 124M the fork, not the HVP, is the scarce resource [A].** Exhaustive patching stays at proxy. HVP and AtP
   scores, validated by adaptive group testing on a few 124M instances, are the scalable route.
5. **New close prior art to cite (all [V] this session):**
   - [2607.21005](https://arxiv.org/abs/2607.21005): module-level mass of the top eigenvector plus a module-selective
     weight-decay intervention.
   - Biswas, NeurIPS 2026 OPT workshop: row-level v-excess editing ≈ full reset.
   - TIC, ICML 2026: ghost AdamW updates from a frozen state, used for detection.
   - [2609.18314](https://arxiv.org/abs/2609.18314): superquadratic geometry.
   - [2605.23476](https://arxiv.org/abs/2605.23476): non-normal update operator (commutator).
   - [2607.07206](https://arxiv.org/abs/2607.07206): factorial optimizer interventions.
   - [2603.16731](https://arxiv.org/abs/2603.16731): optimizer-state resets under quantization.
   - [2606.00888](https://arxiv.org/abs/2606.00888): selective optimizer warm-up for regrown sparse parameters.

   None of them localizes poisoned m/v directions after a spike, or repairs only those, with counterfactual forks.
6. **Data alignment constrains two repairs.** B7 re-warm must not consume extra training batches. B14 rollback-skip
   changes data order, so it stays a baseline with a stated confound.

## 5 · Weakest families (deprioritize)
- **A10 Shapley:** too many forks at fine grain.
- **A15 learned probe:** circular.
- **A16 DAS:** 40 p-h and an interpretability-illusion risk.
- **A14 SBFL:** correlational.
- **B13 β2 lowering:** reported weaker than clearing the excess (Biswas).

---

## 6 · Search log (2026-10-22)

| Q | Tool · query | Result |
|---|---|---|
| Q1 | WebSearch (extended) · `loss spike Adam second moment localize poisoned parameters selective reset optimizer state 2026` | Bai 2026, SPAM, 2609.18314. **No localization/repair hit** |
| Q2 | WebSearch · `"optimizer state" patching causal intervention training instability counterfactual fork` | 2607.07206, 2607.06648 (activation patching across checkpoints), pyvene. **No hit** |
| Q3 | WebSearch · Neuron Shapley Ghorbani Zou | 2002.09815 confirmed |
| Q4 | WebSearch · AutoTrainer ICSE 2021 | confirmed; repairs by config changes |
| Q5 | arXiv API id_list · 2609.18314, 2607.07206, 2606.16768, 2608.19762, 2606.28116 | none edits m/v or localizes directions |
| Q6 | Semantic Scholar API · `loss spike localization optimizer state repair` | **HTTP 429**, no data |
| Q7 | OpenAlex · `loss spike Adam second moment reset` | only spiking-neural-network noise, unusable |
| Q8 | WebSearch · `"loss spike" "attribution patching" OR "activation patching" optimizer training dynamics Adam` | **No hit**; Liu & Li closest |
| Q9 | WebSearch (extended) · per-layer optimizer-state reset after loss spike | SPAM, 2603.16731, 2403.08763. **No per-group post-spike reset** |
| Q10 | WebSearch · Hessian eigenvector localization / transformer loss spike | 2607.21005, 2303.06296, 2410.16682 |
| Q11 | WebSearch · Shapley OR group testing OR delta debugging, training divergence | DeepLocalize, cs/0012009. **No hit** |
| Q12 | arXiv API · 2607.21005, 2603.16731, cs/0012009, 2606.00888, 2405.02803, 2602.11185, 2609.17380 | abstracts checked |
| Q13 | WebFetch · arxiv.org/html/2607.21005v1 | module-mass decomposition + MLP-only weight-decay exclusion confirmed (§6.2, Fig. 9) |
| Q14 | WebSearch · learned spike predictor from telemetry | TIC (ICML 2026), 2510.04202, K2-V2 |
| Q15 | WebSearch · data attribution / TracIn for loss spikes | **No hit** |
| Q16 | WebSearch + WebFetch · TIC ICML page | detection only; no arXiv ID found |
| Q17 | arXiv API · 2510.04202, 2605.23476, 2609.29379 | monitoring / theory only |
| Q18 | WebSearch · zero-LR re-estimate of the second moment | Biswas (NeurIPS 2026 workshop), 2412.02153 |
| Q19 | WebSearch · ε increase / β2 lowering for spikes | Bai recommends lower β2 [V]; ε claim only secondary, **UNVERIFIED** |
| Q20 | WebSearch · spectrum-based fault localization, Ochiai | Abreu et al. 2006/2007 |
| Q21 | WebFetch + WebSearch · NeurIPS 2026 poster 161497 | abstract confirmed; no arXiv version found |
| Q22 | WebSearch · selective row-level second-moment edit | Biswas, SlimAdam, Ember, Coupled Adam. **No spike-repair hit** |
| Q23 | WebSearch · Fisher / per-layer spike diagnosis | **No hit**; Wortsman patch-embedding |
| Q24 | WebSearch · PaLM restart + skip | confirmed (≈100 steps back, skip 200–500 batches; replay from an earlier state did not spike) |
| Q25 | arXiv API · 2601.13160, 2410.05052 | no state repair |
| Q26–29 | arXiv API batch verification of ~70 IDs; OpenAlex DOI filter (BH, KM, Imai, TOST, Zeller); ACL Anthology W04-3250; Nanda blog | all citations above confirmed. Muon blog fetch failed (SSL), so it is not cited |

**Limits.** Semantic Scholar was rate-limited, and OpenAlex keyword search returned noise. Novelty checks therefore
rest on WebSearch plus the arXiv API, and they are not exhaustive. Redo them with Semantic Scholar before submission.
