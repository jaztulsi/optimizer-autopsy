# 12 — Novelty fact-check (Skeptic pass) for PLAN_V8 claims N1–N15

Date: 2026-10-22 (from session context; no shell was used).
Role: adversarial novelty reviewer. The aim was to find the closest prior work for each claim and to argue it down.
Method: WebSearch (standard and extended), arXiv API listing queries (`export.arxiv.org/api/query`), and WebFetch on arXiv abstract and HTML pages to confirm what each paper actually says.
Coverage limits (read these first):
- The shared WebSearch budget for this turn ran out partway through. After that, I only fetched papers by IDs I already had, to check them. I ran no new searches.
- As a result, N13, N14 and N15 got fewer search phrasings than N1–N12.
- OpenAlex and OpenReview were not queried. Semantic Scholar was not tried.
- A follow-up pass on OpenReview (ICLR/NeurIPS 2026 workshops) and OpenAlex is recommended before any claim is labeled NEW in a submission.

The "already-known" list in the brief was not re-reported as a finding. Known items are cited only where I checked their content against a claim. One case needed this: Wortsman 2023 small-scale proxies (2309.14322) is a different paper from the listed Wortsman 2304.13013, so it counts as a new finding.

---

## Summary table

| ID | Claim (short) | Verdict | Closest prior work (most damaging first) | Safe claim |
|---|---|---|---|---|
| N1 | 2×2×2 (w,m,v) cross-world factorial over cut times → damage-share curve | **PARTIAL** | Xu 2026 (2608.20442) first-moment / params+first-moment transplant; Chimera 2026 (2607.06628) donor/recipient moments; Guo 2026 (2608.24593) crossed checkpoint × future-path; Stewart 2026 (2605.17590) memory-only vs parameter-only correction vs counterfactual oracle; Ibrahim 2024 (2403.08763 App. A.4) keep-vs-discard state washes out | First time-resolved, full (w,m,v) factorial for LLM *loss spikes* |
| N2 | Poisoned (m,v) on clean weights, dose κ | **PARTIAL (core claim near-TAKEN)** | Xu 2026 (2608.20442): first moment alone, with parameters unchanged at the cut, is the causal carrier | First *dose–response* of spike-poisoned (m,v) on clean weights in pretraining |
| N3 | State-only retroactive deletion from m,v; residual splits direct vs indirect | **PARTIAL** | Stewart 2026 (2605.17590) "deletion horizon" + memory-only vs parameter-only corrections (online L-BFGS, convex); Stewart 2026b (2604.23046); SWIFT (known); Unlearning-at-Scale (known, replay); DeltaGrad (2006.14755) | Closed-form Adam-moment deletion used as a *mediation probe* for spikes; not "first state-only unlearning" |
| N4 | Signed support-limited localization in m/v; group-testing oracle; AtP over state; two-sided g²/v test | **PARTIAL** | SPAM (2501.06842): one-sided g²/V>θ detector (θ=5000); Guo 2026 (2608.24593) differentiates future loss through AdamW state; Liu & Li 2026 (known) signed response operator; AtP/AtP* (2403.00745); ACDC (2304.14997); Zhang-Bengio-Singer 2019 (1902.01996) per-layer reset oracle | The composite (causal group-testing oracle over *optimizer-state* groups + cheap AtP/innovation predictor scored against it) |
| N5 | Smallest v-edit on support s.t. λmax(Ĥ) < (1−δ)·2/η | **PARTIAL + threshold error** | Cohen et al. 2022 (2207.14484): Adam threshold is 38/η for β1=0.9, not 2/η; Kalra & Barkeshli 2024 (2406.09405) GI-Adam edits v0 to lower preconditioned sharpness; Fong & Yang 2026 (2608.20638) v as negative feedback to the boundary | Constrained, support-limited post-spike v-edit with a curvature certificate (fix the threshold first) |
| N6 | Jump-robust shadow v (bipower / median) as localizer + repair source | **PARTIAL (estimator NEW)** | TAdam (2003.00179), AdaTerm (2201.06714) robust moments; AdaShift (1810.00143) v from shifted past gradients; SPAM v-relative clipping; Kourkoutas-β, AFOR (known) | First use of bipower-variation jump-share for Adam-state localization (no ML use of BV found) |
| N7 | Bit-flip vulnerability map of Adam state; localized restore | **PARTIAL (injection map NEW)** | Altenbernd 2026 (known, 2604.00726): corrupted gradient drives v→∞ and freezes updates; TrainSDC (known) explicitly does *not* modify optimizer state; Ma et al. 2025 (2502.12340); LayerCheck 2026 (2609.27193) per-layer mixed-timestamp restore with matching optimizer state | First direct fault-injection study *into* m/v; restore part is not new |
| N8 | Recover offending batch's tokens from Δlog v of embedding rows | **PARTIAL (mechanism near-TAKEN)** | Melis et al. 2019 (1805.04049): non-zero embedding-gradient rows reveal batch words; consecutive *snapshots* diffed; FILM (2205.08514); AEGIS 2026 (2608.19534); Zanella-Béguelin 2020 (1912.07942) snapshot-diff leakage; NeuroImprint 2026 (2606.20553) treats Adam state mixing as an obstacle | Optimizer-state (not gradient/weight) channel for spike forensics; must handle tied embeddings |
| N9 | Spike-damage scaling law over width/depth, SP vs μP/CompleteP, pre-registered held-out rungs | **PARTIAL** | Wortsman et al. 2023 *small-scale proxies* (2309.14322) — LR sensitivity across scale, μParam, predicting instabilities from norm scaling; Everett 2024 (2407.05872); CompleteP (2505.01618) | Pre-registered held-out-rung prediction of *spike damage share*; avoid "scaling law" with <5 rungs |
| N10 | Fork from released Pythia optimizer states to study induced spikes at 160M | **TAKEN (as technique)** | Pythia paper (2304.01373 §3.1): resumes from "a checkpoint and optimizer state" for an intervention; `EleutherAI/neox-ckpt-pythia-*` repos exist "for research that requires access to optimizer states"; Unlearning-at-Scale (known) uses Pythia-160M step143000 | A setting, not a contribution: "we use Pythia's released states" |
| N11 | Integrity scan of optimizer.pt via \|m\| ≤ C√v | **PARTIAL (invariant TAKEN)** | Patitucci & Mokhtari 2026 (2602.10204) Remark 3.1 states exactly this bound; standard in Adam convergence proofs; no optimizer-state supply-chain scanner found | A cheap *necessary* consistency check; not a defense against an adaptive adversary |
| N12 | Spike-dominated coordinate drifts ≈5.84η regardless of spike size | **PARTIAL + overstated** | Kingma & Ba 2015 (1412.6980 §2.1) per-step bound; Patitucci & Mokhtari 2026 (2602.10204) single-spike model and carry-over at t⋆ | Closed-form *saturating* cumulative drift, (1−β1)/(√(1−β2)(1−β1/√β2))·η; "fixed regardless of size" is wrong |
| N13 | Spikes as natural availability attacks on optimizer state | **PARTIAL (framing only)** | Altenbernd 2026 (v→∞ freeze); Bai 2026 / Wortsman 2304.13013 (v too small → instability); Xu 2026 (m steers); Xiao 2026 (2606.00888) cold-start v → spikes; Shumailov 2021 (2104.09667) availability attacks via ordering | A taxonomy/analogy; claim no novelty |
| N14 | Spike phase × LR cooldown | **PARTIAL** | Wen et al. 2024 (2410.05192): "decaying near a loss spike can lead to a significant performance loss"; Hägele 2024 (2405.18392); WSD-cooldown dynamics 2025 (2508.01483) | Causal (w,m,v) decomposition of spike damage vs schedule phase |
| N15 | SpikeState dataset | **NEW (low confidence, limited search)** | LLM360 K2 (known); LLM360 Amber did *not* save optimizer states (2312.06550); Pythia neox-ckpt states (no spikes) | First public spike instances with (w,m,v,v̄) + clean twins + oracle maps |

Net: **only N15 survives as NEW**, and that verdict rests on a thin search. **N10 is TAKEN.** Every other claim is PARTIAL. Two of them, N5 and N12, also have a technical error that a reviewer will catch.

---

## Per-claim sections

### N1 — Damage-migration curve (2×2×2 cross-world factorial over w, m, v at several cut times)

**Verdict: PARTIAL.** I found no paper running a full (w,m,v) factorial at multiple cut times after an LLM loss spike. The methodology, though, is already in use: cross-run transplants of optimizer state, with each component crossed against the others.

Closest prior work:
1. **Xu 2026, "Stored in Optimizer State, Valued by Later Training: A Causal Account of Subliminal Trait Transfer"**, arXiv 2608.20442 (known, but closer than the brief implies).
   - Overlap: it transplants the first moment alone and then parameters + first moment, and shows the first moment is the causal carrier while parameters are unchanged at the cut. It also reports effects growing from 8 to 16 updates, which is a time-since-cut axis.
   - Gap: no second-moment arm, no full factorial, subliminal-trait fine-tuning rather than pretraining spikes, and no damage-share curve.
2. **Chimera 2026**, arXiv 2607.06628 (known). It runs cross-trajectory swaps with three conditions for the continuation moments (reset / recipient's / donor's). The setting is grokking.
3. **Guo 2026, "Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions"**, arXiv 2608.24593.
   - Overlap: "crossed checkpoint–future-path tests attribute this reordering to the interaction between optimizer state and near-future data". It also ablates moment-state transport.
   - Gap: no spikes, and it crosses checkpoint × data rather than w × m × v.
4. **Stewart 2026, "Form and Function: Machine Unlearning as a Problem of Misaligned States"**, arXiv 2605.17590.
   - Overlap: it benchmarks memory-only and parameter-only corrections against a counterfactual oracle. It reports a "deletion horizon": influence persists indirectly through later parameters after the contaminated memory leaves the window.
   - Gap: online L-BFGS, convex, not Adam, not spikes.
5. **Ibrahim et al. 2024, "Simple and Scalable Strategies to Continually Pre-train LLMs"**, arXiv 2403.08763, App. A.4.
   - Finding: "keeping optimizer states does not affect continual pre-training compared to discarding them" (405M, β1=0.9, β2=0.95). The authors attribute this to fast EMA decay.
   - Why it matters: this is the reviewer's strongest weapon. It predicts the m and v shares collapse to ≈0 within tens of steps. With β2=0.95 the half-life of v is ≈13.5 steps (m: ≈6.6), so the curve may be predictable a priori.

Reviewer attacks to pre-empt:
- With interactions present, "share carried by each of w, m, v" is not unique. Pre-register the decomposition (Shapley over 3 factors, or main effects from the factorial ANOVA) before running.
- Expect a reviewer to say "trivially predicted by EMA half-lives." The novel content is the *indirect* path (state → weights → lasting damage), which N3 has to measure.

Queries run: see Q1–Q4, Q17–Q19, Q34–Q37 in the log.

Recommended wording: "We give the first time-resolved causal decomposition of lasting loss-spike damage across weights, first moment and second moment, using a full 2×2×2 cross-world factorial on a bitwise-deterministic fork harness. Prior transplant studies (Xu 2026; Chimera 2026) cross subsets of these components in fine-tuning or grokking settings." Do not say "first to show optimizer state carries causal effect".

### N2 — Poisoned-state transplant with dose–response κ

**Verdict: PARTIAL. The headline "optimizer state alone causes damage on clean weights" is near-TAKEN.**

Closest prior work:
- **Xu 2026 (2608.20442).** "Transplanting only the first moment leaves parameters, hidden states, and outputs unchanged at the cut, yet it still acts as the causal carrier." That is the same logical claim (state alone is causal on unchanged weights) in another setting.
- **Chimera 2026 (2607.06628)**: donor-moment arm.
- **Xiao et al. 2026, "Memory-Efficient LLM Training with Dynamic Sparsity"**, arXiv 2606.00888. A cold-start (fresh) Adam state on regrown parameters causes loss spikes, which shows state alone can cause spikes on otherwise-fine weights.
- Counter-evidence: **Ibrahim 2024 (2403.08763 A.4)**. Foreign or discarded state washes out at β2=0.95.

Dose–response: I found no prior work with a graded poison dose κ (interpolating spiked vs clean m,v).

Queries: Q4, Q17, Q34–Q36.

Recommended wording: "We extend state-transplant causality (Xu 2026) to pretraining loss spikes and give the first dose–response curve: interpolating spike-poisoned (m, v) into clean weights at dose κ." Drop "optimizer state alone causes damage" as a novelty claim.

### N3 — State-only retroactive deletion (forward compensation of m, v; direct vs indirect split)

**Verdict: PARTIAL.**

Closest prior work:
- **Stewart 2026 (2605.17590).** The concept (memory-only vs parameter-only correction against a counterfactual oracle, with residual influence persisting *indirectly* through later parameters) is the direct-vs-indirect split. Differences: L-BFGS, convex theory, no spikes.
- **Stewart 2026b, "Shape of Memory: a Geometric Analysis of Machine Unlearning in Second-Order Optimizers"**, arXiv 2604.23046. Second-order optimizer state retains residual deleted information that first-order checks miss.
- **SWIFT (2302.06173, known)**: one-step inversion ancestor.
- **Amnesiac ML (2010.10981, known)**: subtracts stored weight updates (the weights-side analogue).
- **Unlearning-at-Scale (2508.12220, known)**: replay, bit-exact on model and optimizer state, with no closed-form state edit. Verified: it "replays", and states that optimizer state is part of the exactness target.
- **DeltaGrad** (Wu, Dobriban, Davidson 2020), arXiv 2006.14755: cached-gradient correction, SGD/GD only.
- **Bae et al. 2024, SOURCE**, arXiv 2405.12186: approximate unrolling that accounts for Adam's implicit bias. That is a first-order counterfactual removal through the trajectory, but not a state-only edit.

I found no prior Adam-specific closed-form subtraction of a past window from m and v with weights held fixed (two phrasings plus API queries).

Technical caveats to state in the plan:
- `v −= (1−β2)β2^k (g_s² − v̄)` is a *replacement* (the spike step's g² replaced by v̄), not an exact counterfactual removal.
- The edit can drive v negative where v̄ is mis-estimated, so it needs a clamp.
- If global-norm clipping fired at step s, the logged contribution must be the *clipped* g_s.

Queries: Q5, Q20–Q24, Q37.

Recommended wording: "We use closed-form forward compensation of Adam's moments (weights untouched) as a mediation probe, separating the direct (state-borne) from the indirect (weight-borne) path of spike damage. Related state-alignment views of unlearning exist for L-BFGS (Stewart 2026) and via replay for Adam (Unlearning-at-Scale)."

### N4 — Signed, support-limited localization in m/v; group-testing oracle; AtP over state; two-sided g²/v test

**Verdict: PARTIAL. Every component has prior art; only the composition is new.**

- **Detector.** SPAM (2501.06842), verified from the HTML: spike set 𝒢 = {g_i : g_i²/V_i > θ}, θ=5000, clipped to sign(g_i)·√(θV_i). That is the upper tail of N4's normalized-innovation test.
  - The lower tail (g² ≪ v, i.e. stale or frozen coordinates) relates to Topollai 2026 staleness (known) and Altenbernd 2026's v→∞ freeze (known).
  - "Two-sided" is a modest extension. Kalman-filter χ²/NIS gating is textbook. Outlier-insensitive KF papers (2309.09505, 2210.06083) note χ² tests need careful false-alarm tuning.
- **AtP over optimizer state.**
  - AtP comes from Nanda's 2023 post (https://neelnanda.io/attribution-patching), refined as AtP* (Kramár et al. 2024, arXiv 2403.00745).
  - Gradients of future loss with respect to AdamW state already appear in Guo 2026 (2608.24593), which differentiates 8-step AdamW trajectories "through the full model and optimizer state", and in Liu & Li 2026 (2608.19762, known): a linearized signed response operator of a gradient perturbation through the (w,m,v) system.
  - So "first-order attribution through optimizer state" is PARTIAL-taken. Applying it as a per-group *patch estimate* scored against a causal oracle is new.
- **Causal oracle by adaptive group testing / delta debugging.**
  - DelBugV (2305.18558) applies delta debugging to NN verifiers.
  - ACDC (Conmy et al. 2023, arXiv 2304.14997) does iterative patch-based pruning over components.
  - Zhang, Bengio, Singer 2019 "Are All Layers Created Equal?" (1902.01996) uses a per-layer re-initialization criticality oracle on weights.
  - I found no group testing over *optimizer-state* groups.
- AtP* reports that AtP has significant false negatives. Expect the cheap predictor to under-detect, and pre-register that AtP will be scored against the oracle, not assumed.

Queries: Q25–Q28.

Recommended wording: "We localize spike poison inside stored Adam moments with a causal group-testing oracle over tensor/row/head groups, and evaluate two cheap predictors against it: first-order attribution through optimizer state (cf. AtP; Guo 2026; Liu & Li 2026) and a two-sided normalized-innovation test generalizing SPAM's one-sided g²/V rule."

### N5 — Stability-certified restoration (smallest v-edit to bring λmax(Ĥ) under (1−δ)·2/η)

**Verdict: PARTIAL. As written, the threshold is likely wrong.**

- **Cohen et al. 2022, "Adaptive Gradient Methods at the Edge of Stability"**, arXiv 2207.14484. Verified abstract: "For Adam with step size η and β1 = 0.9, this stability threshold is 38/η". In general the threshold is (2+2β1)/((1−β1)η).
  - A certificate at 2/η is either ~19× too conservative, or it is defined on a differently normalized Ĥ, which the plan has to state.
  - Bai 2026 (2506.04805, known) also uses the 38 threshold.
- **Kalra & Barkeshli 2024, "Why Warmup the Learning Rate?"**, arXiv 2406.09405. GI-Adam *edits the second-moment initialization* specifically to lower preconditioned sharpness and avoid early instability. This is prior art for "edit v to control λmax(P⁻¹H)" (at init rather than post-spike, and not support-limited).
- **Fong & Yang 2026, "Adam at the Edge of Stability: Adaptive Feedback, Provable…"**, arXiv 2608.20638 (v2 Oct 2026). Second-moment adaptation is the negative-feedback mechanism that returns the dynamics to the stability boundary. So a reviewer will ask why an explicit edit beats Adam's own feedback. The answer has to be the *lasting* damage from the overshoot.
- I found no constrained minimal-norm, support-limited, eigenvalue-certified v-edit after a spike.

Queries: Q29–Q31.

Recommended wording: "A support-limited, minimum-change edit of v that certifies λmax(P⁻¹H) below (1−δ) of the Adam stability threshold (2+2β1)/((1−β1)η) (Cohen et al. 2022), generalizing GI-Adam-style second-moment control (Kalra & Barkeshli 2024) from initialization to post-spike repair." **Fix the 2/η in the plan.**

### N6 — Jump-robust "shadow v" (bipower-variation EMA of |g_t||g_{t−1}|, or median)

**Verdict: PARTIAL overall. The specific estimator looks NEW.** I found no ML or optimizer use of bipower variation in two phrasings plus a lagged-product phrasing.

Prior art on the ingredients:
- **TAdam** (Ilboudo, Kobayashi, Sugimoto 2020), arXiv 2003.00179: Student-t robust first moment.
- **AdaTerm** (Ilboudo, Kobayashi, Matsubara 2022/23), arXiv 2201.06714: t-distribution-derived robust estimates of "all the associated statistics", including the second moment.
- **AdaShift** (Zhou et al., ICLR 2019), arXiv 1810.00143: v from temporally shifted g_{t−n}. That is a lagged-gradient v (motivated by decorrelation, not jump-robustness).
- **MaxVA** (2006.11918): an alternative v-tracking rule.
- SPAM: v-relative clipping. Kourkoutas-β and AFOR (known): spike- or quality-adaptive β2.

Caveats:
- BV is jump-robust only for *isolated* jumps. A spike that spans ≥2 consecutive steps (the repo's `SpikeRecipe.width`) puts G² into the |g_t||g_{t−1}| product and defeats the estimator.
- Use a scaling constant (π/2 under Gaussian) or calibrate on normal steps.

Queries: Q32–Q34.

Recommended wording: "We introduce a bipower-variation shadow second moment (borrowed from jump-robust volatility estimation) as a zero-fork localizer and repair source. Robust-moment Adam variants (TAdam, AdaTerm) target heavy-tailed noise, not post-hoc localization."

### N7 — Bit-flip vulnerability map of Adam state

**Verdict: PARTIAL. Direct injection into m/v looks NEW; the mechanism and the restore are not.**

- **Altenbernd 2026 (2604.00726, known).** Verified: injections target GEMM HMMA only. Without clipping, "a single corrupted gradient causes the optimizer's second moment to become infinity, freezing parameter updates". That is the "v too large = frozen coordinate" mechanism, reached through the gradient.
- **TrainSDC (2608.30769, known).** Verified: "Parameters, optimizer states, and stored training data are not modified." This *strengthens* N7's novelty for direct state injection.
- **Ma et al. 2025, "Understanding Silent Data Corruption in LLM Training"** (ACL 2025), arXiv 2502.12340: optimizer-step granularity, computations rather than state.
- **LayerCheck 2026** (Sun et al., IEEE CLUSTER 2026), arXiv 2609.27193: recovery via a "mixed-timestamp composite model state" per layer with matching optimizer state, i.e. partial restore instead of full rollback. That is prior art for the "localized restore" half.
- Inference-time bit-flip attacks on weights (PrisonBreak 2412.07192, GenBFA 2411.13757) and training-time Rowhammer on weights (WBP, ECCV 2024) do not target optimizer state.

Technical errors to fix in the claim:
- An upward exponent flip that yields v=Inf is a *permanent* freeze (β2·Inf=Inf), not ~ln κ/(1−β2) steps. The recovery time is ln κ/ln(1/β2).
- A *downward* exponent flip in v produces an immediate large update (a spike), not silent poison.
- "v flips are silent" is therefore true only for upward, finite flips.

Queries: Q6–Q12.

Recommended wording: "First fault-injection map of AdamW *state* (exponent/mantissa × m/v × direction). It extends gradient-path SDC findings (Altenbernd 2026; TrainSDC, which leaves optimizer state untouched) and compares localized restore with rollback (cf. LayerCheck's per-layer restore)."

### N8 — Forensic token recovery from Δlog v of embedding rows

**Verdict: PARTIAL. The mechanism is near-TAKEN.**

- **Melis et al. 2019, "Exploiting Unintended Feature Leakage in Collaborative Learning"** (IEEE S&P), arXiv 1805.04049. The embedding gradient is non-zero only for words in the batch, and an adversary *comparing consecutive model snapshots* sees which words were present. Δlog v is the same support signal routed through v: absent rows decay by exactly log β2.
- **FILM** (Gupta et al., NeurIPS 2022), arXiv 2205.08514: recovers the bag of words from non-zero embedding-gradient rows, then reorders.
- **AEGIS 2026**, arXiv 2608.19534: "embedding gradient's row-norm sparsity directly reveals which tokens are present."
- **Zanella-Béguelin et al. 2020** (CCS), arXiv 1912.07942: snapshot differences of LMs leak training-data updates.
- **NeuroImprint** (Shi et al. 2026), arXiv 2606.20553: names Adam/AdamW "cross-step mixing" as something the attack must engineer around. That is partial counter-evidence on feasibility across many steps.
- **Habani, Refael, Lindenbaum 2026**, arXiv 2610.18294: optimizer choice changes checkpoint leakage, and Adam is the most resistant. That covers weights only, not stored state.
- Verification-of-training-data literature: Proof-of-Learning (2103.05633); Proof-of-Training-Data (Choi, Shavit, Duvenaud 2023, 2307.00682).
- Searches for "optimizer state as a training-data leakage channel" found no direct prior work (three phrasings plus an API query).

**Technical problem specific to this repo:**
- nanoGPT ties `wte.weight == lm_head.weight`. The LM-head softmax gives a *dense* gradient over every vocabulary row each step, so the support channel (non-zero rows) Melis and FILM exploit **does not exist** for tied embeddings.
- What remains is a magnitude anomaly (target/input token rows get larger g²). That is a harder, statistical recovery problem.
- Across k>1 steps between snapshots, the signal is a recency-weighted union over k·B·T tokens. A spike batch helps here, because its g² dominates.

Queries: Q13–Q16, Q38–Q39.

Recommended wording: "Stored second moments are a token-leakage channel, extending embedding-gradient leakage (Melis 2019; FILM 2022) from gradients and weight snapshots to optimizer state, and used here for forensic identification of the spike-causing batch without data logs. With tied embeddings the channel is magnitude-based, not support-based." Add a privacy note: shipping optimizer.pt leaks recent tokens.

### N9 — Spike-damage scaling law across width/depth (SP vs μP/CompleteP), pre-registered held-out rungs

**Verdict: PARTIAL.**

- **Wortsman et al. 2023, "Small-scale proxies for large-scale Transformer training instabilities"** (ICLR 2024), arXiv 2309.14322.
  - This is a different paper from the known 2304.13013, so it is a new finding for this list.
  - Content: LR-sensitivity-vs-scale curves, μParam among the mitigations, and "the scaling behavior of model activation and gradient norms" used to predict instabilities before they emerge. This is the direct ancestor of "predict spike behaviour at held-out scale".
- **Everett et al. 2024, "Scaling Exponents Across Parameterizations and Optimizers"**, arXiv 2407.05872: four parameterizations (incl. SP, μP), 14 sizes to 26.8B, epsilon underflow and Adam-atan2.
- **Dey et al. 2025, CompleteP**, arXiv 2505.01618: claims only α=1 gives stable depth scaling.
- **Diehl Martinez, Lesci, Buttery 2024** (Findings EMNLP), arXiv 2410.11451: small Pythia models converge less stably.
- I found no work on how *damage share or recovery after a spike* scales with width/depth across parameterizations.

Reviewer attack: a "scaling law" fitted on 3–4 rungs is not credible. Call it a scaling *trend*, and pre-register the functional form plus one held-out rung.

Queries: Q40–Q42.

Recommended wording: "We pre-register and test predictions of spike-damage share at a held-out scale rung under SP vs μP/CompleteP, extending small-scale instability proxies (Wortsman et al. 2023, 2309.14322) from LR sensitivity to optimizer-state-borne damage."

### N10 — Forking from publicly released optimizer states (Pythia)

**Verdict: TAKEN as a technique; the spike setting is not a contribution.**

- **Biderman et al. 2023, Pythia**, arXiv 2304.01373, §3.1 (verified via ar5iv): "we take a checkpoint and optimizer state 21B tokens (7%) prior to the end of training" and continue with an intervention (pronoun swapping) on 70M–6.9B models. That is fork-from-released-optimizer-state plus intervention, by the release authors themselves.
- **`EleutherAI/neox-ckpt-pythia-160m`** (https://huggingface.co/EleutherAI/neox-ckpt-pythia-160m): `zero_pp_rank_*_optim_states.pt`, "intended for research that requires access to optimizer states."
  - Conflict: the model card lists 154 branches, while the Pythia README says only final-step optimizer states are served at scale and intermediate ones are "on request".
  - Check that the files exist for each branch you need before planning.
- **Unlearning-at-Scale (known)** trains from Pythia-160M step143000 with bit-exact state hashes.
- Huang et al. 2024 (arXiv 2407.17817) could not reproduce Pythia's optimizer state and synthesized one. That is a useful contrast if the released shards are used directly.
- **Confound.** The final-step state sits at the end of Pythia's cosine decay (LR near minimum). Spikes induced there land in the cooldown regime, which entangles N10 with N14.

Queries: Q43–Q46.

Recommended wording: "We use EleutherAI's released ZeRO optimizer shards for Pythia-160M (as Biderman et al. 2023 did for their data intervention) to test whether proxy findings transfer to a third-party trained state." Do not list it as a contribution.

### N11 — Optimizer-state integrity scanning of shipped optimizer.pt

**Verdict: PARTIAL. The invariant is TAKEN; the application looks NEW but is weak.**

- The bound |m_t| ≤ (1−β1)/√(1−β2) · 1/√(1−β1²/β2) · √v_t is stated as **Remark 3.1** of Patitucci & Mokhtari 2026, arXiv 2602.10204 (verified: "uniform bound … valid when β2 > β1²").
  - It follows from Cauchy–Schwarz on the two EMAs and is a standard lemma behind the β1<√β2 conditions in Adam convergence proofs.
  - Do not present the inequality as a contribution.
- No supply-chain or tamper scanner for optimizer state was found (three phrasings plus an API query).
  - Closest: optimization-triggered checkpoint backdoors (arXiv 2605.20641, compiler or backend, not moments).
  - TriShield 2026 (2607.27940) uses optimizer-state entanglement as a *defense*.
  - A March 2026 practitioner post-mortem of a silent "optimizer state not loaded" resume bug (https://www.kroonen.ai/blog/genesis-optimizer-state-bug/).
- **Reviewer attacks:**
  - (i) The invariant is necessary, not sufficient. An adversary can satisfy it trivially, e.g. by inflating v to freeze coordinates, which is exactly N13's "freeze" attack and passes the check.
  - (ii) Legitimate pipelines violate it:
    - quantized or 8/4-bit states;
    - factored v (Adafactor);
    - v-only resets, including this repo's own Bvbar / v-reset baselines, where m is kept and v is zeroed so |m| > 0 = C√v;
    - SPAM resets;
    - ZeRO reshard bugs.

  So the false-positive policy must be specified.

Queries: Q47–Q50.

Recommended wording: "A cheap consistency check (the standard EMA Cauchy–Schwarz bound, cf. Patitucci & Mokhtari 2026 Rem. 3.1, plus parity residuals) for detecting corrupted or inconsistent optimizer.pt files. We do not claim robustness to adaptive adversaries."

### N12 — "Anti-windup invariance": spike-dominated coordinate drifts ≈5.84η regardless of spike size

**Verdict: PARTIAL. The claim is also overstated as written.**

- **Kingma & Ba 2015**, arXiv 1412.6980 §2.1 (not re-fetched; ar5iv returned no text): per-step bound |Δ| ≤ α(1−β1)/√(1−β2) and invariance to gradient rescaling.
- **Patitucci & Mokhtari 2026**, arXiv 2602.10204, Theorem 3.3 (verified): single-spike model g0 = M·u, then g_t = u. Defines t⋆ as the first t with (1−β2)β2^t M² ≤ 1, and lower-bounds the *post-dominance carry-over* |Δ_{t⋆}| ≥ c(1−β1)β1^{t⋆}M. That is precisely the regime the "regardless of spike size" claim ignores. They do not compute cumulative drift.
- I found no paper stating the closed-form cumulative drift Σ_k (1−β1)/√(1−β2)·(β1/√β2)^k = (1−β1)/(√(1−β2)(1−β1/√β2)) ≈ 5.84 (β=0.9/0.95; ≈31.8 for β2=0.999). It is an elementary geometric series and a reviewer will treat it that way.

Corrections:
- The dominated-phase drift *saturates* toward 5.84η as G→∞. It grows with G for moderate G, and t⋆ grows as ln(G)/ln(1/β2).
- The constant diverges as β2→β1².
- It ignores bias correction (early training) and decoupled weight decay.
- "Explains why clipping only softens spikes" holds only if per-coordinate post-clip g still dominates v. Make that a stated condition.

Queries: Q51–Q53.

Recommended wording: "Elementary but useful: under spike dominance Adam's cumulative per-coordinate drift saturates at (1−β1)/(√(1−β2)(1−β1/√β2))·η (≈5.84η for β=(0.9,0.95)), independent of spike magnitude in the limit. This explains why norm clipping reduces but cannot remove spike-induced drift. Cf. per-step bounds (Kingma & Ba 2015) and single-spike carry-over analysis (Patitucci & Mokhtari 2026)."

### N13 — "Spikes as natural availability attacks on optimizer state"

**Verdict: PARTIAL, as framing only. Each of the three legs is already documented.**

- **Freeze (v too large):** Altenbernd 2026 (known): v→∞ freezes updates.
- **Destabilize (v too small):**
  - Wortsman 2304.13013 (known): second moment underestimates g².
  - Bai 2026 (known): preconditioner decoupling.
  - Xiao et al. 2026 (2606.00888): cold-start v on regrown parameters causes spikes.
- **Steer (m poisoned):** Xu 2026 (2608.20442).
- **Attack framing:**
  - Shumailov et al. 2021, "Manipulating SGD with Data Ordering Attacks", arXiv 2104.09667 (verified): availability attacks, "a single adversarially ordered epoch can reset training progress", using only training dynamics.
  - SPAM, AdaGC (2502.11034) and GradientStabilizer (2502.17055) all frame spikes as "contaminating" optimizer state.
- I found no paper that frames *natural* spikes as availability attacks on optimizer state (two phrasings).

Queries: Q54–Q55 (thin coverage; the search budget ran out after Q57).

Recommended wording: use it as an organizing taxonomy in the discussion, not as a claimed contribution.

### N14 — Spike phase × LR cooldown (cosine, WSD)

**Verdict: PARTIAL.**

- **Wen, Li, Wang, Hall, Liang, Ma 2024, "Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective"**, arXiv 2410.05192 (verified quote): "decaying near a loss spike can lead to a significant performance loss".
- **Hägele et al. 2024**, arXiv 2405.18392: cooldown and WSD scaling.
- **"Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler" (2025)**, arXiv 2508.01483 (authors not verified in this pass): β2 choice during cooldown matters about as much as cooldown shape.
- I found no causal test of whether the decay phase erases *optimizer-state-borne* poison specifically (one phrasing; thin coverage).

Queries: Q56 (thin coverage).

Recommended wording: "We test causally, via (w,m,v) forks, whether LR decay erases spike damage and through which component, building on the observation that decaying near a spike hurts final loss (Wen et al. 2024)."

### N15 — SpikeState public dataset

**Verdict: NEW, but at low confidence because the search budget ran out before any dedicated query.** Only facts already gathered are used here:
- **LLM360 K2** (known) releases checkpoints around natural spikes.
- **LLM360 Amber** (arXiv 2312.06550) "did not manage to save the optimizer states" and saw NaN losses in 4/360 chunks. So its spikes have no state.
- **Pythia neox-ckpt** has states but no spike annotations.
- I know of no dataset combining spike instances, (w,m,v,v̄) snapshots, clean twins and oracle damage maps.

Note for reviewers: a resource is judged on utility and documentation, not novelty.

Recommended wording: "We release SpikeState …". Run an OpenReview / HF-datasets check before claiming "first".

---

## Full query log

S = WebSearch (standard), X = WebSearch (extended), A = arXiv API listing query, F = targeted WebFetch of a known ID or URL for verification.

| # | Type | Query / URL | Used for |
|---|---|---|---|
| Q1 | A | `abs:"optimizer state" AND abs:spike` (sorted by date) | N1,N2,N4 |
| Q2 | A | `abs:"optimizer states" AND abs:"loss spike"` | N1,N2 |
| Q3 | X | loss spike "optimizer state" weights causal swap which component carries damage Adam moments | N1,N2 |
| Q4 | A | `abs:"optimizer state" AND (abs:transplant OR abs:swap OR abs:swapping OR abs:interchange)` | N1,N2 |
| Q5 | A | `(abs:unlearning OR abs:"data removal") AND (abs:Adam OR abs:momentum) AND (abs:"optimizer state" OR abs:"second moment" OR abs:"moment estimates")` → 0 results | N3 |
| Q6 | A | `abs:"bit flip" AND (abs:optimizer OR abs:Adam OR abs:training) AND (abs:"silent data corruption" OR abs:"soft error" OR abs:"fault injection")` | N7 |
| Q7 | X | fault injection "optimizer state" Adam momentum bit flip training resilience silent data corruption | N7 |
| Q8 | S | "Understanding and Mitigating Hardware Failures in Deep Learning Training" optimizer state bit flip INF NaN | N7 |
| Q9 | S | "Understanding Silent Data Corruption in LLM Training" optimizer | N7 |
| Q10 | X | "bit flips" "optimizer states" OR "Adam states" training corruption vulnerability second moment exponent | N7 |
| Q11 | S | soft error resilience DNN training "momentum" "fault injection" PyTorchFI training phase optimizer weights gradients SGD Adam study | N7 |
| Q12 | S | rowhammer attack during training "optimizer" momentum buffer backdoor DeepVenom transient perturbation fine-tuning | N7,N11 |
| Q13 | X | "optimizer states" leak training data privacy "second moment" reconstruction tokens embedding | N8 |
| Q14 | S | gradient leakage embedding layer nonzero rows reveal tokens in batch federated learning language model word set recovery | N8 |
| Q15 | S | "Analyzing Information Leakage of Updates to Natural Language Models" snapshot difference | N8 |
| Q16 | X | membership inference OR data reconstruction from "Adam" "momentum" optimizer state checkpoint released "optimizer state" privacy risk | N8 |
| Q17 | F | arxiv.org/abs/2608.24593 (Guo 2026) | N1,N2,N4 |
| Q18 | F | arxiv.org/abs/2610.18294 (Habani 2026) | N8 |
| Q19 | F | arxiv.org/abs/2608.20442, 2608.19762, 2608.02091, 2603.16731, 2609.24233 (content check of known items) | N1–N4 |
| Q20 | X | machine unlearning Adam optimizer "momentum" history subtract stored gradients "exp_avg" OR "first moment" OR "second moment" remove contribution of batch | N3 |
| Q21 | S | DeltaGrad rapid retraining machine learning Adam extension momentum "unrolled" removal of training samples counterfactual trajectory | N3 |
| Q22 | F | arxiv.org/abs/2508.12220 and html v2 (Unlearning-at-Scale: replay? Pythia state source?) | N3,N10 |
| Q23 | S | "training data attribution" unrolled differentiation Adam optimizer state "second moment" counterfactual removal step SOURCE Bae 2024 | N3 |
| Q24 | S | unlearning "optimizer state" Adam moments "stale" deleted data influence remains in momentum buffers after unlearning fine-tuning resume; + F 2605.17590, 2604.23046 | N3 |
| Q25 | X | delta debugging OR "group testing" localize faulty parameters neural network training divergence bisection parameter groups | N4 |
| Q26 | S | "attribution patching" optimizer state OR "training dynamics" OR checkpoint gradient-based approximation activation patching parameters | N4 |
| Q27 | S | Adam "g^2/v" OR "squared gradient to second moment ratio" outlier detection two-sided normalized innovation Kalman filter optimizer anomaly | N4 |
| Q28 | F | arxiv.org/html/2501.06842 (SPAM clipping rule), 1902.01996, 2304.14997, 2403.00745 | N4 |
| Q29 | X | "preconditioned sharpness" Adam edit second moment restore stability below threshold "2/η" OR "38/η" intervention after spike | N5 |
| Q30 | A | `abs:"preconditioned sharpness"` → 0 results | N5 |
| Q31 | S | Kalra Barkeshli "Why Warmup the Learning Rate" GI-Adam initialize second moment preconditioned sharpness; + F 2207.14484, 2608.20638 | N5 |
| Q32 | X | "bipower variation" gradient optimizer OR "Adam" OR "second moment estimate" neural network training robust jump | N6 |
| Q33 | S | robust Adam heavy-tailed gradient outliers median second moment estimate "robust" adaptive optimizer TAdam student-t moment | N6 |
| Q34 | S | Adam second moment product of consecutive gradients "\|g_t\|\|g_{t-1}\|" OR "lag-one" OR "cross-product" variance estimator optimizer; + F 2003.00179, 2201.06714, 1810.00143 | N6 |
| Q35 | X | spike mitigation reset Adam second moment only "reset v" vs "reset m" ablation which moment matters after loss spike experiment | N1,N2 |
| Q36 | S | "mismatched optimizer state" OR "optimizer state from a different run" OR "hybrid checkpoint" weights one run moments another run experiment training dynamics; + F 2606.31591 | N1,N2 |
| Q37 | A | `abs:"second moment" AND abs:"loss spike"` | N1,N3,N5 |
| Q38 | S | Melis "Exploiting Unintended Feature Leakage in Collaborative Learning" embedding gradient words batch | N8 |
| Q39 | S | FILM "Recovering Private Text in Federated Learning of Language Models" bag of words embedding gradient; + F 2606.20553 | N8 |
| Q40 | X | scaling law loss spike damage recovery model width depth μP CompleteP instability predicted from small models | N9 |
| Q41 | A | `abs:"loss spikes" AND (abs:muP OR abs:"maximal update" OR abs:"scaling law" OR abs:"hyperparameter transfer")` | N9 |
| Q42 | S | how loss spike frequency or severity scales with model size "loss spikes" increase with scale empirical study parameterization standard vs muP attention logit growth; + F 2309.14322, 2407.05872 | N9 |
| Q43 | X | Pythia optimizer states released checkpoints ZeRO "optimizer states" huggingface neox resume training EleutherAI | N10 |
| Q44 | F | huggingface.co/EleutherAI/neox-ckpt-pythia-160m | N10 |
| Q45 | S | resume Pythia training from released optimizer states intermediate checkpoint experiment "optimizer states" Pythia continued pretraining rewarm; + F 2403.08763 (App. A.4) | N10,N1,N2 |
| Q46 | S | "neox-ckpt-pythia" paper; + F ar5iv 2304.01373 §3.1; S "Tending Towards Stability"...; S OLMo OR LLM360 Amber released checkpoints "optimizer state"... | N10,N15 |
| Q47 | X | malicious optimizer state checkpoint attack resume fine-tuning "optimizer state" poisoned supply chain backdoor | N11 |
| Q48 | A | `abs:"optimizer state" AND (abs:attack OR abs:poisoning OR abs:backdoor OR abs:tampering OR abs:integrity)` | N11 |
| Q49 | A | `all:"optimizer state" AND (all:backdoor OR all:trojan OR all:malicious)` | N11 |
| Q50 | A | `all:"optimizer states" AND (all:privacy OR all:leakage OR all:inversion OR all:reconstruction)` | N8,N11,N3 |
| Q51 | X | Adam response to single large gradient spike cumulative update "β1/√β2" OR "beta1/sqrt(beta2)" geometric sum independent of gradient magnitude bounded drift | N12 |
| Q52 | S | "A Theory on Adam Instability in Large-Scale Machine Learning" Molybog time-domain correlation update magnitude spike | N12 |
| Q53 | F | arxiv.org/abs/2602.10204 + html (Theorem 3.3, Remark 3.1) | N12,N11 |
| Q54 | X | poisoning attack exploiting Adam second moment freeze learning "availability" OR "denial-of-service" training attack adaptive optimizer vulnerability large gradient injection | N13 |
| Q55 | S | Byzantine attack federated learning adaptive optimizer Adam server state "second moment" attack slows convergence FedAdam vulnerability | N13 |
| Q56 | X | loss spike during learning rate cooldown decay phase WSD recovery "cooldown" erases instability damage optimizer state annealing spike timing; + F 2410.05192 html, 2405.18392 | N14 |
| Q57 | S | "Manipulating SGD with Data Ordering Attacks" Shumailov … → **refused: WebSearch budget exhausted**; ID verified by F 2104.09667 | N13 |
| Q58 | F | 2604.00726 html, 2608.30769 html, 2604.10390, 2502.12340, 2609.27193 (fault-injection targets; restore style) | N7 |
| Q59 | F | 2307.00682, 2103.05633 (training-data verification) | N8,N11 |

Not run (gaps): OpenReview search, OpenAlex, Semantic Scholar, Google Scholar, GitHub code search, HF datasets search (N15), extra phrasings for N13–N15.
