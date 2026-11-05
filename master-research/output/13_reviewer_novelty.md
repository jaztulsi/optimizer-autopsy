# 13 · Reviewer pass: novelty and methodology (2026-10-23)

**Author:** reviewer session (optimizer-autopsy-95), working alongside the builder session (optimizer-autopsy-42).
**Basis:** PLAN_V8, `05`, `11`, the builder's transcript (user turns since 2026-08-29), and primary-source re-reads
listed in §4.
**Nothing here has run anywhere.** All experiments are proposals for the AMD MI300X once it's handed over (no
CUDA, no Kaggle, no local runs).

Labels: **[V]** re-read at the primary source this pass · **[I]** derivation/inference · **[UNV]** not confirmed.

---

## 1 · Verdict in one paragraph

Every component of the plan has precedent, and the builder's fact-check already says so honestly. As framed in V8 the
headline ("where does damage live: w, m or v?") has a predictable answer: by end of schedule it lives in w, and the
state share decays at the EMA rate. A reviewer will call that linear algebra. The **non-obvious** question one step
over is **who wrote the weight damage**: the spike step itself (direct effect), or the optimizer's memory of the spike
over the following ~1/(1−β) steps (indirect effect, mediated by m's echo and v's stall)? Closed-form analysis predicts
that ~92% of a dominated coordinate's drift is written *after* the spike step at (0.9, 0.95), and that v-inflation is
*protective* against the m-echo. Both are testable, falsifiable and quotable. That reframe, plus a retroactive repair
whose niche is tested honestly against skip-step, is the most defensible route to ICML.

**Working title:** *Most of a Loss Spike's Damage Is Done After the Spike: A Causal Mediation Autopsy of Adam.*

---

## 2 · The mediation headline

### 2.1 Closed form (builder's derivation, checked) [I]
On a spike-dominated coordinate (g₀ = G ≫ ḡ), ignoring bias correction and ε:

- u_k/η = (1−β1)/√(1−β2) · r^k, with **r = β1/√β2**
- total drift = (1−β1)/√(1−β2)/(1−r) = **5.84η** at (0.9, 0.95), which is K7
- the fraction written after the spike step = **r** = 0.923 at (0.9, 0.95); 0.82 at (0.8, .95); 0.975 at (.95, .95);
  0.900 at (.9, .999)

**Prior art:** this is a corollary of Patitucci & Mokhtari 2602.10204, Thm 3.3 (same single-spike setup: m decays as
β1^t·M, v forgets as β2^t·M²). Their Remark 3.1 condition β2 > β1² is exactly r < 1 [V]. Cite it; claim the
*loss-space measurement on LLM spikes*, not the formula.

### 2.2 What the formula misses
1. **The v-stall channel.** Inflated v suppresses ordinary learning on the spiked coordinates for
   ≈ ln(G²(1−β2)/ḡ²)/(−ln β2) steps (≈ 31 at G/ḡ = 10, β2 = 0.95). That is opportunity-cost damage, with the opposite
   sign to the echo's extra push.
2. **m × v interaction.** The echo stays bounded (r < 1) *only because* v also holds the spike. Prediction: cleaning v
   while keeping m's echo makes things worse than doing nothing. The mediation therefore needs the 2×2 undo arms
   {none, m, v, m+v} with Shapley shares, not one undo arm.
3. **Truncation.** The series holds while the spike dominates. For moderate spikes, m stops dominating almost at once.
   State r as the G/ḡ → ∞ limit and report the measured indirect share as a curve over the κ dose arms.
4. **Loss ≠ drift.** The spike step's update may cost more loss per unit of drift than the echo. So pre-register the
   *direction* (indirect share rises with β1 across arms), not the value r.

### 2.3 Estimator
- Δ_X = L_val(X, end) − L_val(B\*, end) on held-out batches ≥ 1000 (PLAN_V8 §6.1).
- **Direct-only:** undo the window's m and v contribution at t1 = s1 (τ = 0). The spike step's own weight update is
  kept; its echo and stall are removed.
- **Indirect share** = (Δ_B0 − Δ_direct-only)/Δ_B0. The τ sweep {1, 10, 30, 60, 150} shows how fast the memory writes
  damage.
- **Undo arms at each τ:** none / m-only / v-only / m+v, giving Shapley φ_m, φ_v and the interaction.

### 2.4 Why Ibrahim and Xu don't kill it
- **Ibrahim 2403.08763 App. A.4 [V]:** continual pretraining at a dataset transition (405M, β = (0.9, 0.95)). Moments
  "quickly decay to 0"; after 1000 steps inherited moments contribute < 0.0044%. Not about spikes, and about what the
  state *holds*, not what it *wrote into w* before decaying. Use it as the null. (The phrase "washes out" does not
  appear; fix the quote in PLAN_V8 §5 K2.)
- **Xu 2608.20442 [V abstract]:** m-only transplant carries a subliminal trait (θ unchanged at the cut, 8 + 16
  updates), plus a transport/valuation split via a full-horizon costate. It is the closest methodological precedent for
  an "m-mediated" effect. Ours differs in phenomenon (spikes), channel (v-stall, m × v interaction), horizon (end of
  schedule) and identification (β1 arms). Cite it as the method's origin.

---

## 3 · Confirmatory hypotheses for P1 (Holm, at most 4)

Everything else in H1–H18 becomes exploratory (BH).

| ID | Hypothesis | Refuted if |
|---|---|---|
| **C-1 Mediation** | The indirect (state-written, post-spike) share of end-of-schedule Δ has a CI excluding 0 for ≥ 1 data-driven spike type, **and** rises monotonically across the β1 arms (0.8 < 0.9 < 0.95) | CI ∋ 0 for all types, or the ordering fails |
| **C-2 Protective v** | Undo-v-only gives Δ ≥ Δ_B0, while undo-m+v gives Δ < Δ_B0 | Undo-v-only beats B0 |
| **C-3 Signed localization (relative)** | The signed predictor beats the magnitude-only control by ≥ 0.05 AUROC vs the causal oracle, paired over instances | Difference CI ∋ 0 |
| **C-4 Retroactive niche** | State-only undo at a *detector-timed* lag is non-inferior to the best non-oracle baseline **and** beats OLMo-6σ skip-step on the same spikes | TOST fails, or skip-step catches every spike |

**Why C-4 matters most for significance.** If grad-norm skip-step fires on the spike step, the spike never enters m/v
and a retroactive tool has nothing to fix. The niche that must be tested on purpose is (a) sub-threshold multi-step
data spikes (`repeat_ngram`, width ≥ 3) where no single step trips the rule, and (b) loss-detected spikes where
detection lags. The spike menu must contain spikes in that niche, or C-4 is untestable.

---

## 4 · Fact-check results (re-read this pass)

| Item | Finding | Action for PLAN_V8 |
|---|---|---|
| OPEN-1B 2609.17380 | §3.4.2 [V]: clip 1.0; skip 5 optimizer steps when pre-clip norm > 5. **Only 2 activations in 80,957 steps** (step 1,782, norm 5.78; step 43,737, norm 6.88). Clipping on 0.16% of steady-state steps. §3.2: moments stored and hashed in FP32. Does **not** say whether skipped steps touch m/v; `spike_protocol.json` not mentioned. Table 9 shows norm 7.06 at step 247 with no activation (unreconciled) | Drop "real natural-spike moments at 1.6B". Reframe as §5.1 below. Intro: well-tuned runs spike rarely, so motivate with aggressive LR, low precision and SDC |
| Patitucci 2602.10204 | Remark 3.1 bounds the *update* (equivalent to \|m\| ≤ C√v modulo ε). Thm 3.3 = single spike. No cumulative-drift bound. Experiments include GPT-2 124M / OpenWebText | "follows from Remark 3.1"; cite Thm 3.3 for the r-law; K7's cumulative sum stays ours |
| Ibrahim 2403.08763 | See §2.4 | Fix the quote; use as the null |
| Pythia 2304.01373 | §3.1 [V]: resumes from "a checkpoint and optimizer state" and trains on "the exact same data" with an intervention. §2.7 doesn't mention releasing optimizer states | K1 must cite Pythia as precedent for the aligned-data counterfactual. Keep "160M ZeRO shards exist" only with a repo-level check |
| Wen 2410.05192 | §5.2.2 verbatim quote verified; val loss "worse by 1e-2 or even more"; recommends rollback | ✅ |
| LayerCheck 2609.27193 | IEEE CLUSTER 2026; post-training; per-layer restore "together with their matching optimizer states"; no spikes/SDC | ✅ |
| Melis 1805.04049 | [I] tied wte = lm_head → softmax makes every row nonzero; the input-side excess survives | ✅ as written |
| Biswas (OpenReview VJFfO4tLMB) | OpenReview browser check blocks automated reads | Keep [UNV]; jaz opens it in a browser |
| ICML 2027 | No CFP. Unofficial estimate: abstract Jan 16, paper Jan 22, 2027 AoE | Keep [UNV] |
| **New threat:** Tang 2609.18314 | β1/β2 mismatch and moment memory timescales set the spike *onset* phase boundaries | Cite next to the β1×β2 cross; position ours as carry-over, theirs as onset |
| **New threat:** Xu 2608.20442, Guo 2608.24593 | Both differentiate through the complete AdamW state (costate; 8-step windows) | "Adjoint/gradient through optimizer state" is TAKEN as a technique; don't propose it as new |
| K8 SpikeState | LLM360 K2-Spike-1 (now `IFM/K2-Spike-1`) has ckpt branches with **weights only** (27 shards, no optimizer files) [V: HF API tree] | Still NEW (low confidence). Cite Ling 2503.05139: spikes are "triggered by specific interactions between the data and optimizer states" |
| K9 fault injection | No study found that injects into exp_avg / exp_avg_sq. TrainSDC 2608.30769 abstract covers forward/backward only; Altenbernd 2604.00726 is closest | PARTIAL, leaning NEW. Read TrainSDC's scope section in full before claiming |

---

## 5 · Ideas this pass adds (all for AMD, none run)

### 5.1 "Replay the spike the production run skipped" (OPEN-1B)
- Fork OPEN-1B at the step-1,700 and step-43,700 checkpoints and replay through steps 1,782 and 43,737 twice: with
  the skip (the run as released) and without it.
- That gives 2 real natural-gradient events at 1.6B with an exact data-aligned B\*. It's the only natural-spike
  counterfactual in the plan, and it directly answers reviewer feedback #1 ("need at least one natural spike").
- Only fork-vs-fork determinism on AMD matters; we don't need to match their H100 hashes.
- **Cost [A]:** ≈ 4.9M tokens/step (400B / 80,957) → roughly 100+ s/step on one MI300X in bf16; ~200 steps × 2
  branches + horizon eval ≈ 10–30 GPU-h per event. T3.
- **Prerequisite:** confirm the public checkpoints include m/v and that the data order is reproducible from the
  released codebase.
- **Risk:** post-clip the update is bounded (clip 1.0), so the effect may be small. A small effect is still a
  reportable answer to "did the skip matter?"

### 5.2 Echo-aware amnesiac subtraction (builder's "counterfactual weight back-projection")
**PARTIAL.** Precedents:
- Amnesiac ML (Graves et al., AAAI 2021) subtracts stored per-batch updates.
- SWIFT 2302.06173 inverts one Adam step, weights included.
- DeltaGrad 2006.14755 replays with cached updates.

**What's new:** the batch's contribution includes its Adam echo (m/v-mediated future updates), accumulated online
over the window. The open- vs closed-loop error is measured against a bitwise skip fork.

### 5.3 Kept out deliberately
"Sparse optimal control of optimizer state via the adjoint" was considered and dropped: the gradient through AdamW
state is TAKEN (Xu, Guo). It could come back only as a *baseline localizer* ("adjoint-state AtP over the window"),
not as a claimed contribution.

---

## 6 · Scope advice
- The catalog (~140 ideas) is the right *backlog* for the user's "test many things" goal. Every row needs a P1 / P2 /
  P3 / backlog tag.
- **P1 must read as one story:** mediation (C-1, C-2) → where it sits (C-3) → how to undo it in the niche skip-step
  misses (C-4). Everything else is appendix or P2/P3.
- **Order of AMD work for P1:** T0 gate → MVE with the 2×2 undo arms at τ ∈ {0, 10, 30} on `corrupt_batch` +
  `repeat_ngram` (width 3) + `v_inflate` → β1 arms → localization → C-4 battery.

## 7 · Systematic OpenReview + OpenAlex pass (2026-10-23, second round)

**Method.**
- **OpenReview:** API v2 full-text search (`api2.openreview.net/notes/search`) on 10 queries, plus exact-title lookups.
- **OpenAlex:** `title_and_abstract.search` since 2024 on 5 queries. This used the shared anonymous budget; no API key
  was requested, because getting one means sending the user's email.
- Titles were read first, then the abstracts of every plausible threat.
- Still not exhaustive, because full texts weren't read.

### 7.1 New threats (all public since 2026-09, so they count as prior work for ICML 2027)

| Paper | Venue | What it does | Hits | Severity |
|---|---|---|---|---|
| **"Newer Is Not Always Better: The Joint Role of AdamW Moment Histories"** (OpenReview `P0oFJZXAh8`) | ICLR 2027 submission | Controlled AdamW moment transplants from 34M to 7B. A *newer m with an older v* is far more disruptive than an older coherent pair. Explained by **history coverage**: when v covers the gradients in m, a weighted Cauchy–Schwarz bound controls m/√v. Proposes REBUILD_M (keep v, re-accumulate m from zero); "1B/7B recovery with the original batches, schedules, and data cursors" | **C-2** (protective v is their coverage mechanism); K3 (transplants); K1 (aligned-cursor recovery at 1B/7B); the hybrid feasibility bound (#10) | **HIGH** |
| **"Understanding and Mitigating AdamW Loss Spikes"** (OpenReview `pAjrfibDB9`) | ICLR 2027 submission | GPT-2 355M, **19 captured spike-onset states** with replay. The fraction of coordinates newly crossing \|u\| > 1 is 170× higher at onsets. **Selectively limiting new tail entrants beats global shrinking at equal norm.** "Replay favors **sustained over onset-only** control." The method (coordinate clip + tensor-relative growth constraint) gives zero post-warmup spikes at 774M | **C-1** (sustained > onset-only is a control-side version of "damage is written after the spike"); **C-3** (selective coordinates beat global at equal norm); RML; the spike-state capture instrument | **HIGH** |
| **"Training-Data Leakage from Optimizer State in Resumable Checkpoints"** (OpenReview `seU1LYAiwm`) | ICLR 2027 submission | One checkpoint's optimizer state reveals which tokens or classes were trained on; the persistence is called "privacy memory". Two adjacent checkpoints → signed state differencing recovers the gradient, which enables gradient inversion | **A12 FTA / R3: TAKEN.** **SpikeState (K8) release risk:** releasing (w, m, v) snapshot pairs leaks training data | **HIGH** for A12; a **release-policy blocker** for P3 |
| "Transport Instability Coordinates" (TIC; OpenReview `6BWFYVa5E3`) | ICML 2026 workshop poster | Ghost AdamW transitions from a frozen optimizer state as early-warning features; PolyPythias-410M audit | H9 detector baseline (already cited) | medium |
| "Adam Moment Routing and Protection Strength in Continual Language Modeling" (OpenReview `mn75U33Dj6`) | ACL ARR Oct 2026 | Applies gradient protection to both moments vs m only vs the final update; m-only reduces forgetting ≈ 4% | Routing edits through m vs v (C-2 framing) | low |
| "Arnold's Bifurcation Principle Applied to AdamW Training: … Early Warning of Second-Moment Collapse Preceding Loss Spikes" | 2026 (OpenAlex only; not on OpenReview) | Early warning of v collapse before spikes | H9 lower-tail detector | unread, so check before claiming H9 |
| AdamMAD (Data Sci. Eng. 2026, doi 10.1007/s41019-025-00341-7) | journal | Second moment from the **mean absolute deviation** instead of g² | **BiH-Adam (A3):** a robust-scale v exists. Bipower/MedRV specifically is still not found | medium for A3 |

### 7.2 What changes
- **C-2 is no longer a novel mechanism.** "Newer Is Not Always Better" owns the coverage explanation and the
  asymmetry. Reframe C-2 as an **out-of-sample test of the coverage theory on spike-poisoned states**: undo-v-only
  creates exactly a "newer m, older v" pair. That is still confirmatory and still worth running, but say
  "replicates and extends to spikes". Asadi 2306.17833 §4.3 (deep RL, β2 = 0.999) found a **v-only reset "quite
  effective"**. That doesn't contradict us, because their m carried no spike, so coverage holds there. Cite it as the
  boundary condition.
- **C-1 must be positioned against "Understanding and Mitigating AdamW Loss Spikes".** Their claim is about
  **immediate** loss and control effectiveness. Ours is the causal decomposition of **lasting held-out** damage into
  the direct effect and the memory-written effect, with β1 identification. Use their "sustained > onset-only" result
  as converging evidence.
- **C-3's relative test** (signed > magnitude-only) remains distinct from their "tail entrants > global". Add their
  tail-entrant selector as a **named baseline localizer**.
- **A12 FTA → TAKEN.** Drop it as a contribution.
- **SpikeState release:** publishing adjacent (m, v) snapshots enables gradient inversion on the training batches.
  Release only snapshots from public-data runs (our nanoGPT stream; Pythia/Pile; OPEN-1B's public set), state the
  privacy-memory risk in the datasheet, and never release states from private data.

### 7.3 Asadi and SPAM (re-read)
- **Asadi, Fakoor & Sabach, "Resetting the Optimizer in Deep RL: An Empirical Study", arXiv 2306.17833** [V].
  - Alg. 1 resets i = 0, m = 0, v = 0.
  - §4.3 / Fig. 5: m-only "is not effective"; v-only "is quite effective"; both is best (β2 = 0.999).
- **SPAM 2501.06842** [V]: resets **both** m and v periodically; it has **no m-only or v-only ablation**. App. F says
  resetting all moments beats keeping overlapping ones. Warmup is used "to mitigate potential instability caused by
  momentum reset".

Sources (added): [Newer Is Not Always Better](https://openreview.net/forum?id=P0oFJZXAh8) ·
[Understanding and Mitigating AdamW Loss Spikes](https://openreview.net/forum?id=pAjrfibDB9) ·
[Training-Data Leakage from Optimizer State](https://openreview.net/forum?id=seU1LYAiwm) ·
[TIC](https://openreview.net/forum?id=6BWFYVa5E3) · [Adam Moment Routing](https://openreview.net/forum?id=mn75U33Dj6) ·
[AdamMAD](https://doi.org/10.1007/s41019-025-00341-7) · [Asadi et al.](https://arxiv.org/html/2306.17833) ·
[SPAM](https://arxiv.org/html/2501.06842)

## 8 · Kill-search of the system-identification framing (M1–M4), 2026-10-23

**Searches:**
- 3 extended web searches: system ID / impulse-response of training; Adam "refractory" suppression after a large
  gradient; two-spike interference.
- 7 OpenReview API queries: impulse response of training; damage kernel; refractory optimizer; two-spike
  interaction; antithetic gradient perturbation; intervention timing after a spike; rollback recovery window.

| Idea | Verdict | Nearest work | Note |
|---|---|---|---|
| **M1 damage-writing kernel k(τ)** (undo-lag sweep, writing time) | **NEW** (medium confidence) | Liu & Li 2608.19762: a linearized *observation-time* response; #36730: "sustained over onset-only control"; #17880: "effects that outlast inherited moment differences" | No paper measures *when* lasting damage is written by sweeping the lag of a state-only undo. The closest are coarser or linearized |
| **M3 refractory curve χ(Δt)** | **NEW as an instrument** (medium confidence); the qualitative effect is TAKEN | AdamMAD and folklore: large g² inflates v → smaller effective LR (stall); SPAM: spikes have a "prolonged" effect on moments | Measuring a phase-response curve with probe perturbations and checking it against 1/√(1 + x·β2^Δt) was not found |
| **M4 two-spike interference** (same- vs opposite-sign) | **NEW** (medium confidence) | SPAM: spikes "accumulate" in momentum; Tang 2609.18314: the β1/β2 boundary for spiking | Nothing on the interaction term I(Δt) or on sign-dependent cancellation |
| **M2 parity split** | **NEW in context** (low-medium confidence) | antithetic sampling and central differences (generic); Liu & Li (first-order signed response) | The technique is generic. The novelty is measuring the even (nonlinear) share of spike damage |
| — | (related work, not a collision) | **"Spiking, Fast and Slow: The Anatomy of Loss Spikes in Language Model Pretraining"** (ICLR 2027 submission, OpenReview `ptNmbmCdj3`; record saved in `raw/openreview/spiking_fast_slow.json`) | Fast vs "time-bomb" spikes driven by LM-head logit growth × Adam's preconditioner under finite precision; mitigations act by reducing LM-head quantization error. That's a **cause** story (like Marin). Cite it in the cause-vs-carrier paragraph; our claim stays "carried and written by state" |
| — | (related) | "Simulating Training Spikes of Deep Neural Networks by Layer-wise Random Feature Models" (arXiv 2610.13524) | A spike-simulation model. Unread; cite if relevant |

**Bottom line.** The new P1 framing survives the kill-search. M1, M3 and M4 have no direct precedent in the
corpus or in this pass. The labels stay "NEW (medium confidence, bounded search)", and the claim must rest on the
measurement, not on the closed forms, which are credited to Patitucci & Mokhtari.

Sources: [OPEN-1B](https://arxiv.org/html/2609.17380) · [Patitucci & Mokhtari](https://arxiv.org/html/2602.10204) ·
[Ibrahim et al.](https://arxiv.org/html/2403.08763) · [Pythia](https://arxiv.org/html/2304.01373) ·
[Wen et al.](https://arxiv.org/html/2410.05192) · [LayerCheck](https://arxiv.org/abs/2609.27193) ·
[Xu](https://arxiv.org/abs/2608.20442) · [Guo](https://arxiv.org/abs/2608.24593) ·
[Tang](https://arxiv.org/html/2609.18314) · [TrainSDC](https://arxiv.org/abs/2608.30769) ·
[Altenbernd et al.](https://arxiv.org/html/2604.00726) · [K2-Spike-1](https://huggingface.co/LLM360/K2-Spike-1) ·
[Ling MoE](https://arxiv.org/pdf/2503.05139) ·
[ICML 2027 estimate](https://www.opencurious.com/ai-conference-deadlines/icml-2027)
