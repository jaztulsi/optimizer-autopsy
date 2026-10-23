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

Sources: [OPEN-1B](https://arxiv.org/html/2609.17380) · [Patitucci & Mokhtari](https://arxiv.org/html/2602.10204) ·
[Ibrahim et al.](https://arxiv.org/html/2403.08763) · [Pythia](https://arxiv.org/html/2304.01373) ·
[Wen et al.](https://arxiv.org/html/2410.05192) · [LayerCheck](https://arxiv.org/abs/2609.27193) ·
[Xu](https://arxiv.org/abs/2608.20442) · [Guo](https://arxiv.org/abs/2608.24593) ·
[Tang](https://arxiv.org/html/2609.18314) · [TrainSDC](https://arxiv.org/abs/2608.30769) ·
[Altenbernd et al.](https://arxiv.org/html/2604.00726) · [K2-Spike-1](https://huggingface.co/LLM360/K2-Spike-1) ·
[Ling MoE](https://arxiv.org/pdf/2503.05139) ·
[ICML 2027 estimate](https://www.opencurious.com/ai-conference-deadlines/icml-2027)
