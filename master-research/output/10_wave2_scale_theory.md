# 10 · Wave 2: scale transfer and theory (2026-10-22)

*Wave-2 researcher (scale transfer + theory). Answers Skeptic **F1** (proxy regime too small) and **M11** (the horizon
decides the result), building on `09_analysis.md` (Adam memory math, anti-windup invariance, cost model) and
`06_methodology.md` (H1–H9). Web research and reading only. **No code, scripts, tests or python were run.** The
only shell commands were `date` and one read-only `ls` of `~/.claude/agents/web-search-modules/` before the search
modules were read; that `ls` broke the date-only rule. No repo file was edited except this one.*

**Labels.** [V] = **verified** this session against the linked source (abstract, HTML or model card actually
fetched). [V-snip] = verified only from a search-engine snippet or summary of the source, not from a page I fetched.
[I] = **inferred** (my arithmetic or reasoning; it can be wrong). [A] = **assumed**. **UNVERIFIED** = could not
confirm; it supports no claim. Costs use 09's "10× P100" MI300X scenario [A] (≈ 37 TFLOP/s effective, back-solved
from 09's 6.6 s per 327,680-token step at 124M) and scale by FLOPs per token [I]. `step_timer` on AMD replaces all
of them.

**Decision this serves.** How should the scale ladder (proxy → ~30M mid → 124M, n ≥ 3) be designed so that a
reviewer cannot say "a 3-layer toy at step 160"? Which theory predictions are sharp enough to pre-register? If
the answer were "nothing transfers", the paper should be rescoped to a proxy-only mechanism study (V7 core value 6).

---

## 0 · Bottom line

1. **No prior work studies spike damage, or optimizer-state damage, under μP or across a width/depth ladder.** Five
   kill-searches found nothing (§7). The adjacent facts are all verified:
   - Under μP the top (preconditioned) Hessian eigenvalues are width-consistent, but loss and the lower part of the
     spectrum pick up finite-size effects (Noci et al. 2024) [V].
   - μParam fixes the optimal LR but does **not** reduce LR sensitivity, and it does not remove the need for
     qk-layernorm (Wortsman et al. 2023) [V].
   - Depth raises LR sensitivity faster than width does (Wortsman et al. 2023) [V].
   - Natural spikes appear only at 410M in PolyPythias: 2 of 10 seeds, versus 0 of 40 runs at 14M–160M [V].
2. **What transfers splits into three levels** (this is the proposed scale argument, §4):
   - **L0, algebraic.** Decay rates and drift constants are set by (β1, β2) alone, so they hold at any scale.
   - **L1, parametrization.** Amplitudes collapse across width/depth only under μP or CompleteP; under SP they follow
     a fit-able power law.
   - **L2, finite-size.** Long-horizon damage in flat modes needs real scale.
3. **New closed-form result [I]: at the project's default betas, two memories are indistinguishable.**
   - Closed-loop, Adam's EMA momentum makes every weight mode with ηλ̃ ∈ [0.026, 38] decay at **√β1 per step**
     (half-life 13.2 steps at β1 = 0.9).
   - That is within 2.7% of the β2 = 0.95 v-memory rate (half-life 13.5 steps).
   - So 09's H2 "slope ≈ −ln β2" test cannot tell v-memory from weight/momentum memory at (0.9, 0.95). A **β1 arm**
     separates them (§3, T3).
   - The same algebra corrects 09 §1.4(3): closed-loop m-carried damage at s1+50 is ≈ 7% of its initial size, not
     ≈ 0.5%.
4. **The rungs differ in tokens per step, and that changes Adam's memory.**
   - The proxy runs 4,096–16,384 tokens/step; the 124M runs 327,680 [V: config, 09 §6].
   - At β2 = 0.95, v's memory is 13.5 steps. That is ≈ 55k–221k tokens at the proxy and ≈ 4.4M tokens at 124M, a
     20–80× mismatch [I].
   - **Fix:** run every rung at 16,384 tokens/step. This also makes a 300-step 124M fork ≈ 20× cheaper (≈ 100 s
     instead of ≈ 0.55 h) [I/A]. With the cost saved, n ≥ 5 at 124M becomes affordable.
5. **A real, fully trained 12L/768 trunk with Adam moments already exists.**
   - `EleutherAI/neox-ckpt-pythia-160m-deduped-v1` holds the final step's 32 ZeRO optimizer shards (2.27 GB
     listed) [V].
   - It was trained with Adam (0.9, 0.95), ε = 1e-8, cosine to 6e-5, 143k steps, 12 layers, d = 768 [V: config
     yml].
   - Forking from it gives a 124M-class rung at about 3,500 tokens per non-embedding parameter for ≈ 4 MI300X-h,
     against ≈ 34–45 h to train our own (09 §6) [I/A].
   - Nobody has used these released states for spike or state-damage work as far as the searches show (§7).
   - Intermediate-step states are "on request" by email [V].
6. **Theory gives a repair target, with a caveat.** Per the central flow (Cohen et al. 2024), after a spike v relaxes
   at rate (1−β2)/β2 toward the **stationary preconditioner of the current weights** [V eq. 34]. Once w has moved,
   the right target is "fresh g² at current w", not the pre-spike v̄ [I]. This backs 06's "fresh g²" reference and
   predicts that the v̄-target repair weakens at late cut times.

---

## 1 · μP, Tensor Programs, depth-μP, u-μP, CompleteP: does spike behaviour transfer?

| Work (link · date) | What it gives us | Status |
|---|---|---|
| **Noci, Meterez, Hofmann, Orvieto**, *Super Consistency of Neural Network Landscapes and LR Transfer* — [2402.17457](https://arxiv.org/abs/2402.17457) · NeurIPS 2024 | (1) Under μP and Depth-μP, sharpness "reaches a width-independent value which is close to the EoS threshold". (2) Under NTP, sharpness shrinks progressively with width. (3) Loss and NTK eigenvalues "accumulate finite-size effects"; lower-order Hessian eigenvalues do too, early in training. (4) For Adam, App. J.1 tracks a per-layer-LR-scaled preconditioned Hessian. Super consistency is **"expected"** there, not proven. (5) A 2-layer post-LN transformer on WikiText-2 under μP+Adam is super consistent. (6) Depth-μP GPT-2 without 1/√L LR scaling is "not super consistent". | [V] |
| **Wortsman et al.**, *Small-scale proxies for large-scale Transformer training instabilities* — [2309.14322](https://arxiv.org/abs/2309.14322) · 25 Sep 2023, ICLR 2024 | (1) μParam "does succeed in stabilizing the optimal LR … However, μParam does not improve loss or reduce LR sensitivity". (2) Without qk-LN, μParam "does not alleviate the need for this intervention at high learning rates". (3) "scaling depth increases LR sensitivity at a faster rate than scaling width". (4) Without qk-LN, the divergence LR "becomes smaller with increasing model size". (5) A quadratic fit of max attention logits across scales forecast a 4.8B divergence at LR 1e-2: "diverged as predicted". (6) Grad RMS reaches AdamW ε at scale; lowering ε from 1e-8 to 1e-15 helps a 4.8B model. | [V] |
| **Everett et al.**, *Scaling Exponents Across Parameterizations and Optimizers* — [2407.05872](https://arxiv.org/abs/2407.05872) · 8 Jul 2024, ICML 2024 | (1) HP transfer works for all four parameterizations when LRs are per-layer. (2) Per-layer SP beats μP. (3) Adam ε must be scaled to avoid "gradient underflow". (4) Proposes **Adam-atan2**, which removes ε. (5) Models up to 26.8B. | [V] |
| **Yang, Yu, Zhu, Hayou**, *Tensor Programs VI* (Depth-μP) — [2310.02244](https://arxiv.org/abs/2310.02244) · Oct 2023, ICLR 2024 | Divide each residual block and its update by √depth; validated to depth 2¹⁰. | [V-snip] |
| **Bordelon, Noci, Li, Hanin, Pehlevan**, *Depthwise HP transfer in residual networks* — [2309.16620](https://arxiv.org/abs/2309.16620) · Sep 2023, ICLR 2024 | μP plus a 1/√L branch scale gives width and depth transfer (DMFT limit). | [V-snip] |
| **Dey et al.**, *CompleteP* — [2505.01618](https://arxiv.org/abs/2505.01618) · 2 May 2025 (v4 19 Jan 2026), NeurIPS 2025 | α = 1 residual scale m_L⁻¹; hidden AdamW LR η·m_N⁻¹ (no depth factor); **AdamW ε for residual blocks = ε_base·m_N⁻¹·m_L⁻¹**, embedding/unembedding ε = ε_base·m_N⁻¹ (Table 1); "Only CompleteP (α=1) ensures stable training … as N→∞, L→∞". No loss-spike discussion. Minimal nanoGPT code: [EleutherAI/nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup) `completep` branch, MIT. | [V] |
| **Blake et al.**, *u-μP* — [2407.17465](https://arxiv.org/abs/2407.17465) · 24 Jul 2024 (v3 10 Jan 2025), ICLR 2025 | Unit scaling plus μP; FP8 out of the box. The abstract does not mention spikes. A search snippet says μP's "combined mults" phase made loss spike; that claim is UNVERIFIED. | [V] abstract |
| **Lingle**, *An Empirical Study of μP LR Transfer* — [2404.05728](https://arxiv.org/abs/2404.05728) · 8 Apr 2024 (v6 13 Feb 2025) | μ-Transfer "works as intended for most important cases" up to 10B / 190B tokens. | [V-snip] |
| **Vyas et al.**, *Feature-learning networks are consistent across widths at realistic scales* — [2305.18411](https://arxiv.org/abs/2305.18411) · NeurIPS 2023 | Edge-of-stability and large-LR effects look consistent across widths. On language modelling and long training, finite-width deviations grow. | [V-snip] |
| **Haas, Bordt, von Luxburg, Vankadara**, *Large LRs under standard width scaling* — [2505.22491](https://arxiv.org/abs/2505.22491) · 28 May 2025, NeurIPS 2025 | Under SP with cross-entropy there is a "controlled divergence regime" (logits diverge, gradients stay stable). Width scaling predicts the exponent of the maximal stable LR. | [V] |
| **Yang, Simon, Bernstein**, *A Spectral Condition for Feature Learning* — [2310.17813](https://arxiv.org/abs/2310.17813) · v2 14 May 2024 | ‖W‖ and ‖ΔW‖ should both scale as Θ(√(fan-out/fan-in)). Adam processes gradients entrywise. Updates "have low rank and align to incoming vectors". | [V] |
| **Dey, Anthony, Hestness**, *Practitioner's Guide to μP* — [blog.eleuther.ai/mutransfer](https://blog.eleuther.ai/mutransfer/) · 19 Sep 2024 | Exact μP for a tied-embedding decoder: hidden init σ²/m_d, hidden Adam LR η/m_d, attention 1/d_head, output ×α_out/m_d. Coordinate check over 10 steps. μP "does not completely solve the problem of instability". | [V] |

**Answer to "do instabilities and Adam second-moment behaviour transfer under μP?"**
- **Partly, and nobody has measured it for spikes [V for the parts; the gap is bounded by the §7 searches].**
- **Sharpness and EoS: consistent.** The top preconditioned sharpness is width-consistent under μP. For Adam this is
  shown at small scale and stated as an expectation [V].
- **LR sensitivity and attention-logit blow-ups: not fixed by μP.** μP leaves LR sensitivity unchanged, and the
  need for qk-LN remains [V].
- **ε is width-sensitive.** Hidden gradients shrink with scale, so a fixed ε starts to dominate √v at width.
  CompleteP and Everett prescribe ε ∝ 1/width or Adam-atan2 [V].

**Can a μP proxy make damage-migration curves collapse by construction? [I]**
- **Only for L1 amplitudes.** Decay *rates* already collapse under any parameterization (L0, §3).
- **Derivation sketch [I].** 09 §2.3 shows a spike-dominated coordinate drifts 5.84η in a fixed sign, independent of
  the gradient's size G.
  - If the spike gradient on a hidden matrix is low rank (verified in general by Yang–Simon–Bernstein), the drift
    matrix ≈ 5.84η·sign(G). That is close to rank 1, because the sign of a rank-1 outer product is exactly rank 1.
  - Acting on an aligned input, it changes pre-activations by ∝ η·fan_in.
  - μP sets η_hidden ∝ 1/fan_in, which cancels this. SP at fixed η does not.
- **Prediction.** Spike damage in function space is flat with width under μP. Under SP at fixed η it grows (up to ∝
  width) until LayerNorm or softmax saturation, or Haas's controlled-divergence regime, caps it.
- **The same argument for depth.** CompleteP's m_L⁻¹ residual scale keeps the summed per-block drift
  depth-invariant. Under SP it grows with L.
- **What must be verified before claiming collapse:**
  - (a) A μP coordinate check on the nanoGPT fork (EleutherAI's tooling).
  - (b) Width-consistent λ_grad(Ĥ) / λ_max(Ĥ) on calm steps (power iteration, ~5 HVPs per probe).
  - (c) The L0 rates match theory at every width.
  - (d) The dominated-coordinate fraction ρ_dom and the dimensionless damage observables collapse within a
    pre-registered tolerance.
  - (e) ε is scaled as CompleteP prescribes, or ε is small enough that the fraction of coordinates with
    √v̂ < 10ε is < 1% at every width. Otherwise ε, not μP, drives the trend.

---

## 2 · Scaling of instability and spikes (incidence, sensitivity, critical LR)

| Work (link · date) | Finding we can use | Status |
|---|---|---|
| **Wortsman, Dettmers, Zettlemoyer, Morcos, Farhadi, Schmidt**, *Stable and low-precision training for large-scale vision-language models* — [2304.13013](https://arxiv.org/abs/2304.13013) · 25 Apr 2023, NeurIPS 2023 | (1) "loss spikes increase when increasing model size", and also with batch size and LR (§3.3). (2) "Reducing AdamW β2 from its default in PyTorch of 0.999 mitigates loss spikes". (3) **"28 out of 30 detected loss spikes follow an RMS spike by 1-8 iterations"**, where RMS_t = √E[g²/v]. (4) Mostly in the patch embedding. (5) StableAdamW = per-tensor update clipping by max(1, RMS_t). | [V] |
| **Molybog et al.**, *A Theory on Adam Instability in Large-Scale ML* — [2304.09871](https://arxiv.org/abs/2304.09871) · 19 Apr 2023 | (1) "any reproduction of these spikes at a smaller scale is not necessarily caused by or remediated by the same factors as in larger scales" (§1). (2) Early-layer gradient norm "comparable to ε". (3) The update ratio goes bimodal near ±β1/√β2. (4) More likely with depth and large batch. Runs at 7B–546B. | [V] |
| **van der Wal et al.**, *PolyPythias* — [2503.09543](https://arxiv.org/abs/2503.09543) · 12 Mar 2025, ICLR 2025 | (1) 50 runs over 14M–410M; seeds change both init and batch composition. (2) **"only for these very seeds we observe 'loss spikes'"**: 410M seeds 3 and 4 only. (3) ~7k checkpoints released. Whether optimizer states are released is UNVERIFIED. | [V] |
| **Takase et al.**, *Spike No More* — [2312.16903](https://arxiv.org/abs/2312.16903) · 28 Dec 2023, COLM 2025 | Stability needs "small sub-layers and large shortcut" (Jacobian spectral norms). The width dependence is not in the abstract. | [V] abstract |
| **Tang, Chen, Liu**, *Beyond Quadratic Loss: The Stability Phase Diagram of Adam* — [2609.18314](https://arxiv.org/abs/2609.18314) · 16 Sep 2026 | (1) Spiky vs non-spiky boundary ≈ 1−β2 = C(1−β1)^p. (2) The spiky phase is on the larger-β2 side. (3) Neural fit C ≈ 2.21, p ≈ 0.92 (modular division); other models p ≈ 0.75–1.05. (4) "periods tend to increase as β2 approaches one". (5) Models ≤ a 65k-parameter char-transformer; no width study. | [V] |
| **Bai et al.**, *Adaptive Preconditioners Trigger Loss Spikes in Adam* — [2506.04805](https://arxiv.org/abs/2506.04805) · 5 Jun 2025 (v2 25 May 2026), ICML 2026 | (1) Ĥ_t = [1/(1−β1ᵗ)]·[(1−β1)/(1+β1)]·diag(1/(√v̂+ε))·H, with threshold **2/η** for Ĥ (eq. 7). (2) The raw-Hessian equivalent is 38/η at β1 = 0.9 (Prop. 4.2). (3) Largest model "187M parameter LLaMA-structured transformer on 100B tokens" (§6.2). | [V] |
| **Porian et al.**, *Resolving Discrepancies in Compute-Optimal Scaling* — [2406.19146](https://arxiv.org/abs/2406.19146) · NeurIPS 2024 | Tuning AdamW β2 "is essential at lower batch sizes". | [V-snip] |
| **Bergsma et al.**, *Power Lines* — [2505.13738](https://arxiv.org/abs/2505.13738) · 19 May 2025, NeurIPS 2025 | AdamW timescale τ = B/(ηλD). Optimal τ is a power law in tokens/param. B_opt and B_crit scale with D, "independent of model size". | [V] |
| **Wang & Aitchison**, *How to set AdamW's weight decay as you scale* — [2405.13698](https://arxiv.org/abs/2405.13698) · ICML 2025 | AdamW weights are an EMA of updates. The optimal timescale in epochs is roughly constant across scale. | [V-snip] |

**Implications [I].**
- **(a) Natural spike incidence rises with scale at a fixed recipe** (CLIP: size, batch, LR; PolyPythias: only
  410M). So proxy-scale natural spikes need stress presets. That is the honest reason induced spikes dominate at
  the proxy, and it is citable.
- **(b) Molybog's sentence is the reviewer's F1 in print.** Our answer is §4's three levels: claim transfer only
  where theory says it must hold (L0), test it where a parameterization predicts it (L1), and measure the rest at
  scale (L2).
- **(c) No spike-incidence scaling law exists** (one search, §7). Fitting one from PolyPythias/K2/OLMo logs is
  possible but off-thesis. **Skip.**

---

## 3 · Theory of Adam's response to a perturbation: sharp, testable predictions

### 3.1 Sources

| Work (link · date) | What it gives | Status |
|---|---|---|
| **Liu & Li**, *Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW* — [2608.19762](https://arxiv.org/abs/2608.19762) · 20 Aug 2026 | (1) A linearised input–state–output response operator over (θ, m, v). (2) Isolated θ/m/v responses peak near horizons 6, 16 and 26. (3) "increasing β1 from 0.5 to 0.99 multiplies accumulated response by 14.17 and 11.96". (4) Loop gains "shift the poles of the joint system away from the uncoupled values". (5) Pythia-410M/1B/1.4B are *continued 500 updates* (lr 1e-5, β = 0.9/0.999, batch 1) "to create" moment state; it is not stated that the released states were loaded. (6) No closed-form decay rates. | [V] |
| **Cohen et al.**, *Adaptive Gradient Methods at the Edge of Stability* — [2207.14484](https://arxiv.org/abs/2207.14484) · 29 Jul 2022 (v2 15 Apr 2024) | Frozen-preconditioner Adam is a linear recurrence. The threshold "For Adam with step size η and β1 = 0.9 … is 38/η". | [V] |
| **Cohen, Damian, Talwalkar, Kolter, Lee**, *Central Flows* — [2410.24206](https://arxiv.org/abs/2410.24206) · 31 Oct 2024 (v2 25 Sep 2025), ICLR 2025 | RMSProp central flow (eq. 34): dν/dt = ((1−β2)/β2)[∇L⊙² + (4ν/η²)⊙diag Σ − ν]. The stationary preconditioner solves min tr(P)+… s.t. H ⪯ 2P (eq. 35). Quasistationarity holds at β2 = 0.99. No momentum (RMSProp only); deterministic. | [V] |
| **Regis & Chewi**, *A Rod Flow Model for Adam at EoS* — [2605.06821](https://arxiv.org/abs/2605.06821) · 7 May 2026 | A continuous-time model of Adam at EoS in (w, m) phase space, with v smooth. More accurate than stable flow. CNN/MLP/ViT on CIFAR-10. | [V] abstract |
| **Malladi, Lyu, Panigrahi, Arora**, *SDEs and Scaling Rules for Adaptive Algorithms* — [2205.10287](https://arxiv.org/abs/2205.10287) · NeurIPS 2022 | Adam square-root scaling rule for batch κB: η√κ, β′ = 1−κ(1−β), ε/√κ (Def. 5.2). Time per step = η². Needs β → 1 and noise-dominated gradients. | [V] |
| **Marek et al.**, *Small Batch Size Training for LMs* — [2507.07101](https://arxiv.org/abs/2507.07101) · NeurIPS 2025 | Hold β2's half-life fixed **in tokens**: β2* = β2^(B*/B) (eq. 2). Models from 30M to 1.3B. | [V] |
| **Compagnoni et al.**, *Adaptive Methods through the Lens of SDEs* — [2411.15958](https://arxiv.org/abs/2411.15958) · ICLR 2025 | SDEs for SignSGD and Adam(W), with a stationary-regime expression. The exact scaling rules were not retrieved. | [V-snip] |
| **Bock & Weiß** — [2102.09804](https://arxiv.org/abs/2102.09804) (local convergence via the Jacobian) · [2210.02070](https://arxiv.org/abs/2210.02070) (period-2 limit cycles for all hyperparameters, even on quadratics, with bias correction) | Adam does not always settle to a fixed point after a perturbation. | [V-snip] |
| **Kalra, He, Barkeshli**, *Universal Sharpness Dynamics* — [2311.02076](https://arxiv.org/abs/2311.02076) | Period-doubling route to chaos on the EoS manifold. | [V-snip] |
| **Altıntaş, Kwok, Raffel, Rolnick**, *The Butterfly Effect* — [2506.13234](https://arxiv.org/abs/2506.13234) · ICML 2025 | (1) One perturbed weight early in training makes runs diverge. (2) "stability continues to increase throughout training". (3) The shallow-wide model is "most stable". (4) Divergence is "not … exponential", contrary to a linearised model. | [V] |
| **Herrmann et al.**, *Chaotic Dynamics are Intrinsic to NN Training with SGD* — NeurIPS 2022, [proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/222a2a46018a1e7b55ba48ba11932d04-Abstract.html) | Chaos is concentrated early. Late divergence looks like a random walk plus linear growth. | [V-snip] |
| **Redman et al.**, *Identifying Equivalent Training Dynamics* (Koopman) — [2302.09160](https://arxiv.org/abs/2302.09160) · NeurIPS 2024 | Koopman eigenvalues test whether two training runs are topologically conjugate; includes a width study. | [V-snip] |
| **Lewkowycz et al.**, catapult — [2003.02218](https://arxiv.org/abs/2003.02218) | For GD with 2/S < η < 4/S, the loss spikes and sharpness drops. | [V-snip] |

### 3.2 Predictions table

Rows marked NEW are derivations made in this document. Adam has no ε-floor unless stated; (β1, β2) = (0.9, 0.95)
unless stated.

| # | Prediction (closed form) | Numbers | Source / status | Test (which experiment) |
|---|---|---|---|---|
| **T1** | Excess v decays as β2ᵏ (open loop). Central-flow form: ν relaxes at (1−β2)/β2 toward the **stationary ν̄(w)** of the *current* w. | Half-life 13.5 / 69.0 / 692.8 steps at β2 = .95 / .99 / .999 | 09 §1 [I]; eq. 34 [V] | v-only patch recovery vs cut time (E2) |
| **T2** | Excess m decays as β1ᵏ, **open loop only** (future gradients unchanged) | Half-life 6.6 steps | 09 §1 [I] | m-only patch at s1+5 |
| **T3 (NEW)** | **Closed loop with a frozen preconditioner.** Each mode of D⁻¹H with a = ηλ̃ follows x_{t+1} = (1+β1−(1−β1)a)x_t − β1x_{t−1}. **(i)** Instability at a = 2(1+β1)/(1−β1). **(ii)** Underdamped band (1−√β1)²/(1−β1) < a < (1+√β1)²/(1−β1): \|r\| = **√β1**, the same rate for every curvature in the band. **(iii)** Flat modes (a below the band) decay at ≈ a per step. **(iv)** So long-lived weight damage lives only in flat modes, while damage in the sharp directions C2 searches (top-k of Ĥ) decays within ~13 steps or oscillates at EoS. | (i) 38 at β1 = 0.9 (= Cohen's 38/η ✓) (ii) band a ∈ [0.0263, 37.97]; half-life 13.2 steps (β1 = .9), 6.2 (β1 = .8), 27.0 (β1 = .95) | [I] (heavy-ball algebra; threshold cross-checked [V]; Liu & Li's "poles shift" statement [V] is consistent) | Δ_B0(τ) curve, w-only patch recovery, by β1 arm (E2) |
| **T3′ (NEW)** | **Degeneracy at the defaults.** −½ln β1 = 0.0527/step vs −ln β2 = 0.0513/step: the underdamped (w, m) rate and the v rate differ by 2.7%. H2's slope test cannot attribute damage to v at (0.9, 0.95). | Separating arms: (0.8, 0.95): 0.112 vs 0.051. (0.95, 0.95): 0.026 vs 0.051. (0.9, 0.99): 0.053 vs 0.010 | [I] | E2: β1 × β2 cross |
| **T3″ (NEW)** | Closed-loop m-carried damage at s1+50 ≈ √β1⁵⁰ ≈ **7%** of its initial underdamped amplitude, not β1⁵⁰ ≈ 0.5%. This corrects 09 §1.4(3)'s "free sanity prediction". | 0.9487⁵⁰ = 0.072; 0.9⁵⁰ = 0.005 | [I] | m-only patch at s1+50 |
| **T4** | Anti-windup drift on spike-dominated coordinates: D(β1, β2) = (1−β1)/[√(1−β2)·(1−β1/√β2)]·η, independent of G. Dominance needs G/ḡ ≫ 1/√(1−β2). | D = **5.84η** (.9/.95), **10.47η** (.9/.99), **31.8η** (.9/.999), 4.99η (.8/.95), 8.83η (.95/.95) → drift ratio across the β2 arms **1 : 1.79 : 5.44**. Dominance thresholds 4.5×, 10×, 31.6× | 09 §2.3 [I]; generalisation [I] | Post-spike update RMS on dominated coordinates (E2) |
| **T5 (NEW)** | **Width.** Function-space drift from T4 on a low-rank-gradient hidden matrix ∝ η·fan_in. μP (η ∝ 1/fan_in) cancels it; **SP at fixed η: damage grows with width** until saturation. | 192 → 768: ≈ flat (μP) vs up to 4× (SP) | [I] from [V] ingredients (spectral condition; μP LR table) | E1 SP vs μP |
| **T6 (NEW)** | **Depth.** CompleteP's m_L⁻¹ residual scale keeps the summed per-block drift depth-invariant. Under SP it grows between √L (incoherent blocks) and L (coherent). | L = 3 → 12: flat vs 2–4× | [I] | E1 depth arm |
| **T7** | **Repair target.** Once w ≠ w_pre, the stationary preconditioner ν̄(w_now) ≠ v̄_pre. The v̄-restore error grows with cut time; a fresh-g² or central-flow target does not. | — | [I] from eq. 34–35 [V] | 06 ablation "reference: v̄ vs fresh g²" at t1 = s1 + {10, 50, 150} |
| **T8** | **Phase boundary** (Tang 2026): spiky iff 1−β2 < C(1−β1)^p. Spike period grows as β2 → 1. | At β1 = .9, C = 2.21, p = .92: spiky iff β2 ≳ **0.73**. So .95/.99/.999 are all spiky; β2 = 0.6 is predicted non-spiky | [V] paper; [I] transfer to LMs | E8 negative control on the natural stress preset |
| **T9** | **Precursor.** An RMS_t = √E[g²/v] spike precedes the loss spike by 1–8 steps (CLIP: 28/30). This is the same under-estimated-v regime as Bai's decay stage. | Lead time 1–8 steps | [V] (CLIP ViT) | H9 lead time; log RMS_t per tensor (free) |
| **T10** | **Batch.** Marek: β2* = β2^(B*/B). Malladi: β2′ = 1−κ(1−β2). These agree to first order in (1−β2); they disagree on η (√κ vs much weaker in practice). | Matching 124M's token half-life at 16k tokens/step: β2* = 0.95^0.05 = 0.9974 | [V] formulas; equivalence [I] | Design rule in §4 |
| **T11** | **ε floor.** Where √v̂ ≲ ε, the T4 drift shrinks by ≈ √v̂/(√v̂+ε). At fixed ε this fraction grows with width (grad RMS falls with scale), so SP damage can *fall* artificially at width. | — | [I] from [V] (Everett; Wortsman; CompleteP ε table) | Log the fraction with √v̂ < 10ε per tensor and rung (free); E4 on Pythia |
| **T12** | **Feasibility.** Honest Adam states satisfy \|m\| ≤ C√v, with C = 1.165 at (.9, .95). | Violation rate must be 0 up to fp32 rounding | 09 §2.4 [I] | E4: 5 Pythia sizes. Doubles as the ZeRO-shard parser check |
| **T13** | **Placebo spread** shrinks with schedule progress (Butterfly). Compare rungs at matched *fraction of schedule*, not matched step. | — | [V] qualitative | M8 placebo per site |
| **T14** | **Catapult (lr_bump, factor f).** Afterwards λ_max(Ĥ) falls below the threshold at fη. Part of the damage is flattening, so Δ_B0 can turn negative at long horizon. | — | [I]; the Adam analogue is UNVERIFIED | λ_grad probes before and after the bump (E9) |

**Unit check [I].** Bai's Ĥ absorbs (1−β1)/(1+β1) and 1/(1−β1ᵗ), so its threshold is 2/η. On the raw D⁻¹H the
threshold is 38/η at β1 = 0.9. Certified restoration (06 §3.2, λ* = (1−δ)·2/η·(bc)) must use **Bai's Ĥ
throughout**. Mixing the two is a 19× error.

---

## 4 · Proposed scale-transfer protocol (STP)

### 4.1 The three-level claim structure (proposed paper framing)

| Level | What should transfer | Why | How verified | If it fails |
|---|---|---|---|---|
| **L0 algebraic** | Decay rates (T1–T3), drift constants (T4), the feasibility bound (T12) | Set by (β1, β2) alone | Fit at **every** rung, including Pythia; pre-registered tolerance ±15% on rates | ε floor (T11) or non-dominance. Report it; it is diagnostic |
| **L1 parametrization** | Dimensionless amplitudes: state-carried Shapley share φ_state(s1+10), relative damage Δ_B0(300)/L, dominated fraction ρ_dom, gap closed G of the undo repair | Under μP/CompleteP these are O(1) in width/depth (T5, T6). Under SP they follow a power law | E1: width × depth mini-ladder, SP and μP | Report the exponent; it is the result |
| **L2 finite-size** | End-of-schedule Δ; damage in flat modes; natural-spike incidence | Lower-order spectrum and loss accumulate finite-size effects (Noci [V]); incidence rises with size ([V] CLIP, PolyPythias) | 124M n ≥ 3–5 at matched tokens/step; Pythia-160M borrowed trunk (E3) | Claim scope stated plainly |

### 4.2 Matching rules across rungs (hold these fixed; each one closes a confound)

1. **Tokens per optimizer step = 16,384 at every rung**, so Adam's memory is the same in steps and in tokens (T10).
   - For 124M: 8×1024 tokens × grad_accum 2, instead of 40. That is a cfg change, so the determinism gate has to be
     re-run [I].
   - If the batch must differ, apply β2* = β2^(B*/B) (Marek) and report it.
2. **(β1, β2, ε, clip, warmup steps, schedule shape)** identical. Spike sites at matched **fractions** of the
   schedule (proposed: 40% and 70%), not matched steps (T13, M11).
3. **ε:** CompleteP's ε_hidden = ε_base·m_N⁻¹·m_L⁻¹ on the μP arm. On the SP arm, either ε = 1e-12 everywhere or ε
   left as-is with the fraction-below-ε logged. Pick one and pre-register it.
4. **Dose in dimensionless units:** the same corrupted-token fraction and width (5 steps). Report the per-coordinate
   G/ḡ distribution and RMS_t, so doses can be compared across rungs.
5. **qk-LN on and off at each rung** (M4). Log max attention logit (free) for Wortsman-style extrapolation (E7).
6. **η:** SP rungs use the 124M config's LR so that the extrapolation (E1 → 124M) predicts what will actually run.
   μP rungs use η_base at d_base = 192 (the current proxy), so the proxy *is* the μP base model.

### 4.3 Experiments

| ID | Experiment | Pre-registered prediction | Cost [A: 10× scenario] | Person-h [A] |
|---|---|---|---|---|
| **E1** | **Spike-damage scaling law with held-out rungs.** Fit configs: width arm 3L × d ∈ {192, 384, 768}; depth arm d = 192 × L ∈ {3, 6, 12}. Fixed 3,250 steps × 16,384 tokens; warmup+cosine; SP arm (+ μP/CompleteP arm). `corrupt_batch` at 40%/70%; 2 seeds (init + data order). Per instance: B0, B*, state-only, w-only at t1 = s1+10, h = 300, plus 2 placebos. Fit log O = a + γ_d·log d + γ_L·log L per L1 observable. **Before running them**, predict the mid rung (6L/384) and 124M (12L/768, n ≥ 3). | μP: γ_d, γ_L ≈ 0 (\|γ\| < 0.15). SP: γ_d > 0 for Δ_B0 (T5). L0 rates within ±15% at every config. **Pass:** the held-out mid-rung and 124M observations fall inside the bootstrap 95% prediction interval, and the sign of H1 is preserved | Trunks ≈ 0.25–0.5 h per seed per parametrization over 5 configs [I: FLOPs ≈ 66M–359M per token]. Forks ≈ 1 h. **≈ 2–4 MI300X-h** (SP only ≈ 1.5–2.5) | 10–15 (μP: +6–8 via nanoGPT-mup diff; flag as a model change) |
| **E2** | **Two-instrument β identification** at the proxy (realistic F1 trunk): cross design (β1, β2) ∈ {(.9,.95) centre, (.9,.99), (.9,.999), (.8,.95), (.95,.95)}. Cut times τ ∈ {1, 10, 30, 60, 150, 400}; 4 factorial cells | **v-only patch recovery** decays at −ln β2 (varies across the β2 arms, flat across the β1 arms). **w-only recovery** of the underdamped part decays at −½ln β1 (varies across the β1 arms, flat across the β2 arms). Drift ratio 1 : 1.79 : 5.44 (T4). Refuted if both curves track the same β | 5 trunks × 2 seeds ≈ 0.7 h; 20 instances × 24 forks ≈ 1 h → **≈ 1.7 h** (09's β2 sweep already budgets ≈ +3 GPU-h; the β1 arms add ≈ 0.7 h) | 4–6 |
| **E3** | **Borrowed trunk: Pythia-160M final state** (and optionally 410M). Convert ZeRO-1 shards to the name-keyed snapshot; GPT-NeoX model module; determinism gate; 200-step burn-in at 16,384 tokens/step on a deduped-Pile slice (v re-equilibrates to the smaller batch: 3/(1−β2) ≈ 60 steps [I]). Induce `corrupt_batch` / `v_inflate`; run the 4-cell factorial plus undo plus placebos | L0 rates match T1–T4 at 160M. The H1 sign matches the proxy. This is the **end-of-schedule regime** (LR 6e-5), which Bai places spikes in ("late in training") | ≈ 0.38 s/step [I: ≈ 855M FLOP/token incl. attention] → 300-step fork ≈ 2 min; n = 10 × 10 forks ≈ 3.2 h → **≈ 4 MI300X-h**. 410M: ≈ 2.5× that | **20–30** (risk: ZeRO-1 / pipeline-format conversion; the DeepSpeed universal-checkpoint tooling may help, but its ZeRO-1 support is UNVERIFIED) |
| **E4** | **Offline optimizer-state census** of Pythia final states (70M, 160M, 410M, 1B, 1.4B; ≈ 37 GB [I: 12 B/param; the 160M shards total 1.95 GB ✓]). Measure the T12 violation rate, the fraction with √v̂ < 10ε per tensor class (T11), √v RMS vs width (an empirical exponent for the ε concern), embedding-row v sparsity (m14) | T12 holds at 100% (else the parser is wrong). The ε-fraction rises with width for hidden matrices. Rare-token rows dominate any magnitude-based score (m14) | CPU job on Kaggle or the MI300X host, < 1 h. **Not on the Mac** (CLAUDE.md §1) | 6–10 |
| **E5** | **Ask EleutherAI** (README contacts) for (a) Pythia-160M/410M optimizer states at mid-training steps, and (b) PolyPythias-410M seeds 3/4 states bracketing their loss spikes | Gives natural-spike, real-scale m/v for offline C2 scoring (H5) and a natural-spike test of F3's sign classification | 0 GPU; an email | 1 |
| **E6** | OPEN-1B offline (Skeptic F1(d)). New verified facts: AdamW (0.9, 0.95, 1e-8), checkpoints **every 100 steps**, "both Adam moments … are stored in FP32", "an audit replay can reconstruct the moments exactly", skip protocol fired on 2 batches | Same as E4/E5. **Whether the public release contains the moments is UNVERIFIED** | ≤ 1 GPU-h + download | 10–15 |
| **E7** | Wortsman-style extrapolation: fit max-attention-logit and RMS_t growth across the E1 configs and forecast 124M (qk-LN off) | 124M instability at our LR, predicted before running | 0 extra (telemetry) | 2 |
| **E8** | Tang negative control: natural stress preset at β2 = 0.6 (β1 = .9) | No macroscopic natural spikes (T8). If spikes appear, the LM is outside Tang's regime: report it | ≈ 0.1–0.3 h | 1–2 |
| **E9** | lr_bump catapult probe: λ_grad(Ĥ) and λ_max(Ĥ) before and after the bump (~5 HVPs per probe) | Post-bump λ below threshold. Δ_B0 can turn negative at the end of schedule (T14) | ≈ 0.1 h | 2 |

**Budget [I/A].**
- **Totals.** E1 + E2 + E3 + E4 ≈ 8–12 MI300X-h at the 10× scenario. At 2×, multiply by about 5.
- **Savings.** Matched tokens/step cuts 09's 124M primary contrast from ≈ 34–45 h (n = 3) to ≈ 6–8 h (n = 5): the
  trunk stays ≈ 5.6 h, and the forks fall from ≈ 0.55 h to ≈ 0.03 h each.
- **Caveat.** Small batches lose GPU efficiency [A], and Marek found β2 = 0.95 suboptimal at small batch [V]. That
  is a quality cost to the trunk, not a validity cost.

### 4.4 What a reviewer gets (F1 rebuttal, drafted)

> The β-determined quantities (L0) are verified at five model configurations and at a released, fully trained
> 160M-class Pythia state. The amplitudes (L1) follow a power law fitted on small configurations, which predicted
> the held-out 30M and 124M values before they were run. The parts that do not transfer (L2: end-of-schedule Δ and
> natural-spike incidence) are measured at 124M (n = 5) and stated as scope.

This answers Molybog's caveat [V] directly. It does not claim that the proxy *is* a large model; it claims that the
specific quantities we report are predicted to transfer and were tested out of sample. [I]

---

## 5 · Idea cards (mechanism · closest prior · kill-search verdict · plug-in)

Verdicts are bounded by the searches in §7 only.

1. **Spike-damage scaling law with held-out rungs (E1)**
   - **Mechanism:** fit L1 damage observables vs (d, L) on cheap configs; predict mid-rung and 124M values before
     running them [I].
   - **Closest prior:** Wortsman 2309.14322, attention-logit quadratic fit with a held-out 4.8B divergence [V]
     (predicts *occurrence*, not *damage carried in state*).
   - **Verdict:** **NEW** for state-carried damage; **PARTIAL** for the method family. Queries Q1, Q2, Q4, Q5, Q11.
   - **Plug-in:** the observables are H1's φ_state, Δ_B0, and the undo repair's G. The pre-registered prediction
     interval turns H8 from descriptive into falsifiable.
2. **μP/CompleteP "spike coordinate check" (E1 μP arm)**
   - **Mechanism:** the T4 sign-coherent drift ∝ η·fan_in; μP's η ∝ 1/fan_in cancels it [I].
   - **Closest prior:** Noci 2402.17457 (sharpness super-consistency, no spikes) [V]; Wortsman 2309.14322 (μParam
     doesn't reduce LR sensitivity) [V]; CompleteP 2505.01618 [V].
   - **Verdict:** **NEW**. Queries Q1–Q5.
   - **Plug-in:** if the L1 observables collapse, the proxy's damage-migration curves transfer to width by
     construction, and the certified-restoration threshold is width-free (Ĥ super-consistent).
3. **Two-instrument β1 × β2 identification (E2)**
   - **Mechanism:** the v rate (−ln β2) and the underdamped (w, m) rate (−½ln β1) coincide at (0.9, 0.95). Crossing
     the betas separates the carriers (T3′) [I].
   - **Closest prior:** Liu & Li 2608.19762 (β1 sweep multiplies the response 14.17/11.96×; poles shift) [V]; Tang
     2609.18314 (β-plane spike boundary) [V]; 09 §1.4 (β2 sweep).
   - **Verdict:** **PARTIAL**. β sweeps exist; carrier identification and the default-beta degeneracy look new.
     Queries Q8, Q9.
   - **Plug-in:** gives H2 an identifiable slope. Without it, a matched "−ln β2" slope at the defaults proves
     nothing about v.
4. **Borrowed trunk from Pythia's released optimizer state (E3)**
   - **Mechanism:** fork a real 12L/768 Adam state (β = 0.9/0.95) instead of training a trunk [V for the
     availability].
   - **Closest prior:** Liu & Li continue Pythia for 500 updates to *build* the moments [V]; Gupta 2308.04014
     rewarms Pythia-410M [V-snip]; the TIC workshop poster audits PolyPythias-410M with "ghost updates from a frozen
     optimizer state" [V].
   - **Verdict:** **NEW** for spike/state-damage forks; **PARTIAL** as an instrument. Queries Q6, Q7, Q12.
   - **Plug-in:** the full factorial, undo and certified repair at 160M on a 300B-token trunk. It answers "really
     undertrained" (Stable-SPAM reviewer) for the end-of-schedule regime.
5. **Matched tokens/step ladder (§4.2 rule 1)**
   - **Mechanism:** Adam's memory is per step; data is per token. Matching batch makes both agree and makes forks
     ≈ 20× cheaper at 124M [I].
   - **Closest prior:** Marek 2507.07101 (token half-life rule) [V]; Malladi 2205.10287 (SRSR) [V].
   - **Verdict:** **TAKEN** (method). This is an application.
   - **Plug-in:** makes n ≥ 5 at 124M affordable. The migration-curve x-axis means the same thing at every rung.
6. **Central-flow repair target (T7)**
   - **Mechanism:** v relaxes to the stationary ν̄(w_now), so repair should aim at fresh-g² / ν̄(w), not pre-spike
     v̄ [I from V].
   - **Closest prior:** Cohen 2410.24206 (stationary preconditioner) [V]; SPAM 2501.06842 (zeroes moments)
     [V-snip]; Bai (refill ends the spike) [V].
   - **Verdict:** **PARTIAL**. The theory exists; using it as a repair target was not found. Query Q10.
   - **Plug-in:** a sharp prediction for 06's reference ablation (v̄ vs fresh g² diverge as t1 − s1 grows), and a
     principled target for certified restoration.
7. **Offline optimizer-state census across sizes (E4)**
   - **Mechanism:** measure ε-floor fraction, √v width exponent, embedding sparsity and the feasibility bound on
     released states [I].
   - **Closest prior:** Everett and Wortsman grad-RMS vs scale (their own runs) [V]; SlimAdam SNR of v 2503.01843
     [V-snip].
   - **Verdict:** **PARTIAL**. Queries Q7, Q13.
   - **Plug-in:** checks T11/T12 at 70M–1.4B with zero training; calibrates m14; validates the snapshot converter E3
     needs.
8. **Natural-spike state at 410M via PolyPythias (E5)**
   - **Mechanism:** spikes appear only at 410M seeds 3/4 [V]; request bracketing m/v.
   - **Closest prior:** K2-Spike releases (weights; optimizer content UNVERIFIED) [V]; TIC poster [V].
   - **Verdict:** **PARTIAL**. Queries Q6, Q12.
   - **Plug-in:** a natural-spike external check for C2 (H5) and F3's signed classification. No B\* is needed
     (06 §7 option b).
9. **Tang phase-boundary negative control (E8)**
   - **Mechanism:** spiky iff 1−β2 < C(1−β1)^p [V]; test at LM scale.
   - **Closest prior:** Tang 2609.18314 [V].
   - **Verdict:** **PARTIAL** (replication at a new scale).
   - **Plug-in:** separates "natural spike from Adam's timescale mismatch" from other causes in the stress presets.
10. **Placebo spread as a finite-time sensitivity per rung (T13)**
    - **Mechanism:** sensitivity falls with training progress and with width [V qualitative].
    - **Closest prior:** Altıntaş 2506.13234 [V]; Herrmann NeurIPS 2022 [V-snip]; FTLE 2405.00743 [V-snip].
    - **Verdict:** **PARTIAL**. Query Q14.
    - **Plug-in:** the M8 placebo distribution, compared at matched schedule fractions.
11. **Koopman conjugacy of B0-vs-B\* difference dynamics across widths**
    - **Mechanism:** test whether the damage dynamics are conjugate across widths (Redman 2302.09160 [V-snip]).
    - **Verdict:** **PARTIAL**. **Rejected:** heavy fitting, small n, and a reviewer gains little over L0/L1 rate
      fits.
12. **Catapult view of lr_bump (E9)**
    - **Mechanism:** a Lewkowycz-style flattening after the bump [V-snip for GD]. The Adam analogue is UNVERIFIED.
    - **Verdict:** **PARTIAL**.
    - **Plug-in:** explains a possibly negative long-horizon Δ for `lr_bump` (M9, H6).
13. **Spike-incidence scaling law from public logs**
    - **Verdict:** NEW-ish (one search). **Rejected** as off-thesis.

---

## 6 · Corrections and flags for existing docs (for team sign-off; files not edited)

1. **09 §1.4(3)** "m irrelevant beyond ~30–50 steps" holds only open loop. Closed loop, the underdamped (w, m) modes
   decay at √β1 per step, leaving ≈ 7% at s1+50 (T3″) [I].
2. **09 §1.4(2) / 06 H2.** At (0.9, 0.95) the β2-slope test is degenerate with the weight-mode rate (T3′). Add the
   β1 arms (E2) [I].
3. **09 §6 / 124M config.** 327,680 tokens/step against the proxy's 4k–16k changes v's token memory 20–80× and costs
   ≈ 20× per fork. Proposed: 16,384 tokens/step at every rung (§4.2) [I]. The determinism gate must be re-run after
   the grad_accum change.
4. **06 §3.2 λ\*.** Use Bai's Ĥ (threshold 2/η) consistently. Raw D⁻¹H has 38/η at β1 = 0.9 [V both].
5. **ε.** Log the fraction of coordinates with √v̂ < 10ε per tensor and rung. Pre-register an ε policy (§4.2
   rule 3) [I].
6. **Limitations section.** Quote Molybog's small-scale caveat [V] and answer it with the L0/L1/L2 structure.
7. **Related work.** Add Noci 2024, Everett 2024, CompleteP 2025, Wortsman 2023 (CLIP), PolyPythias 2025, Tang 2026,
   central flows 2024, Malladi 2022 and Marek 2025. All are older than the ~2026-11-22 concurrent-work cutoff from
   02 §1 [I], so all count as prior work.

---

## 7 · Search log (2026-10-22; WebSearch unless noted)

| # | Query | Outcome |
|---|---|---|
| Q1 | `maximal update parametrization loss spike instability transfer across width` (extended) | μP docs, u-μP, EleutherAI guide; no spike-transfer study |
| Q2 | `"loss spikes" "μP" OR "muP" width scaling Adam second moment recovery small proxy model` (extended) | Found Tang 2026; nothing combining μP and spike recovery |
| Q3 | `"coordinate check" perturbation spike damage width scaling μP standard parametrization gradient outlier` | No coord-check-for-spikes work |
| Q4 | `"loss spike" OR "loss spikes" "maximal update" width proxy predicts large model instability 2026 arXiv` (extended) | Only Wortsman 2023 (occurrence forecast) |
| Q5 | `loss spike recovery time versus model width depth transformer Adam experiments "recovery" spike scaling` (extended) | No recovery-vs-width study |
| Q6 | `induced loss spike pretrained checkpoint optimizer state counterfactual fork width ladder "spike" injection mid-size language model 2026` | Found the TIC workshop poster; no fork-from-released-state study |
| Q7 | `Pythia released optimizer states used to study loss spike or Adam second moment analysis "optimizer states" Pythia experiment` | Nothing |
| Q8 | `vary β1 and β2 separately to identify which optimizer memory carries perturbation damage decay rate …` | Liu & Li closest |
| Q9 | `Adam EMA momentum linearized dynamics characteristic polynomial underdamped decay rate sqrt(beta1) …` | Threshold confirmed; the √β1 band result not found in a source (my derivation) |
| Q10 | `reset Adam second moment to stationary preconditioner after loss spike repair optimizer state "fresh" …` | SPAM (zeroing) only |
| Q11 | `scaling law loss spike frequency model size pretraining instability incidence "spikes" empirical study 2025 2026` (extended) | No incidence scaling law |
| Q12 | `LLM360 K2 loss spike checkpoints released …`; `Pythia checkpoints optimizer states released "neox-ckpt-pythia" huggingface` | K2-Spike branches (contents unknown); Pythia final-step states confirmed |
| Q13 | `μP Adam second moment epsilon underflow width hidden gradients shrink …` | Everett; MoE μP 2605.14200 (ε ∝ grad RMS) [V-snip] |
| Q14 | `finite-time Lyapunov exponent of training dynamics measured from paired runs … width transformer` | No width study |
| Others | Super Consistency; Everett; CompleteP; TP-VI; Bordelon; Vyas; Redman; Kalra; catapult μP; Malladi; central flows; Bock & Weiß; Compagnoni; Adam impulse response; Marek; Wang & Aitchison; Porian; Lingle; spectral condition; model growth with optimizer state; SlimAdam; Butterfly; Herrmann; Koopman; LLM360 Amber; OLMo optimizer state ×2; released checkpoints with Adam moments; Gupta rewarm; DeepSpeed universal checkpointing; PolyPythias | Used for §1–§4 |

**WebFetch (read):**
- arXiv abstracts or HTML for 2309.14322 (ar5iv), 2402.17457 v2 (two passes), 2407.05872, 2505.01618 (+ v4 HTML),
  2407.17465, 2609.18314 (+ HTML), 2608.19762 (+ HTML, two passes), 2205.10287 (HTML), 2410.24206 (+ HTML, two
  passes), 2304.13013 (+ ar5iv), 2304.09871 (+ ar5iv), 2507.07101 (HTML), 2312.16903, 2506.04805 (+ HTML),
  2207.14484, 2605.06821, 2505.22491, 2310.17813 (HTML), 2506.13234 (HTML), 2609.17380 (+ HTML, two passes),
  2503.09543 (+ HTML), 2505.13738, 2606.28116 (not a rival), 2608.15824 (marginal).
- Code and model pages: github.com/EleutherAI/pythia, EleutherAI/nanoGPT-mup, allenai/OLMo, the OLMo-2-0425-1B
  CSV, HF LLM360/Amber, LLM360/K2, LLM360/K2-Spike-1, HF neox-ckpt-pythia-160m-deduped-v1 tree + yml,
  blog.eleuther.ai/mutransfer, icml.cc/virtual/2026/80057.
- **Failed:** olmo-checkpoints.org (TLS error ×2), so **whether OLMo-2 releases optimizer state is UNVERIFIED**;
  kindxiaoming.github.io (TLS error); arxiv PDF 2309.14322 (too large; ar5iv used instead).

---

## 8 · Optimizer-state availability (item 4), per source

| Release | Adam moments released? | Evidence |
|---|---|---|
| **Pythia (all sizes)** | **Final step only**, NeoX format, `EleutherAI/neox-ckpt-pythia-xxx-deduped-v1`. Other steps: "please email" hailey@ / stella@eleuther.ai | README [V]. 160M-deduped-v1 tree: 32 × 60.9 MB `zero_pp_rank_*_optim_states.pt` [V]. 1.4B (non-deduped) repo also has shards [V-snip] |
| **PolyPythias** | UNVERIFIED (~7k checkpoints; tokenised data per seed) | arXiv HTML [V for what is stated] |
| **LLM360 Amber** | Not mentioned ("360 model checkpoints and the full data sequence") | Model card [V] |
| **LLM360 K2 / K2-Spike-1/2** | Not stated; spike and normal checkpoints on branches (`spike_ckpt_160`, …) | Model cards [V] |
| **OLMo-2** | UNVERIFIED. The CSVs list `…/stepN-unsharded/` directories (≈ 195 rows for 1B); directory contents unreadable here | README + CSV [V]; directory fetch failed |
| **OPEN-1B** | Moments are checkpointed and hashed in FP32 every 100 steps. Whether they are *publicly released* is UNVERIFIED | arXiv HTML §1, §3.2 [V] |

---

## 9 · Limits of this pass

- Most sources were read at abstract or HTML-excerpt depth through a fetch tool that paraphrases and truncates. No
  PDF was read end to end. Quotes are short fragments as returned.
- T3, T3′, T3″, T5, T6, T7 and T11 are my derivations [I], resting on [V] ingredients. T3 assumes a frozen
  preconditioner, a locally quadratic loss and deterministic gradients. T5 assumes low-rank, sign-coherent spike
  gradients, and nonlinearity can cap it. All need the experiments above.
- All costs are FLOP-scaled from 09's assumed 10× scenario. Small configs will be launch-bound (09 §6), so their
  real times may be 2–5× higher [A].
- The E3 engineering risk (ZeRO-1 → name-keyed snapshot; GPT-NeoX parallel-residual/rotary model in our harness;
  fp16-trained weights replayed in fp32) is unmeasured.
- "NEW" verdicts are bounded by the 14 query families in §7. This is not a systematic review.

---

## 10 · Top 5 (ranked by expected reduction in F1/M11 risk per GPU-hour)

1. **Spike-damage scaling law with held-out rungs (E1, SP arm first).** It turns 124M from "n = 3, descriptive"
   into an out-of-sample test of a pre-registered prediction. ≈ 1.5–2.5 MI300X-h [A]. Verdict NEW (state damage).
2. **Matched tokens/step = 16,384 at every rung.** It removes a 20–80× confound in Adam's token memory and makes 124M
   forks ≈ 20× cheaper, so n = 5 at 124M becomes affordable (≈ 6–8 h instead of 34–45 h) [I/A]. Verdict TAKEN
   (method), highest leverage.
3. **Two-instrument β1 × β2 design (E2).** It fixes a real identifiability hole in H2 at the default betas (T3′),
   with closed-form rate and drift predictions (T3, T4: 1 : 1.79 : 5.44). +0.7 h over 09's β2 sweep. Verdict
   PARTIAL.
4. **Borrowed trunk from Pythia-160M's released optimizer state (E3), with the E4 census as its parser check.** A
   fully trained 124M-class rung with real Adam moments for ≈ 4 MI300X-h and 20–30 person-h. Verdict NEW (spikes),
   PARTIAL (instrument). Send the E5 email the same day: it costs nothing.
5. **μP/CompleteP arm of E1 ("spike coordinate check").** If the L1 observables collapse, the transfer argument holds
   by construction. If they don't, the measured exponent is itself the finding. +1–1.5 h and a flagged model change.
   Verdict NEW.

---

## Sources (accessed 2026-10-22; dates are submission / venue)

- Noci et al., Super Consistency (NeurIPS 2024) — https://arxiv.org/abs/2402.17457
- Wortsman et al., Small-scale proxies (25 Sep 2023; ICLR 2024) — https://arxiv.org/abs/2309.14322
- Wortsman et al., Stable and low-precision training for VLMs (25 Apr 2023; NeurIPS 2023) — https://arxiv.org/abs/2304.13013
- Everett et al., Scaling Exponents (8 Jul 2024; ICML 2024) — https://arxiv.org/abs/2407.05872
- Yang, Yu, Zhu, Hayou, TP-VI (Oct 2023; ICLR 2024) — https://arxiv.org/abs/2310.02244
- Yang et al., Tensor Programs V (Microsoft PDF) — https://www.microsoft.com/en-us/research/wp-content/uploads/2021/11/TP5.pdf [V-snip]
- Bordelon et al., Depthwise HP transfer (ICLR 2024) — https://arxiv.org/abs/2309.16620
- Dey et al., CompleteP (2 May 2025; NeurIPS 2025) — https://arxiv.org/abs/2505.01618 · code https://github.com/EleutherAI/nanoGPT-mup
- Blake et al., u-μP (24 Jul 2024; ICLR 2025) — https://arxiv.org/abs/2407.17465
- Lingle, μP LR transfer (8 Apr 2024) — https://arxiv.org/abs/2404.05728
- Vyas et al. (NeurIPS 2023) — https://arxiv.org/abs/2305.18411
- Haas et al. (28 May 2025; NeurIPS 2025) — https://arxiv.org/abs/2505.22491
- Yang, Simon, Bernstein, Spectral condition — https://arxiv.org/abs/2310.17813
- Dey, Anthony, Hestness, Practitioner's Guide to μP (19 Sep 2024) — https://blog.eleuther.ai/mutransfer/
- Molybog et al. (19 Apr 2023) — https://arxiv.org/abs/2304.09871
- Takase et al., Spike No More (COLM 2025) — https://arxiv.org/abs/2312.16903
- Bai et al. (ICML 2026) — https://arxiv.org/abs/2506.04805
- Tang, Chen, Liu, Stability Phase Diagram of Adam (16 Sep 2026) — https://arxiv.org/abs/2609.18314
- Liu & Li, AdamW ISO (20 Aug 2026) — https://arxiv.org/abs/2608.19762
- Cohen et al., Adaptive EoS (29 Jul 2022) — https://arxiv.org/abs/2207.14484
- Cohen et al., Central Flows (31 Oct 2024; ICLR 2025) — https://arxiv.org/abs/2410.24206
- Regis & Chewi, Rod flow for Adam (7 May 2026) — https://arxiv.org/abs/2605.06821
- Malladi et al., SDEs and scaling rules (NeurIPS 2022) — https://arxiv.org/abs/2205.10287
- Marek et al., Small batch LM training (NeurIPS 2025) — https://arxiv.org/abs/2507.07101
- Compagnoni et al. (ICLR 2025) — https://arxiv.org/abs/2411.15958
- Bock & Weiß — https://arxiv.org/abs/2102.09804 · https://arxiv.org/abs/2210.02070
- Kalra, He, Barkeshli — https://arxiv.org/abs/2311.02076
- Lewkowycz et al., catapult — https://arxiv.org/abs/2003.02218
- Altıntaş et al., Butterfly Effect (ICML 2025) — https://arxiv.org/abs/2506.13234
- Herrmann et al. (NeurIPS 2022) — https://proceedings.neurips.cc/paper_files/paper/2022/hash/222a2a46018a1e7b55ba48ba11932d04-Abstract.html
- Redman et al. (NeurIPS 2024) — https://arxiv.org/abs/2302.09160
- Porian et al. (NeurIPS 2024) — https://arxiv.org/abs/2406.19146
- Bergsma et al., Power Lines (NeurIPS 2025) — https://arxiv.org/abs/2505.13738
- Wang & Aitchison (ICML 2025) — https://arxiv.org/abs/2405.13698
- van der Wal et al., PolyPythias (ICLR 2025) — https://arxiv.org/abs/2503.09543
- Pythia README — https://github.com/EleutherAI/pythia · 160M NeoX checkpoint — https://huggingface.co/EleutherAI/neox-ckpt-pythia-160m-deduped-v1
- OPEN-1B (15 Sep 2026) — https://arxiv.org/abs/2609.17380
- LLM360 Amber — https://huggingface.co/LLM360/Amber · K2 — https://huggingface.co/LLM360/K2 · K2-Spike-1 — https://huggingface.co/LLM360/K2-Spike-1
- OLMo README — https://github.com/allenai/OLMo · CSV — https://raw.githubusercontent.com/allenai/OLMo/main/configs/official-0425/OLMo-2-0425-1B.csv
- Kim, TIC (ICML 2026 Hypothesis Testing Workshop) — https://icml.cc/virtual/2026/80057
- Kalra et al., SlimAdam — https://arxiv.org/abs/2503.01843 [V-snip]
- Gupta et al., rewarming — https://arxiv.org/abs/2308.04014 [V-snip]
- DeepSpeed universal checkpointing — https://deepspeed.ai/tutorials/universal-checkpointing [V-snip]
- SPAM — https://arxiv.org/abs/2501.06842 [V-snip]
