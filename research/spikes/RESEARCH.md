# Spike research — what is known (October 2026) and the ideas we adopt

Written 2026-10-07/08 for the spike-engineer role. The plan these ideas feed is `PLAN_V8.md` (Appendix A; they began as `docs/reference/PLAN_V7.md` §7). Every claim
about a paper below comes from the paper or its official code; links are at the end. Nothing here has been
run yet. Each idea gets a pre-registered test (§6) before any GPU hour is spent on it.

---

## 0 · The ideas, ranked by value per GPU-hour

| # | Idea | What it buys the paper | GPU cost (proxy) | Status |
|---|---|---|---|---|
| 1 | **Optimizer-state patching**: transplant `m`/`v` (or `w`) for a subset `S` from the clean branch `B*` into the damaged `B0` | The cleanest causal test of the thesis (does damage live in the *state* or the *weights*?) and an oracle target for C2 | ~20 short forks per spike instance | design (§4.2) |
| 2 | **Planted-poison benchmark** (`v_inflate`) | Exact ground truth for C2 (precision/recall), plus a recovery-time prediction the theory can fail | negligible | built (`induce.py`) |
| 3 | **Seedable natural spikes** (stress presets) + **cross-seed alignment** | Answers "all spikes are induced" with a *population* of natural spikes, and tells data-caused spikes from optimizer-caused ones | ~120k steps | built (`natural.py`) |
| 4 | **Realistic data cause** (`repeat_ngram`) | Replaces uniform random garbage with a spike cause documented in a production run (OLMo 2) | negligible | built (`induce.py`, in `spike_run`) |
| 5 | **Mechanism-grounded detection signals** (update-RMS, `v̄`-drift, stability margin) | Early warning tied to *why* Adam spikes, not just "loss went up" | free (logged as telemetry) | logging built (`capture.py`) |
| 6 | **Distribution-free false-positive control** (conformal / CUSUM / e-detectors) | A false-alarm guarantee a reviewer can't argue with, replacing grid-swept z-scores | CPU only | design (§3.3) |
| 7 | **Minimal stabilizing repair** | A principled "how much" for C3: repair just enough to restore stability | needs C2's HVPs | design (§5) |
| 8 | **External natural-spike validation** (LLM360 K2-V2 logs, K2-Spike checkpoints) | Detector recall on real 65B spikes at zero GPU cost | CPU only | design (§2.3) |

---

## 1 · Literature map and where the project stands

**Why Adam spikes — the current best account.** Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in
Adam* (ICML 2026), argue the root cause is the second moment decoupling from the gradient: `v_t` keeps
decaying geometrically (rate `β2`) while gradients rise. That inflates the preconditioned Hessian until its top
eigenvalue crosses the stability threshold. They describe five stages: stable descent → preconditioner decay →
onset → preconditioner growth → recovery. They find larger `β2` makes spikes likelier, and they predict onset
with the gradient-directional curvature `λ_grad(Ĥ) = gᵀĤg/‖g‖² > 2/η`. That predictor flags the onset step
itself; it does not give advance warning.

**The stability threshold, two equivalent forms.** Cohen et al. (*Adaptive Gradient Methods at the Edge of
Stability*) show the preconditioned sharpness `λ_max(P⁻¹H)`, with `P = diag(√v̂+ε)`, equilibrates at
`(2+2β1)/((1−β1)η)`, which is 38/η at β1=0.9. Bai et al. fold the momentum factor `(1−β1)/(1+β1)` into `Ĥ` and
write the same condition as `λ_max(Ĥ) < 2/η`. Our PLAN V6 §3.3 operator `H~ = D^-1/2 H D^-1/2` has the same
spectrum as `P⁻¹H` (they are similar matrices). So C2's eigenbasis is exactly the one this literature says
governs spikes.

**Stale second moment, measured.** Wortsman et al. (*Stable and low-precision training…*, 2023) found that
"28 out of 30 detected loss spikes follow an RMS spike by 1-8 iterations" (§3.4), where `RMS_t = √E[g²/v̂]`.
In App. D (ViT-Huge), 14 of 15 followed an RMS spike in the patch-embedding layer (RMS > 2.3). Their reading is that the second-moment estimator is out of date. Molybog et al. (2023)
describe the large-batch failure mode: Adam enters a regime where updates are large and uncorrelated with the
descent direction.

**Instabilities reproduce at small scale.** Wortsman et al. (*Small-scale proxies…*, ICLR 2024) show that
attention-logit growth and output-logit divergence appear in small models at high learning rates, and that
large-scale mitigations (qk-layernorm, z-loss) fix them there too. **This is what makes proxy-scale natural
spikes legitimate.**

**Production evidence.** OLMo 2 traced spikes to batches of repeated n-grams (e.g. `g4ODg4OD…`) and
filters documents with ≥32 repeated n-grams. LLM360 K2 released paired spike/normal checkpoints
(K2-Spike-1/2). K2-V2 reports most spikes fall in the first 40% of training, uses a two-tier alert/restart
monitor, and publishes its W&B logs.

**Mitigations — the baselines.** SPAM (global periodic `m`/`v` reset + spike-aware clipping `g² > θv`),
ZClip (z-score EMA clipping of the global norm), AdaGC (per-tensor EMA clipping). All three act globally or
on gradients. None of them localizes damage inside the optimizer state.

**Detection-only work.** *Mechanism-Aligned Observability for Training Instability of Large Language Models*
(arXiv 2606.28116, v3 2026-10-18; v1 was titled *Mechanism-Driven Monitors…*).
- It reports spectral concentration of the FlashAttention update, plus MoE-router monitors, that respond "before
  degradation becomes visible in global training metrics".
- The v3 abstract gives no quantified lead time; the earlier "thousands of steps" figure is gone.
- It reports no false-positive rates and does no state repair or counterfactual analysis.

**Where we stand.** As of 2026-10-07 I found no work that (i) localizes poisoned directions *inside* Adam's
state, (ii) repairs only those directions, and (iii) verifies the repair causally with deterministic
counterfactual forks. Bai et al. is now the closest prior work: cite it, adopt its `Ĥ`, and position C2/C3 as
the step after explanation, which is localization + repair + causal proof. Caveat: this was a targeted search,
not a systematic review. Redo it before submission.

---

## 2 · Natural spikes we can actually get

### 2.1 Stress presets (built: `natural.py`)
The grid is `{5×, 10×} LR × {β2 0.95, 0.999} × {clip 1.0, none}`, 5 seeds each, 3000 steps (≈120k proxy steps,
~2 P100-h, ≤0.2 MI300X-h).

Why each axis:
- **High constant LR** reproduces small-scale instabilities (Wortsman 2023).
- **β2 = 0.999** lengthens `v`'s memory. The e-folding time of `v` under small gradients is `1/(−ln β2)`:
  ≈19.5 steps at 0.95 versus ≈1000 at 0.999. That makes the decoupling stage of Bai et al. ~50× longer.
- **No clipping** removes the guard that hides spikes.

The selection rule is fixed in advance: the first qualifying spike per run, in preset-then-seed order, until
10 instances exist (`docs/reference/PLAN_V7.md` §4.2, kept by V8).

### 2.2 Cross-seed alignment (built: `natural.alignment_test`)
`get_batch` is a pure function of the step, so all seeds read identical data in identical order. If
natural spikes cluster at the same steps across seeds beyond chance, the cause is in the data (OLMo-style).
If they don't, the cause is the optimizer dynamics.

- **Statistic:** the number of cross-seed spike pairs within ±5 steps.
- **Null:** each seed keeps its spike count, with times redrawn uniformly.
- **Output:** a permutation p-value.

This is a cheap, novel attribution of natural-spike causes, and it decides which repair story applies to
which spike.

### 2.3 External validation, CPU only
- **K2-V2 W&B logs:** run `OnlineDetector` and `find_spikes` on real 65B loss/grad-norm curves. Report
  detector recall against the spikes K2-V2's own monitor flagged.
- **K2-Spike-1/2 paired checkpoints:** measure per-layer `‖w_spike − w_normal‖/‖w_normal‖` as
  natural-spike evidence of where the damage lands. These are weights only; no optimizer state is
  documented. A 65B model is never forked.

---

## 3 · Detection: mechanism-grounded signals and guarantees

### 3.1 Signals logged every step (built: `capture.tensor_signals` → `telemetry.npz`)

Per parameter tensor:

| Signal | Formula | Why |
|---|---|---|
| update RMS | `√mean(g²/max(v̂, ε²))` | Wortsman's stale-`v` indicator. Bounded by `1/√(1−β2)` because `v_t ≥ (1−β2)g_t²`, so use `ρ = RMS·√(1−β2) ∈ [0,1]` across β2 settings |
| spike ratio | `max g²/v̂` | SPAM's per-coordinate spike test (`θ = 5000`) |
| Adam update RMS | `√mean((m̂/(√v̂+ε))²)` | step size actually taken, per tensor |
| `v̄`-drift | `‖v − v̄‖/‖v̄‖` | the quantity PLAN V6 §3.6's poison score is built on |

### 3.2 Stability margin (needs C2's HVP; log it once that exists)
`S_t = η·λ_grad(Ĥ_t)/2`, where `S ≥ 1` means unstable along the gradient direction (Bai et al.'s onset
condition). The hypothesis worth testing: the *trend* of `S_t` (its slope over the preconditioner-decay stage)
gives lead time even though its level only flags onset. Cost: one HVP every k steps on a fixed probe batch.
Because SDPA flash kernels lack double-backward, use the math attention kernel for the HVP pass.

### 3.3 False-positive control with guarantees (CPU only)
- **Conformal run-level threshold.** Take `n` clean held-out runs, each with per-run maximum score
  `M_j = max_t s_t`. Set `τ = M_(⌈(n+1)(1−f)⌉)`, an order statistic. For exchangeable runs (seeds),
  `P(a new clean run ever false-alarms) ≤ f`. With `n = 20` and `f = 0.05`, `τ` is the largest of the 20
  maxima, so the guarantee is ≤ 1/21 ≈ 0.048. This replaces grid-sweeping z on the same runs we score,
  which carries a selection bias.
- **CUSUM / e-detectors.** CUSUM is `C_t = max(0, C_{t−1} + x_t − μ − k)`, alarming at `C_t > h`, where
  `x_t` is a standardized signal. Lorden-optimal delay for mean shifts. The e-detector version (Shin, Ramdas &
  Rinaldo) is nonparametric with average run length to false alarm ≥ 1/α.
  - **Combining signals:** averages (or mixtures) of e-values always stay valid. Products are valid only when each
    e-value is valid conditionally on the others, which is not automatic for signals from the same run.
  - **Caveat:** the guarantee needs μ and σ fixed in advance, not rolling estimates.
  - **Run-level bound:** P(any false alarm in n steps) ≤ n·α.

  Port a single-λ e-CUSUM in about 20 lines of numpy. Don't depend on River or confseq (numpy-pin conflict and a
  C++ build; see `PLAN_V8.md` §9.5).

---

## 4 · Localization ground truth

### 4.1 Planted poison (built: `induce.v_inflate`)
Scale `v` by `κ` on named tensors at `inject_step`. Afterwards
`v_{t+k} ≈ β2^k κ v + (1−β2^k) E[g²]`, so with `v ≈ E[g²]` the update along those tensors shrinks by
`1/√(β2^k κ + 1 − β2^k)`.

- **Half-speed recovery takes** `k½ = ln(κ/3)/(−ln β2)`, ≈158 steps at κ=1e4, β2=0.95.
- **Falsifiable predictions:**
  - **`B0`:** learning stalls in the target tensors for ≈k½ steps.
  - **Restoring `v_S ← v̄_S` (selective repair):** recovers fully.
  - **`Bg` (zero everything):** pays a global ~`1/√(1−β2)` ≈ 4.5× update overshoot.
  - **skip / clip:** do nothing, since no gradient is abnormal.
- **For C2:** the localizer must rank the target tensors first. Score it by precision/recall/AUROC
  against `recipe.params["targets"]`.

### 4.2 Optimizer-state patching (design; the strongest causal tool we have)
At the same step `t`, after the spike window, take the damaged state `Θ⁰ = (w, m, v)` from `B0` and the clean
state `Θ*` from `B*`. Both read the same data, which is what makes this valid. For a subset `S` (tensor
groups, or top-k `H~` directions), build `Θ^S` = `Θ⁰` with `S` replaced by `Θ*_S`, run `k` steps, and measure
`Δ = val_loss − val_loss(B*)`.

- **Sufficiency of `S`:** `[Δ(B0) − Δ(patch S)] / Δ(B0)` ≈ 1.
- **Necessity:** patching the complement `Sᶜ` recovers ≈ 0.
- **The thesis test:** patch `w` only vs `(m, v)` only vs both. This decomposes the damage into weight damage
  and optimizer-state damage, plus their interaction. If `(m, v)`-only patching recovers most of `Δ` after
  the weights have healed, the central claim holds.
- **Oracle localization:** greedy search over ~20 tensor groups (≈20 short forks per instance), which C2's
  unsupervised localizer is then scored against. This is the optimizer-state analogue of activation
  patching / causal tracing in interpretability.

Implementation needs both branches' states at step `t`. Capture them inside the two forks with
`snapshot.capture` in `on_step`, then fork from the patched dict with `run_fork`. **No harness change.**

---

## 5 · Repair math: minimal stabilizing repair (for C3)

Instead of resetting `v` on the localized set `S`, scale it just enough to restore stability. Write the
gradient-direction Rayleigh quotient as contributions from `S` and its complement, `λ = λ_S + λ_Sᶜ`. Scaling
`√v_S` by `a ≥ 1` divides `λ_S` by `a`. The smallest uniform `a` that reaches a target `λ* = (1−δ)·2/η` is

```
a = λ_S / (λ* − λ_Sᶜ)          (needs λ_Sᶜ < λ*; otherwise S is the wrong set — itself a diagnostic)
v_S ← a² · v_S
```

This turns "repair" from a heuristic into a projection onto the stable set. It also ties C3 to the theory
(§3.8 of PLAN V6): the repair succeeds exactly when the undamaged bulk `λ_Sᶜ` is already below threshold,
which is the same condition the ψ_k criterion tries to predict. Cost: one Rayleigh-quotient decomposition
per repair (a few HVPs).

---

## 6 · Pre-registered tests for these ideas

- **Detection signals (§3):** evaluated only on held-out seeds 1–4. A signal replaces the baseline detector
  only if it raises median lead by ≥1 step at conformal run-level FP ≤ 0.05.
- **`repeat_ngram`:** onset policy, calibration round 1 (`spike_run.py`). It enters the 30-instance design
  only as an extra type, never as a substitute.
- **`v_inflate`:** prediction `k½ ≈ 158` steps (κ=1e4, β2=0.95). The theory is counted as supported if the
  measured half-speed time falls within ±30%. A miss is reported, not explained away.
- **Cross-seed alignment:** per preset, α = 0.05, Bonferroni over 8 presets.
- **State patching:** report sufficiency and necessity with paired bootstrap CIs over the 5 seeds per site.
  The weight-vs-state decomposition is reported whatever it shows.

---

## 7 · What not to do

- Don't fork K2 (65B). It is log and checkpoint evidence only.
- Don't call a `corrupt_batch` or `repeat_ngram` onset detection "early warning".
- Don't score recovery on training loss. Held-out loss only (PLAN V7 §1).
- Don't tune detector thresholds on the seeds you report.
- Don't treat stress-preset spikes as "production" spikes. They are natural (nothing injected) but
  hyperparameter-stressed, and the paper says so.

---

## Sources

- Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam* (ICML 2026) — https://arxiv.org/abs/2506.04805
- Cohen et al., *Adaptive Gradient Methods at the Edge of Stability* — https://arxiv.org/abs/2207.14484
- Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (ICLR 2024) — https://arxiv.org/abs/2309.14322
- Wortsman et al., *Stable and low-precision training for large-scale vision-language models* — https://arxiv.org/abs/2304.13013
- Molybog et al., *A Theory on Adam Instability in Large-Scale Machine Learning* — https://arxiv.org/abs/2304.09871
- Huang et al., *SPAM: Spike-Aware Adam with Momentum Reset* — https://arxiv.org/abs/2501.06842
- Kumar et al., *ZClip* — https://arxiv.org/abs/2504.02507 · code https://github.com/bluorion-com/ZClip
- Wang et al., *AdaGC* — https://arxiv.org/abs/2502.11034 · code https://github.com/lishuai-97/AdaGC
- *Mechanism-Aligned Observability for Training Instability of Large Language Models* (v3; v1 titled *Mechanism-Driven Monitors…*) — https://arxiv.org/abs/2606.28116
- OLMo 2 (*2 OLMo 2 Furious*) — https://arxiv.org/abs/2501.00656
- LLM360 K2 — https://arxiv.org/abs/2501.07124 · K2-Spike-1 https://huggingface.co/IFM/K2-Spike-1 (the page now shows the `IFM` org; its checkpoint-branch links still point at `LLM360/K2-Spike-1`)

---

## Changelog
- **v2 (2026-10-22, Master Research run).** The old version is backed up at `master-research/raw/RESEARCH.v1.md`
  (git-excluded).
  - **Corrections:**
    - Wortsman counts: §3.4 has 28/30; App. D (ViT-Huge) has 14/15.
    - 2606.28116 is retitled (v3); the "thousands of steps" claim is gone.
    - e-value products need conditional validity; averages are always safe.
    - The K2-Spike page now lives under the `IFM` org.
  - **New prior art since v1** (each verified in `master-research/output/12_factcheck_citations.md`; BibTeX in
    `master-research/references.bib`):
    - Xu 2026 (optimizer-state surgery; m carries the trait);
    - Liu & Li 2026 (paired control/shock forks);
    - Chimera 2026 (moment-source ablation);
    - Xie 2026 (rank-3 QK projection, energy-matched control);
    - SWIFT 2023/24 (one-step Adam inversion, so our undo log is PARTIAL);
    - Biswas 2026 (v-excess protects recent writes; clearing it on the affected rows ≈ full reset);
    - Altenbernd 2026 (SDC → v→∞);
    - Megatron Core and OPEN-1B (bitwise replay; **OPEN-1B releases full optimizer state every 100 steps**);
    - Tang 2026 (Adam phase diagram);
    - Marin 32B (a counter-case: "rather than in optimizer state").
  - **These ideas now feed `PLAN_V8.md`.** The damage-migration factorial leads; this file's §2–§6 ideas live in
    Appendix A.
- LLM360 K2-V2 — https://arxiv.org/abs/2512.06201 · logs https://wandb.ai/llm360/K2-V2
- Shin, Ramdas & Rinaldo, *E-detectors* — https://arxiv.org/abs/2203.03532
- Thilak et al., *The Slingshot Mechanism* — https://arxiv.org/abs/2206.04817
