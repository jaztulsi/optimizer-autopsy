# Novelty matrix — v2 (2026-10-22)

## v2 additions (Lineage agent + today's kill-searches; most read as abstracts only, so they need full-text reads)

| Claim | New rival | What it owns | Effect on our claim |
|---|---|---|---|
| K4 (state patching) | **Xu 2026** — [2608.20442](https://arxiv.org/abs/2608.20442), *Stored in Optimizer State, Valued by Later Training* | "State surgery" moving m (or θ+m) between runs through matched futures, plus a discrete-adjoint costate. In their setting m carries 89.9%, w 10.6%, v −0.5% (subliminal trait transfer) | **K4 downgraded to PARTIAL.** Patching as a method is theirs. Ours: spikes, the v channel, a cut-time sweep, within-tensor granularity, the bitwise gate. Their "v ≈ 0%" is a falsification target for us. |
| K4 | **Chimera interventions** — [2607.06628](https://arxiv.org/abs/2607.06628) (grokking) | Moment-source ablation (reset / recipient / donor) | Cite as precedent for cross-trajectory hybrids |
| K3 (rank-limited edit) | **Xie et al. 2026** — [2608.02091](https://arxiv.org/abs/2608.02091) | Projects updates off the top-3 QK singular directions vs an energy-matched control | C3's *form* has precedent (on updates and weights). `Br` must be energy- and support-matched |
| K3 (when to reset) | **Topollai & Choromanska 2026** — [2603.16731](https://arxiv.org/abs/2603.16731) | A theory of *when* to reset quantized m/v | We own *which coordinates and how much*, not *when* |
| K1/K4 | **Guo 2026** [2608.24593](https://arxiv.org/abs/2608.24593); **Stewart 2026** [2605.17590](https://arxiv.org/abs/2605.17590); **COIC** [2607.07206](https://arxiv.org/abs/2607.07206) | Differentiating through optimizer state; L-BFGS memory-vs-parameter counterfactuals; factorial optimizer interventions | More precedent for the instrument family. Cite |
| evidence | **Altenbernd et al. 2026** — [2604.00726](https://arxiv.org/abs/2604.00726) (CCGrid) | SDC drives v → ∞ and freezes updates. Lists momentum resets as future work | Natural non-induced v-poisoning evidence; a group moving toward our space |
| K7 / baselines | **CAST** — [2610.19761](https://arxiv.org/abs/2610.19761) (Oct 2026) | Spectral clipping scaled by AdamW's previous v; 19B MoE | Possible newest baseline |
| reviewer risk | **Tang 2026** [2609.18314](https://arxiv.org/abs/2609.18314); **Li, Zhou & Xu 2026** [2607.21005](https://arxiv.org/abs/2607.21005); **Tian & Paquette 2026** [2610.13524](https://arxiv.org/abs/2610.13524) | Non-v spike mechanisms (superquadratic geometry, weight-norm criticality, random-feature models) | Strengthens the cause-vs-carrier framing |

**v2 verdict:** the white space in §4 still holds, but it is narrower. "Post-spike, sign-aware,
support-limited edits of stored m/v, causally verified" is ours. The patching method, bitwise replay and
rank-k edits of updates are not.

---

# Novelty matrix — v1 (2026-10-22)

*Built from the 16 rivals researched in `spike-related-work/results/` (most read in full text plus their
official code). v2 will add the Lineage agent's 2025–26 citing papers. Verdicts are bounded by those
searches. Any "nobody has done X" claim means "not found in the searches listed in the source JSONs".*

## 1 · Our claimed contributions

| ID | Claim |
|---|---|
| K1 | **C1 as a causal instrument**: paired forks from one pre-spike snapshot that differ only by a named optimizer-state intervention. Includes a data-aligned clean counterfactual B\* and a bitwise noop-vs-noop Δ = 0 gate. |
| K2 | **C2 direction-level localization**: which directions *inside* stored m/v are poisoned, scored against a pre-spike v̄ null and matched normal steps. |
| K3 | **C3 selective, rank-limited post-spike repair** of m/v that beats both no-fix and a full reset, verified by forks. |
| K4 | **Optimizer-state patching**: transplant w / m / v subsets from B\* into B0. Gives sufficiency and necessity, and splits lasting damage into weight vs state parts. |
| K5 | **Planted-poison benchmark** (`v_inflate`): exact ground truth plus a falsifiable recovery law k½ = ln(κ/3)/(−ln β2). |
| K6 | **Natural-spike population** via stress presets, plus a **cross-seed alignment test** that separates data-caused from dynamics-caused spikes. |
| K7 | **A detector with a distribution-free false-alarm guarantee** (conformal run-level threshold, e-detectors) on mechanism-grounded signals. |
| K8 | **Minimal stabilizing repair**: the smallest scaling of v on the localized set that puts the preconditioned curvature back under 2/η (a projection onto the stable set). |

## 2 · Claim × closest rival (overlap → delta → verdict)

| Claim | Closest rival(s) | Overlap | Delta (what is ours) | Evidence | Verdict |
|---|---|---|---|---|---|
| K1 | **Liu & Li 2026** (2608.19762) | Paired control/shock forks from a shared snapshot with a shared future; hybrid forks inject the delta into whole θ/m/v channels | Bitwise Δ = 0 gate (they report statistical only); real spikes, not small linear shocks; interventions *edit optimizer state* at sub-channel granularity; data-aligned B\* | JSON: counterfactual_method, determinism_level | **PARTIAL**. Paired forks are not new; our combination is. |
| K1 | **Megatron Core** (blog 2026-10-06), **OPEN-1B** (2609.17380) | Bitwise determinism and resume including optimizer state, at 48–2,432 GPUs | They re-run only the current step (Megatron) or replay for audit (OPEN-1B). No intervention branches, no counterfactuals | JSON: overlap_with_localize/repair = none | **Determinism itself is NOT ours.** Claim the instrument, not the determinism. |
| K2 | **Bai et al. 2026** (ICML) | λ_grad(Ĥ) = gᵀĤg/‖g‖² > 2/η onset predictor; block-level v decay before spikes | Direction-level score of *stored* m/v against a v̄ null; SNR component; validated vs planted ground truth and the patching oracle | JSON: overlap_with_localize "partial … curvature half" | **NEW** at direction level. The curvature *signal* is theirs: cite and adopt their Ĥ. |
| K2 | **Ramasinghe 2026** (2606.16768) | Online top eigenpair of P^-1/2 H P^-1/2 by warm-started power iteration, at up to 3B | Global single eigenvalue vs our per-direction poisoning scores; they never edit state | JSON | **NEW**. Their estimator is reusable tooling. |
| K2 | **Spectra 2026**, **StableAdamW RMS** (2304.13013), **AdaGC** | Top-k subspace of momentum (Spectra); per-tensor stale-v flag (RMS); per-tensor abnormal-gradient flag (AdaGC) | Post-spike poisoning of *stored* state vs preventive per-step signals | JSON | **NEW**. Use them as localizer controls ("energy-only", "RMS-only"). |
| K3 | **SPAM** (ICLR 2025), **Asadi 2023** | Zero m and v (globally; periodic or at known boundaries) | Post-spike, detector-timed, *selective and rank-limited*, causally verified | JSON: granularity = global | **NEW** |
| K3 | **Spectra 2026** | Rank-k edit of optimizer state (flatten the top-k singular values) | Preventive every step on momentum, no v, no spike, no causal test | JSON: overlap_with_repair "medium on form" | **NEW in purpose; closest in form.** Cite as the nearest rank-k edit. |
| K3 | **GI-Adam** (NeurIPS 2024), **Adam-Rel** (NeurIPS 2024) | v ← g² at init; step-counter reset at known events | Mid-training, localized, post-spike | JSON | **NEW**. Both become repair variants or controls. |
| K4 | **Liu & Li 2026** | Channel-level (θ vs m vs v) attribution of a perturbation's delayed effect | Subset- and direction-level sufficiency/necessity for real spikes; weight-vs-state split *after the weights heal* | JSON: overlap_with_localize "channel interventions" | **PARTIAL→NEW**. Channel-level is theirs; finer granularity and spikes are ours. |
| K5 | none of the 16 | — | Planted v-poison with exact ground truth and a closed-form recovery law | not found in the 16 | **NEW (pending Lineage)** |
| K6 | **OLMo 2** (data cause: repeated n-grams), **Marin 32B** (architecture cause), **Wortsman 2024** (small-scale reproduction) | Spike-cause attribution by inspection / ablation | A permutation test across seeds sharing a data order to split data- vs dynamics-driven spikes | RESEARCH.md §2.2 | **NEW (pending Lineage)** |
| K7 | **ZClip** (z-score), **OLMo-core** (6σ / 128), **Megatron** (10× max), **SPAM** (g² > θv) | Online spike triggers | Run-level distribution-free FP guarantee; e-detector combination of signals | JSON: trigger_rule_and_defaults | **PARTIAL**. The statistics exist in other fields; applying them here looks new (pending Adjacent-Field). |
| K8 | **Bai 2026** (global ε/β2 fixes), **Spectra** (flatten top-k) | Restore stability by changing the preconditioner | The *minimal* localized v scaling that hits λ ≤ (1−δ)·2/η: a projection onto the stable set | RESEARCH.md §5 | **NEW (pending Method Hunter)** |

## 3 · Rivals × dimensions

| Rival | State edited | Granularity | Timing | Counterfactual | Determinism | Localization | Causal decomposition | Ground truth | Largest scale |
|---|---|---|---|---|---|---|---|---|---|
| Bai 2026 | none (reads v) | global fixes | preventive / analysis | none | none | block-level | none | none | 187M |
| Liu & Li 2026 | none | channel | analysis | **paired forks** | statistical | channel | **channel** | none | 1.4B (continued) |
| Ramasinghe 2026 | none | global eig | preventive | none | none | global eig | none | none | 3B |
| Spectra 2026 | momentum top-k | **rank-k** | preventive | none | none | top-k subspace | none | none | 8B |
| SPAM | m, v zeroed | global | periodic | none | none | none | none | none | 1B |
| GradientStabilizer | none (g norm) | per-tensor | preventive | none | none | none | none | none | 350M |
| AdaGC | none (g clip) | per-tensor | preventive | none | none | tensor (gradient) | none | none | 10B MoE |
| ZClip | none (g clip) | global | at spike | none | none | none | none | none | 1B |
| StableAdamW | none (per-tensor lr) | per-tensor | preventive | none | none | tensor (RMS) | none | none | ~1B CLIP |
| Asadi 2023 | m, v, t | global | periodic | none | statistical | none | none | none | Atari nets |
| GI-Adam | v init | per-coord, all | init | none | none | none | none | none | small |
| Adam-Rel | step t | global | periodic | none | none | none | none | none | RL nets |
| Marin 32B | none (arch fix) | global | at / after spike | rollback | none | layer diag | none | none | 32B |
| Megatron | none | global step | at spike | rerun | **bitwise** | none | none | none | 2,432 GPUs |
| OPEN-1B | none (skip 5) | global | at spike | replay (audit) | **bitwise** | layer diag | none | none | 1.6B |
| OLMo-core skip | none (skip) | global | at spike | none | none | none | none | none | 32B |
| **Ours** | **m, v (selective)** | **direction / rank-k** | **post-spike** | **paired forks + B\*** | **bitwise gate** | **direction** | **subset w/m/v** | **planted** | 11M + 124M |

## 4 · White space (cells no rival fills)
1. **Post-spike × direction-level × edits stored m/v × verified by paired bitwise forks.** This is the core,
   and nobody occupies it.
2. **Planted ground truth for optimizer-state poisoning.** No rival has any ground truth.
3. **Subset-level causal decomposition** (weights vs m vs v) of *lasting* spike damage. Liu & Li stop at
   whole channels and small perturbations.

## 5 · Where we are weakest
- **Scale:** every rival except GI-Adam runs ≥ 187M; most run ≥ 1B. Our 11M + 124M is the most attackable
  cell. A reviewer will ask whether direction-level poisoning even exists at 1B. Mitigations: the OPEN-1B
  natural-spike checkpoints, if the m/v really are released (1.6B, no training needed), and the
  scale-ladder curves.
- **Cause vs carrier:** Marin shows spikes rooted in the attention stack. Our claim must be "damage is
  carried in optimizer state and repairable there", never "optimizer state is the root cause".
- **Determinism wording:** claim the instrument, not determinism.
