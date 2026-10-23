# Phase 3 · Novelty Engine — 63 candidate ideas, stress-tested and ranked (2026-10-22)

> **v2 corrections (2026-10-22, after wave 2 + fact-check):**
> - **#5 (undo log):** novelty 5 → **3, PARTIAL**. SWIFT (2302.06173, TPDS 2024) inverts Adam one step back. Ours is state-only retroactive deletion of a past window plus the path split.
> - **#12:** cite Biswas 2026 as the closest selective-v precedent.
> - The wave-2 ideas (W1–W12, R1–R35, S1–S12, E1–E9) and the V8 X-ideas live in `PLAN_V8.md` Appendix A.

**Inputs:**
- Phase 2 outputs: `02_lineage.md`, `02_adjacent_fields.md` (T1–T19), `02_method_catalogue.md` (A1–A17,
  B1–B17, C1–C10), `02_skeptic_review.md` (F1–F3, M4–M12, m13–m15).
- The 16 rival files.
- `03_novelty_matrix.md`.
- Four extra kill-searches run today (§3).

**Target:** ICML 2027, deadline about late January 2027 (the Verifier is confirming the date). That leaves
about 13 weeks and ~15 MI300X-h now, ~100 h total.

**Scope rule (user, 2026-10-22):** keep the main idea, localize and repair poisoned Adam state. Adding a
second contribution or reframing is allowed if it makes the project stronger.

**Scoring (1–5, higher is better on every axis):**
- **N** novelty, after the kill-search
- **F** feasibility within our compute and calendar
- **I** impact on an ICML accept
- **R** risk, inverted (5 = low risk)
- **V** validation ease, inverted (5 = easy to validate)
- **D** defensibility against reviewers

The top 15 get a reason for each axis. The rest get one line that names the axes that drive their score.
Novelty verdicts only cover the searches actually run.

---

## 1 · Top 15 (per-axis reasons)

| Rank | Idea | Lens | N | F | I | R | V | D | Σ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Damage-migration curve.** A 2³ factorial of cross-world hybrids over (w, m, v) at cut times spike_end + {1, 10, 30, 60, 150}, measured as end-of-schedule held-out Δ (T2, F2). | decomposition, causal | 3: Xu 2026 does state surgery and Chimera 2607.06628 runs moment-source ablations, but neither covers spikes or a cut-time sweep | 5: about 6 forks per (instance, t1), no harness change | 5: turns the thesis from assumed into tested; answers Marin and F2 | 4: either outcome is publishable | 5: a direct measurement | 4: estimand vocabulary from Pearl and VanderWeele | **26** |
| 2 | **Poisoned-state transplant plus dose-response** ("noising"). Put B0's (m, v) into B\* at varying κ, then into a fresh healthy run. Does poisoned state alone cause a spike on clean weights and data? | inversion, causal | 4: no hit in a targeted search today | 5: a few forks | 4: the most direct "state is a carrier" evidence | 4: a null result is also informative | 5: binary outcome plus dose curve | 4 | **26** |
| 3 | **Decoy-augmented planted benchmark.** `v_inflate` with causally inert decoys (large ‖v−v̄‖ on near-zero-gradient rows or LayerNorm), m-only plants, and κ dose levels (C4). | benchmark | 4: no rival has ground truth | 5: plants go into snapshots, about 0 GPU | 3: an enabler, not the headline | 5 | 5: exact truth | 4: closes the "too easy" objection | **26** |
| 4 | **Placebo replicates.** K = 5 runs with a 1-ulp perturbation per branch, so the repair effect must exceed a matched-placebo spread (M8). | methodology | 3: standard idea, new here | 5: about 1 P100-h | 3 | 5 | 5 | 5: kills the "chaotic single sample" objection | **26** |
| 5 | **Exact undo log.** Subtract the spike window's direct contribution from m and v, using gradients rebuilt from consecutive ring-buffer snapshots. A state-only retroactive skip, weights untouched (T1). | cross-domain (databases) | 5: no hit in 2 searches (T1 plus today) | 5: 3 forks, 0 HVPs, no new logging | 4: a cheap deployable repair, and it splits direct vs indirect path effects | 3: may recover < 20% on `lr_bump` | 4: falsifiable predictions written down | 4 | **25** |
| 6 | **Adaptive group-testing oracle.** Delta debugging plus binary splitting over tensor, row and head groups (T3/A8); SuSiE credible sets when interactions appear. | cross-domain (software engineering) | 4: new for optimizer state | 5: 5–18 forks vs 21–256 | 3: makes a causal oracle affordable at 124M | 4: falls back to ddmin | 5 | 4 | **25** |
| 7 | **QK-Norm ablation of the state-carried fraction** (M4). | failure probing | 3 | 4: a model variant plus a stress re-run, about 2.5 P100-h | 5: meets Marin's counter-evidence head on | 4 | 4 | 5 | **25** |
| 8 | **Strong baseline set:** rollback-and-skip (PaLM), Adam-MR, bias-correct Bg, OLMo-core 6σ skip, GradientStabilizer, StableAdamW. Primary contrast = repair vs best non-oracle baseline, measured as gap closed G (M5). | simplicity test | 1 | 4: about 0.5 P100-h plus about 15 p-h of code | 5: removes the strawman | 5 | 4 | 5 | **24** |
| 9 | **Stats that hold up:** pre-registered primary contrast, TOST equivalence and non-inferiority, mixed model Δ ~ branch + site + (1 \| seed), data-order seeds, pilot SD before locking n (M7, C8). | methodology | 1 | 5 | 4 | 5 | 5 | 5 | **25** |
| 10 | **Hybrid feasibility check.** Every honest Adam state satisfies \|m_i\| ≤ C√v_i (C ≈ 1.165 at 0.9/0.95). Count violations on every patched hybrid; add an off-manifold control (T8, F2). | hygiene | 3 | 5 | 2 | 5 | 5 | 5 | **25** |
| 11 | **Signed localizer:** AtP-state in log-v (one backward) for magnitude, plus the two-sided innovation test (prior-v NIS) for sign. Scored against the oracle (A11/T6 + T5). | composition | 4: AtP over (m, v) for spikes not found; Xu's costate is adjacent | 4 | 4: makes localization cheap and signed | 3: Adam's 1/√v nonlinearity | 4 | 4 | **23** |
| 12 | **Signed, stability-certified, support-limited restoration:** v_S(α) = v_S^(1−α) T_S^α toward v̄ or fresh g², with the largest α whose λ_grad ≤ (1−δ)·2/η (B9). | reformulation | 4 | 4: 6–8 HVPs per repair | 4: fixes both poison signs (F3) with a certificate | 3: depends on the λ_grad proxy | 4 | 4 | **23** |
| 13 | **Realistic proxy:** warmup + cosine to a fixed end at ≥ 20 tokens per *non-embedding* parameter, plus one middle rung (6L/384) (F1). | external validity | 1 | 4: about 5–9 P100-h | 5: F1 is fatal without it | 4 | 4 | 5 | **23** |
| 14 | **SpikeState release:** spike instances plus (w, m, v, v̄) snapshots, clean twins and oracle damage maps, on the HF Hub with Croissant, so localizers can be benchmarked without forks. **Second contribution.** | new benchmark | 4: no public spike-snapshot set with moments found | 4: about 10–15 p-h packaging | 4: raises the paper's floor | 4 | 4 | 4 | **24** |
| 15 | **Parity residual** r_v = v_t − β2v_{t−1} − (1−β2)g_t², zero for honest Adam. It flags non-gradient faults (bit flips, SDC, planted poison) and isolates them with a fault-signature matrix (T8). | cross-domain (FDI) | 4 | 5 | 3: links to SDC (Altenbernd 2026) | 4 | 5 | 3 | **24** |

## 2 · The remaining 48 (one-line reasons)

| # | Idea | Source | N F I R V D | Σ | Driving reason |
|---|---|---|---|---|---|
| 16 | λ_grad decomposition by group as localizer, plus repair-success certificate sign(λ\*−λ_Sᶜ) | A3, #2 | 4 4 4 2 4 3 | 21 | R: superquadratic geometry (2609.18314) may break it |
| 17 | LMI minimal diagonal repair (H ⪯ cD, linear in 1/(√v̂+ε)) | T4 | 4 3 3 3 4 4 | 21 | F: Lanczos plus solver; only covers under-damping |
| 18 | OPEN-1B offline localization on released optimizer-state checkpoints | F1d | 4 2 5 2 3 4 | 20 | F/R: m/v availability unverified, 19 GB, proprietary replay |
| 19 | Data-side attribution of the spike batch → wte rows → data-aligned natural B\* | A17 | 3 4 4 3 3 3 | 20 | I: closes improvement A4; V needs per-example gradients |
| 20 | Muon + AdamW control: does damage persist in Muon momentum? | m13 | 3 4 3 4 4 4 | 22 | D: answers "AdamW-only"; low novelty |
| 21 | Online "state airbag": ring buffer + detector + undo log as a drop-in optimizer wrapper, with overhead measured | ours | 4 3 5 3 3 3 | 21 | I: practitioners use it; F: engineering in 13 weeks |
| 22 | Anti-windup invariance: post-spike drift on spiked coordinates is independent of the clip threshold, which explains why clipping only softens spikes (Marin) | T9 | 4 5 3 3 4 3 | 22 | N: new mechanistic claim; I: medium |
| 23 | Closed-form recovery law for signed poisoning (over- vs under-damping) predicting which repair is needed | ours | 3 4 3 3 4 3 | 20 | V: falsifiable against `v_inflate` k½ |
| 24 | Path-specific effects: direct (spike g → state) vs indirect (via weights), from undo vs patch | T1+T2 | 4 5 3 4 4 4 | 24 | merged into #1/#5 |
| 25 | Recovery-time survival analysis (Kaplan–Meier) vs the k½ prediction | C7 | 2 5 3 4 4 4 | 22 | V: cheap dense evals |
| 26 | Repair-top-k deletion/insertion curves score localizers *for repair* | C5 | 3 4 3 4 4 4 | 22 | D: standard in attribution |
| 27 | Edit-size Pareto with energy- *and* support-matched `Br` (Xie 2026 control) | C10 | 3 5 3 4 4 5 | 24 | D: makes "selective" a measured property |
| 28 | Recovery-per-GPU-second Pareto vs rollback | M10 | 2 5 4 4 4 5 | 24 | I: answers "why not roll back" |
| 29 | Non-adaptive compressed-sensing patch masks (Lasso / SuSiE), parallel on 192 GB | A9 | 3 4 3 3 4 4 | 21 | F: more forks than #6 at small s |
| 30 | Lanczos top-k eigenvector-mass localizer | A4 | 3 3 3 2 3 3 | 17 | R: spurious alignment (Song 2405.16002) |
| 31 | Hutchinson diagonal of H̃ for coordinate-level curvature | A5 | 3 4 2 3 3 3 | 18 | I: noisy at 30 probes |
| 32 | Fresh-gradient Fisher mismatch r = v̂ / Ê[g²] (signed, cheap) | A6 | 3 5 3 4 4 3 | 22 | merged as a sign feature in #11 |
| 33 | Spectral / row-factor analysis of log(v/v̄) | A12 | 3 5 2 4 3 3 | 20 | I: magnitude-only; a control |
| 34 | Per-tensor change-point "first mover" localizer | A13 | 2 5 2 3 3 3 | 18 | correlational |
| 35 | Multi-stream sparse change detection (Xie–Siegmund mixture GLR) | T11 | 3 4 2 3 4 3 | 19 | I: detection side only |
| 36 | Robust v̄ and MAD-based poison z-scores | T13 | 2 5 2 4 4 4 | 21 | hygiene |
| 37 | Principal angles between B0 and B\* top-k H̃ subspaces | T14 | 3 3 2 3 3 3 | 17 | I: descriptive |
| 38 | HVP-free directional curvature via finite differences (runs under flash SDPA) | T15 | 2 5 3 4 4 3 | 21 | DONE as a tool (Kalra 2026; Kim & Lee 2026); adopt |
| 39 | Synthetic control for the natural-spike counterfactual | T10 | 4 3 3 2 3 3 | 18 | R: pre-period fit may fail |
| 40 | Spectrum-based fault localization across the population (Ochiai) | T18/A14 | 2 5 2 3 3 2 | 17 | correlational prior for #6 ordering |
| 41 | Learned poison probe | A15 | 2 4 2 2 3 1 | 14 | D: circular planted → natural |
| 42 | Distributed alignment search for rotated m directions | A16 | 5 1 3 1 2 2 | 14 | F: 40 p-h; interpretability-illusion risk |
| 43 | Shapley over tensor groups | A10 | 2 2 2 3 4 3 | 16 | F: too many forks |
| 44 | m projection off top-k H̃ eigenvectors | B10 | 4 4 2 2 4 3 | 19 | I: m forgets within about 10 steps |
| 45 | v ← mean g² from K fresh micro-batches (GI-Adam idea) | B6 | 3 4 3 3 4 4 | 21 | a repair variant inside #12 |
| 46 | Zero-LR re-warm on S using aligned steps | B7 | 3 4 2 3 4 3 | 19 | alignment constraints |
| 47 | Transient per-tensor update clipping (selective StableAdamW) | B12 | 3 3 3 3 4 3 | 19 | needs a custom step |
| 48 | Selective ε increase | B11 | 2 3 2 3 4 3 | 17 | param-group plumbing |
| 49 | Transient selective β2 lowering | B13 | 2 4 2 3 4 2 | 17 | reported weaker (Biswas 2026) |
| 50 | "Inoculation": a selective v-floor v ← max(v, q·v̄) on groups (AMSGrad-like), to test whether localization is needed at all | inversion | 3 4 3 3 4 3 | 20 | a useful negative-control branch |
| 51 | Negative-result framing if the (m, v) share ≈ 0 after about 50 steps ("damage is weight-carried") | negation | 3 5 4 4 4 4 | 24 | **pivot insurance**, built into #1 |
| 52 | K2-V2 W&B logs: detector recall at 65B, CPU only | RESEARCH §2.3 | 2 4 3 3 3 3 | 18 | external detector check |
| 53 | Untied-embedding proxy variant to remove the wte artifact | m14 | 2 4 2 4 4 4 | 20 | D: an artifact control |
| 54 | Place the stress presets on Tang 2026's (β1, β2) phase diagram | lineage | 2 5 2 4 4 4 | 21 | positioning |
| 55 | Max attention-logit telemetry linking poisoned v to logit growth (Marin bridge) | M4 | 3 5 3 4 4 4 | 23 | free telemetry; with #7 |
| 56 | Token-level story: which wte rows hold poison under `repeat_ngram` | #3 combo | 3 4 3 3 4 3 | 20 | interpretable figure |
| 57 | Concurrent bitwise forks on one 192 GB card | M12 | 2 3 3 3 3 3 | 17 | engineering; determinism unverified |
| 58 | Selective m+v+t reset with per-parameter step reset | B4 | 2 5 3 4 4 4 | 22 | simplest selective baseline |
| 59 | Row-level v restoration for embedding rows | B5-row | 3 5 3 3 4 3 | 21 | inside #12 at row support |
| 60 | Conformal run-level FP guarantee plus e-detector combination on NIS / RMS / v̄-drift | RESEARCH §3.3 | 3 4 3 4 4 4 | 22 | backup B2 core |
| 61 | Lower-tail NIS CUSUM gives lead time on β2 = 0.999 presets (Bai's decay stage) | T5 | 4 4 3 3 4 3 | 21 | backup B2; pre-registered lead test |
| 62 | Micro-batch robust aggregation (trimmed mean) as a prevention baseline | T16 | 2 4 2 3 4 3 | 18 | baseline only |
| 63 | Plasticity transfers: β1 = β2 control optimizer; utility-ranked unit resets | T17 | 2 4 2 3 3 3 | 17 | tangential |

## 3 · Kill-searches run today (WebSearch, 2026-10-22; bounded, not systematic)

| Query (short) | Result | Verdict |
|---|---|---|
| Transplant a spiked run's Adam second moment into a healthy run; dose-response | Only Bai 2026 (mechanism), SPAM, Kourkoutas-β ([2508.12996](https://arxiv.org/abs/2508.12996), adaptive β2) | **NEW** (#2) |
| Retroactively subtract the spike's gradient from Adam's EMAs | Only SPAM (zero spiked gradients, periodic reset) | **NEW** (#5) |
| Inflated v after a spike stalls learning; localized reset | Bai's 5-stage cycle; a batch-size transition paper ([2604.26687](https://arxiv.org/abs/2604.26687)); Molybog | **NEW** for localized sign-aware repair (#12) |
| Causal decomposition of spike damage into weights vs moments with fork patching | Chimera interventions ([2607.06628](https://arxiv.org/abs/2607.06628), grokking); Stewart ([2605.17590](https://arxiv.org/abs/2605.17590), L-BFGS); COIC ([2607.07206](https://arxiv.org/abs/2607.07206)); Xu 2026 (Lineage) | **PARTIAL**: the method exists in other settings; spikes plus the cut-time sweep are new (#1) |

## 4 · Which ideas are renames or already done (said plainly)
- **"Optimizer-state patching" as a method is NOT new.** Xu 2026 ([2608.20442](https://arxiv.org/abs/2608.20442))
  and Chimera ([2607.06628](https://arxiv.org/abs/2607.06628)) both do it. We may claim only its application to
  spikes, the cut-time sweep, row/direction granularity, and the bitwise gate.
- **Bitwise replay is not new** (Megatron, OPEN-1B, Unlearning-at-Scale).
- **HVP-free curvature (#38) is done.** Adopt it, don't claim it.
- **A per-layer "SNR of v" exists** (SlimAdam 2503.01843). C2's SNR must be named *gradient-directional*.
- **"Rank-limited repair of v" is ill-posed**, because v is diagonal. Say **support-limited** for v; rank
  only applies to m.
