# 10 · Wave-2 Novelty Engine: 35 new ideas from lenses not yet used (2026-10-22)

*Wave-2 agent. Reading and web search only. **Nothing was executed.** Every idea here is a candidate, not a result.*

**What this extends:** `04_ideas_ranked.md` (#1–#63), `02_adjacent_fields.md` (T1–T19), `02_method_catalogue.md`
(A1–A17, B1–B17, C1–C10), `05_shortlist.md` (PRIMARY + 2 backups) and `09_analysis.md`. I checked every idea below
against those lists. Where an idea comes close to an existing one, the overlap is named. Ideas that turned out to be
already taken are kept and marked as such, not dropped (SKILL Phase 3C: "keep every candidate").

**The PRIMARY direction these ideas plug into:**
- the w/m/v **damage-migration curve** (#1) and the poisoned-state transplant (#2);
- the localizer: group-testing oracle (#6) plus signed AtP-state (#11);
- the repair: **exact undo log** (#5) and **signed, stability-certified, support-limited restoration** (#12), scored as
  gap closed G against strong baselines (#8).

**Labels:**
- **[verified]**: I read the abstract or page myself today (WebFetch).
- **[verified-snippet]**: confirmed only from search-result text, not the source page. Weaker; re-read before citing.
- **[inferred]**: my own derivation or mapping.
- **[assumed]**: an unmeasured cost or parameter.
- **UNVERIFIED**: could not confirm; kept out of any claim.

Novelty verdicts (**NEW / PARTIAL / TAKEN**) cover only the queries listed. They are not a systematic review. Semantic
Scholar was not used in this wave.

**Scores**, 1–5, higher is better on every axis:
- **N** novelty after the kill-search
- **F** feasibility (13 weeks, ~60–100 MI300X-h)
- **I** impact on an ICML accept
- **R** risk, inverted (5 = low risk)
- **V** validation ease
- **D** defensibility

**Cost unit [assumed]:** one proxy fork ≈ 7.5 s at the "10× P100" MI300X scenario of `09_analysis.md` §6. Nothing has been
measured on the MI300X.

---

## 0 · Ranked summary table

| Rank | ID | Idea (short) | Lens | Verdict | N F I R V D | Σ |
|---|---|---|---|---|---|---|
| 1 | R20 | **Protection law:** a spike's damage persists because the v-excess *protects the spike's own write*; restoration works by removing that protection | negation / mechanism | PARTIAL | 4 5 4 4 4 4 | 25 |
| 2 | R22 | **Bit-flip vulnerability map of m vs v:** v flips cause *silent* state-only poison that no loss detector sees | reliability / SDC | NEW (for moments) | 4 5 3 4 5 4 | 25 |
| 3 | R12 | **Spike phase × cooldown:** the same spike injected at 5 LR phases; does the decay phase erase state poison? | scheduling | NEW | 4 4 4 4 4 4 | 24 |
| 4 | R3 | **Forensic batch reconstruction from Δlog v of wte rows:** recover the offending tokens with no data logs | interpretability / inversion | NEW (for spikes) | 4 4 4 3 5 4 | 24 |
| 5 | R27 | **Stall + decoupled weight decay:** a v→w migration route whose size is set by β2 | mechanism | NEW | 4 5 3 3 5 4 | 24 |
| 6 | R6 | Per-token functional damage map (rare vs frequent tokens) | evaluation | NEW (as a spike endpoint) | 3 5 3 5 5 4 | 25 |
| 7 | R28 | Irreversibility via the linear-mode-connectivity barrier between B0, repaired and B\* | landscape | PARTIAL | 3 5 3 5 5 4 | 25 |
| 8 | R23 | Poison contagion: growth of the support of \|log v⁰/v\*\| over time | epidemiology | PARTIAL | 4 5 3 4 4 4 | 24 |
| 9 | R1 | Spike information half-life: how long w, m and v keep decodable identity of the spike, above the placebo-chaos floor | information theory | NEW | 4 5 3 4 4 3 | 23 |
| 10 | R16 | Universal poison sites across seeds and rungs (a fixed-prior localizer) | population | PARTIAL | 3 4 4 4 4 4 | 23 |
| 11 | R5 | Frequency-calibrated reference v for embedding rows (no v̄ needed) | representation | PARTIAL | 3 5 3 4 4 4 | 23 |
| 12 | R19 | Beneficial spikes: is any benefit state-carried (implicit annealing)? | inversion | PARTIAL | 3 5 3 4 4 4 | 23 |
| 13 | R9 | State format as a poison amplifier: 8-bit blockwise, Adafactor factors, 4-bit ratio codebook | compression | PARTIAL | 4 3 4 3 4 4 | 22 |
| 14 | R14 | Micro-batch-resolved undo | composition | PARTIAL | 3 4 4 3 4 4 | 22 |
| 15 | R11 | AdEMAMix slow momentum as an m-channel instrument | instrument | NEW (for state share) | 3 4 3 4 4 4 | 22 |
| 16 | R29 | Streaming Count-Sketch localization (heavy hitters of Δlog v) | streaming algorithms | NEW | 4 4 3 4 4 3 | 22 |
| 17 | R2 | Rate–distortion of repair (bits per unit of gap closed) | information theory | PARTIAL | 3 5 2 5 4 3 | 22 |
| 18 | R7 | Self-healing dual-timescale v (fast shadow v + consistency test) | self-healing optimizer | PARTIAL | 3 4 4 3 4 3 | 21 |
| 19 | R18 | Optimizer hot-swap on S: sign-momentum for k steps, so poisoned v is bypassed | inversion | NEW (for repair) | 4 4 3 3 4 3 | 21 |
| 20 | R21 | Adversarial optimizer-state poisoning (worst-case minimal state edit) | adversarial | NEW | 4 4 3 3 4 3 | 21 |
| 21 | R32 | Conformal localization sets with coverage of the oracle set | uncertainty | NEW (in context) | 3 4 3 3 4 4 | 21 |
| 22 | R24 | DiLoCo cross-worker donor repair | distributed | NEW | 4 3 3 3 3 3 | 19 |
| 23 | R13 | Localizer-in-the-loop data filter | curriculum | PARTIAL | 3 3 4 3 3 3 | 19 |
| 24 | R15 | Unsupervised spike taxonomy from damage fingerprints | population | PARTIAL | 3 4 3 3 3 3 | 19 |
| 25 | R31 | State drop at stage boundaries as a natural global reset | practice | TAKEN (= Bg / Adam-MR) | 1 5 2 4 4 3 | 19 |
| 26 | R4 | Sparse autoencoder over optimizer-state rows ("what does v know?") | interpretability | NEW | 5 3 3 2 3 2 | 18 |
| 27 | R10 | Kronecker-factored preconditioners (SOAP/Shampoo): rank-limited repair becomes well-posed | generalization | PARTIAL | 4 2 3 2 3 3 | 17 |
| 28 | R17 | Scaling law of the state share φ_state(N) | scale | PARTIAL | 3 3 4 2 3 2 | 17 |
| 29 | R26 | Path patching for optimizer state (cross-tensor influence graph) | causal / interpretability | NEW | 4 2 3 2 3 3 | 17 |
| 30 | R30 | Repair policy from heterogeneous treatment effects | causal ML | NEW (bounded) | 3 3 3 2 3 3 | 17 |
| 31 | R33 | Snapshot-free rewind by inverse Adam steps | reversible computation | PARTIAL | 2 4 2 3 4 2 | 17 |
| 32 | R25 | MoE expert-state poisoning | architecture | PARTIAL | 3 2 3 2 3 3 | 16 |
| 33 | R34 | Vectorized counterfactual forks (vmap over stacked fork states) | systems | TAKEN (as a tool) | 1 3 2 3 4 3 | 16 |
| 34 | R8 | Meta-learned repair operator | learned optimizers | PARTIAL | 3 2 3 2 3 2 | 15 |
| 35 | R35 | Agentic "autopsy" automation | systems / agents | PARTIAL | 2 3 2 3 2 2 | 14 |

---

## 1 · The 35 ideas

### R1 · Spike information half-life (decodability vs damage)
- **Pitch.** Damage tells us how much a spike hurts. Information tells us how long Adam *remembers* which spike happened.
  The two can come apart, for example "v still encodes the spike but the loss no longer cares". Measuring both per
  channel (w, m, v) adds an information-theoretic axis to the migration curve.
- **Mechanism (one line).** Run K spike variants at the same step (K = 8 `corrupt_batch` token seeds). At each cut time,
  decode the variant from Δw, Δm, Δv vs B\* with a leave-one-seed-out linear decoder, giving Î(variant; channel_t) ≤
  log₂K bits as a function of t. **[inferred]**
- **Key subtlety [inferred].** Deterministic replay has no noise, so any fp32 difference is decodable and decoding would
  stay trivially at log₂K. Information must therefore be measured **above the placebo-chaos floor**. Use #4's 1-ulp
  placebo replicates as the noise model. The half-life is the step at which accuracy falls to the
  chance-plus-placebo-confusion level.
- **Minimal experiment.**
  - 8 variants × 4 seeds, reusing #1's cut times {s+1, 10, 30, 60, 150}.
  - About 32 extra forks plus the placebo forks, ≈ 5 min proxy [assumed].
  - Pre-registered prediction [inferred]:
    - the v-information half-life tracks ln2/(−ln β2) on spiked wte rows (13.5 steps at 0.95);
    - m information is lost by about s+50;
    - w information persists the longest.
- **Closest prior work.**
  - Guo 2026, *Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions*,
    [2608.24593](https://arxiv.org/abs/2608.24593) (2026-08-25) [verified]. It treats the moments as a channel that
    transports a loss-weighting change over 8 steps. No MI, no spikes, no repair.
  - Kim 2026, *Second-Moment Memory in Coordinatewise Adam*, [2608.15824](https://arxiv.org/abs/2608.15824)
    (2026-08-16) [verified]. It gives the optimization cost of v's memory length M₂ = (1−β2)⁻¹ and does not address
    information content.
- **Kill-search.** "mutual information optimizer state memory of a training batch decodability Adam moments"; "probing
  what Adam optimizer second moment encodes representation of training history". **NEW.**
- **How it plugs in.** A figure next to the migration curve: bits remaining vs Δ remaining per channel. If v stays
  informative after Δ_v ≈ 0, restoring v is harmless but useless, which tells C3 where to stop.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | no MI-over-moments hit |
| F | 5 | reuses #1's forks |
| I | 3 | a secondary axis |
| R | 4 | the floor definition is the main risk |
| V | 4 | standard decoder |
| D | 3 | reviewers may call it "just a probe" |

### R2 · Rate–distortion curve of repair
- **Pitch.** "Selective" repair should be measured in bits: how many bits of edit description does each repair need to
  close a given share of the gap? This puts undo, which needs a logged gradient, restore, which needs v̄ on S, and the
  oracle patch on one axis.
- **Mechanism.** Description length = log₂C(n, |S|) for the support, plus b bits × |S| for quantized magnitudes. Plot it
  against G and find the R(D) frontier. **[inferred]**
- **Minimal experiment.** Re-score #27's edit-size Pareto forks with the bit cost, then quantize repair magnitudes to b ∈
  {1, 2, 4, 8, 32} bits. That costs 5 forks per repair per instance.
- **Closest prior work.**
  - MDL as an interpretability criterion, as stated in *Structuring Sparsity* ([2606.25234](https://arxiv.org/abs/2606.25234))
    [verified-snippet]. The work it cites (Ayonrinde et al. 2024) is UNVERIFIED.
  - Internal overlap: #27 / C10, the edit-size Pareto.
- **Kill-search.** "minimum description length rate distortion model editing sparse edit bits needed to repair neural
  network". **PARTIAL.** The bit framing is new here, but #27 already covers most of the content.
- **How it plugs in.** An appendix re-plot of #27. The simplicity filter says it should not be a contribution.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | close to #27 |
| F | 5 | no new forks beyond the b-sweep |
| I | 2 | a re-plot |
| R | 5 | low risk |
| V | 4 | straightforward |
| D | 3 | the bit accounting needs justifying |

### R3 · Forensic reconstruction of the offending batch from Δlog v of wte rows
- **Pitch.** The spike batch leaves a token-level fingerprint in the rows of wte's v. Rows of tokens that were present
  get g² that is orders of magnitude larger than their usual tiny v, so two snapshots can recover the bag of offending
  tokens with no data logs. That gives natural spikes a data-identified, aligned B\* (swap only those documents) and
  feeds a data-filtering loop (R13).
- **Mechanism [inferred].**
  - wte is tied to lm_head. The input-side gradient touches only rows of tokens present in the batch.
  - The output-side gradient touches every row ∝ softmax p_j, with a (p−1) term on target rows.
  - So present/target rows separate from the rest in Δlog v = log v_t − log β2·v_{t−1}, most sharply for rare tokens,
    whose baseline v is small.
  - Rank rows by Δlog v and take the top-|set| rows as the reconstruction.
- **Minimal experiment.**
  - On planted `corrupt_batch` and `repeat_ngram` (where truth is the exact token set), report precision and recall of
    the top-k rows. Planting costs 0 GPU and scoring runs on CPU.
  - Then check whether the top rows overlap the oracle damage rows from #6.
  - Prediction [inferred]: recall ≥ 0.9 for `corrupt_batch` (uniform random, mostly rare tokens), and lower for
    `repeat_ngram` with common tokens.
- **Closest prior work.**
  - *MineGrad* ([2608.01521](https://arxiv.org/abs/2608.01521)) [verified-snippet]. A gradient-inversion attack through
    Adam's first-step, sign-like update in LoRA federated fine-tuning.
  - Biswas 2026, *Recent Training is Protected by Adam's Second Moment*, NeurIPS 2026 OPT workshop
    ([page](https://neurips.cc/virtual/2026/161497)) [verified]. The row-level v-excess is "readable from two
    optimiser-state snapshots", but not for data identification.
  - Internal: A17 (TracIn-style data attribution, which needs per-example gradients).
- **Kill-search.** "gradient inversion from optimizer state Adam moments reconstruct training tokens embedding rows";
  "membership inference or data reconstruction from Adam optimizer state moments exp_avg_sq leakage". **NEW** for spike
  forensics. Inversion from *updates* exists in the privacy literature.
- **How it plugs in.**
  - It is a zero-fork, data-side localizer that ties the C2 damage map to its data cause.
  - It closes the natural-spike B\* gap (A4) without per-example gradients.
  - Side note: optimizer-state checkpoints leak the tokens of recent batches. That is a privacy remark for the
    SpikeState release (#14).
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | not found for spikes |
| F | 4 | CPU on existing snapshots |
| I | 4 | links state to data, a reviewer-pleasing figure |
| R | 3 | the output-side gradient may blur rows for common tokens |
| V | 5 | planted truth is exact |
| D | 4 | the mechanism is simple and checkable |

### R4 · Sparse autoencoder over optimizer-state rows ("what does v know?")
- **Pitch.** Treat optimizer state as a representation. Train a sparse dictionary over per-row state trajectories
  ([log v, m/√v] over a window) from many normal and spiked snapshots, and look for a "poison feature" whose ablation is
  a repair.
- **Mechanism.** SAE features over state rows. Validate causally by editing along a feature and forking. **[inferred]**
- **Minimal experiment.** About 50 normal plus 50 spiked proxy snapshots × wte rows. Training the SAE on CPU or GPU takes
  minutes. Score the features on the decoy benchmark (#3) against the A1 magnitude control.
- **Closest prior work.**
  - No SAE on optimizer state found.
  - SAE method: [2309.08600](https://arxiv.org/abs/2309.08600) [verified-snippet].
  - *The Hidden Ratio in Adam* (NeurIPS 2026 poster, [page](https://neurips.cc/virtual/2026/poster/154051)) [verified]
    finds the transformed m/v ratio scale-stable and heavy-tailed, so it is a natural input feature.
- **Kill-search.** "sparse autoencoder on optimizer state Adam second moment interpretability features"; "probing what
  Adam optimizer second moment encodes…". **NEW.**
- **How it plugs in.** A fourth, unsupervised localizer variant, appendix only. The main risk is that the features just
  rediscover magnitude, in which case A1 wins.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 5 | no hit |
| F | 3 | about 20 p-h of new code |
| I | 3 | |
| R | 2 | interpretability-illusion risk |
| V | 3 | needs causal follow-up |
| D | 2 | reviewers will ask "why not the raw magnitude?" |

### R5 · Frequency-calibrated reference v for embedding rows
- **Pitch.** On healthy steps, v of a wte row is close to a smooth function of that token's frequency and the gradient
  scale. Fit that law on matched-normal steps and use its prediction as the restoration target. No pre-spike v̄ is needed,
  so it also works for natural spikes, contaminated v̄ and released checkpoints with no history.
- **Mechanism.** log v_row ≈ a + b·log count_row (fitted). The poison score is the residual, and the restoration target is
  the fitted value. **[inferred]**
- **Minimal experiment.** Fit on 10 matched-normal proxy snapshots using token counts from the aligned stream (`get_batch`
  is a pure function of step). Compare R5 vs v̄ as target T in #12 on `corrupt_batch`: 2 forks per instance.
- **Closest prior work.**
  - *Better Embeddings with Coupled Adam* ([2502.08441](https://arxiv.org/abs/2502.08441)) [verified-snippet]. Per-token
    second moments differ across the vocabulary and cause anisotropy; the fix shares a vocabulary-average v.
  - Ember / *Token Geometry* ([2607.01455](https://arxiv.org/abs/2607.01455)) [verified-snippet]. An optimizer for
    embedding/LM-head matrices.
- **Kill-search.** "Adam second moment token frequency embedding rows rare tokens Coupled Adam anisotropy". **PARTIAL.**
  The v–frequency link is known; using it as a poison reference is not.
- **How it plugs in.** A history-free target T for certified restoration (#12) on row support. It is also a robustness
  check that the repair does not depend on v̄.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 5 | CPU fit |
| I | 3 | |
| R | 4 | the law may be loose for frequent tokens |
| V | 4 | |
| D | 4 | removes the "v̄ is contaminated" objection |

### R6 · Per-token functional damage map
- **Pitch.** A scalar held-out Δ hides *who* was hurt. Stratify the B0 − B\* per-token loss by token frequency and by
  whether the token appeared in the spike batch, and check that each repair restores the right tokens.
- **Mechanism.** Δ_token(f-bucket, in-batch?) on the fixed eval set, compared across branches. **[inferred]**
- **Minimal experiment.** Zero new forks: re-score #1's and #8's branches with per-token losses (the eval already
  computes them).
  - Prediction [inferred]: damage concentrates on rare and in-batch tokens for `corrupt_batch`, and is diffuse for
    `lr_bump`.
- **Closest prior work.** Rare tokens have higher and more sensitive loss in general (Rho-1,
  [2404.07965](https://arxiv.org/abs/2404.07965); *The Fair Language Model Paradox*,
  [2410.11985](https://arxiv.org/abs/2410.11985)) [verified-snippet]. Neither looks at spikes.
- **Kill-search.** "loss spike damage concentrated on rare tokens per-token loss frequency bucket after instability
  language model". **NEW** as a spike-damage endpoint.
- **How it plugs in.** A construct-validity endpoint for G. A repair that closes G by fixing the wrong tokens gets
  flagged. It also pairs with R3 and R5 (row-level story).
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 5 | free |
| I | 3 | |
| R | 5 | |
| V | 5 | |
| D | 4 | |

### R7 · Self-healing dual-timescale v
- **Pitch.** Keep a cheap fast "shadow" v (β2f = 0.5–0.8) next to the slow v. Per coordinate, the ratio v_slow/v_fast is
  a signed online poison score: ≫ 1 means inflated (stale after a spike), ≪ 1 means decayed (Bai's precursor). Blend
  toward the shadow when the ratio leaves a calibrated band.
- **Mechanism.** One extra state tensor plus a ratio test, so the optimizer heals itself online. **[inferred]**
- **Minimal experiment.** Implement as a branch optimizer on the proxy, on the stress presets and induced spikes. Compare
  with #21's airbag, Kourkoutas-β and SPAM on G and false-alarm rate. About 30 forks.
- **Closest prior work.**
  - AdEMAMix: two EMAs for m, not v ([2409.03137](https://arxiv.org/abs/2409.03137)) [verified-snippet].
  - Kourkoutas-β: β2 adapted per layer from a bounded spike signal ([2508.12996](https://arxiv.org/abs/2508.12996))
    [verified-snippet].
  - Adafactor discusses fast vs slow β2 as alternatives ([1804.04235](https://arxiv.org/abs/1804.04235))
    [verified-snippet].
- **Kill-search.** "dual timescale second moment estimator Adam fast slow EMA disagreement detect instability
  self-correcting optimizer"; "Adam shadow second moment estimate consistency check robust v median of EMAs…".
  **PARTIAL.** No fast/slow-v consistency test was found, but Kourkoutas-β is close in spirit.
- **How it plugs in.** The online form of the signed localizer (#11's NIS lower and upper tails) inside the BACKUP-2
  airbag. It is a deployable add-on, not the headline.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | Kourkoutas-β |
| F | 4 | |
| I | 4 | practitioners would use it |
| R | 3 | the band needs calibration |
| V | 4 | |
| D | 3 | |

### R8 · Meta-learned repair operator
- **Pitch.** Learn the repair. A tiny per-coordinate MLP maps (g², v, v̄, m, NIS, lag) to (m′, v′) and is meta-trained on
  planted and induced spikes, using fork-measured Δ or an unrolled k-step validation loss.
- **Mechanism.** A learned-optimizer-style network applied once at repair time. **[inferred]**
- **Closest prior work.** VeLO ([2211.09760](https://arxiv.org/abs/2211.09760)) [verified-snippet] is a general learned
  optimizer with documented generalization limits. Nothing was found on spike repair.
- **Kill-search.** "learned optimizer meta-learned robust to loss spikes training instability recovery VeLO". **PARTIAL.**
- **How it plugs in.** Reject for the main paper. It is circular in the same way as A15 (planted → natural transfer is
  the open question), and it is expensive. It could be a "future work" line.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 2 | meta-training loop |
| I | 3 | |
| R | 2 | circularity |
| V | 3 | |
| D | 2 | |

### R9 · Optimizer-state *format* as a poison amplifier or suppressor
- **Pitch.** Quantized and factored states share statistics across coordinates, so a spike's poison should spread along
  the format's sharing structure:
  - **8-bit blockwise:** a spiked coordinate raises the block's absmax, so the block's other v entries are quantized
    more coarsely (collateral poison).
  - **Adafactor:** one row's inflation enters a row factor and a column factor, so it smears across the matrix.
  - **4-bit "hidden ratio" codebook state.**

  The prediction is that the localization granularity is dictated by the state format.
- **Mechanism.** Re-run identical induced spikes under fp32, blockwise-8-bit, Adafactor and ratio-4-bit states. Measure
  the support of |log v⁰/v\*| and G. **[inferred]**
- **Minimal experiment.**
  - Pure-torch blockwise quantize/dequantize of v at each step, which avoids a bitsandbytes dependency. The block size
    is a parameter. Dettmers' default block size is UNVERIFIED here.
  - 3 formats × 3 spike types × 4 seeds × ~6 forks ≈ 220 forks ≈ 30 min proxy [assumed].
  - A determinism gate is needed per format (new kernels). `determinism.py` is not touched.
  - Prediction [inferred]: for 8-bit, the collateral support equals the spiked block. The undo log loses exactness under
    quantization rounding.
- **Closest prior work.**
  - Dettmers et al., *8-bit Optimizers via Block-wise Quantization* ([2110.02861](https://arxiv.org/abs/2110.02861))
    [verified-snippet]. Blockwise quantization limits the effect of outliers to one block, which is exactly the
    predicted collateral unit.
  - [2407.11722](https://arxiv.org/abs/2407.11722) [verified-snippet]: v quantized with a linear symmetric quantizer
    failed to converge smoothly.
  - Topollai & Choromanska, *Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State
    Staleness and Effectiveness of State Resets* ([2603.16731](https://arxiv.org/abs/2603.16731), v1 2026-03-17)
    [verified]. Quantized EMAs go stale and resets restore responsiveness. No spikes.
  - *Hidden Ratio* (NeurIPS 2026, [page](https://neurips.cc/virtual/2026/poster/154051)) [verified]: a 4-bit codebook
    state.
- **Kill-search.** "8-bit Adam block-wise quantization optimizer state loss spikes instability outlier absmax"; "Adafactor
  factored second moment loss spike row column factor outlier spreads instability". **PARTIAL.** Staleness under
  quantization is known; spike-damage support under quantized or factored states was not found.
- **How it plugs in.** A generalization ablation: does "where damage lives" depend on the state format? It is directly
  relevant to production 8-bit and Adafactor users, and it stress-tests the undo log's exactness.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 3 | new optimizer variants plus gates |
| I | 4 | production relevance |
| R | 3 | the determinism of the new formats is unknown |
| V | 4 | |
| D | 4 | |

### R10 · Kronecker-factored preconditioners: rank-limited repair becomes well-posed
- **Pitch.** For diagonal Adam, "rank-limited repair of v" is ill-posed (`04` §4). For SOAP/Shampoo the poison lives in the
  L and R factor matrices, so a rank-k edit of their top eigendirections is well-posed. That rescues the original
  "rank-limited" wording for a second optimizer family.
- **Mechanism.** Localize poisoned eigendirections of L and R vs B\*, edit them, and fork. **[inferred]**
- **Closest prior work.**
  - *SOAP, Muon, and Beyond* ([2607.20548](https://arxiv.org/abs/2607.20548)) [verified-snippet]. A stale SOAP
    preconditioner causes oscillatory spikes, fixed by a real-time eigenbasis that includes the current gradient.
  - *Purifying Shampoo* (NeurIPS 2025, [2506.03595](https://arxiv.org/abs/2506.03595)) [verified-snippet].
- **Kill-search.** "Shampoo SOAP preconditioner instability stale Kronecker factors loss spike repair". **PARTIAL.** The
  cause is known; post-hoc factor repair is not.
- **How it plugs in.** An extension or appendix only. It is costly (new optimizer, eigensolves, its own determinism gate).
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 2 | |
| I | 3 | |
| R | 2 | |
| V | 3 | |
| D | 3 | |

### R11 · AdEMAMix slow momentum as an m-channel instrument
- **Pitch.** `09` §1.4 uses β2 as an instrument for v. AdEMAMix's slow EMA (β3 = 0.9999) is the matching instrument for
  m. It predicts that the m-share of spike damage becomes large and long-lived, with a half-life of ln2/(−ln 0.9999) ≈
  6,931 steps [inferred], and that undoing m_slow becomes the dominant repair.
- **Minimal experiment.**
  - One AdEMAMix proxy trunk (β3 warmup as the paper requires) with `corrupt_batch` × 4 seeds.
  - The 2³ factorial is extended to (w, m_fast, m_slow, v), so 16 cells; or 4 cells with m split. About 2–3 GPU-h at the
    proxy [assumed].
- **Closest prior work.** AdEMAMix ([2409.03137](https://arxiv.org/abs/2409.03137)) [verified-snippet]: early divergence
  with large β3, fixed by α/β3 warmup. Apertus ([2509.14233](https://arxiv.org/abs/2509.14233)) [verified-snippet]: warmup
  of both is required. Neither covers spike-state shares.
- **Kill-search.** "AdEMAMix slow momentum loss spikes instability early training beta3 warmup". **NEW** for the
  state-share question.
- **How it plugs in.** A sharp, β-determined prediction for the migration curve's m channel. A matched slope is strong
  mechanistic evidence (as in `09` §1.4).
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 4 | |
| I | 3 | |
| R | 4 | |
| V | 4 | |
| D | 4 | the quantitative prediction is hard to fake |

### R12 · Spike phase × cooldown as a natural repair
- **Pitch.** Inject the *same* spike recipe at five LR phases (warmup, peak, mid-decay, late decay, final cooldown) in
  cosine and WSD trunks. Measure the end-of-schedule Δ and the state share φ. This answers when spikes matter most and
  whether the decay phase erases state-carried poison for free.
- **Mechanism [inferred].** The post-spike state-driven drift on dominated coordinates is ≈ 5.84·η (`09` §2.3), so it is
  proportional to the LR at the spike. A cooldown shrinks η, so it caps the state channel but leaves less time to heal
  weight damage.
- **Predictions [inferred].**
  - φ(s+10) increases with η at the spike.
  - The end Δ is largest for spikes injected late, during the cooldown, because there is no recovery time left.
  - In WSD, a spike in the stable phase followed by the decay phase has a smaller end Δ than the same spike in a cosine
    run at equal η (the river-valley view).
- **Minimal experiment.** 5 phases × 2 schedules × 4 seeds × 4 cells (B0, B\*, state-only, weights-only) = 160 forks ≈
  20 min proxy [assumed].
- **Closest prior work.**
  - *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape View* (ICLR 2025,
    [poster](https://www.iclr.cc/virtual/2025/poster/28484)) [verified-snippet].
  - *Critical Learning Periods in Deep Networks* (Achille, Rovere, Soatto; [1711.08856](https://arxiv.org/abs/1711.08856),
    ICLR 2019) [verified-snippet]. Early deficits cause permanent damage.
  - K2-V2 ([2512.06201](https://arxiv.org/abs/2512.06201)) [verified-snippet]. Most natural spikes fall in the first 40%
    of training, which is observational.
  - Ling ([2503.05139](https://arxiv.org/abs/2503.05139)) [verified-snippet]: narrow vs wide spikes.
- **Kill-search.** "effect of loss spike timing warmup versus learning rate decay phase final loss LLM pretraining injected
  perturbation"; "injecting spike at different training steps early vs late controlled perturbation final loss
  sensitivity critical period language model"; "river valley loss landscape cooldown WSD erases perturbation recovery
  from loss spike" (no ML hits). **NEW.** No controlled injected-spike timing × state-share study was found.
- **How it plugs in.** It turns the migration curve into a function of the LR phase. It tells practitioners when repair
  is worth doing and adds a schedule-robustness axis that reviewers will ask for.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 4 | needs a WSD trunk |
| I | 4 | actionable |
| R | 4 | any outcome is informative |
| V | 4 | |
| D | 4 | |

### R13 · Localizer-in-the-loop data filter
- **Pitch.** Close the loop from state to data:
  1. damage map
  2. the token rows it names (R3)
  3. documents containing those tokens
  4. down-weight or mask similar documents in the upcoming stream

  Then measure how often spikes recur.
- **Mechanism.** A deploy-side filter. **It is never used to build B\*.** B\* must keep the aligned data stream (CLAUDE.md
  §1). **[inferred]**
- **Closest prior work.** OLMo 2 ([2501.00656](https://arxiv.org/abs/2501.00656)) [verified-snippet]. A static
  repeated-n-gram filter plus loss masking reduced spikes but did not eliminate them. The same sequence can spike in one
  run and not in another.
- **Kill-search.** "automatically filter offending training data after loss spike pretraining data removal feedback loop".
  **PARTIAL.**
- **How it plugs in.** A "use case" paragraph. It needs a natural-spike population to evaluate, which is the weak point.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 3 | |
| I | 4 | |
| R | 3 | |
| V | 3 | |
| D | 3 | |

### R14 · Micro-batch-resolved undo
- **Pitch.** Refine #5. Instead of undoing a whole step's contribution, undo only the offending micro-batch's. The result
  is a data-aligned "replace just the bad micro-batch" counterfactual for natural spikes.
- **Mechanism [inferred].**
  - From the `grad_hook` micro-gradients, recompute the step gradient without micro-batch k (or with it replaced by a
    held-out clean micro-batch). Re-apply the global clip to that gradient, giving g′_s.
  - Subtract (1−β1)β1^{t−s}(g_s − g′_s) from m and (1−β2)β2^{t−s}(g_s² − g′_s²) from v.
  - This is exact for the direct path (`09` §3). It must be the post-clip g′, and g² is not additive over micro-batches,
    so g′_s has to be recomputed rather than subtracting micro-batch g_k².
- **Cost [inferred].** At 124M, grad_accum = 40, each micro-gradient is ≈ 0.5 GB fp32, so ≈ 20 GB per logged step. That
  fits on 192 GB for a 1–2-step window.
- **Closest prior work.**
  - *Unlearning at Scale: State-Exact Trace-Preserving Deletion in Billion-Parameter Language Models*
    ([2508.12220](https://arxiv.org/abs/2508.12220)) [verified-snippet]. Exact deletion by *replay* with per-micro-batch
    logs, plus micro-checkpoint reverts. It treats the optimizer state as part of the exactness target.
  - Stewart, *Form and Function* ([2605.17590](https://arxiv.org/abs/2605.17590)) [verified-snippet].
- **Kill-search.** "per micro-batch gradient removal from Adam moments retroactive unlearning of a single batch exact
  optimizer state"; "Unlearning at Scale state-exact…". **PARTIAL.** Replay-based exact deletion exists; direct-path
  subtraction at micro-batch resolution for spike repair was not found.
- **How it plugs in.** It is a C3 variant inside the undo family, and it partly answers the natural-spike B\* question.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 4 | |
| I | 4 | |
| R | 3 | off-manifold if g′ ≠ a realizable gradient |
| V | 4 | |
| D | 4 | |

### R15 · Unsupervised spike taxonomy from damage fingerprints
- **Pitch.** Cluster spike instances by fingerprint: (φ_state, e_m vs e_v, sign, support size, layer profile, decay rate).
  Check the clusters against known causes, then assign natural spikes to them.
- **Closest prior work.**
  - Internal T8 fault-signature matrix (residual-based, supervised).
  - K2-V2 ([2512.06201](https://arxiv.org/abs/2512.06201)) [verified-snippet]: transient vs fatal spikes.
  - *Natural Fingerprints* ([2504.14871](https://arxiv.org/abs/2504.14871)) [verified-snippet] uses "fingerprint" in an
    unrelated sense.
- **Kill-search.** "taxonomy classification of loss spike causes LLM training clustering fingerprints diagnosis".
  **PARTIAL** (T8 overlaps).
- **How it plugs in.** An analysis section once there are ≥ 40 instances. Correlational.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 4 | |
| I | 3 | |
| R | 3 | |
| V | 3 | |
| D | 3 | |

### R16 · Universal poison sites across seeds and rungs
- **Pitch.** If the oracle sets from #6 always fall in the same module types across seeds and across proxy, mid and 124M
  (for example wte rows of in-batch tokens, layer-0 attention, ln), a fixed-prior localizer is free and 124M needs fewer
  forks.
- **Minimal experiment.** Compare oracle-set module types across 4 seeds × 2 rungs. Report Jaccard overlap and the
  transfer AUROC of a "module-prior" localizer.
- **Closest prior work.**
  - SPAM ([2501.06842](https://arxiv.org/abs/2501.06842)) [verified-snippet]: embedding and LayerNorm have the most
    gradient spikes at LLaMA-60M.
  - WeSaR ([2410.05052](https://arxiv.org/abs/2410.05052)) [verified-snippet]: W_d and W_o at 13B, with no spikes at
    1.3B / 130M.
  - Bai ([2506.04805](https://arxiv.org/abs/2506.04805)) [verified-snippet]: v of embedding and W_V decays before spikes.
  - Wortsman, *Small-scale proxies* ([2309.14322](https://arxiv.org/abs/2309.14322), ICLR 2024) [verified-snippet].

  None of these reports a per-seed or per-size consistency analysis.
- **Kill-search.** "which layers cause loss spikes consistently embedding layer norm across seeds model sizes transformer".
  **PARTIAL.**
- **How it plugs in.** A scale-transfer test for C2, plus a cheaper 124M oracle via an informed split order for #6.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 4 | |
| I | 4 | |
| R | 4 | |
| V | 4 | |
| D | 4 | |

### R17 · Scaling law of the state share φ_state(N)
- **Pitch.** Fit φ_state against model size across the three rungs and extrapolate to 1B.
- **Closest prior work.** Wortsman 2309.14322 [verified-snippet]: instabilities are predicted from how norms scale with
  model size.
- **Kill-search.** "loss spike recovery cost scaling with model size larger models recover faster from spikes empirical
  study" (no relevant ML hit except the K2-Spike-2 model card, [HF](https://huggingface.co/IFM/K2-Spike-2)
  [verified-snippet]). **PARTIAL.**
- **How it plugs in.** At most a trend figure. Three points do not make a law, and reviewers will say so.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 3 | |
| I | 4 | |
| R | 2 | 3 points |
| V | 3 | |
| D | 2 | |

### R18 · Optimizer hot-swap on the localized set
- **Pitch.** A target-free repair: for k steps after the spike, update only S with sign-momentum (Signum/Lion style, no v)
  at an RMS-matched learning rate, and leave everything else on AdamW. This bypasses the poisoned v without having to
  decide what v *should* be.
- **Mechanism [inferred].**
  - Set lr_S = η·RMS(Adam update on S, pre-spike), so the step size is continuous.
  - Return to AdamW after k = 1/(1−β2) steps, by which time v has re-equilibrated.
  - It is implemented as a parameter group in *all* branches, so the param-group layout stays identical (as in B11).
- **Minimal experiment.** 3 spike types × 4 seeds × k ∈ {10, 20, 50} = 36 forks.
- **Closest prior work.**
  - AOS ([2608.01997](https://arxiv.org/abs/2608.01997)) [verified-snippet]: adaptive switching among AdamW, SGD-M and
    Lion (CIFAR).
  - *Hidden Ratio* (NeurIPS 2026, [page](https://neurips.cc/virtual/2026/poster/154051)) [verified]: Adam reduces to sign
    momentum modulated by a ratio, plus an LR-transfer rule. That rule supplies the RMS-matching.
  - SWATS is cited as Adam → SGD switching in a snippet of 2512.01881. Its own ID is UNVERIFIED here.
- **Kill-search.** "switch optimizer mid-training after instability Adam to sign descent Lion hot swap recovery
  divergence". **NEW** for spike repair. Switching itself is PARTIAL.
- **How it plugs in.** A C3 arm that asks whether v's information is needed at all after a spike. If hot-swap ≈ certified
  restoration, the repair story simplifies (simplicity filter).
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 4 | |
| I | 3 | |
| R | 3 | the switch transient |
| V | 4 | |
| D | 3 | |

### R19 · Beneficial spikes: is any benefit state-carried?
- **Pitch.** Some spikes may *help* (Δ_end < 0). Inflated v acts as per-coordinate LR decay, a form of implicit annealing.
  Pre-register a two-sided test, and decompose any benefit by channel with the factorial.
- **Mechanism.** Count Δ < 0 instances. If there are any, run #1's factorial on them: is e_v < 0? **[inferred]**
- **Closest prior work.**
  - *Catapults in SGD* ([2306.04815](https://arxiv.org/abs/2306.04815)) [verified-snippet]: spikes improve generalization
    through feature learning.
  - *Grokking or Glitching* (ICML 2026, [2605.06152](https://arxiv.org/abs/2605.06152)) [verified-snippet]: slingshot
    spikes from low precision.
  - Bai ([2506.04805](https://arxiv.org/abs/2506.04805)) [verified-snippet]: spike types differ in their effect on
    generalization.
- **Kill-search.** "beneficial loss spikes catapult slingshot mechanism improve generalization inflated second moment
  implicit learning rate decay". **PARTIAL.**
- **How it plugs in.** A "do no harm" endpoint. A repair must not erase a beneficial spike's benefit. It also protects
  the paper from the "spikes are sometimes good" reviewer.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 5 | free on existing forks |
| I | 3 | |
| R | 4 | |
| V | 4 | |
| D | 4 | |

### R20 · Protection law: the spike's write is protected by the v-excess it created
- **Pitch.**
  - Biswas 2026 shows that a training write raises v on its rows, and that this excess *protects* the write from being
    overwritten for a β2-set time. Clearing the excess on those rows removes the protection, and it works as well as a
    full state reset and better than lowering β2.
  - Applied to spikes: the spike is a large write that installs its own protection. That is a *mechanism* for why
    damage persists in v, and for why selective v restoration (#12) works. Restoration removes the protection, so clean
    gradients can overwrite the spike's damage.
- **Mechanism [inferred].** Map Biswas's "first lesson" to the spike's write on S, and the "second lesson" to the
  subsequent clean gradients that would undo it. His integrated-write-ratio law should then predict the self-recovery
  time of each instance. The law exists [verified, abstract]; its exact functional form is UNVERIFIED (full text not
  read).
- **Two new predictions [inferred].**
  1. Across instances, self-recovery time ranks with the integrated-write ratio on S (Spearman ρ ≥ 0.6, pre-registered).
  2. **Collateral forgetting:** restoring v on S also strips the protection of *healthy* recent writes on S. So a repair
     can cause forgetting of recently learned data. Measure the loss on a probe set of the aligned batches from steps
     s−50…s−1. This is evaluation only; no training on them, so data alignment is preserved. It is a new safety endpoint
     for C3.
- **Minimal experiment.**
  - Compute integrated writes on S from B0's logged updates. This is CPU work on telemetry and the ring buffer.
  - Compare predicted with measured recovery time over ≥ 12 instances.
  - Add the recent-batch probe loss to every #8 / #12 branch. 0 extra forks.
- **Closest prior work.**
  - Biswas, *Recent Training is Protected by Adam's Second Moment*, NeurIPS 2026 OPT workshop
    ([page](https://neurips.cc/virtual/2026/161497)) [verified]. A toy two-lesson task. The protection decays within ≈ 8%
    of −1/ln β2. No spikes.
  - It is already cited internally in `02_method_catalogue.md`, but only as "row-level v-excess editing ≈ full reset",
    not as a mechanism for spike persistence.
- **Kill-search.** "Biswas NeurIPS 2026 OPT workshop row-level second moment excess protection Adam"; "probing what Adam
  optimizer second moment encodes representation of training history interpret optimizer state". **PARTIAL.** The
  mechanism is shown on a toy task; the spike-persistence application and the collateral-forgetting endpoint were not
  found.
- **How it plugs in.**
  - It gives the PRIMARY a *mechanistic theory* that ties three pieces together: the migration curve's v share, the
    undo log (which removes the write itself), and certified restoration (which removes the protection).
  - It adds one cheap, falsifiable prediction and one safety endpoint.
  - This is the best "contribution" candidate from wave 2.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | a new application with new predictions |
| F | 5 | CPU plus existing forks |
| I | 4 | a mechanism, not just a measurement |
| R | 4 | if the law fails it is still a reportable contrast |
| V | 4 | |
| D | 4 | it builds on a peer-reviewed workshop result instead of competing with it |

### R21 · Adversarial optimizer-state poisoning (worst-case minimal state edit)
- **Pitch.** Find the smallest edit to (m, log v) under an ℓ0 or ℓ2 budget that maximizes the end-of-schedule Δ. Use the
  AtP-state gradient (#11) as the search direction and validate with forks. The result is a vulnerability map and the
  hardest planted benchmark, one that magnitude-based localizers cannot win by construction.
- **Closest prior work.**
  - *Towards TEE-Certified DP* ([2609.20532](https://arxiv.org/abs/2609.20532)) [verified-snippet]: a forged-gradient
    adversary against AdamW's one-step map.
  - Bit-flip attacks on *weights* (FAT-RABBIT, [record](https://mdsoar.org/items/408c10fe-9948-4f64-9caa-bb94029af39a))
    [verified-snippet].
- **Kill-search.** "adversarial attack on optimizer state Adam moments minimal perturbation cause divergence
  training-time". **NEW.**
- **How it plugs in.** A benchmark plant for #3, plus a repair robustness check: does certified restoration still close
  G on worst-case plants?
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 4 | |
| I | 3 | |
| R | 3 | AtP may be a weak attack direction |
| V | 4 | |
| D | 3 | |

### R22 · Bit-flip vulnerability map of optimizer state: m flips spike, v flips go silent
- **Pitch.** SDC studies inject faults into activations, gradients and weights, but not into Adam's moments. This
  matters because the two moments fail in opposite ways:
  - **m:** an exponent flip makes m huge, so the update is huge and there is an immediate loss spike (weight damage).
  - **v:** an exponent flip toward a large value freezes the coordinate for ≈ ln κ / (−ln β2) steps, with **no loss
    spike at all**. That is a silent, state-only poison that no loss or grad-norm detector sees. For κ = 2¹²⁸ at
    β2 = 0.95 it is ≈ 88.7 / 0.0513 ≈ 1,730 steps [inferred arithmetic].
- **Mechanism.** Flip bits 23–30 of fp32 m, v and w at random coordinates on snapshots, then fork and measure the
  lasting Δ. The result is an AVF-style table by tensor × bit × channel. **[inferred]**
- **Minimal experiment.**
  - 200 flips × {m, v, w} into proxy snapshots (0 GPU to plant), 1 fork each.
  - ≈ 600 forks ≈ 75 min proxy [assumed]. Can be subsampled.
  - Pre-registered [inferred]:
    - v flips have loss-detector recall ≈ 0;
    - the #15 parity residual catches 100% of them;
    - undo (#5) is a no-op, because no gradient caused the fault;
    - certified restoration to v̄ fully repairs.
- **Closest prior work.**
  - *Understanding Silent Data Corruption in LLM Training* ([2502.12340](https://arxiv.org/abs/2502.12340))
    [verified-snippet]: submodule, optimizer-step and whole-run levels.
  - LLM-PRISM ([2604.10390](https://arxiv.org/abs/2604.10390)) [verified-snippet]: injects into activations, input
    gradients and weight gradients, **not the optimizer state**.
  - Altenbernd et al. ([2604.00726](https://arxiv.org/abs/2604.00726)) [verified-snippet]: sensitivity by bit position,
    kernel and stage.
  - ORNL checkpoint alteration ([highlight](https://ornl.gov/research-highlight/understanding-soft-error-sensitivity-deep-learning-models-and-frameworks-through))
    [verified-snippet].
- **Kill-search.** "bit flip fault injection optimizer state Adam training vulnerability hardware failures deep learning
  training"; "silent data corruption optimizer state LLM training sensitivity which tensors exponent bit flip Adam
  moments". **NEW** for the moments specifically, as far as these searches go.
- **How it plugs in.**
  - A *natural* (hardware) source of pure state poison, which answers "are planted plants realistic?".
  - It shows the undo log and certified restoration are complementary: undo handles gradient-caused poison, restoration
    handles state-caused poison.
  - Under ZeRO-1, a faulty rank corrupts only its own shard of m/v, so the shard map is a free prior [inferred].
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 5 | plants are free; forks are cheap |
| I | 3 | |
| R | 4 | |
| V | 5 | exact truth |
| D | 4 | ties to an active SDC literature |

### R23 · Poison contagion: does the poisoned support grow or shrink?
- **Pitch.** After the spike, compute the "infected set" {i : |log v⁰_i/v\*_i| > τ} and its per-tensor arrival order over
  time from forks that already exist. If the set grows (poison spreads through the weights into other coordinates' g²),
  early repair needs a smaller support. If it shrinks, waiting is fine.
- **Mechanism.** Support size S(t), the reproduction ratio of new infections per step, and the spread order across
  tensors. **[inferred]**
- **Minimal experiment.** 0 new forks: dense snapshots along B0 and B\* in #1's forks. Prediction [inferred]: S(t) peaks
  at s+5…s+20 at β2 = 0.95, and the peak timing shifts with β2.
- **Closest prior work.**
  - *The Butterfly Effect* (ICML 2025, [2506.13234](https://arxiv.org/abs/2506.13234)) [verified-snippet]: early-phase
    perturbations diverge, then fade.
  - [2404.05782](https://arxiv.org/abs/2404.05782) [verified-snippet]: non-monotone trajectory divergence.

  Both are whole-trajectory, weight-space measures, with no per-coordinate state contagion.
- **Kill-search.** "spread of perturbation across parameters over training steps divergence of nearby trajectories
  butterfly effect neural network training chaos". **PARTIAL.**
- **How it plugs in.** A free figure that explains *why* the undo benefit decays with lag (`09` §2.3(c)), and that sets
  the support budget of a repair at each lag.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 5 | |
| I | 3 | |
| R | 4 | |
| V | 4 | |
| D | 4 | |

### R24 · DiLoCo cross-worker donor repair
- **Pitch.** In DiLoCo / local SGD each worker runs its own inner AdamW. A spike on one worker's data shard poisons only
  that worker's (m, v), so the other workers' states are a natural, contemporaneous donor for repair: a population v̄.
  That is a real deployment setting where a counterfactual reference exists by construction.
- **Minimal experiment.** Simulate 4 workers at proxy scale on 1 GPU, induce a spike on one worker, and compare donor
  repair with self-v̄ restore and with dropping that worker's pseudo-gradient.
- **Closest prior work.**
  - DiLoCo ([2311.08105](https://arxiv.org/abs/2311.08105)) [verified-snippet]: robust to island failure, i.e.
    availability, not bad updates.
  - HeLoCo ([2606.00271](https://www.alphaxiv.org/abs/2606.00271)) [verified-snippet]: corrects stale pseudo-gradients.
- **Kill-search.** "DiLoCo worker loss spike inner optimizer state outer averaging robustness bad worker" (no hits);
  "DiLoCo local SGD heterogeneous worker failure loss spike robustness outer optimizer pseudo-gradient outlier".
  **NEW.**
- **How it plugs in.** An "applications" extension. It is out of the main 8 pages unless the team targets distributed
  training.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 3 | new training mode |
| I | 3 | |
| R | 3 | |
| V | 3 | |
| D | 3 | |

### R25 · MoE expert-state poisoning
- **Pitch.** Rarely routed experts have stale v, so spikes should localize to experts and their router rows.
- **Closest prior work.**
  - MESH ([2608.04407](https://www.alphaxiv.org/abs/2608.04407)) [verified-snippet]: routed experts are the main failure
    point for stateless normalization.
  - *Excitation* ([2602.21798](https://arxiv.org/abs/2602.21798)) [verified-snippet].
  - Router monitors ([2606.28116](https://arxiv.org/abs/2606.28116)) [verified-snippet].
- **Kill-search.** "mixture of experts loss spike expert stale Adam second moment rarely routed experts instability".
  **PARTIAL.**
- **How it plugs in.** Future work. A proxy MoE would need its own trunk and determinism gate (routing ties).
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 2 | |
| I | 3 | |
| R | 2 | |
| V | 3 | |
| D | 3 | |

### R26 · Path patching for optimizer state (cross-tensor influence graph)
- **Pitch.** Patch only tensor A's state, then measure the induced deviation in tensor B's *weights* and *v* at t+k. The
  result is a directed graph of how poison propagates between modules.
- **Closest prior work.** *Localizing Model Behavior with Path Patching* ([2304.05969](https://arxiv.org/abs/2304.05969))
  [verified-snippet]. It works on activations, not optimizer state.
- **Kill-search.** ""path patching" OR "causal graph" between layers optimizer state influence training dynamics
  cross-layer mediation". **NEW.**
- **How it plugs in.** Appendix only. Exact path patching costs O(G²) forks; an AtP approximation is cheaper but
  unvalidated.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 2 | |
| I | 3 | |
| R | 2 | |
| V | 3 | |
| D | 3 | |

### R27 · Stall plus decoupled weight decay: a β2-dependent v→w migration route
- **Pitch.**
  - On coordinates where v is inflated, Adam's step is ≈ 0 for ≈ ln κ/(−ln β2) steps, but AdamW's decoupled decay
    keeps shrinking w by (1−ηλ) every step.
  - So v-poison migrates into **weight-norm damage** that outlives v's recovery and does not depend on the clip
    threshold.
- **Size [inferred, η and λ assumed as nanoGPT-style 6e-4 / 0.1].**
  - Shrink ≈ 1 − exp(−ηλ·k).
  - At β2 = 0.95 (k½ ≈ 158 for κ = 10⁴) it is ≈ 0.9%, probably negligible.
  - At β2 = 0.999 (k½ ≈ 8,104) it is ≈ 38%, which is large.

  The prediction is therefore sharp and β2-dependent. Schedules lower the effective η, so these are upper bounds.
- **Minimal experiment.**
  - Telemetry only: the row norms of S in B0 vs B\* over time.
  - Plus one ablation: λ = 0 on S in all branches, which should kill this route.
  - Run at β2 ∈ {0.95, 0.999} × 4 seeds. About 16 forks.
- **Closest prior work.**
  - *Weight-norm Criticality* ([2607.21005](https://arxiv.org/abs/2607.21005)) [verified-snippet]: normalization plus
    weight decay drive scale-invariant norms to zero and cause spikes. That is a spike *cause*, not a post-spike
    migration route.
  - *Rotational Equilibrium* ([2305.17212](https://arxiv.org/abs/2305.17212)) [verified-snippet].
- **Kill-search.** "AdamW weight decay dominates when update is small frozen coordinates shrink rotational equilibrium
  weight norm criticality loss spike". **NEW.**
- **How it plugs in.**
  - It names a specific v→w migration route, so the migration curve gets a mechanistic decomposition.
  - It predicts a cheap add-on repair: restore the norms of S's rows.
  - It strengthens the β2-instrument design of `09` §1.4.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 5 | |
| I | 3 | |
| R | 3 | likely null at β2 = 0.95 |
| V | 5 | |
| D | 4 | |

### R28 · Irreversibility via the linear-mode-connectivity barrier
- **Pitch.** Δ says *how much* damage remains. The loss barrier along the line between the final B0 (and the repaired
  branch) and the final B\* says whether the spike pushed training into a different basin. A barrier means the damage
  is permanent, and repair must come early. No barrier means a transient displacement.
- **Minimal experiment.** 20 interpolation evals per pair on finished branches. 0 forks.
- **Closest prior work.**
  - Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis* ([1912.05671](https://arxiv.org/abs/1912.05671))
    [verified-snippet]: instability analysis under SGD noise.
  - *Butterfly Effect* ([2506.13234](https://arxiv.org/abs/2506.13234)) [verified-snippet]: loss barriers between
    perturbed runs.
- **Kill-search.** "linear mode connectivity after loss spike does training return to same basin instability barrier";
  "Frankle linear mode connectivity…". **PARTIAL.** The method is standard; applying it to spike counterfactuals and
  repairs is new.
- **How it plugs in.** A second endpoint for #1 and #8: "repair returns training to B\*'s basin". Cheap and
  reviewer-friendly.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 5 | |
| I | 3 | |
| R | 5 | |
| V | 5 | |
| D | 4 | |

### R29 · Streaming Count-Sketch localization
- **Pitch.** At 1B+ scale a ring buffer of full snapshots is expensive. Insert the per-step Δlog v (or the log NIS
  innovations) into a Count Sketch and recover the top-k poisoned coordinates as heavy hitters. Memory is O(k log n),
  for example 5 × 2²⁰ fp32 counters ≈ 20 MB vs ≈ 0.5 GB per 124M snapshot [inferred].
- **Closest prior work.** FetchSGD ([2007.07682](https://arxiv.org/abs/2007.07682)) [verified-snippet] uses a Count Sketch
  for top-k gradient *compression* in federated learning, not anomaly localization.
- **Kill-search.** "count sketch heavy hitters anomaly detection gradient coordinates streaming training monitoring
  FetchSGD". **NEW.**
- **How it plugs in.** The deployable localizer inside the airbag (#21 / BACKUP 2), scored against the exact localizer
  at proxy with recall@k.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 4 | |
| F | 4 | |
| I | 3 | |
| R | 4 | |
| V | 4 | |
| D | 3 | |

### R30 · Repair policy from heterogeneous treatment effects
- **Pitch.** Learn when to undo, restore, roll back or do nothing, from per-instance features (sign, support, lag, LR
  phase), and evaluate the policy on held-out instances.
- **Closest prior work.** The uplift survey [2007.12769](https://arxiv.org/abs/2007.12769) [verified-snippet]. Nothing
  found for training interventions.
- **Kill-search.** "heterogeneous treatment effect causal forest deciding training intervention per run uplift neural
  network training". **NEW (bounded).**
- **How it plugs in.** A discussion paragraph. With ~40 instances, n is too small for a credible learned policy.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 3 | |
| I | 3 | |
| R | 2 | |
| V | 3 | |
| D | 3 | |

### R31 · State drop at stage boundaries
- **Pitch.** Dropping optimizer state on resume or at a fine-tuning handoff is a natural global reset. Is it better to
  drop poisoned state or keep it?
- **Closest prior work.**
  - LLMTailor ([2602.22158](https://arxiv.org/abs/2602.22158)) [verified-snippet]: omitting moments causes spikes or
    noisy updates on resume.
  - Topollai & Choromanska ([2603.16731](https://arxiv.org/abs/2603.16731)) [verified].
- **Kill-search.** "resuming LLM training without optimizer state reinitialized Adam moments loss spike rewarming
  continual pretraining". **TAKEN in substance.** It is the internal Bg / Adam-MR baselines (#8, `09` §2.2).
- **How it plugs in.** Merge into #8. Do not list it separately.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 1 | |
| F | 5 | |
| I | 2 | |
| R | 4 | |
| V | 4 | |
| D | 3 | |

### R32 · Conformal localization sets
- **Pitch.** Calibrate the localizer on planted and induced instances so that it outputs a set S_α with
  P(S_α ⊇ S\*) ≥ 1−α. Repair S_α, and report the coverage and set size on held-out spike types.
- **Caveat [inferred].** Exchangeability fails across spike *types*, so the coverage is per type or marginal.
- **Closest prior work.** Conformal sets in fault diagnosis ([2507.13022](https://arxiv.org/abs/2507.13022))
  [verified-snippet], which are classification sets. Internal #60 is conformal for *detection* false alarms, not
  localization.
- **Kill-search.** "conformal prediction sets fault localization coverage guarantee root cause localization". **NEW** in
  context.
- **How it plugs in.** It turns C2 into an object with a guarantee. The size-vs-coverage trade-off is one more Pareto
  next to #27.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 3 | |
| F | 4 | |
| I | 3 | |
| R | 3 | |
| V | 4 | |
| D | 4 | |

### R33 · Snapshot-free rewind by inverse Adam steps
- **Pitch.** With the logged updates u_s, w_s = (w_{s+1} + η·u_s)/(1−ηλ) rewinds the spike window without weight
  snapshots. Combined with undo, that gives a full retroactive skip.
- **Limit [verified-snippet].** It is not bitwise in fp32. Maclaurin et al. needed exact arithmetic to reverse
  SGD-momentum ([1502.03492](https://arxiv.org/abs/1502.03492)), and the reversal "fails in practice due to finite
  numerical precision" per a course summary.
- **Kill-search.** "reversible learning exact inversion of optimizer step rewind weights without checkpoint Maclaurin
  momentum". **PARTIAL.** The concept is taken; rollback (#8) covers the use.
- **How it plugs in.** Not recommended. It breaks the Δ = 0 discipline.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 2 | |
| F | 4 | |
| I | 2 | |
| R | 3 | |
| V | 4 | |
| D | 2 | |

### R34 · Vectorized counterfactual forks
- **Pitch.** Run K proxy forks as one batched model, using `torch.func.stack_module_state` and `vmap`, to use the
  192 GB MI300X.
- **Closest prior work.** The PyTorch ensembling tutorial ([docs](https://docs.pytorch.org/tutorials/_sources/intermediate/ensembling.rst.txt))
  [verified-snippet]. **TAKEN** as a tool.
- **Caveat.** Batched kernels differ from unbatched ones. A new noop-vs-noop gate is needed *inside* the batch, and
  `determinism.py` is not touched (CLAUDE.md §1).
- **How it plugs in.** An engineering option next to #57 if the proxy throughput is the bottleneck.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 1 | |
| F | 3 | |
| I | 2 | |
| R | 3 | |
| V | 4 | |
| D | 3 | |

### R35 · Agentic "autopsy" automation
- **Pitch.** An LLM agent drives fork → localize → repair and writes the diagnosis.
- **Closest prior work.**
  - *Interactive Training* ([2510.02297](https://arxiv.org/abs/2510.02297)) [verified-snippet]: an LLM agent adjusts
    the LR from logs and recovers from too high an LR.
  - L4: log-based failure diagnosis. Its arXiv ID is not confirmed, so it is UNVERIFIED.
- **Kill-search.** "LLM agent automatically diagnose training instability loss spike debugging agent". **PARTIAL.**
- **How it plugs in.** Reject for ICML. It adds no scientific claim.
- **Scores.**

| Axis | Score | Reason |
|---|---|---|
| N | 2 | |
| F | 3 | |
| I | 2 | |
| R | 3 | |
| V | 2 | |
| D | 2 | |

---

## 2 · Top 5 (ranked) and where each one plugs in

| Rank | Idea | Role in the PRIMARY | Cost | Strongest objection |
|---|---|---|---|---|
| **1** | **R20 Protection law** | **Contribution (mechanism).** It explains *why* v-carried damage persists (the spike's write protects itself) and why certified restoration works (it removes the protection). It adds a pre-registered prediction (integrated-write ratio ranks self-recovery time) and a safety endpoint (collateral forgetting on recent-batch probes) to C3 | CPU plus 0 extra forks | Biswas's law was shown on a toy two-lesson task, and its form is UNVERIFIED until the full text is read. Read it first |
| **2** | **R22 Bit-flip vulnerability map of m vs v** | **Contribution or benchmark arm.** A *natural* source of state-only poison: v flips are silent (no loss spike), m flips spike. It shows undo (gradient-caused poison) and certified restoration (state-caused poison) are complementary, and validates the parity residual (#15) | ≈ 600 proxy forks, subsampleable | "Bit flips are rare." Answer: SDC is documented at fleet scale (2502.12340, 2604.10390), and the moments are untested there |
| **3** | **R12 Spike phase × cooldown** | **Main-text ablation.** The migration curve as a function of LR phase (cosine and WSD), with a falsifiable prediction from the ≈ 5.84·η drift. It tells practitioners *when* repair matters | ≈ 160 proxy forks plus a WSD trunk | The phase effect may simply track η. Report φ against η explicitly |
| **4** | **R3 Forensic batch reconstruction** | **Localizer add-on (data side).** A zero-fork map from damaged wte rows to offending tokens. It gives natural spikes a data-identified, aligned B\* (A4) and seeds R13's filter loop. It also flags a privacy note for the SpikeState release | CPU on existing snapshots | The tied lm_head's output-side gradient may blur common-token rows. Planted truth measures exactly how much |
| **5** | **R27 Stall + decoupled weight decay** | **Mechanism ablation.** A named v→w migration route with a β2-dependent size (≈ 1% at 0.95, ≈ 38% at 0.999, inferred). The λ = 0-on-S ablation kills it. It pairs with the β2-instrument design | ≈ 16 forks | Probably null at β2 = 0.95. That is why it has to be run at 0.999 as well |

**Zero-cost endpoint bundle (do these regardless):**
- **R6** per-token damage;
- **R28** LMC barrier;
- **R23** contagion support curve.

All three re-score forks the PRIMARY already runs. They add construct validity ("the right tokens recover", "back in the
same basin", "the support needed grows with lag") for about 10 p-h in total [assumed].

**Honorable mention:** R1 (information half-life). It is the most "radical" and novel, but its definition hinges on the
placebo-chaos floor. Keep it as an appendix experiment once #4's placebo replicates exist.

---

## 3 · Said plainly: taken, merged or weak
- **R31 is TAKEN.** "Drop the state" is the Bg / Adam-MR baseline (#8).
- **R34 is TAKEN as a tool** (the PyTorch vmap ensembling tutorial).
- **R33's concept is taken** (Maclaurin 2015), and it breaks the bitwise discipline.
- **R2 is mostly #27 re-plotted.** R15 overlaps T8. R17 cannot be a law with 3 rungs.
- **R7 is close to Kourkoutas-β** (2508.12996). It must be framed as a *consistency test*, not "adaptive β2".
- **R8 and R35 are rejects for ICML** (circular, or no scientific claim).
- **New close prior art from this wave**, none of which addresses spikes, localization or repair (each [verified]):
  - Guo 2608.24593: moments as a transport channel.
  - Kim 2608.15824: the cost of v's memory length.
  - Topollai & Choromanska 2603.16731: quantized-EMA staleness and resets.
  - *Hidden Ratio* (NeurIPS 2026 poster 154051): a compressible ratio state, and Adam ≈ sign momentum.
  - Biswas (NeurIPS 2026 OPT, 161497): v-excess protects recent writes. That is the basis of R20.

  Add all five to the related-work table.

## 4 · Search log (2026-10-22, WebSearch standard mode unless noted)

| # | Query (short) | Closest hit |
|---|---|---|
| 1 | mutual information optimizer state memory … decodability Adam moments | 2608.24593 (fetched [verified]) |
| 2 | membership inference / data reconstruction from Adam optimizer state … exp_avg_sq | none on optimizer state |
| 3 | sparse autoencoder on optimizer state Adam second moment interpretability | none |
| 4 | 8-bit Adam block-wise quantization … loss spikes … absmax | 2110.02861, 2407.11722 |
| 5 | dual timescale second moment estimator … fast slow EMA disagreement … | AdEMAMix, Adafactor |
| 6 | learned optimizer meta-learned robust to loss spikes … VeLO | 2211.09760 |
| 7 | Adam second moment token frequency embedding rows … Coupled Adam | 2502.08441 |
| 8 | Shampoo SOAP preconditioner instability stale Kronecker factors loss spike repair | 2607.20548, 2506.03595 |
| 9 | AdEMAMix slow momentum loss spikes … beta3 warmup | 2409.03137, 2509.14233 |
| 10 | effect of loss spike timing warmup vs decay phase final loss … injected perturbation | WSD river valley (ICLR 2025), 2603.16127 |
| 11 | river valley loss landscape cooldown WSD erases perturbation … | no ML hits |
| 12 | automatically filter offending training data after loss spike … | OLMo 2 2501.00656 |
| 13 | taxonomy classification of loss spike causes … fingerprints | K2-V2, 2504.14871 |
| 14 | which layers cause loss spikes consistently … across seeds model sizes | 2410.05052, SPAM, 2607.21005, Bai |
| 15 | switch optimizer mid-training … Adam to sign descent Lion hot swap | AOS 2608.01997 |
| 16 | beneficial loss spikes catapult slingshot … inflated second moment | 2306.04815, 2605.06152 |
| 17 | Biswas NeurIPS 2026 OPT … second moment excess protection | 2608.15824 (fetched [verified]; not Biswas) |
| 18 | adversarial attack on optimizer state Adam moments … | 2609.20532 |
| 19 | bit flip fault injection optimizer state Adam training vulnerability | ORNL, FAT-RABBIT |
| 20 | silent data corruption optimizer state … exponent bit flip Adam moments | 2502.12340, 2604.10390, 2604.00726 |
| 21 | spread of perturbation … butterfly effect … training chaos | 2506.13234, 2404.05782 |
| 22–23 | DiLoCo worker loss spike inner optimizer state … (2 variants) | 2311.08105, 2606.00271 |
| 24 | mixture of experts loss spike expert stale Adam second moment | 2608.04407, 2602.21798 |
| 25 | AdamW weight decay dominates when update is small … weight norm criticality | 2607.21005, 2305.17212 |
| 26–27 | linear mode connectivity after loss spike …; Frankle LMC 1912.05671 | 1912.05671 |
| 28 | count sketch heavy hitters anomaly detection gradient … FetchSGD | 2007.07682 |
| 29 | conformal prediction sets fault localization coverage … | 2507.13022 |
| 30 | heterogeneous treatment effect causal forest … training intervention | 2007.12769 |
| 31 | resuming LLM training without optimizer state … | 2602.22158, 2603.16731 |
| 32 | reversible learning exact inversion of optimizer step … Maclaurin | 1502.03492 |
| 33–34 | vmap ensembles; LLM agent diagnose spikes (first attempts failed on a usage limit; re-run after reset) | PyTorch tutorial; 2510.02297, 2606.28116 |
| 35 | gradient inversion from optimizer state … embedding rows | 2608.01521 |
| 36 | loss spike damage concentrated on rare tokens … | 2404.07965, 2410.11985 |
| 37 | Adafactor factored second moment loss spike row column factor … | 1804.04235, 2304.13013 |
| 38 | per micro-batch gradient removal from Adam moments … | 2508.12220, 2605.17590 |
| 39 | MDL rate distortion model editing sparse edit bits … | 2606.25234 |
| 40 | loss spike recovery cost scaling with model size … | K2-Spike-2 model card only |
| 41 | "path patching" OR "causal graph" … optimizer state | 2304.05969 |
| 42 | injecting spike at different training steps early vs late … critical period | 2503.05139, 2512.06201 |
| 43 | Adam shadow second moment … consistency check robust v … | 2508.12996 |
| 44 | Achille Critical Learning Periods 1711.08856 | confirmed |
| 45 | Wortsman small-scale proxies … | 2309.14322 |
| 46 | probing what Adam second moment encodes … | Biswas 161497, Hidden Ratio 154051 |
| 47 | Unlearning at Scale … 2508.12220 | confirmed |
| F | WebFetch: 2608.24593, 2608.15824, 2603.16731, NeurIPS 154051, NeurIPS 161497 (all read [verified]); arXiv API query (failed, usage limit; not re-run) | — |

**Limits.**
- No Semantic Scholar or OpenReview search was run.
- Most verdicts rest on one or two WebSearch queries per idea.
- Every `[verified-snippet]` citation should be re-read at the source before it goes into the paper.
- Biswas's law (R20) needs a full-text read before it is pre-registered.

## Changelog
- 2026-10-22: created (wave 2). 35 ideas; top 5 = R20, R22, R12, R3, R27. No other file was edited.
