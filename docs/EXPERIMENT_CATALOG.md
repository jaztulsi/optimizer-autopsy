# Experiment catalog and new algorithms, ready for the GPU

*2026-10-23 · written for the team (jaztulsi, Eddie, the reviewer session) · companion to [`PLAN_V8.md`](../PLAN_V8.md)*

**Nothing in this file has been run.**
- Every algorithm is a design, and every prediction is a hypothesis.
- Every novelty verdict is **pending** until the reviewer session's kill-search (`master-research/output/13_reviewer_*.md`)
  comes back.

**Labels:**
- [I]: our derivation.
- [A]: an assumed cost (one proxy fork ≈ 10 s of MI300X time until `step_timer` replaces it).
- **Paper tag:** **P1** = ICML 2027 core; **P2** = NeurIPS 2027 scale and generality; **P3** = SpikeState / tools
  (datasets track); **BL** = backlog, run when there is time.
- **Pri:** 1 = the GPU queue's first week; 2 = the P1 build; 3 = later.

**What this is.** The user asked for many things to test as a research program, not one-off runs: novel ideas,
different settings and temperatures, different algorithms, and **new algorithms we invent**.
- Part A specifies 16 new algorithms.
- Part B lists about 70 experiment families across every axis.
- Part C gives sweep templates.
- Part D is the GPU day-1 queue.

Confirmatory tests for P1 are only the 4 in PLAN_V8 §7.0. **Everything here is exploratory unless marked C-1…C-4.**

**Notation (per coordinate i):**
- g_t is the post-clip gradient;
- m_t = β1 m_{t−1} + (1−β1) g_t;
- v_t = β2 v_{t−1} + (1−β2) g_t²;
- m̂ and v̂ are bias-corrected;
- the update is u_t = η_t m̂_t/(√v̂_t + ε);
- the weight step is w_{t+1} = (1 − η_t λ) w_t − u_t;
- the spike window is W = [s0, s1); r ≡ β1/√β2.

---

## Part A · New algorithms

**Status: write-and-wait.** The user re-confirmed (2026-10-23): "dont test anythin on cuda or my mac … once i give u
the gpu for amd test them on that". Every algorithm and experiment here is queued for the AMD MI300X.

A key fact drives most of the algorithms. On a spike-dominated coordinate (G/ḡ ≫ 1/√(1−β2)),
u_{s+k}/η ≈ (1−β1)/√(1−β2) · r^k. This is a corollary of Patitucci & Mokhtari 2026's single-spike Thm 3.3
(2602.10204).
- The spike step writes a fraction (1 − r) of the total drift.
- **Adam's memory (the m-echo) writes the remaining fraction r:** 92% at (0.9, 0.95), in the G/ḡ → ∞ limit.

A second indirect channel has the opposite sign: the **v-stall**. Inflated v suppresses ordinary learning for
≈ ln((1−β2)G²/ḡ²)/(−ln β2) steps. Inflated v is also *protective*, because it is what bounds the echo. So repairs
must handle m and v together (C-2). Most of the damage is still preventable *after* the spike, if you act within
about 1/(1−r) ≈ 13 steps. Part A is a family of ways to act in that window, plus tools to measure it.

| ID | Name | One line | Kind | Paper | Verdict (pending) |
|---|---|---|---|---|---|
| A1 | **REC**: Reversible Echo Cancellation | Online, soft, reversible removal of a flagged step's echo from the moments the update uses | repair (online) | P1 | likely NEW |
| A2 | **QV**: Quarantined moments (two-phase commit) | Withhold a flagged gradient from m and v (impute it), then commit or discard it later with exact forward compensation | prevention (online) | P1/P2 | PARTIAL (SPAM clips irreversibly) |
| A3 | **BiH-Adam**: Bipower–Huber Adam | A jump-robust bipower second moment plus Huberized m, so isolated spikes neither inflate v nor dominate m | optimizer | P2 | PARTIAL (TAdam, AdaTerm) |
| A4 | **SAMS**: Spike-Aware Migration Scheduler | Choose the repair by lag τ using the remaining-echo fraction r^τ | policy | P1 | likely NEW (as a policy) |
| A5 | **CWB**: Counterfactual Weight Back-Projection ("echo-aware amnesiac subtraction for Adam") | Retroactive skip without rollback: subtract the echo-written weight drift computed from the same gradients | repair (offline/online) | P1 | **PARTIAL** (reviewer): Amnesiac ML, SWIFT, DeltaGrad. Ours: the Adam echo, plus the measured open- vs closed-loop error |
| A6 | **LRA**: Linear-Response Autopsy | A fork-free estimate of the w/m/v shares and the mediation curve by forward-mode JVP through H unrolled steps | measurement | P1/P2 | PARTIAL (Guo 2026; Liu & Li) |
| A7 | **MomentABFT** | Algorithm-based fault tolerance for Adam state: linear row/column checksums detect, locate and correct single bit flips in m/v | reliability | P2/P3 | likely NEW |
| A8 | **SCR**: Stability-Certified Restoration (two-sided) | A log-space move of v on the support with a Bai-Ĥ certificate; "bulk unstable" is a falsifiable failure flag | repair | P1 | PARTIAL (Biswas; GI-Adam) |
| A9 | **PAR**: Protection-Aware Release | Release v-excess on the rows a spike wrote, ranked by the integrated-write ratio (Biswas law), to speed healing | repair (variant of A8) | P1 | PARTIAL |
| A10 | **IAW-Adam**: Integral Anti-Windup | Clamp each coordinate's *leaky integral* of updates (back-calculation anti-windup), not the per-step update | optimizer / repair | P2 | PARTIAL? (PID optimizers; StableAdamW) |
| A11 | **GT-Oracle**: causal group testing over optimizer-state groups | Adaptive splitting with a placebo-calibrated firing threshold, plus a SuSiE fallback | measurement | P1 | PARTIAL (ACDC, ddmin) |
| A12 | **FTA**: Forensic Token Attribution under tied embeddings | Rank wte rows by Δlog v over an output-side baseline, to recover the spike batch's tokens with no data logs | localization (data side) | BL/P3 | PARTIAL (Melis 2019) |
| A13 | **DT-Gate**: dual-timescale v with a disagreement gate | A fast v detects a jump; the slow v doesn't absorb it (β2_eff = 1) for that coordinate and step | prevention | P2 | PARTIAL (Kourkoutas-β) |
| A14 | **Sketch-REC** | Echo registers in a Count-Sketch when the flagged set is dense | systems | BL | PARTIAL |
| A15 | **Selective β1 drop** | After a flag, lower β1 on the flagged support for k steps, shrinking r and the echo | repair | P1 (ablation) | PARTIAL (SPAM resets globally) |
| A16 | **Shadow-v monitor** | Track v_BV next to v; the jump share J = 1 − v_BV/v is a zero-fork, zero-log localizer | localization | P1 | PARTIAL (estimator use new) |

### A1 · REC: Reversible Echo Cancellation
- **Problem.** The echo of a flagged step (s, with flagged set F_s) keeps writing damage for about 1/(1−r) steps.
  SPAM-style clipping at write time is irreversible, and it is wrong when the "spike" was a real level shift.
- **State.** For each active event, sparse on F_s:
  - echo registers e_m = (1−β1) g_s and e_v = (1−β2)(g_s² − v̄_s), with v̄_s = v_{s−1};
  - a confidence c ∈ [0, 1].
- **Each step k > s:**
  - decay the registers: e_m ← β1 e_m, e_v ← β2 e_v (exactly how those contributions decay inside m and v);
  - compute the update from m′ = m − c·e_m and v′ = max(v − c·e_v, v_floor) on F_s;
  - **the stored m and v are untouched**, so the edit is reversible.
- **Confidence.**
  - c starts at c₀ (from the detector's tail probability).
  - It is updated by a persistence test (W5 outlier router): if g² stays elevated for L steps, it is a level shift
    and c → 0; if g² returns to baseline, it is an additive outlier and c → 1.
- **Commit.** When the decision is final or e_v < tol·v: set m ← m − c e_m and v ← v − c e_v, then free the registers.
- **Exactness [I].** With c = 1, m′ is exactly the m Adam would hold had g_s been 0 (later gradients fixed). v′ is
  the v had g_s² been v̄_s. So REC with c = 1 is the online form of the offline undo (A5 adds the weight side).
- **Cost.** O(|F|) memory and FLOPs; zero when nothing is flagged.
- **Predictions.**
  - `corrupt_batch` with c = 1 removes ≥ 70% of the memory-written damage (C-1's indirect share).
  - On `lr_bump`, the router sets c → 0, so REC does no harm.
- **CPU test.**
  - On a synthetic gradient sequence, REC(c = 1) equals a direct EMA recomputation with g_s zeroed (m) and g_s²
    replaced by v̄ (v).
  - Toggling c 1 → 0 restores Adam's moments exactly.
- **Closest prior work:** SPAM (irreversible clip); SWIFT (one-step inversion); Biswas (an offline clear).
  **To kill-search:** "reversible", "soft deletion", "deferred" plus Adam moments.

### A2 · QV: Quarantined moments (two-phase commit)
- **Why not just quarantine v.** Dropping the spike's excess from v while it stays in m makes m/√v explode. In
  Adam, v's inflation is what bounds the spike step. Quarantine must cover **both** channels.
- **Flagging.** At step t, flag coordinate i if z = g_i²/v̂_{t−1,i} > z\*.
- **While quarantined:**
  - impute g̃_i = m̂_{t−1,i} and g̃_i² = v̂_{t−1,i}, so the coordinate keeps moving with its own momentum;
  - push (t, g_i) to a pending buffer.
- **Decision after L steps** (persistence test):
  - **commit** with forward compensation: m += (1−β1)β1^{t′−t}(g_i − g̃_i), v += (1−β2)β2^{t′−t}(g_i² − g̃_i²);
  - or **discard**.
- **Pairing.** QV is "insert later", the preventive twin of REC's "delete later".
- **Predictions.**
  - Discarding has an indirect share ≈ 0 on `corrupt_batch`.
  - Committing on a level shift costs ≤ L steps of slower adaptation.
- **CPU test.** Commit equals the plain Adam trajectory's moments at t′, exactly, given the same later gradients.
- **Closest prior work:** SPAM's clip (g → sign·√(θ v)); AdaGC; robust M-estimators (TAdam).

### A3 · BiH-Adam: Bipower–Huber Adam (a new optimizer)
- **Robust scale.** s_t = β2 s_{t−1} + (1−β2)·(π/2)·|g_t|·|g_{t−1}| (bipower; ≈ σ² for independent Gaussian noise).
  An isolated jump G enters as ≈ G·ḡ, not G².
- **Huberized gradient.** g̃_t = clip(g_t, ±c·√ŝ_{t−1}) with c ∈ {3, 5, 8}.
- **Moments.** m_t = β1 m_{t−1} + (1−β1) g̃_t.
- **Update.** u_t = η m̂_t/(√ŝ_t + ε).
- **Why both parts.** With s alone, the spike step's update grows like √G (unbounded) [I]. The Huber clip bounds m's
  intake, and s doesn't stall afterwards. Together they remove both echo channels.
- **Variants:**
  - **MedV:** s from med(|g_t|, |g_{t−1}|, |g_{t−2}|)² × ≈ 1.41 (MedRV constant). It survives one jump in 3 steps;
    width w needs a window of ≥ 2w + 1.
  - **Per-row s** (Adam-mini-style) to save memory.
- **Predictions.**
  - Width-1 `corrupt_batch`: indirect share ≈ 0, with no v-stall.
  - Width 5: bipower degrades as derived (the shadow ratio ≈ (π/2)(w−1)/w), so MedV with a window of 11 is needed.
  - Clean training: within ~1% of AdamW's loss [A, to be measured].
- **CPU test.**
  - s is unbiased on i.i.d. Gaussian gradients.
  - A single jump moves s by O(G·ḡ).
  - The spike-step update is bounded by c(1−β1)/√(1−β2).
- **Closest prior work:** TAdam (Student-t), AdaTerm, AdaShift (shifted v), SPAM. No ML use of bipower variation was
  found (12_factcheck_novelty N6).

### A4 · SAMS: Spike-Aware Migration Scheduler (policy)
- **Rule.** Given the lag τ between the flag and the decision:
  - remaining echo fraction ρ(τ) = r^τ (dominated coordinates);
  - if ρ > ρ_hi: state-only repair (REC/undo) is enough;
  - if ρ < ρ_lo: the damage is already in w, so use CWB (A5);
  - in between: both.
- **Thresholds.** With ρ_hi = 0.5 and ρ_lo = 0.1 at r = 0.923: switch points τ ≈ 8.7 and 28.8 steps [I].
- **Predictions.**
  - Undo-only benefit halves every ≈ 8.7 steps of lag.
  - SAMS ≥ the best fixed operator at every lag.
- **CPU test.** The policy picks the operator correctly from (τ, β1, β2).

### A5 · CWB: Counterfactual Weight Back-Projection (retroactive skip without rollback)
- **Counterfactual moments.** For k ≥ s on the flagged support (or all coordinates):
  - m̃_k = m_k − (1−β1)β1^{k−s} g_s;
  - ṽ_k = v_k − (1−β2)β2^{k−s}(g_s² − v̄).
- **Accumulator.** A ← (1 − η_k λ)·A + (u_k − ũ_k), where ũ_k is computed from m̃_k and ṽ_k. Steps k = s … t.
- **Decision at t.** w ← w + A (removes the spike step's own update *and* its echo). Then REC-commit the moments.
- **Open-loop approximation.** Later gradients are assumed unchanged. **The gap to the true skip-step fork
  (`Bskipstep`) measures the closed-loop nonlinearity**, which is itself a measurement for the paper.
- **Cost.** One accumulator plus g_s on the support; can be computed online as the window runs.
- **Predictions.**
  - The error vs `Bskipstep` grows with spike size and lag.
  - It is ≤ 20% of Δ_B0 at τ ≤ 10 for `corrupt_batch` [A].
- **CPU test.** On a quadratic with gradients fixed in advance, CWB equals the exact skip-step trajectory.
- **Closest prior work:** Amnesiac ML (stored per-batch weight updates; no optimizer echo); DeltaGrad (L-BFGS);
  Unlearning-at-Scale (replay); SWIFT (one step).

### A6 · LRA: Linear-Response Autopsy (fork-free factorial)
- **Shares.** With δX = X^{B0}(t1) − X^{B\*}(t1) for X ∈ {w, m, v}, compute
  φ_X(h) ≈ ∂L_val(t1+h)/∂X(t1) · δX for every h ≤ H in one forward-mode (JVP) pass through H unrolled AdamW steps.
  Each step's tangent needs one HVP, run under math SDPA.
- **Mediation.** Mediation shares follow by putting tangents on g_s only.
- **Cost** [A]: ≈ 3–4 forks' worth vs the 8 × |τ| forks of the factorial.
- **Validation.** The LRA-vs-fork agreement as a function of h is a direct measure of linearity. It is first order
  only, with no interaction terms.
- **CPU test.** On a quadratic model, LRA equals the exact factorial.
- **Closest prior work:** Guo 2026 (optimizer-aware derivatives through AdamW state); Liu & Li 2026 (linearized
  response operator).

### A7 · MomentABFT: algorithm-based fault tolerance for optimizer state
- **Checksums** (fp64), for each 2-D state tensor:
  - row checksums R_m ← β1 R_m + (1−β1)(g·1) and column checksums C_m ← β1 C_m + (1−β1)(1ᵀg);
  - the same for v with g⊙g.
  - m and v are linear in g and g², so the checksums update exactly.
- **Check every K steps:** ‖m·1 − R_m‖ and ‖1ᵀm − C_m‖ against a rounding tolerance ∝ √n·eps·|m|.
  - A single corrupted element shows up in exactly one row and one column, which **locates (i, j)**.
  - Correct it: m_ij ← m_ij − residual_i.
- **Cost.** O(rows + cols) memory and O(n) FLOPs per step (fused with the step). It replaces the parity residual's
  full v_{t−1} copy.
- **Predictions.** It detects 100% of exponent flips. It corrects them to rounding, with < 2% step overhead [A].
- **CPU test.** Inject one flip into m or v and check that it is located and corrected; no false alarms over 10⁴
  clean steps.
- **Closest prior work:** ABFT for GEMM (Huang & Abraham 1984); ATTNChecker (attention); LayerCheck (per-layer
  restore). No ABFT for optimizer state was found yet; to be kill-searched.

### A8 · SCR: Stability-Certified Restoration (two-sided)
- **Inputs.** Support S with signs σ, from the localizer. Target T: v̄, or fresh mean g² over K micro-batches at the
  current w (preferred at late τ, by central flows).
- **Path.** v_S(α) = v_S^{1−α} T_S^α (log-linear). Under-preconditioned coordinates move up and over-preconditioned
  ones move down.
- **Search.** α\* is the largest α with λ_grad(Ĥ(α)) ≤ (1−δ)·2/η, using **Bai's normalized Ĥ**. Bisection, about 6
  probes × 5 HVPs.
- **Certificate.** If α = 1 still fails, emit **"bulk unstable"**: the complement is the problem, so a global action
  is needed. That flag is falsifiable.
- **CPU test.** On a synthetic quadratic, α\* is the exact boundary, and the bulk-unstable flag fires when it should.

### A9 · PAR: Protection-Aware Release (variant of A8)
- **Rows.** Rank rows by the spike's integrated-write ratio (the Biswas statistic).
- **Release.** Scale their v-excess by (1−ρ) so clean gradients can overwrite the damage.
- **Prediction.** Faster self-healing on rows above the 0.63 ratio. On healthy recent batches, a forgetting cost
  measured by a probe (the safety endpoint).

### A10 · IAW-Adam: Integral Anti-Windup
- **Leaky integral.** I_i ← γ I_i + u_i with γ = r. It tracks exactly the echo's integration.
- **Normal scale.** s_i is an EMA of |I_i| over calm steps.
- **Back-calculation.** If |I_i| > κ s_i, scale u_i ← u_i · κ s_i/|I_i|.
- **Why.** The damage is the *integral* of post-spike updates (saturating at D·η). Per-step clipping (StableAdamW)
  doesn't bound it.
- **Per-row variant** to save memory.
- **Prediction.** It caps the indirect share at ≈ κ× normal drift, with ≤ 1% clean-loss cost [A].
- **Closest prior work:** StableAdamW; PID optimizers; integrator anti-windup in control. To be kill-searched.

### A11–A16 (short specs)
- **A11 GT-Oracle.**
  - Binary splitting over tensor → row → head groups.
  - A group "fires" if patching it from B\* changes Δ by more than τ_f = max(0.005, 5× the placebo spread).
  - Fork bound: ≤ ⌈log₂ C(G, s)⌉ + s − 1.
  - Under interactions, falls back to SuSiE credible sets.
- **A12 FTA.**
  - With tied embeddings, every wte row gets an output-side gradient each step. Normalize Δlog v_row by an expected
    output-side baseline (unigram probability × typical hidden norm).
  - Rare tokens in the spike batch stand out. Report recall@k on planted batches, tied vs untied.
- **A13 DT-Gate.**
  - A fast v_f (β = 0.5, per row). If v_f/v > ζ, set β2_eff = 1 for that coordinate and step, and route the
    gradient through QV.
- **A14 Sketch-REC.** Count-Sketch registers when |F| is large. The error vs memory is a Pareto curve.
- **A15 Selective β1 drop.**
  - Set β1 → β1′ on F for k steps.
  - Prediction: the indirect share falls to ≈ β1′/√β2.
- **A16 Shadow-v.**
  - Track v_BV only as a monitor (not in the update). J = 1 − v_BV/v is the localizer score.
  - Prediction: Spearman ≥ 0.8 vs the oracle on width-1 spikes; it fails at width ≥ 2 by design.

---

## Part B · Experiment families

Columns:
- **Cost** in forks, ×10 s [A] at the proxy unless a scale is named.
- **Readout:** the primary metric.

The prediction, where one exists, is in the question or the readout.

### B.1 Instrument validity (run first)

| ID | Question | Factors × levels | Readout | Cost | Paper | Pri |
|---|---|---|---|---|---|---|
| E01 | Is noop-vs-noop exactly 0 on AMD? | in-process / cross-process × through a spike window × each rung × each cfg change | max\|Δ\| (must be 0) | 2 forks per cell | all | 1 |
| E02 | How big is the placebo (chaos) spread? | 1-ulp perturbation × K = 5 × sites {40%, 70%} × rungs | SD of Δ | 10 per site | P1 | 1 |
| E03 | Are hybrids on-manifold? | every hybrid cell | violation rate of \|m\| ≤ C√v | 0 (offline) | P1 | 1 |
| E04 | Does the harness see a planted poison? | `v_inflate` κ ∈ {10², 10⁴, 10⁶} × 2 tensors + decoys | v main effect ≥ 0.8 at τ = 1 (G1) | 8 per κ | P1 | 1 |
| E05 | Off-manifold control | B\* state at ±5 steps vs the aligned state | Δ shift | 4 | P1 | 2 |
| E06 | Is the eval set disjoint and stable? | val batches ≥ 1000; 1M vs 4M tokens | Δ eval SD | 0 | P1 | 1 |

### B.2 Mediation and migration (the P1 headline)

| ID | Question | Factors × levels | Readout | Cost | Paper | Pri |
|---|---|---|---|---|---|---|
| **E10** (C-1) | How much damage does the memory write after the spike? | undo **channel** {m, v, m+v} × lag τ ∈ {0, 1, 3, 10, 30, 100} × {corrupt_batch, repeat_ngram (width ≥ 3), lr_bump, sdc_grad} × dose G/ḡ × 10 seeds | indirect share Δ_ind/Δ_B0 (Shapley over m and v) | 18 per instance | P1 | 1 |
| **E10b** (C-2) | Is poisoned v protective? | undo-v-only vs B0 vs undo-m+v, at τ ∈ {0, 3, 10} | Δ(undo-v-only) ≥ Δ_B0 > Δ(undo-m+v)? Also the instability rate | ⊂ E10 | P1 | 1 |
| E11 | Does β1 order the indirect share? | (β1, β2) ∈ {(.8,.95), (.9,.95), (.95,.95), (.9,.99), (.9,.999)} | monotone in β1 (pre-registered direction only); the value vs r reported | E10 × 5 | P1 | 1 |
| E11b | How long is the v-stall? | dose G/ḡ ∈ {3, 10, 30, 100} | lost progress on the spiked coordinates vs ln((1−β2)G²/ḡ²)/(−ln β2) | ⊂ E10 | P1 | 2 |
| E12 | w/m/v share curve | 2³ hybrids × τ ∈ {1, 10, 30, 60, 150, 400} | Shapley shares φ_w, φ_m, φ_v | 6 per τ | P1 | 1 |
| E13 | Does the step counter mediate? | 2⁴ (adds k) at τ = 10 | φ_k | +8 | P1 | 2 |
| E14 | Does poisoned state alone damage clean weights? | B0's m / v / (m, v) on B\*'s w × κ dose {1, 10², 10⁴} | Δ > τ rate; Spearman ρ | 9 per instance | P1 | 2 |
| E15 | How nonlinear is the echo? | CWB (A5) vs `Bskipstep` fork × spike size × lag | closed-loop error | 2 per cell | P1 | 2 |
| E16 | Can a JVP replace the factorial? | LRA (A6) vs E12 × horizon h | agreement vs h | LRA ≈ 3–4 | P1/P2 | 2 |
| E17 | Does drift saturate as derived? | corrupt_batch magnitude × 8 doses | drift on dominated coordinates vs the closed form | 8 | P1 | 2 |
| E18 | Is a large spike clip-invariant? | clip ∈ {∞, 1.0, 0.5, 0.2} × large doses | Δ_B0, drift | 4 per dose | P1 | 2 |
| E19 | Does cooldown erase state poison? | spike at 5 LR phases × {cosine, WSD} | φ_state vs phase (controlling for η) | ≈ 160 total | P1 | 2 |
| E20 | Is weight decay a v→w route? | λ ∈ {0, 0.1} on S × β2 ∈ {.95, .999} | Δ, ‖Δw‖ on S | 16 | P1 | 3 |
| E21 | Does protection predict self-healing? | integrated-write ratio per row × spike types | Spearman vs recovery time | 0 (re-score) | P1 | 2 |
| E22 | Residual memory of v | ‖v − v\*‖ vs τ | decay vs ln 2/(−ln β2) | 0 | P1 | 2 |
| E23 | Zero-cost endpoints | per-token Δ (rare vs frequent), LMC barrier, contagion support curve | three figures | 0 | P1 | 2 |
| E24 | Information half-life | probe decodes "spike happened" from w, m, v vs τ above the placebo floor | half-life per channel | 0 | P2 | 3 |
| E25 | Beneficial spikes | do some spikes *improve* final loss, and is that state-carried? | sign of Δ, φ | 0 (re-score) | BL | 3 |

### B.3 Localization (P1)

| ID | Question | Factors × levels | Readout | Cost | Paper | Pri |
|---|---|---|---|---|---|---|
| E30 | How cheap and stable is the oracle? | GT-Oracle (A11) vs exhaustive per-tensor at the proxy | forks used; Jaccard across placebo seeds | 5–40 per instance | P1 | 2 |
| **E31** (C-3: signed − magnitude-only ≥ 0.05 AUROC vs the oracle, paired; absolute AUROC is exploratory) | Which localizer wins? | AtP-state, two-sided NIS, λ-decomposition, shadow-v J (A16), windup prior, magnitude-only, Fisher mismatch, Lanczos mass, Hutchinson diagonal, random | AUROC / AUPRC / P@k vs the oracle and vs planted truth with decoys | 0 forks (scores) + oracle | P1 | 2 |
| E32 | Granularity | tensor / row / head / coordinate | AUROC; repair-top-k | 0 | P1 | 2 |
| E33 | Reference | B\* / v̄ / fresh g² | AUROC | 0 | P1 | 2 |
| E34 | Forensic tokens | FTA (A12), tied vs untied × rare/frequent tokens | recall@k of batch tokens | 0 (CPU on snapshots) | BL/P3 | 3 |
| E35 | Conformal localization sets | split-conformal over instances | coverage of S\* | 0 | P2 | 3 |
| E36 | Universal poison sites | overlap of S\* across seeds and rungs | Jaccard | 0 | P2 | 3 |
| E37 | Sketch localization | Count-Sketch heavy hitters of Δlog v | recall vs memory | 0 | BL | 3 |

### B.4 Repair (P1; C-4 is the primary contrast)

| ID | Question | Factors × levels | Readout | Cost | Paper | Pri |
|---|---|---|---|---|---|---|
| **E40** (C-4) | Repair bake-off | **ours:** undo (offline), REC (A1), QV (A2), CWB (A5), SCR (A8), PAR (A9), IAW (A10), selective β1 drop (A15), SAMS (A4). **Baselines:** band-clamp, Adam-MR, bias-correct Bg, Bg, Bvreset, Bvbar, SPAM, ZClip, AdaGC, StableAdamW, GradientStabilizer, OLMo 6σ skip, skip-step (oracle + detector), rollback-and-skip, Br, B0 | gap closed G at end of schedule; TOST vs the best baseline. **C-4:** state-only undo at a *detector-timed* lag must be non-inferior to the best non-oracle baseline **and** beat the OLMo 6σ skip-step on the same spikes. The niche spikes are sub-threshold multi-step `repeat_ngram` (width ≥ 3) and loss-detected spikes | ≈ 25 per instance | P1 | 2 |
| E41 | Pareto curves | edit fraction × G; GPU-seconds × G | the Pareto frontier | 0 (re-score) | P1 | 2 |
| E42 | Lag robustness | operators × τ ∈ {0, 3, 10, 30, 100} | G vs τ; SAMS vs fixed | 5 per operator | P1 | 2 |
| E43 | Window misspecification | exact / detector-timed / ±2 steps | G loss | 3 | P1 | 3 |
| E44 | Repair target | v̄ / fresh g² / central-flow target × τ | G | 3 per τ | P1 | 2 |
| E45 | Support size | top-k for k ∈ {0.1, 1, 5, 10, 25}% | G vs k | 5 | P1 | 2 |
| E46 | Certificate validity | SCR's "bulk unstable" flag vs actual divergence | precision / recall of the flag | 0 (re-score) | P1 | 3 |
| E47 | Protection safety | PAR collateral forgetting on recent-batch probes | probe loss change | 0 | P1 | 3 |

### B.5 Detection and monitoring (P1 side product, P2)

| ID | Question | Factors × levels | Readout | Cost | Paper | Pri |
|---|---|---|---|---|---|---|
| E50 | Which detector gives lead time at a fixed false-alarm rate? | lower-tail NIS CUSUM, e-CUSUM (single λ), PageHinkley, conformal max-score, ZClip z, OLMo 6σ, RMS_t (Wortsman), λ_grad (Bai), whiteness clock, shadow-v J | lead time at run-level FP ≤ 0.05; recall | 0 (telemetry) | P1/P2 | 2 |
| E51 | MomentABFT | flip injection × bit × tensor | detect / locate / correct rates; overhead | 0 forks + 1 overhead run | P2/P3 | 2 |
| E52 | External detector check | K2-V2 W&B logs (65B) | recall of the logged spikes | CPU | P2 | 3 |
| E53 | Healed clock | time-to-whiteness vs the fork-measured state decay | Spearman | 0 | P2 | 3 |

### B.6 Reliability: faults in optimizer state (P1 arm, P2)

| ID | Question | Factors × levels | Readout | Cost | Paper | Pri |
|---|---|---|---|---|---|---|
| E55 | Bit-flip map | bit ∈ {exponent 23–30, high mantissa} × {m, v} × {up, down} × layer type × step phase | outcome class (benign / silent freeze / spike / divergence); recovery by undo, SCR, ABFT | ≈ 600 (subsample) | P1/P2 | 2 |
| E56 | Gradient SDC | flip in g before vs after the clip | undo recovery | ≈ 40 | P1 | 2 |
| E57 | Permanent faults | a stuck-at bit in one m/v element | detection latency | ≈ 20 | P2 | 3 |

### B.7 Settings sweeps ("temps": hyperparameters and temperatures)
These run as moderators of E10/E12/E40. Each cell runs the mediation sweep (6 forks) plus B0/B\* on 3 seeds.

| ID | Factor | Levels | Why | Paper | Pri |
|---|---|---|---|---|---|
| E60 | β2 | 0.95, 0.99, 0.999 | v memory; Tang's phase boundary | P1 | 1 |
| E61 | β1 | 0.8, 0.9, 0.95 | sets r and the echo | P1 | 1 |
| E62 | ε | 1e-6, 1e-8, 1e-12 | ε floor (T11); width sensitivity | P1/P2 | 2 |
| E63 | peak LR | × {0.5, 1, 2, 4} | near the edge of stability | P1 | 2 |
| E64 | tokens/step | 4k, 16k, 64k, 256k (with β2\* = β2^{B\*/B} vs without) | Adam memory in tokens | P2 | 2 |
| E65 | precision | fp32, bf16 autocast | numerics as a moderator; bf16 natural spikes | P2 | 2 |
| E66 | schedule | constant, cosine, WSD (decay 10%/20%) | cooldown effect (E19) | P1 | 2 |
| E67 | warmup | 0, 1%, 5% of steps | early-training spikes | P2 | 3 |
| E68 | grad clip | ∞, 1.0, 0.5, 0.2 | saturation (E18) | P1 | 2 |
| **E69** | **attention temperature** (logit scale ×{0.5, 1, 2, 4}) × QK-Norm on/off | attention-logit growth as a spike source and moderator (Wortsman small-scale proxies; Marin) | P1/P2 | 2 |
| E70 | output-logit temperature / z-loss | z-loss ∈ {0, 1e-4} × output scale | output-logit divergence spikes | P2 | 3 |
| E71 | weight decay | 0, 0.01, 0.1 | the v→w route (E20); power-lines timescale | P2 | 3 |
| E72 | init scale | × {0.5, 1, 2} | instability onset | BL | 3 |

### B.8 Architecture (P2)

| ID | Factor | Levels | Readout | Pri |
|---|---|---|---|---|
| E75 | QK-Norm | off / on | incidence; φ_state; indirect share | 2 |
| E76 | embeddings | tied / untied | wte artifact; FTA | 2 |
| E77 | width × depth | d ∈ {192, 384, 768} × L ∈ {3, 6, 12}, SP vs μP/CompleteP | L1 collapse; exponents (H14) | 2 |
| E78 | MLP | GELU / SwiGLU | FP8-style divergence route | 3 |
| E79 | residual | sequential / parallel (NeoX) | Pythia comparability | 3 |
| E80 | norm | pre-LN / post-LN / sub-LN | Takase's "spike no more" conditions | 3 |

### B.9 Optimizers (P2; negative controls included)

| ID | Optimizer | What it tests | Pri |
|---|---|---|---|
| E85 | AdamW (reference) | — | 1 |
| E86 | **SGD + momentum** | **negative control**: no v, so φ_v = 0 and the echo ratio is β1 | 2 |
| E87 | **Lion / sign-momentum** | no v; bounded per-step update | 2 |
| E88 | Adam-mini (reimplemented) | coarse v; how fine-grained is poisoning? | 2 |
| E89 | Muon + AdamW | no v on 2-D; momentum_buffer poisoning | 2 |
| E90 | Adafactor | factored v (rows/cols); no m | 3 |
| E91 | AdEMAMix | slow momentum as an m-channel instrument | 3 |
| E92 | **BiH-Adam (A3)**, **IAW-Adam (A10)**, **QV-Adam (A2)**, **DT-Gate (A13)** | our invented optimizers: clean-loss cost vs spike immunity | 2 |
| E93 | Sophia / SOAP / Shampoo | curvature or Kronecker preconditioners (R10) | 3 |

### B.10 Scale (P2)

| ID | Rung | How | Readout | GPU-h [A] | Pri |
|---|---|---|---|---|---|
| E100 | 5 small configs (E77) | own trunks, SP (+ μP) | fit L1 exponents | 2–4 | 2 |
| E101 | mid rung 6L/384 | own trunk; **predicted before running** | inside the PI? (H13) | 1 | 2 |
| E102 | 124M, n = 5 | own trunk, 16,384 tokens/step | C-1, C-2, and C-4 at scale | 6–8 | 2 |
| E103 | Pythia-160M | released final state + 200-step burn-in | L0 rates; H1 sign | ≈ 4 | 2 |
| E104 | Pythia 410M / 1B / 1.4B / 2.8B | the same | the ladder | ≈ 145 | 3 |
| E105 | Pythia-6.9B | one 192 GB card | top rung | ≈ 115 | 3 |
| E106 | OPEN-1B, **"replay the spike the production run skipped"** | a calm run: only 2 skip events (steps 1,782 and 43,737). Fork at ≈ 1,700 / ≈ 43,700 and run with the skip vs without | 2 real natural-gradient events at 1.6B with an exact aligned B\* (n = 2 case study) | ≈ 20–60 (≈ 4.9M tokens/step) | 3 |
| E107 | own 350M / 1B instrumented trunks | stress presets, multi-GPU trunk, single-GPU forks | natural spikes at scale | 45 / 300+ | 3 |

### B.11 Natural spikes (P2)

| ID | Source | Readout | Pri |
|---|---|---|---|
| E109 | **C-4 niche spikes:** sub-threshold multi-step `repeat_ngram` (width 3–10, dose tuned so no single step trips 6σ / 128) and loss-detected spikes | whether skip-step misses them, and whether undo recovers them | 1 |
| E110 | stress presets (β2 = 0.999, high LR, no QK-Norm, tiny ε) × seeds | population of natural spikes; detector lead | 2 |
| E111 | bf16 + flash-attention explosion at 124M (nanoGPT #303/#524) | reproduces on ROCm? localization | 3 |
| E112 | cross-seed alignment (data- vs dynamics-driven) | permutation test | 2 |
| E113 | data-selected v-inflation (S2) | a natural-like plant with known support | 2 |
| E114 | ordering-attack window (S12) | same data, reordered | 3 |
| E115 | FP8 e4m3fnuz | T4 only | 3 |

### B.12 Transfer domains (backlog / T4)
- **E120:** RL fine-tuning (PPO/GRPO) instabilities.
- **E121:** continued pretraining from released states.
- **E122:** ViT/CLIP patch-embedding spikes (the Wortsman setting).
- **E123:** MoE expert-state poisoning.
- **E124:** DiLoCo cross-worker donor repair.
- **E125:** FedAdam server-state recovery.

### B.13 Security and privacy (P3 / backlog)
- **E130:** integrity scanner on shipped `optimizer.pt` files (invariant + parity + MomentABFT checksums). Its
  false-positive rate is measured on our own legitimate resets.
- **E131:** optimizer-state token leakage (FTA). This is a privacy note for the SpikeState release.
- **E132:** a worst-case minimal state edit (adversarial poisoning, R21).
- **E133:** "spikes as attacks" taxonomy table (writing only).

### B.14 Benchmark (P3)
- **E140:** SpikeState v0. All P1 instances with (w, m, v, v̄), clean twins and oracle maps, plus Croissant + RAI
  metadata.
- **E141:** a leaderboard protocol: localizer AUROC vs the oracle and repair-AUC on held-out instances.
- **E142:** baseline submissions (every E31 localizer and every E40 repair).

---

## Part C · Sweep templates ("temps")

These are written as YAML overrides on `research/experiments/autopsy/*.yaml` (to be created by the implementer;
dotted-key overrides through `load_config(path, overrides)`).

```yaml
# C1 mediation sweep (E10/E11): one instance = one (spike, seed, betas)
base: proxy_realistic.yaml
spike: {kind: corrupt_batch, inject_frac: 0.4, width: 5}
betas: [[0.9, 0.95], [0.8, 0.95], [0.95, 0.95], [0.9, 0.99], [0.9, 0.999]]
undo_lags: [0, 1, 3, 10, 30, 100]
horizon: end_of_schedule
seeds: {tune: [0], heldout: [1, 2, 3, 4, 5, 6, 7, 8, 9]}
eval: {val_batches_from: 1000, tokens: 1_000_000}
placebo: {k: 5, ulp: 1}
```

```yaml
# C2 repair bake-off (E40): every operator on the same instance set
operators: [undo, rec, qv, cwb, scr, par, iaw, beta1_drop, sams,
            band_clamp, adam_mr, bg_biascorrect, bg, bvreset, bvbar, spam, zclip, adagc,
            stableadamw, gradstab, olmo_skip, skipstep, skipstep_det, rollback_skip, br, b0]
primary: {metric: gap_closed_G, test: TOST, margin: max(0.005, 2*placebo_sd), baseline_pick: seed0}
```

```yaml
# C3 bit-flip map (E55)
targets: [m, v]
bits: [30, 29, 28, 27, 26, 25, 24, 23, 22]   # fp32 exponent + top mantissa
directions: [up, down]
layers: [wte, attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj, ln]
phase: [0.2, 0.5, 0.8]
recover: [none, undo, scr, momentabft]
```

```yaml
# C4 attention temperature (E69)
attn_logit_scale: [0.5, 1.0, 2.0, 4.0]
qk_norm: [false, true]
readout: [spike_incidence, max_attn_logit, indirect_share]
```

---

## Part D · GPU day-1 queue (in order; each step gates the next)

1. `python -m research.kaggle.step_timer`: measured speeds replace every [A].
2. E01: determinism, in-process and cross-process, through a spike window (G0).
3. `python -m research.spikes.capture --selfcheck`.
4. E06 + E02: eval set and placebo spread.
5. E04: planted `v_inflate` sanity (G1).
6. **The MVE, on 12 instances:** E10 + E10b. Use the 2×2 undo arms at τ ∈ {0, 10, 30} on `corrupt_batch`,
   `repeat_ngram` (width 3) and `v_inflate` + decoys, plus E12 at τ = 10 (G2/G3). This is the reviewer's order.
7. E11: the β1 arms on the same instances, then localization (E30/E31), then the C-4 battery (E40 + E109).
8. E17/E18: saturation and clip invariance (cheap, mechanistic).
9. E55 (subsampled) + E56: the reliability arm.
10. E30/E31: oracle + localizer bake-off.
11. E40: repair bake-off (C-4).
12. E100–E103: scale (small configs → mid → 124M → Pythia-160M).

## Part E · Algorithm → experiments map

| Algorithm | Tested in |
|---|---|
| A1 REC | E10, E40, E42, E92 |
| A2 QV | E40, E92, E110 |
| A3 BiH-Adam | E92, E10 (as the trunk optimizer), E110 |
| A4 SAMS | E42 |
| A5 CWB | E15, E40, E42 |
| A6 LRA | E16 |
| A7 MomentABFT | E51, E55, E130 |
| A8 SCR / A9 PAR | E40, E44, E46, E47 |
| A10 IAW-Adam | E18, E40, E92 |
| A11 GT-Oracle | E30, E31 |
| A12 FTA | E34, E131 |
| A13 DT-Gate | E92 |
| A14 Sketch-REC | E37, E40 |
| A15 β1 drop | E11, E40 |
| A16 Shadow-v | E31, E50 |
