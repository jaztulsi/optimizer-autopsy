# 10 · Wave 2: estimation-theory transfers (finance, adaptive filtering, data assimilation)

*Written 2026-10-22 by the wave-2 Adjacent-Field agent. Nothing here has been run. Every prediction is a
candidate for pre-registration (PLAN_V7 core value 3), not a result.*

**Labels.**
- **[verified]** I read it on the source page, abstract or arXiv API record today.
- **[verified◐]** I saw it only in a search-result summary or a secondary source, not the source text.
- **[inferred]** my own derivation or mapping.
- **[assumed]** a cost or setup assumption.
- **UNVERIFIED** I couldn't confirm it, so it stays out of every claim.

**Cost units [assumed]:** the same as `02_adjacent_fields.md`. One fork is a 200-step `run_fork` plus one
held-out eval, about 12 s on a P100. One fp32 copy of the proxy's state is about 44 MB (11M params); at
124M it is about 0.5 GB.

**What this wave does NOT repeat.** `02_adjacent_fields.md` already covers:
- the undo log (T1);
- the Kalman two-sided innovation test with covariance matching (T5);
- parity residuals (T8);
- anti-windup in the *integrator* sense (T9);
- multi-stream GLR detection (T11);
- MAD-robust v̄ (T13).

This wave only adds mechanisms from estimation theory that those rows lack. Where one of them sharpens an
existing row, the section says so.

---

## 0 · Three design insights (independent of any single row)

1. **"Should v remember this spike?" can be answered without forks, per spike type, and the answer decides
   whether undo is safe.** In finance, jumps are "distinctly less persistent" than continuous variation, and
   "almost all of the predictability" in future volatility comes from the non-jump part (Andersen,
   Bollerslev & Diebold 2007) [verified]. If a spike's g² carries no information about future gradient
   scale, v's memory of it is pure poison and the undo log (T1) is safe. If it does (a level shift), undo
   deletes real information. W2 measures this by regression and W5 by outlier-type classification; both run
   on telemetry alone [inferred].
2. **Cheap streaming robust estimators have a breakdown width, and it lands right on our recipes** [inferred].
   - Bipower variation and MedRV absorb an *isolated* jump but fail on contiguous jumps of width ≥ 2.
   - A median over a window of L steps survives at most ⌊(L−1)/2⌋ jumps, so width w needs L ≥ 2w+1.
   - The current `corrupt_batch` recipe has width 5.

   So no small-window robust v can replace the undo log on multi-step spikes. The undo log knows W exactly
   (or GLR estimates it, W4). This is a principled answer to "why not just use a robust optimizer?".
3. **Bai et al.'s decay stage is RLS estimator windup** [inferred].
   - In RLS, exponential forgetting is applied uniformly in every direction, but only excited directions get
     new data; without persistent excitation the gain blows up [verified◐: arXiv 2004.03910 and Lai &
     Bernstein, via search summaries].
   - Adam does the same thing: every v_i decays by β2 every step whatever g_i², so 1/√v_i grows on
     low-excitation coordinates.
   - Bai et al. describe the preconditioner decaying on its own even as gradients rise [verified: arXiv
     2506.04805 abstract].

   The windup lens predicts *where* under-preconditioning poison lives: on coordinates with little recent
   excitation, such as rare-token rows. It also turns Skeptic objection m14 ("embedding rows dominate every
   localizer") from an artifact into a mechanism that can be tested (W3).

---

## 1 · Ranked table

Verdict = the kill-search for prior application to Adam, adaptive optimizers or loss spikes:
- **NEW:** none found.
- **PARTIAL:** applied nearby (cited).
- **TAKEN:** already done; cite it and use it as a baseline.

Novelty is bounded by the queries in §3; this is a targeted search, not a systematic one.

| # | Transfer | Field | Mapping onto our objects | Falsifiable prediction (short) | Cost @11M | Verdict | Value |
|---|---|---|---|---|---|---|---|
| **W1** | **Jump/continuous split of v**: a shadow bipower (or median) v, the per-coordinate jump share J = 1 − v_BV/v as a localizer, and v_S ← v_BV,S as a repair | Financial econometrics (Barndorff-Nielsen & Shephard 2004; Andersen, Dobrev & Schaumburg 2012) | v_BV,t = β2·v_BV,t−1 + (1−β2)(π/2)\|g_t\|\|g_{t−1}\|. An isolated jump G enters v as G² but enters v_BV as ~π·G·ḡ | Width-1 spike: J ranks the oracle coordinates with Spearman ≥ 0.8 vs log(v⁰/v\*), and the swap gives ≥ 80% of the logged undo's recovery. **Width ≥ 2: bipower fails** (ratio ≈ (π/2)(w−1)/w) | 0 forks to localize; +2 state copies (88 MB); 1 fork per repair | **NEW** | **5** |
| **W2** | **Persistence test: does a spike's g² predict future g²?** A HAR-style C/J regression per tensor, plus an optional two-component v (fast-decaying jump part) | Financial econometrics (Andersen, Bollerslev & Diebold 2007) | log ḡ²_{t+1:t+k} ~ b_C·log C_t + b_J·log J_t on per-tensor telemetry. b_J ≈ 0 means the spike is noise to v | `corrupt_batch`: b_J's CI contains 0 while b_C > 0. `lr_bump` and natural spikes: b_J > 0. The undo benefit (H6) is larger where b_J ≈ 0 | CPU only (telemetry) | **NEW** | **4** |
| **W3** | **The windup lens**: an excitation-based windup score as a localizer prior, plus a **two-sided band-clamp baseline** v ← clip(v, a·v̄, b·v̄) | Adaptive filtering / system ID (Kulhavý 1987; SIFt-RLS 2024; exponential resetting) | Uniform β2 forgetting under lost excitation causes gain blow-up. Directional forgetting = decay v_i only when coordinate i is excited (LazyAdam is the sparse special case). Bounded information = keep v inside a band around v̄ | `corrupt_batch` row damage correlates with row windup: Spearman ≥ 0.5 vs −log v̄_row. **If band-clamp matches C3's repair within the non-inferiority margin, C2's localizer is unnecessary** | 0 forks (score); 1 fork (baseline) | **PARTIAL** (adaptive β2: AFOR 2609.24233, Kourkoutas-β 2508.12996) | **4** |
| **W4** | **GLR estimate-then-correct**: estimate the jump onset θ̂ and size ν̂ in g² from innovations, then subtract (1−β2)β2^{t−θ̂}ν̂ from v | Control / FDI (Willsky & Jones 1976, GLR family) | T1's undo with an *estimated* window instead of logged gradients, plus a likelihood-ratio statistic for the onset | `corrupt_batch`: θ̂ within ±1 step of inject_step in ≥ 90% of instances; estimated undo gets ≥ 80% of the logged undo's recovery. `lr_bump`: no g² jump at onset (correctly declines to undo) | CPU or elementwise GPU ops; 1 fork per repair | **NEW** (application; incremental on T1) | **4** |
| **W5** | **Outlier-type routing** (AO / IO / TC / LS) of per-tensor log-g² streams decides undo vs keep | Time-series econometrics (Fox 1972; Chen & Liu 1993); outlier-corrected GARCH (Franses & Ghijsels 1999) | Additive outlier → undo. Temporary change → partial undo. Level shift → v should adopt it, so don't repair | ≥ 80% of `corrupt_batch` instances classified AO/TC; ≥ 70% of `lr_bump` LS or long-decay TC. Routed repair is non-inferior to always-undo on `corrupt_batch` and superior on `lr_bump` | CPU only | **NEW** | **4** |
| **W6** | **Innovation whiteness as a fork-free "state healed" clock** | Exponential smoothing theory (Muth 1960); Kalman whiteness tests | The EWMA is the optimal forecast for random walk + noise (Muth). Post-spike, normalized innovations g²/v_prior − 1 become serially correlated; time-to-whiteness = when the state has healed | Across instances, time-to-whiteness tracks the fork-measured state-share decay (H2), Spearman ≥ 0.6. Median ≈ 3 e-folds ≈ 60 steps at β2 = 0.95 | CPU only | **PARTIAL** (extends T5/T11) | **3** |
| **W7** | **Extreme-value-calibrated per-coordinate spike flag** (Lee–Mykland-style threshold) instead of SPAM's fixed θ | Financial econometrics (Lee & Mykland 2008) | L_i = \|g_i\|/√v_prior,i; flag the max over n coordinates against a Gumbel threshold, ≈ √(2 ln n) ≈ 5.7 at n = 1.1·10⁷. SPAM's θ = 5000 is \|g\|/√v > 70.7 | A Gaussian Gumbel threshold over-fires on matched normals by ≥ 3× nominal (heavy tails); an empirical-null fit restores nominal. SPAM's θ misses ≥ 70% of `corrupt_batch` hit coordinates that the calibrated flag catches | CPU only | **PARTIAL** (SPAM 2501.06842, AdaGC thresholds) | **3** |
| **W8** | **Hierarchical adaptive inflation**: a per-group v correction factor with a prior centred on 1, updated from innovations | Data assimilation (Anderson 2009) | Upgrades T5's covariance matching (a point estimate) to a Bayesian estimate with uncertainty, varying across tensors and rows. Damps over-correction from noisy NIS | At equal mean recovery, the Bayesian correction gives a lower placebo spread of Δ than T5's point estimate. On `v_inflate`, it converges to 1/κ within a pre-registered number of steps | CPU + 1 fork per repair | **PARTIAL** (T5) | **3** |
| **W9** | **Innovation-triggered inflation with a Lyapunov-type bound** as a stability certificate that doesn't assume a quadratic | Data assimilation (Tong, Majda & Kelly 2015/16, catastrophic filter divergence) | Grow v_S when NIS_S crosses a threshold, so the update energy stays bounded. An alternative to T4's λ_max ≤ 2/η certificate where Tang 2026's superquadratic geometry breaks it | On stress presets where the quadratic certificate holds before the repair but the run still diverges, the triggered rule prevents divergence | 1 fork per variant | **PARTIAL** (StableAdamW already bounds per-tensor update RMS) | **2** |
| **W10** | **Covariance localization of noisy SNR / direction estimates** | Data assimilation (Hamill, Whitaker & Snyder 2001) | Micro-batch replicates (N = 4–40) give gradient-variance estimates with O(1/√N) noise. Taper (Schur product) by a structural distance: same row > same tensor > same layer | With N = 8, a tapered SNR improves AUROC vs S\* by ≥ 0.05; the optimal taper widens with N (as in DA) | proxy: split-batch grads; CPU | **NEW** | **2** |
| **W11** | **Relaxation-to-prior-spread: interpolation space as a repair ablation** | Data assimilation (Whitaker & Hamill 2012) | Certified restoration interpolates v_S^(1−α)·T^α (log space). Also test relaxing in √v ("spread") space and linear space | The interpolation space changes the α at which λ_grad ≤ λ\* by ≥ 20%, but not the recovery at that α | 1 fork per space | **PARTIAL** (our own B9/#12 already has the α path) | **2** |
| **W12** | **Bounded-innovation-propagation robust v** (BIP-GARCH) | Robust statistics (Muler & Yohai 2008) | v_t = β2·v_{t−1} + (1−β2)·v_{t−1}·ρ(g²/v_{t−1}) with bounded ρ. A clipped ρ is SPAM's per-entry clip | — (baseline only) | — | **TAKEN** in substance (SPAM, AdaGC, GradientStabilizer) | **1** |

**Top 5: W1, W2, W3, W5, W4** (ranked in §4).

---

## 2 · One section per transfer

### W1 · Jump/continuous split of v: a zero-fork localizer and an estimated undo

- **Mechanism (one line).** Products of adjacent absolute increments estimate variance robustly to rare
  jumps, so realized variance minus bipower variation isolates the jump part.
- **Source.**
  - Barndorff-Nielsen & Shephard, *Power and Bipower Variation with Stochastic Volatility and Jumps*, J.
    Financial Econometrics 2(1):1–37, Jan 2004
    ([OUP](https://academic.oup.com/jfec/article/2/1/1/960705)) [verified◐]. Per the summaries,
    "RV − BV estimates the quadratic variation of the jump component" [verified◐].
  - Andersen, Dobrev & Schaumburg, *Jump-robust volatility estimation using nearest neighbor truncation*,
    J. Econometrics 169(1):75–93, 2012 ([NBER w15533](https://www.nber.org/papers/w15533)) [verified◐].
    MinRV uses min over blocks of two and MedRV uses median over blocks of three, with better finite-sample
    robustness to jumps and zeros.
- **Mapping [inferred].** Keep two extra per-coordinate tensors, |g_{t−1}| and v_BV, updated inside the
  existing `pre_opt_step` hook (post-clip gradient):
  - v_BV,t = β2·v_BV,t−1 + (1−β2)·(π/2)·|g_t|·|g_{t−1}|.
  - **Isolated jump G on coordinate i:** v gains (1−β2)G², while v_BV gains ≈ (1−β2)·π·G·ḡ (two products,
    G times a normal neighbour). The share of the jump excess that v_BV absorbs is ≈ π·ḡ/G, about 3% at
    κ = G²/ḡ² = 10⁴.
  - **Localizer:** J_i = 1 − v_BV,i/v_i, computed at the cut t1. Standardize it against matched-normal
    snapshots, because two things bias the ratio even without a jump:
    - the (π/2) constant is exact only for zero-mean, serially independent gradients;
    - a non-zero mean and step-to-step correlation shift the ratio.

    So rank by z(log v − log v_BV), not by raw J.
  - **Repair:** v_S ← v_BV,S on the flagged support. This is a state-only "estimated undo": it needs no
    gradient log, no spike time and no B\*.
  - **Breakdown law:** w contiguous jump steps put G·G products into v_BV (w−1) times, so its excess ratio
    becomes ≈ (π/2)(w−1)/w, which is ≥ 1 for w ≥ 3. **Bipower fails completely on `corrupt_batch` width 5.**
    A median over L steps (MedRV generalized) survives width w only if L ≥ 2w+1. That is 11 state copies
    for width 5: 0.5 GB at 11M, about 5.5 GB at 124M.
- **Prediction.**
  1. Width-1 `corrupt_batch` (a new width setting of the existing recipe): Spearman(J-score,
     log v⁰/v\*) ≥ 0.8 over coordinates at t1 = s1+1, and the swap repair recovers ≥ 80% of T1's logged
     undo.
  2. Width 5: bipower J has Spearman ≤ 0.3. The width-11 median variant recovers ≥ 0.8.
  3. `v_inflate`: J ≈ 0, because no gradient jump occurred, and the parity residual (T8) catches it instead.

  Prediction (2) is the sharp one. It fails if bipower still localizes width-5 spikes well, which would mean
  the spike gradients aren't contiguous on the hit coordinates.
- **Cost.** 0 forks to localize; 1 fork per repair; +2 fp32 state copies (88 MB proxy, ~1 GB at 124M);
  elementwise ops per step.
- **Closest prior work / kill-search.**
  - Queries: "bipower variation Adam optimizer second moment robust jump gradient"; "jump robust second
    moment Adam optimizer 'jump' component heavy-tailed gradient spikes separate persistence continuous
    component".
  - Closest hits:
    - an RL value-estimation paper using a "mean-square bipower variation error" (arXiv
      [2411.11697](https://arxiv.org/abs/2411.11697); title not checked) [verified◐];
    - SPAM's per-entry clip g² > θv ([2501.06842](https://arxiv.org/abs/2501.06842)), a truncation, not a
      jump/continuous split;
    - AdaMax (∞-norm v);
    - heavy-tailed-noise Adam convergence ([2607.27383](https://arxiv.org/abs/2607.27383)) [verified◐].

  **NEW**: nothing uses a bipower/median shadow of v as a post-spike localizer or repair.
- **Plugs into.**
  - C2: a magnitude-free, fork-free localizer variant, signed by construction (J > 0 means a jump inflated
    v).
  - C3: a third repair arm next to undo and certified restoration.
  - Breakdown law: a pre-registered result that explains why the logged undo is needed for multi-step
    spikes.
  - Damage migration: the shadow v_BV gives a "jump-free v" to patch into hybrids, an extra cell between
    v⁰ and v\*.

### W2 · Persistence test: should v remember the spike at all?

- **Mechanism (one line).** Split realized variation into continuous and jump parts, then regress future
  variance on both. In markets, the jump coefficient is small: jumps barely forecast future volatility.
- **Source.** Andersen, Bollerslev & Diebold, *Roughing it up: Including jump components in the
  measurement, modeling and forecasting of return volatility*, REStat 89(4):701–720, Nov 2007 (NBER WP
  [w11775](https://www.nber.org/papers/w11775), Nov 2005) [verified, NBER abstract page]. Quotes: "jumps are
  both highly prevalent and distinctly less persistent than the continuous sample path variation process";
  "almost all of the predictability in daily, weekly, and monthly return volatilities comes from the
  non-jump component."
- **Mapping [inferred].**
  - Per tensor, build C_t (bipower part, W1) and J_t = max(RV_t − C_t, 0) from per-tensor mean g².
    [assumed]: per-tensor mean g² either already sits in `capture.tensor_signals` or is a one-line addition
    in `on_step`; not checked.
  - Fit a HAR-style regression over normal steps plus spikes: log mean g²_{t+1:t+k} = a + b_C·log C_t +
    b_J·log(1 + J_t) for k ∈ {10, 50}.
  - **Reading:**
    - b_J ≈ 0: the spike's g² says nothing about future gradient scale. v's memory of it is pure poison,
      and removal (T1, W1, W4) is safe.
    - b_J ≈ b_C: the spike is a real level shift, so v *should* adopt it, and undo deletes information.
  - **Optional variant:** a two-component v = v_C(β2) + v_J(β_J) with β_J ≪ β2. It is an online version of
    "let the jump part forget fast". Use it as an ablation, not a headline (prevention baselines already
    crowd that space).
  - A regime-switching (Markov-switching) variant is possible, but its finance sources weren't checked here
    (UNVERIFIED), and it is subsumed by the two-component form.
- **Prediction.**
  1. `corrupt_batch` and `repeat_ngram` (additive-outlier type): b_J's 95% CI contains 0 while b_C > 0.
  2. `lr_bump` and natural β2 = 0.999 spikes: b_J > 0, because the weights moved and later gradients really
     are larger.
  3. Across instances, H6's undo benefit correlates negatively with b_J (Spearman ≤ −0.5).

  It is refuted if b_J is large for `corrupt_batch`. That would mean random-token batches leave lasting
  gradient-scale changes, which is itself a finding about where the damage lives.
- **Cost.** CPU on telemetry already logged during trunks and forks; 0 forks.
- **Closest prior work / kill-search.**
  - Queries: as for W1, plus "outlier-corrected GARCH analogy Adam optimizer squared gradient EMA jump
    removal loss spike".
  - Hits: only Bai 2026 and Wortsman 2023 (mechanism and RMS detector). Neither asks whether spike g²
    *predicts* future g².
  - Kourkoutas-β ([2508.12996](https://arxiv.org/abs/2508.12996)) shortens β2 during spiky phases (the
    abstract says a layer-wise β2 is driven by a "sunspike" ratio) [verified]. That is the opposite policy:
    it assumes spikes *should* be tracked faster.

  **NEW.**
- **Plugs into.**
  - The damage-migration story: a fork-free, per-type prior for H1/H6.
  - The routing rule for C3: undo only where b_J ≈ 0.
  - Kourkoutas-β becomes a natural contrast baseline.

### W3 · Bai's decay stage as RLS estimator windup: a localizer prior and a band-clamp baseline

- **Mechanism (one line).** RLS with exponential forgetting discounts old information equally in every
  direction but only replaces it where the input excites the system. Without persistent excitation the
  covariance (gain) winds up. Directional forgetting and resetting keep it bounded.
- **Sources.**
  - Kulhavý, *Restricted exponential forgetting in real-time identification*, Automatica 23:589–600, 1987
    [verified◐ via citing papers].
  - Kulhavý & Kárný, directional forgetting, 9th IFAC World Congress, 1984 [verified◐].
  - Lai & Bernstein, *SIFt-RLS*, arXiv [2404.10844](https://arxiv.org/abs/2404.10844), 2024-04-16:
    forgets only within the regressor's row space, with covariance bounds without persistent excitation
    [verified, arXiv API].
  - Exponential and cyclic resetting RLS (Lai & Bernstein, ACC 2023,
    [PDF](https://dsbaero.engin.umich.edu/wp-content/uploads/sites/441/2023/08/BrianRLSACC2023.pdf))
    [verified◐].
  - Salgado, Goodwin & Middleton 1988 (EFRA) is **UNVERIFIED**: not found.
- **Mapping [inferred].**
  - The preconditioner gain is ∝ 1/√v, and v is an information accumulator with uniform forgetting β2.
  - A coordinate whose recent g² is small keeps decaying (v ∝ β2^k), so its gain winds up. When excitation
    returns (a rare token appears; the gradient grows again), the step is too large. That is Bai's
    decoupling [verified: the abstract says the preconditioner decays autonomously despite rising
    gradients].
  - **(a) Windup score:** w_i = log(v̄_i / v_i) over the pre-spike window, or "steps since last strong
    excitation". It is a fork-free prior for the *under-preconditioned* sign.
    - In tied nanoGPT, every wte row gets a small gradient each step through lm_head (softmax), so windup is
      partial, not total [inferred].
    - Rows of rare tokens are the most wound up, so `corrupt_batch` (uniform random tokens; `09_analysis` §7)
      hits exactly those.
  - **(b) Band-clamp baseline:** v ← clip(v, a·v̄, b·v̄), with a and b pre-registered (proposal: a = 0.25,
    b = 4). It edits only out-of-band coordinates, so it localizes automatically, and it handles **both**
    poison signs. It is the "bounded information" cure.
  - **(c) Directional forgetting as a preventive variant:** decay v_i only in proportion to the current
    excitation, e.g. v_i ← v_i + (1−β2)·e_i·(g_i² − v_i) with e_i ∈ [0, 1]. LazyAdam's "update the
    accumulators only for indices present in the batch" is the sparse special case [verified◐, TensorFlow
    Addons docs].
- **Prediction.**
  1. Row-level `corrupt_batch` damage from the oracle correlates with windup: Spearman(per-row damage,
     −log v̄_row) ≥ 0.5. An untied-embedding variant raises it, because there is no lm_head floor.
  2. On natural β2 = 0.999 spikes, the top-decile windup coordinates hold ≥ 50% of the oracle S\*.
  3. **Kill condition for C2:** if the band-clamp is non-inferior to C3's best repair (margin 0.005 nats) on
     the primary contrast, the localizer adds nothing, and the paper must say so.
- **Cost.** The score is 0 forks (v̄ already exists in `capture.py`); the baseline is 1 fork; preventive
  variants need one stress run each.
- **Closest prior work / kill-search.**
  - Queries: "'estimator windup' OR 'covariance windup' Adam optimizer…"; "'directional forgetting' OR
    'restricted forgetting' Adam optimizer…"; "recursive least squares windup forgetting factor analogy Adam
    optimizer loss spike persistent excitation"; "rare token embedding Adam second moment decay…
    LazyAdam".
  - No windup framing or directional forgetting applied to Adam was found. Closest:
    - Bai 2026 (mechanism, global ε/β2 fixes);
    - **AFOR** ([2609.24233](https://arxiv.org/abs/2609.24233), 2026-09-21): estimates β2 online per tensor
      for non-stationary EEG data [verified, arXiv API];
    - **Kourkoutas-β** (layer-wise adaptive β2) [verified];
    - Kim 2026, *Second-Moment Memory in Coordinatewise Adam* ([2608.15824](https://arxiv.org/abs/2608.15824)):
      long v memory slows optimization [verified];
    - AMSGrad's running max, and idea #50's one-sided v-floor in `04_ideas_ranked.md`.

  **PARTIAL.** Adaptive *uniform* forgetting exists. Excitation-gated forgetting, the windup localizer prior
  and the two-sided band-clamp baseline were not found.
- **Plugs into.**
  - C2: a signed prior for the "−" sign.
  - C3: the band-clamp is the easy-to-beat-by-accident baseline the plan needs (`06_methodology` §5).
  - Skeptic m14: the windup lens predicts the embedding-row effect is real.
  - Theory: a named mechanism with known cures.

### W4 · GLR estimate-then-correct: undo without logging the window

- **Mechanism (one line).** For a linear filter, a jump of unknown time and size leaves a known signature
  in the innovations. A GLR matched filter estimates (θ̂, ν̂), and the filter state is then corrected by the
  jump's propagated effect instead of being restarted.
- **Source.** Willsky & Jones 1976, a GLR approach to the detection and estimation of jumps in linear
  systems. The **venue and pages are UNVERIFIED**: secondary sources disagree between IEEE TAC and
  Automatica.
  - The GLR processes Kalman residuals to estimate the jump time and magnitude, and the 1976 paper discusses
    a sliding-window approximation [verified◐: DTIC report ADA112875 and an IFAC 2002 paper, via search
    summaries].
  - A later GLR extension writes the optimal estimate as the jump-free estimate plus a term in the
    estimated jump [verified◐].
  - Whether the 1976 paper itself contains the state-correction step is **UNVERIFIED**. The lecture notes
    that would settle it were not read.
- **Mapping [inferred].**
  - The innovation for v on coordinate (or tensor) i is e_t = g_t² − v_{t−1}.
  - An additive jump ν in g² at θ shifts e_θ by ν, then shifts later innovations by −(1−β2)β2^{t−θ−1}ν,
    because v absorbed the jump.
  - Maximize the likelihood ratio over (θ, w) inside a sliding window of the last ~2/(1−β2) steps. That
    gives the onset θ̂, the width ŵ, the size ν̂ and a test statistic.
  - Correct with v ← v − (1−β2)·Σ_{s∈Ŵ} β2^{t−s}·(ν̂_s − v̄). That is T1's formula with an *estimated*
    window and size.
  - The same machinery runs on m with g instead of g².
- **Prediction.**
  1. `corrupt_batch`: θ̂ = inject_step ± 1 in ≥ 90% of instances, and the estimated undo recovers ≥ 80% of
     the logged undo.
  2. `lr_bump`: no g² jump at onset (gradients are normal); a later, smaller "jump" appears as the loss
     rises. The rule correctly declines to undo at onset, consistent with H6's < 20%.
  3. This gives the "detector-timed undo window" ablation (`06_methodology` §6) a principled estimator
     instead of ±2 steps.
- **Cost.** Elementwise work over a sliding window: about 2/(1−β2) = 40 steps at β2 = 0.95. On per-tensor
  telemetry it is CPU only; per coordinate it needs a ring buffer of g² (the same memory as T1). 1 fork per
  repair.
- **Closest prior work / kill-search.**
  - Queries: "Willsky Jones 1976 'A generalized likelihood ratio approach to the detection and estimation of
    jumps in linear systems'…"; "Willsky Jones GLR jump detection Kalman filter 'state estimate' corrected
    by estimated jump magnitude…"; plus `02`'s FDI queries (#1, #31).
  - No ML or optimizer application found.

  **NEW** as an application. It is incremental over our own T1 (and gives T1 a control-theory lineage).
- **Plugs into.** C3 undo becomes deployable without knowing the spike window, which is the BACKUP-2
  "airbag". The θ̂ estimates also provide onset labels for natural spikes, where inject_step doesn't exist.

### W5 · Outlier-type routing: undo additive outliers, keep level shifts

- **Mechanism (one line).** Time-series outliers come in types with different dynamics: additive (one
  observation), innovational (propagates through the model), temporary change and level shift. Joint
  estimation classifies and adjusts each, and volatility models fitted after correcting additive outliers
  forecast better.
- **Sources.**
  - Fox, *Outliers in time series*, JRSS-B 34:350–363, 1972 (AO vs IO) [verified◐].
  - Chen & Liu, *Joint estimation of model parameters and outlier effects in time series*, JASA
    88(421):284–297, Mar 1993 ([PDF via ISTAT](https://www.istat.it/it/files/2014/06/Joint-Estimation-of-Model-Parameters-and-Outlier-Effects-in-Time-Series.pdf));
    four types IO/AO/TC/LS [verified◐, IMSL documentation].
  - Franses & Ghijsels, *Additive outliers, GARCH and forecasting volatility*, IJF 15(1):1–9, 1999, doi
    [10.1016/S0169-2070(98)00053-3](https://doi.org/10.1016/S0169-2070(98)00053-3) ([EUR
    repository](https://repub.eur.nl/pub/2174)). Outlier-corrected GARCH forecasts better across samples,
    criteria and four markets [verified◐].
- **Mapping [inferred].**
  - The per-tensor log-g² stream is the series, and each spike event is classified by Chen–Liu's
    standardized statistics:
    - **AO** (one bad batch: `corrupt_batch`, `repeat_ngram`) → undo: T1, W1 or W4.
    - **TC** (decaying) → partial undo.
    - **LS** (`lr_bump` after onset; natural spikes where the weights moved) → don't repair v; v *should*
      track the new scale.
    - **IO** → the outlier feeds the dynamics, so look at the weights.
  - `v_inflate` doesn't appear in the g stream at all. It is an AO in the *state*, caught by T8's parity
    residual.
  - Franses & Ghijsels is the finance precedent for T1: correct the outlier, then rerun the variance
    recursion.
  - Link to W2: the type answers "should v remember?" categorically, and W2 answers it by regression. They
    should agree.
  - Kourkoutas-β's "shorten memory during spikes" is the right policy only for LS, and the wrong one for AO.
- **Prediction.**
  1. ≥ 80% of `corrupt_batch` instances classified AO/TC, and ≥ 70% of `lr_bump` instances LS or
     long-decay TC.
  2. The routed repair (undo iff AO/TC) is non-inferior to always-undo on `corrupt_batch` and superior on
     `lr_bump` (where undo deletes useful history).
  3. On natural spikes, the classifier's labels agree with the cross-seed alignment test (data-caused →
     AO): Cohen's κ ≥ 0.5.
- **Cost.** CPU only (iterative fits on short series); 0 forks for classification.
- **Closest prior work / kill-search.**
  - Queries: "'additive outlier' OR 'innovational outlier' gradient stream neural network training loss
    spike classification optimizer"; "outlier-corrected GARCH analogy Adam…".
  - Hits: robust NAR estimators for time-series *data*, and *Outliers with Opposing Signals* (ICLR 2024),
    which uses "outlier" for groups of training examples, not a gradient-stream type.

  **NEW.**
- **Plugs into.** It is the decision rule that makes C3 safe across spike types (it removes the H6
  asymmetry by construction), and a fork-free cause classifier that complements K6's alignment test.

### W6 · Innovation whiteness as a fork-free clock for when the state has healed

- **Mechanism (one line).** The EWMA is the optimal forecast for a random walk plus noise (Muth). With a
  well-specified gain, the one-step innovations are white; persistent autocorrelation means the estimate is
  biased.
- **Source.** Muth, *Optimal properties of exponentially weighted forecasts*, JASA 55:299–306, 1960
  [verified◐]. A Kalman reformulation is in [QuantEcon](https://python-advanced.quantecon.org/muth_kalman.html)
  [verified◐]. The Mehra whiteness test is cited in `02` T5 (Mehra 1970) [verified◐ there]. Desroziers 2005
  was not checked (UNVERIFIED) and is not used.
- **Mapping [inferred].**
  - Normalized innovations: u_t = g_t²/v_{t−1} − 1 per tensor.
  - After an over-inflating spike, u stays negative for a run of steps, because v is too large. After a
    decay-stage event, it stays positive.
  - Define **time-to-whiteness** as the first step after s1 at which the lag-1 autocorrelation of u over a
    trailing window falls inside its matched-normal 95% band.
  - This is an *observable*, fork-free definition of "the optimizer state has healed", to set against the
    fork-measured state-share decay (H2).
- **Prediction.**
  1. Across instances, time-to-whiteness and the fork-measured half-life of the state share have Spearman
     ≥ 0.6.
  2. At β2 = 0.95, the median time-to-whiteness is ≈ 3 e-folds ≈ 60 steps [inferred from `09` §1.1].
  3. Under the β2 instrument (`09` §1.4), time-to-whiteness scales ∝ 1/(−ln β2).

  It is refuted if whiteness returns while the fork-measured state share is still > τ. That would mean
  innovations miss state damage, which is useful to know for the deployable detector.
- **Cost.** CPU on telemetry.
- **Closest prior work / kill-search.** It covers the same ground as T5/T11 in `02` (their queries apply).
  No whiteness or recovery-clock use for optimizer state was found. **PARTIAL.**
- **Plugs into.** H2 gains a cheap observable, and the BACKUP-2 detector gains an "all-clear" signal
  (when to stop a repair window).

### W7 · Extreme-value-calibrated per-coordinate spike flag

- **Mechanism (one line).** Test each observation's standardized size, and set the threshold from the
  extreme-value (Gumbel) distribution of the maximum over n tests, so the family-wise false-flag rate is
  controlled.
- **Source.**
  - Lee & Mykland, *Jumps in Financial Markets: A New Nonparametric Test and Jump Dynamics*, RFS 2008, doi
    [10.1093/rfs/hhm056](https://doi.org/10.1093/rfs/hhm056) [verified◐]. The Gumbel-based threshold is
    described in follow-ups [verified◐].
  - Nunes & Ruas 2024 (Finance Research Letters) correct the norming constants: use the maximum of
    *folded* normals ([IDEAS](https://ideas.repec.org/a/eee/finlet/v59y2024ics1544612323011868.html))
    [verified◐].
- **Mapping [inferred].**
  - L_i = |g_i|/√v_prior,i, using the **prior** v as T5 does, because the posterior bounds the ratio.
  - Gumbel's leading term is ≈ √(2 ln n): 5.69 at n = 1.1·10⁷ (proxy) and 6.11 at n = 1.24·10⁸ (124M).
  - SPAM's θ = 5000 on g²/v is |g|/√v > 70.7, about 12× higher in z units. Gradients are heavy-tailed, so
    fit the null on matched-normal steps (an empirical extreme-value or conformal fit) instead of assuming
    Gaussian. Together with T13's MAD scale, that gives a calibrated per-step coordinate flag.
- **Prediction.**
  1. Gaussian Gumbel at α = 0.01 over-fires on matched-normal steps by ≥ 3× nominal; the empirical-null fit
     stays within 2×.
  2. On `corrupt_batch`, the calibrated flag catches ≥ 70% of the hit coordinates that SPAM's θ misses.
  3. Repairing only the calibrated-flag coordinates closes more gap than SPAM-θ coordinates at an equal
     edit budget.
- **Cost.** CPU / elementwise; 0 forks for the flag.
- **Kill-search.** Covered by the `02` T13/T5 queries plus this wave's SPAM checks. Thresholded
  per-coordinate flags exist (SPAM θ; AdaGC's 1.04·γ per tensor); no family-wise calibrated threshold was
  found. **PARTIAL.**
- **Plugs into.** It supplies the support S for W1/W4 repairs and a calibrated version of a component of
  SPAM's baseline.

### W8 · Hierarchical adaptive inflation (an upgrade to T5)

- **Mechanism (one line).** Treat each state element's covariance-inflation factor as a random variable
  with a prior, and update it from innovation statistics, so the inflation varies in space and time and is
  damped when the evidence is weak.
- **Source.** Anderson, *Spatially and temporally varying adaptive covariance inflation for ensemble
  filters*, Tellus A 61A(1):72–83, 2009, doi
  [10.1111/j.1600-0870.2008.00361.x](https://doi.org/10.1111/j.1600-0870.2008.00361.x)
  ([UCAR](https://www.cisl.ucar.edu/opensky-publications/articles%3A17973/spatially-and-temporally-varying-adaptive-covariance-inflation-for-ensemble-filters)).
  A normally distributed inflation variable per state element; better mean and variance estimates than
  other inflation methods in low-order tests [verified◐].
- **Mapping [inferred].**
  - Per group S (tensor, row or head), keep λ_S ~ N(μ_S, σ_S²) with prior μ = 1.
  - Each step, update by the likelihood of the observed mean g²_S under λ_S·v̂_prior,S, with σ_S floored so
    it keeps adapting.
  - Repair: v_S ← μ_S·v_S. λ > 1 means v is too small (grow it); λ < 1 means v is too large (shrink it).
    So it is signed, like T5, but with uncertainty: groups with few coordinates or noisy NIS barely move.
- **Prediction.** At equal mean recovery on `v_inflate` and `corrupt_batch`, the hierarchical estimate's
  placebo spread of Δ is ≤ 0.7× T5's point-estimate repair. On `v_inflate` (κ = 10⁴), μ_S reaches 1/κ
  within a pre-registered number of steps set by the prior SD.
- **Cost.** CPU arithmetic, plus 1 fork per repair.
- **Kill-search.** Query: "'covariance inflation' OR 'covariance localization' OR 'relaxation to prior
  spread' Adam optimizer second moment neural network training". Hits: only v-initialization papers
  ([2412.02153](https://arxiv.org/abs/2412.02153)) and structured preconditioners. **PARTIAL**, relative to
  our own T5 (Mehra covariance matching).
- **Plugs into.** It stabilizes T5's covariance-matching repair, and it is a principled way to set "how much"
  for signed restoration without HVPs.

### W9 · Innovation-triggered inflation with a Lyapunov-type guarantee

- **Mechanism (one line).** Ensemble Kalman filters can diverge catastrophically to machine infinity.
  Adaptive inflation triggered by innovation size (and cross-covariance) restores nonlinear stability, with
  Lyapunov functions and geometric ergodicity.
- **Source.**
  - Tong, Majda & Kelly, *Nonlinear stability of the ensemble Kalman filter with adaptive covariance
    inflation*, arXiv [1507.08319](https://arxiv.org/abs/1507.08319), Jul 2015 (Commun. Math. Sci. 2016;
    volume and pages UNVERIFIED) [verified◐].
  - Companion: arXiv [1507.08307](https://arxiv.org/abs/1507.08307) [verified◐].
- **Mapping [inferred, loose].** Grow v_S when NIS_S exceeds a threshold, scaled so the per-step update
  energy stays below a pre-set bound. That gives a stability certificate that doesn't rely on the local
  quadratic model behind T4/B9, which Tang 2026's superquadratic geometry threatens.
  - **Caveat:** their proof uses EnKF structure (observation operator, ensemble covariance) that doesn't
    transfer one-to-one. Treat this as a design heuristic until someone proves an Adam analogue.
- **Prediction.** On stress presets where certified restoration's quadratic check passes but the run still
  diverges within 50 steps, the triggered rule prevents divergence in ≥ 80% of instances.
- **Cost.** 1 fork per variant.
- **Kill-search.** The same DA query as W8 found nothing. But StableAdamW already bounds the per-tensor
  update RMS (Wortsman 2023), which is a bounded-update mechanism. **PARTIAL.**
- **Plugs into.** A fallback certificate for BACKUP 1's weak point (Tang 2026).

### W10 · Covariance localization of noisy SNR estimates

- **Mechanism (one line).** Small ensembles produce spurious long-range covariances. Multiply the sample
  covariance elementwise by a compactly supported correlation function, with a taper length that grows with
  ensemble size.
- **Source.** Hamill, Whitaker & Snyder, *Distance-dependent filtering of background error covariance
  estimates in an ensemble Kalman filter*, MWR 129:2776–2790, 2001
  ([NOAA PSL](https://psl.noaa.gov/psd1/pubs/2001/details/hamill_01abs.html)) [verified◐]. Sampling noise
  dominates for small ensembles or weak covariances, and the optimal length grows with ensemble size
  [verified◐].
- **Mapping [inferred].**
  - The "ensemble" is the micro-batch gradients: N = 40 at 124M via `grad_hook`, and split-batch at the
    proxy.
  - The "distance" is structural: same row (0), same tensor (1), same layer (2), other layer (3).
  - Taper any directional variance, covariance or score map C2 builds from them.
  - Payoff is only material if C2 uses *directional* (non-diagonal) statistics. Diagonal SNR needs no
    localization.
- **Prediction.** With N = 8, a tapered directional SNR improves AUROC vs S\* by ≥ 0.05 over untapered. The
  best taper length increases monotonically with N ∈ {4, 8, 16, 40}.
- **Cost.** CPU; proxy split-batch gradients add memory [assumed].
- **Kill-search.** Same DA query: no hit. **NEW**, but low value.
- **Plugs into.** C2's SNR component, as an ablation.

### W11 · Relaxation to prior spread: interpolation space as an ablation

- **Source.** Whitaker & Hamill, *Evaluating methods to account for system errors in ensemble data
  assimilation*, MWR 140(9):3078–3089, 2012, doi
  [10.1175/MWR-D-11-00276.1](https://doi.org/10.1175/MWR-D-11-00276.1) (citation from the reference list
  of arXiv 1409.0589) [verified◐].
  - Their 2011 ECMWF workshop version proposes relaxing the posterior *spread* back toward the prior,
    instead of the perturbations (as in Zhang et al.'s relaxation-to-prior, which is UNVERIFIED)
    [verified◐].
  - The claim that spread-space relaxation *performs better* is **UNVERIFIED** and is not used.
- **Mapping and prediction [inferred].** Certified restoration already walks v_S(α) = v_S^(1−α)·T^α (log
  space). Add √v and linear-v paths as an ablation. Prediction: the space changes α\* (where
  λ_grad ≤ λ\*) by ≥ 20%, but not the recovery at α\*. If recovery does change, the space matters and must be
  pre-registered.
- **Cost.** 1 fork per space.
- **Verdict.** **PARTIAL**: our own B9/#12 already has the α path.

### W12 · Bounded-innovation-propagation robust v

- **Source.** Muler & Yohai, *Robust estimates for GARCH models*, J. Stat. Planning & Inference
  138:2918–2940, 2008, doi [10.1016/j.jspi.2007.11.003](https://doi.org/10.1016/j.jspi.2007.11.003)
  (citation via the CRAN `robustGarch` docs; the BIP description is from a secondary source) [verified◐].
- **Mapping [inferred].** A bounded ρ inside the variance recursion bounds each observation's effect on
  every later variance. A clipped ρ *is* SPAM's per-entry clip (g² ≤ θv); AdaGC and GradientStabilizer
  prevent the same contamination upstream.
- **Verdict.** **TAKEN** in substance. The only residual idea is a *redescending* ρ, where extreme jumps
  contribute almost nothing instead of θv. That's an optimizer variant, out of scope. Use SPAM as the
  baseline.

### Considered and not pursued
- STA/LTA seismic triggers with the long-term average frozen during events, and Holt-Winters
  aberrant-behaviour detection that skips anomalous points in the baseline. Both look like precedents for
  T13's trimmed v̄, but neither source was checked (**UNVERIFIED**), so no claim is made.
- Variable forgetting factor RLS (Fortescue et al. 1981): **UNVERIFIED**. Its role is covered by AFOR and
  Kourkoutas-β (W3) and by the AO/LS argument (W5).

---

## 3 · Search log (all 2026-10-22)

Semantic Scholar was not used in this wave, because it returned HTTP 429 earlier today (see `02` §4).
WebSearch, WebFetch and the arXiv API did all the work. Retrieved text was treated as data.

| # | Engine | Query / URL | Result → use |
|---|---|---|---|
| 1 | WebSearch | bipower variation Adam optimizer second moment robust jump gradient | No Adam use; RL MSBVE paper (2411.11697) → W1 **NEW** |
| 2 | WebSearch | Barndorff-Nielsen Shephard 2004 "Power and bipower variation…" JFEC | JFEC 2(1):1–37 → W1 source |
| 3 | WebSearch | Andersen Bollerslev Diebold 2007 "Roughing it up"… REStat | REStat 89(4):701–720 → W2 source |
| 4 | WebFetch | nber.org/papers/w11775 | Abstract read; quotes verified → W2 |
| 5 | WebSearch | Lee Mykland 2008 "Jumps in Financial Markets…" Gumbel threshold | RFS doi; Nunes & Ruas 2024 correction → W7 |
| 6 | WebSearch | Andersen Dobrev Schaumburg 2012 "Jump-robust volatility estimation…" MedRV MinRV | J. Econometrics 169(1) → W1 |
| 7 | WebSearch | Franses Ghijsels 1999 "Additive outliers, GARCH and forecasting volatility" | IJF 15(1):1–9 → W5 |
| 8 | WebSearch | Kulhavý 1987 "Restricted exponential forgetting…" directional forgetting windup | Automatica 23:589–600 (via citers) → W3 |
| 9 | WebSearch | Salgado Goodwin Middleton 1988 "Modified least squares… exponential resetting and forgetting" | Not found → **UNVERIFIED**; Lai & Bernstein resetting instead |
| 10 | WebSearch | "estimator windup" OR "covariance windup" Adam optimizer deep learning second moment decay preconditioner blow-up | None; Bai 2026 closest → W3 |
| 11 | WebFetch | arxiv.org/abs/2506.04805 | Bai abstract: the preconditioner decays autonomously despite rising gradients → W3 mapping |
| 12 | WebSearch | "directional forgetting" OR "restricted forgetting" Adam optimizer neural network training second moment | None for Adam; AFOR, 2608.15824, FOGO → W3 **PARTIAL** |
| 13 | WebSearch | recursive least squares windup forgetting factor analogy Adam optimizer loss spike persistent excitation | RLS only; no Adam → W3 |
| 14 | WebSearch | rare token embedding Adam second moment decay instability… LazyAdam sparse | LazyAdam docs; Kourkoutas-β → W3 |
| 15 | arXiv API | id_list = 2608.15824, 2609.24233, 2508.12996, 2404.10844 | Titles, authors and dates verified |
| 16 | WebSearch | Willsky Jones 1976 "A generalized likelihood ratio approach…" | Secondary sources only; venue conflicting → W4 (UNVERIFIED venue) |
| 17 | WebSearch | Chen Liu 1993 "Joint estimation of model parameters and outlier effects…" | JASA 88(421) → W5 |
| 18 | WebSearch | Fox 1972 "Outliers in time series" JRSS-B | 34:350–363 → W5 |
| 19 | WebFetch | dspace.mit.edu R-0645 (a GLR report) | HTTP 429, not read |
| 20 | WebSearch | "additive outlier" OR "innovational outlier" gradient stream neural network training loss spike… | None → W5 **NEW** |
| 21 | WebSearch | outlier-corrected GARCH analogy Adam optimizer squared gradient EMA jump removal loss spike | None → W2/W5 **NEW** |
| 22 | WebSearch | Willsky Jones GLR… "state estimate" corrected by estimated jump magnitude… | Secondary only → W4 correction step UNVERIFIED |
| 23 | WebSearch | Basseville Nikiforov "Detection of Abrupt Changes" 1993 free pdf | Book is free online; not read |
| 24 | WebFetch | rt.isy.liu.se GLR lecture 8 | Redirect; the follow-up fetch was declined, so it was not read |
| 25 | WebSearch | Anderson 2009 "Spatially and temporally varying adaptive covariance inflation…" | Tellus A 61A:72–83 → W8 |
| 26 | WebSearch | Tong Majda Kelly 2016 "Nonlinear stability of the EnKF with adaptive covariance inflation" | arXiv 1507.08319 → W9 |
| 27 | WebSearch | Hamill Whitaker Snyder 2001 "Distance-dependent filtering…" | MWR 129:2776–2790 → W10 |
| 28 | WebSearch | "covariance inflation" OR "covariance localization" OR "relaxation to prior spread" Adam optimizer second moment… | None → W8/W9/W10 |
| 29 | WebSearch | Muler Yohai 2008 "Robust estimates for GARCH models" BIP… | JSPI 138 via CRAN → W12 |
| 30 | WebSearch | Whitaker Hamill 2012 "Evaluating methods…" RTPS MWR | MWR 140(9) via a reference list → W11 |
| 31 | WebSearch | Muth 1960 "Optimal properties of exponentially weighted forecasts" | JASA 55:299–306 → W6 |
| 32 | WebSearch | jump robust second moment Adam optimizer "jump" component heavy-tailed gradient spikes… | None; 2607.27383, SPAM, AdaMax → W1/W2 **NEW** |

**Gaps.**
- Every NEW verdict comes from 1–3 targeted queries, not a systematic review. Re-run W1, W2, W4 and W5 with
  Semantic Scholar (with a key) and Google Scholar before claiming "first" in the paper.
- Most finance and DA sources were confirmed only from bibliographic records and abstracts ([verified◐]).
  Read the full texts of ABD 2007 and BN-S 2004 before writing the related-work paragraph.

---

## 4 · Top 5, ranked

1. **W1 · Jump/continuous split of v.** It gives a fork-free, signed localizer, a third repair arm (an
   estimated undo with no gradient log), and a sharp, falsifiable **breakdown law**: bipower fails at
   width ≥ 2, and a median needs a window ≥ 2w+1. That law explains why the logged undo is needed for
   multi-step spikes. NEW. Cost: 2 state copies, 0 forks to localize.
2. **W2 · Persistence test** ("does a spike's g² predict future g²?"). This is a CPU-only measurement that
   decides, per spike type, whether v *should* remember the spike, and therefore whether undo is safe. It is
   grounded in a verified empirical regularity from finance: jumps are less persistent and carry almost no
   forecast power. NEW.
3. **W3 · The windup lens.** It names Bai's mechanism (estimator windup under uniform forgetting), predicts
   where "−"-signed poison lives (low-excitation, rare-token rows; testable against the oracle), and
   supplies the **two-sided band-clamp baseline**, the cheapest control that could make C2's localizer
   unnecessary. PARTIAL (adaptive β2 exists), but its baseline and localizer parts are new.
4. **W5 · Outlier-type routing.** AO/TC → undo; LS → keep. It removes the H6 asymmetry by construction and
   gives a fork-free cause classifier that can be checked against the cross-seed alignment test. NEW.
   Finance precedent for T1: Franses & Ghijsels 1999.
5. **W4 · GLR estimate-then-correct.** It makes the undo log deployable without knowing the spike window
   (onset and size estimated from innovations), provides onset labels for natural spikes, and gives T1 a
   control-theory lineage. NEW as an application; incremental on T1.

**Suggested pre-registration additions (proposals, for PLAN_V7 §4.7):**
- the W1 breakdown prediction (width 1 vs 5);
- the W2 regression spec (k ∈ {10, 50}; b_J CI rule);
- the band-clamp constants (a = 0.25, b = 4) and its role as a required baseline in H7;
- the W5 routing rule, fixed on seed 0.
