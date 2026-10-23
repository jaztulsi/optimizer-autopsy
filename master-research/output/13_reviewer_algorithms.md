# 13b · Reviewer pass: kill-search and technical check of the invented algorithms (2026-10-23)

**Scope:** `docs/EXPERIMENT_CATALOG.md` Part A, A1–A16, as specified by the builder session.
**Method:**
- Technical errors were checked by hand [I].
- Novelty was checked with targeted web searches this pass (ABFT × optimizer state; bipower/MedRV × Adam;
  anti-windup/PID × optimizers; deferred/reversible moment edits), plus the abstracts of Liu & Li 2608.19762 and
  Guo 2608.24593.
- This is not a systematic review. OpenReview and OpenAlex were not queried.

**Nothing here has run. Everything is queued for the AMD MI300X.**

---

## 0 · Headline advice: three contributions, not sixteen names

Sixteen named algorithms read as name inflation to an AC. A1, A2, A4, A5, A13 and A15 are one idea: **a per-coordinate,
reversible log of what each flagged gradient wrote into m and v, with a deferred decision** (insert later, delete
later, or project the weight side back out). Present them as **one algorithm with operators and ablations**. A working
name is *Reversible Moment Log (RML)*. That gives three separable contributions:

| Contribution | Members | Paper |
|---|---|---|
| **RML**: reversible, deferred per-coordinate moment edits | A1 REC (delete-later), A2 QV (insert-later), A5 CWB (weight side), A4 policy, A15 β1-drop ablation, A13 gate | P1 (C-4) |
| **BiH-Adam**: a jump-robust optimizer | A3 (+ A16 as its monitor) | P2 / separate optimizer paper |
| **MomentABFT**: integrity of stored optimizer state | A7 (+ the honest-Adam invariant) | P2/P3, or a systems venue (SC, MLSys) |

The measurement tools (A6, A11, A12) and repairs (A8, A9) stay as *instruments*, not inventions. A10 and A14 go to
the backlog (§A10, §A14).

---

## 1 · Verdicts

| ID | Verdict | Nearest prior work | Safe wording | Technical issue (must fix) |
|---|---|---|---|---|
| **A7 MomentABFT** | **NEW** (medium-low confidence: no ABFT for optimizer state found in 2 searches) | ABFT for GEMM (Huang & Abraham 1984); V-ABFT 2602.08043; "Calibrating the Checksum" 2610.02240; DRS ABFT 2307.10244; ATTNChecker | "To our knowledge, the first checksum-based detect-locate-correct scheme for stored Adam moments" | **Yes, three (§A7)** |
| **A1 REC** | **PARTIAL**, leaning NEW as a mechanism (low confidence) | SPAM 2501.06842 (same g²/v detector; irreversible clip + momentum reset); SWIFT 2302.06173 (one-step inversion); OLMo skip-step | "A reversible, deferred alternative to spike clipping: the flagged step's echo is cancelled in the update while the stored moments stay intact" | **Yes: "exact" is open-loop only; the persistence test is confounded (§A1)** |
| **A2 QV** | **PARTIAL** | SPAM; OLMo/OPEN-1B skip-step (whole step, irreversible); AdaGC | "Per-coordinate skip-step with a deferred commit/discard and exact forward compensation of the moments" | **Yes: momentum imputation is ballistic (§A2)** |
| **A3 BiH-Adam** | **Bipower v: NEW** (low confidence, ~3 searches incl. N6). **Huber clip: TAKEN** (it *is* SPAM's clip, sign·√(θv)) | TAdam 2003.00179; AdaTerm; AdaShift; SPAM | "The first use of a jump-robust realized-variance estimator (bipower/MedRV) as Adam's second moment" | **Yes: biased under signal; "no v-stall" over-claims (§A3)** |
| **A10 IAW-Adam** | **PARTIAL** | SPI-Optimizer 1812.11305 (integral-*separated* PI optimizer; integral separation is a classic anti-windup technique); PID optimizers (An et al. CVPR 2018; Dai 2023, IET); StableAdamW | "Back-calculation anti-windup on a leaky per-coordinate update integral" | **Yes: can't separate echo from normal drift on high-SNR coordinates (§A10). Backlog** |
| **A6 LRA** | **TAKEN as a method** | Liu & Li 2608.19762: linearizes AdamW with state (θ, m, v) into "a signed response operator that maps a localized gradient perturbation to its future loss effects", validated on paired trajectories differing in one update. Guo 2608.24593 differentiates 8-step AdamW through the full state | Don't name it as ours. Use "Liu & Li's response operator" as a fork-free *baseline estimator* of the mediation curve, scored against our bitwise factorial | Global-norm clip is active exactly at the spike: the JVP must include the clip's Jacobian (piecewise). Math SDPA only |
| A4 SAMS | PARTIAL (trivial composition) | — | An ablation/policy inside RML, not a named algorithm | Open-loop and dominated-coordinate only; r^τ is an upper bound on the remaining *weight* share |
| A5 CWB | PARTIAL (already logged) | Amnesiac ML (Graves et al., AAAI 2021); SWIFT; DeltaGrad 2006.14755 | "Echo-aware amnesiac subtraction for Adam" | **The CPU test is self-contradictory (§A5)** |
| A8 SCR | PARTIAL (as logged) | Biswas 2026 [UNV]; GI-Adam 2406.09405; Bai 2506.04805 | as in PLAN_V8 §5 K6 | none new |
| A9 PAR | PARTIAL; **depends on Biswas, which is still [UNV]** | Biswas | Keep exploratory until Biswas is read in full | The 0.63 threshold comes from an unread paper |
| A11 GT-Oracle | PARTIAL | ACDC 2304.14997; ddmin; group testing | "a causal group-testing oracle over optimizer-state groups" | none |
| A12 FTA | PARTIAL | Melis 1805.04049 | as logged | none |
| A13 DT-Gate | PARTIAL | Kourkoutas-β 2508.12996 (adaptive β2); AdaShift | fold into RML as the detector variant | β2_eff = 1 also freezes v's ordinary decay; state that it's deliberate |
| A14 Sketch-REC | **PARTIAL, near TAKEN** for the sketching part | **Spring et al., "Compressing Gradient Optimizers via Count-Sketches", ICML 2019 (arXiv 1902.00179)**: count-sketch storage of Adam/momentum state | Cite Spring et al. and claim only the echo-register use | Backlog |
| A15 β1 drop | PARTIAL | SPAM (global momentum reset); adaptive-β1 schemes | An RML ablation | Lowering β1 also raises the intake weight (1−β1′) of the next ordinary gradients, so noise rises during the drop |
| A16 Shadow-v | PARTIAL (estimator use new) | finance BV/MedRV; TAdam | as logged | Same bias caveat as A3 |

---

## 2 · Technical notes

### A7 MomentABFT
1. **v's harm is relative; a linear checksum is absolute.** A down-flip of a small v_ij (rare-token wte rows, tiny
   gradients) changes v_ij by orders of magnitude, so the update m/√v explodes. But the *absolute* change sits below
   the √n·eps·‖v‖ tolerance, so the checksum misses exactly the most dangerous faults. **Fix:** pair the checksum
   with an elementwise *relative* check. The honest-Adam invariant |m_ij| ≤ C√v_ij (C = 1.165 at (0.9, 0.95))
   catches v down-flips wherever m_ij isn't tiny. A v floor tracked as β2^k·(min v) catches the rest. Absolute
   checksums handle m and v up-flips; the invariant handles v down-flips.
2. **"Detects 100% of exponent flips" over-claims.** A calibrated one-sided ABFT on bf16 GEMM caught 304/400 (76%)
   single exponent-bit flips and 0/600 low-mantissa flips (2610.02240). Pre-register: "detects every flip whose
   induced error exceeds τ_round; undetected flips are bounded by τ_round in m, and by the invariant in v."
3. **Scope is memory faults in stored state between steps.** If SDC corrupts g itself (`sdc_grad`), the checksum is
   updated from the same corrupted g and stays consistent, so it isn't detected. That's what the undo/CWB path is
   for. Also, a flip in m that sits K steps before the check has already moved w; correcting m doesn't undo that
   drift, so pair it with CWB.
4. Use deterministic fp64 reductions for g·1 and 1ᵀg, or the checksums will disagree across forks. A sign-bit flip in
   v produces a NaN under √; that check is trivial and should come first.

### A1 REC
1. **"Exact" is open-loop.** With c = 1, m′ equals Adam's m with g_s = 0 *given the realized later gradients*. The
   true counterfactual's later gradients differ, because w differs. Wording: "exact in moment space for the realized
   gradient sequence". The gap to the skip fork is the closed-loop error (the same estimand as CWB).
2. **REC never removes the spike step's own update** (direct effect) unless it acts at step s itself. That's fine,
   because it targets the indirect share, but say so. Combine with CWB for the full retroactive skip.
3. **The persistence test is confounded; this is the most important fix.** After a real spike, the *training* gradient
   stays elevated for several steps *because the weights were damaged*. So "g² stays high for L steps → level shift →
   c → 0" will cancel REC in exactly the damaging cases. **Fix:** run the persistence test on the data side, not on
   g². Use the gradient of a *fresh probe batch* at the current w (disjoint from Δ-eval batches), or the W2
   persistence regression on probe gradients. Alternatively, test whether the flagged *batch's* loss is an outlier
   under the current model.

### A2 QV
1. **Imputing g̃ = m̂ is ballistic.** With g̃ = m̂, m converges to m̂ and the coordinate moves at constant velocity
   m̂/√v̂ for L steps. Near a minimum (the true gradient reverses), that overshoots, and with many quarantined
   coordinates it can destabilize. Safer default: g̃ = 0 into m (decay only: per-coordinate skip-step) and
   g̃² = v̂ into v (keep the scale). Keep m̂-imputation as an ablation.
2. m̂ is bias-corrected but m is not. Impute with m_{t−1}/(1−β1^{t−1}) consistently, or with the raw-scale
   equivalent, and state which.
3. The CPU test must say "commit reproduces Adam's **moments** at t′ given the realized gradients". The trajectory
   differs, because w moved with imputed updates.
4. Memory is O(|F|·L) pending entries. Cap L.

### A3 BiH-Adam
1. **The bipower scale is unbiased only for zero-mean, independent, Gaussian gradients.** With a persistent signal μ,
   (π/2)|g_t||g_{t−1}| → (π/2)μ² in the high-SNR limit, a 1.57× over-estimate. So the effective step shrinks by up to
   1/√1.57 ≈ 0.80, depending on each coordinate's SNR. That is a silent per-coordinate LR change versus Adam. Heavy
   tails and the lag-1 autocorrelation that Molybog reports also shift the constant. Pre-register "LR re-tuned per
   optimizer", and report s/v by SNR bin.
2. **"No v-stall" over-claims.** An isolated jump enters two products, (t, t−1) and (t+1, t), each ≈ G·ḡ. The excess
   is ≈ (1−β2)(π/2)·2Gḡ, against Adam's (1−β2)G². At G/ḡ = 100 and β2 = 0.95, the stall factor (√ of the inflated
   scale over the calm one) is √(1 + 15.7) ≈ 4.1 for BiH vs √(1 + 500) ≈ 22 for Adam. Wording: "stall reduced from
   O(G²) to O(Gḡ)", not "no stall".
3. **The Huber clip at c√ŝ is SPAM's clip** (g → sign(g)√(θ v) when g² > θ v). Claim only the bipower/MedRV second
   moment, and cite SPAM for the clip.
4. Memory: bipower stores |g_{t−1}| (+1 full tensor); MedV stores two lags (+2). Report it next to Adam-mini-style
   per-row variants.

### A5 CWB
**The CPU test is self-contradictory.** "On a quadratic with gradients fixed in advance": on a quadratic the
gradients depend on w, so they can't be fixed in advance.
- With an exogenous gradient sequence, CWB equals the skip trajectory exactly. That is the identity test.
- On a quadratic (closed loop), CWB is approximate with error O(η‖H‖·‖A‖). That is the error-growth test.

Write both. The (1 − η_kλ) factor in the accumulator is right for decoupled weight decay under a schedule.

### A10 IAW-Adam: an issue that reaches K7
The saturating drift D·η = 5.84η is **small per coordinate**. A coordinate with a consistent ordinary gradient moves
≈ 1/(1−γ) ≈ 13η over the same window. So a per-coordinate integral threshold can't separate the spike echo from
ordinary progress on high-SNR coordinates. It can only fire on noise-dominated ones, where calm |I| ~ √13·η.

Spike damage is **collective**: many dominated coordinates move together in the spike gradient's direction. A
per-coordinate magnitude isn't the right harm measure. **Implication for PLAN_V8 K7/H10 framing:** measure echo harm
as the projection of the echo drift onto the loss gradient (or Bai's top Ĥ directions), not as per-coordinate |Δw|.
Move A10 to the backlog, and stratify any test by SNR bin.

---

## 3 · Kill-search log (this pass)

| Query theme | Result |
|---|---|
| ABFT / checksum × optimizer state, Adam, momentum, training soft errors | Nothing on optimizer state. ABFT-for-GEMM and recommendation-model papers only. 2610.02240 gives detection-rate evidence (76% of exponent flips, 0% low-mantissa) |
| Bipower / MedRV × Adam second moment | Nothing combining them. TAdam is the closest robust Adam (it robustifies momentum) |
| Anti-windup / PID / integral × optimizers | SPI-Optimizer (integral separation), the PID optimizer lineage. No leaky-integral back-calculation on Adam found |
| Deferred / reversible / retroactive per-coordinate moment edits | Nothing. AdaGC and SPAM are preventive or irreversible |
| Linearized response through (θ, m, v) | **Liu & Li 2608.19762 owns it** (→ A6 TAKEN as a method) |
| Count-sketch optimizer state | Spring, Kyrillidis, Mohan & Shrivastava, arXiv 1902.00179 [V]: "a linear sketch to compress the auxiliary variables", including Adam in the embedding and softmax layers. Venue: I believe ICML 2019 (PMLR 97), but the arXiv page only says "initially submitted to WWW 2019", so verify before citing a venue |

Sources: [V-ABFT](https://arxiv.org/html/2602.08043v1) · [Calibrating the Checksum](https://arxiv.org/abs/2610.02240) ·
[DRS ABFT](https://arxiv.org/pdf/2307.10244) · [TAdam](https://arxiv.org/pdf/2003.00179) ·
[SPI-Optimizer](https://arxiv.org/pdf/1812.11305) ·
[PID adaptive optimizer (Dai 2023)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cth2.12404) ·
[SPAM](https://arxiv.org/pdf/2501.06842) · [AdaGC](https://arxiv.org/html/2502.11034v3) ·
[Liu & Li](https://arxiv.org/abs/2608.19762) · [Guo](https://arxiv.org/abs/2608.24593)
