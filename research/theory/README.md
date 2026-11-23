# Theory

Notes backing the two contributions.

- **C1 — the instrument (localizer).** Why SNR of the Adam update `m / (sqrt(v)+eps)` and local
  curvature localize the poison: what a spike does to the second-moment estimate, and why the
  implicated directions are recoverable. TODO.
- **C2 — attribution science (repair).** The counterfactual-fork argument: repairing exactly the
  localized groups is *necessary and sufficient* for recovery, vs random/global controls. TODO.

References and derivations go here as the empirics land.

## R-F2: the m+v coupling law (PLAN_V8 Appendix A.3; written 2026-11-21)

**Status.** A frozen, exploratory prediction. The code is `coupling.py`; `tests/test_coupling.py` runs in CI, CPU
only. It is scored on runs prereg-v3 already plans (E1, E2 and the β1 arms), so it needs no extra GPU. It is not in
prereg-v3's tables. Adding it as a secondary row before the tag is the team's decision. The function and the scoring
rule below are fixed by the commit that adds them, which lands before any prereg-v3 run, and they are not changed
after data arrives. Nothing here was fitted to prereg-v2's seed 0, which logged no dose x̄ to fit against.

### The question
prereg-v3 runs four arms at each dose: none, undo-m, undo-v and undo-m+v. It predicts that undoing v alone is worse
than doing nothing (P2), and that v cancels most of m's damage, with the protection share π = −I/E_m above 0.5 and
rising with dose (P3). R-F2 predicts π, and the ratio R_v = Y(undo-v)/Y(undo-m+v), from β1, β2, the undo lag, the
steps left in the run and the logged dose x̄. It has no fitted parameter.

### The law, on one coordinate
A spike at step s adds δ to the gradient. Clean v sits at v̄. The spike writes (1−β1)δ into m and inflates v by a
factor 1 + x, where x = (1−β2)·dv/v̄ is the dose meter's x. j steps later, while v still carries the spike, the
spike's share of the update is −η·z·(1−β1)·β1^j·q_j, with z = δ/√v̄ and q_j = (1 + x·β2^j)^(−1/2). Undoing v
removes the q_j damping but leaves m's echo running. Undoing at lag τ fires at the top of step s + 1 + τ, as
`undo.undo_step` does. Up to the end of the run (j ≤ H), the spike moves the coordinate by η·z times:

| Arm | Displacement |
|---|---|
| undo-m+v (and undo-m, along δ) | A_mv = Σ_{j≤τ} (1−β1) β1^j q_j |
| undo-v | A_v = A_mv + β1^(τ+1) − β1^(H+1) |
| none | A_none = Σ_{j≤H} (1−β1) β1^j q_j |

Take the loss as quadratic in the displacement. That is prereg-v3's own working model, since P4 has m's damage
growing like dose squared. Then Y(arm) ∝ A², and:

- R_v = (A_v / A_mv)²
- π = 1 − (A_none² − A_mv²) / (A_v² − A_mv²)

At τ = 0 and a long run, R_v = (1 + k·√(1+x))², with k = β1/(1−β1). As x → ∞, x·(1−π) → K, where
K = ((1−β1)/β1)²·((1−r)^(−2) − 1) and r = β1/√β2 (the r-law, PLAN_V8 §6.3 item 0). `predict` computes both for
finite H; `pi_gap_limit` computes K.

### Assumptions (each is a way the law can fail)
- **A1, open loop.** The gradients after the spike do not depend on the displacement. The error is first order in
  η‖H‖·A. CWB's test (b) measures the same kind of error.
- **A2, stationary v.** Clean v is close to v̄ over the run's remaining steps.
- **A3, dominated coordinates.** The spike's gradient is much larger than the clean one. Keeping v inflated also
  slows the clean updates (the v-stall). The stall's square cancels exactly in I. Its cross term with the spike's echo
  is O(1/z) = O(√((1−β2)/x)) relative to the echo.
- **A4, one effective coordinate.** The loss is quadratic near the clean path. Restoring is the same in every arm,
  which needs the curvature timescale to be long against 1/(1−β1). Curvature is uncorrelated with dose across
  coordinates, so the |u|-weighted x̄ stands in for the whole set. If P4(a) fails (b(E_m) outside [1.5, 2.5]), the
  quadratic model fails, and R-F2's score is reported as "assumption failed", not as a pass or a fail.
- **A5, ε ≪ √v.** Bias correction is the same in every arm, and x does not depend on it.
- **The meter's x.** The meter uses v before the spike step, so the law's 1 + x is (β2 + x̄)/(β2 + x̄/c²). That
  is x̄ − (1−β2) when G² ≈ v. The law is evaluated at the logged x̄ as it stands. The offset is at most 0.05 and
  moves π by less than 0.01 for x̄ ≥ 3 (inferred from the slope of the table below).

### Predictions
**π at the two planned doses** (τ = 0, H = 199, β2 = 0.95). These were computed by hand; CI pins the first column
to ±0.01. c = 10 and 30 give x ≈ 5 and 45 when G² ≈ v̄. The scored values use the logged x̄.

| β1 | x = 5 | x = 45 | K |
|---|---|---|---|
| 0.8 | 0.75 | 0.96 | 1.88 |
| 0.9 | 0.73 | 0.96 | 2.09 |
| 0.95 | 0.65 | 0.93 | 4.32 |

At β1 = 0.9: K = 1.34 at β2 = 0.99 and 1.23 at β2 = 0.999. **R_v at x = 5** is about 531 for β1 = 0.9 (from
(1 + 9√6)²). It is larger at higher β1 and larger x.

1. **π rises with dose and approaches 1.** This agrees with P3. 1 − π falls roughly like 1/x̄.
2. **β1 order (S2).** The order at c = 10 and step 1800 is π(0.95) < π(0.9) ≈ π(0.8), so the law predicts what
   S2's hand theory predicts. The 0.8 and 0.9 values differ by less than 0.05, too little to call. At c = 30 all the
   gaps are below 0.05.
3. **β2 (S1).** At large x̄, the β2 rows one v-memory after the spike give x̄·(1 − π) of about 1.77 (β2 = 0.95,
   H = 19), 1.34 (0.99, H = 99) and 1.23 (0.999, H = 999). These are the large-x forms; `predict` gives the exact
   values. So S1's 0.15 tolerance should hold once x̄ is above about 4, and can fail below that.
4. **D is tiny.** Y(undo-m+v) is R_v times smaller than Y(undo-v), about 500 times at β1 = 0.9 and x = 5, so D will
   likely sit inside the chaos floor. The law also predicts b(E_m) → 2, in line with P4(a), and b(D) → 0 because D
   saturates: b(D) is about 0.15 between c = 10 and 30 at β1 = 0.9.

### Scoring (frozen at this commit)
- **Cells.** E1 at steps 1000 and 1800 (β = 0.9/0.95, H = 999 and 199), the β1 arms (β1 = 0.8 and 0.95, H = 199)
  and the four E2 rows (H = 19, 99, 999, 999). In every cell, c ∈ {10, 30} and τ = 0. That is 16 cells.
- **Per seed.** Compute π_meas as prereg-v3 does; it is undefined when E_m ≤ 0.002 nats. Set π_pred =
  `predict(x̄, β1, β2, H)["pi"]` at that seed's logged x̄ for that cell's Px arm. A cell with fewer than 2 seeds
  that have π_meas is "not testable".
- **L-a (level).** In each testable cell, the mean over seeds of |π_meas − π_pred| is at most 0.15. Report k of n
  cells. The law "holds" only if every testable cell passes.
- **L-b (order).** For each pair of β1 cells, and each pair of β2 rows, at the same c: if the seed-mean π_pred values
  differ by at least 0.05, the seed-mean π_meas values fall in the same order. Pairs closer than 0.05 have no order
  prediction.
- **L-c (direction of failure).** Keeping v inflated also slows the restoring force in the none and undo-m arms.
  By a persistence ratio ρ ≥ 1, that gives π = 1 − ρ·(A_none² − A_mv²)/(A_v² − A_mv²) ≤ π_pred. So a miss with
  π_meas < π_pred is the known bias, while π_meas > π_pred by more than 0.15 contradicts the mechanism.
- **R_v check (one-sided).** Let σ be the seed's chaos floor at the end of the run: the SD over its 8 placebo forks,
  as in v2. If D > 2σ, the law fails this check when Y(undo-v)/D is below R_v_pred/2 or above 2·R_v_pred. If D ≤ 2σ,
  it fails when Y(undo-v)/(2σ) is above 2·R_v_pred, which a predicted R_v above 100 (every cell at x̄ ≥ 5) makes
  very unlikely. So
  this check has teeth only if D turns out to be resolvable.
- **Reporting.** Results go out as k of n cells and k of 3 seeds, exploratory, with no p-values. If prereg-v3 logs
  quantiles of x (open decision 6), π_pred is also shown at the 10% and 90% quantiles to show the spread. The score
  itself uses x̄.

### Prior art (searched 2026-11-21)
No source found derives the undo-v / undo-m+v ratio or the protection share. This is bounded by five standard web
searches and the reads listed below.

| Work | Overlap | Verdict |
|---|---|---|
| Liu & Li 2026, [2608.19762](https://arxiv.org/abs/2608.19762) | A linearised (θ, m, v) response to one minibatch shock, plus hybrid-state channel tests. Their operator puts the shock's own (1−β2)ξ² write to v into the quadratic remainder (§3.3, Eq. 1). That term is x, the quantity this law is built on. They say β1 and β2 "do not by themselves determine" the coupled gain (§3.4). No v-only versus m+v comparison appears in the parts read. Read: abstract and §3.3–3.4; the rest is a summary. | PARTIAL |
| Patitucci & Mokhtari 2026, [2602.10204](https://arxiv.org/abs/2602.10204) | MVN-Grad's "temporal coupling": the normaliser dips while momentum keeps pushing. This is the same m/v timing mismatch, which here comes from undoing v. Thm 3.3 analyses a single spike (references.bib, from the reviewer session's full-text read). | PARTIAL, the closest mechanism |
| Bai et al., ICML 2026, [2506.04805](https://arxiv.org/abs/2506.04805) | v as the trigger of spikes, through preconditioned stability (Prop. 4.2). It asks what causes spikes, not how to repair them. | PARTIAL, a neighbour |
| Huang et al., ICLR 2025, SPAM, [2501.06842](https://arxiv.org/abs/2501.06842) | Resets m and v together. It does not analyse v-only against m+v. | PARTIAL |
| Zhang et al. 2026, [2603.02092](https://arxiv.org/abs/2603.02092) | Adam converges when β1 < √β2. This is the same r = β1/√β2 that sets K here. Read: abstract only. | Context, not the claim |
