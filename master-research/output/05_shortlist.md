# Phase 3E · Shortlist: one primary direction plus two backups (2026-10-22)

> **v2 (2026-10-22):** superseded by `PLAN_V8.md` §2/§5.
> - Say "**state-only retroactive deletion**" (cite SWIFT), not "exact undo".
> - Say "**stability-certified** restoration" (Crab owns "certified recovery").
> - Marin 32B is a counter-case, not support.

All three keep the main idea (localize and repair poisoned Adam state) and add one separable
contribution. Numbers in brackets are idea IDs from `04_ideas_ranked.md`.

---

## PRIMARY — "Where does spike damage live in Adam, and how do we undo it?"

**Two-sentence pitch.** Loss-spike fixes (clip, skip, reset, rollback) treat the optimizer as a black box,
so nobody knows how much of a spike's lasting damage is carried by the weights and how much by Adam's m
and v. Using bitwise-deterministic counterfactual forks, we measure that split over time, localize the
carrying coordinates with a causal oracle and a cheap signed predictor, and repair only those coordinates
with an exact "undo" of the spike's contribution and a stability-certified restoration.

**Contributions (three, sized to fit 8 pages):**
1. **Measurement (headline).**
   - A damage-migration curve: the share of end-of-schedule held-out Δ carried by w, m and v as a function
     of time since the spike, from a 2³ cross-world factorial [1].
   - A poisoned-state transplant with a dose-response [2].
   - Placebo replicates [4] and hybrid-feasibility checks [10].
   - Run on a warmup+cosine proxy and a middle rung [13], with and without QK-Norm [7, 55].
2. **Localization.**
   - A causal oracle via adaptive group testing at tensor, row and head granularity [6].
   - A cheap signed predictor: AtP-state in log-v plus the two-sided innovation test [11].
   - Scored on the decoy-augmented planted benchmark [3] and on induced and natural spikes.
3. **Repair.**
   - Exact undo log, a state-only retroactive skip [5].
   - Signed, support-limited, stability-certified restoration [12].
   - Compared against the strong baseline set [8] on gap closed G, with cost and edit-size Pareto
     curves [27, 28] and the statistics in [9].

**Second contribution:** SpikeState, a released set of spike instances with (w, m, v, v̄) snapshots, clean
twins and oracle maps [14]. It goes in the appendix or a companion D&B submission.

**Pivot insurance [51].** If the (m, v) share is ≈ 0 beyond about 50 steps, the paper becomes "spike damage
is weight-carried after ~k steps; state repair helps only inside that window", with the undo log as the
in-window remedy. That is still a measured, useful result.

**Scores:**

| N | F | I | R | V | D |
|---|---|---|---|---|---|
| 4 | 4 | 5 | 4 | 4 | 4 |

The combination is new even though several pieces have precedent (see `03_novelty_matrix.md` v2).

**Strongest objection:** scale (F1). An 11M-parameter proxy, even made realistic, is small for ICML.
- Answer: the middle rung, n ≥ 3 at 124M on the primary contrast if the follow-on hours land, and OPEN-1B
  offline if its m/v are released.
- State it as a limitation, not a hidden weakness.

**Cost:**
- About 8–14 P100-equivalent h at proxy plus about 6–10 MI300X-h at 124M. That sits inside the ~15 h ask
  only if the MI300X runs at least 2× a P100 on our shapes (unmeasured).
- About 120–160 person-hours.

---

## BACKUP 1 — "One quantity localizes and repairs" (theory-tied)

**Pitch.** Split Bai et al.'s gradient-direction preconditioned curvature over groups of optimizer state.
That decomposition names the culprit coordinates and gives the smallest v change that restores stability,
via the LMI repair [16, 17, 12]. It comes with a falsifiable certificate: the repair fails exactly when the
untouched bulk is already unstable.

**When to switch:** if AtP-state fails against the oracle (ρ < 0.5), or if the team wants a stronger theory
section.

**Scores:**

| N | F | I | R | V | D |
|---|---|---|---|---|---|
| 4 | 4 | 4 | 2 | 4 | 3 |

**Strongest objection:** superquadratic geometry (Tang 2026) and spurious eigenspace alignment. The
quadratic picture may not hold at spike scale.

**Cost:** about 0.5 GPU-h proxy plus about 1 GPU-h per 124M instance; about 35 p-h.

---

## BACKUP 2 — "Fork-free hedge": signed innovation detection with guarantees, plus an online undo-log airbag

**Pitch.** Adam's own normalized innovations (g² over the prior v), tested in both tails, detect both kinds
of v-poisoning. Combined with a conformal run-level false-alarm bound and e-detectors, they give lead time
on decay-stage spikes. A drop-in wrapper then undoes the flagged window's contribution to m and v online
[60, 61, 5, 21].

**When to switch:** if ROCm can't reach Δ = 0 and the team won't ship a tolerance-only causal paper (M12).
Detection and the undo log need no bitwise forks; their evaluation uses statistical replication.

**Scores:**

| N | F | I | R | V | D |
|---|---|---|---|---|---|
| 3 | 5 | 4 | 4 | 4 | 4 |

**Strongest objection:** the novelty of a detector is moderate (ZClip, OLMo, SPAM, TIC already exist). The
guarantee and the signed lower tail have to carry it.

**Cost:** CPU for detection plus about 2–4 GPU-h for evaluation; about 60 p-h.

---

## Recommendation
**Go with the PRIMARY, and build BACKUP 2's detector as a side product.** It is cheap, it uses telemetry
`capture.py` already logs, and it protects us if ROCm determinism fails. Keep BACKUP 1's λ-decomposition as
one of the three localizer variants V7 already budgets.

The first two weeks' de-risking experiment (the minimal viable experiment, Phase 5) tests the headline: the
damage-migration curve on a dozen proxy instances.
