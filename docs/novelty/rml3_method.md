# RML-3: three additions to the core method

*Written 2026-11-12. A proposal for team sign-off. Nothing here has been run, and no number below is a result
unless it cites the seed-0 files. The survey behind it is `technique_map.md` (75 techniques, 15 combinations).
Labels: **[V]** id and title checked today; **[abs]** abstract read today; **[snip]** search snippet only;
**[L]** in `gap_ledger.md`; **[I]** my inference; **[S]** speculative; **UNVERIFIED** could not be checked.*

The brief names "V6". `PLAN_V8.md` governs and keeps V6's four components: **C1** the fork-and-compare instrument,
**C2** the localizer, **C3** the repair, and the theory. Each addition below names the component it extends.

---

## 0. Summary

| | What it is | Borrowed from | Extends | Novelty after today's searches |
|---|---|---|---|---|
| **K1 Appeal** | Withhold the gradient share of flagged examples, keep a record, and add it back into m and v exactly if later batches agree with it | private training (ghost norms), data valuation (gradient alignment), databases (undo log) | C3 repair, the detector | The combination was **not found**. Its scoring machinery is taken |
| **K2 Tournament** | After a spike, replay a short menu of remedies from a snapshot on the same data and keep the best. "Do nothing" is always on the menu | population-based training, model-predictive control, speculative execution | C1 becomes a controller; C3 | **Partial**: one 2026 paper chooses loss schedules by 8-step rollouts; another supervises training with accept-or-rollback and a bound on degradation |
| **K3 Ceiling** | The gradient of final loss with respect to (w, m, v) bounds what any state edit can gain; its predictions plus a few exact forks give valid intervals | metagradients, prediction-powered inference | C2 localizer, theory | **Partial**: the gradient exists in two 2026 papers; the ceiling and the intervals were not found |

How the three fit together: K1 is the repair that runs all the time. K2 is the safety net that runs when a spike
gets through anyway, and K1 is one item on its menu. K3 is offline science that says whether any repair has room to
do better than the menu.

**What changes from RML-1 and RML-2.**

| | Receipt | Resolution | Decision | Safety |
|---|---|---|---|---|
| RML-1 (in the repo) | spiked gradient minus the clean twin's | the whole step | none: it is an oracle for measurement | none needed |
| RML-2 (`method.md`) | innovation, no twin needed | coordinates | forecast arms it; later gradients give a verdict | none |
| **RML-3** | the flagged examples' share | **examples**, then coordinates | later batches, calibrated against ordinary examples | **the tournament** |

RML-1 stays unchanged as the measuring instrument. RML-2's forecast idea (a tiny model that scores risk before the
spike) becomes a layer of three small predictors: when to arm, which rows look odd, and which remedy to try first.
Its design and prior-art check are in `appeal_build_plan.md` §10. Forecasting itself is already published and is
not claimed.

**An honest limit before the details.** On the proxy at β2 = 0.95 a real spike leaves almost nothing to repair
(`docs/prereg_v2_results.md`). So none of the tests below is "lower final loss than skip-step". They test
discrimination (junk dropped, real data kept), safety (never worse than doing nothing) and headroom (how much any
repair could gain).

---

## 1. K1 Appeal: reversible per-example trimming  **[NEW, extends C3 and the detector]**

### 1.1 Idea
A bad batch usually holds a few bad examples. Skip-step throws away the good ones with them. Clipping shrinks the
whole gradient and still feeds the bad share into m and v. Trimming the bad examples is the obvious fix, and it has
the obvious flaw: a hard but real example looks like a bad one, and once trimmed it is gone.

Appeal makes the trim reversible. The withheld share is recorded. For the next K steps the later batches act as the
jury: if stepping along the withheld gradient would have lowered their loss about as much as an ordinary example
does, the share is added back. Because m and v are running averages, adding it back K steps late is exact for the
state (the same identity RML uses to remove a step, run in the other direction).

### 1.2 Inputs, what is logged, what is computed
- **Inputs:** the batch (B rows), per-row losses ℓ_i, the final gradient ĝ and the pre-update m, v at each step.
- **Flag.** Score each row by a robust z-score of its loss against the median and spread of its own batch. A fork
  starts at the spike step, so there is no history to score against. Rows above a threshold q are flagged. If more
  than half the batch is flagged it is a whole-batch event, and Appeal leaves it to skip-step or the tournament.
  Later refinements: a threshold set from the ranks of recent scores, so that a chosen fraction α of ordinary rows
  is flagged (conformal calibration [V, 2107.07511]), and per-row gradient norms from the ghost-norm trick
  [V, 1510.01799; 2110.05679] as a stronger score at higher cost.
- **Record** (one per flagged step): the flagged rows' mean gradient g_F; the gradient ĝ_0 that Adam was actually
  fed; the clip factor; the number of flagged rows n. Two vectors on the support. At scale g_F can be stored as per-token factors,
  which is where the low-rank idea in V6 fits (K9 in the map).
- **Shadow records.** At the same step, also record the mean gradient of M random groups of kept rows (M = 8), each
  the size of the flagged group, without withholding anything. They share the step, the weights, the group size
  and the later batches with the flagged record, so they are the matched reference for the verdict.
- **Evidence.** For each open record R at step s + k, for k = 1 to K:
  T_R += ⟨ĝ_{s+k}, g_R⟩_P / (‖ĝ_{s+k}‖_P · ‖g_R‖_P), with P = 1/(√v + ε) taken from the pre-spike v.
  This is a cosine in Adam's own scaling: the first-order drop in the current batch's loss per unit of step length
  along the recorded gradient. A cosine is used because hard rows have large gradients, and a score divided by the
  record's squared norm would penalise them for size alone. It is the alignment score of Learning to Reweight
  Examples [L] and of in-run data valuation [V, 2406.11011], with later training batches in place of a clean
  validation set.

### 1.3 The rule
```
pre_step(step, ctx):                          # top of the step; may replace the batch
    x, y = the aligned batch for this step
    ℓ    = per-row loss (one forward pass, no gradients, eval mode so it draws nothing from the RNG)
    F    = rows whose score exceeds q ;  n = |F|
    if F:        g_F = gradient of the mean loss over the rows in F    # torch.autograd.grad; p.grad is untouched
                 y[F] = −1 ;  ctx["batch"] = (x, y)    # the loss ignores target −1 and averages over the kept rows
                 open.append(Record(step, g_F, n))
                 repeat M times:  open.append(Record(step, mean gradient of n random kept rows, n, shadow of F))

pre_opt_step(step, ctx):                      # sees the final gradient ĝ and the pre-update m, v
    for R in open:
        if R.step == step and not R.shadow:   R.g0, R.clip = ĝ, clip factor
        elif R.step < step:                   R.T += <ĝ, R.g>_P / (‖ĝ‖_P ‖R.g‖_P)
    for R in open, not a shadow, with step − R.step == K:
        p = share of R's own shadow records with T below R.T
        φ = clip((p − 0.05) / 0.20, 0, 1)
        ḡ = R.g0 / R.clip                               # the kept rows' mean, before the clip
        ĝ_φ = clip( ((B − n)·ḡ + φ·n·R.g) / (B − n + φ·n) )    # φ = 1 gives the full-batch gradient
        m += β1^(K−1) (1−β1) (ĝ_φ − R.g0)               # the edit fires before step s+K's own update
        v += β2^(K−1) (1−β2) (ĝ_φ² − R.g0²)
        close R and its shadows
```
Three outcomes follow from the rule. A share that scores below every one of its shadow groups is dropped. A share
that ranks like an ordinary group is restored in full, even if its alignment is near zero, because ordinary groups
score near zero too. Anything between is restored in part. With eight shadow groups p moves in steps of 1/8, so φ
is 0, 0.375 or 1. The 0.05 and 0.25 cut-offs are proposals.

Details the code forces (checked in `research/model/nanogpt.py`, `research/harness/trunk.py` and
`research/autopsy/undo.py`; the full list is in `appeal_build_plan.md` §3). The loss is a mean over unmasked
tokens, so masking the flagged rows feeds Adam the mean of the kept rows (a trimmed mean), not a partial sum; the
restore formula weights the two shares by their row counts for that reason. A fork starts at the spike step, so the
reference cannot come from earlier steps; the shadow groups are taken at the flagged step itself. The restore is
the existing RML undo run with the entry (ĝ_0 − ĝ_φ, ĝ_0² − ĝ_φ²). The proxy runs with `grad_accum: 1` and
`dropout: 0.0`; with accumulation the batch override applies to every micro-batch, so Appeal would need a
per-micro-batch hook there.

### 1.4 The experiment inside the fork framework
- **Spike sources.** `repeat_ngram` already replaces a fraction of the rows of a real batch with a junk n-gram
  (`research/spikes/induce.py`, default 25%). A new source, **hard rows**, replaces the same fraction with real rows
  drawn from the top tenth by loss of a sampled pool. Both are flagged by the score; only one should be dropped.
- **Arms.** none; skip-step with oracle timing; oracle row mask (only the junk rows removed: the ceiling); trim
  without appeal (φ = 0 always); delayed keep (φ = 1 always, which prices the K-step delay by itself); Appeal;
  global clip; SPAM; RML at lag 3.
- **Read-outs.** (i) verdicts per row type; (ii) end-of-run loss against the clean twin; (iii) loss on a held-out
  hard set, which measures whether real signal was thrown away; (iv) the cost of Appeal on a stretch with no spike.
- **Design.** Seeds 1 to 7, injection at steps 1000 and 1800, as in prereg-v3. The amplified doses (c = 10, 30)
  apply to the junk rows' share.

### 1.5 What would prove it wrong (thresholds are proposals, to be fixed before running)
| # | Prediction | Wrong if |
|---|---|---|
| A1 | Junk rows are dropped | fewer than 90% get φ ≤ 0.1 |
| A2 | Hard real rows are restored | fewer than 70% get φ ≥ 0.9 |
| A3 | Appeal keeps what trimming loses | on the hard set, Appeal closes less than 80% of the gap between "trim without appeal" and "keep everything" (not testable if that gap is under 0.005 nats) |
| A4 | It costs nothing on clean training | a no-spike stretch with Appeal on ends more than 0.003 nats from the same stretch with it off |
| A5 | At amplified dose it matches skip-step | Y(Appeal) > Y(oracle skip-step) + 0.002 nats, or Y(Appeal) ≥ Y(none) |

### 1.6 Cost
One extra forward pass per step for the row losses in a hooks-only build (about a third of a step [I]); close to
zero if the loop exposes per-row losses, which needs a change in `trunk.train_forward` and a re-run of the `Δ==0`
gate. On a flagged step, one small backward pass for the flagged rows and one per shadow group. Memory: about
eleven gradient-sized vectors for K steps at eight shadow groups. GPU for the experiment: about 1 hour at proxy
scale [estimate].

### 1.7 Closest prior work
| Work | What it has | What it lacks |
|---|---|---|
| GREATS [snip]; OPUS [snip, 2602.05400]; Filter-then-Weight [abs, 2604.00001] | per-example gradient inner products at low cost; Filter-then-Weight scores under the current optimizer state and uses outer-product factors | a validation or target set is required; the choice is final; nothing is withheld and restored |
| Data Shapley in One Training Run [V, 2406.11011] | in-run alignment scores | used after the run; Adam left to future work [L] |
| Dynamic loss-based reweighting [abs, 2502.06733] | online down-weighting by loss in LLM pretraining | no reversal, no optimizer-state edit |
| Ling [snip, 2503.05139] | skips a spiking update and feeds the data again later | unconditional; details UNVERIFIED |
| Gradient agreement filtering [L]; Early Cutting [snip, 2502.08227] | dropping by disagreement; a later model re-judging early picks | irreversible; multi-epoch |
| Swift, the RML undo [L] | exact removal from m and v | removal only, never late admission |
| Roulet et al. [abs, 2510.00236] | cheap per-example statistics inside optimizers | not aimed at outliers or spikes (abstract only) |

---

## 2. K2 Tournament: choosing the remedy by replay  **[NEW, extends C1 and C3]**

### 2.1 Idea
The project's instrument answers "which remedy would have been best" after the fact, by forking. The same machinery
can answer it during the run. Keep a recent snapshot in memory. When the detector fires, restore the snapshot,
replay the next stretch once per remedy on the same data, score each on fresh batches, and continue from the
winner. Because the replay starts before the spike, every remedy acts at the spike step itself, with hindsight.
Because "do nothing" is on the menu, the choice is never worse than doing nothing at the horizon judged.

Seed 0 shows why a fixed rule is not enough. At realistic dose and step 1000, undoing m alone left the loss
0.117 nats above the clean run 19 steps later, ten times the spike's own excess of 0.0115, and 0.005 at the end.
At step 160 the same arm was the best one at +19 steps. No single remedy was safe in all three phases.

### 2.2 The rule
```
ring: an in-memory snapshot of (w, m, v, RNG, step) every N steps
on detector fire at t0, flagged step s:
    for a in menu = [none, skip(s), undo_state(s), appeal(s), band_clamp(t0)]:
        restore(ring)
        run_fork(hooks = a, steps = (t0 − ring.step) + H)          # same batches for every arm
        J[a] = mean loss on the judge batches ;  keep the end state
    best = argmin J ;  if J[none] − J[best] < δ:  best = none       # ties go to doing nothing
    continue from best's end state at step t0 + H
```
- **Judge:** a fixed set of fresh batches, the same for every arm.
- **δ:** twice the spread of the same comparison among placebo forks at that point of training (the floor curve, E3
  in prereg-v3).
- **H:** a multiple of v's memory, 1/(1−β2) steps. Short rollouts are known to favour greedy choices
  [V, 1803.02021], so H is the quantity under test.
- **Extrapolated judge (optional):** fit each arm's excess loss over the rollout and predict its end value, as
  AutoLRS does for candidate learning rates [abs, 2105.10762]. It can be tested offline on the saved horizons.

### 2.3 The experiment
**Offline, no extra GPU.** Every fork in the battery already records the held-out loss at fixed horizons for every
arm. So for each instance: pick the arm that looks best at horizon H, then read what that arm did at the end.
- regret(H) = end damage of the pick − end damage of the best arm;
- harm(H) = end damage of the pick − end damage of "none".

**Pilot on seed 0** (real spike, the four undo arms; held-out loss minus the clean run, nats):

| Step | Arm | +19 | +99 | +199 | End |
|---|---|---|---|---|---|
| 160 | none | 0.0068 | 0.0040 | 0.0038 | 0.0005 |
| 160 | undo-m | 0.0005 | 0.0011 | −0.0006 | 0.0013 |
| 160 | undo-v | 0.0083 | 0.0049 | 0.0041 | 0.0023 |
| 160 | undo-m+v | 0.0016 | 0.0016 | −0.0000 | 0.0021 |
| 1000 | none | 0.0012 | −0.0012 | 0.0016 | −0.0001 |
| 1000 | undo-m | **0.1173** | 0.0235 | 0.0220 | 0.0050 |
| 1000 | undo-v | 0.0016 | −0.0007 | 0.0025 | −0.0004 |
| 1000 | undo-m+v | 0.0024 | 0.0050 | 0.0033 | 0.0026 |
| 1800 | none | 0.0000 | 0.0028 | — | 0.0024 |
| 1800 | undo-m | 0.0008 | 0.0010 | — | 0.0023 |
| 1800 | undo-v | 0.0011 | 0.0025 | — | 0.0040 |
| 1800 | undo-m+v | 0.0004 | 0.0016 | — | 0.0020 |

Source: `sealed.json`, the `c2` entries per horizon. A pick at +19 steps would have chosen undo-m, none and none.
Its regret at the end is 0.0008, 0.0003 and 0.0005 nats (from the unrounded values), and it never chooses the arm
that ended worst. This is one
seed, four arms, and differences mostly inside the scatter between arms; it shows the analysis is possible, not that
the method works. The 0.117 figure is unexplained and should be looked at before anything is built on it.

**Online demonstration.** Three seeds, two phases, the menu above plus two arms known to do harm on seed 0 (v reset
and SPAM), to check that the tournament rejects them. The accept-or-rollback supervisor of Or 2026
[page, 2601.17483] runs beside it as a baseline.

### 2.4 What would prove it wrong
| # | Prediction | Wrong if |
|---|---|---|
| T1 | A rollout of one v-memory picks well | mean regret at H = 20 steps exceeds 0.003 nats over seeds 1 to 7 |
| T2 | It does no harm | the pick ends more than 0.005 nats worse than "none" in more than one of seven seeds, in either phase |
| T3 | It rejects harmful remedies | v reset or SPAM is picked in any event where it ends more than 0.01 nats worse than "none" |

### 2.5 Cost
Per spike: (menu size) × (steps since the snapshot + H). With five arms, N = 50 and H = 20 that is at most 350 steps
per spike. PaLM's manual remedy restarts about 100 steps back and skips 200 to 500 batches [L]. Memory: one extra
copy of (w, m, v) while training. The offline analysis costs nothing; the online demonstration is under 0.2 GPU-hours
at proxy scale [estimate].

### 2.6 Closest prior work
| Work | What it has | What it lacks |
|---|---|---|
| Guo 2026 [abs, 2608.24593] | chooses among candidate interventions by 8-step AdamW rollouts on committed future batches | the candidates are loss schedules; no spike trigger, no remedies, no "do nothing" guarantee, 8 steps only |
| Or 2026 [page, 2601.17483] | a supervisor that accepts or rolls back each update from a snapshot of weights and optimizer state, triggered by a held-out probe loss, with a bound on how far the probe loss can rise | it evaluates only accept and rollback and names richer strategies as future work; remedies are never compared |
| AutoLRS [abs, 2105.10762] | tries each candidate learning rate for a few steps and extrapolates the validation loss | scheduled, learning rates only, no spike trigger |
| Jiang et al. 2026 [abs, 2608.04408] | replays error states through continuation and rollback branches and lets the outcomes steer training | the states are a student's reasoning steps in distillation, not optimizer state |
| Population Based Training [V, 1711.09846] | copy the better member and continue | not triggered by spikes; perturbs hyperparameters |
| K2-V2 [snip, 2512.06201] | a rule: narrow spikes continue, wide ones roll back | no comparison of what each choice would have done |
| Ling [snip]; PaLM [L] | one fixed remedy | no choice |
| Short-horizon bias [V, 1803.02021] | the known failure of short rollouts | not measured for remedies |

---

## 3. K3 Ceiling: how much could any repair gain?  **[NEW, extends C2 and the theory]**

### 3.1 Idea
V6's hypothesis is that damage sits in a few directions and a surgical edit can remove it. Before building a
localizer and a repair, ask what the best possible state edit could gain. The gradient of the final loss with
respect to the state at the repair step, λ = ∂J/∂(w, m, v), answers that to first order:

- the effect of any edit δ is predicted by ⟨λ, δ⟩;
- under a per-coordinate budget |δ_i| ≤ ρ_i, the best first-order gain is Σ_i ρ_i |λ_i|. That is the **ceiling**;
- the scores s_i = λ_i δ_i rank coordinates by how much of the removable damage each carries. This is the
  concentration test for V6's hypothesis, and a yardstick for every cheap localizer, with no Lanczos run.

Predictions are cheap and exact forks are few. Prediction-powered inference [abs, 2301.09633] is built for that
case: many predictions, debiased by a random sample of true labels, give valid confidence intervals.

### 3.2 What is computed
```
forward:  replay steps s+L … T' and store (w, m, v, batch id) at each          # deterministic
backward: λ = ∇ J(x_T') ;  for t = T'−1 … s+L:  λ = (∂F_t/∂x_t)ᵀ λ             # one double-backward per step
outputs:  pred(δ) = <λ, δ> ;  ceiling(ρ) = Σ ρ_i |λ_i| ;  scores s_i = λ_i δ_i
intervals: θ̂ = mean over N patches of pred  +  mean over n random exact forks of (Y − pred)
```
F_t is one AdamW step with the clip. Near-zero v needs the smoothed form √(v + ε²), as noted for AdamW's
augmented-state Jacobian [abs, 2606.30788].

### 3.3 The experiment
On instances from prereg-v3's seeds, at c = 1, 10 and 30:
1. predicted against measured effects for the four undo arms and 20 random per-tensor patches;
2. the ceiling against the gain of the exact undo and of skip-step;
3. one fork that applies the ceiling's own edit, to see whether the bound can be reached;
4. interval coverage: the prediction-powered interval against the brute-force mean from many forks.

The 2 × 2 already says where this must fail: the first-order prediction is additive in m and v, and the measured
interaction at c = 30 is as large as the main effect (`upgraded_plan.md` §2). The size of that mismatch is itself
the read-out: it says up to which dose a linear theory of repair is valid.

### 3.4 What would prove it wrong
| # | Prediction | Wrong if |
|---|---|---|
| C1 | At realistic dose the linear prediction holds | correlation between predicted and measured effects is below 0.8 at c = 1 in phases where the placebo floor is under 1e-5 nats |
| C2 | The intervals are valid | 95% intervals cover the brute-force value in fewer than 90% of resamples |
| C3 | The ceiling can be reached | the fork that applies the ceiling's edit does worse than the exact undo at c = 10 |

### 3.5 Cost
About three step-equivalents per step of horizon for the backward sweep, plus storage of every state on the way. At
proxy scale and a 200-step horizon that is under 0.1 GPU-hours per instance [estimate]. The engineering is the cost:
a hand-checked adjoint of the AdamW step, verified against finite-difference forks. V8 already flags HVP work as its
largest engineering risk.

### 3.6 Closest prior work
| Work | What it has | What it lacks |
|---|---|---|
| Xu 2026 [abs, 2608.20442] | a full-horizon costate through optimizer state that predicts the sign of an effect | trait transfer, not spikes; no ceiling, no best edit |
| Guo 2026 [abs, 2608.24593] | derivatives through model and optimizer state, used to shortlist candidates for exact rollouts | 8 steps; schedules; no intervals |
| Metagradient descent [abs, 2503.13751] | exact gradients through whole runs | taken with respect to data weights and hyperparameters; needs smooth training |
| AtP\* [abs, 2403.00745] | first-order patch estimates with a bound on missed components | no interval on effect sizes |
| Prediction-powered inference [abs, 2301.09633] | the estimator | not found combined with patching (two searches and the independent check) |

---

## 4. Novelty check

Three passes: my own 13 searches and 5 page reads, the arXiv listing for 63 ids, and an independent checker that
was told to assume each idea already exists and made 18 web calls to show it. Its notes
are in `master-research/raw/novelty-2026-11/results/kill_4_rml3.md` (a local file, not in git). A fourth pass on
2026-11-13 (8 searches, 12 abstracts from the arXiv listing, 2 page reads) covered the forecast layer and found
two more neighbours of the tournament (Or 2026, AutoLRS). All reading was at abstract or snippet depth.

| Approach | Verdict | Most damaging prior work | What may still be claimed |
|---|---|---|---|
| K1 Appeal | **Not found as a combination.** Parts are taken | Filter-then-Weight and GREATS (the scoring); Ling (a second chance for skipped data) | "A sample rejection that is revised later, from the training stream itself, and applied exactly to Adam's state." Never "the first per-example method" |
| K2 Tournament | **Partial** | Or 2026 (rollback supervisor with a bound; names richer strategies as future work); Guo 2026 (choice by short rollouts) | "Spike-triggered choice among several remedies by exact replay on the same batches, with a measured short-horizon regret." A bound on degradation is not new |
| Forecast layer (`appeal_build_plan.md` §10) | Forecasting and small-model scoring are **already done**; the remedy policy is **partial**; a forecast-started tournament was **not found** | six forecasters including TIC; RHO-LOSS; Jiang 2026 | only the couplings listed in the build plan §10.2 |
| K3 Ceiling | **Partial** | Xu 2026 and Guo 2026 (the costate itself) | "A first-order ceiling on state repair, and valid intervals for patch effects." The costate is not ours |

What the searches do not cover: full texts; OpenReview submissions from the last two months; code repositories.
"Not found" means not found in these searches.

---

## 5. Additions to the methodology

Section numbers refer to `PLAN_V8.md` §6. **NEW** marks additions. Everything not listed is **unchanged**.

### §6.2 Spike sources
- **Unchanged:** `corrupt_batch`, `repeat_ngram`, `lr_bump`, the planted dose.
- **NEW use:** `repeat_ngram` with a quarter of the rows replaced becomes the test bed for per-row repair, because
  it is the one source where rows differ in quality.
- **NEW:** `hard_rows`, real rows from the top tenth by loss, as the control that should not be dropped.

### §6.3 Measurement instruments
- **Unchanged:** the fork harness, the placebo floor, the RML undo as an oracle.
- **NEW:** per-horizon values kept for every arm in the report (they are already measured).
- **NEW:** the regret and harm of a horizon-H pick (§2.3).
- **NEW:** the costate, its ceiling and its scores (§3), with the finite-difference check as its own gate.

### §6.4 Localization (C2)
- **Unchanged:** the `Localization` schema and the no-HVP localizers.
- **NEW:** every localizer is scored by rank agreement with s_i = λ_i δ_i, and by the share of the ceiling its
  selected set reaches.
- **Proposed change:** Lanczos moves behind this. If the ceiling shows no headroom over the exact undo, C2 needs no
  curvature tool.

### §6.5 Repair (C3)
- **Unchanged:** RML with its variants; the band clamp.
- **NEW:** Appeal (§1) as the always-on operator.
- **NEW:** the tournament (§2) as the outer loop, with every other remedy as a menu item.

### §6.7 Baselines
- **Unchanged:** skip-step (oracle and detector-timed), clip, SPAM, ZClip, AdaGC, rollback. GradientStabilizer is in
  the ledger and not yet in the battery.
- **NEW:** trim without appeal; delayed keep; oracle row mask; the two split SPAM arms from prereg-v3; the
  accept-or-rollback supervisor of Or 2026.

**How each baseline is expected to be beaten, and the test that decides it [I, not results].**

| Baseline | Its limit | The test |
|---|---|---|
| Skip-step | discards the good rows of a mixed batch; cannot act late | A3, A5 |
| Clip family (global clip, ZClip, AdaGC, GradientStabilizer) | shrinks good and bad rows together; the bad share still enters m and v | A1 with A5 |
| SPAM | clips coordinates, then resets all moments on a clock | prereg-v3 P5; T3 |
| Rollback | exact, but recomputes and needs a snapshot | the tournament includes it and reports cost beside damage |
| Any single remedy | safe in one phase, harmful in another (seed 0, §2.1) | T2 |

### §6.8 Metrics
- **NEW:** verdict accuracy by row type; retention on the hard set; regret and harm at horizon H; prediction
  correlation; interval coverage; overhead in time and memory.

### §7.2 Thresholds set before running
A1 to A5, T1 to T3 and C1 to C3 above. They are proposals until a pre-registration carries them. They are
exploratory under prereg-v3, which covers only E1 to E3 and Extension A.

### Threats to validity
- **No damage to repair.** At realistic dose the proxy self-heals, so final-loss gains cannot be shown. The tests
  are built on discrimination, safety and headroom.
- **Rare real data.** Later batches may never contain a rare token again. Appeal treats "no evidence" like an
  ordinary example and restores it; A2 and A3 test exactly this.
- **Stale gradients.** A share restored K steps late was computed at older weights. Asynchronous SGD tolerates
  small delays [abs, 1506.08272], but that analysis is for SGD.
- **The reference is not quite neutral.** Shadow groups are fed to Adam at the flagged step and the withheld share
  is not, which tilts the verdict slightly toward restoring [I]. A rank-calibrated flag threshold, if added later,
  assumes recent scores resemble current ones, and training drifts.
- **Clipping.** The global clip couples the flagged and kept shares; the record stores the clip factor for that
  reason.
- **Short horizons.** The tournament's guarantee holds at H, not at the end. T1 and T2 measure the gap.
- **Snapshot memory.** One extra copy of the full state. Trivial here, real at billions of parameters.
- **Linearisation.** The ceiling is first-order. The measured interaction says it fails at high dose.
- **One model, one seed so far.** Everything rests on seed 0 and a small proxy.
- **Shallow searches.** §4.

---

## 6. Where the code would go (nothing is written)

| File | Content | Hooks it uses |
|---|---|---|
| `research/autopsy/appeal.py` | flagging, records, evidence, verdict (build plan: `appeal_build_plan.md`) | `pre_step` (replace the batch), `pre_opt_step` (edit m and v) |
| `research/autopsy/tournament.py` | snapshot ring, menu, judge, commit | `run_fork`, `snapshot.capture` and `restore` |
| `research/analysis/costate.py` | functional AdamW step, backward sweep, ceiling, intervals | none; offline |
| `research/spikes/induce.py` | the `hard_rows` recipe | existing recipe table |
| `research/analysis/regret.py` | regret and harm from saved horizons | none; offline |

The two repair files sit next to the RML undo they reuse. `research/repair/` is under Eddie's proposed role in
`PLAN_V8.md` §11.1, so they move there only if he wants them there.

`determinism.py` and `preflight.py` are not touched. A hooks-only Appeal needs no change to the loop. The cheaper
build needs per-row losses from `trunk.train_forward`; that change must be flagged and the `Δ==0` gate re-run.
Tests run in CI; the GPU self-checks are the finite-difference check for the costate and the gate for any new loop.

---

## 7. The three answers the brief asks for

1. **The single most novel combination.** K1 Appeal: ghost-norm flagging from private training, alignment scoring
   from data valuation, and the undo-log identity from this project, joined so that a rejected example can be
   restored later, exactly, on the evidence of the training stream itself.
2. **Why it has not been done.** Three communities each hold one piece and none needs the other two. Data-selection
   work scores examples to save compute, against a validation set, and never has to undo a choice. Robust
   aggregation treats rejection as final. The exact algebra of removing a past gradient from Adam's averages has
   been used only to delete, in unlearning. And spike remedies act on whole batches, where there is nothing to
   restore. This is my reading of the gap [I]; the searches found no counter-example.
3. **The first experiment to run.** `repeat_ngram` at 25% of rows against `hard_rows` at 25%, steps 1000 and 1800,
   one seed first: do the verdicts separate the two (A1, A2)? It needs the hooks-only build, about 20 forks, and
   under 0.2 GPU-hours [estimate]. If the verdicts do not separate, Appeal is dropped and K2 becomes the lead.
   Before any GPU: run the offline regret analysis for K2 on the existing seed-0 files in the analysis code, which
   costs nothing.
