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
do better than the menu. An offline check on seed 0 (§2.3) narrows K2: it works as insurance against a harmful
remedy, it needs a horizon near 100 steps, and it did not beat the best fixed remedy.

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

Seed 0 shows what it is for, and what it is not for (§2.3). Some remedies are harmful in one phase and harmless in
another. Undoing m alone left the loss 0.117 nats above the clean run 19 steps after the spike at step 1000, ten
times the spike's own excess of 0.0115, while at step 160 it was the best undo arm at that horizon. A full moment
reset helped at step 160 and hurt at steps 1000 and 1800. But ZClip and the three RML variants never ended more than
0.003 nats above doing nothing in any phase. So the tournament is insurance against a harmful remedy. It is not a
source of gain when a safe remedy is already known.

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
- **H:** five times v's memory, 5/(1−β2) steps (100 at β2 = 0.95). Short rollouts are known to favour greedy
  choices [V, 1803.02021]. The first proposal was one v-memory; the offline check in §2.3 shows that is too short.
- **Free judge (to test):** the mean training loss over the rollout. Every arm sees the same batches, each for the
  first time, so it is a paired held-out estimate that costs no extra forward pass.
- **Extrapolated judge (optional):** fit each arm's excess loss over the rollout and predict its end value, as
  AutoLRS does for candidate learning rates [abs, 2105.10762]. Not run yet: it needs a curve fit, and only `jq`
  runs on the workstation.

### 2.3 The experiment
**Offline, no extra GPU.** Every fork in the battery already records the held-out loss at fixed horizons for every
arm. So for each instance: pick the arm that looks best at horizon H, then read what that arm did at the end.
- regret(H) = end damage of the pick − end damage of the best arm;
- harm(H) = end damage of the pick − end damage of "none".

**Offline check on seed 0 (run 2026-11-13; exploratory).** Data: the raw fork files of the prereg-v2 run
(`results/mi300x_2026-11-10/2026-11-10_prereg_v2/`, the `*_core.json` and `*_repair.json` files). They hold every
arm's held-out loss 1, 20, 100 and 200 steps into the fork and at the end, and its training-loss trace. Script:
`tournament_offline.jq` (`jq` only; nothing else ran). Nineteen events, all from one trunk:
- the real spike at steps 160, 1000 and 1800, with 17 arms (four undo arms and thirteen remedies);
- the real spike at β1 = 0.8 and 0.95, with the four undo arms;
- fourteen planted-dose events with five arms (none, undo-m, undo-v, undo-m+v, cooldown).

The pick is the arm with the lowest held-out loss at H. All values are nats against the clean run.

| Menu | H | Events | Mean regret | Worst regret | Mean harm vs none | Worst harm | End damage: pick / none / best arm |
|---|---|---|---|---|---|---|---|
| Real spike, 17 arms | 20 | 3 | 0.0184 | 0.0492 | +0.0164 | **+0.0489** | 0.0173 / 0.0009 / −0.0011 |
| Real spike, 17 arms | 100 | 3 | 0.0020 | 0.0033 | 0.0000 | +0.0030 | 0.0009 / 0.0009 / −0.0011 |
| Real spike, 17 arms | 200 | 2 | 0.0020 | 0.0033 | +0.0002 | +0.0030 | 0.0004 / 0.0002 / −0.0015 |
| Planted dose, 5 arms | 20 | 14 | 0.0034 | 0.0130 | −0.0063 | +0.0009 | 0.0044 / 0.0107 / 0.0011 |
| Planted dose, 5 arms | 100 | 14 | 0.0033 | 0.0130 | −0.0063 | +0.0009 | 0.0044 / 0.0107 / 0.0011 |
| Planted dose, 5 arms | 200 | 10 | 0.0011 | 0.0056 | −0.0081 | +0.0009 | 0.0025 / 0.0106 / 0.0014 |

What it shows:
1. **A horizon of 20 steps is too short.** At step 1000 the pick at H = 20 was SPAM, whose moment reset fell on
   that step. SPAM was 0.0085 nats *below* the clean run at H = 20, 0.0245 above it at H = 100, and 0.0488 above
   it at the end. That is the short-horizon bias in a single event. The thresholds first proposed at H = 20 fail
   on seed 0: T1 and T3 fail outright, and the harm T2 guards against occurs at step 1000.
2. **At 100 steps it does no harm on this seed.** The worst harm over all nineteen events is 0.0030 nats, and no
   pick is one of the damaging arms (v reset, SPAM, AdaGC, rollback-k).
3. **At realistic dose it gains nothing.** The pick ends at 0.0009 nats, the same as doing nothing. The best fixed
   arm (ZClip) ends 0.001 nats below doing nothing on average. This agrees with prereg-v2.
4. **At amplified dose a fixed rule beats it.** Always applying the exact undo of m and v ends at 0.0020 nats on
   average. The tournament at H ≤ 100 ends at 0.0044, because a short horizon favours undoing m alone (worst
   regret 0.0130, at step 1000 and c = 30). Against doing nothing (0.0107) it recovers 66% of the available gain at
   H ≤ 100 and 88% at H = 200.
5. **A free judge did as well.** Picking by mean training loss over the rollout gave a mean regret of 0.0024
   (H = 20) and 0.0020 (H = 100) on the real spike, and did not pick SPAM at H = 20. On the dose events it gave
   0.0025, 0.0034 and 0.0014 at H = 20, 100 and 200, against 0.0034, 0.0033 and 0.0011 for the held-out judge. One
   event carries the difference at H = 20. Using only the second half of the rollout did worse (it picked AdaGC
   twice, with harm up to 0.0103).

Limits of this check: one seed, and the nineteen events share one trunk, so they are not independent. H = 100 was
chosen after seeing this data, so only new seeds can test it. The rollback arms are compared at matched compute
and are left out of the training-loss judge, because their data stream is shifted. The dose arms use the exact
undo, which a replay can recompute. The 0.117-nat transient from undoing m alone at step 1000 is still unexplained.

**Online demonstration.** Three seeds, two phases, the menu above plus two arms known to do harm on seed 0 (v reset
and SPAM), to check that the tournament rejects them. The accept-or-rollback supervisor of Or 2026
[page, 2601.17483] runs beside it as a baseline.

### 2.4 What would prove it wrong
| # | Prediction | Wrong if |
|---|---|---|
| T1 | A rollout of five v-memories picks well | on the real spike, mean regret at H = 100 steps exceeds 0.003 nats over seeds 1 to 7 |
| T2 | It does no harm | the pick at H = 100 ends more than 0.005 nats worse than "none" in more than one of seven seeds, in either phase |
| T3 | It rejects harmful remedies | at H = 100, v reset or SPAM is picked in any event where it ends more than 0.01 nats worse than "none" |

History, so the change is on record: T1 to T3 were first proposed at H = 20. Seed 0 fails them there (§2.3).
H = 100 was chosen after seeing seed 0, so seed 0 cannot count as a test of it. No claim is made that the tournament
beats the best fixed remedy; on seed 0 it does not (§2.3, point 4).

### 2.5 Cost
Per spike: (menu size) × (steps since the snapshot + H). With five arms, N = 50 and H = 100 that is at most 750 steps
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
| A remedy of unknown quality | can be harmless in one phase and harmful in another (seed 0, §2.3) | T2, T3 |

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
   under 0.2 GPU-hours [estimate]. If the verdicts do not separate, Appeal is dropped. K2 would not carry the
   work alone: its offline check has been run (§2.3) and supports it as insurance at H = 100, not as a source of
   gain.
