# Pre-registration v3 (DRAFT): replication, the 2 × 2 split, and the β regime

**DRAFT, written 2026-11-12. Not signed off. Not tagged. It governs nothing until Jasraj signs it and the git tag
`prereg-v3` exists.** Every threshold below is a proposal the team can change before the tag and not after. Background
and reasoning: `docs/novelty/upgraded_plan.md`. `docs/prereg_v2.md` is frozen and was not edited.

## Status of what came before
- The T1 pilot (2026-10-27) is exploratory.
- prereg-v2 (tag `prereg-v2`, run 2026-11-10, seed 0) was a dose-finding run. Its verdicts stand as recorded in
  `docs/prereg_v2_results.md`: the third outcome applies ("no lasting damage at realistic dose"); H-A passed; H-A2,
  H-B, H-C, H-E and H-F failed as written; H-D read "coordinate-selective".
- v2 names "replication on seeds 1–4 at the chosen doses" as the confirmatory step. This document defines that step.
- It also adds hypotheses that were formed **after looking at seed 0** (the 2 × 2 reading in `upgraded_plan.md` §2).
  Seed 0 is therefore exploratory for everything new here and is excluded from every test below.

## Design (proxy, `corrupt_batch`, width 1, constant LR, batch 64, max_steps 2000; config and container as v2)

**Core (E1): seeds 1 to 7.** Seeds 1–4 are v2's confirmatory seeds. Seeds 5–7 are an extension declared here so that
an exact test can pass (see Statistics).
- **Phases:** inject at 1000 and 1800. Step 160 is not in the core (see Extension A).
- **Doses:** c ∈ {1, 10, 30}. c = 3 is dropped; on seed 0 its cells sat inside the scatter between arms.
- **Arms at each dose, τ = 0:** none, undo-m, undo-v, undo-m+v, global cooldown (the five arms the battery runs).
- **Chaos floor:** K = 8 placebo forks per phase per seed, w channel (v2's primary channel).
- **Remedy battery at c = 1, lag 3:** v2's arms and comparators, plus two new comparators:
  `Bspam_clip` (SPAM with `delta_t = 0`, no reset) and `Bspam_reset` (SPAM with `theta = 0`, no clip). PR #28
  wires both in `fork.make_baseline_branches`, with CI tests; it must merge before the tag.

**β2 arms (E2): seeds 1 to 3, β1 = 0.9.** The end of the run is one v-memory, 1/(1−β2) steps, after the spike.

| β2 | Inject step | Fork length | Doses c | Extra |
|---|---|---|---|---|
| 0.95 | 1980 | 20 | 1, 10, 30 | uses the E1 trunk |
| 0.99 | 1900 | 100 | 1, 22.4, 67.1 | |
| 0.999 | 1000 | 1000 | 1, 70.7, 212 | remedy battery at c = 1, lag 3 |
| 0.99 | 1000 | 1000 | 1, 22.4, 67.1 | ten memories; compared with β2 = 0.95 at step 1800 from E1 |

The scaled doses are 10 and 30 times √(0.05/(1−β2)), which matches x = (1−β2)(cG)²/v̄ if v̄ were equal across trunks.
Whether it matched is read from the logged dose x̄ (`probes.dose_meter`), not assumed. K = 8 placebos per row.

**β1 arms: seeds 1 to 3, β2 = 0.95.** β1 ∈ {0.8, 0.95}, inject at 1800, c ∈ {1, 10, 30}, K = 8. Compared with β1 = 0.9
at step 1800 from E1. (v2 ran these at step 160; step 1800 is used here because its forks are nine times shorter.)

**Floor curve (E3): seed 1.** `--battery chaos`, K = 8, forks from steps 160, 500, 1000, 1400, 1800 and 1950.
Descriptive only.

**Extension A: step 160 on seeds 1 to 4**, same doses, arms, floor and remedy battery as the core. It runs only
after E1, E2, the β1 arms and E3, and only if the GPU ledger then shows at least 3.5 hours beyond a 3-hour reserve.
That rule depends on the ledger alone.

## Quantities
For one seed, one injection step and one dose, with Y(arm) = held-out loss at the end of the run minus the clean
twin's, in nats:

- D = Y(undo-m+v); E_m = Y(undo-v) − Y(undo-m+v); E_v = Y(undo-m) − Y(undo-m+v);
- I = Y(none) − Y(undo-v) − Y(undo-m) + Y(undo-m+v), so Y(none) = D + E_m + E_v + I exactly;
- protection share π = −I / E_m. If E_m ≤ 0.002 nats, π is not computed and that seed counts against any hypothesis
  that uses it;
- dose slope b(Q) = ln(Q(c=30)/Q(c=10)) / ln 3. If either value is ≤ 0 the slope is undefined and that seed counts
  against the hypothesis.

## Hypotheses (directions and thresholds fixed at the tag)

Two families, each tested in the order shown at one-sided α = 0.025. Within a family, testing stops at the first
hypothesis that fails; the later ones are reported as "not tested".

**Family A: the realistic dose (c = 1).**

| ID | Prediction | Per-seed statistic | Pass criterion |
|---|---|---|---|
| P1 | No lasting damage from the real spike | mean over steps 1000 and 1800 of Y(none) | two one-sided t-tests put the mean over seeds inside ±0.01 nats |
| P5 | Clearing v does harm; SPAM's cost comes from its reset | (a) min over both steps of Y(`Bvreset`) − Y(none) − 0.01; (b) at step 1000, Y(`Bspam_reset`) − Y(`Bspam_clip`) | both (a) and (b) positive by the exact test |

At c = 1, "none" is B0, the real spike, as in v2's third outcome; the planted `Px1` arm is reported beside it.
P5(b) is read at step 1000 only. SPAM resets when `(step + 1) % 500 == 0`, so at steps 499, 999, 1499 and 1999.
A fork from step 1800 meets only the reset at step 1999, its last step, which runs at 1% LR (the start of the
warmup), so it cannot show what a reset costs. The fork from step 1000 meets the resets at 1499 and 1999. Every arm
with resets ends on one, and its end-of-run Y includes it (open decision 9).

**Family B: the mechanism at high dose (c = 10 and 30).**

| ID | Prediction | Per-seed statistic | Pass criterion |
|---|---|---|---|
| P2 | Undoing v alone is worse than doing nothing | min over the four cells (2 steps × 2 doses) of Y(undo-v) − Y(none) | positive by the exact test |
| P3 | v cancels most of m's damage, more so at higher dose | (a) min over steps of π(10) − 0.5; (b) min over steps of π(30) − π(10) | both positive by the exact test |
| P4 | m's damage grows like dose squared; the total grows slower than dose | (a) mean over steps of b(E_m); (b) 1 − max over steps of b(total) | (a) the 95% interval of the mean over seeds lies inside [1.5, 2.5]; (b) positive by the exact test |

**Secondary (reported with intervals, no pass/fail weight on the paper's abstract).**

| ID | Prediction | Read as |
|---|---|---|
| S1 | One v-memory after the spike, π depends on the dose x̄ and not on β2 | for dose pairs whose logged x̄ agree within a factor of 2, the seed-mean π differs by less than 0.15 between every pair of β2 values. Pairs that do not match are "not testable" |
| S2 | The hand theory says π falls as β1 rises; seed 0 showed the opposite at step 160 | two-sided: report the order of the seed-mean π(c = 10, step 1800) across β1 = 0.8, 0.9, 0.95 |

**v2's hypotheses on seeds 1 to 4, as written in v2.** H-B, H-C, H-D, H-E and H-F are scored per seed with that
seed's own floor, at the phases and doses run, and reported as "holds in k of 4 seeds". Two narrowings are declared
now: H-A is scored on the phases run (two, or three if Extension A runs) instead of "≥ 2 of 3"; H-A2 is scored at
step 1800 on seeds 1–3 instead of step 160. No v2 threshold changes.

## Statistics
- **Unit:** the seed. Steps and doses within a seed share a trunk and are combined into one number per seed, as in
  the tables above.
- **Exact test:** one-sided sign-flip (randomization) test on the seven per-seed statistics. Its smallest possible
  p is 1/128 = 0.0078. With five seeds it would be 1/32 = 0.031, which cannot pass α = 0.025; that is why seeds
  5–7 are added.
- **Effect sizes:** every per-seed value is shown, with the mean and a bootstrap interval.
- v2's rules stay: effects with |d| < 2 are "within chaos"; |Δ| < 0.01 nats is below practical size; non-survivors
  are reported, never dropped.

## Rules
- All seeds in the design are run whatever the early seeds show. No hypothesis result changes the design.
- **Cost rule (depends on the ledger only).** Seed 1 of E1 runs first and its GPU time is read. If seven seeds would
  then cost more than 6 hours, only seeds 1–5 run, and the primary test for P2, P3, P4(b) and P5 becomes a one-sided
  paired t-test at α = 0.025. This is decided before seed 2 starts and is written into the results note.
- **Exclusion.** A seed is excluded only if its `Δ==0` gate fails, or its clean run ends more than 0.3 nats above
  seed 0's clean run (2.833). Excluded seeds are listed with the reason.
- **Order of runs:** E1, E2, β1 arms, E3, Extension A. Anything else (a lag sweep, Lion, `lr_bump`) is exploratory and
  is not covered by this document.
- Deviations are logged in the results note the way v2's were.

## What was known when this draft was written
- All of seed 0: `docs/prereg_v2_results.md`, the sealed report, the 2 × 2 reading and the scored blind predictions
  (`docs/novelty/method.md` §6.1a). On seed 0, P1 to P4 and P5(a) hold. That is why seed 0 is excluded. P5(b)'s two
  arms have not been run on any seed.
- No run of this design on any seed other than 0.
- The T1 pilot used seed 0 only (`results/mi300x_2026-10-27/t1_pilot/run.sh`, `--seed 0`, inject at 160), so seeds
  1–7 have not been run in any form.

## Open decisions before the tag (for the team)
Each decision has a suggested answer (Jasraj's draft, 2026-11-20). None is decided until the team agrees and the
text above is changed to match.

1. Seven seeds, or five with t-tests from the start.
   *Suggested: seven.* Five seeds cannot pass the exact test (smallest p 1/32 > 0.025), and a t-test on five values
   rests on a normality assumption that five values cannot check. The cost rule already falls back to five if seed 1
   shows seven would cost more than 6 hours. Seeds 6 and 7 cost about 1.2 hours [estimate].
2. Whether Extension A (step 160) comes before or after a lag sweep at β2 = 0.999.
   *Suggested: neither runs under this document.* PLAN_V8 §11.2 gives the hours after the core to the RML block
   (4.0 h). Extension A's ledger rule would still fire: after G0 and the core, about 6.8 hours remain [estimate],
   above the 6.5 hours the rule asks for, so it would take the RML block's hours. Either remove Extension A (step 160
   stays exploratory and H-A is scored on two phases), or make its rule hold back the RML block's 4.0 hours as well.
   The lag sweep stays exploratory, as the Rules already say.
3. P4(a)'s interval [1.5, 2.5]. Seed 0 gave 1.95 and 2.31 at these two steps.
   *Suggested: keep it.* It is centred on the theory's exponent 2, not on seed 0, and it excludes both rival
   readings, a slope of 1 and a slope of 3. Narrowing it toward seed 0's values would tune it to an excluded seed.
   It tests precision as well as the mean: with seven seeds, the interval fits only if the per-seed slopes have a
   standard deviation below about 0.4 to 0.5 [estimate]. That spread is unknown.
4. S1's tolerances (0.15 in π, a factor of 2 in x̄).
   *Suggested: keep both.* S1 is secondary, so a loose tolerance costs little. No run has measured how much π varies
   between seeds, so a tighter number would be a guess. The measured differences are reported with intervals either
   way.
5. **Answered:** Eddie wires `Bspam_clip` and `Bspam_reset` in `fork.make_baseline_branches`, so every repair
   battery gets them (PR #28, CI green, not merged yet).
6. Whether to log the 10%, 50% and 90% quantiles of x on the hit support as well as x̄ (a logging change only).
   *Suggested: yes, if it lands before the tag.* It needs no GPU: a change to `probes.dose_meter` with a CPU test in
   CI. It helps S1, because two trunks can share x̄ while their spread of x differs. If it is not ready, skip it
   rather than hold the tag; S1 is secondary. **Built 2026-11-20:** `dose_meter` now writes `x_q10`, `x_q50` and
   `x_q90` next to `xbar` (unweighted, window steps pooled), with a CPU test. A yes means merging it before the tag.
7. The exclusion threshold of 0.3 nats.
   *Suggested: keep it.* The rule is meant to catch a broken run (wrong data, a diverged trunk), not a seed that ends
   a little worse. Only seed 0 has run this configuration, so the normal spread between seeds is unknown, and a tight
   threshold could drop honest seeds. 0.3 nats is about 10% of seed 0's 2.833.
8. Whether the raw fork files stay committed beside the report. They hold every arm's held-out loss at each horizon
   and its training-loss trace; the sealed report keeps per-horizon values only for the four undo arms. The
   exploratory regret analysis in `novelty/rml3_method.md` §2.3 was run from them. No extra GPU.
   *Suggested: keep them committed.* v2's 104 result files are 7.2 MB in all, and snapshots stay out of git
   (`*.safetensors` is in `.gitignore`). v3 runs about four times v2's fork-steps, so a few tens of MB [estimate].
   They are the only record of the non-undo arms at each horizon.
9. Whether P5(b)'s Y should avoid SPAM's last reset at step 1999: read Y one step earlier, or pick a `max_steps`
   that does not end on a reset. As written, the reset at 1999 is inside the end-of-run Y of `Bspam_reset`.
   *Suggested, in two parts.*
   - **The last reset:** read P5(b) at step 1998 by adding 999 to the remedy battery's `--horizons` for the fork
     from step 1000 (`mve` reads a horizon h as step inject + h − 1). No GPU. Changing `max_steps` would also move
     every other result and break the comparison with v2.
   - **A bigger problem, found while drafting this:** the trunk is plain AdamW, but `spam.phase` counts SPAM's
     schedule from global step 0. In the fork from step 1000, `cur = 1001`, so the arms with resets (`Bspam`,
     `Bspam_reset`) start in the warmup after a reset that never happened in the fork. The spike step runs at about
     1% LR, and the LR ramps back up over steps 1000 to 1148. Full `Bspam` doesn't clip the spike either:
     1001 % 500 = 1, inside `grace` = 20. `Bspam_clip` clips it at full LR. Y(`Bspam_reset`) − Y(`Bspam_clip`)
     therefore mixes the cost of the reset at 1499 with a warmup that mutes the spike itself. Either the SPAM arms
     count their reset and warmup clock from the fork start and apply the post-reset grace only after a reset the
     fork actually ran, so every SPAM arm clips from the fork's first step (Eddie's code, so this went to him as
     feedback on PR #28 on 20 Nov), or P5(b) moves to the secondary table. If neither is done before the tag, move it. v2's `Bspam` at step 1000 ran with the same
     warmup; v2 is frozen, so the v3 results note should say so.

## Budget [estimates, not measured]
Basis: the v2 session ran roughly 200,000 fork-steps in 2.1 GPU-hours, by my reconstruction of what it ran, which
gives about 100,000 fork-steps per hour. The session also ran gates, tests and restarts, so the real cost per fork
should be lower.

| Block | Fork-steps | GPU-h |
|---|---|---|
| E1, 7 seeds × about 61,000 | 430,000 | 4.3 |
| E2, β2 arms and one remedy battery | 260,000 | 2.6 |
| β1 arms | 46,000 | 0.5 |
| E3, floor curve | 48,000 | 0.5 |
| Reserve | | 3.0 |
| Extension A | 340,000 | 3.4 |
| **Total** | | **14.3 of 14.9** |

The fork-step counts include replaying the trunk to the injection step for every invocation, and assume about 29
forks per core invocation and 17 per remedy battery. Both counts are my reading of the battery, not a measurement.
With Extension A the plan leaves 0.6 hours, so any exploratory run competes with it (open decision 2).

## Run (draft commands; none has been run)
```bash
CFG=results/mi300x_2026-10-27/t1_pilot/config.yaml
A="--config $CFG --kind corrupt_batch --data-dir $DATA"
# E1: seeds 1..7, phases 1000 / 1800
python -m research.autopsy.mve $A --seed $SEED --inject-step $INJ --horizons 1,20,100,200 --taus 0 --c2-taus 0 \
    --dts "" --plant-coefs 1,10,30 --dose-taus 0 --placebo 8 --placebo-channels w --out $OUT/s${SEED}_inj$INJ
python -m research.autopsy.mve $A --seed $SEED --inject-step $INJ --horizons 1,20,100,200 --battery repair --lag 3 \
    --out $OUT/s${SEED}_repair_inj$INJ
# E2: seeds 1..3. (B2, INJ, COEFS, HOR) = (0.95, 1980, "1,10,30", "1,10"), (0.99, 1900, "1,22.4,67.1", "10,50"),
#     (0.999, 1000, "1,70.7,212", "100,500"), (0.99, 1000, "1,22.4,67.1", "100,500")
python -m research.autopsy.mve $A --seed $SEED --betas 0.9,$B2 --inject-step $INJ --horizons $HOR --taus 0 \
    --c2-taus 0 --dts "" --plant-coefs $COEFS --dose-taus 0 --placebo 8 --placebo-channels w \
    --out $OUT/s${SEED}_b2_${B2}_inj$INJ
python -m research.autopsy.mve $A --seed $SEED --betas 0.9,0.999 --inject-step 1000 --horizons 100,500 \
    --battery repair --lag 3 --out $OUT/s${SEED}_b2_0.999_repair
# beta1 arms: seeds 1..3, B1 in 0.8, 0.95
python -m research.autopsy.mve $A --seed $SEED --betas $B1,0.95 --inject-step 1800 --horizons 1,20,100 --taus 0 \
    --c2-taus 0 --dts "" --plant-coefs 1,10,30 --dose-taus 0 --placebo 8 --placebo-channels w \
    --out $OUT/s${SEED}_b1_${B1}
# E3: the chaos battery of docs/chaos_floor.md with --seed 1 and --inject-step in 160 500 1000 1400 1800 1950
```
The flags are the ones `research/autopsy/mve.py` accepts today. The commands were not dry-run; check them in CI or
on the GPU box before the tag. `research/gpu/exea_prereg_v3.sh` runs these same commands on Exea Reserve (`exea.md`),
together with G0 and the cost rule; if one changes, change the other to match.
