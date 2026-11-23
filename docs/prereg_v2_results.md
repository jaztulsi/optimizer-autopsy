# prereg-v2 results and deviations (MI300X, 2026-11-10 UTC)

Governing rules: `docs/prereg_v2.md` at tag `prereg-v2` (d823a4b). That file is frozen and was not edited. Evidence:
`results/mi300x_2026-11-10/`, final report at `2026-11-10_prereg_v2/report_final/`. Torch 2.10.0+rocm7.14.1 in the
pinned container, one MI300X VF. All numbers are seed 0, proxy model, `corrupt_batch`, width 1. This is a dose-finding
run; the prereg says replication on seeds 1–4 is the confirmatory step.

**Bottom line.** The frozen third outcome applies: at c = 1, |Δ(B0)| < 0.01 nats in all 3 phases, so the abstract is
"no lasting damage at realistic dose", whatever the higher doses show. Above that, at c ≥ 10, the protective-v and
memory-writes-the-damage patterns are large and consistent. Repair at lag 3 did not meet its bar.

## Effects (Δ = val loss at end − clean run B\*, nats; B\* = 2.83315)

σ is the w-channel 1-ulp floor at end of schedule. Step 1800 uses its own K = 16; steps 160 and 1000 use stage 4's K = 8.

| Phase (σ) | c | none | undo-m | undo-v | undo-m+v | cooldown | d(none) | memory share |
|---|---|---|---|---|---|---|---|---|
| 160 (1.04e-4) | 1 | +0.00040 | +0.00126 | +0.00225 | +0.00209 | −0.00243 | 3.8 | −4.21 |
| | 3 | +0.00312 | −0.00073 | −0.00252 | +0.00195 | −0.00363 | 30 | 0.37 |
| | 10 | +0.00778 | +0.00437 | +0.04744 | +0.00214 | +0.02504 | 75 | 0.73 |
| | 30 | +0.02499 | +0.01414 | +0.60050 | +0.00230 | +0.06821 | 240 | 0.91 |
| 1000 (4.01e-7) | 1 | −0.00014 | +0.00505 | −0.00044 | +0.00263 | −0.00036 | −354 | n/a |
| | 3 | +0.00431 | +0.00465 | +0.00354 | +0.00167 | +0.01058 | 10,726 | 0.61 |
| | 10 | +0.01137 | +0.00369 | +0.02553 | +0.00178 | +0.02201 | 28,313 | 0.84 |
| | 30 | +0.01538 | +0.01406 | +0.20295 | +0.00106 | +0.05594 | 38,303 | 0.93 |
| 1800 (2.71e-8) | 1 | +0.00243 | +0.00234 | +0.00397 | +0.00195 | +0.00252 | 89,721 | 0.20 |
| | 3 | +0.00590 | +0.00075 | +0.00754 | +0.00001 | +0.00867 | 217,660 | 1.00 |
| | 10 | +0.01104 | +0.00588 | +0.04506 | −0.00027 | +0.02223 | 407,426 | 1.02 |
| | 30 | +0.02444 | +0.01138 | +0.57190 | −0.00063 | +0.08331 | 902,359 | 1.03 |

The "none" arm at each dose is the planted arm `Pxc`. At c = 1 it equals the real spike B0 at steps 1000 and 1800
(to 1e-7), and is close at step 160 (B0 = +0.00053, d = 5.1). Memory share = 1 − Δ(undo-m+v)/Δ(none).

## Verdicts, as frozen

| ID | Rule | Result | Verdict |
|---|---|---|---|
| H-A | d ≥ 2 at some c ≤ 30 in ≥ 2 of 3 phases | d ≥ 2 in all 3. c\* = 1 at 160 and 1800, 3 at 1000. Damage rises with dose in every phase | **Pass**, but see the ruler problem below. Practical size (0.01 nats) is first reached at c = 30 (step 160) and c = 10 (steps 1000, 1800) |
| H-A2 | Δ at c = 10 larger for β1 = 0.95 than 0.8 | β1 = 0.95: +0.01431. β1 = 0.8: +0.02428 | **Fail** (opposite direction). Direction only: these arms have no floor of their own |
| H-B | At every dose with d ≥ 2, undo-v worse than none | True at 9 of 11 doses. False at c = 3 in steps 160 and 1000. At c ≥ 10: 6 of 6, 2.2× to 24× worse | **Fail as written**; holds at c ≥ 10 |
| H-C | At every dose with d ≥ 2, memory share ≥ 0.5 | True at 8 of 11. False at c = 1 and 3 in step 160, c = 1 in step 1800. At c ≥ 10: 6 of 6 (0.73–1.03) | **Fail as written**; holds at c ≥ 10 |
| H-D | \|cooldown − none\| ≤ 1σ means global schedule | Larger than 1σ at every dose | **"Coordinate-selective"** by the rule (see below) |
| H-E | Best RML closes ≥ 80% where detector-timed skip closes ≤ 20% | Step 1800: RML-full closes 68%, skip closes −107%. Step 160: RML overshoots (closes > 100%), skip −243%, on a 0.0005-nat effect. Step 1000: not testable (d < 2) | **Fail** at 1800; the 160 pass is formal only |
| H-F | RML-sel within 1σ of RML-full | Step 1800: 0.00017 apart (6,000σ). Step 160: 0.00063 apart (6σ) | **Fail** |
| Third outcome | c = 1, all phases: d < 2 or \|Δ\| < 0.01 | +0.00053, −0.00014, +0.00243 | **Applies** |

## Memory length (step 160, c = 10; no placebos)

| β1 | Clean val | none | undo-m | undo-v | undo-m+v | cooldown | memory share |
|---|---|---|---|---|---|---|---|
| 0.8 | 2.79466 | +0.02428 | +0.01675 | +0.07122 | +0.00679 | +0.05188 | 0.72 |
| 0.9 | 2.83315 | +0.00778 | +0.00437 | +0.04744 | +0.00214 | +0.02504 | 0.73 |
| 0.95 | 2.88238 | +0.01431 | +0.01296 | +0.04502 | +0.00470 | +0.04195 | 0.67 |

C-1's prediction that the memory share rises with β1 is not seen here.

## Repair battery (c = 1, lag 3, Δ in nats)

| Arm | step 160 | step 1000 | step 1800 |
|---|---|---|---|
| none (B0) | +0.00053 | −0.00014 | +0.00243 |
| RML-full | −0.00022 | +0.00227 | +0.00077 |
| RML-cool | −0.00204 | +0.00198 | +0.00191 |
| RML-sel | −0.00085 | +0.00169 | +0.00094 |
| skip-step, oracle timing | −0.00130 | +0.00314 | −0.00015 |
| skip-step, detector timing | +0.00182 | +0.00291 | +0.00504 |
| skip batch (Bs) | +0.00076 | +0.00285 | +0.00196 |
| clip (Bg) | −0.00205 | +0.00192 | +0.00983 |
| rollback | +0.00384 | −0.00004 | +0.00653 |
| rollback k | +0.01315 | +0.00860 | +0.01052 |
| v̄ reset | +0.07403 | +0.05288 | +1.59753 |
| SPAM | +0.07991 | +0.04875 | +0.00600 |
| ZClip | −0.00263 | −0.00014 | +0.00243 |
| AdaGC | +0.01357 | +0.00658 | +0.01268 |

ZClip did not fire at steps 1000 and 1800. SPAM and AdaGC ran with default settings and were not tuned here, so "worse
than none" is a statement about those settings. RML uses the clean gradient as an oracle receipt, which real training
does not have.

## The ruler problem (exploratory, added after seeing effects)

The 1-ulp floor is 3e-8 to 1e-4 nats, so d ≥ 2 passes almost anything. In the last 20 minutes, with GPU room free, 10
data-order forks were added at step 1800 (the same 200 batches, new order, from the tag; `2026-11-10_stages/dord_inj1800/`).
- Their SD at the end is **0.0047 nats**, about 170,000 times the 1-ulp floor at that phase.
- In that unit the c = 1, 3, 10, 30 damage at step 1800 is 0.5, 1.3, 2.4 and 5.2σ. Undo-v at c = 30 is 122σ.
- At c = 1, no RML arm differs from none, or from skip, by as much as 1σ.
- All 10 forks ended below B\*, by 0.026–0.040 (mean −0.033), so the original batch order is not typical. The cause was
  not established. The size of that shift is larger than the c = 30 damage.
- It was measured at step 1800 only. Whether this is the right ruler for paired forks is open: it answers "is this as
  big as ordinary data-order variation", while 1-ulp answers "is this above numerical chaos".

The cooldown result also needs more than the frozen rule gives. By that rule it is "coordinate-selective", but a global
cooldown recovers a large part of what undo-v loses, (undo-v − cooldown)/(undo-v − none): 25–67% at c = 10 and 78–93% at
c = 30. That ratio is post hoc.

## Checks run on the same box

- G0: Δ = 0 on the changed training loop. Stage 3 fork, cross-process and baseline checks passed.
- Capture self-check from the tag: `fork@40 t0=40 replay max|Δ|=0` (the earlier MI300X crash is fixed).
- All zero-perturbation controls replayed B\* exactly (max |D| = 0).
- Test suite on the GPU: 84 passed, 6 failed, all with "CUDA is already initialized". 3 pass alone. The other 3 seed twice
  in one process, a test problem on GPU machines. No run was affected. The repair test's resume half stays CI-only.

## Deviations from the frozen plan

The MI300X had a hard end at 04:35 UTC. All cuts were made to fit it.
1. **Floors at steps 160 and 1000 are K = 8, borrowed** from stage 4's no-spike forks, instead of the frozen K = 16
   run alongside. The g-channel floor there is also stage 4's. Only step 1800 has its own K = 16 in w and g. Stage 4
   ran before the tag, on ebfc7ed with the `mve` preflight fix that became 3133daf. The clean run B\* matches the tag's
   to every digit (2.833148365).
2. **β1 arms have no placebos**, so H-A2 is direction only.
3. **Not run:** the K = 111 chaos floor (about 7–8 h), the capture battery, the natural-spike search, and data-order
   forks at steps 160 and 1000.
4. **Added, not preregistered:** the step 1800 data-order forks above. Exploratory.
5. **Restarted runs.** The first attempts at `main_inj160`, `main_inj1000` and `beta0.8` were stopped and restarted
   without placebos, so the dose arms ran first. Their partial output is in `2026-11-10_prereg_v2_stopped/`.
6. **Disclosure.** Early arm values were visible while the run went on, including when the cut was made. The cut
   depended only on the clock. The analysis rules were not changed.
7. **The tag is local only** at the time of writing; it has not been pushed.
8. Fork-start snapshots (`*.safetensors`) stayed on the box and are gone. They can be rebuilt from the tag because the
   trunk replays bitwise.

GPU time for this session: about 2.1 hours (02:29–04:32 UTC). Project total: about 5.1 of 20 hours.
