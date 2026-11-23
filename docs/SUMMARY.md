# Optimizer Autopsy: the project in one place

*Updated 2026-11-22. Two versions: a short one (under 2,000 characters) and a long one. Both describe the
project itself: the idea, what is new about it, how it is built, and what it has found so far. Update both
together whenever the code or the results change.*

---

## Short version

**Idea.** When an LLM's training loss spikes, the damage that lasts may not come from the spike step. It may come
from Adam's *memory* of it. Adam keeps two running averages: m (momentum) and v (gradient scale). After a spike, m
keeps echoing the bad gradient into the weights (≈ 92% of the drift at β = (0.9, 0.95) for a large spike). The
inflated v stalls learning but also caps that echo. So cleaning v alone should be *worse* than doing nothing.

**Approach.** A bitwise-deterministic fork harness replays a spiked run (B0) and its data-aligned clean twin
(B\*) from one snapshot. Retroactive undo edits, the Reversible Moment Log (RML), remove the spike's share of m, v
or both at a chosen lag, weights untouched. Damage is the held-out loss gap to B\*, read against a
chaos floor from 1-ulp placebo forks. Four tests: the memory-written share (ordered by β1), protective v,
interference between two spikes, and RML repair vs baselines.

**Tactics.**
- Undo at lags 0…64 traces when memory writes the damage.
- Re-planting the spike gradient gives dose, ±sign, two-spike and refractory probes.
- A w/m/v swap factorial and a β1 × β2 cross separate the channels.
- An effect counts only at \|Δ/σ_chaos\| ≥ 2; thresholds are fixed before data.

**What's new.** Each part has precedent; new is the causal, time-resolved split of a spike's lasting damage into
direct, m-echo and v-stall parts.

**Built and run (2026-11-10).**
- Harness: Δ = 0 on CUDA and the AMD MI300X.
- Pre-registered dose sweep (proxy, seed 0): at realistic dose (c = 1) damage is ≤ 0.0025 nats, so the frozen outcome
  "no lasting damage at realistic dose" applies.
- At 10× and 30×: undoing m+v removes 73–103% of the damage; undoing v alone is 2–24× worse.
- RML at lag 3 closed 68% at step 1800 (bar: 80%).
- Next: prereg-v3 (seeds 1–7, β arms; tagged 11-23, not run), then RML-1, CWB (RML-1 + w), RML-2 (built, not run) and RML-3 (unbuilt). R-F2 predicts v3's β arms. Localizer: CPU-tested only.

---

## Long version

Labels: **[V]** verified in the repo or a primary source, **[I]** our inference, **[A]** an assumption,
**[UNV]** not confirmed.

### 1 · The idea

LLM pretraining loss spikes leave lasting damage. Every standard fix (clip, skip, reset, roll back) acts at or
before the spike step, and treats the optimizer as a black box. This project asks **who writes the lasting
damage**. Is it the spike step itself (direct), or Adam's *memory* of the spike over the next ~1/(1−β) steps
(indirect)?

The indirect path has two channels that pull in opposite directions:
- **m-echo** (extra push). On a spike-dominated coordinate, the momentum echo writes a fraction r = β1/√β2 of the
  total drift as the spike grows (G/ḡ → ∞). That is ≈ 92% at (0.9, 0.95), a corollary of Patitucci & Mokhtari
  2026, Thm 3.3 [V].
- **v-stall** (lost progress). Inflated v suppresses ordinary learning for ≈ ln((1−β2)G²/ḡ²)/(−ln β2) steps,
  ≈ 31 at G/ḡ = 10 [I].
- **v is also protective.** Its inflation is what keeps the echo bounded. The sharp consequence: undoing v alone,
  while m's echo stays, should be *worse* than doing nothing.

If the direct part dominates instead, acting at the spike step (clip or skip) is enough, and a retroactive state
repair has no job. That outcome is measured and reported the same way.

A spike-dominated coordinate's total drift also *saturates*, at ≈ 5.84·η for (0.9, 0.95), so clipping a large
spike barely changes its damage [I]. Honest Adam states satisfy \|m\| ≤ 1.165·√v (a known invariant), which is
used as a feasibility check on edited states.

### 2 · The approach

- **Fork-and-replay instrument.** A spiked run B0 and its clean twin B\* fork from one snapshot.
  - B\* reads the same data slots with the spike off. It never "skips to the next batch", because that would
    confound the spike with a data-order change.
  - Two noop forks must agree bit for bit (Δ = 0). Without that, a fork's Δ measures nondeterminism instead of
    the edit.
- **Damage.** Δ = val_loss(branch) − val_loss(B\*), measured at end of schedule on held-out slots that are
  token-disjoint from every probe batch.
- **Retroactive undo (RML, the Reversible Moment Log).**
  - It logs what the spike window's gradients wrote into m and v, then subtracts that contribution at lag τ and
    keeps the weights.
  - Undo at τ = 0 isolates the direct effect. A log grid τ ∈ {0, 1, 2, 4, 8, 16, 32, 64} traces the
    **damage-writing kernel** D(τ): how much lasting damage the memory writes at each lag.
  - Variants: undo m, undo v, undo both; undo v only on the top 1% of coordinates; and undo v plus a matched
    **global cooldown** (one LR scale that mimics what inflated v does on average).
- **Planted probes.** The measured corrupt-batch gradient delta G is re-planted on the clean stream. That gives:
  - sign parity (+G vs −G);
  - dose (c·G);
  - **interference** between two spikes Δt apart (same vs opposite sign);
  - a **refractory** probe (how much a spike blunts a later small kick).
- **Chaos floor.** K forks of B\* with a 1-ulp nudge in the weights (or the gradient) measure how far tiny
  rounding differences alone move the final loss, σ_chaos. Effects are reported as d = Δ/σ_chaos, and \|d\| < 2
  counts as "within chaos".
- **Supporting measurements:**
  - the 2³ cross-world (w, m, v) factorial, which splits damage into weight-, m- and v-carried shares at a cut
    time;
  - a β1 × β2 cross, to separate the √β1 and β2 decay rates;
  - transplants of poisoned (m, v) onto clean weights;
  - spikes at different points of the LR schedule.

### 3 · What is tested

**Four core tests:**

| ID | Test | Prediction |
|---|---|---|
| C-1 | Memory-written share, from the damage-writing kernel | It is above zero for at least one data-driven spike type and rises with β1 (0.8 < 0.9 < 0.95) |
| C-2 | Protective v | Undoing v alone leaves *more* damage than doing nothing, while undoing m+v leaves less |
| C-3 | Spike interference, with no state patching | Opposite-sign pairs interfere less than same-sign pairs for Δt ≤ 13 |
| C-4 | Retroactive repair at a detector-timed lag | RML matches the best non-oracle baseline and beats skip-step on the same spikes |

**The experiment run on 2026-11-10** ([`prereg_v2.md`](prereg_v2.md), results in
[`prereg_v2_results.md`](prereg_v2_results.md)) is a dose sweep on `corrupt_batch`:
- planted dose c ∈ {1, 3, 10, 30};
- three phases of training (descent, plateau, late);
- none / undo-m / undo-v / undo-m+v, plus the global-cooldown arm;
- β1 ∈ {0.8, 0.95} at c = 10;
- 16 placebos per phase as written (8 at steps 160 and 1000 in the actual run, because of a hardware deadline).

It asks:
- **H-A, H-A2:** Is there a **forgetting threshold**, the smallest dose c\* whose damage clears chaos, and does it
  move with β1?
- **H-B:** Is v protective?
- **H-C:** Does memory write ≥ 50% of the damage?
- **H-D:** Does v act like a global LR cooldown, or is its protection coordinate-selective?
- **H-E, H-F:** Does RML repair work after the spike was applied (≥ 80% of the damage closed, where skip-step
  closes ≤ 20%)? Is editing ≤ 2% of v enough?

**Repair comparators:** skip, clip, moment resets, v̄ reset, skip-step (oracle- and detector-timed), SPAM, ZClip,
AdaGC, and rollback-and-skip at matched compute.

### 4 · What is new, and what is not

**Not ours (cited, never claimed):**
- state transplants (Xu 2026, Chimera, Guo);
- deleting from optimizer memory (SWIFT's one-step Adam inversion; Stewart's L-BFGS deletion);
- the single-spike Adam dynamics (Patitucci & Mokhtari);
- bitwise replay (Megatron, OPEN-1B);
- forking from released Pythia states;
- the history-coverage account of mismatched m and v ("Newer Is Not Always Better"), which is C-2's mechanism;
- spike-onset capture and selective limiting ("Understanding and Mitigating AdamW Loss Spikes").

**New:**
- The first causal, time-resolved split of a loss spike's **lasting** damage into direct, m-echo and v-stall
  parts, on a bitwise counterfactual instrument, with β1-ordered predictions written down before the data.
- Three measurement ideas that the forks make possible:
  - the damage-writing kernel;
  - the refractory period;
  - spike interference.
- Ideas rated NEW on thin searches (none of them has been checked on OpenReview yet):
  - spike damage measured in chaos units, with a forgetting threshold;
  - v as an emergent cooldown;
  - direct bit-flip faults in m and v;
  - predicting a spike's damage from the placebo ensemble's expanding directions, before forking
    (ensemble-Lyapunov triage).

**Invented algorithms** (designs in [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md), Part A):
- **RML:** reversible per-coordinate moment log with delete-later / insert-later.
- **BiH-Adam:** a jump-robust bipower second moment.
- **MomentABFT:** checksums that detect, locate and correct corrupted Adam moments.
- 13 more: echo cancellation, quarantined moments, counterfactual weight back-projection, stability-certified
  restoration, integral anti-windup and others.

### 5 · Implementation

| Piece | State | Where |
|---|---|---|
| Harness: determinism, name-keyed (w, m, v) snapshots (fp32, tied weights once), trunk with hooks, fork + Δ = 0 gate | ✅ GPU-verified on T4, P100 and the MI300X | `research/harness/` |
| Cross-process + co-tenancy gate through spike windows | ✅ MI300X: 3 processes bitwise equal | `research/gpu/xproc_gate.py` |
| Spike recipes + detector (`lr_bump`, `corrupt_batch`, `repeat_ngram`, `v_inflate`, `tiny_eps`, precision) | ✅ CI; three recipes ran on the MI300X | `research/spikes/` |
| Spike capture (v̄, t0 / fork / matched-normal snapshots) | ✅ Self-check passed on the MI300X from the tag (2026-11-10). Since 11-22 it also saves v̄ at each normal step (6 by default), which the localizer needs; that change has not run on a GPU. The full capture battery has not run | `research/spikes/capture.py` |
| Baselines: skip, clip, resets, v̄, SPAM, ZClip, AdaGC, skip-step | ✅ CI; ran on the MI300X in the repair battery (2026-11-10), default settings. SPAM's two halves for prereg-v3, `Bspam_clip` and `Bspam_reset`, are wired (PR #28, merged 11-22) | `research/baselines/` |
| Rollback-and-skip at matched compute | ✅ CI-tested, merged (PRs #20 and #21); ran on the MI300X from the tag. Bitwise replay check owed at the next session | `research/baselines/`, `research/autopsy/mve.py` |
| Autopsy code: RML undo (m / v / m+v, top-k v, cooldown), planted probes, placebo floor, dose meter, GPU driver `mve`, sealed report | ✅ CI-tested; placebo, dose, cooldown and repair arms ran on the MI300X (2026-11-10) | `research/autopsy/` |
| Chaos-floor benchmark: 1-ulp placebo forks in the weight, gradient and optimizer-state arms at three phases, data-order forks, a zero-perturbation check, a per-fork release table, and d for spike effects from a matching floor | ◐ K = 8 and K = 16 ran on the MI300X; data-order forks at step 1800 only; K = 111 not run ([`chaos_floor.md`](chaos_floor.md)) | `research/autopsy/` (`--battery chaos`) |
| Held-out eval (token-disjoint slots, horizon probe) | ✅ ran on the MI300X | `research/analysis/eval.py` |
| Stats (TOST, bootstrap, Holm), attribution | ◐ partial; the exact one-sided sign-flip test prereg-v3 uses is written (11-22, CPU test) | `research/analysis/` |
| Model: GPT-2-style nanoGPT, proxy (≈ 11M params, ≈ 1.3M outside the tied token embedding) and 124M | ✅ proxy trains (held-out loss 10.8 → 2.83 in 2,000 steps) | `research/model/` |
| LR schedule (warmup + cosine / WSD), applied inside `train_forward`, so trunk, gate and every fork follow it; `lr_bump` and the cooldown arm scale it. No `schedule` key = constant LR, loop unchanged | ◐ written 2026-11-05, CI-tested (fdff507); not run on a GPU | `research/harness/schedule.py` |
| Bit-flip faults: one fp32 bit of a gradient (`sdc_grad`), of v or of m, at a known step; a parity residual (AdamW's moment recurrence, checked each step) on B\* and B0 | ◐ written 2026-11-05, wired into `mve`, CI-tested (fdff507) | `research/autopsy/faults.py` |
| Localizer (which coordinates carry the damage), repair operator, theory | ◐ Localizer: `curvature.py` (HVP on the GPT loss, the preconditioned H̃, top-k eigsh) and `poison.py`'s v̄ accumulator (PR #15), then directional SNR, the poison score and selection, the end-to-end pipeline, eigenbasis stability checks, spike-vs-normal validation and a GPU driver (PR #40), all by Avyank and merged 11-22; CPU tests, nothing run on a GPU. Repair operator: stub; RML-1 covers part of the repair; RML-2's core (coordinate probation) is in `autopsy/probation.py` (PR #27), wired into `mve` as the `RML2` and `RML2@win` arms (PR #30, merged 11-20; CPU tests, not run on a GPU). RML-3 (row appeal) is designed, not built, and waits on its kill test; it restores through the same `undo.undo`. CWB (RML-1 + w, EXPERIMENT_CATALOG A5) is in `autopsy/cwb.py`, behind `mve --cwb` as `CWB@lag` and `RML-w@lag`: it also moves the weights, so the spike's update and its echo go without a rollback (exploratory; CPU tests, not run on a GPU). Damage triage (PLAN_V8 R-F6, exploratory) is in `repair/triage.py`: it predicts which spikes leave damage and repairs nothing (PR #32, merged 11-20; CPU tests, not run on a GPU). Theory has one law: R-F2's m+v coupling law (`theory/coupling.py`, frozen 11-21) predicts the protection share π and Y(undo-v)/Y(undo-m+v) from β1, β2 and the logged dose, with no fitted parameter, scored on prereg-v3's β arms (exploratory; CPU test only). The localizer's output format is fixed in PLAN_V8 §6.4 | `research/localizer/`, `repair/`, `theory/` |
| Factorial and causal-oracle modules; GPT-NeoX + Pythia state loader; QK-Norm / μP variants; figures | ⬜ not written | — |

**Known code issues:**
- Fixed in PR #30 (merged 11-20; CPU tests in CI, nothing run on a GPU): `mve` no longer plants phase-B arms for `sdc_grad` at bit 30, which on a GPU planted a reversed gradient; the test suite stays on the CPU on a GPU machine (`research/tests/conftest.py`), so its seeding tests no longer fail there; `trunk.py`'s `main` runs the environment check in a child process, as `mve` does, instead of crashing after CUDA starts.
- Fixed before the prereg-v3 tag (PR #43): SPAM's warmup clock counted from the global step, so a fork from step 1000 started `Bspam` and `Bspam_reset` inside the warmup after SPAM's step-999 reset, which the fork never ran (the spike step at about 1% LR, and full `Bspam` inside its 20-step grace, so no clipping). Warmup and grace now follow only resets the fork itself ran; forks from 160, 480 and 1800 are unchanged. prereg-v2's `Bspam` at step 1000 ran with the old clock (`start=0` reproduces it). The SPAM split stays secondary S3 in prereg-v3 (decision 9 in [`prereg_v3.md`](prereg_v3.md)).

### 6 · Results so far (AMD MI300X, 2026-10-27)

Evidence: [`results/mi300x_2026-10-27/`](../results/mi300x_2026-10-27/).

- **Determinism holds on AMD** [V]:
  - in-process fork gate, Δ = 0;
  - 3 processes (two of them sharing the GPU) bitwise equal for the noop, `lr_bump` and `corrupt_batch`
    recipes, through the spike window;
  - trunk → capture → save/load → noop gate, Δ = 0.
- **Speed** [V]: 22.2 ms/step solo for the proxy at batch 64, ≈ 33 ms/step per deterministic fork when forks share
  the GPU, and ≈ 1.3× throughput from packing two forks. That is ≈ 43 s (0.012 GPU-h) per end-of-schedule proxy
  branch.
- **GPU budget:** 20 MI300X-hours in total, about 5.1 used so far. The next session was booked on 11-20; prereg-v3 (tagged 11-23) E1 runs in it.
- **Pilot** [V, exploratory]: 120 of 120 branches with 0 crashes. Width-1 spikes at step 160, seed 0, constant LR,
  with `corrupt_batch` and `repeat_ngram` at β1 = 0.9, `lr_bump`, and `corrupt_batch` at β1 = 0.8 and 0.95.
  - **Spikes this small mostly wash out.** Lasting damage was 0.0003–0.0117 nats, while the 1-ulp chaos scale
    (+G planted vs B0) reached 0.0027.
  - **One clear case:** `corrupt_batch` at β1 = 0.8 (Δ = 0.0117). Undoing m+v at τ = 0 removed ~98% of the
    damage, and undoing v alone made it worse (0.0134). That is the sign C-2 predicts, with n = 1.
  - Its thresholds were fixed after the data existed, so it is a pilot and confirms nothing.
- **What it changed:** a measured chaos floor comes before any reading of an effect, and the spike dose has to
  rise until damage clears that floor. Hence the dose sweep in §3.

**Pre-registered dose sweep (MI300X, 2026-11-10; proxy, seed 0, `corrupt_batch`).** Full tables and deviations:
[`prereg_v2_results.md`](prereg_v2_results.md).
- **Realistic dose (c = 1):** damage was +0.0005, −0.0001 and +0.0024 nats at steps 160, 1000 and 1800, all below 0.01.
  The frozen outcome "no lasting damage at realistic dose" applies.
- **Dose response:** damage rises with dose in every phase and reaches 0.01 nats at c = 10 to 30.
- **Protective v (C-2):** at c ≥ 10, undoing v alone is 2.2× to 24× worse than doing nothing (6 of 6). The strict test
  across all doses fails, 9 of 11.
- **Memory writes the damage:** at c ≥ 10, undoing m+v removes 73–103% (6 of 6). The strict test fails, 8 of 11.
- **β1 ordering (C-1):** the memory share did not rise with β1 (0.72, 0.73, 0.67). D(τ) and interference (C-3) were
  not run.
- **Cooldown:** a global LR cooldown does not replace v by the frozen rule, but it recovers 25–93% of what undo-v loses.
- **Repair (C-4), c = 1, lag 3:** RML closes 68% at step 1800 (bar: 80%). At this dose the damage is too small to rank
  repairs.
- **Ruler:** 10 data-order forks at step 1800 (exploratory) have an SD of 0.0047 nats, about 170,000 times the 1-ulp
  floor there. In that unit, c = 1 damage is 0.5σ and c = 30 damage is 5.2σ.

### 7 · What is new since PLAN_V8 (2026-10-22)

| Date | Change |
|---|---|
| 10-23 | The question became the mediation question (direct vs m-echo vs v-stall) with the four core tests. Localization moved to supporting analysis |
| 10-23 | RML, BiH-Adam and MomentABFT specified as the three invented algorithms |
| 10-27 | Autopsy code (RML undo, planted probes, GPU driver, sealed report), held-out eval and the cross-process gate |
| 10-27 | First MI300X runs: determinism verified, exploratory pilot |
| 10-27 | Placebo floor (weight and gradient channels), dose arms and the global-cooldown arm added; the dose-sweep experiment written down before it runs |
| 10-28 | Ten new research directions with novelty searches: forgetting threshold, chaos-calibrated stats, v as cooldown, rollback and prevention baselines, larger models, m/v bit-flips, natural spikes, a zero-parameter migration law, a damage atlas by parameter type, ensemble-Lyapunov triage ([`p1_upgrades.md`](p1_upgrades.md)) |
| 10-28 | Capture self-check fix |
| 10-29 | Rollback-and-skip baseline at matched compute |
| 11-05 | LR schedule moved into the training loop; `lr_bump` and the cooldown arm now scale it |
| 11-05 | Bit-flip faults in g, m and v, plus a parity residual; `mve` skips the undo arms that equal B0 for m / v flips |
| 11-05 | `mve` refuses to resume a result file written under a different config |
| 11-08 | That refusal now also covers a different commit or command line; `report` keeps β1 groups per experiment and stops on two result files with one name |
| 11-10 | prereg-v2 amended, tagged and run on the MI300X (dose, undo, cooldown and repair arms, 3 phases); Δ = 0 re-passed on the changed loop |
| 11-16 | The paper tests three repair methods side by side on one stress bed: RML-1 (undo with a clean twin), RML-2 (withhold suspect coordinates until later gradients decide, [`novelty/method.md`](novelty/method.md)) and RML-3 (appeal single rows, [`novelty/rml3_method.md`](novelty/rml3_method.md)). prereg-v3 (seeds 1–7, β arms, floor curve) drafted ([`prereg_v3.md`](prereg_v3.md)) |
| 11-17 | RML-2's core (coordinate probation) is in `autopsy/probation.py` with CPU tests (PR #27, merged 11-16); not yet wired into `mve`. RML-3 narrowed to its Appeal step, held until its kill test passes (PR #25) |
| 11-20 | prereg-v3's SPAM reset schedule corrected (resets at steps 499, 999, 1499 and 1999, so P5(b) is read on the fork from step 1000 only), suggested answers to its open decisions drafted, and the SPAM warmup problem at fork start found (PR #29, open). `Bspam_clip` and `Bspam_reset` wired (PR #28, open) |
| 11-20 | RML-2 wired into `mve` as the `RML2` and `RML2@win` arms; G0's inert-hook option (`xproc_gate --probation-inert`); three code issues fixed: `sdc_grad` bit 30 planting, GPU test order, `trunk.py`'s preflight order (CPU tests only) |
| 11-20 | PRs #29 and #30 merged into main; CI installs scipy and statsmodels so `analysis/stats.py` can be tested |
| 11-20 | Damage triage proposed (PLAN_V8 R-F6, Appendix A.3.1). At a detected spike, a small model predicts whether the spike will leave lasting damage, so only those spikes get repaired. Labels come from prereg-v3's dose arms with no extra GPU runs. `repair/triage.py` and a read-only hook on `mve`'s dose arms (PR #32, merged 11-20; 102 tests green in CI on main, nothing run on a GPU) |
| 11-20 | Next MI300X session booked. Open decision 6 built: `dose_meter` logs the 10/50/90% quantiles of x next to x̄ (PR #34, CPU test; it merges if the team says yes). Arya re-invited; proposed role: attribution (`analysis/attribution.py`) |
| 11-21 | CWB built (RML-1 + w, EXPERIMENT_CATALOG A5, PLAN_V8 R-F7): a shadow AdamW state skips the spike window (`CWB@lag`, counterfactual skip-step) or steps on the clean twin's gradient (`RML-w@lag`, counterfactual B\*), then overwrites w, m and v at RML-1's lag. In `mve` behind `--cwb`; exploratory, gated offline by damage triage (CPU tests only, nothing run on a GPU) |
| 11-21 | R-F2 written (PLAN_V8 Appendix A.3): the m+v coupling law. A closed-form linear response of Adam to one spike predicts the protection share π and Y(undo-v)/Y(undo-m+v) from β1, β2, lag, the steps left and the logged dose x̄, with no fitted parameter. `theory/coupling.py`, CPU test against AdamW through the real undo hooks; prediction and scoring rule frozen in `theory/README.md` before prereg-v3 runs. Exploratory; prior art PARTIAL (Liu & Li 2026 keep the dose term out of their linear operator) |
| 11-21 | Exea Reserve runner drafted: the root `exea.md` (setup / start / resume / stop / save) and `gpu/exea_prereg_v3.sh` run prereg-v3 from its tag on Exea's MI300X in the daily 10:00–13:00 PST window. G0 runs on day 1, and each later day adds a fork that must match day 1 bit for bit. The cost rule is applied before seed 2; finished instances are skipped and unfinished ones resume branch by branch. `mve`'s results file and `snapshot.save` now write atomically. Not run anywhere |
| 11-22 | PR #37 merged (R-F2 coupling law, Exea runner; `test_coupling.py` passes in CI). PR #34 merged, so open decision 6 is answered yes: `dose_meter` logs the 10/50/90% quantiles of x. Dependabot #9–#13 held, with a comment on each, until prereg-v3 has run (pins are checked by `check_env`, and any bump needs the Δ==0 gate on the MI300X). PR #15 merged: Avyank's curvature localizer (HVP on the GPT loss, the preconditioned H̃, top-k eigsh) and a v̄ accumulator, with 10 CPU tests in CI; not run on a GPU. Follow-ups (a `jvp` HVP, two v̄ implementations) are in a comment on the PR |
| 11-22 | prereg-v3's open decisions answered in PR #42: seven seeds; Extension A (step 160) removed, so the plan uses 10.9 of 14.9 GPU-hours and the RML block's 4.0 hours get their own pre-registration; P4(a)'s [1.5, 2.5], S1's tolerances and the 0.3-nat exclusion kept; raw fork files stay committed; the SPAM split moves from P5(b) to secondary S3, read at horizon 999. PR #28 (SPAM's two halves) and PR #40 (Avyank's localizer, weeks 4–8) merged. `analysis/stats.py` gets the exact sign-flip test; `spikes/capture.py` saves a debiased v̄ at each normal step, which the localizer's driver needs, and `k2.py` records them (PR #41, merged; the localizer reads this v̄). PR #43 fixes SPAM's fork clock. CPU tests only; nothing run on a GPU |
| 11-23 | PRs #42 and #43 merged; prereg-v3 signed off by Jasraj and tagged `prereg-v3` (UTC). S3 stays secondary. The run commands were checked against `mve`'s argparse by reading, not dry-run. Nothing run on a GPU |

### 8 · Where things live

| Need | File |
|---|---|
| Full design, math, novelty ledger, all hypotheses | [`PLAN_V8.md`](../PLAN_V8.md) |
| The frozen experiment, and its results | [`prereg_v2.md`](prereg_v2.md), [`prereg_v2_results.md`](prereg_v2_results.md) |
| The ten new directions | [`p1_upgrades.md`](p1_upgrades.md) |
| Invented algorithms and experiment families | [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md) |
| Harness interfaces and guardrails | [`CLAUDE.md`](../CLAUDE.md) |
| Code layout and module status | [`research/README.md`](../research/README.md) |
| Evidence and verified citations | [`master-research/output/`](../master-research/output/), [`references.bib`](../master-research/references.bib) |
