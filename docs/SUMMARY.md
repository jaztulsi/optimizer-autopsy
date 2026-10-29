# Optimizer Autopsy: the project in one place

*Updated 2026-10-29. Two versions: a short one (under 2,000 characters) and a long one. Both describe the
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
or both at a chosen lag and leave the weights untouched. Damage is the held-out loss gap to B\*, read against a
chaos floor from 1-ulp placebo forks. Four tests: the memory-written share (ordered by β1), protective v,
interference between two spikes, and RML repair vs skip-step and other baselines.

**What's new.** Every component has precedent: state transplants, bitwise replay, single-spike Adam math. The new
part is the causal, time-resolved split of a spike's lasting damage into direct, m-echo and v-stall parts.

**Built and run (2026-10-29).**
- Harness: Δ = 0 on CUDA and on an AMD MI300X, in-process and across processes.
- Autopsy code: RML undo, planted probes, placebo floor, GPU driver.
- First MI300X pilot (exploratory, 120 branches): width-1 spikes left 0.0003–0.0117 nats, about the size of the
  chaos floor (≤ 0.0027). In the one clear case, undoing m+v removed ~98% and undoing v alone made it worse
  (n = 1).
- Next experiment: a spike-dose sweep, effects in chaos units, a v-cooldown arm.
- Not built: localizer, repair module, theory.

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

**The next experiment** ([`prereg_v2.md`](prereg_v2.md)) is a dose sweep on `corrupt_batch`:
- planted dose c ∈ {1, 3, 10, 30};
- three phases of training (descent, plateau, late);
- none / undo-m / undo-v / undo-m+v, plus the global-cooldown arm;
- β1 ∈ {0.8, 0.95} at c = 10;
- 8 placebos per phase in the weight and gradient channels.

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
| Cross-process + co-tenancy gate through spike windows | ✅ MI300X: 3 processes bitwise equal | `research/kaggle/xproc_gate.py` |
| Spike recipes + detector (`lr_bump`, `corrupt_batch`, `repeat_ngram`, `v_inflate`, `tiny_eps`, precision) | ✅ CI; three recipes ran on the MI300X | `research/spikes/` |
| Spike capture (v̄, t0 / fork / matched-normal snapshots) | ◐ The self-check crashed on the MI300X (eps read back as a string). Fix committed (f751e14); not yet re-run on the GPU | `research/spikes/capture.py` |
| Baselines: skip, clip, resets, v̄, SPAM, ZClip, AdaGC, skip-step | ✅ CI; not yet run on a GPU | `research/baselines/` |
| Rollback-and-skip at matched compute | ◐ written and CI-tested (PR #20, not merged) | branch `baselines/rollback-skip` |
| Autopsy code: RML undo (m / v / m+v, top-k v, cooldown), planted probes, placebo floor, dose meter, GPU driver `mve`, sealed report | ◐ CI-tested; the core battery ran on the MI300X; the placebo, dose and cooldown arms have not run yet | `research/autopsy/` |
| Held-out eval (token-disjoint slots, horizon probe) | ✅ ran on the MI300X | `research/analysis/eval.py` |
| Stats (TOST, bootstrap, Holm), attribution | ◐ partial | `research/analysis/` |
| Model: GPT-2-style nanoGPT, proxy (≈ 11M non-embedding) and 124M | ✅ proxy trains (held-out loss 10.8 → 2.83 in 2,000 steps) | `research/model/` |
| LR schedule (warmup + cosine / WSD) | ⬜ constant-LR stub | `research/autopsy/schedule.py` |
| Localizer (which coordinates carry the damage), repair operator, theory | ⬜ stubs. RML covers part of the repair. The localizer's output format is fixed in PLAN_V8 §6.4 | `research/localizer/`, `repair/`, `theory/` |
| Factorial, causal oracle, faults (bit-flip) modules; GPT-NeoX + Pythia state loader; QK-Norm / μP variants; figures | ⬜ not written | — |
| Toy Hessian-vector product (double-backward) | ◐ written, needs fixes (PR #15) | — |

**Known code issues:**
- `lr_bump` sets an absolute LR instead of multiplying the schedule. That is harmless at constant LR.
- `fork.py:351` still records every exception as a branch failure.
- The box runs torch 2.14 while the repo pins 2.10, so the pilot skipped the environment preflight.

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
- **Pilot** [V, exploratory]: 120 of 120 branches with 0 crashes. Width-1 spikes at step 160, seed 0, constant LR,
  with `corrupt_batch` and `repeat_ngram` at β1 = 0.9, `lr_bump`, and `corrupt_batch` at β1 = 0.8 and 0.95.
  - **Spikes this small mostly wash out.** Lasting damage was 0.0003–0.0117 nats, while the 1-ulp chaos scale
    (+G planted vs B0) reached 0.0027.
  - **One clear case:** `corrupt_batch` at β1 = 0.8 (Δ = 0.0117). Undoing m+v at τ = 0 removed ~98% of the
    damage, and undoing v alone made it worse (0.0134). That is the sign C-2 predicts, with n = 1.
  - Its thresholds were fixed after the data existed, so it is a pilot and confirms nothing.
- **What it changed:** a measured chaos floor comes before any reading of an effect, and the spike dose has to
  rise until damage clears that floor. Hence the dose sweep in §3.

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

### 8 · Where things live

| Need | File |
|---|---|
| Full design, math, novelty ledger, all hypotheses | [`PLAN_V8.md`](../PLAN_V8.md) |
| Next experiment, written down before it runs | [`prereg_v2.md`](prereg_v2.md) |
| The ten new directions | [`p1_upgrades.md`](p1_upgrades.md) |
| Invented algorithms and experiment families | [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md) |
| Harness interfaces and guardrails | [`CLAUDE.md`](../CLAUDE.md) |
| Code layout and module status | [`research/README.md`](../research/README.md) |
| Evidence and verified citations | [`master-research/output/`](../master-research/output/), [`references.bib`](../master-research/references.bib) |
