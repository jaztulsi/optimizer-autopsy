# OPTIMIZER AUTOPSY — PLAN V6
### Complete project reference · AMD-only restart · audited GPU ask ~15–40 GPU-h (~500 build-effort h) · reallocated budget, go/no-go gate, and a written cut order
### Self-contained: everything needed to understand and continue this project is in this one file.

> **Governing plan.** V6 is the current source of truth for hardware, compute, budget, timeline, and
> Definitions of Done. The 26 task-level build prompts in `BUILD_PLAN.md` remain the per-file science
> detail, but where they disagree with V6 on hardware (AMD, not Kaggle CUDA), on the spike DoD
> (≥2 recipes, not 4), or on the front-of-Phase-1 go/no-go gate, V6 wins. Current banked state is in
> `context-ai.md` §14; the plain-English mirror is `EXPLANATION.md`.

---

## 0 · What changed from V5, and why

Quick changelog for anyone who read V5:

- **Budget reallocated, not expanded.** Same 500-hour cap, different split. Localizer, repair, and baselines
  gained hours; short-fork calibration and the full attribution battery gave some back; contingency grew
  from 8% to 12% of the total. See §6.
- **A go/no-go gate now sits at the very front of Phase 1**, before the 30-hour AMD determinism
  re-verification is spent in full: a small smoke test on a single forked pair, checked against the exact
  operations this pipeline actually uses (Hessian-vector products, Adam moment updates), not assumed just
  because a determinism flag exists somewhere in the stack. See §6, §8.
- **Phase 2's Definition of Done changed** from "all four spike recipes working" to "at least two solid
  recipes." Both the short-fork calibration and the full attribution battery now explicitly scale with
  whatever recipe count survives — V5 only made the battery adaptive, not the calibration. See §6.
- **A pre-registered decision protocol now exists** for the two places in this project where an ambiguous
  result could get quietly nudged toward the more exciting outcome: the cheap-fix kill-test, and the ψ_k
  repair-sufficiency threshold from §3.8. See §8.
- **A cut order exists in writing**, ranked, decided now rather than mid-crunch. See §6.
- **Three open items now block finalizing this budget**, listed explicitly rather than assumed away: the
  team-size figure needs to resolve to one accurate number; Exea Labs' realistic single-project allocation
  needs confirming rather than assumed at 500–600; GPU parallelism and expiration terms need confirming,
  because they determine whether GPU-hours or personal dev-hours is the actual binding constraint. See §5.
- **Three budget tiers now exist** (Floor / Core / Stretch) so this plan doesn't need to be renegotiated
  from scratch if the eventual grant lands smaller or larger than hoped. See §7.
- **§9's timeline claims are corrected.** The NeurIPS 2026 dates are confirmed accurate. The "NeurIPS 2027,
  roughly 8–9 months out" line is a reasonable prediction, not a verified date — the actual pattern (2024:
  May 22 → 2025: May 16 → 2026: May 6) has been moving earlier every year, not holding steady.

Everything else — the core thesis, the math in §3, the C1 rationale in §2, the codebase status in §12 —
carries forward from V5 essentially unchanged, because none of it was where the actual risk lived.

---

## 1 · The project, in plain terms

Pretraining a large model means repeatedly measuring how wrong its predictions are (the **loss**) and
nudging every internal weight to be slightly more accurate. Nearly every large model uses **Adam** or
**AdamW**, which is not memoryless: for every weight it keeps two running averages that persist across
steps — **`m`** (a smoothed sense of recent direction) and **`v`** (a smoothed sense of recent volatility).
Together these are the **optimizer state**.

Occasionally a **loss spike** hits — a sudden jump in loss — and the run is either destroyed or left
degraded for a long stretch afterward. Current practice routes around a spike (skip the batch, clip the
gradient, or roll back to an earlier checkpoint and discard everything since) rather than diagnosing it.

**This project's thesis:** a spike plausibly corrupts `m` and `v` in ways that persist after the weights
themselves recover, and that damage is concentrated in a small number of specific directions rather than
spread evenly. If true, the damaged directions can be found and repaired surgically — recovering the run
as well as a full memory reset while discarding almost none of the optimizer's healthy accumulated memory.

**The three contributions:**
- **C1 — the instrument.** A harness that freezes a training moment and relaunches multiple parallel
  continuations from that identical frozen point, so the only way branches differ is through a deliberate,
  named intervention. `research/harness/`.
- **C2 — the localizer.** Identifies which specific directions of `m`/`v` are damaged at a detected
  failure point. `research/localizer/` — not yet built.
- **C3 — the repair.** Corrects only the identified damaged directions, proven via C1 to beat both doing
  nothing and a full reset. `research/repair/` — not yet built.

---

## 2 · Why the instrument (C1) is built the way it is

**The core problem:** if a run improves after a repair, that could be the repair, the run recovering on
its own, or noise between two separately-run trainings. Only **fork and replay** distinguishes these:
**replay** means two identical runs produce byte-for-byte identical results; **fork** means launching
multiple continuations from one frozen **snapshot** (weights, `m`/`v`, and every RNG state), so branches
can only differ through their deliberate intervention. The clean control branch must read the exact same
data in the exact same order as the branch it's compared against — never "skip ahead," which would
confound the measurement with a difference in training data.

**Why exact-zero, not approximately-zero:** if two untouched branches drift apart on their own by any
tiny background amount, there's no way to know whether a later measured difference is real or noise. GPUs
are normally allowed to reorder arithmetic for speed, which breaks bit-identity — this project includes a
protected setup routine forcing strict, repeatable execution, part of which must run before the deep
learning framework's math library initializes. This code is fragile: an innocuous cleanup could silently
break the guarantee, after which every later causal claim in the project is quietly untrustworthy.

**The gate rule:** no cause-and-effect claim is permitted until "do-nothing vs. do-nothing" reads exactly
zero, at real model scale. Clearing this on the original (CUDA) hardware was the project's biggest
completed milestone before this restart. **On AMD, this guarantee is being re-earned, not assumed** — see
the go/no-go protocol in §6 and §8 before any further hours are spent on it.

One implementation detail worth flagging explicitly for the localizer build in C2: the eigensolve tooling
under consideration appears to run its actual iterative solve on the host (via scipy), farming only the
matrix-vector products themselves out to the GPU. If that holds, the determinism burden narrows
considerably — you need one Hessian-vector product (a single forward+backward pass) to be bit-reproducible,
not the entire multi-step solve. Confirm this before assuming the harder version of the problem.

**Definition of Done (DoD):** the specific, pre-agreed check a piece of work must pass before being called
finished. "Two untouched forked branches matched to exactly zero difference over 15+ steps, at real model
dimensions, on real GPU hardware" is a DoD. "It ran without crashing" is not.

---

## 3 · The mathematics

**3.1 AdamW.** Per coordinate at step `t`: `m ← β1·m + (1−β1)·g`, `v ← β2·v + (1−β2)·g²`, bias-corrected
`m̂ = m/(1−β1^t)`, `v̂ = v/(1−β2^t)`, update `θ ← θ − η·(m̂/(√v̂+ε) + λ_wd·θ)`. `betas/eps/wd` must match
exactly across the original run and every forked branch, or bitwise replay breaks silently.

**3.2 Why `v` is the primary suspect.** A spike inflates `v` in a few directions; since the update divides
by `√v`, an over-inflated `v` shrinks the effective step size there for roughly `1/(1−β2)` steps. If `v` is
corrupted downward toward underflow instead, `1/(√v+ε)` can explode. `m`/`v` are stored at full 32-bit
precision for the exact-zero proof; a compact 16-bit format is reserved only for a later, larger run, and
must never risk the underflow explosion.

**3.3 The geometric basis.** Adam behaves locally like preconditioned gradient descent in the metric
`D = diag(√v + ε)`, so the operator governing local stability is the preconditioned Hessian
`H~ = D^(-1/2) H D^(-1/2)`, not the raw loss Hessian. This project's working hypothesis: spikes originate,
and poison concentrates, in `H~`'s top eigenvectors. Localization and repair operate in this
frozen-at-failure-point eigenbasis.

**3.4 Hessian-vector products without forming the Hessian.** Only Hessian-times-vector is ever computed,
via a double-backward pass through the same autodiff machinery already used for training (Pearlmutter's
trick), sandwiched with the cheap diagonal `D^(-1/2)` term for the preconditioned version.

**3.5 Extracting the top eigenpairs.** At proxy scale, a Lanczos-based solver recovers the top-k eigenpairs
from matvec access alone. At larger scale, a GPU-native iterative method is required instead — the small
solver's memory footprint won't fit. One fixed probe batch must be used across every matvec within a
single solve; a different batch per product makes the operator non-symmetric and the solver silently
returns noise instead of an error.

**3.6 The poison score.** Maintaining a pre-spike running average `v̄`, at the detected failure step and
within the frozen top-k eigenbasis: a per-direction poison score compares how far `v` has moved from `v̄`;
a spectral-mass statistic `ψ_k` describes what fraction of total movement is captured within the top-k
subspace. Combined with a directional signal-to-noise t-statistic, poisoned directions are those with
both high poison score and low signal. Thresholds are derived from matched *normal* training steps, never
set as arbitrary constants.

**3.7 The repair operator.** A rank-limited projection removing the poison component of `m`/`v` (and
optionally `w`) exactly along identified poisoned directions, ramped in gradually, with `v` clamped
non-negative. Deliberately a projection onto a small identified subspace, not coordinate-wise.

**3.8 When selective repair suffices.** Modeling a spike's perturbation in each eigendirection as a
second-order recursion, each direction has a spectral radius: below one, it decays on its own; above one,
it grows. The central theoretical claim, expressed via `ψ_k`, predicts whether a rank-limited repair drives
every direction below one (matching a full reset) or whether the undamaged bulk carries too much mass for
any selective repair to work. Caveat, stated from the start: this assumes the raw Hessian and the
preconditioner are simultaneously diagonalizable, which isn't exactly true — the commutator (coupling)
error from this assumption is measured and reported explicitly. **§8 defines, in writing, what a
convincing ψ_k signal looks like before this criterion is used to decide anything.**

**3.9 Causal effect and statistics.** Every reported effect is
`Δ = held-out validation loss(branch) − held-out validation loss(clean counterfactual)`, using held-out
data so a branch can't appear to "recover" by memorizing more of the training stream. The repaired branch
and its random-subspace control share the same seed and data — a **paired** measurement, cancelling
shared variance and substantially more sensitive than treating them as independent samples.

---

## 4 · Path A vs. Path B

**Path A — full ground-up rebuild.** Rewrite everything from a blank starting point, treating prior CUDA
work as informal prior art only. **Cost:** an estimated 150–250 hours of pure re-implementation on top of
everything else, producing no new scientific insight. **Realistic total: 650–900+ hours — exceeds the cap.**

**Path B — port existing logic, re-earn only the hardware-specific guarantee (recommended).** The
already-correct, hardware-independent code (tokenization, model architecture, spike-triggering recipes,
the fork experimental design) carries over unchanged. Only genuinely hardware-specific pieces are
re-derived: the strict-mode setup routine, and the exact-zero replay proof itself, both fresh on AMD.
**Cost:** 30 hours (see §6's go/no-go framing for why this number needs a checkpoint, not blind trust).

**Risk, stated honestly:** any subtle flaw already present in the original design carries forward rather
than getting caught by a rewrite. Judged low, since the design has been reviewed carefully multiple times
— but that judgment comes from the same person who wrote the original code. **Cheap insurance:** before
fully committing to Path B, spend a few hours having the PhD mentor (or another qualified reader) read the
fork/branch design fresh, specifically hunting for confounds, rather than relying solely on self-review.

**Recommendation: Path B**, unchanged from V5. It is also the right call against what is already banked:
Tasks 0–7 (determinism primitive, trunk, snapshot, and the fork `Δ==0` gate) are done, GPU-verified, and
committed with evidence under `results/` — re-deriving them from scratch would reproduce reviewed work.

---

## 5 · Three things to resolve before this budget is final

**1. The team-size figure.** Different numbers have circulated for this team (five, with a mentor; eight,
per outreach framing) against a commit history showing one author, unconfirmed against real roles. This
may have an innocent explanation — a mentor or collaborators contributing to design rather than commits —
but it needs to resolve to one accurate figure before it appears in any funding or compute request. A
mismatch here costs nothing to fix now and costs real credibility with a compute partner if it's ever
checked later.

**2. Exea Labs' realistic single-project allocation.** Exea Labs is a legitimate, AMD-backed program built
specifically to give high-school researchers free GPU access, and by its own count spans several hundred
researchers across dozens of active projects. That scale is good news for legitimacy — it also means
500–600 hours shouldn't be assumed as the default single-project grant. Ask directly what a realistic
allocation looks like for a project at this budget's scope.

**3. GPU parallelism and grant expiration.** "500–600 GPU-hours" and "24 weeks" are only the same
constraint if there's one GPU running continuously. Ask two specific questions: how many GPUs can run in
parallel, and is there an expiration window on the hours? If it's one GPU with no expiry, §9's ~20–25
hour/week pacing is right. If it's several GPUs, the schedule compresses and the real constraint becomes
personal dev-hours available around school — worth stating that as its own explicit line item once known
(e.g., "X hours/week available"), rather than quietly assuming unlimited personal bandwidth.

---

## 6 · The V6 budget: ~500 build-effort hours; audited AMD GPU ask ~15–40 GPU-h

> **⚠ AUDITED 2026-08-31 — the compute ask is ~15–40 GPU-hours, not 500.** A bottom-up recount of the
> table below, grounded in a **measured 0.0619 s/step** at proxy scale (11M params, batch 16, Tesla P100;
> committed under `research/kaggle/step_timer_results.md`) and a code audit of which line items are actually
> GPU-bound vs engineering-bound, found that the "500" conflated two different resources: **build-effort
> (person-hours)** and **GPU-compute**. The table's hours are best read as **build-effort**, which drives
> the ~24-week wall-clock. The real **AMD GPU-compute ask is ~15–40 GPU-h**: ~5–10h to re-earn bit-exact
> determinism on AMD/ROCm (the one line nothing has tested — every measurement so far ran on NVIDIA/CUDA
> free-tier Kaggle; a P100 grad_accum=40 gate has since confirmed max|Δ|=0 on **CUDA**, not AMD), plus
> ~3–30h of proxy-scale GPU-bound science (short-fork calibration ~2–14h + the attribution battery
> ~0.4–4.8h; the rest is dev-validation runs). **The 124M robustness battery (~71–302 GPU-h/battery) stays
> on free-tier Kaggle over multiple weeks per Task 22 — off the AMD ask.** Full per-line breakdown and the
> measured numbers behind it: `research/kaggle/step_timer_results.md`.

The line below are **build-effort estimates** (person-hours), not GPU-hours; see the audit note above for the GPU-compute figures.

| Phase | V5 hours | V6 hours | Why it changed |
|---|---|---|---|
| AMD determinism re-verification | 30 | 30 | Same total — but see the go/no-go gate below before spending it |
| Finishing spike induction + detection | 30 | 30 | Same total — Definition of Done changed to "≥2 solid recipes," not 4 |
| The cheap-fix kill-test | 15 | 15 | Same — but run only after the §8 threshold is written down |
| Building the localizer (C2) | 60 | 70 | Never-built, highest scientific risk in the project — was under-resourced relative to novelty |
| Building the repair operator (C3) | 50 | 60 | Same reasoning as the localizer |
| Reimplementing comparison baselines | 35 | 45 | ~7–9h/method to faithfully reproduce 4–5 published methods was tight |
| Calibrating a short-fork proxy | 80 | 55 | An engineering-efficiency step, not core science — use a faster heuristic pilot |
| The full attribution battery | 140 | 120 | Back-solve with the formula below before locking this; treat as a ceiling |
| Measuring the theoretical correction term | 20 | 15 | Valuable, but shouldn't compete with the two still-unbuilt core components |
| Contingency | 40 | 60 | 8% → 12% of total; an AMD port plus two never-built components deserves more than a routine-migration buffer |
| **Total** | **500** | **500** | |

**If 600 hours materializes:** don't put the extra 100 entirely into calibration/battery as V5 originally
proposed. Split it: +40 contingency, +30 localizer/repair, +30 calibration/battery.

**The go/no-go gate on Phase 1 (new in V6).** Before spending the full 30-hour determinism line: run a
~6-hour smoke test — one forked pair, run one training step, compare `m`, `v`, and weights bit-for-bit.
Check it against the specific operations this pipeline actually touches (the Hessian-vector product
double-backward pass, the Adam moment update), not just whatever operation category AMD's own release
notes happen to cover that quarter. If it fails cleanly, that's the signal to pivot Phase 1's remaining
24 hours toward a tolerance-based statistical framework (paired seeds, many-run averaging) instead of a
single-run bitwise proof, rather than discovering the wall in week three of a nominally-solved 30-hour
phase.

**Back-solve formula for the full attribution battery (new in V6).** This document is meant to be
self-contained, but the one number every hour estimate in this table ultimately depends on — measured
seconds-per-step at proxy scale and at the larger scale — isn't stated anywhere. Pull it from the prior
CUDA runs' logs (they are committed under `results/`), then check:

```
(recipe count) × (branches per site: control + repair + baselines + random-subspace control, ≈7)
  × (seeds per branch) × (calibrated fork length in steps) × (seconds/step) ÷ 3600
  ≤ 120 hours
```

If 5 seeds don't fit with 2 recipes at the calibrated fork length, that's the moment to trim seeds or
sites — not after most of the hours are already spent.

**A cut order, decided now, not mid-crunch:**
1. **First cut:** shrink to whichever spike recipes actually work (2, not 4) — this alone shrinks
   calibration and the battery without threatening the core causal claim.
2. **Second cut:** reduce seed count in the full battery. Less statistical power, honestly flagged as a
   limitation in the writeup.
3. **Third cut:** drop the two adaptive-clipping baselines. Keep skip+clip and naive full-reset — the two
   any reader will expect regardless.
4. **Fourth cut:** measure the commutator error once (as V4 did), not across every confirmed spike site.
5. **Never cut:** the exact-zero replay proof, or the held-out validation methodology. Lose either and no
   claim in the paper is trustworthy.

---

## 7 · Three versions of this plan, depending on what actually materializes

These tiers are **scope / build-effort** tiers (person-hours), not GPU-hours — the audited GPU-compute
ask is ~15–40 GPU-h at every tier (see §6's audit note and `research/kaggle/step_timer_results.md`).

| Version | Build-effort h | Scope | Deliverable |
|---|---|---|---|
| **Floor** | ~150–200 | Determinism + go/no-go + 2 recipes + kill-test + a cruder heuristic-subspace repair (skip the full eigensolve if it isn't cheap on AMD) + 1 baseline comparison + a 1–2-site battery | The "cheap fix wins" or "why global reset wins" paper — still real, still TMLR-shaped |
| **Core** | 500 | Full C1→C2→C3 pipeline, reallocated as in §6 | Benchmark paper or full paper, decided by what the kill-test and ψ_k criterion actually show |
| **Stretch** | 600 | Core + deeper contingency + a real multi-site commutator-error characterization | The strongest honest version of the full paper |

Whichever tier ends up funded, run the go/no-go determinism smoke test in week one regardless — its
result should decide which tier you're actually building toward, and it's far cheaper to learn that in
week one than week eight.

---

## 8 · Pre-registered decision protocol

Two places in this project are exactly where an ambiguous result could get quietly nudged toward whichever
outcome feels more exciting — not through dishonesty, just through the ordinary human pull of the
"outcome-robust, there's a paper either way" framing in §1. Writing the thresholds down now, before either
test runs, removes that pull.

**Cheap-fix kill-test.** Before running it, define in writing: what held-out-loss recovery counts as
"the cheap fix already wins" (repair work becomes unnecessary — a legitimate, valuable finding on its
own), what counts as "the cheap fix clearly falls short" (repair work is justified), and what counts as
genuinely ambiguous — and what happens in the ambiguous case (e.g., default to proceeding with the
repair build, treating the kill-test as inconclusive rather than as a win for either side).

**The ψ_k repair-sufficiency criterion (§3.8).** Before computing it on real data, define in writing what
value of `ψ_k`, measured against how many matched normal-training steps, counts as a convincing signal
that selective repair should suffice — versus a signal that only a full reset will work. Report the
commutator (coupling) error alongside it every time, not just when it's small enough to be flattering.

**Stop-and-write-up trigger.** Regardless of hours remaining, define now what "good enough to stop and
write up" looks like for each of the three outcome branches from §1, so there's a pre-committed answer
when the localizer or repair is close-but-not-quite-there near the end of the budget — rather than an
open-ended pull to keep going past the cap the plan itself set.

**Phase-overrun trigger.** If actual hours spent on any phase exceed roughly 1.3x its budgeted line
before that phase's Definition of Done is met, that's an automatic checkpoint to formally reassess scope
— not a cue to push through on momentum. (Every prior version of this plan, V3 through V5, has needed
external review to catch its own over-optimism; this is the mechanism for V6 to catch it internally.)

---

## 9 · Realistic timeline

As of this writing, late August 2026: the **NeurIPS 2026 main-track deadline (May 6, 2026) has passed** —
confirmed against the official call for papers. There is no single fixed "NeurIPS workshop deadline";
NeurIPS notifies organizers of accepted workshop proposals by September 29, 2026, and individual workshops
set their own paper deadlines around that window — if a specific workshop's deadline is in view, confirm
it on that workshop's own page. **NeurIPS 2027's deadline is not yet published**; conference trackers list
it as a prediction based on the last three years' pattern (2024: May 22 → 2025: May 16 → 2026: May 6),
which has been moving earlier each cycle, not holding steady — treat "roughly 8–9 months from now" as a
reasonable bet with a few weeks of slack, not a locked date. **TMLR** remains a strong structural fit and
its own turnaround claims check out: reviews within about four weeks, a decision within about two months.

**Revised phase timeline (Core / 500h version):**
- **Week 1:** Go/no-go determinism smoke test (6h) — decision point on bitwise vs. statistical replay.
- **Weeks 1–3:** Remaining AMD re-verification (24h) + finishing spike induction to ≥2 recipes (30h).
- **Weeks 4–5:** Cheap-fix kill-test (15h), against the §8 pre-registered threshold.
- **Weeks 6–12:** Localizer (70h) + repair operator (60h) + baselines (45h).
- **Weeks 13–15:** Short-fork calibration (55h), back-solved via the §6 formula.
- **Weeks 16–20:** Full attribution battery (120h) + commutator-error measurement (15h).
- **Weeks 21–24:** Writing, internal review, submission to TMLR and/or NeurIPS 2027 prep.

---

## 10 · Honest scoring, with no inflation

No reallocation of a fixed compute budget produces a guaranteed high conference acceptance rate. Review
outcomes are genuinely noisy even for strong papers, and the very top tier of conference outcomes requires
a different category of contribution than a well-executed, well-combined application of existing tools to
a new but scoped question — which is an accurate description of this project even at its most successful.
This project's realistic ceiling, executed as well as the budget allows, is a genuinely strong,
publishable, technically sound single contribution — the kind TMLR is explicitly built to reward, and a
reasonable NeurIPS Datasets & Benchmarks or workshop candidate. Not, and this plan does not claim it to
be, a near-certain top-tier acceptance.

---

## 11 · Team, resources, and current outreach status

**Team:** open item — see §5.1. Use a conservative single-contributor-plus-mentor assumption for
scheduling until a verified team with assigned roles is confirmed and reconciled across every place it's
been stated.

**Compute:** requesting AMD GPU-hours (MI300X or equivalent) from Exea Labs — the **audited ask is
~15–40 GPU-h** (see §6's audit note; the old 500–600 was build-effort, not GPU-compute). See §5.2 and §5.3
for the questions to resolve (realistic allocation size, parallelism, expiration) before this budget's
tier (§7) is locked. Azure credits remain in reserve for backup compute and checkpoint storage; get the
actual amount specified rather than leaving it open-ended. No OpenAI credits are needed — the pipeline
uses local, offline tokenization only.

**Current outreach status:** an inquiry has been sent to an Exea Labs founder. Nothing in §6 or §7 above
should be treated as locked until that inquiry resolves and §5's three open items are answered.

---

## 12 · Codebase and directory status

Unchanged from V5 except where noted. Current banked evidence is committed under `results/` (per-task JSON
+ Kaggle logs) and mirrored in `context-ai.md` §14 and `EXPLANATION.md` Part C.

- **Strict-mode replay setup and exact-zero proof** — done and verified on original CUDA hardware
  (`results/task4_determinism.json`, `max|Δ|=0` over 50 steps on a Kaggle T4). Must be re-earned fresh on
  AMD via the §6 go/no-go gate before the remainder of the 30-hour line is spent.
- **Training loop, tokenized-data pipeline, model architecture, freeze-and-restore, fork-and-compare** —
  done and verified, carried forward under Path B. The fork `Δ==0` gate (C1) reads
  `max|Δ|=0` on T4 (`results/task7_fork_gate.json`, re-confirmed in `task8_forkgate_reconfirm.json`).
- **Deliberate spike-triggering and detection** — V6's two-recipe policy gate is implemented. `lr_bump`
  qualifies predictively with two-step lead; instantaneous `corrupt_batch` uses a distinct zero-lead
  onset policy and never earns an early-warning claim. Regrading the committed held-out T4 evidence
  qualifies 2/4 with no false positives, but the policy was resolved after that run was inspected, so
  `results/task8_v6_policy_regrade.json` is explicitly provisional pending a fresh prospective GPU run.
  `tiny_eps` remains too weak at proxy scale and `precision` has not produced a clean spike.
- **The cheap-fix kill-test** — built, passing internal logic checks, never run on real data. **Run it
  only after the §8 threshold is written down.**
- **The localizer, the repair operator, comparison baselines, statistical analysis/figures, the theory
  notebook** — not yet built.
- **New in V6:** the go/no-go smoke test itself (§6) — not yet built; should be the first thing built on
  AMD hardware, before anything else in Phase 1.

---

## 13 · Glossary

Carried forward from V5, plus:

- **Go/no-go gate** — a cheap, fast checkpoint run before committing to a larger, more expensive line item,
  specifically designed to surface a fatal blocker early rather than late.
- **Descoping ladder / cut order** — a ranked list of what to trim first if hours run short, decided in
  advance so the decision isn't made under time pressure.
- **Pre-registration** — writing down, before a result is known, what will count as each possible outcome,
  to prevent an ambiguous result from being interpreted after the fact toward whichever conclusion is more
  convenient.

(All other terms — weights, loss, loss spike, optimizer state, snapshot, fork, bitwise determinism,
localizer, repair operator, held-out validation loss, paired statistics — as defined in V5 / the
`context-ai.md` §18 glossary.)

---

## 14 · Version history

- **V1/V2:** original 16-week solo-oriented execution plan; core science design.
- **V3:** 8-person, 26-week, two-paper expansion; self-assessed at 90/100, correctly noting a top-tier
  outcome would need a larger theoretical claim, not more scope. Team size never independently confirmed
  against authorship history.
- **V4:** reallocated V3's scope to a 500-hour ceiling; correctly matched claim size to budget, but
  omitted the cheap-fix kill-test and assumed a spike-recipe count not yet achieved.
- **V5:** full restart on AMD hardware, reconciling V4's budget-matching instinct with the project's
  actual state; grounded timeline claims in verified dates.
- **V6 (this document):** reallocates V5's budget toward the genuinely novel, highest-risk components
  (localizer, repair, baselines); adds a go/no-go gate ahead of the AMD determinism re-verification; adds
  a written cut order and pre-registered decision protocol; surfaces three open items (team size, realistic
  Exea Labs allocation, GPU parallelism/expiry) that block finalizing the budget tier; adds three explicit
  budget tiers so the plan survives a smaller-or-larger-than-hoped compute grant without a rewrite;
  corrects the NeurIPS 2027 timeline claim from "verified" to "a reasonable, historically-grounded
  prediction."
- **V6.1 audit (2026-08-31):** bottom-up recount of the §6 budget against measured proxy step-time
  (0.0619 s/step, P100) and a code audit of GPU-bound vs engineering-bound line items. Finding: the
  "500–600 GPU-hours" conflated **build-effort (person-hours)** with **GPU-compute**; the real **AMD GPU
  ask is ~15–40 GPU-h** (~5–10h AMD/ROCm determinism re-verification + ~3–30h proxy-scale GPU-bound
  science). The 124M robustness battery (~71–302 GPU-h/battery) is confirmed off the AMD ask (free-tier
  Kaggle, Task 22). Also measured: 124M step-time 1.08 s/step (batch 8, P100) and a P100 grad_accum=40
  determinism gate at max|Δ|=0 (on CUDA — AMD/ROCm still untested). Data + full breakdown:
  `research/kaggle/step_timer_results.md`. Numbers propagated to `index.html`, `README.md`, and the docs.

*End of document.*
