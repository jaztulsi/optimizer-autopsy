# EXPLANATION.md — the whole project, in plain English

This document explains what this project is, why it is built the way it is, and exactly what has
been built and checked so far. It is written for a smart person who is *not* a machine-learning
specialist. Wherever a technical word is unavoidable, it is defined in a sentence right where it
first appears. Nothing here is dumbed down — it is just said plainly.

If you only remember one sentence: **when a big AI model's training goes briefly haywire, the damage
doesn't just land on the model — it also poisons the "auto-pilot" that steers the training, and this
project is trying to prove we can repair just the poisoned part instead of throwing the whole thing
away.**

---

## Part A — What this project is actually trying to figure out

### The setting: training a model, and the "auto-pilot" that steers it

To train a large language model, you show it text over and over and nudge its internal numbers
(called **weights** — the millions of adjustable dials that make up the model) so it gets a little
better each step. "A little better" is measured by the **loss**: a single number that says how
wrong the model's predictions were on the latest chunk of text. Lower loss = better. Training is
just: look at the loss, figure out which direction to nudge every dial, take a small step, repeat
millions of times.

The thing that decides *how* to nudge the dials is called the **optimizer**. The standard one is
called **Adam** (or **AdamW**, a common variant). Here is the key thing about Adam that makes this
whole project necessary: **Adam is not memoryless.** It doesn't just react to the current step. It
keeps a running summary of the recent history of the training, so it can steer smoothly instead of
jerking around. Specifically it keeps two running averages for every single dial:

- **`m`** — roughly, "which direction has this dial been moving lately, on average?" (a smoothed
  sense of momentum, so the optimizer keeps rolling in a consistent direction instead of zig-zagging).
- **`v`** — roughly, "how big and how jumpy have the recent nudges to this dial been?" Adam uses this
  to take *small, careful* steps on dials that have been jumpy and *bigger, confident* steps on dials
  that have been calm.

Together, `m` and `v` are called the **optimizer state**. Think of the optimizer as an experienced
driver and `m`/`v` as the driver's built-up sense of the road — their feel for the speed and the
curves. That built-up feel is what makes the driving smooth.

### The problem: loss spikes

Sometimes, during a long training run, the loss suddenly jumps — a **loss spike**. For a few steps
the model's predictions get dramatically worse, then (usually) recover. Spikes are common and
annoying in large-scale training; a bad one can knock a run off course or waste days of compute.

Now here is the insight this whole project is built around, and it is the part most people miss:

> **A loss spike doesn't only damage the weights. It also poisons the optimizer state (`m` and `v`).**

Go back to the driver analogy. A spike is like the car hitting a sudden patch of ice. Two things get
hurt: the car's position on the road (the weights), and — more subtly — the driver's *feel* for the
road (the `m`/`v` history). Because `m` and `v` are running averages, a violent few steps get
"baked in." For a while afterward the optimizer is steering based on a memory of chaos that already
passed. Even after the loss looks recovered, the auto-pilot is still spooked and still steering
badly. That lingering, baked-in damage is what we call the **poison**.

### The bet: repair only the poisoned part

The usual reactions to a spike are blunt: skip the bad batch, clip (shrink) the nudges, or in the
worst case reset the optimizer's memory entirely and start it fresh. Resetting the memory throws
away *all* of the driver's hard-won feel for the road just to get rid of the small poisoned part —
it's like giving your veteran driver amnesia to cure a moment of fright.

This project's bet is that we can do something far more surgical. The claim has three parts, and the
whole repository is a machine for testing them in order:

1. **The poison is real and it lives in the optimizer state, not just the weights.** A spike
   measurably damages `m`/`v` in a way that hurts the run going forward.
2. **The poison is *localizable*** — meaning it lives in only a few specific "directions" of the
   optimizer state, not smeared evenly across everything. (A "direction" here is a particular
   combination of dials that tend to move together; some of these combinations get poisoned and most
   don't.) If we can *find* which few directions are poisoned, that's the hard part.
3. **The poison is *repairable* surgically** — we can fix just those few poisoned directions and
   leave the rest of the optimizer's healthy memory intact, and this beats the blunt fixes (skip,
   clip, full reset).

In the project's shorthand these three are the three contributions:

- **C1 — the harness:** the trustworthy laboratory that lets us make *before/after* comparisons we
  can actually believe (Part B explains why this is hard and comes first).
- **C2 — the localizer:** the instrument that *finds* which directions of `m`/`v` are poisoned.
- **C3 — the repair operator:** the surgery that *fixes* just those directions.

Everything below is about building C1 first — because, as the next section explains, none of the
C2/C3 claims can even be *measured* honestly until C1 exists.

---

## Part B — Why it's built the way it is

### The core difficulty: proving cause, not coincidence

Suppose we invent a repair and the run gets better afterward. How do we know *our repair* caused the
improvement, rather than the run just happening to recover on its own, or some random difference
between the two runs? Training has a lot of built-in randomness (which examples get shown, how
numbers get shuffled on the hardware). If we run "with repair" and "without repair" as two separate
training runs, they'll differ for a hundred reasons that have nothing to do with our repair, and any
comparison is worthless. This is the central obstacle, and it dictates the entire design.

The answer is a technique the project calls **fork and replay**.

### Fork and replay

**Replay** means: run the *exact same* training run twice and get *byte-for-byte identical* results
both times. Same starting point, same data in the same order, same everything — the two runs come
out literally identical, down to the last digit.

**Fork** means: take a saved moment in a training run — a **snapshot**, which is a complete freeze-
frame of everything (all the weights, the full `m`/`v` optimizer memory, and the exact state of all
the random-number generators) — and launch *several* continuations from that identical starting
point. Like a "what-if" tree. One branch does nothing (our control). Another branch applies our
repair. Another applies a baseline fix like "skip the bad batch." Because every branch starts from
the *same* frozen snapshot and replays deterministically, **the only thing that differs between
branches is the intervention we deliberately made.** So when the branches' losses diverge, that
difference is *caused* by the intervention and nothing else. That is how you turn "it got better" into
"our change *made* it better."

An important, easy-to-get-wrong detail: the "do-nothing" control branch (the project calls it **B\***,
the clean counterfactual) must keep reading the *same* data in the *same* slots as the spike branch —
just with the spike switched off. It must **never** be implemented as "skip ahead to the next batch,"
because skipping shifts which data the model sees, and then you can no longer tell whether a
difference came from your intervention or just from the model seeing different text. Keeping the data
stream perfectly aligned is a rule the whole harness is built to respect.

### Bitwise determinism (and why "almost identical" is useless)

For fork-and-replay to work, replay has to be **bitwise deterministic** — the two runs must match to
the *exact bit*, a difference of *exactly zero*, not "very close." Here's why "very close" is not
good enough: the effects we're hunting for (the poison, the repair) can be small. If two supposedly-
identical runs already drift apart on their own by some tiny amount, we can't tell whether a small
measured difference is our repair working or just that background drift. The only way to be sure the
difference we measure is *entirely* due to our intervention is to first prove that, with *no*
intervention, the difference is *exactly zero*. That zero is the bedrock the whole project stands on.

Getting a computer — especially a GPU (the specialized chip that does the training math) — to produce
*exactly* identical results twice is surprisingly hard. GPUs are allowed to do arithmetic in slightly
different orders on different runs for speed, and that reorders tiny rounding errors, which breaks
bit-identity. So the project has a small piece of load-bearing setup code that forces the hardware
into a strict, repeatable mode. Part of this must happen *before* the math library is even switched
on, which is why that code is treated as fragile and protected: an innocent-looking "cleanup" of it
could silently break the exact-zero guarantee, and then every causal claim downstream would quietly
become untrustworthy.

### The gate discipline

Because the exact-zero property is so load-bearing, the project follows a strict rule: **no cause-and-
effect claim is allowed until the "do nothing vs. do nothing" comparison reads exactly zero at the
real scale we'll be working at.** This is the **determinism gate**. Until it passes, the laboratory
isn't trusted, so nothing measured inside it counts. This is why the early tasks are all about
building and *proving* the laboratory (C1) before any of the actual science (C2/C3) is attempted. The
project would rather move slowly and be believable than move fast and produce numbers no one can
trust.

A recurring phrase you'll see is **DoD — "Definition of Done."** It just means: the specific,
pre-agreed check a task must pass before it's allowed to be called finished. "The code runs without
crashing" is *not* a DoD; "two do-nothing runs matched to exactly zero over 50 steps on a real GPU"
*is*.

---

## Part C — What's actually been built and verified so far (Tasks 4–8)

This section walks through the work task by task. For each: **what it was trying to prove**, **what
actually happened** (with the real committed numbers), and **whether it passed** — including where a
result is only partial. The honesty here matters: some of this is a clean pass, and one of it (Task 8)
is explicitly *not* finished, and this document says so plainly rather than dressing it up.

A note on where the numbers come from: every number below is read from a committed results file in
the `results/` folder (a small machine-written record of an actual run), not from someone's memory.
All the real runs happen on a free cloud GPU (a "Tesla T4"), never on the local laptop.

### Task 4 — Can we replay a run to *exactly* zero difference? (the determinism primitive)

**What it was trying to prove.** The most basic building block of the whole laboratory: that we can
run a short stretch of training twice and get a difference of *exactly* zero — the bitwise
determinism from Part B, in its simplest form. This is called the "primitive" because it's the raw
capability everything else is built on.

**What actually happened.** A 50-step training run was executed twice on the GPU and compared step by
step. The biggest difference found between the two runs, at *every one* of the 50 steps, was
`0.0` — exactly zero, not approximately. (Recorded in `results/task4_determinism.json`: `max|Δ|=0`
over 50 steps on the GPU, where `max|Δ|` just means "the largest difference we saw anywhere.")

**Did it pass?** ✅ **Yes, cleanly.** This is the foundation stone, and it's solid.

### Task 5 — Does the training loop actually learn? (the trunk)

**What it was trying to prove.** Before hunting subtle poison, we need to confirm the basic training
loop — the project calls the main run the **trunk** — genuinely works: that loss actually goes *down*
when we train. A laboratory that can replay perfectly but can't actually learn anything would be
useless.

**What actually happened.** The small stand-in model (a "proxy" — a deliberately tiny model used for
cheap, fast experiments before spending money on a big one) was trained for 200 steps on the GPU.
The loss fell from **10.85 at the first step to 4.73 by the last step** — it more than halved, which
is exactly the healthy "it's learning" shape you want. (Recorded in `results/task5_trunk.json`.)

**Did it pass?** ✅ **Yes.** The training loop learns.

### Task 6 — Can we freeze and perfectly restore a full training moment? (the snapshot)

**What it was trying to prove.** Forking requires being able to save a *complete* freeze-frame — not
just the weights, but the entire optimizer memory (`m` and `v`) and all the random-number-generator
state — and later reload it so precisely that training resumes as if it was never interrupted. The
tricky part is bookkeeping: the optimizer's memory has to be matched back to the right weights *by
name*, tied weights (two parts of the model that deliberately share the same dials) must be handled
so they aren't double-counted, and everything must round-trip without losing a single bit.

**What actually happened.** The snapshot machinery was built to capture `{weights, m, v, step
counters, random state}`, keyed by parameter name, with tied weights handled explicitly, all stored
at full precision at proxy scale. It carries its own built-in self-check. Its real proof, though,
comes *through Task 7*: the fork gate below can only read exactly zero if the snapshot's freeze-and-
restore is itself perfect, because a fork *is* "snapshot, then continue." So the snapshot is
validated indirectly but rigorously — if it lost even one bit, Task 7 could not have read zero.

**Did it pass?** ✅ **Yes, but validated indirectly** — there is no standalone Task 6 results file;
its correctness is demonstrated by Task 7 passing at exactly zero (which is impossible unless the
snapshot round-trips perfectly). Worth being precise about: it's proven *as a consequence of* Task 7,
not by its own dedicated experiment.

### Task 7 — Does the *fork* itself replay to exactly zero? (the determinism GATE — this is C1)

**What it was trying to prove.** This is *the* gate from Part B, and the single most important
checkpoint in the project so far. Task 4 proved a plain run can replay to zero. Task 7 proves the
full **fork** path — freeze a snapshot, then launch a continuation branch from it — also replays to
exactly zero when the branch does nothing. In the project's words: **"do-nothing vs. do-nothing must
equal exactly zero"** at realistic model shapes. Until this reads zero, no cause-and-effect claim is
allowed. Passing it is what lets the project start using the word "caused."

**What actually happened.** Two do-nothing branches were forked from the same snapshot and run for 15
steps at the real proxy model shapes on the GPU. The largest difference between them, checked at
every step, was **exactly `0.0`** (recorded in `results/task7_fork_gate.json`). It was then
**re-confirmed a second time** after a later change to the training code, to make sure that change
hadn't quietly broken the guarantee — again exactly zero (`results/task8_forkgate_reconfirm.json`).

**Did it pass?** ✅ **Yes — this is the big one.** The laboratory (C1) is now trustworthy. The project
has officially cleared the gate that permits causal claims. Everything after this is allowed to talk
about cause and effect.

### Task 8 — Can we deliberately cause spikes and catch them early? (spike induction + detector)

This is the first task that is **honestly unfinished**, and this document is going to be straight
about that rather than round it up.

**What it was trying to prove.** Two things. First, that we can *deliberately* trigger loss spikes on
demand, in known, repeatable ways — because to study poison and prove a repair, we need spikes whose
exact cause and timing we *already know* (ground truth to grade ourselves against). Second, that a
simple automatic **detector** can *notice* a spike happening — ideally a couple of steps *before* the
loss actually peaks, so a future repair could act in time.

The project defined four ways to induce a spike ("recipes"), each triggered at a known step:

- **`lr_bump`** — briefly crank the learning rate (the step-size dial) way up, shoving the run over
  the edge of stability. This is the strong, reliable one.
- **`tiny_eps`** — shrink a tiny internal safety number in Adam so the "how jumpy has this dial been"
  machinery can blow up.
- **`precision`** — run a few steps in lower numerical precision (fewer digits of accuracy) to
  provoke a numerical blow-up.
- **`corrupt_batch`** — feed the model a batch of pure garbage (random nonsense text) for one step, so
  the loss shoots up on that step.

The detector's success bar (its DoD) is strict: to count, it must catch **at least 3 of the 4**
recipes, and for a caught spike it must give **at least 2 steps of early warning** before the peak
("lead ≥ 2"), while raising essentially no false alarms on the calm stretches. Two runs are used per
recipe: one to *tune* the detector's sensitivity, and a separate *held-out* run to *grade* it — so
it's graded on a spike it wasn't tuned on, which is the honest way to test.

**What actually happened — the honest version.** After two rounds of calibration, the current result
is **1 out of 4** recipes detected — the bar is 3. Concretely:

- **`lr_bump`** — ✅ works end to end. The spike reliably happens (the loss roughly tripled), and the
  detector catches it **2 steps before the peak** with **zero false alarms**. This single recipe
  fully validates the whole idea: induce → detect → get early warning. The mechanism is proven to
  work at least once.
- **`tiny_eps`** — the induced perturbation doesn't actually produce a spike at this small scale; the
  blow-up it targets needs conditions that essentially never occur here. This looks like a genuine
  "this recipe just doesn't bite at proxy scale" result, not a tuning failure — flagged as an open
  question about whether the recipe should be redefined or accepted as a documented negative.
- **`precision`** — hasn't produced a clean spike yet at the current settings; still an open recipe
  question (its injection was moved earlier in training to a spot where lower precision is more likely
  to blow up, but that's not yet resolved).
- **`corrupt_batch`** — the spike itself is real, but there's a **structural** problem with the
  *grading*, which is the most interesting finding here. A corrupt batch causes an *instantaneous*
  shock — the loss jumps on the very step the garbage is fed in — unlike `lr_bump`, whose damage
  builds up over a few steps. Because the shock and the peak land on the same step, there is *no
  earlier moment* at which a detector could have given "2 steps of warning." So holding this recipe to
  the "warn ≥ 2 steps before the peak" bar may be **asking for a warning that physically cannot
  exist**. This is now flagged in the results file itself as a real open question for the project's
  specification: the honest bar for an instantaneous recipe is probably *"detected at or before the
  peak"*, not *"detected with lead."* This is a decision to be made about the *rules*, not something
  to keep endlessly re-tuning against.

**Did it pass?** 🟡 **No — partial.** The detector currently passes for **only 1 of the 4** spike
types (`lr_bump`), against a bar of 3, so Task 8's Definition of Done is **not met**. The result is
committed honestly, labeled **"CALIBRATION-IN-PROGRESS — NOT a passed DoD,"** precisely so nobody
mistakes it for a finished result. What *is* solidly established is that the core loop works for at
least one recipe (proving the approach is sound), and the remaining three each have a clearly
diagnosed reason — one likely-inert recipe, one still-unresolved recipe, and one that may be measured
against the wrong standard.

**A footnote on the most recent attempt (infrastructure, not science).** The latest fix — a small,
correct change to grade `corrupt_batch` at its injection step (matching the earlier working behavior)
— is written, tested, and committed, but has **not yet produced a fresh graded result.** The reason is
purely mechanical: the free cloud service (Kaggle) has started handing out an older GPU model (a
"P100") that the current, version-pinned math library no longer supports, so the run crashes before it
can do any training. Earlier Task 8 runs succeeded only because the service happened to assign a
newer, supported GPU (the "T4"). Every attempt to force the newer GPU has so far still been given the
older one. This is a scheduling/hardware issue on the cloud provider's side, entirely separate from
whether the code or the science is correct — and it's the current blocker on getting Task 8 a fresh
number.

---

## Where things stand, in one honest paragraph

The **laboratory is built and trustworthy**: runs replay to exactly zero (Task 4), the training loop
learns (Task 5), full training moments freeze and restore perfectly (Task 6), and — the milestone —
the *fork* itself replays to exactly zero, which unlocks all cause-and-effect claims (Task 7, which is
contribution C1). The project is now at the **start of the actual science**: Task 8 is deliberately
inducing spikes and learning to catch them, and it's **genuinely partway** — 1 of 4 spike types fully
works (induce → detect → early warning), and the other three each have a specific, well-understood
reason they don't yet, including one (`corrupt_batch`) that has surfaced a real question about whether
the grading rule itself is fair for an instantaneous shock. Still entirely ahead, and not yet started
in earnest, is the heart of the project: the **localizer** (C2, which finds *where* the poison lives)
and the **repair operator** (C3, which fixes just that part and must beat the blunt baselines). In
short: the instrument is proven, the first measurements are underway and honestly reported as
partial, and the headline scientific claims remain to be tested.
