# UPDATE — where the project is, in plain English

This is your simple status page. No jargon. It answers three questions: **what's done**, **what I'm
doing right now**, and **what's next**. (The nerdy task-by-task checklist is at the very bottom if you
ever want it.)

**Last updated:** 2026-08-02

---

## The one-line version

The project has a solid foundation and just got its most important safety check working. We're
**~17% of the way** through the build. Everything built so far **works and is tested**. Nothing is
waiting on you right now.

---

## What is this project again? (the plain version)

Big AI models sometimes "blow up" during training — the loss suddenly spikes and the run gets
damaged. Our project is like a **detective + surgeon** for that moment:

1. **Rewind** training to the instant it broke.
2. **Find** exactly which part of the model's "memory" got poisoned.
3. **Repair** just that part and prove the repair is what fixed it.

To do that honestly, we need to be able to **replay training perfectly** — run it twice and get the
*exact* same numbers. That "perfect replay" is the thing we just got working today.

---

## ✅ What you've accomplished so far

Think of this as laying the foundation of a house. It's done and inspected:

1. **All your accounts and keys are set up.** Kaggle (free GPUs), Hugging Face (cloud storage),
   Weights & Biases (charts), Colab. Your secret keys are safely stored — never on your laptop,
   never in the code.
2. **A private cloud storage box exists** for the big files (`optimizer-autopsy-artifacts`).
3. **The code knows how to run on Kaggle's free GPUs** and checks the environment is correct before
   it starts — we confirmed this on a real Kaggle GPU (a Tesla T4). It printed "OK".
4. **The data pipeline is built.** It turns text into the numbers the model trains on, and — this is
   important — it can hand back the *exact same* batch of data every time, which is what makes
   perfect replay possible. We proved it gives identical results across two separate runs.
5. **Secrets are safe.** There's an automatic guard that scans the whole project and refuses to let a
   password or key get saved into it by accident.
6. **🔑 Perfect replay works (today's big one).** We can now take a snapshot of a model mid-training,
   run it forward 50 steps, rewind, run it again — and get *byte-for-byte identical* results. Zero
   difference. This is the single most important piece: without it, none of the "find the poison"
   science would be trustworthy.

---

## 🔨 What I'm doing right now

I just finished **#6 above (perfect replay)** and tested it on your laptop's CPU — it passed with
zero difference at every step.

**The catch:** the plan requires this same test to also pass on a real **Kaggle GPU** (GPUs do math
slightly differently than CPUs, so we can't assume it works there just because it works locally).
So the next small thing I need from you is to run one command on Kaggle — I've put it in `todo.md`.

After that, I move on to building the **model itself** and the **training loop** (the engine that
actually trains the mini AI we'll experiment on).

---

## ⏭️ What's next (in order)

1. **You:** run the replay test on Kaggle GPU (see `todo.md`) so we know it works there too.
2. **Me:** build the small practice model + the training loop.
3. **Me:** build the "snapshot" system (save/reload a model's full state to the cloud).
4. **Me:** build the "fork" system — the heart of the project, where we test a fix and prove it
   worked by comparing against a perfect replay.

---

## Progress bar

```
Foundation  ████████████████████  DONE (4 of 4 steps)
The engine  ██░░░░░░░░░░░░░░░░░░░  starting now
Everything else  ░░░░░░░░░░░░░░░░  not started
```

**Overall: ~17% built. Status: 🟢 everything passing.**

---

## Anything I should worry about?

- Kaggle's free computers have **newer software** than the plan expected (that's fine, just noted so
  nothing surprises us later).
- The replay test passed on CPU; **still needs the Kaggle-GPU thumbs-up** (that's your one to-do).
- Everything else is green.

---
---

## Appendix: the technical checklist (skip unless you want detail)

Legend: ✅ done · 🟡 partial · ⬜ not started · ⛔ needs paid GPUs

**Phase 0 · Foundation — 4/4 ✅**
- ✅ Task 0 — repo scaffold (imports clean)
- ✅ Task 1 — env check `check_env()` (passes on Kaggle T4)
- ✅ Task 2 — fixed tokenized data + `get_batch` (bitwise-identical across two processes)
- ✅ Task 3 — secrets loader + no-token-in-git test (3/3 pass)

**Phase 1 · The instrument — 1/4**
- ✅ Task 4 — deterministic replay + test (CPU **PASS**, max|Δ|=0 over 50 steps; Kaggle-GPU run pending — `todo.md`)
- ⬜ Task 5 — proxy nanoGPT model + trunk training loop  ← *next*
- ⬜ Task 6 — snapshot/restore of (weights + optimizer + RNG) to HF Hub
- ⬜ Task 7 — fork driver + the Δ==0 gate

**Phases 2–6 — not started:** spike induction, kill-test (T8–9), localizer (T10–12), repair +
baselines (T13–15), attribution + theory (T16–19), scale + figures + paper (T20–24). ⛔ Task 25
(410M) needs paid GPUs.

**Totals:** ✅ 5 · 🟡 1 · ⬜ 19 · ⛔ 1  →  **5 fully done / 26.**

**Known technical caveats:** Kaggle image is newer than planned (numpy 2.0.2, datasets 5.0.0 —
watch for API drift); `prepare.py` dataset revisions still `"main"` (need pinned commit shas);
`<5 min` proxy prep is a design target, not yet timed on Kaggle; Task 4 GPU leg unverified.
