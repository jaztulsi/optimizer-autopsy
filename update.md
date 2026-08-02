# UPDATE — where the project is, in plain English

This is your simple status page. No jargon. It answers three questions: **what's done**, **what I'm
doing right now**, and **what's next**. (The nerdy task-by-task checklist is at the very bottom if you
ever want it.)

**Last updated:** 2026-08-02

---

## The one-line version

The project has a solid foundation and its most important safety check is now **confirmed working on
a real GPU**. We're **~19% of the way** through the build. Everything built so far **works and is
tested**. Nothing is waiting on you right now.

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
6. **🔑 Perfect replay works — confirmed on a real GPU.** We can take a snapshot of a model
   mid-training, run it forward 50 steps, rewind, run it again — and get *byte-for-byte identical*
   results. Zero difference, on both your laptop and a real Kaggle GPU. This is the single most
   important piece: without it, none of the "find the poison" science would be trustworthy. It's now
   locked in.

---

## 🔨 What I'm doing right now

Perfect replay is **done and GPU-verified** — you ran it on Kaggle and it came back
`max|Δ|=0` (zero difference). That closes out the entire "safety foundation" part of the project.

I'm now starting the **engine**: the small practice AI model we'll experiment on, plus the
**training loop** that teaches it. Think of it as building the little car we're going to
deliberately crash and then repair.

---

## ⏭️ What's next (in order)

1. **Me (now):** build the small practice model + the training loop (the engine).
2. **Me:** build the "snapshot" system (save/reload a model's full state to the cloud).
3. **Me:** build the "fork" system — the heart of the project, where we test a fix and prove it
   worked by comparing against a perfect replay.

(You'll get one more Kaggle to-do when the snapshot system is ready — saving/reloading needs a
real-GPU check too.)

---

## Progress bar

```
Foundation  ████████████████████  DONE (4 of 4 steps)
Safety gate ████████████████████  DONE (perfect replay, GPU-verified)
The engine  ██░░░░░░░░░░░░░░░░░░░  building now
Everything else  ░░░░░░░░░░░░░░░░  not started
```

**Overall: ~19% built. Status: 🟢 everything passing.**

---

## Anything I should worry about?

- Kaggle's free computers have **newer software** than the plan expected (that's fine, just noted so
  nothing surprises us later).
- The replay test now passes on **both CPU and Kaggle GPU** — the big risk is retired.
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
- ✅ Task 4 — deterministic replay + test (**PASS on CPU and Kaggle GPU**, max|Δ|=0 over 50 steps)
- 🟡 Task 5 — proxy nanoGPT model + trunk training loop  ← *building now*
- ⬜ Task 6 — snapshot/restore of (weights + optimizer + RNG) to HF Hub
- ⬜ Task 7 — fork driver + the Δ==0 gate

**Phases 2–6 — not started:** spike induction, kill-test (T8–9), localizer (T10–12), repair +
baselines (T13–15), attribution + theory (T16–19), scale + figures + paper (T20–24). ⛔ Task 25
(410M) needs paid GPUs.

**Totals:** ✅ 5 · 🟡 1 · ⬜ 19 · ⛔ 1  →  **5 fully done / 26.**

**Known technical caveats:** Kaggle image is newer than planned (numpy 2.0.2, datasets 5.0.0 —
watch for API drift); `prepare.py` dataset revisions still `"main"` (need pinned commit shas);
`<5 min` proxy prep is a design target, not yet timed on Kaggle.
