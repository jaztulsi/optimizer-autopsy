# UPDATE — where the project is, in plain English

This is your simple status page. No jargon. It answers three questions: **what's done**, **what's
changing**, and **what's next**. (The nerdy task-by-task checklist is at the very bottom.)

**Last updated:** 2026-08-23

---

## The one-line version

The hard part of the lab is finished and proven: we can rewind a training run and replay it perfectly,
and we've built the "fork" machinery that lets us test a fix and prove it was the fix. Now the project is
**switching hardware** — off Kaggle's free GPUs (we kept running out of hours) and onto a dedicated AMD
GPU budget — and then starting the actual science: finding and repairing the poison.

---

## What is this project again? (the plain version)

Big AI models sometimes "blow up" during training — the loss suddenly spikes and the run gets damaged.
This project is a **detective + surgeon** for that moment:

1. **Rewind** training to the instant it broke.
2. **Find** exactly which part of the model's "memory" got poisoned.
3. **Repair** just that part and prove the repair is what fixed it.

To do that honestly, we need to **replay training perfectly** — run it twice and get the *exact* same
numbers. Perfect replay is the thing we already got working.

---

## ✅ What's done (and proven on a real GPU)

Think of this as a finished, inspected laboratory:

1. **Accounts, keys, and cloud storage** are all set up; secret keys are never on the laptop or in the code.
2. **The code runs on a real GPU** and checks the environment before it starts.
3. **The data pipeline** hands back the *exact same* batch every time — the foundation of perfect replay.
4. **A secret-safety guard** scans the whole project and refuses to let a password slip in.
5. **Perfect replay works — confirmed on a real GPU.** Snapshot mid-training, run forward, rewind, run
   again → byte-for-byte identical, zero difference.
6. **The training engine learns.** The small practice model trained for 200 steps and its error fell from
   **10.85 → 4.73**.
7. **🔑 The "fork" gate passed — this is the big one.** Two do-nothing copies forked from the same snapshot
   came out *exactly identical* (zero difference). That's what makes every future "the repair worked" claim
   a real measurement instead of a guess. This is contribution **C1**, and it's done.

All of these have their actual run records saved in the `results/` folder, not just claimed from memory.

---

## 🔄 What's changing (the restart)

We built everything above on **Kaggle's free GPUs**, but the free tier only gives ~30 hours a week and
kept interrupting the real runs. So the plan (now written up as **PLAN_V6.md**) is:

- **Move to a dedicated AMD GPU** (an MI300X, via a program called Exea Labs — we've asked, not confirmed
  yet). More hours, no weekly cliff.
- **Don't rebuild — port.** All the code we already proved carries over as-is. The *only* thing we have to
  re-earn on the new hardware is the "perfect replay" guarantee, because AMD's chips do the math a bit
  differently than the old ones.
- **Check the risky part first.** Before spending real time, we run a tiny 6-hour test on the AMD GPU to
  see if perfect replay is even possible there. If yes, full speed ahead. If not, we switch to a
  "very-close-instead-of-exact" statistical approach — and we find that out in week one, not week eight.
- **Budget in tiers** (a small version, a full version, a stretch version) so the plan survives whether the
  grant ends up bigger or smaller than hoped.
- **Timeline:** the big conference deadline for this year already passed, so the honest targets are TMLR (a
  journal with no deadline) and next year's NeurIPS cycle.

---

## ⏭️ What's next (in order)

1. **AMD "can we replay perfectly here?" test** — the 6-hour go/no-go check. This decides everything else.
2. **Prospectively confirm the spike-maker** — the V6 policy now qualifies 2 recipes on the existing
   held-out T4 evidence; rerun once on the target GPU because the instantaneous-recipe rule was fixed
   after that evidence was inspected.
3. **The cheap-fix test** — check whether a simple existing fix already works everywhere. If it does, the
   fancy repair isn't needed, and that's a real, publishable finding on its own.
4. **Then the actual science:** build the "localizer" (finds the poison) and the "repair" (fixes just it).

---

## Progress bar

```
The laboratory (C1)   ████████████████████  DONE, proven on a real GPU
Spike-maker           ████████████████████  V6 policy: 2 qualify; fresh GPU confirmation pending
Cheap-fix test        ████░░░░░░░░░░░░░░░░  built, not yet run
The science (C2/C3)   ░░░░░░░░░░░░░░░░░░░░  not started — the heart of it
Hardware restart      ░░░░░░░░░░░░░░░░░░░░  moving to AMD; replay test is step one
```

**Status: 🟢 the lab is proven; now switching hardware and starting the science.**

---

## Anything to worry about?

- **The AMD replay test is the real unknown.** Everything rests on perfect replay, and AMD's chips have no
  exact copy of the trick we used on the old ones. That's exactly why we test it first and cheaply.
- **The AMD grant isn't confirmed yet** — we've reached out but nothing is locked. The budget in PLAN_V6
  stays a plan until it is.
- Everything already built is green and has its run records saved.

---
---

## Appendix: the technical checklist

Legend: ✅ done · 🟡 partial · ⬜ not started · ⛔ needs paid GPUs

**Phase 0 · Foundation — 4/4 ✅** — scaffold, env check, fixed data + `get_batch`, secrets + no-token test.

**Phase 1 · The instrument (C1) — ✅ complete, CUDA-verified**
- ✅ Task 4 — deterministic replay (`max|Δ|=0` over 50 steps, CPU + Kaggle T4)
- ✅ Task 5 — proxy model + trunk loop (loss 10.85 → 4.73 / 200 steps on T4)
- ✅ Task 6 — snapshot/restore of (weights + optimizer + RNG), name-keyed, tied-weight safe
- ✅ Task 7 — fork driver + the `Δ==0` gate (noop-vs-noop `max|Δ|=0` on T4)

**Phase 2 · Spikes + kill-test — 🟡 partial**
- 🟡 Task 8 — spike induction + detector: V6 policy now distinguishes predictive `lr_bump` (lead 2)
  from instantaneous `corrupt_batch` (onset, lead 0). Existing held-out T4 evidence qualifies 2/4,
  provisionally; a fresh prospective GPU run is still required.
- 🟡 Task 9 — cheap-fix battery (B0/Bg/Bs/B*) + PROCEED/PIVOT rule: built, not yet run to a verdict

**Hardware restart (PLAN V6):** ⬜ AMD go/no-go determinism smoke test · ⬜ ROCm strict-mode setup + pins

**Phases 3–6 — ⬜ not started:** localizer (C2), repair + baselines (C3), attribution + theory, scale +
figures + paper. ⛔ 410M needs paid GPUs.

**Known caveats:** `prepare.py` dataset revisions still `"main"` (need pinned commit shas); proxy `<5 min`
prep not yet timed; on AMD the recipe numerics may shift and need re-checking. Governing plan: `PLAN_V6.md`.
