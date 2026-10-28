# OPTIMIZER AUTOPSY — PLAN V8
### Current source of truth · 2026-10-22 · V7 + the Master Research run (waves 1–2 + fact-check) · targets ICML 2027 → NeurIPS 2027

> **Read this first, every session.** V8 supersedes V7 ([`docs/reference/PLAN_V7.md`](docs/reference/PLAN_V7.md)) where
> they differ. V7 §4's locked decisions (spike design, branch menu, ROCm tolerance, natural-spike rule, ZClip/AdaGC
> shadow warm-up, team rule) still hold unless §7 or §8 here changes them. V6 stays the long-form math reference
> ([`docs/reference/PLAN_V6.md`](docs/reference/PLAN_V6.md)).
>
> **The evidence trail for every claim here** is in [`master-research/output/`](master-research/output/):
> - wave-1 audits `02_*`;
> - novelty matrix `03`;
> - 63 ranked ideas `04`;
> - methodology `06`;
> - validation `07`;
> - blueprint and starter code `08`;
> - closed-form analysis `09`;
> - wave-2 transfers `10_wave2_*`;
> - skeptical review `11`;
> - fact-checks `12_*`.
>
> **Labels:**
> - **[V]** verified in the repo or a primary source;
> - **[I]** our inference or derivation;
> - **[A]** an assumption, such as a cost before `step_timer`;
> - **[UNV]** not confirmed, so it supports no claim.
>
> **Nothing in this plan has run on the AMD GPU yet. Every number is a prediction or a proposal.**

---

## 0 · What changed from V7 (one screen)

1. **The question is sharper.** V7 asked "can we localize and repair poisoned state?". V8's headline (revised
   2026-10-23 after the reviewer session) is a **mediation** question: **who wrote the lasting damage?** Was it the
   spike step itself (direct), or the optimizer's *memory* of the spike over the next ~1/(1−β) steps (indirect,
   through m carrying the bad gradient and v stalling or under-preconditioning)?
   - **Estimator:** 2 × 2 undo arms (none / m / v / m+v) at lag τ, with a Shapley split. Sweep τ and the spike dose
     G/ḡ.
   - **Two indirect channels with opposite signs:**
     - the **m-echo** (extra push), a fraction r = β1/√β2 of the total drift in the G/ḡ → ∞ limit (92% at
       (0.9, 0.95)). This is a corollary of Patitucci & Mokhtari 2026's single-spike Thm 3.3
       [V: reviewer re-read], so cite it;
     - the **v-stall** (lost progress): inflated v suppresses ordinary learning for
       ≈ ln((1−β2)G²/ḡ²)/(−ln β2) steps, ≈ 31 at G/ḡ = 10 [I].
   - **v is also protective:** its inflation is what keeps the echo bounded. Hence the sharp confirmatory prediction
     **C-2**: undo-v-only is *worse* than doing nothing.
   - **Pre-registered as a direction only** (loss ≠ drift): the indirect share rises with β1 across the arms
     (0.8 < 0.9 < 0.95).
   - **Supporting measurement:** the w/m/v share curve from the 2³ cross-world factorial. On its own it is
     predictable (state "washes out": Ibrahim 2024).

   Localization and repair keep their place as contributions 2 and 3.
2. **The novelty claims were narrowed by fact-checking.**
   - Bitwise replay is not ours (Megatron, OPEN-1B).
   - Optimizer-state patching is not ours (Xu 2026, Chimera).
   - One-step Adam inversion is not ours: SWIFT, TPDS 2024 [V: ar5iv §IV, Alg. 6/8].
   - What is ours is in §5.
3. **The instrument gets four fixes before any data.**
   - Held-out val-loss Δ, not the last training loss.
   - A non-degenerate cheap fix: `Bs` clip below the trunk clip.
   - `lr_bump` multiplies the schedule.
   - The Δ eval set is disjoint from the probe batches.

   Results are stamped, and the thresholds are pre-registered (§7.2).
4. **Identification fixes from theory.**
   - At (β1, β2) = (0.9, 0.95), the closed-loop weight/momentum decay rate √β1 sits within 2.7% of v's rate β2, so
     the "slope ≈ −ln β2" test can't tell them apart [I]. V8 adds a **β1 × β2 cross**.
   - The certified-repair threshold must use **Bai's Ĥ (2/η)**, not raw D⁻¹H (38/η at β1 = 0.9).
     [V: Bai eq. 7 defines Ĥ; 38/η is stated by Cohen et al. 2207.14484 and matches Bai Prop. 4.2 at β1 = 0.9.]
5. **The scale protocol has three levels.**
   - **L0:** constants set by β alone; they hold at any scale.
   - **L1:** amplitudes; they collapse under μP and follow a fitted law under SP.
   - **L2:** finite-size effects; measured at scale.

   It adds **matched 16,384 tokens per step** at every rung, which makes 124M forks about 20× cheaper [I/A], and a
   **scaling law predicted on held-out rungs before they run**.
6. **Borrowed real trunks.** We fork from Pythia's released final optimizer states: 160M first, then 410M, 1B, 1.4B,
   2.8B and up to 6.9B on one 192 GB MI300X [V: the 160M ZeRO shards exist; larger sizes are being checked]. That
   is a real scale ladder without training trunks.

   **OPEN-1B** (1.6B) publicly releases its **full optimizer state every 100 steps**, along with a skip-step spike
   log [V: HF card plus an anonymous bucket listing, `gs://gensyn-open-1b/ckpt`; step 100 is ≈ 19.3 GB ≈ 3× the fp32
   weights; shards not opened yet]. **But it is a calm run** [V: reviewer re-read of 2609.17380 §3.4.2, Table 7].
   - Its protocol "skips five optimizer steps when the pre-clip gradient norm exceeds 5". Only two batches ever
     triggered it: step 1,782 (norm 5.78) and step 43,737 (6.88).
   - So there is no population of naturally poisoned states.
   - **The opportunity:** "replay the spike the production run skipped". Fork at steps ≈ 1,700 and ≈ 43,700 and run
     once with the skip and once without. That gives two real natural-gradient events at 1.6B with an exact aligned
     B\*: an n = 2 case study at T3 cost.
   - Whether the skips left m/v untouched is not stated [UNV].
7. **New spike sources.**
   - Silent data corruption: `sdc_grad`, `v_bitflip`, `m_bitflip`.
   - Data-selected v-inflation.
   - An ordering-attack spike.
   - FP8 (FNUZ) instability.
8. **A three-paper program** instead of one (§4): ICML 2027 core → NeurIPS 2027 main (scale + generality) →
   NeurIPS 2027 datasets/benchmarks (SpikeState + toolkit).
9. **The compute ladder runs from ~5 to ~2,000 MI300X-h** (§8). Each tier is a complete, publishable unit. The GPU is
   not the constraint (user, 2026-10-22).
10. **All ideas are kept** (Appendix A, about 140 entries with honest novelty labels). The gates in §8.3 decide what
    runs first. They never delete an idea.

---

## 1 · Core values (non-negotiable)

1. **No causal claim without replay.** Noop vs noop must read exactly zero, or on AMD pass the pre-registered
   tolerance (V7 §4.4), and this must be measured *through spike windows and across processes*.
2. **A clean counterfactual and held-out loss.** `B*` keeps the data stream aligned. Every effect is
   `Δ = val_loss(branch) − val_loss(B*)` on a fixed held-out token set that is disjoint from every probe batch.
3. **Pre-register before you look** (§7.2). Commit the thresholds before AMD data exists, and never move a goalpost.
4. **Verify your own work; try to break it first.** A CPU check is never reported as GPU-verified.
5. **Compute discipline.** No training or tests on personal machines. All GPU work goes to the AMD MI300X. **No
   Kaggle** (user, 2026-10). CI runs CPU tests only.
6. **Honest scoring.** Claim the *finding* and the *instrument*, not determinism and not "root cause". Every novelty
   claim carries its kill-search status.
7. **Free tools, licenses respected.**
   - SPAM and AdaGC: reimplement.
   - ZClip: Apache-2.0, can be wrapped.
   - ARS skill material: CC BY-NC, ideas only.
8. **Load-bearing code is guarded.** Don't touch `determinism.py` or `preflight.py` without flagging. ROCm env vars go
   in the AMD launcher, not in `determinism.py`, unless signed off. `fork.py` belongs to the harness owner.
9. **Team hygiene.** Each person commits as themselves, through PRs. No AI attribution. Dev packages go in the root
   `requirements-dev.txt` only.
10. **Small verifiable steps.** Every unit ships with a check that fails if its logic breaks.
11. **Keep every idea; route, don't kill.** Gates order the work. A failed gate turns an idea into a reported result
    or a later-tier item, never a silent deletion.
12. **Fact-check before claiming novelty.** A NEW verdict needs a logged kill-search (`12_factcheck_novelty.md`).
    Citations come from `master-research/references.bib`, which holds verified entries only.

---

## 2 · The project in one paragraph

LLM pretraining loss spikes leave lasting damage, and every standard fix (clip, skip, reset, roll back) acts at or
before the spike step. We ask **who writes that damage**. Is it the spike step itself, or Adam's *memory* of the
spike over the next ~1/(1−β) steps?
- the **m-echo**, an extra push;
- the **v-stall**, lost progress, though also protective.

With a bitwise-deterministic counterfactual fork harness (C1, built), we identify Adam's **damage response** the
way one identifies a physical system. We measure how much damage the memory writes at each lag, what part depends on
the spike's sign, the refractory window a spike opens, and how two spikes interfere (§2.5). We then:
- **localize** the coordinates that carry the memory-written damage (C2);
- **undo** it retroactively with a reversible moment log, in the niche that step-time skip and clip cannot reach
  (C3).

**Headline claim (P1, to be tested):** most of a loss spike's lasting damage is written *after* the spike step by
Adam's memory of it, at a share ordered by β1.

**The null is still a result:** if direct ≫ indirect, acting at the spike step (clip or skip) is enough, and a
retroactive state repair has no niche. We would report exactly that.

**Title (P1, adopted):** *The Afterlife of a Loss Spike: System Identification of How Adam's Memory Writes Lasting
Damage*. The earlier "Where Does a Loss Spike Live?" is dropped.

**P1 reads as one story:** who wrote the damage (C-1, C-2) → how the two memory channels interact (C-3) → how to undo it in the niche
skip-step misses (C-4). Everything else goes to the appendix, P2 or P3. The invented algorithms and the full test
matrix are in [`docs/EXPERIMENT_CATALOG.md`](docs/EXPERIMENT_CATALOG.md).

### 2.5 · Novelty core: identifying Adam's damage response (reviewed and slotted 2026-10-23)
The mediation share stops being a single number. It becomes one readout of three quotable objects that bitwise forks
make measurable: a **damage-writing kernel**, a **refractory period** and **spike interference**. Specs are in the
catalog, Part A0; the reviewer's verdicts are in `13_reviewer_plan.md` §11. Novelty is judged against our corpus: no
hard collisions.

| ID | Measurement | Slot | Novelty vs the corpus, and the distinction to write | Key fix from the review |
|---|---|---|---|---|
| **M1** | Damage-writing kernel: cumulative D(τ), plus k(τ) = D(τ+1) − D(τ), "the lasting damage added by letting the memory act one more step, given the closed-loop path" | **C-1** | NEW. Liu & Li give when an effect is *observed* (output time, linearized); k(τ) gives when lasting damage is *written* (mediator time, exact). Cite #36730's "sustained > onset-only control" as a coarse precedent for τ½, and #17880 as converging evidence | Log grid τ ∈ {0, 1, 2, 4, 8, 16, 32, 64}. Cumulative D(τ) with CIs is primary. Pre-register the k(τ) family a·r^τ − (stall term) |
| **M2** | Gradient-parity split (±G): Δ_odd vs Δ_even | exploratory, with its own figure | NEW as an instrument. It measures exactly where Liu & Li's linear signed operator fails (Δ_even) | **Clip-match the arms**: apply the +G arm's clip factor to both, or inject post-clip at the accumulated gradient. Pre-register "≫" as \|Δ_odd\| ≤ 0.2·Δ_even |
| **M3** | Refractory period χ(Δt) from small probes | **supporting evidence for C-2** (not independent) | NEW as an instrument. The effect of inflated v is known qualitatively (Wortsman, SPAM) | Restrict or weight the probe to the dominated coordinates. Size P above the placebo spread. Loss-level χ ≠ update-level χ |
| **M4** | Spike interference I(Δt), same vs opposite sign | **C-3** | NEW (nothing like it in the corpus) | Δt ∈ {2, 8, 32}. Pre-registered: I_opp < I_same for Δt ≤ 1/(1−r) ≈ 13. It equals the second-order kernel only for small G |
| **M5** | Universal collapse vs the dose x̄ | **P2** | NEW (claim). Loss also needs scale-free curvature | **The dose metric is pre-registered now**: x̄ = Σ_i \|u_i(s)\|·x_i / Σ_i \|u_i(s)\|, the spike-step-update-weighted mean of x_i = (1−β2)G_i²/v̄_i. No metric shopping |
| **M6** | Mediator-dynamics DiD (β changed only in the echo window) | **P1-stretch** | PARTIAL (A15; Tang is onset) | With bitwise forks the DiD *is* the exact interaction contrast, with no parallel-trends assumption. Bias-correction convention: the EMA uses the β in effect at each step, and the correction is 1 − Π_t β_t over the β's actually used |
| **M7** | Dead time in natural spike trains | **P2** | PARTIAL (Tang's period; Bai's cycle) | Run it on the AMD host or in CI, never the Mac. Few natural spikes, so a weak hazard estimate |
| **M8** | Staged repair (RML operator) | **exploratory** | PARTIAL | — |

**Fork budget for P1-core** [A] (reviewer's count, converted by the builder at the 10× P100 scenario: ≈ 10 s per
end-of-schedule proxy fork; 124M as in §8):
- **Per instance:** ≈ 32 forks (M1 10 + C-2 4 + M3 4 + M4 12 + M2 2).
- **Totals:**
  - 30 core instances ≈ 960 forks;
  - β1 arms ≈ 300;
  - repair battery ≈ 360;
  - ≈ 1,600 proxy forks ≈ 4.5 h.
- **Plus:** 124M (n = 5 × 5 forks + its trunk) ≈ 6–7 h; T0 bring-up 3–5 h; proxy trunks < 0.5 h.
- **Total ≈ 12–19 MI300X-h**, inside the ≤ 20 h core line.
- **At the pessimistic 2× scenario** the proxy part is ≈ 5× more expensive. Then, in order: cut M3 to 2 Δt values, M4
  to Δt ∈ {2, 8}, and 124M to n = 3. `step_timer` on day 1 decides.

### 2.6 · Paper skeleton (P1), written backwards (drafted by the reviewer session)

**Title:** *The Afterlife of a Loss Spike: System Identification of How Adam's Memory Writes Lasting Damage*

**Placeholder abstract** (fill the brackets from Tab 2):
> Loss spikes in LLM pretraining are handled at the spike step: clip it, skip it, or roll back. We ask who writes a
> spike's *lasting* damage, the spike step or Adam's memory of it, and answer with system identification on a
> bitwise-deterministic counterfactual fork harness. Undoing the spike's remaining contribution to Adam's moments at
> increasing lags identifies a damage-writing kernel: [X]% [CI] of end-of-schedule held-out damage is written
> *after* the spike step, within an action window of τ½ ≈ [Y] steps, and this share rises with β1 as a closed-form
> echo law predicts. The two memory channels pull in opposite directions. The first moment's echo pushes the weights
> further. The inflated second moment stalls learning (a refractory period of ≈ [Z] steps, measured directly with
> probe perturbations) and also bounds the echo, so cleaning the second moment alone makes damage *worse*
> ([W] vs [V] nats). Pairs of spikes interfere exactly as this channel structure predicts. Acting inside the window,
> a reversible moment log recovers [R]% of the oracle while editing [E]% of coordinates, and beats skip-step on
> multi-step data spikes that per-step rules miss. Results hold from 11M to 124M parameters and on a Pythia-160M
> trunk. All thresholds were pre-registered.

**Null version** (if C-1 fails, direct ≫ indirect): "…we find that [X]% of lasting damage is written at the spike
step itself; Adam's memory adds little after τ ≈ [Y]. Spike handling therefore belongs at the spike step, and we
quantify the window in which retroactive repair still helps." **Both versions stay in the repo**, so the result can't
steer the framing.

| # | Element | Shows | Experiment (forks per instance) | Gate / test |
|---|---|---|---|---|
| **Fig 1** | Damage-writing kernel: cumulative D(τ) with CIs, and the fitted k(τ), split into direct / m-echo / v-stall, per spike type | the C-1 headline; τ½ | M1, undo-m+v at τ ∈ {0, 1, 2, 4, 8, 16, 32, 64}, plus B0 and B\* (10) | G2; **C-1** |
| **Fig 2** | Theory vs measurement across the β1 arms (0.8 / 0.9 / 0.95) and the G/ḡ doses | "the theory predicts the ordering before the run" | M1 on the β1 arms; κ doses | C-1 (ordering) |
| **Fig 3** | Protective v: none / undo-m / undo-v / undo-m+v at τ ∈ {0, 10}; inset: refractory χ(Δt) vs its prediction | the C-2 sign; the v time course | undo arms at 2 τ (4); M3 at Δt ∈ {2, 8, 32} + P alone (4) | **C-2**; M3 supporting |
| **Fig 4** | Spike interference I_same vs I_opp for Δt ∈ {2, 8, 32} | the channel structure, with no state patching | M4 (12) | **C-3** |
| **Fig 5** | Repair Pareto (gap closed vs edit fraction vs GPU-s): RML vs OLMo 6σ skip, REBUILD_M, tail-entrant limiting, ZClip, AdaGC, SPAM, Adam-MR, band-clamp, on niche spikes | C-4 | repair battery (≈ 12) | **C-4** |
| **Fig 6** | Scale: indirect share at proxy → mid → 124M (n = 5), plus Pythia-160M | external validity | M1 subset τ ∈ {0, 8, 64} (5) | G6 |
| Tab 1 | Instrument validity: AMD noop Δ (in-process and cross-process, through spike windows), placebo spread, hybrid violations | credibility | T0/T1 | G0, G1, G3 |
| Tab 2 | C-1…C-4 with CIs, Holm-adjusted, per spike type, never pooled | — | all | — |
| App. | M2 parity (clip-matched); localization vs the oracle (exploratory); the pre-registration text and tag | exploratory | M2 (2) | BH |

**Writing order** (it can start before the GPU; jaz to approve): related work (corpus 02–13) → method (harness,
estimands, the M1/M3/M4 definitions) → pre-registration appendix → intro around the placeholder abstract. Results get
filled in as gates pass. Reviewer simulation at week 8.

---

## 3 · Verified state on 2026-10-22

| Piece | State | Evidence |
|---|---|---|
| C1 determinism and the fork `Δ==0` gate | ✅ CUDA-verified (T4 proxy; P100 124M at grad_accum = 40). **AMD untested** | `results/task4_*.json`, `task7_fork_gate.json`, `research/kaggle/step_timer_results.md` |
| AMD MI300X | Provided by Avneh. **Not yet handed over to the build; nothing has run on it** | user, 2026-10 |
| Spike pipeline | `capture.py`, `natural.py`, `k2.py`, `induce.py` (`repeat_ngram`, `v_inflate`) and `test_spike_pipeline.py` are written and **committed on local branch `plan-v8`** (826b03c; not pushed, never run) | `git log plan-v8` |
| Baselines | skip, clip, Bg, SPAM, ZClip, AdaGC, skip-step (oracle and detector-timed), v-only reset, `Bvbar`. Merged in PR #14; CI-tested; never run on a GPU | `research/baselines/` |
| Kill-test (Task 9) | Built and logic-tested. **Four instrument defects found** (§9.3); never run | `docs/improvement.md` A1–A8 |
| Localizer (C2), repair (C3), theory | ⬜ unbuilt. The output contract is now defined (§6.4) | `research/localizer/`, `repair/`, `theory/` |
| Research corpus | Waves 1–2, about 150 ideas and 2 fact-check passes | `master-research/output/` |

---

## 4 · Paper program (escalation path)

| Paper | Venue (dates [UNV] until CFPs are out) | Core claim | Compute tier | Status |
|---|---|---|---|---|
| **P1** | **ICML 2027** (abstract ≈ Jan 16, paper ≈ Jan 22, 2027 [UNV]) | Who wrote the damage. **C-1** mediation (the M1 kernel; β1-ordered indirect share); **C-2** protective v (2×2 undo arms + the M3 refractory curve); **C-3** spike interference (M4: I_opp < I_same for Δt ≤ 13, no state patching); **C-4** retroactive RML undo at detector lag vs the best baseline and the OLMo skip-step on niche spikes; 124M n = 5 on C-1 | P1-core ≤ 20 MI300X-h [A]; stretch +30–60 h | planned |
| **P2** | **NeurIPS 2027 main** (≈ mid-May 2027 [UNV]) | *Scaling laws of optimizer-state damage*: out-of-sample predicted rungs; Pythia ladder to 2.8B–6.9B; μP/CompleteP collapse; natural spikes at ≥ 410M; SDC/bit-flip reliability; Muon / Adafactor / Adam-mini / AdEMAMix generality; RL fine-tuning and ViT transfer | Tier 3 (+ Tier 4) | planned |
| **P3** | **NeurIPS 2027 datasets/benchmarks track** (the 2026 track was "Evaluations & Datasets", with Croissant + RAI metadata [V]) | **SpikeState**: ≥ 200 spike instances with (w, m, v, v̄) snapshots, clean twins, oracle maps and a leaderboard; plus the open autopsy toolkit (fork engine, undo-log wrapper, integrity scanner). **Privacy:** optimizer state leaks training tokens (seU1LYAiwm), so release only public-data runs and document the risk | Tier 3 (reuse) | planned |
| fallbacks | TMLR (rolling); ICLR 2027 / ICML 2027 workshops | any subset that is complete | any | — |

**Rule:** P1 must be submittable on **P1-core alone**: ≤ 20 MI300X-h [A]. That fits the 15 h ask and Avneh's
~10 h comment. P2 and P3 reuse P1's harness and every instance P1 produces.
- **P1-core:**
  - T0/T1 validity;
  - 2×2 undo × τ ∈ {0, 10, 30} × 3 spike types × n = 10, plus the M1 kernel on a subset;
  - 3 β1 arms;
  - the oracle + **no-HVP** localizers;
  - RML + baselines;
  - 124M n = 5 on C-1.
- **P1-stretch (+30–60 h):**
  - HVP-based pieces (λ-decomposition, SCR), the biggest engineering risk and needed by no confirmatory test;
  - M2–M4;
  - Pythia-160M.
- **→ P2:** everything else.

**GPU-delay rule:** each week of delay drops one stretch row, and the core is protected. If the GPU arrives after
Dec 1, P1 moves to NeurIPS 2027, plus an ICML-workshop paper on C-1.

---

## 5 · Contributions and novelty ledger (fact-checked)

Two passes back these verdicts:
- `12_factcheck_citations.md` re-read 25 cited claims at the source; 5 were corrected.
- `12_factcheck_novelty.md` was an adversarial kill-search on 15 claims.

They are **not a systematic review**. That pass ran out of search budget before N13–N15, and OpenReview and
OpenAlex are still to query (open issue §13).

> **Honest novelty statement.** Every component of this project has precedent:
> - state transplants (Xu 2026, Chimera 2026, Guo 2026);
> - deletion from optimizer memory (Stewart 2026 for L-BFGS; SWIFT's one-step Adam inversion);
> - selective v clearing (Biswas 2026, toy setting);
> - the single-spike Adam response (Patitucci & Mokhtari 2026);
> - forking from released optimizer states (the Pythia paper);
> - joint m/v history effects and aligned recovery up to 7B ("Newer Is Not Always Better", ICLR 2027 submission
>   P0oFJZXAh8);
> - spike-onset state capture and replay at 355M, and selective coordinate limiting beating global shrinking at
>   equal norm, which is prior art for the selectivity premise of RML and the exploratory localizer ("Understanding and Mitigating AdamW Loss
>   Spikes", pAjrfibDB9);
> - optimizer-state data leakage (seU1LYAiwm).
>
> **What is new is the study and what it finds:**
> - the first time-resolved, causal (w, m, v) decomposition of *LLM loss-spike* damage, with a bitwise counterfactual
>   instrument, β-identified rates and pre-registered predictions;
> - localization scored against a causal oracle;
> - repairs that must beat the strongest baselines;
> - a released benchmark.
>
> This is an empirical-science contribution in the ICML sense. We never claim a component as an invention.

| # | Claim | Verdict | Closest prior work (most damaging first) | What we may safely claim |
|---|---|---|---|---|
| K1 | Counterfactual fork instrument on optimizer state (aligned B\*, bitwise gate) | PARTIAL | Pythia [2304.01373](https://arxiv.org/abs/2304.01373) §3.1 (resume from "a checkpoint and optimizer state" on "the exact same data" with an intervention: **the precedent for an aligned B\***); Megatron Core bitwise determinism (2026); OPEN-1B [2609.17380](https://arxiv.org/abs/2609.17380); Liu & Li paired forks [2608.19762](https://arxiv.org/abs/2608.19762); Gemini §3 replay | Interventions that edit stored m/v on real spikes, checked through spike windows. Claim the instrument's *use*, not determinism |
| **K2** | **Mediation of spike damage over time** (direct vs memory-written; m-echo vs v-stall), with the w/m/v share curve as support | **PARTIAL** (the headline delta is the loss-space mediation on LLM spikes) | **Nearest to the headline itself:** #17880 P0oFJZXAh8 ("effects that outlast inherited moment differences", for transplants) and #36730 pAjrfibDB9 ("replay favors sustained over onset-only control"; timescale mismatch at onset), both [V-abstract]. Then: Patitucci & Mokhtari [2602.10204](https://arxiv.org/abs/2602.10204) Thm 3.3 (single-spike m/v dynamics; the r-law follows; they also run GPT-2 124M); Xu [2608.20442](https://arxiv.org/abs/2608.20442) (transplants); Chimera [2607.06628](https://arxiv.org/abs/2607.06628); Guo [2608.24593](https://arxiv.org/abs/2608.24593); Stewart [2605.17590](https://arxiv.org/abs/2605.17590); Ibrahim 2024 [2403.08763](https://arxiv.org/abs/2403.08763) App. A.4 (continual pretraining at a dataset switch, not spikes: inherited moments "quickly decay to 0", < 0.0044% after 1,000 steps); Tang [2609.18314](https://arxiv.org/abs/2609.18314) (β phase diagram for *onset*; ours is carry-over) | "The first causal, time-resolved mediation of LLM loss-spike damage into direct, m-echo and v-stall channels, with a bitwise instrument and β1-ordered predictions." **Ibrahim's decay is the null C-1 must beat** |
| K3 | Poisoned-state transplant with dose-response | PARTIAL (near TAKEN for "state alone causes damage") | Xu 2026: m alone, with weights unchanged at the cut, carries the effect. P0oFJZXAh8: moment transplants across checkpoint ages, 34M–7B | "The first dose–response of *spike-poisoned* (m, v) on clean weights in pretraining" |
| K4 | State-only retroactive deletion (undo log) and the direct/indirect split; plus its weight-side extension (counterfactual weight back-projection) | PARTIAL | SWIFT [2302.06173](https://arxiv.org/abs/2302.06173) (one step, weights included); Amnesiac ML [2010.10981](https://arxiv.org/abs/2010.10981) (subtract stored per-batch updates); DeltaGrad [2006.14755](https://arxiv.org/abs/2006.14755); Stewart 2026 (memory-only vs parameter-only deletion, L-BFGS) | "Closed-form Adam-moment window deletion as a mediation probe and repair." The weight side is **"echo-aware amnesiac subtraction for Adam"**: the batch's contribution includes its m/v-mediated future updates, and its open-loop error is measured against a bitwise skip fork |
| K5 | Signed, support-limited localization vs a causal group-testing oracle | PARTIAL | SPAM (one-sided g²/v > θ); Guo 2026 (gradients through AdamW state); AtP\* [2403.00745](https://arxiv.org/abs/2403.00745); ACDC [2304.14997](https://arxiv.org/abs/2304.14997); Zhang–Bengio–Singer [1902.01996](https://arxiv.org/abs/1902.01996) (per-layer reset oracle) | The composite: a causal oracle over *optimizer-state* groups, with cheap signed predictors scored against it |
| K6 | Stability-certified restoration | PARTIAL | Biswas 2026 (clearing excess v on the affected rows ≈ full reset; toy); GI-Adam [2406.09405](https://arxiv.org/abs/2406.09405) (edits v at init to lower preconditioned sharpness); Fong & Yang [2608.20638](https://arxiv.org/abs/2608.20638) | A post-spike, support-limited, signed v edit with a logged certificate. The threshold is **2/η for Bai's normalized Ĥ** (which absorbs (1−β1)/(1+β1)); raw D⁻¹H is 38/η (Cohen 2022). State which matrix every time |
| K7 | Anti-windup drift | PARTIAL; **saturating, not size-free** | Patitucci & Mokhtari [2602.10204](https://arxiv.org/abs/2602.10204) (per-update bound in Remark 3.1; single-spike Thm 3.3; **no cumulative bound**); Kingma & Ba §2.1 | "Cumulative drift on a dominated coordinate *saturates* at (1−β1)/(√(1−β2)(1−β1/√β2))·η = 5.84η as G/ḡ → ∞, a corollary of their Thm 3.3. Smaller spikes drift less." Clipping lowers G, but large spikes stay near saturation |
| K8 | SpikeState dataset | **NEW** (low confidence; one thin search) | LLM360 K2 (no moments); Amber (no optimizer states); Pythia neox-ckpt (states, no spikes); OPEN-1B bucket (states, uncurated) | "The first curated public spike instances with (w, m, v, v̄), clean twins and oracle maps", once re-searched |
| K9 | Fault injection directly into m/v; localized restore | PARTIAL (the injection map looks new) | Altenbernd 2026 (gradient SDC → v→∞ freeze); TrainSDC (explicitly leaves optimizer state alone); LayerCheck [2609.27193](https://arxiv.org/abs/2609.27193) (per-layer restore instead of rollback) | "The first direct fault-injection study into Adam's moments." **Physics:** a v up-flip freezes the coordinate (permanently if v = ∞); a v down-flip under-preconditions it and spikes at once |
| K10 | Scale transfer of state damage (held-out-rung prediction) | PARTIAL | Wortsman 2023 small-scale proxies [2309.14322](https://arxiv.org/abs/2309.14322); Everett 2024; CompleteP 2025 | "A pre-registered held-out-rung prediction of spike damage shares". Don't call it a "scaling law" with fewer than 5 rungs |
| — | Forking from released Pythia states | **TAKEN (a setting)** | Pythia [2304.01373](https://arxiv.org/abs/2304.01373) §3.1 | "We use Pythia's released states." Not a contribution |
| — | Jump-robust shadow v | PARTIAL (estimator use looks new) | TAdam, AdaTerm, AdaShift (robust moments) | "The first bipower-variation jump share for Adam-state localization" |
| — | Optimizer-state forensics (tokens from Δv) | **TAKEN** | **"Training-Data Leakage from Optimizer State in Resumable Checkpoints"** (ICLR 2027 submission, OpenReview seU1LYAiwm, [V-abstract]): a single checkpoint's state reveals the tokens trained on, and differencing adjacent checkpoints recovers gradients. Also Melis 2019 [1805.04049](https://arxiv.org/abs/1805.04049); FILM; AEGIS 2026 | Not a contribution. **SpikeState release constraint:** release states only from public-data runs, and put the leakage risk in the datasheet |
| — | Joint m/v history mismatch (newer m, older v) | **TAKEN** (mechanism) | "Newer Is Not Always Better" (P0oFJZXAh8, [V-abstract]) | Cite it; C-2 tests it out of sample on spikes |
| — | Spike-onset coordinate tail entrants; selective limiting | **TAKEN** (for immediate control) | "Understanding and Mitigating AdamW Loss Spikes" (pAjrfibDB9, [V-abstract]) | An exploratory-localization and C-4 baseline; our delta is the lasting-damage mediation |
| — | Integrity scan of `optimizer.pt` | PARTIAL (the invariant is stated in Patitucci & Mokhtari Remark 3.1) | — | A cheap *necessary* check. Not a defense against an adaptive adversary (our own v-reset baselines fail it legitimately) |
| — | Spike phase × cooldown | PARTIAL | Wen 2024 [2410.05192](https://arxiv.org/abs/2410.05192) ("decaying near a loss spike can lead to a significant performance loss") | The causal w/m/v decomposition vs schedule phase |
| — | "Spikes are availability attacks on state" | framing only | Altenbernd; Bai; Xu; Shumailov 2021 | A taxonomy for the discussion section. Claim no novelty |

**Not ours (cite, never claim):**
- Bai's mechanism and λ_grad;
- Wortsman's stale-v/RMS precursor;
- bitwise replay;
- optimizer-state patching as a method;
- one-step Adam inversion;
- HVP-free curvature;
- adaptive β2 (Kourkoutas-β [2508.12996](https://arxiv.org/abs/2508.12996); AFOR [2609.24233](https://arxiv.org/abs/2609.24233), an EEG paper);
- the 38/η threshold (Cohen 2022);
- the |m| ≤ C√v invariant (Patitucci & Mokhtari Remark 3.1; the standard proof technique);
- forking from released Pythia states.

---

## 6 · Methodology

### 6.1 Objects and estimands
- **State.** Θ_t = (w_t, m_t, v_t, k_t), where k is Adam's step counter. A spike event E occupies the window
  W = [s0, s1).
- **Branches.**
  - B0: the trajectory with E.
  - B\*: the data-aligned counterfactual (the hook omitted).
  - Cut time: t1 ≥ s1. Both branches read identical data after t1.
- **Damage.** Δ_X(h) = L_val(X, t1+h) − L_val(B\*, t1+h) on ≥ 1M held-out tokens from val batches ≥ 1000. The
  primary horizon is end of schedule.
- **Cross-world hybrids.** Θ^{abc} = (w^a, m^b, v^c, k) for a, b, c ∈ {0, \*}, which gives Möbius effects and the
  Shapley shares φ_w, φ_m, φ_v. The **2⁴ variant** adds k^d (bias correction) as a fourth mediator, because Adam-MR
  and Bg reset k (new in V8, X1).
- **Hybrid feasibility.** Every honest Adam state satisfies |m_i| ≤ C√v_i with
  C = (1−β1)/√(1−β2)/√(1−β1²/β2) = 1.165 at (0.9, 0.95) [I: Cauchy–Schwarz, re-derived in V8]. Violations are
  counted on every hybrid, and an off-manifold control is added.
- **Signed poison.** σ_i = sign(log v_i⁰ − log v_i\*): "+" means over-preconditioned (stall); "−" means
  under-preconditioned (Bai's instability).
- **Causal ground truth.** S\*, the set whose patching recovers Δ (sufficiency) while its complement does not
  (necessity). It is found by adaptive group testing.

### 6.2 Spike sources (each instance is paired with its own B\*)

| Source | Type | Ground truth | Note |
|---|---|---|---|
| `lr_bump` | predictive | none | **Fix:** multiply the *scheduled* LR (today it sets an absolute LR, induce.py:52-60 [V]) |
| `corrupt_batch` | onset, data-driven | none | width 5 |
| `repeat_ngram` | data-driven (the OLMo cause class) | none | |
| `v_inflate` + **decoys** + m-only plants | planted | exact | κ ∈ {1, 10², 10⁴} |
| natural (stress presets) | dynamics or data | none | excluded from causal contrasts (option b); used for localization and detection |
| **`sdc_grad`** | exponent flip in one gradient element | exact | undo should fix it (K9) |
| **`v_bitflip`** / **`m_bitflip`** | memory fault in stored state | exact | v up-flip: silent freeze (permanent if v = ∞). v down-flip: under-preconditioned, spikes at once. m flip: spikes. The parity residual flags all of them |
| **data-selected v-inflation** | data-realizable plant | known support | bridges planted and natural |
| **ordering-attack window** | same data, reordered | none | B\* is the unpermuted window (design check needed) |
| **bf16 + flash-attention loss explosion** (GPT-2 124M, ≈ 6.6k steps) | precision-driven natural | none | T3. nanoGPT issues #303/#524; Qiu & Yao [2510.04212](https://arxiv.org/abs/2510.04212). Whether it reproduces on ROCm is [UNV] |
| FP8 (e4m3fnuz) instability | precision-driven natural | none | T4 only. torchao works on MI300X [V-vendor], but the small-scale evidence shows a loss gap, not spikes, and torchao wants `torch.compile`, which breaks our determinism recipe [V: tools pass] |

### 6.3 Measurement instruments
0. **Mediation sweep (the P1 headline).**
   - **Arms:** B0; B\*; and "undo at lag τ" for τ ∈ {0, 1, 3, 10, 30, 100}, **for each of undo-m, undo-v and
     undo-m+v** (a Shapley split; C-2's protective-v test). "Undo at τ" removes the spike window's contribution to
     that channel at t1 = s1 + τ and keeps w.
   - **Dose arms** report the indirect share as a curve over G/ḡ. The r-law only holds as G/ḡ → ∞.
   - **Estimates:**
     - Δ_direct = Δ(undo at τ = 0): the spike step's own update, kept;
     - Δ_indirect(τ) = Δ(undo at τ) − Δ_direct: the damage the memory wrote within τ steps.
   - **Weight-space analogue,** fork-free: counterfactual weight back-projection (EXPERIMENT_CATALOG A5). Its gap to
     the true skip-step fork measures closed-loop nonlinearity.
   - **Prediction:** the indirect share of dominated-coordinate drift is β1/√β2 [I].
1. **Damage-migration factorial.**
   - Cut times τ = t1 − s1 ∈ {1, 10, 30, 60, 150, 400}.
   - 6 new hybrids per τ (B0 and B\* exist already). Add +8 cells for the 2⁴ k-arm on a subset.
2. **β1 × β2 identification cross.**
   - Arms (β1, β2) ∈ {(.9,.95), (.9,.99), (.9,.999), (.8,.95), (.95,.95)}.
   - **Predictions:**
     - v-only patch recovery decays at −ln β2;
     - the underdamped w/m part decays at −½ ln β1;
     - the drift ratio is 1 : 1.79 : 5.44 across the β2 arms [I].
3. **Poisoned-state transplant.** B0's m, v or (m, v) onto B\*'s w, with doses κ ∈ {1, 10², 10⁴}. Endpoints: a
   binomial rate of "Δ > τ" plus Spearman ρ.
4. **Placebo replicates.** K = 5 runs with a 1-ulp perturbation per branch. Any effect must exceed the placebo spread.
5. **Free re-scored endpoints** (no extra forks):
   - per-token damage map (R6);
   - linear-mode-connectivity barrier between B0, the repair and B\* (R28);
   - poison-support contagion curve (R23);
   - spike information half-life above the placebo floor (R1);
   - residual memory of v vs ln 2/(−ln β2) (S8);
   - time-to-whiteness of the innovations as a fork-free "healed" clock (W6).
6. **Phase × cooldown.** The same spike injected at 5 LR phases on cosine and WSD schedules (R12).
7. **Telemetry.** Logged on every run, analyzed offline:
   - per-tensor RMS_t = √E[g²/v];
   - max attention logit;
   - fraction of coordinates with √v̂ < 10ε;
   - λ_grad(Ĥ) probes;
   - jump share 1 − v_BV/v.

### 6.4 Localization (C2)
- **Oracle.** Adaptive group testing over tensor → row → head groups (binary splitting, with ddmin as fallback;
  SuSiE credible sets under interactions). Its firing threshold is τ_f = max(0.005 nats, 5× the placebo spread).
- **Cheap predictors (the variants V7 budgets, plus V8 additions):**
  1. **AtP-state.** One backward pass at the ghost-updated weights, in log-v and in m.
  2. **Two-sided NIS sign test.** Lower and upper tails of g²/v_prior.
  3. **λ-decomposition of Bai's λ_grad by group** (BACKUP 1).
  4. **Jump share J = 1 − v_BV/v** from a shadow bipower v. It needs no forks and no logs, and fails by design at
     width ≥ 2 (W1).
  5. **Windup score** (low recent excitation) as a prior (W3).
  6. **Forensic token recovery** from Δlog v of wte rows (R3). This is a data-side localizer.
  7. **Magnitude-only control** ‖log v − log v̄‖.
- **Output contract** (closes CLAUDE.md §7.2):

```
Localization = {"groups": {gid: {"tensor": str, "rows": slice|None, "score": float, "sign": +1|-1,
                                 "channel": "m"|"v"|"m+v"}},
                "selected": [gid, ...], "method": str, "ref": "B*"|"vbar"|"fresh_g2",
                "t": int, "cost_gpu_s": float}
```

### 6.5 Repair (C3)

| Operator | What it does | Slot | Prior art / role |
|---|---|---|---|
| **Undo log** (state-only retroactive deletion) | m −= Σ(1−β1)β1^{t1−s}g_s; v −= Σ(1−β2)β2^{t1−s}(g_s² − v̄), clamped ≥ 0; w untouched | `intervention` at t1 | SWIFT ancestor; **ours: window, state only, path split** |
| **GLR estimate-then-correct** | Undo with an *estimated* onset and size, so no gradient log is needed (W4) | same | new application |
| **Outlier-type routing** | Additive outlier → undo; level shift → keep; transient → partial (W5) | policy | removes the undo-vs-`lr_bump` problem by design |
| **Stability-certified restoration** | v_S(α) = v_S^{1−α}·T_S^α, with the largest α such that λ_grad(Ĥ) ≤ (1−δ)·2/η, using **Bai's Ĥ** | `intervention` | target T = v̄ or **fresh g² at the current w** (central flows predict that v̄ weakens at late cut times [I]) |
| **Shadow-v swap** | v_S ← v_BV,S (W1) | `intervention` | fork-free, width-1 only |
| **Band-clamp baseline** | v ← clip(v, a·v̄, b·v̄) (W3) | `intervention` | **required**: if it matches, the localizer adds nothing |
| Selective re-warm, optimizer hot-swap on S (R18), weight decay off on S (R27) | ablations | `pre_step` | mechanism tests |

### 6.6 Scale-transfer protocol (STP)

**Levels:**
- **L0, algebraic:** decay rates, drift constants and the feasibility bound, checked within ±15% at every rung.
- **L1, parametrization:** φ_state, Δ_B0/L, ρ_dom and G. These should collapse under μP/CompleteP and follow a power
  law under SP.
- **L2, finite-size:** end-of-schedule Δ and natural-spike incidence, measured at scale and stated as scope.

**Matching rules:**
1. **16,384 tokens per optimizer step inside every fork window.**
   - A trunk may train at a large batch for efficiency. It then needs a ≥ 200-step **burn-in at 16,384 tokens/step**
     before the fork snapshot, because v re-equilibrates in ≈ 3/(1−β2) steps [I].
   - Otherwise apply β2\* = β2^{B\*/B} (Marek 2025).
2. Identical (β1, β2, ε, clip, warmup, schedule shape). Sites go at matched schedule fractions (40% and 70%).
3. A pre-registered ε policy, with the fraction of coordinates where √v̂ < 10ε logged.
4. Doses in dimensionless units, with the G/ḡ distributions reported.
5. qk-LN on and off at each rung.
6. SP rungs use the 124M learning rate. The μP base is the d = 192 proxy.

**Held-out-rung law:** fit L1 observables on 5 small configs. **Predict the mid rung and 124M before running them.**
It passes if the observations fall inside the 95% prediction interval.

### 6.7 Baselines (each tuned on seed 0 only)
- **Strong:**
  - rollback-and-skip (PaLM; labelled data-shifting);
  - Adam-MR (resets m, v and t);
  - bias-correct Bg;
  - v-only reset;
  - `Bvbar`;
  - OLMo-core 6σ skip;
  - `Bskipstep_det` (the OPEN-1B rule);
  - ZClip, AdaGC, SPAM;
  - GradientStabilizer;
  - StableAdamW;
  - CAST [2610.19761](https://arxiv.org/abs/2610.19761) if code exists;
  - the band-clamp.
- **Controls:**
  - B0;
  - a clip-tighter `Bs` (0.5 × the trunk clip);
  - `Br` (random, matched in support and energy);
  - a magnitude-only localizer;
  - Spectra-style damping;
  - a step-counter-only reset.
- **Primary contrast:** the best of our repairs vs the best non-oracle baseline, on gap closed
  G = (Δ_B0 − Δ_X)/Δ_B0 at end of schedule.
- **State of the code (2026-10-22, checked against `main` @ 59d329b).**
  - **Merged in PR #14 (Eddie):** SPAM, ZClip, AdaGC (with `prime_trunk` shadow priming), `Bskipstep` /
    `Bskipstep_det`, `Bvreset` and `Bvbar`. `set_v_to_vbar` is strict: it raises `KeyError` with the parameter name
    and checks shapes [V: reset.py:35-53].
  - **`Bg` and `Bvreset` keep Adam's step counter.** Their first post-reset updates are therefore about 4.5× too
    large at β2 = 0.95 (documented in reset.py). V8 adds **Adam-MR** (also reset `state["step"]`) and a
    **bias-correct `Bg`** as the fair versions.
  - **Still to write (baselines owner):**
    - Adam-MR / bias-correct Bg;
    - band-clamp;
    - step-counter-only control;
    - `Br` (matched in support and energy);
    - StableAdamW;
    - GradientStabilizer;
    - OLMo 6σ skip;
    - rollback-and-skip;
    - **REBUILD_M** (keep v, re-accumulate m from 0; P0oFJZXAh8);
    - **new-tail-entrant limiting** (pAjrfibDB9).
  - **Muon caveat:** `zero_moments` silently skips params with no `exp_avg`, so Bg would do nothing on Muon's
    `momentum_buffer` (T3).

### 6.8 Metrics and analyses
- **Damage:**
  - Δ(h) curves;
  - G;
  - % of oracle recovery;
  - Kaplan–Meier recovery time;
  - recovery per GPU-second vs rollback.
- **Localization:**
  - AUROC, AUPRC and P@k vs S\* and vs planted truth with decoys;
  - repair-top-k curves;
  - oracle stability across placebo seeds (Jaccard).
- **Validity:**
  - hybrid violation rate;
  - placebo spread;
  - parity residual.
- **Persistence regression** (W2): does a spike's g² predict future g²? This decides per type whether undo is safe.
- **Reporting rule:** everything is reported **per spike type and per scale, never pooled**.

### 6.9 Statistics
- **Unit and clustering.** The unit is the spike instance, clustered by data-order seed × init seed. Seeds now permute
  the data order, not just the init.
- **Primary analysis.**
  - Mixed model Δ ~ branch + site + (1 | seed), plus paired bootstrap CIs.
  - **One** primary contrast (TOST non-inferiority at margin max(0.005, 2× the placebo spread), then superiority).
  - Holm across H1–H18; BH for exploratory ablations.
- **Sample size.** A 4-instance pilot measures the SD, then n is locked: n = max(10, ⌈8.6(σ̂/δ)²⌉) per type.

### 6.10 Leakage and contamination controls
- The Δ eval set uses val batches ≥ 1000. The localizer probe uses val batches 0–3. Assert that `get_batch`'s modulo
  wrap can't make them overlap.
- Seed 0 tunes; seeds 1–9 are held out.
- Decoys and plants come from a seed used for nothing else.
- Every JSON is stamped with the git SHA, cfg, argv, torch/ROCm/hipBLASLt/AOTriton versions and the device.

---

## 7 · Hypotheses and pre-registration

### 7.0 Confirmatory vs exploratory (agreed with the reviewer session, 2026-10-23)
Holm across 18 hypotheses leaves little power, so P1 pre-registers **exactly 4 CONFIRMATORY** tests, Holm-corrected
among themselves:

| ID | Confirmatory test | Prediction | Refuted if |
|---|---|---|---|
| **C-1 mediation, via the M1 damage-writing curve** | Cumulative D(τ) on the log grid τ ∈ {0, 1, 2, 4, 8, 16, 32, 64}. Its integral Δ_B0 − D(0) is the indirect (memory-written) share of end-of-schedule held-out Δ. Pre-registered family for k(τ): a·r^τ (m-echo) minus a v-stall term | Indirect share CI excludes 0 for ≥ 1 data-driven spike type, **and** it rises monotonically across the β1 arms (0.8 < 0.9 < 0.95). The τ½ ordering is secondary | CI ∋ 0 for all types, or a non-monotone order |
| **C-2 protective v.** An out-of-sample test of history coverage (anonymous ICLR 2027 #17880, P0oFJZXAh8) on spike-poisoned states, a perturbation class its abstract doesn't cover | Undo-v-only vs B0 vs undo-m+v (the sign test). M3's refractory curve is **supporting evidence only**: it measures the v-channel's time course (stall cost), not the sign of the protection benefit | Δ(undo-v-only) ≥ Δ_B0 (cleaning v while m's echo stays is *worse* than nothing), while Δ(undo-m+v) < Δ_B0 | Δ(undo-v-only) < Δ_B0 |
| **C-3 interference (M4), channel split without state patching** | Two planted spikes at Δt ∈ {2, 8, 32}, same sign (+G, +G) vs opposite sign (+G, −G). I(Δt) = Δ(both) − Δ(first) − Δ(second). No state is patched, so it is immune to the off-manifold / coverage critique of the undo arms | Derived from the r-law: the m-echo writes most of the drift, and opposite-sign pairs cancel echoes while v adds. So **I_opp(Δt) < I_same(Δt) for Δt ≤ 1/(1−r) ≈ 13** | I_opp ≈ I_same (stall-dominated), or the order reverses |
| **C-4 retroactive repair** | State-only undo (RML) at a **realistic, detector-timed lag** vs (a) the best non-oracle baseline (including pAjrfibDB9's tail-entrant limiting and P0oFJZXAh8's REBUILD_M) and (b) the OLMo 6σ skip-step **on the same spikes** | Non-inferior to (a); better than (b) | Either fails |

**Localization (formerly C-3) is now exploratory.** It has the weakest remaining novelty and the heaviest engineering
(the oracle), and no confirmatory test depends on it. The single-comparator rule still applies, for its exploratory
report.

**Prior work that reshapes C-1 and C-2** (three ICLR 2027 submissions, public since Sept 2026, so prior work for ICML
2027).
- **Status:** [V-abstract]. The OpenReview API records (title, abstract, TLDR, keywords, submission numbers #17880
  and #36730) are saved in `master-research/raw/openreview/*.json` (git-excluded), and both sessions read them.
- **Full texts are blocked** by OpenReview's browser check, so **jaz must read them in a browser** (§13).
- **Two abstract lines that sit close to us:**
  - P0oFJZXAh8: "extended continuations reveal effects that outlast inherited moment differences". We read that as
    memory-written lasting damage, but for checkpoint transplants, not spikes [I].
  - pAjrfibDB9: "Mismatched first- and second-moment time scales can let momentum outpace its normalization,
    amplifying the adaptive update".
    - Their cause, the β1-vs-β2 timescale mismatch, is the same quantity as our r = β1/√β2. A reviewer will say our
      β1 ordering is anticipated.
    - **Answer:** their mismatch concerns spike *onset* and immediate loss. Our β1 ordering predicts the post-spike,
      memory-written share of *lasting* Δ.
- **Our delta stays** the causal decomposition of a spike's *lasting* held-out damage into direct, m-echo and
  v-stall, with β1 ordering.
- **"Newer Is Not Always Better: The Joint Role of AdamW Moment Histories"** (OpenReview P0oFJZXAh8).
  - **What it shows:** moment transplants from 34M to 7B. A newer m with an older v is far more disruptive, explained
    by **history coverage** (v must cover m's gradients; a weighted Cauchy–Schwarz bound on m/√v). REBUILD_M keeps v
    and re-accumulates m. Aligned recovery at 1B/7B with "the original batches, schedules, and data cursors".
  - **What it means for us:** C-2's mechanism is theirs. Undo-v-only creates exactly their newer-m/older-v pair, so
    C-2 is **an out-of-sample test of their mechanism on spike-poisoned states**. Also cite them for K1 (aligned
    recovery), K3 (transplants) and the feasibility bound.
- **"Understanding and Mitigating AdamW Loss Spikes"** (pAjrfibDB9).
  - **What it shows:** GPT-2 355M, 19 captured spike-onset states plus replay. Coordinates newly crossing |u| > 1 are
    170× more frequent at onsets. Selectively limiting new tail entrants beats global shrinking at equal norm.
    "Replay favors sustained over onset-only control".
  - **What it means for us:** for C-1, theirs is *immediate* loss and control effectiveness; ours is the causal
    direct/memory decomposition of *lasting* held-out Δ, with β1 identification. Their sustained > onset result is
    converging evidence. Their selector is an exploratory-localization baseline, and their method a C-4 baseline.
- **Asadi 2023** (2306.17833, deep RL, β2 = 0.999). m-only reset "not effective", v-only "quite effective", both
  best. It doesn't contradict C-2: their m carried no spike, so coverage held. Cite it as the **boundary condition**.
  SPAM has no m-only or v-only ablation.

**C-4's niche must actually be in the spike menu.** If a grad-norm skip-step at the spike step catches every spike,
damage never enters the state and a retroactive tool has no job. So the menu includes:
- **sub-threshold multi-step data spikes** (`repeat_ngram`, width ≥ 3, no single step tripping the 6σ rule);
- **loss-detected spikes**.

**Everything else is EXPLORATORY** (BH-corrected, reported as such): absolute AUROC, H2–H18 below, and every catalog
experiment.

### 7.1 Hypotheses (proxy unless noted; all are proposals for team sign-off)

| ID | Hypothesis | Prediction | Refuted if |
|---|---|---|---|
| H1 carrier | Optimizer state carries part of the lasting damage | (m, v) share of Δ(end) at τ = 10 is ≥ 0.3 for `corrupt_batch` and `repeat_ngram` | CI upper bound < 0.1 |
| H2 migration | The state share decays at a β-set rate | v-arm slope ≈ −ln β2; w/m-arm slope ≈ −½ ln β1 (β cross) | both track the same β |
| H3 transplant | Poisoned state alone damages clean weights | Δ > τ in ≥ 80% of instances; monotone in κ | < 50%, or ρ ≤ 0 |
| H4 QK-Norm | QK-Norm cuts incidence; the share is reported | descriptive | — |
| H5 localization | S\* is sparse and signed predictors find it | \|S\*\| ≤ 3 tensors or ≤ 5% of rows; AUROC ≥ 0.85 vs S\*; +0.10 over magnitude-only on decoys | AUROC < 0.7 |
| H6 undo asymmetry | Undo fixes data-driven spikes, not LR-driven ones | ≥ 0.7 of oracle recovery on `corrupt_batch` (lag ≤ 5); < 0.2 on `lr_bump` | either reverses |
| **H7 PRIMARY** | Selective repair matches or beats the best baseline while editing little | Non-inferior; G ≥ 0.5; ≤ 10% of coordinates edited | TOST fails |
| H8 scale | The H1 sign holds up the ladder | mid rung CI > 0; same sign at 124M (n = 5) | sign flips |
| H9 detector | The lower-tail NIS gives lead time on decay-stage spikes. **Threat:** "Arnold's Bifurcation Principle … Early Warning of Second-Moment Collapse Preceding Loss Spikes" (2026, OpenAlex only, unread); read it before claiming anything | median lead ≥ 1 step at conformal FP ≤ 0.05 | no lead |
| **H10** anti-windup | Post-spike drift on dominated coordinates saturates at D(β1, β2)·η once G/ḡ ≫ 1/√(1−β2), so it is clip-invariant in that regime. **Harm is collective**, so measure it as the drift's projection onto ∇L_val and Bai's top-Ĥ directions, not per-coordinate \|Δw\| (reviewer) | per-coordinate drift within ±20% of 5.84η / 10.47η / 31.8η; projected harm saturates with G | no saturation |
| **H11** SDC split | Undo fixes `sdc_grad`. The parity residual flags `v_bitflip`. v *up*-flips cause no loss spike; v *down*-flips do | ≥ 0.8 / ≥ 0.95 / ≥ 90% silent (up-flips) | any reverses |
| **H12** protection | The integrated v-excess on written rows predicts self-recovery time (R20; read Biswas in full first) | Spearman ≥ 0.5 | ρ ≤ 0 |
| **H13** held-out law | L1 observables at the mid rung and 124M fall inside the 95% PI fitted on 5 small configs | inside the PI | outside |
| **H14** μP collapse | Under μP/CompleteP, \|γ_width\|, \|γ_depth\| < 0.15 for the L1 observables; under SP, γ_width > 0 for Δ_B0 | as stated | SP flat and μP sloped |
| **H15** borrowed trunk | L0 rates at Pythia-160M (and up the ladder) match T1–T4; the H1 sign matches the proxy | ±15% | mismatch |
| **H16** persistence | g² from `corrupt_batch` does not predict future g² (b_J's CI ∋ 0); `lr_bump` does (b_J > 0) | as stated | reversed |
| **H17** jump share | For width-1 spikes, Spearman(J, oracle) ≥ 0.8; it fails at width ≥ 2 as derived | as stated | succeeds at width ≥ 2 (then the derivation is wrong) |
| **H18** cooldown | The LR decay phase reduces the state-carried share, after controlling for η (R12) | φ falls with phase beyond the η effect | no phase effect |

### 7.2 Decision thresholds (to commit before any AMD data; proposals)
- **Kill-test τ:** 0.01 nats.
- **Oracle τ_f:** max(0.005, 5× the placebo spread).
- **Non-inferiority margin:** max(0.005, 2× the placebo spread).
- **n:** n_min = 10 per type, locked after the pilot.
- **Horizons:** {50, 150, 500, end}; the primary horizon is end of schedule.
- **Seeds:** seed 0 tunes; 1–9 are held out; data-order seeds.
- **Natural-spike B\*:** excluded (option b).
- **`Bs` clip:** 0.5 × the trunk's `grad_clip`.
- **β cross:** the arms in §6.3.
- **Matched tokens:** 16,384 per step.
- **ε policy:** pick one before running (§6.6 rule 3).
- **ROCm tolerance:** V7 §4.4 unchanged.

---

## 8 · Experiment program and compute ladder

Costs are MI300X-hours. The working assumptions [A] are:
- fork mode (deterministic, fp32 math): ≈ 37 TFLOP/s effective per GPU;
- trunk mode (bf16 autocast): ≈ 390 TFLOP/s (30% of the 1,307 TFLOP/s dense peak);
- FLOPs ≈ 6·N·tokens.

**`step_timer` replaces all of these on AMD day 1.** At the Oct 2026 median of $2.90/GPU-h, 100 h ≈ $290 and
2,000 h ≈ $5.8k; the GPU itself is free to us.

### 8.1 Tiers (each tier is a complete, publishable unit)

| Tier | Goal | Contents | GPU-h [A] | Feeds |
|---|---|---|---|---|
| **T0 · Bring-up** (week 1) | A trustworthy instrument on AMD | `step_timer`; fork gate in-process **and cross-process through a spike window**; math-SDPA HVP determinism; pinned image `rocm/pytorch:rocm7.14.1_…_pytorch_release_2.10.0` by digest [V: tools pass]; if Δ ≠ 0, try torch 2.13 / AOTriton 0.12b | 3–5 | all |
| **T1 · MVE** (week 2) | Does the headline exist? | 12 proxy instances: `corrupt_batch`, `repeat_ngram` (width 3) and `v_inflate` + decoys, × 4 seeds. **2 × 2 undo arms (none / m / v / m+v) at τ ∈ {0, 10, 30}** (C-1, C-2), plus the w/m/v factorial at τ = 10 and placebos; ≈ 400 forks. Then the β1 arms → localization → the C-4 battery (reviewer's order) | 1–2 | G1–G3 |
| **T2 · ICML core** (weeks 3–10) | P1 | Realistic proxy (warmup + cosine, ≥ 20 tokens/non-embedding param) plus the 6L/384 mid rung, QK-Norm and untied variants. β cross (≈ 1.7 h). Full factorial, transplant, undo and placebos over 5 types × n = 10. Localizers + oracle (3–5 h). Repairs + full baseline battery (6–10 h). SDC/bit-flip arm (≈ 1–2 h). Phase × cooldown (≈ 0.5–1 h). E1 scaling law, SP arm (1.5–2.5 h). **124M n = 5 at matched tokens (6–8 h)**. **Pythia-160M borrowed trunk (≈ 4 h)**. +30% reruns | **50–80** | P1 |
| **T3 · NeurIPS scale-up** (Feb–May 2027) | P2 + P3 | **Pythia ladder:** 410M (≈ 10), 1B (≈ 35), 1.4B (≈ 50), 2.8B n = 5 (≈ 50). μP/CompleteP arm (+1.5). Own 350M rung, 3 seeds (bf16 trunk ≈ 10 h each + forks ≈ 12). Natural-spike program at 124M–410M with stress presets (≈ 100). Optimizer generality (≈ 15–25): **Adam-mini first** (reimplemented in about 30 lines with AdamW state keys; its default name grouping mis-groups our fused `c_attn`), then Muon + AdamW, Adafactor (no m; the lr cap clips `lr_bump`), AdEMAMix. bf16/flash-attention natural explosion at 124M (≈ 10–20). SpikeState production (≥ 200 instances, mostly reuse) | **200–450** | P2, P3 |
| **T4 · Moonshot** (8× MI300X node, 2027) | Real-scale natural spikes, plus breadth | Own **instrumented 1B trunks** (20B tokens, bf16, 8 GPUs ≈ 85 GPU-h per seed) × 3–5 stress seeds. Natural spikes forked single-GPU (1B fits; ≈ 3.3 h per instance). **Pythia-6.9B** borrowed trunk on one 192 GB card (≈ 23 h per instance, n = 5). RL fine-tuning (PPO/GRPO) and ViT/CLIP transfer (50–100). 3B trunk stretch (≈ 770) | **500–2,000** | P2 extension, follow-ups |

**Why the MI300X matters here.**
- One 192 GB card holds fp32 w, g, m and v for models up to about 7B. Forks can stay **single-GPU**, which avoids
  multi-GPU reduction-order nondeterminism [I].
- Trunks can be multi-GPU, because B0 and B\* both fork from the same snapshot. **Only fork-vs-fork determinism
  matters** [I].
- An 8-GPU node runs all 8 factorial cells in one fork's wall-clock time.

### 8.2 Data and models
- **Data:** our nanoGPT stream for the proxy, mid and 124M rungs; a deduped-Pile slice for the Pythia burn-in.
- **Pythia states:** final step only; intermediate states are available on request. **Email EleutherAI in week 1**
  for mid-training states. PolyPythias observed spikes only in 410M seeds 3 and 4, but it **releases no optimizer
  states** [V], so ask about those too.
- **OPEN-1B (now a key asset).**
  - Full optimizer state is public every 100 steps at `gs://gensyn-open-1b/ckpt` [V: HF card + bucket listing].
  - The run used AdamW (0.9, 0.95, 1e-8, wd 0.1) and ended at step 80,957. Each checkpoint has a
    `spike_protocol.json` skip log [V].
  - **Week 1:** read `spike_protocol.json`. **Did the skipped spike steps touch m/v?** Open one DCP shard on the GPU
    host (never the Mac), confirm m and v are present, and run the T12 census (G8).
  - **Then, only if the states around spikes are poisoned:** fork natural OPEN-1B spikes single-GPU (1.6B fits) for
    localization and detection on real natural spikes (H5, H9).
  - **Otherwise:** use the states for pre-spike precursor analysis (Bai's decay stage, W6 whiteness, H9 lead time).

### 8.3 Gates (they route work; they never delete ideas)

| Gate | When | Criterion | If it fails → route |
|---|---|---|---|
| G0 determinism | T0, and after every config change per rung | Cross-process Δ = 0 through a spike window, or the V7 §4.4 tolerance measured *in* spike windows | ROCm checklist → torch 2.13 → tolerance-only claims; BACKUP 2 (fork-free detector + online undo) leads |
| G1 harness sanity | T1 | `v_inflate`: v main effect ≥ 0.8 of Δ at τ = 1; violations ≈ 0 | fix the harness first |
| G2 headline | T1 | H1 CI excludes 0 for ≥ 1 data-driven type at τ = 10 | negative-result framing (§2 pivot); keep building |
| G3 placebo | T1 | placebo spread ≤ ½ the median \|Δ_B0\| | raise K; lengthen the horizon |
| G4 interference sanity (replaces the old localizer gate, which is now exploratory and non-gating) | week 5 | I_same(Δt = 2) < 0 on ≥ 1 spike type | check the M4 harness before reading C-3. The localizer AUROC ≥ 0.7 is reported, not gating |
| G5 repair | week 7 | H7 non-inferiority | measurement-led P1 |
| G6 scale | week 8 | mid-rung sign matches | narrow the claim |
| G7 held-out law | week 9 | H13 inside the PI | report the exponent as the finding |
| G8 Pythia parser | before any borrowed fork | T12 invariant holds on 100% of coordinates | fix the converter |
| G9 multi-GPU | T4 only | trunk resume is bitwise, *or* single-GPU forks only | single-GPU forks only |

---

## 9 · Implementation

### 9.1 Layout (new code; nothing is scaffolded until it's written for real)

```
research/autopsy/schedule.py      warmup+cosine / WSD pre_step; lr_bump that multiplies the schedule
research/autopsy/factorial.py     hybrid() (asserts same step/adam_step; rng from B0), violation_rate(), shares(), 2^4 k-arm
research/autopsy/oracle.py        group_test() over tensor/row/head masks; ddmin fallback
research/autopsy/localize.py      atp_state(), nis_sign(), lambda_decomp(), jump_share(), windup(), magnitude_ctrl() -> Localization
research/autopsy/repair.py        record_grads(), undo(), glr_undo(), route(), certified_restore(), band_clamp()
research/autopsy/faults.py        sdc_grad / v_bitflip / m_bitflip recipes; parity_residual()
research/autopsy/mve.py           GPU-only driver: python -m research.autopsy.mve --kind corrupt_batch --seed 1
research/analysis/eval.py         val_loss(), horizon_probe()          (fills the TODO stub)
research/analysis/stats.py        tost(), paired_bootstrap(), mixed(), holm(), bh()
research/model/neox.py            GPT-NeoX (parallel residual, rotary) for Pythia borrowed trunks
research/tools/pythia_state.py    ZeRO-1 shards -> name-keyed fp32 snapshot (+ T12 census)
research/experiments/autopsy/*.yaml   proxy_realistic, mid_rung, beta_cross, width/depth ladder, pythia_160m
research/tests/test_autopsy.py    CPU tests (§9.2); run in GitHub CI only
```

Starter-code sketches for every helper above are in `master-research/output/08_blueprint.md` §4. They have not been
run.

### 9.2 Tests

**CPU tests, in CI only.** Each one fails if its logic breaks:
- `undo()` equals a direct EMA recomputation with the window replaced (fp64, synthetic gradients);
- forward compensation has no error growth with lag;
- the |m| ≤ C√v invariant holds on random sequences;
- `hybrid()` rejects mismatched step / adam_step;
- `lr_bump` multiplies the schedule;
- the Δ eval batches and the probe batches are disjoint under the modulo wrap;
- the Shapley shares sum to D(111);
- `group_test` recovers a planted S\* within its fork bound;
- bipower shadow v absorbs an isolated jump and fails at width 2;
- bit-flip recipes are deterministic and reversible;
- the parity residual is 0 on honest Adam and nonzero on an injected edit;
- the Localization schema validates;
- `certified_restore` finds the largest α on a synthetic quadratic;
- the outlier router labels AO and LS correctly on synthetic series;
- every JSON is stamped;
- (when Muon is added) snapshot round-trips `momentum_buffer`;
- the NeoX shard loader round-trips a tiny synthetic shard.

**GPU gates on AMD:**
- `step_timer`;
- in-process gate;
- cross-process gate through a spike window;
- math-SDPA HVP determinism;
- the per-rung gate after a grad_accum change;
- `capture --selfcheck`;
- the T12 census on the Pythia shards.

### 9.3 Instrument fixes, before T1
Each fix carries its `docs/improvement.md` ID.
- **A1 (held-out Δ).** `_summarize` uses the last training loss [V: fork.py:306-310]. Δ must come from `val_loss` on
  disjoint batches.
- **A2 (cheap-fix degeneracy).** `Bs` clip = trunk clip, so `Bs` ≡ B0 for `lr_bump` and ≡ B\* for `corrupt_batch`
  [V]. Fix: clip below the trunk clip, or a best-cheap-fix gate.
- **A3 (pre-registration).** Commit τ and the other thresholds (§7.2).
- **Spike role:** the `lr_bump` schedule fix (induce.py).
- **Harness owner, flagged:** the `except Exception` at fork.py:351.

### 9.4 Changes that need sign-off (flagged; not done)
- **ROCm env vars** (`ROCBLAS_DEFAULT_ATOMICS_MODE=0`, `ROCBLAS_USE_HIPBLASLT=0`, `preferred_blas_library`, a fixed
  hipBLASLt workspace, TunableOp off, `TORCH_COMPILE_DISABLE=1`, TF32 off) go **in the AMD launcher**. Only touch
  `determinism.py` with sign-off.
- `snapshot.py`: `momentum_buffer` support (Muon); the bf16 *weight* storage path for ≥ 1B (Task 22; m/v stay fp32;
  never fp16).
- Model variants: QK-Norm, untied embedding, μP/CompleteP (EleutherAI nanoGPT-mup, MIT) and NeoX.
- 124M grad_accum 40 → 2 (matched tokens). This requires re-running the gate.
- `requirements-dev.txt`: drop `pyhessian==0.1`. It calls the removed `torch.eig`, carries GPL headers, and its job
  is covered by eigenthings and curvlinops [V: tools pass].

### 9.5 Tooling decisions (from the 13-item tools pass, `master-research/raw/tools-extensions/results/`)
- **HVP.**
  - The core HVP is about 10 lines of `torch.func` `jvp(grad)`, run inside `sdpa_kernel(SDPBackend.MATH)` **around
    the HVP only, never around training**. Every fused SDPA backend lacks both double-backward and forward-AD [V].
  - Check `jvp` against `vjp` once on the real model (PyTorch #200600).
  - curvlinops 3.0.1 is a dev-only test oracle; its next release changes the API. hessian-eigenthings 1.0.0a5 is
    fine, with `reorthogonalize=True`.
- **ROCm.**
  - Keep `TORCH_ROCM_FA_PREFER_CK` unset: the CK mem-efficient backward hard-codes `deterministic=false` [V].
  - Keep `torch==2.10.0` in `research/requirements.txt`. `check_env()` strips the `+rocm` suffix, so the pin still
    matches.
  - Avoid the `rocm/primus` images: their torch strings break `check_env()`.
  - Pin the image by digest in the launcher.
- **Detectors.**
  - River 0.26 needs numpy ≥ 2.2.5 but our pin is 2.0.2. Port PageHinkley (about 15 lines) and a single-λ e-CUSUM
    (about 20 lines) into `tune_detector.py` instead, and run River only in an isolated CI cross-check.
  - The run-level bound P(any false alarm) ≤ n·α holds only with μ and σ fixed in advance [I].
- **Release (P3).**
  - The NeurIPS 2026 track was "Evaluations & Datasets", and its deadlines have passed [V]. Plan for NeurIPS 2027.
  - HF auto-generates Croissant only for Parquet-convertible data. So: export the manifest as JSONL/Parquet, add a
    FileSet for the safetensors snapshots and the 8 RAI fields by hand, and check it with the hosted validator.
  - `snapshot.push_to_hub` overwrites `latest.safetensors`. The release needs a dataset repo with write-once paths.
  - The proxy battery is about 20–40 GB [A].

---

## 10 · New instruments and tools to release

**Invented algorithms, packaged as three contributions** (reviewer, `13_reviewer_algorithms.md`; full specs in
`docs/EXPERIMENT_CATALOG.md` Part A):
- **RML, Reversible Moment Log** (P1, C-4): a per-coordinate reversible log of what each flagged gradient wrote into
  m and v, with a deferred delete-later / insert-later / weight back-projection decision. PARTIAL, leaning NEW as a
  mechanism.
- **BiH-Adam** (P2): a bipower/MedRV jump-robust second moment. Bipower v is NEW at low confidence; its Huber clip
  is SPAM's.
- **MomentABFT** (P2/P3 or a systems venue): checksum detect/locate/correct for stored Adam moments, paired with the
  elementwise invariant for v down-flips. NEW at medium-low confidence.

The tools below (none found in our bounded searches, §5) each ship with tests and docs.

1. **Autopsy report.** For any spike: w/m/v shares over time, the localized signed support and the recommended
   repair, as one JSON plus a figure.
2. **Counterfactual fork engine.** Factorial scheduler; concurrent cells on one 192 GB card or across 8 GPUs; stamped
   outputs; the gate built in.
3. **Undo-log optimizer wrapper ("state airbag").** A ring buffer, detector and online state-only deletion, with
   measured overhead. A sketch-compressed variant (X3) cuts memory.
4. **Shadow-v monitor.** A zero-fork, zero-log jump-share localizer for any Adam run.
5. **Optimizer-state integrity scanner.** The honest-Adam invariant, parity residuals and poison score for shipped
   `optimizer.pt` files (supply-chain angle, S3).
6. **State-fault injector.** `sdc_grad`, `v_bitflip`, `m_bitflip`, with repair instead of rollback.
7. **SpikeState** (P3). Snapshots, clean twins, oracle maps, Croissant metadata and a leaderboard.

---

## 11 · Timeline (week of)

| Week | Dates | Work | Owner |
|---|---|---|---|
| 1 | Oct 26 | T0 bring-up + G0. §9.3 fixes. Commit §7.2. Push the spike pipeline PR. Email EleutherAI. Check OPEN-1B | spike (jaztulsi) + harness |
| 2 | Nov 2 | T1 MVE → G1–G3 | spike |
| 3–4 | Nov 9–20 | Realistic proxy + mid rung; QK-Norm / untied; β cross; SDC recipes; stress re-search | spike + baselines (Eddie) |
| 5–6 | Nov 23–Dec 4 | M4 interference + M3 refractory forks → G4; oracle/localizers only if ahead (exploratory); Pythia converter + census (G8) | C2 owner [TBD] |
| 7–8 | Dec 7–18 | Repairs + full baseline battery → G5; E1 scaling-law fit | C3 owner [TBD] + Eddie |
| 9 | Dec 21 | 124M n = 5; Pythia-160M; G6/G7 | spike |
| 10 | Dec 28 | Buffer, reruns, figures | all |
| 11–12 | Jan 4–15 | Writing; reviewer simulation; ablation fill-ins | all |
| 13 | Jan 18–22 | **ICML 2027 submission** (dates [UNV]) | all |
| — | Feb–May 2027 | T3 → **NeurIPS 2027** P2 + P3 | all |

---

## 12 · Risks and how each is routed

| Risk | Route |
|---|---|
| ROCm can't reach Δ = 0 | torch 2.13, then tolerance-only claims, then BACKUP 2 leads (fork-free detection + online undo) |
| State share ≈ 0 after about 50 steps | Pivot framing (§2); the undo log as the in-window remedy; the H2 curve is the result |
| Localizer weak (AUROC < 0.7) | λ-decomposition or jump share; the oracle is still a contribution |
| Band-clamp matches our repair | Report it. Localization becomes analysis, and the clamp becomes the practical recommendation |
| Scale critique (F1) | L0/L1/L2 framing, the held-out-rung law, Pythia borrowed trunks, 124M n = 5 |
| Cause vs carrier (Marin, Tang, Li 2026) | Claim "damage *carried* in state", with QK-Norm arms. **Marin is a direct counter-case:** it inferred by elimination that its spikes were "rooted in the model's attention stack rather than in optimizer state", and an optimizer-state rebuild ("necromancy") relapsed [V]. Our factorial tests exactly this split, and the QK-Norm arm reports it either way |
| Concurrent work | Re-run the kill-searches monthly; the ~2026-11-22 concurrent-work cutoff from wave 1 [I] |
| Bus factor 1 (spike role) | Assign C2/C3 owners in week 1 |

### 12.5 · Anticipated objections and where the paper answers them (drafted by the reviewer session)

| # | Objection | Answer (where) |
|---|---|---|
| 1 | "Proxy scale; spikes are induced" | Fig 6 (124M n = 5; Pythia-160M). `repeat_ngram` is a data-driven, OLMo-class cause. Scope is stated in the limitations |
| 2 | "Concurrent work did this" (#17880, #36730; Liu & Li) | #36730 = onset / immediate loss / control; #17880 = checkpoint-recovery coverage; Liu & Li = a linearized *observation-time* response. Ours = the exact *writing-time* kernel of *lasting* spike damage, with β1 predictions and channel interference |
| 3 | "The kernel is a tautology (telescoping)" | The identity only guarantees the total. The claims are about the *shape* (r^τ echo minus a stall term), the β1 ordering and τ½, all pre-registered before data (Fig 1, Fig 2) |
| 4 | "The undo arms create off-manifold states (the coverage critique)" | Hybrid violation rates are reported (Tab 1). Undo-m+v removes both channels coherently. **C-3 (interference) uses no state patching** and tests the same channel structure independently (Fig 4) |
| 5 | "±G injection is unphysical" | +G is the measured corrupt-batch gradient delta (data-derived). −G is its planted mirror, a probe for mechanism, not a claim of prevalence. Both arms are clip-matched |
| 6 | "The effect is chaos, not causation" | Placebo replicates (1-ulp); effects must exceed the placebo spread (Tab 1, G3) |
| 7 | "Is determinism on AMD real?" | G0: cross-process noop Δ through spike windows, or the pre-registered tolerance measured *in* spike windows (Tab 1) |
| 8 | "Why not just skip-step?" | C-4 tests exactly this on spikes that per-step rules miss (multi-step, loss-detected), same instances (Fig 5) |
| 9 | "The theory is trivial EMA algebra" | The closed form is credited (Patitucci & Mokhtari Thm 3.3). The contribution is the *loss-space* test of its predictions: ordering across β1, refractory recovery time, interference sign |
| 10 | "Loss ≠ weight drift, so the r-law prediction is loose" | Pre-registered as a *direction* (ordering), not a value. Measured vs predicted is plotted honestly (Fig 2) |
| 11 | "Cost vs rollback" | Fig 5's GPU-s axis; RML is O(\|flagged\|) memory and FLOPs |
| 12 | "Thresholds chosen after the fact" | Git tag `prereg-v1` before T1 (+ OSF if jaz approves); the null-version abstract is kept in the repo |
| 13 | "Small n" | n is locked from a pilot SD (n = max(10, ⌈8.6(σ̂/δ)²⌉)); data-order seeds; a mixed model; never pooled across types |
| 14 | "Results depend on the horizon" | Primary = end of schedule; Δ(h) curves in the appendix |

---

## 13 · Open issues and sign-offs
- **MUST-READ for jaz, in a browser** (both Claude sessions are blocked by OpenReview's browser check). Save each
  PDF into `master-research/raw/openreview/`.
  1. **pAjrfibDB9**, "Understanding and Mitigating AdamW Loss Spikes" (#36730). Do the replay sections measure
     *lasting* held-out Δ per spike? If yes, C-1 must be re-scoped. The abstract only scores "immediate loss".
  2. **P0oFJZXAh8**, "Newer Is Not Always Better" (#17880). Does it include any spike-poisoned states? If yes, C-2
     becomes a pure replication.
  3. **seU1LYAiwm**, the optimizer-state leakage paper. What is the threat model? It sets the SpikeState release
     policy.
  4. **Biswas**, NeurIPS 2026 OPT workshop (OpenReview VJFfO4tLMB): the exact protection law (H12, A9).
  5. **"Arnold's Bifurcation Principle … Early Warning of Second-Moment Collapse Preceding Loss Spikes"** (2026,
     found on OpenAlex). Check it before claiming H9.
- **Jaz decides (the agents wait):**
  - **(a) Build the code now?** The critical path, all writable and CI-testable before the GPU, never run locally:
    eval.val_loss → RML/undo → mve driver → stats → oracle → localizers.
  - **(b) Pre-registration:** a git tag `prereg-v1` on §7.0/§7.2 before T1, plus an OSF/AsPredicted entry (an
    external action).
  - **(c) Owners** for C2/C3, stats and writing.
  - **(d) Start writing now** (related work → method → prereg appendix → intro, around both abstracts in §2.6).
- **Team sign-off** on §7.2 thresholds, §9.4 changes and C2/C3 owners.
- **Read in full before writing:**
  - Xu 2026 and Xie 2026 (abstract-level only so far);
  - Biswas (NeurIPS 2026 OPT; the basis of H12);
  - the Liu & Li horizons.
- **Kill-search queue:** the X-series ideas in Appendix A haven't been searched yet.
- **Kill-test fork point:** V7 §5 still stands. Pass the manifest's `fork_snapshot` (step ≤ `inject_step`).

---

## 14 · Change log
- **V8 (2026-10-22):**
  - New question (damage migration first) and a narrowed, fact-checked novelty ledger (SWIFT downgrade, naming).
  - Instrument fixes; β1 × β2 identification; Bai-Ĥ threshold; STP with matched tokens and the held-out-rung law.
  - Pythia borrowed trunks; SDC/bit-flip arm; three-paper program; compute ladder T0–T4; route-don't-kill gates.
  - Full idea portfolio. Supersedes V7 (moved to `docs/reference/`).
- **V7 (2026-10-07):** see `docs/reference/PLAN_V7.md` §8.

---

## Appendix A · Idea portfolio (every idea kept)

**Tier:**
- **Core:** in P1 (T0–T2).
- **Scale:** P2/P3 (T3).
- **Moon:** T4 or a follow-up paper.
- **Ctrl:** a control or baseline.

**Verdict:** NEW / PARTIAL / TAKEN, as of the logged searches. **"unsearched"** means a V8 idea not yet
kill-searched.

**Sources:**
- `#n` = `04_ideas_ranked.md`;
- `T`/`A`/`B`/`C` = wave-1 catalogues (`02_*`);
- `W` = `10_wave2_estimation.md`;
- `R` = `10_wave2_radical.md`;
- `S` = `10_wave2_security_unlearning.md`;
- `E` = `10_wave2_scale_theory.md`;
- `X` = new in V8.

### A.1 Measurement and mechanism

| ID | Idea | Verdict | Tier |
|---|---|---|---|
| #1 | Damage-migration curve (2³ factorial × cut times) | NEW for spikes | Core |
| #2 | Poisoned-state transplant + dose-response | NEW | Core |
| #4 | Placebo replicates (1-ulp) | PARTIAL (new here) | Core |
| #10 | Hybrid feasibility \|m\| ≤ C√v + off-manifold control | NEW use | Core |
| #24 | Direct vs indirect path effects from undo vs patch | NEW | Core |
| #25 | Kaplan–Meier recovery vs the k½ law | PARTIAL | Core |
| #22 / K7 | Anti-windup invariance (5.84η) | NEW | Core |
| #23 | Closed-form recovery law for signed poisoning | NEW [I] | Core |
| E2 | β1 × β2 identification cross | PARTIAL | Core |
| R12 | Spike phase × cooldown (cosine, WSD) | NEW | Core |
| R20 | Protection law (v-excess protects the spike's own write) | PARTIAL | Core |
| R27 | Stall + decoupled weight decay: a v→w route set by β2 | NEW | Core |
| R6 | Per-token functional damage map | NEW (as an endpoint) | Core |
| R28 | Linear-mode-connectivity barrier between B0, repair and B\* | PARTIAL | Core |
| R23 | Poison contagion: support growth over time | PARTIAL | Core |
| R1 | Spike information half-life above the placebo floor | NEW | Scale |
| R19 | Beneficial spikes: is any benefit state-carried? | PARTIAL | Scale |
| S8 | Residual-memory audit of v vs ln 2/(−ln β2) | NEW for Adam | Core |
| W2 | Persistence regression: does spike g² predict future g²? | NEW | Core |
| W6 | Innovation whiteness as a fork-free "healed" clock | PARTIAL | Core |
| #55 | Max attention-logit telemetry (Marin bridge) | PARTIAL | Core |
| #7 | QK-Norm ablation of the state share | PARTIAL | Core |
| #53 | Untied-embedding variant (wte artifact) | Ctrl | Core |
| #54 | Stress presets placed on Tang's (β1, β2) phase diagram | PARTIAL | Core |
| E8 | Tang negative control at β2 = 0.6 | PARTIAL | Core |
| E9 | `lr_bump` catapult probe (λ before and after) | PARTIAL | Core |
| X1 | 2⁴ factorial with Adam's step counter k as a 4th mediator | unsearched | Core |
| X2 | State-mediated data influence: the w/m/v split for *ordinary* batches (link to data attribution) | unsearched | Scale |
| X7 | Damage-migration atlas by parameter type (embedding / QK / VO / MLP / LN) | unsearched | Core |
| X9 | Vulnerability forecasting: does pre-spike state predict where a planted spike will do damage? | unsearched | Scale |
| X11 | Zero-parameter migration law: predict φ_state(τ) from β1, β2 and dominance alone, then test it | unsearched [I] | Core |

### A.2 Localization

| ID | Idea | Verdict | Tier |
|---|---|---|---|
| #6 | Adaptive group-testing oracle (+ SuSiE) | NEW for optimizer state | Core |
| #11 | Signed AtP-state + two-sided NIS | NEW | Core |
| #16 | λ_grad decomposition by group + success certificate | NEW | Core |
| #3 | Decoy-augmented planted benchmark | NEW | Core |
| W1 | Jump/continuous split of v (bipower shadow v; jump share) | NEW | Core |
| W3 | Windup score prior (+ band-clamp baseline) | PARTIAL | Core |
| R3 | Forensic recovery of offending tokens from Δlog v of wte rows | TAKEN (seU1LYAiwm; Melis 2019) | BL (privacy note only) |
| R5 | Frequency-calibrated reference v for embedding rows | PARTIAL | Core |
| R16 | Universal poison sites across seeds and rungs | PARTIAL | Scale |
| R29 | Streaming Count-Sketch localization | NEW | Scale |
| R32 | Conformal localization sets | NEW (in context) | Scale |
| R26 | Path patching for optimizer state (cross-tensor influence graph) | NEW | Moon |
| R4 | Sparse autoencoder over optimizer-state rows | NEW | Moon |
| W10 | Covariance localization of noisy SNR estimates | NEW | Scale |
| #29 | Compressed-sensing patch masks (parallel on 192 GB) | PARTIAL | Scale |
| #30 / #31 / #37 | Lanczos mass / Hutchinson diagonal / principal angles | PARTIAL | Ctrl |
| #32 / #33 / #34 | Fisher mismatch / spectral log(v/v̄) / first-mover change point | PARTIAL | Ctrl |
| #19 / A17 | Data-side attribution → aligned natural B\* | PARTIAL | Scale |
| #39 | Synthetic control for the natural counterfactual | NEW | Scale |
| #40 | Spectrum-based fault localization prior (Ochiai) | PARTIAL | Ctrl |
| #41 / #42 / #43 | Learned probe / DAS / Shapley over groups | PARTIAL | Moon |

### A.3 Repair

| ID | Idea | Verdict | Tier |
|---|---|---|---|
| #5 / S7 | Undo log = state-only retroactive deletion | PARTIAL (SWIFT) | Core |
| #12 | Signed, stability-certified, support-limited restoration | NEW | Core |
| W4 | GLR estimate-then-correct undo (no gradient log) | NEW (application) | Core |
| W5 | Outlier-type routing (AO/IO/TC/LS → undo/keep) | NEW | Core |
| W8 / W9 / W11 | Hierarchical inflation / triggered inflation with a Lyapunov bound / interpolation-space ablation | PARTIAL | Scale |
| #17 | LMI minimal diagonal repair | NEW | Scale |
| #21 | Online state airbag (wrapper) | NEW | Core→P3 |
| R14 | Micro-batch-resolved undo | PARTIAL | Scale |
| X3 | Sketch-compressed undo log (memory–accuracy Pareto) | unsearched | Scale |
| R18 | Optimizer hot-swap on S (sign momentum for k steps) | NEW for repair | Scale |
| R7 | Self-healing dual-timescale v (consistency test) | PARTIAL | Scale |
| R10 | Kronecker-factored preconditioners: rank-limited repair becomes well-posed (SOAP/Shampoo) | PARTIAL | Moon |
| R8 | Meta-learned repair operator | PARTIAL | Moon |
| R30 | Repair policy from heterogeneous treatment effects | NEW (bounded) | Moon |
| X6 | Selective micro-warmup on S after restoration | unsearched | Core |
| #44–#50, #58, #59 | m projection, fresh-g² v, re-warm, selective clip/ε/β2, v-floor inoculation, selective m+v+t reset, row restore | PARTIAL | Ctrl |
| #8 | Strong baseline set; best-baseline primary contrast | TAKEN (baselines) | Core |
| #27 / #28 | Edit-size Pareto with matched `Br`; recovery per GPU-second vs rollback | PARTIAL | Core |

### A.4 Detection (BACKUP 2 side product)

| ID | Idea | Verdict | Tier |
|---|---|---|---|
| #60 | Conformal run-level FP guarantee + e-detectors | PARTIAL | Core |
| #61 / H9 | Lower-tail NIS CUSUM lead time | NEW | Core |
| #35 | Multi-stream sparse GLR | PARTIAL | Scale |
| W7 | Extreme-value-calibrated per-coordinate flag (vs SPAM's θ) | PARTIAL | Core |
| #52 | K2-V2 W&B logs: detector recall at 65B (CPU) | PARTIAL | Scale |
| #15 | Parity residual + fault-signature matrix | NEW | Core |
| River | ADWIN / PageHinkley / KSWIN baselines | TAKEN (tools) | Ctrl |

### A.5 Scale, theory and generality

| ID | Idea | Verdict | Tier |
|---|---|---|---|
| E1 | Spike-damage scaling law with held-out rungs | NEW | Core (SP) / Scale (μP) |
| E1-μP | μP/CompleteP "spike coordinate check" | NEW | Scale |
| E3 | Pythia-160M borrowed trunk | NEW for spikes | Core |
| E3+ | Pythia ladder 410M → 6.9B | NEW | Scale / Moon |
| E4 | Offline optimizer-state census of Pythia (T12, ε fraction, row sparsity) | NEW | Core |
| E5 | Ask EleutherAI for mid-training and PolyPythias spike states | — | Core (week 1) |
| E6 | OPEN-1B states: natural-spike forks if the spike steps touched m/v; otherwise precursor analysis only (pending `spike_protocol.json`) | PARTIAL | Scale |
| E7 | Wortsman-style instability extrapolation to 124M | PARTIAL | Core |
| T3–T14 | Closed-form predictions (√β1 band, degeneracy, drift table, width/depth, repair target, ε floor, placebo vs schedule) | [I] | Core |
| #13 | Realistic proxy + mid rung | — | Core |
| #20 | Muon + AdamW control | PARTIAL | Scale |
| R9 | State format as a poison amplifier (8-bit, Adafactor, 4-bit codebook) | PARTIAL | Scale |
| R11 | AdEMAMix slow momentum as an m-channel instrument | NEW | Scale |
| R17 | Scaling law of φ_state(N) | PARTIAL | Scale |
| R25 | MoE expert-state poisoning | PARTIAL | Moon |
| X4 | Autopsy for RL fine-tuning (PPO/GRPO) and continued pretraining | unsearched | Moon |
| X5 | Cross-modality: ViT/CLIP spikes (Wortsman's patch-embedding setting) | unsearched | Moon |

### A.6 Reliability, security and privacy

| ID | Idea | Verdict | Tier |
|---|---|---|---|
| R22 / S4 | Bit-flip vulnerability map; SDC repair instead of rollback | NEW for moments | Core |
| S1 | Spikes as natural availability attacks on optimizer state (freeze / destabilize / steer) | NEW (framing) | Core (writing) |
| S2 | Data-realizable targeted v-inflation (batch selection) | NEW (generator) | Core |
| S3 | Optimizer-state supply-chain integrity scan | NEW | Scale→P3 |
| R21 | Adversarial optimizer-state poisoning (worst-case minimal edit) | NEW | Moon |
| S5 | FedAdam/FedYogi server-state recovery | NEW | Moon (follow-up) |
| S6 | Byzantine "momentum defends" vs "m/v carry damage" tension | PARTIAL | Core (writing) |
| S12 | Ordering-attack spike recipe | PARTIAL | Scale |
| S10 | Damage-aware optimizer-state compression | PARTIAL | Moon |
| S11 | "ROME for optimizer state" (writing frame) | NEW (framing) | Core (writing) |
| R3-privacy | Released optimizer states leak recent batch tokens (affects the SpikeState release) | NEW for spikes | Scale |
| R24 | DiLoCo cross-worker donor repair | NEW | Moon |

### A.7 Benchmarks, tools and systems

| ID | Idea | Verdict | Tier |
|---|---|---|---|
| #14 | SpikeState release (Croissant, leaderboard) | NEW | P3 |
| X8 | Bitwise counterfactual fork engine as a library (8-cell parallel scheduler) | unsearched | P3 |
| #57 | Concurrent bitwise forks on one 192 GB card | PARTIAL | Scale |
| R34 | vmap-batched forks for launch-bound small configs | TAKEN (tool) | Scale |
| R35 | Agentic "autopsy" automation | PARTIAL | Moon |
| R13 | Localizer-in-the-loop data filter | PARTIAL | Moon |
| R15 | Unsupervised spike taxonomy from damage fingerprints | PARTIAL | Scale |
| R2 | Rate–distortion of repair | PARTIAL | Scale |
| R31 / R33 | State drop at stage boundaries / inverse-Adam rewind | TAKEN | Ctrl |
| #51 | Negative-result framing (pivot insurance) | — | Core |
| #62 / #63 | Trimmed-mean micro-batch aggregation; plasticity-transfer controls | PARTIAL | Ctrl |
| X10 | Repair side effects on calibration and rare-token performance | unsearched | Scale |
