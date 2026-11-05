# 13c · Reviewer pass: make PLAN_V8 a best-paper plan (2026-10-23)

**User instruction:** "no more research — work with the other agent to improve the plan, goal: best paper ever."
So this pass adds no new literature. It covers the plan's structure, focus, feasibility and story.

**What a best paper needs:**
1. one surprising, quotable finding;
2. a causal method reviewers can't poke holes in;
3. theory that predicts numbers *before* the experiment;
4. a practical tool that beats strong baselines;
5. honesty about limits.

PLAN_V8 has all five ingredients, but they're buried under 140 ideas. The changes below are ordered by payoff.

---

## 1 · Make the plan tell the new story everywhere (consistency bugs)
The mediation headline (C-1…C-4) was adopted in §0 and §7.0, but four places still tell the old story:
- **§2 paragraph.** It still says "measure that split … as a damage-migration curve". Rewrite it around "who wrote
  the damage: the spike step or Adam's memory of it".
- **§2 titles list is broken.** The bullet list is interrupted by the "P1 reads as one story" paragraph, and the
  "Where Does a Loss Spike Live?" bullet is orphaned below it. Keep one title; move the old one to the change log.
- **§2 pivot.** It still says "state-carried share ≈ 0 after ~50 steps". The mediation null is "direct ≫ indirect →
  clip/skip at the spike step is enough". Say that.
- **§4 P1 row.** Its core claim still lists "β-identified law; causal oracle; undo log plus certified restoration".
  Replace it with C-1…C-4 verbatim.
- **§3.** "Uncommitted on local `main`" is stale; it's now committed on `plan-v8`.

## 2 · Add a paper skeleton (§2.5, new): write the paper backwards
One table where every row is a figure or table in the paper, tied to an experiment ID, a gate and a GPU cost. If an
experiment feeds no row, it isn't P1.

| # | Paper element | Shows | Experiment | Gate |
|---|---|---|---|---|
| Fig 1 | **Mediation decomposition**: end-of-schedule Δ stacked into direct / m-echo / v-stall, vs lag τ, per spike type | C-1 headline | 2×2 undo arms × τ ∈ {0, 10, 30} | G2 |
| Fig 2 | **Theory vs measurement**: predicted (r = β1/√β2, stall length) vs measured indirect share across the β1 arms and G/ḡ doses | C-1 ordering; "theory predicts" | β1 arms + κ doses | — |
| Fig 3 | **Protective v**: Δ for none / undo-m / undo-v / undo-m+v (C-2), with the coverage prediction overlaid | C-2 | same forks as Fig 1 | — |
| Fig 4 | **Localization**: AUROC vs the causal oracle for the signed predictor vs the single comparator; repair-top-k curve | C-3 | oracle + scores | G4 |
| Fig 5 | **Repair Pareto**: gap closed vs edit fraction vs GPU-s, RML vs the baseline battery and OLMo-6σ skip-step, on niche spikes (`repeat_ngram` width 3, loss-detected) | C-4 | repair battery | G5 |
| Tab 1 | Instrument validity: noop Δ (AMD), placebo spread, hybrid violation rate | credibility | T0/T1 | G0, G1, G3 |
| Tab 2 | Main results with CIs, Holm-adjusted, per spike type | C-1…C-4 | all | — |
| Fig 6 (scale) | Indirect share at proxy → mid rung → 124M (n = 5); one Pythia-160M point | external validity | T2 scale arm | G6 |

Also: a **draft abstract** with placeholders ("X% [CI] of lasting damage is written after the spike…") committed
*now*. It forces every experiment to justify itself.

## 3 · Split T2 into P1-core and P1-stretch (feasibility)
**Today:** 13 weeks to the ICML deadline, the GPU not yet handed over, C2/C3 owners TBD, and `research/autopsy/*`
unwritten. T2 as listed (5 types × n = 10, full factorial, transplant, oracle, HVP localizers, certified
restoration, full battery, SDC arm, phase × cooldown, E1 scaling law, 124M n = 5, Pythia-160M) won't fit.

| P1-core (must ship) | P1-stretch (only if ahead at week 7) | → P2 |
|---|---|---|
| Instrument validity (T0/T1) | Transplant + dose-response (K3) | E1 held-out-rung law, μP |
| 2×2 undo arms × τ, 3 spike types (`corrupt_batch`, `repeat_ngram` w = 3, `v_inflate`) × n = 10 | w/m/v 2³ factorial at all τ | Pythia ladder > 160M |
| β1 arms (3) | QK-Norm arm | SDC / bit-flip arm |
| Oracle + cheap localizers (AtP-state, NIS, magnitude, tail-entrant): **no HVP** | λ-decomposition (HVP) | phase × cooldown |
| RML (undo) + baselines (incl. OLMo-6σ, REBUILD_M, tail-entrant limiting) | SCR certified restoration (HVP) | BiH-Adam, MomentABFT |
| 124M n = 5 on C-1 only | Pythia-160M point | OPEN-1B replay |

**Why HVP goes to stretch:** HVP-based localization and SCR are the biggest engineering risk (math SDPA, jvp/vjp
checks, ROCm). None of C-1…C-4 needs them.

## 4 · State a minimum-compute P1 (the Avneh question)
- **The record:** the grant conversation asked for 15 MI300X-h now and ~100 total. Avneh thought ~10 h might be
  enough. §8 says T2 is 50–80 h, and that the "GPU is not the constraint".
- **Add one line:** "P1-core fits in ≤ 20 MI300X-h [A]; P1-stretch adds ~30–60 h."
- **And a slip rule:** if the measured `step_timer` says otherwise on day 1, cut from the stretch column first.
- **Why it matters:** if the real allocation is smaller than hoped, the paper survives, and Avneh sees a plan sized
  to the hardware.

## 5 · Add a GPU-delay rule to the timeline
§11 assumes T0 in the week of Oct 26. The GPU isn't handed over yet. Add: "each week of GPU delay removes one row of
P1-stretch, in reverse order of the table above; the P1-core set is protected; if the GPU arrives after Dec 1,
retarget P1 to NeurIPS 2027 and submit an ICML 2027 workshop paper on C-1 alone."

## 6 · Use the GPU wait: build the code now (critical path)
Every P1-core experiment needs code that doesn't exist yet. All of it can be written and CPU-tested in CI **now**:
- `analysis/eval.py val_loss`;
- RML/undo in `autopsy/repair.py`;
- `hybrid()` in `autopsy/factorial.py`;
- `autopsy/mve.py`;
- `analysis/stats.py` (TOST, paired bootstrap, Holm);
- the oracle.

**Order by what the MVE needs first:** eval → undo → mve driver → stats → oracle → localizers. This is the
highest-leverage work before the GPU arrives; more ideas are not. Each module gets the CPU test listed in §9.2.

## 7 · Make the pre-registration externally verifiable
- **Before T1:** tag a git commit containing §7.0 and §7.2 (`prereg-v1`) and register the same text on OSF or
  AsPredicted, so it carries a third-party timestamp.
- **In the paper:** cite the tag and the OSF link.
- **Why:** a small team on a proxy-scale model gets credibility from methodology. A verifiable pre-registration is
  rare in ML papers, and reviewers notice it.

## 8 · Add an anticipated-objections table (§12.5, new)
| Objection | Where the paper answers it |
|---|---|
| "Proxy scale; spikes are induced" | Fig 6 (124M n = 5, Pythia-160M); `repeat_ngram` is a data-driven, OLMo-class cause; scope stated |
| "Concurrent work did this" (#17880, #36730) | Related work: theirs is onset/immediate loss and checkpoint coverage; ours is the causal mediation of *lasting* Δ for spikes, with β1 predictions |
| "The effect is chaos, not causation" | Placebo replicates; effects > placebo spread (Tab 1) |
| "Determinism on AMD?" | G0 cross-process through spike windows (Tab 1) |
| "Why not just skip-step?" | C-4 tests exactly that, on niche spikes skip-step misses |
| "The theory is trivial EMA algebra" | Fig 2: the theory predicts an *ordering* across β1 that is checked out of sample; the closed form is credited to Patitucci & Mokhtari |
| "Cost vs rollback" | Fig 5's GPU-s axis |
| "Cherry-picked thresholds" | Pre-registration tag + OSF (§7 above) |

## 9 · Name owners
- §11 says C2/C3 owners are "[TBD]", and the plan says bus factor 1.
- jaz must assign them: localizer, repair/RML, stats/eval, writing lead. These are the user's decisions, not ours.
- Known team: jaztulsi (spike), Eddie (baselines, active), Tanish, Avyank (no commits yet), Solomon, Eknoor,
  Alexander, orange-neon (waiting on hardware).

## 10 · Start writing now
- §11 puts writing in weeks 11–12. Instead, draft intro, related work, method and the pre-registration appendix in
  weeks 1–3, while the GPU work runs.
- Related work is already researched (`12_*`, `13_*`).
- Results sections get filled in as gates pass.
- Run the reviewer simulation (`02_skeptic_review` style) on the draft at week 8, not week 12.

## 11 · Review of the "system identification" proposal (M1–M8)

User directive: "new novel ideas, just focus on novelty". No new searches; everything here is judged against the
corpus (02–13).

| ID | Sound? | Fix needed | Novelty vs the corpus | Slot |
|---|---|---|---|---|
| **M1 kernel k(τ)** | Yes. The telescoping Σk = Δ_B0 − Δ_direct is exact. It's an identity, so its value is the *shape* | k(τ) means "the lasting damage added by letting the memory act one more step, given the closed-loop path", not an isolated step effect. Per-τ differences have √2σ noise, so use a log-spaced τ grid {0, 1, 2, 4, 8, 16, 32, 64}, report the cumulative D(τ) with CIs as primary, and pre-register the parametric family (m-echo r^τ minus a v-stall term) | NEW in framing. **Distinguish from Liu & Li:** their response operator says *when an effect is observed* (output time, linearized); k(τ) says *when the lasting effect is written* (mediator time, exact). #36730's "sustained > onset-only control" is a coarse precedent for τ½; cite it | **C-1** |
| **M2 parity** | Yes: odd = first order (+ odd higher orders), even = curvature + v-stall. The ±G injection is a legitimate *planted* intervention (−G isn't data-realizable, like `sdc_grad`) | **Clip asymmetry:** \|g_clean + G\| ≠ \|g_clean − G\|, so the global clip scales the two arms differently and contaminates odd/even. Apply the +G arm's clip factor to both arms, or inject after clipping. Inject at the accumulated (post-grad_accum) gradient. "≫" needs a number | NEW as a measurement. It measures exactly where Liu & Li's linear operator fails | exploratory (strong figure) |
| **M3 refractory χ(Δt)** | The closed form is right on spiked coordinates: χ ≈ 1/√(1 + x·β2^{Δt}), x = (1−β2)G²/ḡ²; recovery at ln x/(−ln β2) (31 at G/ḡ = 10, β2 = .95) | Restrict P to the dominated coordinates, or weight χ by them. Loss-level χ ≠ update-level χ (curvature at the displaced point). Δ(P alone) is a small, noisy denominator, so size P above the placebo spread. **Overclaim:** M3 measures the v-channel's *time course* (the stall side). It is **not** an independent test of C-2's *sign* (protection). Call it "supporting" | NEW as an instrument; the qualitative effect (stale/inflated v) is known (Wortsman, SPAM) | supporting C-2 |
| **M4 interference** | Yes, but it equals h2(0, Δt) only to second order. With large spikes it is the full interaction term | Derived sharp test: the r-law says the m-echo writes most of the drift, so opposite-sign pairs (echoes cancel, v adds) should be *more* sub-additive than same-sign pairs. **Prediction: I_opp(Δt) < I_same(Δt) for Δt ≤ 1/(1−r) ≈ 13.** If the stall dominated instead, I_opp ≈ I_same. No state patching, so it's immune to the off-manifold/coverage critique of the undo arms | NEW (nothing in the corpus) | **C-3** (replaces localization) |
| M5 collapse | Plausible for drift. For *loss* it also needs the curvature to be scale-free, which is not guaranteed | **Pre-register the dose metric x̄ now.** Picking the metric that collapses best afterwards is a forking path | NEW claim, high risk | P2 |
| M6 β-window DiD | Yes. With bitwise forks the DiD *is* the exact interaction contrast; no parallel-trends assumption is needed | Mid-run β changes alter bias correction: state the convention | PARTIAL (A15 as an instrument; Tang = onset) | P1-stretch |
| M7 dead time | Plausible | "CPU only" analysis still must not run on the Mac: run it on the AMD host or in CI. Few natural spikes, so hazard estimates will be weak | PARTIAL (Tang, Bai) | P2 / backlog |
| M8 staged repair | Sound as a test; the theory is honestly ambiguous | — | NEW operator (RML) | exploratory |

**Recommended confirmatory set:** C-1 = M1 (integral > 0 with CI; β1 ordering of the integral); C-2 = undo-arm sign
(M3 supporting); C-3 = M4 sign test; C-4 unchanged. Localization moves to exploratory. That is the right call: its
novelty is the weakest left, it carries the heaviest engineering (oracle), and no confirmatory test now depends on it.

**Title:** "The Afterlife of a Loss Spike: System Identification of How Adam's Memory Writes Lasting Damage" works.
It replaces the earlier title because it names the instrument (kernel / refractory / interference) and the finding.

---

**Not changed:** the core values, the C-1…C-4 definitions, the statistics, the instrument fixes, and the catalog as
the long-term backlog.
