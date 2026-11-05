# 13d · Paste-ready drafts for PLAN_V8: §2.6 paper skeleton and §12.5 objections (2026-10-23)

Built on the M-ideas that survived review (13_reviewer_plan.md §11):
- **C-1:** M1 kernel;
- **C-2:** undo-arm sign test, with M3 supporting;
- **C-3:** M4 interference sign test;
- **C-4:** RML repair.

Every number in brackets is a placeholder. Nothing has run.

---

## §2.6 · Paper skeleton (P1), written backwards

**Title:** *The Afterlife of a Loss Spike: System Identification of How Adam's Memory Writes Lasting Damage*

**Placeholder abstract** (≈ 200 words; fill brackets from Tab 2; the null version is below):

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
quantify the window in which retroactive repair still helps." Keep both versions in the repo, so the result can't
steer the framing.

### Figures and tables (every P1 experiment must feed a row)

| # | Element | Shows | Experiment (forks per instance) | Gate / test |
|---|---|---|---|---|
| **Fig 1** | **Damage-writing kernel.** Cumulative D(τ) with CIs (primary) and the fitted k(τ), split into direct / m-echo / v-stall, one panel per spike type | C-1 headline; τ½ | M1: undo-m+v at τ ∈ {0, 1, 2, 4, 8, 16, 32, 64}, plus B0 and B\* (10) | G2; **C-1** |
| **Fig 2** | **Theory vs measurement.** Predicted vs measured indirect share and τ½ across the β1 arms (0.8 / 0.9 / 0.95) and the G/ḡ doses | "The theory predicts an ordering before the run" | M1 on the β1 arms; κ doses | C-1 (ordering) |
| **Fig 3** | **Protective v.** Δ for none / undo-m / undo-v / undo-m+v at τ ∈ {0, 10}; inset: refractory χ(Δt) vs the 1/√(1 + x·β2^Δt) prediction | C-2 sign; v time course | undo-m and undo-v at 2 τ (4); M3 probes at Δt ∈ {2, 8, 32} plus P alone (4) | **C-2**; M3 supporting |
| **Fig 4** | **Spike interference.** I_same(Δt) vs I_opp(Δt) for Δt ∈ {2, 8, 32} | Channel structure, with no state patching | M4: both-same, both-opp, second-alone (±) per Δt (12) | **C-3**: I_opp < I_same for Δt ≤ 13 |
| **Fig 5** | **Repair Pareto.** Gap closed vs edit fraction vs GPU-s: RML vs the baseline battery (OLMo-6σ skip-step, REBUILD_M, tail-entrant limiting, ZClip, AdaGC, SPAM, Adam-MR, band-clamp) on niche spikes (`repeat_ngram` width 3, loss-detected) | C-4 | repair battery (≈ 12) | **C-4** |
| **Fig 6** | **Scale.** Indirect share at proxy → mid rung → 124M (n = 5) plus one Pythia-160M point | External validity | M1 subset τ ∈ {0, 8, 64} (5) | G6 |
| Tab 1 | Instrument validity: noop Δ on AMD (in-process and cross-process, through spike windows), placebo spread, hybrid violation rate | Credibility | T0/T1 | G0, G1, G3 |
| Tab 2 | Main results: C-1…C-4 with CIs, Holm-adjusted, per spike type, never pooled | — | all | — |
| App. | Parity Δ_odd / Δ_even (M2, clip-matched); localization vs oracle (exploratory); pre-registration text and tag | Exploratory | M2 (2) | BH |

**Fork budget [A]:**
- **Per instance:** ≈ 10 (M1) + 4 (C-2) + 4 (M3) + 12 (M4) + 2 (M2) ≈ **32 forks**.
- **Core proxy set:** 3 spike types × n = 10 = 30 instances ≈ 960 forks.
- **β1 arms:** M1 only, 2 extra arms × 3 types × n = 5 ≈ 300 forks.
- **Repair battery:** ≈ 12 × 30 ≈ 360 forks.
- **Total:** ≈ 1,600 proxy forks, plus 124M (5 instances × 5 forks).
- **Hours:** fork lengths run to end of schedule, so the builder converts this to MI300X-h with `step_timer` on day 1
  and checks it against the ≤ 20 h P1-core line. If it's over, cut M3 to 2 Δt values and M4 to Δt ∈ {2, 8} first.

**Writing order (start now, before the GPU):** related work (corpus 02–13 is ready) → method (harness, estimands,
the M1/M3/M4 definitions) → pre-registration appendix → intro around the placeholder abstract. Results get filled in
as gates pass. Reviewer simulation at week 8.

---

## §12.5 · Anticipated objections and where the paper answers them

| # | Objection | Answer (where) |
|---|---|---|
| 1 | "Proxy scale; spikes are induced" | Fig 6 (124M n = 5; Pythia-160M). `repeat_ngram` is a data-driven, OLMo-class cause. Scope is stated in the limitations |
| 2 | "Concurrent work did this" (#17880, #36730; Liu & Li) | Related work: #36730 = onset / immediate loss / control; #17880 = checkpoint-recovery coverage; Liu & Li = a linearized *observation-time* response. Ours = the exact *writing-time* kernel of *lasting* spike damage, with β1 predictions and channel interference |
| 3 | "The kernel is a tautology (telescoping)" | The identity only guarantees the total. The paper's claims are about the *shape* (r^τ echo minus a stall term), the β1 ordering and τ½, all pre-registered before data (Fig 1, Fig 2) |
| 4 | "The undo arms create off-manifold states (the coverage critique)" | Hybrid violation rates are reported (Tab 1). Undo-m+v removes both channels coherently. **C-3 (interference) uses no state patching at all** and tests the same channel structure independently (Fig 4) |
| 5 | "±G (opposite-sign) injection is unphysical" | It's a planted probe, like SDC or fault injection, used to identify mechanism, not to claim prevalence. Same-sign pairs use data-realizable spikes. Both arms are clip-matched |
| 6 | "The effect is chaos, not causation" | Placebo replicates (1-ulp); every effect must exceed the placebo spread (Tab 1, G3) |
| 7 | "Is determinism on AMD real?" | G0: cross-process noop Δ through spike windows, or the pre-registered tolerance measured *in* spike windows (Tab 1) |
| 8 | "Why not just skip-step?" | C-4 tests exactly this on spikes that per-step rules miss (multi-step, loss-detected), same instances (Fig 5) |
| 9 | "The theory is trivial EMA algebra" | The closed form is credited (Patitucci & Mokhtari Thm 3.3). The contribution is the *loss-space* test of its predictions: ordering across β1, refractory recovery time, interference sign |
| 10 | "Loss ≠ weight drift, so the r-law prediction is loose" | Pre-registered as a *direction* (ordering), not a value. Measured share vs predicted is plotted honestly (Fig 2) |
| 11 | "Cost vs rollback" | Fig 5's GPU-s axis; RML is O(\|flagged\|) memory and FLOPs |
| 12 | "Thresholds chosen after the fact" | Git tag `prereg-v1` before T1 (+ OSF if jaz approves); null-version abstract kept in the repo |
| 13 | "Small n" | n locked from a pilot SD (n = max(10, ⌈8.6(σ̂/δ)²⌉)); data-order seeds; a mixed model; never pooled across types |
| 14 | "Results depend on the horizon" | Primary = end of schedule; Δ(h) curves in the appendix |
