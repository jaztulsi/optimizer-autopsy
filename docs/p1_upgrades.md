# P1 upgrades: ten changes that most raise acceptance odds (2026-10-27, after the MI300X pilot)

**Evidence state.** The pilot (`results/mi300x_2026-10-27/t1_report`, n = 1 each, exploratory) shows width-1 spikes at step
160 leave lasting damage of 0.0003–0.0028 nats. That is at the 1-ulp chaos scale (P+/B0 gap up to 0.0027). The one
exception is corrupt_batch at β1 = 0.8, at 0.0117. There, undoing m+v at τ0 removes 98% of the damage, and undoing v
alone makes it worse. P1 therefore needs (i) a dose high enough to clear chaos, (ii) chaos-calibrated statistics and
(iii) scale and baselines that a reviewer cannot wave away. Costs assume about 0.012 GPU-h per proxy branch, the
measured co-tenant throughput.

| # | Upgrade (builds on) | Reviewer objection it answers (§12.5) | Novelty / kill-search | GPU-h · dev-days | First concrete test |
|---|---|---|---|---|---|
| 1 | **Forgetting threshold in chaos units.** Planted-G dose c ∈ {1,3,10,30} × phase (descent / plateau / late), with d = Δ/σ_chaos. Output: c\*, and the x̄ (M5) → d triage curve (R1, M5, R12) | #6 chaos; "washes out" (the pilot) | **NEW, thin.** One search found no spike-damage-vs-seed-variance threshold. Cite and distinguish 2602.02400 (noisy data → *divergence probability* over 20 seeds) and the butterfly line (Summers & Dinneen 2021; Frankle 2020; Altıntaş 2025). OpenReview unsearched | 1.2 · 0 (code in PR #18) | prereg-v2 H-A, seed 0, 3 phases |
| 2 | **Chaos-calibrated stats.** Paired K = 8 placebo ensembles per instance; n = max(10, ⌈8.6(σ̂/δ)²⌉). The pilot σ̂ ≤ 0.0027 gives **n = 10** at δ = 0.005 and **n = 16** at δ = 0.002. TOST equivalence with a ±1 σ_chaos margin for every "within chaos" or "no effect" claim; a mixed model (seed random); Holm across C-1…C-4 (G3, §7.2) | #13 small n; #6; #12 | Not a novelty claim | 3.6 (10 seeds × ~30 branches) · 1 | Rerun the pilot's corrupt_batch with `--placebo 8` on seeds 0–2. Measure σ_chaos and fix δ in prereg-v2 |
| 3 | **v as an emergent cooldown (mechanism).** Undo v, then apply a global lr(t) matched to √(v̂_clean/v̂_spiked). If this equals B0, v acts as a schedule; if not, v's protection is coordinate-selective, which ties C-2 to C2 (R20, K2) | #9 "trivial EMA algebra" | **NEW, thin.** One search: Bai 2026 and Wortsman 2023 treat the *opposite* case (stale, too-small v). No post-spike cooldown test found | 0.3 · 0 (in PR #18) | prereg-v2 H-D at c = 10 |
| 4 | **Baselines a reviewer will demand.** PaLM/OPT **rollback + data skip at matched compute** (the industry default), OLMo 6σ skip, Bskipstep_det, SPAM, ZClip, AdaGC, StableAdamW/Adam-atan2, Adam-MR, v-only reset. Plus *prevention* arms (QK-norm, z-loss) to show that repair still matters (§6.7) | #8 why not skip; #11 cost vs rollback | Ctrl | 6–10 · 2 (rollback arm missing) | Add `Brollback(L, skip)` to the repair battery and run H-E at c\* |
| 5 | **Stronger models.** 124M n = 5 at matched tokens; **Pythia-160M borrowed trunk**, then 410M (released Adam states) (E3, E7) | #1 proxy scale | Pythia forking = TAKEN as a setting; the spike results on it are new | 6–8 + 4 (+10 for 410M) · 2 (NeoX shard loader) | Pythia-160M: census of the released states (E4), then a 1-instance fork gate |
| 6 | **Bit-flip / SDC map of m and v**, with invariant-based repair instead of rollback (R22/S4, K9, MomentABFT) | reliability relevance; #1 "induced" | **NEW for moments, thin.** One search: SDC studies (2502.12340; LLM-PRISM) inject into activations and gradients, not optimizer state | 1–2 · 2 | Flip exponent bits in v at 3 sites × 3 parameter types, then run an \|m\| ≤ C√v scan and RML repair |
| 7 | **Natural spikes.** OPEN-1B states (E6) plus 124M stress-preset natural spikes captured by `spikes/capture.py` | #1 "spikes are induced" | PARTIAL | 0 now; 10–20 later · 1 | Fetch OPEN-1B `spike_protocol.json` (no shards). Fix the capture selfcheck yaml bug first |
| 8 | **Zero-parameter migration law.** Predict φ_state(τ) and c\* from β1, β2 and x̄ alone, then test it on the β cross (X11, E2, T3–T14) | #3 tautology; #10 loose r-law | **PARTIAL.** Memory-transient decay rates ρ∨β1, ρ∨β2 (2605.04269), β1 = β2 scale sensitivity (2601.21739), the β1–β2 stability boundary (2609.18314) and the single-spike closed form (Patitucci & Mokhtari). New only as a pre-registered *loss-space* prediction of c\* and φ_state | 1.7 · 2 (theory) | Pre-register the prediction for β1 = 0.95; the pilot ordering is not used |
| 9 | **Damage atlas by parameter type** (embedding / QK / VO / MLP / LN). Group-wise undo at c\* scored against a group-testing oracle (X7, K5) | "where is the damage?"; supports C2 | PARTIAL | 1 · 1.5 | Group undo of m+v at c\*, 5 groups, seed 0 |
| 10 | **Ensemble-Lyapunov triage (new tool).** The K placebo forks already span the run's expanding directions. Project the spike's (w, m, v) delta onto that subspace to predict d *before* forking (R1, X9) | cost; "practical use?" | **PARTIAL (method) / NEW (application), thin.** Micro-ensemble FTLE estimation (2512.16179), ensemble spread as predictability (2604.12809) and input-space FTLEs of DNNs (2306.12548). None of them applies this to training trajectories or spike triage | 0.5 · 3 | On U1's runs, correlate the projected norm with the measured d across doses |

**Ranking (acceptance odds per unit cost):** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10.
- **1–3 decide whether P1 has a finding.** All three are already coded or need only reruns.
- **4–5 decide whether reviewers believe it.**
- **6–10 add breadth.** Skip any of them that slips past the ICML deadline.

**Owners (proposal):**
- **Jasraj:** 1 and 7 (dose and phase recipes, natural spikes, the capture fix).
- **Eddie:** 4 (rollback arm, StableAdamW, prevention arms).
- **Avyank:** 2 and 8 (power, TOST, mixed model, the law).
- **Us (the Claude sessions):** 3, 5, 6, 9 and 10 (code, GPU runs on the MI300X, report).
- **jaz:** signs off prereg-v2 before any run of 1–3.

**Kill-search log (this pass, three standard web queries, abstracts only):**
- threshold: no hit (closest is 2602.02400);
- v-cooldown: no hit (Bai 2506.04805 and Wortsman 2304.13013 cover the opposite case);
- m/v bit-flips: no hit (2502.12340, LLM-PRISM).

Second pass, 2026-10-28, three more queries:
- #8 PARTIAL, as in the table;
- #10 PARTIAL / NEW application, as in the table.

OpenReview could not be searched: a `site:openreview.net` query returned no OpenReview pages, and the browser check
still blocks direct fetches (§13). An OpenReview pass by jaz in a browser is still open for #1, #3, #6, #8 and #10.
None of these is in `references.bib` until verified at the source.
