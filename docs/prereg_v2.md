# Pre-registration v2: spike damage against the chaos floor (DRAFT, needs jaz's sign-off + `git tag prereg-v2`)

Written 2026-10-27 after the first MI300X session, **before** any run it governs. Freeze it by tagging the commit
that contains this file; nothing below may change after the tag.

## Status of what came before
- The T1 pilot of 2026-10-27 (`results/mi300x_2026-10-27/`, tag `prereg-pilot-2026-10-27`) is **exploratory**. Its
  thresholds were frozen only after the data existed, so it is reported as a pilot and never as confirmation.
- What the pilot showed: at end of schedule a width-1 `corrupt_batch` spike at step 160 left 0.0003–0.0117 nats of
  held-out damage, about the size of the 1-ulp chaos floor (P+ vs B0 gap up to 0.0027). The one instance above
  τ = 0.01 (β1 = 0.8) had undo-m+v ≈ 0, and undo-v alone made the damage worse (0.0134 vs 0.0117).

## Design (proxy, corrupt_batch, seed 0, constant LR, batch 64, max_steps 2000)
- **Dose:** planted-G coefficient c ∈ {1, 3, 10, 30} (direction fixed, amplitude only; `probes.plant`).
- **Arms at each dose:** none (B0), undo-m, undo-v, undo-m+v at τ = 0; plus the **global cooldown** arm (undo v, then
  lr(t) = lr · mean_i √(v̂_clean,i / v̂_spiked,i), matched per step).
- **Phase:** inject at 160 (descent), a plateau step, and a late step (last 20% of the schedule).
- **Memory length:** β1 ∈ {0.8, 0.95} (β2 = 0.95) at c = 10 only.
- **Chaos floor:** K = 8 placebo forks per phase. Primary channel: a 1-ulp nudge of w at the fork. Secondary: the same
  nudge in g. σ_chaos(h) = SD of Δ val across the K placebos at horizon h.
- **Effect unit:** d = Δ(arm) / σ_chaos(h_end), using the w-channel floor of the same phase.

## Hypotheses (directions fixed now)
| ID | Prediction | Pass criterion |
|---|---|---|
| H-A | A forgetting threshold exists: d(B0) rises with c | d(B0) ≥ 2 at some c ≤ 30 in ≥ 2 of 3 phases; report c* = smallest such c per phase |
| H-A2 | The threshold moves with memory length | d(B0, c=10) at β1 = 0.95 > at β1 = 0.8 (direction only) |
| H-B | Protective v (C-2) | at every dose with d(B0) ≥ 2: Δ(undo-v) > Δ(none) |
| H-C | Memory writes most lasting damage | at every dose with d(B0) ≥ 2: 1 − Δ(undo-m+v)/Δ(none) ≥ 0.5 |
| H-D | v as an emergent cooldown (two-sided, both outcomes reported) | \|Δ(cooldown) − Δ(none)\| ≤ 1 σ_chaos → "v acts as a global schedule"; otherwise "v's protection is coordinate-selective" |

## Repair battery (C-4), at every dose with d(B0) ≥ 2, acting at detector lag L = 3 after the spike
Arms (all retroactive and state-only, with no recompute and no data shift):
- `RML-full`: undo m and v.
- `RML-cool`: undo m; replace v's inflation with the global cooldown above.
- `RML-sel`: undo m; cap v only on the top 1% of coordinates by x_i = (1−β2)G_i²/v̄_i.

Comparators (already in `fork.make_baseline_branches`): skip-step (detector-timed), SPAM, ZClip, AdaGC. Rollback cost is
counted as L + 1 recomputed steps.

| ID | Prediction | Pass criterion |
|---|---|---|
| H-E | Retroactive repair works after the spike has already been applied | the best RML arm closes ≥ 80% of Δ(none) at lag L, where detector-timed skip-step closes ≤ 20% |
| H-F | Selective is enough | `RML-sel` is within 1 σ_chaos of `RML-full` while editing ≤ 2% of the **v** coordinates (its m undo is dense) |

## Rules
- Seed 0 only. This is a dose-finding run; replication on seeds 1–4 at the chosen doses is the confirmatory step.
- Non-survivors are reported, never dropped.
- Effects with \|d\| < 2 are reported as "within chaos" and never interpreted.
- Prior work to cite and distinguish (verify before adding to `references.bib`): Summers & Dinneen 2021; Frankle et
  al. 2020 (linear mode connectivity / stability); Altıntaş et al. 2025 ("The Butterfly Effect"). The new part is
  spike damage measured in units of that floor, and the m/v decomposition above it.
