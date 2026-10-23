# Attribution (the C2→C3 science layer)

Attribution ties the **localized poison** (from `research/localizer/`) to the **counterfactual fork
outcome** (from `research/harness/fork.py`): does repairing *exactly* the localized directions recover
the run, while repairing random/other directions does not? That is the causal claim the paper makes.

Code: `research/analysis/attribution.py`. This doc is the spec + honest status; trust the code over any
prose that drifts.

## What it computes

- **The effect `Δ`.** For a fork branch, `Δ = val_loss(branch) − val_loss(B*)`, where `B*` is the clean
  counterfactual (spike toggled off, data stream held aligned — never "skip to the next batch"). Δ is
  measured on held-out validation, not the training stream, so a branch can't "recover" by memorizing.
- **The causal map.** A table over (repair site) × (component `w`/`m`/`v`) of `effect(c) = Δ_noop −
  Δ_repair-c`, with the random-subspace control `Br` subtracted and paired 95% CIs per cell. "Causal"
  is earned only where the random-subspace control (`Br`) is beaten — otherwise it's "interventional
  attribution."
- **Necessity & sufficiency.** Is the localized set `P` **necessary** (repairing its complement does
  *not* recover) *and* **sufficient** (repairing just `P` does)?
- **`calibrate()` — mandatory cost control.** Run ~20 full-length forks, score whether a *short* fork
  preserves the **branch-ordering** of Δ (rank-correlation across branches, not just a lead-time
  number), pick the shortest length that preserves ranking, run the big sweep there. This is what keeps
  the attribution battery inside the GPU budget (see `research/kaggle/step_timer_results.md`).

## Gate B — the cheap-fix kill-test verdict (this part is BUILT)

Before the expensive localizer/repair machinery is trusted, `attribution.py` renders a
**PROCEED / PIVOT / INCONCLUSIVE** verdict from the Task-9 cheap-fix battery Δ's:

- `gate_b_verdict(deltas, *, tau, recipes_total=4, confidence=0.95, seed=0) -> dict` — `deltas` is
  `{recipe: {branch: [Δ per matched seed]}}`; a branch **recovers** on a recipe iff its paired-bootstrap
  CI upper bound `≤ tau` (τ = recovery margin). Decision:
  - **PROCEED** — the cheap fix `Bs` (skip+clip) leaves a gap on ≥1 recipe (CI lower bound `> tau`) →
    build the localizer/repair.
  - **PIVOT** — `Bs` recovers on *every* recipe → the cheap fix suffices → C1+C2 paper.
  - **INCONCLUSIVE** — a `Bs` CI straddles τ → need more seeds/recipes.
- The verdict is **PROVISIONAL** unless all `recipes_total` recipes are present, and `render_verdict()`
  hard-labels a partial run as provisional in the output itself (banner + tag), so a partial signal can
  never be misread later as the binding gate.
- `_paired_ci()` is a stdlib-only percentile bootstrap (deterministic via `seed`) — no scipy/statsmodels
  pin needed here, so it runs on the CPU CI runner and the GPU image alike.

## Current state (verify against the code)

| Function | State |
|---|---|
| `gate_b_verdict`, `render_verdict`, `_paired_ci` | ✅ implemented (stdlib-only, CPU-safe) |
| `attribute(snapshot, localization, forks)` | ⬜ TODO — the causal map above |
| `necessity_sufficiency(forks)` | ⬜ TODO |
| `calibrate()` (short-fork length picker) | ⬜ **does not exist anywhere** in the repo yet |

## Interfaces it consumes / must define

- **Forks** — `research/harness/fork.py`: `run_fork(...) -> list[float]` losses; `run_branch`/
  `run_cheap_battery` return `{name, losses, final, survival}` dicts; `make_branches(cfg, recipe, *,
  clip_norm)` builds `B0/B*/Bg/Bs`. Δ's for Gate B come straight from these.
- **A `localization`** — from `research/localizer/` (unbuilt). `attribute()` names `localization` as an
  input, but **its shape is defined nowhere**. Attribution and `repair.as_intervention` both consume it,
  so the localizer/repair roles must define the contract first (e.g. `{param_name: score}` + a selected
  top-`k` group set). See `CLAUDE.md` §7.2.
- **Held-out val loss** — the eval path (`research/analysis/eval.py`, ⬜) must exist for real Δ's beyond
  the kill-test's training-loss proxy.

## Read next

- `CLAUDE.md` §7 — the real harness interfaces and the localizer/repair output-contract obligation.
- `docs/reference/context-ai.md` §15.3 — the causal-map definition this doc summarizes; §4.9 — the effect `Δ`.
- `docs/reference/PLAN_V6.md` §3.6–3.9 — poison score, repair operator, and when selective repair suffices.
- `research/harness/fork.py` — the branch menu and the `Δ==0` gate every Δ rests on.
