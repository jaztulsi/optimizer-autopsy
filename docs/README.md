# Docs map

| Read | File | What it is |
|---|---|---|
| **start** | [`SUMMARY.md`](SUMMARY.md) | **The whole project in one file (updated 2026-10-29):** a short version under 2,000 characters, then a long version covering the idea, approach, tests, what's new, implementation, MI300X results and what changed since V8. It deliberately leaves out papers, timelines and owners; those stay in PLAN_V8 |
| **1st** | [`../PLAN_V8.md`](../PLAN_V8.md) | **Governing plan** (2026-10-22; state updated 2026-10-29): question, novelty ledger, methodology, pre-registration, compute ladder, timeline, idea portfolio |
| 2nd | [`../CLAUDE.md`](../CLAUDE.md) | Guardrails and harness interfaces (binding for every session) |
| 3rd | [`prereg_v2.md`](prereg_v2.md) | **Next GPU run's pre-registration** (draft, needs sign-off + `git tag prereg-v2`): dose × phase × placebo × cooldown, H-A…H-F, the repair battery |
| 4th | [`p1_upgrades.md`](p1_upgrades.md) | Ten ranked P1 upgrades (U1–U10) with kill-search verdicts, costs and owners (2026-10-28) |
| 5th | [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md) | 16 invented algorithms (A1–A16), about 70 experiment families, sweep templates, the GPU day-1 queue |
| 6th | [`../research/README.md`](../research/README.md) | Code layout and per-module status |
| status | `status.html` (on `main`; reaches `plan-v8` when PR #18 merges) | Status page: run order for the next MI300X session, contributor table, dependency map |
| results | [`../results/mi300x_2026-10-27/`](../results/mi300x_2026-10-27/) | First MI300X session: T0 gates, `step_timer`, the exploratory T1 pilot |
| as needed | [`improvement.md`](improvement.md) | Audit of instrument and process holes (A1–C10). Its fixes are folded into PLAN_V8 §9.3 / §7.2 |
| as needed | [`../research/spikes/RESEARCH.md`](../research/spikes/RESEARCH.md) | Spike-role deep research (detection, natural spikes, repair math) |
| as needed | [`../research/analysis/ATTRIBUTION.md`](../research/analysis/ATTRIBUTION.md) | Causal-map definition and the Gate-B verdict |
| evidence | [`../master-research/output/`](../master-research/output/) | Research run: audits (02), novelty (03), ideas (04, 10), methodology (06), validation (07), blueprint (08), analysis (09), review (11), fact-checks (12), reviewer-session pass (13: plan, novelty, drafts, algorithms) |
| citations | [`../master-research/references.bib`](../master-research/references.bib) | Verified entries only |

## reference/ (superseded where they differ from V8; still useful)
- [`PLAN_V7.md`](reference/PLAN_V7.md): its §4 locked decisions still hold unless V8 changes them.
- [`PLAN_V6.md`](reference/PLAN_V6.md): long-form math, Path B, cut order.
- [`BUILD_PLAN.md`](reference/BUILD_PLAN.md): the 26-task per-file plan.
- [`context-ai.md`](reference/context-ai.md): full context packet (§14 is the state as of Aug 2026).

## archive/ (historical snapshots; don't treat as current)
`STATUS_REPORT.md` (2026-08-12), `update.md`, `todo.md`, `EXPLANATION.md` (all 2026-08-31).

## Not in git
- `master-research/raw/`: deep-research JSON outputs and the RESEARCH.md v1 backup.
- `master-research/sources/`: downloaded skill sources.

Both are git-excluded via `.git/info/exclude`.
