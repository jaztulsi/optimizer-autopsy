# Docs map

| Read | File | What it is |
|---|---|---|
| **1st** | [`../PLAN_V8.md`](../PLAN_V8.md) | **Governing plan** (2026-10-22): question, novelty ledger, methodology, pre-registration, compute ladder, timeline, idea portfolio |
| 2nd | [`../CLAUDE.md`](../CLAUDE.md) | Guardrails and harness interfaces (binding for every session) |
| 3rd | [`EXPERIMENT_CATALOG.md`](EXPERIMENT_CATALOG.md) | 16 invented algorithms (A1–A16), about 70 experiment families, sweep templates, the GPU day-1 queue (all write-and-wait for AMD) |
| 4th | [`../research/README.md`](../research/README.md) | Code layout and per-module status |
| as needed | [`improvement.md`](improvement.md) | Audit of instrument and process holes (A1–C10). Its fixes are folded into PLAN_V8 §9.3 / §7.2 |
| as needed | [`../research/spikes/RESEARCH.md`](../research/spikes/RESEARCH.md) | Spike-role deep research (detection, natural spikes, repair math) |
| as needed | [`../research/analysis/ATTRIBUTION.md`](../research/analysis/ATTRIBUTION.md) | Causal-map definition and the Gate-B verdict |
| evidence | [`../master-research/output/`](../master-research/output/) | Research run: audits (02), novelty (03), ideas (04, 10), methodology (06), validation (07), blueprint (08), analysis (09), review (11), fact-checks (12), reviewer-session pass (13) |
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
