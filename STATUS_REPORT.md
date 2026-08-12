# Optimizer Autopsy — Status Report

**Generated:** 2026-08-12 · **Branch:** `main` @ `0d5cd0e` · **Scope:** read-only snapshot of the repo + local environment.

This report is evidence-based. Every claim is tagged **measured** (backed by a run/test I or a prior
session executed) or **planned/assumed** (asserted in docs but not backed by a committed artifact).
Where a number appears, its source is named. Nothing here is rounded up.

> **Provenance caveat, stated once and applied throughout:** the Task 4 / Task 5 result numbers
> (`max|Δ|=0`, `loss 10.85→4.73`) exist **only in committed markdown** (`update.md`, `todo.md`),
> self-reported from prior Kaggle sessions. **No run log, results table, or W&B export is committed
> to the repo** (`git ls-files` finds zero result/log/figure artifacts). So those two results are
> *credible but not independently verifiable from the repo alone*. This is the single biggest
> rigor gap and is called out again in §5.

---

## 1. Environment & Tooling Audit

| Tool | Version | Auth / config state |
|---|---|---|
| git | 2.39.2 | n/a |
| gh | 2.96.0 | **AUTHENTICATED** — `jaztulsi`, scopes `repo, workflow, gist, read:org` |
| kaggle | 2.2.4 | **AUTHENTICATED** — `~/.kaggle/kaggle.json` present (user `jastulsi`) |
| hf (huggingface_hub) | 0.34.1 | **NOT AUTHENTICATED** locally — `hf auth whoami` → "Not logged in" |
| wandb | — | **NOT INSTALLED** locally |
| pre-commit | — | **NOT INSTALLED** locally (config file exists, see below) |
| ruff | 0.16.2 | installed **this session** |
| sloppylint | 0.5.1 | installed **this session** |
| pip | 26.0 | n/a |
| python3 | 3.14.3 (Homebrew) | n/a |

**Finding (not repaired, per read-only instruction):**
- `hf` is **not logged in locally** and `wandb`/`pre-commit` are **not installed locally**. None of
  this blocks work: HF/W&B credentials live in **Kaggle Secrets** and are consumed on the GPU, not
  the driver machine; `wandb` ships in the Kaggle image. Local HF login is only needed if we want to
  manage the artifacts repo from this machine. **No action taken — flagged for your decision.**

**Secrets (presence only, never values):**
- `HF_TOKEN`: **not in local env**; no `.env` file. Source of record = Kaggle Secrets (per `update.md`).
- `WANDB_API_KEY`: **not in local env**; same as above.
- `research/harness/secrets.py` resolves both at runtime from env → Kaggle Secrets → Colab userdata → `.env`.

**Secret-hygiene test — measured:** `python -m research.tests.test_secrets` → **`secrets tests OK`**
(3/3 pass). No secret literal is committed to the tracked tree.

**Requirements files:**
- `research/requirements.txt` — **numerics-critical pins only**, confirmed **zero diff** from HEAD
  (`git diff --quiet` clean). Contents: torch 2.10.0, numpy 2.0.2, scipy 1.16.3, safetensors 0.7.0,
  huggingface_hub 1.11.0, datasets 5.0.0, tiktoken 0.12.0, wandb 0.26.1, pyyaml 6.0.1, tqdm 4.66.4.
- `requirements-dev.txt` (new, repo root, uncommitted) — dev/analysis only, not read by
  `preflight.check_env()`: ruff 0.16.2, sloppylint[rich] 0.5.1, statsmodels 0.14.6,
  curvlinops-for-pytorch 3.0.1, hessian-eigenthings 1.0.0a5, pyhessian 0.1, kaggle 2.2.4.

**Uncommitted working-tree state (from this session, before the read-only instruction):**
- Modified (cosmetic `ruff format` only, reviewed line-by-line as non-semantic): `data/prepare.py`,
  `harness/trunk.py`, `model/nanogpt.py`, `tests/test_determinism.py`, `tests/test_secrets.py`.
  The two load-bearing files (`harness/determinism.py`, `harness/preflight.py`) were **excluded** and
  are untouched.
- New/untracked: `.pre-commit-config.yaml`, `requirements-dev.txt`, `research/SLOPPYLINT_REPORT.json`.

---

## 2. Project Status (ground truth from file contents + tests)

Legend: ✅ done-and-verified · 🟡 code-complete, DoD/verification pending · ⬜ stub (docstring + `TODO`, no implementation).

### Module-by-module

| Module | State | Evidence |
|---|---|---|
| `harness/determinism.py` (59 LOC) | ✅ | Task 4 primitive verified (see §2 gates). No stub markers. |
| `harness/preflight.py` (72 LOC) | ✅ | `check_env()` prints `check_env OK` on Kaggle T4 (per `update.md`, Task 1). |
| `harness/secrets.py` (65 LOC) | ✅ | `test_secrets` passes locally (measured this session). |
| `harness/trunk.py` (152 LOC) | 🟡 | Trains on T4 per docs (loss 10.85→4.73). `resume_trunk()` is an explicit stub (`raise NotImplementedError`, "lands with Task 6"). Result not backed by committed log. |
| `harness/snapshot.py` (7 LOC) | ⬜ | Task 6. Three `TODO`s: `capture/save/load/push_to_hub`. |
| `harness/fork.py` (7 LOC) | ⬜ | Task 7. Three `TODO`s: `run_fork/short_fork/fork_matrix`. |
| `data/prepare.py` (197 LOC) | 🟡 | `get_batch` bitwise-identical across processes (Task 2 DoD per docs). **Caveat:** `hf_revision="main"` in both presets — **not pinned to a commit SHA** (2 `TODO`s). Non-reproducible data source until fixed. |
| `model/nanogpt.py` (128 LOC) | ✅ | 19 defs; trained on T4 (Task 5). No stub markers. |
| `configs/__init__.py` (17 LOC) | ✅ | Config loader. |
| `localizer/{curvature,snr,poison}.py` | ⬜ | The C2 instrument. All ≤6 LOC, docstring + `TODO` only. |
| `repair/operator.py` | ⬜ | The C3 repair operator. Stub. |
| `baselines/{skip,clip,spam,zclip,adagc,reset}.py` | ⬜ | All 2–4 LOC stubs. |
| `spikes/{induce,k2,tune_detector}.py` | ⬜ | Stubs. |
| `analysis/{eval,stats,attribution,figures}.py` | ⬜ | Stubs. |
| `experiments/proxy/smoke.py` | ⬜ | `raise SystemExit("TODO: implement proxy smoke test")`. `config.yaml` exists. |
| `experiments/llm124m/run.py` | ⬜ | `raise SystemExit("TODO: implement 124M run driver")`. `config.yaml` exists. |
| `theory/` | ⬜ | Only `README.md` tracked. No AR(2)/spectral-radius notebook or code exists yet. |
| `tests/{test_determinism,test_secrets}.py` | ✅ | test_secrets measured-pass; test_determinism verified on Kaggle (docs), not runnable locally (torch absent). |

**Summary count:** ✅ 7 · 🟡 2 · ⬜ ~24. The repo is **foundation + partial instrument**; everything from
the localizer onward (the actual scientific contribution C2/C3) is unimplemented.

### Git

- Last commit: `0d5cd0e` (2026-08-11) — "Add reuse map: external tools/prior-art mapped to pipeline tasks…"
- Branch `main`, **in sync with `origin/main`** (0 ahead, 0 behind).
- Uncommitted changes: as listed in §1 (5 cosmetic-format edits + 3 new files).

### Gate status

Gates are defined in `BUILD_PLAN.md`. **Distinguish the determinism *primitive* (Task 4, proven)
from the fork *gate* (Task 7, which consumes it but has not been run because `fork.py` is a stub).**

| Gate | Defined | Status | Evidence / why |
|---|---|---|---|
| Determinism **primitive** (Task 4) | bitwise replay, `max|Δ|=0` over 50 steps, CPU + Kaggle GPU | **PASSED (measured, doc-sourced)** | `bitwise replay OK on cuda: 50 steps, max\|Δ\|=0` — per `update.md`/`todo.md`; **no committed log**. |
| Trunk-trains DoD (Task 5) | loss decreases on real GPU | **PASSED (measured, doc-sourced)** | `loss 10.851→4.730 / 200 steps`, Tesla T4; **no committed log**. |
| **Fork determinism GATE (Task 7)** — noop-vs-noop `Δ==0` | THE product gate; nothing causal proceeds until `Δ==0` at proxy | **NOT YET RUN** | `fork.py` is a stub. The primitive is proven, but the gate that runs two noop branches through the fork driver has never executed. |
| Cheap kill-test / "method is dead" (Tasks 8–9) | does skip+clip already beat everything? PROCEED/PIVOT verdict | **NOT YET RUN** | `spikes/induce.py`, `baselines/*` are stubs. |
| Calibration gate (Task 17) | short fork preserves Δ branch-ordering vs full fork | **NOT YET RUN** | depends on fork driver + attribution (all stubs). |
| GO/NO-GO #1 (Task 18) | `Bv` beats `B0` and random control `Br` (paired CIs) | **NOT YET RUN** | localizer/repair unimplemented. |
| Proxy smoke gate (Task 20) | whole pipeline in <10 min, asserts `Δ==0`, emits table+figure | **NOT YET RUN** | `smoke.py` raises `SystemExit("TODO")`. |

---

## 3. Current Conclusions

**Empirically established (measured, with the provenance caveat that logs aren't committed):**
1. Bit-identical replay is achievable in this codebase: `max|Δ|=0` over 50 steps on **both** CPU and
   a Kaggle T4. This is the necessary precondition for every downstream causal claim.
2. The proxy nanoGPT + AdamW trunk loop learns: cross-entropy `10.85 → 4.73` over 200 steps on a T4.
3. Secret hygiene holds: no token literal is in the tracked tree (independently re-verified this session).
4. The data reader `get_batch` is deterministic/offset-correct (Task 2 DoD, doc-sourced).

**Still hypothesis / not yet measured:**
- That the **fork driver** reproduces `Δ==0` for noop-vs-noop (Task 7) — *the* gate. Unproven; `fork.py` is empty.
- Everything the project is actually *about*: that loss spikes poison `(m, v)` in localizable
  directions (C2), and that rank-limited repair of those directions recovers the run better than
  baselines (C3). **Zero evidence exists yet** — the localizer, repair operator, spike inducer, and
  baselines are all stubs.

**Which decision branch is the project on?** It is **before its first scientific decision gate.**
Per `BUILD_PLAN.md`'s own rule — *"No causal word is allowed until the proxy gate reads Δ==0"* — the
project is still in the **instrument-construction phase** (Phase 1), having cleared the determinism
*primitive* but not the fork gate. No PROCEED/PIVOT/delocalized-poison outcome has been reached
because no kill-test or attribution has been run. Any statement about the method "working" would be
unsupported today.

---

## 4. Remaining Work (dependency order)

Estimates assume Kaggle free tier (30 GPU-h/week, 12 h/session) and the existing proxy scale.

| # | Task | Unblocks | Definition of Done | Est. |
|---|---|---|---|---|
| 1 | **Pin `hf_revision` to commit SHAs** (`data/prepare.py`) | reproducible data for every downstream run | both presets use a fixed SHA, not `"main"` | ~30 min, no GPU |
| 2 | **Task 6 — snapshot** (`snapshot.py`) | forks, localizer (needs `(w,m,v)`) | capture/save/load round-trip bit-identical; bf16(124M)/fp32(proxy); HF Hub push/pull | 1–2 days; short GPU checks |
| 3 | **Task 7 — fork driver + determinism GATE** (`fork.py`) | **all** causal work; this is C1 | noop-vs-noop `Δ==0` on proxy/fp32, committed as a test | 2–3 days; GPU |
| 4 | **Tasks 8–9 — spike induction + cheap kill-test** | the biggest de-risk: is the method even needed? | PROCEED/PIVOT verdict prints with a Δ table | 1–2 days; GPU |
| 5 | **Tasks 10–12 — localizer** (`snr.py`, `curvature.py`, `poison.py`) | repair; C2 novelty | high-signal directions score above noise floor; can wrap `curvlinops` for HVPs (see §5) | 3–5 days; GPU |
| 6 | **Tasks 13–15 — repair operator + baselines** | attribution; C3 novelty | `fork(snapshot, 8 branches)` emits Δ + effect(w/m/v); baselines wired as forks | 4–6 days; GPU |
| 7 | **Tasks 16–18 — eval + attribution + calibration + GO/NO-GO #1** | scale decision | bitwise val-loss; short-fork ordering calibrated; `Bv` vs `Br` paired CIs | 3–4 days; GPU-heavy |
| 8 | **Task 19 — theory notebook** (AR(2)/spectral-radius) | paper §4 | notebook predicts spike onset; predictions vs measured overlaid | 2–3 days; mostly CPU |
| 9 | **Task 20 — proxy smoke gate** (`smoke.py`) | gates any 124M spend | one command runs whole pipeline <10 min, asserts `Δ==0`, emits table+figure | 1 day; CPU/T4 |
| 10 | **Tasks 21–24 — 124M robustness, figures, paper, repro package** | submission | per BUILD_PLAN | weeks; budgeted GPU |
| — | **Task 25 (410M)** | — | ⛔ credits-gated, **not free** — out of scope | — |

**Stall flag (called out explicitly):** `harness/trunk.py` is marked "done" (Task 5) but its companion
`resume_trunk()` has been a `NotImplementedError` stub since it was written, and the **entire pipeline
past Task 5 has not advanced** — snapshot (Task 6) is the next gate and is still an empty file. The
foundation has been polished repeatedly (README, CI, tooling, reuse map — all this session) while the
**first unbuilt scientific component, the snapshot/fork spine, has not been started.** That is the
critical path; the recent commits do not touch it.

---

## 5. Recommendations to Materially Improve the Repo (prioritized)

**P0 — Close the evidence gap (rigor; low cost, high credibility).**
The Task 4/5 numbers are doc-only. Commit the actual artifacts: the Kaggle run log and a small
`results/` JSON (step, loss, `max|Δ|`) for the determinism + trunk runs. **Impact:** a skeptical
reviewer can verify the two things that are currently just asserted. **Cost:** ~1 hr (pull existing
Kaggle output, commit under `results/`). *Not done here — this report is read-only.*

**P0 — Pin the dataset revisions.** `hf_revision="main"` makes the corpus non-reproducible; a
TinyStories re-push would silently change every shard. **Impact:** removes a reproducibility landmine
before any result depends on it. **Cost:** 30 min.

**P1 — Reuse reference code where it doesn't touch novelty.** The paper's novelty is C1+C2+C3, not
baseline/curvature plumbing. Concretely (licenses verified in `research/EXTERNAL_TOOLS.md`):
- `curvature.py` → wrap **curvlinops** (MIT) for Hessian/GGN matvecs instead of hand-rolling
  HVP+eigsh. Saves ~2–3 days and reduces a correctness-risk surface. Cross-check with
  **pytorch-hessian-eigenthings**.
- `stats.py` → use `scipy.stats.bootstrap` (already pinned) + `statsmodels.multipletests`. Near-zero risk.
- `baselines/zclip.py` → wrap **ZClip** (Apache-2.0). **SPAM/AdaGC must be reimplemented** (SPAM has
  **no license** → cannot vendor; AdaGC is PaddlePaddle). This is a real constraint, not optional.

**P1 — Determinism CI cannot currently catch a regression.** `test_determinism.py` runs in CI, but on
**CPU only** — a change that breaks determinism *on GPU* (the environment that matters) would pass CI
green. **Impact:** the project's load-bearing property can silently regress. **Mitigation:** add a
scheduled/manual Kaggle job that runs the determinism test on a T4 and reports back (the harness to do
this headlessly already exists). **Cost:** ~half day.

**P2 — Every wrapped optimizer/baseline must pass the determinism gate before it counts.** External
optimizer code frequently introduces nondeterministic kernels; wiring it into a fork without re-running
`Δ==0` would corrupt every Δ silently. Make "passes noop-vs-noop `Δ==0`" a required check in
`fork.py`'s test, not a manual step.

**P2 — Build the repro package incrementally (Task 24), not at the end.** Each time a gate passes,
commit its log + the exact config + seed under `results/<task>/`. Retrofitting a reproducibility
package after 20 tasks is far more expensive than appending to it now.

**P3 — Theory-vs-measurement gap.** The AR(2)/spectral-radius theory (Task 19) does not exist yet
(`theory/` is a README). When built, its predicted spike-onset step must be overlaid against a
*measured* induced spike — otherwise §4 of the paper is a derivation with no empirical anchor. Plan the
theory notebook to consume the same induced-spike logs the kill-test produces, so prediction and
measurement share inputs.

---

*Read-only snapshot. No installs, auth changes, downloads, or commits were performed to produce this
report beyond the local `ruff`/`sloppylint` install already disclosed in §1, which predated the
read-only instruction.*
