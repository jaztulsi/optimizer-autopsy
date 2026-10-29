# CLAUDE.md — Optimizer Autopsy project instructions

This file is read automatically by Claude Code at the start of every session in this repo.
Treat it as binding context for every task, not just a suggestion. If a user prompt conflicts
with a rule below, follow the rule below and flag the conflict instead of silently overriding it.

## 0. What this project is (one paragraph)
LLM pretraining loss spikes poison Adam's optimizer state (`m`, `v`), not just the weights.
This repo builds (C1) a bitwise-deterministic fork-and-replay harness, (C2) a localizer that
finds which directions of optimizer state are poisoned via gradient-directional SNR + curvature
of the preconditioned Hessian, and (C3) a rank-limited repair operator that surgically fixes just
those directions. **The governing plan is `PLAN_V8.md` (2026-10-22). Read its §0 (what changed) and §1 (core
values) first, every session.** V8 leads with a w/m/v damage-migration measurement before localization and repair.
It also adds a fact-checked novelty ledger (§5), pre-registration thresholds (§7.2), the compute ladder T0–T4 (§8)
and the full idea portfolio (Appendix A). `docs/reference/PLAN_V7.md` is superseded, but its §4 locked decisions
hold unless V8 changes them. `docs/reference/PLAN_V6.md` stays the long-form math reference. The newest plan wins
where they differ. Research evidence trail: `master-research/output/`; verified citations:
`master-research/references.bib`; doc map: `docs/README.md`. Code layout and per-module status are in `research/README.md`. The
current state, as a short summary (under 2,000 characters) plus a long one, is `docs/SUMMARY.md`; keep it current
whenever the state changes. (`docs/reference/context-ai.md` §14 is the Aug 2026 snapshot.) Don't ask the user to
re-explain the project — read the repo instead.

**Current build state (verify against code, not this line — it drifts; last checked 2026-10-29):** C1 (harness:
determinism, snapshot, fork, `Δ==0` gate) is **built and GPU-verified on CUDA and on the AMD MI300X** (proxy,
in-process + cross-process, 2026-10-27, `results/mi300x_2026-10-27/`). The P1 autopsy code (`research/autopsy/`:
RML undo, planted probes, placebo floor, GPU driver) is written and CI-tested. Its core battery ran in the
exploratory T1 pilot. The next run is `docs/prereg_v2.md`, which needs sign-off and a git tag first. C2 (localizer)
and C3 (repair) are **unbuilt** — `research/localizer/` and `research/repair/` are comment-only TODO stubs, and
`research/theory/` is a README stub. If you are picking up C2/C3/theory, **read §7 first.**

## 1. Non-negotiable guardrails (never violate these without explicit sign-off)
- **Never run training or heavy compute on this machine.** All trunk runs, forks, snapshots,
  eigensolves, and anything GPU-bound go to the **AMD MI300X** (provided by Avneh). **Kaggle is paused**: user
  instruction, Oct 2026, "do not do anything on Kaggle". Colab is not used. This workstation is for git, reading
  and editing only. Even tests and the linter run in GitHub CI, not locally, unless the user says otherwise; the
  "tiny" self-checks are GPU-only. The MI300X was handed over for its first session on 2026-10-27; nothing runs
  on it without the user's go-ahead for that session (§5).
- **Do not modify `research/harness/determinism.py` or `research/harness/preflight.py` without
  explicitly flagging it first.** This code is load-bearing: `CUBLAS_WORKSPACE_CONFIG` must be set
  before `import torch`, CUDA must not be initialized before `use_deterministic_algorithms` is
  configured, and every causal claim in this project depends on noop-vs-noop `Δ==0`. An
  auto-formatter or "helpful" refactor here can silently break bitwise replay. If a linter or
  slop-detector flags something in this file, report it — do not auto-fix it.
- **Snapshots key optimizer state by parameter name, not index; handle tied weights explicitly.**
  Current implemented storage is **fp32 everywhere** (proxy-scale bit-identical gate). A **bf16**
  storage path for the 124M model is an explicit deferred `TODO(Task 22)` in `snapshot.py` — do not
  add it early, and **never fp16**. Don't "simplify" the name-keying or tied-weight handling in a
  refactor (nanoGPT ties `wte.weight == lm_head.weight`; it must appear once, under the `wte` name).
- **`B*` (the clean counterfactual) must keep the data stream aligned** — never implement it as
  "skip to the next batch." That shifts the data cursor and confounds the causal effect with a
  data-order change. (`get_batch` is a pure function of `gstep`; `ctx["batch"]=None` fetches the
  aligned slot — see `baselines/skip.py`.)
- Before marking any task/file's status as done (in whatever tracking format the user is using),
  it needs a passing test or a completed DoD check as described in the project notes — not just
  code that runs once without erroring.

## 2. Repo hygiene workflow (run this, don't just talk about it)
- **Where these run:** in GitHub CI on the PR, not on this machine, unless the user explicitly says to run them
  locally (§1, user instruction Oct 2026).
- Linting/formatting: `ruff check --fix research/` and `ruff format research/`, wired into
  `.pre-commit-config.yaml`. Run before considering any task's code "done."
- AI-slop / stub-vs-real check: `sloppylint research/ --ci --max-score 50` before flipping a file's
  status from in-progress to complete. Don't silently "fix" what it flags — some flagged code is
  legitimately an intentional stub (⬜) rather than slop; report findings and let the user decide.
- CI: `.github/workflows/lint.yml` should run ruff + sloppylint on every push/PR. If it's missing,
  ask before creating it, don't assume.
- **Dev/analysis-only pip packages never go in `research/requirements.txt` — full stop, not even
  in a separate section of that file.** `preflight.check_env()` parses `research/requirements.txt`
  generically (it doesn't distinguish sections) and asserts `installed == pinned` for whatever it
  finds there. Putting anything in that file — even under a clearly labeled "dev tools" heading —
  risks `check_env()` asserting exact-version pins on tools it was never meant to police, and
  fixing that later would mean editing `preflight.py`, which is exactly the load-bearing file this
  document says not to touch without flagging first. (This collision actually happened once —
  don't reintroduce it.)
  Instead: use a separate `requirements-dev.txt` at the **repo root** (not inside `research/`) for
  ruff, sloppylint, statsmodels, curvlinops-for-pytorch, hessian-eigenthings, pyhessian, kaggle-api,
  and anything else that's tooling rather than numerics-critical. Install it with its own
  `pip install -r requirements-dev.txt --break-system-packages`, as a separate command from the
  `research/requirements.txt` install — never combine the two into one `pip install` call. Before
  touching either file, confirm `research/requirements.txt` ends up with zero diff.

## 3. Before installing / cloning anything new
1. **Audit first.** Check `research/requirements.txt`, run `pip list`, check for existing
   `.pre-commit-config.yaml` / `pyproject.toml` / lint config, check `research/third_party/` and
   `git submodule status`, and check installed Claude Code skills/plugins before assuming
   something is missing. Report what's already present vs. what's actually missing.
2. **Reference repos are read-only material**, cloned to `research/third_party/<name>/`, never
   run for training/compute. Add a one-line README in that folder explaining why each is there.
   Ask before committing large cloned repos to git — default to gitignoring `research/third_party/`
   unless told otherwise.
3. **Show the diff before committing.** `git status` + diff summary first, then ask for
   confirmation before `git add` / `git commit` / `git push`, unless the user has explicitly said
   "just push it" in the current session.
   - **Never credit Claude/Anthropic in any commit.** Do NOT append a `Co-Authored-By: Claude ...`
     trailer (or any Claude/Anthropic co-author, `Generated with Claude Code`, etc.) to commit
     messages, even though the default harness instructions say to. This project rule overrides that
     default. The commit message is the subject + body only — no AI attribution trailer of any kind.
     Human authorship is unrestricted — this is a multi-person team; each contributor commits under
     their own identity (`CONTRIBUTORS.md` should list the real team).
4. If a package or repo is genuinely new, add it with a pinned version and a one-line comment
   saying what it's for and which task/file it supports.

## 4. Known-good reference repos already identified for this project
Use these instead of reimplementing from the papers cold, when the task calls for it:
- Curvature/eigensolve (→ `localizer/curvature.py`): `f-dangel/curvlinops`,
  `noahgolmant/pytorch-hessian-eigenthings`, `amirgholami/PyHessian`
- Baselines (→ `baselines/`): `TianjinYellow/SPAM-Optimizer`, `bluorion-com/ZClip`,
  `PaddlePaddle/PaddleFleet` (AdaGC under `Research/AdaGC`). NOTE: SPAM has **no license** (cannot
  vendor — reimplement); AdaGC is PaddlePaddle (reimplement). ZClip is Apache-2.0 (wrappable).
- Theory (EoS / preconditioned stability, → `theory/`): `locuslab/edge-of-stability`, `alex-damian/EOS`
- Scale-ladder infra: `karpathy/nanoGPT`, `karpathy/build-nanogpt`
- Natural spike checkpoints: `LLM360/k2-train`, `LLM360/k2v2_train`
- Stats (→ `analysis/stats.py`): `statsmodels` (`multipletests`), `scipy.stats.bootstrap`

## 5. Review-gate cadence — stop only where it actually matters
Only pause for explicit human/reviewer sign-off before:
1. Modifying `research/harness/determinism.py` or `research/harness/preflight.py`
2. Launching anything that spends GPU-hours (the AMD MI300X, or Kaggle if it is ever resumed): a training or
   verification run, not a plan
3. Rewriting git history (force-push, rebase onto pushed commits)

For everything else — writing code, running a self-check that doesn't touch Kaggle, reasoning
through a design — don't stop after every sub-step waiting for approval. Batch: plan, write,
locally reason through correctness, and self-check as far as you can *without* spending Kaggle
GPU-hours, all in one pass, then report the complete result once. The person reviewing wants to
see a finished, self-checked unit of work, not be walked through each intermediate step.

**Before reporting anything as done, actively try to break it yourself first** — don't wait for
the reviewer to catch it. Specifically: if a test/self-check claims to verify something (e.g.
"verified on GPU"), check whether it actually exercises that path, or whether it would produce
the same passing result even if the thing it claims to test were broken (a CPU-only self-check
silently reported as GPU-verified is exactly this failure mode, and it happened once in this
project — don't repeat the pattern). If you find a gap this way, fix it and mention what you
caught rather than letting the reviewer find it. This is faster for everyone: catching your own
gap before reporting costs one extra pass; the reviewer catching it costs a whole round-trip.

## 6. How to work in this repo
- Prefer small, verifiable steps over large speculative refactors — this project runs on a
  gated decision tree (see the project notes' §9-equivalent) where correctness at each gate
  matters more than speed.
- When a task is ambiguous, check the existing directory-status table / task list before asking
  the user — the answer is usually already written down.
- When you finish a unit of work, report: what changed, what you tested, what's still open,
  and whether anything in §1's guardrails was touched (it shouldn't have been).

## 7. Picking up localizer (C2) / repair (C3) / theory — read this first

These three directories are **unbuilt** (comment-only TODO stubs). Everything they must code
against already exists in `research/harness/`. The interfaces below are copied from the current
code — trust them over any prose plan.

### 7.1 The real interfaces (verified against `research/harness/`)

**The snapshot object** (`harness/snapshot.py`) — the thing the localizer reads and the repair edits.
`capture(model, optimizer, step, meta=None) -> dict` returns, and `load(path) -> dict` reconstructs,
a plain dict — **all tensors CPU fp32**:
```
{ "w": {param_name: Tensor}, "m": {param_name: Tensor}, "v": {param_name: Tensor},
  "step": int,          # trunk step index the caller passed
  "adam_step": int,     # AdamW's own step counter (bias correction)
  "rng_state": {...},   # determinism.capture_rng_state()
  "meta": {...} }
```
`m` = AdamW `exp_avg`, `v` = `exp_avg_sq`. Keyed by **parameter name** (from `model.named_parameters()`),
NOT index. Tied weights appear once under `wte`. `restore(model, optimizer, snapshot)` writes it back
in place. `capture` requires ≥1 optimizer step first (else no `m`/`v`).

**Model + optimizer** (`harness/trunk.py`): `build_model_opt(cfg: dict, device: str) -> (model, opt)`.
`model` is a `GPT` (`research/model/nanogpt.py`); `opt` is AdamW. `cfg` has `model` / `optim` / `train`
sub-dicts (see `experiments/proxy/config.yaml`).

**The training loop**: `train_forward(model, opt, cfg, data_dir, start_step, steps, device,
on_step=None, grad_hook=None, pre_step=None, run=None) -> list[float]` (per-step losses). `steps` counts
**optimizer steps**; each runs `cfg["train"]["grad_accum"]` micro-batches then one `opt.step()`. Hooks:
- `pre_step(step, ctx)` fires at the **top** of each step; `ctx = {"model","opt","cfg","batch","autocast_dtype"}`,
  mutable (set `ctx["batch"]=(x,y)`, mutate `opt`, set `ctx["autocast_dtype"]`).
- `grad_hook(step, micro, model)` fires after each micro-batch's backward with that micro's grad alone.
- `on_step(step, info)` fires after each optimizer step; `info = {"loss","grad_norm","lr","model","opt"}`.

**The determinism gate** (`harness/fork.py`): `fork_determinism_gate(cfg, data_dir, snapshot, steps,
device=None, seed=True) -> float`. Runs two noop forks, **asserts** they are bitwise identical at every
step, returns `max|Δ|` (0.0 on success; raises on any nonzero). Already passes at proxy (fp32, CUDA) and
at 124M grad_accum=40 (P100) — see `research/kaggle/step_timer_results.md` — and on the AMD MI300X at proxy,
in-process and across 3 processes through spike windows (`research/kaggle/xproc_gate.py`,
`results/mi300x_2026-10-27/t0_*`).

**Fork + intervention** (`harness/fork.py`): `run_fork(cfg, data_dir, snapshot, intervention=None,
steps=None, start_step=None, device=None, on_step=None, grad_hook=None, pre_step=None, seed=True)
-> list[float]`. Two intervention slots:
- `intervention(model, optimizer)` — fires **once** at fork start (default `_noop`).
- `pre_step(step, ctx)` — fires every step (the spike/skip/reset channel).

**The done-baseline callback pattern** (what C3's repair must mirror):
- `baselines/reset.py`: `as_pre_step(at_step) -> pre_step`, and `zero_moments(opt)` mutates
  `opt.state[p]["exp_avg"] / ["exp_avg_sq"]` in place. **This is the exact pattern for editing
  optimizer moments inside a fork.**
- `baselines/skip.py`: `as_pre_step(inject_step, width) -> pre_step` sets `ctx["batch"]=None`.
- `baselines/clip.py`: `apply_to_cfg(cfg, max_norm) -> cfg` (clip is a cfg setting, not a hook —
  grads don't exist when `pre_step` fires).
- The battery builder `make_branches(cfg, recipe, *, clip_norm)` and `Branch(name, pre_step, cfg)`
  show how interventions are wired into `run_cheap_battery`.

### 7.2 Dependencies + unresolved contracts (settle these before coding to a timeline)

- **Update 2026-10-08:** `research/spikes/capture.py` now keeps `v̄` and saves the fork-time copy plus t0 / fork /
  matched-normal snapshots per spike, and `spikes/k2.py` records them in the K2 manifest (`vbar_uri`,
  `normal_snapshot_uris`). The paragraph below describes the gap it fills. **2026-10-29:** its `--selfcheck`
  crashed on the MI300X (eps read back as a string). The fix (f751e14) is committed, but the GPU re-run is still
  pending.
- **`v̄` (pre-spike running average of `v`) does NOT exist yet.** PLAN_V6 §3.6 defines the poison score
  against a pre-spike running-average `v̄` and thresholds "derived from matched *normal* training steps."
  Nothing captures that: `snapshot.capture` grabs `v` at **one** step, proxy `config.yaml` has
  `snapshot_steps: []`, and the detector tracks a **grad-norm EMA + loss z-score** (for firing `t0`),
  not a running average of `v`. **The localizer role needs snapshots at `t0` AND at matched normal
  steps** — the capture mechanism exists, the orchestration (a driver that captures at multiple steps,
  or a `v̄` accumulator in the trunk loop) does **not**. `experiments/proxy/smoke.py` and
  `experiments/llm124m/run.py` (the drivers that would do this) are both `SystemExit("TODO")` stubs.
  This is a real prerequisite, not a detail.
- **The detected failure step `t0` IS available** from `spikes/tune_detector.py` (its scoring returns
  `{"t0","peak_step","lead",...}`) and `spikes/induce.py` exposes `SpikeRecipe(kind, inject_step, width,
  pre_step)`. So "where the spike is" is a solved input; "a snapshot captured there" is not yet wired.
- **No downstream code hard-codes the localizer/repair output shape** — a repo-wide grep for
  `localize` / `poison_score` / `localization` / `as_intervention` / `repair(` outside those two dirs
  returns nothing. `analysis/attribution.py` only *names* `attribute(snapshot, localization, forks)` in
  a TODO. **Upside:** C2/C3 are free to define their output contract. **Obligation:** they MUST define
  it (what a "localization" is — e.g. `{param_name: score}` + a selected group set), because attribution
  (Task 17) and `repair.as_intervention` will consume it. **Since V8 it is documented:** the `Localization`
  schema in PLAN_V8 §6.4.
- **Unresolved design question for C3's callback slot.** `repair/operator.py`'s TODO says
  `as_intervention(localization, mode) -> callable usable by run_fork`. Which slot depends on which
  snapshot the localizer operates on: if repair edits a **post-spike** snapshot at fork start, it's the
  once-at-start `intervention(model, opt)` slot; if it follows the Task-9 protocol (fork from a
  **pre-spike** `t0` snapshot, replay the spike forward, then fix), it must be an `as_pre_step(at_step)`
  firing after the spike window — exactly `reset.py`'s shape but **selective** on localized groups.
  Settle pre- vs post-spike before writing the signature. **Precedent (2026-10-27):** the RML undo in
  `autopsy/undo.py` already takes the pre-spike route. `mve` forks from the snapshot at `inject_step`, and
  `as_undo_pre_step(log, at_step, channels)` fires after the window, so a selective `repair/` operator that
  follows it is consistent with the P1 code.

### 7.3 "If you're about to work on X, read Y first"

- **localizer (C2)** → read `harness/snapshot.py` (the dict you score), `model/nanogpt.py`
  (`named_parameters()` grouping), `spikes/induce.py` + `spikes/tune_detector.py` (how `t0` and ground
  truth arrive), PLAN_V6 §3.3–3.7 + context-ai §4.3–4.7 (`H~`, HVP, eigensolve, `ψ_k`), and §7.2 above
  (the `v̄` gap). Reference code: curvlinops / pytorch-hessian-eigenthings for HVP + top-k.
- **repair (C3)** → read `baselines/reset.py` (the in-place moment-edit pattern), `harness/fork.py`
  (`run_fork` intervention slots, `make_branches`), `harness/snapshot.py` (what you're editing), and
  §7.2's pre-vs-post-spike question + output-contract obligation. PLAN_V6 §3.7–3.8.
- **theory** → read PLAN_V6 §3.8 + §3.4 (`ψ_k`, AR(2) recovery criterion, the commutator caveat) and
  context-ai §4.10. It must consume the same induced-spike logs the localizer/kill-test produce, so its
  predictions overlay measured spikes — coordinate its inputs with C2's outputs.

Confirm the file is updated (and pushed, once the user signs off) before we move to the next task.
