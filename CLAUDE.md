# CLAUDE.md — Optimizer Autopsy project instructions

This file is read automatically by Claude Code at the start of every session in this repo.
Treat it as binding context for every task, not just a suggestion. If a user prompt conflicts
with a rule below, follow the rule below and flag the conflict instead of silently overriding it.

## 0. What this project is (one paragraph)
LLM pretraining loss spikes poison Adam's optimizer state (`m`, `v`), not just the weights.
This repo builds (C1) a bitwise-deterministic fork-and-replay harness, (C2) a localizer that
finds which directions of optimizer state are poisoned via gradient-directional SNR + curvature
of the preconditioned Hessian, and (C3) a rank-limited repair operator that surgically fixes just
those directions. Full details live in `research/README.md` and the task list in the tracked
issues/notes — don't ask the user to re-explain the project; read the repo instead.

## 1. Non-negotiable guardrails (never violate these without explicit sign-off)
- **Never run training or heavy compute on this machine.** All trunk runs, forks, snapshots,
  eigensolves, and anything GPU-bound go to Kaggle or Colab. This workstation is for git, reading,
  editing, linting, and CPU-only unit tests only — even the "tiny" self-checks are treated as
  GPU-only from here on.
- **Do not modify `research/harness/determinism.py` or `research/harness/preflight.py` without
  explicitly flagging it first.** This code is load-bearing: `CUBLAS_WORKSPACE_CONFIG` must be set
  before `import torch`, CUDA must not be initialized before `use_deterministic_algorithms` is
  configured, and every causal claim in this project depends on noop-vs-noop `Δ==0`. An
  auto-formatter or "helpful" refactor here can silently break bitwise replay. If a linter or
  slop-detector flags something in this file, report it — do not auto-fix it.
- **Snapshots key optimizer state by parameter name, not index; handle tied weights explicitly;
  `m`/`v` are stored in bf16, never fp16** (fp32 only for the exact proxy-scale gate). Don't
  "simplify" this in a refactor.
- **`B*` (the clean counterfactual) must keep the data stream aligned** — never implement it as
  "skip to the next batch." That shifts the data cursor and confounds the causal effect with a
  data-order change.
- Before marking any task/file's status as done (in whatever tracking format the user is using),
  it needs a passing test or a completed DoD check as described in the project notes — not just
  code that runs once without erroring.

## 2. Repo hygiene workflow (run this, don't just talk about it)
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
4. If a package or repo is genuinely new, add it with a pinned version and a one-line comment
   saying what it's for and which task/file it supports.

## 4. Known-good reference repos already identified for this project
Use these instead of reimplementing from the papers cold, when the task calls for it:
- Curvature/eigensolve: `f-dangel/curvlinops`, `noahgolmant/pytorch-hessian-eigenthings`,
  `amirgholami/PyHessian`
- Baselines: `TianjinYellow/SPAM-Optimizer`, `bluorion-com/ZClip`,
  `PaddlePaddle/PaddleFleet` (AdaGC under `Research/AdaGC`)
- Theory (EoS / preconditioned stability): `locuslab/edge-of-stability`, `alex-damian/EOS`
- Scale-ladder infra: `karpathy/nanoGPT`, `karpathy/build-nanogpt`
- Natural spike checkpoints: `LLM360/k2-train`, `LLM360/k2v2_train`
- Stats: `statsmodels` (`multipletests`), `scipy.stats.bootstrap`

## 5. Review-gate cadence — stop only where it actually matters
Only pause for explicit human/reviewer sign-off before:
1. Modifying `research/harness/determinism.py` or `research/harness/preflight.py`
2. Pushing anything to Kaggle that spends GPU-hours (a training/verification run — not a plan)
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

Confirm the file is updated and pushed before we move to the next task.
