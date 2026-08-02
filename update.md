# UPDATE — OPTIMIZER AUTOPSY project status

Running status report. I update this after every unit of work. Companion: `todo.md` (things I need
*you* to do) and `BUILD_PLAN.md` (the 26-task plan this tracks against).

**Last updated:** 2026-08-01

---

## At a glance

| Metric | Value |
|---|---|
| **Overall progress** | **~15%** (4 of 26 build tasks landed) |
| **Phases complete** | **1 of 7** (Phase 0 complete) |
| **Current status** | 🟢 **PASS** — everything built so far passes its DoD (verified on Kaggle T4) |
| **Blocked on you** | Nothing — `todo.md` is clear. |
| **Next up** | Task 4 (deterministic replay + smoke test — the load-bearing gate) |

**Pass/fail of what exists:**
- Scaffold imports cleanly — ✅ PASS
- `get_batch` bitwise-identical across two processes — ✅ PASS (verified with 2 separate processes)
- `check_env()` — ✅ PASS on Kaggle Tesla T4 (`check_env OK`); pins now match the real image.
- secrets: load-from-env, missing-required-raises, and no-token-in-git — ✅ PASS (3/3).

---

## What I've been implementing (newest first)

- **Task 3 — secrets hygiene.** `research/harness/secrets.py::get_secret()` resolves in order
  env → Kaggle Secrets → Colab userdata → `.env`; never prints values; raises with the fix when a
  required secret is missing. `hf_token()`/`wandb_key()` wrappers. `research/tests/test_secrets.py`
  greps the git-tracked tree for `hf_...` / `KEY="..."` literals (detector verified non-vacuous),
  plus env-load and missing-raises tests.
- **Task 1 — env pin + preflight.** `research/harness/preflight.py::check_env()`: parses
  `requirements.txt` (single source of truth) and asserts installed versions of torch/numpy/scipy/
  datasets/safetensors/huggingface_hub/tiktoken/wandb; asserts `CUBLAS_WORKSPACE_CONFIG` ∈
  `(":4096:8", ":16:8")`; asserts CUDA not yet initialized; prints the GPU name. Wired into
  `smoke.py` and `run.py` entrypoints.
- **Task 2 — fixed data shard.** `research/data/prepare.py`: GPT-2 BPE (tiktoken) → flat `uint16`
  memmap; deterministic tail val split; `get_batch(split, step, batch, block)` indexes by integer
  offset (pure function of `step`, no streaming/shuffle, O(1) resume); HF upload/pull by fixed
  revision; Kaggle `/kaggle/input` → `/kaggle/working` → `./data` path resolution.
- **Task 0 — scaffold.** Full `research/` package tree (harness, data, model, localizer, repair,
  baselines, spikes, analysis, experiments, theory, configs), pinned `requirements.txt`, READMEs,
  both notebook launchers (set `CUBLAS_WORKSPACE_CONFIG` before torch), `.gitignore`. Root
  `index.html` left untouched.

---

## BUILD_PLAN task tracker

Legend: ✅ done · 🟡 partial/stubbed · ⬜ not started · ⛔ credits-gated

### Phase 0 · Foundation — 3/3 ✅
- ✅ **Task 0** — scaffold repo (DoD: imports clean — PASS)
- ✅ **Task 1** — env pin + `check_env()` (DoD: clear-message-on-mismatch PASS; passes-on-Kaggle **PASS** — `GPU: Tesla T4 / check_env OK`)
- ✅ **Task 2** — fixed tokenized shard + `get_batch` (DoD: cross-process bitwise-identical PASS; <5 min prep by design, not yet timed on Kaggle)
- ✅ **Task 3** — secrets hygiene + no-token-in-git test (DoD: env-load PASS; no-secrets-in-git PASS)

### Phase 1 · The instrument — 0/4
- ⬜ **Task 4** — deterministic replay + smoke test *(LOAD-BEARING — the determinism gate)*
- ⬜ **Task 5** — proxy nanoGPT model + trunk loop
- ⬜ **Task 6** — snapshot/restore of (w, m, v, RNG, cursor) + HF Hub
- ⬜ **Task 7** — fork driver + the Δ==0 determinism GATE

### Phase 2 · Cheapest kill-test — 0/2
- ⬜ **Task 8** — spike induction + detector tuning
- ⬜ **Task 9** — cheap-branch battery + method-is-dead check

### Phase 3 · Localizer — 0/3
- ⬜ **Task 10** — directional SNR
- ⬜ **Task 11** — curvature: HVP + memory-bounded top-k
- ⬜ **Task 12** — poison score, ψ_k, poisoned set P

### Phase 4 · Repair, battery, baselines — 0/3
- ⬜ **Task 13** — repair operator (rank-|P| projection)
- ⬜ **Task 14** — full branch battery
- ⬜ **Task 15** — baselines (skip/clip/spam/zclip/adagc/reset)

### Phase 5 · The science — 0/4
- ⬜ **Task 16** — held-out val-loss eval harness
- ⬜ **Task 17** — attribution battery + calibration + paired stats
- ⬜ **Task 18** — GO/NO-GO #1
- ⬜ **Task 19** — theory notebook (AR(2), Thms 1 & 2)

### Phase 6 · Scale, figures, paper — 0/6
- 🟡 **Task 20** — end-to-end proxy smoke gate *(entrypoint + `check_env()` stubbed; pipeline TODO)*
- ⬜ **Task 21** — natural spikes (LLM360 K2 + corrupted-batch)
- ⬜ **Task 22** — 124M on Kaggle within free limits
- ⬜ **Task 23** — analysis & figures
- ⬜ **Task 24** — workshop paper + repro package
- ⛔ **Task 25** — 410M mechanism transfer *(credits-gated)*

**Totals:** ✅ 4 · 🟡 1 · ⬜ 20 · ⛔ 1  →  **4 fully done / 26.**

---

## What's left (the short version)

1. **Build the instrument (Phase 1):** determinism gate → model+trunk → snapshot → fork Δ==0 gate.
   This is the tall pole; nothing causal is allowed until Task 7 reads Δ==0 on the proxy.
3. **Week-one kill-test (Phase 2):** find out if the repair method is even needed before building it.
4. Then localizer → repair → attribution/theory → 124M/figures/paper.

## Known caveats / risks
- **Pins verified on Kaggle** ✅ — matched to the real image (torch 2.10.0, numpy 2.0.2, scipy
  1.16.3, datasets 5.0.0, safetensors 0.7.0, huggingface_hub 1.11.0, tiktoken 0.12.0, wandb 0.26.1).
  Note this is a *newer* image than the plan assumed — watch for numpy-2 / datasets-5 API drift.
- **Dataset revisions unpinned** — `prepare.py` presets use `"main"` (TODO: real commit shas) so a
  shard is only reproducible once those are pinned.
- **`<5 min` proxy prep** is a design target (streaming + token cap), not yet measured on real hardware.
