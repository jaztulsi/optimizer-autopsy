# Optimizer Autopsy — Improvement Plan

*Draft for team review · 2026-10-19 · nothing in here has been implemented yet; every fix is a proposal.*

> **Update 2026-10-22.** The governing plan is now [`PLAN_V8.md`](../PLAN_V8.md).
> - **Folded into V8:** A1–A3 → V8 §9.3; the thresholds → V8 §7.2; B-items → V8 §6.7 / §9.4; the five investment
>   areas → the V8 tiers (§8).
> - **C1 (doc sprawl) is done:** old plans are in `docs/reference/`, status files in `docs/archive/`, and the map is
>   at `docs/README.md`.
> - **Still open:** A5, pushing the spike pipeline through a PR.
>
> Rows below keep their original wording; where a row says "PLAN_V7 §4.7", read "PLAN_V8 §7.2".

---

## 0 · Bottom line

1. **The kill-test can't be trusted yet, and it decides half the project.** As written, its "cheap fix"
   branch is bit-for-bit the same as one of the other branches, it compares training loss instead of held-out
   loss, and its pass/fail threshold was never written down. Fix these before the battery runs (§1, A1–A4).
2. **Most of the current work exists only on one laptop.** `PLAN_V7.md`, the spike pipeline and its tests
   are uncommitted. Push them through a PR so CI runs and the team can see the plan (A5).
3. **Two things will break on AMD day 1 unless prepared:** the determinism setup only knows CUDA settings,
   and every Hessian library will fail on our attention kernel (A6, A7).
4. **Novelty still looks intact for C2/C3, but not for C1 on its own.** A 16-item deep pass found no
   published method that localizes poisoned Adam moments and repairs only those. But bitwise-deterministic
   replay is now shown at production scale (NVIDIA Megatron Core, Oct 2026; OPEN-1B, Sep 2026). So C1
   must be claimed as a *causal instrument* (counterfactual forks that edit optimizer state), not as
   determinism. Three 2026 papers sit close to C2 and must be cited and contrasted (§2.3).
5. **Five investments** (§2): a trustworthy kill-test · a measured AMD bring-up · C2 built against a
   planted-poison oracle · a stronger comparison set · a public spike benchmark.

---

## 1 · Things to fix

Legend — **✔** = re-checked by reading the code/repo while writing this doc · **◐** = audit finding, not
independently re-checked · owner names are suggestions.

### A · Before the first AMD run (blockers)

| # | Problem | Evidence | Proposed fix | Owner · effort |
|---|---|---|---|---|
| A1 ✔ | **The kill-test's cheap fix `Bs` does nothing.** `Bs` = skip + clip, and every caller passes `clip_norm=1.0`, which is already the trunk's `grad_clip`, so the clip half changes nothing. Skip is a no-op for `lr_bump`, so `Bs` ≡ `B0` there. For `corrupt_batch`, skip restores the clean batch, so `Bs` ≡ `B*`. The verdict is decided before any data exists. | `research/harness/fork.py:196`, `research/experiments/proxy/config.yaml:18`, `research/tests/test_kill_test.py:128`, `research/baselines/skip.py` docstring | Pre-register a `Bs` clip below the trunk clip (e.g. Marin 32B tightened 1.0 → 0.2), or define Gate B on the strongest cheap fix (skip-step, v-only reset). Add an assertion that `Bs.cfg != B0.cfg`. | harness owner + jaztulsi · 2 h |
| A2 ✔ | **No held-out loss anywhere in the battery.** `_summarize` returns the last *training* loss, but core value 2 and `gate_b_verdict` need Δ = held-out val_loss(branch) − val_loss(B\*). `analysis/eval.py` and `stats.py` contain only TODO comments. No driver goes manifest → branches → Δ per seed → verdict. | `fork.py:306-310`, `research/analysis/attribution.py:20,44`, `research/analysis/eval.py` | Add `eval.val_loss`, reusing the probe pattern at `research/kaggle/corrupt_batch_probe.py:83`, evaluated at a pre-registered fork length. Add one Gate-B driver script. | harness owner · 6–10 h |
| A3 ✔ | **Decision thresholds are not pre-registered.** None of these is written down anywhere: the kill-test recovery margin τ (the only value is `tau=0.05` in a test), the ψ_k repair-sufficiency criterion, the stop-and-write-up trigger, the fork length / eval horizon, the `Bs` clip value, minimum n, and power / minimum detectable effect (MDE). | `attribution.py:39-40`, PLAN_V6 §8 (asks for them, gives no numbers), grep of every plan doc | Add a **PLAN_V7 §4.7 "Decision thresholds"** table with numbers, signed off and committed before any AMD data. Starting proposal: τ = 0.01 nats (the §4.4 claimed-effect floor), n_min = 10 per type, eval horizon set from `step_timer` cost. | jaztulsi + team · 3 h + sign-off |
| A4 ✔ | **Natural spikes have no clean counterfactual (B\*).** K2 defines B\* as "the same run with the hook omitted", but natural runs have no hook. 10 of the 30 instances per intervention (V7 §4.2) cannot produce a Δ. | `research/spikes/k2.py:6`, `fork.make_branches` needs `recipe.pre_step` | Decide and pre-register now: either define a natural B\* (e.g. data-driven spikes: aligned clean replacement of the offending batches, found by the cross-seed alignment test), or drop natural spikes from Gate B and say so. | jaztulsi · 1 h decision |
| A5 ✔ | **The source of truth isn't in git.** About 1,670 lines are untracked: `PLAN_V7.md`, `RESEARCH.md`, `ATTRIBUTION.md`, `capture.py`, `natural.py`, `corrupt_batch_probe.py`, `test_spike_pipeline.py`. About 490 more lines of edits are uncommitted. Teammates can't see V7, CI has never run the spike tests, and one lost laptop loses it all. | `git status` | Branch + PR now (diff shown first, no AI attribution). | jaztulsi · 30 min |
| A6 ✔/◐ | **ROCm determinism isn't prepared, and the deterministic flag reaches less on AMD.** On ROCm, `use_deterministic_algorithms` only turns off rocBLAS atomics ◐. It doesn't reach hipBLASLt, MI300X's default matrix-multiply library ◐. In our pinned torch 2.10, the attention backward accepts `deterministic` and ignores it ✔ (read in the 2.10 source). On ROCm, `CUBLAS_WORKSPACE_CONFIG` only sets a workspace size ◐, so our preflight check passes without guaranteeing anything. The §4.4 tolerance path doesn't exist either: the gate hard-asserts exactly zero, with no 5-pair noop-drift measurement. | `fork.py:131-135`; torch 2.10 `mha_all_aot.hip`; checklist below | (1) **Flag-and-approve** a ROCm block in `determinism.py` (CLAUDE.md §1), per the checklist below. (2) Run forks under `sdpa_kernel(SDPBackend.MATH)`, or move the AMD pin to torch ≥ 2.13 (AOTriton deterministic backward) ◐. (3) Add `noop_drift(pairs=5)` + the §4.4 comparison so a non-zero ROCm result falls back cleanly instead of stalling the queue. | harness owner · 4–8 h, **needs sign-off** |
| A7 ✔/◐ | **Hessian-vector products (HVPs) will fail on our attention kernel.** The model uses `F.scaled_dot_product_attention` ✔, whose flash and memory-efficient kernels have no double-backward ◐. Every HVP library (curvlinops, eigenthings, PyHessian) will error, and the V6 go/no-go wanted an HVP determinism check that nobody has written. | `research/model/nanogpt.py:63`; PyTorch forum reports | Force `sdpa_kernel(SDPBackend.MATH)` inside HVP calls. Add an HVP bitwise check to `capture --selfcheck` (AMD queue item 1b). | C2 owner · 2–4 h |
| A8 ✔ | **Open Dependabot PRs would break preflight, and the pin itself is stale.** #9–#13 (open since 08-19) bump the pins that `check_env()` asserts (torch 2.10 → 2.13). Merging any of them breaks every GPU run. `research/requirements.txt` says it is "pinned to Kaggle's default GPU image", but Kaggle is on hold. | `gh pr list`; `research/requirements.txt:1` | Close the Dependabot PRs; set Dependabot to ignore `/research` pip. Then choose the **AMD** pin deliberately (AMD ROCm wheel or `rocm/pytorch-training` image) before the first gate, so all AMD results come from one stack. | jaztulsi · 15 min + pin decision |

**ROCm determinism checklist for A6.** It comes from a source-level research pass ◐ except where marked ✔.
Everything must be re-verified on our MI300X, and none of it is in the harness yet.
- **Matrix multiplies:**
  - set `ROCBLAS_DEFAULT_ATOMICS_MODE=0` and `ROCBLAS_USE_HIPBLASLT=0`;
  - to force rocBLAS over hipBLASLt, call `torch.backends.cuda.preferred_blas_library("cublas")` (this selects rocBLAS on ROCm);
  - pin `HIPBLASLT_WORKSPACE_SIZE` (and the `CUBLAS_WORKSPACE_CONFIG` size) identically in the trunk and every fork process.
- **Why process separation matters:** hipBLASLt can pick a different fallback kernel per process for untuned shapes, and that has bitten other projects. It has only been reported on gfx1201 (a Radeon card) and is untested on MI300, but `capture` runs one process per run, so treat it as a risk until our cross-process gate passes.
- **Attention:**
  - forks run under `sdpa_kernel(SDPBackend.MATH)`, or the pin moves to torch ≥ 2.13;
  - never set `TORCH_ROCM_FA_PREFER_CK=1`. In torch 2.10 the CK *memory-efficient* backward passes
    `deterministic=false`; the CK flash path passes the flag through, but the safe rule is still to avoid
    CK in forks.
- **Autotuning off:** TunableOp, hipBLASLt tuning, and `TORCH_COMPILE_DISABLE=1`.
- **Fixed numerics:**
  - TF32/XF32 off, so fp32 matrix multiplies stay IEEE;
  - `TENSILE_SOLUTION_SELECTION_METHOD=0` (no Stream-K).
- **Record in run metadata:** the MI300X partition mode (`amd-smi`; SPX = 304 CUs). The GPU's random-number streams depend on the CU count.
- **Embedding-backward race:** there is a known race (pytorch #196164, fix not yet released). It only fires for 2033–3072 indices. Our shapes are safe (proxy 16,384; 124M 12,288), but add a preflight assert so a config change can't land inside that window.
- **Snapshots:** snapshots taken on CUDA will not replay bit-for-bit on ROCm. Regenerate everything on AMD.

### B · Before the battery

| # | Problem | Evidence | Proposed fix |
|---|---|---|---|
| B1 ✔/◐ | **Natural-spike selection ignores the pre-registered rule.** The rule is "first qualifying spike per run, preset then seed order, stop at 10". `build_k2` admits every qualifying trigger ✔. `max_triggers=3` with a 50-step cooldown can be used up by early false alarms ◐. Normal snapshots stop at the first trigger ✔, so matched normals can be empty. | `k2.py:130`, `capture.py:181,189` | Add `select_natural()` that applies the §4.2 rule. Keep saving normal snapshots after the first trigger. |
| B2 ◐ | **`gate_b_verdict` gaps.** `recipes_total=4`, but V7 has 3 types. There is no minimum n: with n=1 the CI is a point, so one seed can give a final verdict. A NaN `Bs` reads INCONCLUSIVE instead of PROCEED. | `attribution.py:39-82` | Module constants `TAU`, `RECIPES`, `N_MIN`. Treat survival = 0 as a gap. |
| B3 ✔ | **Real bugs get recorded as science.** `run_cheap_battery` catches *every* exception as `survival=0`, so a `KeyError` becomes "branch diverged". | `fork.py:351` | Catch only non-finite / kernel errors; re-raise everything else. |
| B4 ✔ | **Results can't be traced to code.** `results/*.json` hold no git SHA, config, argv or torch/HIP version. The 124M P100 Δ=0 gate exists only as prose, and ran on torch 2.7.1, not the pinned 2.10.0. `capture`/`natural` default `--out` to the tracked `results/` folder. | `results/task7_fork_gate.json` keys; `research/kaggle/step_timer_results.md` | A stamp helper writes SHA, cfg, argv and versions into every JSON. Add a `results/README.md` mapping JSON → command. Point `--out` outside the repo or gitignore run dirs and `*.npz`. |
| B5 ◐ | **Some tests would pass even if the code were broken.** `test_determinism` uses a toy MLP and never touches `snapshot`/`run_fork`. CI never runs the fork gate or snapshot save/load (it doesn't install safetensors). One baselines test uses a no-op spike and asserts survival only. The `prime_trunk` test stops at step 4, still inside both observers' warmup. | `research/tests/*`, `.github/workflows/ci.yml` | Add 3 small CPU tests: snapshot round-trip through safetensors; `Bs` ≠ `B0`; primed observers past warmup. |
| B6 ✔ resolved | **SPAM fidelity.** The paper's Algorithm 1 bug is real: m/v are zeroed before the spike test, so training would freeze. The official code avoids it with an undocumented 20-step grace window (author-confirmed, issue #2). Eddie's `spam.py` matches the official schedule exactly: reset timing, grace window, cosine warmup, step counter kept. | deep pass, read against `TianjinYellow/SPAM-Optimizer` | Nothing to fix in code. In the paper, say we follow the **code**, not Algorithm 1. Also note the official launch script passes the 1000-step LR warmup as SPAM's warmup (issue #6). |

### C · Hygiene, docs, team

| # | Problem | Proposed fix |
|---|---|---|
| C1 ✔ | **Too many top-level docs (9+), several stale.** README still calls V6 the governing plan. Its quick start runs a `SystemExit("TODO")` stub (`README.md:162`). It says baselines are "not started" (`README.md:204`). | Keep: PLAN_V7, CLAUDE.md, README (refreshed), RESEARCH.md, ATTRIBUTION.md. Banner "superseded by V7": PLAN_V6, BUILD_PLAN. Merge into V7 §3: update.md, todo.md (→ GitHub issues), context-ai §14. Archive to `docs/archive/`: STATUS_REPORT, EXPLANATION, rest of context-ai, index.html. |
| C2 ◐ | **Compute policy drift.** `CLAUDE.md:24-26` and `:105` still say Kaggle/Colab. Several docs say 124M stays on Kaggle. | Point all of them to V7 §4.1 (AMD). |
| C3 ✔ | **`main` is unprotected**, most history is direct pushes, and merged branches linger. | Protect `main` (PR + CI required); delete merged branches. |
| C4 ◐ | **CI gaps.** CI's ruff isn't pinned to 0.16.2, there's no `ruff format --check`, sloppylint isn't run, and CLAUDE.md expects a `lint.yml` that doesn't exist. `SLOPPYLINT_REPORT.json` dates from 2026-08-11. | Pin ruff, add a format check, decide on sloppylint, regenerate the report. |
| C5 ✔/◐ | **`.gitignore` gaps.** It lacks `research/third_party/`, `*.npz` and `.DS_Store` ◐. A tool config `.claude/proven-config.json` is tracked in the public repo ✔. | Add the ignores; untrack the tool config. |
| C6 ✔ | **Credit and team.** `CITATION.cff` lists only jaztulsi. The `CONTRIBUTORS.md` update is uncommitted. aryakumarn's invite expired; Avyank's was re-sent 2026-10-19. Roles for Orange-neon and tanishp3280 are unknown, and there is one spike engineer (bus factor 1). | Match CITATION to CONTRIBUTORS. Assign owners for harness, C2, C3 and analysis. Resend aryakumarn's invite if still wanted. |
| C7 ◐ | **Stubs disagree with the plan.** `snr.py` defines SNR as \|m\|/(√v+ε), not gradient-directional SNR. `curvature.py` targets plain H, not the preconditioned H̃. Several `__init__`s label the components C1/C2, but V7 calls them C2/C3. | Fix when C2 starts (the doc comments, not code). |
| C8 ◐ | **124M config.** `max_steps: null` makes `run_fork(steps=None)` crash, and `snapshot.dtype: bf16` is never read. | Fix with Task 22. |
| C9 ✔ (corrected by the Verifier) | **Citation locator.** arXiv v2 of Wortsman 2023 contains *both* counts. §3.4 has 28 of 30 (all runs); App. D / Fig. 16 has 14 of 15 (ViT-Huge, RMS ≥ 2.3 in `visual.conv1.weight`). `RESEARCH.md:42` and `capture.py:61` are correct but give no locator. | Add "App. D, ViT-Huge" to the citation. |
| C10 ✔/◐ | **Stale reference notes.** CLAUDE.md §4 says AdaGC is PaddlePaddle-only ("reimplement"). An Apache-2.0 PyTorch reference by a co-author now exists (`lishuai-97/AdaGC`) ◐, and the paper's PaddleFleet link is dead ◐. Our `adagc.py` matches the PyTorch defaults ◐. | Update CLAUDE.md §4. Say in the paper which reference we follow. |

---

## 2 · Five areas to invest in

### 2.1 Make the kill-test trustworthy — *highest leverage*
- **Why:** Gate B decides whether C2/C3 get built at all (V6 §7, Floor vs Core). Right now it would return
  PROCEED on `lr_bump` for structural reasons (A1), so we could spend ~130 h building a repair the data
  never asked for.
- **What:** A1–A4 + B2 + B3, plus the §4.7 thresholds table.
- **Cost:** ~15–20 person-hours, 0 GPU-h until it runs.
- **First step:** write the §4.7 table and get sign-off this week.

### 2.2 AMD bring-up as a measured, pre-registered gate
- **Why:** every causal claim rests on noop-vs-noop Δ = 0, and that has never been tested on ROCm. The
  15 GPU-h ask assumes MI300X speed that hasn't been measured.
- **What:** A6 + A7 + the A6 checklist. Pin the environment with AMD's ROCm wheels or a
  `rocm/pytorch-training` image, so `check_env()` stays exact. Then run `step_timer` first (V7 §4.1). Run the
  noop-vs-noop gate **across two processes**, not just within one: that's where the hipBLASLt kernel-choice
  risk shows up. To diagnose a mismatch, log the chosen kernels (`HIPBLASLT_LOG_LEVEL`, `AOTRITON_DEBUG_LEVEL`)
  in each process and diff the logs. If the GPU is an AMD Developer Cloud VM,
  sync snapshots off the VM as they're written: per AMD's terms, the VM is destroyed when credits run out ◐.
- **Cost:** 4–8 person-hours + the V7 3–5 GPU-h bring-up line.
- **First step:** SSH access to the MI300X (setup already drafted), then `step_timer` and `capture --selfcheck`.

### 2.3 Build C2 against a planted-poison oracle, positioned against 2026 prior art
- **Why:** C2/C3 are the novel contribution. A planted `v_inflate` poison gives exact ground truth, so the
  localizer can be scored before natural spikes exist (RESEARCH.md §7.1–7.2).
- **Prior art to cite and contrast** (none localizes and repairs; all read in full text):
  - **Bai et al. ICML 2026** — the spike mechanism and a curvature predictor `gᵀĤg/‖g‖² > 2/η`. Their Ĥ is
    left-preconditioned with a `(1−β1)/(1+β1)` factor, not our symmetric `D^-1/2 H D^-1/2`, so a scale
    factor is needed to compare numbers. Their Transformer predictor takes a min over steps t−1…t+1, which
    uses a future step, so it isn't an online detector as written. Largest run 187M. The code is in the
    conference supplementary material, and there is no public repository. Localization is only block-level
    ◐, and there is no repair.
  - **Liu & Li 2026** (arXiv 2608.19762) — paired control/shock runs with shared later data, like our C1. Its
    code injects the shock into whole channels {θ, m, v}, never directions within them, and works in a
    small-perturbation regime, not spikes. The code has no license (reimplement, don't vendor).
  - **Ramasinghe et al. 2026** (arXiv 2606.16768) — tracks the same symmetric operator we plan
    (`P^-1/2 H P^-1/2`) with a warm-started power iteration (about 5 HVPs per step), up to 3B. A good reference
    design for `localizer/curvature.py`. It tracks one global eigenvalue and never edits m/v.
  - **Spectra 2026** (arXiv 2602.11185, ICML 2026) — a spike-subspace *suppression* optimizer. It shrinks
    only the top ~1.5% singular values of the momentum toward the tail's scale and leaves the tail alone.
    The paper explicitly contrasts this with Muon's full-spectrum flattening. It reports about 97% of v's spectral energy in that subspace: prior evidence that
    poisoned v may be low-rank. Use it as a cheap **"energy-only" localizer control** (damp the top singular
    directions, no SNR or curvature) next to `Br`.
- **Counter-evidence a reviewer will raise:** Marin 32B (2025) traced its recurring spikes to the attention
  stack. Clip tightening, update-norm clipping, skip-step and a Muon swap only softened them; QK-Norm
  removed them. Our nanoGPT proxy has **no QK-Norm**, so proxy spikes may be architecture-driven. Plan a
  QK-Norm ablation of the proxy, and say plainly that we measure damage carried in optimizer state, not that
  optimizer state is the root cause.
- **Repair variants and controls worth pre-registering:**
  - **"v ← g²_post" (GI-Adam idea, Kalra & Barkeshli, NeurIPS 2024, MIT).** It sits between v-only reset
    (v = 0) and `Bvbar` (v = v̄), so it tests whether v̄'s history matters. It needs the `pre_opt_step` hook,
    because gradients don't exist at `pre_step`.
  - **A step-counter-only reset (Adam-Rel, Ellis et al., NeurIPS 2024)** as a control. It shows a repair
    effect isn't a bias-correction artifact. Adam-Rel resets *only* t; the m + v + t reset is Asadi's Adam-MR.
- **Precedent, weaker than it looks:** Asadi et al. (NeurIPS 2023) found a second-moment-only reset "quite
  effective" in deep RL. But their method also resets Adam's step counter, and they used β2 = 0.999 (about
  1,000 steps of memory) against our 0.95 (about 20). So it doesn't transfer directly.
- **Cost:** V6 budget, 70 h C2 + 60 h C3.
- **First step:** settle the localization output contract (CLAUDE.md §7.2) and the HVP math-backend check (A7).

### 2.4 Strengthen the comparison set
- **Why:** a missing baseline is a common rejection reason. Adding one is cheap before the battery and
  expensive after it.
- **Add (each only if pre-registered before any battery data):**
  - **GradientStabilizer** (formerly Stable-SPAM; ICML 2026; MIT). Keeps the gradient direction, replaces
    its norm with a ratio of running norm averages. Reimplement in about 15 lines with state keyed by name
    (the official state is keyed by position). It acts every step, so it needs the same shadow warm-up as
    ZClip/AdaGC.
  - **StableAdamW / per-tensor update clipping** (Wortsman 2023). Reimplement: the official gist has no
    license; `optimi` is MIT. It keeps no extra state, so forking is exact. It must be a cfg-level switch
    like `clip.py`. Add it or Marin's update-norm clipping, not both.
  - **OLMo-core's skip rule** (6σ over a 128-step window on loss and pre-clip grad norm; Apache-2.0) and
    **OPEN-1B's rule** (skip 5 steps when grad norm exceeds a threshold). Add these as **trigger variants of
    `Bskipstep_det`**, not new branches. Their state semantics already match ours.
  - **Muon**, built into PyTorch with no `v`, as the control optimizer for V7's "2 extra optimizers".
  - **Adam-mini** as an ablation of how fine-grained v-poisoning is.
- **Reference implementations confirmed:** ZClip and AdaGC defaults in our ports match the official code;
  `spam.py` matches SPAM's code (B6).
- **Cost:** ~10–20 person-hours + small proxy GPU time inside the existing ask.

### 2.5 Release a public spike benchmark
- **Why:** it raises the paper's floor whatever the kill-test says, and fits the NeurIPS Evaluations &
  Datasets track. We found no *curated* public dataset of spike snapshots that includes Adam moments ◐. The closest is
  OPEN-1B's bucket, which appears to hold fp32 optimizer state every 100 steps around two grad-norm events.
  But those steps were skipped and AdaGC-clipped, so they are not left-to-spike instances.
- **What:** induced + natural spike snapshots (w, m, v, v̄), the K2 manifest and the fork harness, hosted on
  the Hugging Face Hub with Croissant metadata (required by the track). Mind the Hub's file-count limits.
- **External natural spikes with optimizer state ◐:** OPEN-1B (arXiv 2609.17380, Apache-2.0 weights) reports
  checkpoints every 100 steps with "full optimizer state", bracketing two natural grad-norm events (ckpt 1700
  → step 1782; ckpt 43700 → step 43737). If the bucket really holds m/v, that is a real-world natural-spike
  source at 1.6B with no fork needed for C2 scoring. Verify the bucket contents before planning on it.
- **Cost:** ~10–15 person-hours packaging; storage is free.
- **First step:** after the battery, freeze the K2 manifest schema (`k2.py`) as the dataset card's schema.

---

## 3 · Tools we have

| Area | What we have | Use it for |
|---|---|---|
| **Instrument (C1)** | `harness/`: determinism, name-keyed fp32 snapshots, `run_fork`, Δ==0 gate (CUDA-verified on T4/P100) | every causal claim |
| **Baselines** | `baselines/`: skip, clip, m+v reset, v-only reset, v ← v̄, SPAM, ZClip, AdaGC, skip-step (oracle + detector-timed); ZClip/AdaGC warm-up wired into capture | branch menu V7 §4.3 |
| **Spike pipeline** | `spikes/`: 6 recipes (incl. `repeat_ngram`, `v_inflate`), `OnlineDetector`, capture driver (ring buffer, v̄, normals, telemetry), natural stress search, K2 manifest | the 30-instance battery |
| **Decision code** | `analysis/attribution.py`: `gate_b_verdict` + one-page report; `kaggle/corrupt_batch_probe.py`, `spike_run.py` | Gate B, recipe confirmation |
| **Quality gates** | ruff 0.16.2 (pre-commit + CI), gitleaks, CPU pytest in CI, sloppylint 0.5.1 (installed, not in CI) | every PR |
| **Pinned dev tools** | `requirements-dev.txt`: statsmodels, curvlinops 3.0.1, hessian-eigenthings 1.0.0a5, pyhessian 0.1 | stats, C2 curvature |
| **Built into PyTorch** | `torch.func` (HVP), `sdpa_kernel` (math backend), `torch.optim.Muon`, `torch.optim.Adafactor` | C2, control optimizers — no new deps |
| **Compute** | MI300X via Avneh (access being set up); V7 ask 15 GPU-h now, ~100 total. Kaggle on hold. | everything GPU-bound |
| **Reference code** | CLAUDE.md §4 repos (curvlinops, eigenthings, PyHessian, SPAM, ZClip, AdaGC, EoS, nanoGPT, LLM360 K2) | reimplementation checks |
| **Claude toolbox** | Deep Research skills (`/research` → `/research-deep` → `/research-report`) and `web-search-agent` (used for this doc); Explore/Plan agents (repo audits); graphify (knowledge graph of repo + papers); `gh` (PRs, invites, branch protection); browser automation; Google Drive / Claude Docs connectors; Artifacts (shareable pages, e.g. a battery-results dashboard) | research, audits, team sharing |

---

## 4 · Timeline

| When | What | Owner |
|---|---|---|
| **This week** (before any AMD run) | A5 push the spike work via PR · A8 close the Dependabot PRs · A3 thresholds table + sign-off · A1/A2/A4 decisions · A7 HVP math-backend check · A6 ROCm block (flagged) · MI300X SSH access | jaztulsi, harness owner |
| **Next 2 weeks** | AMD day-1 queue (`step_timer` → `capture --selfcheck` → `spike_run` → `corrupt_batch_probe` → `natural --search` → `capture --battery`) · B1–B5 · C1–C6 cleanup · roles assigned | jaztulsi, Eddie, team |
| **Weeks 3–6** | Gate B run → verdict. If PROCEED: C2 on `v_inflate` + optimizer-state patching; add the §2.4 baselines. If PIVOT: write the "cheap fix wins" paper (V6 Floor). | team |
| **Later** | C3, full battery, benchmark release (§2.5), 124M/350M (~85 more GPU-h), paper → TMLR / NeurIPS 2027 | team |

## 5 · Open questions for the team
1. Natural spikes: define a B\* or drop them from Gate B (A4)?
2. τ, `Bs` clip value, eval horizon, n_min: accept the A3 starting proposal?
3. Who owns the harness, C2, C3 and analysis? What are Orange-neon's and tanishp3280's roles?
4. Is the MI300X an AMD Developer Cloud VM (credit-limited, destroyed on expiry) or a dedicated machine?

## 6 · How this doc was made
- Two read-only code/doc audits and my own re-check of the top items (marked ✔).
- Web research with the Deep Research skills: a supplement pass on 3 topics (ROCm determinism, related
  work, tools) and a 16-item deep pass on related work (16/16 validated, 100% field coverage; most read in
  full text plus official code). Findings marked ◐ come from one source, an abstract
  or an agent's source reading, and still need checking. The single most consequential ROCm claim (the
  torch 2.10 attention backward ignores `deterministic`) was re-read in the source ✔.
- Raw research output lives in `./spike-related-work/` (16 JSON files). The tools and ROCm topics stopped
  at the supplement pass to save usage. All of it is local and uncommitted.

## 7 · Sources
- Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam*, ICML 2026 — https://proceedings.mlr.press/v306/bai26o.html · https://arxiv.org/abs/2506.04805
- Liu & Li, *Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW* — https://arxiv.org/abs/2608.19762
- Ramasinghe et al., *Taming Curvature* — https://arxiv.org/abs/2606.16768
- Spectra — https://arxiv.org/abs/2602.11185
- GradientStabilizer / Stable-SPAM — https://arxiv.org/abs/2502.17055 · https://github.com/TianjinYellow/GradientStabilizer
- SPAM — https://github.com/TianjinYellow/SPAM-Optimizer · review: https://pith.science/paper/2501.06842
- AdaGC — https://arxiv.org/abs/2502.11034 · PyTorch reference https://github.com/lishuai-97/AdaGC · ZClip — https://github.com/bluorion-com/ZClip
- SPAM issues #2 (grace window) and #6 (warmup) — https://github.com/TianjinYellow/SPAM-Optimizer/issues/2 · https://github.com/TianjinYellow/SPAM-Optimizer/issues/6
- Wortsman et al., *Stable and low-precision training* (StableAdamW) — https://arxiv.org/abs/2304.13013 · optimi (MIT) https://github.com/warner-benjamin/optimi
- Marin 32B spike issue #1368 — https://github.com/marin-community/marin/issues/1368
- Liu & Li code (no license) — https://github.com/Kanyooo/Loss_ISO · Spectra code — https://github.com/kimmichtank/spectra
- Asadi et al., *Resetting the Optimizer in Deep RL* — https://arxiv.org/abs/2306.17833
- Kalra & Barkeshli, *Why Warmup the Learning Rate?* — https://arxiv.org/abs/2406.09405
- Ellis et al., *Adam on Local Time* — https://arxiv.org/abs/2412.17113
- Marin 32B retrospective — https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/
- NVIDIA, bitwise-deterministic Megatron Core — https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/
- OPEN-1B — https://arxiv.org/abs/2609.17380 · OLMo-core SkipStepOptimizer — https://olmo-core.readthedocs.io/en/latest/_modules/olmo_core/optim/skip_step_optimizer.html
- AMD Primus determinism guide — https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/determinism-and-reproducibility.html
- torch 2.10 AOTriton attention backward (`deterministic` unused) — https://raw.githubusercontent.com/pytorch/pytorch/release/2.10/aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip
- PyTorch BLAS handle setup on ROCm — https://raw.githubusercontent.com/pytorch/pytorch/release/2.10/aten/src/ATen/cuda/CublasHandlePool.cpp
- hipBLASLt per-process kernel choice — https://github.com/ROCm/rocm-libraries/issues/13476 · hipBLASLt env vars — https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html · rocBLAS env vars — https://rocm.docs.amd.com/projects/rocBLAS/en/develop/reference/env-variables.html
- AOTriton releases (0.12b deterministic backward) — https://github.com/ROCm/aotriton/releases
- Embedding backward race — https://github.com/pytorch/pytorch/issues/196164 · fix https://github.com/pytorch/pytorch/pull/199569
- MI300X partition modes — https://instinct.docs.amd.com/projects/amdgpu-docs/en/develop/gpu-partitioning/mi300x/overview.html
- SDPA double-backward reports — https://discuss.pytorch.org/t/runtimeerror-derivative-for-aten-scaled-dot-product-is-not-implemented/200233
- `torch.optim.Muon` — https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html
- Hugging Face Hub storage limits — https://huggingface.co/docs/hub/en/storage-limits · NeurIPS 2026 E&D hosting — https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting
- AMD Developer Cloud — https://amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html
