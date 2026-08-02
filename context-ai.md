# CONTEXT-AI — Optimizer Autopsy: full project context packet

**Purpose of this file.** Drop this into any LLM (or hand it to any human) and it will understand the
*entire* project: the scientific question, the mathematics, the engineering, the file layout, exactly
what is built vs. not, and — critically — *why each choice was made the way it was and not another
way*. It is written to be self-contained: you should not need to read any other file to follow it,
but every claim points at the code or plan that backs it. Read top to bottom once and you can reason
about any decision in this codebase.

**One-sentence version.** We fork a language-model training run at the exact step it "blows up",
surgically edit the optimizer's internal memory to remove the damage, and *prove* the edit caused the
recovery by comparing against a bit-for-bit identical replay that did nothing.

**Status at a glance (2026-08-02):** Foundation + first instrument piece done. Tasks 0–4 fully
complete and GPU-verified; Task 5 (proxy model + trunk training loop) is **code-complete and passing
its local self-check**, waiting only on its final Definition-of-Done run on a Kaggle GPU. ~19% of the
build. Everything built so far is tested and green.

---

## 0. The compass: what problem, why it matters, what's novel

### 0.1 The phenomenon
When you train large neural networks (LLMs), the training loss sometimes **spikes**: it is descending
smoothly, then suddenly jumps by a large amount and the run is damaged or destroyed. These "loss
spikes" are a well-documented, expensive failure mode of large-scale pretraining. People currently
respond with blunt instruments — skip the bad batch, clip the gradients, or in the worst case restart
from an old checkpoint and throw away compute.

### 0.2 The thesis
A loss spike is not just a bad step of the *weights*; it **poisons the optimizer's internal state**.
Adam (the standard optimizer) keeps a running memory of gradient statistics — `m` (first moment,
momentum) and `v` (second moment, per-coordinate variance estimate). A spike injects garbage into
`m` and especially `v`, and because Adam divides by `sqrt(v)`, that garbage keeps steering the
optimizer *after* the spike is over. The damage is **localized**: it lives in a small number of
directions in parameter space, not smeared everywhere. If that is true, you don't need a global
reset — you can do **rank-limited surgery** on just the poisoned directions and recover as well as a
full reset, at a fraction of the cost, while keeping everything the model had already learned.

### 0.3 The three contributions (this is what a paper claims)
- **C1 — the instrument.** A *fork-and-intervene harness* that can rewind to a snapshot `(weights,
  m, v, RNG, data-cursor)`, branch into several counterfactual continuations, and replay each one
  **bit-for-bit identically** except for the one intervention under test. This determinism is the
  entire epistemic foundation: without it, no causal claim is trustworthy. **This harness IS the
  artifact** the paper ships.
- **C2 — the attribution science.** A *localizer* that, at the moment of failure, identifies *which*
  directions in optimizer state are poisoned, using two signals: gradient **directional SNR** (is
  this direction real signal or noise?) and **curvature** (the preconditioned Hessian's top
  eigendirections). Plus a **spectral-mass** statistic `ψ_k` that measures what fraction of the
  poison lives in the top-k directions — the quantity that decides whether selective repair can even
  work.
- **C3 — the repair (contingent).** A *repair operator* that projects the poison out of `m`/`v`/`w`
  along the identified directions, and a battery of experiments proving it beats (a) doing nothing and
  (b) — the crucial control — repairing *random* directions of matched rank and strength. C3 is
  explicitly allowed to "die": if a cheap existing fix already recovers everywhere, the project
  pivots to shipping C1+C2 as a science/benchmark paper. That kill-test is run in *week one*, before
  building the expensive machinery.

### 0.4 Why it's publishable (NeurIPS framing)
The originality is C1 + C2 *at proxy and 124M scale* — a reproducible causal instrument for a failure
mode people currently treat as a black box, plus an honest attribution map of where the damage lives,
backed by a small theory (Theorems 1 & 2 below) that predicts *when* selective repair suffices vs.
when a global reset is unavoidable. The word "causal" is earned only where the random-direction
control (`Br`) licenses it. If repair turns out unnecessary, "the poison is delocalized: why global
reset wins" is *still* a real result. The design is built so that every branch of the outcome is a
paper.

---

## 1. The mathematics (the part you can present)

Notation: parameters `θ ∈ R^d` (weights `w`). Adam keeps `m` (EMA of gradient) and `v` (EMA of
squared gradient), both in `R^d`. The Adam update at step t (per coordinate) is roughly
`θ ← θ − η · m̂ / (sqrt(v̂) + ε)`, with bias-corrected `m̂, v̂`. Define the **preconditioner**
`D = diag(sqrt(v) + ε)`. Adam is (approximately) preconditioned gradient descent with metric `D`.

### 1.1 Why the second moment `v` is the crime scene
A spike is a burst of enormous gradients in a few directions. Those inflate `v` in exactly those
coordinates. Because the update divides by `sqrt(v)`, an over-inflated `v` *shrinks* the effective
step in the poisoned directions long after the spike — the optimizer goes partially blind there. If
`v` is corrupted downward (underflow — see §2.3 on why bf16 not fp16), `1/(sqrt(v)+ε)` *explodes* and
you get a secondary blow-up. Hence: **`v` is stored carefully, and repair targets `v` first.**

### 1.2 The right basis: eigendirections of the *preconditioned* Hessian
Damage and dynamics are naturally described in the eigenbasis of the operator that actually governs
Adam's local behavior — not the raw Hessian `H`, but the **preconditioned Hessian**
```
H~ = D^{-1/2} H D^{-1/2}
```
Its top eigenpairs `(λ~_i, u_i)` are the directions where the optimizer's effective dynamics are
stiffest — the "edge of stability" directions where spikes originate and live. We compute `H~`'s
top-k eigenpairs via **Hessian-vector products (HVP)** using Pearlmutter's double-backward trick
(never forming the `d×d` Hessian, which is astronomically large), fed to an eigensolver
(`scipy.sparse.linalg.eigsh` at proxy scale; `torch.lobpcg` on GPU at 124M, because eigsh's Lanczos
basis OOMs host RAM). **Invariant:** use ONE fixed probe batch across every matvec of a single
eigensolve — a per-matvec stochastic batch makes the operator non-symmetric and the eigenpairs become
noise *with no error raised*.

### 1.3 Directional signal-to-noise ratio (SNR) — is a direction real?
For a set of directions `U = {u_i}` and the per-microbatch gradients `g_1..g_m` (exposed by the trunk
loop's grad hook), project each gradient onto each direction and compute a t-statistic:
```
SNR_{u_i} = |mean_j (g_j · u_i)| / sqrt( var_j(g_j · u_i) / m )
```
High SNR ⇒ the direction carries consistent learning signal (don't touch it). Low SNR ⇒ the direction
is noise/poison (a repair candidate). Computed *on the fly* — project each microbatch grad and keep
only the `m × |U|` scalars, never materialize `m` full gradient vectors (that OOMs at 124M). Caveat:
this is a t-stat with `m−1` dof and `m` (grad-accum microbatches) is small (4–8), so signals may be
aggregated over a few steps.

### 1.4 Poison score and the spectral-mass statistic ψ_k
Maintain a pre-spike EMA baseline `v_bar` of the second moment. At the trigger step `t0`, in the
FROZEN top-k eigenbasis of `H~`:
```
p_i   = |u_i · (v_t − v_bar)| / (|u_i · v_bar| + ε)          # per-direction poison score
ψ_k   = || Π_k (v_t − v_bar) || / || v_t − v_bar ||          # fraction of poison in the top-k subspace
P     = { i : p_i > τ_p  AND  SNR_{u_i} < τ_s }              # the poisoned set to repair
```
`Π_k` projects onto the top-k eigenbasis. `ψ_k` is *the* decision variable: it says how much of the
damage is captured by a rank-k repair. Thresholds `τ_p, τ_s` are set from the **normal-step null
distribution** (e.g. 95th percentile of `p_i`/SNR measured on matched normal steps) — never magic
constants. The denominator in `p_i` is robust because `u_i` has mixed-sign entries.

### 1.5 The repair operator: rank-|P| projection (not coordinate-wise)
Repair removes the poison component along the identified directions, ramped over `R` steps by a
schedule `c_i ∈ [0,1]`:
```
v_t ← v_t − Σ_{i∈P} c_i (u_i u_iᵀ)(v_t − v_bar) ;  then CLAMP v_t ≥ 0   (it is a variance)
m_t ← m_t − Σ_{i∈P} (m_t · u_i) u_i
```
This is a **rank-|P| projection onto the frozen eigenbasis**, NOT a diagonal/coordinate-wise op —
using the HVP/eigsh machinery to then do something coordinate-wise would waste it. The eigenbasis
`u_i` is computed once at `t0` and frozen through the repair; afterward we recompute `H~` and report
the **eigenbasis rotation angle** (the commutator correction the theory needs).

### 1.6 The theory: AR(2) recovery, spectral radius, Theorems 1 & 2
Model a spike perturbation, per eigen-direction, as a **second-order autoregressive (AR(2))**
recursion — momentum (`β1`) + preconditioner give a 2-step memory. Write it as a companion matrix;
its **spectral radius** `ρ_i(λ~_i, η, β)` governs whether the perturbation in direction `i` decays
(`ρ_i < 1`, recovers) or grows (`ρ_i > 1`, diverges).
- **Theorem 1 (selective repair suffices).** If `ψ_k ≥ ψ*` (enough poison mass is in the top-k),
  a rank-k repair drives `ρ_i < 1` in every direction and recovers *as well as a global reset*.
- **Theorem 2 (reset is necessary).** If the bulk (outside top-k) carries too much mass, *any* rank-k
  repair leaves some `ρ_i > 1` — only a global reset works.
This predicts, from a measurable quantity `ψ_k`, which regime a given spike is in. **Honesty caveats
baked in:** the scalar-`ρ_i` decoupling assumes `H` and `D` are simultaneously diagonalizable — the
theory notebook MEASURES the commutator error and plots the off-diagonal coupling term so `ρ_i`'s
validity regime is explicit; and it states the linearize-around-fixed-point (`v` locally constant)
caveat. Verified on toy quadratics (free CPU) — this is Task 19.

---

## 2. The engineering (how the science is made real and honest)

### 2.1 The determinism spine — why this is load-bearing
Every causal claim reduces to: *branch A did X, branch B did nothing, and they differ ONLY because of
X.* That is only true if a "do-nothing" branch reproduces the trunk **bit-for-bit**. So the whole
harness is engineered around bitwise-reproducible replay:
- `CUBLAS_WORKSPACE_CONFIG` must be set **before** `import torch` (cuBLAS reads it once at init); the
  code hard-asserts this and that CUDA is not yet initialized. (`research/harness/determinism.py`.)
- `seed_everything(seed, deterministic)` seeds python/numpy/torch(CPU+CUDA), and when deterministic
  sets `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `benchmark=False`.
- `dropout = 0` everywhere on the deterministic path — removes an entire class of RNG divergence.
- **The gate (Task 7):** two "noop" branches must give `Δ == 0` at proxy/fp32. If not, everything
  stops until determinism is fixed. `Δ < ε` is tolerated only at 124M/bf16 against a *measured* floor.
- Trunk may run with determinism OFF (faster); forks flip it ON. Bitwise identity holds only on the
  **same GPU model in one session** — so all branches of a fork run together.

### 2.2 Data as a pure function of `step`
There is **no streaming, no shuffle buffer, no RNG in the read path**. The corpus is tokenized once
(GPT-2 BPE via tiktoken) into a flat `uint16` memmap. `get_batch(split, step, batch, block, dir)`
indexes by integer offset: row i reads a `block+1` window at `(step*batch*block + i*block) mod span`.
Consequences that the whole project leans on:
- A batch is a **pure function of `step`** ⇒ two processes with the same args read identical bytes.
- The data "cursor" **IS the step** ⇒ resume is exact and O(1) (just pass the step you left off at).
- Forks stay **data-aligned**: every branch sees identical subsequent batches. (`data/prepare.py`.)

### 2.3 Snapshots that actually restore (Task 6 — the subtle one)
A snapshot bundles: weights; Adam `m` (exp_avg) and `v` (exp_avg_sq); **per-param `step` tensor**;
param-group `betas/eps/wd`; RNG state; data cursor (= step); meta incl. `get_device_name()`. The
non-obvious correctness rules:
- **Key optimizer state by PARAMETER NAME** (from `named_parameters()`), not optimizer index. On
  restore, rebuild groups with the *same* grouping function, map name→param→state, and assert
  `m.shape == v.shape == p.shape` for every param. (Optimizer index ordering is fragile.)
- **Tied weights:** `wte.weight` IS `lm_head.weight` (same tensor). Use `save_model` (dedupes) or
  clone-and-drop-duplicate, and restore the tie on load — otherwise you double-store and break identity.
- **Store `m,v` in bf16, never fp16.** bf16 has fp32's exponent range, so tiny `v` values don't
  underflow; fp16 would flush small `v` toward 0 and blow up `1/(sqrt(v)+ε)`. fp32 allowed for the
  exact proxy gate.
- Push to a **rotating** `latest.safetensors` on the HF Hub — not a new versioned commit every K steps
  (upload time, not storage, is the real limit).

### 2.4 The fork driver and the `B*` counterfactual (Task 7)
`fork(snapshot, branches, steps)` restores the SAME snapshot for each branch, applies that branch's
intervention to `(w,m,v)`, runs `steps` with identical seed + data order (determinism ON), NaN-safe
(a branch that goes Inf/NaN records `survival=0` and the battery continues). The key definition:
- **`B*` = the aligned clean counterfactual**: the same data stream with the spike toggled off
  (induced spikes) or the clean version of the same batch slot (corrupted-batch). Every branch sees
  identical *subsequent* batches.
- **Never "skip to the next batch"** — that shifts the cursor and confounds the effect `Δ` with data
  order. All effects are `Δ = final_loss(branch) − final_loss(B*)`, using **held-out val loss**
  (Task 16), never training loss.

### 2.5 The free-compute stack (nothing heavy on the laptop)
- **Code** → this GitHub repo (text only). **Compute** → **Kaggle** (30 GPU-h/week, 12h/session,
  persistent 20GB `/kaggle/working`) for real runs; **Colab** for quick proxy iteration.
- **Artifacts** (checkpoints, `(w,m,v)` snapshots, tokenized shards) → **HuggingFace Hub**, private,
  LFS-backed (`jaztulsi/optimizer-autopsy-artifacts`, a Dataset repo).
- **Logs/metrics** → **Weights & Biases**, scalars only, `group=spike_id, job_type=branch`. NEVER log
  tensors/snapshots to W&B — those go to HF.
- **Scale reality:** proxy (1–3M params) does the full science on a T4 in minutes–1h. 124M is a
  *robustness section* (spike windows + short forks, ~1–3h each, quota-serialized). 410M is
  credits-gated (TRC/Lambda/Modal) and NOT part of the free MVP.

> **⚠️ Operational constraint (this environment):** never run training/heavy compute on the user's
> Mac — it is sensitive and has crashed under load. All training and Definition-of-Done runs go on
> Kaggle/Colab GPU. Local machine is for git, reading, and linting only.

---

## 3. Repository map (every module, what it does, and its status)

Repo root keeps `index.html` (the GitHub Pages site) untouched. All code lives under `research/`.
Legend: ✅ done · 🟡 in progress · ⬜ stub (docstring + TODO only).

```
research/
  harness/
    determinism.py   ✅ seed_everything, capture/restore RNG, the CUBLAS/CUDA hard-asserts (§2.1)
    preflight.py     ✅ check_env(): assert pinned versions + CUBLAS var + CUDA-not-init; print device
    secrets.py       ✅ load HF_TOKEN / WANDB_API_KEY from env or Kaggle/Colab store; never print
    trunk.py         🟡 AdamW training loop; two hooks (per-microbatch grad, per-step callback).
                        CODE COMPLETE, self-check passes; awaiting Kaggle DoD run (Task 5)
    snapshot.py      ⬜ bundle/restore (w,m,v,step,RNG,cursor); param-name keying; tied weights (Task 6)
    fork.py          ⬜ fork-and-intervene driver + the Δ==0 determinism gate + B* (Task 7)
  data/
    prepare.py       ✅ tokenize→uint16 memmap; get_batch(step) pure fn; HF up/download; path resolve
  model/
    nanogpt.py       🟡 config-driven GPT (proxy 1-3M / GPT2_124M); tied embeddings; AdamW 2 groups.
                        CODE COMPLETE (Task 5)
  localizer/
    snr.py           ⬜ directional SNR t-statistic (Task 10)
    curvature.py     ⬜ HVP (Pearlmutter) + top-k of preconditioned Hessian; eigsh/lobpcg (Task 11)
    poison.py        ⬜ poison score p_i, spectral mass ψ_k, poisoned set P from the null (Task 12)
  repair/
    operator.py      ⬜ rank-|P| projection repair of v/m/w; freeze basis; report rotation (Task 13)
  baselines/
    skip.py clip.py  ⬜ skip+reinject, global-norm clip (Task 15, do first)
    spam.py zclip.py adagc.py reset.py ⬜ published baselines, each cites its arXiv id (Task 15)
  spikes/
    induce.py        ⬜ reproducible spike recipes: high-LR, tiny-eps, precision, corrupted-batch (T8)
    tune_detector.py ⬜ sweep detector thresholds for max lead-time at bounded FP rate (Task 8)
    k2.py            ⬜ pull LLM360 K2 real spike/normal checkpoint pairs; ψ_k on real spikes (T21)
  analysis/
    eval.py          ⬜ val_loss(model): deterministic mean loss over fixed val shard (Task 16)
    stats.py         ⬜ paired bootstrap CIs; minimum-detectable-effect (Task 17)
    attribution.py   ⬜ (sites)×(branches)×(seeds) sweep + short-fork calibration (Task 17)
    figures.py       ⬜ the five paper figures, colorblind-safe (Task 23)
  experiments/
    proxy/config.yaml   🟡 proxy hyperparams (updated for Task 5: n_layer=3,n_head=6,n_embd=192)
    proxy/smoke.py      ⬜ whole-pipeline <10min gate; asserts noop-vs-noop Δ==0 (Task 20)
    llm124m/config.yaml ⬜ 124M config
    llm124m/run.py      ⬜ 124M trunk to spike windows; bf16 snapshots; short forks; GPU-h ledger (T22)
  theory/README.md   ⬜ (→ recovery.ipynb) AR(2)/companion/spectral radius; Thm 1&2 (Task 19)
  configs/__init__.py ✅ load_config(path, overrides): yaml + dotted-key overrides
  tests/
    test_determinism.py ✅ snapshot→50 steps twice→bitwise identical at each step (Task 4)
    test_secrets.py     ✅ grep tracked tree, assert no token literal committed (Task 3)
  requirements.txt   ✅ pinned to Kaggle's image (torch 2.10.0, numpy 2.0.2, ...)
notebooks/{kaggle_runner,colab_runner}.ipynb  — thin launchers: set CUBLAS var BEFORE import torch,
                                                 clone repo, load secrets, expose run(cmd)
```

### 3.1 What Task 5's code actually contains (the current frontier)
- **`model/nanogpt.py`** — a faithful nanoGPT: `GPTConfig` dataclass; `LayerNorm` (optional bias);
  `CausalSelfAttention` (fused QKV, `scaled_dot_product_attention`, causal); `MLP` (4× GELU);
  pre-norm `Block`; `GPT` with tied `wte`/`lm_head`, GPT-2 init (+ scaled residual-proj init
  `std=0.02/sqrt(2·n_layer)`); `configure_optimizers()` → AdamW with two param groups (decay for ≥2D
  tensors, no-decay for biases/LayerNorm), betas/eps/wd **explicit** so trunk and fork share
  identical optimizer semantics. Presets `PROXY` (n_layer=3,n_head=6,n_embd=192,block=256) and
  `GPT2_124M`.
- **`harness/trunk.py`** — `run_trunk(cfg, data_dir, ...)`: AdamW loop over `get_batch(step)`, grad
  accumulation, grad-clip+norm, W&B scalar logging, and the two hooks that make it the substrate for
  the rest: `grad_hook(step, micro, model)` fires after each microbatch's backward with `p.grad`
  holding THAT microbatch's grad alone (no extra backprop — the localizer's SNR consumes this), and
  `on_step(step, info)` fires after each optimizer step (where fork/snapshot/detector logic will
  hang). `resume_trunk` is stubbed until snapshot (Task 6). A `_selfcheck()` trains a tiny GPT on a
  synthetic repeating pattern and asserts loss drops + the grad hook fires clean — this PASSES locally
  (`loss 2.207 → 0.000, grad_hook fired 400x clean`).
- **`configs/__init__.py`** — `load_config(path, overrides)`: yaml load + flat dotted-key overrides.
- **`experiments/proxy/config.yaml`** — proxy hyperparams updated to match the model presets, adds
  `eps: 1e-8` and `grad_accum: 1`.

**The only thing left for Task 5** is its Definition of Done on a real GPU:
`python -m research.harness.trunk --config research/experiments/proxy/config.yaml --steps 200`
should train on the fixed proxy shard with loss decreasing, in under ~2 minutes. That needs the proxy
data shard prepared and the run done on Kaggle (not the laptop).

---

## 4. Build roadmap — the 26 tasks, in dependency order

Build order = dependency order: **env → data → determinism → snapshot → fork gate → cheap kill-test →
localizer → repair → theory → scale/figures/paper.** No "causal" word is allowed until the proxy gate
reads `Δ == 0`.

| Phase | Tasks | What it delivers | Status |
|---|---|---|---|
| 0 · Foundation | 0 scaffold, 1 env preflight, 2 fixed data, 3 secrets | the reproducible substrate | ✅ done |
| 1 · The instrument | 4 determinism replay, 5 model+trunk, 6 snapshot, 7 fork+GATE | C1: bit-identical fork harness | 4 ✅, 5 🟡, 6–7 ⬜ |
| 2 · Cheap kill-test | 8 spike induction+detector, 9 cheap-branch battery+"method-dead?" | keep/kill signal in week one | ⬜ |
| 3 · Localizer | 10 SNR, 11 curvature (HVP/eigsh/lobpcg), 12 poison/ψ_k/P | C2: where the poison lives | ⬜ |
| 4 · Repair + baselines | 13 repair operator, 14 full battery (+Br control), 15 baselines | C3: the surgery + controls | ⬜ |
| 5 · Science | 16 val eval, 17 attribution+paired stats+calibration, 18 GO/NO-GO, 19 theory | the causal map + theorems | ⬜ |
| 6 · Scale & paper | 20 proxy smoke gate, 21 natural K2 spikes, 22 124M on Kaggle, 23 figures, 24 paper, 25⛔ 410M | robustness + the writeup | ⬜ |

**Timeline (solo, free tier):** arXiv/workshop MVP (proxy C1+C2 + theory) ≈ 5–7 weeks focused / 3–4
months part-time. 410M + full sweep is not free (+2–4 weeks and GPU credits).

---

## 5. Decision criteria and kill-switches (decided now, not in rebuttal)

- **Determinism gate (Task 7):** noop-vs-noop `Δ` must be `0` at proxy/fp32. Nothing proceeds until
  it is. This is the single most important number in the project.
- **Method-is-dead test (Task 9, pulled to week one):** if cheap skip+clip (`Bs`, ~0 cost) already
  recovers final loss *everywhere*, the repair contribution (C3) is dead → pivot to shipping C1+C2 as
  a science/benchmark paper (NeurIPS Datasets & Benchmarks), drop repair as the headline.
- **GO/NO-GO #1 (Task 18):** if `Bv` (repair v) doesn't beat `B0` (noop) AND beat `Br` (random
  subspace) by more than the paired CIs → pivot.
- **Delocalized-poison fallback:** if selective repair never beats global reset → "the poison is
  delocalized: why global reset wins" is still a real paper (this is Theorem 2's regime, empirically).

---

## 6. Load-bearing invariants (do not let these regress)

1. Data is a **fixed pre-tokenized memmap**; a batch is a **pure function of `step`** (no streaming).
2. `CUBLAS_WORKSPACE_CONFIG` is set **before** `import torch`; the code asserts it.
3. Snapshots key optimizer state by **param name** (+ shape asserts); handle **tied weights**;
   restore **per-param `step`**; store `m,v` in **bf16** (never fp16).
4. `B*` keeps the data stream **aligned** (toggle the spike off / clean the same slot) — never "skip".
5. Curvature: **one fixed probe batch** across all matvecs of an eigensolve; **lobpcg** (not eigsh) at
   124M.
6. Attribution uses **held-out val loss** and **paired** CIs; the `Br` random-subspace control is what
   licenses the word "causal".

---

## 7. "Why this and not that" — the design-decision FAQ

**Why fork-and-replay instead of just watching one run?** Because "the repair helped" is only
meaningful against a counterfactual that is *identical except for the repair*. One run can't tell you
what would have happened otherwise. The fork + bit-identical `B*` is the counterfactual.

**Why obsess over bitwise determinism — isn't `Δ ≈ 0` fine?** No. If a do-nothing branch drifts from
the trunk on its own, you can't distinguish "the repair caused recovery" from "the run was going to
recover / diverge anyway." `Δ == 0` for noop-vs-noop is the proof the instrument is trustworthy.
`Δ < ε` is only accepted at 124M where a *measured* hardware floor makes exact identity impossible.

**Why a tiny 1–3M "proxy" model at all?** It exhibits the same spike/recovery phenomenology but runs
the *entire* attribution battery in minutes on free compute. It's the primary iteration + keep/kill
signal. 124M is there to show the story survives scale-up, not to be the contribution.

**Why data as a pure function of `step` instead of a normal shuffled DataLoader?** A shuffled loader
has hidden RNG/iterator state that makes exact resume and cross-branch alignment nearly impossible.
Integer-offset indexing makes the data cursor equal to the step — resume is O(1) and every branch is
automatically aligned.

**Why key optimizer state by param name, not index?** Optimizer param ordering is an implementation
detail that can change; a name→param→state map with shape asserts is robust and catches mismatches
loudly instead of silently restoring the wrong tensor into the wrong slot.

**Why bf16 for `m,v` and never fp16?** Adam divides by `sqrt(v)+ε`. fp16's small exponent range flushes
tiny `v` toward 0, making `1/(sqrt(v)+ε)` explode. bf16 keeps fp32's exponent range, so small values
survive. (fp32 only for the exact proxy gate.)

**Why the preconditioned Hessian `H~`, not the raw Hessian `H`?** Adam's *effective* dynamics live in
the `D`-metric. The directions that matter for stability and for where poison acts are the eigenvectors
of `H~ = D^{-1/2} H D^{-1/2}`, not of `H`.

**Why one fixed probe batch for the eigensolve?** A different batch per matvec makes the operator
non-symmetric; the eigensolver then returns noise *without raising an error* — a silent correctness
trap. One fixed batch keeps the operator symmetric.

**Why lobpcg at 124M when eigsh works at proxy?** `eigsh`'s Lanczos basis (~ncv vectors) needs
~10–20GB host RAM at 124M and OOMs Kaggle. `torch.lobpcg` runs on the GPU in fp32 within a T4's budget.

**Why a random-subspace control (`Br`)?** Repairing *some* directions might help just because you
perturbed the optimizer at all. `Br` repairs random directions of matched rank and strength; only if
targeted repair (`Bv`) beats `Br` (paired CIs) did *location* matter — that's what earns "causal".

**Why paired bootstrap statistics?** `Bv` and `Br` share seed + data, so their difference is a paired
measurement — pairing cancels shared variance and gives a far more sensitive, honest test than
comparing independent means.

**Why run the "is the method even needed" kill-test in week one?** Because the expensive machinery
(localizer, repair, theory, 124M) is only worth building if a cheap existing fix doesn't already solve
everything. Front-loading the kill-test avoids months of work on a dead contribution.

**Why is repair (C3) allowed to fail?** The project is structured so every outcome is a paper: repair
works → C1+C2+C3; cheap fix wins → C1+C2 benchmark paper; poison is delocalized → "why global reset
wins." The science is designed to be falsifiable and still publishable.

---

## 8. Glossary (fast reference)

- **Spike / blow-up** — sudden large jump in training loss that damages a run.
- **`m`, `v`** — Adam's first moment (momentum) and second moment (per-coord variance estimate).
- **`v_bar`** — pre-spike EMA baseline of `v`; the "clean" reference the poison is measured against.
- **`D`** — Adam preconditioner `diag(sqrt(v)+ε)`.
- **`H~`** — preconditioned Hessian `D^{-1/2} H D^{-1/2}`; its top eigenvectors `u_i` are the working
  basis.
- **HVP** — Hessian-vector product (Pearlmutter double-backward); computes `H v` without forming `H`.
- **SNR_{u_i}** — directional gradient signal-to-noise (a t-statistic); high = real signal, low = noise.
- **`p_i`** — per-direction poison score; **`ψ_k`** — fraction of poison mass in the top-k subspace;
  **`P`** — the set of directions selected for repair.
- **`ρ_i`** — spectral radius of the per-direction AR(2) recovery recursion; `<1` recovers, `>1`
  diverges.
- **Trunk** — the main training trajectory we snapshot and fork from.
- **Fork / branch** — a continuation from a snapshot with one intervention applied.
- **`B*`** — the aligned clean counterfactual branch; all effects are `Δ = loss(branch) − loss(B*)`.
- **`B0/Bg/Bs/Bw/Bm/Bv/Br`** — noop / global-reset / skip+clip / repair-w / repair-m / repair-v /
  random-subspace-control branches.
- **Proxy** — the 1–3M-param model that carries the science on free compute; **124M** — the robustness
  scale; **410M** — credits-gated extension.

---

*This file is the single source of context. If you're deciding "why is it done this way?", §7 is the
first stop; if you need the math, §1; the current frontier is Task 5 in §3.1. Keep this file updated
as tasks complete — it is the packet that lets any model or person walk in cold and understand the
whole project.*
