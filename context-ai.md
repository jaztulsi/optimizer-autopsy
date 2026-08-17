# CONTEXT-AI — Optimizer Autopsy: the complete project context packet

> **What this file is.** A single, self-contained briefing dense enough that any LLM or human can read
> it once and then reason about *any* decision, file, equation, or future step in this project without
> opening anything else. It covers: the scientific question; the external literature, theories, math,
> libraries and tools we build on; the engineering spine and *why* it is built that way; a walk of the
> **entire directory** (every file — what it does now, and what it will do); the **future code** yet to
> be written, task by task, with planned signatures; every decision gate and **what happens on each
> possible outcome including failure**; the correctness invariants; a "why this and not that" FAQ; and a
> glossary. When in doubt about *why*, jump to §12. When you need the math, §4. When you need "where is
> X and what's its status", §6–§8.

> **Reading map.** §1 compass · §2 external landscape (papers, theories, libraries, tools) · §3 glossary-
> up-front of symbols · §4 the mathematics in full · §5 the engineering spine · §6 the entire directory,
> file by file · §7 the future code, task by task · §8 the end-to-end data/experiment lifecycle · §9 the
> decision-gate outcome tree (incl. every failure branch) · §10 free-tier survival mechanics · §11
> load-bearing invariants · §12 "why this not that" FAQ · §13 how to run · §14 current state + next action.

---

## 1. Compass — the whole project in one page

### 1.1 The phenomenon
When large language models are pretrained, the training loss occasionally **spikes**: it descends
smoothly, then jumps sharply, and the run is damaged or destroyed. These loss spikes are a real,
expensive, recurring failure mode at scale. The field's current responses are blunt: **skip** the bad
batch, **clip** the gradients, or (worst case) **roll back** to an earlier checkpoint and burn the
compute in between. Nobody routinely asks *where inside the optimizer the damage lives* or *whether you
can excise just that*.

### 1.2 The thesis (the bet this project makes)
A spike is not merely a bad step of the **weights** `w`; it **poisons the optimizer's internal state**.
Adam/AdamW keep two running per-coordinate statistics: `m` (first moment — momentum, an EMA of the
gradient) and `v` (second moment — an EMA of the squared gradient). The Adam update divides by
`sqrt(v)`, so a burst of huge gradients inflates `v` in a few directions and the optimizer stays
*mis-scaled in those directions long after the spike is over*. The core empirical bet: **that damage is
low-rank** — concentrated in a small number of directions in parameter space — so you can do
**rank-limited surgery** on `m`/`v` (and optionally `w`) in exactly those directions and recover as well
as a full reset, at a fraction of the cost, **without discarding what the model already learned**.

### 1.3 The three contributions
- **C1 — the instrument.** A *fork-and-intervene harness*: rewind to a snapshot `(w, m, v, per-param
  step, RNG, data-cursor)`, branch into several counterfactual continuations, and replay each **bit-for-
  bit identically** except for the single intervention under test. Bitwise determinism is the entire
  epistemic foundation — it is what makes "the repair caused the recovery" a *measurement* and not a
  story. **The harness is itself the shipped artifact.**
- **C2 — the attribution science.** A *localizer* that, at the failure step `t0`, identifies *which*
  directions of optimizer state are poisoned using two orthogonal signals — gradient **directional SNR**
  (is this a real-signal direction or noise?) and **curvature** (the top eigendirections of the
  *preconditioned* Hessian) — combined into a **poison score** and a **spectral-mass** statistic `ψ_k`
  that quantifies how much of the damage lives in the top-k directions. `ψ_k` is the variable the theory
  turns into a prediction.
- **C3 — the repair (explicitly contingent).** A *repair operator* = a **rank-|P| projection** that
  removes the poison component of `m`/`v`/`w` along the identified directions, plus a battery of forks
  proving targeted repair beats (a) doing nothing (`B0`) and (b) — the decisive control — repairing
  *random* directions of matched rank and strength (`Br`). If a cheap existing fix already recovers
  everywhere, C3 "dies" and the project pivots to a C1+C2 science/benchmark paper. That kill-test is run
  in **week one**, before the expensive machinery is built.

### 1.4 Why it is publishable (NeurIPS framing)
Novelty = C1 + C2 at proxy **and** 124M scale: a reproducible causal instrument for a failure people
treat as a black box, plus an honest map of where the damage lives, backed by a small theory
(Theorems 1 & 2, §4.10) that predicts, from a measurable `ψ_k`, *when* selective repair suffices vs.
when a global reset is unavoidable. "Causal" appears **only** where the `Br` random-direction control
earns it, using **paired** statistics on **held-out val loss**. Crucially the design is
**outcome-robust**: repair works → C1+C2+C3; a cheap fix wins → C1+C2 benchmark paper (NeurIPS Datasets
& Benchmarks); the poison is delocalized → "why global reset wins" is still a real result (Theorem 2's
regime, empirically confirmed). Every branch of reality is a paper (see the full tree in §9).

### 1.5 Scale ladder and compute reality
- **Proxy (1–3M params)** — same spike phenomenology, runs the *entire* attribution battery in minutes–
  1h on a free T4. This is the **primary iteration + keep/kill signal** and the free MVP.
- **124M nanoGPT** — a **robustness section**, not the contribution: reproduce spike windows + short
  forks (500–2000 steps), ~1–3h each, snapshots ~0.75 GB bf16 to HF Hub; quota-serialized by Kaggle's
  ~30 GPU-h/week.
- **410M** — ⛔ credits-gated (TRC/Lambda/Modal), out of scope for the free MVP; a transfer-of-mechanism
  robustness check only.

---

## 2. External landscape — the theories, papers, libraries and tools we stand on

### 2.1 The theory lineage (the week-1 reading order, and what each gives us)
This is the conceptual dependency chain the whole method rests on; the BUILD_PLAN §6 packet teaches each
from scratch, but here is what each contributes and *where it enters our code*:

1. **Adam's `m`/`v` and why `v` lags.** `v` is an EMA with decay `β2` (≈0.999), so after a gradient
   burst it stays inflated for `~1/(1−β2)` steps. This lag is the mechanism of the spike's persistence
   and the reason repairing `v` is the primary lever. → drives `localizer/poison.py`, `repair/operator.py`.
2. **Hessian eigenpairs.** The loss curvature `H = ∇²L` has eigenvalues `λ_i` / eigenvectors; the top
   eigendirections are the "sharp" directions where dynamics are unstable. → `localizer/curvature.py`.
3. **Edge of Stability (EoS).** Plain gradient descent is stable in a direction only while `η·λ < 2`;
   training naturally rides the edge where the top `λ` sits near `2/η`. Spikes are EoS excursions. →
   motivates operating on the *top* eigendirections.
4. **Preconditioned / adaptive EoS.** For Adam the relevant stability bound is on the **preconditioned**
   Hessian `H~ = D^{-1/2} H D^{-1/2}` (with `D = diag(sqrt(v)+ε)`); the adaptive edge sits near a
   constant (`≈38/η` in the literature we cite). → this is *why* we diagonalize `H~`, not `H`.
5. **HVP / Pearlmutter (1994).** Compute `H·vec` via a double-backward without ever forming `H` (which
   is `d×d`, impossible at scale). → `localizer/curvature.py::hvp`.
6. **Power iteration / Lanczos.** Extract top-k eigenpairs of a matrix accessible only through matvecs.
   `scipy.sparse.linalg.eigsh` (Lanczos) at proxy scale; `torch.lobpcg` (GPU) at 124M. → `curvature.py`.
7. **Gradient SNR.** A t-statistic over per-microbatch gradient projections separates consistent signal
   directions from noise directions. → `localizer/snr.py`.
8. **AR(2) / companion matrix / spectral radius.** A spike perturbation per eigendirection obeys a
   second-order recursion (momentum + preconditioner = 2-step memory); its spectral radius `ρ_i` decides
   decay vs. divergence. → `theory/recovery.ipynb` (Thm 1 & 2).
9. **Spike causes & fixes; small-scale proxies; deterministic replay.** → the induced-spike recipes,
   the proxy design, and the determinism spine respectively.

### 2.2 Key papers (cite prominently; stake claims against these)
- **2506.04805 — spike onset.** Tier-1; the onset mechanism. Cite prominently and **stake our recovery
  claims against it** (they explain onset; we act *at* onset and recover).
- **2501.06842 — SPAM (Spike-Aware Adam with Momentum reset).** The puzzle C2 resolves — SPAM resets
  momentum globally; we ask whether *selective* beats *global*. Also a baseline (`baselines/spam.py`).
- **2504.02507 — ZClip (adaptive z-score gradient-norm clipping).** Baseline (`baselines/zclip.py`).
- **2502.11034 — AdaGC (per-parameter adaptive gradient clipping).** Baseline (`baselines/adagc.py`).
- **LLM360 K2** — public checkpoint series with **natural** spike/normal pairs (optimizer state where
  available). Our source of *real* (not induced) spikes to test transfer (`spikes/k2.py`).
- **Karpathy nanoGPT / "reproduce GPT-2 (124M)"** — the architectural and 124M-infra template
  (`model/nanogpt.py`, `experiments/llm124m/`).

### 2.3 The library stack (exact pins and why each) — `research/requirements.txt`
Pinned to Kaggle's default GPU image; `harness/preflight.py::check_env()` parses this file and asserts
`installed == pinned` for the numerics-critical subset. **Single source of truth.**

| Library | Pin | Why it's here / what it does |
|---|---|---|
| `torch` | 2.10.0 | model, autograd (HVP double-backward), AdamW, `use_deterministic_algorithms`, `lobpcg`, `scaled_dot_product_attention` |
| `numpy` | 2.0.2 | the tokenized `uint16` memmap; `get_batch` slicing; RNG state |
| `scipy` | 1.16.3 | `sparse.linalg.eigsh` (Lanczos) top-k of `H~` at proxy scale |
| `safetensors` | 0.7.0 | snapshot serialization; `save_model` dedupes **tied weights** |
| `huggingface_hub` | 1.11.0 | push/pull snapshots + shards to the private artifacts repo |
| `datasets` | 5.0.0 | stream the source corpus during tokenization (`data/prepare.py`) |
| `tiktoken` | 0.12.0 | GPT-2 BPE tokenizer (vocab 50257 → fits `uint16`) |
| `wandb` | 0.26.1 | scalar logging only (loss/grad-norm/lr), grouped by spike/branch |
| `pyyaml` | 6.0.1 | experiment configs (`configs/__init__.py::load_config`) |
| `tqdm` | 4.66.4 | progress bars (non-numeric; not version-checked) |

> **Version drift note.** The Kaggle image is *newer* than the original plan assumed (numpy 2.0.2,
> datasets 5.0.0). `check_env()` strips torch's local build tag (e.g. `2.10.0+cu128`) so a CPU box and a
> CUDA box both pass the same pin. Watch for `datasets`/`numpy` API drift.

### 2.4 External services / tools (the "free stack")
- **GitHub** — this repo (text only; the laptop holds nothing heavy).
- **Kaggle** — real compute: T4×2 / P100, **30 GPU-h/week, 12 h/session**, persistent 20 GB
  `/kaggle/working`, read-only `/kaggle/input`. Phone-verified for GPU + Internet.
- **Google Colab** — quick proxy iteration; secrets via `google.colab.userdata`.
- **Hugging Face Hub** — private **Dataset** repo `jaztulsi/optimizer-autopsy-artifacts` holds
  snapshots, checkpoints, tokenized shards, spike sets (LFS-backed). The real limit is **upload time**,
  not storage → rotating `latest.safetensors`, never a commit per K steps.
- **Weights & Biases** — scalar metrics only; `wandb.init(group=spike_id, job_type=branch)`. **Never**
  log tensors/snapshots (those go to HF).
- **Secrets** — `HF_TOKEN`, `WANDB_API_KEY` live in Kaggle Secrets + Colab userdata (never in git).

> **⚠️ Operational constraint (this workstation).** Never run training or heavy compute on the user's
> Mac — it is sensitive and has crashed under load mid-meeting. **All training and Definition-of-Done
> runs go to Kaggle/Colab GPU.** The local machine is for git, reading, editing, and linting only. Even
> the tiny CPU self-checks are treated as GPU-only from here on.

---

## 3. Symbols, up front (full glossary in §15 for terms)

| Symbol | Meaning |
|---|---|
| `w`, `θ` | model weights / parameters, `∈ R^d` |
| `m` | Adam first moment (EMA of gradient) = `exp_avg` |
| `v` | Adam second moment (EMA of squared gradient) = `exp_avg_sq` |
| `v_bar` | pre-spike EMA baseline of `v` (the "clean" reference) |
| `β1, β2` | Adam EMA decay rates (≈0.9, 0.95–0.999) |
| `η` | learning rate; `ε` | Adam denominator epsilon |
| `D` | Adam preconditioner `diag(sqrt(v)+ε)` |
| `H` | Hessian `∇²L`; `λ_i` its eigenvalues |
| `H~` | preconditioned Hessian `D^{-1/2} H D^{-1/2}`; `(λ~_i, u_i)` its top eigenpairs |
| `Π_k` | projector onto the top-k eigenbasis `{u_i}` |
| `g_j` | gradient of microbatch `j` (exposed by the trunk grad hook) |
| `SNR_{u_i}` | directional gradient signal-to-noise (a t-statistic) along `u_i` |
| `p_i` | per-direction poison score; `P` the selected poisoned set |
| `ψ_k` | spectral-mass: fraction of poison in the top-k subspace |
| `ρ_i` | spectral radius of the AR(2) recovery recursion in direction `i` |
| `t0` | detector-fired pre-spike trigger step (where the autopsy happens) |
| `Δ` | causal effect `= val_loss(branch) − val_loss(B*)` |
| `B0/Bg/Bs/Bw/Bm/Bv/Br/B*` | branch types (§5.4) |

---

## 4. The mathematics — in full

### 4.1 AdamW, precisely (what the code implements)
Per coordinate at step `t`: `m ← β1·m + (1−β1)·g`, `v ← β2·v + (1−β2)·g²`,
bias-corrected `m̂ = m/(1−β1^t)`, `v̂ = v/(1−β2^t)`, update `θ ← θ − η·(m̂/(sqrt(v̂)+ε) + λ_wd·θ)`
(weight decay `λ_wd` decoupled — the "W" in AdamW). Our `configure_optimizers()`
(`model/nanogpt.py`) puts **≥2D tensors** (matmuls, embeddings) in the decayed group and **biases +
LayerNorm** in a `weight_decay=0` group, with `betas/eps/wd` passed **explicitly** so the trunk and
every fork share *identical* optimizer semantics — a silent mismatch here would break bitwise replay.

### 4.2 Why `v` is the crime scene
A spike is a burst of huge `g` in a few directions → `v` inflates there. Since the update divides by
`sqrt(v)`, an over-inflated `v` **shrinks** the effective step in poisoned directions long after the
burst (the optimizer goes partially blind there); the `β2` EMA makes this persist for `~1/(1−β2)`
steps. Worse, if `v` is corrupted *downward* (underflow), `1/(sqrt(v)+ε)` **explodes** into a secondary
blow-up — which is precisely why snapshots store `m,v` in **bf16, never fp16** (§5.3).

### 4.3 The right basis: the preconditioned Hessian `H~`
Adam is (locally) preconditioned gradient descent in the `D`-metric, so the operator that governs its
stability is `H~ = D^{-1/2} H D^{-1/2}`, not `H`. Its top eigenvectors `u_i` are the stiffest effective
directions — where spikes originate and where the poison concentrates. We work in this frozen-at-`t0`
eigenbasis for localization *and* repair.

### 4.4 Hessian-vector products (Pearlmutter)
Forming `H` (`d×d`) is impossible; we only ever need `H·vec`. Pearlmutter's trick: `H·vec =
∇_θ( (∇_θ L)·vec )` via a double-backward (`torch.autograd.grad(..., create_graph=True)` or
`torch.func.hvp`). For `H~·vec` we sandwich with `D^{-1/2}` (cheap, diagonal from live Adam `v`). Memory
control at 124M: a **small fixed probe batch** (4–8 sequences) + optional gradient checkpointing so
double-backward fits a 16 GB T4.

### 4.5 Top-k eigensolve — eigsh vs. lobpcg
- **Proxy:** `scipy.sparse.linalg.eigsh` on a `LinearOperator` wrapping the HVP. Recovers top-k to ~1e-4
  on a quadratic.
- **124M:** `torch.lobpcg` on GPU in fp32 (**primary at scale**) — eigsh's ~ncv-vector Lanczos basis is
  ~10–20 GB host RAM and OOMs Kaggle. An `estimate_mem(dim,k,ncv)` guard refuses/routes to lobpcg +
  parameter-subset localization (e.g. one block) when over budget.
- **Non-negotiable:** ONE **fixed** probe batch across every matvec of a single solve. A per-matvec
  stochastic batch makes the operator non-symmetric → eigenpairs become noise **with no error raised**.

### 4.6 Directional gradient SNR (a t-statistic)
For directions `U={u_i}` and per-microbatch grads `g_1..g_m`:
```
SNR_{u_i} = |mean_j (g_j · u_i)| / sqrt( var_j(g_j · u_i) / m )
```
High ⇒ consistent learning signal (protect it). Low ⇒ noise/poison (repair candidate). Computed **on the
fly** — project each microbatch grad onto `U`, keep only the `m×|U|` scalars, never materialize `m` full
grad vectors (OOM at 124M). It's a t-stat with `m−1` dof; grad-accum `m` is small (4–8), so a knob
aggregates over a few steps.

### 4.7 Poison score, spectral mass `ψ_k`, poisoned set `P`
Maintain a pre-spike EMA `v_bar`. At `t0`, in the FROZEN top-k eigenbasis of `H~`:
```
p_i   = |u_i · (v_t − v_bar)| / (|u_i · v_bar| + ε)      # robust denom: u_i has mixed signs
ψ_k   = || Π_k (v_t − v_bar) || / || v_t − v_bar ||       # fraction of poison in the top-k subspace
P     = { i : p_i > τ_p  AND  SNR_{u_i} < τ_s }           # repair only high-poison, low-signal dirs
```
Thresholds `τ_p, τ_s` come from the **normal-step null** (e.g. 95th percentile of `p_i`/SNR on matched
normal steps), never magic constants. `ψ_k` is the decision variable the theory consumes.

### 4.8 The repair operator — rank-|P| projection (not coordinate-wise)
Ramped over `R` steps by a schedule `c_i ∈ [0,1]`:
```
v_t ← v_t − Σ_{i∈P} c_i (u_i u_iᵀ)(v_t − v_bar) ;  then CLAMP v_t ≥ 0   (it is a variance)
m_t ← m_t − Σ_{i∈P} (m_t · u_i) u_i
```
This is a **rank-|P| projection onto the frozen eigenbasis** — deliberately not a diagonal op (using the
HVP/eigsh machinery to then act coordinate-wise would waste it). `u_i` is computed once at `t0`, frozen
through the repair; afterward we recompute `H~` and report the **eigenbasis rotation angle** (the
commutator correction §4.10 needs). Exposed as `repair_v / repair_m / repair_w / repair_all`.

### 4.9 The counterfactual and the effect `Δ`
Every causal number is `Δ = val_loss(branch) − val_loss(B*)`, on **held-out val** (never train loss),
where `B*` is the **data-aligned** clean counterfactual (§5.4). Attribution effect of a component `c`:
`effect(c) = Δ_noop − Δ_repair-c`; the decisive contrast is `Bv` (repair v) vs. `Br` (random subspace,
matched rank+strength). Because `Bv` and `Br` share seed+data, effects use **paired** bootstrap CIs
(§4.11).

### 4.10 The theory: AR(2), spectral radius, Theorems 1 & 2
Model a spike perturbation per eigendirection as a **second-order autoregressive** recursion (momentum
`β1` + preconditioner ⇒ 2-step memory). Its **companion matrix** has spectral radius
`ρ_i(λ~_i, η, β)`; `ρ_i < 1` ⇒ the perturbation decays (recovers), `ρ_i > 1` ⇒ it grows (diverges).
- **Theorem 1 (selective suffices).** If `ψ_k ≥ ψ*`, a rank-k repair drives `ρ_i < 1` in *every*
  direction ⇒ recovers **as well as global reset**.
- **Theorem 2 (reset necessary).** If the bulk (outside top-k) carries too much mass, *any* rank-k
  repair leaves some `ρ_i > 1` ⇒ only a global reset works.
`ψ*` is plotted against the top-k/bulk curvature gap. **Honesty (baked into the notebook):** the scalar-
`ρ_i` decoupling assumes `H` and `D` are simultaneously diagonalizable — the notebook **measures the
commutator error** and plots the **off-diagonal coupling term** so `ρ_i`'s validity regime is explicit;
it also states the linearize-around-fixed-point (`v` locally constant) caveat. Verified on toy quadratics
on free CPU (`theory/recovery.ipynb`, Task 19).

### 4.11 Statistics — paired, honest, corrected
Because `Bv` and `Br` (and each branch vs. `B*`) share seed+data, the honest test is a **paired
bootstrap** over seeds: pairing cancels shared variance and sharply raises sensitivity. Report paired
95% CIs for `effect(w/m/v)` and `Bv−Br`, the **minimum-detectable-effect** at `n=5–8` seeds, and
**multiple-comparison correction** across the method matrix. (`analysis/stats.py`.)

---

## 5. The engineering spine — and why it is built this way

### 5.1 Determinism (load-bearing) — `harness/determinism.py`, `harness/preflight.py`
Every causal claim = "branch A did X, branch B did nothing, they differ ONLY because of X." True only if
a do-nothing branch reproduces the trunk **bit-for-bit**. So:
- `CUBLAS_WORKSPACE_CONFIG` must be one of `":4096:8" / ":16:8"` and set **before** `import torch`
  (cuBLAS reads it once at init). Both `determinism._assert_preconditions()` and `preflight.check_env()`
  hard-assert this; the notebook launchers set it in their first cell.
- CUDA must **not** be initialized when deterministic algorithms are configured (else scheduled work
  isn't covered) — hard-asserted; `check_env()` prints the GPU name **last** because that call
  initializes CUDA.
- `seed_everything(seed, deterministic)` seeds python/numpy/torch(CPU+CUDA); when `deterministic`,
  sets `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `benchmark=False`. **Trunk may
  pass `deterministic=False` for speed; forks MUST use the default `True`.**
- `capture_rng_state()/restore_rng_state()` cover python, numpy, torch CPU, torch CUDA (all devices).
- `dropout = 0` everywhere on the deterministic path (removes a whole class of RNG divergence).
- **Bitwise identity holds only on the same GPU model in one session** → all branches of a fork run
  together. The proof lives in `tests/test_determinism.py`: snapshot → 50 steps → restore → 50 steps →
  assert `max|Δ| == 0` at *every* step (fp32, same device). GPU-verified: `max|Δ|=0`.

### 5.2 Data as a pure function of `step` — `data/prepare.py`
No streaming, no shuffle buffer, no RNG in the read path. The corpus is tokenized **once** (GPT-2 BPE
via tiktoken) into a flat `uint16` memmap per split. `get_batch(split, step, batch, block, dir, device)`
indexes by integer offset: row `i` reads a `block+1` window at `(step*batch*block + i*block) mod span`.
Consequences the whole project leans on:
- A batch is a **pure function of `step`** ⇒ two processes with the same args read identical bytes.
- The data "cursor" **IS the step** ⇒ resume is exact and O(1).
- Forks stay **data-aligned** automatically ⇒ every branch sees identical subsequent batches.
Shard determinism comes from pinning upstream: a fixed HF dataset **revision (commit sha)**, fixed
document order (no shuffle), a fixed token cap, the fixed GPT-2 tokenizer. Shards are gitignored; they
live on `/kaggle/input` (read-only) or are pulled to `/kaggle/working` by fixed revision. `_selfcheck()`
asserts determinism, target-shift (`y` = `x` shifted one token), correct offset, and cursor advance.

### 5.3 Snapshots that actually restore — `harness/snapshot.py` (Task 6 ✅)
A snapshot bundles: weights; `m` (exp_avg); `v` (exp_avg_sq); **per-param `step` tensor**; param-group
`betas/eps/wd`; RNG state; data cursor (= step); meta incl. `torch.cuda.get_device_name()`. Subtle
correctness rules:
- **Key optimizer state by PARAMETER NAME** (from `named_parameters()`), not optimizer index. On restore
  rebuild groups with the *same* grouping fn, map name→param→state, assert `m.shape==v.shape==p.shape`
  for every param.
- **Tied weights:** `wte.weight IS lm_head.weight` (same tensor). Use `safetensors.torch.save_model`
  (dedupes) or clone+drop-duplicate and restore the tie on load — else you double-store and break
  identity.
- **`m,v` in bf16, never fp16** (§4.2). fp32 allowed for the exact proxy gate.
- Upload to a **rotating** `latest.safetensors` (upload time, not storage, is the real limit).

### 5.4 The fork driver and the branch menu — `harness/fork.py` (Task 7 ✅, GPU-verified `max|Δ|=0`)
`fork(snapshot, branches, steps)` restores the SAME snapshot per branch, applies that branch's
intervention to `(w,m,v)`, runs `steps` with identical seed + data order (determinism ON), logs each to
its own W&B run grouped by `spike_id`, and is **NaN-safe** (Inf/NaN loss ⇒ record `survival=0`, continue
the battery). The branch menu:

| Branch | Intervention | Role |
|---|---|---|
| `B0` | noop | the do-nothing baseline; two `B0`s are the **Δ==0 gate** |
| `B*` | clean counterfactual (spike toggled off / clean same batch slot) | the reference all `Δ` subtract |
| `Bg` | global reset: zero `m,v` | the blunt upper-bound baseline |
| `Bs` | skip + clip | the **cheap fix**; if it recovers everywhere, C3 dies |
| `Bw/Bm/Bv` | repair `w` / `m` / `v` (localized) | the surgery under test |
| `Br` | **random** subspace repair, rank+strength matched to `Bv` | the control that earns "causal" |

**`B*` alignment rule:** `B*` is the SAME data stream with the spike toggled off (induced) or the clean
version of that same batch slot (corrupted-batch) — **never "skip to the next batch"** (that shifts the
cursor and confounds `Δ` with data order). **The gate:** two `B0` branches must give `Δ==0` (proxy/fp32)
or `Δ<ε` (124M/bf16, `ε` from a measured floor). If not 0 at proxy: STOP and fix determinism.

### 5.5 Metrics & artifacts routing
Scalars (loss/grad-norm/lr, and later `Δ`, `ψ_k`, effects) → W&B. Tensors (snapshots, checkpoints,
shards, spike sets, figures) → HF Hub. The laptop holds only the git repo.

---

## 6. The entire directory — file by file (current state + intended role)

Legend: ✅ done · 🟡 code-complete, DoD pending · ⬜ stub (docstring + TODO from Task-0 scaffold).
Repo root `index.html` = the GitHub Pages site — **leave untouched**. All code under `research/`.

### 6.1 `research/harness/` — the instrument
- **`determinism.py`** ✅ — `seed_everything(seed, deterministic=True)`, `capture_rng_state()`,
  `restore_rng_state(state)`, `_assert_preconditions()`. The determinism spine (§5.1).
- **`preflight.py`** ✅ — `check_env()` parses `requirements.txt`, asserts pinned versions
  (strips torch local build tag), asserts `CUBLAS_WORKSPACE_CONFIG` + CUDA-not-initialized, prints GPU
  name last. `_pinned()` is the parser. Called at the top of every entrypoint.
- **`secrets.py`** ✅ — `get_secret(name, required=True)` resolving env → Kaggle Secrets → Colab
  userdata → `.env`; `hf_token()`, `wandb_key()`. Never prints values; missing-required fails loudly.
- **`trunk.py`** ✅ (Task 5) — `run_trunk(cfg, data_dir, steps, start_step, on_step, grad_hook,
  use_wandb, deterministic, device) -> list[float]`. AdamW loop over `get_batch(step)`; grad accumulation
  with **per-microbatch grad exposure** (`grad_hook(step, micro, model)` sees `p.grad` = that microbatch
  alone, no extra backprop), a **per-step callback** (`on_step(step, info)`), and a **`pre_step` injection
  hook** (added for Task 8 spike recipes); grad-clip + norm; W&B scalar logging. `resume_trunk(...)` now
  wired to `snapshot.restore`. `build_model_opt/train_forward` extracted so `fork.py` reuses them.
  GPU-verified DoD on T4: `loss 10.851 → 4.730` over 200 steps.
- **`snapshot.py`** ✅ (Task 6) — `capture(model, optimizer, step, meta) -> dict`; `save/load`
  safetensors round-trip (fp32 @proxy; bf16 @124M per §5.3), name-keyed with per-param `step` + tied-weight
  safe; `push_to_hub/pull_from_hub`. Asserts all params share one AdamW step; `_selfcheck()` runs on CUDA
  when available and probes a post-restore `opt.step`. Restore path is also exercised by the Task 7 fork
  gate on T4 (below).
- **`fork.py`** ✅ (Task 7 — **C1, the shipped instrument**) — `run_fork(snapshot, intervention, cfg)`,
  `short_fork/full_fork`, `fork_matrix`, and `fork_determinism_gate` (the noop-vs-noop `Δ==0` gate).
  **GPU-verified on T4: `max|Δ|=0`** — the bitwise fork replay that licenses every causal claim.

### 6.2 `research/data/` — the determinism foundation
- **`prepare.py`** ✅ — `DataConfig` dataclass; `PRESETS` (`proxy`=TinyStories ~100M tok,
  `llm124m`=FineWeb-Edu ~1B tok — revisions still `"main"`, **TODO: pin to commit shas**); `prepare(cfg,
  out_root)` streams+tokenizes→`{train,val}.bin`+`meta.json`; `get_batch(...)` pure fn (§5.2);
  `upload_shards/pull_shards` (fixed revision); `resolve_data_dir(name)` searches
  `/kaggle/input`→`/kaggle/working`→`./data`; `_selfcheck()`.

### 6.3 `research/model/` — the network
- **`nanogpt.py`** 🟡 (Task 5) — `GPTConfig` (block_size, vocab_size=50304, n_layer, n_head, n_embd,
  dropout, bias); presets `PROXY`(3/6/192/256) & `GPT2_124M`(12/12/768/1024); `LayerNorm`(optional
  bias), `CausalSelfAttention`(fused QKV + `scaled_dot_product_attention`, causal), `MLP`(4×GELU),
  pre-norm `Block`, `GPT` (tied `wte`/`lm_head`, GPT-2 init + scaled residual-proj init
  `0.02/sqrt(2·n_layer)`), `num_params(non_embedding)`, `forward(idx, targets) -> (logits, loss)`,
  `configure_optimizers(weight_decay, lr, betas, eps)` → AdamW two groups (≥2D decayed / else no-decay),
  betas/eps/wd **explicit**.

### 6.4 `research/localizer/` — C2, the instrument (all ⬜)
> Note: the Task-0 stub TODOs describe a *simpler first cut* than the eventual BUILD_PLAN spec; both are
> given so you know the direction of travel.
- **`snr.py`** ⬜ — stub TODO: elementwise `|m|/(sqrt(v)+ε)` + group aggregation. **Task 10 (real):**
  directional SNR t-statistic over per-microbatch grads projected onto directions `U`, on-the-fly, `m×|U|`
  scalars only. DoD: high-signal dir scores high, random ~O(1); unit test asserts ordering.
- **`curvature.py`** ⬜ — stub TODO: `hvp`, `curvature_probe`, `curvature_score`. **Task 11 (real):**
  `hvp(loss, params, vec)` Pearlmutter; operator `H~ = D^{-1/2} H D^{-1/2}`; fixed probe batch;
  eigsh(proxy)/lobpcg(124M); `estimate_mem(dim,k,ncv)` guard. DoD: eigsh recovers top-3 to 1e-4 on a
  quadratic; 124M top-5 via lobpcg within T4 mem in <5 min.
- **`poison.py`** ⬜ — stub TODO: combine snr+curvature → ranked groups. **Task 12 (real):** `p_i`,
  `ψ_k`, `P` in the frozen top-k eigenbasis, thresholds from the normal-step null. DoD: `ψ_k` on a spike
  measurably > matched normal step; test asserts the gap + null-derived thresholds.

### 6.5 `research/repair/` — C3, the surgery (⬜)
- **`operator.py`** ⬜ — stub TODO: `repair(snapshot, localization, mode)`, modes
  `reset_v/rescale_v/reset_m/...`, `as_intervention(...)`. **Task 13 (real):** rank-|P| **projection**
  onto the frozen eigenbasis (§4.8), ramp `c_i`, clamp `v≥0`, `repair_v/m/w/all`, report eigenbasis
  rotation angle. DoD: repaired dirs' `v` within tol of `v_bar`, untouched dirs unchanged, `v≥0` (unit
  test).

### 6.6 `research/baselines/` — fork interventions citing prior work (all ⬜, Task 15)
- **`skip.py`** ⬜ — skip/replace the spike batch (data-side). **`clip.py`** ⬜ — global grad-norm clip.
  **`spam.py`** ⬜ — SPAM momentum reset (2501.06842). **`zclip.py`** ⬜ — z-score EMA grad-norm clip
  (2504.02507). **`adagc.py`** ⬜ — per-param adaptive clip vs weight norm (2502.11034). **`reset.py`**
  ⬜ — naive global `m,v→0` (the blunt version our localized repair must beat). Each exposes
  `as_intervention(cfg)` usable by `fork.run_fork`. DoD: each survives an induced proxy spike + logs to
  W&B.

### 6.7 `research/spikes/` — ground-truth failures (all ⬜)
- **`induce.py`** 🟡 (**Task 8, partial**) — four recipes (trunk cfg + expected spike step): (a) high-LR
  bump `lr_bump`, (b) tiny Adam eps 1e-12, (c) precision stress, (d) corrupted-batch. Injected via the
  trunk `pre_step` hook. Ground-truth scoring in place.
- **`tune_detector.py`** 🟡 (**Task 8, partial**) — `detect_spike()` (loss z-score / grad-norm EMA)
  fires pre-spike `t0`; `tune()` sweeps thresholds to maximize lead-time subject to an FP cap. **DoD NOT
  met:** on the real proxy trunk only **1 of 4** recipes (`lr_bump`) passes (bar is 3); result committed
  honestly as *"CALIBRATION-IN-PROGRESS."* Open issues: `precision`/`tiny_eps` diagnosed (one likely-inert,
  one unresolved); `corrupt_batch` is an instantaneous shock and may be graded against the wrong bar
  (flagged as a **spec question**: "detected at/before peak" vs. "detected with lead ≥ L"). Current blocker
  is mechanical, not scientific — Kaggle keeps assigning P100 (unsupported by the pinned torch build)
  instead of T4, crashing the run before training.
- **`k2.py`** ⬜ — **Task 21:** pull LLM360 K2 real spike/normal checkpoint pairs from HF, diff `(m,v)`,
  compute `ψ_k` on **real** spikes; run through the same attribution/repair pipeline (transfer stated as
  inference, not causation).

### 6.8 `research/analysis/` — the science layer (all ⬜)
- **`eval.py`** ⬜ — **Task 16:** `val_loss(model)` = deterministic mean loss over the FIXED val shard
  (identical batches for every branch). ALL `Δ` use this, never train loss.
- **`stats.py`** ⬜ — **Task 17:** `paired_test`, `bootstrap_ci`, multiple-comparison `correct`;
  minimum-detectable-effect at `n=5–8` seeds.
- **`attribution.py`** ⬜ — **Task 17:** run (4–6 sites)×(8 branches)×(5–8 seeds) on the proxy; mandatory
  `calibrate()` first: ~20 full-length forks, find the shortest fork length that preserves the **branch
  ordering** of `Δ` (rank-correlation), then run the big sweep at that length. `necessity_sufficiency`.
- **`figures.py`** ⬜ — **Task 23:** the five paper figures (causal map w/ paired CIs & Br subtracted;
  `ψ_k` histogram per spike class; theory-predicts-practice scatter; final-loss recovery vs `B*`;
  compute-to-recover). Colorblind-safe SVG+PNG to HF.

### 6.9 `research/experiments/` — the runnable drivers
- **`proxy/config.yaml`** 🟡 — proxy hyperparams (updated Task 5: model 3/6/192/256, `eps 1e-8`,
  `grad_accum 1`, `lr 3e-4`, `wd 0.1`, `betas [0.9,0.95]`, `grad_clip 1.0`, `batch_size 64`,
  `max_steps 5000`).
- **`proxy/smoke.py`** ⬜ — **Task 20:** whole pipeline in <10 min (induce→detect→snapshot→battery→`Δ`+
  one paired attribution row→one figure); asserts noop-vs-noop `Δ==0`. **Green here gates any 124M spend.**
  Currently calls `check_env()` then raises the TODO SystemExit.
- **`llm124m/config.yaml`** ⬜ — GPT-2 124M shape; `lr 6e-4`, `batch 12`, `grad_accum 40`, bf16
  snapshots. `hub_repo`/`hf_revision` **TODO**.
- **`llm124m/run.py`** ⬜ — **Task 22:** parse `"trunk"|"forks"`, train to spike windows (not
  convergence), bf16 snapshots to HF, short forks; disconnect-survival auto-resume from rotating
  `latest`; GPU-hour ledger printed vs the 30h cap. Currently `check_env()` + TODO SystemExit.

### 6.10 `research/theory/` and `research/tests/` and configs
- **`theory/README.md`** ⬜ → **`recovery.ipynb`** (Task 19): AR(2)/companion/spectral radius; Thm 1&2;
  `ψ*` + commutator/coupling plots on toy quadratics (free CPU).
- **`tests/test_determinism.py`** ✅ — the bitwise-replay gate (§5.1); GPU-verified `max|Δ|=0`.
- **`tests/test_secrets.py`** ✅ — `test_no_secrets_in_git` (regex-greps the tracked tree for `hf_…`
  tokens / assigned high-entropy literals), `test_loads_from_env`, `test_missing_required_raises`.
- **`configs/__init__.py`** ✅ — `load_config(path, overrides)`: yaml load + flat dotted-key overrides
  (`{"train.max_steps": 200}`).
- **`requirements.txt`** ✅ — the pins (§2.3). **`README.md`** ✅ — layout, free stack, secrets,
  determinism, run commands.
- **`notebooks/{kaggle_runner,colab_runner}.ipynb`** — thin launchers: set `CUBLAS_WORKSPACE_CONFIG`
  **before** importing torch, clone the repo, load secrets, expose `run(cmd)`.

---

## 7. The future code — task by task (what will be written, and its DoD)

Phase 0 (Tasks 0–3) ✅, Task 4 ✅, **Tasks 5–7 ✅ (GPU-verified)** are done; **Task 8 is in progress**.
Remaining, in dependency order:

- **Task 5 ✅ — proxy model + trunk loop.** GPU DoD ran on T4: `loss 10.851 → 4.730` over 200 steps.
- **Task 6 ✅ — snapshot/restore + HF Hub** (`harness/snapshot.py`). §5.3 rules; name-keyed, tied-weight
  safe, per-param step. Restore exercised by the Task 7 gate on T4.
- **Task 7 ✅ — fork driver + the Δ==0 GATE (C1, the shipped instrument)** (`harness/fork.py`). §5.4.
  **GATE A PASSED: noop-vs-noop `max|Δ|=0` on T4.** The causal instrument is trustworthy; science unlocked.
- **Task 8 🟡 (in progress) — spike induction + detector tuning** (`spikes/induce.py`,
  `spikes/tune_detector.py`). §6.7. Four recipes + online detector built; **DoD not met (1/4 recipes pass,
  need 3)**; `corrupt_batch` grading bar is an open spec question; current blocker is Kaggle P100-vs-T4
  scheduling, not the code.
- **Task 9 — cheap-branch battery + "method-is-dead" check** (register `B0/Bg/Bs/B*` in `fork.py`). Run
  across induced spikes; if cheap `Bs` recovers everywhere → **PIVOT** to C1+C2. Emits a one-page
  markdown verdict with the `Δ` table + PROCEED/PIVOT. *This is the week-one keep/kill.*
- **Task 10 — directional SNR** (`localizer/snr.py`). §4.6.
- **Task 11 — curvature: HVP + top-k of `H~`** (`localizer/curvature.py`). §4.4–4.5.
- **Task 12 — poison score, `ψ_k`, `P`** (`localizer/poison.py`). §4.7.
- **Task 13 — the repair operator** (`repair/operator.py`). §4.8.
- **Task 14 — full branch battery** (add `Bw/Bm/Bv/Br` to `fork.py`). §5.4; effect = `Δ_noop −
  Δ_repair-c` with `Br` subtracted. DoD: 8-branch fork runs on the proxy + emits `Δ` per branch +
  `effect(w/m/v)`.
- **Task 15 — baselines** (`baselines/*`; skip+clip first). §6.6.
- **Task 16 — held-out eval** (`analysis/eval.py`). §6.8.
- **Task 17 — attribution battery + calibration + paired stats** (`analysis/attribution.py`,
  `analysis/stats.py`). §4.11, §6.8.
- **Task 18 — GO/NO-GO #1** (extend the Task 9 verdict with localizer results). PROCEED iff `Bv` beats
  `B0` **and** `Br` beyond the paired CIs.
- **Task 19 — theory notebook** (`theory/recovery.ipynb`). §4.10.
- **Task 20 — proxy smoke gate** (`experiments/proxy/smoke.py`). §6.9. Green gates 124M.
- **Task 21 — natural K2 spikes** (`spikes/k2.py`). §6.7.
- **Task 22 — 124M on Kaggle within free limits** (`experiments/llm124m/run.py`). §6.9.
- **Task 23 — analysis & figures** (`analysis/figures.py`). §6.8.
- **Task 24 — workshop paper + reproducibility package** (`paper/`). Every claim slot → a Task 23
  figure; "causal" only where `Br` earned it; one-command replay per table; the harness runs standalone.
- **Task 25 ⛔ (credits-gated) — 410M mechanism transfer + full benchmark.** Robustness check; states the
  transfer claim with its 3-seed caveat.

---

## 8. End-to-end lifecycle — a single spike's journey through the pipeline

1. **Prepare** (`data/prepare.py`): tokenize the fixed corpus once → `uint16` memmap on HF/Kaggle.
2. **Trunk** (`harness/trunk.py`, determinism OFF for speed): train, logging loss/grad-norm/lr; the
   `grad_hook` exposes per-microbatch grads; the `on_step` callback runs the detector.
3. **Induce + detect** (`spikes/induce.py`, `spikes/tune_detector.py`): a recipe triggers a spike at a
   known step; `detect_spike()` fires the pre-spike trigger `t0`.
4. **Snapshot** (`harness/snapshot.py`) at `t0`: bundle `(w,m,v,per-param step,RNG,cursor)` → HF Hub
   (`latest.safetensors`), keyed by param name, tied weights handled, `m,v` bf16 (fp32 for the proxy
   gate).
5. **Localize** (`localizer/`): fix a probe batch → HVP → top-k eigenpairs of `H~` (eigsh/lobpcg);
   project per-microbatch grads for `SNR_{u_i}`; compute `p_i`, `ψ_k`, and the poisoned set `P` from the
   normal-step null.
6. **Fork the battery** (`harness/fork.py`, determinism ON): restore the SAME snapshot per branch, apply
   `{B0, B*, Bg, Bs, Bw, Bm, Bv, Br}` to `(w,m,v)`, run the calibrated short length, NaN-safe.
7. **Measure** (`analysis/eval.py`, `stats.py`): `Δ = val_loss(branch) − val_loss(B*)` on the fixed val
   shard; paired bootstrap CIs; `effect(w/m/v)` with `Br` subtracted.
8. **Decide + attribute** (`analysis/attribution.py`, the verdicts): PROCEED/PIVOT; build the causal map.
9. **Predict** (`theory/recovery.ipynb`): does the measured `ψ_k` land in Thm 1's regime (selective
   suffices) or Thm 2's (reset necessary)? Plot theory-vs-practice.
10. **Scale + write** (`experiments/llm124m/`, `analysis/figures.py`, `paper/`): repeat at 124M for
    robustness; regenerate the five figures; wire each paper claim to a figure.

---

## 9. Decision gates and possible outcomes — including every failure branch

The project is a decision tree with hard gates. Each gate has a defined action on failure, so no outcome
is a dead end.

- **GATE A — Determinism (Task 7): is noop-vs-noop `Δ == 0` at proxy/fp32?**
  - **Yes →** proceed; the instrument is trustworthy.
  - **No → STOP.** Fix determinism first (RNG source, a non-deterministic op flagged by
    `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, dropout leak). Nothing causal is allowed
    until this reads 0. This is non-negotiable — a nonzero `Δ` means every downstream number is measuring
    nondeterminism, not the intervention.

- **GATE B — Method-is-dead (Task 9, week one): does cheap `Bs` (skip+clip) recover final loss
  *everywhere*?**
  - **Yes → PIVOT.** The repair *method* (C3) is not needed. Ship **C1+C2 as a science/benchmark paper**
    (NeurIPS Datasets & Benchmarks): the instrument + the attribution map are the contribution; drop
    repair as the headline. Still a strong paper.
  - **No →** proceed to build the localizer/repair — there is a gap a cheap fix can't close.

- **GATE C — GO/NO-GO #1 (Task 18): does `Bv` beat `B0` AND beat `Br` beyond the paired CIs?**
  - **Yes →** the causal map is real; "causal" is licensed; full attribution + theory + scale.
  - **No (Bv ≈ Br) →** location did **not** matter → the poison is **delocalized**. Fallback paper:
    **"the poison is delocalized: why global reset wins"** — empirically Theorem 2's regime. Report the
    `ψ_k` distribution showing bulk mass dominates. Still a real result.

- **GATE D — Calibration (Task 17): does a SHORT fork preserve the branch *ordering* of `Δ`?**
  - **Yes →** run the big sweep at the shortest length that preserves ranking (huge compute saving),
    full-length only on a decision subset.
  - **No →** short forks don't preserve ranking → must run longer forks (more compute, fewer sites/seeds)
    or restrict scope to the proxy; state the limitation.

- **GATE E — Smoke gate (Task 20): does the whole proxy pipeline pass in <10 min with `Δ==0`?**
  - **Yes →** greenlight 124M spend.
  - **No →** do **not** spend Kaggle GPU-hours on 124M; fix the proxy pipeline first.

- **GATE F — Scale/quota (Task 22): can a 124M spike + one 8-branch short-fork battery complete within
  free Kaggle limits (across sessions)?**
  - **Yes →** robustness section stands.
  - **No →** report proxy-only results as the MVP; mark 124M as future work; the arXiv/workshop paper does
    not depend on it. 410M (Task 25) is credits-gated regardless.

- **Theory outcome (Task 19):** either the toy-quadratic regimes reproduce Thm 1 & Thm 2 (theory-predicts-
  practice scatter is tight) → strong theory section; or the commutator/coupling term is large →
  **honestly report** `ρ_i`'s narrowed validity regime (the notebook plots this on purpose).

**Net:** repair works → C1+C2+C3 full paper; cheap fix wins → C1+C2 benchmark paper; delocalized → "why
reset wins" paper; scale blocked → proxy MVP + future-work. There is no branch where the project
produces nothing.

---

## 10. Free-tier survival mechanics
- **Snapshot budget:** proxy `(w,m,v)` fp32 ≈ 12–36 MB; 124M bf16 ≈ 0.75 GB. Keep 5–10 fork-point
  snapshots on HF (LFS). Real limit = **upload time** → rotating `latest.safetensors`, not per-K-step
  commits.
- **Determinism:** trunk OFF (fast), forks ON; all branches in one session on one GPU model; `dropout=0`;
  exact `Δ==0` @proxy/fp32, `Δ<ε` @124M.
- **Curvature at scale:** `torch.lobpcg` on GPU (not `eigsh` — host-RAM OOM); ONE fixed probe batch;
  small (4–8 seq) HVP batch.
- **Kaggle:** 12 h/session, 30 h/week; run the sweep at the calibrated SHORT length; **print a GPU-hour
  estimate before each launch**; shard via `/kaggle/input` or HF pull.
- **W&B:** scalars only; `group=spike_id, job_type=branch` keeps hundreds of runs navigable; never log
  tensors.
- **Disconnect survival (124M):** checkpoint the **full** optimizer state (not just weights, or resume
  diverges) to a rotating `latest` on HF every ~1000 steps; auto-resume on restart; pin the GPU type per
  experiment; append GPU-hours to a persistent ledger.

---

## 11. Load-bearing invariants (do not let these regress)
1. Data is a **fixed pre-tokenized memmap**; a batch is a **pure function of `step`** (no streaming).
2. `CUBLAS_WORKSPACE_CONFIG` set **before** `import torch`; asserted in two places.
3. Snapshots key optimizer state by **param name** (+ shape asserts); handle **tied weights**; restore
   **per-param `step`**; store `m,v` in **bf16** (never fp16).
4. `B*` keeps the data stream **aligned** (toggle the spike off / clean the same slot) — never "skip".
5. Curvature: **one fixed probe batch** across all matvecs of an eigensolve; **lobpcg** (not eigsh) at
   124M.
6. Attribution uses **held-out val loss** and **paired** CIs; the `Br` random-subspace control is what
   licenses the word "causal".
7. Trunk and every fork share **identical optimizer semantics** (explicit betas/eps/wd, same grouping fn)
   — a mismatch silently breaks replay.

---

## 12. "Why this and not that" — the design-decision FAQ

**Why fork-and-replay instead of watching one run?** "The repair helped" is only meaningful against a
counterfactual identical except for the repair. One run can't tell you what would have happened otherwise.

**Why obsess over *bitwise* determinism — isn't `Δ ≈ 0` fine?** No. If a do-nothing branch drifts on its
own, you can't separate "the repair caused recovery" from "it would have recovered/diverged anyway."
`Δ==0` for noop-vs-noop *proves* the instrument. `Δ<ε` is only accepted at 124M against a measured floor.

**Why a tiny 1–3M proxy at all?** It shows the same spike/recovery phenomenology but runs the full battery
in minutes on free compute — the primary iteration + keep/kill signal. 124M shows the story survives
scale; it is not the contribution.

**Why data as a pure function of `step`, not a shuffled DataLoader?** A shuffled loader hides RNG/iterator
state that makes exact resume and cross-branch alignment nearly impossible. Integer-offset indexing makes
the cursor equal to the step → resume is O(1) and every branch is auto-aligned.

**Why key optimizer state by param *name*, not index?** Param ordering is an implementation detail that
can shift; a name→param map with shape asserts catches mismatches loudly instead of restoring the wrong
tensor into the wrong slot.

**Why bf16 for `m,v`, never fp16?** Adam divides by `sqrt(v)+ε`. fp16's small exponent range flushes tiny
`v` toward 0 → `1/(sqrt(v)+ε)` explodes. bf16 keeps fp32's exponent range. (fp32 only for the exact proxy
gate.)

**Why the *preconditioned* Hessian `H~`, not `H`?** Adam's effective dynamics live in the `D`-metric; the
stability-relevant and poison-carrying directions are eigenvectors of `H~ = D^{-1/2} H D^{-1/2}`.

**Why one fixed probe batch for the eigensolve?** A different batch per matvec makes the operator
non-symmetric; the solver then returns noise **without raising** — a silent correctness trap.

**Why lobpcg at 124M when eigsh works at proxy?** `eigsh`'s Lanczos basis needs ~10–20 GB host RAM at
124M and OOMs Kaggle; `torch.lobpcg` runs on the GPU within a T4's budget.

**Why HVP instead of forming the Hessian?** `H` is `d×d`; at even proxy scale that's infeasible. We only
ever need `H·vec`, which Pearlmutter's double-backward gives without materializing `H`.

**Why a rank-|P| *projection* repair, not coordinate-wise?** We paid for the HVP/eigsh machinery to find
the right *basis*; acting coordinate-wise afterward would throw that away. The projection removes the
poison exactly along the identified directions and leaves the rest untouched.

**Why a *random-subspace* control (`Br`)?** Perturbing the optimizer at all might help. `Br` repairs
random directions of matched rank+strength; only if `Bv` beats `Br` (paired CIs) did *location* matter —
that's what earns "causal".

**Why paired bootstrap statistics?** `Bv` and `Br` share seed+data, so their difference is a paired
measurement; pairing cancels shared variance → a far more sensitive, honest test than independent means.

**Why held-out val loss for `Δ`, not train loss?** Train loss can look "recovered" while generalization is
wrecked; the honest recovery metric is deterministic val loss over a fixed shard.

**Why define `B*` as the aligned clean stream, never "skip to next batch"?** Skipping shifts the data
cursor, so `Δ` would confound the intervention with a different data order. Toggling the spike off / using
the clean same slot keeps every branch on identical subsequent batches.

**Why run the "is the method even needed" kill-test in *week one*?** The expensive machinery
(localizer/repair/theory/124M) is only worth building if a cheap fix doesn't already win everywhere.
Front-loading the kill-test avoids months on a dead contribution.

**Why is repair (C3) *allowed* to fail?** The design is outcome-robust (§9): repair works → C1+C2+C3;
cheap fix wins → C1+C2 benchmark; delocalized → "why reset wins". Every branch is a paper.

**Why calibrate short-fork length by *ranking* preservation, not a lead-time number?** The science needs
the *ordering* of `Δ` across branches to be right; the shortest length that preserves that ordering is the
cheapest honest sweep length.

**Why pin exact library versions and assert them?** Determinism is torch-version-sensitive; "works on
Colab, silently different on Kaggle" is a real failure class. `check_env()` makes drift loud, not silent.

**Why keep everything off the laptop / in the cloud?** Free GPUs live on Kaggle/Colab; artifacts belong on
HF; the laptop holds only text (git). And this workstation specifically must not run heavy compute (§2.4).

---

## 13. How to run (all real runs on Kaggle/Colab, never the Mac)
Notebook launcher (`notebooks/kaggle_runner.ipynb` / `colab_runner.ipynb`) first cell sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** importing torch, clones the repo, loads secrets, exposes
`run(cmd)`. Then:
```
pip install -r research/requirements.txt
python -m research.harness.preflight                          # check_env OK + GPU name
python -m research.data.prepare proxy --upload <HF repo>      # build + push the proxy shard
python -m research.harness.trunk --config research/experiments/proxy/config.yaml --steps 200
python -m research.tests.test_determinism                     # bitwise replay OK, max|Δ|=0
python -m research.experiments.proxy.smoke                    # (once built) the keep/kill gate
python -m research.experiments.llm124m.run trunk              # (once built) 124M robustness, budgeted
```
Local (Mac) is fine only for: git, reading/editing, `--selfcheck` logic review — **not** training.

---

## 14. Current state and the immediate next action (2026-08-17)
- **Done & GPU-verified (T4):**
  - Tasks 0–4 (scaffold, env preflight, fixed data + `get_batch`, secrets + no-token-in-git,
    deterministic bitwise replay `max|Δ|=0` on CPU **and** Kaggle T4).
  - **Task 5** — proxy model + trunk loop; GPU DoD ran on T4 (`loss 10.851 → 4.730`, 200 steps).
  - **Task 6** — snapshot/restore + HF Hub; name-keyed, tied-weight safe, per-param step.
  - **Task 7 — the milestone: C1 is shipped.** The fork driver's noop-vs-noop gate reads
    **`max|Δ|=0` on T4** (GATE A passed). This is the bitwise fork replay that turns every downstream
    number into a *measurement* rather than a story — the whole causal program is now unlocked.
- **Task 8 (spike induction + detector): 🟡 IN PROGRESS — the current frontier.** Four induced-spike
  recipes (`lr_bump`, `tiny_eps`, `precision`, `corrupt_batch`) injected through the trunk `pre_step`
  hook, plus an online z-score/grad-norm detector with ground-truth scoring. **DoD not met:** on the real
  proxy trunk only **1/4** recipes (`lr_bump`) currently passes (bar is 3); committed honestly as
  *"CALIBRATION-IN-PROGRESS — NOT a passed DoD."* Three diagnosed sub-issues: (i) `precision`/`tiny_eps`
  need retuning (one likely-inert, one unresolved), (ii) `corrupt_batch` is an *instantaneous* shock, so
  "warn ≥ L steps before peak" may be the wrong bar — flagged as a **project spec question** ("detected
  at/before peak" is the honest bar), not something to keep re-tuning. **Current blocker is mechanical,
  not scientific:** Kaggle keeps assigning P100 (unsupported by the pinned torch build) instead of T4,
  crashing the run before training; earlier Task 8 numbers came from sessions that happened to get a T4.
- **Immediate next action:** get a fresh graded Task 8 run on a T4 (retry until Kaggle assigns one, or
  pin/guard the GPU type), then either (a) close Task 8 at ≥3/4 or (b) formally adopt the "at/before peak"
  bar for instantaneous recipes. After Task 8: Task 9 (cheap-branch battery `B0/Bg/Bs/B*` + the week-one
  **method-is-dead** keep/kill), then the localizer (Tasks 10–12).
- **Overall: instrument (C1) complete and trustworthy; project is at the *start of the actual science*.
  The heart — localizer (C2) and repair (C3) — is entirely ahead and not yet begun.**

*Known open TODOs to not forget:* pin `data/prepare.py` `PRESETS` revisions to real commit shas (both
currently `"main"`); set `llm124m/config.yaml` `hub_repo`/revision; the proxy `<5 min` prep is a design
target not yet timed on Kaggle; resolve the Kaggle P100-vs-T4 assignment so Task 8 can be re-graded;
decide the `corrupt_batch` detector bar. There is also a companion plain-English guide, `EXPLANATION.md`,
kept in sync with this state.

> **Repo note (not a project fact):** `main` currently shows 50↔50 divergence with `origin/main` — the
> local history was re-authored (stripping AI co-author trailers per CLAUDE.md §3) so the commits carry
> the same content under new SHAs. Reconcile with a force-push **only** with explicit sign-off (CLAUDE.md
> §5 review gate #3), not a merge.

---

## 15. Methods in detail — the parts that need their own paragraph

### 15.1 The four induced-spike recipes (`spikes/induce.py`, Task 8)
Each recipe returns a modified trunk config + the step at which a spike is expected, so the failure has
**known ground truth** (location + cause) to score the localizer against.
- **(a) High-LR bump** — transiently multiply `η` for a window; pushes the top preconditioned-Hessian
  direction past the adaptive edge of stability → a clean, tunable spike.
- **(b) Tiny Adam eps (1e-12)** — shrink `ε` so `1/(sqrt(v)+ε)` becomes explosive wherever `v` is small;
  stresses the exact `v`-underflow pathway repair targets.
- **(c) Precision stress** — run the sensitive step in reduced precision to induce numerical blow-up; the
  most "natural-looking" numerical spike.
- **(d) Corrupted-batch** — inject a duplicated/garbage batch at a known step; the **cheapest reproducible
  natural class**, and the one with the cleanest `B*` (the clean version of that same slot).

### 15.2 The online spike detector (`spikes/tune_detector.py`, Task 8)
`detect_spike()` thresholds a statistic over the streaming trunk logs — a **loss z-score** and/or a
**grad-norm EMA** — to fire a **pre-spike trigger `t0`** *before* the loss peak (so the snapshot captures
the poisoned-but-not-yet-exploded state). `tune()` sweeps thresholds over the labeled induced spikes and
picks the operating point that **maximizes lead-time** (`t0` before peak) subject to a **false-positive
cap** on normal steps. DoD: median lead ≥ `L` at FP ≤ `f` on held-out spikes for ≥3 of 4 recipes.

### 15.3 What "attribution" concretely computes (`analysis/attribution.py`, Task 17)
The output is a **causal map**: a table over (repair site) × (component `w/m/v`) of `effect(c) = Δ_noop −
Δ_repair-c` with `Br` subtracted and paired 95% CIs. Plus `necessity_sufficiency`: is the localized set
`P` **necessary** (repairing its complement doesn't recover) *and* **sufficient** (repairing just `P`
does)? The `calibrate()` pre-step (Task 17) is a mandatory cost control: run ~20 full-length forks, score
whether a short fork preserves the **branch ordering** of `Δ` (rank-correlation across branches), pick the
shortest length that preserves ranking, and run the big sweep there.

### 15.4 The five paper figures (`analysis/figures.py`, Task 23)
(1) **causal map** — `effect(w/m/v)`, paired CIs, `Br` subtracted; (2) **`ψ_k` histogram** per spike
class; (3) **theory-predicts-practice scatter** — measured `ψ_k` vs. whether selective repair beat global
reset (Thm 1 vs Thm 2 regime); (4) **final-loss recovery vs `B*`** across methods; (5) **compute-to-
recover**. Colorblind-safe, SVG+PNG to HF Hub.

---

## 16. Appendix A — real code signatures (copyable references across the tree)

These are the *actual* signatures in the repo today (✅) or the planned ones (⬜ / 🟡). Use them as the
canonical reference for how modules connect.

```python
# research/harness/determinism.py  ✅
def seed_everything(seed: int = 1337, deterministic: bool = True) -> None: ...
def capture_rng_state() -> dict: ...        # {python, numpy, torch, cuda}
def restore_rng_state(state: dict) -> None: ...

# research/harness/preflight.py  ✅
def check_env() -> None: ...                # asserts pins + CUBLAS var + CUDA-not-init; prints GPU last

# research/harness/secrets.py  ✅
def get_secret(name: str, required: bool = True) -> str | None: ...
def hf_token(required: bool = True) -> str | None: ...
def wandb_key(required: bool = True) -> str | None: ...

# research/data/prepare.py  ✅
@dataclass
class DataConfig: name; hf_dataset; hf_revision; hf_split="train"; text_key="text";
                  target_tokens=100_000_000; val_fraction=0.005; encode_batch=1024
def prepare(cfg: DataConfig, out_root: str = "data") -> str: ...          # writes {train,val}.bin+meta
def get_batch(split, step, batch, block, data_dir, device="cpu"): ...     # -> (x, y) int64; pure fn(step)
def upload_shards(out_dir, repo_id, revision=None) -> None: ...
def pull_shards(repo_id, revision, local_dir) -> str: ...
def resolve_data_dir(name: str) -> str: ...                              # /kaggle/input -> working -> ./data

# research/model/nanogpt.py  🟡 (Task 5)
@dataclass
class GPTConfig: block_size=256; vocab_size=50304; n_layer=3; n_head=6; n_embd=192; dropout=0.0; bias=False
PROXY = GPTConfig(3, 6, 192, 256); GPT2_124M = GPTConfig(12, 12, 768, 1024)
class GPT(nn.Module):
    def forward(self, idx, targets=None): ...                            # -> (logits, loss)
    def num_params(self, non_embedding: bool = True) -> int: ...
    def configure_optimizers(self, weight_decay, lr, betas, eps=1e-8): ...# AdamW, 2 param groups

# research/harness/trunk.py  🟡 (Task 5)
def run_trunk(cfg, data_dir, steps=None, start_step=0, on_step=None, grad_hook=None,
              use_wandb=False, deterministic=False, device=None) -> list[float]: ...
def resume_trunk(cfg, data_dir, from_snapshot, steps=None, **kw): ...     # NotImplementedError until Task 6
#   grad_hook(step, micro, model): p.grad holds THIS microbatch's grad alone (localizer SNR input)
#   on_step(step, {"loss","grad_norm","lr","model","opt"}): fork/snapshot/detector hook

# research/configs/__init__.py  ✅
def load_config(path, overrides: dict | None = None) -> dict: ...         # yaml + dotted-key overrides

# ---- planned (⬜) ----
# research/harness/snapshot.py  (Task 6)
def capture(model, optimizer, step, rng) -> dict: ...                     # {w,m,v,step,rng_state,meta}
def save(snapshot, path) -> None: ...; def load(path) -> dict: ...        # safetensors; bf16@124M/fp32@proxy
def push_to_hub(snapshot, repo_id) -> None: ...; def pull_from_hub(repo_id, step) -> dict: ...

# research/harness/fork.py  (Task 7)
def run_fork(snapshot, intervention, cfg): ...
def fork_matrix(snapshot, interventions, seeds): ...                      # N x methods x seeds
#   the Δ==0 gate: two "noop" branches must return identical final loss (proxy/fp32)

# research/localizer/snr.py (Task 10) / curvature.py (Task 11) / poison.py (Task 12)
def snr(U, micro_grads) -> dict: ...                                      # SNR_{u_i} t-statistic
def hvp(loss, params, vec): ...                                          # Pearlmutter double-backward
def top_k_preconditioned_hessian(model, probe_batch, D, k): ...           # eigsh(proxy)/lobpcg(124M)
def poison_score(snapshot, probe_batch) -> dict: ...                      # p_i, psi_k, P from the null

# research/repair/operator.py (Task 13)
def repair_v(snapshot, P, U, v_bar, c): ...; def repair_m(...); def repair_w(...); def repair_all(...)
def as_intervention(localization, mode): ...                             # callable for fork.run_fork

# research/analysis/eval.py (Task 16) / stats.py (Task 17) / attribution.py (Task 17)
def val_loss(model) -> float: ...                                        # deterministic, fixed val shard
def paired_test(a, b): ...; def bootstrap_ci(values): ...; def correct(pvalues): ...
def attribute(snapshot, localization, forks): ...; def calibrate(forks): ...
```

## 17. Appendix B — the branch/effect algebra (quick reference)
```
Δ_X            = val_loss(B_X) − val_loss(B*)          # effect of branch X vs the aligned clean run
effect(c)      = Δ_noop − Δ_repair-c                    # how much repairing component c ∈ {w,m,v} helps
causal signal  = effect(Bv) − effect(Br)               # targeted minus random-subspace (paired CI)
GATE (Task 7)  = |val_loss(B0#1) − val_loss(B0#2)|  ==  0   at proxy/fp32   (else STOP, fix determinism)
PROCEED (T18)  iff  Δ_Bv < Δ_B0 − CI  AND  Δ_Bv < Δ_Br − CI    (Bv beats both noop and random control)
Thm1 regime    iff  ψ_k ≥ ψ*   (rank-k repair drives every ρ_i<1 → selective ≈ global reset)
Thm2 regime    iff  ψ_k < ψ*   (bulk mass too large → any rank-k leaves some ρ_i>1 → reset necessary)
```

## 18. Appendix C — full glossary
- **Loss spike / blow-up** — a sudden large jump in training loss that damages/destroys a run.
- **AdamW** — Adam with decoupled weight decay; the optimizer whose state we autopsy.
- **`m` / `v`** — Adam first moment (momentum, EMA of grad) / second moment (EMA of grad²); `exp_avg` /
  `exp_avg_sq` in torch.
- **`v_bar`** — pre-spike EMA baseline of `v`; the clean reference the poison is measured against.
- **`β1, β2, η, ε, λ_wd`** — momentum decay, variance decay, learning rate, denominator epsilon, weight
  decay.
- **`D`** — Adam preconditioner `diag(sqrt(v)+ε)`.
- **Hessian `H`** — `∇²L`; `λ_i` its eigenvalues; top eigendirections are the sharp/unstable directions.
- **Preconditioned Hessian `H~`** — `D^{-1/2} H D^{-1/2}`; the operator governing Adam's local stability;
  its top eigenpairs `(λ~_i, u_i)` are the working basis for localization and repair.
- **HVP** — Hessian-vector product (Pearlmutter double-backward); computes `H·vec` without forming `H`.
- **Edge of Stability (EoS)** — GD is stable in a direction only while `η·λ<2`; training rides the edge;
  spikes are excursions past it (adaptive edge `≈38/η` for Adam).
- **Lanczos / power iteration** — top-k eigenpairs of a matvec-only operator; `eigsh` (proxy) / `lobpcg`
  (124M) here.
- **Probe batch** — the small **fixed** batch used across every matvec of one eigensolve (must be fixed or
  the operator is non-symmetric).
- **Directional SNR `SNR_{u_i}`** — t-statistic of per-microbatch grad projections onto `u_i`; high =
  signal, low = noise/poison.
- **Poison score `p_i`** — per-direction measure of how far `v` moved from `v_bar` along `u_i`.
- **Spectral mass `ψ_k`** — fraction of the total poison captured by the top-k eigen-subspace; the theory's
  decision variable; `ψ*` the threshold from Thm 1/2.
- **Poisoned set `P`** — `{ i : p_i > τ_p and SNR_{u_i} < τ_s }`, the directions repair acts on; thresholds
  from the normal-step null.
- **Rank-|P| projection repair** — remove the poison component of `m/v/w` along the `|P|` frozen
  eigendirections (not a coordinate-wise op).
- **Eigenbasis rotation angle** — how much the `H~` eigenbasis moved after repair; the commutator
  correction the theory tracks.
- **AR(2) / companion matrix / spectral radius `ρ_i`** — the per-direction recovery recursion (momentum +
  preconditioner = 2-step memory); `ρ_i<1` recovers, `ρ_i>1` diverges.
- **Commutator / coupling term** — the error from assuming `H` and `D` are simultaneously diagonalizable;
  measured and plotted to bound `ρ_i`'s validity.
- **Trunk** — the main training trajectory we snapshot and fork from (may run determinism OFF).
- **Fork / branch** — a continuation from a snapshot with one intervention applied (determinism ON).
- **`t0`** — the detector-fired pre-spike trigger step; where the snapshot + autopsy happen.
- **`Δ`** — causal effect `= val_loss(branch) − val_loss(B*)`, on held-out val loss.
- **Branch menu** — `B0` noop · `B*` aligned clean counterfactual · `Bg` global reset · `Bs` skip+clip
  (the cheap fix) · `Bw/Bm/Bv` localized repair of w/m/v · `Br` random-subspace control.
- **The gate** — noop-vs-noop `Δ==0` at proxy/fp32; nothing causal proceeds until it reads 0.
- **Snapshot** — `(w, m, v, per-param step, betas/eps/wd, RNG, data cursor, meta)`; keyed by param name;
  tied weights handled; `m,v` bf16 (fp32 for the proxy gate).
- **Proxy / 124M / 410M** — the 1–3M free-MVP model / the robustness scale / the credits-gated extension.
- **K2** — LLM360's checkpoint series providing **natural** spike/normal pairs (the real-spike test set).
- **Baselines** — skip / clip / SPAM (2501.06842) / ZClip (2504.02507) / AdaGC (2502.11034) / naive reset,
  each a fork intervention we benchmark localized repair against.

---

*This file is the single source of context. `why?` → §12 · math → §4 · methods detail → §15 · "where is X
/ its status" → §6 · signatures → §16 · future code → §7 · "what if it fails" → §9 · current frontier →
§14 · glossary → §18. Keep it updated as tasks close — it is the packet that lets any model or person walk
in cold and understand the entire project.*
