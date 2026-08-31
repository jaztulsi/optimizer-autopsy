# OPTIMIZER AUTOPSY — Build Plan
**Fork training at the moment of failure, find where the poison lives, repair it.**
Solo-friendly execution guide. Companion to the research plan at
<https://jaztulsi.github.io/optimizer-autopsy/>. Every task below is a **ready-to-paste Claude
Code prompt** — open Claude Code *inside this repo* and run them in order.

> **⚠ Superseded on hardware/budget/DoD by `PLAN_V6.md` (the governing plan).** This file is still the
> per-file *science* detail — the task-by-task build prompts and their code-level DoDs are current. But
> where it says **Kaggle / free-tier CUDA**, read **AMD MI300X (Path B: port everything, re-earn only the
> ROCm determinism guarantee via a go/no-go smoke test first); audited GPU ask ~15–40 GPU-h, not the old
> 500–600 — see `research/kaggle/step_timer_results.md`**; where
> Task 8's DoD says **≥3 of 4 spike recipes**, read the relaxed V6 bar of **≥2**; and the calibration +
> attribution battery scale with the surviving recipe count per V6's cut order. Current banked state and
> committed run evidence: `context-ai.md` §14 + `results/`. Tasks 0–7 (the instrument, C1) are done and
> CUDA-verified.

> **How to use this doc.** Do the one-time setup (§1). Then work top-to-bottom through §2 — paste
> one prompt into Claude Code, let it build + run its own check, commit, move on. Each task names
> its **Definition of Done (DoD)**. Tasks marked **⛔ credits-gated** need paid GPUs; skip them for
> the free MVP. The ordering is deliberately front-loaded to give you a **keep/kill signal in week
> one** (§2 Phase 2) before you build the expensive machinery.

---

## 0. Reality check — what free tier can and can't do

| Workload | Free tier? | How |
|---|---|---|
| Tiny **proxy** (1–3 M params): spikes, full attribution battery, short + full forks | ✅ easy | Kaggle/Colab T4, minutes–1 h; snapshots ~12–36 MB |
| **124 M** nanoGPT: spike windows + short forks (500–2000 steps) | ✅ budgeted | Kaggle P100/T4, ~1–3 h each; snapshots ~0.75 GB (bf16) → HF Hub |
| **124 M** *full-convergence* forks (~30–40) | ⚠ a few only | GPU-hours; spread across weeks or gate behind credits |
| **410 M** attribution + full N≥10×methods sweep | ⛔ credits-gated | TRC / Lambda / Modal credits; not free |

**The originality is the instrument (C1) + the attribution science (C2), both at proxy + 124 M.**
That is the free MVP and the arXiv/workshop paper. 124 M is a *robustness section*, not the
contribution; 410 M is credits-gated.

### The free stack (nothing on your laptop)
- **Code** → this GitHub repo. **Compute** → **Kaggle** (30 h/wk GPU, 12 h/session, persistent 20 GB
  `/kaggle/working`) for real runs; **Colab** for quick proxy iteration.
- **Artifacts** (checkpoints, `(w,m,v)` snapshots, tokenized shards, spike sets) → **HuggingFace Hub**
  (private, LFS-backed). **Logs/metrics** → **Weights & Biases** (scalars only — never tensors).
- Your laptop only holds this git repo (text). Everything heavy is born and dies in the cloud.

---

## 0.5 How long this takes (solo, free tier → workshop-scope MVP)

Two clocks: **build time** (coding with Claude Code + debugging) and **compute wall-clock** (free-GPU
quota + disconnects, which you wait on).

| Phase | Tall pole? | Focused (~full-time) | Part-time |
|---|---|---|---|
| Foundation + instrument (env, data, determinism, snapshot, fork gate) | **yes** — determinism is fiddly | 4–6 days | 1.5–2 weeks |
| Cheap kill-test (is the method even needed?) | — | 1–2 days | ~half week |
| Localizer (SNR, HVP/eigsh, ψ_k) | mem-bounded curvature | 3–4 days | ~1 week |
| Repair + battery + baselines | 5 baselines | 4–6 days | 1–1.5 weeks |
| Attribution + theory | many cheap proxy runs | 1.5–2 weeks | 3–4 weeks |
| 124 M on Kaggle + figures + paper | **quota-serialized** | 2–3 weeks | 4–6 weeks |

- **arXiv/workshop MVP** (proxy C1+C2 + theory): **~5–7 weeks focused**, **~3–4 months part-time.**
- **⛔ 410 M + full sweep:** not free — **+2–4 weeks and GPU credits.**

Sets the clock: Kaggle's ~30 GPU-h/week serializes 124 M work; determinism + snapshot round-trip is
the tall pole and gates everything; nothing about 410 M is free. Published "16 weeks" = 5 people +
paid A100s.

---

## 1. One-time setup (~45 min, manual — do this first)

1. **Accounts:** GitHub (have it), Kaggle, HuggingFace, W&B, Google (Colab/Drive), Zotero. All free.
2. **Tokens** (create once, store as *platform secrets*, never in git / on laptop):
   HF write token → Kaggle Secret `HF_TOKEN` + Colab Secret `HF_TOKEN`; W&B key → `WANDB_API_KEY`.
3. **HF storage:** create a **private** HF repo `jaztulsi/optimizer-autopsy-artifacts` (snapshots,
   checkpoints, tokenized shards all go here).
4. **Kaggle:** phone-verify to enable GPU; confirm "GPU T4 x2 / P100" accelerator.
5. **Read once:** bitwise-identical forks hold only on the **same GPU model in one session** — run
   all branches of a fork together. The trunk may train with determinism OFF (faster); only the
   forks need it ON.

**▶ Task 0 — scaffold the repo:**
```
We're building OPTIMIZER AUTOPSY research code inside this repo. Keep index.html at the repo root
untouched (GitHub Pages site). Put ALL code under research/. Create this skeleton with docstrings +
TODOs (no logic yet):

research/
  harness/{__init__,determinism,preflight,secrets,snapshot,trunk,fork}.py
  data/{__init__,prepare}.py
  model/{__init__,nanogpt}.py
  localizer/{__init__,snr,curvature,poison}.py
  repair/{__init__,operator}.py
  baselines/{__init__,skip,clip,spam,zclip,adagc,reset}.py
  spikes/{__init__,induce,tune_detector,k2}.py
  analysis/{__init__,eval,stats,attribution,figures}.py
  experiments/{proxy/{config.yaml,smoke.py},llm124m/{config.yaml,run.py}}
  theory/README.md
  configs/__init__.py
  requirements.txt   # PINNED versions matching Kaggle's image: torch, numpy, scipy, safetensors,
                     # huggingface_hub, datasets, tiktoken, wandb, pyyaml, tqdm
  README.md          # free stack, secrets, how to run on Kaggle/Colab
notebooks/{kaggle_runner,colab_runner}.ipynb   # thin launchers: set CUBLAS_WORKSPACE_CONFIG BEFORE
                     # importing torch, clone repo, load secrets, expose run(cmd)
.gitignore           # __pycache__, *.pt, *.safetensors, wandb/, data/, .env

DoD: `python -c "import research.harness, research.localizer, research.repair, research.analysis"`
imports cleanly.
```

---

## 2. The build — task by task (each = one Claude Code prompt)

### Phase 0 · Foundation (env, data, secrets) — before anything else

**▶ Task 1 — environment pin + preflight.** *Determinism is torch-version-sensitive.*
```
Pin exact versions in research/requirements.txt matching Kaggle's default image (torch, numpy,
scipy, datasets, safetensors, huggingface_hub, tiktoken, wandb). Implement
research/harness/preflight.py check_env(): assert those versions; assert
os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"); assert NOT
torch.cuda.is_initialized() at the time determinism is configured; print
torch.cuda.get_device_name(). Call check_env() at the top of every entrypoint.
DoD: check_env() raises a clear message on any mismatch and passes on Kaggle.
```

**▶ Task 2 — fixed pre-tokenized data shard (the determinism foundation).** *Do NOT stream.*
```
Implement research/data/prepare.py (nanoGPT-style): tokenize a FIXED corpus with GPT-2 BPE (tiktoken)
to a flat uint16 memmap. proxy: ~50-100M tokens of TinyStories or a FineWeb-Edu sample; 124M: a
larger fixed shard. Deterministic train/val split. Implement get_batch(split, step, batch, block)
that indexes the memmap by INTEGER OFFSET (no streaming, no shuffle buffer): a batch is a pure
function of step, so the data "cursor" is just the step and resume is exact and O(1). Upload shards
to the HF artifacts repo and pull by fixed revision (they live on Kaggle /kaggle/input read-only or
pulled to /kaggle/working). DoD: get_batch is bitwise-identical across two processes given the same
(split, step); proxy shard prepares in <5 min.
```

**▶ Task 3 — secrets hygiene.**
```
Implement research/harness/secrets.py: load HF_TOKEN / WANDB_API_KEY from env or the Kaggle/Colab
secret store; never print them; fail loudly if missing. Add a test that greps the tracked tree and
asserts no token literal is committed. DoD: tokens load from env; the no-secrets-in-git test passes.
```

### Phase 1 · The instrument (determinism → trunk → snapshot → fork GATE)

**▶ Task 4 — deterministic replay + smoke test (LOAD-BEARING).**
```
Implement research/harness/determinism.py: seed_everything(seed) seeds python/numpy/torch(cpu+cuda),
sets torch.use_deterministic_algorithms(True), cudnn.deterministic=True/benchmark=False, and
HARD-ASSERTS CUBLAS_WORKSPACE_CONFIG is already set and CUDA not yet initialized (it must be set in
the launcher BEFORE `import torch`). capture_rng_state()/restore_rng_state() cover python, numpy,
torch, torch.cuda. All snapshot/fork runs use dropout=0 (removes a whole class of RNG divergence).
Write tests/test_determinism.py: build a tiny model, snapshot, run N=50 steps INCLUDING
optimizer.step() twice from that snapshot with NO intervention, assert weights are bitwise-identical
(max abs diff == 0) at EACH step, fp32, same device. Catch+print any op that raises under
use_deterministic_algorithms. DoD: passes on CPU and on Kaggle GPU.
```

**▶ Task 5 — proxy model + trunk loop.**
```
Implement research/model/nanogpt.py: config-driven GPT (n_layer/n_head/n_embd/block_size). PROXY
config ~1-3M params (n_layer=3, n_embd=192, block_size=256, dropout=0); GPT2_124M config.
Implement research/harness/trunk.py: AdamW loop with EXPLICIT betas/eps/weight_decay and the two
param groups (decay / no-decay), over the Task 2 memmap via get_batch(step). Grad accumulation with
a hook that, after EACH microbatch, reads p.grad to expose per-microbatch grads (for the localizer
later) WITHOUT extra backprop. W&B logs loss/grad-norm/lr (scalars only). Step-callback hook for
localizer/fork logic. Trunk may run with determinism OFF for speed; forks flip it ON. DoD:
`python -m research.harness.trunk --config research/experiments/proxy/config.yaml --steps 200`
trains on the fixed proxy shard, loss decreases, CPU <2 min.
```

**▶ Task 6 — snapshot / restore of (w, m, v, RNG, cursor) + HF Hub.**
```
Implement research/harness/snapshot.py. Bundle: model weights; Adam m (exp_avg) and v (exp_avg_sq);
PER-PARAM step tensor; param-group betas/eps/wd; RNG state; data cursor (= step); meta incl.
torch.cuda.get_device_name(). CRITICAL: key all optimizer tensors by PARAMETER NAME (from
named_parameters()), not optimizer index; on restore rebuild groups with the SAME grouping fn, map
name->param->state, and assert m.shape==v.shape==p.shape for every param. Handle TIED weights
(wte.weight == lm_head.weight): use safetensors.torch.save_model (dedupes) or clone+drop-duplicate
and restore the tie on load. Store m,v in bf16 (fp32's exponent range — no underflow; fp16 would
flush small v toward 0 and blow up 1/(sqrt(v)+eps)); allow fp32 for the exact proxy gate. push_hf()/
pull_hf() to the private artifacts repo; upload to a ROTATING latest.safetensors path (not a new
versioned commit every K steps). DoD: snapshot->push_hf->pull_hf->restore reproduces a
bitwise-identical 20-step continuation at proxy/fp32.
```

**▶ Task 7 — fork-and-intervene driver + the DETERMINISM GATE (C1, the product).**
```
Implement research/harness/fork.py: fork(snapshot, branches, steps) restores the SAME snapshot for
each branch, applies that branch's intervention fn to (w,m,v), runs `steps` with identical seed +
data order (determinism ON), logs each to its own W&B run grouped by spike_id. Register only "noop"
and "B_star" for now. DEFINE B* TO PRESERVE DATA ALIGNMENT: B* is the SAME data stream with the
spike toggled off (induced spikes) or the clean version of that same batch slot (corrupted-batch) —
so every branch sees identical SUBSEQUENT batches. Do NOT "skip to the next batch" (that shifts the
cursor and confounds Δ with data order). Δ = L_final(branch) − L_final(B*). Wrap each branch NaN-safe
(Inf/NaN loss -> record survival=0, continue the battery). THE GATE: two "noop" branches must give
Δ == 0 (proxy/fp32) or Δ < ε (124M/bf16, ε from a measured floor). If not 0 at proxy, STOP and fix
determinism — every causal claim depends on this. DoD: noop-vs-noop Δ==0 on the proxy.
```

### Phase 2 · The cheapest kill-test FIRST (no localizer needed — biggest de-risk)

**▶ Task 8 — spike induction + detector tuning.**
```
Implement research/spikes/induce.py: reproducible recipes returning a trunk config + expected spike
step: (a) high-LR bump, (b) tiny Adam eps 1e-12, (c) precision stress, (d) corrupted-batch (inject a
duplicated/garbage batch — cheapest natural class). Implement detect_spike() (loss z-score /
grad-norm EMA) firing a pre-spike trigger t0. Implement research/spikes/tune_detector.py: sweep
thresholds over labeled induced spikes, pick the operating point maximizing lead-time (t0 before
peak) subject to a false-positive cap on normal steps. DoD: each recipe produces a visible spike;
detector reports median lead >= L steps at FP rate <= f on held-out spikes for >=3 of 4 recipes.
```

**▶ Task 9 — cheap-branch battery + the "method-is-dead" check.** *Learn keep/kill in week one.*
```
In fork.py register the interventions that need NO localizer: B0 (noop), Bg (global reset: zero m,v),
Bs (skip+clip), B* (clean counterfactual). Run them across the induced spikes and answer, using the
held-out val metric (Task 16 defines it; use train-loss placeholder until then): does cheap Bs
(~0 cost) already recover final loss EVERYWHERE? If yes -> the repair METHOD contribution is dead;
pivot now to shipping C1+C2 as a science/benchmark paper (NeurIPS D&B). Emit a one-page markdown
verdict with the Δ table and a PROCEED/PIVOT recommendation. DoD: the verdict prints with numbers.
```

### Phase 3 · Localizer (C2)

**▶ Task 10 — directional SNR.**
```
Implement research/localizer/snr.py. Consume the per-microbatch grads exposed by the trunk hook;
for directions U (columns u_i) compute SNR_{u_i} = |mean_i(g·u_i)| / sqrt(var_i(g·u_i)/m). Project
each microbatch grad onto U ON THE FLY and keep only the m×|U| scalars — never materialize m full
grad vectors (OOM at 124M). Note SNR is a t-statistic with m-1 dof; grad-accum m is small (4-8), so
expose a knob to aggregate over a few steps. DoD: high-signal directions (mean-grad direction) score
high SNR, random directions ~O(1); a unit test asserts the ordering.
```

**▶ Task 11 — curvature: HVP + memory-bounded top-k of the preconditioned Hessian.**
```
Implement research/localizer/curvature.py. hvp(loss, params, vec) via Pearlmutter double-backward
(torch.autograd.grad create_graph=True or torch.func.hvp). Operator = preconditioned Hessian
H~ = D^{-1/2} H D^{-1/2}, D = diag(sqrt(v)+eps) from the live Adam state. FIX ONE clean probe batch
for the ENTIRE eigen-solve (a stochastic per-matvec batch makes the operator non-symmetric and the
eigenpairs noise — with no error thrown); use the same probe batch across spikes. Use a SMALL probe
batch (4-8 sequences) + optional grad checkpointing so double-backward fits a 16GB T4. Two paths:
proxy -> scipy.sparse.linalg.eigsh (LinearOperator); 124M -> torch.lobpcg on GPU fp32 (PRIMARY at
scale — eigsh's ~ncv Lanczos basis is ~10-20GB host RAM and OOMs Kaggle). Add estimate_mem(dim,k,ncv)
that refuses/routes to lobpcg + parameter-subset localization (e.g. one block) when over budget. DoD:
on a small quadratic, eigsh recovers top-3 eigenpairs to 1e-4; 124M top-5 via lobpcg returns within
T4 memory in <5 min.
```

**▶ Task 12 — poison score, spectral-mass ψ_k, poisoned set P.**
```
Implement research/localizer/poison.py. Maintain a pre-spike EMA v_bar. At t0, in the top-k eigenbasis
u_i of H~ (FROZEN at t0, pre-repair):
  p_i   = |u_i · (v_t - v_bar)| / (|u_i · v_bar| + eps)     # robust denominator (u_i has mixed signs)
  psi_k = ||Pi_k (v_t - v_bar)|| / ||v_t - v_bar||
  P     = { i : p_i > tau_p and SNR_{u_i} < tau_s }
Set tau_p, tau_s from the NORMAL-STEP NULL (e.g. 95th percentile of p_i / SNR on matched normal
steps), not magic constants. DoD: psi_k on an induced spike is measurably higher than on a matched
normal step; a test asserts the gap and that thresholds derive from the null.
```

### Phase 4 · Repair, full battery, baselines (C3)

**▶ Task 13 — the repair operator.**
```
Implement research/repair/operator.py. Repair = RANK-|P| PROJECTION onto the frozen top-k eigenbasis
(NOT a coordinate-wise "diagonal" op — that would waste the HVP/eigsh machinery), ramped over R steps
by schedule c_i in [0,1]:
  v_t <- v_t - sum_{i in P} c_i (u_i u_i^T)(v_t - v_bar) ; then CLAMP v_t >= 0 (it's a variance)
  m_t <- m_t - sum_{i in P} (m_t · u_i) u_i
Expose repair_v/repair_m/repair_w/repair_all. Compute u_i once at t0, freeze, repair, then recompute
H~ and REPORT the eigenbasis rotation angle (this is the commutator correction Task 19 needs). DoD:
repaired directions' v sit within tol of v_bar, untouched directions unchanged, v>=0 (unit test).
```

**▶ Task 14 — register the full branch battery.**
```
In fork.py add Bw (repair_w), Bm (repair_m), Bv (repair_v), and Br (RANDOM-subspace repair, rank +
strength matched to Bv but random directions — the CONTROL that proves LOCATION mattered).
Attribution: effect(c) = Δ_noop − Δ_repair-c, with the key contrast Bv vs Br. DoD: fork(snapshot,
all 8 branches, steps) runs on the proxy and emits Δ per branch + effect(w/m/v) with Br subtracted.
```

**▶ Task 15 — baselines (skip+clip first, rest can trail).**
```
Implement research/baselines/: skip.py (skip+reinject) and clip.py (global-norm clip) FIRST; then
spam.py (momentum reset, 2501.06842), zclip.py (z-score EMA grad-norm clip, 2504.02507), adagc.py
(per-param adaptive clip, 2502.11034), reset.py (global m,v reset). Each is a drop-in trunk callback
citing its arXiv id. DoD: each survives an induced proxy spike (loss recovers) and logs to W&B.
```

### Phase 5 · The science (attribution, theory)

**▶ Task 16 — held-out eval harness (defines "final loss").**
```
Implement research/analysis/eval.py val_loss(model): deterministic mean loss over the FIXED val shard
(identical batches for every branch). ALL Δ = val_loss(branch) − val_loss(B*) use this, never
training loss. DoD: val_loss is bitwise-reproducible across calls and is wired into fork.py's Δ.
```

**▶ Task 17 — attribution battery + proxy-fork calibration + paired stats.**
```
Implement research/analysis/stats.py: because Bv and Br share seed+data, compute effect via PAIRED
differences (paired bootstrap over seeds) — cuts variance sharply and is the honest test; report
paired 95% CIs for effect(w/m/v) and Bv−Br, plus the minimum-detectable-effect at n=5-8 seeds.
Implement research/analysis/attribution.py to run (4-6 sites)x(8 branches)x(5-8 seeds) on the proxy.
FIRST the mandatory cost control calibrate(): run ~20 FULL-length forks, then score whether a SHORT
fork (500-2000 steps) preserves the BRANCH ORDERING of Δ (rank-correlation of Δ across branches, not
just a lead-time number); pick the shortest length that preserves ranking; run the big sweep at that
length, full-length only on a decision subset. Metrics in strict order: (1) final val-loss vs B*,
(2) compute-to-recover, (3) survival. DoD: one command emits the causal-map table (paired CIs) + the
calibration plot, entirely on free-tier proxy compute.
```

**▶ Task 18 — GO/NO-GO #1 (full attribution sanity).**
```
Extend the Task 9 verdict with the localizer results: (1) does Bv beat B0 AND beat Br by more than
the PAIRED CIs -> causal map is real, proceed; (2) re-check the method-is-dead test with real numbers.
Emit an updated one-page verdict with a clear PROCEED / PIVOT recommendation and the supporting table.
DoD: verdict prints PROCEED or PIVOT with paired-CI evidence.
```

**▶ Task 19 — theory notebook (AR(2) / Theorems 1 & 2).**
```
Create research/theory/recovery.ipynb (free CPU). Model a spike perturbation as a per-direction AR(2)
recursion (momentum + preconditioner => 2-step memory); companion matrix; spectral radius
rho_i(lambda~_i, eta, beta). Verify Thm 1 (psi_k >= psi* => rank-k repair drives rho_i<1 everywhere,
recovers as well as global reset) and Thm 2 (bulk mass too large => any rank-k repair leaves rho_i>1,
reset necessary). Plot psi* vs the top-k/bulk curvature gap. HONESTY: the scalar-rho_i decoupling
assumes H and D are simultaneously diagonalizable; MEASURE the commutator error AND plot the
off-diagonal coupling term so rho_i's validity regime is explicit; state the linearize-around-fixed-
point (v locally constant) caveat. DoD: notebook reproduces the Thm1/Thm2 regimes on toy quadratics
and plots psi* + the coupling term.
```

### Phase 6 · Scale, figures, paper

**▶ Task 20 — end-to-end proxy smoke gate (must pass before any Kaggle spend).**
```
Implement research/experiments/proxy/smoke.py: one command running the WHOLE pipeline on the proxy in
<10 min on CPU/Colab — induce a spike, detect t0, snapshot, run the branch battery, compute Δ +
one attribution row (paired), emit one figure. DoD: `python -m research.experiments.proxy.smoke`
exits 0, asserts noop-vs-noop Δ==0, writes a results table + figure. Green here gates 124M.
```

**▶ Task 21 — natural spikes (LLM360 K2 + corrupted-batch).**
```
Implement research/spikes/k2.py: pull LLM360's K2 spike/normal checkpoint pairs from HF, load
optimizer state where available, diff (m,v) spike-vs-normal, compute psi_k on REAL spikes. Wire the
corrupted-batch recipe as the cheapest reproducible natural class. Run both through the SAME
attribution + repair pipeline; report transfer from the induced sandbox as inference, not causation.
DoD: a psi_k value + attribution row for >=1 real K2 spike appears in the results table.
```

**▶ Task 22 — 124 M on Kaggle within free limits (survival built in).**
```
Implement research/experiments/llm124m/run.py + a Kaggle notebook: train a 124M trunk only to hit
spike windows (NOT convergence), snapshot (w,m,v) bf16 to HF Hub, run SHORT forks (calibrated length)
for the battery. Disconnect survival: checkpoint the FULL optimizer state (not just weights, or
resume diverges) to a rotating latest on HF every ~1000 steps and auto-resume on restart; pin the GPU
type per experiment; append GPU-hours to a persistent ledger and print the weekly Kaggle total vs the
30h cap before launching. Use lobpcg + a small probe batch for curvature (Task 11). DoD: a 124M spike
is reproduced and one 8-branch short-fork battery completes (across sessions if needed), all artifacts
on HF Hub, nothing on the laptop.
```

**▶ Task 23 — analysis & figures.**
```
Implement research/analysis/figures.py from the W&B/HF results: (1) causal map (effect w/m/v, paired
CIs, Br subtracted), (2) psi_k histogram per spike class, (3) theory-predicts-practice scatter
(measured psi_k vs whether selective repair beat global reset), (4) final-loss recovery vs B* across
methods, (5) compute-to-recover. Colorblind-safe, SVG+PNG to HF Hub. DoD:
`python -m research.analysis.figures` regenerates all five from stored results.
```

**▶ Task 24 — workshop paper + reproducibility package.**
```
Create paper/ (NeurIPS-workshop skeleton: intro, C1 protocol, C2 attribution, C3 repair, theory
sketch, metrics, honesty section) where every claim slot points at a Task 23 figure and "causal"
appears ONLY where the Br control earned it. Repro README: exact configs, seeds, spike recipes, HF
artifact ids, one-command replay per table. Make research/harness runnable standalone — the fork
harness IS the artifact. DoD: skeleton renders with figure placeholders wired to real files.
```

**▶ Task 25 ⛔ credits-gated — 410 M mechanism transfer + full benchmark.**
```
(Only with GPU credits — TRC/Lambda/Modal.) Reuse everything at 410M: a 3-seed attribution battery
testing whether the causal map transfers across scale, plus the full N>=10-spikes/class x all-methods
benchmark. Robustness check, not a new contribution. DoD: the 410M causal map is compared to 124M and
the transfer claim is stated with its 3-seed caveat.
```

---

## 3. Free-tier survival tactics
- **Snapshot budget:** proxy `(w,m,v)` fp32 ≈ 12–36 MB; 124 M bf16 ≈ 0.75 GB. Keep 5–10 fork-point
  snapshots on HF (LFS, fine). The real limit is **upload time**, not storage — use a rotating
  `latest.safetensors`, don't version every K steps.
- **Determinism:** trunk OFF (fast), forks ON; all branches of a fork in one session on one GPU
  model; `dropout=0` everywhere on the deterministic path; exact `Δ==0` at proxy/fp32, `Δ<ε` at 124M.
- **Curvature at scale:** `torch.lobpcg` on GPU (not scipy `eigsh` — it OOMs host RAM); one FIXED
  probe batch across all matvecs; small (4–8 seq) HVP batch.
- **Kaggle:** 12 h/session, 30 h/week — run the big sweep at the calibrated SHORT-fork length; print
  a GPU-hour estimate before each launch; tokenized shard via `/kaggle/input` or HF pull.
- **W&B:** scalars only; `wandb.init(group=spike_id, job_type=branch)` keeps hundreds of runs
  navigable; never log tensors/snapshots (those go to HF).

## 4. Milestones & kill criteria (decided now, not in rebuttal)
- **Determinism gate (Task 7):** no-op vs no-op Δ must be 0 at proxy. Nothing proceeds until it is.
- **Method-is-dead test (Task 9, pulled to week one):** if cheap `Bs` (skip+clip) recovers final loss
  everywhere → ship C1+C2 as a science/benchmark paper (NeurIPS D&B), drop repair as the headline.
- **GO/NO-GO #1 (Task 18):** if `Bv` doesn't beat `B0` and the random control `Br` (paired CIs) → pivot.
- **Delocalized-poison fallback:** if selective repair never beats global reset → "the poison is
  delocalized: why global reset wins" — still a real paper.

## 5. Credits-gated extensions (the "full plan" parts)
410 M scale (Task 25), the full multi-scale sweep, and the 5-person division of labor (Workstream A
harness/attribution, B repair/baselines, C theory). None are needed for the arXiv + workshop MVP.

## 6. Week-1 reading (pointer)
In order (teaching packet has the from-scratch version): Adam m/v + why v lags → Hessian eigenpairs →
Edge of Stability (`η·λ<2`) → preconditioned/adaptive EoS (`≈38/η`) → HVP/Pearlmutter → power
iteration/Lanczos → gradient SNR → AR(2)/companion matrix/spectral radius → spike causes & fixes →
small-scale proxies → deterministic replay. Tier-1 papers first: **2506.04805** (onset — cite
prominently, stake recovery against it) and **SPAM 2501.06842** (the puzzle C2 resolves). Then
Karpathy's "reproduce GPT-2 (124 M)" = your 124 M infra tutorial.

## 7. Load-bearing correctness invariants (don't let these regress)
1. Data is a **fixed pre-tokenized memmap**; a batch is a pure function of `step` (no streaming).
2. `CUBLAS_WORKSPACE_CONFIG` is set **before** `import torch`; assert it.
3. Snapshots key optimizer state by **param name** (+ shape asserts); handle **tied weights**;
   restore **per-param `step`**; store m,v in **bf16** (never fp16).
4. `B*` keeps the data stream **aligned** (toggle the spike off / clean the same slot) — never skip.
5. Curvature: **one fixed probe batch** across all matvecs; **lobpcg** (not eigsh) at 124 M.
6. Attribution uses **held-out val loss** and **paired** CIs; `Br` random-subspace control is what
   licenses the word "causal".

---

*Build order is dependency order: env → data → determinism → snapshot → fork gate, THEN the week-one
cheap kill-test, THEN the localizer/repair/theory. No causal word is allowed until the proxy gate
reads Δ==0.*
