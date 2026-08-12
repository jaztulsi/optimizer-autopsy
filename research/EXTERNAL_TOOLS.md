# External tools & prior art — reuse map

Repos/libraries mapped to the pipeline tasks, so we **wrap proven code instead of rebuilding**.

**Adoption legend**
- **WRAP** — permissive license, import/adapt directly; replaces code we'd otherwise hand-write.
- **REFERENCE** — read/cross-check against, don't depend on.
- **REIMPLEMENT** — algorithm we must write ourselves (no usable license, or wrong framework), using the repo only to validate behavior.
- **RELATED** — cite in the paper's related work, not code to adopt.
- **DATA** — a source of checkpoints/curves.

**Two hard project rules that override "just wrap it":**
1. **Determinism gate.** Anything touching the training step (every baseline, any optimizer wrapper) must pass `research/tests/test_determinism.py` (`max|Δ| = 0` on a bit-identical replay) *before it counts*. External code often introduces nondeterministic kernels or its own RNG — verify, don't assume.
2. **License.** We are MIT. `pip`-depending on a permissive package is fine; **copying/adapting source requires an MIT/BSD/Apache-2.0 license.** No-license repos are all-rights-reserved — reimplement from the paper instead.

---

## Task 11 — `localizer/curvature.py` (Hessian/GGN spectrum → poison score)

| Tool | Adopt | License | Note |
|---|---|---|---|
| **f-dangel/curvlinops** | **WRAP** | MIT | `pip install curvlinops-for-pytorch`. scipy `LinearOperator` wrappers for Hessian/GGN/Fisher matvecs → straight into `eigsh`/`lobpcg`. Replaces the HVP glue we planned to hand-roll. **Top pick.** |
| noahgolmant/pytorch-hessian-eigenthings | REFERENCE | — | Lanczos/power-iteration top-k via matvec — cross-check our eigenvalues. |
| amirgholami/PyHessian | REFERENCE | — | Top eigenvalues, Hutchinson trace, full ESD — validate spectral density beyond top-k. |
| amirgholami/HessianFlow | REFERENCE | — | Lighter Hessian-through-training logging reference. |

## Task 15 — `baselines/*.py` (fork interventions to benchmark against)

| Tool | Adopt | License | Note |
|---|---|---|---|
| **bluorion-com/ZClip** | **WRAP** | Apache-2.0 | Official EMA z-score adaptive clipping. Lightning-based — use its **direct-PyTorch `.step(model)`** path, not the callback. → `baselines/zclip.py`. |
| TianjinYellow/SPAM-Optimizer | **REIMPLEMENT** | **none (all-rights-reserved)** | Official but **no LICENSE file** — do NOT copy. SPAM = momentum reset + spike-aware clipping; small enough to clean-room from the paper. Repo = behavior reference only. → `baselines/spam.py`. |
| PaddlePaddle AdaGC | **REIMPLEMENT** | Apache-2.0 but PaddlePaddle | Wrong framework; port the per-tensor adaptive-clip rule to torch. → `baselines/adagc.py`. |
| zoq/Awesome-Optimizer | REFERENCE | — | Index if we need a 6th/7th baseline beyond skip/clip/SPAM/ZClip/AdaGC/reset. |

## Task 19 — `theory/` (spike onset, preconditioned EoS, AR(2)/spectral-radius)

| Tool | Adopt | Note |
|---|---|---|
| locuslab/edge-of-stability | REFERENCE | Cohen et al. ICLR'21 sharpness-tracking scripts to adapt for our eigenvalue-vs-step plots. |
| alex-damian/EOS | REFERENCE | Self-stabilization / "frozen Adam" analysis underlying the Adaptive-EoS result we cite. |
| centralflows (Central Flow) | REFERENCE | Continuous-time Adam-near-EoS flow — sanity-check baseline for the AR(2)/spectral-radius theory. |
| "Adaptive Preconditioners Trigger Loss Spikes in Adam" (arXiv 2506.04805) | RELATED | Already cited; check for a linked repo when writing the theory section. |

## Task 21 — `spikes/k2.py` (natural spike test set)

| Tool | Adopt | Note |
|---|---|---|
| LLM360/k2-train, k2v2_train, Analysis360 | **DATA (curves only)** | Source of *documented* real spike locations + loss curves. **Reality:** K2-65B optimizer state is far beyond free-tier download/compute — use their published spike steps/curves as ground truth, not the full `(w,m,v)`. Our real runs stay proxy + 124M. |

## Task 5 / 22 — `model/nanogpt.py`, `experiments/llm124m/`

| Tool | Adopt | Note |
|---|---|---|
| karpathy/nanoGPT | REFERENCE | Diff our `model/nanogpt.py` against upstream for drift. |
| karpathy/build-nanogpt | REFERENCE | Commit-by-commit build-up — bisect a subtle attention/init bug against known-good. |
| KellerJordan/modded-nanogpt | REFERENCE | Throughput tricks (fused ops, muP scaling) to shrink the 124M GPU-hour budget under Kaggle's 30h/wk cap. |
| karpathy/llm.c | REFERENCE | C/CUDA fallback if Python overhead ever bottlenecks 124M. |

## Task 6 / 7 — harness spine (snapshot, determinism)

| Tool | Adopt | Note |
|---|---|---|
| PyTorch reproducibility docs | REFERENCE | Canonical `use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` / cuDNN checklist — diff `determinism.py` against it on each torch-pin bump. |
| safetensors tied-weight issues | REFERENCE | Check `save_model` tied-embedding dedup gotchas *before* building `snapshot.py`. |
| huggingface_hub `upload_large_folder` | **WRAP** | Current best-practice for large LFS pushes — use for the rotating-`latest.safetensors` uploader (Task 6). |
| W&B **Artifacts** (not scalar logging) | WRAP (optional) | Version snapshot→fork lineage without touching HF, if we want lineage tracking. |

## Task 17 — `analysis/stats.py`

| Tool | Adopt | License | Note |
|---|---|---|---|
| **scipy.stats.bootstrap** | **WRAP** | BSD | Paired BCa bootstrap CI built in — scipy is **already pinned**. Use directly for `bootstrap_ci`. |
| **statsmodels.multipletests** | **WRAP** | BSD | Off-the-shelf Holm/FDR multiple-comparison correction for the method matrix. |
| arch (Sheppard) | REFERENCE | BSD | Stationary/circular block bootstrap if paired seeds show autocorrelation across forks. |

## Ops (§10, free-tier survival)

| Tool | Adopt | Note |
|---|---|---|
| kaggle/kaggle-api | **WRAP** | Already wired — our headless compute path. |
| huggingface_hub multi-commit/large-folder | WRAP | See Task 6 above. |

---

## Related work (cite, don't adopt)

Adam-mini · APOLLO · GaLore — all "selective / rank-limited intervention on optimizer state." Same flavor as our C3 repair operator but for a *different reason* (memory vs. poison). One related-work sentence each contrasting motive.

## Highest-leverage first

1. **curvlinops** (Task 11) — deletes the most hand-written glue, MIT, clean.
2. **scipy.stats.bootstrap + statsmodels** (Task 17) — already-pinned/BSD, zero-risk wins.
3. **ZClip** (Task 15) — one of three baselines becomes a wrap; SPAM/AdaGC stay reimplements for license/framework reasons.
4. **edge-of-stability / EOS / centralflows** (Task 19) — working reference code beats deriving the theory cold.

_Verify exact repo paths + licenses at adoption time; this list was curated from a broad scan and a few names may need confirmation._
