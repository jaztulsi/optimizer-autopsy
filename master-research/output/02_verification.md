# 02 · Verification of RESEARCH.md and improvement.md claims

Verifier pass, run 2026-10-22. Primary sources only (arXiv abs/HTML, official repos/issues, PyTorch source at the named release branch, vendor docs, icml.cc).
No code was executed. Quotes are ≤25 words.

Verdicts: **PASS** = primary source confirms · **FAIL** = primary source contradicts (correct value given) · **UNVERIFIED** = no primary source reached or claim not checkable.
`PASS*` = confirmed only through a search engine's copy of the primary page, because a direct fetch timed out.
Notes in *italics* flag version drift or nuance that doesn't change the verdict.

---

## 1 · research/spikes/RESEARCH.md

| # | Claim | Doc · line | Verdict | Evidence | Correct value / note |
|---|---|---|---|---|---|
| R1 | Bai et al., *Adaptive Preconditioners Trigger Loss Spikes in Adam*, ICML 2026 | L26, L223 | PASS | arXiv 2506.04805 comments: "Accepted to ICML 2026"; ICML poster https://icml.cc/virtual/2026/poster/63930; PMLR v306 `bai26o` | — |
| R2 | Root cause: `v_t` decouples from `g²`, pushing λ_max of preconditioned Hessian past 2/η | L27-29 | PASS | https://arxiv.org/abs/2506.04805: "the adaptive preconditioner v_t fails to track the instantaneous squared gradients g_t^2"; "beyond the stability threshold 2/η" | — |
| R3 | Five stages: stable descent → preconditioner decay → onset → preconditioner growth → recovery | L29-30 | PASS | https://arxiv.org/html/2506.04805v2: stages "Stable Loss Decrease, Preconditioner Decay, Spike Onset, Preconditioner Growth, Loss Decrease" | *The paper calls stage 5 "Loss Decrease", not "recovery".* |
| R4 | Larger β2 makes spikes likelier | L30 | PASS | arXiv HTML v2: decoupling "is exacerbated by larger β2"; reducing β2 mitigates spikes | — |
| R5 | Predictor `λ_grad(Ĥ) = gᵀĤg/‖g‖² > 2/η`; flags onset, not advance warning | L31-32 | PASS | arXiv HTML v2 defines λgrad as ∇Lᵀ H ∇L/‖∇L‖² and §6.2 calls crossing 2/η "signaling the spike onset" | *The abstract says "anticipating spikes". The body supports the onset reading.* |
| R6 | Cohen et al.: adaptive EoS threshold `(2+2β1)/((1−β1)η)` = 38/η at β1=0.9 | L35-36 | PASS | https://ar5iv.labs.arxiv.org/html/2207.14484 §4: "This threshold is (2+2β1)/((1−β1)η) = 38/η for our choice of β1=0.9" | — |
| R7 | `P = diag(√v̂+ε)` | L35 | PASS | Same source: frozen Adam has "P = diag(ν^1/2) + εI" | *With bias correction, Eq. 2 also multiplies P by (1−β1^(t+1)). That's minor.* |
| R8 | Bai folds `(1−β1)/(1+β1)` into Ĥ, giving the same condition λ_max(Ĥ) < 2/η | L36-37 | PASS | arXiv HTML v2 Eq. 7: Ĥ_t = [1/(1−β1^t)]·[(1−β1)/(1+β1)]·D_t·H_t. Algebra: 38/η × (1−β1)/(1+β1) = 2/η | — |
| R9 | `D^-1/2 H D^-1/2` has the same spectrum as `P⁻¹H` (similar matrices) | L37-38 | PASS | Bai footnote 1 factors D^{1/2}(D^{1/2}HD^{1/2})D^{-1/2}: "is similar to a symmetric matrix" | — |
| R10 | Wortsman 2023: "14 of 15 loss spikes followed an RMS spike in patch-embedding (RMS > 2.3) by 1–8 iterations" | L41-43 | PASS | https://arxiv.org/html/2304.13013v2 App. D / Fig. 16 (ViT-Huge): "14 out of 15 come 1-8 iterations after an RMS spike in the patch embedding layer"; threshold "{t : RMS_t ≥ 2.3}" | **Version question settled:** v2 contains *both* counts. Main text §3.4: "28 out of 30 detected loss spikes follow an RMS spike by 1-8 iterations". App. D: ViT-Huge 14/15 and ViT-Large 13/15 (14+13=27, so 28/30 is not their sum). v1 text could not be machine-read (PDF only), so v1 is UNVERIFIED. Cite "App. D, ViT-Huge" for 14/15 or "§3.4" for 28/30. The layer is `visual.conv1.weight`. |
| R11 | `RMS_t = √E[g²/v̂]`; second moment is "out of date" | L42-43 | PASS | ar5iv 2304.13013: RMS_t = √E[g_t²/u_t]; "stuck-in-the-past" scenario | — |
| R12 | Molybog 2023: large-batch regime, updates large and uncorrelated with descent direction | L43-45 | PASS | https://arxiv.org/abs/2304.09871: update vector with large norm, nearly uncorrelated with descent direction; more likely with deep models and large batches | — |
| R13 | Wortsman, *Small-scale proxies…* (ICLR 2024): attention-logit growth and output-logit divergence at small scale; qk-layernorm and z-loss fix them | L47-49 | PASS | https://arxiv.org/html/2309.14322 Fig. 2: qk-layernorm "is equally effective in the small-scale regime"; §3.1.2: "z-loss resolves this instability". ICLR 2024 oral: https://iclr.cc/virtual/2024/poster/18273 | — |
| R14 | OLMo 2 traced spikes to repeated n-grams, e.g. `g4ODg4OD…` | L52 | PASS | https://arxiv.org/html/2501.00656 §3.1: example "g4ODg4ODg4ODg4ODg4ODg4ODg4OD..." | — |
| R15 | OLMo 2 filters documents with ≥32 repeated n-grams | L52-53 | PASS | Same §3.1: "removes all documents with a sequence of 32 or more repeated n-grams" | *An n-gram here is a span of 1–13 tokens.* |
| R16 | LLM360 K2 released paired spike/normal checkpoints K2-Spike-1/2 | L53-54, L233 | PASS | https://huggingface.co/LLM360/K2-Spike-1 card: "For every spike checkpoint, we also uploaded the corresponding normal checkpoint for easy comparison" | *Both repos now resolve under the **IFM** org (IFM/K2-Spike-1, IFM/K2-Spike-2). The cards don't mention optimizer state, which matches RESEARCH.md §2.3's "weights only".* |
| R17 | K2-V2: most spikes in the first 40% of training | L54 | PASS | https://arxiv.org/html/2512.06201 §3.5.3: "the vast majority of loss spikes occur during the first 40% of training" | *The paper is now at v3 (2026-09-17).* |
| R18 | K2-V2 two-tier alert/restart monitor | L54-55 | PASS | Same, §3.5.2: "The restart tier uses a wider window and stricter thresholds to confirm sustained divergence" | — |
| R19 | K2-V2 publishes W&B logs (wandb.ai/llm360/K2-V2) | L55, L234 | PASS | The paper's front-matter "Training Log" link is https://wandb.ai/llm360/K2-V2 | *The URL now 301-redirects to forge.coreweave.com/wandb/llm360/K2-V2, which did not render for me. Check public access by hand before planning §2.3 on it.* |
| R20 | SPAM: periodic global m/v reset + spike-aware clipping `g² > θv` | L57 | PASS | https://arxiv.org/html/2501.06842: G = {g_i : g_i²/V_i > θ}; "By resetting the momentum terms at regular intervals of ΔT" | — |
| R21 | SPAM θ = 5000 | L119 | PASS | Same, §4: "GSS threshold θ=5000"; Table 8 uses 5000 at all sizes | *Appendix I gives the good range as 1000–5000.* |
| R22 | ZClip = z-score EMA clipping of the global norm | L58 | PASS | https://github.com/bluorion-com/ZClip README: "tracking running statistics of gradient norms through Exponential Moving Averages (EMA)"; Apache-2.0 | — |
| R23 | AdaGC = per-tensor EMA clipping | L58 | PASS | https://arxiv.org/abs/2502.11034: "an adaptive per-tensor gradient clipping scheme… tensor-wise exponential moving average" | — |
| R24 | AdaGC code at github.com/lishuai-97/AdaGC | L230 | PASS | The repo exists: "PyTorch implementation of AdaGC… built on Megatron-LM", Apache-2.0 | *The paper's own link (PaddlePaddle/PaddleFleet) now returns 404. It's unconfirmed that the owner is co-author Shuai Li.* |
| R25 | *Mechanism-Driven Monitors for Preemptive Detection of LLM Training Instability*, arXiv 2606.28116 | L61-62, L231 | PASS (v1) | https://arxiv.org/abs/2606.28116v1 has that title | **Title drift:** v3 (2026-10-18) is retitled *"Mechanism-Aligned Observability for Training Instability of Large Language Models"*, with a rewritten abstract. Cite as v1 or update. |
| R26 | …spectral signals of the attention update, lead times of thousands of steps on low-precision faults | L62-63 | PASS (v1) | v1 abstract: "spectral entropy of a QK bilinear decomposition"; "triggering thousands of steps before loss divergence". v1 HTML: Δ2 collapse "preceded loss divergence by approximately 17,000 steps" | *The v3 abstract no longer gives step counts.* |
| R27 | …no false-positive rates, no repair or counterfactual analysis | L63-64 | PASS (v1) | v1 HTML: no false-alarm measurement and no mitigation is evaluated. Only healthy-vs-faulted control runs | — |
| R28 | `v` e-folding `1/(−ln β2)`: ≈19.5 (0.95) vs ≈1000 (0.999), ~50× | L82-83 | PASS (math) | 1/0.05129 = 19.50 and 1/0.0010005 = 999.5. The ratio is 51.3 | — |
| R29 | Update RMS bounded by `1/√(1−β2)` since `v_t ≥ (1−β2)g_t²` | L118 | PASS (math) | v̂ = v/(1−β2^t) ≥ (1−β2)g²/(1−β2^t) ≥ (1−β2)g² ⇒ g²/v̂ ≤ 1/(1−β2) | — |
| R30 | Conformal: τ = M_(⌈(n+1)(1−f)⌉); n=20, f=0.05 ⇒ τ = max, guarantee ≤ 1/21 ≈ 0.048 | L130-134 | PASS (math) | ⌈21×0.95⌉ = ⌈19.95⌉ = 20 = n. By exchangeability, P(M_new > M_(20)) ≤ 1/21 = 0.0476 | *Needs the alarm defined as s_t > τ (strict), tie-free scores, and identical run horizons, so the M_j are exchangeable. With f=0.05, n < 19 gives an index > n, so no finite τ exists.* |
| R31 | CUSUM is Lorden-optimal for mean shifts | L136 | PASS | https://arxiv.org/html/2203.03532 §simulations: oracle CUSUM is "known to be minimax optimal, in the sense of minimizing the worst case delay" (cites Moustakides 1986) | *This holds for known pre/post-change distributions.* |
| R32 | E-detectors (Shin, Ramdas & Rinaldo) are nonparametric with ARL ≥ 1/α | L136-137 | PASS | arXiv 2203.03532 Thm 2.4: alarm at e-detector ≥ 1/α gives ARL ≥ 1/α; abstract: "nonparametrically specified (and thus composite)" | — |
| R33 | …several signals combine through **products** or averages of e-values without losing validity | L137-138 | **FAIL (partial)** | Prop. 2.3: "an average (or 'a mixture') of e-detectors is also an e-detector"; §6.3: "minimum or scaled summation (average) of e-detectors also forms a valid e-detector" | **Correct:** averages/mixtures and the minimum are valid. The paper never claims products are. Products of e-detectors built on the same, dependent telemetry are not generally valid. Drop "products". |
| R34 | `k½ = ln(κ/3)/(−ln β2)` ≈ 158 at κ=1e4, β2=0.95 | L149 | PASS (math) | Solve β2^k κ + 1 − β2^k = 4 ⇒ k = ln((κ−1)/3)/(−ln β2) = 8.1114/0.051293 = 158.1 | — |
| R35 | `Bg` overshoot ≈ `1/√(1−β2)` ≈ 4.5× | L153 | PASS (math) | 1/√0.05 = 4.47 | — |
| R36 | SDPA flash kernels lack double-backward | L127 | PASS | https://discuss.pytorch.org/t/…/200233: "derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented" | — |
| R37 | Source list: Huang (SPAM), Kumar (ZClip), Wang (AdaGC), Thilak (Slingshot), Shin/Ramdas/Rinaldo, Molybog first authors and titles | L223-236 | PASS | arXiv abs pages 2501.06842, 2504.02507, 2502.11034, 2206.04817, 2203.03532, 2304.09871 match | — |
| R38 | "No work localizes poisoned directions inside Adam's state + repairs + causal forks" | L66-70 | UNVERIFIED | This is a negative claim and can't be confirmed from sources | *It's consistent with every paper read here, and the doc already flags it as "a targeted search".* |

---

## 2 · improvement.md — §1 A6 ROCm checklist (◐ items) and A6 row

| # | Claim | Doc · line | Verdict | Evidence | Correct value / note |
|---|---|---|---|---|---|
| I1 | On ROCm, `use_deterministic_algorithms` only turns off rocBLAS atomics | L42 | PASS | torch release/2.10 `CublasHandlePool.cpp`, `getCurrentCUDABlasHandle()` ROCm branch: "if (at::globalContext().deterministicAlgorithms()) { hipblas_mode = HIPBLAS_ATOMICS_NOT_ALLOWED;" | *This is the BLAS scope only. The flag still governs non-BLAS ops elsewhere.* |
| I2 | It doesn't reach hipBLASLt, MI300X's default matmul library | L42 | PASS | Same file: the ROCm `getCurrentCUDABlasLtHandle()` only creates a handle and has no determinism code. `Context.cpp` (2.10) `blasPreferredBackend()`: allowlist "gfx90a", "gfx942" ⇒ `BlasBackend::Cublaslt` (hipBLASLt) | *If devices are mixed, the default falls back to rocBLAS.* |
| I3 | On ROCm, `CUBLAS_WORKSPACE_CONFIG` only sets a workspace size | L42 | PASS | `parseChosenWorkspaceSize()` ROCm branch reads CUBLAS_/HIPBLAS_/ROCBLAS_WORKSPACE_CONFIG; "/* 32MiB default, 128MiB for gfx94x/gfx95x */" | — |
| I4 | torch 2.10 AOTriton attention backward accepts `deterministic` and ignores it | L42, L245 | PASS | release/2.10 `mha_all_aot.hip`: in `mha_bwd_aot` / `mha_varlen_bwd_aot`, `deterministic` appears only in the signature | — |
| I5 | Set `ROCBLAS_DEFAULT_ATOMICS_MODE=0` | L49 | PASS | https://rocm.docs.amd.com/projects/rocBLAS/en/develop/reference/env-variables.html: 0 ⇒ default `rocblas_atomics_not_allowed` | — |
| I6 | Set `ROCBLAS_USE_HIPBLASLT=0` | L49 | PASS | Same page: at 0, "Tensile is always used as the GEMM backend" | *This controls rocBLAS's internal backend only. It does not stop PyTorch from calling hipBLASLt directly; I7 does that.* |
| I7 | `preferred_blas_library("cublas")` selects rocBLAS on ROCm | L50 | PASS | `CublasHandlePool.cpp` ROCm shim: "rocblas_set_workspace((rocblas_handle)handle, addr, size)". `BlasBackend::Cublas` is the hipBLAS/rocBLAS handle path | — |
| I8 | Pin `HIPBLASLT_WORKSPACE_SIZE` | L51 | PASS | `parseCUDABlasLtWorkspaceSize()` ROCm: reads "HIPBLASLT_WORKSPACE_SIZE"; default "76 MB for hipBLASLt" | *This is a PyTorch env var. It isn't on the hipBLASLt env-var page.* |
| I9 | hipBLASLt can pick a different fallback kernel per process for untuned shapes | L52 | PASS | https://github.com/ROCm/rocm-libraries/issues/13476 (open, 2026-10-12): "Default heuristic runs a heap-layout-dependent kernel for untuned shapes, so outputs differ between runs" | *The fix (#11340) is on `develop` and not yet in a release.* |
| I10 | "It wasn't reproduced on MI300 with default settings" | L52 | UNVERIFIED | The issue was reproduced on a Radeon AI PRO R9700 (gfx1201) only. It has no comments and doesn't mention MI300 | Reword as "only reported on gfx1201; untested on MI300". |
| I11 | Never set `TORCH_ROCM_FA_PREFER_CK=1`; the CK backward hard-codes non-deterministic | L55 | PASS (partial scope) | The env var exists (`Context.cpp` comment: "env vars such as TORCH_ROCM_FA_PREFER_CK"). release/2.10 `attention_backward.cu` CK mem-efficient path calls `mem_eff_backward_ck` with "false, // deterministic" | *The **flash** CK backward (`flash_api.h` → `mha_bwd_ck`) forwards `deterministic` instead of hard-coding it. The hard-code is in the **mem-efficient** CK path. The advice stands.* |
| I12 | Autotuning off: TunableOp, hipBLASLt tuning, `TORCH_COMPILE_DISABLE=1` | L56 | PASS (partial) | AMD Primus determinism guide: `PRIMUS_DETERMINISTIC=1` sets "TORCH_COMPILE_DISABLE=1", "ROCBLAS_DEFAULT_ATOMICS_MODE=0", and disables hipBLASLt tuning | *The guide doesn't mention TunableOp. That part is a sensible recommendation, not a sourced claim.* |
| I13 | TF32/XF32 off | L58 | UNVERIFIED | This is a recommendation and no source was checked | — |
| I14 | `TENSILE_SOLUTION_SELECTION_METHOD=0` (no Stream-K) | L59 | PASS | https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html: "0: Default (standard tuned libraries, no Stream-K)"; 0 is the default | *It has "no effect on the AMD Instinct MI350 series. Stream-K is always used." On MI300X, 0 is already the default.* |
| I15 | MI300X SPX = 304 CUs, query with `amd-smi` | L60 | PASS | https://instinct.docs.amd.com/…/mi300x/overview.html: "8 XCDs per GPU → 304 total CUs"; mode adjustable "using command-line tools like amd-smi" | *The page lists modes SPX, DPX and CPX.* |
| I16 | GPU random-number streams depend on CU count | L60 | PASS | release/2.10 `DistributionTemplates.h` `calc_execution_policy`: grid uses `multiProcessorCount`, and "counter_offset = ((numel - 1) / (block_size * grid.x * unroll_factor) + 1)" | *Partition mode changes the CU count, so it changes philox offsets.* |
| I17 | Embedding-backward race pytorch #196164, fires only for 2033–3072 indices | L61 | PASS (for MI300X) | https://github.com/pytorch/pytorch/issues/196164 maintainer root cause: "1009..3072 indices on wave32 ROCm GPUs, 2033..3072 on wave64 ROCm GPUs and 2017..3072 on CUDA" | *MI300X (CDNA3) is wave64, so 2033–3072 applies. On CUDA (current T4/P100 gates) the window is **2017**–3072. Use the widest window (1009–3072) in the preflight assert to be safe.* |
| I18 | …fix not yet released (PR #199569) | L61, L249 | PASS | https://github.com/pytorch/pytorch/pull/199569: "Fixes #196164", merged to `main` 2026-10-05 (commit e511672). The PR page names no release | *A third-party Megatron-LM commit says it's "not in releases through 2.15". Not independently confirmed.* |
| I19 | AOTriton 0.12b has a deterministic backward | L42, L54, L248 | PASS | https://github.com/ROCm/aotriton/releases 0.12b: "attn_options::deterministic = true forces the deterministic split-kernel backward path" | — |
| I20 | Moving the pin to torch ≥ 2.13 gets it | L42, L54 | PASS | release/2.13 `cmake/External/aotriton.cmake`: `set(__AOTRITON_VER "0.12b")`. release/2.13 `mha_all_aot.hip`: "options.deterministic = deterministic;" in both backward functions | — |
| I21 | CUDA snapshots won't replay bit-for-bit on ROCm | L62 | UNVERIFIED | No primary source; this is an engineering expectation | *Almost certainly true (different kernels and reduction orders). Treat it as a rule, not a citation.* |

---

## 3 · improvement.md — §2.2 AMD bring-up

| # | Claim | Doc · line | Verdict | Evidence | Correct value / note |
|---|---|---|---|---|---|
| I22 | `rocm/pytorch-training` image exists | L105-106 | PASS | Docker Hub API: "Unified PyTorch base container optimized for training with ROCm", updated 2026-01-28 | **Note:** Docker Hub says this repo "will be deprecated in favor of `rocm/primus`". Choose the pin with that in mind. |
| I23 | Diagnose with `HIPBLASLT_LOG_LEVEL`, `AOTRITON_DEBUG_LEVEL` | L108 | PASS | hipBLASLt env-var page: "Controls the verbosity level of hipBLASLt logging output". AOTriton README: "Set AOTRITON_DEBUG_LEVEL=<n> to enable runtime logging" (1–5) | — |
| I24 | AMD Developer Cloud VM is destroyed when credits run out | L110 | PASS* | AMD FAQ text, retrieved via search index (direct amd.com fetches timed out ×4): "your credit expires and you have no active payment method on file, your AMD GPU VM will be destroyed" | *The trigger is credit expiry with no payment method on file. AMD pages disagree on expiry: 10 days or 30 days from deposit (the newer pages say 30). DigitalOcean's AMD credit docs don't cover deletion. Re-confirm on amd.com before relying on it.* |

---

## 4 · improvement.md — §2.3 prior art (no ◐ marks, checked anyway)

| # | Claim | Doc · line | Verdict | Evidence | Correct value / note |
|---|---|---|---|---|---|
| I25 | Bai: Ĥ left-preconditioned with `(1−β1)/(1+β1)`, not symmetric | L119 | PASS | Bai Eq. 7: Ĥ_t = …·D_t·H_t (left-multiplied); footnote 1 notes similarity to a symmetric matrix | — |
| I26 | Bai Transformer predictor takes min over t−1…t+1, so it uses a future step | L120-121 | PASS | arXiv HTML v2: the "sustained spike predictor" takes the minimum of λgrad over t−1, t, t+1 | — |
| I27 | Bai largest run 187M | L121 | PASS | arXiv HTML v2: "187M parameter LLaMA-structured transformer" on 100B SlimPajama tokens | — |
| I28 | Bai "no code" | L121 | **FAIL** | arXiv HTML v2 App. B: "Code and additional implementation details are made available in the supplementary materials." | **Correct:** code was in the conference supplementary material. No public repo link was found. Say "no public repository". |
| I29 | Bai localization only block-level; no repair | L122 | UNVERIFIED | Not checked in the full text | — |
| I30 | Liu & Li (2608.19762): paired control/shock runs with shared later data; small-perturbation regime | L123-124 | PASS | https://arxiv.org/abs/2608.19762: "paired trajectories that differ only in one gradient update and share the same subsequent training sequence"; "Linearizing the joint dynamics" | — |
| I31 | Liu & Li code injects shock into whole channels {θ, m, v}, never directions | L124-125 | UNVERIFIED | https://github.com/Kanyooo/Loss_ISO holds only `spike_anonymous.zip`, which wasn't opened (no code or unzip allowed) | — |
| I32 | Liu & Li code has no license | L125 | PASS | The repo file list is only `spike_anonymous.zip`, with no LICENSE file | — |
| I33 | Ramasinghe (2606.16768): tracks `P^-1/2 H P^-1/2` with warm-started power iteration, ~5 HVPs/step, up to 3B, one global eigenvalue, no m/v edits | L126-128 | PASS | https://arxiv.org/html/2606.16768: "We use five power iterations with warm-start for online curvature tracking"; "models scaling up to 3-billion parameters"; "estimate the top eigenpair"; preconditioner applied on both sides | *Title: "Taming Curvature: Architecture Warm-Up for Stable Transformer Training".* |
| I34 | Spectra (2602.11185) is ICML 2026 | L129 | PASS | https://icml.cc/virtual/2026/poster/61015; PMLR v306 `huang26d` | — |
| I35 | Spectra: top 1.5% directions; ~97% of v's spectral energy in that subspace | L129-131 | PASS | https://arxiv.org/html/2602.11185 §2.3: "the spike subspace already explains about 97% of the spectral energy in V"; "roughly the top 1.5% directions" | *For M (momentum) it's about 50%.* |
| I36 | Spectra is "a Muon-like optimizer that flattens" the top directions | L129-130 | **FAIL (wording)** | Same: Muon is described as "full spectrum-flattening"; Spectra "only shrinks the spike singular values toward the tail's average scale" | **Correct:** a spike-subspace *suppression* optimizer, explicitly contrasted with Muon. It shrinks the top singular values toward the tail scale and leaves the tail unchanged. |
| I37 | Marin 32B: clip 1.0→0.2, update-norm clipping, skip-step, Muon swap only softened spikes; QK-Norm removed them; rooted in attention stack | L133-135 | PASS (nuance) | https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/: tightened "max grad norm to 0.2"; clipping and skipping "softened the spikes but never removed them"; with QK-Norm "the loss spikes disappeared entirely"; "rooted in the model's attention stack rather than in optimizer state" | *The Muon swap gave temporary relief, then turned "into gradient ascents" and was dropped. Update-norm clipping "may have reduced their severity". The attention cause is inferred, not isolated.* |
| I38 | GI-Adam (Kalra & Barkeshli, NeurIPS 2024) = v initialized with g² | L139 | PASS | https://arxiv.org/abs/2406.09405 "Accepted at NeurIPS 2024"; ar5iv: "Gradient Initialized Adam (GI-Adam), which initializes the second moment using the gradient squared" | *The "MIT" license is UNVERIFIED: no code link was found in the paper.* |
| I39 | Adam-Rel (Ellis et al., NeurIPS 2024) resets only t; the m+v+t reset is Asadi's "Adam-MR" | L142-143 | PASS | https://arxiv.org/html/2412.17113v1: "retaining momentum estimates and only resetting t, which we refer to as Adam-Rel"; "We refer to this method as Adam-MR." NeurIPS 2024 poster 94469 | *The name "Adam-MR" is Ellis et al.'s. Asadi's paper doesn't use it.* |
| I40 | Asadi (NeurIPS 2023): second-moment-only reset "quite effective"; method also resets step counter; β2 = 0.999 | L144-146 | PASS | ar5iv 2306.17833 Fig. 5: "only resetting the second-order moment is quite effective even when we do not reset the first-order moment"; Alg. 1 resets "i=0, m⁰=0, v⁰=0"; "β1=0.9 and β2=0.999" | — |

---

## 5 · improvement.md — §2.5 benchmark release

| # | Claim | Doc · line | Verdict | Evidence | Correct value / note |
|---|---|---|---|---|---|
| I41 | No public dataset of spike snapshots that includes Adam moments | L172 | UNVERIFIED | This is a negative claim | *A near-counterexample: OPEN-1B's public bucket appears to hold fp32 optimizer state every 100 steps around two natural grad-norm events (I44). It isn't a curated spike dataset, but weaken the wording to "no curated…".* |
| I42 | Croissant metadata required by the NeurIPS E&D track | L174 | PASS | https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting: "we require submissions with dataset contributions to provide a Croissant file" | *Preferred hosts: Harvard Dataverse, Kaggle, Hugging Face, OpenML (they auto-generate Croissant; RAI fields are added by hand).* |
| I43 | Hugging Face Hub file-count limits | L174 | PASS | https://huggingface.co/docs/hub/en/storage-limits: files per repo "<100k"; "The maximum number of files per folder cannot exceed 10k"; file "<200GB" recommended, 500GB hard | *There's no per-repo size limit. Free public storage is "best-effort" beyond a few GB.* |
| I44 | OPEN-1B (2609.17380): Apache-2.0 weights, 1.6B | L175 | PASS | arXiv HTML: "1.61B total parameters". HF API `Gensyn/open-1b-base`: "license":"apache-2.0" | *The paper itself is CC BY 4.0.* |
| I45 | …checkpoints every 100 steps with "full optimizer state" | L175-176 | PASS (substance) | arXiv HTML: "each intermediate checkpoint at intervals of 100 steps"; "Optimizer state is checkpointed and hashed in FP32; thus, an audit replay can reconstruct the moments exactly." | *The phrase "full optimizer state" isn't in the paper. In the bucket, `ckpt/step_000001700/dcp/` has 8 shards of 2.414 GB ≈ 19.3 GB ≈ 1.61B × 12 B, consistent with fp32 w+m+v. That's an inference from sizes; tensor names weren't read.* |
| I46 | …at `gs://gensyn-open-1b/ckpt` | (prompt) | PASS | https://github.com/gensyn-ai/open-transformers README: "gs://gensyn-open-1b/ckpt", "Checkpoints (810, all 7 run segments)". The GCS JSON listing is public | — |
| I47 | …bracketing grad-norm events: ckpt 1700 → step 1782, ckpt 43700 → step 43737 | L176-177 | PASS | arXiv HTML: pre-clip norms "5.78 at step 1,782 and 6.88 at step 43,737". Bucket prefixes `ckpt/step_000001700/`, `…001800/`, `…043700/` exist | **Caveat for use as natural spikes:** OPEN-1B "skips five optimizer steps when the pre-clip gradient norm exceeds 5", and `meta.json` shows AdaGC clipping (`adagc_*`, betas [0.9, 0.95]). These events were skipped and clipped, not left to spike. |

---

## 6 · improvement.md C9 (the Wortsman count dispute)

| # | Claim | Doc · line | Verdict | Evidence | Correct value |
|---|---|---|---|---|---|
| I48 | "`RESEARCH.md:42` … cite Wortsman 2023 as '14 of 15' spikes; arXiv v2 says 28 of 30" | L87 | **FAIL** | arXiv 2304.13013v2 contains both: §3.4 "28 out of 30 detected loss spikes follow an RMS spike by 1-8 iterations" and App. D/Fig. 16 (ViT-Huge) "14 out of 15…" | This isn't a v1-vs-v2 conflict. Both counts are in v2, in different sections (main-text aggregate vs the ViT-Huge appendix). RESEARCH.md's 14/15 with RMS > 2.3 is correct if it cites App. D. |

---

## 7 · ICML 2027

| # | Item | Verdict | Evidence |
|---|---|---|---|
| C1 | ICML 2027 announced? | **Not announced** | https://icml.cc/Conferences/2027 and /2027/Dates return 404. https://icml.cc/Conferences/FutureMeetings says only "2027 -- South America" (no city, no dates) |
| C2 | ICML 2027 deadlines | UNVERIFIED | No official CFP exists. A third-party tracker (opencurious.com) *estimates* abstract 2027-01-16 and paper 2027-01-22 AoE. That's an estimate, not official |
| C3 | ICML 2026 deadlines (reference) | PASS | https://icml.cc/Conferences/2026/Dates: abstract "Jan 23 '26", full paper "Jan 28 '26" (AoE); author response Mar 30; notification Apr 30 |
| C4 | ICML 2026 location/dates | PASS | CFP: "held in Seoul, South Korea, July 6-11". Main conference Jul 7–9, tutorials Jul 6, workshops Jul 10–11 |
| C5 | Page limit | PASS | CFP: "8 pages for the main paper, with unlimited pages for references, the impact statement, and appendices". Camera-ready gets +1 page (9) |
| C6 | Supplementary/code | PASS | https://icml.cc/Conferences/2026/AuthorInstructions: "The supplementary code can be submitted as either a zip file or a pdf"; anonymous GitHub allowed (branch frozen after deadline); "must also be anonymized"; appendices go in the main PDF; no separate supplementary deadline; "There is NO camera-ready supplementary material of any kind" |
| C7 | LLM policy | PASS | CFP: "Authors are allowed to use generative AI tools such as Large Language Models (LLMs) to assist in writing or research". LLMs can't be authors; prompt injection means desk rejection. Reviewing (https://icml.cc/Conferences/2026/LLM-Policy): Policy A bans LLMs, Policy B allows them for comprehension and polish. Authors may require Policy A for their paper |
| C8 | Reciprocal reviewing | PASS | CFP: "All submissions must have at least one author who agrees to serve as a reviewer for ICML". Each author can be the reciprocal reviewer for ≤2 submissions, and authors with ≥4 submissions have an extra requirement. Negligent reviewers "may have their own submissions desk-rejected" |
| C9 | Preprints | PASS | CFP: "Authors are allowed to post versions of their work on preprint servers such as arXiv" (don't advertise it as an ICML submission) |

---

## Sources (all fetched 2026-10-22)
- https://arxiv.org/abs/2506.04805 · https://arxiv.org/html/2506.04805v2 · https://icml.cc/virtual/2026/poster/63930
- https://ar5iv.labs.arxiv.org/html/2207.14484
- https://arxiv.org/html/2304.13013v2 · https://ar5iv.labs.arxiv.org/html/2304.13013
- https://arxiv.org/html/2309.14322 · https://iclr.cc/virtual/2024/poster/18273
- https://arxiv.org/abs/2304.09871 · https://arxiv.org/html/2501.06842 · https://arxiv.org/abs/2504.02507 · https://github.com/bluorion-com/ZClip
- https://arxiv.org/abs/2502.11034 · https://github.com/lishuai-97/AdaGC · https://github.com/PaddlePaddle/PaddleFleet (404)
- https://arxiv.org/abs/2606.28116v1 · https://arxiv.org/html/2606.28116v1 · https://arxiv.org/abs/2606.28116 (v3)
- https://arxiv.org/html/2501.00656
- https://arxiv.org/abs/2501.07124 · https://huggingface.co/LLM360/K2-Spike-1 · https://huggingface.co/LLM360/K2-Spike-2
- https://arxiv.org/html/2512.06201 · https://wandb.ai/llm360/K2-V2 (301 → forge.coreweave.com)
- https://arxiv.org/html/2203.03532 · https://arxiv.org/abs/2206.04817
- https://discuss.pytorch.org/t/runtimeerror-derivative-for-aten-scaled-dot-product-is-not-implemented/200233
- PyTorch source: release/2.10 `aten/src/ATen/cuda/CublasHandlePool.cpp`, `aten/src/ATen/Context.cpp`, `…/hip/flash_attn/aot/mha_all_aot.hip`, `…/hip/flash_attn/flash_api.h`, `…/hip/flash_attn/ck/mha_bwd_ck.hip`, `…/transformers/cuda/attention_backward.cu`, `aten/src/ATen/native/cuda/DistributionTemplates.h`; release/2.13 `mha_all_aot.hip`, `cmake/External/aotriton.cmake` (via raw.githubusercontent.com)
- https://rocm.docs.amd.com/projects/rocBLAS/en/develop/reference/env-variables.html · https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html
- https://github.com/ROCm/rocm-libraries/issues/13476 · https://github.com/ROCm/aotriton/releases · https://raw.githubusercontent.com/ROCm/aotriton/main/README.md
- https://github.com/pytorch/pytorch/issues/196164 · https://github.com/pytorch/pytorch/pull/199569
- https://instinct.docs.amd.com/projects/amdgpu-docs/en/develop/gpu-partitioning/mi300x/overview.html
- https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/determinism-and-reproducibility.html
- https://hub.docker.com/v2/repositories/rocm/pytorch-training/
- AMD Developer Cloud FAQ (search-index copy; amd.com timed out) · https://docs.digitalocean.com/products/amd/details/credits/
- https://arxiv.org/abs/2608.19762 · https://github.com/Kanyooo/Loss_ISO
- https://arxiv.org/html/2606.16768 · https://arxiv.org/html/2602.11185 · https://icml.cc/virtual/2026/poster/61015 · https://github.com/kimmichtank/spectra
- https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/ · https://github.com/marin-community/marin/issues/1368
- https://ar5iv.labs.arxiv.org/html/2306.17833 · https://ar5iv.labs.arxiv.org/html/2406.09405 · https://arxiv.org/html/2412.17113v1
- https://arxiv.org/html/2609.17380 · https://github.com/gensyn-ai/open-transformers · https://huggingface.co/api/models/Gensyn/open-1b-base · https://storage.googleapis.com/storage/v1/b/gensyn-open-1b/o?prefix=ckpt/ · https://storage.googleapis.com/gensyn-open-1b/ckpt/step_000001700/meta.json
- https://huggingface.co/docs/hub/en/storage-limits · https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting
- https://icml.cc/Conferences/FutureMeetings · https://icml.cc/Conferences/2026/CallForPapers · https://icml.cc/Conferences/2026/Dates · https://icml.cc/Conferences/2026/AuthorInstructions · https://icml.cc/Conferences/2026/LLM-Policy
