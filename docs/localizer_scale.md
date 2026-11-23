# Localizer at larger scale: which eigensolver (Week 8 sketch)

Status: a proposal for the team to decide, not a decision. Owner: Avyank (C2).

## The problem

`research/localizer/curvature.py` solves the top-k eigenpairs of the preconditioned Hessian
H~ = D^(-1/2) H D^(-1/2) with `scipy.sparse.linalg.eigsh` (ARPACK Lanczos). ARPACK keeps its Lanczos
basis in host RAM in float64: dim × ncv × 8 bytes.

| Model | dim | k = 5 (ncv = 20) |
|---|---|---|
| proxy | 11,035,968 | 1.77 GB, which fits |
| GPT-2 124M | about 124M | about 19.9 GB, more than a T4 box's host RAM |

`estimate_mem` refuses the solve over budget instead of running out of memory an hour in.
context-ai §4.5 planned `torch.lobpcg` for 124M, but that cannot work: it needs the matrix as an explicit tensor
and does not accept a matvec (see `hvp_project/WEEKS.md`).

## What changed: the hardware

The plan for this was written for a 16 GB T4. The project now runs on the AMD MI300X, which has 192 GB of
GPU memory. At 124M a k = 5 basis is about 9.9 GB in fp32 or 19.9 GB in fp64, so it fits **on the GPU**
next to the model and the double-backward graph.

## Options, best first

1. **No eigensolve at all: the no-HVP localizer.** `localize_spike(..., use_eig=False)` uses only the drift of
   v from v_bar per group. It costs nothing at any scale and already runs. Use it at 124M until option 2 is
   built, and as the control that the curvature path must beat (PLAN_V8 §6.4, item 7).
2. **A GPU-native Lanczos on the existing matvec.** Write a block Lanczos (or LOBPCG) that keeps its basis on
   the GPU, with full reorthogonalization, on top of `HTilde.matvec`. The operator, the symmetry guard and the
   Ritz-residual check all stay the same. `localizer/stability.py` (Week 4) and `dense_reference` are the
   tests it must pass before anyone trusts it: the same top-k as eigsh on the proxy, to the same residuals.
3. **Parameter-subset localization.** Solve H~ restricted to one block at a time (for example one
   transformer layer) with the existing eigsh. This is context-ai §4.5's own fallback. It is cheap to build,
   but it misses directions that span blocks, and the answer then depends on how the blocks were chosen.

## Recommendation

Use option 1 now at every scale. Build option 2 only if the proxy results (Weeks 6–7) show that the curvature
path separates spikes from normal steps better than option 1 does (`localize_report.json`, the AUC of
`max_score` against `max_drift_ratio`). If it does not, the eigensolve is not worth its GPU time at 124M.
Keep option 3 as the fallback if option 2 cannot pass the stability tests.

## Open questions

- PLAN_V8 §11.2 puts localizers after prereg-v3 and marks them "only if ahead". GPU time for any of this
  comes out of the 20 MI300X-hour budget, and Jasraj prices it.
- Does the MI300X ROCm build run double backward through the MATH attention backend at 124M without
  running out of memory at probe batch 4? This is untested.
