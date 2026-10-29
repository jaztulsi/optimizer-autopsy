# OPTIMIZER AUTOPSY — research code

Fork training at the moment a loss spike hits, measure who wrote the lasting damage (the spike step or Adam's
memory of it), and undo it.
Plan: [PLAN_V8.md](../PLAN_V8.md) · state: [docs/SUMMARY.md](../docs/SUMMARY.md) · site: <https://jaztulsi.github.io/optimizer-autopsy/> · reuse map: [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md)

All code lives under `research/`. The repo-root `index.html` is the GitHub Pages site, so leave it alone.

## Layout and status (2026-10-29)

| Package | Role | Status |
|---|---|---|
| `harness/` | **C1**, the instrument: determinism, preflight, secrets, name-keyed `(w,m,v)` snapshots, trunk and fork runs, the Δ = 0 gate | ✅ Δ = 0 on T4, P100 and the AMD MI300X |
| `autopsy/` | P1 code: RML undo (m / v / m+v, top-k v, cooldown), planted probes, 1-ulp placebo floor, dose meter, the GPU driver `mve`, the sealed report | ◐ CI-tested; the core battery ran in the 2026-10-27 pilot; placebo, dose and cooldown arms not yet GPU-run |
| `spikes/` | Spike recipes, detector, capture driver (v̄ + snapshots), natural-spike search, K2 manifest | ◐ 3 recipes ran on the MI300X; capture self-check fix awaits a GPU re-run |
| `baselines/` | skip, clip, resets, v̄, SPAM, ZClip, AdaGC, skip-step, as fork interventions | ◐ CI-tested; never GPU-run; `Brollback` in PR #20 |
| `analysis/` | held-out eval (token-disjoint slots), stats, attribution, figures | ◐ eval done; stats partial; figures stub |
| `localizer/` | **C2**: which state coordinates carry the damage (SNR, curvature, poison score) | ⬜ stubs (output contract: PLAN_V8 §6.4) |
| `repair/` | **C3**: edit the localized optimizer state | ⬜ stub; RML in `autopsy/` covers part of it |
| `theory/` | closed-form predictions (ψ_k, AR(2), echo / stall laws) | ⬜ README only |
| `data/` | tokenize + shard the corpus (uint16 memmap; a batch is a pure function of the step) | ✅ |
| `model/` | nanoGPT: proxy (≈ 11M non-embedding params) and 124M | ✅ proxy trains on the MI300X |
| `experiments/` | `proxy/` and `llm124m/` configs; their drivers (`smoke.py`, `run.py`) are TODO stubs; use `autopsy.mve` | ◐ |
| `kaggle/` | `step_timer`, the cross-process gate `xproc_gate.py`, legacy CUDA scripts (Kaggle is paused) | ✅ ran on the MI300X |
| `tests/` | CPU tests, run in GitHub CI only | ✅ |

## Compute

- **GPU:** the AMD MI300X only, and only with the user's go-ahead. **Kaggle is paused, Colab is unused, and
  nothing runs on personal machines** (not even tests).
- **Tests and lint:** GitHub CI.
- **Artifacts** (checkpoints, snapshots, shards, spike sets): the Hugging Face Hub.
- **Logs:** wandb.

## Secrets

Never committed. `harness.secrets` resolves them from env, Kaggle Secrets, Colab userdata or `.env`:
- `HF_TOKEN`: Hugging Face Hub read/write.
- `WANDB_API_KEY`: wandb logging.

## Determinism (read this first)

`CUBLAS_WORKSPACE_CONFIG=:4096:8` **must be set before torch is imported**: on the MI300X, pass it with
`docker exec -e` (see `results/mi300x_2026-10-27/t1_pilot/run.sh`). `harness.determinism` handles the rest (seeds,
deterministic algorithms). Forks require bit-identical replay. Don't edit `harness/determinism.py` or
`harness/preflight.py` without flagging it first (`CLAUDE.md` §1).

## Run (MI300X)

```
python -m research.kaggle.step_timer
python -m research.autopsy.mve --config results/mi300x_2026-10-27/t1_pilot/config.yaml \
    --kind corrupt_batch --inject-step 160 --seed 0 --out results/autopsy           # one P1 instance
python -m research.autopsy.mve ... --placebo 8 --plant-coefs 1,3,10,30          # prereg-v2 arms (after the tag)
python -m research.autopsy.mve ... --battery repair --lag 3                     # C-4 repair battery
```

Install: `pip install -r research/requirements.txt`. Dev tools go in the root `requirements-dev.txt`, never here.
