# OPTIMIZER AUTOPSY — research code

Fork training at the moment of failure, find where the poison lives, repair it.
Plan: [BUILD_PLAN.md](../BUILD_PLAN.md) · site: <https://jaztulsi.github.io/optimizer-autopsy/> · reuse map: [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md)

All code lives under `research/`. The repo-root `index.html` is the GitHub Pages site — leave it alone.

## Layout

| Package | Role |
|---|---|
| `harness/` | determinism, preflight, secrets, `(w,m,v)` snapshots, trunk + fork runs |
| `data/` | tokenize + shard the corpus |
| `model/` | nanoGPT (proxy 1-3M and 124M) |
| `localizer/` | **C1** the instrument: SNR + curvature -> poison score |
| `repair/` | **C2** the surgery: edit localized optimizer state |
| `baselines/` | skip / clip / SPAM / ZClip / AdaGC / naive reset, as fork interventions |
| `spikes/` | induce spikes, tune the detector, build the K2 spike set |
| `analysis/` | eval, stats, attribution, figures |
| `experiments/` | `proxy/` (free MVP) and `llm124m/` (robustness) configs + drivers |

## Free stack

- **Code** → this repo. **Compute** → Kaggle (30 h/wk GPU, 12 h/session, 20 GB `/kaggle/working`)
  for real runs; Colab for proxy iteration.
- **Artifacts** (checkpoints, snapshots, shards, spike sets) → Hugging Face Hub.
- **Logs** → wandb.

## Secrets

Never committed. Resolved by `harness.secrets` from env / Kaggle Secrets / Colab userdata / `.env`:
- `HF_TOKEN` — Hugging Face Hub read/write.
- `WANDB_API_KEY` — wandb logging.

## Determinism (read this first)

`CUBLAS_WORKSPACE_CONFIG=:4096:8` **must be set before torch is imported** — the notebook launchers
(`notebooks/kaggle_runner.ipynb`, `notebooks/colab_runner.ipynb`) do this. `harness.determinism`
handles the rest (seeds, deterministic algorithms). Forks require bit-identical replay.

## Run

Kaggle/Colab: open the matching notebook launcher, which sets the env var, clones this repo, loads
secrets, and exposes `run(cmd)`. Then:

```
run("python -m research.experiments.proxy.smoke")          # free MVP keep/kill signal
run("python -m research.experiments.llm124m.run trunk")    # 124M robustness (budgeted)
```

Install: `pip install -r research/requirements.txt`.
