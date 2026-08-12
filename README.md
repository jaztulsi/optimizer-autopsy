# 🔬 Optimizer Autopsy

**Rewind a neural-net training run to the instant its loss blew up, find exactly which slice of optimizer state got poisoned, repair only that slice — and prove the repair is what fixed it.**

[![site](https://img.shields.io/badge/site-github.io-2b7bb9)](https://jaztulsi.github.io/optimizer-autopsy/)
[![compute](https://img.shields.io/badge/compute-Kaggle%20T4%20(free)-20beff)](#free-stack)
[![status](https://img.shields.io/badge/status-%7E23%25%20built%20%C2%B7%20all%20green-brightgreen)](update.md)
[![replay](https://img.shields.io/badge/perfect%20replay-GPU%20verified%20max%7CΔ%7C%3D0-success)](#why-perfect-replay-is-the-whole-game)

---

## The problem

Large models sometimes **spike**: the loss suddenly jumps, the run gets damaged, and today the only
fixes are blunt — clip the gradient, skip the batch, or roll back and pray. Nobody localizes *what
actually broke inside the optimizer*. This project treats a spike like a crime scene:

1. **Rewind** — return training to the exact step it broke.
2. **Localize** — score every parameter's optimizer state (`SNR + curvature`) to find the poison.
3. **Repair** — edit *only* the localized state, then **fork** the run and compare against a
   bit-identical replay to prove the edit is what recovered it.

## Why perfect replay is the whole game

The science is only trustworthy if "run it again" gives **byte-for-byte identical** numbers —
otherwise you can't tell a real repair from RNG noise. That determinism is built and verified:

> `bitwise replay OK on cuda: 50 steps, max|Δ| = 0` — on both CPU **and** a real Kaggle T4.

Every intervention (ours and the baselines) is applied as a **fork** off a shared snapshot, so the
only difference between two runs is the edit under test. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set
before torch imports; `harness.determinism` locks seeds + deterministic kernels.

## Repo layout

Research code lives under [`research/`](research/) (see [`research/README.md`](research/README.md)).
The root `index.html` is the [GitHub Pages site](https://jaztulsi.github.io/optimizer-autopsy/).

| Package | Role |
|---|---|
| `harness/` | determinism · preflight · secrets · `(w,m,v)` snapshots · trunk + fork runners |
| `model/` | nanoGPT proxy (1–3M) and 124M |
| `data/` | tokenize + shard the corpus, reproducibly |
| `localizer/` | **the instrument** — SNR + curvature → poison score |
| `repair/` | **the surgery** — edit the localized optimizer state |
| `baselines/` | skip · clip · SPAM · ZClip · AdaGC · naive-reset, each as a fork intervention |
| `spikes/` | induce spikes, tune the detector, build the K2 spike set |
| `analysis/` | eval · stats · attribution · figures |
| `experiments/` | `proxy/` (free MVP) and `llm124m/` (robustness) configs + drivers |

## Free stack

Everything runs on **free tiers, no local GPU** — code here, compute in the cloud, artifacts on the Hub.

| Piece | Where |
|---|---|
| Code | this repo |
| Compute | **Kaggle** T4 (30 h/wk, headless via the Kaggle CLI) · Colab for proxy iteration |
| Checkpoints / snapshots / shards | Hugging Face Hub (`optimizer-autopsy-artifacts`) |
| Loss curves / logs | Weights & Biases |

### Secrets — never committed

Resolved at runtime by `research/harness/secrets.py` from **env → Kaggle Secrets → Colab userdata →
`.env`**, in that order. `.env` is gitignored and a test (`test_no_secrets_in_git`) fails the build if
a token ever lands in the tracked tree. Needed: `HF_TOKEN`, `WANDB_API_KEY`.

## Run

Open a notebook launcher (`notebooks/kaggle_runner.ipynb` / `notebooks/colab_runner.ipynb`) — it sets
the determinism env var, clones the repo, loads secrets, and gives you `run(...)`:

```python
run("python -m research.experiments.proxy.smoke")        # free MVP keep/kill signal
run("python -m research.experiments.llm124m.run trunk")  # 124M robustness (budgeted)
```

Deps: `pip install -r research/requirements.txt` (torch 2.10, numpy 2.0.2 — pinned to the Kaggle image).

## Status

**~23% built · everything passing.** Foundation + safety gate done; the training engine trains on a
real T4 (loss **10.85 → 4.73** in 200 steps). Next: snapshot/restore, then the fork driver + `Δ==0`
gate. Live detail in [`update.md`](update.md); full plan in [`BUILD_PLAN.md`](BUILD_PLAN.md).

```
Foundation   ████████████████████  done
Safety gate  ████████████████████  perfect replay, GPU-verified
The engine   ████████████░░░░░░░░  trunk done · snapshot/fork next
Everything else  ░░░░░░░░░░░░░░░░  not started
```

## License & credits

Built by **jaztulsi**. See [`CONTRIBUTORS.md`](CONTRIBUTORS.md).
