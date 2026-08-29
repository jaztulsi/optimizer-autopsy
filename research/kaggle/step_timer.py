"""Measure REAL proxy-scale GPU seconds/step, and turn it into a GPU-hours estimate for the
fork-replay attribution battery. Justification for the Exea Labs AMD compute request.

The unit we time is exactly what a fork branch repeats: one optimizer step end-to-end =
forward + backward + grad-clip + AdamW `opt.step`. We reuse the trunk's own `build_model_opt`
(research/harness/trunk.py) at the real proxy shapes (research/experiments/proxy/config.yaml:
n_layer=3 n_head=6 n_embd=192 block=256 batch=64) -- no invented architecture, no duplicated model.
Data is a synthetic batch of the correct shape/dtype (int64 token ids, values < vocab); step time is
shape/dtype-bound, not data-bound, so no shard download is needed for a throughput number.

Copy this whole file into ONE Kaggle notebook cell (GPU T4/P100) and run. Or: python -m research.kaggle.step_timer
Requires the repo importable as `research.*` (same as the other research/kaggle/* scripts).
"""

from __future__ import annotations

import os

# Must precede any torch import (harness convention); a launcher that already set it wins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ----------------------------------------------------------------------------------------
# KNOBS -- edit these. Timing knobs first, then the GPU-hours scenario.
# ----------------------------------------------------------------------------------------
CONFIG_PATH = "research/experiments/proxy/config.yaml"  # the real proxy config; don't invent one
WARMUP_STEPS = 10  # unmeasured: GPU/cudnn/allocator warmup, first-call overhead
TIMED_STEPS = 100  # measured; report mean/std over these

# GPU-hours scenario for the attribution battery (all four easy to change):
RECIPES = 4  # number of spike recipes
BRANCHES = 7  # branches per site (fork sites); default 7
SEEDS = 3  # seeds per branch
FORK_LENGTH = 200  # fork length in steps


def main(config_path=CONFIG_PATH, warmup=WARMUP_STEPS, timed=TIMED_STEPS) -> None:
    import statistics
    import time

    import torch

    from research.configs import load_config
    from research.harness.determinism import seed_everything
    from research.harness.trunk import _grad_norm, build_model_opt

    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 1337), deterministic=False)  # throughput run, not a replay gate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA -- this number is CPU and not usable for the compute request.")

    model, opt = build_model_opt(cfg, device)
    model.train()
    t, block = cfg["train"], model.cfg.block_size
    batch, grad_clip = t["batch_size"], cfg["optim"].get("grad_clip", 0.0)

    print("GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
    print(f"params (non-embedding): {model.num_params():,}")
    m = cfg["model"]
    print(
        f"proxy step: n_layer={m['n_layer']} n_head={m['n_head']} n_embd={m['n_embd']} "
        f"block={block} batch={batch} grad_clip={grad_clip}  (warmup={warmup}, timed={timed})"
    )

    # Synthetic batch of the exact shape/dtype data.prepare.get_batch returns: (batch, block) int64
    # token ids with values < vocab. Step time is shape/dtype-bound, not data-bound -> no shard needed.
    x = torch.randint(0, 50000, (batch, block), dtype=torch.int64, device=device)
    y = torch.randint(0, 50000, (batch, block), dtype=torch.int64, device=device)

    def one_step():
        # The exact fork-replayed unit, mirroring train_forward's inner body (grad_accum=1 on proxy).
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        _grad_norm(model.parameters(), grad_clip)  # clips in place, same as the trunk
        opt.step()

    for _ in range(warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(timed):
        if device == "cuda":
            torch.cuda.synchronize()  # ensure prior GPU work is done before the clock starts
        t0 = time.perf_counter()
        one_step()
        if device == "cuda":
            torch.cuda.synchronize()  # wait for THIS step's GPU work before stopping the clock
        times.append(time.perf_counter() - t0)

    mean = statistics.mean(times)
    std = statistics.stdev(times)
    print("\n=== measured ===")
    print(f"mean : {mean:.6f} s/step")
    print(f"std  : {std:.6f} s/step")
    print(f"rate : {1.0 / mean:.2f} steps/sec")

    # ----- GPU-hours calculator -----
    total_steps = RECIPES * BRANCHES * SEEDS * FORK_LENGTH
    gpu_hours = total_steps * mean / 3600.0
    print("\n=== GPU-hours estimate (edit RECIPES/BRANCHES/SEEDS/FORK_LENGTH at top) ===")
    print(f"recipes={RECIPES}  branches/site={BRANCHES}  seeds/branch={SEEDS}  fork_length={FORK_LENGTH} steps")
    print(f"total fork steps : {total_steps:,}")
    print(f"GPU-hours        : {gpu_hours:.2f}  (= {total_steps:,} x {mean:.6f}s / 3600)")


if __name__ == "__main__":
    main()
