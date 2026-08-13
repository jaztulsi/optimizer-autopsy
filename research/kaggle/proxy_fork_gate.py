"""Proxy-scale confirmation of the C1 fork determinism gate (Task 7 DoD).

Runs `fork_determinism_gate` at the REAL proxy tensor shapes (research/experiments/proxy/config.yaml:
n_layer=3, n_head=6, n_embd=192, block_size=256, batch_size=64) so SDPA is exercised in the shape/
dtype range the DoD is about -- unlike the tiny fork._selfcheck (n_embd=64, block=32). Data is
synthetic: the determinism question is shape-dependent, not data-dependent, so no shard download is
needed. Short by design (a handful of steps answers determinism, not convergence).

Run on Kaggle:  python -m research.kaggle.proxy_fork_gate
"""

from __future__ import annotations

import os

# Must precede any torch import; a launcher that already set it wins (setdefault).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import tempfile  # noqa: E402


def main(warmup: int = 10, gate_steps: int = 15) -> None:
    import numpy as np
    import torch

    from research.configs import load_config
    from research.harness.determinism import seed_everything
    from research.harness.fork import fork_determinism_gate
    from research.harness.snapshot import capture, load, save
    from research.harness.trunk import build_model_opt, train_forward

    cfg = load_config("research/experiments/proxy/config.yaml")
    cfg["train"]["max_steps"] = warmup + gate_steps  # short: this answers determinism, not convergence

    seed_everything(cfg.get("seed", 1337), deterministic=True)  # ONCE, before any CUDA op
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
    print("cuda_available:", torch.cuda.is_available())
    mcfg, tcfg = cfg["model"], cfg["train"]
    print(
        f"proxy shapes: n_layer={mcfg['n_layer']} n_head={mcfg['n_head']} n_embd={mcfg['n_embd']} "
        f"block={mcfg['block_size']} batch={tcfg['batch_size']} (warmup={warmup}, gate_steps={gate_steps})"
    )

    with tempfile.TemporaryDirectory() as d:
        # Synthetic shard sized for warmup+gate at proxy batch/block; tokens < vocab (50304).
        block, batch = mcfg["block_size"], tcfg["batch_size"]
        n_tok = batch * block * (warmup + gate_steps + 2) + block + 1
        np.mod(np.arange(n_tok), 50000).astype(np.uint16).tofile(f"{d}/train.bin")

        model, opt = build_model_opt(cfg, device)
        train_forward(model, opt, cfg, d, 0, warmup, device)  # populate (m, v)
        snap = capture(model, opt, step=warmup, meta={"scale": "proxy-fork-gate"})
        path = os.path.join(d, "latest.safetensors")
        save(snap, path)
        snap = load(path)  # exercise the disk round-trip too

        max_d = fork_determinism_gate(cfg, d, snap, steps=gate_steps, device=device, seed=False)
        assert max_d == 0.0
    print(f"PROXY FORK GATE OK on {device}: noop-vs-noop max|Δ|=0 at proxy shapes")


if __name__ == "__main__":
    main()
