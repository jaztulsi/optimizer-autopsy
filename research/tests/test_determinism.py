"""The determinism gate: two no-intervention runs from the same snapshot must be BITWISE identical.

If this fails at fp32 on a given device, nothing causal downstream is trustworthy — a fork's Δ
would be measuring RNG/nondeterminism, not the intervention. DoD: passes on CPU and on Kaggle GPU.

Run:  `pytest research/tests/test_determinism.py`  or  `python -m research.tests.test_determinism`.
"""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere in the process; setdefault keeps a launcher's value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy  # noqa: E402

from research.harness.determinism import (  # noqa: E402
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


def _tiny_model(device):
    import torch

    # No dropout (deterministic-path convention). fp32, same device for both runs.
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16)).to(device).float()


def _run_steps(model, opt, x, y, steps):
    """Run `steps` of (forward, backward, opt.step) and return a per-step weight fingerprint.

    Catches+prints the offending op if use_deterministic_algorithms rejects it, then re-raises.
    """
    import torch

    loss_fn = torch.nn.MSELoss()
    traj = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        try:
            opt.step()
        except RuntimeError as e:
            print(f"[determinism] non-deterministic op at step {i}: {e}")
            raise
        # Fingerprint = clone of every parameter, flattened. Bitwise comparison, no tolerance.
        traj.append(torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()]))
    return traj


def test_bitwise_replay(steps: int = 50) -> list[float]:
    """Assert bit-identical replay and return the per-step max|Δ| (all 0.0) for logging."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(1337)
    model = _tiny_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 16, device=device)
    y = torch.randn(8, 16, device=device)

    # Snapshot BEFORE stepping: weights, optimizer state, and all RNG.
    snap = {
        "model": copy.deepcopy(model.state_dict()),
        "opt": copy.deepcopy(opt.state_dict()),
        "rng": capture_rng_state(),
    }

    traj_a = _run_steps(model, opt, x, y, steps)

    # Restore the exact snapshot and replay with NO intervention.
    model.load_state_dict(snap["model"])
    opt.load_state_dict(snap["opt"])
    restore_rng_state(snap["rng"])
    traj_b = _run_steps(model, opt, x, y, steps)

    diffs = []
    for i, (a, b) in enumerate(zip(traj_a, traj_b)):
        max_abs = (a - b).abs().max().item()
        assert max_abs == 0.0, f"step {i} diverged on {device}: max|Δ|={max_abs} (must be 0)"
        diffs.append(max_abs)
    print(f"bitwise replay OK on {device}: {steps} steps, max|Δ|=0 at every step")
    return diffs


if __name__ == "__main__":
    test_bitwise_replay()
