"""Deterministic training setup: seeds, deterministic algorithms, and RNG capture/restore.

This is the load-bearing module: every causal claim in the project rests on a fork being a
BIT-IDENTICAL replay of the trunk. `CUBLAS_WORKSPACE_CONFIG` must be set BEFORE torch is imported
(the notebook launchers do this); `seed_everything` handles everything settable after import and
HARD-ASSERTS the two footguns — the env var, and CUDA not being initialized yet.

Convention: all snapshot/fork runs use dropout=0, which removes a whole class of RNG divergence.
"""

from __future__ import annotations

import os
import random

import numpy as np

_VALID_CUBLAS = (":4096:8", ":16:8")


def _assert_preconditions() -> None:
    """The two things that silently break determinism if wrong."""
    import torch

    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas not in _VALID_CUBLAS:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG={cublas!r}; must be one of {_VALID_CUBLAS} and set BEFORE "
            "`import torch` (in the launcher's first cell). cuBLAS reads it once at init."
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA is already initialized; call seed_everything() before any CUDA op, or "
            "use_deterministic_algorithms cannot cover work already scheduled."
        )


def seed_everything(seed: int = 1337, deterministic: bool = True) -> None:
    """Seed python/numpy/torch (CPU+CUDA) and, if `deterministic`, lock the algorithms.

    Trunk runs may pass deterministic=False for speed; forks MUST use the default (True). When
    deterministic, preconditions are asserted first, then CUDA may initialize during seeding.
    """
    import torch

    if deterministic:
        _assert_preconditions()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:
    """Snapshot every RNG we touch: python, numpy, torch CPU, torch CUDA (all devices)."""
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Restore what capture_rng_state() saved, so the next draws replay identically."""
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
