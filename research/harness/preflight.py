"""Preflight: verify the environment matches pinned expectations before a real run.

Catches the "works on Colab, silently different on Kaggle" class of bug, and the determinism
footguns: `CUBLAS_WORKSPACE_CONFIG` unset, or CUDA already initialized before deterministic
algorithms were configured. Call `check_env()` at the top of every entrypoint.
"""

from __future__ import annotations

import importlib.metadata as md
import os
from pathlib import Path

# The subset whose exact version affects reproducibility. Pins come from requirements.txt so there
# is one source of truth (pyyaml/tqdm are pinned there too but don't move numerics, so we skip them).
_CHECKED = ("torch", "numpy", "scipy", "datasets", "safetensors", "huggingface_hub", "tiktoken", "wandb")
_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"
_VALID_CUBLAS = (":4096:8", ":16:8")


def _pinned() -> dict[str, str]:
    """Parse `pkg==ver` lines from requirements.txt into {normalized-name: version}."""
    pins: dict[str, str] = {}
    for line in _REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[name.strip().lower().replace("_", "-")] = ver.strip()
    return pins


def check_env() -> None:
    """Assert the environment is the pinned, determinism-safe one. Raise a clear message otherwise.

    Checks, in order (each raises with an actionable message):
      1. installed versions of the reproducibility-critical packages == requirements.txt pins.
      2. CUBLAS_WORKSPACE_CONFIG is a value cuBLAS accepts for deterministic GEMMs.
      3. CUDA is NOT yet initialized (so deterministic algorithms still take effect).
    Then prints the GPU name (which itself initializes CUDA — hence it comes last).
    """
    pins = _pinned()
    mismatches = []
    for pkg in _CHECKED:
        want = pins.get(pkg.replace("_", "-"))
        if want is None:
            mismatches.append(f"{pkg}: no pin in requirements.txt")
            continue
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            mismatches.append(f"{pkg}: NOT INSTALLED (pinned {want})")
            continue
        # Strip the local build tag (e.g. torch "2.10.0+cu128") so a CPU box and a CUDA box both
        # pass against the same pin; the public version is what governs numerics.
        if have.split("+", 1)[0] != want:
            mismatches.append(f"{pkg}: installed {have} != pinned {want}")
    if mismatches:
        raise RuntimeError(
            "environment does not match research/requirements.txt:\n  "
            + "\n  ".join(mismatches)
            + "\nInstall the pins, or update requirements.txt if this is a new Kaggle image."
        )

    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas not in _VALID_CUBLAS:
        raise RuntimeError(
            f"CUBLAS_WORKSPACE_CONFIG={cublas!r}; must be one of {_VALID_CUBLAS} and set BEFORE torch "
            "is imported (do it in the notebook launcher's first cell — see notebooks/)."
        )

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA is already initialized; call check_env()/enable_deterministic() before any CUDA op, "
            "or torch.use_deterministic_algorithms will not cover work already scheduled."
        )
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name())  # initializes CUDA — must be last.
    else:
        print("GPU: none (CPU-only)")


if __name__ == "__main__":
    check_env()
    print("check_env OK")
