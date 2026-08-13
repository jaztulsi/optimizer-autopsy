"""Evidence run: turn the Task 4 / Task 5 claims into committable artifacts.

Until now the determinism primitive (`max|Δ|=0`, Task 4) and the trunk-trains claim
(`loss 10.85 -> 4.73`, Task 5) existed only as self-reported numbers in `update.md`/`todo.md`,
with no run log or results file in the repo. This module regenerates both on a real GPU and writes:

  results/task4_determinism.json   per-step max|Δ| over the bitwise-replay gate (all 0.0)
  results/task5_trunk.json         per-step loss for a 200-step proxy trunk (batch 16, T4)

Run on Kaggle:  `python -m research.kaggle.evidence_run`  (see the evidence kernel launcher).
Output dir defaults to ./results; override with EVIDENCE_OUT.
"""

from __future__ import annotations

import os

# Must precede any torch import; a launcher that already set it wins (setdefault).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _dump(out_dir: str, name: str, payload: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(payload, f, indent=2)
    brief = {k: v for k, v in payload.items() if k != "per_step"}
    print(f"wrote {os.path.join(out_dir, name)}: {json.dumps(brief)}")


def task4_determinism(out_dir: str, steps: int = 50) -> dict:
    """Run the bitwise-replay gate and record its per-step max|Δ| (asserts each == 0)."""
    from research.tests.test_determinism import test_bitwise_replay

    diffs = test_bitwise_replay(steps)
    payload = {
        "claim": "bitwise replay max|Delta|=0 over N steps (Task 4 primitive)",
        "device": _device(),
        "steps": steps,
        "max_over_run": max(diffs) if diffs else None,
        "per_step": [{"step": i, "max_abs_delta": d} for i, d in enumerate(diffs)],
    }
    _dump(out_dir, "task4_determinism.json", payload)
    return payload


def task5_trunk(out_dir: str, steps: int = 200, batch_size: int = 16) -> dict:
    """Train the proxy trunk `steps` and record per-step loss (the doc claim uses batch 16)."""
    from research.configs import load_config
    from research.data.prepare import PRESETS, prepare, resolve_data_dir
    from research.harness.trunk import run_trunk

    try:
        data_dir = resolve_data_dir("proxy")  # shard already staged (e.g. /kaggle/input)
    except FileNotFoundError:
        data_dir = prepare(PRESETS["proxy"])  # prepare on the PINNED revision

    cfg = load_config("research/experiments/proxy/config.yaml", {"train.batch_size": batch_size})
    losses = run_trunk(cfg, data_dir, steps=steps)  # trunk default: deterministic=False
    payload = {
        "claim": "trunk loss decreases on real GPU (Task 5)",
        "device": _device(),
        "steps": steps,
        "batch_size": batch_size,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "per_step": [{"step": i, "loss": loss} for i, loss in enumerate(losses)],
    }
    _dump(out_dir, "task5_trunk.json", payload)
    return payload


def main() -> None:
    out_dir = os.environ.get("EVIDENCE_OUT", "results")
    t4 = task4_determinism(out_dir)
    t5 = task5_trunk(out_dir)
    print(
        f"EVIDENCE DONE on {t4['device']}: task4 max|Delta|={t4['max_over_run']}, "
        f"task5 loss {t5['loss_first']:.3f} -> {t5['loss_last']:.3f}"
    )


if __name__ == "__main__":
    main()
