"""Curvature evidence run: turn the Task 11 DoD claims into committable artifacts.

Same shape and same reason as research.kaggle.evidence_run (which did Tasks 4/5/7/8): until a DoD
has a results file in the repo it is a self-reported number, and CLAUDE.md §1 says a task is not
done without a passing check. This module runs the Task 11 / Week 2-3 checks on a real box and
writes:

  results/task11_curvature_cpu.json    cheap checks -- no GPU, no shard, seconds
  results/task11_curvature_proxy.json  real scale -- needs the proxy shard, uses the GPU

TWO TIERS, deliberately separate. The CPU tier spends no GPU quota and needs no data, so run it
first and every time; if it fails, the proxy tier cannot tell you anything useful. The proxy tier
trains a few trunk steps to populate (m, v), so it spends GPU-hours -- CLAUDE.md §5 makes that a
review gate.

Run on Kaggle (CPU session is enough for the first):
    python -m research.kaggle.curvature_run --cpu
    python -m research.kaggle.curvature_run            # both tiers
Output dir defaults to ./results; override with EVIDENCE_OUT.
"""

from __future__ import annotations

import os

# Must precede any torch import; a launcher that already set it wins (setdefault).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _dump(out_dir: str, name: str, payload: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


def task11_cpu(out_dir: str) -> dict:
    """The three cheap DoD checks plus the v_bar properties. No GPU, no shard.

    Every one of these asserts internally, so a regression raises here rather than being written
    out as a passing-looking JSON.
    """
    from research.localizer.curvature import (
        _selfcheck_asymmetry_is_caught,
        _selfcheck_precond,
        _selfcheck_quadratic,
    )
    from research.localizer.poison import _selfcheck_vbar

    payload = {
        "task": 11,
        "tier": "cpu",
        "device": _device(),
        "quadratic": _selfcheck_quadratic(),
        "preconditioner": _selfcheck_precond(),
        "asymmetry_guard": _selfcheck_asymmetry_is_caught(),
        "vbar": _selfcheck_vbar(),
    }
    _dump(out_dir, "task11_curvature_cpu.json", payload)
    return payload


def task11_proxy(out_dir: str, steps: int = 3, probe: int = 4, k: int = 3) -> dict:
    """The real-scale tier: FD sweep + top-k of H~ on the model build_model_opt builds.

    Prepares the proxy shard if it is not already staged, the same way evidence_run does.
    """
    from research.localizer.curvature import _selfcheck_proxy

    payload = {"task": 11, "tier": "proxy", **_selfcheck_proxy(steps=steps, probe=probe, k=k)}
    _dump(out_dir, "task11_curvature_proxy.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 11 curvature DoD evidence run.")
    ap.add_argument("--cpu", action="store_true", help="cheap tier only; no GPU, no shard")
    ap.add_argument("--proxy-only", action="store_true", help="skip the cheap tier (not advised)")
    ap.add_argument("--steps", type=int, default=3, help="trunk steps before probing, to fill m/v")
    ap.add_argument("--probe", type=int, default=4, help="probe batch size (sequences)")
    ap.add_argument("-k", type=int, default=3, help="how many eigenpairs")
    args = ap.parse_args()

    out_dir = os.environ.get("EVIDENCE_OUT", "results")
    cpu = None if args.proxy_only else task11_cpu(out_dir)
    if args.cpu:
        q = cpu["quadratic"]
        print(
            f"\nCURVATURE CPU TIER DONE on {cpu['device']}: "
            f"quadratic top-{q['k']} rel={q['eigval_rel_err']:.2e} (DoD {q['tolerance']:.0e}), "
            f"sandwich rel={cpu['preconditioner']['sandwich_vs_dinv_a_rel_err']:.2e}, "
            f"asymmetry guard fired={cpu['asymmetry_guard']['guard_fired']}, "
            f"vbar debias err={cpu['vbar']['debias_max_abs_err']:.2e}"
        )
        return

    px = task11_proxy(out_dir, steps=args.steps, probe=args.probe, k=args.k)
    b = px["fd"]["best"]
    print(
        f"\nCURVATURE DONE on {px['device']}: dim={px['dim']:,}, "
        f"FD valley h={b['h']:.1e} rel_l2={b['rel_l2']:.3e} cos={b['cos']:.9f}, "
        f"top-{px['eigen']['k']} lambda_0={px['eigen']['eigvals'][0]:+.4e} "
        f"(residual {px['eigen']['rel_residuals'][0]:.2e}), "
        f"bit-reproducible={px['hvp_bit_reproducible']}"
    )


if __name__ == "__main__":
    main()
