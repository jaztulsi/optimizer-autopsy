"""GPU bitwise checks for the rollback baseline's harness pieces (PR #20). GPU-only: run on the MI300X.

    python -m research.kaggle.baseline_gpu_checks --out DIR

Two checks, both on the real proxy model at grad_accum = 40 (the 124M config's accumulation, where a
single ctx["batch"] override would be wrong), each must be BITWISE (max|Δ| = 0, losses equal):

  1. data_offset: training steps s.. with ctx["data_offset"] = k  ==  training steps s+k.. outright.
  2. Brollback rebuild: the Brollback branch (real lr_bump spike, detector at inject + L) == an independent
     rebuild (restore the fork snapshot, no spike, train on the next unseen batches). The branch is also
     run twice, which must match itself (in-process replay of a reused branch).

Synthetic random tokens: the question is numerics, not learning. Writes DIR/baseline_gpu_checks.json and
exits 1 if any check fails. `--tiny` swaps in a 1-layer model for a quick CPU wiring check only -- it says
nothing about GPU numerics and must never be reported as a GPU result.
"""

from __future__ import annotations

import os

# Must precede any torch import; a launcher that already set it wins (setdefault).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402

GRAD_ACCUM = 40
INJECT, WIDTH, LAG, BUDGET = 6, 2, 2, 6  # fork at 6, lr_bump over 6..7, detector at 8, 6-step budget
OFFSET_START, OFFSET_K, OFFSET_STEPS = 3, 5, 4


def _cfg(tiny: bool, batch: int):
    from research.configs import load_config

    cfg = load_config("research/experiments/proxy/config.yaml")
    if tiny:
        cfg["model"].update({"n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8, "vocab_size": 64})
    cfg["train"]["batch_size"] = batch
    cfg["train"]["grad_accum"] = GRAD_ACCUM
    return cfg


def _data(out, cfg):
    import numpy as np

    path = os.path.join(out, "train.bin")
    vocab = cfg["model"].get("vocab_size") or 50000
    steps = INJECT + BUDGET + OFFSET_START + OFFSET_K + OFFSET_STEPS + LAG + WIDTH + 4
    n_tok = cfg["train"]["batch_size"] * cfg["model"]["block_size"] * GRAD_ACCUM * steps * 2
    np.random.default_rng(0).integers(0, min(vocab, 50000), n_tok).astype(np.uint16).tofile(path)
    return out


def _weights(model):
    import torch

    return torch.cat([p.detach().reshape(-1).float().cpu() for p in model.parameters()])


def _max_abs(a, b) -> float:
    return float((a - b).abs().max()) if a.numel() else 0.0


def _last_weights():
    box = {}

    def on_step(step, info):
        box["w"] = _weights(info["model"])

    return box, on_step


def check_brollback(cfg, data, device) -> dict:
    from research.baselines import rollback
    from research.harness.fork import make_baseline_branches, prime_trunk, run_branch, run_fork
    from research.spikes.induce import induce

    snap = prime_trunk(cfg, data, INJECT, device=device)  # seeds ONCE, deterministically (first CUDA touch)
    recipe = induce("lr_bump", cfg, inject_step=INJECT, width=WIDTH, factor=50.0)
    t0 = INJECT + LAG
    branch = make_baseline_branches(cfg, recipe, detector_t0=t0, **snap["meta"]["baseline_priming"])["Brollback"]
    wasted, skip = rollback.plan(INJECT, t0)  # whatever the default skip is, rebuild with the same one

    box1, on1 = _last_weights()
    run1 = run_branch(data, snap, branch, steps=BUDGET, device=device, seed=False, on_step=on1)
    box2, on2 = _last_weights()
    run2 = run_branch(data, snap, branch, steps=BUDGET, device=device, seed=False, on_step=on2)
    boxr, onr = _last_weights()
    ref = run_fork(
        cfg, data, snap, steps=BUDGET - wasted, start_step=INJECT + skip, device=device, seed=False, on_step=onr
    )
    d_ref, d_rerun = _max_abs(box1["w"], boxr["w"]), _max_abs(box1["w"], box2["w"])
    ok = run1["losses"] == ref and run1["losses"] == run2["losses"] and d_ref == 0.0 and d_rerun == 0.0
    ok = ok and len(ref) == BUDGET - wasted and run1.get("wasted_steps") == wasted
    return {
        "PASS": ok,
        "wasted_steps": wasted,
        "data_skip": skip,
        "steps_run": len(run1["losses"]),
        "losses_equal_rebuild": run1["losses"] == ref,
        "losses_equal_rerun": run1["losses"] == run2["losses"],
        "max_abs_w_vs_rebuild": d_ref,
        "max_abs_w_vs_rerun": d_rerun,
        "losses": run1["losses"],
    }


def check_data_offset(cfg, data, device) -> dict:
    import torch

    from research.baselines import rollback
    from research.harness.trunk import build_model_opt, train_forward

    def fresh():
        torch.manual_seed(0)  # identical init for both runs (determinism is already configured)
        return build_model_opt(cfg, device)

    m1, o1 = fresh()
    shifted = train_forward(
        m1, o1, cfg, data, OFFSET_START, OFFSET_STEPS, device, pre_step=rollback.as_pre_step(OFFSET_K)
    )
    m2, o2 = fresh()
    direct = train_forward(m2, o2, cfg, data, OFFSET_START + OFFSET_K, OFFSET_STEPS, device)
    d_w = _max_abs(_weights(m1), _weights(m2))
    d_v = max(
        _max_abs(o1.state[p1]["exp_avg_sq"].float().cpu(), o2.state[p2]["exp_avg_sq"].float().cpu())
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
    return {
        "PASS": shifted == direct and d_w == 0.0 and d_v == 0.0,
        "losses_equal": shifted == direct,
        "max_abs_w": d_w,
        "max_abs_v": d_v,
        "losses": shifted,
    }


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4, help="per-microbatch size; x40 accumulation per step")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tiny", action="store_true", help="CPU wiring check only; NOT a GPU result")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and not args.tiny:
        raise SystemExit("no GPU visible: this is a GPU check (use --tiny only for a CPU wiring check)")
    cfg = _cfg(args.tiny, args.batch)
    data = _data(args.out, cfg)

    rb = check_brollback(cfg, data, device)  # first: prime_trunk does the one deterministic seed
    off = check_data_offset(cfg, data, device)
    summary = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "tiny_cpu_wiring_check": args.tiny,
        "grad_accum": GRAD_ACCUM,
        "brollback_rebuild": rb,
        "data_offset": off,
        "PASS": rb["PASS"] and off["PASS"],
    }
    with open(os.path.join(args.out, "baseline_gpu_checks.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for name, r in (("brollback_rebuild", rb), ("data_offset", off)):
        print(f"{name}: {'PASS' if r['PASS'] else 'FAIL'} {json.dumps({k: v for k, v in r.items() if k != 'losses'})}")
    tag = " (TINY CPU WIRING CHECK, not a GPU result)" if args.tiny else ""
    print(f"BASELINE GPU CHECKS {'PASS' if summary['PASS'] else 'FAIL'} on {device}{tag}")
    raise SystemExit(0 if summary["PASS"] else 1)


if __name__ == "__main__":
    main()
