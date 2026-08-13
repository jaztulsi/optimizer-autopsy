"""Task 8 real proxy run: induce each of the 4 spike recipes on the REAL proxy trunk (TinyStories),
score the online detector against MEASURED ground truth, and report the DoD on held-out spikes.

Two seeds per recipe: seed 0 tunes the detector thresholds, seed 1 is held out for the DoD check
(context §15.2: median lead >= L at FP <= f for >= 3/4 recipes). Short runs (a handful of steps past
injection) -- this answers "does the recipe spike + does the detector catch it", not convergence.

Run on Kaggle:  python -m research.kaggle.spike_run
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

INJECT = 30  # known cause step (>= detector warmup, leaves a clean prefix for FP)
STEPS = 60  # run long enough to see the post-inject peak + tail
BATCH = 16  # fits the T4 (batch 64 OOMs at this vocab); SDPA shapes set by block/head_dim
L, F = 2, 0.05  # DoD: median lead >= L steps at false-positive rate <= F
WIDTHS = {"lr_bump": 3, "tiny_eps": 3, "precision": 1, "corrupt_batch": 1}


def _run_recipe(kind: str, data_dir: str, seed: int, device: str) -> dict:
    from research.configs import load_config
    from research.harness.trunk import run_trunk
    from research.spikes.induce import induce

    cfg = load_config("research/experiments/proxy/config.yaml", {"train.batch_size": BATCH, "seed": seed})
    recipe = induce(kind, cfg, inject_step=INJECT, width=WIDTHS[kind])

    losses, grads = [], []

    def on_step(step, info):
        losses.append(info["loss"])
        grads.append(info["grad_norm"])

    run_trunk(cfg, data_dir, steps=STEPS, on_step=on_step, pre_step=recipe.pre_step, deterministic=False, device=device)
    return {"kind": kind, "seed": seed, "inject_step": INJECT, "losses": losses, "gradnorms": grads}


def main() -> None:
    import torch

    from research.data.prepare import PRESETS, prepare, resolve_data_dir
    from research.spikes.induce import RECIPES
    from research.spikes.tune_detector import DetectorParams, dod_check, score_spike, spike_occurred, tune

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("cuda_available:", torch.cuda.is_available())

    try:
        data_dir = resolve_data_dir("proxy")
    except FileNotFoundError:
        data_dir = prepare(PRESETS["proxy"])  # pinned-revision TinyStories shard

    tune_runs, test_runs, per_recipe = [], [], {}
    for kind in RECIPES:
        r0 = _run_recipe(kind, data_dir, seed=0, device=device)
        torch.cuda.empty_cache()
        r1 = _run_recipe(kind, data_dir, seed=1, device=device)
        torch.cuda.empty_cache()
        tune_runs.append(r0)
        test_runs.append(r1)
        occ = spike_occurred(r1["losses"], INJECT)  # ground truth on the held-out run
        per_recipe[kind] = {"spike_occurred": occ["occurred"], "peak_step": occ["peak_step"], "ratio": occ["ratio"]}
        print(
            f"[{kind}] held-out spike_occurred={occ['occurred']} peak_step={occ['peak_step']} ratio={occ['ratio']:.2f}"
        )

    best = tune(tune_runs, L=L, f=F)
    params = best["params"] if best else DetectorParams()
    print(
        f"tuned params: z={params.z} ema_mult={params.ema_mult} warmup={params.warmup} "
        f"(tune-set: {'passed' if best else 'NO passing point -> defaults'})"
    )

    detail = {}
    for r in test_runs:
        s = score_spike(r["losses"], r["gradnorms"], INJECT, params, min_lead=L)
        detail[r["kind"]] = {
            k: s[k] for k in ("occurred", "t0", "peak_step", "lead", "detected", "fp_steps", "clean_steps")
        }
        print(
            f"[{r['kind']}] held-out: occurred={s['occurred']} t0={s['t0']} peak={s['peak_step']} "
            f"lead={s['lead']} detected={s['detected']}"
        )

    dod = dod_check(test_runs, params, L=L, f=F)
    print(
        f"DoD (held-out): passed={dod['passed']} detected={dod['n_detected']}/{len(test_runs)} "
        f"(need {dod['need']}) median_lead={dod['median_lead']} FP={dod['fp_rate']:.3f} [L={L}, f={F}]"
    )

    out_dir = os.environ.get("EVIDENCE_OUT", "results")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "claim": "Task 8: induced-spike recipes + online detector DoD on the real proxy trunk",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "inject_step": INJECT,
        "steps": STEPS,
        "batch_size": BATCH,
        "L": L,
        "f": F,
        "detector_params": {"z": params.z, "ema_mult": params.ema_mult, "warmup": params.warmup, "win": params.win},
        "ground_truth_heldout": per_recipe,
        "detector_heldout": detail,
        "dod": dod,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "task8_spike_detector.json"), "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"SPIKE RUN DONE on {device}: DoD passed={dod['passed']} ({dod['n_detected']}/{len(test_runs)})")


if __name__ == "__main__":
    main()
