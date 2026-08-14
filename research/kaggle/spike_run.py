"""Task 8 real proxy run: induce each of the 4 spike recipes on the REAL proxy trunk (TinyStories),
score the online detector against MEASURED ground truth, and report the DoD on held-out spikes.

Calibration history: round 1 came in 0/4 (spikes masked by a prefix-median baseline during the steep
descent). Round 2 (LOCAL baseline + plateau injection + x50 lr_bump) reached 1/4 -- lr_bump fully
validated the mechanism (detected, lead 2), and it surfaced two recipe-specific issues now fixed here:
  * lr_bump    -- plateau (160), x50; validated -> unchanged.
  * corrupt_batch -- spikes reliably but its acute loss lands ON the injection step (no lead); score
    the AFTERMATH window [inject+1, inject+window) instead (peak_offset=1) -- the persistence in
    optimizer state is exactly what the localizer will act on, not the acute batch.
  * precision  -- fp16 at the plateau doesn't overflow (settled weights); inject EARLY (step 15, steep
    descent) where fp16 is likelier to actually blow up. One escalation; if flat, that is a real answer.
  * tiny_eps   -- held as-is (early, step 20). Appears near-mechanistically-inert at this scale
    (1/(sqrt(v)+eps) only explodes at v~eps^2~1e-24); flagged as an open spec question, not engineered around.
DoD thresholds are unchanged (L=2, f=0.05): these fixes make real spikes visible, they do not lower
the bar. Two seeds per recipe: seed 0 tunes the detector thresholds, seed 1 is held out for the DoD.

Run on Kaggle:  python -m research.kaggle.spike_run
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

STEPS = 200  # long enough to reach the plateau before the (plateau) injections
BATCH = 16  # fits the T4 (batch 64 OOMs at this vocab); SDPA shapes set by block/head_dim
L, F = 2, 0.05  # DoD: median lead >= L steps at false-positive rate <= F (UNCHANGED)

# Per-recipe injection + measurement:
#   inject      -- step to perturb (plateau ~160, or early where the mechanism is more plausible)
#   peak_offset -- start the peak window this many steps AFTER injection (1 = aftermath, skips the
#                  acute injection-step loss; the persistence in optimizer state is what the localizer acts on)
SPEC = {
    "lr_bump": {"inject": 160, "width": 3, "params": {"factor": 50.0}, "peak_offset": 0},
    "tiny_eps": {"inject": 20, "width": 5, "params": {}, "peak_offset": 0},  # held as-is (near-inert; documented)
    "precision": {"inject": 15, "width": 3, "params": {}, "peak_offset": 0},  # early descent: fp16 likelier to overflow
    "corrupt_batch": {
        "inject": 160,
        "width": 1,
        "params": {},
        "peak_offset": 1,
    },  # score the aftermath, not the bad batch
}


def _run_recipe(kind: str, data_dir: str, seed: int, device: str) -> dict:
    from research.configs import load_config
    from research.harness.trunk import run_trunk
    from research.spikes.induce import induce

    spec = SPEC[kind]
    cfg = load_config("research/experiments/proxy/config.yaml", {"train.batch_size": BATCH, "seed": seed})
    recipe = induce(kind, cfg, inject_step=spec["inject"], width=spec["width"], **spec["params"])

    losses, grads = [], []

    def on_step(step, info):
        losses.append(info["loss"])
        grads.append(info["grad_norm"])

    run_trunk(cfg, data_dir, steps=STEPS, on_step=on_step, pre_step=recipe.pre_step, deterministic=False, device=device)
    return {
        "kind": kind,
        "seed": seed,
        "inject_step": spec["inject"],
        "peak_offset": spec["peak_offset"],
        "losses": losses,
        "gradnorms": grads,
    }


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
        occ = spike_occurred(r1["losses"], r1["inject_step"], peak_offset=r1["peak_offset"])  # held-out ground truth
        per_recipe[kind] = {
            "inject_step": r1["inject_step"],
            "peak_offset": r1["peak_offset"],
            "spike_occurred": occ["occurred"],
            "peak_step": occ["peak_step"],
            "ratio": occ["ratio"],
        }
        print(
            f"[{kind}] inject={r1['inject_step']} held-out spike_occurred={occ['occurred']} "
            f"peak_step={occ['peak_step']} ratio={occ['ratio']:.2f}"
        )

    best = tune(tune_runs, L=L, f=F)
    params = best["params"] if best else DetectorParams()
    print(
        f"tuned params: z={params.z} ema_mult={params.ema_mult} warmup={params.warmup} "
        f"(tune-set: {'passed' if best else 'NO passing point -> defaults'})"
    )

    detail = {}
    for r in test_runs:
        s = score_spike(r["losses"], r["gradnorms"], r["inject_step"], params, min_lead=L, peak_offset=r["peak_offset"])
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
        "claim": "Task 8: induced-spike recipes + online detector DoD on the real proxy trunk (calibrated)",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "spec": SPEC,
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
