"""corrupt_batch lead-time probe (Task 8). GPU-only: run it on the AMD MI300X (or any CUDA/ROCm GPU),
never on the laptop:   python -m research.kaggle.corrupt_batch_probe

WHY corrupt_batch shows zero lead today (read from the code): `spikes/induce.py` sets
`ctx["batch"] = (garbage, garbage)`, so the loss LOGGED at the injection step is the loss on random tokens,
not a measurement of model damage. Its "peak" therefore sits on the injection step by construction, and no
detector can fire before its own cause. v3 scored the aftermath of the TRAINING loss (peak_offset=1) and
it never reached the 2x bar (commit bf5d42f).

WHAT this measures instead: model quality on CLEAN data. After every optimizer step the model is scored on
a fixed held-out probe (val batches 0..3), and each corrupted run is compared with its paired clean
counterfactual B* (same seed, same data stream, hook omitted). If the poisoned optimizer state keeps
damaging the model after the bad batch (momentum carrying the garbage direction), the probe peaks LATER
than the injection step -- that gap is real lead for a detector that fires on the garbage step itself.
The detector is unchanged: it fires from training loss / grad-norm, exactly as in `spike_run.py`.

Injection-timing variants: one bad batch at the plateau (160) and in steep descent (40), plus 3- and
5-batch bursts at 160. Seeds 0 (inspect) and 1 (held out). Each run is its own process so determinism can
be ON (`seed_everything` asserts CUDA is untouched); the pre-injection paired max|Δ| is then an exact-zero
check that the comparison is clean. If ROCm cannot run deterministically (the AMD go/no-go), pass
`--no-deterministic` and read that pre-injection max|Δ| as a noise floor instead.

PRE-DECLARED decision rule (fixed before any run; the 2.0 rise bar is not lowered after seeing data):
corrupt_batch is upgraded from `onset` to `predictive` iff on held-out seed 1 the single-batch plateau
variant (i160_w1) shows (a) a probe spike, probe peak / pre-injection baseline >= 2.0; (b) a detector
trigger t0 with probe_peak - t0 >= L=2; (c) zero false-positive steps in the clean prefix. Burst variants
are reported but cannot upgrade the recipe on their own, since part of their "lead" is corruption still
being injected. Otherwise corrupt_batch stays an onset recipe (lead 0, never called early warning).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before any torch import (harness convention)

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

STEPS, BATCH, L, RISE, WINDOW = 200, 16, 2, 2.0, 25
SEEDS = (0, 1)
HELDOUT_SEED, DECISION_VARIANT = 1, "i160_w1"
# name -> (inject_step, width); "clean" is the paired B* for its seed
VARIANTS = {
    "clean": (None, None),
    "i160_w1": (160, 1),
    "i160_w3": (160, 3),
    "i160_w5": (160, 5),
    "i40_w1": (40, 1),
}


# --------------------------------------------------------------------------------------
# GPU collection (one run per process)
# --------------------------------------------------------------------------------------


def _run_one(seed: int, variant: str, data_dir: str, out: str, deterministic: bool) -> None:
    """Train one proxy trunk (optionally with corrupt_batch) and log train loss, grad-norm and the clean
    probe loss after every optimizer step. Must be the process's first CUDA touch."""
    import torch

    from research.configs import load_config
    from research.data.prepare import get_batch
    from research.harness.trunk import run_trunk
    from research.spikes.induce import induce

    if not torch.cuda.is_available():  # does not initialize CUDA
        raise SystemExit("GPU required -- run on the AMD/Kaggle GPU, never on the laptop.")
    cfg = load_config("research/experiments/proxy/config.yaml", {"train.batch_size": BATCH, "seed": seed})
    inject, width = VARIANTS[variant]
    pre_step = induce("corrupt_batch", cfg, inject_step=inject, width=width).pre_step if inject is not None else None

    loss, gradnorm, probe_loss, probe = [], [], [], []

    def on_step(step, info):
        model = info["model"]
        if not probe:  # built lazily: CUDA may only be touched after run_trunk has seeded
            dev = next(model.parameters()).device
            probe.extend(get_batch("val", k, BATCH, cfg["model"]["block_size"], data_dir, dev) for k in range(4))
        loss.append(float(info["loss"]))
        gradnorm.append(float(info["grad_norm"]))
        with torch.no_grad():
            probe_loss.append(sum(model(x, y)[1].item() for x, y in probe) / len(probe))

    run_trunk(cfg, data_dir, steps=STEPS, on_step=on_step, pre_step=pre_step, deterministic=deterministic)
    rec = {
        "seed": seed,
        "variant": variant,
        "inject": inject,
        "width": width,
        "loss": loss,
        "gradnorm": gradnorm,
        "probe": probe_loss,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "deterministic": deterministic,
    }
    with open(out, "w") as fp:
        json.dump(rec, fp)


# --------------------------------------------------------------------------------------
# Scoring (pure numpy -- CPU-testable, no torch)
# --------------------------------------------------------------------------------------


def score(runs: list[dict], params) -> list[dict]:
    """Score every corrupted run against its seed's paired clean run. Returns one row per corrupted run."""
    import numpy as np

    from research.spikes.tune_detector import detection_qualifies, spike_occurred, trigger_mask

    clean = {r["seed"]: r for r in runs if r["variant"] == "clean"}
    rows = []
    for r in runs:
        if r["variant"] == "clean":
            continue
        inj, width = r["inject"], r["width"]
        b = clean[r["seed"]]
        probe, bprobe = np.asarray(r["probe"]), np.asarray(b["probe"])
        occ = spike_occurred(probe, inj, rise=RISE, window=WINDOW)
        floor = spike_occurred(bprobe, inj, rise=RISE, window=WINDOW)  # same window, no injection
        mask = trigger_mask(r["loss"], r["gradnorm"], params)
        t0 = next((t for t, fired in enumerate(mask) if fired), None)
        detected, lead = detection_qualifies(
            occurred=occ["occurred"],
            inject_step=inj,
            trigger_step=t0,
            peak_step=occ["peak_step"],
            min_lead=L,
            allow_at_peak=False,
        )
        d = probe - bprobe
        post = d[inj : inj + WINDOW]
        rows.append(
            {
                "seed": r["seed"],
                "variant": r["variant"],
                "inject": inj,
                "width": width,
                "pre_inject_paired_max_abs_delta": float(np.abs(d[:inj]).max()) if inj else 0.0,
                "probe_baseline": occ["baseline"],
                "probe_peak_step": occ["peak_step"],
                "probe_peak_ratio": occ["ratio"],
                "probe_spike_occurred": occ["occurred"],
                "clean_same_window_ratio": floor["ratio"],
                "paired_delta_peak": float(post.max()),
                "paired_delta_peak_step": inj + int(post.argmax()),
                "train_loss_at_inject": float(r["loss"][inj]),
                "t0": t0,
                "raw_lead": None if t0 is None else occ["peak_step"] - t0,
                "fp_steps": int(sum(mask[params.warmup : inj])),
                "peak_after_corruption_window": occ["peak_step"] >= inj + width,
                "predictive_detected": detected,
                "predictive_lead": lead,
            }
        )
    return rows


def decide(rows: list[dict]) -> dict:
    """Apply the pre-declared rule (module docstring) to the held-out single-batch plateau row."""
    row = next((x for x in rows if x["seed"] == HELDOUT_SEED and x["variant"] == DECISION_VARIANT), None)
    if row is None:
        return {"upgrade_to_predictive": False, "reason": f"missing {DECISION_VARIANT} seed {HELDOUT_SEED}"}
    ok = row["probe_spike_occurred"] and row["predictive_detected"] and row["fp_steps"] == 0
    return {
        "upgrade_to_predictive": bool(ok),
        "policy": "predictive" if ok else "onset",
        "basis": f"{DECISION_VARIANT}, seed {HELDOUT_SEED}",
        "probe_peak_ratio": row["probe_peak_ratio"],
        "predictive_lead": row["predictive_lead"],
        "fp_steps": row["fp_steps"],
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--one", nargs=2, metavar=("SEED", "VARIANT"), help="internal: run a single trace")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-deterministic", action="store_true")
    a = ap.parse_args()
    det = not a.no_deterministic
    if a.one:
        _run_one(int(a.one[0]), a.one[1], a.data, a.out, det)
        return

    from research.data.prepare import PRESETS, prepare, resolve_data_dir
    from research.spikes.tune_detector import load_params

    try:
        data_dir = resolve_data_dir("proxy")
    except FileNotFoundError:
        data_dir = prepare(PRESETS["proxy"])
    out_dir = os.environ.get("EVIDENCE_OUT", "results")
    trace_dir = os.path.join(out_dir, "corrupt_batch_probe_traces")
    os.makedirs(trace_dir, exist_ok=True)

    runs = []
    for seed in SEEDS:
        for variant in VARIANTS:
            path = os.path.join(trace_dir, f"seed{seed}_{variant}.json")
            cmd = [sys.executable, "-m", "research.kaggle.corrupt_batch_probe", "--one", str(seed), variant]
            cmd += ["--data", data_dir, "--out", path] + ([] if det else ["--no-deterministic"])
            subprocess.run(cmd, check=True)
            runs.append(json.load(open(path)))
            print(f"[probe] seed={seed} {variant} done", flush=True)

    rows = score(runs, load_params(out_dir))
    for x in rows:
        print(
            f"[{x['variant']} seed={x['seed']}] pre-inject max|Δ|={x['pre_inject_paired_max_abs_delta']:.3g} "
            f"probe peak@{x['probe_peak_step']} ratio={x['probe_peak_ratio']:.3f} (clean floor "
            f"{x['clean_same_window_ratio']:.3f}) "
            f"paired Δpeak={x['paired_delta_peak']:.4f}@{x['paired_delta_peak_step']} "
            f"t0={x['t0']} raw_lead={x['raw_lead']} predictive={x['predictive_detected']} FP={x['fp_steps']}"
        )
    decision = decide(rows)
    print("DECISION:", decision)
    with open(os.path.join(out_dir, "task8_corrupt_batch_probe.json"), "w") as fp:
        json.dump(
            {
                "claim": "Task 8: does corrupt_batch damage persist on clean data long enough to give real lead?",
                "gpu": runs[0]["gpu"],
                "torch": runs[0]["torch"],
                "deterministic": det,
                "steps": STEPS,
                "batch_size": BATCH,
                "L": L,
                "rise": RISE,
                "variants": VARIANTS,
                "rows": rows,
                "decision": decision,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
            },
            fp,
            indent=2,
        )


# --------------------------------------------------------------------------------------
# CPU-only self-check (numpy; no torch, no training)
# --------------------------------------------------------------------------------------


def _selfcheck() -> None:
    """Exercise score() + decide() on synthetic traces: a delayed clean-probe peak must upgrade the
    recipe; damage that only lands on the injection step must not; FP and paired-Δ paths must count."""
    import numpy as np

    from research.spikes.tune_detector import DetectorParams

    n, inj = STEPS, 160
    steps = np.arange(n)
    base = (3.0 + 4.0 * np.exp(-steps / 25.0)).tolist()

    def run(variant, seed, probe_bump=None):
        loss, grad, probe = list(base), [1.0] * n, list(base)
        if variant != "clean":
            loss[inj], grad[inj] = 10.8, 20.0  # garbage-batch loss + grad-norm burst at the cause
            for k, v in (probe_bump or {}).items():
                probe[inj + k] += v
        i, w = VARIANTS[variant]
        return {
            "seed": seed,
            "variant": variant,
            "inject": i,
            "width": w,
            "loss": loss,
            "gradnorm": grad,
            "probe": probe,
        }

    delayed = {0: 1.0, 1: 2.5, 2: 4.0, 3: 5.5, 4: 4.0, 5: 2.0}  # probe peaks 3 steps after the bad batch
    acute = {0: 1.0}  # damage only on the injection step, well under 2x
    p = DetectorParams()

    rows = score([run("clean", 1), run("i160_w1", 1, delayed)], p)
    assert rows[0]["pre_inject_paired_max_abs_delta"] == 0.0
    assert rows[0]["probe_peak_step"] == inj + 3 and rows[0]["t0"] == inj, rows[0]
    assert rows[0]["predictive_detected"] and rows[0]["predictive_lead"] == 3 and rows[0]["fp_steps"] == 0
    assert abs(rows[0]["paired_delta_peak"] - 5.5) < 1e-9 and rows[0]["paired_delta_peak_step"] == inj + 3, rows[0]
    assert rows[0]["peak_after_corruption_window"] and 0.99 < rows[0]["clean_same_window_ratio"] <= 1.0, rows[0]
    assert decide(rows)["upgrade_to_predictive"] is True

    # A pre-injection false fire must block the upgrade even when the probe spike and lead look fine.
    noisy = run("i160_w1", 1, delayed)
    noisy["gradnorm"][100] = 20.0
    rows = score([run("clean", 1), noisy], p)
    assert rows[0]["fp_steps"] >= 1 and decide(rows)["upgrade_to_predictive"] is False, rows[0]

    rows = score([run("clean", 1), run("i160_w1", 1, acute)], p)
    assert not rows[0]["probe_spike_occurred"] and decide(rows)["policy"] == "onset", rows[0]

    assert decide([])["upgrade_to_predictive"] is False  # missing held-out row never upgrades
    print("corrupt_batch_probe selfcheck OK: delayed probe peak -> predictive (lead 3); acute-only -> onset")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _selfcheck()
    else:
        main()
