"""Run the localizer on captured spikes and write the results (Week 6: one spike; Week 7: many seeds).

Input: the run directories `research.spikes.capture` writes (run.json + t0 / fork / normal snapshots
+ v_bar files). For every trigger in a run it calls `localizer.pipeline.localize_spike` and writes

  <out>/<run_id>_t<t0>.json      the Localization (PLAN_V8 §6.4) + provenance + eigensolve evidence

and, unless --no-normals, it also scores each matched normal step AS IF it were a spike (leave one
out: the other normal steps are its null), so spikes and normal steps can be compared. With
--battery it does that for every run under a folder and writes <out>/localize_report.json
(`localizer.validate.report`: spike-vs-normal AUC for psi_k / max_score / max_drift_ratio, per seed
and pooled, and how much the selected groups agree across seeds).

GPU ONLY (CLAUDE.md §1): it trains nothing, but the eigensolve is hundreds of double-backward HVPs
on the 11M-parameter proxy, and every MI300X session needs the user's go-ahead (CLAUDE.md §5).

    python -m research.gpu.localize_run --run results/spikes/<run_id>
    python -m research.gpu.localize_run --battery results/spikes          # Week 7: every run, every seed
    python -m research.gpu.localize_run --run ... --no-eig                # the no-HVP localizer only

KNOWN GAP (needs a change to research/spikes/capture.py, which is not this file's to change):
capture saves a v_bar only at fork points, not at the normal steps. The thresholds come from the
normal steps and need (v, v_bar) at each of them, so this driver stops with an explanation if
run.json has no "normal_vbars" list. Running capture with n_normal >= 4 is also needed for the
leave-one-out normal scoring (each normal step needs >= 3 others as its null).
"""

from __future__ import annotations

import os

# Must precede any torch import; a launcher that already set it wins (setdefault).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

PROBE_STEP = 10_000  # the fixed probe batch, the same one curvature_run uses

GAP = (
    "run.json has no 'normal_vbars': research/spikes/capture.py saved the normal-step snapshots without "
    "their v_bar, so there is nothing to derive thresholds from. The fix is in capture.py (save the v_bar "
    "next to each normal snapshot and list the paths as run['normal_vbars']); then re-capture."
)


def _load_vbar(path):
    from safetensors.torch import load_file

    return load_file(path)


def _normals(run):
    snaps = run.get("normal_snapshots") or []
    vbars = run.get("normal_vbars")
    if vbars is None:
        raise SystemExit(GAP)
    if len(vbars) != len(snaps):
        raise SystemExit(f"{len(snaps)} normal snapshots but {len(vbars)} normal v_bars in run.json")
    return list(zip(snaps, vbars))


def run_one(run_dir: str, args, out_dir: str) -> list[dict]:
    """Localize every trigger in one capture run (+ the leave-one-out normal steps). Returns the rows."""
    import torch

    from research.configs import load_config
    from research.data.prepare import PRESETS, get_batch, prepare, resolve_data_dir
    from research.harness.determinism import seed_everything
    from research.harness.snapshot import load
    from research.harness.trunk import build_model_opt
    from research.localizer.pipeline import localize_spike, summary_stats, trunk_microbatches

    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    kind = run["recipe"]["kind"] if run.get("recipe") else "natural"
    normal_paths = _normals(run)

    cfg = load_config(run["config_path"], run.get("overrides") or {})
    seed_everything(cfg.get("seed", 1337), deterministic=True)
    device = "cuda"
    try:
        data_dir = resolve_data_dir(run.get("data_preset", "proxy"))
    except FileNotFoundError:
        data_dir = prepare(PRESETS[run.get("data_preset", "proxy")])
    model, opt = build_model_opt(cfg, device)

    probe = get_batch("train", PROBE_STEP, args.probe, cfg["model"]["block_size"], data_dir, device)
    micro = trunk_microbatches(cfg, data_dir, n_micro=args.micro, device=device)
    normals = [(load(s), _load_vbar(v)) for s, v in normal_paths]

    common = dict(
        probe_batch=probe,
        microbatches=micro,
        use_eig=not args.no_eig,
        k=args.k,
        eps=cfg["optim"].get("eps", 1e-8),
        groups=args.groups,
        n_head=cfg["model"]["n_head"],
        k_select=args.k_select,
        min_steps=args.min_steps,
        mem_budget_gb=args.mem_gb,
    )

    rows = []

    def emit(name, loc, label, t):
        payload = {
            "run_id": run["run_id"],
            "seed": run["seed"],
            "kind": kind,
            "label": label,
            "t": t,
            "localization": loc,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {path}  selected={loc['selected']}")
        rows.append({**summary_stats(loc), "label": label, "seed": run["seed"], "kind": kind, "run_id": run["run_id"]})

    for trig in run["triggers"]:
        loc = localize_spike(
            model, opt, load(trig["t0_snapshot"]), _load_vbar(trig["vbar"]), normals, t=trig["t0"], **common
        )
        emit(f"{run['run_id']}_t{trig['t0']}.json", loc, "spike", trig["t0"])

    if not args.no_normals:
        for i, (snap, vb) in enumerate(normals):
            others = normals[:i] + normals[i + 1 :]
            if len(others) < args.min_steps:
                print(f"skip normal {snap['step']}: {len(others)} other normal steps < {args.min_steps}")
                continue
            loc = localize_spike(model, opt, snap, vb, others, **common)
            emit(f"{run['run_id']}_normal{snap['step']}.json", loc, "normal", int(snap["step"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Localizer on captured spikes (Weeks 6-7).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="one capture run directory (has run.json)")
    src.add_argument("--battery", help="a folder; every */run.json under it is localized")
    ap.add_argument("--out", default=os.path.join(os.environ.get("EVIDENCE_OUT", "results"), "localize"))
    ap.add_argument("-k", type=int, default=5, help="eigenpairs of H~")
    ap.add_argument("--k-select", type=int, default=5, help="max groups to select")
    ap.add_argument("--groups", choices=("tensor", "head"), default="tensor")
    ap.add_argument("--probe", type=int, default=4, help="probe batch size (sequences)")
    ap.add_argument("--micro", type=int, default=4, help="microbatches per step for the directional SNR")
    ap.add_argument("--min-steps", type=int, default=3, help="minimum normal steps behind a threshold")
    ap.add_argument("--mem-gb", type=float, default=6.0, help="host RAM budget for the Lanczos basis")
    ap.add_argument("--no-eig", action="store_true", help="no-HVP localizer (drift only)")
    ap.add_argument("--no-normals", action="store_true", help="skip leave-one-out scoring of normal steps")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():  # does not initialize CUDA
        raise SystemExit("GPU required -- run on the AMD GPU, never on a laptop (CLAUDE.md §1).")

    os.makedirs(args.out, exist_ok=True)
    run_dirs = (
        [args.run]
        if args.run
        else sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.battery, "**", "run.json"), recursive=True))
    )
    if not run_dirs:
        raise SystemExit(f"no run.json found under {args.battery}")

    rows = []
    for d in run_dirs:
        rows.extend(run_one(d, args, args.out))

    from research.localizer.validate import report

    rep = {"runs": run_dirs, "args": vars(args), "rows": rows, **report(rows)}
    path = os.path.join(args.out, "localize_report.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {path}")
    for stat, s in rep["separation"].items():
        print(f"  {stat:>16}: spike-vs-normal AUC = {s['auc']}  ({s['n_spike']} spikes, {s['n_normal']} normal)")


if __name__ == "__main__":
    main()
