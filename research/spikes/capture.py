"""Spike capture: run one proxy trunk with a spike source and save everything C2/C3 need.

GPU-only, one run per process (deterministic mode must configure CUDA before its first use), so `--battery`
and natural.py launch every run as its own subprocess (`--one <json spec>`). One run:
  * streams the detector (`tune_detector.OnlineDetector`, identical to offline scoring);
  * INDUCED spike: pins the fork snapshot just before the injection (step == inject_step) -- a t0 snapshot
    already contains the injection, so forks cannot replay from it. NATURAL run: keeps a RAM ring of
    snapshots and takes the newest one at least `lookback` steps before t0;
  * keeps v̄, a running average (EMA, `vbar_beta`) of Adam's v, and saves its fork-time copy -- the pre-spike
    baseline of the localizer's poison score (PLAN V6 §3.6);
  * keeps the newest NORMAL snapshots (every `save_every` steps) until the first trigger; the `n_normal`
    newest at least `normal_gap` steps before the first fork point are kept, the rest deleted;
  * on each qualifying trigger (>= inject_step for induced; `cooldown` apart; at most `max_triggers`) saves
    the t0 snapshot, the fork snapshot and v̄;
  * logs per-tensor telemetry every step (RESEARCH.md §3) for offline detector research;
  * writes run.json + telemetry.npz; `k2.build_k2` turns run.json files into manifest entries.

Run on the AMD GPU (never on a laptop):
  python -m research.spikes.capture --selfcheck   # seconds; the first thing to run on new hardware
  python -m research.spikes.capture --battery     # the PLAN V7 §4.2 induced instances -> K2 manifest
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from collections import deque  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

CONFIG = "research/experiments/proxy/config.yaml"
BATCH = 16
SITES = (160, 480)  # PLAN V7 §4.2 (600-step trunk)
SEEDS = (0, 1, 2, 3, 4)
BATTERY_KINDS = ("lr_bump", "corrupt_batch")
POST = 60  # steps past the injection: covers spike_occurred's 25-step window plus detector lead
SIGNALS = ("update_rms", "max_g2_over_v", "adam_update_rms", "vbar_drift")
KNOBS = {  # capture defaults; a spec may override any of them
    "vbar_beta": 0.99,
    "snap_every": 10,
    "ring": 8,
    "lookback": 20,
    "save_every": 40,
    "n_normal": 3,
    "normal_gap": 20,
    "cooldown": 50,
    "max_triggers": 3,
}


def tensor_signals(g, m, v, vbar, step: int, betas, eps: float):
    """One tensor's telemetry: [RMS(g/√v̂), max g²/v̂, RMS(m̂/(√v̂+ε)), ‖v−v̄‖/‖v̄‖].

    v̂, m̂ are Adam's bias-corrected moments after this step; g²/v̂ floors v̂ at ε² (Wortsman et al. 2023's
    RMS_t, whose spikes preceded 14/15 loss spikes). The last entry is the drift from the running average
    that PLAN V6 §3.6's poison score is built on."""
    import torch

    b1, b2 = betas
    vh = v / (1 - b2**step)
    mh = m / (1 - b1**step)
    r = g.float().square() / vh.clamp_min(eps * eps)
    upd = mh / (vh.sqrt() + eps)
    drift = (v - vbar).norm() / vbar.norm().clamp_min(1e-30)
    return torch.stack([r.mean().sqrt(), r.max(), upd.square().mean().sqrt(), drift])


def fork_from_ring(ring_steps: list[int], t0: int, lookback: int) -> int | None:
    """Index of the newest ring snapshot whose `step` (next step to run) is <= t0 + 1 - lookback."""
    ok = [i for i, s in enumerate(ring_steps) if s <= t0 + 1 - lookback]
    return ok[-1] if ok else None


def pick_normals(pool_steps: list[int], fork_step: int, n_normal: int, gap: int) -> list[int]:
    """The `n_normal` newest pool steps at least `gap` steps before the fork point."""
    return [s for s in sorted(pool_steps) if s <= fork_step - gap][-n_normal:]


def run_one(spec: dict) -> dict:
    """Train one trunk under `spec` and write `<out_dir>/run.json` (+ snapshots, telemetry.npz)."""
    import numpy as np
    import torch

    if not torch.cuda.is_available():  # does not initialize CUDA
        raise SystemExit("GPU required -- run on the AMD GPU, never on a laptop.")

    from research.baselines import adagc, zclip
    from research.configs import load_config
    from research.data.prepare import PRESETS, prepare, resolve_data_dir
    from research.harness.fork import _baseline_max_norm
    from research.harness.snapshot import capture
    from research.harness.snapshot import save as save_snapshot
    from research.harness.trunk import run_trunk
    from research.spikes.induce import induce
    from research.spikes.tune_detector import OnlineDetector, load_params, trigger_mask

    k = {**KNOBS, **spec.get("knobs", {})}
    out = spec["out_dir"]
    os.makedirs(out, exist_ok=True)
    data_dir = spec.get("data_dir")
    if data_dir is None:
        try:
            data_dir = resolve_data_dir("proxy")
        except FileNotFoundError:
            data_dir = prepare(PRESETS["proxy"])
    overrides = {"train.batch_size": BATCH, **spec.get("overrides", {}), "seed": spec["seed"]}
    cfg = load_config(spec.get("config", CONFIG), overrides)

    rec = spec.get("recipe")
    inject = rec["inject_step"] if rec else None
    pre_step = None
    if rec:
        params = dict(rec.get("params", {}))
        if rec["kind"] == "repeat_ngram":
            params.update(data_dir=data_dir, seed=spec["seed"])
        pre_step = induce(rec["kind"], cfg, inject_step=inject, width=rec["width"], **params).pre_step

    det = OnlineDetector(load_params(spec.get("detector_results", "results")))  # spike_run.py's tuned point
    losses, gradnorms, telemetry, triggers = [], [], [], []
    st = {"vbar": None, "named": None, "groups": None, "pinned": None, "first_fork": None, "last": None}
    ring: deque = deque(maxlen=k["ring"])
    pool: deque = deque(maxlen=k["n_normal"] + 2 + (k["lookback"] + k["normal_gap"]) // k["save_every"])
    saved: dict = {}  # snapshot step -> (snapshot path, vbar path), so a shared fork point is saved once

    def save_pair(step, snap, vbar_cpu, tag):
        from safetensors.torch import save_file

        if step not in saved:
            sp, vp = os.path.join(out, f"{tag}_{step}.safetensors"), os.path.join(out, f"vbar_{step}.safetensors")
            save_snapshot(snap, sp)
            save_file({n: t.contiguous() for n, t in vbar_cpu.items()}, vp)
            saved[step] = (sp, vp)
        return saved[step]

    def vbar_cpu():
        return {n: t.detach().to("cpu", torch.float32).clone() for n, t in st["vbar"].items()}

    # ZClip/AdaGC shadow priming (PR #14, same as harness.fork.prime_trunk): they watch every trunk step
    # without touching grads; each fork snapshot carries their state in meta["baseline_priming"].
    max_norm = _baseline_max_norm(cfg)
    zc, ag = zclip.ZClip(max_norm=max_norm), adagc.AdaGC(max_norm=max_norm)
    watchers = (zc.observer(), ag.observer(cfg["optim"].get("grad_clip") or 0.0))

    def primed():
        return {"baseline_priming": {"zclip_state": zc.state_dict(), "adagc_state": ag.state_dict()}}

    def on_step(s, info):
        model, opt = info["model"], info["opt"]
        for w in watchers:  # before any capture below, so a snapshot's priming includes step s
            w(s, info)
        if st["named"] is None:
            st["named"] = list(model.named_parameters())
            st["groups"] = {id(p): g for g in opt.param_groups for p in g["params"]}
            st["vbar"] = {n: opt.state[p]["exp_avg_sq"].detach().clone() for n, p in st["named"]}
        losses.append(float(info["loss"]))
        gradnorms.append(float(info["grad_norm"]))
        rows = []
        with torch.no_grad():
            for n, p in st["named"]:
                state, grp, vb = opt.state[p], st["groups"][id(p)], st["vbar"][n]
                v = state["exp_avg_sq"]
                vb.mul_(k["vbar_beta"]).add_(v, alpha=1 - k["vbar_beta"])
                if p.grad is None:
                    rows.append(torch.full((len(SIGNALS),), float("nan"), device=v.device))
                    continue
                step = int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
                rows.append(tensor_signals(p.grad, state["exp_avg"], v, vb, step, grp["betas"], grp["eps"]))
            telemetry.append(torch.stack(rows).tolist())
        nxt = s + 1  # a snapshot taken now has step = the next step to run
        meta = {"run_id": spec["run_id"], "seed": spec["seed"]}
        if inject is not None and nxt == inject:  # induced: the ideal fork point, just before the injection
            st["pinned"] = (nxt, capture(model, opt, nxt, meta={**meta, "role": "fork", **primed()}), vbar_cpu())
        if inject is None and nxt % k["snap_every"] == 0:
            ring.append((nxt, capture(model, opt, nxt, meta={**meta, "role": "ring", **primed()}), vbar_cpu()))
        if not triggers and nxt % k["save_every"] == 0:
            path = os.path.join(out, f"normal_{nxt}.safetensors")
            save_snapshot(capture(model, opt, nxt, meta={**meta, "role": "normal"}), path)
            if len(pool) == pool.maxlen:
                os.remove(pool[0][1])
            pool.append((nxt, path))

        fired = det.update(info["loss"], info["grad_norm"])
        eligible = fired and (inject is None or s >= inject) and len(triggers) < k["max_triggers"]
        if not eligible or (st["last"] is not None and s - st["last"] < k["cooldown"]):
            return
        if inject is not None:
            fork = st["pinned"]
        else:
            i = fork_from_ring([r[0] for r in ring], s, k["lookback"])
            fork = ring[i] if i is not None else None
        if fork is None:
            return  # no snapshot far enough back (trigger too early in the run); counted via n_fired
        t0_path = os.path.join(out, f"t0_{s}.safetensors")
        save_snapshot(capture(model, opt, nxt, meta={**meta, "role": "t0"}), t0_path)
        fork_path, vbar_path = save_pair(fork[0], fork[1], fork[2], "fork")
        triggers.append(
            {
                "t0": s,
                "stat": dict(det.stat),
                "t0_snapshot": t0_path,
                "fork_snapshot": fork_path,
                "fork_step": fork[0],
                "vbar": vbar_path,
            }
        )
        if st["first_fork"] is None:
            st["first_fork"] = fork[0]
        st["last"] = s

    n_steps = spec["steps"]
    run_trunk(cfg, data_dir, steps=n_steps, on_step=on_step, pre_step=pre_step, deterministic=True)

    keep = pick_normals([p[0] for p in pool], st["first_fork"], k["n_normal"], k["normal_gap"]) if triggers else []
    for step, path in pool:
        if step not in keep:
            os.remove(path)
    normals = [os.path.join(out, f"normal_{s}.safetensors") for s in keep]
    np.savez_compressed(
        os.path.join(out, "telemetry.npz"),
        signals=np.asarray(telemetry, dtype=np.float32),
        names=np.asarray([n for n, _ in st["named"]]),
        signal_names=np.asarray(SIGNALS),
    )
    run = {
        "run_id": spec["run_id"],
        "config_path": spec.get("config", CONFIG),
        "overrides": overrides,
        "seed": spec["seed"],
        "stress": spec.get("stress"),
        "recipe": None if rec is None else {k_: v for k_, v in rec.items() if k_ != "data_dir"},
        "data_preset": "proxy",
        "data_revision": PRESETS["proxy"].hf_revision,
        "steps": n_steps,
        "losses": losses,
        "gradnorms": gradnorms,
        "detector_params": asdict(det.p),
        "n_fired": int(sum(trigger_mask(losses, gradnorms, det.p))),  # every firing step, false alarms included
        "triggers": triggers,
        "normal_snapshots": normals,
        "knobs": k,
        "telemetry": os.path.join(out, "telemetry.npz"),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "backend": "rocm" if torch.version.hip else "cuda",
            "deterministic": True,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out, "run.json"), "w") as fp:
        json.dump(run, fp)
    return run


def _battery_specs(out_root: str) -> list[dict]:
    """The PLAN V7 §4.2 induced instances: 2 recipes x 2 sites x 5 seeds, policies from spike_run.SPEC."""
    from research.kaggle.spike_run import SPEC

    specs = []
    for kind in BATTERY_KINDS:
        sp = SPEC[kind]
        for site in SITES:
            for seed in SEEDS:
                run_id = f"{kind}-i{site}-s{seed}"
                specs.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "steps": site + POST,
                        "out_dir": os.path.join(out_root, run_id),
                        "recipe": {
                            "kind": kind,
                            "inject_step": site,
                            "width": sp["width"],
                            "params": sp["params"],
                            "min_lead": sp["min_lead"],
                            "allow_at_peak": sp["allow_at_peak"],
                            "peak_offset": sp["peak_offset"],
                        },
                    }
                )
    return specs


def _selfcheck() -> None:
    """GPU-only end-to-end check on a tiny synthetic shard: capture an lr_bump spike, then prove the saved
    fork snapshot is fork-ready -- replaying the recipe from it must reproduce the trunk's losses bitwise."""
    import tempfile

    import numpy as np

    from research.harness.fork import run_fork
    from research.harness.snapshot import load
    from research.spikes.induce import induce

    with tempfile.TemporaryDirectory() as d:
        vocab = 32
        np.tile(np.arange(vocab, dtype=np.uint16), 4000).tofile(f"{d}/train.bin")
        np.tile(np.arange(vocab, dtype=np.uint16), 100).tofile(f"{d}/val.bin")
        import yaml

        cfg_path = os.path.join(d, "cfg.yaml")
        with open(cfg_path, "w") as fp:
            yaml.safe_dump(  # not json.dump: YAML 1.1 reads json's `1e-08` back as a string, and AdamW rejects it
                {
                    "seed": 0,
                    "model": {
                        "n_layer": 2,
                        "n_head": 2,
                        "n_embd": 64,
                        "block_size": 32,
                        "vocab_size": vocab,
                        "dropout": 0.0,
                        "bias": False,
                    },
                    "optim": {"lr": 3e-3, "weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip": 1.0},
                    "train": {"batch_size": 16, "grad_accum": 1, "max_steps": 80},
                },
                fp,
            )
        inject = 40
        spec = {
            "run_id": "selfcheck",
            "seed": 0,
            "steps": 80,
            "out_dir": os.path.join(d, "run"),
            "config": cfg_path,
            "data_dir": d,
            "detector_results": d,  # no tuned result here -> default thresholds, independent of results/
            "overrides": {"train.batch_size": 16},
            "knobs": {"save_every": 10, "normal_gap": 5},
            "recipe": {
                "kind": "lr_bump",
                "inject_step": inject,
                "width": 3,
                "params": {"factor": 100.0},
                "min_lead": 2,
                "allow_at_peak": False,
                "peak_offset": 0,
            },
        }
        run = run_one(spec)
        assert len(run["losses"]) == 80 and run["triggers"], "no trigger captured -- lr x100 should fire"
        tr = run["triggers"][0]
        assert tr["fork_step"] == inject <= tr["t0"], tr
        t0_snap, fork_snap = load(tr["t0_snapshot"]), load(tr["fork_snapshot"])
        assert t0_snap["step"] == tr["t0"] + 1 and fork_snap["step"] == inject
        assert run["normal_snapshots"] and all(os.path.exists(p) for p in run["normal_snapshots"])
        tel = np.load(run["telemetry"])
        assert tel["signals"].shape == (80, len(tel["names"]), len(SIGNALS))

        from research.configs import load_config

        cfg = load_config(cfg_path, {"train.batch_size": 16, "seed": 0})
        recipe = induce("lr_bump", cfg, inject_step=inject, width=3, factor=100.0)
        k_steps = tr["t0"] + 1 - inject
        replay = run_fork(cfg, d, fork_snap, steps=k_steps, seed=False, pre_step=recipe.pre_step)
        delta = max(abs(a - b) for a, b in zip(replay, run["losses"][inject : tr["t0"] + 1]))
        assert delta == 0.0, f"fork replay from the captured snapshot drifted: max|Δ|={delta}"
        print(f"capture selfcheck OK on GPU: fork@{inject} t0={tr['t0']} replay max|Δ|=0 over {k_steps} steps")


def main() -> None:
    ap = argparse.ArgumentParser(description="Spike capture driver (GPU-only).")
    ap.add_argument("--one", help="internal: JSON spec for a single run")
    ap.add_argument("--battery", action="store_true", help="run the PLAN V7 §4.2 induced instances")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default=os.environ.get("EVIDENCE_OUT", "results"))
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
    elif a.one:
        run_one(json.loads(a.one))
    elif a.battery:
        from research.spikes.k2 import build_k2, save_k2

        paths = []
        for spec in _battery_specs(os.path.join(a.out, "capture_runs")):
            subprocess.run([sys.executable, "-m", "research.spikes.capture", "--one", json.dumps(spec)], check=True)
            paths.append(os.path.join(spec["out_dir"], "run.json"))
        manifest = build_k2(paths)
        save_k2(manifest, os.path.join(a.out, "k2_manifest_induced.json"))
        print(f"battery: {len(manifest.entries)}/{len(paths)} runs gave a qualifying spike -> k2_manifest_induced.json")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
