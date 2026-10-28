"""GPU driver for one P1-core instance (PLAN_V8 §2.6 skeleton; §7.0 C-1..C-4). Run on the AMD MI300X only.

    python -m research.autopsy.mve --config research/experiments/proxy/config.yaml \
        --kind corrupt_batch --inject-step 160 --out results/autopsy
    python -m research.autopsy.mve ... --battery repair --lag 3          # C-4 repair battery

One instance = one (spike kind, inject step, width, seed, betas). Everything forks from ONE snapshot at
`inject_step` (pre-spike, primed for ZClip/AdaGC) and runs to end of schedule; every branch gets the same
held-out val probe at the same horizons, so Delta = val(branch) - val(B*).

Phase A (the data spike itself):
  B*            clean counterfactual (same data slots, spike off)
  B0            spike on; also RECORDS the window's (g - g*) -> G for phase B (recording changes nothing)
  U_mv@tau      C-1 kernel: undo m+v at lag tau (tau=0 -> direct effect only)
  U_m@tau, U_v@tau   C-2 arms (protective v: U_v is predicted WORSE than B0)
Phase B (planted on the clean stream, G = the measured corrupt-batch gradient delta, post-clip):
  P+ / P-       M2 parity (P+ should reproduce B0 up to fp rounding -- a free sanity check)
  I_same@dt, I_opp@dt, S2+@dt, S2-@dt   C-3 interference: I = D(both) - D(first) - D(second)
  R@dt, Ponly@dt   M3 refractory: chi = (D(R) - D(P+)) / D(Ponly)

Results go to <out>/<instance>.json, rewritten after every branch; a rerun skips finished branches.
Derived quantities (kernel, Shapley, interference, chi, stats) are computed offline from the JSON.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import subprocess
import sys
import time


def _stamp(cfg: dict) -> dict:
    """Provenance for every result file (PLAN_V8 §6.10)."""
    import torch

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "argv": sys.argv,
        "cfg": cfg,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "device": dev,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _with_schedule(branches: dict, cfg: dict) -> dict:
    """Prepend the LR schedule to every branch (fork.make_branches' B* has pre_step=None, which would
    silently drop a cosine/WSD schedule)."""
    from research.autopsy import schedule
    from research.harness.fork import Branch, compose

    sched = schedule.as_pre_step(cfg)
    return {n: Branch(b.name, compose(sched, b.pre_step), b.cfg, b.pre_opt_step) for n, b in branches.items()}


def phase_a(cfg, recipe, data_dir, taus, c2_taus, placebo=0, channels=()):
    """{name: Branch} for the data-spike arms, plus the B0 record log (filled when B0 runs).
    `Pl_<ch>#k`: K 1-ulp placebo forks of B* per channel (w / g) -> the chaos floor sigma_chaos(h)."""
    from research.autopsy import undo
    from research.autopsy.probes import ulp_placebo
    from research.harness.fork import Branch, compose

    window = list(range(recipe.inject_step, recipe.inject_step + recipe.width))
    last = window[-1]

    def undo_branch(name, tau, ch):
        log = undo.new_log()
        pre = compose(recipe.pre_step, undo.as_undo_pre_step(log, undo.undo_step(last, tau), ch))
        return Branch(name, pre, cfg, undo.as_record_hook(log, window, data_dir=data_dir))

    log0 = undo.new_log()
    br = {
        "B*": Branch("B*", None, cfg),
        "B0": Branch("B0", recipe.pre_step, cfg, undo.as_record_hook(log0, window, data_dir=data_dir)),
    }
    for t in taus:
        br[f"U_mv@{t}"] = undo_branch(f"U_mv@{t}", t, "mv")
    for t in c2_taus:
        br[f"U_m@{t}"] = undo_branch(f"U_m@{t}", t, "m")
        br[f"U_v@{t}"] = undo_branch(f"U_v@{t}", t, "v")
    for ch in channels:
        for k in range(placebo):
            h = ulp_placebo(k, recipe.inject_step, ch)
            br[f"Pl_{ch}#{k}"] = Branch(f"Pl_{ch}#{k}", h if ch == "w" else None, cfg, h if ch == "g" else None)
    return br, log0


def phase_b(cfg, recipe, log0, dts, eps, coefs=(), dose_taus=(), extra=None):
    """{name: Branch} for the planted probes, built from B0's recorded window deltas. A width-w spike is
    planted as its w per-step deltas; the second spike/probe is the same window shifted by dt.
    Dose arms `Px<c>` plant c*G (same direction, pure amplitude) and meter its M5 dose x_bar into
    extra[name]; `Px<c>|U_<ch>@tau` undo it from m/v; `Px<c>|cool@tau` undo v + global cooldown (H-D)."""
    from research.autopsy import undo
    from research.autopsy.probes import chain_opt, dose_meter, plant, planted_clean_fn, spike_delta
    from research.harness.fork import Branch

    extra = {} if extra is None else extra

    window = list(range(recipe.inject_step, recipe.inject_step + recipe.width))
    deltas = {s: spike_delta(log0, s) for s in window}

    def planted(name, *shifts):  # shifts: (dt, coef) pairs
        hooks = [plant(deltas[s], {s + dt: c}) for dt, c in shifts for s in window]
        return Branch(name, None, cfg, chain_opt(*hooks))

    br = {"P+": planted("P+", (0, 1.0)), "P-": planted("P-", (0, -1.0))}
    for dt in dts:
        if dt < recipe.width:
            raise ValueError(f"dt={dt} overlaps the width-{recipe.width} window")
        br[f"I_same@{dt}"] = planted(f"I_same@{dt}", (0, 1.0), (dt, 1.0))
        br[f"I_opp@{dt}"] = planted(f"I_opp@{dt}", (0, 1.0), (dt, -1.0))
        br[f"S2+@{dt}"] = planted(f"S2+@{dt}", (dt, 1.0))
        br[f"S2-@{dt}"] = planted(f"S2-@{dt}", (dt, -1.0))
        br[f"R@{dt}"] = planted(f"R@{dt}", (0, 1.0), (dt, eps))
        br[f"Ponly@{dt}"] = planted(f"Ponly@{dt}", (dt, eps))
    for c in coefs:
        plants = [plant(deltas[s], {s: c}) for s in window]
        extra[f"Px{c:g}"] = {}
        br[f"Px{c:g}"] = Branch(f"Px{c:g}", None, cfg, chain_opt(*plants, dose_meter(deltas, c, extra[f"Px{c:g}"])))
        for t in dose_taus:
            for ch in ("m", "v", "mv", "cool"):
                name, log = f"Px{c:g}|{'cool' if ch == 'cool' else 'U_' + ch}@{t}", undo.new_log()
                rec = undo.as_record_hook(
                    log, window, clean_fn=lambda s, ctx, c=c: planted_clean_fn(deltas[s], {s: c})(s, ctx)
                )
                at = undo.undo_step(window[-1], t)
                if ch == "cool":
                    extra[name] = {}
                    pre = undo.as_cooldown_pre_step(log, at, "v", extra[name])
                else:
                    pre = undo.as_undo_pre_step(log, at, ch)
                br[name] = Branch(name, pre, cfg, chain_opt(*plants, rec))
    return br


def repair_battery(cfg, recipe, data_dir, snap, lag, detector_t0=None, extra=None):
    """C-4: the Task-9 battery (Bs clip at 0.5x trunk clip, PLAN_V8 §7.2) + published baselines + RML
    (state-only, fired `lag` steps after the window -- a detector-timed lag, not the oracle):
    RML@lag = full m+v undo; RML-cool = m+v undo + global cooldown; RML-sel = m undo + v undo on the top
    1% of coordinates by x_i (docs/prereg_v2.md H-E, H-F). Edit stats land in extra[name]."""
    from research.autopsy import undo
    from research.harness.fork import Branch, compose, make_baseline_branches, make_branches

    clip = cfg["optim"].get("grad_clip") or 1.0
    br = make_branches(cfg, recipe, clip_norm=0.5 * clip)
    br.update(make_baseline_branches(cfg, recipe, detector_t0=detector_t0, **snap["meta"].get("baseline_priming", {})))
    window = list(range(recipe.inject_step, recipe.inject_step + recipe.width))
    extra = {} if extra is None else extra
    at = undo.undo_step(window[-1], lag)

    def sel(log, out):
        def pre_step(step, ctx):
            if step == at:
                undo.undo(ctx["model"], ctx["opt"], log, at, "m")
                out["v_edit_frac"] = undo.undo_v_top(ctx["model"], ctx["opt"], log, at, 0.01)

        return pre_step

    for name in (f"RML@{lag}", f"RML-cool@{lag}", f"RML-sel@{lag}"):
        log, extra[name] = undo.new_log(), {}
        fix = (
            undo.as_undo_pre_step(log, at, "mv")
            if name.startswith("RML@")
            else undo.as_cooldown_pre_step(log, at, "mv", extra[name])
            if "cool" in name
            else sel(log, extra[name])
        )
        br[name] = Branch(name, compose(recipe.pre_step, fix), cfg, undo.as_record_hook(log, window, data_dir=data_dir))
    return br


def run_instance(args) -> str:
    import torch

    from research.analysis.eval import heldout_steps, horizon_probe
    from research.configs import load_config
    from research.data.prepare import resolve_data_dir
    from research.harness.determinism import seed_everything
    from research.harness.fork import prime_trunk, run_branch
    from research.harness.snapshot import load, save
    from research.spikes.induce import induce

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.betas:
        cfg["optim"]["betas"] = [float(b) for b in args.betas.split(",")]
    data_dir = args.data_dir or cfg.get("data", {}).get("dir") or resolve_data_dir(args.shard)
    params = {"data_dir": data_dir} if args.kind == "repeat_ngram" else {}
    recipe = induce(args.kind, cfg, args.inject_step, args.width, **params)

    b1, b2 = cfg["optim"]["betas"]
    tag = f"{args.kind}_w{args.width}_inj{args.inject_step}_seed{cfg['seed']}_b{b1}-{b2}_{args.battery}"
    os.makedirs(args.out, exist_ok=True)
    out_path, snap_path = os.path.join(args.out, f"{tag}.json"), os.path.join(args.out, f"{tag}.snap.safetensors")

    if cfg["optim"].get("schedule") and "pre_step" not in inspect.signature(prime_trunk).parameters:
        # ponytail: prime_trunk (fork.py, harness owner) trains the trunk with no pre_step, so a schedule
        # would silently be dropped before the fork. Needs a pre_step passthrough there; refuse until then.
        raise NotImplementedError("cfg has an LR schedule but fork.prime_trunk cannot apply it (flagged)")
    if os.path.exists(snap_path):  # resume: seed once ourselves, reuse the primed pre-spike snapshot
        seed_everything(cfg.get("seed", 1337), deterministic=True)
        snap = load(snap_path)
    else:
        extra = {}
        if "pre_step" in inspect.signature(prime_trunk).parameters:
            from research.autopsy import schedule

            extra["pre_step"] = schedule.as_pre_step(cfg)
        snap = prime_trunk(cfg, data_dir, args.inject_step, meta={"instance": tag}, **extra)
        save(snap, snap_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = json.load(open(out_path)) if os.path.exists(out_path) else {"stamp": _stamp(cfg), "branches": {}}
    steps = cfg["train"]["max_steps"] - args.inject_step
    last = args.inject_step + steps - 1
    horizons = sorted({min(args.inject_step + h - 1, last) for h in args.horizons} | {last})
    val = heldout_steps(data_dir, cfg["train"]["batch_size"], cfg["model"]["block_size"])
    res["val_slots"], res["horizons"] = val, horizons

    extra: dict = {}  # per-branch metrics filled by hooks (x_bar, lr factor, edit fraction)

    def run(name, br):
        if name in res["branches"]:
            return
        vals: dict = {}
        t = time.time()
        r = run_branch(
            data_dir,
            snap,
            br,
            steps,
            device=device,
            seed=False,
            on_step=horizon_probe(data_dir, val, cfg["train"]["batch_size"], horizons, vals),
        )
        res["branches"][name] = {
            "val": {str(k): v for k, v in vals.items()},
            "survival": r["survival"],
            "final_train_loss": r["final"],
            "train_losses": r["losses"],
            "seconds": round(time.time() - t, 1),
            "metrics": extra.get(name, {}),
        }
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"[{tag}] {name}: val@end={vals.get(last)} survival={r['survival']}", flush=True)

    if args.battery == "repair":
        for name, br in _with_schedule(
            repair_battery(cfg, recipe, data_dir, snap, args.lag, args.detector_t0, extra), cfg
        ).items():
            run(name, br)
        return out_path

    a, log0 = phase_a(cfg, recipe, data_dir, args.taus, args.c2_taus, args.placebo, args.placebo_channels)
    b_names = _b_names(args)  # empty for lr_bump: its gradient delta is 0 by construction
    if any(n not in res["branches"] for n in b_names):
        res["branches"].pop("B0", None)  # B0 reruns in-process so its record log (G) exists for phase B
    for name, br in _with_schedule(a, cfg).items():
        run(name, br)
    if any(n not in res["branches"] for n in b_names):
        b = phase_b(cfg, recipe, log0, args.dts, args.eps_probe, args.plant_coefs, args.dose_taus, extra)
        for name, br in _with_schedule(b, cfg).items():
            run(name, br)
    return out_path


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _b_names(args) -> list[str]:
    if args.kind == "lr_bump":
        return []
    names = ["P+", "P-"] + [f"{k}@{dt}" for dt in args.dts for k in ("I_same", "I_opp", "S2+", "S2-", "R", "Ponly")]
    for c in args.plant_coefs:
        names += [f"Px{c:g}"] + [f"Px{c:g}|{ch}@{t}" for t in args.dose_taus for ch in ("U_m", "U_v", "U_mv", "cool")]
    return names


def expected_branches(args) -> list[str]:
    """Branch names one instance writes (report.py checks completion against this)."""
    if args.battery == "repair":
        return []  # ponytail: battery names come from fork.make_*branches; completion unchecked for repair
    names = ["B*", "B0"] + [f"U_mv@{t}" for t in args.taus]
    names += [f"U_{c}@{t}" for t in args.c2_taus for c in ("m", "v")]
    names += [f"Pl_{ch}#{k}" for ch in args.placebo_channels for k in range(args.placebo)]
    return names + _b_names(args)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="P1-core instance: C-1 kernel, C-2 arms, C-3/M2/M3 probes, C-4 battery")
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", required=True, choices=["corrupt_batch", "repeat_ngram", "lr_bump"])
    ap.add_argument("--inject-step", type=int, required=True)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--betas", default="", help="override, e.g. 0.8,0.95 (the beta1 arms)")
    ap.add_argument("--taus", type=_ints, default=_ints("0,1,2,4,8,16,32,64"))
    ap.add_argument("--c2-taus", type=_ints, default=_ints("0,10"))
    ap.add_argument("--dts", type=_ints, default=_ints("2,8,32"))
    ap.add_argument("--eps-probe", type=float, default=0.05)
    ap.add_argument("--horizons", type=_ints, default=_ints("50,150,500"), help="steps after inject; end always added")
    ap.add_argument("--placebo", type=int, default=0, help="K 1-ulp placebo forks of B* per channel (chaos floor)")
    ap.add_argument("--placebo-channels", type=lambda s: [c for c in s.split(",") if c], default=["w", "g"])
    ap.add_argument(
        "--plant-coefs",
        type=lambda s: [float(c) for c in s.split(",") if c],
        default=[],
        help="dose arms: plant c*G on the clean stream, e.g. 1,3,10,30",
    )
    ap.add_argument("--dose-taus", type=_ints, default=_ints("0"), help="undo lags for the dose arms")
    ap.add_argument("--battery", choices=["core", "repair"], default="core")
    ap.add_argument("--lag", type=int, default=3, help="repair battery: RML undo lag after the window")
    ap.add_argument("--detector-t0", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--shard", default="proxy")
    ap.add_argument("--out", default="results/autopsy")
    ap.add_argument("--skip-preflight", action="store_true", help="tests only; never on the GPU")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)

    if not args.skip_preflight:
        from research.harness.preflight import check_env

        check_env()  # entrypoint: pinned env + determinism before any CUDA work
    print(run_instance(args))


if __name__ == "__main__":
    main()
