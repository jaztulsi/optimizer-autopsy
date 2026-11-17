"""GPU driver for one P1-core instance (PLAN_V8 §2.6 skeleton; §7.0 C-1..C-4). Run on the AMD MI300X only.

    python -m research.autopsy.mve --config research/experiments/proxy/config.yaml \
        --kind corrupt_batch --inject-step 160 --out results/autopsy
    python -m research.autopsy.mve ... --battery repair --lag 3          # C-4 repair battery
    python -m research.autopsy.mve --config ... --battery chaos --inject-step 1000 \
        --placebo 111 --placebo-channels w,g,opt --data-forks 8      # chaos floor only (docs/chaos_floor.md)

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

Fault kinds (sdc_grad, v_bitflip, m_bitflip): B* (the honest floor) and B0 also store faults.parity_residual
raw values under "parity" (H11). v_bitflip / m_bitflip run no U_* arms and no phase B (G - G* = 0).

Results go to <out>/<instance>.json, rewritten after every branch; a rerun skips finished branches.
Derived quantities (kernel, Shapley, interference, chi, stats) are computed offline from the JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

from research.autopsy.faults import TARGET

FAULTS = ("sdc_grad", "v_bitflip", "m_bitflip")
# These leave the gradient untouched, so G - G* = 0: nothing to plant in phase B. sdc_grad flips g itself, so it
# keeps phase B like corrupt_batch (the record hook's clean recompute doesn't re-flip: the hook fires once).
NO_GRAD_DELTA = {"lr_bump", "v_bitflip", "m_bitflip"}
# With G - G* = 0 every U_* arm subtracts exact zeros, i.e. is bitwise B0. Dropped for the m/v flips; lr_bump keeps
# them so its branch set (the pilot / prereg comparison) doesn't change.
NO_UNDO = NO_GRAD_DELTA - {"lr_bump"}


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
        # hash of the uncommitted edits to tracked files (new untracked files are not in `git diff HEAD`)
        "git_diff": hashlib.sha256(d.encode()).hexdigest()[:16] if (d := git("diff", "HEAD")) else None,
        "argv": sys.argv,
        "cfg": cfg,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "device": dev,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _resume_mismatch(old: dict, new: dict) -> list[str]:
    """Which of cfg / commit / uncommitted diff / command line differ between a result file's stamp and this run."""
    new = json.loads(json.dumps(new))  # a stored stamp has been through JSON
    bad = [k for k in ("cfg", "git_sha", "git_diff") if old.get(k) != new.get(k)]
    return bad + ["argv"] * (old["argv"][1:] != new["argv"][1:])  # argv[0] is the interpreter's view of the script


def placebo_branches(cfg, at, k, channels, data_dir=None, data_forks=0):
    """{name: Branch} for the chaos floor, all forks of B* from step `at`: `Pl_0` (no nudge: must replay B*
    exactly), `Pl_<ch>#i` K 1-ulp placebos per channel (w / g / opt -> sigma_chaos(h)) and `Dord#i`
    data-order forks (the seed-level spread)."""
    from research.autopsy.probes import data_order, ulp_placebo
    from research.harness.fork import Branch

    br = {"Pl_0": Branch("Pl_0", None, cfg)} if k else {}
    for ch in channels:
        for i in range(k):
            h = ulp_placebo(i, at, ch)
            br[f"Pl_{ch}#{i}"] = Branch(f"Pl_{ch}#{i}", None if ch == "g" else h, cfg, h if ch == "g" else None)
    for i in range(data_forks):
        br[f"Dord#{i}"] = Branch(f"Dord#{i}", data_order(i, at, cfg, data_dir), cfg)
    return br


def phase_a(cfg, recipe, data_dir, taus, c2_taus, placebo=0, channels=(), data_forks=0):
    """{name: Branch} for the data-spike arms, plus the B0 record log (filled when B0 runs), plus the
    chaos-floor forks of `placebo_branches`."""
    from research.autopsy import undo
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
    br.update(placebo_branches(cfg, recipe.inject_step, placebo, channels, data_dir, data_forks))
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
    1% of coordinates by x_i (docs/prereg_v2.md H-E, H-F). Edit stats land in extra[name].
    `detector_t0` defaults to the step the same detector lag implies (window end + lag), so the detector-timed
    baselines (skip-step, rollback) act on the same information as RML instead of dropping out of the battery.
    `snap["rollback_snapshot"]` (set by run_instance) adds Brollback_k."""
    from research.autopsy import undo
    from research.harness.fork import Branch, compose, make_baseline_branches, make_branches

    window = list(range(recipe.inject_step, recipe.inject_step + recipe.width))
    at = undo.undo_step(window[-1], lag)
    if detector_t0 is None:
        detector_t0 = at - 1  # the detector saw this step's loss; RML acts at the top of the next one
    clip = cfg["optim"].get("grad_clip") or 1.0
    br = make_branches(cfg, recipe, clip_norm=0.5 * clip)
    priming = snap["meta"].get("baseline_priming", {})
    rollback_snapshot = snap.get("rollback_snapshot")
    br.update(
        make_baseline_branches(cfg, recipe, detector_t0=detector_t0, rollback_snapshot=rollback_snapshot, **priming)
    )
    extra = {} if extra is None else extra

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


def _prime_parity(on_step, at: int):
    """`pre_step`, composed BEFORE the fault: at the fork's first step `at`, hand parity_residual's `on_step` the
    restored m, v as its previous copy. Without it the flip at `at` has nothing to compare against and reads
    clean. parity_residual skips grad-less params, so each gets a zero grad for the call, then None again
    (train_forward's zero_grad would drop it anyway); w, m, v are untouched."""
    import torch

    def pre_step(step, ctx):
        if step != at:
            return
        bare = [p for p in ctx["model"].parameters() if p.grad is None]
        for p in bare:
            p.grad = torch.zeros_like(p)
        on_step(at - 1, ctx)
        for p in bare:
            p.grad = None

    return pre_step


def _parity_summary(pr: dict, inject: int) -> dict:
    """Raw residuals for steps inject-2..inject+10 (those the fork ran) and the max over every other step. The
    flag threshold belongs to the analysis, not the driver."""
    win = range(inject - 2, inject + 11)
    rest = [r for s, r in pr.items() if s not in win]
    return {
        "window": {str(s): pr[s] for s in win if s in pr},
        "max_outside": {k: max((r[k] for r in rest), default=None) for k in ("m", "v")},
    }


def _undo_taus(args) -> tuple:
    """(taus, c2_taus) for the U_* arms; none for NO_UNDO kinds."""
    return ((), ()) if args.kind in NO_UNDO else (args.taus, args.c2_taus)


def run_instance(args) -> str:
    from dataclasses import replace

    import torch

    from research.analysis.eval import heldout_steps, horizon_probe
    from research.autopsy.faults import parity_residual
    from research.baselines.rollback import ROLLBACK_K
    from research.configs import load_config
    from research.data.prepare import resolve_data_dir
    from research.harness.determinism import seed_everything
    from research.harness.fork import Branch, compose, prime_trunk, run_branch
    from research.harness.snapshot import load, save
    from research.spikes.induce import induce

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.betas:
        cfg["optim"]["betas"] = [float(b) for b in args.betas.split(",")]
    data_dir = args.data_dir or cfg.get("data", {}).get("dir") or resolve_data_dir(args.shard)
    params = {"data_dir": data_dir} if args.kind == "repeat_ngram" else {}
    kind = args.kind or "chaos"
    if args.kind in FAULTS:
        if args.fault_bit is None:  # no default: bit 30 zeroes a whole sdc_grad step on GPU (faults.py docstring)
            raise SystemExit(f"--kind {args.kind} needs --fault-bit (0 = mantissa LSB, 30 = exponent top)")
        params = {"target": args.fault_target, "index": args.fault_index, "bit": args.fault_bit}
        kind += f"-{args.fault_target}-i{args.fault_index}-bit{args.fault_bit}"  # one file per flipped bit
    recipe = induce(args.kind, cfg, args.inject_step, args.width, **params) if args.kind else None

    b1, b2 = cfg["optim"]["betas"]
    tag = f"{kind}_w{args.width}_inj{args.inject_step}_seed{cfg['seed']}_b{b1}-{b2}_{args.battery}"
    os.makedirs(args.out, exist_ok=True)
    out_path, snap_path = os.path.join(args.out, f"{tag}.json"), os.path.join(args.out, f"{tag}.snap.safetensors")
    rollback_path = os.path.join(args.out, f"{tag}.rollback.safetensors")  # snapshot.save drops Brollback_k's snapshot
    repair = args.battery == "repair"

    # An optim.schedule needs nothing here: trunk.train_forward applies it, so prime_trunk, B* and every fork get it.
    if os.path.exists(snap_path):  # resume: seed once ourselves, reuse the primed pre-spike snapshot
        seed_everything(cfg.get("seed", 1337), deterministic=True)
        snap = load(snap_path)
        if repair:
            if not os.path.exists(rollback_path):
                raise SystemExit(f"{rollback_path} is missing: Brollback_k can't run; use a new --out to prime again")
            snap["rollback_snapshot"] = load(rollback_path)
    else:
        k = ROLLBACK_K if repair else None
        snap = prime_trunk(cfg, data_dir, args.inject_step, meta={"instance": tag}, rollback_k=k)
        if repair:  # before snap_path, so a snap on disk means its rollback file is there too
            save(snap["rollback_snapshot"], rollback_path)
        save(snap, snap_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stamp = _stamp(cfg)
    if os.path.exists(out_path):  # resume: the tag doesn't encode lr / schedule / arms, and report.py reads argv
        res = json.load(open(out_path))
        if bad := _resume_mismatch(res["stamp"], stamp):
            raise SystemExit(f"{out_path} differs in {', '.join(bad)}: resume would mix runs; use a new --out")
    else:
        res = {"stamp": stamp, "branches": {}}
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
        # A rollback branch runs steps - wasted, so horizon h is probed at matched compute: step h - shift,
        # where shift = wasted (Brollback) or k + wasted (Brollback_k, which also starts k steps earlier)
        wasted = br.wasted_steps
        shift = wasted + (snap["step"] - br.snapshot["step"] if br.snapshot is not None else 0)
        on_step = horizon_probe(data_dir, val, cfg["train"]["batch_size"], [h - shift for h in horizons], vals)
        pr: dict = {}  # {step: {"m", "v"}} parity residuals: fault kinds' B* and B0 only
        if args.kind in FAULTS and name in ("B*", "B0"):
            par = parity_residual(pr)
            on_step = compose(on_step, par)
            br = replace(br, pre_step=compose(_prime_parity(par, args.inject_step), br.pre_step))
        r = run_branch(data_dir, snap, br, steps, device=device, seed=False, on_step=on_step)
        res["branches"][name] = {
            "val": {str(k + shift): v for k, v in vals.items()},
            "survival": r["survival"],
            "final_train_loss": r["final"],
            "train_losses": r["losses"],
            "seconds": round((time.time() - t) * steps / (steps - wasted), 1),
            "wasted_steps": wasted,
            "horizon_shift": shift,
            "metrics": extra.get(name, {}),
        }
        if pr:
            res["branches"][name]["parity"] = _parity_summary(pr, args.inject_step)
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"[{tag}] {name}: val@end={vals.get(last - shift)} survival={r['survival']}", flush=True)

    if args.battery == "repair":
        for name, br in repair_battery(cfg, recipe, data_dir, snap, args.lag, args.detector_t0, extra).items():
            run(name, br)
        return out_path
    if args.battery == "chaos":
        br = {"B*": Branch("B*", None, cfg)}
        br |= placebo_branches(cfg, args.inject_step, args.placebo, args.placebo_channels, data_dir, args.data_forks)
        for name, b in br.items():
            run(name, b)
        return out_path

    a, log0 = phase_a(cfg, recipe, data_dir, *_undo_taus(args), args.placebo, args.placebo_channels, args.data_forks)
    b_names = _b_names(args)  # empty for NO_GRAD_DELTA kinds: their gradient delta is 0 by construction
    if any(n not in res["branches"] for n in b_names):
        res["branches"].pop("B0", None)  # B0 reruns in-process so its record log (G) exists for phase B
    for name, br in a.items():
        run(name, br)
    if any(n not in res["branches"] for n in b_names):
        b = phase_b(cfg, recipe, log0, args.dts, args.eps_probe, args.plant_coefs, args.dose_taus, extra)
        for name, br in b.items():
            run(name, br)
    return out_path


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _b_names(args) -> list[str]:
    if args.kind in NO_GRAD_DELTA:
        return []
    names = ["P+", "P-"] + [f"{k}@{dt}" for dt in args.dts for k in ("I_same", "I_opp", "S2+", "S2-", "R", "Ponly")]
    for c in args.plant_coefs:
        names += [f"Px{c:g}"] + [f"Px{c:g}|{ch}@{t}" for t in args.dose_taus for ch in ("U_m", "U_v", "U_mv", "cool")]
    return names


def _placebo_names(args) -> list[str]:
    names = ["Pl_0"] if args.placebo else []
    names += [f"Pl_{ch}#{k}" for ch in args.placebo_channels for k in range(args.placebo)]
    return names + [f"Dord#{k}" for k in range(args.data_forks)]


def expected_branches(args) -> list[str]:
    """Branch names one instance writes (report.py checks completion against this)."""
    if args.battery == "repair":
        return []  # ponytail: battery names come from fork.make_*branches; completion unchecked for repair
    if args.battery == "chaos":
        return ["B*"] + _placebo_names(args)
    taus, c2_taus = _undo_taus(args)
    names = ["B*", "B0"] + [f"U_mv@{t}" for t in taus]
    names += [f"U_{c}@{t}" for t in c2_taus for c in ("m", "v")]
    return names + _placebo_names(args) + _b_names(args)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="P1-core instance: C-1 kernel, C-2 arms, C-3/M2/M3 probes, C-4 battery")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--kind",
        choices=["corrupt_batch", "repeat_ngram", "lr_bump", *FAULTS],
        help="spike recipe (omit for --battery chaos)",
    )
    ap.add_argument("--fault-target", default=TARGET, help="bit-flip kinds: parameter name whose g / m / v is hit")
    ap.add_argument("--fault-index", type=int, default=0, help="bit-flip kinds: flat element index")
    ap.add_argument("--fault-bit", type=int, help="bit-flip kinds (required): 0 = mantissa LSB, 30 = exponent top")
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
    ap.add_argument(
        "--placebo-channels", type=lambda s: [c for c in s.split(",") if c], default=["w", "g"], help="w / g / opt"
    )
    ap.add_argument("--data-forks", type=int, default=0, help="data-order forks of B* (same batches, new order)")
    ap.add_argument(
        "--plant-coefs",
        type=lambda s: [float(c) for c in s.split(",") if c],
        default=[],
        help="dose arms: plant c*G on the clean stream, e.g. 1,3,10,30",
    )
    ap.add_argument("--dose-taus", type=_ints, default=_ints("0"), help="undo lags for the dose arms")
    ap.add_argument("--battery", choices=["core", "repair", "chaos"], default="core")
    ap.add_argument("--lag", type=int, default=3, help="repair battery: RML undo lag after the window")
    ap.add_argument(
        "--detector-t0", type=int, default=None, help="repair battery: detector step (default: window end + --lag)"
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--shard", default="proxy")
    ap.add_argument("--out", default="results/autopsy")
    ap.add_argument("--skip-preflight", action="store_true", help="tests only; never on the GPU")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.kind is None and args.battery != "chaos":
        ap.error("--kind is required unless --battery chaos")

    if not args.skip_preflight:
        # entrypoint: pinned env + determinism before any CUDA work. In a child process, because check_env ends by
        # touching the GPU and seed_everything refuses to start in a process where CUDA is already initialized.
        subprocess.run(
            [sys.executable, "-c", "from research.harness.preflight import check_env; check_env()"], check=True
        )
    print(run_instance(args))


if __name__ == "__main__":
    main()
