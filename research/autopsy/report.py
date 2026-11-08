"""Report over mve instance JSONs, blinded per PLAN_V8 §7.2 until prereg-v2 is tagged.

    python -m research.autopsy.report runs/t1_pilot runs/t1_beta --out runs/t1_report

Writes into <root>: report.md + report.json (INSTRUMENT only: completion, survival, P+ vs B0 parity,
runtime, chaos floors, the zero-perturbation replay check), sealed.json (effects: C-1 kernel D(tau) + removed
share, C-2 arms, M4 interference, M3 chi, effects in chaos units d) and chaos_forks.jsonl (one row per
placebo / data-order fork per horizon: the chaos-floor release). sealed.json is never printed.
D(X)@h = val(X)@h - val(B*)@h at each probe horizon h. Safe on partial JSONs. An instance without its own
placebos borrows the w-channel floor of a chaos run forked from the same trunk (same cfg, same inject step).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics

from research.autopsy.mve import expected_branches, parser


def _num(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def _instance(res: dict, sig_fallback=None) -> tuple[dict, dict]:
    args = parser().parse_args(res["stamp"]["argv"][1:])
    br, hs = res["branches"], [str(h) for h in res["horizons"]]

    def D(name, h):
        try:
            return _num(br[name]["val"][h] - br["B*"]["val"][h])
        except (KeyError, TypeError):
            return None

    def op(f, *xs):
        return None if any(x is None for x in xs) else _num(f(*xs))

    exp = expected_branches(args)
    parity = [
        op(lambda a, b: abs(a - b), br.get("P+", {}).get("val", {}).get(h), br.get("B0", {}).get("val", {}).get(h))
        for h in hs
    ]
    secs = [b["seconds"] for b in br.values() if "seconds" in b]
    inst = {
        "kind": args.kind,
        "seed": res["stamp"]["cfg"].get("seed"),
        "done": len(br),
        "expected": len(exp),
        "missing": [n for n in exp if n not in br],
        "non_survivors": [n for n, b in br.items() if not b["survival"]],
        "parity_P+_vs_B0_max_abs_val": max((p for p in parity if p is not None), default=None),
        "sec_per_branch_mean": round(sum(secs) / len(secs), 1) if secs else None,
        "Bstar_val": {h: br.get("B*", {}).get("val", {}).get(h) for h in hs},
        "c2_mv_arm_present": {t: f"U_mv@{t}" in exp for t in args.c2_taus},
        "betas": res["stamp"]["cfg"]["optim"]["betas"],
    }
    sealed = {}
    for h in hs:
        d0 = D("B0", h)
        s = {"D_B0": d0, "kernel": {}, "removed_share": {}, "c2": {}, "interference": {}, "chi": {}}
        for t in args.taus:
            k = D(f"U_mv@{t}", h)
            s["kernel"][t] = k
            s["removed_share"][t] = op(lambda a, b: 1 - a / b, k, d0) if d0 else None
        s["indirect_share"] = s["removed_share"].get(0)  # 1 - D(U_mv@0)/D(B0): share carried by m, v
        for t in args.c2_taus:
            s["c2"][t] = {"none": d0, "m": D(f"U_m@{t}", h), "v": D(f"U_v@{t}", h), "mv": D(f"U_mv@{t}", h)}
        p = D("P+", h)
        for dt in args.dts:
            s["interference"][dt] = {
                "same": op(lambda a, b, c: a - b - c, D(f"I_same@{dt}", h), p, D(f"S2+@{dt}", h)),
                "opp": op(lambda a, b, c: a - b - c, D(f"I_opp@{dt}", h), p, D(f"S2-@{dt}", h)),
            }
            s["chi"][dt] = op(lambda r, a, q: (r - a) / q if q else None, D(f"R@{dt}", h), p, D(f"Ponly@{dt}", h))
        sealed[h] = s

    # chaos floor (instrument): SD over the K placebo forks of D at each horizon, per channel; "data" is the
    # data-order spread it is compared against. Pl_0 (no nudge) must replay B* exactly.
    prefixes = {"w": "Pl_w#", "g": "Pl_g#", "opt": "Pl_opt#", "data": "Dord#"}
    pl = {ch: [n for n in br if n.startswith(p)] for ch, p in prefixes.items()}
    floor = {ch: {h: _sd([D(n, h) for n in ns]) for h in hs} for ch, ns in pl.items() if ns}
    inst["sigma_chaos"] = floor
    if "Pl_0" in br:
        z = [abs(x) for x in (D("Pl_0", h) for h in hs) if x is not None]
        inst["zero_pert_max_abs_D"] = max(z, default=None)
        inst["zero_pert_exact"] = bool(z) and max(z) == 0.0
    end = hs[-1] if hs else None
    sig, src = floor.get("w", {}).get(end), "own"  # prereg-v2 effect unit: w-channel floor at the end horizon
    if sig is None:
        sig, src = sig_fallback, ("borrowed" if sig_fallback is not None else None)
    inst["sigma_unit"] = {"value": sig, "source": src}

    def d(x):
        return op(lambda a: a / sig, x) if sig else None

    if "B0" in br and end is not None:  # spike damage and the C-2 arms in chaos units, end horizon
        sealed["_d_end"] = {
            "B0": d(D("B0", end)),
            "c2": {t: {a: d(x) for a, x in sealed[end]["c2"][t].items()} for t in args.c2_taus},
        }

    def seconds(n):
        return br.get(n, {}).get("seconds")

    # dose triage (sealed): c -> x_bar (M5) -> d(none); plus the arms at each dose
    for c in args.plant_coefs:
        p = f"Px{c:g}"
        arms = {"none": D(p, end)}
        arms |= {f"{a}@{t}": D(f"{p}|{a}@{t}", end) for t in args.dose_taus for a in ("U_m", "U_v", "U_mv", "cool")}
        t0 = args.dose_taus[0] if args.dose_taus else None
        sealed.setdefault("_dose", {})[f"{c:g}"] = {
            "xbar": br.get(p, {}).get("metrics", {}).get("xbar"),
            "D": arms,
            "d": {a: d(x) for a, x in arms.items()},
            "indirect_share": op(lambda a, b: 1 - a / b if b else None, arms.get(f"U_mv@{t0}"), arms["none"]),
            "cool_min_lr_factor": br.get(f"{p}|cool@{t0}", {}).get("metrics", {}).get("min_lr_factor"),
        }
    # repair battery (sealed): closure of D(B0), extra seconds vs B0, edit fraction
    if args.battery == "repair":
        d0 = D("B0", end)
        sealed["_repair"] = {
            n: {
                "D": D(n, end),
                "d": d(D(n, end)),
                "closes": op(lambda a, b: 1 - a / b if b else None, D(n, end), d0),
                "extra_seconds_vs_B0": op(lambda a, b: a - b, seconds(n), seconds("B0")),
                **br[n].get("metrics", {}),
            }
            for n in br
            if n not in ("B*", "B0")
        }
    return inst, sealed


def _sd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else None


def _floor_key(res: dict) -> tuple:
    """A placebo floor transfers only between instances forked from the same trunk at the same step."""
    cfg = {k: v for k, v in res["stamp"]["cfg"].items() if k != "data"}
    return parser().parse_args(res["stamp"]["argv"][1:]).inject_step, json.dumps(cfg, sort_keys=True)


def _fork_rows(tag: str, res: dict):
    """chaos_forks.jsonl rows: every placebo / data-order fork at every horizon, val in nats/token."""
    args = parser().parse_args(res["stamp"]["argv"][1:])
    bstar = res["branches"].get("B*", {}).get("val", {})
    for name, b in res["branches"].items():
        if name == "Pl_0":
            arm, k = "zero", 0
        elif name.startswith(("Pl_", "Dord#")):
            arm, k = ("data", name[5:]) if name.startswith("Dord#") else name[3:].split("#")
        else:
            continue
        for h, x in b["val"].items():
            y = bstar.get(h)
            yield {
                "instance": tag,
                "inject_step": args.inject_step,
                "seed": res["stamp"]["cfg"].get("seed"),
                "betas": res["stamp"]["cfg"]["optim"]["betas"],
                "arm": arm,
                "fork": int(k),
                "horizon": int(h),
                "val_nats": x,
                "D": _num(x - y) if _num(x) is not None and _num(y) is not None else None,
            }


def main(roots: list[str], out: str, unseal: bool = False) -> None:
    out_names = {"report.json", "sealed.json"}
    files = sorted(
        f
        for root in roots
        for f in glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
        if os.path.basename(f) not in out_names
    )
    runs = {}
    for f in files:
        res = json.load(open(f))
        if "branches" in res:
            runs[os.path.basename(f)[: -len(".json")]] = res
    floors: dict = {}  # own w-channel end floors, lent to same-trunk instances that ran no placebos
    for res in runs.values():
        sig = _instance(res)[0]["sigma_unit"]["value"]
        if sig is not None:
            floors.setdefault(_floor_key(res), sig)
    inst, sealed, beta = {}, {}, {}
    for tag, res in runs.items():
        inst[tag], sealed[tag] = _instance(res, floors.get(_floor_key(res)))
        hz = [h for h in sealed[tag] if h.isdigit()]
        end = sealed[tag][max(hz, key=int)] if hz else {}
        key = f"{inst[tag]['kind']}_seed{inst[tag]['seed']}_b2={inst[tag]['betas'][1]}"
        beta.setdefault(key, {})[inst[tag]["betas"][0]] = end.get("indirect_share")
    # beta1 arms (Fig 2): indirect share at the end horizon per beta1, and its ordering -- sealed
    sealed["_beta_arms"] = {
        k: {
            "indirect_share_by_beta1": v,
            "beta1_by_indirect_share": sorted((b for b in v if v[b] is not None), key=v.get),
        }
        for k, v in beta.items()
        if len(v) > 1
    }
    os.makedirs(out, exist_ok=True)
    json.dump(inst, open(os.path.join(out, "report.json"), "w"), indent=1)
    json.dump(sealed, open(os.path.join(out, "sealed.json"), "w"), indent=1)
    with open(os.path.join(out, "chaos_forks.jsonl"), "w") as fh:
        for tag, res in runs.items():
            for row in _fork_rows(tag, res):
                fh.write(json.dumps(row) + "\n")
    rows = [
        "| instance | done/exp | non-survivors | P+ vs B0 max abs val | s/branch | B* val@end |",
        "|---|---|---|---|---|---|",
    ]
    for tag, r in inst.items():
        end = list(r["Bstar_val"].values())[-1] if r["Bstar_val"] else None
        rows.append(
            f"| {tag} | {r['done']}/{r['expected']} | {r['non_survivors'] or '-'} | "
            f"{r['parity_P+_vs_B0_max_abs_val']} | {r['sec_per_branch_mean']} | {end} |"
        )
    rows += [
        "",
        "| instance | sigma_chaos @end: w | g | opt | data-order | zero-pert max abs D | sigma unit |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, r in inst.items():
        sc = r["sigma_chaos"]
        cells = [list(sc[ch].values())[-1] if ch in sc else "-" for ch in ("w", "g", "opt", "data")]
        unit = f"{r['sigma_unit']['value']} ({r['sigma_unit']['source']})" if r["sigma_unit"]["source"] else "-"
        cells += [r.get("zero_pert_max_abs_D", "-"), unit]
        rows.append(f"| {tag} | " + " | ".join(map(str, cells)) + " |")
    if unseal:  # only after the prereg tag for this run exists
        rows += [
            "",
            "## Triage (sealed until tagged): dose c -> x_bar (M5) -> d = D / sigma_chaos",
            "",
            "| instance | c | x_bar | d(none) | indirect share |",
            "|---|---|---|---|---|",
        ]
        for tag, s in sealed.items():
            for c, v in s.get("_dose", {}).items() if isinstance(s, dict) else ():
                rows.append(f"| {tag} | {c} | {v['xbar']} | {v['d']['none']} | {v['indirect_share']} |")
    md = "# Instrument report (effects sealed in sealed.json until the prereg tag)\n\n" + "\n".join(rows) + "\n"
    open(os.path.join(out, "report.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="run dirs searched recursively for instance JSONs")
    ap.add_argument("--out", help="where report.md/report.json/sealed.json go (default: first root)")
    ap.add_argument("--unseal", action="store_true", help="add the triage table to report.md (after the prereg tag)")
    a = ap.parse_args()
    main(a.roots, a.out or a.roots[0], a.unseal)
