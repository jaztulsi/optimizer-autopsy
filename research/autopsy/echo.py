"""X4 echo checks (docs/novelty/method.md §6.2): plant a spike of size z on one tensor at a fork's first step and
measure how far it moves that tensor over the fork, against the echo law (analysis/damage_law.py).

The plant is G = z · sqrt(v_bar) · s on every entry of `target`: s a fixed random sign, v_bar the raw exp_avg_sq
before the step (Adam's bias correction goes into the logged step sizes). It is added to the final gradient, after
the trunk's clip, as autopsy/probes.plant does. Each (arm, z) runs as two forks, +G and -G. Each logs
d = mean(s · (w_end - w_start)) over `target`, and the echo is (d- - d+) / 2 (a step moves w against G), which
cancels the stall and the drift (both even in G). The arms:
  none    nothing: (a, b) = (1, 1), predicted Ψ(z)
  clip    at the spike step, the whole gradient scaled back to its pre-plant norm (a clip at the normal size, as
          ZClip and AdaGC aim for): (c, c²) with c logged, predicted Ψ(cz)
  spam    SPAM's spike clip every step (baselines/spam.spike_clip_, θ = 5000): z -> min(z, sqrt(θ))
  stable  StableAdamW's update clipping every step (Wortsman et al. 2023, arXiv 2304.13013, Alg. 2: per tensor,
          lr / max(1, RMS), RMS = sqrt(mean(g² / u)), u the debiased second moment after this step's update, weight
          decay scaled too): predicted Ψ(z), the law saying it trims only the spike step (<= 8%)
  undo_v  the spike never enters v, its own step included: (1, 0), predicted E(1, 0), about z
X4's row names Ψ(z) and Ψ(cz); for undo_v the law's own point (1, 0) is the prediction.
"""

from __future__ import annotations

import math

import torch

from research.analysis.damage_law import echo
from research.baselines import spam

ARMS = ("none", "clip", "spam", "stable", "undo_v")
TARGET = "transformer.h.0.attn.c_attn.weight"


def name(arm: str, z: float, sign: int) -> str:
    return f"X4_{arm}_z{z:g}{'+' if sign > 0 else '-'}"


def signs(shape, seed: int = 0) -> torch.Tensor:
    """The plant's fixed random sign s, the same in the +G and -G forks."""
    return torch.randint(0, 2, tuple(shape), generator=torch.Generator().manual_seed(seed)).float() * 2 - 1


def _norm(grads) -> float:
    return math.sqrt(sum(float(g.double().square().sum()) for g in grads))


def make(arm: str, z: float, sign: int, at: int, steps: int, target: str = TARGET, seed: int = 0):
    """(pre_opt_step, out) for one X4 fork: the spike at step `at`, the fork `steps` long. out is JSON-safe: lrs
    (Adam's step size on `target` at each step, bias correction included), c (clip), rms (stable: `target`'s at the
    spike step) and, after the last step, d."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    last = at + steps - 1
    out = {"arm": arm, "z": z, "sign": sign, "lrs": []}
    st = {}

    def hook(step, ctx):
        model, opt = ctx["model"], ctx["opt"]
        p = dict(model.named_parameters())[target]
        group = next(g for g in opt.param_groups if any(q is p for q in g["params"]))
        b1, b2 = group["betas"]
        t = float(opt.state[p]["step"]) + 1  # Adam's count after this update
        if step == at:
            out["lrs"].clear()  # in place: extra holds this dict
        out["lrs"].append(group["lr"] * math.sqrt(1 - b2**t) / (1 - b1**t))
        post = []
        with torch.no_grad():
            if step == at:
                v = opt.state[p]["exp_avg_sq"]
                s = signs(p.shape, seed).to(p.device)  # unsigned: d is read against s in both forks
                G = (z * sign) * v.sqrt() * s
                grads = [q.grad for q in model.parameters() if q.grad is not None]
                n0 = _norm(grads)
                if arm == "undo_v":  # pre-compensate: this step's v = b2 v + (1 - b2) g² then holds no G
                    v.sub_(((p.grad + G).square() - p.grad.square()) * ((1 - b2) / b2))
                p.grad.add_(G)
                if arm == "clip":
                    out["c"] = min(1.0, n0 / _norm(grads))
                    for g in grads:
                        g.mul_(out["c"])
                st["s"], st["w0"] = s, p.detach().clone()
            if arm == "spam":
                spam.spike_clip_(opt)
            if arm == "stable":
                scale = {}
                for grp in opt.param_groups:
                    gb2 = grp["betas"][1]
                    for q in grp["params"]:
                        if q.grad is None or "exp_avg_sq" not in opt.state.get(q, {}):
                            continue
                        tq = float(opt.state[q]["step"]) + 1
                        u = (gb2 * opt.state[q]["exp_avg_sq"] + (1 - gb2) * q.grad.square()) / (1 - gb2**tq)
                        rms = float((q.grad.square() / u.clamp_min(1e-30)).mean().sqrt())  # 0/0 -> 0 where g = 0
                        if q is p and step == at:
                            out["rms"] = rms
                        if rms > 1:
                            scale[q] = (1 / rms, q.detach().clone())
                post.append(lambda: [q.copy_(w0 + (q - w0) * f) for q, (f, w0) in scale.items()])
            if step == last:
                post.append(lambda: out.__setitem__("d", float(((p.double() - st["w0"].double()) * st["s"]).mean())))

        def undo():
            with torch.no_grad():
                for f in post:
                    f()

        return undo if post else None

    return hook, out


def branches(cfg: dict, at: int, zs, steps: int, extra: dict, target: str = TARGET, seed: int = 0) -> dict:
    """Every arm at every z, +G and -G, as mve Branches; each fork's log is extra[name] (mve's branch metrics)."""
    from research.harness.fork import Branch

    out = {}
    for arm in ARMS:
        for z in zs:
            for sign in (1, -1):
                n = name(arm, z, sign)
                hook, extra[n] = make(arm, z, sign, at, steps, target, seed)
                out[n] = Branch(n, None, cfg, hook)
    return out


def summarize(branches: dict, betas, theta: float = spam.THETA) -> dict:
    """Per (arm, z) with both forks done: the echo (d- - d+) / 2, the law's prediction for the arm, their ratio, and
    `left`, the arm's echo over the none arm's at that z. The verdict reads X4's row (method.md §6.2): pass if every
    ratio is within ±20% and SPAM leaves >= 90% at z >= 30, fail if any is off by more than 40%, else neither."""
    m = {k: b["metrics"] for k, b in branches.items() if k.startswith("X4_")}
    cells = {}
    for k, plus in m.items():
        if not k.endswith("+") or k[:-1] + "-" not in m:
            continue
        minus, arm, z, lrs = m[k[:-1] + "-"], plus["arm"], plus["z"], plus["lrs"]
        if arm == "clip":
            c = (plus["c"] + minus["c"]) / 2
            pred = echo(z, c, c * c, betas, lrs)
        elif arm == "undo_v":
            pred = echo(z, 1.0, 0.0, betas, lrs)
        else:  # spam's clip only bites above sqrt(theta); stable is predicted to leave the echo alone
            pred = echo(min(z, math.sqrt(theta)) if arm == "spam" else z, betas=betas, lrs=lrs)
        d = (minus["d"] - plus["d"]) / 2  # Adam steps against the gradient: +G moves w by -E s
        cells[k[3:-1]] = {"arm": arm, "z": z, "echo": d, "pred": pred, "ratio": d / pred}
    for c in cells.values():
        none = cells.get(f"none_z{c['z']:g}")
        c["left"] = c["echo"] / none["echo"] if none else None
    off = [abs(c["ratio"] - 1) for c in cells.values()]
    spam_ok = all((c["left"] or 0) >= 0.9 for c in cells.values() if c["arm"] == "spam" and c["z"] >= 30)
    verdict = "fail" if any(o > 0.4 for o in off) else "pass" if all(o <= 0.2 for o in off) and spam_ok else "neither"
    return {"cells": cells, "verdict": verdict}
