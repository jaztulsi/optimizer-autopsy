"""RML-2 probation: withhold a suspicious gradient's direction, let later gradients judge it, restore what they confirm.

Spec: docs/novelty/method.md §2.1, §2.5 and §3; PLAN_V8 §6.5 (the RML family). One `pre_opt_step` hook, run on the
final (accumulated, clipped) gradient with the pre-update m and v:

  arm         x_i = (1-b2)(g_i - m_i)^2 / v_i ; support S = {i : x_i >= x_arm}; receipt R = (s, S, g_S, m_S, v_S)
  quarantine  feed Adam g := m on S (direction withheld); after opt.step(), v_S += (1-b2)(g_S^2 - m_S^2) (scale in)
  evidence    at age k: y_k = sum_S (g_now - R.m)(R.g - R.m) / R.v ; Q = sum_S (R.g - R.m)^2 / R.v
  verdict     t = sum y / sqrt(kQ). Decided when |t| >= level, or at k = K.
                phi_hat = mean y / Q, se = 1/sqrt(kQ); phi = clip(phi_hat - level*se, 0, 1)
                t <= -level: "unstable" (phi = 0, scale kept); phi > 0: "confirm"; phi = 0: "refute" (exact skip)
  restore     m_S += phi * e_m and v_S -= (1-phi) * e_v, aged to the step it fires at, through `undo.undo`
              (e_m = (1-b1)(g - m), e_v = (1-b2)(g^2 - v), the pre-spike m and v)

Corners are exact (method.md §2.1): phi = 0 leaves the moments a run that skipped step s on S would hold, phi = 1 the
run that kept it, given the gradients seen since (open loop; Adam's step counter stays one ahead of a true skip).
Installed but never armed, the hook is a bitwise no-op: the null-repair identity.

The weights in Q, y and x are 1/v with the raw `exp_avg_sq`, not bias-corrected. Mid-training the correction is 1.

RML-3 (row Appeal, docs/novelty/rml3_method.md) restores through the same `undo.undo`, with the entry
(g0 - g_phi, g0^2 - g_phi^2). Its flagging and shadow code is not written: §8.3 of that doc holds it until the kill
test passes.

Hooks are plain closures (`fork.run_branch` deep-copies hooks; a closure keeps its state, a bound method would not).
"""

from __future__ import annotations

import math

from research.autopsy.undo import undo

_TINY = 1e-300


def _flat(t):
    return t.detach().reshape(-1).double()


def arm(model, opt, step: int, x_arm: float) -> dict | None:
    """Receipt for the coordinates with x_i >= x_arm, or None if there are none. Reads only."""
    import torch

    if not x_arm > 0:
        raise ValueError(f"x_arm must be > 0, got {x_arm}")
    b2 = opt.param_groups[0]["betas"][1]
    params, Q = {}, 0.0
    for name, p in model.named_parameters():
        st = opt.state.get(p, {})
        if p.grad is None or "exp_avg_sq" not in st:
            continue
        g, m, v = _flat(p.grad), _flat(st["exp_avg"]), _flat(st["exp_avg_sq"])
        x = torch.where(v > 0, (1 - b2) * (g - m).square() / v.clamp_min(_TINY), torch.zeros_like(v))
        idx = (x >= x_arm).nonzero().flatten()
        if idx.numel():
            gs, ms, vs = g[idx], m[idx], v[idx]
            params[name] = (idx, gs, ms, vs)
            Q += float(((gs - ms).square() / vs).sum())
    return {"step": step, "params": params, "Q": Q, "ys": []} if params else None


def quarantine(model, opt, R: dict):
    """Feed Adam g := m on S; return the `undo()` that admits the scale after `opt.step()`."""
    import torch

    named = dict(model.named_parameters())
    for name, (idx, _, ms, _) in R["params"].items():
        g = named[name].grad
        g.view(-1).index_copy_(0, idx, ms.to(g.dtype))

    def admit_scale():
        b2 = opt.param_groups[0]["betas"][1]
        with torch.no_grad():
            for name, (idx, gs, ms, _) in R["params"].items():
                v = opt.state[named[name]]["exp_avg_sq"].view(-1)
                new = (v[idx].double() + (1 - b2) * (gs.square() - ms.square())).clamp_min_(0)
                v.index_copy_(0, idx, new.to(v.dtype))

    return admit_scale


def evidence(model, R: dict) -> float:
    """y_k for the current final gradient."""
    named = dict(model.named_parameters())
    y = 0.0
    for name, (idx, gs, ms, vs) in R["params"].items():
        y += float(((_flat(named[name].grad)[idx] - ms) * (gs - ms) / vs).sum())
    return y


def verdict(R: dict, K: int, level: float) -> tuple[str, float] | None:
    """(name, phi) once decided, else None."""
    k, Q = len(R["ys"]), R["Q"]
    t = sum(R["ys"]) / math.sqrt(k * Q)
    if abs(t) < level and k < K:
        return None
    if t <= -level:
        return "unstable", 0.0
    phi = min(1.0, max(0.0, sum(R["ys"]) / (k * Q) - level / math.sqrt(k * Q)))
    return ("confirm" if phi > 0 else "refute"), phi


def close(model, opt, R: dict, phi: float, keep_scale: bool, at_step: int) -> None:
    """Give back phi of the direction, remove (1 - phi) of the scale (none if keep_scale), before `at_step`'s update."""
    import torch

    entry = {}
    for name, p in model.named_parameters():
        if name not in R["params"]:
            continue
        idx, gs, ms, vs = R["params"][name]
        dm = torch.zeros(p.numel(), dtype=torch.float64, device=gs.device)
        dv = torch.zeros_like(dm)
        dm[idx] = -phi * (gs - ms)  # undo subtracts, so a negative entry adds the direction back
        if not keep_scale:
            dv[idx] = (1 - phi) * (gs.square() - vs)
        entry[name] = (dm.view(p.shape), dv.view(p.shape))
    undo(model, opt, {R["step"]: entry}, at_step, "mv")


def as_pre_opt_step(x_arm: float, K: int = 3, level: float = 3.0, arm_steps=None, out: dict | None = None):
    """The RML-2 `pre_opt_step`. `arm_steps`: steps it may arm at (None = any). `out` collects
    `out["verdicts"]` = [(s, closed_at, name, phi, |S|)] and `out["open"]` (the open receipts)."""
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    st = out if out is not None else {}
    st.update(open=[], verdicts=[], last=None)

    def hook(step, ctx):
        if st["last"] is not None and step <= st["last"]:  # a reused branch starting a new run
            st["open"].clear()
            st["verdicts"].clear()
        st["last"] = step
        model, opt = ctx["model"], ctx["opt"]
        for R in list(st["open"]):
            R["ys"].append(evidence(model, R))
            v = verdict(R, K, level)
            if v is not None:
                name, phi = v
                close(model, opt, R, phi, keep_scale=name == "unstable", at_step=step)
                st["open"].remove(R)
                st["verdicts"].append((R["step"], step, name, phi, sum(len(t[0]) for t in R["params"].values())))
        # ponytail: one receipt at a time, so a 2nd event within K steps is missed; per-coordinate exclusion if needed
        if st["open"] or (arm_steps is not None and step not in arm_steps):
            return None
        R = arm(model, opt, step, x_arm)
        if R is None:
            return None
        st["open"].append(R)
        return quarantine(model, opt, R)

    return hook
