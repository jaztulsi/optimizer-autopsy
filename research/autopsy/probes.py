"""Planted gradient perturbations for the system-ID probes (M2 parity, M3 refractory, M4 interference).

A plant adds coef * G to the FINAL gradient (after accumulation and the global clip), right before
`opt.step()`. Injecting post-clip makes +G and -G arms clip-matched by construction (13_reviewer_plan
§11, M2 fix). With dropout 0 a batch touches training only through its gradient, so planting
G = g_corrupt - g*_clean (from `undo.record`) on the clean stream reproduces the data spike up to fp
rounding -- the +G arm is data-derived; -G is its planted mirror (a probe, not a prevalence claim).

    G = spike_delta(log, s)                                    # from a recorded B0 run
    plant(G, {s: +1.0})                                        # M2 "+G"   (and {s: -1.0} for "-G")
    plant(G, {s: +1.0, s + dt: +1.0})                          # M4 same-sign pair (opp: -1.0 second)
    plant(G, {s: +1.0, s + dt: eps})                           # M3 spike + probe P = eps*G at s+dt
"""

from __future__ import annotations


def spike_delta(log: dict, step: int) -> dict:
    """{name: g_s - g*_s} (fp32) at a recorded window step -- the planted spike direction G."""
    if step not in log:
        raise KeyError(f"step {step} not in the record log (has {sorted(log)})")
    return {n: dm.float() for n, (dm, _) in log[step].items()}


def plant(G: dict, at: dict):
    """`pre_opt_step` adding coef * G[name] to each param's final gradient at every step in `at`
    ({step: coef}). Params absent from G are untouched."""
    at = {int(k): float(c) for k, c in at.items()}

    def hook(step, ctx):
        c = at.get(step)
        if c is None:
            return None
        for n, p in ctx["model"].named_parameters():
            if n in G and p.grad is not None:
                p.grad.add_(G[n].to(p.grad.device, p.grad.dtype), alpha=c)
        return None

    return hook


def planted_clean_fn(G: dict, at: dict):
    """`clean_fn` for `undo.as_record_hook` on a planted run: g* = the gradient minus the plant.
    Chain it AFTER `plant` so it sees the planted gradient."""
    at = {int(k): float(c) for k, c in at.items()}

    def clean(step, ctx):
        c = at[step]
        return {
            n: p.grad.detach() - (c * G[n].to(p.grad.device, p.grad.dtype) if n in G else 0)
            for n, p in ctx["model"].named_parameters()
            if p.grad is not None
        }

    return clean


def dose_meter(G_by_step: dict, coef: float, out: dict):
    """Read-only `pre_opt_step` (chain it AFTER the plant): the pre-registered M5 dose (PLAN_V8 §2)
    x_bar = sum_i |u_i| x_i / sum_i |u_i|, x_i = (1-b2)(coef G_i)^2 / v_bar_i, v_bar = v before the spike
    step, u = that step's weight update; summed over the window. Writes out["xbar"]."""

    def hook(step, ctx):
        G = G_by_step.get(step)
        if G is None:
            return None
        opt, b2 = ctx["opt"], ctx["opt"].param_groups[0]["betas"][1]
        pre = {}
        for n, p in ctx["model"].named_parameters():
            st = opt.state.get(p, {})
            if n in G and "exp_avg_sq" in st:
                x = (1 - b2) * (coef * G[n].to(p.device)) ** 2 / st["exp_avg_sq"].clamp_min(1e-30)
                pre[n] = (p, p.detach().clone(), x)

        def after():
            for p, w0, x in pre.values():
                u = (p.detach() - w0).abs()
                out["num"] = out.get("num", 0.0) + float((u * x).sum())
                out["den"] = out.get("den", 0.0) + float(u.sum())
            out["xbar"] = out["num"] / out["den"] if out.get("den") else None

        return after

    return hook


def nudge_ulp(tensors, seed: int) -> None:
    """In place: every element moves exactly 1 ulp, up or down by a coin from `seed` (a CPU generator, so
    the signs are identical on any device). The chaos-floor placebo's whole perturbation."""
    import torch

    g = torch.Generator().manual_seed(seed)
    for t in tensors:
        up = torch.randint(0, 2, t.shape, generator=g).bool().to(t.device)
        inf = torch.tensor(float("inf"), device=t.device, dtype=t.dtype)
        t.copy_(torch.nextafter(t, torch.where(up, inf, -inf)))


def ulp_placebo(seed: int, at: int, channel: str):
    """Hook nudging every weight (channel "w": use as a `pre_step`) or every final gradient (channel "g":
    use as a `pre_opt_step`) by 1 ulp at step `at`, on the clean stream. K seeds -> sigma_chaos(h)."""
    import torch

    def hook(step, ctx):
        if step == at:
            with torch.no_grad():
                ts = [p if channel == "w" else p.grad for _, p in ctx["model"].named_parameters()]
                nudge_ulp([t for t in ts if t is not None], seed)
        return None

    return hook


def chain_opt(*hooks):
    """Compose `pre_opt_step` hooks left-to-right (None dropped); their returned `undo()` callables
    all run after `opt.step()`, last-in first-out. All-None -> None."""
    hooks = [h for h in hooks if h is not None]
    if not hooks:
        return None

    def hook(step, ctx):
        undos = [u for u in (h(step, ctx) for h in hooks) if u is not None]
        if not undos:
            return None

        def undo_all():
            for u in reversed(undos):
                u()

        return undo_all

    return hook
