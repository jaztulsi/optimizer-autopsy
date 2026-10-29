"""State-only undo of a spike window's contribution to Adam's moments (the RML core; C-1, C-2, C-4).

During the spike window, `as_record_hook` stores, per parameter, what the spiked gradient g_s wrote
into AdamW's moments beyond what the counterfactual gradient g*_s would have written:

    dm_s = g_s - g*_s            dv_s = g_s**2 - g*_s**2

Later, `undo(model, opt, log, at_step)` (fired at the TOP of step `at_step`, i.e. on the moments left by
step at_step-1) subtracts the part of that contribution still alive:

    m -= sum_s (1-b1) * b1**(at_step-1-s) * dm_s
    v -= sum_s (1-b2) * b2**(at_step-1-s) * dv_s      (clamped >= 0)

Weights and Adam's step counter are untouched, so this is exactly the moments Adam would hold had g*_s
been fed instead -- given the gradients actually seen afterwards (open loop; the closed-loop gap is
what the forks measure). Lag tau = at_step - (last window step) - 1: tau = 0 keeps only the spike step's
own update (the direct effect); tau -> inf removes nothing (D(inf) = Delta_B0).

g*_s, the counterfactual gradient:
  * data spikes (`corrupt_batch`, `repeat_ngram`): the CLEAN batch of the same data slot, at the same w,
    averaged over grad_accum and globally clipped like the trunk -- an extra forward/backward inside the
    hook that leaves model and optimizer state bitwise untouched (dropout 0, no RNG consumed);
  * planted spikes (`probes.plant`): the gradient before the plant, passed in directly;
  * `lr_bump`: the clean batch IS the batch, so dm = dv = 0 and undo is a no-op -- by design (the LR
    spike writes its damage into w directly; H6).

Hooks are plain closures, never bound methods: `fork.run_branch` deep-copies `pre_opt_step`, and a
deep-copied object would record into a copy the undo never sees.
"""

from __future__ import annotations

CHANNELS = {"m": ("exp_avg",), "v": ("exp_avg_sq",), "mv": ("exp_avg", "exp_avg_sq")}


def new_log() -> dict:
    """{step: {param_name: (dm, dv)}} -- filled by a record hook, read by `undo`."""
    return {}


def clean_final_grads(model, cfg, data_dir, step) -> dict:
    """{name: g*} -- the clean data slot's gradient at the current w, accumulated and clipped exactly
    like `trunk.train_forward`. Leaves every `p.grad` as it found it."""
    import torch

    from research.data.prepare import get_batch

    t = cfg["train"]
    ga, b, blk = t.get("grad_accum", 1), t["batch_size"], model.cfg.block_size
    dev = next(model.parameters()).device
    params = list(model.named_parameters())
    saved = [p.grad for _, p in params]
    acc = [torch.zeros_like(p) for _, p in params]
    for micro in range(ga):
        for _, p in params:
            p.grad = None
        x, y = get_batch("train", step * ga + micro, b, blk, data_dir, dev)
        _, loss = model(x, y)
        loss.backward()
        for a, (_, p) in zip(acc, params):
            if p.grad is not None:
                a.add_(p.grad)
    for a, (_, p) in zip(acc, params):
        p.grad = a.div_(ga)
    clip = cfg["optim"].get("grad_clip", 0.0)
    torch.nn.utils.clip_grad_norm_([p for _, p in params], clip if clip and clip > 0 else float("inf"))
    out = {n: p.grad.detach().clone() for n, p in params}
    for (_, p), g in zip(params, saved):
        p.grad = g
    return out


def record(log: dict, model, step: int, clean: dict) -> None:
    """Store (g - g*, g**2 - g*^2) for every param with a gradient, in fp64 (the subtraction later
    cancels large terms; fp64 keeps the residual at the fp32 rounding the run itself had)."""
    entry = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g, c = p.grad.detach().double(), clean[n].double()
        entry[n] = (g - c, g * g - c * c)
    log[step] = entry


def as_record_hook(log: dict, steps, data_dir=None, clean_fn=None):
    """`pre_opt_step` recording each step in `steps`. Data spikes: pass `data_dir` (g* = clean slot).
    Planted spikes: pass `clean_fn(step, ctx) -> {name: g*}`. The log is cleared at the first window
    step, so a reused branch never mixes runs."""
    steps = sorted(steps)
    if (data_dir is None) == (clean_fn is None):
        raise ValueError("pass exactly one of data_dir (data spikes) or clean_fn (planted spikes)")

    def hook(step, ctx):
        if step not in steps:
            return None
        if step == steps[0]:
            log.clear()
        clean = clean_fn(step, ctx) if clean_fn else clean_final_grads(ctx["model"], ctx["cfg"], data_dir, step)
        record(log, ctx["model"], step, clean)
        return None

    return hook


def undo(model, opt, log: dict, at_step: int, channels: str = "mv") -> int:
    """Subtract the logged window's surviving contribution from m and/or v, in place, at the top of
    `at_step`. Returns how many tensors were edited. Raises if the log is empty or ahead of `at_step`."""
    import torch

    if not log:
        raise RuntimeError("undo: empty log -- did the record hook run over the spike window?")
    if max(log) >= at_step:
        raise ValueError(f"undo at step {at_step} but the log has step {max(log)}: undo must come after the window")
    b1, b2 = opt.param_groups[0]["betas"]
    keys = CHANNELS[channels]
    named = dict(model.named_parameters())
    edited = 0
    with torch.no_grad():
        for name, p in named.items():
            st = opt.state.get(p, {})
            if "exp_avg" not in st:
                continue
            dm = dv = None
            for s, entry in log.items():
                if name not in entry:
                    continue
                age = at_step - 1 - s
                em, ev = entry[name]
                dm = (1 - b1) * b1**age * em + (0 if dm is None else dm)
                dv = (1 - b2) * b2**age * ev + (0 if dv is None else dv)
            if dm is None:
                continue
            if "exp_avg" in keys:
                st["exp_avg"].copy_((st["exp_avg"].double() - dm).to(st["exp_avg"].dtype))
                edited += 1
            if "exp_avg_sq" in keys:
                st["exp_avg_sq"].copy_((st["exp_avg_sq"].double() - dv).clamp_min_(0).to(st["exp_avg_sq"].dtype))
                edited += 1
    return edited


def _alive_dv(log: dict, name: str, at_step: int, b2: float):
    """v's surviving inflation for one param at the top of `at_step` (fp64), or None if not logged."""
    dv = None
    for s, entry in log.items():
        if name in entry:
            dv = (1 - b2) * b2 ** (at_step - 1 - s) * entry[name][1] + (0 if dv is None else dv)
    return dv


def undo_v_top(model, opt, log: dict, at_step: int, frac: float) -> float:
    """RML-sel's v edit: remove v's surviving inflation ONLY on the top `frac` of coordinates (global
    ranking) by x_i = (1-b2) G_i^2 / v*_i, G = the summed window delta, v* = the de-inflated v. Returns the
    edited fraction of all logged coordinates."""
    import torch

    b2 = opt.param_groups[0]["betas"][1]
    xs = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            st = opt.state.get(p, {})
            dv = _alive_dv(log, n, at_step, b2)
            if dv is None or "exp_avg_sq" not in st:
                continue
            G = sum(e[n][0] for e in log.values() if n in e)
            xs[n] = (p, dv, ((1 - b2) * G * G / (st["exp_avg_sq"].double() - dv).clamp_min(1e-30)).float())
        flat = torch.cat([x.flatten() for _, _, x in xs.values()])
        thr = torch.topk(flat, max(1, int(frac * flat.numel()))).values[-1]
        edited = 0
        for p, dv, x in xs.values():
            v, mask = opt.state[p]["exp_avg_sq"], x >= thr
            v.copy_(torch.where(mask, (v.double() - dv).clamp_min(0), v.double()).to(v.dtype))
            edited += int(mask.sum())
    return edited / flat.numel()


def as_cooldown_pre_step(log: dict, at_step: int, channels: str, out: dict):
    """`pre_step`: at `at_step` undo `channels` ("v": the H-D arm; "mv": RML-cool), then every step set ONE
    global lr = lr0 * mean_i (sqrt(v^_i)+eps) / (sqrt(v^_i + dv^_i)+eps), dv = the undone inflation still
    alive -- the step shrink the inflated v would have applied, spread uniformly over coordinates.
    A per-coordinate match would just rebuild B0; the global scalar is the test (v as a schedule?)."""
    import torch

    if "v" not in channels:
        raise ValueError("the cooldown replaces v's inflation, so channels must include v")

    def pre_step(step, ctx):
        if step < at_step:
            return
        opt, model = ctx["opt"], ctx["model"]
        g0 = opt.param_groups[0]
        b2, eps = g0["betas"][1], g0["eps"]
        if step == at_step:
            undo(model, opt, log, at_step, channels)
            out["lr0"] = g0["lr"]  # ponytail: constant-LR rung; with a schedule, scale the schedule's lr
        f = 1.0
        if (1 - b2) * b2 ** (step - 1 - max(log)) > 1e-12:  # past that, the inflation is gone: lr0
            num = den = 0.0
            with torch.no_grad():
                for n, p in model.named_parameters():
                    st = opt.state.get(p, {})
                    dv = _alive_dv(log, n, step, b2)
                    if dv is None or "exp_avg_sq" not in st:
                        continue
                    bc = 1 - b2 ** (int(st["step"]) + 1)
                    v = st["exp_avg_sq"].double() / bc
                    num += float(((v.sqrt() + eps) / ((v + dv / bc).sqrt() + eps)).sum())
                    den += v.numel()
            f = num / den if den else 1.0
        for g in opt.param_groups:
            g["lr"] = out["lr0"] * f
        out["min_lr_factor"] = min(f, out.get("min_lr_factor", 1.0))

    return pre_step


def as_undo_pre_step(log: dict, at_step: int, channels: str = "mv"):
    """`pre_step` firing `undo` once at the top of `at_step`."""

    def pre_step(step, ctx):
        if step == at_step:
            undo(ctx["model"], ctx["opt"], log, at_step, channels)

    return pre_step


def undo_step(window_last: int, tau: int) -> int:
    """The step whose top the undo fires at, for lag `tau` after a window ending at `window_last`."""
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    return window_last + 1 + tau
