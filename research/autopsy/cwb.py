"""CWB: Counterfactual Weight Back-Projection (docs/EXPERIMENT_CATALOG.md A5; PLAN_V8 §6.5). It repairs a spike
window after the fact, with no rollback. RML-1 (`undo.py`) edits m and v only; CWB also moves w, so it removes the
spike's own update and its echo too.

A shadow AdamW state (w~, m~, v~ and Adam's step count, fp64, keyed by parameter name) forks from the live state
at the first window step. It replays the live run's own final gradients (accumulated and clipped), except inside
the window, where it does one of two things:
  * mode "skip": no update at all. The counterfactual is `Bskipstep`. It needs no clean twin, so a trainer could
    run it on a natural spike.
  * mode "clean": a step on the clean slot's gradient. The counterfactual is B*, with RML-1's oracle clean twin.
At the top of `at_step` the shadow overwrites w, m, v and the step count. The shadow's w~ equals w + A, where A is
A5's accumulator, A <- (1 - eta_k lambda) A + eta_k (u_k - u~_k). Keeping w~ itself avoids the cancellation.

It is open loop: after the window the shadow sees gradients taken at the live w, not at w~. With gradients that
don't depend on w, CWB lands exactly on its counterfactual. On a real loss, the gap to the true fork (`Bskipstep`,
or B*) is the closed-loop error, first order in eta ||H|| ||A||. That gap is itself a measurement (A5).
Cost: three fp64 copies of the model while the shadow runs; the clean mode adds RML-1's clean recompute per window
step. The hooks are plain closures sharing one dict: `fork.run_branch` deep-copies `pre_opt_step`, and a closure
copies as itself.
"""

from __future__ import annotations

from research.harness.schedule import lr_at

MODES = ("skip", "clean")


def as_hooks(window, at_step: int, mode: str = "skip", data_dir=None, clean_fn=None, out=None):
    """(pre_step, pre_opt_step) for one CWB repair of the spike `window` (its steps), committed at the top of
    `at_step` (> the last window step; `undo.undo_step` gives it from a lag). Mode "clean" takes the clean
    gradients the way `undo.as_record_hook` does: `data_dir` (undo.clean_final_grads) or
    `clean_fn(step, ctx) -> {name: grad}`, exactly one. `out` (a dict) gets "a_norm" = ||w - w~|| at the commit
    and "w_norm" = ||w||. The shadow uses the clean hyperparameters: lr_at(step, cfg), and betas and eps from
    cfg; weight decay comes from each parameter's group."""
    import torch

    from research.autopsy import undo

    window = sorted(window)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if at_step <= window[-1]:
        raise ValueError(f"at_step={at_step} must come after the window's last step {window[-1]}")
    if mode == "clean" and (data_dir is None) == (clean_fn is None):
        raise ValueError("mode 'clean' needs exactly one of data_dir / clean_fn")
    inside = set(window)
    sh: dict = {}  # name -> {"w", "m", "v", "t", "wd"}

    def adam(step, cfg, grads):  # one torch-AdamW step of the shadow on {name: grad}
        o = cfg["optim"]
        (b1, b2), eps, lr = o["betas"], o.get("eps", 1e-8), lr_at(step, cfg)
        for n, g in grads.items():
            s = sh.get(n)
            if s is None:
                continue
            g = g.double()
            s["t"] += 1
            s["w"].mul_(1 - lr * s["wd"])
            s["m"].lerp_(g, 1 - b1)  # torch's own form, so the shadow tracks AdamW to the last bit it can
            s["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (s["v"].sqrt() / (1 - b2 ** s["t"]) ** 0.5).add_(eps)
            s["w"].addcdiv_(s["m"], denom, value=-lr / (1 - b1 ** s["t"]))

    def pre_opt_step(step, ctx):
        if step < window[0] or step >= at_step:
            return
        model, opt = ctx["model"], ctx["opt"]
        if step == window[0]:  # fork the shadow from the live state; a rerun of the branch starts afresh
            sh.clear()
            wd = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
            for n, p in model.named_parameters():
                st = opt.state.get(p, {})
                if "exp_avg" in st:
                    sh[n] = {
                        "w": p.detach().double().clone(),
                        "m": st["exp_avg"].double().clone(),
                        "v": st["exp_avg_sq"].double().clone(),
                        "t": int(st["step"]),
                        "wd": wd[id(p)],
                    }
        if step not in inside:
            grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        elif mode == "skip":
            return
        elif clean_fn is not None:
            grads = clean_fn(step, ctx)
        else:  # a forward and backward pass, so it must stay outside no_grad
            grads = undo.clean_final_grads(model, ctx["cfg"], data_dir, step)
        with torch.no_grad():
            adam(step, ctx["cfg"], grads)

    def pre_step(step, ctx):
        if step != at_step:
            return
        if not sh:
            raise RuntimeError(f"CWB: nothing to commit at step {at_step}; the window {window} never ran")
        model, opt = ctx["model"], ctx["opt"]
        a2 = w2 = 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                s = sh.get(n)
                if s is None:
                    continue
                st = opt.state[p]
                a2 += float((p.double() - s["w"]).square().sum())
                w2 += float(p.double().square().sum())
                p.copy_(s["w"])
                st["exp_avg"].copy_(s["m"])
                st["exp_avg_sq"].copy_(s["v"])
                st["step"].fill_(s["t"])
        sh.clear()  # frees the three copies for the rest of the fork
        if out is not None:
            out.update(a_norm=a2**0.5, w_norm=w2**0.5)

    return pre_step, pre_opt_step
