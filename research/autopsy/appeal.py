"""RML-3 (K1 Appeal): the measurement hook for the kill test (docs/novelty/rml3_method.md §1.2 and §8.2;
docs/novelty/appeal_build_plan.md §7's gate). It only reads, so a run with it is bitwise the run without it
(tests/test_appeal.py).

At every step it scores each row of the batch by a robust z of its loss inside that batch (median and MAD, both
sides: junk the model can copy scores low). At a flagged step it records the gradient of each flagged row and of up
to 16 random kept rows (the shadows: same step, weights and later batches, so they are the matched reference). Over
the next K steps each record collects evidence: the cosine, in Adam's scaling P = 1/(sqrt(v) + eps) with v from
before the flagged step's update, between the batch gradient Adam is fed and the recorded one. At K it closes with
p = share of that step's shadows scoring below the row and phi = clip((p - lo) / (hi - lo), 0, 1).

Not here, on purpose (the gate: nothing past the measurement until the kill test passes): withholding the flagged
rows, the restore through undo.undo, force_phi. So these verdicts come from a run that trained on the flagged rows,
as on the shadows; under Appeal the flagged rows would be withheld (build plan §9 question 3 is that asymmetry).
"""

from __future__ import annotations

import torch

MAD_SCALE = 1.4826  # MAD of a normal is 0.6745 sigma, so z reads in sigmas


def row_losses(model, x, y) -> torch.Tensor:
    """Mean loss of each row, ignoring targets of -1. One forward pass, no gradients, eval mode (no RNG drawn)."""
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits, _ = model(x, y)
            tok = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1, reduction="none"
            ).view(y.shape)
    finally:
        model.train(was)
    keep = y != -1
    return (tok * keep).sum(1) / keep.sum(1).clamp_min(1)


def robust_z(losses: torch.Tensor) -> torch.Tensor:
    med = losses.quantile(0.5)
    return (losses - med) / (MAD_SCALE * (losses - med).abs().quantile(0.5)).clamp_min(1e-12)


def flag_rows(losses: torch.Tensor, z: float = 6.0) -> torch.Tensor:
    """Rows whose robust z is at least `z` in absolute value."""
    return robust_z(losses).abs() >= z


def cosine_p(a: torch.Tensor, b: torch.Tensor, P: torch.Tensor) -> float:
    """<a, b>_P / (|a|_P |b|_P); 0 when either vector is zero."""
    norm = ((a * P * a).sum() * (b * P * b).sum()).sqrt()
    return float((a * P * b).sum() / norm) if norm > 0 else 0.0


def p_score(t: float, shadow_t: list[float]) -> float:
    return sum(u < t for u in shadow_t) / len(shadow_t)


def phi(p: float, lo: float = 0.05, hi: float = 0.25) -> float:
    return min(1.0, max(0.0, (p - lo) / (hi - lo)))


def _vec(ts) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1).double() for t in ts])


def _row_grad(model, params, x, y, i) -> torch.Tensor:
    """Gradient of row i's mean loss, through autograd.grad: p.grad is never touched."""
    was = model.training
    model.eval()
    try:
        _, loss = model(x[i : i + 1], y[i : i + 1])
        gs = torch.autograd.grad(loss, params, allow_unused=True)
    finally:
        model.train(was)
    return _vec(torch.zeros_like(p) if g is None else g for g, p in zip(gs, params))


def make(data_dir, lag=10, z=6.0, n_shadow=16, lo=0.05, hi=0.25, seed=0, flag_fn=None):
    """Returns (pre_step, pre_opt_step, state) for train_forward / run_fork; compose the spike recipe first.

    flag_fn(step, losses) -> bool mask, if given, names the recorded rows (the kill test names the injected ones);
    the z flag is still logged. state (JSON-safe apart from "open"):
      "verdicts": one entry per closed flagged step: step, n, and per flagged row (parallel lists) row, z, T, p, phi,
                  with the shadows' rows and T;
      "flagged": [step, row, z] for every row the z flag catches, at every step (the false-flag rate comes from here);
      "batch_events": steps where more than half the batch was flagged, so nothing was recorded.
    """
    from research.data.prepare import get_batch

    if lag < 1:
        raise ValueError(f"lag must be at least 1 (evidence is read from the steps after the flag), got {lag}")
    st ={"last": None, "open": {}, "verdicts": [], "flagged": [], "batch_events": []}

    def pre_step(step, ctx):
        if st["last"] is not None and step <= st["last"]:  # the hooks reused for another run
            for k in ("open", "verdicts", "flagged", "batch_events"):
                st[k].clear()  # in place: callers hold these lists
        st["last"] = step
        model, c = ctx["model"], ctx["cfg"]
        if c["train"].get("grad_accum", 1) != 1:
            raise ValueError("appeal reads one batch per step: grad_accum must be 1")
        if ctx["batch"] is not None:
            x, y = ctx["batch"]
        else:
            dev = next(model.parameters()).device
            slot = step + ctx.get("data_offset", 0)
            x, y = get_batch("train", slot, c["train"]["batch_size"], model.cfg.block_size, data_dir, dev)
        losses = row_losses(model, x, y)
        zs = robust_z(losses)
        auto = zs.abs() >= z
        st["flagged"] += [[step, int(i), float(zs[i])] for i in auto.nonzero().flatten()]
        mask = auto if flag_fn is None else flag_fn(step, losses)
        rows = mask.nonzero().flatten().tolist()
        if not rows:
            return
        if 2 * len(rows) > len(losses):
            st["batch_events"].append(step)
            return
        kept = (~mask).nonzero().flatten().cpu()
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)  # local: the global RNG is never drawn
        shadows = kept[torch.randperm(len(kept), generator=gen)[:n_shadow]].tolist()
        params = list(model.parameters())
        st["open"][step] = {
            "rows": rows,
            "z": [float(zs[i]) for i in rows],
            "g": [_row_grad(model, params, x, y, i) for i in rows],
            "sg": [_row_grad(model, params, x, y, i) for i in shadows],
            "shadows": shadows,
            "P": None,
        }

    def pre_opt_step(step, ctx):
        if not st["open"]:
            return None
        opt, params = ctx["opt"], list(ctx["model"].parameters())
        g = None
        for s, rec in list(st["open"].items()):
            if rec["P"] is None:  # the flagged step itself: v is still from before its update
                eps = {id(p): grp["eps"] for grp in opt.param_groups for p in grp["params"]}
                rec["P"] = _vec(1 / (opt.state[p]["exp_avg_sq"].sqrt() + eps[id(p)]) for p in params)
                rec["T"], rec["sT"] = [0.0] * len(rec["g"]), [0.0] * len(rec["sg"])
                continue
            if g is None:
                g = _vec(torch.zeros_like(p) if p.grad is None else p.grad for p in params)
            rec["T"] = [t + cosine_p(g, r, rec["P"]) for t, r in zip(rec["T"], rec["g"])]
            rec["sT"] = [t + cosine_p(g, r, rec["P"]) for t, r in zip(rec["sT"], rec["sg"])]
            if step - s == lag:
                ps = [p_score(t, rec["sT"]) for t in rec["T"]]
                st["verdicts"].append(
                    {
                        "step": s,
                        "n": len(rec["rows"]),
                        "rows": rec["rows"],
                        "z": rec["z"],
                        "T": rec["T"],
                        "p": ps,
                        "phi": [phi(p, lo, hi) for p in ps],
                        "shadows": rec["shadows"],
                        "shadow_T": rec["sT"],
                    }
                )
                del st["open"][s]
        return None

    return pre_step, pre_opt_step, st
