"""The localizer end to end: snapshots in, PLAN_V8 §6.4 `Localization` out (Weeks 5-7).

`localize_spike` is the whole pipeline for one spike:

  1. restore the t0 snapshot (the one taken at the detected failure step) into a live model;
  2. [curvature path] solve the top-k eigenpairs of H~ at t0 on ONE fixed probe batch, and check
     the Ritz residuals -- a basis that did not converge is refused, not used;
  3. [curvature path] directional SNR along that frozen basis, from per-microbatch gradients at t0;
  4. for every matched normal step: restore it, and compute the same quantities along the SAME
     frozen basis (that is what makes them a null for this basis);
  5. `poison.localize` -> thresholds from the normal steps, p_i, psi_k, P, group scores, selection.

With `use_eig=False` steps 2-3 are skipped and it is the no-HVP localizer (drift only), which needs
no eigensolve at all.

`microbatches(snapshot)` is a callable the caller supplies, so this file does not decide where data
comes from; `trunk_microbatches` is the one the GPU driver uses. Torch is imported inside functions,
so importing this module is cheap.

This runs anywhere torch runs; research/gpu/localize_run.py is the driver that runs it on real
captured spikes (GPU only, CLAUDE.md §1). research/tests/test_localizer_pipeline.py runs it on a
tiny GPT on the CPU with a planted spike.
"""

from __future__ import annotations

import time

from research.localizer import poison, snr


def make_groups(layout, how="tensor", n_head=None) -> dict:
    if how == "tensor":
        g = snr.groups_by_tensor(layout)
    elif how == "head":
        if n_head is None:
            raise ValueError("groups='head' needs n_head")
        g = snr.groups_by_head(layout, n_head)
    else:
        raise ValueError(f"groups must be 'tensor' or 'head', got {how!r}")
    snr.check_groups(g, sum(e["numel"] for e in layout))
    return g


def trunk_microbatches(cfg, data_dir, *, n_micro=4, device="cpu"):
    """`microbatches(snapshot)` for real runs: the trunk's own batch for the last step the snapshot ran.

    A snapshot's `step` is the NEXT step to run, so the step that produced its state is `step - 1`.
    get_batch is a pure function of the step, so this is the clean data that step read (an induced
    corrupt_batch spike replaced it with corrupted data; the clean batch is the right one for asking
    whether a direction carries real signal). The batch's rows are split into `n_micro` equal
    chunks, because the proxy runs with grad_accum = 1 and the directional SNR needs >= 2 gradients.
    """
    from research.data.prepare import get_batch

    t = cfg["train"]
    accum = int(t.get("grad_accum", 1))
    block = cfg["model"]["block_size"]

    def fn(snapshot):
        data_step = int(snapshot["step"]) - 1
        if data_step < 0:
            raise ValueError("snapshot step 0 has no previous step to take data from")
        out = []
        for micro in range(accum):
            x, y = get_batch("train", data_step * accum + micro, t["batch_size"], block, data_dir, device)
            if x.shape[0] % n_micro:
                raise ValueError(f"batch of {x.shape[0]} rows does not split into {n_micro} microbatches")
            out.extend(zip(x.chunk(n_micro), y.chunk(n_micro)))
        return out

    return fn


def localize_spike(
    model,
    optimizer,
    t0_snapshot,
    vbar,
    normals,
    *,
    probe_batch=None,
    microbatches=None,
    use_eig=True,
    k=5,
    eps=1e-8,
    groups="tensor",
    n_head=None,
    k_select=5,
    q_p=0.95,
    q_s=0.5,
    q_d=0.95,
    min_steps=3,
    max_residual=1e-3,
    mem_budget_gb=6.0,
    t=None,
) -> dict:
    """One spike -> a validated `Localization` (PLAN_V8 §6.4), with the eigensolve evidence attached.

    `t0_snapshot` and each normal snapshot are harness.snapshot dicts; `vbar` and each normal's
    v_bar are {param_name: tensor}. `normals` is a list of (snapshot, vbar) pairs. `t` defaults to
    the t0 snapshot's step. The model is left holding the t0 weights.
    """
    import torch

    from research.harness.snapshot import restore
    from research.localizer.curvature import layout, param_index, top_k_htilde

    if not normals:
        raise ValueError("no matched normal steps: the thresholds have nothing to come from")
    if use_eig and (probe_batch is None or microbatches is None):
        raise ValueError("the curvature path needs probe_batch and microbatches (or pass use_eig=False)")

    started = time.perf_counter()
    restore(model, optimizer, t0_snapshot)
    index = param_index(model)
    lay = layout(index)
    grp = make_groups(lay, groups, n_head)

    def flat(named):
        return snr.flat_from_named(named, lay)

    U = dir_snr = None
    eigen = None
    if use_eig:
        out = top_k_htilde(model, probe_batch, k=k, snapshot=t0_snapshot, eps=eps, mem_budget_gb=mem_budget_gb)
        worst = max(out["residuals"]) if out["residuals"] else float("inf")
        if not out["converged"] or not worst <= max_residual:
            raise RuntimeError(
                f"eigensolve at t0 is not usable (converged={out['converged']}, worst Ritz residual {worst:.3e} "
                f"> {max_residual:.0e}); a localization on this basis would be noise"
            )
        U = snr.as_np(out["eigvecs"])
        dir_snr = snr.directional_snr(snr.project_microbatch_grads(model, microbatches(t0_snapshot), U, index=index))
        eigen = {
            "eigvals": [float(x) for x in out["eigvals"]],
            "residuals": [float(r) for r in out["residuals"]],
            "symmetry_worst_rel": out["symmetry"]["worst_rel"] if out["symmetry"] else None,
            "n_matvec": int(out["n_matvec"]),
        }

    null = []
    for snap_n, vbar_n in normals:
        row = {"v": flat(snap_n["v"]), "vbar": flat(vbar_n), "step": int(snap_n["step"])}
        if use_eig:
            restore(model, optimizer, snap_n)
            row["dir_snr"] = snr.directional_snr(
                snr.project_microbatch_grads(model, microbatches(snap_n), U, index=index)
            )
        null.append(row)
    if use_eig:
        restore(model, optimizer, t0_snapshot)  # leave the model at t0, as documented

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    cost = time.perf_counter() - started

    loc = poison.localize(
        v=flat(t0_snapshot["v"]),
        vbar=flat(vbar),
        groups=grp,
        null=null,
        t=int(t0_snapshot["step"] if t is None else t),
        U=U,
        dir_snr=dir_snr,
        k_select=k_select,
        q_p=q_p,
        q_s=q_s,
        q_d=q_d,
        min_steps=min_steps,
        cost_gpu_s=cost if torch.cuda.is_available() else 0.0,
    )
    loc["provenance"].update(
        eigen=eigen,
        k=int(k) if use_eig else None,
        groups=groups,
        normal_steps=[r["step"] for r in null],
        wall_s=cost,
    )
    return loc


def summary_stats(loc) -> dict:
    """The numbers Week 7 compares between spikes and normal steps, from one Localization."""
    prov = loc["provenance"]
    tau_d = prov["thresholds"]["tau_d"]
    return {
        "t": loc["t"],
        "max_drift_ratio": max(prov["drift"].values()) / tau_d,
        "max_score": max(g["score"] for g in loc["groups"].values()),
        "psi_k": prov.get("psi_k"),
        "n_selected": len(loc["selected"]),
        "selected": list(loc["selected"]),
    }
