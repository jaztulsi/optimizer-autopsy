"""Curvature localizer: local sharpness / Hessian-vector proxies at the snapshot.

High curvature in an implicated direction distinguishes a genuine sharp region from Adam noise.

Task 11. This is the merge of hvp_project/week2_hvp.py and week3_eigen.py, in dependency order:

  1. A parameter INDEX + flat layout (`param_index`, `layout`). Every flat vector here -- probe
     vector, HVP output, eigenvector -- is indexed against the same ordered list, taken from
     `model.named_parameters()`. nanoGPT ties `wte.weight` to `lm_head.weight`, so that list holds
     the shared tensor ONCE under `transformer.wte.weight` (see harness.snapshot's docstring).
     `layout()` is the bridge back to the name-keyed `m`/`v` dicts the repair operator writes.
  2. `hvp(model, batch, vec)` -- Pearlmutter: H.vec via double backward, H never formed (§3.4),
     plus `fd_check` (§3.1's verification) and `symmetry_check`.
  3. `HTilde` + `top_k_htilde` -- the preconditioned operator `H~ = D^{-1/2} H D^{-1/2}`,
     `D = diag(sqrt(v) + eps)`, and its top-k eigenpairs by Lanczos (§3.3, §3.5).

WHY THE SANDWICH and not `D^{-1} H`: Adam is locally preconditioned GD in the D-metric, so the
operator governing its stability is `D^{-1} H`. That is not symmetric, and Lanczos requires
symmetry. `D^{-1/2} H D^{-1/2}` is a similarity transform of `D^{-1} H` -- identical eigenvalues,
symmetric -- so it is the thing you can actually hand an eigensolver. `_selfcheck_precond` asserts
both halves of that claim numerically.

THREE WAYS THIS OPERATOR GOES SILENTLY NON-SYMMETRIC. Lanczos raises no error; it returns noise.
  * A different probe batch per matvec. PLAN §3.5 / context-ai §4.5 call this out. `HTilde` binds
    one batch at construction and never re-reads it.
  * dropout > 0. A fresh mask per matvec is the same failure with a different source, and the plan
    does NOT mention it. `HTilde.__init__` refuses to construct.
  * A non-deterministic attention kernel. Forced to MATH, which is also the only twice-
    differentiable one -- see `_math_sdpa`.
`symmetry_check` runs before every solve by default, `_selfcheck_asymmetry_is_caught` proves that
guard actually fires, and Ritz residuals come back with every eigenpair. Those are the only
evidence a solve worked.

Self-checks here are GPU/Kaggle-only per CLAUDE.md §1; nothing in this module has been run on the
workstation.

# TODO: curvature_probe(snapshot, batch, directions) -> curvature along candidate poison directions.
# TODO: curvature_score(snapshot, batch) -> normalized per-group score for the poison combiner.
"""

from __future__ import annotations

import contextlib

# --------------------------------------------------------------------------------------
# 1. Parameter index and flat layout
# --------------------------------------------------------------------------------------


def param_index(model) -> list[tuple[str, object]]:
    """Ordered [(name, param)] that every flat vector in this module is indexed against.

    `named_parameters()` dedups by tensor identity, so the tied wte/lm_head tensor appears exactly
    ONCE, under `transformer.wte.weight` (module traversal reaches `self.transformer` before
    `self.lm_head`). Those are the same keys harness.snapshot.capture writes, so an eigenvector
    from this module lines up with a snapshot's `m`/`v` without any renaming.

    Never rebuild this from `state_dict()` keys: state_dict exposes BOTH names for the shared
    tensor, which double-counts 9.66M of the proxy's 11.04M parameters in every dot product and
    silently destroys symmetry. The duplicate check below is what fails loudly if that happens.
    """
    seen: dict[int, str] = {}
    out: list[tuple[str, object]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:
            raise RuntimeError(
                f"duplicate parameter tensor: {name!r} is the same object as {seen[id(p)]!r}. "
                "A tied weight must appear once -- did this come from state_dict() or "
                "named_parameters(remove_duplicate=False)?"
            )
        seen[id(p)] = name
        out.append((name, p))
    if not out:
        raise RuntimeError("model has no trainable parameters")
    return out


def assert_tied_once(model) -> None:
    """The tying invariant, asserted the way harness.snapshot's selfcheck asserts it."""
    names = {n for n, _ in param_index(model)}
    assert "transformer.wte.weight" in names, "wte weight missing from the index"
    assert "lm_head.weight" not in names, "tied weight indexed twice (wte and lm_head)"


def flat_dim(index) -> int:
    return sum(p.numel() for _, p in index)


def layout(index) -> list[dict]:
    """[{name, shape, offset, numel}] -- how a flat vector slices back into named tensors.

    Task 12/13 need this: eigenvectors come out of the solver flat, but `m`/`v` in a snapshot are
    keyed by name, so the projection in §4.8 has to know which slice belongs to which tensor.
    """
    out, off = [], 0
    for name, p in index:
        n = p.numel()
        out.append({"name": name, "shape": tuple(p.shape), "offset": off, "numel": n})
        off += n
    return out


def flatten_params(index):
    import torch

    return torch.cat([p.reshape(-1) for _, p in index])


def flatten(tensors):
    import torch

    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten(vec, index):
    """Slice a flat vector into parameter-shaped views, not copies."""
    out, i = [], 0
    for _, p in index:
        n = p.numel()
        out.append(vec[i : i + n].view_as(p))
        i += n
    if i != vec.numel():
        raise ValueError(f"vector has {vec.numel()} elements, index needs {i}")
    return out


def set_flat_params(index, flat):
    import torch

    with torch.no_grad():
        for p, part in zip((p for _, p in index), unflatten(flat, index)):
            p.copy_(part)


def to_named(vec, index) -> dict:
    """Flat vector -> {param_name: tensor}, the form snapshot/repair consume."""
    return {name: t for (name, _), t in zip(index, unflatten(vec, index))}


# --------------------------------------------------------------------------------------
# 2. Hessian-vector products (Pearlmutter)
# --------------------------------------------------------------------------------------


def _math_sdpa():
    """Force the MATH attention backend for the duration of a double backward.

    The flash and mem-efficient SDPA kernels have NO double backward: the second
    `autograd.grad` dies with "derivative for aten::_scaled_dot_product_*_backward is not
    implemented". nanoGPT's attention goes through `F.scaled_dot_product_attention`, so every HVP
    in this project hits that. MATH is built from ordinary differentiable primitives, so it is
    twice-differentiable -- and it is the deterministic backend, which the bitwise-replay
    requirement (one HVP must be bit-reproducible) wants regardless.
    """
    import torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(SDPBackend.MATH)
    except Exception:  # pragma: no cover - older torch
        try:
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            return contextlib.nullcontext()


def compute_loss(model, x, y):
    """nanoGPT's forward returns (logits, loss).

    AdamW's weight decay is decoupled and NOT in this loss, so H is the pure loss Hessian with no
    decay term to correct for.
    """
    _, loss = model(x, y)
    return loss


def hvp(model, batch, vec, *, index=None, loss_fn=None):
    """H . vec by double backward, without ever forming H (§3.4).

    `batch` is a single (x, y) pair -- the SAME pair for every matvec in a solve (see the module
    docstring). `vec` is either a flat 1-D tensor of length `flat_dim(index)` or a list of tensors
    shaped like the parameters; the return matches whichever was passed.

    The tied embedding needs no special handling HERE: wte and lm_head are one tensor, so autograd
    accumulates both paths into its single gradient entry by itself. What needs care is the index
    (above) -- count it twice and every dot product is wrong, in a way that still produces a
    plausible-looking vector.
    """
    import torch

    index = param_index(model) if index is None else index
    ps = [p for _, p in index]
    loss_fn = loss_fn or compute_loss

    flat_in = torch.is_tensor(vec) and vec.dim() == 1
    v = unflatten(vec, index) if flat_in else list(vec)

    x, y = batch
    with _math_sdpa():
        model.zero_grad(set_to_none=True)  # so a stale .grad can't be mistaken for this pass
        loss = loss_fn(model, x, y)

        # FIRST backward: the gradient. create_graph=True keeps the graph for the second pass.
        g = torch.autograd.grad(loss, ps, create_graph=True)

        # grad . v as ONE scalar; d/dtheta of that IS H @ v. Summed per-tensor rather than on a
        # concatenated vector: torch.cat on 11M params would allocate a second full copy inside
        # the graph for nothing.
        grad_dot_v = sum((gi * vi).sum() for gi, vi in zip(g, v))

        # SECOND backward: this is the Hessian-vector product.
        hv = torch.autograd.grad(grad_dot_v, ps, allow_unused=True)

    hv = [torch.zeros_like(p) if h is None else h.detach() for h, p in zip(hv, ps)]
    return flatten(hv) if flat_in else hv


def grad_flat(model, batch, *, index=None, loss_fn=None):
    """Flat first-order gradient, same index and same attention backend as `hvp`."""
    import torch

    index = param_index(model) if index is None else index
    ps = [p for _, p in index]
    x, y = batch
    with _math_sdpa():
        model.zero_grad(set_to_none=True)
        loss = (loss_fn or compute_loss)(model, x, y)
        g = torch.autograd.grad(loss, ps, allow_unused=True)
    return flatten([torch.zeros_like(p) if gi is None else gi.detach() for gi, p in zip(g, ps)])


# --------------------------------------------------------------------------------------
# 3. Verification: finite differences and symmetry
# --------------------------------------------------------------------------------------


def finite_diff_hvp(model, batch, v, h=1e-2, *, index=None, loss_fn=None):
    """(g(w + h v) - g(w - h v)) / 2h ~= H v, on the SAME batch as the HVP.

    A different batch on either side turns this into a measurement of gradient noise instead of a
    check of the Hessian. Weights are restored in a `finally`, so a raise mid-check cannot leave
    the model perturbed -- not something you want to discover later at 11M params on a GPU.
    """
    import torch

    index = param_index(model) if index is None else index
    ps = [p for _, p in index]
    theta0 = [p.detach().clone() for p in ps]
    vparts = unflatten(v, index)

    try:
        with torch.no_grad():
            for p, t, dv in zip(ps, theta0, vparts):
                p.copy_(t).add_(dv, alpha=h)
        gp = grad_flat(model, batch, index=index, loss_fn=loss_fn)

        with torch.no_grad():
            for p, t, dv in zip(ps, theta0, vparts):
                p.copy_(t).add_(dv, alpha=-h)
        gm = grad_flat(model, batch, index=index, loss_fn=loss_fn)
    finally:
        with torch.no_grad():
            for p, t in zip(ps, theta0):
                p.copy_(t)

    return (gp - gm) / (2.0 * h)


def fd_check(model, batch, *, steps=None, seed=0, index=None, loss_fn=None, vec=None):
    """Central-difference check of `hvp` at real scale, swept over step size.

    READ THE SWEEP, never a single h. The error is O(h^2) truncation plus
    O(eps_machine * ||g|| / h) cancellation, so it falls, bottoms out in a valley, then rises
    again as h shrinks. In fp32 that valley sits around h ~ 1e-2..1e-3 and bottoms at a relative
    error of ~1e-3, NOT 1e-6 -- that is the expected result here, not a failure. A single-h
    pass/fail (which is fine in float64 on a toy model) tells you nothing at this precision.

    `cos` is the more robust column: the direction survives cancellation long after the magnitude
    stops being accurate. Norm-relative error is used rather than an elementwise max because over
    11M entries most of H.v is near zero, so a per-element max is dominated by near-zero
    denominators and blows up even when the vector is right.
    """
    import torch

    index = param_index(model) if index is None else index
    ps = [p for _, p in index]
    dim = flat_dim(index)
    steps = list(steps) if steps is not None else [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

    dev, dt = ps[0].device, ps[0].dtype
    if vec is None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        v = torch.randn(dim, generator=gen, dtype=torch.float32).to(dev, dt)
    else:
        v = vec.to(dev, dt)
    v = v / v.norm()

    hv = hvp(model, batch, v, index=index, loss_fn=loss_fn)
    g0 = grad_flat(model, batch, index=index, loss_fn=loss_fn)
    den = hv.norm().item()

    rows = []
    for h in steps:
        fd = finite_diff_hvp(model, batch, v, h, index=index, loss_fn=loss_fn)
        cos = torch.nn.functional.cosine_similarity(fd.float().unsqueeze(0), hv.float().unsqueeze(0)).item()
        rows.append(
            {
                "h": h,
                "rel_l2": (fd - hv).norm().item() / den if den else float("nan"),
                "cos": cos,
                "fd_norm": fd.norm().item(),
                "hv_norm": den,
            }
        )

    return {
        "dim": dim,
        "grad_norm": g0.norm().item(),
        "theta_norm": flatten_params(index).norm().item(),
        "rows": rows,
        "best": min(rows, key=lambda r: r["rel_l2"]),
    }


def print_fd_check(report) -> None:
    print(f"dim={report['dim']:,}  ||theta||={report['theta_norm']:.4g}  ||g||={report['grad_norm']:.4g}")
    print(f"{'h':>10} {'rel_l2':>12} {'cos':>14} {'||fd||':>12} {'||Hv||':>12}")
    for r in report["rows"]:
        print(f"{r['h']:>10.1e} {r['rel_l2']:>12.3e} {r['cos']:>14.9f} {r['fd_norm']:>12.4e} {r['hv_norm']:>12.4e}")
    b = report["best"]
    print(f"valley: h={b['h']:.1e}  rel_l2={b['rel_l2']:.3e}  cos={b['cos']:.9f}")


def symmetry_check(matvec, dim, *, n=3, seed=0, device="cpu", dtype=None):
    """<u, A w> vs <w, A u>. The non-negotiable guard from context-ai §4.5.

    An operator that is not symmetric makes Lanczos return noise WITH NO ERROR RAISED, so this is
    not optional diagnostics -- it is the only thing standing between a bug and a published
    eigenbasis. Costs 2n matvecs.

    Expect `rel` ~1e-6..1e-4 from fp32 rounding alone. A value near 1 means a genuine asymmetry:
    the batch is changing between matvecs, dropout is on, or a non-deterministic kernel is in play.
    """
    import torch

    dtype = dtype or torch.float32
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows = []
    for _ in range(n):
        u = torch.randn(dim, generator=gen, dtype=torch.float32).to(device, dtype)
        w = torch.randn(dim, generator=gen, dtype=torch.float32).to(device, dtype)
        u /= u.norm()
        w /= w.norm()
        a = torch.dot(u, matvec(w)).item()
        b = torch.dot(w, matvec(u)).item()
        rows.append({"uAw": a, "wAu": b, "rel": abs(a - b) / max(abs(a), abs(b), 1e-30)})
    worst = max(r["rel"] for r in rows)
    return {"rows": rows, "worst_rel": worst, "symmetric": worst < 1e-3}


# --------------------------------------------------------------------------------------
# 4. The preconditioned operator H~ = D^{-1/2} H D^{-1/2}
# --------------------------------------------------------------------------------------


def adam_v_flat(index, *, optimizer=None, snapshot=None):
    """Adam's second moment `v` as one flat vector in `index` order.

    Two sources, because the localizer runs at a FROZEN point: a live optimizer (`exp_avg_sq`,
    keyed by param OBJECT) or a snapshot dict (`snapshot["v"]`, keyed by param NAME -- the
    authoritative form per CLAUDE.md §1).
    """
    import torch

    if (optimizer is None) == (snapshot is None):
        raise ValueError("pass exactly one of optimizer= or snapshot=")

    parts = []
    if snapshot is not None:
        v = snapshot["v"]
        for name, p in index:
            if name not in v:
                raise RuntimeError(f"snapshot has no v for {name!r}")
            t = v[name]
            if tuple(t.shape) != tuple(p.shape):
                raise RuntimeError(f"v[{name!r}] is {tuple(t.shape)}, param is {tuple(p.shape)}")
            parts.append(t.to(p.device, torch.float32).reshape(-1))
    else:
        for name, p in index:
            st = optimizer.state.get(p, {})
            if "exp_avg_sq" not in st:
                raise RuntimeError(
                    f"no AdamW state for {name!r}; build D after >=1 opt.step() (same rule as harness.snapshot.capture)"
                )
            parts.append(st["exp_avg_sq"].detach().to(p.device, torch.float32).reshape(-1))
    return torch.cat(parts)


def d_inv_sqrt(index, *, optimizer=None, snapshot=None, eps=1e-8, adam_step=None, beta2=None):
    """Flat `D^{-1/2}` for `D = diag(sqrt(v) + eps)`, i.e. elementwise `(sqrt(v) + eps) ** -0.5`.

    `eps` must be the SAME eps AdamW is running with (`cfg["optim"]["eps"]`, 1e-8 in both the proxy
    and 124M configs). A different eps here means this is not the metric the optimizer steps in,
    and the eigenbasis is of the wrong operator.

    Bias correction is OFF by default because PLAN §3.3 writes `D = diag(sqrt(v) + eps)` with raw
    `v`. AdamW's effective preconditioner uses `v / (1 - beta2**t)`, which differs by a scalar that
    only matters in the first ~1/(1-beta2) steps (~20 at beta2=0.95). Pass `adam_step` and `beta2`
    to apply it; note it rescales every eigenvalue by the same factor, so it changes the
    eigenVALUES and not the eigenVECTORS.
    """
    v = adam_v_flat(index, optimizer=optimizer, snapshot=snapshot)
    if adam_step is not None and beta2 is not None:
        v = v / (1.0 - float(beta2) ** int(adam_step))
    d = v.sqrt().add_(eps)
    if not bool(d.gt(0).all()):
        raise RuntimeError("D has a non-positive entry; D^{-1/2} undefined")
    return d.rsqrt()


class HTilde:
    """Matvec access to `H~ = D^{-1/2} H D^{-1/2}` against ONE fixed probe batch.

    The batch is bound here and never re-read, which is the whole point: a per-matvec batch breaks
    symmetry and the eigensolver will not tell you (context-ai §4.5). dropout is refused for the
    same reason, and attention is pinned to MATH inside every `hvp`.

    Construct with either `optimizer=` (live run) or `snapshot=` (frozen at t0).
    """

    def __init__(
        self,
        model,
        batch,
        *,
        optimizer=None,
        snapshot=None,
        eps=1e-8,
        adam_step=None,
        beta2=None,
        loss_fn=None,
        index=None,
    ):
        dropout = getattr(getattr(model, "cfg", None), "dropout", 0.0)
        if dropout:
            raise ValueError(
                f"dropout={dropout} makes H~ non-symmetric: a fresh mask per matvec is a different "
                "operator each time, and Lanczos returns noise without raising. The proxy and 124M "
                "configs both set dropout: 0.0 -- use those."
            )
        self.model = model
        self.batch = batch  # bound once, deliberately
        self.index = param_index(model) if index is None else index
        self.dim = flat_dim(self.index)
        self.loss_fn = loss_fn
        self.eps = eps
        self.dinv = d_inv_sqrt(
            self.index,
            optimizer=optimizer,
            snapshot=snapshot,
            eps=eps,
            adam_step=adam_step,
            beta2=beta2,
        )
        self.n_matvec = 0
        p0 = self.index[0][1]
        self.device, self.dtype = p0.device, p0.dtype

    def matvec(self, x):
        """torch flat vector -> torch flat vector. D^{-1/2} applied on both sides."""
        self.n_matvec += 1
        xd = self.dinv * x.to(self.device, self.dtype)
        return self.dinv * hvp(self.model, self.batch, xd, index=self.index, loss_fn=self.loss_fn)

    def h_matvec(self, x):
        """The RAW Hessian on the same batch -- for the commutator / basis-rotation work in §4.10."""
        self.n_matvec += 1
        return hvp(self.model, self.batch, x.to(self.device, self.dtype), index=self.index, loss_fn=self.loss_fn)

    def as_linear_operator(self):
        """scipy LinearOperator, float64.

        ARPACK is double-only and the model is fp32, so every matvec round-trips
        float64 -> float32 -> float64. That caps attainable accuracy at the fp32 noise floor
        (~1e-6 relative) regardless of `tol`, which is why `top_k_htilde` defaults `tol=1e-6` and
        not 0. `tol=0` asks ARPACK for machine precision the matvec cannot deliver, and it will
        spend the whole iteration budget failing to get there.
        """
        import numpy as np
        import torch
        from scipy.sparse.linalg import LinearOperator

        def mv(x):
            xt = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(-1))
            return self.matvec(xt).detach().to("cpu", torch.float64).numpy()

        return LinearOperator((self.dim, self.dim), matvec=mv, dtype=np.float64)


# --------------------------------------------------------------------------------------
# 5. Top-k eigenpairs
# --------------------------------------------------------------------------------------


def default_ncv(dim, k):
    return int(min(max(2 * k + 1, 20), dim))


def estimate_mem(dim, k, ncv=None, *, bytes_per=8):
    """Host RAM the ARPACK Lanczos basis needs: dim * ncv * 8 bytes, in float64.

    AT THE PROXY SHAPE THIS IS NOT SMALL, and the config comment is misleading about why.
    `build_model_opt` resolves `vocab_size: null` to 50304, so the tied embedding alone is
    50304 * 192 = 9,658,368 of 11,035,968 total parameters. The model is ~11.0M params, not the
    "1-3M" the config says -- that figure is the non-embedding count, 1,328,448. So:

        k=5,  ncv=20  ->  1.77 GB
        k=10, ncv=21  ->  1.85 GB
        k=20, ncv=41  ->  3.62 GB

    before the model, the double-backward graph, or the probe batch. Budget for that on a 13 GB box.
    """
    ncv = default_ncv(dim, k) if ncv is None else int(ncv)
    basis = dim * ncv * bytes_per
    return {
        "dim": int(dim),
        "k": int(k),
        "ncv": ncv,
        "basis_bytes": int(basis),
        "basis_gb": basis / 1e9,
        "per_vector_mb": dim * bytes_per / 1e6,
    }


def top_k_htilde(
    model,
    batch,
    *,
    k=5,
    optimizer=None,
    snapshot=None,
    eps=1e-8,
    adam_step=None,
    beta2=None,
    which="LA",
    tol=1e-6,
    ncv=None,
    maxiter=None,
    seed=0,
    mem_budget_gb=6.0,
    check_symmetry=True,
    residuals=True,
    loss_fn=None,
    op=None,
):
    """Top-k eigenpairs of `H~` from matvec access alone (§3.5).

    `eigsh` IS Lanczos -- ARPACK's implicitly restarted version. Hand-rolling the three-term
    recurrence would mean also hand-rolling full reorthogonalization and restart logic, which is
    where hand-rolled Lanczos actually goes wrong.

    `which="LA"` = largest ALGEBRAIC: the stiffest positive-curvature directions, which is what
    the stability story in §3.8 is about. "LM" (largest magnitude) would mix in large NEGATIVE
    eigenvalues -- unstable for a different reason, and not belonging in the same top-k.

    Returns {eigvals, eigvecs (k, dim) float32, residuals, symmetry, converged, n_matvec, layout},
    where `residuals[i] = ||H~ u_i - lambda_i u_i|| / |lambda_i|`. Those residuals and the symmetry
    row are the ONLY evidence the solve worked -- eigsh returns confident-looking numbers either
    way. Read them before using the basis for anything.
    """
    import numpy as np
    import torch
    from scipy.sparse.linalg import ArpackNoConvergence, eigsh

    op = op or HTilde(
        model,
        batch,
        optimizer=optimizer,
        snapshot=snapshot,
        eps=eps,
        adam_step=adam_step,
        beta2=beta2,
        loss_fn=loss_fn,
    )
    dim = op.dim
    ncv = default_ncv(dim, k) if ncv is None else int(ncv)
    mem = estimate_mem(dim, k, ncv)
    if mem["basis_gb"] > mem_budget_gb:
        raise MemoryError(
            f"ARPACK basis would take {mem['basis_gb']:.2f} GB (dim={dim:,}, ncv={ncv}), over a "
            f"{mem_budget_gb} GB budget. Options, best first: lower k; localize on a parameter "
            "SUBSET (one block) and solve there -- context-ai §4.5's own fallback; or raise "
            "mem_budget_gb if the box really has the RAM. NOTE: torch.lobpcg is NOT an escape "
            "hatch. It requires an explicit tensor A and will not accept a matvec or a "
            "LinearOperator, so the 124M route as written in §4.5 cannot be built on this "
            "operator. That needs a decision, not a workaround."
        )

    sym = None
    if check_symmetry:
        sym = symmetry_check(op.matvec, dim, seed=seed, device=op.device, dtype=op.dtype)
        if not sym["symmetric"]:
            raise RuntimeError(
                f"H~ is not symmetric (worst <u,Aw> vs <w,Au> mismatch {sym['worst_rel']:.3e}). "
                "Lanczos would return noise and raise nothing. Check, in order: one fixed probe "
                "batch, dropout 0, deterministic attention kernel."
            )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    v0 = torch.randn(dim, generator=gen, dtype=torch.float64).numpy()

    converged = True
    try:
        vals, vecs = eigsh(op.as_linear_operator(), k=k, which=which, ncv=ncv, tol=tol, maxiter=maxiter, v0=v0)
    except ArpackNoConvergence as e:
        # Partial results beat losing an hour of Kaggle GPU time to an exception.
        converged = False
        vals, vecs = e.eigenvalues, e.eigenvectors
        if vals.size == 0:
            raise

    order = np.argsort(vals)[::-1]
    vals = np.ascontiguousarray(vals[order])
    vecs = np.ascontiguousarray(vecs[:, order])
    U = torch.from_numpy(vecs.T.astype(np.float32)).contiguous()  # (k, dim)

    res = None
    if residuals:
        res = []
        for i in range(U.shape[0]):
            u = U[i].to(op.device, op.dtype)
            r = (op.matvec(u) - float(vals[i]) * u).norm().item()
            res.append(r / max(abs(float(vals[i])), 1e-30))

    return {
        "eigvals": vals,
        "eigvecs": U,
        "residuals": res,
        "symmetry": sym,
        "converged": bool(converged),
        "which": which,
        "backend": "scipy.sparse.linalg.eigsh (ARPACK implicitly-restarted Lanczos)",
        "mem": mem,
        "dim": dim,
        "n_matvec": op.n_matvec,
        "layout": layout(op.index),
    }


def print_eigen(result) -> None:
    print(f"backend = {result['backend']}")
    print(f"dim={result['dim']:,}  matvecs={result['n_matvec']}  ncv={result['mem']['ncv']}  ", end="")
    print(f"basis={result['mem']['basis_gb']:.2f} GB  converged={result['converged']}")
    if result["symmetry"]:
        s = result["symmetry"]
        print(f"symmetry: worst_rel={s['worst_rel']:.3e}  symmetric={s['symmetric']}")
    for i, lam in enumerate(result["eigvals"]):
        r = result["residuals"][i] if result["residuals"] else float("nan")
        print(f"  lambda_{i} = {lam:+.6e}   rel_residual = {r:.3e}")


# --------------------------------------------------------------------------------------
# 6. Self-checks -- GPU/Kaggle only (CLAUDE.md §1). Not run on the workstation.
# --------------------------------------------------------------------------------------


def _quad_model(A, dim, seed=0):
    """A real nn.Module whose loss is 0.5 x^T A x, so H == A exactly.

    Built as a Module on purpose: the checks then run through the same `hvp` / `HTilde` /
    `top_k_htilde` path the proxy model uses. A numpy shortcut would pass even if `hvp` were broken.
    """
    import torch
    import torch.nn as nn

    class Quad(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = type("cfg", (), {"dropout": 0.0})()
            self.x = nn.Parameter(torch.randn(dim, generator=torch.Generator().manual_seed(seed)))

        def forward(self, _x, _y):
            return None, 0.5 * (self.x @ (A @ self.x))

    return Quad()


def _selfcheck_quadratic(dim=400, k=3, seed=0) -> dict:
    """Task 11 DoD part 1: eigsh recovers the top-3 of a known quadratic to 1e-4.

    `v` is all ones here, so D = (1 + eps) I and H~ = A / (1 + eps). That isolates the HVP and the
    solver from the preconditioner; D gets its own check next.
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    B = torch.randn(dim, dim, dtype=torch.float64)
    A = ((B + B.T) / 2).float()

    m = _quad_model(A, dim, seed)
    idx = param_index(m)
    batch = (None, None)

    v = torch.randn(dim)
    v /= v.norm()
    got, want = hvp(m, batch, v, index=idx), A @ v
    err = (got - want).norm().item() / want.norm().item()
    assert err < 1e-5, f"hvp != A v: rel {err:.3e}"

    class FakeOpt:  # v = 1 everywhere -> D = (1 + eps) I
        state = {idx[0][1]: {"exp_avg_sq": torch.ones(dim)}}

    op = HTilde(m, batch, optimizer=FakeOpt(), eps=1e-8)
    sym = symmetry_check(op.matvec, dim, seed=seed)
    assert sym["symmetric"], f"operator not symmetric: {sym['worst_rel']:.3e}"

    # tol=1e-7, not 0: the operator is fp32, so float64 machine precision is unreachable.
    out = top_k_htilde(m, batch, k=k, op=op, tol=1e-7, seed=seed)
    ref = np.sort(np.linalg.eigvalsh(A.double().numpy()))[::-1][:k]
    rel = np.max(np.abs(out["eigvals"] * (1.0 + 1e-8) - ref) / np.abs(ref))
    assert rel < 1e-4, f"top-{k} eigenvalues off by {rel:.3e} (DoD: 1e-4)"
    assert max(out["residuals"]) < 1e-5, f"Ritz residuals too large: {max(out['residuals']):.3e}"
    print(f"quadratic DoD OK: hvp rel {err:.2e}, top-{k} rel {rel:.2e}, {out['n_matvec']} matvecs")
    return {
        "claim": "hvp == A v, and eigsh recovers top-k of a known quadratic to 1e-4 (Task 11 DoD)",
        "dim": dim,
        "k": k,
        "hvp_rel_err": float(err),
        "eigval_rel_err": float(rel),
        "max_ritz_residual": float(max(out["residuals"])),
        "symmetry_worst_rel": float(sym["worst_rel"]),
        "n_matvec": int(out["n_matvec"]),
        "eigvals": [float(x) for x in out["eigvals"]],
        "eigvals_reference": [float(x) for x in ref],
        "tolerance": 1e-4,
        "passed": True,
    }


def _selfcheck_precond(dim=200, k=3, seed=0) -> dict:
    """Task 11 DoD part 2, and the check part 1 CANNOT do.

    With a non-trivial `v`, asserts both halves of the sandwich claim:
      (a) the solve returns the spectrum of D^{-1/2} A D^{-1/2};
      (b) that spectrum equals the spectrum of the non-symmetric D^{-1} A -- the operator Adam
          actually moves in.
    Catches D built in the wrong power (D^{-1} where D^{-1/2} belongs), which part 1 is blind to
    because v=1 makes D the identity there.
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    B = torch.randn(dim, dim, dtype=torch.float64)
    A = ((B + B.T) / 2).float()
    vdiag = torch.rand(dim) * 9.0 + 1.0  # v in [1, 10) -> sqrt(v) spans ~3x

    m = _quad_model(A, dim, seed)
    idx = param_index(m)

    class FakeOpt:
        state = {idx[0][1]: {"exp_avg_sq": vdiag}}

    op = HTilde(m, (None, None), optimizer=FakeOpt(), eps=1e-8)
    out = top_k_htilde(m, (None, None), k=k, op=op, tol=1e-7, seed=seed)

    dm = (vdiag.double().sqrt() + 1e-8).rsqrt()
    ref = np.sort(np.linalg.eigvalsh((torch.diag(dm) @ A.double() @ torch.diag(dm)).numpy()))[::-1][:k]
    rel = np.max(np.abs(out["eigvals"] - ref) / np.abs(ref))
    assert rel < 1e-4, f"preconditioned top-{k} off by {rel:.3e}"

    dinv_a = torch.diag(1.0 / (vdiag.double().sqrt() + 1e-8)) @ A.double()
    ref2 = np.sort(np.real(np.linalg.eigvals(dinv_a.numpy())))[::-1][:k]
    rel2 = np.max(np.abs(out["eigvals"] - ref2) / np.abs(ref2))
    assert rel2 < 1e-3, f"spec(D^-1/2 A D^-1/2) != spec(D^-1 A): {rel2:.3e}"
    print(f"preconditioner DoD OK: rel {rel:.2e}, sandwich == spec(D^-1 A) to {rel2:.2e}")
    return {
        "claim": "top-k of D^-1/2 A D^-1/2 is correct AND equals spec(D^-1 A) (the sandwich claim)",
        "dim": dim,
        "k": k,
        "eigval_rel_err": float(rel),
        "sandwich_vs_dinv_a_rel_err": float(rel2),
        "n_matvec": int(out["n_matvec"]),
        "eigvals": [float(x) for x in out["eigvals"]],
        "tolerance": 1e-4,
        "passed": True,
    }


def _selfcheck_asymmetry_is_caught(dim=300, k=3, seed=0) -> dict:
    """The guard has to actually fire. Hands `top_k_htilde` a deliberately batch-dependent
    operator and asserts it REFUSES rather than returning plausible-looking eigenvalues.

    Without this, `check_symmetry=True` is an untested claim -- and an untested guard against a
    silent failure is the same as no guard.
    """
    import torch

    torch.manual_seed(seed)
    B = torch.randn(dim, dim, dtype=torch.float64)
    A = ((B + B.T) / 2).float()
    m = _quad_model(A, dim, seed)
    idx = param_index(m)

    class FakeOpt:
        state = {idx[0][1]: {"exp_avg_sq": torch.ones(dim)}}

    op = HTilde(m, (None, None), optimizer=FakeOpt(), eps=1e-8)
    clean = op.matvec
    drift = {"n": 0}

    def wobbling_matvec(x):  # stands in for "a new probe batch every matvec"
        drift["n"] += 1
        return clean(x) + 0.05 * drift["n"] * x.roll(1)

    op.matvec = wobbling_matvec
    try:
        top_k_htilde(m, (None, None), k=k, op=op, seed=seed)
    except RuntimeError as e:
        assert "not symmetric" in str(e), f"wrong error: {e}"
        print("asymmetry guard OK: a drifting operator is refused, not silently solved")
        return {
            "claim": "a non-symmetric (batch-drifting) operator is REFUSED, not silently solved",
            "dim": dim,
            "guard_fired": True,
            "error": str(e).split(".")[0],
            "passed": True,
        }
    raise AssertionError("asymmetry guard did NOT fire -- a non-symmetric operator was solved")


def _selfcheck_proxy(device=None, steps=3, probe=4, k=3) -> dict:
    """The real-scale deliverable: the FD sweep and the top-k solve on the model
    `build_model_opt` builds, with a real data batch and real AdamW state behind it.

    GPU/Kaggle only (CLAUDE.md §1). Needs a tokenized proxy shard.
    """
    import torch

    from research.configs import load_config
    from research.data.prepare import PRESETS, get_batch, prepare, resolve_data_dir
    from research.harness.determinism import seed_everything
    from research.harness.trunk import build_model_opt, train_forward

    # Load the config file rather than inlining a copy of it: an inlined copy silently drifts from
    # research/experiments/proxy/config.yaml, and then this check is measuring a different model
    # than the trunk trains.
    cfg = load_config("research/experiments/proxy/config.yaml")
    seed_everything(cfg.get("seed", 1337), deterministic=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        data_dir = resolve_data_dir("proxy")
    except FileNotFoundError:
        data_dir = prepare(PRESETS["proxy"])

    model, opt = build_model_opt(cfg, device)
    assert_tied_once(model)
    train_forward(model, opt, cfg, data_dir, 0, steps, device)  # so m, v exist

    idx = param_index(model)
    dim = flat_dim(idx)
    nemb = next(p.numel() for n, p in idx if n == "transformer.wte.weight")
    print(f"dim = {dim:,} over {len(idx)} tensors (tied embedding counted once)")
    print(f"  of which transformer.wte.weight = {nemb:,} ({100 * nemb / dim:.1f}%)")

    # A SMALL fixed probe batch, not the training batch: double backward holds a second graph.
    pb = get_batch("train", 10_000, probe, cfg["model"]["block_size"], data_dir, device)

    report = fd_check(model, pb, index=idx)
    print_fd_check(report)
    best = report["best"]
    # fp32 valley is expected near rel_l2 ~ 1e-3 (see fd_check); 1e-2 leaves margin, cos is the robust column.
    assert best["cos"] > 0.999 and best["rel_l2"] < 1e-2, (
        f"FD check failed at real scale: best rel_l2={best['rel_l2']:.3e}, cos={best['cos']:.6f}"
    )

    # One HVP must be bit-reproducible on a fixed batch (§3.1).
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(dim, generator=g).to(device)
    reproducible = torch.equal(hvp(model, pb, v, index=idx), hvp(model, pb, v, index=idx))
    assert reproducible, "HVP is not bit-reproducible on a fixed batch"
    print("HVP bit-reproducible: OK")

    op = HTilde(model, pb, optimizer=opt, eps=cfg["optim"].get("eps", 1e-8), index=idx)
    mem = estimate_mem(op.dim, k)
    print("\nmem:", mem)
    eig = top_k_htilde(model, pb, k=k, op=op)
    print_eigen(eig)

    return {
        "claim": "FD check passes at real scale on build_model_opt; top-k of H~ solved (Task 11)",
        "device": device,
        "dim": dim,
        "n_tensors": len(idx),
        "embedding_numel": int(nemb),
        "embedding_fraction": nemb / dim,
        "trunk_steps_before_probe": steps,
        "probe_batch": probe,
        "block_size": cfg["model"]["block_size"],
        "hvp_bit_reproducible": bool(reproducible),
        "fd": {
            "theta_norm": report["theta_norm"],
            "grad_norm": report["grad_norm"],
            "best": report["best"],
            "rows": report["rows"],
        },
        "eigen": {
            "k": k,
            "which": eig["which"],
            "backend": eig["backend"],
            "converged": eig["converged"],
            "ncv": mem["ncv"],
            "basis_gb": mem["basis_gb"],
            "n_matvec": eig["n_matvec"],
            "symmetry_worst_rel": eig["symmetry"]["worst_rel"] if eig["symmetry"] else None,
            "eigvals": [float(x) for x in eig["eigvals"]],
            "rel_residuals": eig["residuals"],
        },
    }


if __name__ == "__main__":
    _selfcheck_quadratic()
    _selfcheck_precond()
    _selfcheck_asymmetry_is_caught()
