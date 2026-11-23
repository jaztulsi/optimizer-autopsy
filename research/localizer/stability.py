"""Is the top-k eigenbasis of H~ stable? (Week 4: stress-test the Lanczos solve.)

A basis that changes when only the solver's random start vector changes is a solver problem. A
basis that changes when only the probe batch changes is a property of the model: the localizer's
"frozen eigenbasis" then depends on which batch was used, and that has to be reported.

`eigen_stability` runs `curvature.top_k_htilde` once per (probe batch, start seed) pair. It never
stops at the first failure: a solve that raises, does not converge, or comes back with large Ritz
residuals is recorded with its reason and the rest still run. It then compares every pair of
successful solves:

  * subspace overlap  = ||U_a U_b^T||_F^2 / k, in [0, 1]. 1 means the same k-dimensional subspace
    (individual eigenvectors may still rotate inside it when eigenvalues are close, which is why
    the subspace is compared and not vector by vector);
  * eigenvalue spread = max |lambda_a - lambda_b| / |lambda_b| over the k values.

Split two ways: same batch / different seed (solver repeatability) and different batch (batch
sensitivity). `dense_reference` checks a solve against a full `eigh` of the explicit matrix for a
small model, as a second solver to compare with.

The numpy parts (`subspace_overlap`, `summarize_pairs`) are tested without torch.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from research.localizer.snr import as_np


def subspace_overlap(Ua, Ub) -> float:
    """||Ua Ub^T||_F^2 / k for two (k, dim) blocks with orthonormal rows. 1 = same subspace, ~k/dim = unrelated."""
    a, b = as_np(Ua), as_np(Ub)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    k = a.shape[0]
    return float(np.sum((a @ b.T) ** 2) / k)


def eigval_spread(la, lb) -> float:
    a, b = as_np(la), as_np(lb)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))


def summarize_pairs(solves) -> dict:
    """Pairwise comparison of successful solves.

    `solves` is a list of dicts with keys `batch` (any label), `seed`, `ok`, and for ok ones
    `eigvals` and `eigvecs`. Returns min/mean overlap and max eigenvalue spread, split into
    same-batch pairs and cross-batch pairs (None where there are no pairs of that kind).
    """
    good = [s for s in solves if s["ok"]]
    same, cross = [], []
    for a, b in combinations(good, 2):
        row = {
            "a": (a["batch"], a["seed"]),
            "b": (b["batch"], b["seed"]),
            "overlap": subspace_overlap(a["eigvecs"], b["eigvecs"]),
            "eigval_spread": eigval_spread(a["eigvals"], b["eigvals"]),
        }
        (same if a["batch"] == b["batch"] else cross).append(row)

    def agg(rows):
        if not rows:
            return None
        ov = [r["overlap"] for r in rows]
        return {
            "n_pairs": len(rows),
            "min_overlap": float(min(ov)),
            "mean_overlap": float(np.mean(ov)),
            "max_eigval_spread": float(max(r["eigval_spread"] for r in rows)),
        }

    return {
        "n_solves": len(solves),
        "n_ok": len(good),
        "failures": [{"batch": s["batch"], "seed": s["seed"], "reason": s["reason"]} for s in solves if not s["ok"]],
        "same_batch": agg(same),
        "cross_batch": agg(cross),
        "pairs": same + cross,
    }


def eigen_stability(
    model,
    probe_batches,
    *,
    seeds=(0, 1, 2),
    k=5,
    optimizer=None,
    snapshot=None,
    eps=1e-8,
    max_residual=1e-3,
    **solve_kw,
) -> dict:
    """Solve top-k of H~ for every (probe batch, start seed) pair and report how stable the answer is.

    `probe_batches` is a list of (x, y) pairs; each is held fixed for every matvec of its own solve.
    A solve counts as failed if it raises, if ARPACK did not converge, or if any relative Ritz
    residual is above `max_residual`. Extra keyword arguments go to `top_k_htilde`.
    """
    from research.localizer.curvature import top_k_htilde

    solves = []
    for bi, batch in enumerate(probe_batches):
        for seed in seeds:
            row = {"batch": bi, "seed": int(seed), "ok": False, "reason": None}
            try:
                out = top_k_htilde(
                    model, batch, k=k, optimizer=optimizer, snapshot=snapshot, eps=eps, seed=int(seed), **solve_kw
                )
            except Exception as e:  # recorded, not raised: one bad solve must not hide the others
                row["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
                solves.append(row)
                continue
            res = out["residuals"] or []
            worst = max(res) if res else float("nan")
            row.update(
                eigvals=np.asarray(out["eigvals"], dtype=np.float64),
                eigvecs=as_np(out["eigvecs"]),
                residuals=[float(r) for r in res],
                converged=out["converged"],
                n_matvec=int(out["n_matvec"]),
            )
            if not out["converged"]:
                row["reason"] = "ARPACK did not converge"
            elif not res or not (worst <= max_residual):
                row["reason"] = f"Ritz residual {worst:.3e} > {max_residual:.0e}"
            else:
                row["ok"] = True
            solves.append(row)

    summary = summarize_pairs(solves)
    summary["k"] = int(k)
    summary["seeds"] = [int(s) for s in seeds]
    summary["n_batches"] = len(probe_batches)
    summary["solves"] = [
        {
            key: (val.tolist() if isinstance(val, np.ndarray) and key == "eigvals" else val)
            for key, val in s.items()
            if key != "eigvecs"
        }
        for s in solves
    ]
    return summary


def dense_reference(op, k, *, max_dim=4000) -> dict:
    """Top-k of H~ from a full `numpy.linalg.eigh` of the explicit matrix (small models only).

    Builds H~ column by column from `op.matvec` (dim matvecs), symmetrizes it, and solves it
    exactly. This is a second, independent solver to compare `top_k_htilde` against.
    """
    import torch

    dim = op.dim
    if dim > max_dim:
        raise ValueError(f"dim={dim:,} is too large for a dense reference (max {max_dim:,})")
    cols = []
    for j in range(dim):
        e = torch.zeros(dim, dtype=op.dtype, device=op.device)
        e[j] = 1.0
        cols.append(as_np(op.matvec(e)))
    A = np.stack(cols, axis=1)
    asym = float(np.abs(A - A.T).max() / max(np.abs(A).max(), 1e-30))
    w, V = np.linalg.eigh((A + A.T) / 2)
    order = np.argsort(w)[::-1][:k]
    return {"eigvals": w[order], "eigvecs": V[:, order].T, "asymmetry": asym}
