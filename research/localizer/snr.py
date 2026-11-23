"""SNR localizer: how much real learning signal each part of the optimizer state carries (Week 4, Task 10).

Two SNRs, because the plans use the word for two different things:

  1. UPDATE SNR (no gradients needed, works from a snapshot alone):
         snr_i = |m_hat_i| / (sqrt(v_hat_i) + eps)
     the size of Adam's own step relative to its noise scale. When v is inflated by a spike, this
     collapses in the poisoned coordinates. `snr`, `group_snr`, `snr_score`.

  2. DIRECTIONAL SNR (context-ai §4.6), a t-statistic along each direction u_i:
         SNR_i = |mean_j (g_j . u_i)| / sqrt(var_j(g_j . u_i) / n)
     over n per-microbatch gradients g_j. High = a consistent learning signal (protect it); low =
     noise (a repair candidate). `directional_snr` does the statistic; `project_microbatch_grads`
     gets the n x k projections without ever holding n full gradients (context-ai §4.6).

Groups. A group is one contiguous slice of the flat parameter vector, in the order of
`curvature.param_index` / `curvature.layout`, so the tied wte/lm_head tensor is ONE group under
`transformer.wte.weight`. `groups_by_tensor` gives one group per tensor; `groups_by_head` also
splits each attention `c_attn` weight into its q/k/v rows per head. Every group carries `tensor`
and `rows`, the two fields PLAN_V8 §6.4's `Localization` needs.

Everything except `project_microbatch_grads` is numpy, so it is tested without torch.
"""

from __future__ import annotations

import numpy as np


def as_np(x, dtype=np.float64) -> np.ndarray:
    """A torch tensor (any device) or array-like -> a numpy array, copying as needed."""
    if hasattr(x, "detach"):
        x = x.detach().to("cpu").numpy()
    return np.asarray(x, dtype=dtype)


# --------------------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------------------


def groups_by_tensor(layout) -> dict:
    """One group per tensor. `layout` is `curvature.layout(index)` (or the `layout` a solve returns)."""
    return {
        e["name"]: {"tensor": e["name"], "rows": None, "start": e["offset"], "stop": e["offset"] + e["numel"]}
        for e in layout
    }


def groups_by_head(layout, n_head: int) -> dict:
    """Like `groups_by_tensor`, but each `*.attn.c_attn.weight` (and bias) is split into q/k/v rows per head.

    nanoGPT's c_attn is a Linear(C, 3C): its weight is (3C, C), rows [0, C) are q, [C, 2C) are k,
    [2C, 3C) are v, and inside each block head h owns rows [h*hd, (h+1)*hd) with hd = C / n_head.
    Rows are contiguous in the flat (row-major) layout, so each head group is still one slice.
    """
    out = {}
    for e in layout:
        name, shape, off = e["name"], tuple(e["shape"]), e["offset"]
        if not name.endswith("attn.c_attn.weight") and not name.endswith("attn.c_attn.bias"):
            out[name] = {"tensor": name, "rows": None, "start": off, "stop": off + e["numel"]}
            continue
        three_c = shape[0]
        if three_c % 3 or (three_c // 3) % n_head:
            raise ValueError(f"{name}: {three_c} rows do not split into 3 x {n_head} heads")
        c = three_c // 3
        hd = c // n_head
        row = e["numel"] // three_c  # elements per row (C for the weight, 1 for the bias)
        for part, tag in enumerate("qkv"):
            for h in range(n_head):
                r0 = part * c + h * hd
                r1 = r0 + hd
                out[f"{name}[{tag}{h}]"] = {
                    "tensor": name,
                    "rows": (r0, r1),
                    "start": off + r0 * row,
                    "stop": off + r1 * row,
                }
    return out


def check_groups(groups, dim: int) -> None:
    """Groups must tile [0, dim) exactly once: no gaps, no overlaps."""
    spans = sorted((g["start"], g["stop"]) for g in groups.values())
    pos = 0
    for a, b in spans:
        if a != pos or b <= a:
            raise ValueError(f"groups do not tile the flat vector: gap or overlap at {pos} (next span {a}..{b})")
        pos = b
    if pos != dim:
        raise ValueError(f"groups cover {pos} of {dim} entries")


def flat_from_named(named: dict, layout) -> np.ndarray:
    """{param_name: tensor} (a snapshot's `m`, `v` or a v_bar) -> one flat float64 vector in layout order."""
    parts = []
    for e in layout:
        if e["name"] not in named:
            raise KeyError(f"no entry for {e['name']!r}")
        t = as_np(named[e["name"]]).reshape(-1)
        if t.size != e["numel"]:
            raise ValueError(f"{e['name']!r} has {t.size} entries, layout says {e['numel']}")
        parts.append(t)
    return np.concatenate(parts) if parts else np.zeros(0)


# --------------------------------------------------------------------------------------
# 1. Update SNR
# --------------------------------------------------------------------------------------


def snr(m, v, eps=1e-8, *, adam_step=None, betas=None) -> np.ndarray:
    """Elementwise |m_hat| / (sqrt(v_hat) + eps).

    With `adam_step` and `betas`, m and v are bias-corrected the way AdamW corrects them, so the
    number is the size of the step AdamW actually takes. Without them, raw m and v are used.
    """
    m, v = as_np(m), as_np(v)
    if np.any(v < 0):
        raise ValueError("v has a negative entry; it is a variance")
    if adam_step is not None and betas is not None:
        b1, b2 = betas
        m = m / (1.0 - b1 ** int(adam_step))
        v = v / (1.0 - b2 ** int(adam_step))
    return np.abs(m) / (np.sqrt(v) + eps)


def group_snr(snapshot, groups, layout, eps=1e-8, *, betas=None) -> dict:
    """Mean update SNR per group. `snapshot` is a harness.snapshot dict (`m`, `v`, `adam_step`)."""
    step = snapshot.get("adam_step") if betas is not None else None
    s = snr(
        flat_from_named(snapshot["m"], layout), flat_from_named(snapshot["v"], layout), eps, adam_step=step, betas=betas
    )
    return {gid: float(s[g["start"] : g["stop"]].mean()) for gid, g in groups.items()}


def snr_score(group_values: dict) -> dict:
    """Normalized per-group implication score from `group_snr`: log(median SNR / group SNR).

    > 0 means the group has LESS signal than the typical group (more implicated); 0 is typical.
    A group with zero SNR gets +inf. Scale-free, so it can be compared across steps and runs.
    """
    vals = np.array(list(group_values.values()), dtype=np.float64)
    if vals.size == 0:
        return {}
    med = float(np.median(vals))
    out = {}
    for gid, x in group_values.items():
        out[gid] = float("inf") if x <= 0 else float(np.log(med / x)) if med > 0 else float("-inf")
    return out


# --------------------------------------------------------------------------------------
# 2. Directional SNR
# --------------------------------------------------------------------------------------


def directional_snr(proj) -> np.ndarray:
    """t-statistic per direction from an (n, k) array of projections g_j . u_i (n microbatches).

    Uses the unbiased variance (n - 1 degrees of freedom), so n >= 2 is required. Zero variance
    with a nonzero mean gives +inf (a perfectly consistent signal); zero mean and zero variance
    gives 0.
    """
    p = as_np(proj)
    if p.ndim != 2:
        raise ValueError(f"expected an (n, k) array, got shape {p.shape}")
    n = p.shape[0]
    if n < 2:
        raise ValueError(f"directional SNR needs >= 2 microbatch gradients, got {n}")
    mean = p.mean(axis=0)
    se = np.sqrt(p.var(axis=0, ddof=1) / n)
    out = np.empty_like(mean)
    for i in range(mean.size):
        if se[i] > 0:
            out[i] = abs(mean[i]) / se[i]
        else:
            out[i] = np.inf if mean[i] != 0 else 0.0
    return out


def project_microbatch_grads(model, batches, U, *, index=None, loss_fn=None) -> np.ndarray:
    """(n, k) array of g_j . u_i, one row per microbatch in `batches`, one column per row of `U`.

    `U` is a (k, dim) eigenvector block (torch or numpy) in `curvature.param_index` order. Each
    gradient is projected and dropped before the next is computed, so memory is one gradient, not n.
    """
    import torch

    from research.localizer.curvature import grad_flat, param_index

    index = param_index(model) if index is None else index
    p0 = index[0][1]
    Ut = U if torch.is_tensor(U) else torch.from_numpy(np.asarray(U))
    Ut = Ut.to(p0.device, p0.dtype)
    rows = []
    for b in batches:
        g = grad_flat(model, b, index=index, loss_fn=loss_fn)
        rows.append((Ut @ g).detach().to("cpu", torch.float64).numpy())
    model.zero_grad(set_to_none=True)
    return np.stack(rows) if rows else np.zeros((0, Ut.shape[0]))
