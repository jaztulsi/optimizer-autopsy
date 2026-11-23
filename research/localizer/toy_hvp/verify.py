import math
import sys

import torch

from research.localizer.toy_hvp.hvp import compute_loss, flatten_params, hvp, set_flat_params
from research.localizer.toy_hvp.model import TinyModel

TOLERANCE = 1e-3  # max relative difference allowed between HVP and finite differences


def finite_diff_hvp(model, x, y, v, eps=1e-4):
    """Slow, independent check: nudge params by +-eps*v, see how
    the gradient changes. (grad_plus - grad_minus) / (2*eps) ~= H @ v."""
    params = list(model.parameters())
    theta0 = flatten_params(model).clone()

    set_flat_params(model, theta0 + eps * v)
    loss_plus = compute_loss(model, x, y)
    grads_plus = torch.autograd.grad(loss_plus, params)
    flat_grad_plus = torch.cat([g.reshape(-1) for g in grads_plus])

    set_flat_params(model, theta0 - eps * v)
    loss_minus = compute_loss(model, x, y)
    grads_minus = torch.autograd.grad(loss_minus, params)
    flat_grad_minus = torch.cat([g.reshape(-1) for g in grads_minus])

    set_flat_params(model, theta0)  # restore original params

    return (flat_grad_plus - flat_grad_minus) / (2 * eps)


def check(seed=0, eps=1e-4):
    """Compare the HVP with finite differences. Returns the max relative difference.

    Raises AssertionError if the result is not finite or is above TOLERANCE, so a broken HVP
    fails a test instead of only printing FAIL.
    """
    # Seed here, not at import time: importing this file must not reset anyone else's random stream.
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)  # TinyModel's weight init uses the global stream

    # float64: finite differences divides by a small eps, which
    # amplifies float32 rounding noise. float64 avoids that.
    model = TinyModel().double()
    x = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    y = torch.randn(5, 2, generator=gen, dtype=torch.float64)

    n_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(n_params, generator=gen, dtype=torch.float64)
    v = v / v.norm()

    Hv_exact = hvp(model, x, y, v)
    Hv_approx = finite_diff_hvp(model, x, y, v, eps=eps)

    abs_diff = (Hv_exact - Hv_approx).abs()
    rel_diff = abs_diff / (Hv_exact.abs() + 1e-8)
    worst = rel_diff.max().item()

    print(f"Max absolute difference: {abs_diff.max().item():.2e}")
    print(f"Max relative difference: {worst:.2e}")
    print(f"Mean relative difference: {rel_diff.mean().item():.2e}")

    assert math.isfinite(worst), "HVP or finite-difference result is not finite (NaN or inf)"
    assert worst < TOLERANCE, f"HVP does not match finite differences: max rel diff {worst:.2e} >= {TOLERANCE:.0e}"
    return worst


if __name__ == "__main__":
    try:
        check()
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
    print("\nPASS: double-backward HVP matches finite differences.")
