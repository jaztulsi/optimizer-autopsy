import torch
from model import TinyModel
from hvp import hvp, flatten_params, set_flat_params, compute_loss

torch.manual_seed(0)


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


if __name__ == "__main__":
    # float64: finite differences divides by a small eps, which
    # amplifies float32 rounding noise. float64 avoids that.
    model = TinyModel().double()
    x = torch.randn(5, 4).double()
    y = torch.randn(5, 2).double()

    n_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(n_params).double()
    v = v / v.norm()

    Hv_exact = hvp(model, x, y, v)
    Hv_approx = finite_diff_hvp(model, x, y, v, eps=1e-4)

    abs_diff = (Hv_exact - Hv_approx).abs()
    rel_diff = abs_diff / (Hv_exact.abs() + 1e-8)

    print(f"Max absolute difference: {abs_diff.max().item():.2e}")
    print(f"Max relative difference: {rel_diff.max().item():.2e}")
    print(f"Mean relative difference: {rel_diff.mean().item():.2e}")

    if rel_diff.max().item() < 1e-3:
        print("\nPASS: double-backward HVP matches finite differences.")
    else:
        print("\nFAIL: something is wrong with the HVP implementation.")
        
