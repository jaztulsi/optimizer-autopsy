import torch
from model import TinyModel

torch.manual_seed(0)


def flatten_params(model):
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].reshape(p.shape))
        idx += n


def compute_loss(model, x, y):
    out = model(x)
    return ((out - y) ** 2).mean()


def hvp(model, x, y, v):
    """Hessian-vector product via Pearlmutter's trick (double-backward)."""
    params = list(model.parameters())

    # FIRST backward pass: get the gradient.
    # create_graph=True keeps the graph alive so we can backward again.
    loss = compute_loss(model, x, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    # Dot the gradient with v to get ONE scalar.
    # d/dtheta of (grad . v) == H @ v  -- this is the whole trick.
    grad_dot_v = torch.dot(flat_grad, v)

    # SECOND backward pass: this IS the Hessian-vector product.
    hvp_grads = torch.autograd.grad(grad_dot_v, params)
    flat_hvp = torch.cat([g.reshape(-1) for g in hvp_grads])

    return flat_hvp


if __name__ == "__main__":
    model = TinyModel()
    x = torch.randn(5, 4)
    y = torch.randn(5, 2)

    n_params = sum(p.numel() for p in model.parameters())
    v = torch.randn(n_params)
    v = v / v.norm()

    Hv = hvp(model, x, y, v)
    print(f"H @ v shape: {Hv.shape}")
    print(f"H @ v (first 5 entries): {Hv[:5]}")
    
