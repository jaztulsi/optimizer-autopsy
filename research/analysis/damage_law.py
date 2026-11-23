"""The echo law of docs/novelty/method.md §2.2, the prediction X4 (§6.2) checks.

A spike of size z = |G| / sqrt(v_bar) on a coordinate, with a share a of its innovation kept in m and b in v, moves
that coordinate, summed over the steps after it, by

    E(a, b) = a · eps1 · z · Σ_k lr_k · beta1^k / sqrt(1 + b · x · beta2^k),   x = eps2 · z²,  eps_i = 1 - beta_i

In units of the learning rate (lr_k = 1, k to infinity), Ψ(z) = E(1, 1) <= min(z, D). Clipping by c lands at
(c, c²), so its echo is Ψ(cz); undoing v alone lands at (1, 0), echo z. The stall law (§2.3) is not here: nothing
built so far needs it.
"""

from __future__ import annotations

import math


def ceiling(betas=(0.9, 0.95)) -> float:
    """D = eps1 / (sqrt(eps2) (1 - beta1 / sqrt(beta2))), the largest echo any spike leaves: 5.84 at (0.9, 0.95)."""
    b1, b2 = betas
    return (1 - b1) / (math.sqrt(1 - b2) * (1 - b1 / math.sqrt(b2)))


def echo(z: float, a: float = 1.0, b: float = 1.0, betas=(0.9, 0.95), lrs=None) -> float:
    """E(a, b) for a spike of size z, step k weighted by lrs[k] (Adam's step size k steps after the spike, bias
    correction included). lrs=None sums 2,000 steps at lr 1, which is Ψ(z) at a = b = 1."""
    b1, b2 = betas
    x = (1 - b2) * z * z
    lrs = [1.0] * 2000 if lrs is None else lrs
    return a * (1 - b1) * z * sum(lr * b1**k / math.sqrt(1 + b * x * b2**k) for k, lr in enumerate(lrs))
