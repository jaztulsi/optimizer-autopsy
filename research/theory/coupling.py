"""R-F2: the m+v coupling law (PLAN_V8 Appendix A.3; derivation and frozen predictions in research/theory/README.md).

Linear response of Adam to one spike step s, on a coordinate the spike dominates. Clean v sits at v_bar; the spike
adds delta to the gradient, so (1 - b1) * delta to m and (1 - b2) * dv to v, with x = (1 - b2) * dv / v_bar (the
dose meter's x; x ~ (1 - b2) * z**2 for z = delta / sqrt(v_bar)). j steps later the spike's share of the update is
-eta * z * (1 - b1) * b1**j * q_j, q_j = (1 + x * b2**j) ** -0.5, while v still holds the spike, and the same
without q_j once v is undone and m is not. Undo at lag tau fires at the top of step s + 1 + tau (`undo.undo_step`).
To the end of the run (j <= horizon) the spike moves the coordinate by eta * z times
    A_mv   = sum_{j <= tau} (1 - b1) b1**j q_j       undo-m+v, and undo-m along delta
    A_v    = A_mv + b1**(tau + 1) - b1**(horizon + 1)  undo-v: m's echo, no longer damped by v
    A_none = sum_{j <= horizon} (1 - b1) b1**j q_j    no undo
With loss quadratic in the displacement (prereg-v3 P4) and the v-stall (which moves the clean update, not delta)
cancelling in the 2x2 design, Y(arm) is proportional to A**2. So, with no free parameter:
    R_v = Y(undo-v) / Y(undo-m+v) = (A_v / A_mv)**2
    pi  = -I / E_m = 1 - (A_none**2 - A_mv**2) / (A_v**2 - A_mv**2)     (prereg-v3's protection share)
"""

from __future__ import annotations

import math


def predict(x: float, b1: float, b2: float, horizon: int, lag: int = 0) -> dict:
    """{"R_v", "pi"} at dose x (>= 0) for a spike `horizon` steps before the end of the run, undone at `lag`."""
    if x < 0 or not 0 <= lag < horizon:
        raise ValueError(f"need x >= 0 and 0 <= lag < horizon, got x={x}, lag={lag}, horizon={horizon}")
    s = [(1 - b1) * b1**j / math.sqrt(1 + x * b2**j) for j in range(horizon + 1)]
    a_mv, a_none = math.fsum(s[: lag + 1]), math.fsum(s)
    a_v = a_mv + b1 ** (lag + 1) - b1 ** (horizon + 1)
    return {"R_v": (a_v / a_mv) ** 2, "pi": 1 - (a_none**2 - a_mv**2) / (a_v**2 - a_mv**2)}


def pi_gap_limit(b1: float, b2: float) -> float:
    """lim x * (1 - pi) as x -> inf (lag 0, long horizon). r = b1 / sqrt(b2) is the plan's r-law (§6.3 item 0)."""
    r = b1 / math.sqrt(b2)
    return ((1 - b1) / b1) ** 2 * ((1 - r) ** -2 - 1)
