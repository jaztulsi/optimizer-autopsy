"""CPU checks for research/theory/coupling.py (R-F2): the law against torch AdamW driven through the real undo hooks,
and its large-dose limit against the plan's r-law."""

import pytest
import torch

from research.autopsy import undo
from research.theory.coupling import pi_gap_limit, predict

B1, B2, S, H = 0.9, 0.95, 300, 60  # the spike at step S, once v has settled; H steps of echo after it


def _clean(step):
    # Gradients of +-1 that flip sign every step: v sits at exactly 1, and the two coordinates' m are opposite, so
    # their mean displacement drops the clean m's cross-term with the inflated v (the v-stall, which the 2x2 cancels).
    return torch.tensor([1.0, -1.0], dtype=torch.float64) * (-1) ** step


def _displacement(delta, arm, lag):
    """Mean over the two coordinates of w(arm) - w(clean) after step S + H. The spike adds `delta` to both at step S;
    arm is "none" or an undo channel ("m", "v", "mv") fired at lag `lag`."""

    def run(spiked, channel=None):
        model = torch.nn.Module()
        model.w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(B1, B2), eps=1e-8, weight_decay=0.0)
        log = {}
        rec = undo.as_record_hook(log, [S], clean_fn=lambda step, ctx: {"w": _clean(step)})
        fix = channel and undo.as_undo_pre_step(log, undo.undo_step(S, lag), channel)
        for step in range(S + H + 1):
            ctx = {"model": model, "opt": opt}
            if fix:
                fix(step, ctx)
            model.w.grad = _clean(step) + (delta if spiked and step == S else 0.0)
            rec(step, ctx)
            opt.step()
        return model.w.detach()

    return float((run(True, None if arm == "none" else arm) - run(False)).mean())


@pytest.mark.parametrize(("delta", "lag"), [(10.0, 0), (40.0, 0), (40.0, 3)])
def test_law_matches_adamw_through_the_undo_hooks(delta, lag):
    # Y = squared displacement, combined into prereg-v3's quantities exactly as the GPU runs will combine losses.
    y = {arm: _displacement(delta, arm, lag) ** 2 for arm in ("none", "m", "v", "mv")}
    i = y["none"] - y["v"] - y["m"] + y["mv"]
    got = {"R_v": y["v"] / y["mv"], "pi": -i / (y["v"] - y["mv"])}
    # The dose meter's x, with v_bar = 1. The two coordinates' x differ by a factor (1 +- 2/delta), as the clean
    # gradient enters dv; that spread is the error budget (about 1% on R_v at delta = 10).
    want = predict((1 - B2) * delta**2, B1, B2, H, lag)
    assert got == pytest.approx(want, rel=0.05)
    assert want["R_v"] > 100 and 0.5 < want["pi"] < 1  # not a vacuous match: undo-v really is far worse


@pytest.mark.parametrize("b1", [0.8, 0.9, 0.95])
def test_large_dose_limit_is_the_r_law(b1):
    x = 1e8
    assert x * (1 - predict(x, b1, B2, 5000)["pi"]) == pytest.approx(pi_gap_limit(b1, B2), rel=0.01)


@pytest.mark.parametrize(("b1", "pi"), [(0.8, 0.75), (0.9, 0.73), (0.95, 0.65)])
def test_readme_table(b1, pi):
    # research/theory/README.md's x = 5 column was computed by hand; this pins it (and catches a slip in either).
    assert predict(5.0, b1, B2, 199)["pi"] == pytest.approx(pi, abs=0.01)


def test_predict_rejects_bad_arguments():
    for args in ((-1.0, B1, B2, 10), (1.0, B1, B2, 10, 10)):
        with pytest.raises(ValueError):
            predict(*args)
