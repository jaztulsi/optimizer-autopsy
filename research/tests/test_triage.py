"""CPU tests for damage triage (research/repair/triage.py): exact feature values, the read-only guarantee (alone and
inside the real training loop), AUC, the leave-one-seed-out report, and the dataset reader."""

from __future__ import annotations

import os

# Must be set before torch is imported anywhere in the process; setdefault keeps a launcher's value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json  # noqa: E402
import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from research.autopsy import probation  # noqa: E402
from research.repair import triage  # noqa: E402


def test_features_exact_and_read_only():
    b1, b2, lr = 0.9, 0.95, 3e-3
    model = torch.nn.Linear(4, 2, bias=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(b1, b2))
    model(torch.ones(1, 4)).sum().backward()
    opt.step()  # a real step, so the state exists
    st = opt.state[model.weight]
    st["exp_avg"].zero_()
    st["exp_avg_sq"].fill_(1.0)
    g = torch.tensor([[6.0, 6.0, 1.0, 1.0], [0.5, 0.5, 0.25, 0.25]])
    model.weight.grad = g.clone()

    def snap():
        return model.weight.detach().clone(), model.weight.grad.clone(), {k: t.clone() for k, t in st.items()}

    before = snap()
    out = {}
    hook = triage.features_hook([7], out)
    assert hook(8, {"model": model, "opt": opt}) is None and out == {}  # not in steps: nothing written
    assert hook(7, {"model": model, "opt": opt}) is None
    after = snap()
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
    assert all(torch.equal(before[2][k], after[2][k]) for k in before[2])

    f = out["triage"]["7"]
    assert list(f) == list(triage.FEATURES) and all(isinstance(v, float) for v in f.values())
    assert math.isclose(f["log_x_mean"], math.log((1 - b2) * float(g.double().square().mean())), rel_tol=1e-12)
    assert math.isclose(f["log_x_max_tensor"], f["log_x_mean"], rel_tol=1e-12)  # one tensor
    assert f["frac_knee"] == 2 / 8  # (1-b2)*36 = 1.8 >= x_knee(0.9, 0.95) = 1.70; 1.0 gives 0.05
    assert f["cos_gm"] == 0.0  # m = 0
    assert math.isclose(f["log_lr"], math.log(lr)) and math.isclose(f["log_mem"], -math.log(1 - b2))
    assert math.isclose(f["log_step"], math.log(7))
    assert probation.x_knee(b1, b2) < 1.8


def test_hook_is_read_only_inside_the_real_training_loop():
    from research.harness.fork import run_fork
    from research.harness.snapshot import capture
    from research.harness.trunk import train_forward
    from research.tests.test_baselines import _CFG, _data_dir, _fresh, _tmpdir

    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh()
        train_forward(model, opt, _CFG, data, 0, 3, "cpu")
        snap = capture(model, opt, step=3)

        def fork(hook=None):
            return run_fork(_CFG, data, snap, steps=3, device="cpu", seed=False, pre_opt_step=hook)

        plain = fork()
        out = {}
        assert fork(triage.features_hook([4], out)) == plain  # bitwise identical losses
        assert list(out["triage"]) == ["4"]
        assert all(math.isfinite(v) for v in out["triage"]["4"].values())


def test_auc():
    assert triage.auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert triage.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert math.isnan(triage.auc([0.1, 0.2], [1, 1]))
    # within strata: pooled, 0.2 (positive) < 0.8 (negative) costs a pair; within a / b both pairs are right, and
    # stratum c (one class) adds no pair
    s, y = [0.9, 0.8, 0.2, 0.1, 0.5], [1, 0, 1, 0, 1]
    assert triage.auc(s[:4], y[:4]) == 0.75
    assert triage.auc(s, y, ["a", "a", "b", "b", "c"]) == 1.0
    assert math.isnan(triage.auc(s, y, ["a", "b", "c", "d", "e"]))


def _synthetic(label_col):
    rng = np.random.default_rng(0)
    n = 240
    X = rng.normal(size=(n, len(triage.FEATURES)))
    y = (X[:, label_col] > 0).astype(int)
    return X, y, np.repeat(np.arange(4), n // 4), np.tile([1.0, 10.0, 30.0], n // 3)


def test_report_keeps_a_gain_and_flags_the_ceiling():
    X, y, groups, dose = _synthetic(label_col=1)  # label from a non-baseline column; column 0 is noise
    r = triage.report(X, y, groups, dose)
    assert r["auc"] > 0.95 and abs(r["auc_baseline"] - 0.5) < 0.15 and r["keep"] and not r["ceiling"]
    masks = [dose == c for c in (1, 10, 30)]
    assert r["pairs_within_stratum"] == sum(int(y[k].sum()) * int((1 - y[k]).sum()) for k in masks)
    X, y, groups, dose = _synthetic(label_col=0)  # spike size alone says it all
    r = triage.report(X, y, groups, dose)
    assert r["ceiling"] and not r["keep"] and r["n"] == len(y) and r["positives"] == int(y.sum())


def test_fit_stays_finite_on_one_class():
    X = np.random.default_rng(1).normal(size=(10, 3))
    w = triage.fit(X, np.zeros(10, int))["w"]
    # the penalised optimum has the intercept near -1.6; without the intercept penalty Newton walks it toward -inf
    assert np.all(np.isfinite(w)) and np.abs(w).max() < 5


def test_dataset(tmp_path):
    def feats(x):
        return dict.fromkeys(triage.FEATURES, x)

    def arm(v, tri=None):
        return {"val": {"50": v - 0.5, "99": v}, "metrics": {"triage": tri} if tri else {}}

    paths = []
    for seed, (v1, v30) in ((1, (2.001, 2.02)), (2, (2.002, 2.005))):
        bad = feats(float("nan")) if seed == 2 else feats(0.5)
        res = {
            "stamp": {"cfg": {"seed": seed, "optim": {"betas": [0.9, 0.95]}}},
            "branches": {
                "B*": arm(2.0),
                "Px1": arm(v1, {"20": feats(-9.0), "21": feats(0.0)}),
                "Px30": arm(v30, {"20": feats(1.0), "21": feats(9.0)}),
                "Px30|U_v@0": arm(v30, {"20": feats(1.0)}),  # not a dose arm: ignored
                "Px10": arm(2.0),  # a dose arm from before the hook: counted, not used
            },
        }
        if seed == 2:
            res["branches"]["Px30"]["metrics"]["triage"]["20"] = bad  # onset has a NaN: skipped
        p = tmp_path / f"r{seed}.json"
        p.write_text(json.dumps(res))  # json writes NaN as a bare NaN, which json.load reads back
        paths.append(p)
    X, y, groups, rows, skipped = triage.dataset(paths)
    assert X.shape == (3, 7) and skipped == {"no_triage": 2, "nonfinite": 1}
    assert y.tolist() == [0, 1, 0] and groups.tolist() == [1, 1, 2]  # Y(none): .001, .02, .002 vs 0.01
    assert [r["branch"] for r in rows] == ["Px1", "Px30", "Px1"] and rows[1]["c"] == 30.0
    assert rows[0]["stratum"] == rows[2]["stratum"] == "Px1|b0.9-0.95|inj20"  # same dose, betas and onset
    assert X[0, 0] == -9.0 and X[1, 0] == 1.0  # features come from the smallest step
