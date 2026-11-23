"""CPU checks for research/autopsy/appeal.py (RML-3's kill-test measurement hook), the hard_rows recipe and mve's
appeal battery: appeal_build_plan.md §6 tests 1, 2, 6, 7, 8 and 9. Whether junk rows score low and hard rows high is
the GPU kill test's question, not CI's."""

import json
import math

import numpy as np
import pytest
import torch

from research.autopsy import appeal
from research.data.prepare import get_batch
from research.harness.fork import run_fork
from research.harness.snapshot import capture
from research.harness.trunk import train_forward
from research.spikes.induce import induce
from research.tests.test_baselines import _CFG, _data_dir, _fresh, _tmpdir


def test_flag_rule_is_two_sided_and_robust():
    base = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0, 2.02, 1.98]  # median 2.0, MAD 0.05 with any one row added
    assert appeal.flag_rows(torch.tensor(base + [9.0])).tolist() == [False] * 8 + [True]
    assert appeal.flag_rows(torch.tensor(base + [0.01])).tolist() == [False] * 8 + [True]  # a copied n-gram
    assert not appeal.flag_rows(torch.tensor(base + [2.2])).any()  # z = 2.7
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])  # median 3, MAD 1
    assert torch.allclose(appeal.robust_z(x), (x - 3) / 1.4826)


def test_cosine_p_and_phi_on_hand_made_numbers():
    a, b, one = (torch.tensor(v, dtype=torch.float64) for v in ([1.0, 0.0], [1.0, 1.0], [1.0, 1.0]))
    assert appeal.cosine_p(a, b, one) == pytest.approx(1 / math.sqrt(2))
    assert appeal.cosine_p(a, -b, one) == pytest.approx(-1 / math.sqrt(2))
    assert appeal.cosine_p(a, b, torch.tensor([4.0, 1.0], dtype=torch.float64)) == pytest.approx(2 / math.sqrt(5))
    assert appeal.cosine_p(0 * a, b, one) == 0.0
    assert appeal.p_score(0.5, [0.0, 1.0, 0.2, 0.7]) == 0.5  # strictly below only
    assert appeal.p_score(0.2, [0.2, 0.2]) == 0.0
    assert [appeal.phi(p) for p in (0.0, 0.05, 0.15, 0.25, 1.0)] == pytest.approx([0, 0, 0.5, 1, 1])
    assert len({round(appeal.phi(k / 16), 12) for k in range(17)}) == 5  # sixteen shadows: five values


def test_row_losses_are_each_rows_own_loss_and_keep_train_mode():
    model, _ = _fresh()
    gen = torch.Generator().manual_seed(0)
    x, y = torch.randint(0, 16, (4, 8), generator=gen), torch.randint(0, 16, (4, 8), generator=gen)
    y[1, :3] = -1  # ignored targets drop out of the row mean
    got = appeal.row_losses(model, x, y)
    assert model.training
    want = torch.stack([model(x[i : i + 1], y[i : i + 1])[1] for i in range(4)]).detach()
    assert torch.allclose(got, want, atol=1e-6)


def test_hooked_run_is_bitwise_the_plain_run_and_logs_one_verdict_per_flagged_step():
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh()
        train_forward(model, opt, _CFG, data, 0, 3, "cpu")
        snap = capture(model, opt, step=3)

        def fork(pre=None, pre_opt=None):
            return run_fork(_CFG, data, snap, steps=4, device="cpu", seed=False, pre_step=pre, pre_opt_step=pre_opt)

        plain = fork()
        pre, pre_opt, st = appeal.make(data, z=1e9)  # nothing flagged
        assert fork(pre, pre_opt) == plain and st["verdicts"] == st["flagged"] == []

        def row0_at_3(step, losses):
            mask = torch.zeros(len(losses), dtype=torch.bool)
            mask[0] = step == 3
            return mask

        pre, pre_opt, st = appeal.make(data, lag=2, flag_fn=row0_at_3)
        assert fork(pre, pre_opt) == plain  # recording rows and shadows leaves training alone
        (v,) = st["verdicts"]
        assert (v["step"], v["n"], v["rows"], sorted(v["shadows"])) == (3, 1, [0], [1, 2, 3])  # all kept rows < 16
        assert len(v["shadow_T"]) == 3 and all(abs(t) <= 2 + 1e-9 for t in v["T"] + v["shadow_T"])  # 2 cosines
        assert v["p"] == [appeal.p_score(v["T"][0], v["shadow_T"])] and v["phi"] == [appeal.phi(v["p"][0])]
        first = json.loads(json.dumps(st["verdicts"]))
        assert fork(pre, pre_opt) == plain and st["verdicts"] == first  # reused for a second run: no leak

        pre, pre_opt, st = appeal.make(data, flag_fn=lambda s, losses: torch.tensor([True, True, True, False]))
        assert fork(pre, pre_opt) == plain and st["batch_events"] == [3, 4, 5, 6] and st["verdicts"] == []


def test_grad_accum_above_one_raises():
    pre, _, _ = appeal.make("unused")
    model, opt = _fresh()
    cfg = {**_CFG, "train": {**_CFG["train"], "grad_accum": 2}}
    with pytest.raises(ValueError, match="grad_accum"):
        pre(0, {"model": model, "opt": opt, "cfg": cfg, "batch": None})
    with pytest.raises(ValueError, match="lag"):
        appeal.make("unused", lag=0)  # K = 0 would never close a record


def test_hard_rows_injects_the_pools_top_rows_into_the_real_batch():
    with _tmpdir() as d:
        data = _data_dir(d)
        model, _ = _fresh()
        r = induce("hard_rows", _CFG, inject_step=3, width=2, frac=0.5, data_dir=data)
        seen = set()
        for step in (3, 4):
            ctx = {"model": model, "cfg": _CFG, "batch": None}
            r.pre_step(step, ctx)
            x, y = ctx["batch"]
            slots = [2 * 6 + 4 * (step - 3) + j for j in range(4)]  # past 2 * max_steps, four fresh slots a step
            px, py = (torch.cat(t) for t in zip(*(get_batch("train", s, 4, 8, data) for s in slots)))
            loss = torch.stack([model(px[i : i + 1], py[i : i + 1])[1] for i in range(16)]).detach()
            picked = [next(i for i in range(16) if torch.equal(px[i], x[row])) for row in range(2)]
            assert set(picked) == set(loss.topk(2).indices.tolist()) and torch.equal(y[:2], py[picked])
            rx, ry = get_batch("train", step, 4, 8, data)
            assert torch.equal(x[2:], rx[2:]) and torch.equal(y[2:], ry[2:])  # the rest is the real batch
            seen |= {tuple(px[i].tolist()) for i in picked}
        assert len(seen) == 4  # no row repeats across steps
        ctx = {"model": model, "cfg": _CFG, "batch": None}
        r.pre_step(5, ctx)
        assert ctx["batch"] is None  # outside the window

        big = {**_CFG, "train": {**_CFG["train"], "max_steps": 1000}}  # pool starts past the 8000-token shard
        with pytest.raises(ValueError, match="wraps"):
            induce("hard_rows", big, inject_step=3, data_dir=data).pre_step(3, {"model": model, "cfg": big})
    with pytest.raises(ValueError, match="data_dir"):
        induce("hard_rows", _CFG, inject_step=3)


def test_mve_appeal_battery_logs_verdicts_for_the_injected_rows(tmp_path, monkeypatch):
    from research.autopsy import mve

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for split in ("train", "val"):
        np.random.default_rng(0).integers(0, 16, size=8000, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    cfg = {**_CFG, "train": {**_CFG["train"], "max_steps": 10}}
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg))  # JSON is valid YAML
    argv = ["--config", str(tmp_path / "cfg.yaml"), "--inject-step", "3", "--battery", "appeal", "--appeal-k", "3"]
    argv += ["--horizons", "4", "--data-dir", str(tmp_path), "--out", str(tmp_path / "out"), "--skip-preflight"]
    mve.main(argv + ["--kind", "hard_rows"])
    br = json.loads(next((tmp_path / "out").glob("*.json")).read_text())["branches"]
    m = br["B0"]["metrics"]
    assert (m["k"], m["injected_rows"], [v["rows"] for v in m["verdicts"]]) == (3, 1, [[0]])
    assert br["B0"]["train_losses"][0] != br["B*"]["train_losses"][0]  # the recipe fired
    with pytest.raises(SystemExit):
        mve.main(argv + ["--kind", "corrupt_batch"])
