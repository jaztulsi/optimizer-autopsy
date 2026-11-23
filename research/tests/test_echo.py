"""CPU checks for X4's echo law (research/analysis/damage_law.py), its arms (research/autopsy/echo.py) and mve's echo
battery. The end-to-end test runs at lr 1e-6, where the law is near exact, so it checks the instrument; whether the law
holds at the real learning rate is X4's GPU question, not CI's."""

import json
import math

import numpy as np
import pytest
import torch

from research.analysis import damage_law
from research.autopsy import echo
from research.baselines import spam
from research.harness.fork import run_fork
from research.harness.snapshot import capture
from research.harness.trunk import train_forward
from research.tests.test_baselines import _CFG, _data_dir, _fresh, _tmpdir


def test_echo_law_values():
    assert damage_law.ceiling() == pytest.approx(5.837, abs=1e-3)
    assert damage_law.ceiling((0.9, 0.999)) == pytest.approx(31.77, abs=0.01)
    assert damage_law.echo(2) == pytest.approx(1.877, abs=0.005)
    assert damage_law.echo(10) == pytest.approx(4.92, abs=0.02)
    assert damage_law.echo(1e6) == pytest.approx(damage_law.ceiling(), rel=1e-3)
    for z, c in ((10, 0.3), (30, 0.5)):
        assert damage_law.echo(z, c, c * c) == pytest.approx(damage_law.echo(c * z))  # a clip by c is Ψ(cz)
    assert damage_law.echo(7, 1, 0) == pytest.approx(7)  # undo v alone: the whole spike comes back


def _close(a, b):
    return torch.allclose(a, b, rtol=1e-4, atol=1e-6 * float(b.abs().max()))


def test_each_arm_on_the_spike_step():
    T, (b1, b2), lr = echo.TARGET, _CFG["optim"]["betas"], _CFG["optim"]["lr"]
    with _tmpdir() as d:
        data = _data_dir(d)
        model, opt = _fresh()
        train_forward(model, opt, _CFG, data, 0, 3, "cpu")
        snap = capture(model, opt, step=3)
        w0, m0, v0 = (snap[k] for k in ("w", "m", "v"))

        def fork(arm=None, z=10.0):
            hook, out = echo.make(arm, z, 1, at=3, steps=1) if arm else (None, {})
            got = {}

            def grab(step, info):
                for n, p in info["model"].named_parameters():
                    s = info["opt"].state[p]
                    got[n] = (p.detach().clone(), s["exp_avg"].clone(), s["exp_avg_sq"].clone())

            run_fork(_CFG, data, snap, steps=1, device="cpu", seed=False, on_step=grab, pre_opt_step=hook)
            return got, out

        plain, _ = fork()
        none, out = fork("none")
        s = echo.signs(w0[T].shape)
        assert _close(none[T][1] - plain[T][1], (1 - b1) * 10 * v0[T].sqrt() * s)  # m took ε1 G
        assert out["lrs"] == pytest.approx([lr * math.sqrt(1 - b2**4) / (1 - b1**4)])  # Adam's 4th step
        assert out["d"] == pytest.approx(float(((none[T][0].double() - w0[T].double()) * s).mean()))

        undo_v, _ = fork("undo_v")
        assert torch.equal(undo_v[T][1], none[T][1]) and _close(undo_v[T][2], plain[T][2])  # in m, not in v

        clip, out = fork("clip")
        assert 0 < out["c"] < 1
        for n in (T, "transformer.wte.weight"):  # the whole gradient is scaled, the plant with it
            ref = none if n == T else plain
            assert _close(clip[n][1] - b1 * m0[n], out["c"] * (ref[n][1] - b1 * m0[n]))

        sp, _ = fork("spam", z=1000.0)  # far past sqrt(θ) on every entry, whatever g is
        assert _close((sp[T][1] - b1 * m0[T]) / (1 - b1), s * (spam.THETA * v0[T]).sqrt())

        stable, out = fork("stable")
        g = (none[T][1] - b1 * m0[T]) / (1 - b1)
        rms = float((g.square() / (none[T][2] / (1 - b2**4))).mean().sqrt())  # from the stepped state
        assert out["rms"] == pytest.approx(rms, rel=1e-4) and rms > 1
        assert _close(stable[T][0] - w0[T], (none[T][0] - w0[T]) / rms)
    with pytest.raises(ValueError, match="arm"):
        echo.make("zclip", 1.0, 1, 0, 1)


def test_mve_echo_battery_matches_the_law_at_small_lr(tmp_path, monkeypatch):
    from research.autopsy import mve

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for split in ("train", "val"):
        np.random.default_rng(0).integers(0, 16, size=8000, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    cfg = {**_CFG, "optim": {**_CFG["optim"], "lr": 1e-6}, "train": {**_CFG["train"], "max_steps": 160}}
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg))  # JSON is valid YAML
    argv = ["--config", str(tmp_path / "cfg.yaml"), "--inject-step", "100", "--battery", "echo", "--echo-z", "6,30"]
    argv += ["--horizons", "60", "--data-dir", str(tmp_path), "--out", str(tmp_path / "out"), "--skip-preflight"]
    mve.main(argv)
    out = next((tmp_path / "out").glob("*.json"))
    res = json.loads(out.read_text())
    cells = res["echo"]["cells"]
    assert len(cells) == 10 and res["echo"]["verdict"] == "pass"
    for k, c in cells.items():
        if c["arm"] != "stable":
            assert c["ratio"] == pytest.approx(1, abs=0.1), (k, cells)
    for z in ("6", "30"):
        assert cells[f"spam_z{z}"]["left"] == pytest.approx(1, abs=1e-6)  # below sqrt(θ) = 70.7: untouched
        assert 0.8 <= cells[f"stable_z{z}"]["left"] < 1  # trims the spike step only
        assert all(res["branches"][f"X4_clip_z{z}{s}"]["metrics"]["c"] < 1 for s in "+-")
    mve.main(argv)  # resume: every fork is on disk, the summary is recomputed
    assert json.loads(out.read_text())["echo"] == res["echo"]
    with pytest.raises(SystemExit):
        mve.main(argv + ["--kind", "corrupt_batch"])
