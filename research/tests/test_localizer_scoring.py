"""Numpy-only tests for the localizer's scoring layer (Weeks 4-5): groups, SNR, stability summaries,
poison ratios, psi_k, null thresholds and `localize`. No torch, no model -- synthetic vectors with a
known answer, including a planted v-inflation the localizer must find.
"""

import numpy as np
import pytest

from research.localizer import poison, snr, stability


def _layout(sizes):
    out, off = [], 0
    for i, n in enumerate(sizes):
        out.append({"name": f"t{i}", "shape": (n,), "offset": off, "numel": n})
        off += n
    return out


def _orthonormal(k, dim, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, k)))
    return q.T


# ---- groups ----------------------------------------------------------------------------------------


def test_groups_by_tensor_tile_the_vector():
    lay = _layout([3, 5, 2])
    g = snr.groups_by_tensor(lay)
    snr.check_groups(g, 10)
    assert g["t1"] == {"tensor": "t1", "rows": None, "start": 3, "stop": 8}


def test_groups_by_head_splits_c_attn_rows():
    C, H = 4, 2  # hd = 2
    lay = [
        {"name": "wte", "shape": (6, C), "offset": 0, "numel": 6 * C},
        {"name": "h.0.attn.c_attn.weight", "shape": (3 * C, C), "offset": 24, "numel": 3 * C * C},
    ]
    g = snr.groups_by_head(lay, n_head=H)
    snr.check_groups(g, 24 + 3 * C * C)
    k1 = g["h.0.attn.c_attn.weight[k1]"]  # k block starts at row C, head 1 at +hd
    assert k1["rows"] == (C + 2, C + 4)
    assert k1["start"] == 24 + (C + 2) * C and k1["stop"] == 24 + (C + 4) * C
    assert len([x for x in g if x.startswith("h.0.attn.c_attn.weight[")]) == 3 * H


def test_check_groups_catches_a_gap():
    with pytest.raises(ValueError):
        snr.check_groups({"a": {"start": 0, "stop": 2}, "b": {"start": 3, "stop": 5}}, 5)


def test_flat_from_named_order_and_size():
    lay = _layout([2, 1])
    flat = snr.flat_from_named({"t1": [9.0], "t0": [1.0, 2.0]}, lay)
    assert flat.tolist() == [1.0, 2.0, 9.0]
    with pytest.raises(ValueError):
        snr.flat_from_named({"t0": [1.0], "t1": [9.0]}, lay)


# ---- update SNR -----------------------------------------------------------------------------------


def test_update_snr_values_and_bias_correction():
    m, v = np.array([0.1, -0.2]), np.array([0.01, 0.04])
    assert np.allclose(snr.snr(m, v, eps=0.0), [1.0, 1.0])
    b1, b2 = 0.9, 0.95
    want = (np.abs(m) / (1 - b1)) / np.sqrt(v / (1 - b2))
    assert np.allclose(snr.snr(m, v, eps=0.0, adam_step=1, betas=(b1, b2)), want)
    with pytest.raises(ValueError):
        snr.snr(m, -v)


def test_group_snr_and_score():
    lay = _layout([2, 2])
    groups = snr.groups_by_tensor(lay)
    snap = {"m": {"t0": [1.0, 1.0], "t1": [0.1, 0.1]}, "v": {"t0": [1.0, 1.0], "t1": [1.0, 1.0]}}
    gs = snr.group_snr(snap, groups, lay, eps=0.0)
    assert gs == pytest.approx({"t0": 1.0, "t1": 0.1})
    sc = snr.snr_score(gs)
    assert sc["t1"] > 0 > sc["t0"]  # less signal than the median -> implicated


def test_directional_snr():
    proj = np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]])  # col0: mean 2, sd sqrt(2); col1 const; col2 zeros
    got = snr.directional_snr(proj)
    assert got[0] == pytest.approx(2.0 / np.sqrt(2.0 / 2))
    assert got[1] == np.inf and got[2] == 0.0
    with pytest.raises(ValueError):
        snr.directional_snr(np.ones((1, 3)))


# ---- stability ----------------------------------------------------------------------------------------


def test_subspace_overlap():
    U = _orthonormal(3, 20)
    R, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))
    assert stability.subspace_overlap(U, R @ U) == pytest.approx(1.0)  # rotated inside the subspace
    Q = _orthonormal(6, 20, seed=2)
    assert stability.subspace_overlap(Q[:3], Q[3:]) == pytest.approx(0.0, abs=1e-12)


def test_summarize_pairs_splits_and_keeps_failures():
    U = _orthonormal(2, 10)
    ok = {"ok": True, "eigvals": np.array([2.0, 1.0]), "eigvecs": U}
    solves = [
        {"batch": 0, "seed": 0, **ok},
        {"batch": 0, "seed": 1, **ok},
        {"batch": 1, "seed": 0, **ok, "eigvals": np.array([2.2, 1.0])},
        {"batch": 1, "seed": 1, "ok": False, "reason": "boom"},
    ]
    s = stability.summarize_pairs(solves)
    assert s["n_ok"] == 3 and s["failures"] == [{"batch": 1, "seed": 1, "reason": "boom"}]
    assert s["same_batch"]["n_pairs"] == 1 and s["cross_batch"]["n_pairs"] == 2
    assert s["cross_batch"]["max_eigval_spread"] == pytest.approx(0.2 / 2.2)  # |a - b| / |b|, b = batch 1


# ---- poison ratios and psi_k -------------------------------------------------------------------------


def test_poison_ratios_and_psi_k():
    U = np.eye(4)[:2]  # directions e0, e1
    vbar = np.ones(4)
    v = np.array([3.0, 1.0, 1.0, 5.0])
    assert poison.poison_ratios(U, v, vbar, eps=0.0).tolist() == [2.0, 0.0]
    assert poison.psi_k(U, v, vbar) == pytest.approx(2.0 / np.sqrt(4.0 + 16.0))
    assert poison.psi_k(U, vbar, vbar) == 0.0
    with pytest.raises(ValueError):
        poison.psi_k(2 * U, v, vbar)  # rows not unit length


def test_group_drift_sign_and_share():
    groups = snr.groups_by_tensor(_layout([2, 2]))
    v, vbar = np.array([4.0, 4.0, 0.5, 0.5]), np.ones(4)
    d = poison.group_drift(v, vbar, groups)
    assert d["t0"] == pytest.approx(np.log(4.0)) and d["t1"] == pytest.approx(np.log(2.0))
    assert poison.group_sign(v, vbar, groups) == {"t0": 1, "t1": -1}
    U = np.array([[1.0, 0, 0, 0], [0, 0, 0.6, 0.8]])
    assert poison.curvature_share(U, [1], groups) == pytest.approx({"t0": 0.0, "t1": 1.0})
    assert poison.curvature_share(U, [0, 1], groups) == pytest.approx({"t0": 0.5, "t1": 0.5})
    assert poison.curvature_share(U, [], groups) == {"t0": 0.0, "t1": 0.0}


# ---- thresholds -------------------------------------------------------------------------------------


def test_null_thresholds_refuse_missing_or_thin_data():
    groups = snr.groups_by_tensor(_layout([2, 2]))
    with pytest.raises(ValueError, match="calibration"):
        poison.null_thresholds([], groups=groups)
    row = {"v": np.ones(4), "vbar": np.ones(4)}
    with pytest.raises(ValueError, match="need >="):
        poison.null_thresholds([row, row], groups=groups, min_steps=3)  # 2 normal steps < 3
    with pytest.raises(ValueError, match="dir_snr"):
        poison.null_thresholds([row] * 5, U=np.eye(4)[:2], groups=groups)


# ---- localize: a planted v-inflation must be found --------------------------------------------------


def _planted_case(kappa=100.0, n_normal=8, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [40, 30, 50, 20, 60, 25]
    lay = _layout(sizes)
    groups = snr.groups_by_tensor(lay)
    dim = sum(sizes)
    vbar = rng.uniform(1e-4, 1e-3, dim)

    def noisy():
        return vbar * np.exp(rng.normal(0.0, 0.05, dim))

    # A frozen basis whose first direction lives in t3 (the planted tensor); the rest are generic.
    t3 = groups["t3"]
    u0 = np.zeros(dim)
    u0[t3["start"] : t3["stop"]] = rng.standard_normal(t3["stop"] - t3["start"])
    rest = rng.standard_normal((dim, 3))
    q, _ = np.linalg.qr(np.column_stack([u0, rest]))
    U = q.T
    if U[0] @ u0 < 0:
        U[0] = -U[0]

    null = [{"v": noisy(), "vbar": vbar, "dir_snr": rng.uniform(1.0, 4.0, 4)} for _ in range(n_normal)]
    v = noisy()
    v[t3["start"] : t3["stop"]] *= kappa
    dir_snr = np.array([0.2, 5.0, 6.0, 7.0])  # only u0 is below the null SNR range [1, 4)
    return lay, groups, U, v, vbar, null, dir_snr


def test_localize_finds_planted_tensor_with_eigenbasis():
    lay, groups, U, v, vbar, null, dir_snr = _planted_case()
    loc = poison.localize(v=v, vbar=vbar, groups=groups, null=null, t=123, U=U, dir_snr=dir_snr, k_select=3)
    assert loc["selected"][0] == "t3"
    assert loc["groups"]["t3"]["sign"] == 1 and loc["ref"] == "vbar" and loc["t"] == 123
    assert 0 in loc["provenance"]["poisoned_directions"]
    assert loc["provenance"]["curvature_share"]["t3"] > 0.9


def test_localize_no_hvp_path_finds_planted_tensor():
    lay, groups, U, v, vbar, null, _ = _planted_case()
    loc = poison.localize(v=v, vbar=vbar, groups=groups, null=null, t=5)
    assert loc["selected"][0] == "t3" and loc["method"] == "drift (no HVP)"
    assert "poisoned_directions" not in loc["provenance"]


def test_localize_false_positive_rate_on_normal_steps():
    # A fresh normal step should select nothing most of the time: q_d = 0.95 is meant as the chance a
    # normal step selects ANY group. 8 normal steps make the threshold coarse, so allow up to 25%.
    hits = 0
    for seed in range(100):
        lay, groups, U, v, vbar, null, dir_snr = _planted_case(seed=seed)
        rng = np.random.default_rng(10_000 + seed)
        fresh = vbar * np.exp(rng.normal(0.0, 0.05, vbar.size))
        hits += bool(poison.localize(v=fresh, vbar=vbar, groups=groups, null=null, t=0)["selected"])
    assert hits <= 25, f"{hits}/100 normal steps selected a group"


def test_localize_selects_nothing_when_nothing_moved():
    lay, groups, U, v, vbar, null, dir_snr = _planted_case()
    loc = poison.localize(v=vbar.copy(), vbar=vbar, groups=groups, null=null, t=0, U=U, dir_snr=dir_snr)
    assert loc["selected"] == [] and loc["provenance"]["psi_k"] == 0.0


def test_localize_without_calibration_raises():
    lay, groups, U, v, vbar, null, dir_snr = _planted_case()
    with pytest.raises(ValueError, match="calibration"):
        poison.localize(v=v, vbar=vbar, groups=groups, null=[], t=0, U=U, dir_snr=dir_snr)
    with pytest.raises(ValueError, match="dir_snr is required"):
        poison.localize(v=v, vbar=vbar, groups=groups, null=null, t=0, U=U)


def test_validate_localization_rejects_bad_output():
    lay, groups, U, v, vbar, null, dir_snr = _planted_case()
    loc = poison.localize(v=v, vbar=vbar, groups=groups, null=null, t=0)
    bad = {k: v for k, v in loc.items() if k != "ref"}
    with pytest.raises(ValueError, match="ref"):
        poison.validate_localization(bad)
    loc["selected"] = ["nope"]
    with pytest.raises(ValueError, match="not in groups"):
        poison.validate_localization(loc)
    assert poison.rows_slice([2, 5]) == slice(2, 5) and poison.rows_slice(None) is None


# ---- Week 7: spike-vs-normal separation and cross-seed agreement --------------------------------------


def test_auc_and_jaccard():
    from research.localizer import validate

    assert validate.auc([3, 4], [1, 2]) == 1.0
    assert validate.auc([1], [1]) == 0.5
    assert validate.auc([], [1]) is None
    assert validate.auc([2, None, float("nan")], [1]) == 1.0
    assert validate.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert validate.jaccard([], []) == 1.0


def test_report_per_seed_and_agreement():
    from research.localizer import validate

    rows = [
        {"label": "spike", "seed": 0, "kind": "lr_bump", "psi_k": 0.9, "selected": ["a", "b"]},
        {"label": "spike", "seed": 1, "kind": "lr_bump", "psi_k": 0.2, "selected": ["a"]},
        {"label": "normal", "seed": 0, "psi_k": 0.1, "selected": []},
        {"label": "normal", "seed": 1, "psi_k": 0.3, "selected": []},
    ]
    sep = validate.separation(rows, "psi_k")
    assert sep["auc"] == pytest.approx(3 / 4)
    assert sep["per_seed"][0]["auc"] == 1.0 and sep["per_seed"][1]["auc"] == 0.0
    agree = validate.selection_agreement(rows)
    assert agree["lr_bump"] == {"n_spikes": 2, "mean_jaccard": 0.5, "empty_selections": 0}
