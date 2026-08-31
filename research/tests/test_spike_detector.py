"""Regression tests for Task 8's topology-aware spike-detection policies."""

from research.spikes.tune_detector import DetectorParams, detection_qualifies, dod_check, score_spike


def _instantaneous_run(inject_step=40):
    n = 70
    losses = [2.0] * n
    gradnorms = [1.0] * n
    losses[inject_step] = 6.0
    gradnorms[inject_step] = 20.0
    return {
        "losses": losses,
        "gradnorms": gradnorms,
        "inject_step": inject_step,
        "min_lead": 0,
        "allow_at_peak": True,
    }


def _delayed_run(inject_step=40):
    n = 70
    losses = [2.0] * n
    gradnorms = [1.0] * n
    gradnorms[inject_step] = 20.0
    losses[inject_step + 3] = 6.0
    return {
        "losses": losses,
        "gradnorms": gradnorms,
        "inject_step": inject_step,
        "min_lead": 2,
        "allow_at_peak": False,
    }


def _clean_run(inject_step=40):
    return {
        "losses": [2.0] * 70,
        "gradnorms": [1.0] * 70,
        "inject_step": inject_step,
        "min_lead": 2,
        "allow_at_peak": False,
    }


def test_instantaneous_recipe_is_onset_detected_with_zero_lead():
    run = _instantaneous_run()
    score = score_spike(
        run["losses"],
        run["gradnorms"],
        run["inject_step"],
        DetectorParams(),
        min_lead=0,
        allow_at_peak=True,
    )
    assert score["occurred"]
    assert score["detected"]
    assert score["lead"] == 0
    assert score["detection_mode"] == "onset"


def test_onset_exception_cannot_relax_a_predictive_recipe():
    run = _instantaneous_run()
    score = score_spike(
        run["losses"],
        run["gradnorms"],
        run["inject_step"],
        DetectorParams(),
        min_lead=2,
        allow_at_peak=False,
    )
    assert score["occurred"]
    assert not score["detected"]
    assert score["lead"] is None
    assert score["detection_mode"] == "predictive"


def test_v6_gate_accepts_two_policy_compliant_recipes_only():
    runs = [_delayed_run(), _instantaneous_run(), _clean_run(45), _clean_run(50)]
    verdict = dod_check(runs, DetectorParams(), L=2, f=0.05, min_recipes=2)
    assert verdict["passed"]
    assert verdict["need"] == 2
    assert verdict["n_detected"] == 2
    assert verdict["fp_rate"] == 0.0


def test_original_three_of_four_default_is_preserved():
    runs = [_delayed_run(), _instantaneous_run(), _clean_run(45), _clean_run(50)]
    verdict = dod_check(runs, DetectorParams(), L=2, f=0.05)
    assert not verdict["passed"]
    assert verdict["need"] == 3
    assert verdict["n_detected"] == 2


def test_committed_corrupt_batch_summary_regrades_as_onset_not_prediction():
    detected, lead = detection_qualifies(
        occurred=True,
        inject_step=160,
        trigger_step=160,
        peak_step=160,
        min_lead=0,
        allow_at_peak=True,
    )
    assert detected and lead == 0

    predictive, predictive_lead = detection_qualifies(
        occurred=True,
        inject_step=160,
        trigger_step=160,
        peak_step=160,
        min_lead=2,
        allow_at_peak=False,
    )
    assert not predictive and predictive_lead is None
