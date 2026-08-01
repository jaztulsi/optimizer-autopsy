"""Tune the online spike detector against induced spikes (known ground truth).

Calibrates thresholds so the detector fires at the spike onset, not late, with low false positives.
"""

# TODO: detector_signal(loss_history) -> the statistic thresholded to flag a spike (e.g. z-score).
# TODO: tune(trunk_logs, labels) -> pick thresholds maximizing detection @ fixed false-positive rate.
