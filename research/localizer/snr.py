"""SNR localizer: signal-to-noise of the Adam update m / (sqrt(v) + eps) per parameter/group.

A collapsing SNR flags directions where the second moment has gone stale relative to the first.
"""

# TODO: snr(m, v, eps) -> elementwise |m| / (sqrt(v) + eps).
# TODO: group_snr(snapshot, grouping) -> aggregate SNR per layer / head / tensor.
# TODO: snr_score(snapshot) -> normalized per-group implication score for the poison combiner.
