"""Poison score: combine SNR + curvature (+ others) into a single localization map.

Output is a ranked set of parameters/groups the repair operator will act on.
"""

# TODO: poison_score(snapshot, batch, weights) -> combined per-group score from snr + curvature.
# TODO: localize(snapshot, batch, k) -> top-k implicated groups (the repair target set).
