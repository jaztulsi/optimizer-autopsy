"""Trunk run: the main training trajectory we snapshot and later fork from.

Trains a model while periodically capturing (w, m, v) snapshots and logging the spike detector,
so a failure at step T can be reconstructed and forked.
"""

# TODO: run_trunk(cfg) -> train loop; snapshot at cfg.snapshot_steps; log loss + detector signal.
# TODO: resume_trunk(cfg, from_snapshot) -> continue a trunk from a saved snapshot.
