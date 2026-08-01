"""Induce loss spikes so we have ground-truth failures to autopsy.

Controlled triggers (LR bump, bad batch, high-norm injection) give known spike locations/causes.
"""

# TODO: induce(kind, cfg) -> intervention that reliably triggers a spike during a trunk run.
# TODO: kinds: "lr_bump", "bad_batch", "norm_inject" -> the menu of controlled triggers.
