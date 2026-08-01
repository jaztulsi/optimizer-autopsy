"""Curvature localizer: local sharpness / Hessian-vector proxies at the snapshot.

High curvature in an implicated direction distinguishes a genuine sharp region from Adam noise.
"""

# TODO: hvp(model, batch, vec) -> Hessian-vector product via double backward.
# TODO: curvature_probe(snapshot, batch, directions) -> curvature along candidate poison directions.
# TODO: curvature_score(snapshot, batch) -> normalized per-group score for the poison combiner.
