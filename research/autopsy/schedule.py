"""LR-schedule pre_step for forks (PLAN_V8 §9 layout). The proxy rung is constant-LR, so None.

# ponytail: warmup+cosine / WSD not written -- fork.prime_trunk cannot apply a schedule to the trunk yet
# (mve.run_instance refuses scheduled cfgs). Add both together at the T2 realistic-proxy rung.
"""


def as_pre_step(cfg: dict):
    if cfg["optim"].get("schedule"):
        raise NotImplementedError("LR schedules need fork.prime_trunk pre_step passthrough first (flagged)")
    return None
