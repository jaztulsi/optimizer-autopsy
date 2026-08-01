"""Deterministic training setup: seeds, cudnn/cublas flags, deterministic algorithms.

CUBLAS_WORKSPACE_CONFIG must be set BEFORE torch is imported (see notebooks/).
This module handles everything settable after import.
"""

# TODO: seed_everything(seed) -> set python/numpy/torch/cuda RNG seeds.
# TODO: enable_deterministic() -> torch.use_deterministic_algorithms(True),
#       cudnn.deterministic=True, cudnn.benchmark=False; warn if CUBLAS_WORKSPACE_CONFIG unset.
# TODO: rng_state() / load_rng_state() -> capture+restore full RNG state for exact fork replay.
