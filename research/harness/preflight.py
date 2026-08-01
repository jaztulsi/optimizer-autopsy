"""Preflight: verify environment matches pinned expectations before a real run.

Catches the "works on Colab, silently different on Kaggle" class of bugs.
"""

# TODO: check_versions() -> assert torch/numpy/etc match requirements.txt pins; warn on drift.
# TODO: check_determinism() -> run two tiny forward/backward passes, assert bit-identical.
# TODO: check_gpu() -> report device name, VRAM, bf16 support; used to pick snapshot dtype.
# TODO: preflight() -> run all checks, raise on hard failures, return an env report dict.
