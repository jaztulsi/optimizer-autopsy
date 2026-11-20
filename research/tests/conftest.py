"""The suite is CPU-only (CI has no GPU), and stays on the CPU on a GPU box too. Otherwise the first test that picks
"cuda" initializes it, and every later `seed_everything(deterministic=True)` in the same process refuses to run
("CUDA is already initialized"): 6 failures on the MI300X, 2026-11-10 (results/mi300x_2026-11-10/.../tests.log)."""

import torch

# ponytail: GPU checks are the research/gpu/ scripts, not pytest; every device pick in the code goes through this
torch.cuda.is_available = lambda: False
