"""(w, m, v) snapshots: weights, optimizer first moment, second moment at a training step.

The atomic unit of the autopsy. A snapshot is everything needed to resume/fork a run
bit-identically, plus the AdamW state the localizer inspects. Stored to HF Hub (safetensors).
"""

# TODO: capture(model, optimizer, step, rng) -> dict of {w, m, v, step, rng_state, meta}.
# TODO: save(snapshot, path) / load(path) -> safetensors round-trip; bf16 for 124M, fp32 for proxy.
# TODO: push_to_hub(snapshot, repo_id) / pull_from_hub(repo_id, step) -> HF Hub transfer.
