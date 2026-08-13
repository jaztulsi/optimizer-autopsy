"""(w, m, v) snapshots: weights, optimizer first moment, second moment at a training step.

The atomic unit of the autopsy. A snapshot is everything needed to resume/fork a run
bit-identically, plus the AdamW state the localizer inspects. Stored to HF Hub (safetensors).

Layout of the in-memory snapshot dict (all tensors fp32, CPU):
    {"w": {name: tensor}, "m": {name: tensor}, "v": {name: tensor},
     "step": int,          # trunk step index the caller passed
     "adam_step": int,     # AdamW's own step counter (drives bias correction)
     "rng_state": {...},   # from determinism.capture_rng_state()
     "meta": {...}}

Keyed by PARAMETER NAME, not index: `m`/`v` join back to weights (and to the localizer's
scores) by name, and AdamW's per-object state dict is translated through model.named_parameters().
Tied weights (nanoGPT ties wte.weight == lm_head.weight) appear once, under the wte name, because
named_parameters() dedups by identity -- so safetensors never sees the shared storage it rejects.

fp32 everywhere at proxy scale, no exceptions: Task 6/7's DoD is bit-identical resume.
# TODO(Task 22): bf16 storage path for the 124M model. Deferred -- do not add until then.
"""

from __future__ import annotations

import json
import os
import tempfile


# --------------------------------------------------------------------------------------
# Capture / restore (in-memory <-> live model+optimizer)
# --------------------------------------------------------------------------------------


def capture(model, optimizer, step: int, meta: dict | None = None) -> dict:
    """Snapshot (w, m, v) + AdamW step + RNG from a live model and its AdamW optimizer.

    Must be called after >=1 optimizer step (so m/v exist). All tensors are detached, moved to
    CPU fp32, and cloned, so the returned dict is independent of the live run.
    """
    import torch

    from research.harness.determinism import capture_rng_state

    name_by_id = {id(p): n for n, p in model.named_parameters()}
    w: dict = {}
    m: dict = {}
    v: dict = {}
    steps: dict = {}  # name -> AdamW step; must all agree (see assert below)
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = name_by_id.get(id(p))
            if name is None:  # optimizer holds a param the model doesn't name -- shouldn't happen
                raise RuntimeError("optimizer param not found in model.named_parameters()")
            st = optimizer.state.get(p, {})
            if "exp_avg" not in st:
                raise RuntimeError(f"no AdamW state for {name!r}; capture after >=1 opt.step()")
            w[name] = p.detach().to("cpu", torch.float32).clone()
            m[name] = st["exp_avg"].detach().to("cpu", torch.float32).clone()
            v[name] = st["exp_avg_sq"].detach().to("cpu", torch.float32).clone()
            s = st["step"]
            steps[name] = int(s.item() if torch.is_tensor(s) else s)

    # Every param must have stepped the same number of times. A disagreement means a frozen
    # param, per-param skipping, or inconsistent grad-accum -- a real bug, not something to
    # paper over by picking one value. Surface it loudly, naming the offenders.
    distinct = set(steps.values())
    if len(distinct) > 1:
        by_step: dict = {}
        for name, s in steps.items():
            by_step.setdefault(s, []).append(name)
        raise RuntimeError(f"AdamW step disagreement across params: {by_step}")
    adam_step = distinct.pop()

    return {
        "w": w,
        "m": m,
        "v": v,
        "step": int(step),
        "adam_step": int(adam_step),
        "rng_state": capture_rng_state(),
        "meta": dict(meta or {}),
    }


def restore(model, optimizer, snapshot: dict) -> None:
    """Write a snapshot back into a live model + AdamW optimizer, in place (the resume path).

    Weights take the model's dtype; AdamW moments stay fp32 (they always are, even for bf16
    params). RNG is restored last so the next draws replay identically.
    """
    import torch

    from research.harness.determinism import restore_rng_state

    by_name = {n: p for n, p in model.named_parameters()}
    step_t = torch.tensor(float(snapshot["adam_step"]))
    for name, p in by_name.items():
        p.data.copy_(snapshot["w"][name].to(p.device, p.dtype))
        st = optimizer.state.setdefault(p, {})
        st["exp_avg"] = snapshot["m"][name].to(p.device, torch.float32).clone()
        st["exp_avg_sq"] = snapshot["v"][name].to(p.device, torch.float32).clone()
        st["step"] = step_t.clone()
    restore_rng_state(snapshot["rng_state"])


# --------------------------------------------------------------------------------------
# Save / load (safetensors, single file -- tensors in the body, the rest in __metadata__)
# --------------------------------------------------------------------------------------


def _pyrandom_to_json(state):
    version, keys, gauss = state
    return [version, list(keys), gauss]


def _json_to_pyrandom(j):
    return (j[0], tuple(j[1]), j[2])


def _numpy_to_json(state):
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


def _json_to_numpy(j):
    import numpy as np

    return (j[0], np.array(j[1], dtype=np.uint32), j[2], j[3], j[4])


def save(snapshot: dict, path: str) -> None:
    """Write a snapshot to one safetensors file: w/m/v + RNG byte tensors in the body, the
    scalar step, meta, and python/numpy RNG state as a JSON blob in the header metadata."""
    from safetensors.torch import save_file

    tensors: dict = {}
    for kind in ("w", "m", "v"):
        for name, t in snapshot[kind].items():
            tensors[f"{kind}/{name}"] = t.contiguous()

    rng = snapshot["rng_state"]
    tensors["rng/torch"] = rng["torch"].clone()
    n_cuda = len(rng["cuda"]) if rng.get("cuda") else 0
    for i in range(n_cuda):
        tensors[f"rng/cuda/{i}"] = rng["cuda"][i].clone()

    sidecar = {
        "step": snapshot["step"],
        "adam_step": snapshot["adam_step"],
        "meta": snapshot["meta"],
        "rng_python": _pyrandom_to_json(rng["python"]),
        "rng_numpy": _numpy_to_json(rng["numpy"]),
        "n_cuda_rng": n_cuda,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(tensors, path, metadata={"sidecar": json.dumps(sidecar)})


def load(path: str) -> dict:
    """Inverse of save(). Tensors come back on CPU (RNG restore needs CPU byte tensors; restore()
    moves w/m/v onto the model's device)."""
    from safetensors import safe_open

    w: dict = {}
    m: dict = {}
    v: dict = {}
    torch_rng = None
    cuda_rng: dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        sidecar = json.loads(f.metadata()["sidecar"])
        for key in f.keys():
            t = f.get_tensor(key)
            if key.startswith("w/"):
                w[key[2:]] = t
            elif key.startswith("m/"):
                m[key[2:]] = t
            elif key.startswith("v/"):
                v[key[2:]] = t
            elif key == "rng/torch":
                torch_rng = t
            elif key.startswith("rng/cuda/"):
                cuda_rng[int(key.rsplit("/", 1)[1])] = t

    n_cuda = sidecar["n_cuda_rng"]
    rng_state = {
        "python": _json_to_pyrandom(sidecar["rng_python"]),
        "numpy": _json_to_numpy(sidecar["rng_numpy"]),
        "torch": torch_rng,
        "cuda": [cuda_rng[i] for i in range(n_cuda)] if n_cuda else None,
    }
    return {
        "w": w,
        "m": m,
        "v": v,
        "step": sidecar["step"],
        "adam_step": sidecar["adam_step"],
        "rng_state": rng_state,
        "meta": sidecar["meta"],
    }


# --------------------------------------------------------------------------------------
# HF Hub (single rotating latest.safetensors per repo -- push overwrites, pull grabs latest)
# --------------------------------------------------------------------------------------

_LATEST = "latest.safetensors"


def push_to_hub(snapshot: dict, repo_id: str) -> None:
    """Save `snapshot` and overwrite the repo's single latest.safetensors object."""
    from huggingface_hub import HfApi

    from research.harness.secrets import hf_token

    with tempfile.TemporaryDirectory() as d:
        local = os.path.join(d, _LATEST)
        save(snapshot, local)
        HfApi().upload_file(
            path_or_fileobj=local,
            path_in_repo=_LATEST,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token(),
        )


def pull_from_hub(repo_id: str, local_dir: str) -> dict:
    """Download and load the repo's latest.safetensors."""
    from huggingface_hub import hf_hub_download

    from research.harness.secrets import hf_token

    f = hf_hub_download(
        repo_id=repo_id,
        filename=_LATEST,
        repo_type="model",
        local_dir=local_dir,
        token=hf_token(),
    )
    return load(f)


# --------------------------------------------------------------------------------------
# Self-check (runs on Kaggle: no torch locally per project rules)
# --------------------------------------------------------------------------------------


def _selfcheck() -> None:
    """Round-trip a real (tied-weight) nanoGPT snapshot: capture -> save -> load must be
    bit-identical, RNG must replay, and restore() into a fresh model+opt must reproduce it."""
    import torch

    from research.harness.determinism import restore_rng_state, seed_everything
    from research.model.nanogpt import GPT, GPTConfig

    seed_everything(0, deterministic=False)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)

    def fresh():
        model = GPT(cfg)
        opt = model.configure_optimizers(weight_decay=0.1, lr=1e-3, betas=(0.9, 0.95))
        return model, opt

    model, opt = fresh()
    for _ in range(3):  # populate (m, v)
        opt.zero_grad(set_to_none=True)
        x = torch.randint(0, 64, (4, 16))
        _, loss = model(x, x)
        loss.backward()
        opt.step()

    snap = capture(model, opt, step=3, meta={"scale": "proxy-selfcheck"})
    assert "transformer.wte.weight" in snap["w"], "wte weight missing"
    assert "lm_head.weight" not in snap["w"], "tied weight double-stored"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, _LATEST)
        save(snap, path)
        got = load(path)

    for kind in ("w", "m", "v"):
        assert snap[kind].keys() == got[kind].keys(), f"{kind} names changed"
        for name in snap[kind]:
            assert torch.equal(snap[kind][name], got[kind][name]), f"{kind}/{name} not bit-identical"
    assert got["step"] == 3 and got["adam_step"] == snap["adam_step"], "step counters changed"

    restore_rng_state(snap["rng_state"])
    a = torch.rand(5)
    restore_rng_state(got["rng_state"])
    b = torch.rand(5)
    assert torch.equal(a, b), "RNG did not round-trip"

    model2, opt2 = fresh()
    restore(model2, opt2, got)
    by_name = {n: p for n, p in model2.named_parameters()}
    for name in snap["w"]:
        p = by_name[name]
        assert torch.equal(p.detach(), snap["w"][name]), f"restore w {name}"
        assert torch.equal(opt2.state[p]["exp_avg"], snap["m"][name]), f"restore m {name}"
        assert torch.equal(opt2.state[p]["exp_avg_sq"], snap["v"][name]), f"restore v {name}"
    print("snapshot selfcheck OK: capture->save->load bit-identical, RNG + restore verified")


if __name__ == "__main__":
    _selfcheck()
