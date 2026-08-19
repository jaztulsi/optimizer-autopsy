"""Fork runs: branch off a trunk snapshot to test interventions counterfactually.

A fork loads (w, m, v) at step T, applies an intervention (repair operator or baseline), and trains
forward. Comparing forks isolates the causal effect of the intervention.

C1 -- the determinism GATE -- lives here: two NOOP forks (identity intervention) from the same
snapshot must produce BITWISE-identical trajectories (max|Δ|=0). Until that holds on the proxy at
fp32, no Δ from any real intervention is trustworthy, so nothing causal downstream may proceed.

Ordering (load-bearing): the gate configures determinism ONCE, before any CUDA op -- `seed_everything`
asserts CUDA is not yet initialized. Both noop branches then build a FRESH model+opt and restore the
same snapshot; `restore()` rewrites (w, m, v) AND the RNG, so a differing weight-init or init-time
RNG draw between branches is fully overwritten, and with dropout=0 the forward/backward/step consume
no RNG -- so the two branches are identical by construction. Data stays aligned: both branches train
the same step range and `get_batch` is a pure function of step, so a fork never "skips a batch".
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _noop(model, optimizer) -> None:
    """The identity intervention: change nothing. The gate forks two of these against each other."""


def run_fork(
    cfg: dict,
    data_dir: str,
    snapshot,
    intervention=None,
    steps: int | None = None,
    start_step: int | None = None,
    device: str | None = None,
    on_step=None,
    grad_hook=None,
    pre_step=None,
    seed: bool = True,
) -> list[float]:
    """One fork: (optionally seed) -> build model+opt -> restore snapshot -> apply intervention ->
    train forward. Returns the per-step loss list.

    `intervention(model, optimizer)` mutates state in place ONCE at fork start (default: noop);
    `pre_step(step, ctx)` fires every step (the spike-replay / skip / reset channel, Task 9 -- see
    `trunk.train_forward`). `snapshot` is a path or an in-memory snapshot dict. `seed=True` configures
    determinism first and MUST be the process's first CUDA touch; the gate/battery seed once
    themselves and call their branches with seed=False.
    """
    import torch

    from research.harness.determinism import seed_everything
    from research.harness.snapshot import load as load_snapshot
    from research.harness.snapshot import restore
    from research.harness.trunk import build_model_opt, train_forward

    if seed:
        seed_everything(cfg.get("seed", 1337), deterministic=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    snap = load_snapshot(snapshot) if isinstance(snapshot, str) else snapshot
    model, opt = build_model_opt(cfg, device)
    restore(model, opt, snap)
    (intervention or _noop)(model, opt)

    start = snap["step"] if start_step is None else start_step
    steps = steps if steps is not None else cfg["train"]["max_steps"]
    return train_forward(
        model, opt, cfg, data_dir, start, steps, device, on_step=on_step, grad_hook=grad_hook, pre_step=pre_step
    )


def _fork_fingerprints(cfg, data_dir, snap, steps, device, intervention=None):
    """Run one fork and return a per-step weight fingerprint (all params flattened onto CPU, for
    bitwise comparison). Built fresh each call; `restore()` makes the start state a pure function of
    `snap`, so two calls with the same snapshot must fingerprint identically."""
    import torch

    from research.harness.snapshot import restore
    from research.harness.trunk import build_model_opt, train_forward

    model, opt = build_model_opt(cfg, device)
    restore(model, opt, snap)
    (intervention or _noop)(model, opt)

    fps: list = []

    def on_step(step, info):
        m = info["model"]
        fps.append(torch.cat([p.detach().reshape(-1).cpu().clone() for p in m.parameters()]))

    train_forward(model, opt, cfg, data_dir, snap["step"], steps, device, on_step=on_step)
    return fps


def fork_determinism_gate(cfg, data_dir, snapshot, steps, device=None, seed=True):
    """THE C1 gate: two noop forks from `snapshot` must be bitwise-identical at every step.

    Returns max|Δ| over the run (0.0 on success) and asserts it is exactly 0. `seed=True` configures
    determinism first (must be the process's first CUDA op); pass seed=False when the caller already
    seeded and produced the snapshot in-process (a second deterministic seed would trip the
    'CUDA already initialized' precondition).
    """
    import torch

    from research.harness.determinism import seed_everything
    from research.harness.snapshot import load as load_snapshot

    if seed:
        seed_everything(cfg.get("seed", 1337), deterministic=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    snap = load_snapshot(snapshot) if isinstance(snapshot, str) else snapshot

    a = _fork_fingerprints(cfg, data_dir, snap, steps, device)
    b = _fork_fingerprints(cfg, data_dir, snap, steps, device)
    assert len(a) == len(b) == steps and steps > 0, f"fingerprint count mismatch: {len(a)}/{len(b)}/{steps}"

    max_d = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        d = (x - y).abs().max().item()
        max_d = max(max_d, d)
        assert d == 0.0, f"noop-vs-noop diverged at fork step {i} on {device}: max|Δ|={d} (must be 0)"
    print(f"fork determinism GATE OK on {device}: {steps} steps, noop-vs-noop max|Δ|=0")
    return max_d


def short_fork(cfg, data_dir, snapshot, intervention=None, steps=1000, **kw):
    """Convenience: a short fork (default 1000 steps) for calibration / branch-ordering signal."""
    return run_fork(cfg, data_dir, snapshot, intervention=intervention, steps=steps, **kw)


def full_fork(cfg, data_dir, snapshot, intervention=None, **kw):
    """Convenience: a full-convergence fork (runs to cfg['train']['max_steps'])."""
    return run_fork(cfg, data_dir, snapshot, intervention=intervention, steps=None, **kw)


# --------------------------------------------------------------------------------------
# Task 9 -- the cheap-fix kill-test battery (B0, B*, Bg, Bs), forked from a pre-spike (t0)
# snapshot and replaying the spike forward. See context-ai.md §5.4 (branch menu) / §9 (Gate B).
# --------------------------------------------------------------------------------------


@dataclass
class Branch:
    """One kill-test branch: a name, a per-step `pre_step` hook (None = clean / no spike), and the
    cfg it runs under (Bs runs a clip-augmented cfg; the others reuse the trunk cfg)."""

    name: str
    pre_step: object
    cfg: dict


def compose(*pre_steps):
    """Chain `pre_step(step, ctx)` hooks left-to-right; None entries are dropped, all-None -> None.
    Order is LOAD-BEARING: the spike hook goes first (it sets LR / `ctx['batch']`); a fix hook runs
    after and overrides it (skip resets `ctx['batch']` to the clean slot; reset zeroes moments)."""
    hooks = [h for h in pre_steps if h is not None]
    if not hooks:
        return None

    def pre_step(step, ctx):
        for h in hooks:
            h(step, ctx)

    return pre_step


def make_branches(cfg, recipe, *, clip_norm):
    """Build `{name: Branch}` for the Task-9 cheap-fix battery -- all forked from the SAME pre-spike
    (t0) snapshot and replaying `recipe` forward, EXCEPT `B*`, which toggles the spike off (the clean
    counterfactual every Δ subtracts). `Bg` zeroes moments once just after the spike window; `Bs` adds
    skip (data-side) + clip (cfg grad-norm). `recipe` is a `spikes.induce.SpikeRecipe`."""
    from research.baselines import clip, reset, skip

    spike = recipe.pre_step
    inj, w = recipe.inject_step, recipe.width
    return {
        "B0": Branch("B0", spike, cfg),  # damaged do-nothing (spike on, no fix)
        "B*": Branch("B*", None, cfg),  # clean counterfactual (spike off)
        "Bg": Branch("Bg", compose(spike, reset.as_pre_step(inj + w)), cfg),  # blunt global reset
        "Bs": Branch("Bs", compose(spike, skip.as_pre_step(inj, w)), clip.apply_to_cfg(cfg, clip_norm)),
    }


def _summarize(name: str, losses: list) -> dict:
    """NaN-safe branch summary: `survival=0` if any loss is non-finite (or the list is empty), else
    `survival=1` and `final` = the last loss. The battery never lets a NaN branch abort the others."""
    survival = int(bool(losses) and all(math.isfinite(x) for x in losses))
    return {"name": name, "losses": losses, "final": (losses[-1] if survival else float("nan")), "survival": survival}


def run_branch(data_dir, snapshot, branch: Branch, steps, device=None, seed=False, on_step=None) -> dict:
    """Run one `Branch` from `snapshot` (restore -> its cfg + pre_step -> train forward) and return a
    NaN-safe result dict. `seed=False`: the battery seeds ONCE up front (a 2nd deterministic seed
    would trip the 'CUDA already initialized' precondition)."""
    losses = run_fork(
        branch.cfg, data_dir, snapshot, steps=steps, device=device, seed=seed, pre_step=branch.pre_step, on_step=on_step
    )
    return _summarize(branch.name, losses)


def run_cheap_battery(data_dir, snapshot, branches: dict, steps, device=None, seed=True) -> dict:
    """Run the whole cheap-fix battery NaN-safely: seed once, then each branch with `seed=False`; a
    branch that NaNs or raises records `survival=0` and the battery CONTINUES. Returns `{name: result}`.
    All branches restore the SAME snapshot, so two `B0`s replay bitwise (the Δ==0 determinism gate)."""
    import torch

    from research.harness.determinism import seed_everything
    from research.harness.snapshot import load as load_snapshot

    any_cfg = next(iter(branches.values())).cfg
    if seed:
        seed_everything(any_cfg.get("seed", 1337), deterministic=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    snap = load_snapshot(snapshot) if isinstance(snapshot, str) else snapshot

    results = {}
    for name, br in branches.items():
        try:
            results[name] = run_branch(data_dir, snap, br, steps, device=device, seed=False)
        except Exception as e:  # a NaN kernel, transient device error, etc. -- keep the battery going
            results[name] = {"name": name, "losses": [], "final": float("nan"), "survival": 0, "error": repr(e)}
    return results


def fork_matrix(*args, **kwargs):
    """The N-branch x methods x seeds sweep. Deferred: needs the baseline/repair intervention set,
    which is unbuilt (Tasks 13-15). run_fork + fork_determinism_gate are the Task 7 deliverables."""
    raise NotImplementedError("fork_matrix lands with the intervention set (Task 13+)")


def _selfcheck() -> None:
    """End-to-end on synthetic data (GPU on Kaggle; no local compute per project rules):
    seed once -> short deterministic trunk -> capture -> save/load round-trip -> noop gate max|Δ|=0.

    `seed_everything` runs ONCE at the top, so producing the snapshot in-process does not trip the
    'CUDA already initialized' precondition that a second deterministic seed would.
    """
    import os
    import tempfile

    import numpy as np
    import torch

    from research.harness.determinism import seed_everything
    from research.harness.snapshot import capture, load, save
    from research.harness.trunk import build_model_opt, train_forward

    seed_everything(0, deterministic=True)  # ONCE, before any CUDA op
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "seed": 0,
        "model": {
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 32,
            "vocab_size": 32,
            "dropout": 0.0,
            "bias": False,
        },
        "optim": {"lr": 3e-3, "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0},
        "train": {"batch_size": 16, "grad_accum": 1, "max_steps": 40},
    }

    with tempfile.TemporaryDirectory() as d:
        np.tile(np.arange(32, dtype=np.uint16), 4000).tofile(f"{d}/train.bin")  # learnable pattern

        # Produce a real (w, m, v) snapshot at step 10.
        model, opt = build_model_opt(cfg, device)
        train_forward(model, opt, cfg, d, 0, 10, device)
        snap = capture(model, opt, step=10, meta={"scale": "fork-selfcheck"})
        path = os.path.join(d, "latest.safetensors")
        save(snap, path)
        snap = load(path)  # exercise the disk round-trip too

        # The gate: determinism already configured above, so seed=False.
        max_d = fork_determinism_gate(cfg, d, snap, steps=15, device=device, seed=False)
        assert max_d == 0.0
    print(f"fork selfcheck OK on {device}: trunk->capture->save/load->noop gate, max|Δ|=0")


if __name__ == "__main__":
    _selfcheck()
