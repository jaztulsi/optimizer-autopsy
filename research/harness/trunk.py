"""Trunk run: the main training trajectory we snapshot and later fork from.

An AdamW loop over the fixed memmap (data.get_batch — a pure function of step). Two hooks make it the
substrate for the rest of the project:
  * `grad_hook(step, micro, model)` fires after EACH microbatch's backward, with `p.grad` holding
    THAT microbatch's gradient alone (no extra backprop) — this is what the localizer's SNR consumes.
  * `on_step(step, info)` fires after each optimizer step — where fork/snapshot/detector logic hangs.

The trunk may run with determinism OFF for speed (default); forks flip it ON. Dropout is 0 on the
deterministic path (proxy/124M configs set it), so RNG divergence has one fewer source.
"""

from __future__ import annotations

import argparse

import torch

from research.data.prepare import get_batch, resolve_data_dir
from research.harness.determinism import seed_everything
from research.model.nanogpt import GPT, GPTConfig


def _grad_norm(params, clip: float) -> float:
    """Total grad L2 norm; also clips in place when clip>0 (0/None -> measure only)."""
    max_norm = clip if clip and clip > 0 else float("inf")
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


def build_model_opt(cfg: dict, device: str):
    """Construct the GPT + its AdamW exactly as the trunk does, so resumes/forks share identical
    model and optimizer semantics (a mismatch here would silently break bitwise replay)."""
    m = dict(cfg["model"])
    if not m.get("vocab_size"):
        m["vocab_size"] = 50304  # GPT-2 vocab padded to a multiple of 64
    gptcfg = GPTConfig(**{k: m[k] for k in GPTConfig.__dataclass_fields__ if k in m})
    model = GPT(gptcfg).to(device)
    o = cfg["optim"]
    opt = model.configure_optimizers(
        weight_decay=o["weight_decay"], lr=o["lr"], betas=o["betas"], eps=o.get("eps", 1e-8)
    )
    return model, opt


def train_forward(
    model,
    opt,
    cfg: dict,
    data_dir: str,
    start_step: int,
    steps: int,
    device: str,
    on_step=None,
    grad_hook=None,
    run=None,
) -> list[float]:
    """Run `steps` optimizer steps from `start_step` on an ALREADY-BUILT model+opt; return per-step
    losses. The single loop the trunk, resume, and forks all share -- one implementation so a fork's
    numerics are identical to the trunk's. With dropout=0 (the deterministic fork path) it consumes
    no RNG, which is what lets two noop forks replay bitwise. See the module docstring for the hooks.
    """
    block = model.cfg.block_size
    t = cfg["train"]
    batch = t["batch_size"]
    grad_accum = t.get("grad_accum", 1)
    grad_clip = cfg["optim"].get("grad_clip", 0.0)

    losses: list[float] = []
    for step in range(start_step, start_step + steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        acc = None  # manual grad accumulator, only when grad_accum > 1
        for micro in range(grad_accum):
            # Each microbatch is a distinct, reproducible data slot (pure function of gstep).
            gstep = step * grad_accum + micro
            x, y = get_batch("train", gstep, batch, block, data_dir, device)
            if grad_accum > 1:
                for p in model.parameters():
                    p.grad = None  # so p.grad holds THIS microbatch's grad, clean, for the hook
            _, loss = model(x, y)
            loss.backward()
            if grad_hook is not None:
                grad_hook(step, micro, model)  # p.grad = this microbatch only, no extra backprop
            total_loss += loss.item()
            if grad_accum > 1:
                if acc is None:
                    acc = [torch.zeros_like(p) for p in model.parameters()]
                for a, p in zip(acc, model.parameters()):
                    if p.grad is not None:
                        a.add_(p.grad)
        if grad_accum > 1:  # ponytail: one extra grad copy only when accumulating; proxy uses accum=1
            for a, p in zip(acc, model.parameters()):
                p.grad = a.div_(grad_accum)

        grad_norm = _grad_norm(model.parameters(), grad_clip)
        opt.step()

        step_loss = total_loss / grad_accum
        lr = opt.param_groups[0]["lr"]
        losses.append(step_loss)
        if run is not None:
            run.log({"loss": step_loss, "grad_norm": grad_norm, "lr": lr}, step=step)
        if on_step is not None:
            on_step(step, {"loss": step_loss, "grad_norm": grad_norm, "lr": lr, "model": model, "opt": opt})
    return losses


def run_trunk(
    cfg: dict,
    data_dir: str,
    steps: int | None = None,
    start_step: int = 0,
    on_step=None,
    grad_hook=None,
    use_wandb: bool = False,
    deterministic: bool = False,
    device: str | None = None,
) -> list[float]:
    """Train from scratch and return the per-step loss list. See module docstring for the hooks."""
    seed_everything(cfg.get("seed", 1337), deterministic=deterministic)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, opt = build_model_opt(cfg, device)
    steps = steps if steps is not None else cfg["train"]["max_steps"]

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(project="optimizer-autopsy", config=cfg, job_type="trunk")

    losses = train_forward(
        model, opt, cfg, data_dir, start_step, steps, device, on_step=on_step, grad_hook=grad_hook, run=run
    )
    if run is not None:
        run.finish()
    return losses


def resume_trunk(
    cfg: dict,
    data_dir: str,
    from_snapshot,
    steps: int | None = None,
    start_step: int | None = None,
    on_step=None,
    grad_hook=None,
    deterministic: bool = True,
    device: str | None = None,
) -> list[float]:
    """Continue a trunk from a snapshot: seed, build model+opt, restore (w, m, v, RNG), train forward.

    `from_snapshot` is a path to a saved snapshot or an in-memory snapshot dict. `deterministic`
    defaults True (resumes/forks need bitwise replay); because `seed_everything` asserts CUDA is not
    yet initialized, call this before any other CUDA op in the process. Data stays aligned: training
    continues from the snapshot's step, so `get_batch` reads the same stream the trunk would have next
    -- never a "skip to the next batch".
    """
    from research.harness.snapshot import load as load_snapshot
    from research.harness.snapshot import restore

    seed_everything(cfg.get("seed", 1337), deterministic=deterministic)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, opt = build_model_opt(cfg, device)
    snap = load_snapshot(from_snapshot) if isinstance(from_snapshot, str) else from_snapshot
    restore(model, opt, snap)
    start = snap["step"] if start_step is None else start_step
    steps = steps if steps is not None else cfg["train"]["max_steps"]
    return train_forward(model, opt, cfg, data_dir, start, steps, device, on_step=on_step, grad_hook=grad_hook)


def _selfcheck() -> None:
    """Train the loop on a tiny synthetic shard and assert loss drops + the grad hook fires clean.

    Synthetic data is a short repeating pattern a tiny GPT memorizes fast, so a real training loop
    MUST reduce loss in a couple hundred CPU steps. Proves the loop trains and both hooks work,
    without needing the downloaded proxy shard.
    """
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        vocab = 32
        pattern = np.tile(np.arange(vocab, dtype=np.uint16), 4000)  # learnable, low-entropy
        pattern.tofile(f"{d}/train.bin")
        pattern[:2000].tofile(f"{d}/val.bin")
        cfg = {
            "seed": 0,
            "model": {
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "block_size": 32,
                "vocab_size": vocab,
                "dropout": 0.0,
                "bias": False,
            },
            "optim": {"lr": 3e-3, "weight_decay": 0.1, "betas": [0.9, 0.95], "grad_clip": 1.0},
            "train": {"batch_size": 16, "grad_accum": 2, "max_steps": 200},
        }
        fired = {"n": 0, "clean": True}

        def grad_hook(step, micro, model):
            fired["n"] += 1
            if not any(p.grad is not None for p in model.parameters()):
                fired["clean"] = False

        losses = run_trunk(cfg, d, steps=200, grad_hook=grad_hook, device="cpu")
        first = sum(losses[:10]) / 10
        last = sum(losses[-10:]) / 10
        assert last < first * 0.7, f"loss did not decrease: {first:.3f} -> {last:.3f}"
        assert fired["n"] == 200 * 2, f"grad_hook fired {fired['n']} times, expected 400"
        assert fired["clean"], "grad_hook saw empty grads"
        print(f"selfcheck OK: loss {first:.3f} -> {last:.3f}, grad_hook fired {fired['n']}x clean")


def main():
    ap = argparse.ArgumentParser(description="Train an OPTIMIZER AUTOPSY trunk.")
    ap.add_argument("--config", help="path to experiment config.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--data-dir", default=None, help="shard dir; else resolved from --shard")
    ap.add_argument("--shard", default="proxy", help="shard name for auto-resolution")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck or not args.config:
        _selfcheck()
        return

    from research.configs import load_config
    from research.harness.preflight import check_env

    check_env()  # entrypoint: verify env + determinism before any CUDA work
    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg.get("data", {}).get("dir") or resolve_data_dir(args.shard)
    losses = run_trunk(cfg, data_dir, steps=args.steps, use_wandb=args.wandb)
    print(f"trunk done: {len(losses)} steps, loss {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
