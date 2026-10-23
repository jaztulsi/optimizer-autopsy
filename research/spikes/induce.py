"""Induce loss spikes so we have ground-truth failures to autopsy.

Reproducible recipes (context §15.1), each pinned to a KNOWN `inject_step` so the failure has ground
truth (location + cause) to score the localizer/detector against:
  (a) lr_bump      -- transiently multiply the LR over a window (adaptive edge-of-stability push)
  (b) tiny_eps     -- shrink Adam eps to 1e-12 over a window (stresses the v-underflow pathway)
  (c) precision    -- run the windowed step(s) in reduced precision (numerical blow-up)
  (d) corrupt_batch-- feed a garbage batch over a window (cheapest natural class; cleanest B*)
  (e) repeat_ngram -- replace rows of the REAL batch with a repeated junk n-gram: the spike cause OLMo 2
                      documented in production (e.g. 'g4ODg4OD...'), a realistic upgrade on uniform garbage
  (f) v_inflate    -- planted optimizer-state poison: scale Adam's v on NAMED tensors. Not a loss spike --
                      it stalls learning there, regaining half speed after ~ln(factor/3)/(-ln beta2) steps
                      (~158 at factor 1e4, beta2 0.95; RESEARCH.md §4.1) -- PLAN V6 §3.2's over-inflated-v
                      pathway -- and gives the localizer (C2) exact ground truth: `targets`.

Each recipe is delivered as a `pre_step(step, ctx)` hook for `trunk.train_forward` -- it perturbs
ONLY inside `[inject_step, inject_step+width)` and is otherwise a no-op. This keeps the base `cfg`
and data stream untouched, so the clean counterfactual **B*** is just the SAME run with the hook
omitted (`pre_step=None`): identical data slots, spike toggled off -- exactly the alignment rule the
project requires (never "skip to the next batch").

The pure control logic (window timing, LR/eps mutation, precision flag) is CPU-only unit-testable
with a fake optimizer and no torch (see `_selfcheck`); only `corrupt_batch` touches torch, and only
inside its firing branch.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class SpikeRecipe:
    """A reproducible induced spike. `pre_step` is passed to `trunk.train_forward`; `inject_step` is
    the known cause location the detector/localizer are scored against."""

    kind: str
    cfg: dict
    inject_step: int
    width: int
    cause: str
    pre_step: Callable[[int, dict], None]
    params: dict = field(default_factory=dict)  # the recipe kwargs (v_inflate's `targets` = ground truth)


def _in_window(step: int, inject_step: int, width: int) -> bool:
    return inject_step <= step < inject_step + width


def _lr_bump(cfg, inject_step, width, factor=10.0):
    base = cfg["optim"]["lr"]

    def pre_step(step, ctx):
        lr = base * factor if _in_window(step, inject_step, width) else base
        for g in ctx["opt"].param_groups:  # reset every step -> idempotent, restores after the window
            g["lr"] = lr

    return pre_step, f"LR x{factor} over steps [{inject_step},{inject_step + width})"


def _tiny_eps(cfg, inject_step, width, eps=1e-12):
    base = cfg["optim"].get("eps", 1e-8)

    def pre_step(step, ctx):
        e = eps if _in_window(step, inject_step, width) else base
        for g in ctx["opt"].param_groups:
            g["eps"] = e

    return pre_step, f"Adam eps -> {eps} over steps [{inject_step},{inject_step + width})"


def _precision(cfg, inject_step, width, dtype="float16"):
    def pre_step(step, ctx):
        if _in_window(step, inject_step, width):
            ctx["autocast_dtype"] = dtype  # train_forward resolves the string to a torch dtype

    return pre_step, f"reduced precision ({dtype}) over steps [{inject_step},{inject_step + width})"


def _corrupt_batch(cfg, inject_step, width):
    def pre_step(step, ctx):
        if not _in_window(step, inject_step, width):
            return
        import torch  # only on a real run; keeps the module import-light and the self-check torch-free

        m = ctx["model"]
        b = ctx["cfg"]["train"]["batch_size"]
        blk, vocab = m.cfg.block_size, m.cfg.vocab_size
        dev = next(m.parameters()).device
        garbage = torch.randint(0, vocab, (b, blk), device=dev)  # uniform-random tokens -> high loss
        ctx["batch"] = (garbage, garbage)

    return pre_step, f"corrupt (random-token) batch over steps [{inject_step},{inject_step + width})"


def _repeated_rows(x, y, n, frac, vocab, gen):
    """Replace the first ceil(frac*B) rows of (x, y) with one random n-gram per row, tiled to the block,
    targets = next token. `gen` is a local CPU torch.Generator, so no global RNG is consumed."""
    import torch

    b, blk = x.shape
    k = max(1, math.ceil(frac * b))
    gram = torch.randint(0, vocab, (k, n), generator=gen)
    seq = gram.repeat(1, blk // n + 2)[:, : blk + 1].to(x.device)
    x, y = x.clone(), y.clone()
    x[:k], y[:k] = seq[:, :-1], seq[:, 1:]
    return x, y


def _repeat_ngram(cfg, inject_step, width, n=8, frac=0.25, data_dir=None, seed=0):
    if not 0 < frac <= 1 or n < 1:
        raise ValueError(f"need 0<frac<=1 and n>=1, got frac={frac}, n={n}")
    if frac < 1 and data_dir is None:
        raise ValueError("repeat_ngram with frac<1 needs data_dir for the real rows")

    def pre_step(step, ctx):
        if not _in_window(step, inject_step, width):
            return
        import torch

        from research.data.prepare import get_batch

        m, c = ctx["model"], ctx["cfg"]
        b, blk = c["train"]["batch_size"], m.cfg.block_size
        dev = next(m.parameters()).device
        if data_dir is None:
            x = y = torch.zeros(b, blk, dtype=torch.long, device=dev)  # frac == 1: every row replaced
        else:  # the step's own first data slot, so the untouched rows are the real batch
            x, y = get_batch("train", step * c["train"].get("grad_accum", 1), b, blk, data_dir, dev)
        gen = torch.Generator().manual_seed(seed * 1_000_003 + step)
        vocab = min(m.cfg.vocab_size, 50257)  # real GPT-2 ids only (50257+ is padding, never in data)
        ctx["batch"] = _repeated_rows(x, y, n, frac, vocab, gen)

    return pre_step, f"repeated {n}-gram rows (frac={frac}) over steps [{inject_step},{inject_step + width})"


def _v_inflate(cfg, inject_step, width, factor=1e4, targets=("transformer.h.0.attn.c_attn.weight",)):
    targets = tuple(targets)

    def pre_step(step, ctx):
        if not _in_window(step, inject_step, width):
            return
        named = dict(ctx["model"].named_parameters())
        missing = [t for t in targets if t not in named]
        if missing:
            raise KeyError(f"v_inflate targets not in model.named_parameters(): {missing}")
        for t in targets:
            st = ctx["opt"].state.get(named[t], {})
            if "exp_avg_sq" not in st:
                raise RuntimeError(f"v_inflate at step {step}: no Adam state for {t!r} yet (inject after step 0)")
            st["exp_avg_sq"].mul_(factor)

    return pre_step, f"Adam v x{factor:g} on {list(targets)} over steps [{inject_step},{inject_step + width})"


_BUILDERS = {
    "lr_bump": _lr_bump,
    "tiny_eps": _tiny_eps,
    "precision": _precision,
    "corrupt_batch": _corrupt_batch,
    "repeat_ngram": _repeat_ngram,
    "v_inflate": _v_inflate,
}
RECIPES = tuple(_BUILDERS)


def induce(kind: str, cfg: dict, inject_step: int, width: int = 1, **params) -> SpikeRecipe:
    """Build a `SpikeRecipe` of `kind` that perturbs `[inject_step, inject_step+width)`. Extra kwargs
    tune the recipe (factor= for lr_bump/v_inflate, eps= for tiny_eps, dtype= for precision, n=/frac=/
    data_dir= for repeat_ngram, targets= for v_inflate)."""
    if kind not in _BUILDERS:
        raise ValueError(f"unknown spike kind {kind!r}; choose from {RECIPES}")
    if inject_step < 0 or width < 1:
        raise ValueError(f"need inject_step>=0 and width>=1, got {inject_step}, {width}")
    pre_step, cause = _BUILDERS[kind](cfg, inject_step, width, **params)
    return SpikeRecipe(
        kind=kind, cfg=cfg, inject_step=inject_step, width=width, cause=cause, pre_step=pre_step, params=params
    )


def _selfcheck() -> None:
    """CPU-only, torch-free verification of the recipe control logic with a fake optimizer.

    Exercises real code paths: a broken window boundary, a missing reset, or a wrong flag fails here.
    corrupt_batch's torch branch is only checked outside its window (no torch needed there).
    """
    from types import SimpleNamespace

    base_lr, base_eps = 3.0e-4, 1e-8
    cfg = {"optim": {"lr": base_lr, "eps": base_eps}, "train": {"batch_size": 16}}

    def fake_opt():  # opt.param_groups is just a list of dicts, exactly what recipes mutate
        return SimpleNamespace(param_groups=[{"lr": base_lr, "eps": base_eps}, {"lr": base_lr, "eps": base_eps}])

    assert set(RECIPES) == {"lr_bump", "tiny_eps", "precision", "corrupt_batch", "repeat_ngram", "v_inflate"}

    # (a) lr_bump: base outside [5,7), 10x inside, restored after.
    r = induce("lr_bump", cfg, inject_step=5, width=2, factor=10.0)
    assert r.inject_step == 5 and r.width == 2 and r.kind == "lr_bump"
    for step, want in [(4, base_lr), (5, base_lr * 10), (6, base_lr * 10), (7, base_lr)]:
        opt = fake_opt()
        r.pre_step(step, {"opt": opt})
        assert all(g["lr"] == want for g in opt.param_groups), (step, want, opt.param_groups)

    # (b) tiny_eps: base outside, 1e-12 inside.
    r = induce("tiny_eps", cfg, inject_step=8, width=1)
    for step, want in [(7, base_eps), (8, 1e-12), (9, base_eps)]:
        opt = fake_opt()
        r.pre_step(step, {"opt": opt})
        assert all(g["eps"] == want for g in opt.param_groups), (step, want)

    # (c) precision: autocast flag set only inside the window.
    r = induce("precision", cfg, inject_step=3, width=2)
    for step, want in [(2, None), (3, "float16"), (4, "float16"), (5, None)]:
        ctx = {"autocast_dtype": None}
        r.pre_step(step, ctx)
        assert ctx["autocast_dtype"] == want, (step, want, ctx)

    # (d) corrupt_batch: outside its window it is a no-op (and needs no torch).
    r = induce("corrupt_batch", cfg, inject_step=6, width=1)
    assert r.cause.startswith("corrupt")
    ctx = {"batch": None}
    r.pre_step(2, ctx)  # outside window -> untouched, no torch import
    assert ctx["batch"] is None

    # (e, f) repeat_ngram / v_inflate: outside their windows they are no-ops (no torch, no model needed).
    for kind in ("repeat_ngram", "v_inflate"):
        r = induce(kind, cfg, inject_step=6, width=1, **({"frac": 1.0} if kind == "repeat_ngram" else {}))
        ctx = {"batch": None}
        r.pre_step(2, ctx)
        assert ctx["batch"] is None, kind

    assert set(RECIPES) == set().union(*[{k} for k in _BUILDERS])  # all builders dispatchable
    print("induce selfcheck OK: 6 recipes, window timing + LR/eps/precision control verified (torch-free)")


if __name__ == "__main__":
    _selfcheck()
