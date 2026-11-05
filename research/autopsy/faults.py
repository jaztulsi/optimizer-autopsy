"""Silent-data-corruption recipes and the parity residual (PLAN_V8 §6.2 K9/H11, E55/E56, #15).

One fp32 bit is XOR-flipped at a known step, so each fault has exact ground truth (tensor, index, bit):
  sdc_grad   a bit of one gradient element, before the global clip (a fault in backward / all-reduce).
             Under grad_accum > 1 it hits the first microbatch's gradient only.
  v_bitflip  a bit of one stored exp_avg_sq element (a memory fault between steps)
  m_bitflip  the same, in exp_avg
They are ordinary spikes.induce recipes: a pre_step that fires once at `inject_step` (width 1), so B* is the same
run without it. Bits 23-30 are the exponent. Bit 30, the top one, multiplies |x| < 1 by 2^128 (an "up" flip),
turns [1, 2) into inf/NaN and divides |x| >= 2 by 2^128 ("down"). A lower exponent bit's direction depends on the
value: on [2^-63, 2) bit 29 is set, so flipping it divides by 2^64 (a near-zeroing, not a spike). Pick the bit from
the value: g ~ 1e-3 (exponent 117) has bits 26 (x256) and 24 (x4) clear; v ~ 1e-8 has 29, 28 and 25 set (down).
Each flip prints before -> after, so a log shows which way it went.

What bit 30 does (check grad_norm / survival in the logs; CPU tests can't see the GPU case):
  sdc_grad   |g| ~ 1e-3 becomes ~3e35. Its square overflows fp32, so where the clip's norm accumulates in fp32
             (CUDA/ROCm) grad_norm = inf, the clip coefficient is 0, and the WHOLE step's gradient is zeroed: a lost
             batch, not a one-coordinate fault. With the clip off (trunk._grad_norm's measure-only path, the ZClip /
             AdaGC branches) the coefficient is inf/inf = NaN and every gradient goes NaN. A CPU norm accumulated in
             double may stay finite. For a one-coordinate fault, flip a clear lower exponent bit (above).
  m_bitflip  m ~ 3e34, so m_hat / sqrt(v_hat) can reach fp32 max: expect inf/NaN weights (E55's divergence class).
  v_bitflip  small v becomes ~1e30: the coordinate freezes silently (H11's up-flip case).

Torch is imported inside the functions, so spikes.induce stays torch-free.
"""

from __future__ import annotations

TARGET = "transformer.h.0.attn.c_attn.weight"


def flip_bit_(t, index: int, bit: int) -> tuple[float, float]:
    """XOR bit `bit` (0 = mantissa LSB, 23-30 = exponent, 31 = sign) of flat element `index` of fp32 `t`, in
    place. Returns (before, after). Flipping the same bit again restores `t` bitwise."""
    import torch

    if t.dtype != torch.float32 or not t.is_contiguous():
        raise TypeError(f"bit flips need a contiguous fp32 tensor, got {t.dtype} contiguous={t.is_contiguous()}")
    if not 0 <= bit <= 31:
        raise ValueError(f"bit must be in 0..31, got {bit}")
    flat = t.view(-1)
    before = float(flat[index])
    flat.view(torch.int32)[index] ^= -(1 << 31) if bit == 31 else 1 << bit  # bit 31 = INT32_MIN in int32
    return before, float(flat[index])


def _param(ctx, target: str):
    named = dict(ctx["model"].named_parameters())
    if target not in named:
        raise KeyError(f"fault target {target!r} is not a parameter; e.g. {list(named)[:3]}")
    return named[target]


def _check(width: int, bit: int) -> None:
    if width != 1:
        raise ValueError(f"a bit flip is a one-step fault: width must be 1, got {width}")
    if not 0 <= bit <= 31:
        raise ValueError(f"bit must be in 0..31, got {bit}")


def sdc_grad(cfg, inject_step, width, target=TARGET, index=0, bit=30):
    """Flip one bit of `target`'s gradient at `inject_step`, before the clip (a one-shot tensor hook)."""
    _check(width, bit)
    handles: list = []  # closure state: a recipe's pre_step is reused across forks (deepcopy keeps functions)

    def pre_step(step, ctx):
        if step < inject_step:
            return
        while handles:  # drop a spent hook (or one left on a previous fork's model)
            handles.pop().remove()
        if step != inject_step:
            return
        import torch

        fired: list = []

        def hook(g):
            if fired:  # grad_accum > 1, or a clean-gradient recompute in the same step: flip once
                return None
            fired.append(True)
            g = g.clone(memory_format=torch.contiguous_format)
            before, after = flip_bit_(g, index, bit)
            print(f"[sdc_grad] step {step} grad {target}[{index}] bit {bit}: {before:.3e} -> {after:.3e}", flush=True)
            return g

        handles.append(_param(ctx, target).register_hook(hook))

    return pre_step, f"bit {bit} of grad {target}[{index}] flipped before the clip at step {inject_step}"


def _state_flip(key: str, name: str):
    def build(cfg, inject_step, width, target=TARGET, index=0, bit=30):
        _check(width, bit)

        def pre_step(step, ctx):
            if step != inject_step:
                return
            st = ctx["opt"].state.get(_param(ctx, target), {})
            if key not in st:
                raise RuntimeError(f"{name} at step {step}: no Adam state for {target!r} yet (inject after step 0)")
            before, after = flip_bit_(st[key], index, bit)
            print(f"[{name}] step {step} {key} {target}[{index}] bit {bit}: {before:.3e} -> {after:.3e}", flush=True)

        return pre_step, f"bit {bit} of {key} {target}[{index}] flipped at step {inject_step}"

    return build


v_bitflip = _state_flip("exp_avg_sq", "v_bitflip")
m_bitflip = _state_flip("exp_avg", "m_bitflip")


def parity_residual(out: dict):
    """`on_step` hook: out[step] = {"m": r_m, "v": r_v}, the max over coordinates of AdamW's moment-recurrence
    residual |x_t - (b x_{t-1} + (1-b) u_t)| / (|b x_{t-1}| + |(1-b) u_t|), u = g for m and g^2 for v, against a
    copy of the previous step's m, v. Honest steps sit at fp32 rounding (~1e-7); a flip in stored m / v between
    steps reads O(1) or more (NaN counts as inf). A corrupted g (sdc_grad) moves both sides consistently, so it
    reads ~0 here by design (undo's job, not parity's). Costs a full copy of m and v on the device.
    Not for skip-step branches: a skipped update leaves the state unchanged, which reads as a fault."""
    import torch

    prev: dict = {}

    def rel(cur, a, b):
        r = (cur.double() - (a + b)).abs() / (a.abs() + b.abs()).clamp_min(1e-300)
        return float(torch.nan_to_num(r, nan=float("inf")).max())

    def on_step(step, info):
        opt, model = info["opt"], info["model"]
        b1, b2 = opt.param_groups[0]["betas"]  # ponytail: nanoGPT's two groups share betas
        rm = rv = 0.0
        checked = False  # the first step has no previous copy: nothing to check, nothing recorded
        with torch.no_grad():
            for n, p in model.named_parameters():
                st = opt.state.get(p, {})
                if "exp_avg" not in st or p.grad is None:
                    continue
                if n in prev:
                    m0, v0 = prev[n]
                    g = p.grad.double()
                    rm = max(rm, rel(st["exp_avg"], b1 * m0.double(), (1 - b1) * g))
                    rv = max(rv, rel(st["exp_avg_sq"], b2 * v0.double(), (1 - b2) * g * g))
                    checked = True
                prev[n] = (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
        if checked:
            out[step] = {"m": rm, "v": rv}

    return on_step
