"""Cross-process + co-tenancy fork gate THROUGH a spike window (PLAN_V8 §8.1 T0 / G0, Tab 1). GPU-only:
run it on the AMD MI300X, never on the laptop.

    python -m research.kaggle.xproc_gate prime   --out DIR     # trunk -> snapshot at the fork step
    python -m research.kaggle.xproc_gate fork    --out DIR --tag a
    python -m research.kaggle.xproc_gate compare --out DIR

`fork_determinism_gate` compares two forks inside ONE process. This compares forks run by SEPARATE
processes (own HIP context, allocator, kernel caches), each replaying the same spike recipes from the same
snapshot, so the gate covers clipped, large-gradient steps and not just calm ones. Launch two `fork`s at
once and a third solo: if all match bitwise, concurrent forks can be packed onto one GPU. Each process also
runs every recipe twice (in-process check) and times the warm pass (deterministic fork-mode s/step).
Synthetic data: the question is numerics, not learning.
"""

from __future__ import annotations

import os

# Must precede any torch import; a launcher that already set it wins (setdefault).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402

PRIME, STEPS, INJECT, WIDTH = 30, 40, 40, 3  # fork at 30, spike window 40..42, fork ends at 69
RECIPES = {"noop": None, "lr_bump": {"factor": 50.0}, "corrupt_batch": {}}


def _cfg(batch):
    from research.configs import load_config

    cfg = load_config("research/experiments/proxy/config.yaml")
    cfg["train"]["batch_size"] = batch
    cfg["train"]["max_steps"] = PRIME + STEPS
    return cfg


def _data(out, cfg):
    import numpy as np

    path = os.path.join(out, "train.bin")
    if not os.path.exists(path):
        n_tok = cfg["train"]["batch_size"] * cfg["model"]["block_size"] * (PRIME + STEPS + 2) * 2
        np.random.default_rng(0).integers(0, 50000, n_tok).astype(np.uint16).tofile(path)
    return out


def _state_hash(model, opt) -> str:
    import hashlib

    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        st = opt.state[p]
        for t in (p, st["exp_avg"], st["exp_avg_sq"]):
            h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prime(out, batch):
    import torch

    from research.harness.determinism import seed_everything
    from research.harness.snapshot import capture, save
    from research.harness.trunk import build_model_opt, train_forward

    os.makedirs(out, exist_ok=True)
    cfg = _cfg(batch)
    seed_everything(cfg.get("seed", 1337), deterministic=True)
    model, opt = build_model_opt(cfg, "cuda")
    train_forward(model, opt, cfg, _data(out, cfg), 0, PRIME, "cuda")
    save(capture(model, opt, step=PRIME, meta={"scale": "xproc-gate"}), os.path.join(out, "snap.safetensors"))
    print("primed:", torch.cuda.get_device_name(0), torch.__version__)


def fork(out, batch, tag):
    import time

    import torch

    from research.harness.determinism import seed_everything
    from research.harness.fork import run_fork
    from research.harness.snapshot import load
    from research.spikes.induce import induce

    cfg = _cfg(batch)
    seed_everything(cfg.get("seed", 1337), deterministic=True)  # ONCE; every fork below uses seed=False
    snap = load(os.path.join(out, "snap.safetensors"))
    rec = {"tag": tag, "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "recipes": {}}
    for kind, params in RECIPES.items():
        runs = []
        for _ in range(2):  # in-process repeat
            gn, end = [], {}

            def on_step(step, info, gn=gn, end=end):
                gn.append(float(info["grad_norm"]).hex())
                if step == PRIME + STEPS - 1:
                    end["hash"] = _state_hash(info["model"], info["opt"])

            pre = None if params is None else induce(kind, cfg, INJECT, WIDTH, **params).pre_step
            t = time.perf_counter()
            losses = run_fork(cfg, out, snap, steps=STEPS, device="cuda", on_step=on_step, pre_step=pre, seed=False)
            torch.cuda.synchronize()
            assert end.get("hash"), "on_step never saw the last step: a missing hash would compare equal"
            dt = (time.perf_counter() - t) / STEPS
            runs.append({"loss": [float(x).hex() for x in losses], "gn": gn, "hash": end.get("hash"), "s_per_step": dt})
        same = all(runs[0][k] == runs[1][k] for k in ("loss", "gn", "hash"))
        rec["recipes"][kind] = {**runs[1], "in_process_equal": same}
        print(f"[{tag}] {kind}: in-process equal={same} warm {runs[1]['s_per_step'] * 1e3:.1f} ms/step")
    with open(os.path.join(out, f"fork_{tag}.json"), "w") as f:
        json.dump(rec, f)


def compare(out):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "fork_*.json")))]
    ok = len(recs) >= 2
    summary = {"tags": [r["tag"] for r in recs], "recipes": {}}
    for kind in RECIPES:
        rs = [r["recipes"][kind] for r in recs]
        xproc = all(r[k] == rs[0][k] for r in rs for k in ("loss", "gn", "hash"))
        inproc = all(r["in_process_equal"] for r in rs)
        a = [float.fromhex(x) for x in rs[0]["loss"]]
        dmax = max(abs(float.fromhex(x) - y) for r in rs for x, y in zip(r["loss"], a, strict=True))
        summary["recipes"][kind] = {"cross_process_equal": xproc, "in_process_equal": inproc, "max_abs_dloss": dmax,
                                    "ms_per_step": [round(r["s_per_step"] * 1e3, 2) for r in rs]}
        ok = ok and xproc and inproc
    summary["PASS"] = ok
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print("XPROC GATE", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prime", "fork", "compare"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    {"prime": lambda: prime(a.out, a.batch), "fork": lambda: fork(a.out, a.batch, a.tag),
     "compare": lambda: compare(a.out)}[a.mode]()
