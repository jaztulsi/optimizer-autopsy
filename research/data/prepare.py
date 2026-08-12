"""Tokenize a FIXED corpus into a flat uint16 memmap, and read it by integer offset.

nanoGPT-style. GPT-2 BPE (tiktoken) -> one contiguous `uint16` token stream per split, written
with `ndarray.tofile` and read back via `np.memmap`. There is no streaming loader, no shuffle
buffer, and no RNG in the read path: `get_batch(split, step, ...)` is a pure function of `step`, so
the data "cursor" IS the step and resume is exact and O(1) — the property forks need for bit-identical
replay.

Determinism of a shard comes from pinning everything upstream: a fixed HF dataset revision, fixed
document order (no shuffle), a fixed token cap, and the fixed GPT-2 tokenizer. Shards are gitignored;
they live on Kaggle `/kaggle/input` (read-only) or are pulled to `/kaggle/working` from the HF
artifacts repo by fixed revision.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

# GPT-2 BPE: vocab 50257 (< 2**16) so tokens fit in uint16; 50256 is <|endoftext|>.
DTYPE = np.uint16
EOT = 50256


@dataclass
class DataConfig:
    """Everything that fixes a shard. Change any field -> a different (pinned) shard."""

    name: str                 # shard name -> out_dir basename
    hf_dataset: str           # HF dataset id, e.g. "roneneldan/TinyStories"
    hf_revision: str          # pinned dataset commit sha (NOT a mutable branch)
    hf_split: str = "train"   # source split to draw documents from
    text_key: str = "text"    # column holding the document text
    target_tokens: int = 100_000_000  # total tokens to keep (train + val)
    val_fraction: float = 0.005       # tail fraction held out for val
    encode_batch: int = 1024          # docs per tiktoken batch call


# Presets. proxy = ~100M tokens TinyStories (prepares in <5 min via streaming). 124M = a larger
# fixed FineWeb-Edu shard. TODO: replace revisions with the exact commit shas you pin against.
PRESETS = {
    "proxy": DataConfig(
        name="proxy",
        hf_dataset="roneneldan/TinyStories",
        hf_revision="main",  # TODO: pin to a commit sha
        target_tokens=100_000_000,
    ),
    "llm124m": DataConfig(
        name="llm124m",
        hf_dataset="HuggingFaceFW/fineweb-edu",
        hf_revision="main",  # TODO: pin to a commit sha
        hf_split="train",
        target_tokens=1_000_000_000,
    ),
}


# --------------------------------------------------------------------------------------
# Prepare (writes shards)
# --------------------------------------------------------------------------------------

def prepare(cfg: DataConfig, out_root: str = "data") -> str:
    """Tokenize `cfg`'s corpus and write {out_root}/{name}/{train,val}.bin + meta.json.

    Returns the shard directory. Streams the source dataset (no full download) and stops once
    `target_tokens` are collected, so the proxy shard finishes in minutes.
    """
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    out_dir = os.path.join(out_root, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(
        cfg.hf_dataset, split=cfg.hf_split, revision=cfg.hf_revision, streaming=True
    )

    chunks: list[np.ndarray] = []
    total = 0
    batch: list[str] = []

    def flush(texts: list[str]) -> None:
        nonlocal total
        for ids in enc.encode_ordinary_batch(texts):
            ids.append(EOT)  # doc separator
            chunks.append(np.asarray(ids, dtype=DTYPE))
            total += len(ids)

    for row in ds:
        batch.append(row[cfg.text_key])
        if len(batch) >= cfg.encode_batch:
            flush(batch)
            batch = []
        if total >= cfg.target_tokens:
            break
    if batch and total < cfg.target_tokens:
        flush(batch)

    tokens = np.concatenate(chunks)[: cfg.target_tokens]
    split_at = int(len(tokens) * (1.0 - cfg.val_fraction))
    tokens[:split_at].tofile(os.path.join(out_dir, "train.bin"))
    tokens[split_at:].tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "dtype": "uint16",
        "tokenizer": "gpt2",
        "eot": EOT,
        "n_train": int(split_at),
        "n_val": int(len(tokens) - split_at),
        "config": asdict(cfg),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


# --------------------------------------------------------------------------------------
# Read (pure function of step)
# --------------------------------------------------------------------------------------

_MMAP: dict[tuple[str, str], np.memmap] = {}


def _memmap(data_dir: str, split: str) -> np.memmap:
    key = (os.path.abspath(data_dir), split)
    mm = _MMAP.get(key)
    if mm is None:
        mm = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=DTYPE, mode="r")
        _MMAP[key] = mm
    return mm


def get_batch(split, step, batch, block, data_dir, device="cpu"):
    """Return (x, y) int64 tensors for `step` — a pure function of (split, step, batch, block).

    Row i of the batch reads a contiguous `block+1` window starting at
        (step*batch*block + i*block) mod (n - block - 1)
    so consecutive steps consume disjoint spans and the whole thing wraps cleanly. No RNG, no shared
    state: two processes given the same args read the same bytes and produce bitwise-identical tensors.
    Resume is O(1) — just pass the step you left off at.
    """
    import torch

    data = _memmap(data_dir, split)
    span = len(data) - block - 1
    if span <= 0:
        raise ValueError(f"{split} shard too small: {len(data)} tokens for block {block}")

    base = (step * batch * block) % span
    starts = [(base + i * block) % span for i in range(batch)]
    x = np.stack([np.asarray(data[s : s + block], dtype=np.int64) for s in starts])
    y = np.stack([np.asarray(data[s + 1 : s + 1 + block], dtype=np.int64) for s in starts])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# --------------------------------------------------------------------------------------
# HF artifacts repo (upload / pull by fixed revision) + Kaggle path resolution
# --------------------------------------------------------------------------------------

def upload_shards(out_dir: str, repo_id: str, revision: str | None = None) -> None:
    """Push a prepared shard dir to the HF artifacts repo (dataset repo)."""
    from huggingface_hub import HfApi

    from research.harness.secrets import hf_token

    HfApi().upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=hf_token(),
    )


def pull_shards(repo_id: str, revision: str, local_dir: str) -> str:
    """Pull a shard from the HF artifacts repo at a FIXED revision. Returns the local dir."""
    from huggingface_hub import snapshot_download

    from research.harness.secrets import hf_token

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        token=hf_token(),
    )


def resolve_data_dir(name: str) -> str:
    """Find the shard `name` across the usual free-stack locations (first hit wins)."""
    for root in (f"/kaggle/input/{name}", "/kaggle/working/data", "data"):
        d = root if os.path.basename(root) == name else os.path.join(root, name)
        if os.path.exists(os.path.join(d, "train.bin")):
            return d
    raise FileNotFoundError(f"no shard '{name}' on /kaggle/input, /kaggle/working, or ./data")


# --------------------------------------------------------------------------------------
# CLI + self-check
# --------------------------------------------------------------------------------------

def _selfcheck() -> None:
    """get_batch is deterministic and correctly offset. No RNG in the read path => two processes
    with the same args are bitwise-identical, so single-process determinism is the honest check."""
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as d:
        n = 5000
        np.arange(n, dtype=DTYPE).tofile(os.path.join(d, "train.bin"))
        _MMAP.clear()
        b, t, step = 4, 8, 3
        x1, y1 = get_batch("train", step, b, t, d)
        x2, y2 = get_batch("train", step, b, t, d)
        assert torch.equal(x1, x2) and torch.equal(y1, y2), "get_batch not deterministic"
        # y is x shifted by one token.
        assert torch.equal(x1[:, 1:], y1[:, :-1]), "targets not shifted inputs"
        # Row 0 starts at the documented offset; data is arange so token value == index.
        span = n - t - 1
        assert x1[0, 0].item() == (step * b * t) % span, "wrong step offset"
        # Different step => different window.
        x3, _ = get_batch("train", step + 1, b, t, d)
        assert not torch.equal(x1, x3), "step did not advance the cursor"
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare / upload OPTIMIZER AUTOPSY data shards.")
    ap.add_argument("preset", nargs="?", choices=list(PRESETS), help="which shard to prepare")
    ap.add_argument("--out-root", default="data")
    ap.add_argument("--upload", metavar="REPO_ID", help="HF dataset repo to upload the shard to")
    ap.add_argument("--revision", help="revision for --upload")
    ap.add_argument("--selfcheck", action="store_true", help="run the get_batch determinism check")
    args = ap.parse_args()

    if args.selfcheck or not args.preset:
        _selfcheck()
    if args.preset:
        out = prepare(PRESETS[args.preset], out_root=args.out_root)
        print("wrote", out)
        if args.upload:
            upload_shards(out, args.upload, args.revision)
            print("uploaded to", args.upload)
