"""Shared config helpers. Experiment configs live under experiments/*/config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a yaml config into a plain nested dict, then apply flat dotted-key overrides.

    Override keys use dots for nesting, e.g. {"train.max_steps": 200}. Missing intermediate keys
    are created. Returns the merged dict.
    """
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, val in (overrides or {}).items():
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = val
    return cfg
