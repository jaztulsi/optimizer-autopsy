"""Secret hygiene: tokens load from env, and NO secret literal is committed to the tracked tree.

Runnable two ways:  `pytest research/tests/test_secrets.py`  or  `python -m research.tests.test_secrets`.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from research.harness import secrets

# Patterns for an actually-leaked secret. Kept specific to avoid firing on env-var *names*:
#  - HF tokens are `hf_` + a long alnum blob.
#  - a KEY/TOKEN/SECRET assigned a >=20-char high-entropy string literal.
_HF = re.compile(r"hf_[A-Za-z0-9]{30,}")
_ASSIGN = re.compile(
    r"(?i)(token|api[_-]?key|secret|password)\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}['\"]"
)


def _tracked_files() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True
    )
    return [root / p for p in out.stdout.splitlines() if p]


def test_no_secrets_in_git() -> None:
    """Grep every git-tracked file; assert no HF/W&B token literal is present."""
    hits = []
    for f in _tracked_files():
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary or gone
        for pat in (_HF, _ASSIGN):
            for m in pat.finditer(text):
                hits.append(f"{f}: {m.group(0)[:12]}...")
    assert not hits, "possible committed secret(s):\n  " + "\n  ".join(hits)


def test_loads_from_env() -> None:
    """get_secret resolves from the process env (first source in the chain)."""
    name = "OPTIMIZER_AUTOPSY_TEST_SECRET"
    os.environ[name] = "sentinel-value"
    try:
        assert secrets.get_secret(name) == "sentinel-value"
    finally:
        del os.environ[name]


def test_missing_required_raises() -> None:
    """A required-but-absent secret fails loudly, not silently."""
    os.environ.pop("OPTIMIZER_AUTOPSY_DEFINITELY_UNSET", None)
    try:
        secrets.get_secret("OPTIMIZER_AUTOPSY_DEFINITELY_UNSET")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for a missing required secret")


if __name__ == "__main__":
    test_no_secrets_in_git()
    test_loads_from_env()
    test_missing_required_raises()
    print("secrets tests OK")
