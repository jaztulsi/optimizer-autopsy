"""Secret loading for the free stack (HF Hub token, wandb key).

Resolution order: process env -> Kaggle Secrets -> Colab userdata -> `.env` file. Secrets are never
printed and never committed (`.env` is gitignored; `test_no_secrets_in_git` guards the tracked tree).
Missing a required secret fails loudly with the fix, never silently.
"""

from __future__ import annotations

import os
from pathlib import Path

# Canonical secret names, also the env-var names the notebook launchers export.
HF_TOKEN = "HF_TOKEN"
WANDB_API_KEY = "WANDB_API_KEY"


def _from_env(name: str) -> str | None:
    return os.environ.get(name) or None


def _from_kaggle(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient  # only present on Kaggle
    except Exception:
        return None
    try:
        return UserSecretsClient().get_secret(name) or None
    except Exception:
        return None  # secret not attached to this kernel


def _from_colab(name: str) -> str | None:
    try:
        from google.colab import userdata  # only present on Colab
    except Exception:
        return None
    try:
        return userdata.get(name) or None
    except Exception:
        return None  # not set / access not granted


def _from_dotenv(name: str) -> str | None:
    # Minimal .env parse (KEY=VALUE, ignore comments/blanks). Walk up from cwd to find the file.
    for base in (Path.cwd(), *Path.cwd().parents):
        env = base / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip().strip("'\"") or None
            break  # first .env found wins
    return None


_RESOLVERS = (_from_env, _from_kaggle, _from_colab, _from_dotenv)


def get_secret(name: str, required: bool = True) -> str | None:
    """Return the secret `name` from the first source that has it. Never logs the value.

    Raises RuntimeError with an actionable message if `required` and nowhere provides it.
    """
    for resolve in _RESOLVERS:
        val = resolve(name)
        if val:
            os.environ.setdefault(name, val)  # cache for libs that read env directly (hf, wandb)
            return val
    if required:
        raise RuntimeError(
            f"secret {name!r} not found. Set it as an env var, a Kaggle Secret, a Colab userdata "
            f"entry, or a line in .env (all named {name!r}). See research/README.md#secrets."
        )
    return None


def hf_token(required: bool = True) -> str | None:
    return get_secret(HF_TOKEN, required=required)


def wandb_key(required: bool = True) -> str | None:
    return get_secret(WANDB_API_KEY, required=required)
