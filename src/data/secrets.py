"""Securely load credentials (Kaggle) from .env, the OS environment, or Streamlit secrets.

Resolution order (first hit wins), so the same code works locally, in CI, and on
Streamlit Cloud:

  1. Process environment   - KAGGLE_USERNAME / KAGGLE_KEY already exported
  2. Streamlit st.secrets  - for `streamlit run` deployments (Settings -> Secrets)
  3. Project .env file     - local dev; this file is git-ignored and never committed

Nothing here ever prints, logs, or returns a secret's value to a caller that would
display it - callers get a boolean "are creds available" via `have_kaggle()` and the
values are pushed straight into os.environ for the Kaggle client to consume.
"""
from __future__ import annotations

import os

# .env lives at the repo root (two levels up from src/data/)
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".env")

# Keys we are willing to hydrate from .env / st.secrets into the process environment.
_KEYS = ("KAGGLE_USERNAME", "KAGGLE_KEY")


def _parse_env_file(path: str) -> dict:
    """Minimal, dependency-free .env parser (KEY=VALUE per line, # comments, optional quotes)."""
    out: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    out[key] = val
    except OSError:
        pass
    return out


def _from_streamlit() -> dict:
    """Read keys from st.secrets, but ONLY when actually running inside a Streamlit
    script run - otherwise touching st.secrets emits a noisy bare-mode warning on every CLI run."""
    try:
        from streamlit.runtime import exists as _st_running  # noqa: PLC0415
        if not _st_running():
            return {}
        import streamlit as st  # noqa: PLC0415
        return {k: str(st.secrets[k]) for k in _KEYS if k in st.secrets}
    except Exception:  # noqa: BLE001 - st not installed, no secrets file, etc.
        return {}


def load_env(override: bool = False) -> None:
    """Hydrate KAGGLE_* into os.environ from st.secrets then .env, without clobbering
    anything already exported (unless override=True). Safe to call repeatedly."""
    sources = (_from_streamlit(), _parse_env_file(_ENV_PATH))
    for src in sources:
        for key in _KEYS:
            if key in src and src[key] and (override or not os.environ.get(key)):
                os.environ[key] = src[key]


def have_kaggle() -> bool:
    """True iff both Kaggle credentials are available (after attempting to load them).
    Never reveals the values."""
    load_env()
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
