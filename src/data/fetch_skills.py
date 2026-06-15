"""Download EA Sports FC / FIFA player-rating data for the squad-skill model factor.

Two complementary sources, both cached in ./data/ :

  * ea_players_legacy.csv  (~96 MB)  - Kaggle: stefanoleone992/ea-sports-fc-24-complete-player-dataset
        `male_players.csv`, one row per player PER edition (fifa_version 15..24). Used for
        AS-OF-DATE training: for a past match we read the edition active at that date. Needs
        Kaggle credentials (see src/data/secrets.py + .env.example).

  * ea_players_current.csv (~30 MB) - GitHub: ismailoksuz/EAFC26-DataHub (no auth)
        Current FC26 ratings used to score each player in the official 26-man squads (matched by
        name) for the live 2026 squad-skill factor.

Run:  python -m src.data.fetch_skills   (add --force to refresh)
The legacy file requires Kaggle creds; the current file does not, so the no-auth
download still succeeds for users who only want live squads.
"""
from __future__ import annotations

import base64
import io
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
import zipfile

from .secrets import have_kaggle, load_env

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

LEGACY_FILE = "ea_players_legacy.csv"
CURRENT_FILE = "ea_players_current.csv"

_KAGGLE_DATASET = "stefanoleone992/ea-sports-fc-24-complete-player-dataset"
_KAGGLE_MEMBER = "male_players.csv"
_CURRENT_URL = "https://raw.githubusercontent.com/ismailoksuz/EAFC26-DataHub/main/data/players.csv"

_CTX = ssl.create_default_context()


def _write(dest: str, payload: bytes) -> int:
    with open(dest, "wb") as fh:
        fh.write(payload)
    return len(payload)


def _download(url: str, headers: dict | None = None, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026-predictor", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
        return resp.read()


def _kaggle_auth_header() -> dict:
    load_env()
    u, k = os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"]
    return {"Authorization": "Basic " + base64.b64encode(f"{u}:{k}".encode()).decode()}


def fetch_current(force: bool = False) -> str | None:
    """Download the current FC26 ratings file (no auth). Returns path or None on failure."""
    dest = os.path.join(DATA_DIR, CURRENT_FILE)
    if os.path.exists(dest) and not force:
        print(f"  [cached]  {CURRENT_FILE}  ({os.path.getsize(dest):,} bytes)")
        return dest
    t0 = time.time()
    try:
        payload = _download(_CURRENT_URL)
        n = _write(dest, payload)
        print(f"  [fetched] {CURRENT_FILE}  ({n:,} bytes in {time.time() - t0:.1f}s)")
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAILED]  {CURRENT_FILE}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def fetch_legacy(force: bool = False) -> str | None:
    """Download the historical multi-edition ratings file from Kaggle. Returns path or None.

    Skips gracefully (returns None) when Kaggle creds are absent so the rest of the
    pipeline can still run on the current-only data."""
    dest = os.path.join(DATA_DIR, LEGACY_FILE)
    if os.path.exists(dest) and not force:
        print(f"  [cached]  {LEGACY_FILE}  ({os.path.getsize(dest):,} bytes)")
        return dest
    if not have_kaggle():
        print(f"  [skip]    {LEGACY_FILE}: no Kaggle credentials "
              "(set KAGGLE_USERNAME / KAGGLE_KEY in .env) — skill factor will train on "
              "current ratings only.", file=sys.stderr)
        return None
    t0 = time.time()
    try:
        url = (f"https://www.kaggle.com/api/v1/datasets/download/"
               f"{_KAGGLE_DATASET}/{urllib.parse.quote(_KAGGLE_MEMBER)}")
        payload = _download(url, headers=_kaggle_auth_header(), timeout=600)
        # Single-file downloads usually arrive zipped; unwrap if so.
        if payload[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                payload = zf.read(zf.namelist()[0])
        n = _write(dest, payload)
        print(f"  [fetched] {LEGACY_FILE}  ({n:,} bytes in {time.time() - t0:.1f}s)")
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAILED]  {LEGACY_FILE}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def fetch_all(force: bool = False) -> dict:
    """Fetch both files; returns {'legacy': path|None, 'current': path|None}."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return {"legacy": fetch_legacy(force=force), "current": fetch_current(force=force)}


if __name__ == "__main__":
    print("Fetching EA Sports FC / FIFA skill data...")
    out = fetch_all(force="--force" in sys.argv)
    print("Done:", {k: bool(v) for k, v in out.items()})
