"""Download the raw data the model is built on.

Sources (all free, no API key):
  * martj42/international_results  -> every international match 1872-2026 (results.csv, shootouts.csv)
  * eloratings.net                 -> live World Football Elo (cross-check only)

Run:  python -m src.data.fetch_data
The files land in ./data/ and are cached; re-run to refresh.
"""
from __future__ import annotations

import os
import ssl
import sys
import time
import urllib.request

# eloratings.net presents a valid cert but some corporate proxies mangle it;
# we fall back to an unverified context so the cross-check download never blocks a run.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

SOURCES = {
    "results.csv": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "goalscorers.csv": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
    "shootouts.csv": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
    "elo_world.tsv": "https://www.eloratings.net/World.tsv",
}


def _download(url: str, dest: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (wc2026-predictor)"})
    with urllib.request.urlopen(req, timeout=90, context=_CTX) as resp:
        payload = resp.read()
    with open(dest, "wb") as fh:
        fh.write(payload)
    return len(payload)


def fetch_all(force: bool = False) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in SOURCES.items():
        dest = os.path.join(DATA_DIR, name)
        if os.path.exists(dest) and not force:
            print(f"  [cached]  {name}  ({os.path.getsize(dest):,} bytes)")
            continue
        t0 = time.time()
        try:
            size = _download(url, dest)
            print(f"  [fetched] {name}  ({size:,} bytes in {time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - report and continue; elo is optional
            print(f"  [FAILED]  {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if name == "results.csv":
                raise


if __name__ == "__main__":
    print("Fetching FIFA WC 2026 model data...")
    fetch_all(force="--force" in sys.argv)
    print("Done.")
