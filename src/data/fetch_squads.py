"""Fetch the official 2026 FIFA World Cup squads (Wikipedia) into a JSON cache.

So the squad-skill factor scores the players a nation actually selected — not the top-rated
players by EA overall (the old fallback could include stars who weren't called up). One page
request, parsed into the real 26-man lists, then matched to EA ratings in features/skills.py.

Output: data/squads_2026.json
    { "fetched_at": "...", "source": "wikipedia",
      "teams": { "<our data_name>": [ {"name": "Player", "pos": "GK|DEF|MID|FWD"}, ... ] } }

No credentials needed. Run:  python -m src.data.fetch_squads [--force]
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from io import StringIO

import pandas as pd

from .config_loader import load_teams
from ..features.availability import build_team_matcher, match_team

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SQUADS_CACHE = os.path.join(DATA_DIR, "squads_2026.json")
_PAGE = "2026 FIFA World Cup squads"
_POS_MAP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}


def _wiki_html(page: str) -> str:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "parse", "page": page, "prop": "text", "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026-predictor/1.0 (forecast)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())["parse"]["text"]["*"]


def _clean(name: str) -> str:
    """Strip '(captain)', footnote markers and surrounding whitespace from a cell."""
    name = re.sub(r"\(.*?\)", "", str(name))
    name = re.sub(r"\[.*?\]", "", name)
    return name.strip()


def fetch_squads(force: bool = False) -> dict | None:
    """Fetch + write the official-squads cache. Returns the dict, or None on failure.
    Cached: re-reads the existing file unless force=True."""
    if os.path.exists(SQUADS_CACHE) and not force:
        print(f"  [cached]  squads_2026.json ({os.path.getsize(SQUADS_CACHE):,} bytes)")
        with open(SQUADS_CACHE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415 - only needed here
        html = _wiki_html(_PAGE)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAILED] squads fetch: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    teams = load_teams()
    matcher = build_team_matcher(teams["data_name"])
    soup = BeautifulSoup(html, "html.parser")
    out_teams: dict[str, list] = {}
    cur = None
    for el in soup.find_all(["h2", "h3", "table"]):
        if el.name in ("h2", "h3"):
            cur = match_team(_clean(el.get_text()), matcher)
        elif el.name == "table" and "wikitable" in (el.get("class") or []) and cur:
            try:
                df = pd.read_html(StringIO(str(el)))[0]
            except ValueError:
                continue
            if "Player" not in df.columns or cur in out_teams:
                continue
            pos_col = next((c for c in df.columns if str(c).strip().lower().startswith("pos")), None)
            players = []
            for i in range(len(df)):
                name = _clean(df["Player"].iloc[i])
                raw_pos = str(df[pos_col].iloc[i]).strip().upper() if pos_col else ""
                if name:
                    players.append({"name": name, "pos": _POS_MAP.get(raw_pos, "")})
            out_teams[cur] = players
            cur = None

    out = {
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "wikipedia",
        "teams": out_teams,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SQUADS_CACHE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    missing = [t for t in teams["data_name"] if t not in out_teams]
    print(f"  [fetched] squads_2026.json — {len(out_teams)}/48 teams"
          + (f" | MISSING: {missing}" if missing else ""))
    return out


if __name__ == "__main__":
    fetch_squads(force="--force" in sys.argv)
