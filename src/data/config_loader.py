"""Load the editable JSON configuration (teams, weights, bracket)."""
from __future__ import annotations

import json
import os

import pandas as pd

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "config")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _load_json(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_teams() -> pd.DataFrame:
    """Return the 48-team table indexed by team name."""
    cfg = _load_json("teams_2026.json")
    df = pd.DataFrame(cfg["teams"]).set_index("team")
    assert len(df) == 48, f"expected 48 teams, got {len(df)}"
    assert df["group"].value_counts().eq(4).all(), "every group must have exactly 4 teams"
    return df


def load_weights() -> dict:
    cfg = _load_json("weights.json")
    w = cfg["factor_weights"]
    total = sum(w.values())
    assert abs(total - 1.0) < 1e-6, f"factor weights must sum to 1.0 (got {total})"
    return cfg


def load_bracket() -> dict:
    return _load_json("bracket_2026.json")


def load_managers() -> dict:
    return _load_json("managers_2026.json")


def load_results() -> pd.DataFrame:
    """Historical international results, played matches only (scored, chronological)."""
    path = os.path.join(DATA_DIR, "results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("data/results.csv missing - run `python -m src.data.fetch_data` first")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def load_shootouts() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "shootouts.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    return pd.read_csv(path, parse_dates=["date"])
