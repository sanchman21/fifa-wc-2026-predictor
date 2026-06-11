"""Parse the WC 2026 fixture list out of the results dataset and expose played/unplayed
matches. martj42's results.csv already contains every 2026 fixture (scores are NaN until
played and fill in live during the tournament), so this is the hook for live updates.
"""
from __future__ import annotations

import pandas as pd

REF_DATE = "2026-06-11"


def wc2026_matches(results_all: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """All WC-2026 matches (played or scheduled), with display names and group/stage tags.

    `results_all` must be the RAW results (incl. unplayed NaN-score rows), not the
    played-only frame used for Elo.
    """
    dn2disp = {row["data_name"]: t for t, row in teams.iterrows()}
    dn2group = {row["data_name"]: row["group"] for _, row in teams.iterrows()}

    wc = results_all[(results_all["tournament"] == "FIFA World Cup")
                     & (results_all["date"] >= pd.Timestamp("2026-01-01"))].copy()
    wc["home"] = wc["home_team"].map(dn2disp)
    wc["away"] = wc["away_team"].map(dn2disp)
    wc = wc.dropna(subset=["home", "away"])           # keep only matches between our 48
    wc["played"] = wc["home_score"].notna() & wc["away_score"].notna()
    gh = wc["home_team"].map(dn2group); ga = wc["away_team"].map(dn2group)
    wc["group"] = gh.where(gh == ga)                  # group letter if same group, else NaN
    wc["stage"] = wc["group"].notna().map({True: "group", False: "knockout"})
    wc = wc.rename(columns={"home_team": "home_dn", "away_team": "away_dn"})
    cols = ["date", "home", "away", "home_dn", "away_dn",
            "home_score", "away_score", "played", "group", "stage"]
    return wc[cols].sort_values("date").reset_index(drop=True)


def known_group_results(fixtures: pd.DataFrame) -> dict:
    """{(home, away): (home_score, away_score)} for PLAYED group matches."""
    g = fixtures[(fixtures["stage"] == "group") & fixtures["played"]]
    return {(r.home, r.away): (int(r.home_score), int(r.away_score)) for r in g.itertuples()}


def known_knockout_results(fixtures: pd.DataFrame, teams: pd.DataFrame,
                           shootouts: pd.DataFrame | None = None) -> dict:
    """{(home, away): winner} for PLAYED knockout ties (penalty shootouts resolved)."""
    dn2disp = {row["data_name"]: t for t, row in teams.iterrows()}
    k = fixtures[(fixtures["stage"] == "knockout") & fixtures["played"]]
    out = {}
    for r in k.itertuples():
        if r.home_score > r.away_score:
            out[(r.home, r.away)] = r.home
        elif r.away_score > r.home_score:
            out[(r.home, r.away)] = r.away
        elif shootouts is not None and len(shootouts):     # draw -> penalties
            m = shootouts[(shootouts["date"] == r.date)
                          & (shootouts["home_team"] == r.home_dn)
                          & (shootouts["away_team"] == r.away_dn)]
            if len(m):
                out[(r.home, r.away)] = dn2disp.get(m.iloc[0]["winner"])
    return {k: v for k, v in out.items() if v is not None}
