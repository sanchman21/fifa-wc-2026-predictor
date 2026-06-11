"""Data-driven manager / coaching factor.

We never ask the user to score a manager. Instead we measure, purely from results, how much
a team OVER- or UNDER-performs its Elo-implied expectation. Elo expectation already encodes
"how good the players are on paper", so the residual isolates the part attributable to
coaching, organisation and tournament management.

  * For TRAINING: each match gets each side's trailing-window residual (as-of-date).
  * For 2026: the residual accumulated specifically under the *current* manager's tenure
    (using appointment dates), so a strong new appointment isn't judged on the old regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIL_WINDOW = "550D"     # ~18 months trailing window for the as-of-date coaching signal


def _residual_long(pre: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, side) with the team's over/under-performance vs Elo expectation."""
    hs, as_ = pre["home_score"].to_numpy(), pre["away_score"].to_numpy()
    actual_home = np.where(hs > as_, 1.0, np.where(hs == as_, 0.5, 0.0))
    we_home = pre["pre_we_home"].to_numpy()
    home = pd.DataFrame({"orig": pre.index, "date": pre["date"].to_numpy(),
                         "team": pre["home_team"].to_numpy(),
                         "resid": actual_home - we_home, "side": "H"})
    away = pd.DataFrame({"orig": pre.index, "date": pre["date"].to_numpy(),
                         "team": pre["away_team"].to_numpy(),
                         "resid": (1.0 - actual_home) - (1.0 - we_home), "side": "A"})
    return pd.concat([home, away], ignore_index=True)


def build_coach_asof(pre: pd.DataFrame) -> pd.DataFrame:
    """Return coach_home / coach_away aligned to `pre.index` (trailing-window residual)."""
    long = _residual_long(pre).sort_values(["team", "date"])
    long = long.set_index("date")
    trailing = (long.groupby("team")["resid"]
                .rolling(TRAIL_WINDOW, closed="left").mean()
                .reset_index(level=0, drop=True))
    long["coach"] = trailing.fillna(0.0).to_numpy()
    long = long.reset_index()
    home = long[long.side == "H"].set_index("orig")["coach"]
    away = long[long.side == "A"].set_index("orig")["coach"]
    return pd.DataFrame({"coach_home": home.reindex(pre.index),
                         "coach_away": away.reindex(pre.index)}).fillna(0.0)


def current_coach(pre: pd.DataFrame, appointments: dict, ref_date: str,
                  fallback_days: int = 550) -> pd.Series:
    """Mean residual under each manager's current tenure (per team `data_name`)."""
    long = _residual_long(pre)
    ref = pd.Timestamp(ref_date)
    out = {}
    for team, sub in long.groupby("team"):
        appt = appointments.get(team)
        start = pd.Timestamp(appt) if appt else ref - pd.Timedelta(days=fallback_days)
        window = sub[(sub.date >= start) & (sub.date < ref)]
        if len(window) < 3:   # too few games under tenure -> blend with trailing fallback
            window = sub[(sub.date >= ref - pd.Timedelta(days=fallback_days)) & (sub.date < ref)]
        out[team] = float(window["resid"].mean()) if len(window) else 0.0
    return pd.Series(out, name="coach")
