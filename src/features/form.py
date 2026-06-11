"""Recent-form factor: trailing time-windowed share of points.

As-of-date (for training) and current (for 2026) versions share one definition, so the
feature the model learns on is exactly the feature it predicts with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORM_WINDOW = "730D"   # ~2 years


def _points_long(pre: pd.DataFrame) -> pd.DataFrame:
    hs, as_ = pre["home_score"].to_numpy(), pre["away_score"].to_numpy()
    pts_home = np.where(hs > as_, 1.0, np.where(hs == as_, 0.5, 0.0))
    home = pd.DataFrame({"orig": pre.index, "date": pre["date"].to_numpy(),
                         "team": pre["home_team"].to_numpy(), "pts": pts_home, "side": "H"})
    away = pd.DataFrame({"orig": pre.index, "date": pre["date"].to_numpy(),
                         "team": pre["away_team"].to_numpy(), "pts": 1.0 - pts_home, "side": "A"})
    return pd.concat([home, away], ignore_index=True)


def asof_form(pre: pd.DataFrame, window: str = FORM_WINDOW) -> pd.DataFrame:
    """Trailing points share for home & away, aligned to pre.index (excludes current match)."""
    long = _points_long(pre).sort_values(["team", "date"]).set_index("date")
    trail = (long.groupby("team")["pts"].rolling(window, closed="left").mean()
             .reset_index(level=0, drop=True))
    long["form"] = trail.fillna(0.5).to_numpy()
    long = long.reset_index()
    home = long[long.side == "H"].set_index("orig")["form"]
    away = long[long.side == "A"].set_index("orig")["form"]
    return pd.DataFrame({"form_home": home.reindex(pre.index),
                         "form_away": away.reindex(pre.index)}).fillna(0.5)


def current_form(pre: pd.DataFrame, ref_date: str, window_days: int = 730) -> pd.Series:
    """Each team's points share over the trailing window ending at ref_date."""
    long = _points_long(pre)
    ref = pd.Timestamp(ref_date)
    lo = ref - pd.Timedelta(days=window_days)
    w = long[(long.date >= lo) & (long.date < ref)]
    out = w.groupby("team")["pts"].mean()
    return out.rename("form")
