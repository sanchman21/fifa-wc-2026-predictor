"""World-Football-Elo engine + data-driven calibration of the goals model.

We replay every played international (chronologically) and update Elo ratings with
match-importance weighting and a goal-difference multiplier. Replaying also lets us
record each match's *pre-match* rating gap against its actual scoreline, which we use
to calibrate the Poisson goals model empirically (no hand-tuned magic numbers).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Match-importance K factors (World Football Elo convention).
K_WORLD_CUP = 60.0
K_CONTINENTAL = 50.0
K_QUALIFIER = 40.0
K_MINOR_CUP = 30.0
K_FRIENDLY = 20.0

DEFAULT_HOME_ADV = 65.0  # Elo points added to the non-neutral home side.


def _k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "fifa world cup" in t and "qualif" not in t:
        return K_WORLD_CUP
    if any(s in t for s in ("uefa euro", "copa am", "african cup", "afc asian cup",
                            "gold cup", "confederations", "uefa nations league")) and "qualif" not in t:
        return K_CONTINENTAL
    if "qualif" in t:
        return K_QUALIFIER
    if "friendly" in t:
        return K_FRIENDLY
    return K_MINOR_CUP


def _goal_multiplier(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


@dataclass
class EloResult:
    ratings: dict          # team -> final Elo
    elo_per_goal: float    # Elo points equivalent to one goal of supremacy (calibrated)
    total_goals: float     # baseline expected goals per match (calibrated)
    n_matches: int
    pre: pd.DataFrame = field(repr=False, default=None)   # per-match pre-match ratings & expectations
    history: pd.DataFrame = field(repr=False, default=None)


def compute_elo(results: pd.DataFrame, home_adv: float = DEFAULT_HOME_ADV,
                calibration_since: str = "2006-01-01") -> EloResult:
    """Replay history; return final ratings, an empirical goals calibration, and the
    pre-match rating / win-expectation of every match (used for training & the coach factor)."""
    ratings: dict[str, float] = {}
    cal_dr, cal_gd, cal_tot = [], [], []
    cutoff = pd.Timestamp(calibration_since)

    home = results["home_team"].to_numpy()
    away = results["away_team"].to_numpy()
    hs = results["home_score"].to_numpy()
    as_ = results["away_score"].to_numpy()
    neutral = results["neutral"].to_numpy()
    dates = results["date"].to_numpy()
    ks = results["tournament"].map(_k_factor).to_numpy()

    n = len(results)
    pre_h = np.empty(n); pre_a = np.empty(n); pre_we = np.empty(n)
    for i in range(n):
        h, a = home[i], away[i]
        rh = ratings.get(h, 1500.0)
        ra = ratings.get(a, 1500.0)
        ha = 0.0 if neutral[i] else home_adv
        dr = rh + ha - ra
        we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
        pre_h[i] = rh; pre_a[i] = ra; pre_we[i] = we
        gd = hs[i] - as_[i]
        w = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        delta = ks[i] * _goal_multiplier(gd) * (w - we)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

        if dates[i] >= cutoff:
            cal_dr.append(dr)
            cal_gd.append(gd)
            cal_tot.append(hs[i] + as_[i])

    cal_dr = np.asarray(cal_dr, float)
    cal_gd = np.asarray(cal_gd, float)
    slope = float(np.cov(cal_dr, cal_gd)[0, 1] / np.var(cal_dr))
    elo_per_goal = 1.0 / slope
    total_goals = float(np.mean(cal_tot))

    pre = pd.DataFrame({
        "date": results["date"].to_numpy(),
        "home_team": home, "away_team": away,
        "pre_elo_home": pre_h, "pre_elo_away": pre_a, "pre_we_home": pre_we,
        "home_score": hs, "away_score": as_, "neutral": neutral,
    })
    hist = pd.DataFrame({"rating_gap": cal_dr, "goal_diff": cal_gd})
    return EloResult(ratings=ratings, elo_per_goal=elo_per_goal, total_goals=total_goals,
                     n_matches=n, pre=pre, history=hist)
