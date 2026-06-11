"""Single-match predictions from the trained model: win/draw/loss, expected goals,
and most-likely scorelines. Same math the simulator uses, exposed per fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .match_model import scoreline_matrix
from .train import FACTORS

MIN_LAMBDA = 0.05


def _supremacy(power, model, home, away):
    """Expected goal supremacy of `home` over `away` (host edge auto-applied)."""
    zc = [f"z_{f}" for f in FACTORS]
    b = np.array([model.betas[f] for f in FACTORS])
    zh = power.loc[home, zc].to_numpy(float)
    za = power.loc[away, zc].to_numpy(float)
    host_h = float(power.loc[home, "host"]); host_a = float(power.loc[away, "host"])
    return float((zh - za) @ b + model.beta_home * (host_h - host_a))


def predict_match(power: pd.DataFrame, model, home: str, away: str, max_goals: int = 10) -> dict:
    sup = _supremacy(power, model, home, away)
    T = model.total_goals
    la = max(MIN_LAMBDA, (T + sup) / 2.0)
    lb = max(MIN_LAMBDA, (T - sup) / 2.0)
    m = scoreline_matrix(la, lb, model.rho, max_goals)
    p_home = float(np.tril(m, -1).sum()); p_draw = float(np.trace(m)); p_away = float(np.triu(m, 1).sum())
    # top scorelines
    flat = [((i, j), m[i, j]) for i in range(max_goals + 1) for j in range(max_goals + 1)]
    flat.sort(key=lambda x: -x[1])
    return {"home": home, "away": away, "exp_home": la, "exp_away": lb,
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "scorelines": [(f"{i}-{j}", p) for (i, j), p in flat[:6]]}


def predict_fixtures(power: pd.DataFrame, model, fixtures: pd.DataFrame,
                     only_unplayed: bool = True, stage: str | None = None) -> pd.DataFrame:
    """Predict every fixture (optionally a single stage / only the unplayed ones)."""
    f = fixtures
    if only_unplayed:
        f = f[~f["played"]]
    if stage:
        f = f[f["stage"] == stage]
    rows = []
    for r in f.itertuples():
        pr = predict_match(power, model, r.home, r.away)
        fav = max([(r.home, pr["p_home"]), ("Draw", pr["p_draw"]), (r.away, pr["p_away"])],
                  key=lambda x: x[1])
        rows.append({"date": r.date.date(), "group": r.group, "stage": r.stage,
                     "home": r.home, "away": r.away,
                     "P(home win)": pr["p_home"], "P(draw)": pr["p_draw"], "P(away win)": pr["p_away"],
                     "most likely": pr["scorelines"][0][0], "favorite": fav[0]})
    return pd.DataFrame(rows)
