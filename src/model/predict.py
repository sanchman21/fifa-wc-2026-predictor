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


def predict_match(power: pd.DataFrame, model, home: str, away: str, max_goals: int = 10,
                  sup_scale: float = 1.0, goal_scale: float = 1.0) -> dict:
    """`sup_scale` rescales the goal-supremacy and `goal_scale` the expected goal total —
    1.0 is the raw model; values learned from the model's own track record (see
    model.tracking.calibration_scale) self-correct over/under-confidence and goal-volume
    bias. sup_scale <1 tempers favourites, >1 sharpens them; goal_scale >1 lifts the total
    (also lowering the draw rate)."""
    sup = _supremacy(power, model, home, away) * sup_scale
    T = model.total_goals * goal_scale
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


def _pen_skill(power: pd.DataFrame, team: str) -> float:
    """Log-odds penalty rating for a team (0 = average) if the column is present."""
    if "pen_skill" in power.columns:
        return float(power.loc[team, "pen_skill"])
    return 0.0


def predict_knockout(power: pd.DataFrame, model, home: str, away: str,
                     sup_scale: float = 1.0, goal_scale: float = 1.0) -> dict:
    """A knockout tie: the most-likely RESULT and the single winner it gives.

    The winner is the most likely of {home win, draw, away win} — an aggregate over the full
    scoreline distribution (`scorelines` carries each exact score's probability). A predicted
    draw is the only case that goes to penalties, decided by penalty SKILL: a Bradley-Terry coin
    on the log-odds shootout ratings. We deliberately do NOT collapse the distribution to a
    single "predicted score": the most-probable exact score is a draw (usually 1-1) for almost
    every match, while the most-probable result is a win, so no one score represents both. The
    caller shows the score distribution (and the most-likely exact score with its probability)
    alongside the result. (Inside the Monte-Carlo simulation every level tie still goes to pens.)
    """
    pr = predict_match(power, model, home, away, sup_scale=sup_scale, goal_scale=goal_scale)
    sh, sa = _pen_skill(power, home), _pen_skill(power, away)
    p_home_pens = 1.0 / (1.0 + np.exp(-(sh - sa)))
    ph, pd_, pa = pr["p_home"], pr["p_draw"], pr["p_away"]
    if pd_ > ph and pd_ > pa:                                   # most likely result is a draw
        result, winner, decided_on_pens = "draw", (home if p_home_pens >= 0.5 else away), True
    elif ph >= pa:                                             # most likely result is a home win
        result, winner, decided_on_pens = "home", home, False
    else:                                                       # most likely result is an away win
        result, winner, decided_on_pens = "away", away, False
    pr.update({"pen_home": sh, "pen_away": sa, "p_home_pens": float(p_home_pens),
               "result": result, "winner": winner, "decided_on_pens": bool(decided_on_pens)})
    return pr


def predict_fixtures(power: pd.DataFrame, model, fixtures: pd.DataFrame,
                     only_unplayed: bool = True, stage: str | None = None,
                     sup_scale: float = 1.0, goal_scale: float = 1.0) -> pd.DataFrame:
    """Predict every fixture (optionally a single stage / only the unplayed ones)."""
    f = fixtures
    if only_unplayed:
        f = f[~f["played"]]
    if stage:
        f = f[f["stage"] == stage]
    rows = []
    for r in f.itertuples():
        if r.stage == "knockout":
            # A knockout can't end level: the favourite is whoever ADVANCES (draws -> penalties),
            # so the most-likely scoreline and the predicted winner stay consistent.
            pr = predict_knockout(power, model, r.home, r.away, sup_scale=sup_scale, goal_scale=goal_scale)
            fav = pr["winner"]
        else:
            pr = predict_match(power, model, r.home, r.away, sup_scale=sup_scale, goal_scale=goal_scale)
            fav = max([(r.home, pr["p_home"]), ("Draw", pr["p_draw"]), (r.away, pr["p_away"])],
                      key=lambda x: x[1])[0]
        rows.append({"date": r.date.date(), "group": r.group, "stage": r.stage,
                     "home": r.home, "away": r.away,
                     "P(home win)": pr["p_home"], "P(draw)": pr["p_draw"], "P(away win)": pr["p_away"],
                     "most likely": pr["scorelines"][0][0], "favorite": fav})
    return pd.DataFrame(rows)
