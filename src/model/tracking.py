"""Track the model's own predictions and grade them as real scores come in.

The flow is deliberately honest:

1. While a WC-2026 fixture is still **unplayed**, we snapshot the model's pre-match
   prediction (win/draw/loss probabilities + expected goals + supremacy) and *lock it*
   into a persistent ledger (``output/prediction_log.json``). Once locked it is never
   overwritten, so the recorded forecast is genuinely out-of-sample.
2. When that fixture is **played** (the refreshed results feed fills in the score), we
   attach the actual result to the locked entry and grade it — outcome hit-rate, Brier
   score, log-loss and expected-goal error.
3. Those graded entries give a running success rate AND let us fit a single calibration
   scale on the supremacy term, so the live forecast can self-correct if it is running
   over- or under-confident. The scale is heavily shrunk toward 1.0 so a handful of
   matches can't whipsaw it.

A prediction made for a match that was *already* played before we first tracked it is
still recorded, but flagged ``locked_pre_match=False`` and excluded from grading — the
current power table already reflects that result, so scoring it would be circular.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from .match_model import scoreline_matrix
from .predict import MIN_LAMBDA, _supremacy

LEDGER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "output", "prediction_log.json")

_OUTCOMES = ("home", "draw", "away")
_UNIFORM_LOGLOSS = math.log(3)            # skill baseline: a coin-flip over 3 outcomes


# --------------------------------------------------------------------------- core math
def probs_from_supremacy(model, sup: float, max_goals: int = 10) -> dict:
    """Win/draw/loss probabilities + expected goals for a given goal-supremacy.

    Identical Poisson/Dixon-Coles math to :func:`predict.predict_match`, but driven by a
    raw supremacy number so the calibration search can re-price a locked prediction under
    a different supremacy scale without re-reading the power table.
    """
    T = model.total_goals
    la = max(MIN_LAMBDA, (T + sup) / 2.0)
    lb = max(MIN_LAMBDA, (T - sup) / 2.0)
    m = scoreline_matrix(la, lb, model.rho, max_goals)
    return {
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "exp_home": la,
        "exp_away": lb,
    }


def outcome(home_score, away_score) -> str | None:
    if home_score is None or away_score is None or pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return "draw"


def _match_key(date, home: str, away: str) -> str:
    d = pd.Timestamp(date).date().isoformat()
    return f"{d}|{home}|{away}"


# --------------------------------------------------------------------------- grading
def grade_entry(e: dict) -> dict | None:
    """Per-match scores for a resolved, genuinely-locked prediction (else ``None``)."""
    if not e.get("resolved") or not e.get("locked_pre_match", True):
        return None
    act = e["actual_outcome"]
    p = {o: e[f"p_{o}"] for o in _OUTCOMES}
    pred = max(p, key=p.get)
    eps = 1e-12
    brier = sum((p[o] - (1.0 if o == act else 0.0)) ** 2 for o in _OUTCOMES)
    logloss = -math.log(max(eps, p[act]))
    return {
        "predicted": pred,
        "actual": act,
        "correct": int(pred == act),
        "p_actual": p[act],
        "brier": brier,
        "logloss": logloss,
        "abs_goal_err": abs((e["exp_home"] - e["exp_away"])
                            - (e["actual_home_score"] - e["actual_away_score"])),
    }


# --------------------------------------------------------------------------- ledger I/O
def load_ledger(path: str = LEDGER_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_ledger(ledger: dict, path: str = LEDGER_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=True)


def update_ledger(power: pd.DataFrame, model, fixtures: pd.DataFrame,
                  run_date: str, path: str = LEDGER_PATH) -> dict:
    """Lock new pre-match predictions, attach results to matches that have since played.

    Returns the up-to-date ledger (also written to ``path``).
    """
    ledger = load_ledger(path)
    for r in fixtures.itertuples():
        if r.home not in power.index or r.away not in power.index:
            continue
        key = _match_key(r.date, r.home, r.away)
        played = bool(r.played)

        if key not in ledger:
            sup = _supremacy(power, model, r.home, r.away)
            pr = probs_from_supremacy(model, sup)
            ledger[key] = {
                "date": pd.Timestamp(r.date).date().isoformat(),
                "home": r.home, "away": r.away,
                "group": (None if pd.isna(r.group) else r.group), "stage": r.stage,
                "predicted_at": run_date,
                "locked_pre_match": not played,   # contaminated if the match already happened
                "supremacy": sup,
                "p_home": pr["p_home"], "p_draw": pr["p_draw"], "p_away": pr["p_away"],
                "exp_home": pr["exp_home"], "exp_away": pr["exp_away"],
                "resolved": False,
            }

        e = ledger[key]
        if played and not e.get("resolved"):
            hs, as_ = int(r.home_score), int(r.away_score)
            e.update(actual_home_score=hs, actual_away_score=as_,
                     actual_outcome=outcome(hs, as_), resolved=True)
    save_ledger(ledger, path)
    return ledger


# --------------------------------------------------------------------------- aggregation
def graded_frame(ledger: dict) -> pd.DataFrame:
    """One row per graded match with its scores (empty frame if none yet)."""
    rows = []
    for e in ledger.values():
        g = grade_entry(e)
        if g is None:
            continue
        rows.append({
            "date": e["date"], "match": f"{e['home']} v {e['away']}",
            "stage": e["stage"],
            "pick": e[f"p_{g['predicted']}"], "predicted": g["predicted"], "actual": g["actual"],
            "score": f"{e['actual_home_score']}-{e['actual_away_score']}",
            **{k: g[k] for k in ("correct", "p_actual", "brier", "logloss", "abs_goal_err")},
        })
    df = pd.DataFrame(rows)
    return df.sort_values("date").reset_index(drop=True) if len(df) else df


def calibration_bins(df: pd.DataFrame, edges=(0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)) -> pd.DataFrame:
    """Reliability table: for matches grouped by the model's confidence in its pick,
    how often was that pick right?"""
    if not len(df):
        return pd.DataFrame(columns=["bucket", "n", "avg_confidence", "hit_rate"])
    cut = pd.cut(df["pick"], list(edges), right=False)
    out = (df.groupby(cut, observed=True)
             .agg(n=("correct", "size"), avg_confidence=("pick", "mean"),
                  hit_rate=("correct", "mean")).reset_index())
    out["bucket"] = out["pick"].astype(str)
    return out[["bucket", "n", "avg_confidence", "hit_rate"]]


def track_record(ledger: dict) -> dict:
    """Headline success metrics over all graded matches."""
    df = graded_frame(ledger)
    n = len(df)
    n_pending = sum(1 for e in ledger.values()
                    if e.get("locked_pre_match", True) and not e.get("resolved"))
    if not n:
        return {"n_graded": 0, "n_pending": n_pending, "frame": df}
    logloss = float(df["logloss"].mean())
    return {
        "n_graded": n,
        "n_pending": n_pending,
        "accuracy": float(df["correct"].mean()),
        "brier": float(df["brier"].mean()),
        "logloss": logloss,
        "logloss_vs_uniform": _UNIFORM_LOGLOSS - logloss,   # >0 ⇒ better than a coin-flip
        "skill_pct": 100.0 * (1 - logloss / _UNIFORM_LOGLOSS),
        "mean_goal_err": float(df["abs_goal_err"].mean()),
        "calibration": calibration_bins(df),
        "frame": df,
    }


# --------------------------------------------------------------------------- calibration
def calibration_scale(ledger: dict, model, prior_strength: float = 12.0,
                      lo: float = 0.6, hi: float = 1.4, steps: int = 81) -> dict:
    """Fit a single multiplier on supremacy that minimises log-loss on graded matches.

    Heavily shrunk toward 1.0: the objective adds ``prior_strength * (s-1)**2`` so it
    behaves like ``prior_strength`` pseudo-matches voting for "no change". With only a few
    real results the scale barely moves; it earns its keep once dozens of matches resolve.
    ``scale < 1`` ⇒ the model was over-confident (favourites too strong), ``> 1`` ⇒ under.
    """
    graded = [e for e in ledger.values()
              if e.get("resolved") and e.get("locked_pre_match", True)]
    grid = np.linspace(lo, hi, steps)
    if not graded:
        return {"scale": 1.0, "n": 0, "applied": False,
                "baseline_logloss": None, "calibrated_logloss": None}

    def mean_logloss(s):
        tot = 0.0
        for e in graded:
            p = probs_from_supremacy(model, s * e["supremacy"])
            tot += -math.log(max(1e-12, p[f"p_{e['actual_outcome']}"]))
        return tot / len(graded)

    n = len(graded)
    losses = np.array([mean_logloss(s) for s in grid])
    penalty = prior_strength * (grid - 1.0) ** 2 / n
    best = grid[int(np.argmin(losses + penalty))]
    return {
        "scale": float(best),
        "n": n,
        "applied": n >= 12 and abs(best - 1.0) > 1e-6,   # need real signal before nudging
        "baseline_logloss": float(mean_logloss(1.0)),
        "calibrated_logloss": float(mean_logloss(best)),
    }
