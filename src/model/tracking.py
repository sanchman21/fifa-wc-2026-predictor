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

A group prediction made for a match that was *already* played before we first tracked it is
still recorded, but flagged ``locked_pre_match=False`` and excluded from grading — the
current power table already reflects that result, so scoring it would be circular.

Knockout ties never get a pre-match lock at all: they only enter the feed once finished. So
their pre-kickoff forecast is *reconstructed* by rolling the two sides' Elo back to their
pre-match value (every other factor is frozen at the tournament-start snapshot) — exactly what
the model would have predicted before kickoff — and graded out-of-sample. See
:func:`_played_knockout_pre_probs`.
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd

from .match_model import scoreline_matrix, scoreline_prob
from .predict import MIN_LAMBDA, _supremacy

# Where the locked-prediction ledger lives. Defaults to the in-repo ``output/`` dir, but can be
# pointed elsewhere via ``WC_LEDGER_PATH`` (useful on hosts that mount the repo read-only).
LEDGER_PATH = os.environ.get(
    "WC_LEDGER_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "output", "prediction_log.json"),
)
# Fallback used only when the primary path is not writable (e.g. Streamlit Community Cloud serves
# the cloned repo from a read-only mount, so writing into ``output/`` raises OSError).
_FALLBACK_PATH = os.path.join(tempfile.gettempdir(), "wc2026_prediction_log.json")

_OUTCOMES = ("home", "draw", "away")
_UNIFORM_LOGLOSS = math.log(3)            # skill baseline: a coin-flip over 3 outcomes


# --------------------------------------------------------------------------- core math
def probs_from_supremacy(model, sup: float, total: float | None = None,
                         max_goals: int = 10) -> dict:
    """Win/draw/loss probabilities + expected goals for a given goal-supremacy.

    Identical Poisson/Dixon-Coles math to :func:`predict.predict_match`, but driven by a
    raw supremacy number so the calibration search can re-price a locked prediction under
    a different supremacy scale (and, via ``total``, a different goal total) without
    re-reading the power table. ``total`` defaults to the model's learned goal total.
    """
    T = model.total_goals if total is None else total
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
    total_pred = e["exp_home"] + e["exp_away"]
    total_actual = e["actual_home_score"] + e["actual_away_score"]
    return {
        "predicted": pred,
        "actual": act,
        "correct": int(pred == act),
        "p_actual": p[act],
        "p_draw": e["p_draw"],
        "brier": brier,
        "logloss": logloss,
        # margin error (goal difference) vs total-goals error (volume) — tracked separately
        "abs_goal_err": abs((e["exp_home"] - e["exp_away"])
                            - (e["actual_home_score"] - e["actual_away_score"])),
        "total_pred": total_pred,
        "total_actual": total_actual,
        "total_err": total_actual - total_pred,   # +ve ⇒ the model under-predicted goals
    }


# --------------------------------------------------------------------------- ledger I/O
def load_ledger(path: str = LEDGER_PATH) -> dict:
    """Read the ledger, preferring ``path`` but falling back to the temp-dir copy a previous
    run may have written when ``path`` was read-only."""
    for p in (path, _FALLBACK_PATH):
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                continue
    return {}


def save_ledger(ledger: dict, path: str = LEDGER_PATH) -> None:
    """Persist the ledger. If ``path`` is not writable (read-only repo mount on some hosts),
    transparently fall back to a temp-dir copy; if even that fails, give up quietly — the
    in-memory ledger still drives this run's metrics, only cross-run persistence is lost."""
    for p in (path, _FALLBACK_PATH):
        if not p:
            continue
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(ledger, fh, indent=2, sort_keys=True)
            return
        except OSError:
            continue


def _played_knockout_pre_probs(power: pd.DataFrame, model, elo, fixtures: pd.DataFrame) -> dict:
    """Reconstruct the *pre-kickoff* forecast for every PLAYED knockout tie.

    Knockout ties never appear in the results feed until they have already finished (the feed
    only pre-loads the group schedule; a knockout row is appended by the live overlay once the
    score is in). So the only prediction otherwise available for them is contaminated — the
    power table's Elo has already absorbed the tie's own result — which is why they were written
    off ``locked_pre_match=False`` and never graded.

    But that contamination is *entirely* in the Elo factor: every other factor (squad skill,
    form, coach, attack talent, host edge) is frozen at the tournament-start snapshot, and the
    power table is built once per run, so two teams' predictions differ from their pre-match
    form only by their own Elo movement. The Elo engine already records each match's pre-match
    ratings (``elo.pre``), and the Elo term enters supremacy linearly with fixed standardisation
    (``z = (elo - mean) / sd``), so rolling the two sides' Elo back to their pre-match value
    yields *exactly* the forecast the model would have committed to before kickoff — a genuine
    out-of-sample prediction we can grade. Returns ``{(home, away): {supremacy, p_*, exp_*}}``
    keyed by display name for each played knockout tie we could reconstruct.
    """
    out: dict = {}
    pre = getattr(elo, "pre", None)
    if elo is None or pre is None or "elo" not in getattr(model, "betas", {}):
        return out
    ko = fixtures[(fixtures["stage"] == "knockout") & fixtures["played"]]
    if not len(ko):
        return out
    b = model.betas["elo"] / model.sds["elo"]   # supremacy goals per Elo point (z is linear)
    pre_lookup = {}
    for row in pre.itertuples():
        d = pd.Timestamp(row.date).date().isoformat()
        pre_lookup[(row.home_team, row.away_team, d)] = (float(row.pre_elo_home),
                                                         float(row.pre_elo_away))
    for r in ko.itertuples():
        if r.home not in power.index or r.away not in power.index:
            continue
        d = pd.Timestamp(r.date).date().isoformat()
        pre_pair = pre_lookup.get((r.home_dn, r.away_dn, d))
        if pre_pair is None:
            continue
        pre_h, pre_a = pre_pair
        cur_h = float(elo.ratings.get(r.home_dn, 1500.0))   # Elo the power table actually used
        cur_a = float(elo.ratings.get(r.away_dn, 1500.0))
        sup = _supremacy(power, model, r.home, r.away) + b * ((pre_h - pre_a) - (cur_h - cur_a))
        out[(r.home, r.away)] = {"supremacy": float(sup), **probs_from_supremacy(model, sup)}
    return out


def update_ledger(power: pd.DataFrame, model, fixtures: pd.DataFrame,
                  run_date: str, path: str = LEDGER_PATH, elo=None) -> dict:
    """Lock new pre-match predictions, attach results to matches that have since played.

    Group fixtures are present in the feed before kickoff, so their forecast is locked while
    unplayed and graded once the score lands. Knockout ties only ever enter the feed *already
    finished*, so they get no pre-match lock; instead — when ``elo`` is supplied — their
    pre-kickoff forecast is reconstructed (see :func:`_played_knockout_pre_probs`) so they grade
    out-of-sample just like the group stage. A knockout tie recorded as contaminated by an
    earlier run is re-based onto that honest forecast in place.

    Returns the up-to-date ledger (also written to ``path``).
    """
    ledger = load_ledger(path)
    ko_pre = _played_knockout_pre_probs(power, model, elo, fixtures)
    for r in fixtures.itertuples():
        if r.home not in power.index or r.away not in power.index:
            continue
        key = _match_key(r.date, r.home, r.away)
        played = bool(r.played)
        ko_pr = ko_pre.get((r.home, r.away)) if r.stage == "knockout" else None

        e = ledger.get(key)
        if e is None:
            if ko_pr is not None:                          # reconstructed pre-kickoff forecast
                sup, pr, locked, recon = ko_pr["supremacy"], ko_pr, True, True
            else:
                sup = _supremacy(power, model, r.home, r.away)
                pr, locked, recon = probs_from_supremacy(model, sup), not played, False
            ledger[key] = e = {
                "date": pd.Timestamp(r.date).date().isoformat(),
                "home": r.home, "away": r.away,
                "group": (None if pd.isna(r.group) else r.group), "stage": r.stage,
                "predicted_at": run_date,
                "locked_pre_match": locked,   # contaminated if a group match already happened
                "reconstructed": recon,       # knockout forecast rebuilt from pre-kickoff Elo
                "supremacy": sup,
                "p_home": pr["p_home"], "p_draw": pr["p_draw"], "p_away": pr["p_away"],
                "exp_home": pr["exp_home"], "exp_away": pr["exp_away"],
                "resolved": False,
            }
        elif ko_pr is not None and not e.get("reconstructed") and not e.get("locked_pre_match", True):
            # A knockout tie an earlier run recorded as contaminated (its only prediction had
            # already absorbed the result via Elo). Re-base it on the honest pre-kickoff forecast
            # so it grades out-of-sample like the rest of the track record.
            e.update(supremacy=ko_pr["supremacy"], reconstructed=True, locked_pre_match=True,
                     p_home=ko_pr["p_home"], p_draw=ko_pr["p_draw"], p_away=ko_pr["p_away"],
                     exp_home=ko_pr["exp_home"], exp_away=ko_pr["exp_away"])

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
            **{k: g[k] for k in ("correct", "p_actual", "p_draw", "brier", "logloss",
                                 "abs_goal_err", "total_pred", "total_actual", "total_err")},
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
        "mean_goal_err": float(df["abs_goal_err"].mean()),   # mean |margin| error (goal diff)
        # goal-volume diagnostics: are real matches out-scoring the model, and is the draw rate right?
        "goals_pred": float(df["total_pred"].mean()),
        "goals_actual": float(df["total_actual"].mean()),
        "goals_bias": float(df["total_err"].mean()),         # +ve ⇒ model under-predicts goals
        "draw_rate_pred": float(df["p_draw"].mean()),        # model's mean P(draw)
        "draw_rate_actual": float((df["actual"] == "draw").mean()),
        "calibration": calibration_bins(df),
        "frame": df,
    }


# --------------------------------------------------------------------------- calibration
def calibration_scale(ledger: dict, model, prior_strength: float = 12.0,
                      s_lo: float = 0.6, s_hi: float = 1.4,
                      g_lo: float = 0.7, g_hi: float = 1.5, steps: int = 41) -> dict:
    """Jointly fit a supremacy scale ``s`` and a total-goals scale ``g`` that minimise the
    Dixon-Coles *scoreline* negative log-likelihood on graded matches.

    Scoring the full scoreline (not just win/draw/loss) calibrates BOTH how lopsided the
    model is (``s``, the goal *difference*) AND how many goals it expects (``g``, the goal
    *total*). The goal-total knob also corrects the draw frequency — raising the total lowers
    the Poisson probability of an exact tie — so the two halves of this feature share one fit.

    Both scales are heavily shrunk toward 1.0: the objective adds
    ``prior_strength * ((s-1)**2 + (g-1)**2)``, behaving like ``prior_strength`` pseudo-matches
    voting for "no change", so a handful of early results can't whipsaw the live forecast.
    ``s < 1`` ⇒ favourites were too strong; ``g > 1`` ⇒ the model under-predicted goals.
    """
    graded = [e for e in ledger.values()
              if e.get("resolved") and e.get("locked_pre_match", True)]
    if not graded:
        return {"sup_scale": 1.0, "goal_scale": 1.0, "n": 0, "applied": False,
                "baseline_logloss": None, "calibrated_logloss": None,
                "baseline_scoreline_nll": None, "calibrated_scoreline_nll": None}

    n = len(graded)
    T0 = model.total_goals
    rho = model.rho

    def scoreline_nll(s, g):
        """Mean negative log-likelihood of the actual scorelines under scales (s, g)."""
        T = g * T0
        tot = 0.0
        for e in graded:
            sup = s * e["supremacy"]
            la = max(MIN_LAMBDA, (T + sup) / 2.0)
            lb = max(MIN_LAMBDA, (T - sup) / 2.0)
            p = scoreline_prob(la, lb, e["actual_home_score"], e["actual_away_score"], rho)
            tot += -math.log(max(1e-12, p))
        return tot / n

    def outcome_logloss(s, g):
        """Mean 3-way (win/draw/loss) log-loss — the interpretable headline metric."""
        tot = 0.0
        for e in graded:
            p = probs_from_supremacy(model, s * e["supremacy"], total=g * T0)
            tot += -math.log(max(1e-12, p[f"p_{e['actual_outcome']}"]))
        return tot / n

    best_s, best_g, best_obj = 1.0, 1.0, math.inf
    for s in np.linspace(s_lo, s_hi, steps):
        for g in np.linspace(g_lo, g_hi, steps):
            obj = scoreline_nll(s, g) + prior_strength * ((s - 1.0) ** 2 + (g - 1.0) ** 2) / n
            if obj < best_obj:
                best_obj, best_s, best_g = obj, float(s), float(g)

    moved = abs(best_s - 1.0) > 1e-6 or abs(best_g - 1.0) > 1e-6
    return {
        "sup_scale": best_s,
        "goal_scale": best_g,
        "n": n,
        "applied": n >= 12 and moved,   # need real signal before nudging the live forecast
        "baseline_logloss": float(outcome_logloss(1.0, 1.0)),
        "calibrated_logloss": float(outcome_logloss(best_s, best_g)),
        "baseline_scoreline_nll": float(scoreline_nll(1.0, 1.0)),
        "calibrated_scoreline_nll": float(scoreline_nll(best_s, best_g)),
    }
