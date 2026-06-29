"""Penalty-shootout skill ratings from historical shootout results.

A knockout tie still level after extra time is decided on penalties — much closer to a
team-specific *skill* (nerve, goalkeeping, kicking preparation) than to open-play strength.
England's repeated shootout exits and Germany's near-perfect record are the canonical
examples the open-play ratings can't see.

We rate each team by its historical shootout win rate, shrunk toward 50% with a Beta prior
(teams with few or no shootouts stay near neutral), and expose it as a log-odds *penalty
rating* (0 = average). In a shootout between A and B, P(A wins) = sigmoid(rating_A - rating_B)
— a Bradley-Terry coin. This is what resolves drawn knockout ties in the simulator and the
per-match predictor, so the predicted scoreline and the predicted winner stay consistent:
a tie on the scoreboard is broken by who is better at penalties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pseudo-shootouts of regression toward 50% (a = b = PRIOR_STRENGTH/2). At 6 this is like
# adding 3 wins + 3 losses, so a 1-from-1 record barely moves and a 0/0 team sits at neutral.
PRIOR_STRENGTH = 6.0


def shootout_records(shootouts: pd.DataFrame) -> pd.DataFrame:
    """Per-team (data_name space) wins/total/win_rate over all historical shootouts."""
    cols = ["wins", "total", "win_rate"]
    if shootouts is None or not len(shootouts):
        return pd.DataFrame(columns=cols)
    played = pd.concat([shootouts["home_team"], shootouts["away_team"]])
    total = played.value_counts()
    wins = shootouts["winner"].dropna().value_counts()
    rec = pd.DataFrame({"total": total.astype(float)})
    rec["wins"] = wins.reindex(rec.index).fillna(0.0)
    rec["win_rate"] = rec["wins"] / rec["total"]
    return rec[cols]


def penalty_table(shootouts: pd.DataFrame, teams: pd.DataFrame,
                  prior: float = PRIOR_STRENGTH) -> pd.DataFrame:
    """Penalty record + log-odds rating per DISPLAY team name.

    Columns: pen_wins, pen_total, pen_win_rate (raw, NaN if none), pen_skill (shrunk log-odds;
    0 = average, +ve = strong on penalties). `teams` maps display name -> data_name to join
    the shootout history onto the model's team space.
    """
    rec = shootout_records(shootouts)
    a = prior / 2.0
    rows = []
    for disp, row in teams.iterrows():
        dn = row["data_name"]
        w = float(rec["wins"].get(dn, 0.0)) if len(rec) else 0.0
        n = float(rec["total"].get(dn, 0.0)) if len(rec) else 0.0
        rate = (w + a) / (n + 2 * a)                       # shrunk toward 0.5
        rows.append({"team": disp, "pen_wins": int(w), "pen_total": int(n),
                     "pen_win_rate": (w / n if n else np.nan),
                     "pen_skill": float(np.log(rate / (1.0 - rate)))})
    return pd.DataFrame(rows).set_index("team")


def shootout_win_prob(skill_home, skill_away):
    """P(home wins the shootout) from the two log-odds penalty ratings (scalar or array)."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(skill_home, float) - np.asarray(skill_away, float))))
