"""Player-performance factors from real goal-by-goal data (martj42 goalscorers.csv).

Instead of squad market *value* (which the user rightly notes can misprice cheap-but-good
players), we measure what players actually *do*: goals, weighted by the scorer's proven
track record, with recency decay. A goal by a player with 40 international goals counts far
more than a debutant's. Two factors:

  * attack_talent : recency-weighted (half-life ~3y), scorer-quality-weighted goal output.
  * wc_player     : the nation's World Cup goal pedigree (half-life ~16y), same weighting,
                    capturing players who deliver specifically on the biggest stage.

Everything is computed strictly from goals *before* a given date, so the exact same code
produces (a) as-of-date features to TRAIN the model and (b) the current 2026 values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HL_ATTACK_DAYS = 3 * 365.25
HL_WC_DAYS = 16 * 365.25
DAY = np.timedelta64(1, "D")


def load_goals(results: pd.DataFrame, goals_path: str) -> pd.DataFrame:
    """Goal events joined to results to recover the tournament; sorted chronologically."""
    g = pd.read_csv(goals_path, parse_dates=["date"])
    r = results[["date", "home_team", "away_team", "tournament"]]
    g = g.merge(r, on=["date", "home_team", "away_team"], how="left")
    g = g.dropna(subset=["scorer"])
    g["own_goal"] = g["own_goal"].fillna(False).astype(bool)
    g["is_wc"] = g["tournament"] == "FIFA World Cup"
    return g.sort_values("date").reset_index(drop=True)


class _Accumulator:
    """Exponentially time-decayed, scorer-quality-weighted goal accumulators per team."""

    def __init__(self):
        self.career: dict[str, int] = {}          # scorer -> goals so far (quality proxy)
        self.att: dict[str, float] = {}           # team -> attack accumulator
        self.att_t: dict[str, np.datetime64] = {}
        self.wc: dict[str, float] = {}            # team -> WC-goal accumulator
        self.wc_t: dict[str, np.datetime64] = {}

    @staticmethod
    def _decay(val, last_t, now, hl):
        if last_t is None:
            return 0.0
        dt = (now - last_t) / DAY
        return val * 0.5 ** (dt / hl)

    def value(self, team, now):
        a = self._decay(self.att.get(team), self.att_t.get(team), now, HL_ATTACK_DAYS)
        w = self._decay(self.wc.get(team), self.wc_t.get(team), now, HL_WC_DAYS)
        return a, w

    def add_goal(self, team, scorer, t, is_wc, own_goal):
        if own_goal:
            return
        prior = self.career.get(scorer, 0)
        quality = 1.0 + np.log1p(prior)          # proven scorers weighted higher
        self.career[scorer] = prior + 1
        # attack accumulator
        self.att[team] = self._decay(self.att.get(team), self.att_t.get(team), t, HL_ATTACK_DAYS) + quality
        self.att_t[team] = t
        if is_wc:
            self.wc[team] = self._decay(self.wc.get(team), self.wc_t.get(team), t, HL_WC_DAYS) + quality
            self.wc_t[team] = t


def asof_player_features(matches: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    """For each match (chronological), the home & away pre-match attack/WC factors."""
    acc = _Accumulator()
    g_dates = goals["date"].to_numpy()
    g_team = goals["team"].to_numpy(); g_scorer = goals["scorer"].to_numpy()
    g_wc = goals["is_wc"].to_numpy(); g_og = goals["own_goal"].to_numpy()
    gi, ng = 0, len(goals)

    m = matches.sort_values("date").reset_index()
    md = m["date"].to_numpy()
    out = np.zeros((len(m), 4))
    for i in range(len(m)):
        now = md[i]
        while gi < ng and g_dates[gi] < now:           # apply all goals strictly before kickoff
            acc.add_goal(g_team[gi], g_scorer[gi], g_dates[gi], g_wc[gi], g_og[gi])
            gi += 1
        ah, wh = acc.value(m["home_team"].iat[i], now)
        aa, wa = acc.value(m["away_team"].iat[i], now)
        out[i] = (ah, aa, wh, wa)
    res = pd.DataFrame(out, columns=["att_home", "att_away", "wc_home", "wc_away"])
    res.index = m["index"].to_numpy()
    return res.reindex(matches.index)


def current_player_factors(goals: pd.DataFrame, ref_date: str) -> pd.DataFrame:
    """Snapshot every team's attack_talent and wc_player as of `ref_date`."""
    acc = _Accumulator()
    now = np.datetime64(ref_date)
    sub = goals[goals["date"].to_numpy() < now]
    for team, scorer, t, is_wc, og in zip(sub["team"], sub["scorer"],
                                           sub["date"].to_numpy(), sub["is_wc"], sub["own_goal"]):
        acc.add_goal(team, scorer, t, is_wc, og)
    teams = sorted(set(goals["team"]))
    rows = {t: acc.value(t, now) for t in teams}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["attack_talent", "wc_player"])


def top_scorers(goals: pd.DataFrame, team: str, since: str, n: int = 6) -> pd.DataFrame:
    """Recent top scorers for a team (for the UI / explainability)."""
    sub = goals[(goals["team"] == team) & (goals["date"] >= since) & (~goals["own_goal"])]
    tab = (sub.groupby("scorer").agg(goals=("scorer", "size"),
                                     wc_goals=("is_wc", "sum"))
           .sort_values("goals", ascending=False).head(n))
    return tab
