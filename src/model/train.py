"""Learn the factor weights from historical matches (not from user inputs).

We compute every factor *as of each match's date* over ~15 years of internationals, then fit
a model that predicts the actual outcome from the gap between the two teams' factors. The
fitted, standardized coefficients ARE the weights — assigned by the data, not chosen by hand.

Primary model: ridge regression of goal difference on standardized factor gaps (gives an
expected-supremacy function that feeds the Poisson simulator). A multinomial logistic on
Win/Draw/Loss is fit alongside purely to report out-of-sample accuracy and to show how much
the player/coach factors add on top of Elo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from ..features.elo import EloResult
from ..features.form import asof_form, current_form
from ..features.manager import build_coach_asof, current_coach
from ..features.players import asof_player_features, current_player_factors
from ..features.skills import asof_skill_features, current_skill_factors

FACTORS = ["elo", "attack_talent", "wc_player", "form", "skill", "coach"]
FACTOR_LABELS = {
    "elo": "Elo (match results)",
    "attack_talent": "Attacking talent (players)",
    "wc_player": "World Cup pedigree (players)",
    "form": "Recent form",
    "skill": "Squad skill (EA player ratings)",
    "coach": "Manager / coaching effect",
}


@dataclass
class TrainedModel:
    betas: dict                 # factor -> effective coefficient (goals per SD gap) = slope*weight
    beta_home: float            # home/host advantage in goals
    means: dict                 # factor -> training mean (team-level)
    sds: dict                   # factor -> training SD (team-level)
    total_goals: float
    rho: float
    weights: dict               # factor -> Shapley relative-importance share (sums to 1)
    metrics: dict               # fit/validation metrics
    n_train: int
    field: dict = field(default_factory=dict)

    def supremacy(self, fz_home: np.ndarray, fz_away: np.ndarray, home_flag=0.0):
        b = np.array([self.betas[f] for f in FACTORS])
        return float((fz_home - fz_away) @ b + self.beta_home * home_flag)


def _r2(X: np.ndarray, y: np.ndarray) -> float:
    """OLS R^2 with intercept (X already includes any controls)."""
    A = np.column_stack([X, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    sse = float(np.sum((y - A @ coef) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - sse / sst


def shapley_importance(Z: np.ndarray, control: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LMG / Shapley-regression relative importance of each column of Z (controlling for
    `control`), i.e. each factor's fairly-shared contribution to explained variance.

    This is the principled way to weight CORRELATED predictors: Elo, form, attack and
    coaching all overlap, so a plain regression hands Elo everything. Shapley averages each
    factor's marginal R^2 gain over every ordering, splitting shared credit fairly.
    """
    k = Z.shape[1]
    idx = list(range(k))
    phi = np.zeros(k)
    # cache R^2 for every subset
    cache = {}
    def r2_of(subset):
        key = frozenset(subset)
        if key not in cache:
            cols = [Z[:, j] for j in subset]
            X = np.column_stack(cols + [control]) if cols else control.reshape(-1, 1)
            cache[key] = _r2(X, y)
        return cache[key]
    for i in idx:
        others = [j for j in idx if j != i]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                w = factorial(len(S)) * factorial(k - len(S) - 1) / factorial(k)
                phi[i] += w * (r2_of(list(S) + [i]) - r2_of(list(S)))
    phi = np.clip(phi, 0.0, None)
    return phi / phi.sum() if phi.sum() > 0 else np.ones(k) / k


def _assemble_asof(pre: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    """Per-match home/away raw factor values, computed strictly from prior data."""
    df = pd.DataFrame(index=pre.index)
    df["date"] = pre["date"].to_numpy()
    df["elo_home"] = pre["pre_elo_home"].to_numpy()
    df["elo_away"] = pre["pre_elo_away"].to_numpy()
    df["gd"] = pre["home_score"].to_numpy() - pre["away_score"].to_numpy()
    df["neutral"] = pre["neutral"].to_numpy()
    fm = asof_form(pre); df["form_home"] = fm["form_home"]; df["form_away"] = fm["form_away"]
    co = build_coach_asof(pre); df["coach_home"] = co["coach_home"]; df["coach_away"] = co["coach_away"]
    pl = asof_player_features(pre, goals)
    df["attack_talent_home"] = pl["att_home"]; df["attack_talent_away"] = pl["att_away"]
    df["wc_player_home"] = pl["wc_home"]; df["wc_player_away"] = pl["wc_away"]
    sk = asof_skill_features(pre)   # EA squad ratings from the edition active at kickoff
    df["skill_home"] = sk["skill_home"]; df["skill_away"] = sk["skill_away"]
    return df


def train(pre: pd.DataFrame, goals: pd.DataFrame, since="2011-01-01", seed=2026) -> TrainedModel:
    asof = _assemble_asof(pre, goals)
    # established teams only: both sides have a real Elo (played before) and we're in the modern era
    mask = (asof["date"] >= pd.Timestamp(since)) & (asof["elo_home"] != 1500.0) & (asof["elo_away"] != 1500.0)
    d = asof[mask].copy()

    # pooled team-level standardization (home and away observations together). A factor with NO
    # usable data (e.g. squad skill when the historical EA file couldn't be downloaded — no Kaggle
    # creds on a cloud cold start) yields an all-NaN column; guard mean/SD to finite values so it
    # degrades to a neutral (zero-contribution) factor instead of poisoning the regression.
    means, sds = {}, {}
    for f in FACTORS:
        pooled = np.concatenate([d[f"{f}_home"].to_numpy(float), d[f"{f}_away"].to_numpy(float)])
        has = np.any(np.isfinite(pooled))
        mu = float(np.nanmean(pooled)) if has else 0.0
        sd = float(np.nanstd(pooled)) if has else 0.0
        means[f] = mu if np.isfinite(mu) else 0.0
        sds[f] = sd if (np.isfinite(sd) and sd > 0) else 1.0

    # standardized home-minus-away gap per factor; NaN factor values (a team with no EA skill
    # rating, or pre-2014 matches before the first edition) are imputed to the factor mean so
    # their standardized gap is 0. nan_to_num is a final guard so no NaN/inf can reach lstsq.
    Z = np.column_stack([
        ((d[f"{f}_home"].fillna(means[f]) - means[f]) / sds[f])
        - ((d[f"{f}_away"].fillna(means[f]) - means[f]) / sds[f])
        for f in FACTORS])
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    home = (~d["neutral"]).astype(float).to_numpy()
    y_gd = d["gd"].to_numpy(float)
    y_cls = np.where(d["gd"] > 0, 2, np.where(d["gd"] == 0, 1, 0))   # 0=away,1=draw,2=home

    # --- WEIGHTS: Shapley relative importance (fair across correlated factors) ----------
    phi = shapley_importance(Z, home, y_gd)
    weights = {f: float(phi[i]) for i, f in enumerate(FACTORS)}

    # --- GOAL SCALE: calibrate the composite -> goals mapping + home advantage ----------
    composite = Z @ phi
    A = np.column_stack([composite, home, np.ones(len(y_gd))])
    coef, *_ = np.linalg.lstsq(A, y_gd, rcond=None)
    slope, beta_home = float(coef[0]), float(coef[1])
    betas = {f: slope * weights[f] for f in FACTORS}   # effective goals-per-SD per factor

    # --- VALIDATION: does adding player+coach factors beat Elo alone? --------------------
    Xfull = np.column_stack([Z, home])
    Xelo = np.column_stack([Z[:, 0], home])
    Xtr, Xte, ctr, cte, etr, ete = train_test_split(Xfull, y_cls, Xelo, test_size=0.2,
                                                     random_state=seed)
    full = LogisticRegression(max_iter=3000).fit(Xtr, ctr)
    eonly = LogisticRegression(max_iter=3000).fit(etr, ctr)
    metrics = {
        "full_accuracy": float(accuracy_score(cte, full.predict(Xte))),
        "full_logloss": float(log_loss(cte, full.predict_proba(Xte))),
        "elo_only_accuracy": float(accuracy_score(cte, eonly.predict(ete))),
        "elo_only_logloss": float(log_loss(cte, eonly.predict_proba(ete))),
        "composite_r2_goal_diff": _r2(np.column_stack([composite, home]), y_gd),
        "home_adv_goals": beta_home,
    }
    total_goals = float((pre.loc[d.index, "home_score"] + pre.loc[d.index, "away_score"]).mean())

    return TrainedModel(betas=betas, beta_home=beta_home, means=means, sds=sds,
                        total_goals=total_goals, rho=-0.12, weights=weights,
                        metrics=metrics, n_train=len(d))


def current_factor_table(teams: pd.DataFrame, elo: EloResult, pre: pd.DataFrame,
                         goals: pd.DataFrame, appointments_dn: dict, ref_date: str,
                         skill: pd.Series | None = None) -> pd.DataFrame:
    """Raw current factor values per 2026 team (indexed by team display name).

    `skill` may be supplied to override; otherwise it is the current squad-skill per team
    (from the official 26-man squads matched to EA ratings)."""
    dn = teams["data_name"]
    pf = current_player_factors(goals, ref_date)
    fm = current_form(pre, ref_date)
    co = current_coach(pre, appointments_dn, ref_date)
    sk = current_skill_factors(teams) if skill is None else skill.reindex(teams.index)
    out = pd.DataFrame(index=teams.index)
    out["elo"] = [elo.ratings.get(d, 1500.0) for d in dn]
    out["attack_talent"] = [float(pf["attack_talent"].get(d, 0.0)) for d in dn]
    out["wc_player"] = [float(pf["wc_player"].get(d, 0.0)) for d in dn]
    out["form"] = [float(fm.get(d, 0.5)) for d in dn]
    out["skill"] = sk.astype(float)
    out["coach"] = [float(co.get(d, 0.0)) for d in dn]
    return out


def build_power_table(teams: pd.DataFrame, factors: pd.DataFrame, model: TrainedModel) -> pd.DataFrame:
    """Per-team power rating (expected goal supremacy vs the field) + factor contributions."""
    df = teams.copy()
    for f in FACTORS:
        df[f] = factors[f]
        # impute any missing factor value to its training mean -> standardized 0 (neutral),
        # so e.g. an absent skill rating never NaN-propagates through the whole power table.
        df[f"z_{f}"] = (factors[f].fillna(model.means[f]) - model.means[f]) / model.sds[f]
    # contribution of each factor to the team's rating, centred on the 48-team field mean
    field_meanz = {f: df[f"z_{f}"].mean() for f in FACTORS}
    for f in FACTORS:
        df[f"contrib_{f}"] = model.betas[f] * (df[f"z_{f}"] - field_meanz[f])
    df["power_score"] = sum(model.betas[f] * df[f"z_{f}"] for f in FACTORS)   # goal-supremacy units
    s = df["power_score"]
    df["power_index"] = 100 * (s - s.min()) / (s.max() - s.min())
    df = df.sort_values("power_score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
