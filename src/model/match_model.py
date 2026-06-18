"""Match outcome model: rating gap -> expected goals -> scoreline distribution.

Expected goals come from the calibrated Elo->goals mapping. Exact-score probabilities
use a Dixon-Coles-adjusted double Poisson (the low-score correction that fixes the
classic independent-Poisson under-prediction of draws). All functions accept scalars
or numpy arrays so the same math drives both single-match reports and the vectorised
Monte-Carlo simulation.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import poisson

MIN_LAMBDA = 0.15

# Pre-computed log-factorials for the numpy-only fast path below (calibration calls it tens of
# thousands of times, so we avoid scipy's per-call overhead there).
_MAXG = 10
_K = np.arange(_MAXG + 1)
_LOG_FACT = np.array([math.lgamma(k + 1) for k in _K])


def _dc_score_matrix(la, lb, rho):
    """Normalised Dixon-Coles exact-score matrix (numpy-only fast path; scalar la/lb > 0)."""
    gh = np.exp(-la + _K * np.log(la) - _LOG_FACT)
    ga = np.exp(-lb + _K * np.log(lb) - _LOG_FACT)
    m = np.outer(gh, ga)
    m[0, 0] *= 1.0 - la * lb * rho
    m[0, 1] *= 1.0 + la * rho
    m[1, 0] *= 1.0 + lb * rho
    m[1, 1] *= 1.0 - rho
    return m / m.sum()


def scoreline_prob(la, lb, i, j, rho=-0.12):
    """Probability of the exact scoreline (i, j) under the Dixon-Coles double Poisson.

    Goal counts above the internal grid are clamped to its edge (negligible probability).
    Used by the calibration fit, which scores locked predictions against real scorelines.
    """
    m = _dc_score_matrix(la, lb, rho)
    return float(m[min(int(i), _MAXG), min(int(j), _MAXG)])


def expected_goals(r_home, r_away, elo_per_goal, total_goals, net_home_adv=0.0):
    """Return (lambda_home, lambda_away) expected goals from the rating gap."""
    gap = np.asarray(r_home, float) + net_home_adv - np.asarray(r_away, float)
    supremacy = gap / elo_per_goal
    la = np.clip((total_goals + supremacy) / 2.0, MIN_LAMBDA, None)
    lb = np.clip((total_goals - supremacy) / 2.0, MIN_LAMBDA, None)
    return la, lb


def _dc_tau(i, j, la, lb, rho):
    """Dixon-Coles low-score adjustment factor for scoreline (i, j)."""
    if i == 0 and j == 0:
        return 1.0 - la * lb * rho
    if i == 0 and j == 1:
        return 1.0 + la * rho
    if i == 1 and j == 0:
        return 1.0 + lb * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def scoreline_matrix(la, lb, rho=-0.12, max_goals=10):
    """(max_goals+1) x (max_goals+1) matrix of exact-score probabilities (scalar la/lb)."""
    gh = poisson.pmf(np.arange(max_goals + 1), la)
    ga = poisson.pmf(np.arange(max_goals + 1), lb)
    m = np.outer(gh, ga)
    for i in (0, 1):
        for j in (0, 1):
            m[i, j] *= _dc_tau(i, j, la, lb, rho)
    return m / m.sum()


def outcome_probs(la, lb, rho=-0.12, max_goals=10):
    """Return (P_home_win, P_draw, P_away_win) for a single match."""
    m = scoreline_matrix(la, lb, rho, max_goals)
    p_home = np.tril(m, -1).sum()   # home goals > away goals
    p_draw = np.trace(m)
    p_away = np.triu(m, 1).sum()
    return float(p_home), float(p_draw), float(p_away)
