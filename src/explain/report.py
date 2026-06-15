"""Turn the model outputs into explainable artifacts: CSVs, charts, and a report."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..model.train import FACTORS, FACTOR_LABELS

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
_COLORS = {"elo": "#1f77b4", "attack_talent": "#ff7f0e", "wc_player": "#9467bd",
           "form": "#d62728", "skill": "#17becf", "coach": "#2ca02c"}


def _ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_csvs(power: pd.DataFrame, probs: pd.DataFrame):
    _ensure_out()
    keep = (["rank", "group", "position", "confederation", "power_index", "power_score"]
            + FACTORS + [f"z_{f}" for f in FACTORS] + [f"contrib_{f}" for f in FACTORS])
    power[keep].round(4).to_csv(os.path.join(OUT_DIR, "strength_ratings.csv"))
    probs.round(4).to_csv(os.path.join(OUT_DIR, "predictions.csv"))


def chart_champion_odds(probs, top=15):
    d = probs.sort_values("P_champion", ascending=False).head(top)[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(d.index, d["P_champion"] * 100, color="#c8102e")
    for y, v in enumerate(d["P_champion"] * 100):
        plt.text(v + 0.1, y, f"{v:.1f}%", va="center", fontsize=8)
    plt.xlabel("Win probability (%)"); plt.title("FIFA World Cup 2026 — Title odds")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "champion_odds.png"), dpi=130); plt.close()


def chart_power_breakdown(power, top=20):
    d = power.head(top)[::-1]
    plt.figure(figsize=(11, 9))
    lp = np.zeros(len(d)); ln = np.zeros(len(d))
    for f in FACTORS:
        vals = d[f"contrib_{f}"].to_numpy()
        base = np.where(vals >= 0, lp, ln)
        plt.barh(d.index, vals, left=base, color=_COLORS[f], label=FACTOR_LABELS[f])
        lp += np.where(vals >= 0, vals, 0); ln += np.where(vals < 0, vals, 0)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("Contribution to power rating (goals of supremacy vs an average team)")
    plt.title("What drives each team's rating — learned factor contributions")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "power_breakdown.png"), dpi=130); plt.close()


def chart_weights(model):
    plt.figure(figsize=(7, 7))
    plt.pie([model.weights[f] for f in FACTORS], labels=[FACTOR_LABELS[f] for f in FACTORS],
            autopct="%1.0f%%", colors=[_COLORS[f] for f in FACTORS], startangle=90, counterclock=False)
    plt.title("Learned factor weights (trained on match history)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "factor_weights.png"), dpi=130); plt.close()


def chart_stage_heatmap(probs, top=20):
    cols = ["P_reach_R32", "P_reach_R16", "P_reach_QF", "P_reach_SF", "P_reach_Final", "P_champion"]
    labels = ["R32", "R16", "QF", "SF", "Final", "Win"]
    d = probs.sort_values("P_champion", ascending=False).head(top)
    arr = d[cols].to_numpy() * 100
    plt.figure(figsize=(9, 10))
    plt.imshow(arr, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    plt.colorbar(label="probability (%)")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(d)), d.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", fontsize=7,
                     color="black" if arr[i, j] < 60 else "white")
    plt.title("Stage-by-stage advancement probability (%)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "stage_heatmap.png"), dpi=130); plt.close()


def make_charts(power, probs, model):
    _ensure_out()
    chart_champion_odds(probs); chart_power_breakdown(power)
    chart_weights(model); chart_stage_heatmap(probs)


def write_report(power, probs, model, chalk, elo, n_sims):
    _ensure_out()
    pct = lambda x: f"{x * 100:.1f}%"
    m = model.metrics

    wt_rows = "\n".join(
        f"| {FACTOR_LABELS[f]} | **{pct(model.weights[f])}** | {model.betas[f]:+.3f} |"
        for f in sorted(FACTORS, key=lambda f: -model.weights[f]))

    grp_lines = []
    for g in sorted(power["group"].unique()):
        sub = probs.join(power["group"]).query("group == @g").copy()
        sub["P_adv"] = sub["P_group_winner"] + sub["P_group_runnerup"]
        sub = sub.sort_values("P_adv", ascending=False)
        grp_lines.append(f"\n**Group {g}**\n")
        grp_lines.append("| Team | Win grp | Runner-up | Advance | Reach R16 |")
        grp_lines.append("|---|---|---|---|---|")
        for t, r in sub.iterrows():
            grp_lines.append(f"| {t} | {pct(r.P_group_winner)} | {pct(r.P_group_runnerup)} "
                             f"| {pct(r.P_adv)} | {pct(r.P_reach_R16)} |")

    stage_favs = []
    for col, name in [("P_reach_R16", "Round of 16"), ("P_reach_QF", "Quarter-finals"),
                      ("P_reach_SF", "Semi-finals"), ("P_reach_Final", "Final"),
                      ("P_champion", "Champion")]:
        top = probs.sort_values(col, ascending=False).head(8)
        stage_favs.append(f"- **{name}:** " + ", ".join(f"{t} ({pct(v)})" for t, v in top[col].items()))

    pr = power.head(12)
    pr_tbl = ["| # | Team | Power Index | Elo | Strongest factor | Weakest factor |",
              "|---|---|---|---|---|---|"]
    for t, r in pr.iterrows():
        c = {f: r[f"contrib_{f}"] for f in FACTORS}
        up = max(c, key=c.get); dn = min(c, key=c.get)
        pr_tbl.append(f"| {int(r['rank'])} | {t} | {r['power_index']:.1f} | {r['elo']:.0f} "
                      f"| {FACTOR_LABELS[up]} ({c[up]:+.2f}) | {FACTOR_LABELS[dn]} ({c[dn]:+.2f}) |")

    sf = chalk["rounds"]["semi_finals"]; fn = chalk["rounds"]["final"][0]
    qf = chalk["rounds"]["quarter_finals"]
    champ = probs.index[0]; champ_p = probs.iloc[0]["P_champion"]

    md = f"""# FIFA World Cup 2026 — Statistical Forecast

*Explainable Monte-Carlo model with weights LEARNED from {model.n_train:,} historical matches.*
*Model run date: 2026-06-11. {n_sims:,} simulated tournaments.*

## Headline

- **Most likely champion: {champ} ({pct(champ_p)})**
- The 48-team tournament is simulated {n_sims:,} times — group stage (FIFA tiebreakers),
  the 8-best-third-placed allocation, and the full knockout bracket exactly as drawn.

## The weights are trained, not chosen

Each factor is computed **as of each match's date** over {model.n_train:,} internationals since
2011. Because Elo, form, attacking talent and coaching all overlap, a plain regression would hand
Elo almost all the weight; instead a **Shapley / LMG relative-importance decomposition** fairly
splits each factor's shared contribution to explained variance. Those shares ARE the weights —
no hand-tuning:

| Factor | Learned weight | Coef (goals / SD gap) |
|---|---|---|
{wt_rows}
| *Home/host advantage* | — | {model.beta_home:+.3f} goals |

**Does it beat Elo alone?** On a held-out 20% test set, the full model (with the player and
coaching factors) scores **{pct(m['full_accuracy'])}** outcome accuracy / log-loss
**{m['full_logloss']:.3f}**, versus **{pct(m['elo_only_accuracy'])}** / **{m['elo_only_logloss']:.3f}**
for an Elo-only model — so player and coaching data add real signal. Composite goal-difference
R² = {m['composite_r2_goal_diff']:.3f}; baseline ~{model.total_goals:.2f} goals/match.

The two **player factors come from real goal-by-goal data** (every international goal, weighted
by the scorer's proven track record, recency-decayed — so cheap-but-prolific players score
highly and market value is never used). The **coaching factor** is each team's results
over/under-performance versus Elo expectation *under the current manager's tenure* — measured
from data, not a subjective rating.

See `output/factor_weights.png`, `output/power_breakdown.png`, `output/stage_heatmap.png`,
`output/champion_odds.png`. Explore everything interactively with `streamlit run app.py`.

## Power ranking (top 12) with factor attribution

{chr(10).join(pr_tbl)}

## Stage-by-stage favorites

{chr(10).join(stage_favs)}

## Predicted single most-likely path

- **Semi-finals:** {sf[0][0]} vs {sf[0][1]} → **{sf[0][2]}**; {sf[1][0]} vs {sf[1][1]} → **{sf[1][2]}**
- **Final:** {fn[0]} vs {fn[1]} → **🏆 {fn[2]}**

*Quarter-finalists (chalk):* {", ".join(sorted({x for mm in qf for x in mm[:2]}))}

## Group-stage qualification probabilities
{chr(10).join(grp_lines)}

## Files

- `output/predictions.csv` — every team's probability of reaching each stage.
- `output/strength_ratings.csv` — power rating + each factor's value, z-score and contribution.

## Caveats

- Player factors use goals only (the richest reliable free goal-level feed); they capture
  attacking output and big-stage pedigree, not defensive/midfield contribution.
- The coaching factor is bounded by the current manager's tenure length; very new managers
  fall back to the team's recent overperformance.
- The best-third → R32 allocation reproduces a valid bracket via constraint matching.
"""
    path = os.path.join(OUT_DIR, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    return path
