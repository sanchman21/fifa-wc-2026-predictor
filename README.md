# FIFA World Cup 2026 — Explainable Statistical Forecast

A reproducible, **explainable** model that forecasts how all **48 teams** progress through
the FIFA World Cup 2026 — from the group stage to the Round of 32, Round of 16,
Quarter-finals, Semi-finals, Final, and the eventual champion.

What makes it different:
- **The factor weights are LEARNED from data, not chosen by hand.** Each factor is computed
  *as of every historical match's date* over ~15 years of internationals, and their importance
  is assigned by a **Shapley / LMG relative-importance analysis** of how well they predict
  real results.
- **Player strength comes from real player performance, not market value.** Every international
  goal (martj42 `goalscorers.csv`) is weighted by the scorer's proven track record and decayed
  by recency — so a cheap-but-prolific player rates highly. Past **World Cup** scoring is a
  separate factor.
- **Manager quality is measured, not guessed.** No subjective score: the coaching factor is each
  team's results over/under-performance versus Elo expectation *under the current manager's
  tenure* (real appointment dates).
- An **interactive Streamlit UI** to explore weights, ratings, groups, knockouts, players and
  managers.

> **Format note.** WC 2026 has **48 teams in 12 groups of 4**. Top 2 of each group (24) + the
> **8 best third-placed teams** (32 total) reach the Round of 32, then R16 → QF → SF → Final.

---

## Quick start

```bash
pip install -r requirements.txt

python -m src.data.fetch_data     # download source data into ./data (one time)
python -m src.main                # full run -> ./output (report, CSVs, charts)
streamlit run app.py              # interactive dashboard
```

Flags: `--sims 50000` (more simulations), `--refresh` (re-download data), `--seed N`.

---

## How the model works

### 1. Factors (all computed from data, as-of-date)
| Factor | Source | What it is |
|---|---|---|
| **Elo** | match history (49k games) | World-Football-Elo replayed over every international, with importance-weighted K-factors and goal-difference scaling. |
| **Attacking talent** | `goalscorers.csv` | Recency-weighted goals, each weighted by the **scorer's proven quality** (career goals to date). Real player performance — not market value. |
| **WC pedigree (players)** | `goalscorers.csv` | The nation's World Cup goal record, same quality/recency weighting — players who deliver on the biggest stage. |
| **Recent form** | match history | Trailing ~2-year share of points. |
| **Manager / coaching** | match history + appointment dates | Results over/under-performance vs Elo expectation **under the current manager**. |

The two player factors and the coaching factor replace the previous version's squad *value* and
subjective manager score — both now come from real performance data.

### 2. The weights are trained (`src/model/train.py`)
We build a feature for each of ~15k matches since 2011 from the **home-minus-away gap** in each
(standardized) factor and fit:
- **Relative importance via Shapley/LMG decomposition** → the factor **weights**. This is the
  correct way to weight *correlated* predictors: Elo, form, attack and coaching overlap, so a
  plain regression would hand Elo ~90% of the weight. Shapley averages each factor's marginal
  contribution to explained variance over every ordering, splitting the shared signal fairly.
- A **calibration regression** mapping the weighted composite (and a home/host dummy) to actual
  goal difference → the goals scale used by the simulator.
- A **multinomial logistic** on Win/Draw/Loss (held-out test set) to validate that the player +
  coaching factors genuinely add accuracy over an Elo-only baseline.

Typical learned weights: **Elo ≈ 58% · form ≈ 20% · attacking talent ≈ 15% · WC pedigree ≈ 6% ·
coaching ≈ 1%.** (Coaching is small because manager overperformance beyond squad quality is
genuinely hard to detect statistically — an honest data finding, reported as-is.)

### 3. Match model & simulation
The trained composite gives an **expected goal supremacy** per matchup, fed to a **Dixon-Coles
double-Poisson** for realistic scorelines and draw rates. **20,000 tournaments** are simulated
(vectorised numpy, a few seconds) reproducing the real competition: group round-robins → FIFA
tiebreakers → 8-best-third allocation → the official R32→Final bracket.

Invariants that always hold: stage probabilities sum to exactly 32/16/8/4/2/1; each group's
winner/runner-up probabilities sum to 1; advancement is monotone per team.

---

## Match predictions & live updating
- **Per-match predictions** (`src/model/predict.py`, UI **Match Predictor** tab): pick any two
  teams (or filter the real 2026 fixture list) for win/draw/loss, expected goals and the most
  likely scorelines — same Dixon-Coles math the simulator uses.
- **Live updating as the tournament progresses:**
  - **Real results** — click **🔄 Refresh data** (or `python -m src.main --refresh`). martj42's
    `results.csv`/`goalscorers.csv` fill in actual scores and goals live, so Elo, attacking
    talent, form and coaching all re-derive, and the simulation **conditions on every played
    match** (played group games are fixed; decided knockout ties are locked by team identity).
  - **What-if** (UI **Live / What-if** tab) — type in hypothetical group scores and recompute
    instantly to see how qualification and title odds shift, before any real data exists.

## Interactive UI (`streamlit run app.py`)
Ten tabs: **Overview** · **Match Predictor** · **Live / What-if** · **Model & Weights**
(learned weights + validation vs Elo-only) · **Power Ratings** (per-team factor attribution) ·
**Groups** · **Knockouts** (stage heatmap, predicted bracket) · **Players** (top scorers,
attack-talent ranks) · **Managers** (tenure + coaching effect) · **Data** (downloads). Training
is cached; re-simulate, predict matches, refresh to live scores, or run what-ifs from the UI.

---

## Data sources
| Data | Source | Notes |
|---|---|---|
| International results 1872–2026 | [martj42/international_results](https://github.com/martj42/international_results) `results.csv` | CC BY-NC-SA 4.0. |
| Goal-by-goal data (incl. World Cups) | same repo `goalscorers.csv` | 47k goals with scorer, penalty, own-goal. Powers the player factors. |
| Elo cross-check | [eloratings.net](https://www.eloratings.net/) | Reference only; the model computes its own Elo. |
| 2026 groups & bracket | FIFA draw (2025-12-05) + Mar 2026 playoff results | `config/teams_2026.json`, `config/bracket_2026.json`. |
| Managers & appointment dates | Wikipedia / FIFA / news (June 2026) | `config/managers_2026.json`. |

---

## Project layout
```
config/   teams_2026.json · managers_2026.json · bracket_2026.json   (editable inputs)
data/     downloaded source data (git-ignored, re-fetchable)
src/
  data/      fetch_data.py, config_loader.py
  features/  elo.py · players.py · form.py · manager.py
  model/     train.py (LEARNS the weights) · match_model.py · tournament.py (Monte-Carlo)
  explain/   report.py
  main.py    pipeline: prepare_model() + run_simulation()
app.py    Streamlit UI
output/   generated report.md, CSVs, charts
```

---

## Caveats
- Player factors use **goals only** (the richest reliable free goal-level feed): they capture
  attacking output and big-stage pedigree, not defensive/midfield contribution.
- The coaching factor is bounded by the current manager's tenure; very new appointments fall back
  to the team's recent overperformance. A few 2026 appointments (e.g. Morocco, Ghana, Saudi
  Arabia) are weeks old, so their coaching signal is thin — by design.
- `teams_2026.json` still carries `squad_value_m`/`manager_score` fields for reference; the model
  no longer uses them.
- The best-third → R32 allocation reproduces a valid bracket via constraint matching.
- A "most likely champion" at ~20–28% means the field is genuinely open — as it should be.
