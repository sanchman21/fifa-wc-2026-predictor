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
- **Squad skill covers the whole pitch, from EA player ratings.** Goals only see attackers, so a
  **squad-skill** factor (EA Sports FC / FIFA overall ratings, the *skill*-based — not monetary —
  measure) captures defenders, midfielders and goalkeepers too. Its weight is learned as-of-date
  like every other factor.
- **It scores the players actually selected.** Squad skill is built from the **official 26-man
  squads** (Wikipedia), each player matched to his EA rating — so only called-up players count,
  not a nation's top-rated names who stayed home. No user input, no API key.
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

### Squad-skill setup (Kaggle credentials)

The current FC26 ratings download with **no auth**. The **historical** editions (for training the
squad-skill weight as-of-date) come from Kaggle and need a free token:

1. Create a token at <https://www.kaggle.com/settings> → **Create New Token** (downloads `kaggle.json`).
2. Copy `.env.example` to `.env` and fill in:
   ```
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key
   ```
   `.env` is **git-ignored** — it is never committed. On Streamlit Community Cloud, put the same
   two keys in **Settings → Secrets** instead (the app reads `st.secrets` automatically).

Without credentials everything still runs — the skill factor simply falls back to neutral (no
crash), and you lose only the historical training signal for that one factor.

### Official squads (no key needed)

The squad-skill factor uses the official 26-man squads from the
[2026 FIFA World Cup squads](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads) Wikipedia
page — fetched automatically (one request, no auth) and cached to `data/squads_2026.json`.
`python -m src.data.fetch_squads` (or the sidebar **🔄 Refresh**) refreshes it. If the cache is
missing, the factor falls back to each nation's top-rated EA players.

---

## How the model works

### 1. Factors (all computed from data, as-of-date)
| Factor | Source | What it is |
|---|---|---|
| **Elo** | match history (49k games) | World-Football-Elo replayed over every international, with importance-weighted K-factors and goal-difference scaling. |
| **Attacking talent** | `goalscorers.csv` | Recency-weighted goals, each weighted by the **scorer's proven quality** (career goals to date). Real player performance — not market value. |
| **WC pedigree (players)** | `goalscorers.csv` | The nation's World Cup goal record, same quality/recency weighting — players who deliver on the biggest stage. |
| **Recent form** | match history | Trailing ~2-year share of points. |
| **Squad skill** | EA Sports FC / FIFA ratings | Rank-weighted mean of a nation's strongest players' **overall** ratings — covering defence, midfield and goalkeeping that goal data can't. Read **as-of-date** from the EA edition active at each match; current squads use real call-ups where EA licenses them. |
| **Manager / coaching** | match history + appointment dates | Results over/under-performance vs Elo expectation **under the current manager**. |

The two player factors and the coaching factor replace the previous version's *subjective* manager
score; squad strength now comes from both real goal output (attacking talent) and skill ratings
(squad skill) rather than market value.

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

Typical learned weights: **Elo ≈ 56% · form ≈ 18% · attacking talent ≈ 14% · WC pedigree ≈ 5% ·
squad skill ≈ 5% · coaching ≈ 1%.** (Coaching is small because manager overperformance beyond
squad quality is genuinely hard to detect statistically — an honest data finding, reported as-is.
Adding squad skill lifts held-out W/D/L accuracy from ~58.3% Elo-only to ~59.1%.)

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
  - **Official squads** (UI **Squad** tab) — the squad-skill factor is built from each nation's
    official 26-man squad (matched to EA ratings), refreshed alongside scores. Read-only.

## Interactive UI (`streamlit run app.py`)
Eleven tabs: **Overview** · **Match Predictor** · **Live / What-if** · **Model & Weights**
(learned weights + validation vs Elo-only) · **Power Ratings** (per-team factor attribution) ·
**Groups** · **Knockouts** (stage heatmap, predicted bracket) · **Players** (top scorers,
attack-talent ranks) · **Squad** (official 26-man squads + EA ratings) · **Managers** (tenure +
coaching effect) · **Data** (downloads). Training is cached; re-simulate, predict matches, refresh
to live scores, or run what-ifs from the UI.

---

## Data sources
| Data | Source | Notes |
|---|---|---|
| International results 1872–2026 | [martj42/international_results](https://github.com/martj42/international_results) `results.csv` | CC BY-NC-SA 4.0. |
| Goal-by-goal data (incl. World Cups) | same repo `goalscorers.csv` | 47k goals with scorer, penalty, own-goal. Powers the player factors. |
| Player skill ratings (historical, FIFA 15→FC24) | Kaggle [stefanoleone992/ea-sports-fc-24-complete-player-dataset](https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset) | One snapshot per edition; powers the as-of-date squad-skill factor. **Needs Kaggle credentials** (see *Squad-skill setup* below). |
| Player skill ratings (current, FC26) | [ismailoksuz/EAFC26-DataHub](https://github.com/ismailoksuz/EAFC26-DataHub) `data/players.csv` | No auth. Current ratings used to score each squad player. |
| Official 26-man squads | [2026 FIFA World Cup squads](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads) (Wikipedia) | No auth. Cached to `data/squads_2026.json`; defines who counts toward each team's squad skill. |
| Elo cross-check | [eloratings.net](https://www.eloratings.net/) | Reference only; the model computes its own Elo. |
| 2026 groups & bracket | FIFA draw (2025-12-05) + Mar 2026 playoff results | `config/teams_2026.json`, `config/bracket_2026.json`. |
| Managers & appointment dates | Wikipedia / FIFA / news (June 2026) | `config/managers_2026.json`. |

---

## Project layout
```
config/   teams_2026.json · managers_2026.json · bracket_2026.json   (editable inputs)
data/     downloaded source data (git-ignored, re-fetchable)
src/
  data/      fetch_data.py · fetch_skills.py · fetch_squads.py · secrets.py · config_loader.py
  features/  elo.py · players.py · form.py · skills.py · availability.py · manager.py
  model/     train.py (LEARNS the weights) · match_model.py · tournament.py (Monte-Carlo)
  explain/   report.py
  main.py    pipeline: prepare_model() + run_simulation()
app.py    Streamlit UI
.env      Kaggle credentials (git-ignored; see .env.example)
output/   generated report.md, CSVs, charts
```

---

## Caveats
- The two *goal-based* player factors use **goals only** (attacking output + big-stage pedigree);
  the **squad-skill** factor adds defence/midfield/goalkeeping via EA ratings, which carry a mild
  popularity / big-league bias and thin coverage for a few smaller nations (their few EA-rated
  players still give a correctly low rating).
- Official-squad scoring relies on cross-source **name matching** (Wikipedia squad names → EA
  players). It's robust to nicknames/accents and resolves namesakes by rating, but a selected
  player absent from EA is counted at a fringe baseline (shown in the Squad tab) — so the rare
  miss slightly *under*-rates a team rather than wrongly inflating it.
- The coaching factor is bounded by the current manager's tenure; very new appointments fall back
  to the team's recent overperformance. A few 2026 appointments (e.g. Morocco, Ghana, Saudi
  Arabia) are weeks old, so their coaching signal is thin — by design.
- `teams_2026.json` still carries `squad_value_m`/`manager_score` fields for reference; the model
  doesn't use them yet (a market-value factor is a planned addition alongside squad skill).
- The best-third → R32 allocation reproduces a valid bracket via constraint matching.
- A "most likely champion" at ~20–28% means the field is genuinely open — as it should be.
