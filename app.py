"""Interactive UI for the FIFA World Cup 2026 forecast.

    streamlit run app.py

Training (Elo + learned weights) is cached once. You can re-simulate with different
counts/seeds, predict individual matches, refresh to the latest real scores, and run
"what-if" scenarios that condition the forecast on results you enter.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.data import fetch_data, fetch_skills, fetch_squads
from src.features.players import top_scorers
from src.features.skills import DEFAULT_OVERALL, squad_rating
from src.main import prepare_model, run_simulation
from src.model.predict import predict_fixtures, predict_match
from src.model.train import FACTORS, FACTOR_LABELS

st.set_page_config(page_title="WC 2026 Forecast", page_icon="🏆", layout="wide")

STAGES = ["P_reach_R32", "P_reach_R16", "P_reach_QF", "P_reach_SF", "P_reach_Final", "P_champion"]
STAGE_LABELS = ["R32", "R16", "QF", "SF", "Final", "Champion"]


@st.cache_resource(show_spinner="Loading data, computing Elo, training weights…")
def _prep():
    return prepare_model(verbose=False)


@st.cache_data(show_spinner="Simulating tournaments…")
def _sim(sims: int, seed: int, known_items=None, sup_scale: float = 1.0, goal_scale: float = 1.0):
    prep = _prep()
    kg = dict(known_items) if known_items else None
    probs, chalk = run_simulation(prep, sims=sims, seed=seed, known_group=kg,
                                  sup_scale=sup_scale, goal_scale=goal_scale)
    return probs, chalk


def pct(x):
    return f"{x*100:.1f}%"


prep = _prep()
teams, model, power, goals, managers, results, fixtures = (
    prep["teams"], prep["model"], prep["power"], prep["goals"],
    prep["managers"], prep["results"], prep["fixtures"])
squads, squads_info = prep["squads"], prep["squads_meta"]
# Tracking is an optional overlay; tolerate its absence so the forecast still renders.
from src.model import tracking as _tracking  # noqa: E402
track = prep.get("track_record") or _tracking.track_record({})
calib = prep.get("calibration") or _tracking.calibration_scale({}, model)

st.title("🏆 FIFA World Cup 2026 — Explainable Forecast")
st.caption(f"48 teams · weights LEARNED from {model.n_train:,} matches · "
           f"data through {results.date.max().date()} · player factors from real goal data")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh data (pull latest scores)", use_container_width=True):
        with st.spinner("Downloading latest results, goals, squad ratings & official squads…"):
            fetch_data.fetch_all(force=True)
            fetch_skills.fetch_all(force=True)
            fetch_squads.fetch_squads(force=True)   # refresh official 26-man squads
        _prep.clear(); _sim.clear()
        st.rerun()
    sims = st.select_slider("Simulations", [5000, 10000, 20000, 30000, 50000], value=20000)
    seed = st.number_input("Random seed", value=2026, step=1)
    n_played = int(fixtures["played"].sum())
    st.caption(f"{n_played} of {len(fixtures)} WC-2026 matches played so far "
               "(forecast auto-conditions on real results).")
    st.divider()
    st.subheader("Self-calibration")
    can_calibrate = calib["n"] > 0
    use_calib = st.toggle(
        f"Apply learned calibration (gap ×{calib['sup_scale']:.3f}, goals ×{calib['goal_scale']:.3f})"
        if can_calibrate else "Apply learned calibration",
        value=bool(calib["applied"]), disabled=not can_calibrate,
        help="Rescales the rating gap (who wins, by how much) AND the expected goal total (how many "
             "goals — which also sets the draw rate), fit jointly from the model's own locked "
             "predictions vs the real scorelines. Needs ≥12 graded matches before it nudges anything.")
    if can_calibrate:
        st.caption(f"Fitted on {calib['n']} graded match(es). "
                   + ("Self-correcting margin & goal volume." if calib["applied"]
                      else "Too few results to apply automatically — toggle to preview."))
    else:
        st.caption("No graded matches yet — calibration unlocks once results come in.")
    sup_scale = float(calib["sup_scale"]) if (use_calib and can_calibrate) else 1.0
    goal_scale = float(calib["goal_scale"]) if (use_calib and can_calibrate) else 1.0
    st.divider()
    st.subheader("Learned weights")
    for f in sorted(FACTORS, key=lambda f: -model.weights[f]):
        st.progress(model.weights[f], text=f"{FACTOR_LABELS[f]} — {pct(model.weights[f])}")

probs, chalk = _sim(int(sims), int(seed), sup_scale=sup_scale, goal_scale=goal_scale)

tabs = st.tabs(["🏆 Overview", "🔮 Match Predictor", "📈 Track Record", "📡 Live / What-if",
                "📊 Model & Weights", "💪 Power Ratings", "🅰️ Groups", "🗺️ Knockouts",
                "⚽ Players", "🩹 Squad", "🧑‍💼 Managers", "📥 Data"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    mlf = chalk["most_likely_final"]
    runner_up = mlf["away"] if mlf["champion"] == mlf["home"] else mlf["home"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted champion", mlf["champion"], pct(probs.loc[mlf["champion"], "P_champion"]))
    c2.metric("Predicted final", f"{mlf['home']} vs {mlf['away']}",
              help="The two semi-final winners feeding the final come from opposite halves of "
                   "the draw, so this matchup is always bracket-valid and the champion is one of "
                   f"these two. This exact final occurs in {pct(mlf['p_this_exact_final'])} of sims.")
    c3.metric("Runner-up", runner_up)
    st.caption(f"Most likely to reach the final from each half: **{mlf['home']}** "
               f"({pct(mlf['p_home_reaches_final'])}) vs **{mlf['away']}** "
               f"({pct(mlf['p_away_reaches_final'])}). The predicted champion is the more likely "
               "title winner of the two — guaranteed to be one of the predicted finalists.")
    st.subheader("Title odds")
    d = probs.sort_values("P_champion", ascending=False).head(16).rename_axis("team").reset_index()
    d["pct"] = d["P_champion"] * 100
    st.altair_chart(alt.Chart(d).mark_bar(color="#c8102e").encode(
        x=alt.X("pct:Q", title="Win probability (%)"), y=alt.Y("team:N", sort="-x", title=None),
        tooltip=["team", alt.Tooltip("pct:Q", format=".1f")]).properties(height=460),
        use_container_width=True)

# ---------------------------------------------------------------- Match Predictor
with tabs[1]:
    st.subheader("Predict any match")
    cc = st.columns(3)
    home = cc[0].selectbox("Home / first team", list(power.index), index=list(power.index).index("Spain"))
    away = cc[1].selectbox("Away / second team", list(power.index), index=list(power.index).index("Germany"))
    cc[2].caption("Host nations automatically get home advantage; otherwise treated as neutral.")
    if home == away:
        st.info("Pick two different teams.")
    else:
        pr = predict_match(power, model, home, away, sup_scale=sup_scale, goal_scale=goal_scale)
        m = st.columns(3)
        m[0].metric(f"{home} win", pct(pr["p_home"]))
        m[1].metric("Draw", pct(pr["p_draw"]))
        m[2].metric(f"{away} win", pct(pr["p_away"]))
        st.write(f"**Expected score:** {home} {pr['exp_home']:.2f} – {pr['exp_away']:.2f} {away}")
        sl = pd.DataFrame(pr["scorelines"], columns=["score", "prob"])
        st.altair_chart(alt.Chart(sl).mark_bar().encode(
            x=alt.X("prob:Q", axis=alt.Axis(format="%"), title="probability"),
            y=alt.Y("score:N", sort="-x", title="most likely scorelines"),
            tooltip=["score", alt.Tooltip("prob:Q", format=".1%")]).properties(height=240),
            use_container_width=True)
    st.divider()
    st.subheader("Predicted scheduled fixtures")
    only_unplayed = st.checkbox("Only unplayed", value=True)
    gsel = st.selectbox("Group filter", ["All"] + sorted(power["group"].unique()))
    pf = predict_fixtures(power, model, fixtures, only_unplayed=only_unplayed,
                          sup_scale=sup_scale, goal_scale=goal_scale)
    if gsel != "All":
        pf = pf[pf["group"] == gsel]
    st.dataframe(pf.style.format({"P(home win)": "{:.0%}", "P(draw)": "{:.0%}", "P(away win)": "{:.0%}"}),
                 use_container_width=True, hide_index=True, height=380)

# ---------------------------------------------------------------- Track Record
with tabs[2]:
    st.subheader("How the model's own predictions have scored")
    st.caption("Every WC-2026 prediction is *locked* while the match is unplayed, then graded "
               "when the real score arrives — so this is a genuine out-of-sample record, not "
               "hindsight. The learned calibration scale (sidebar) is fit from exactly these results.")
    if not track["n_graded"]:
        st.info(f"No matches graded yet — **{track['n_pending']}** pre-match prediction(s) are "
                "locked in and waiting for results. Click **🔄 Refresh data** once matches are "
                "played to pull the scores and start grading.")
    else:
        k = st.columns(4)
        k[0].metric("Matches graded", track["n_graded"], f"{track['n_pending']} pending")
        k[1].metric("Outcome hit-rate", pct(track["accuracy"]))
        k[2].metric("Log-loss", f"{track['logloss']:.3f}", f"{track['skill_pct']:+.0f}% skill vs coin-flip",
                    delta_color="normal")
        k[3].metric("Brier score", f"{track['brier']:.3f}", help="0 = perfect, lower is better")
        st.caption(f"Mean margin error: {track['mean_goal_err']:.2f} goals/match (error in the goal "
                   f"*difference*). A 3-way coin-flip scores log-loss {1.0986:.3f}.")

        st.markdown("##### Goal volume & draws — is the model scoring the right *amount*?")
        g2 = st.columns(2)
        g2[0].metric("Goals/match — actual vs predicted", f"{track['goals_actual']:.2f}",
                     f"{track['goals_bias']:+.2f} vs {track['goals_pred']:.2f} predicted",
                     delta_color="off",
                     help="Positive bias ⇒ real matches are out-scoring the model. The goals "
                          "calibration (sidebar) lifts the forecast's goal total to close this.")
        g2[1].metric("Draw rate — actual vs predicted", pct(track['draw_rate_actual']),
                     f"{(track['draw_rate_actual'] - track['draw_rate_pred']) * 100:+.0f} pts vs "
                     f"{pct(track['draw_rate_pred'])} predicted", delta_color="off",
                     help="Actual share of draws vs the model's average draw probability. Raising the "
                          "goal total lowers predicted draws, so the goals scale tunes this too.")

        st.markdown("##### Calibration — when the model is *this* confident, how often is it right?")
        cal = track["calibration"]
        if len(cal):
            cal_v = cal.assign(avg_confidence=cal["avg_confidence"] * 100,
                               hit_rate=cal["hit_rate"] * 100).melt(
                id_vars=["bucket", "n"], value_vars=["avg_confidence", "hit_rate"],
                var_name="metric", value_name="pct")
            cal_v["metric"] = cal_v["metric"].map({"avg_confidence": "Model confidence",
                                                   "hit_rate": "Actual hit-rate"})
            st.altair_chart(alt.Chart(cal_v).mark_bar().encode(
                x=alt.X("bucket:N", title="Confidence bucket", sort=list(cal["bucket"])),
                xOffset="metric:N",
                y=alt.Y("pct:Q", title="%"), color=alt.Color("metric:N", title=None),
                tooltip=["bucket", "metric", alt.Tooltip("pct:Q", format=".0f"), "n"]).properties(height=280),
                use_container_width=True)
            st.caption("Bars at equal height ⇒ well-calibrated. Confidence consistently above the "
                       "hit-rate ⇒ over-confident (scale < 1 corrects it); below ⇒ under-confident.")

        if calib["baseline_logloss"] is not None:
            st.markdown("##### Learned calibration (jointly fit on real scorelines)")
            cc = st.columns(4)
            cc[0].metric("Supremacy scale", f"{calib['sup_scale']:.3f}",
                         "applied" if calib["applied"] else "preview only",
                         help="<1 tempers favourites, >1 sharpens them.")
            cc[1].metric("Goals scale", f"{calib['goal_scale']:.3f}",
                         "applied" if calib["applied"] else "preview only",
                         help=">1 lifts the expected goal total (and lowers the draw rate).")
            cc[2].metric("Scoreline log-loss @ raw", f"{calib['baseline_scoreline_nll']:.3f}",
                         help="The quantity the calibration minimises — negative log-likelihood of "
                              "the actual scorelines (captures margin, goal volume AND draws).")
            cc[3].metric("@ calibrated", f"{calib['calibrated_scoreline_nll']:.3f}",
                         f"{calib['calibrated_scoreline_nll'] - calib['baseline_scoreline_nll']:+.3f}",
                         delta_color="inverse")

        st.markdown("##### Graded predictions")
        df = track["frame"].copy()
        df["hit"] = df["correct"].map({1: "✅", 0: "❌"})
        show = df[["date", "match", "stage", "predicted", "pick", "score", "actual", "hit",
                   "p_actual", "brier", "logloss"]].rename(
            columns={"pick": "confidence", "p_actual": "P(actual)"})
        st.dataframe(show.style.format({"confidence": "{:.0%}", "P(actual)": "{:.0%}",
                     "brier": "{:.3f}", "logloss": "{:.3f}"}),
                     use_container_width=True, hide_index=True, height=360)
        st.download_button("⬇ track_record.csv", df.to_csv(index=False).encode(),
                           "track_record.csv", "text/csv")

# ---------------------------------------------------------------- Live / What-if
with tabs[3]:
    st.subheader("Update the forecast with results")
    st.markdown(
        "- **Real results:** click **🔄 Refresh data** in the sidebar to pull the latest scores "
        "and goals (from martj42). Elo, player attack-talent, form and coaching all re-derive, and "
        "the simulation conditions on every played match.\n"
        "- **What-if:** edit group scores below and recompute instantly — no data needed. Played "
        "matches are fixed; the rest are simulated.")
    gf = fixtures[fixtures["stage"] == "group"][["group", "home", "away", "home_score", "away_score"]].copy()
    edited = st.data_editor(gf, use_container_width=True, height=300, hide_index=True,
                            disabled=["group", "home", "away"], key="whatif_editor")
    known_items = tuple(sorted(
        (r.home, r.away, int(r.home_score), int(r.away_score))
        for r in edited.itertuples() if pd.notna(r.home_score) and pd.notna(r.away_score)))
    cset = st.columns([1, 3])
    if cset[0].button("Apply & recompute", type="primary"):
        st.session_state["whatif_active"] = known_items
    active = st.session_state.get("whatif_active")
    if active:
        kg_dict_items = tuple(((h, a), (hs, as_)) for (h, a, hs, as_) in active)
        wp, _ = _sim(int(sims), int(seed), kg_dict_items, sup_scale=sup_scale, goal_scale=goal_scale)
        st.caption(f"Conditioned on {len(active)} entered result(s).")
        comp = pd.DataFrame({"baseline": probs["P_champion"], "conditioned": wp["P_champion"]})
        comp["Δ"] = comp["conditioned"] - comp["baseline"]
        comp = comp.sort_values("conditioned", ascending=False).head(12)
        st.dataframe(comp.style.format("{:.1%}"), use_container_width=True)
    else:
        st.caption("Enter at least one score and click **Apply & recompute**.")

# ---------------------------------------------------------------- Model & Weights
with tabs[4]:
    st.subheader("The weights are trained, not chosen")
    st.markdown(
        f"Each factor is computed **as of every match's date** over **{model.n_train:,} "
        "internationals since 2011** (squad skill uses the EA edition active at kickoff). Because "
        "Elo, form, attacking talent, squad skill and coaching overlap, a plain regression would "
        "hand Elo almost everything — so we use **Shapley / LMG relative-importance "
        "decomposition**, the standard way to fairly split shared predictive credit among "
        "correlated factors.")
    wdf = pd.DataFrame({"factor": [FACTOR_LABELS[f] for f in FACTORS],
                        "weight": [model.weights[f] for f in FACTORS],
                        "coef_goals_per_SD": [model.betas[f] for f in FACTORS]}).sort_values("weight", ascending=False)
    cc = st.columns([2, 1])
    cc[0].altair_chart(alt.Chart(wdf).mark_bar().encode(
        x=alt.X("weight:Q", axis=alt.Axis(format="%"), title="Learned weight"),
        y=alt.Y("factor:N", sort="-x", title=None), color=alt.Color("factor:N", legend=None),
        tooltip=["factor", alt.Tooltip("weight:Q", format=".1%")]).properties(height=300),
        use_container_width=True)
    cc[1].dataframe(wdf.set_index("factor").style.format({"weight": "{:.1%}", "coef_goals_per_SD": "{:+.3f}"}),
                    use_container_width=True)
    mm = model.metrics
    st.subheader("Validation — does player + coaching data beat Elo alone?")
    k = st.columns(4)
    k[0].metric("Full model accuracy", pct(mm["full_accuracy"]), f"vs {pct(mm['elo_only_accuracy'])} Elo-only")
    k[1].metric("Full model log-loss", f"{mm['full_logloss']:.3f}", f"{mm['full_logloss']-mm['elo_only_logloss']:+.3f}",
                delta_color="inverse")
    k[2].metric("Goal-diff R²", f"{mm['composite_r2_goal_diff']:.3f}")
    k[3].metric("Home/host edge", f"{model.beta_home:+.2f} goals")

# ---------------------------------------------------------------- Power Ratings
with tabs[5]:
    st.subheader("Composite power rating & factor attribution")
    show = power[["rank", "group", "power_index"] + FACTORS].copy()
    st.dataframe(show.style.format({"power_index": "{:.1f}", "elo": "{:.0f}", "attack_talent": "{:.1f}",
                 "wc_player": "{:.1f}", "form": "{:.2f}", "skill": "{:.1f}", "coach": "{:+.3f}"}),
                 use_container_width=True, height=430)
    st.subheader("What drives each team's rating (top 20)")
    top = power.head(20)
    rows = [{"team": t, "factor": FACTOR_LABELS[f], "contribution": r[f"contrib_{f}"]}
            for t, r in top.iterrows() for f in FACTORS]
    st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
        x=alt.X("contribution:Q", title="Contribution to rating (goals of supremacy vs avg team)"),
        y=alt.Y("team:N", sort=list(top.index), title=None), color=alt.Color("factor:N", title="Factor"),
        tooltip=["team", "factor", alt.Tooltip("contribution:Q", format="+.2f")]).properties(height=560),
        use_container_width=True)

# ---------------------------------------------------------------- Groups
with tabs[6]:
    st.subheader("Group-stage qualification probabilities")
    g = st.selectbox("Group", sorted(power["group"].unique()), key="grp")
    sub = probs.join(power["group"]).query("group == @g").copy()
    sub["P_advance (top 2)"] = sub["P_group_winner"] + sub["P_group_runnerup"]
    sub = sub.sort_values("P_advance (top 2)", ascending=False)
    view = sub[["P_group_winner", "P_group_runnerup", "P_advance (top 2)", "P_reach_R16", "P_reach_QF"]]
    view.columns = ["Win group", "Runner-up", "Advance (top 2)", "Reach R16", "Reach QF"]
    st.dataframe(view.style.format("{:.1%}"), use_container_width=True)
    st.caption("Top 2 of each group + the 8 best third-placed teams reach the Round of 32.")

# ---------------------------------------------------------------- Knockouts
with tabs[7]:
    st.subheader("Stage-by-stage advancement probability")
    heat = probs.sort_values("P_champion", ascending=False).head(24)[STAGES].copy()
    heat.columns = STAGE_LABELS
    st.dataframe(heat.style.format("{:.0%}").background_gradient(cmap="YlOrRd", axis=None),
                 use_container_width=True, height=560)
    mlf = chalk["most_likely_final"]
    st.subheader("Bracket-aware predicted final")
    st.write(f"**Final:** {mlf['home']} vs {mlf['away']}  →  🏆 **{mlf['champion']}**")
    st.caption(f"Each half's modal finalist: {mlf['home']} reaches the final from the top half in "
               f"{pct(mlf['p_home_reaches_final'])} of sims, {mlf['away']} from the bottom half in "
               f"{pct(mlf['p_away_reaches_final'])}. Because the two come from opposite halves of "
               "the draw, this is always a valid matchup and the champion is one of the two.")
    st.divider()
    st.subheader("Single most-likely path (chalk — stronger team always advances)")
    for label, key in [("Semi-finals", "semi_finals"), ("Final", "final")]:
        for mm2 in chalk["rounds"][key]:
            st.write(f"**{label}:** {mm2[0]} vs {mm2[1]}  →  **{mm2[2]}**")
    st.caption(f"Deterministic illustrative path; champion **{chalk['champion']}**.")

# ---------------------------------------------------------------- Players
with tabs[8]:
    st.subheader("Player attacking data (real goal-by-goal records)")
    st.caption("Attack-talent weights each goal by the scorer's proven track record, recency-decayed. "
               "Player *value* is never used — a cheap prolific scorer rates highly.")
    t = st.selectbox("Team", list(power.index), key="plteam")
    dn = teams.loc[t, "data_name"]
    a, b, c = st.columns(3)
    a.metric("Attack-talent rank", f"#{int(power['attack_talent'].rank(ascending=False)[t])} / 48")
    b.metric("WC-pedigree rank", f"#{int(power['wc_player'].rank(ascending=False)[t])} / 48")
    c.metric("Power Index", f"{power.loc[t, 'power_index']:.1f}")
    ts = top_scorers(goals, dn, since="2022-01-01", n=8)
    if len(ts):
        ts = ts.reset_index().rename(columns={"scorer": "Player", "goals": "Goals (since 2022)",
                                              "wc_goals": "WC goals"})
        st.dataframe(ts, use_container_width=True, hide_index=True)
    else:
        st.info("No recent goal records found for this team.")

# ---------------------------------------------------------------- Squad
with tabs[9]:
    st.subheader("Squads (official 26-man lists)")
    if squads_info:
        st.caption(f"Squad skill is built from the official {squads_info.get('source','')} 26-man "
                   f"squads (updated {squads_info.get('fetched_at','?')}), each player matched to his "
                   "EA rating — so only players actually selected count toward a team's strength.")
    else:
        st.info("Official squads not loaded — using each nation's top-rated EA players as a "
                "fallback. Click **🔄 Refresh data** to fetch the official squads.")
    st.caption("Squad skill (a learned model factor) is a rank-weighted mean of these players' EA "
               "overall ratings — covering defence, midfield and goalkeeping that goal data can't see.")

    head = st.columns([2, 1])
    team = head[0].selectbox("Team", list(power.index), key="squadteam")
    dn = teams.loc[team, "data_name"]
    head[1].metric("Squad-skill rank", f"#{int(power['skill'].rank(ascending=False)[team])} / 48")
    sq = squads.get(dn)

    if sq is None or not len(sq):
        st.info("No squad data for this team.")
    else:
        official = bool(sq["callup"].iloc[0])
        n_fringe = int((sq["overall"] == DEFAULT_OVERALL).sum())
        mc = st.columns(3)
        mc[0].metric("Squad skill", f"{squad_rating(sq['overall'].to_numpy()):.1f}")
        mc[1].metric("Players", f"{len(sq)}")
        mc[2].metric("Source", "official" if official else "EA top-rated")
        view = sq.rename(columns={"player": "Player", "pos": "Pos"})
        view["EA overall"] = view["overall"].map(
            lambda v: "— (fringe)" if v == DEFAULT_OVERALL else f"{v:.0f}")
        st.dataframe(view[["Player", "EA overall", "Pos"]],
                     use_container_width=True, hide_index=True, height=380)
        if n_fringe:
            st.caption(f"{n_fringe} selected player(s) aren't in the EA database — counted at a "
                       "fringe baseline so they don't inflate the rating.")

# ---------------------------------------------------------------- Managers
with tabs[10]:
    st.subheader("Manager / coaching factor (data-derived)")
    st.caption("No subjective rating: the coaching factor is each team's results over/under-performance "
               "versus Elo expectation, measured under the current manager's tenure.")
    mrows = []
    for t in power.index:
        info = managers["managers"].get(t, {})
        mrows.append({"Team": t, "Manager": info.get("manager", "—"), "Appointed": info.get("appointed", "—"),
                      "Career PPG": info.get("career_ppg"), "Coaching effect (z)": power.loc[t, "z_coach"],
                      "Contribution (goals)": power.loc[t, "contrib_coach"]})
    mdf = pd.DataFrame(mrows).set_index("Team")
    st.dataframe(mdf.style.format({"Coaching effect (z)": "{:+.2f}", "Contribution (goals)": "{:+.3f}",
                 "Career PPG": "{:.2f}"}, na_rep="—"), use_container_width=True, height=520)

# ---------------------------------------------------------------- Data
with tabs[11]:
    st.subheader("All probabilities")
    st.dataframe(probs.style.format("{:.1%}"), use_container_width=True, height=520)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ predictions.csv", probs.to_csv().encode(), "predictions.csv", "text/csv")
    c2.download_button("⬇ strength_ratings.csv", power.to_csv().encode(), "strength_ratings.csv", "text/csv")
