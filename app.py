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

from src.data import fetch_data
from src.features.players import top_scorers
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
def _sim(sims: int, seed: int, known_items=None):
    prep = _prep()
    kg = dict(known_items) if known_items else None
    probs, chalk = run_simulation(prep, sims=sims, seed=seed, known_group=kg)
    return probs, chalk


def pct(x):
    return f"{x*100:.1f}%"


prep = _prep()
teams, model, power, goals, managers, results, fixtures = (
    prep["teams"], prep["model"], prep["power"], prep["goals"],
    prep["managers"], prep["results"], prep["fixtures"])

st.title("🏆 FIFA World Cup 2026 — Explainable Forecast")
st.caption(f"48 teams · weights LEARNED from {model.n_train:,} matches · "
           f"data through {results.date.max().date()} · player factors from real goal data")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh data (pull latest scores)", use_container_width=True):
        with st.spinner("Downloading latest results & goals…"):
            fetch_data.fetch_all(force=True)
        _prep.clear(); _sim.clear()
        st.rerun()
    sims = st.select_slider("Simulations", [5000, 10000, 20000, 30000, 50000], value=20000)
    seed = st.number_input("Random seed", value=2026, step=1)
    n_played = int(fixtures["played"].sum())
    st.caption(f"{n_played} of {len(fixtures)} WC-2026 matches played so far "
               "(forecast auto-conditions on real results).")
    st.divider()
    st.subheader("Learned weights")
    for f in sorted(FACTORS, key=lambda f: -model.weights[f]):
        st.progress(model.weights[f], text=f"{FACTOR_LABELS[f]} — {pct(model.weights[f])}")

probs, chalk = _sim(int(sims), int(seed))

tabs = st.tabs(["🏆 Overview", "🔮 Match Predictor", "📡 Live / What-if", "📊 Model & Weights",
                "💪 Power Ratings", "🅰️ Groups", "🗺️ Knockouts", "⚽ Players",
                "🧑‍💼 Managers", "📥 Data"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Most likely champion", probs.index[0], pct(probs.iloc[0]["P_champion"]))
    fin = chalk["rounds"]["final"][0]
    c2.metric("Predicted final (chalk)", f"{fin[0]} vs {fin[1]}")
    c3.metric("Predicted winner (chalk)", fin[2])
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
        pr = predict_match(power, model, home, away)
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
    pf = predict_fixtures(power, model, fixtures, only_unplayed=only_unplayed)
    if gsel != "All":
        pf = pf[pf["group"] == gsel]
    st.dataframe(pf.style.format({"P(home win)": "{:.0%}", "P(draw)": "{:.0%}", "P(away win)": "{:.0%}"}),
                 use_container_width=True, hide_index=True, height=380)

# ---------------------------------------------------------------- Live / What-if
with tabs[2]:
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
        wp, _ = _sim(int(sims), int(seed), kg_dict_items)
        st.caption(f"Conditioned on {len(active)} entered result(s).")
        comp = pd.DataFrame({"baseline": probs["P_champion"], "conditioned": wp["P_champion"]})
        comp["Δ"] = comp["conditioned"] - comp["baseline"]
        comp = comp.sort_values("conditioned", ascending=False).head(12)
        st.dataframe(comp.style.format("{:.1%}"), use_container_width=True)
    else:
        st.caption("Enter at least one score and click **Apply & recompute**.")

# ---------------------------------------------------------------- Model & Weights
with tabs[3]:
    st.subheader("The weights are trained, not chosen")
    st.markdown(
        f"Each factor is computed **as of every match's date** over **{model.n_train:,} "
        "internationals since 2011**. Because Elo, form, attacking talent and coaching overlap, a "
        "plain regression would hand Elo almost everything — so we use **Shapley / LMG "
        "relative-importance decomposition**, the standard way to fairly split shared predictive "
        "credit among correlated factors.")
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
with tabs[4]:
    st.subheader("Composite power rating & factor attribution")
    show = power[["rank", "group", "power_index"] + FACTORS].copy()
    st.dataframe(show.style.format({"power_index": "{:.1f}", "elo": "{:.0f}", "attack_talent": "{:.1f}",
                 "wc_player": "{:.1f}", "form": "{:.2f}", "coach": "{:+.3f}"}),
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
with tabs[5]:
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
with tabs[6]:
    st.subheader("Stage-by-stage advancement probability")
    heat = probs.sort_values("P_champion", ascending=False).head(24)[STAGES].copy()
    heat.columns = STAGE_LABELS
    st.dataframe(heat.style.format("{:.0%}").background_gradient(cmap="YlOrRd", axis=None),
                 use_container_width=True, height=560)
    st.subheader("Predicted most-likely path (chalk bracket)")
    for label, key in [("Semi-finals", "semi_finals"), ("Final", "final")]:
        for mm2 in chalk["rounds"][key]:
            st.write(f"**{label}:** {mm2[0]} vs {mm2[1]}  →  **{mm2[2]}**")
    st.success(f"🏆 Predicted champion: **{chalk['champion']}**")

# ---------------------------------------------------------------- Players
with tabs[7]:
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

# ---------------------------------------------------------------- Managers
with tabs[8]:
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
with tabs[9]:
    st.subheader("All probabilities")
    st.dataframe(probs.style.format("{:.1%}"), use_container_width=True, height=520)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ predictions.csv", probs.to_csv().encode(), "predictions.csv", "text/csv")
    c2.download_button("⬇ strength_ratings.csv", power.to_csv().encode(), "strength_ratings.csv", "text/csv")
