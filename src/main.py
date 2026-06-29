"""End-to-end FIFA World Cup 2026 forecast (weights learned from data).

    python -m src.main                # full run (20k sims), writes ./output
    python -m src.main --sims 50000
    python -m src.main --refresh      # re-download source data first

Or explore interactively:  streamlit run app.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time

from .data import fetch_data, fetch_skills, fetch_squads
from .data.config_loader import (DATA_DIR, load_bracket, load_managers, load_results,
                                 load_results_raw, load_shootouts, load_teams)
from .explain import report
from .features.elo import compute_elo
from .features.fixtures import known_group_results, known_knockout_results, wc2026_matches
from .features.players import load_goals
from .features.skills import current_squads, squads_meta
from .model import tracking
from .model.tournament import TournamentSimulator
from .model.train import build_power_table, current_factor_table, train

REF_DATE = "2026-06-11"


def prepare_model(refresh=False, verbose=True):
    """Load data, compute Elo, train weights, build the power table. (Independent of #sims;
    cache this once.) Returns everything except the simulation."""
    def log(msg):
        if verbose:
            print(msg)

    fetch_data.fetch_all(force=refresh)
    fetch_skills.fetch_all(force=refresh)    # EA player ratings for the squad-skill factor
    fetch_squads.fetch_squads(force=refresh) # official 26-man squads (Wikipedia, no auth)
    teams = load_teams(); bracket = load_bracket(); managers = load_managers()
    results = load_results()
    goals = load_goals(results, os.path.join(DATA_DIR, "goalscorers.csv"))
    log(f"[data] {len(teams)} teams, {len(results):,} matches, {len(goals):,} goals "
        f"({results.date.min().date()} -> {results.date.max().date()})")

    elo = compute_elo(results)
    log(f"[elo] computed; calibrated baseline {elo.total_goals:.2f} goals/match")

    appts = {teams.loc[t, "data_name"]: mgr["appointed"]
             for t, mgr in managers["managers"].items() if t in teams.index}
    model = train(elo.pre, goals)
    log(f"[train] learned weights on {model.n_train:,} matches | "
        + " ".join(f"{k}={v:.0%}" for k, v in model.weights.items()))

    # Squad-skill uses the OFFICIAL 26-man squads: current_squads() reads the Wikipedia cache and
    # matches each selected player to his EA rating, so only players actually called up count
    # (current_factor_table -> current_skill_factors -> current_squads). If the cache is absent it
    # falls back to each nation's top-rated EA players.
    squads = current_squads()
    factors = current_factor_table(teams, elo, elo.pre, goals, appts, REF_DATE)
    power = build_power_table(teams, factors, model)
    official = sum(1 for sq in squads.values() if len(sq) and bool(sq["callup"].iloc[0]))
    log(f"[squads] squad-skill from official lists for {official}/{len(teams)} teams")

    fixtures = wc2026_matches(load_results_raw(), teams)
    known = known_group_results(fixtures)
    known_ko = known_knockout_results(fixtures, teams, load_shootouts())
    if known or known_ko:
        log(f"[live] conditioning on {len(known)} group + {len(known_ko)} knockout result(s)")

    # Track record: lock pre-match predictions, grade the ones that have since played, and
    # learn a calibration scale from how we've done so far (heavily shrunk toward no-change).
    # Tracking is a non-essential overlay — never let it take down the whole forecast (e.g. a
    # read-only filesystem or a corrupt ledger), so degrade to empty/default metrics on failure.
    try:
        ledger = tracking.update_ledger(power, model, fixtures,
                                        run_date=_dt.date.today().isoformat())
    except Exception as exc:  # noqa: BLE001 - tracking must never break the forecast
        log(f"[track] skipped (could not update ledger: {exc})")
        ledger = {}
    record = tracking.track_record(ledger)
    calib = tracking.calibration_scale(ledger, model)
    sup_scale = calib["sup_scale"] if calib["applied"] else 1.0
    goal_scale = calib["goal_scale"] if calib["applied"] else 1.0
    if record["n_graded"]:
        log(f"[track] graded {record['n_graded']} played match(es): "
            f"{record['accuracy']:.0%} hit-rate, log-loss {record['logloss']:.3f} "
            f"({record['skill_pct']:+.0f}% skill vs coin-flip); {record['n_pending']} locked & pending")
        log(f"[track] goals {record['goals_actual']:.2f} actual vs {record['goals_pred']:.2f} "
            f"predicted ({record['goals_bias']:+.2f}); draw rate {record['draw_rate_actual']:.0%} "
            f"actual vs {record['draw_rate_pred']:.0%} predicted")
        if calib["applied"]:
            log(f"[track] applying calibration: supremacy ×{sup_scale:.3f}, goals ×{goal_scale:.3f} "
                f"(log-loss {calib['baseline_logloss']:.3f} → {calib['calibrated_logloss']:.3f})")

    return dict(teams=teams, managers=managers, results=results, goals=goals, elo=elo,
                model=model, power=power, bracket=bracket, fixtures=fixtures, appts=appts,
                squads=squads, squads_meta=squads_meta(), known_group=known, known_knockout=known_ko,
                ledger=ledger, track_record=record, calibration=calib,
                sup_scale=sup_scale, goal_scale=goal_scale)


def run_simulation(prep, sims=20000, seed=2026, known_group=None, forced_knockout=None,
                   sup_scale=None, goal_scale=None):
    """Simulate; by default conditions on real played matches (prep['known_*']).
    Pass known_group / forced_knockout explicitly (e.g. what-if scores) to override.
    sup_scale / goal_scale default to the track-record calibration (prep['sup_scale'],
    prep['goal_scale']); pass 1.0 to disable either or any value to override. goal_scale
    rescales the expected goal total feeding every simulated match."""
    kg = prep.get("known_group", {}) if known_group is None else known_group
    ko = prep.get("known_knockout", {}) if forced_knockout is None else forced_knockout
    s = prep.get("sup_scale", 1.0) if sup_scale is None else sup_scale
    g = prep.get("goal_scale", 1.0) if goal_scale is None else goal_scale
    sim = TournamentSimulator(prep["power"], prep["teams"], prep["bracket"], elo_per_goal=1.0,
                              total_goals=prep["model"].total_goals * g, host_adv=prep["model"].beta_home,
                              rho=prep["model"].rho, pen_k=0.4, sup_scale=s)
    probs = sim.run(n_sims=sims, seed=seed, known_group=kg, forced_knockout=ko)
    chalk = sim.chalk_bracket()
    # Bracket-aware headline final/champion (always a valid two-halves matchup, champion ∈ final).
    chalk["most_likely_final"] = sim.most_likely_final()
    return probs, chalk


def build_forecast(sims=20000, seed=2026, refresh=False, verbose=True):
    """Full pipeline; dict of all artifacts."""
    prep = prepare_model(refresh=refresh, verbose=verbose)
    probs, chalk = run_simulation(prep, sims=sims, seed=seed)
    return {**prep, "probs": probs, "chalk": chalk, "sims": sims}


def main(argv=None):
    ap = argparse.ArgumentParser(description="FIFA World Cup 2026 forecast")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    print("=" * 64)
    print(" FIFA WORLD CUP 2026 — FORECAST (data-trained weights)")
    print("=" * 64)
    R = build_forecast(sims=args.sims, seed=args.seed, refresh=args.refresh)

    report.save_csvs(R["power"], R["probs"])
    report.make_charts(R["power"], R["probs"], R["model"])
    report.write_report(R["power"], R["probs"], R["model"], R["chalk"], R["elo"], R["sims"])
    report.write_track_record(R["track_record"], R["calibration"])

    print("-" * 64)
    print("DONE in %.1fs. See output/report.md  (or: streamlit run app.py)" % (time.time() - t0))
    print("-" * 64)
    print("\nLearned factor weights:")
    for f, w in sorted(R["model"].weights.items(), key=lambda kv: -kv[1]):
        print(f"  {f:<16} {w*100:5.1f}%")
    print("\nTitle odds (top 10):")
    for t, p in R["probs"]["P_champion"].head(10).items():
        print(f"  {t:<24} {p*100:5.1f}%")
    print(f"\nChalk final: {R['chalk']['rounds']['final'][0][0]} vs "
          f"{R['chalk']['rounds']['final'][0][1]}  ->  {R['chalk']['champion']}")

    tr = R["track_record"]
    if tr["n_graded"]:
        print(f"\nTrack record ({tr['n_graded']} graded, {tr['n_pending']} pending): "
              f"{tr['accuracy']:.0%} hit-rate, log-loss {tr['logloss']:.3f} "
              f"({tr['skill_pct']:+.0f}% skill). See output/track_record.md")
    else:
        print(f"\nTrack record: {tr['n_pending']} pre-match prediction(s) locked, awaiting results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
