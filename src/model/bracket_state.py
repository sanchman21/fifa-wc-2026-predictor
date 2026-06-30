"""The CURRENT knockout bracket from REAL results — no predictions.

Fills the official bracket (config/bracket_2026.json) from what has actually happened:
group winners / runners-up come from the real group standings, and every PLAYED knockout
result is advanced up the tree. Slots that aren't decided yet (group incomplete, or a feeder
tie not yet played) are returned as ``None`` so the UI can show them as "TBD".

Third-placed R32 opponents are read from the ACTUAL fixtures rather than computed: FIFA's
third-place→slot allocation is a fixed lookup table this project only approximates, so the
honest way to fill those slots is to observe who each group winner actually played (a group
winner's *earliest* knockout match is, by construction, its Round-of-32 tie). Until that tie
is played the slot stays a labelled placeholder (e.g. "3rd: C/D/F/G/H"). This is purely a
view of the fixtures — the Monte-Carlo forecast lives elsewhere.
"""
from __future__ import annotations

import pandas as pd

_ROUNDS = ["round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"]


def _standings(fixtures: pd.DataFrame) -> dict:
    """{letter: [1st, 2nd, 3rd, 4th] display names} for COMPLETE groups, else [].

    Ranking uses the simulator's tiebreakers (points, goal difference, goals for); the team
    name is a final deterministic tiebreak. Incomplete groups are left empty so we never
    place a team off a half-played table.
    """
    g = fixtures[fixtures["stage"] == "group"]
    groups: dict[str, list] = {}
    for letter, sub in g.groupby("group"):
        members = list(pd.unique(pd.concat([sub["home"], sub["away"]])))
        if not (len(members) == 4 and len(sub) == 6 and bool(sub["played"].all())):
            groups[letter] = []
            continue
        pts = {t: 0 for t in members}; gd = {t: 0 for t in members}; gf = {t: 0 for t in members}
        for r in sub.itertuples():
            hs, as_ = int(r.home_score), int(r.away_score)
            gf[r.home] += hs; gf[r.away] += as_
            gd[r.home] += hs - as_; gd[r.away] += as_ - hs
            if hs > as_:
                pts[r.home] += 3
            elif as_ > hs:
                pts[r.away] += 3
            else:
                pts[r.home] += 1; pts[r.away] += 1
        groups[letter] = sorted(members, key=lambda t: (pts[t], gd[t], gf[t], t), reverse=True)
    return groups


def _third_label(code: str) -> str:
    """'3:C/D/F/G/H' -> '3rd: C/D/F/G/H' for an undecided third-placed slot."""
    return "3rd: " + code[2:]


def actual_bracket(fixtures: pd.DataFrame, bracket: dict, known_knockout: dict) -> dict:
    """Resolve every knockout match id to its current state.

    Returns {match_id: {home, away, home_label, away_label, home_score, away_score,
    winner, played, pens}}. `home`/`away`/`winner` are display names or None (TBD); the
    `*_label` fields carry a helpful placeholder for an undecided side. `known_knockout` is
    the {(home, away): winner} map of decided ties (penalty shootouts already resolved), so
    a drawn-then-pens tie still advances the correct side up the bracket.
    """
    groups = _standings(fixtures)
    slot_team = {}
    for L, ranked in groups.items():
        if ranked:
            slot_team[f"1{L}"] = ranked[0]
            slot_team[f"2{L}"] = ranked[1]

    # Played knockout fixtures, consumed as we match them so a team's R32 tie is never reused
    # for a later round. Sorted by date so a group winner's EARLIEST tie is its Round of 32.
    ko = fixtures[(fixtures["stage"] == "knockout") & fixtures["played"]].sort_values("date")
    remaining = [{"home": r.home, "away": r.away, "hs": int(r.home_score),
                  "as": int(r.away_score), "date": r.date, "pair": frozenset({r.home, r.away})}
                 for r in ko.itertuples()]
    winner_by_pair = {frozenset(pair): w for pair, w in known_knockout.items()}
    match_winner: dict[int, str] = {}   # match id -> advancing team (once the tie is decided)

    def take_by_pair(pair):
        for i, f in enumerate(remaining):
            if f["pair"] == pair:
                return remaining.pop(i)
        return None

    def take_earliest_with(team):
        cands = [i for i, f in enumerate(remaining) if team in f["pair"]]
        return remaining.pop(min(cands, key=lambda i: remaining[i]["date"])) if cands else None

    def resolve(code, mid):
        if code.startswith("W"):
            return match_winner.get(int(code[1:]))
        if code.startswith("3:"):
            return None                       # discovered from the actual fixture below
        return slot_team.get(code)

    state = {}
    for rnd in _ROUNDS:
        for m in bracket[rnd]:
            mid = m["match"]; hc, ac = str(m["home"]), str(m["away"])
            home, away = resolve(hc, mid), resolve(ac, mid)
            info = {"home": home, "away": away,
                    "home_label": None if home else ("TBD" if not hc.startswith("3:") else _third_label(hc)),
                    "away_label": None if away else ("TBD" if not ac.startswith("3:") else _third_label(ac)),
                    "home_score": None, "away_score": None,
                    "winner": None, "played": False, "pens": False}
            f = None
            if home and away:
                f = take_by_pair(frozenset({home, away}))
            elif home and ac.startswith("3:"):          # 1X vs a third — find who they played
                f = take_earliest_with(home)
                if f is not None:
                    away = f["away"] if f["home"] == home else f["home"]
                    info["away"], info["away_label"] = away, None
            if f is not None:
                hs, as_ = (f["hs"], f["as"]) if f["home"] == info["home"] else (f["as"], f["hs"])
                info["home_score"], info["away_score"] = hs, as_
                info["played"] = True
                info["pens"] = hs == as_
                w = winner_by_pair.get(f["pair"])
                info["winner"] = w
                if w is not None:
                    match_winner[mid] = w
            state[mid] = info
    return state
