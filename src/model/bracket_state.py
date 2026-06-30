"""The CURRENT knockout bracket from REAL results — no predictions.

Fills the official bracket (config/bracket_2026.json) from what has actually happened:
group winners / runners-up come from the real group standings, and every PLAYED knockout
result is advanced up the tree. Slots that aren't decided yet (group incomplete, or a feeder
tie not yet played) are returned as ``None`` so the UI can show them as "TBD".

Third-placed R32 opponents are resolved in two ways, ground truth first:
  * A PLAYED third-place tie reveals the real opponent directly — we observe who each group
    winner actually played (a group winner's *earliest* knockout match is, by construction,
    its Round-of-32 tie), so a played slot is always exactly right.
  * Once the GROUP STAGE is complete the eight best third-placed teams are known, so any
    third-place tie not yet played is filled with the same constraint-matched allocation the
    Monte-Carlo simulator uses (src/model/tournament.py) and flagged ``projected``. FIFA's
    third-place→slot table is one specific valid allocation this project approximates, so the
    exact slot may differ in rare combinations; the assignment is PINNED to any third already
    revealed by a played tie, so a projected slot never contradicts a real result or shows a
    team twice. Before every group has finished, these slots stay labelled placeholders
    (e.g. "3rd: C/D/F/G/H").

This is purely a view of the fixtures — the Monte-Carlo forecast lives elsewhere.
"""
from __future__ import annotations

import pandas as pd

from .tournament import _THIRD_SLOTS

_ROUNDS = ["round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final"]


def _group_table(fixtures: pd.DataFrame) -> dict:
    """{letter: {"ranked": [1st..4th names], "stats": {team: (pts, gd, gf)}}} for COMPLETE
    groups; incomplete groups are omitted entirely.

    Ranking uses the simulator's tiebreakers (points, goal difference, goals for); the team
    name is a final deterministic tiebreak. Incomplete groups are skipped so we never place a
    team off a half-played table, nor rank third-placed teams before every group has finished.
    """
    g = fixtures[fixtures["stage"] == "group"]
    table: dict[str, dict] = {}
    for letter, sub in g.groupby("group"):
        members = list(pd.unique(pd.concat([sub["home"], sub["away"]])))
        if not (len(members) == 4 and len(sub) == 6 and bool(sub["played"].all())):
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
        ranked = sorted(members, key=lambda t: (pts[t], gd[t], gf[t], t), reverse=True)
        table[letter] = {"ranked": ranked,
                         "stats": {t: (pts[t], gd[t], gf[t]) for t in members}}
    return table


def _third_allocation(groups: dict, discovered: dict) -> dict:
    """Assign the 8 best third-placed teams to the 8 third-place R32 slots.

    `groups` is _group_table's output; `discovered` is {slot_id: third_team} already revealed
    by a PLAYED tie (ground truth). Returns {slot_id: third_team display name} for ALL eight
    third slots, or {} if fewer than 12 groups are complete (the thirds can't be ranked yet)
    or no valid allocation exists.

    Mirrors the simulator (src/model/tournament.py): the eight best third-placed teams (FIFA
    order — points, then goal difference, then goals for) are matched to the eligible slots in
    _THIRD_SLOTS. Slots whose tie has already been played are PINNED to the group that actually
    filled them, so a projected assignment never contradicts a real result or repeats a team.
    """
    if len(groups) < 12:
        return {}
    letters = sorted(groups)
    third = {L: groups[L]["ranked"][2] for L in letters}
    metric = {L: groups[L]["stats"][third[L]] for L in letters}   # (pts, gd, gf), FIFA order
    grp_of = {third[L]: L for L in letters}

    pinned: dict[int, str] = {}        # slot_id -> group letter, fixed by a played tie
    forced: set[str] = set()
    for slot_id, team in discovered.items():
        L = grp_of.get(team)
        if L is not None:
            pinned[slot_id] = L
            forced.add(L)

    # Best 8 third-place groups: any already proven to qualify (its tie was played) plus the
    # next best by FIFA order. Ground truth wins a tiebreak over a marginally-better metric.
    rest = sorted((L for L in letters if L not in forced), key=lambda L: metric[L], reverse=True)
    qualified = (list(forced) + rest)[:8]

    slots = [sid for sid, _ in _THIRD_SLOTS]
    elig = dict(_THIRD_SLOTS)
    assign = dict(pinned)
    used = set(pinned.values())
    free_slots = [s for s in slots if s not in assign]
    free_groups = [L for L in qualified if L not in used]

    def bt(k: int) -> bool:
        if k == len(free_slots):
            return True
        sid = free_slots[k]
        for L in free_groups:
            if L in used or L not in elig[sid]:
                continue
            used.add(L); assign[sid] = L
            if bt(k + 1):
                return True
            used.discard(L); del assign[sid]
        return False

    if not bt(0):
        return {}
    return {sid: third[assign[sid]] for sid in slots}


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
    groups = _group_table(fixtures)
    slot_team = {}
    for L in groups:
        slot_team[f"1{L}"] = groups[L]["ranked"][0]
        slot_team[f"2{L}"] = groups[L]["ranked"][1]

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
                    "winner": None, "played": False, "pens": False, "projected": False}
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

    # ---- Projected third-place opponents (group stage complete, tie not yet played) -------
    # The eight best third-placed teams are known once every group has finished, so fill any
    # third-place slot still awaiting kickoff with the constraint-matched allocation, pinned to
    # the thirds already revealed by played ties. Played slots keep their real opponent.
    discovered = {m["match"]: state[m["match"]]["away"]
                  for m in bracket["round_of_32"]
                  if str(m["away"]).startswith("3:")
                  and state[m["match"]]["played"] and state[m["match"]]["away"] is not None}
    for mid, third in _third_allocation(groups, discovered).items():
        info = state[mid]
        if info["away"] is None:                 # only an unplayed slot still needs filling
            info["away"], info["away_label"], info["projected"] = third, None, True
    return state
