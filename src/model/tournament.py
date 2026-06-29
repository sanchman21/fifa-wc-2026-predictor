"""Vectorised Monte-Carlo simulation of the whole 48-team tournament.

All `n_sims` tournaments are simulated simultaneously with numpy arrays, so 20k+ runs
finish in seconds. The pipeline mirrors the real competition exactly:
  group stage (round-robin, FIFA tiebreakers) -> best-8 third-placed allocation ->
  Round of 32 -> R16 -> QF -> SF -> Final, following config/bracket_2026.json.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .match_model import expected_goals

# Within-group round-robin pairings (local team indices 0..3).
_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# The 8 Round-of-32 slots filled by a 3rd-placed team, with their eligible groups.
# Order is fixed and reused throughout (matches the 'away' side of these R32 ties).
_THIRD_SLOTS = [
    (74, set("ABCDF")),
    (77, set("CDFGH")),
    (79, set("CEFHI")),
    (80, set("EHIJK")),
    (81, set("BEFIJ")),
    (82, set("AEHIJ")),
    (85, set("EFGIJ")),
    (87, set("DEIJL")),
]


def _bipartite_match(letters, slots_eligible):
    """Return a list assigning one letter to each slot (slot order), or None.

    letters: the 8 qualifying group letters. slots_eligible: list of eligible-letter sets.
    """
    assign = [None] * len(slots_eligible)
    used = set()

    def bt(k):
        if k == len(slots_eligible):
            return True
        for L in letters:
            if L in used or L not in slots_eligible[k]:
                continue
            used.add(L)
            assign[k] = L
            if bt(k + 1):
                return True
            used.remove(L)
            assign[k] = None
        return False

    return assign if bt(0) else None


def _build_allocation_table(group_letters):
    """Precompute, for every possible set of 8 qualifying 3rd-place groups, the
    slot->group letter assignment. Keyed by a 12-bit integer mask for fast lookup."""
    idx = {L: i for i, L in enumerate(group_letters)}
    elig = [s for _, s in _THIRD_SLOTS]
    table = {}
    for combo in combinations(group_letters, 8):
        assign = _bipartite_match(list(combo), elig)
        if assign is None:        # should never happen with FIFA's eligibility design
            continue
        mask = 0
        for L in combo:
            mask |= 1 << idx[L]
        table[mask] = [idx[L] for L in assign]   # letter-index per slot, in _THIRD_SLOTS order
    return table


class TournamentSimulator:
    def __init__(self, power_df: pd.DataFrame, teams: pd.DataFrame, bracket: dict,
                 elo_per_goal: float, total_goals: float, host_adv: float,
                 rho: float = -0.12, sup_scale: float = 1.0):
        # Ratings are in goal-supremacy units (from the trained model), so elo_per_goal=1.0
        # and host_adv is the learned home/host edge in goals. Drawn knockout ties are decided
        # by the teams' penalty-skill ratings (see _play_knockout / self.pen_skill).
        # sup_scale (from the model's own track record) stretches/compresses every rating GAP
        # around the mean to self-correct over/under-confidence, leaving the host edge intact.
        self.bracket = bracket
        self.elo_per_goal = elo_per_goal
        self.total_goals = total_goals
        self.host_adv = host_adv

        self.team_names = list(power_df.index)
        self.idx = {t: i for i, t in enumerate(self.team_names)}
        R = power_df["power_score"].to_numpy(float)
        self.R = R.mean() + sup_scale * (R - R.mean())   # scale gaps, not absolute level
        self.host = teams.reindex(self.team_names)["host"].fillna(False).to_numpy(bool)
        # Penalty-shootout skill (log-odds; 0 = average). Drawn knockout ties are decided by
        # who is better at penalties, not by open-play strength. Absent column -> neutral coin.
        self.pen_skill = (power_df["pen_skill"].to_numpy(float) if "pen_skill" in power_df.columns
                          else np.zeros(len(self.team_names)))

        # group letter -> ordered team indices (by position A1..A4)
        tdf = teams.reindex(self.team_names).copy()
        tdf["gidx"] = [self.idx[t] for t in tdf.index]
        self.group_letters = sorted(tdf["group"].unique())
        self.gpos = {g: i for i, g in enumerate(self.group_letters)}
        self.group_members = {
            g: tdf[tdf.group == g].sort_values("position")["gidx"].to_numpy()
            for g in self.group_letters
        }
        self.alloc_table = _build_allocation_table(self.group_letters)

    # -- single vectorised match (knockout: returns winner idx; resolves draws on pens) --
    def _play_knockout(self, home_idx, away_idx, rng):
        rh, ra = self.R[home_idx], self.R[away_idx]
        net = self.host_adv * (self.host[home_idx].astype(float) - self.host[away_idx].astype(float))
        la, lb = expected_goals(rh, ra, self.elo_per_goal, self.total_goals, net)
        gh = rng.poisson(la)
        ga = rng.poisson(lb)
        # Draw after regulation/ET -> penalties, decided by each team's penalty SKILL (not its
        # open-play rating): Bradley-Terry coin on the log-odds shootout ratings.
        p_home = 1.0 / (1.0 + np.exp(-(self.pen_skill[home_idx] - self.pen_skill[away_idx])))
        pen_home = rng.random(len(home_idx)) < p_home
        return np.where(gh > ga, home_idx,
                        np.where(ga > gh, away_idx,
                                 np.where(pen_home, home_idx, away_idx)))

    def run(self, n_sims=20000, seed=2026, known_group=None, forced_knockout=None):
        """known_group: {(home,away):(hs,as)} played group matches to FIX.
        forced_knockout: {match_id: winning_team_name} decided knockout ties to LOCK."""
        rng = np.random.default_rng(seed)
        n = n_sims
        nteams = len(self.team_names)

        # normalize known group results to both orientations, keyed by (name_i, name_j)
        known = {}
        for (h, a), (hs, as_) in (known_group or {}).items():
            known[(h, a)] = (hs, as_); known[(a, h)] = (as_, hs)
        # decided knockout ties: (teamA, teamB) -> winner, matched by team identity wherever
        # those two teams meet in the bracket (no match-id needed)
        forced_pairs = [(self.idx[x], self.idx[y], self.idx[w])
                        for (x, y), w in (forced_knockout or {}).items()
                        if x in self.idx and y in self.idx and w in self.idx]

        winners = {}   # group letter -> (n,) global idx
        runners = {}
        thirds = {}
        third_metric = np.empty((len(self.group_letters), n))

        # ---------------- group stage ----------------
        for g in self.group_letters:
            t = self.group_members[g]                  # 4 global idx
            pts = np.zeros((4, n)); gd = np.zeros((4, n)); gf = np.zeros((4, n))
            for (a, b) in _PAIRS:
                ia, ib = t[a], t[b]
                key = (self.team_names[ia], self.team_names[ib])
                if key in known:                       # FIX a played match (constant scores)
                    sa, sb = known[key]
                    ga = np.full(n, sa); gb = np.full(n, sb)
                else:
                    net = self.host_adv * (float(self.host[ia]) - float(self.host[ib]))
                    la, lb = expected_goals(self.R[ia], self.R[ib], self.elo_per_goal,
                                            self.total_goals, net)
                    ga = rng.poisson(la, n); gb = rng.poisson(lb, n)
                # points
                pts[a] += np.where(ga > gb, 3, np.where(ga == gb, 1, 0))
                pts[b] += np.where(gb > ga, 3, np.where(gb == ga, 1, 0))
                gf[a] += ga; gf[b] += gb
                gd[a] += ga - gb; gd[b] += gb - ga

            # FIFA order: points, goal difference, goals for, then drawing of lots (noise)
            score = pts * 1e7 + (gd + 1000) * 1e3 + gf + rng.random((4, n)) * 1e-3
            order = np.argsort(-score, axis=0)          # (4, n): row0 = best local idx
            tt = np.asarray(t)
            gi = self.gpos[g]
            winners[g] = tt[order[0]]
            runners[g] = tt[order[1]]
            thirds[g] = tt[order[2]]
            third_local = order[2]
            # metric for ranking thirds across groups (no winner noise dependence beyond ties)
            third_metric[gi] = np.take_along_axis(
                pts * 1e7 + (gd + 1000) * 1e3 + gf, third_local[None, :], axis=0
            )[0] + rng.random(n) * 1e-3

        # ---------------- best 8 third-placed + slot allocation ----------------
        order_thirds = np.argsort(-third_metric, axis=0)     # (12, n) group rows ranked
        qual_rows = order_thirds[:8]                         # (8, n) qualifying group indices
        mask = np.zeros(n, dtype=np.int64)
        for r in range(8):
            mask |= (1 << qual_rows[r].astype(np.int64))

        thirds_stack = np.stack([thirds[g] for g in self.group_letters])  # (12, n) global idx
        slot_letter = np.empty((8, n), dtype=np.int64)        # letter-index feeding each 3rd slot
        for m in np.unique(mask):
            cols = np.where(mask == m)[0]
            alloc = self.alloc_table.get(int(m))
            if alloc is None:                                 # safety fallback (shouldn't trigger)
                alloc = sorted({int(x) for x in qual_rows[:, cols[0]]})[:8]
            for k in range(8):
                slot_letter[k, cols] = alloc[k]
        ar = np.arange(n)
        third_for_slot = {self._slot_id(k): thirds_stack[slot_letter[k], ar] for k in range(8)}

        # qualified-from-group flag (reached R32) per team
        reach = {s: np.zeros(nteams) for s in
                 ["R32", "R16", "QF", "SF", "Final", "Champion"]}
        grp_winner_cnt = np.zeros(nteams)
        grp_runner_cnt = np.zeros(nteams)
        for g in self.group_letters:
            np.add.at(reach["R32"], winners[g], 1)
            np.add.at(reach["R32"], runners[g], 1)
            np.add.at(grp_winner_cnt, winners[g], 1)
            np.add.at(grp_runner_cnt, runners[g], 1)
        for k in range(8):
            np.add.at(reach["R32"], third_for_slot[self._slot_id(k)], 1)

        # ---------------- knockouts ----------------
        slot_team = {}
        for g in self.group_letters:
            slot_team[f"1{g}"] = winners[g]
            slot_team[f"2{g}"] = runners[g]

        match_winner = {}

        def resolve_side(code):
            if code.startswith("W"):
                return match_winner[int(code[1:])]
            if code.startswith("3:"):
                return None  # handled via third_for_slot keyed by match id
            return slot_team[code]

        def _apply_forced(home, away, w):
            for a, b, fw in forced_pairs:              # lock a decided tie wherever it occurs
                w = np.where(((home == a) & (away == b)) | ((home == b) & (away == a)), fw, w)
            return w

        # Round of 32
        for m in self.bracket["round_of_32"]:
            home = resolve_side(m["home"])
            away = (third_for_slot[m["match"]] if str(m["away"]).startswith("3:")
                    else resolve_side(m["away"]))
            w = _apply_forced(home, away, self._play_knockout(home, away, rng))
            match_winner[m["match"]] = w
            np.add.at(reach["R16"], w, 1)

        for rnd, key in [("round_of_16", "QF"), ("quarter_finals", "SF"),
                         ("semi_finals", "Final"), ("final", "Champion")]:
            for m in self.bracket[rnd]:
                home = resolve_side(m["home"])
                away = resolve_side(m["away"])
                w = _apply_forced(home, away, self._play_knockout(home, away, rng))
                match_winner[m["match"]] = w
                np.add.at(reach[key], w, 1)

        # Record, per simulation, which team emerged from EACH bracket half to reach the
        # final, plus the eventual champion. The two semi-final winners feeding the final
        # (W101 / W102) come from opposite halves of the draw by construction, so the modal
        # team in each is a bracket-VALID finalist — used for a coherent headline final whose
        # winner is necessarily one of the two finalists (see most_likely_final()).
        final_m = self.bracket["final"][0]
        self._finalist_top = match_winner[int(str(final_m["home"])[1:])]
        self._finalist_bottom = match_winner[int(str(final_m["away"])[1:])]
        self._champion = match_winner[final_m["match"]]

        # ---------------- assemble probability table ----------------
        out = pd.DataFrame(index=self.team_names)
        out["P_group_winner"] = grp_winner_cnt / n
        out["P_group_runnerup"] = grp_runner_cnt / n
        for s in ["R32", "R16", "QF", "SF", "Final", "Champion"]:
            out[f"P_{s}"] = reach[s] / n
        out = out.rename(columns={
            "P_R32": "P_reach_R32", "P_R16": "P_reach_R16", "P_QF": "P_reach_QF",
            "P_SF": "P_reach_SF", "P_Final": "P_reach_Final", "P_Champion": "P_champion"})
        out.index.name = "team"
        return out.sort_values("P_champion", ascending=False)

    def most_likely_final(self):
        """Bracket-aware predicted final, derived from the Monte-Carlo run.

        The final pits the two semi-finals' modal winners against each other. Because those
        two semi-finals are fed by opposite halves of the draw, the matchup is always a VALID
        bracket final (two teams that really can meet there — never two same-half teams), and
        the predicted champion is, by construction, one of the two finalists. Call after run().
        """
        if getattr(self, "_champion", None) is None:
            raise RuntimeError("most_likely_final() requires a completed run()")
        nteams = len(self.team_names)
        n = len(self._champion)
        top = int(np.bincount(self._finalist_top, minlength=nteams).argmax())
        bottom = int(np.bincount(self._finalist_bottom, minlength=nteams).argmax())
        champ_cnt = np.bincount(self._champion, minlength=nteams)
        # The more likely title winner of the two modal finalists.
        champ = top if champ_cnt[top] >= champ_cnt[bottom] else bottom
        return {
            "home": self.team_names[top],
            "away": self.team_names[bottom],
            "champion": self.team_names[champ],
            "p_home_reaches_final": float((self._finalist_top == top).mean()),
            "p_away_reaches_final": float((self._finalist_bottom == bottom).mean()),
            "p_this_exact_final": float(((self._finalist_top == top)
                                         & (self._finalist_bottom == bottom)).mean()),
            "home_p_champion": float(champ_cnt[top] / n),
            "away_p_champion": float(champ_cnt[bottom] / n),
        }

    @staticmethod
    def _slot_id(k):
        return _THIRD_SLOTS[k][0]

    def _eff(self, i):
        return self.R[i] + self.host_adv * float(self.host[i])

    def chalk_bracket(self):
        """Deterministic single most-likely bracket: rank groups by power rating and
        always advance the stronger side. Used for the narrative path in the report."""
        winners, runners, thirds = {}, {}, {}
        third_strength = {}
        groups_out = {}
        for g in self.group_letters:
            t = list(self.group_members[g])
            ranked = sorted(t, key=lambda i: self.R[i], reverse=True)
            winners[g], runners[g], thirds[g] = ranked[0], ranked[1], ranked[2]
            third_strength[g] = self.R[ranked[2]]
            groups_out[g] = [self.team_names[i] for i in ranked]

        best_third_groups = sorted(self.group_letters,
                                   key=lambda g: third_strength[g], reverse=True)[:8]
        elig = [s for _, s in _THIRD_SLOTS]
        assign = _bipartite_match(best_third_groups, elig)
        third_for_slot = {self._slot_id(k): thirds[assign[k]] for k in range(8)}

        slot_team = {}
        for g in self.group_letters:
            slot_team[f"1{g}"] = winners[g]
            slot_team[f"2{g}"] = runners[g]
        match_winner = {}

        def side(code, match_id=None, is_third=False):
            if is_third:
                return third_for_slot[match_id]
            if code.startswith("W"):
                return match_winner[int(code[1:])]
            return slot_team[code]

        rounds_out = {}
        for m in self.bracket["round_of_32"]:
            h = side(m["home"])
            a = side(m["away"], m["match"], str(m["away"]).startswith("3:"))
            match_winner[m["match"]] = h if self._eff(h) >= self._eff(a) else a
        for rnd in ["round_of_16", "quarter_finals", "semi_finals", "final"]:
            res = []
            for m in self.bracket[rnd]:
                h, a = side(m["home"]), side(m["away"])
                w = h if self._eff(h) >= self._eff(a) else a
                match_winner[m["match"]] = w
                res.append((self.team_names[h], self.team_names[a], self.team_names[w]))
            rounds_out[rnd] = res

        final_match = self.bracket["final"][0]["match"]
        champion = self.team_names[match_winner[final_match]]
        return {"groups": groups_out, "rounds": rounds_out, "champion": champion}
