"""Squad-skill factor from EA Sports FC / FIFA player ratings.

Goals data (players.py) only sees *scorers*; it is blind to defenders, holding midfielders
and goalkeepers. EA player ratings cover every position, so this factor measures a nation's
all-round squad quality.

Two data files (see src/data/fetch_skills.py):
  * legacy  : one rating snapshot per edition (FIFA 15..FC24, dated 2014..2023). Used to build
              AS-OF-DATE training features - for a past match we read the edition active then.
  * current : FC26 ratings, used to score each player named in the official 26-man squads
              (src/data/fetch_squads.py) so only selected players count toward 2026 squad skill.

A team's skill = a rank-weighted mean of its strongest ~23 players' overall ratings (stars
weighted more, so a missing/weaker squad visibly lowers it).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
LEGACY = os.path.join(DATA_DIR, "ea_players_legacy.csv")
CURRENT = os.path.join(DATA_DIR, "ea_players_current.csv")
SQUADS_2026 = os.path.join(DATA_DIR, "squads_2026.json")   # official 26-man squads (Wikipedia)

SQUAD_SIZE = 26          # players considered "in the squad" (real call-ups are 26)
RATE_TOP = 23            # rank-weighted mean is taken over the top-23 of the squad
DEFAULT_OVERALL = 62.0   # rating for a selected player with no EA entry (fringe/uncapped)
_USECOLS = ["fifa_version", "update_as_of", "nationality_name", "overall",
            "player_positions", "short_name", "long_name", "nation_team_id", "nation_position"]

# EA spells some nations differently from the match data (config data_name). Map the few that
# differ; everything else matches by identity. Values list every EA spelling seen across editions.
_TEAM_TO_EA = {
    "South Korea": ["Korea Republic"],
    "Czech Republic": ["Czechia", "Czech Republic"],
    "Turkey": ["Türkiye", "Turkey"],
    "Ivory Coast": ["Côte d'Ivoire"],
    "Cape Verde": ["Cabo Verde", "Cape Verde Islands"],
    "DR Congo": ["Congo DR"],
    "Curaçao": ["Curacao"],
    # broader historical aliases (improve training coverage beyond the 48)
    "United States": ["United States"],
    "China PR": ["China PR"],
    "Republic of Ireland": ["Republic of Ireland"],
}


def _pos_bucket(positions: str) -> str:
    """Coarse position group from the (comma-separated) player_positions string."""
    first = str(positions).split(",")[0].strip().upper()
    if first == "GK":
        return "GK"
    if first in {"CB", "LB", "RB", "LWB", "RWB", "LCB", "RCB"}:
        return "DEF"
    if first in {"CDM", "CM", "CAM", "LM", "RM", "LDM", "RDM", "LCM", "RCM", "LAM", "RAM"}:
        return "MID"
    return "FWD"


def squad_rating(overalls) -> float:
    """Rank-weighted mean of the top RATE_TOP overall ratings (descending weights 23..1).
    Used for each team's current squad value and the as-of-date training editions."""
    vals = np.sort(np.asarray(overalls, dtype=float))[::-1][:RATE_TOP]
    if len(vals) == 0:
        return float("nan")
    w = np.arange(len(vals), 0, -1, dtype=float)
    return float(np.dot(vals, w) / w.sum())


def _ea_to_team_map() -> dict:
    """Reverse the alias table: EA nationality_name -> canonical (match/config) team name.
    Identity for any EA name not explicitly aliased is applied later at lookup time."""
    out = {}
    for team, aliases in _TEAM_TO_EA.items():
        for a in aliases:
            out[a] = team
    return out


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in _USECOLS, low_memory=False)
    df = df.dropna(subset=["nationality_name", "overall"])
    df["overall"] = df["overall"].astype(float)
    alias = _ea_to_team_map()
    df["team"] = df["nationality_name"].map(lambda n: alias.get(n, n))  # alias or identity
    return df


# ---- edition (as-of-date) squad ratings, for TRAINING -------------------------------------

def _edition_table() -> pd.DataFrame:
    """Per (edition, team) squad rating, with the edition's snapshot date. One row per pair."""
    df = _load(LEGACY)
    rows = []
    for (ver, team), grp in df.groupby(["fifa_version", "team"]):
        top = grp.nlargest(SQUAD_SIZE, "overall")
        rows.append((float(ver), team, squad_rating(top["overall"].to_numpy())))
    tab = pd.DataFrame(rows, columns=["fifa_version", "team", "skill"])
    dates = (df.groupby("fifa_version")["update_as_of"].first()
             .pipe(pd.to_datetime).rename("edition_date"))
    return tab.merge(dates, on="fifa_version", how="left")


def asof_skill_features(matches: pd.DataFrame) -> pd.DataFrame:
    """For each match, the home & away squad skill from the EA edition active at kickoff.

    Returns columns skill_home / skill_away aligned to `matches.index`. Teams/dates with no
    edition data get NaN (the trainer imputes those to the factor mean -> zero gap). If the
    legacy file is absent (e.g. no Kaggle creds on a cold start), every value is NaN so the
    skill factor simply contributes nothing rather than crashing the pipeline."""
    if not os.path.exists(LEGACY):
        return pd.DataFrame({"skill_home": np.nan, "skill_away": np.nan}, index=matches.index)
    ed = _edition_table()
    edition_dates = ed[["fifa_version", "edition_date"]].drop_duplicates().sort_values("edition_date")
    ev = edition_dates["fifa_version"].to_numpy()
    ed_t = edition_dates["edition_date"].to_numpy()
    # rating lookup keyed by (edition, team)
    lut = {(r.fifa_version, r.team): r.skill for r in ed.itertuples()}

    def edition_for(date):
        # latest edition whose snapshot date <= match date; clamp to earliest before that
        i = np.searchsorted(ed_t, np.datetime64(date), side="right") - 1
        return ev[max(i, 0)]

    md = matches["date"].to_numpy()
    home = matches["home_team"].to_numpy()
    away = matches["away_team"].to_numpy()
    sh = np.full(len(matches), np.nan)
    sa = np.full(len(matches), np.nan)
    ed_cache = {}
    for i in range(len(matches)):
        e = ed_cache.get(md[i])
        if e is None:
            e = edition_for(md[i]); ed_cache[md[i]] = e
        sh[i] = lut.get((e, home[i]), np.nan)
        sa[i] = lut.get((e, away[i]), np.nan)
    return pd.DataFrame({"skill_home": sh, "skill_away": sa}, index=matches.index)


# ---- current (2026) squads, from the official 26-man lists --------------------------------

def _official_squads() -> dict:
    """Official 26-man squads from the cache -> {team: [{"name","pos"}]}. {} if absent."""
    if not os.path.exists(SQUADS_2026):
        return {}
    try:
        with open(SQUADS_2026, "r", encoding="utf-8") as fh:
            return json.load(fh).get("teams") or {}
    except (OSError, ValueError):
        return {}


def squads_meta() -> dict:
    """{fetched_at, source} for the official-squads cache (for the UI); {} if absent."""
    if not os.path.exists(SQUADS_2026):
        return {}
    try:
        with open(SQUADS_2026, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return {k: blob.get(k) for k in ("fetched_at", "source")}
    except (OSError, ValueError):
        return {}


def current_squads() -> dict:
    """team -> DataFrame(player, full, overall, pos, callup) of its actual squad.

    Preferred: the OFFICIAL 26-man squad (Wikipedia cache), each player matched to EA ratings for
    his overall/position (unmatched fringe players get DEFAULT_OVERALL so they count but don't
    inflate). `callup=True` marks an official squad. Falls back to the top SQUAD_SIZE EA players by
    overall for any team with no official list. Returns {} if the EA ratings file is missing."""
    if not os.path.exists(CURRENT):
        return {}
    from . import availability as av   # local import avoids a module-load cycle
    df = _load(CURRENT)
    df["pos"] = df["player_positions"].map(_pos_bucket)
    official = _official_squads()
    pools = {team: grp for team, grp in df.groupby("team")}

    squads = {}
    for team in set(pools) | set(official):
        pool = pools.get(team)
        off = official.get(team)
        if off and pool is not None and len(pool):
            poolc = (pool[["short_name", "long_name", "overall", "pos"]]
                     .rename(columns={"short_name": "player", "long_name": "full"})
                     .reset_index(drop=True))
            idx, freq = av._index_and_freq(poolc)
            rows = []
            for p in off:
                row = av.match_row(p["name"], poolc, idx, freq)
                if row is not None:
                    rows.append({"player": p["name"], "full": row["full"],
                                 "overall": float(row["overall"]), "pos": p["pos"] or row["pos"]})
                else:
                    rows.append({"player": p["name"], "full": p["name"],
                                 "overall": DEFAULT_OVERALL, "pos": p["pos"] or "MID"})
            sq = pd.DataFrame(rows)
            sq["callup"] = True
            squads[team] = sq.sort_values("overall", ascending=False).reset_index(drop=True)
        elif pool is not None and len(pool):
            chosen = pool.nlargest(SQUAD_SIZE, "overall").copy()
            chosen["callup"] = False
            squads[team] = (chosen[["short_name", "long_name", "overall", "pos", "callup"]]
                            .rename(columns={"short_name": "player", "long_name": "full"})
                            .sort_values("overall", ascending=False).reset_index(drop=True))
    return squads


def current_skill_factors(teams: pd.DataFrame) -> pd.Series:
    """Current squad skill per 2026 team (indexed like `teams`), from FC26 ratings.

    Asserts every one of the 48 resolves to a squad so a silent mapping gap can't zero a team.
    If the current ratings file is missing, returns all-NaN (skill imputed downstream)."""
    if not os.path.exists(CURRENT):
        return pd.Series(np.nan, index=teams.index, name="skill")
    squads = current_squads()
    dn = teams["data_name"] if "data_name" in teams.columns else pd.Series(teams.index, index=teams.index)
    vals, missing = [], []
    for team_disp, name in dn.items():
        sq = squads.get(name)
        if sq is None or len(sq) == 0:   # genuine mapping miss (no EA players at all)
            missing.append(name); vals.append(np.nan)
        else:                            # thin EA squads (minnows) still give valid low ratings
            vals.append(squad_rating(sq["overall"].to_numpy()))
    if missing:
        raise ValueError(f"No EA squad resolved for: {missing} - extend _TEAM_TO_EA in skills.py")
    return pd.Series(vals, index=teams.index, name="skill")
