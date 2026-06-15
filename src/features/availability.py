"""Cross-source name matching for squads.

Matching names across data sources is fiddly: official-squad names (Wikipedia) must map to EA
player ratings, and external team names to our `data_name`. EA uses nicknames/mononyms ('Rodri',
'Vini Jr.') while feeds use full names ('Rodrigo Hernández', 'Vinicius Junior'), and accents and
generic surnames ('Junior', 'Silva') abound. The matchers here normalize text, score candidates
by surname alignment + rarity-weighted shared tokens, and resolve namesakes by EA `overall`.
Everything is pure/deterministic and unit-testable — no network.
"""
from __future__ import annotations

import re
import unicodedata

from .skills import _TEAM_TO_EA


# ---- text normalisation -------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _name_parts(name: str):
    """(first_initial_or_'', last_name) from a player name in either 'K. Mbappe',
    'Kylian Mbappe' or 'Mbappe' form."""
    n = _norm(name)
    if not n:
        return "", ""
    toks = n.split()
    last = toks[-1]
    first_init = toks[0][0] if len(toks) > 1 and toks[0] else ""
    return first_init, last


def _tokens(name: str) -> set:
    """Distinctive normalized tokens (len>=3, drops generic suffixes like 'jr'/'junior')."""
    drop = {"jr", "junior", "filho", "neto", "de", "da", "do", "dos", "el", "al"}
    return {t for t in _norm(name).split() if len(t) >= 3 and t not in drop}


# ---- team-name matching (external -> our data_name) ---------------------------------------

def build_team_matcher(team_data_names) -> dict:
    """Map a normalized external team name -> our canonical data_name.

    Seeds from our data_names plus the EA alias spellings (skills._TEAM_TO_EA) so feeds that
    say 'Korea Republic', 'Czechia', 'Turkiye', etc. still resolve."""
    lut = {}
    for dn in team_data_names:
        lut[_norm(dn)] = dn
    for team, aliases in _TEAM_TO_EA.items():
        for a in aliases:
            lut.setdefault(_norm(a), team)
    return lut


def match_team(external_name: str, matcher: dict):
    """Resolve an external team name to our data_name, or None if unrecognised."""
    return matcher.get(_norm(external_name))


# ---- player matching (API name -> EA squad short_name) ------------------------------------

def _player_index(squad):
    """Build match candidates from a squad DataFrame: each player's display name plus the
    token set drawn from BOTH its short and full names (so nicknames/mononyms like 'Rodri' or
    'Vinicius Junior' still match a feed's full name)."""
    has_full = "full" in squad.columns
    idx = []
    for r in squad.itertuples():
        full = getattr(r, "full", r.player) if has_full else r.player
        toks = _tokens(r.player) | _tokens(full)
        idx.append({"short": r.player, "tokens": toks, "surname": _name_parts(full)[1]})
    return idx


def _score(api_tokens, api_last, cand, freq) -> int:
    """Match score: +2 if surnames align, plus per shared token weighted by rarity within the
    squad (a token unique to one squad player scores 2; a common one like 'silva' scores 1).
    This lets a distinctive mononym ('Vinicius', 'Rodri') match even when the surname is the
    generic 'Junior', without common tokens causing false positives."""
    surname_hit = (api_last and (api_last in cand["tokens"] or cand["surname"] in api_tokens))
    shared = api_tokens & cand["tokens"]
    return (2 if surname_hit else 0) + sum(2 if freq.get(t, 0) == 1 else 1 for t in shared)


def _index_and_freq(squad):
    """Build the match index plus a token-frequency map (for rarity weighting), once per pool."""
    idx = _player_index(squad)
    freq: dict = {}
    for c in idx:
        for t in c["tokens"]:
            freq[t] = freq.get(t, 0) + 1
    return idx, freq


def _best_pos(api_name, idx, freq, tiebreak=None):
    """Position (within idx) of the best match, or None if no candidate clears the bar.

    `tiebreak` (optional, positional, e.g. EA overall) resolves equal-scoring candidates by
    picking the highest value — the selected international is almost always the higher-rated of
    two namesakes. Without it, an ambiguous tie returns None (won't guess)."""
    api_tokens = _tokens(api_name)
    _, api_last = _name_parts(api_name)
    scored = sorted(((_score(api_tokens, api_last, c, freq), p) for p, c in enumerate(idx)),
                    reverse=True)
    if not scored or scored[0][0] < 2:             # need a surname hit or a distinctive token
        return None
    top = scored[0][0]
    tied = [p for s, p in scored if s == top]
    if len(tied) == 1:
        return tied[0]
    if tiebreak is not None:
        return max(tied, key=lambda p: tiebreak[p])
    return None                                    # ambiguous and no tiebreak -> don't guess


def match_player(api_name: str, squad) -> str | None:
    """Best EA-squad player name for an external name, or None if no confident/unique match.

    `squad` may be a squad DataFrame (preferred — uses full names too) or a plain list of names.
    Scores every player by surname alignment + rarity-weighted shared tokens; returns the unique
    best and refuses to guess on ties."""
    if not hasattr(squad, "itertuples"):           # plain list fallback
        import pandas as pd
        squad = pd.DataFrame({"player": list(squad)})
    idx, freq = _index_and_freq(squad)
    pos = _best_pos(api_name, idx, freq)
    return idx[pos]["short"] if pos is not None else None


def match_row(api_name: str, squad_df, idx=None, freq=None):
    """Best-matching ROW (Series) of `squad_df` for `api_name`, or None. Pass precomputed
    (idx, freq) — from `_index_and_freq(squad_df)` — to amortize matching many names against the
    same pool (e.g. official squad names against a nation's EA player pool). Equal-scoring
    namesakes are resolved by EA `overall` when present."""
    if idx is None:
        idx, freq = _index_and_freq(squad_df)
    tb = squad_df["overall"].to_numpy() if "overall" in squad_df.columns else None
    pos = _best_pos(api_name, idx, freq, tiebreak=tb)
    return squad_df.iloc[pos] if pos is not None else None
