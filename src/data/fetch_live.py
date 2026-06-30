"""LIVE same-day World Cup results overlay from ESPN's free, no-auth JSON API.

The primary history source (martj42/international_results, fetched by fetch_data.py)
lags 1-2 days behind kickoff. ESPN publishes finished scores within minutes, so this
module fetches ESPN's hidden scoreboard JSON and overlays FINISHED scores onto the
empty (NA) WC-2026 fixture rows already present in data/results.csv.

Design decisions / caveats:
  * FULL-TIME ONLY. We ingest matches with status STATUS_FULL_TIME exclusively. In-play
    (e.g. STATUS_SECOND_HALF) and scheduled scores are NEVER written -- writing a partial
    or 0-0 scheduled score would corrupt Elo and the group standings.
  * TEAM-PAIR MATCH, not date match. ESPN timestamps are UTC; martj42 uses the LOCAL
    match date. They can differ by a day (e.g. Tunisia 0-4 Japan is "2026-06-21T04:00Z"
    on ESPN but 2026-06-20 locally). So we match on the unordered TEAM PAIR within the WC
    fixtures and use the date only as a +/-1-day tiebreaker (needed for knockout repeats).
  * IDEMPOTENT. A score is filled only if the CSV cell is currently NA; existing scores are
    never overwritten. Re-running fills 0 the second time.
  * Penalty shootouts are OUT OF SCOPE: only the 90'/120' scoreline is written. Knockout
    tie winners on penalties still rely on the existing shootouts.csv source.

Run:  python -m src.data.fetch_live
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import ssl
import sys
import urllib.request

import pandas as pd

# Reuse fetch_data's resilient SSL/UA approach: some corporate proxies mangle certs,
# so we fall back to an unverified context rather than block a live refresh.
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE

_UA = "Mozilla/5.0 (wc2026-predictor)"

_HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(_HERE, "..", "..", "data")
CONFIG_DIR = os.path.join(_HERE, "..", "..", "config")

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={date}"
)

# ESPN displayName -> martj42 data_name, ONLY for names that differ. Verified against the
# live ESPN feed (June 2026 group stage). All other ESPN names match the data_name exactly.
_ESPN_ALIASES = {
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
    # ESPN also reports these with diacritics matching the data_name exactly, listed for clarity:
    "Curaçao": "Curaçao",
    "Cape Verde": "Cape Verde",
    "Ivory Coast": "Ivory Coast",
    "South Korea": "South Korea",
    "Saudi Arabia": "Saudi Arabia",
    "United States": "United States",
    "Iran": "Iran",
    "New Zealand": "New Zealand",
    # Defensive: alternate spellings ESPN has historically used for the same teams.
    "Cabo Verde": "Cape Verde",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Curacao": "Curaçao",
    "Turkiye": "Turkey",
}


def _load_data_names() -> set[str]:
    """The 48 martj42 data_name join keys from the team config."""
    with open(os.path.join(CONFIG_DIR, "teams_2026.json"), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {t["data_name"] for t in cfg["teams"]}


def map_espn_name(espn_name: str, data_names: set[str]) -> str:
    """Map an ESPN displayName to a martj42 data_name.

    Exact data_name matches win first; then the alias table. Raises a clear error if an
    ESPN team that played a WC match cannot be mapped (fail loud -- never silently drop a
    result, which would leave a fixture stuck at NA).
    """
    if espn_name in data_names:
        return espn_name
    mapped = _ESPN_ALIASES.get(espn_name)
    if mapped is not None and mapped in data_names:
        return mapped
    raise ValueError(
        f"Cannot map ESPN team name {espn_name!r} to a martj42 data_name. "
        f"Add it to _ESPN_ALIASES in src/data/fetch_live.py."
    )


def fetch_live_scores(dates: list[str]) -> list[dict]:
    """Fetch ESPN scoreboard for each YYYYMMDD date; return ONLY full-time matches.

    Per-date failures are tolerated (logged to stderr) so a single bad day never aborts
    the overlay. Each returned dict:
        {date_utc: 'YYYY-MM-DD', home_espn, away_espn, home_score:int, away_score:int}
    """
    out: list[dict] = []
    for date in dates:
        url = ESPN_SCOREBOARD.format(date=date)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30, context=_CTX) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - tolerate per-date failures and continue
            print(f"  [warn] ESPN fetch failed for {date}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue

        for ev in payload.get("events", []):
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            st = (comp.get("status") or {}).get("type") or {}
            # Any genuinely FINISHED match -- regulation, extra time OR penalties. Keying off the
            # state/completed flags instead of the single name STATUS_FULL_TIME means knockout ties
            # that end STATUS_FINAL_PEN / STATUS_FINAL_AET are no longer dropped, while in-play
            # ('in') and scheduled ('pre') matches are still rejected.
            if not (st.get("completed") and st.get("state") == "post"):
                continue

            home = away = None
            for c in comp.get("competitors", []):
                side = c.get("homeAway")
                name = (c.get("team") or {}).get("displayName")
                score = c.get("score")
                if score is None or name is None:
                    continue
                # `score` is the 90'/120' scoreline; `shootoutScore` + `winner` resolve a tie
                # that went to penalties (the shootout is NOT added to the written scoreline).
                rec = {"name": name, "score": int(score),
                       "shootout": c.get("shootoutScore"), "winner": bool(c.get("winner"))}
                if side == "home":
                    home = rec
                elif side == "away":
                    away = rec
            if home is None or away is None:
                continue

            # Penalty shootout -> ESPN flags the winner directly and carries the shootout tally.
            pens = home["shootout"] is not None and away["shootout"] is not None
            winner_espn = (home["name"] if home["winner"]
                           else away["name"] if away["winner"] else None)
            addr = (comp.get("venue") or {}).get("address") or {}

            date_utc = (ev.get("date") or "")[:10]  # 'YYYY-MM-DD' from ISO UTC timestamp
            out.append({
                "date_utc": date_utc,
                "home_espn": home["name"],
                "away_espn": away["name"],
                "home_score": home["score"],
                "away_score": away["score"],
                "pens": pens,
                "winner_espn": winner_espn,
                "city": addr.get("city") or "",
                "country": addr.get("country") or "",
            })
    return out


def _default_dates() -> list[str]:
    """Every tournament day from 2026-06-11 through today (capped at the 2026-07-20 final
    window). The tournament runs into July; we generate the whole span so a late re-run
    still catches knockout days."""
    start = _dt.date(2026, 6, 11)
    today = _dt.date.today()
    end = min(max(today, start), _dt.date(2026, 7, 20))
    days = []
    d = start
    while d <= end:
        days.append(d.strftime("%Y%m%d"))
        d += _dt.timedelta(days=1)
    return days


def merge_live_into_results(results_path: str | None = None,
                            dates: list[str] | None = None) -> dict:
    """Overlay ESPN finished scores onto WC-2026 fixture rows in results.csv.

    Matches on the unordered team pair (orientation-agnostic) within WC fixtures, using the
    date as a +/-1-day tiebreaker. Fills home_score/away_score only when currently NA, and
    orients the scores to the matched CSV row's home/away.

    KNOCKOUT matches aren't pre-loaded in the history source (the matchups aren't known until
    the groups finish), so when a finished match has no existing fixture row we APPEND one
    (neutral WC venue) rather than dropping the result -- otherwise no knockout score could
    ever land. Penalty-shootout outcomes are recorded to shootouts.csv (the written scoreline
    stays the 90'/120' draw) so drawn ties still resolve an advancing team downstream.

    Writes the CSV back in the same format (columns, header, NA/TRUE/FALSE representation) and
    verifies by re-reading.

    Returns: {filled, appended, shootouts_added, matched_already_present, unmapped, details}.
    """
    if results_path is None:
        results_path = os.path.join(DATA_DIR, "results.csv")
    if dates is None:
        dates = _default_dates()

    data_names = _load_data_names()

    # Read EVERY column as a raw string and disable pandas' NA inference. This is the only
    # way to round-trip the file byte-for-byte: it preserves the literal "NA" in the score
    # columns and the upper-case TRUE/FALSE in `neutral` (pandas would otherwise coerce the
    # latter to Python bools and re-serialize them as True/False, rewriting every row).
    # An unplayed score is therefore the literal string "NA"; a played one is a digit string.
    df = pd.read_csv(results_path, keep_default_na=False, dtype=str)
    date_ts = pd.to_datetime(df["date"], errors="coerce")

    is_wc = (df["tournament"] == "FIFA World Cup") & (date_ts >= pd.Timestamp("2026-01-01"))
    wc_idx = df.index[is_wc]

    live = fetch_live_scores(dates)

    filled = 0
    appended = 0
    matched_already_present = 0
    unmapped: list[dict] = []
    details: list[dict] = []
    pen_results: list[dict] = []   # penalty-shootout outcomes to record to shootouts.csv

    for m in live:
        try:
            home_dn = map_espn_name(m["home_espn"], data_names)
            away_dn = map_espn_name(m["away_espn"], data_names)
        except ValueError as exc:
            # Only fail loud for teams that are actually in our 48; otherwise (friendly,
            # non-WC entrant appearing on the feed) just record and skip.
            if m["home_espn"] in data_names or m["away_espn"] in data_names:
                raise
            unmapped.append({**m, "reason": str(exc)})
            continue

        pair = {home_dn, away_dn}
        espn_date = pd.to_datetime(m["date_utc"], errors="coerce")

        # Candidate WC rows whose unordered team pair matches.
        cands = [i for i in wc_idx
                 if {df.at[i, "home_team"], df.at[i, "away_team"]} == pair]
        if not cands:
            # No existing fixture (a knockout tie -- the history source only pre-loads the group
            # schedule). Append a fresh WC row, oriented to ESPN's home/away, so the score has
            # somewhere to live. neutral=TRUE matches the source's convention for WC venues.
            new = {c: "" for c in df.columns}
            new.update(date=m["date_utc"], home_team=home_dn, away_team=away_dn,
                       home_score="NA", away_score="NA", tournament="FIFA World Cup",
                       city=m.get("city", ""), country=m.get("country", ""), neutral="TRUE")
            row = int(df.index.max()) + 1
            df.loc[row] = {c: new.get(c, "") for c in df.columns}
            appended += 1
        else:
            # Tiebreaker: pick the candidate with date within +/-1 day of the ESPN UTC date.
            if espn_date is not None and pd.notna(espn_date):
                scored = sorted(cands, key=lambda i: abs((date_ts[i] - espn_date).days))
                row = scored[0]
                if abs((date_ts[row] - espn_date).days) > 1 and len(cands) > 1:
                    # ambiguous knockout repeat with no nearby date -> skip rather than guess
                    unmapped.append({**m, "home_dn": home_dn, "away_dn": away_dn,
                                     "reason": "team pair found but no candidate within +/-1 day"})
                    continue
            else:
                row = cands[0]

        # Orient scores to the matched CSV row.
        if df.at[row, "home_team"] == home_dn:
            hs, as_ = m["home_score"], m["away_score"]
        else:
            hs, as_ = m["away_score"], m["home_score"]

        # Knockout shootout: stash the winner (mapped) so a drawn tie still advances correctly.
        # Recorded even if the scoreline is already present, so a re-run can backfill it.
        if m.get("pens") and m.get("winner_espn"):
            try:
                pen_results.append({"date": str(df.at[row, "date"])[:10],
                                    "home_team": df.at[row, "home_team"],
                                    "away_team": df.at[row, "away_team"],
                                    "winner": map_espn_name(m["winner_espn"], data_names)})
            except ValueError:
                pass  # unmappable winner name -> leave the tie unresolved rather than guess

        cur_h = df.at[row, "home_score"]
        cur_a = df.at[row, "away_score"]
        already = (cur_h not in ("", "NA")) and (cur_a not in ("", "NA"))
        if already:
            matched_already_present += 1
            continue

        df.at[row, "home_score"] = str(int(hs))
        df.at[row, "away_score"] = str(int(as_))
        filled += 1
        details.append({
            "date": df.at[row, "date"],
            "home": df.at[row, "home_team"],
            "away": df.at[row, "away_team"],
            "score": f"{int(hs)}-{int(as_)}",
        })

    if filled or appended:
        # Round-trip verbatim: same columns, header, NA literal and TRUE/FALSE casing.
        df.to_csv(results_path, index=False)
        # Verify by re-reading (with normal NA inference): the rows we just filled must now
        # read back as real, non-null integer scores.
        check = pd.read_csv(results_path)
        for d in details:
            sel = check[(check["date"].astype(str).str[:10] == str(d["date"])[:10])
                        & (check["home_team"] == d["home"])
                        & (check["away_team"] == d["away"])]
            if sel.empty or pd.isna(sel.iloc[0]["home_score"]):
                raise RuntimeError(f"verification failed for {d}")

    shootouts_added = _record_shootouts(pen_results)

    return {
        "filled": filled,
        "appended": appended,
        "shootouts_added": shootouts_added,
        "matched_already_present": matched_already_present,
        "unmapped": unmapped,
        "details": details,
    }


def _record_shootouts(pen_results: list[dict], shootouts_path: str | None = None) -> int:
    """Append penalty-shootout outcomes to shootouts.csv, deduped on (date, team pair).

    Each entry is {date 'YYYY-MM-DD', home_team, away_team, winner} in martj42 data_names,
    oriented to the matching results row. Returns the number of new rows written.
    """
    if not pen_results:
        return 0
    if shootouts_path is None:
        shootouts_path = os.path.join(DATA_DIR, "shootouts.csv")
    if os.path.exists(shootouts_path):
        sdf = pd.read_csv(shootouts_path, keep_default_na=False, dtype=str)
    else:
        sdf = pd.DataFrame(columns=["date", "home_team", "away_team", "winner", "first_shooter"])
    seen = {(str(r["date"])[:10], frozenset({r["home_team"], r["away_team"]}))
            for _, r in sdf.iterrows()}
    added = 0
    for p in pen_results:
        key = (p["date"], frozenset({p["home_team"], p["away_team"]}))
        if key in seen:
            continue
        sdf.loc[len(sdf)] = {c: {"date": p["date"], "home_team": p["home_team"],
                                 "away_team": p["away_team"], "winner": p["winner"]}.get(c, "")
                             for c in sdf.columns}
        seen.add(key)
        added += 1
    if added:
        sdf.to_csv(shootouts_path, index=False)
    return added


if __name__ == "__main__":
    print("Overlaying ESPN live World Cup results onto data/results.csv ...")
    res = merge_live_into_results()
    print(f"\nFilled {res['filled']} new score(s) ({res['appended']} appended knockout row(s), "
          f"{res['shootouts_added']} shootout(s) recorded); "
          f"{res['matched_already_present']} already present.")
    if res["details"]:
        print("Newly filled:")
        for d in res["details"]:
            print(f"  {d['date']}  {d['home']} {d['score']} {d['away']}")
    if res["unmapped"]:
        print(f"\n{len(res['unmapped'])} ESPN match(es) not merged:")
        for u in res["unmapped"]:
            print(f"  {u.get('date_utc')}  {u.get('home_espn')} vs {u.get('away_espn')}"
                  f"  -- {u.get('reason')}")
    print("Done.")
