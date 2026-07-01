"""Auto-update World Cup standings from finished games' Polymarket results.

Reads the *actual* scoreline of each finished group game from its resolved Polymarket
exact-score market and applies it to data/worldcup/teams.json (points, goals, played).
A ledger (data/worldcup/applied_results.json) records what's been applied so a game is
never double-counted, and a snapshot date guards games already baked into the snapshot.

Daily workflow during the group stage:
    python -m src.worldcup_standings     # 1. fold in yesterday's results
    python -m src.worldcup_fetch         # 2. refresh upcoming odds
    python -m src.worldcup_predict       # 3. fresh picks + qualification

Notes / limits:
- teams.json's current snapshot already includes every game BEFORE SNAPSHOT_FROM_DATE, so
  only games on/after that date are applied (the ledger is the primary double-count guard;
  the date is a backstop). If you ever re-baseline teams.json to a later point, bump it.
- If a game resolves on a scoreline Polymarket lumped into "Any Other Score", or has no
  exact-score market (e.g. DR Congo vs Uzbekistan), the exact goals can't be recovered from
  the market alone — that game is reported for MANUAL entry rather than guessed.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import requests

from config import GAMMA_BASE, USER_AGENT
from src.worldcup_bracket import build_bracket

ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "data" / "worldcup"
TEAMS = WC / "teams.json"
LEDGER = WC / "applied_results.json"
KO_LEDGER = WC / "knockout_results.json"    # {match_no: {"winner": team, "score": [h, a]}}
SNAPSHOT_FROM_DATE = "2026-06-25"           # apply only games on/after this date
GAME_RE = re.compile(r"^fifwc-[a-z]{3}-[a-z]{3}-2026-\d\d-\d\d$")
H = {"User-Agent": USER_AGENT}


def _jl(x):
    try:
        return json.loads(x) if isinstance(x, str) else x
    except Exception:
        return x


def _load(path, default):
    return json.load(open(path, encoding="utf-8")) if path.exists() else default


def fetch_fifwc_events(max_offset: int = 1200) -> dict:
    """Sweep open AND recently-closed events; return {slug: event} for fifwc games."""
    events = {}
    for closed in ("false", "true"):
        for off in range(0, max_offset, 100):
            r = requests.get(f"{GAMMA_BASE}/events",
                             params={"closed": closed, "limit": 100, "offset": off,
                                     "order": "volume24hr", "ascending": "false"},
                             headers=H, timeout=25)
            evs = r.json()
            if not evs:
                break
            for e in evs:
                s = e.get("slug", "")
                if s.startswith("fifwc-") and (GAME_RE.match(s) or s.endswith("-exact-score")):
                    events[s] = e
    return events


def resolved_score(es_event: dict | None):
    """(hg, ag) from a resolved exact-score event, 'other' if it resolved to 'Any Other
    Score', or None if not resolved / unavailable."""
    if not es_event:
        return None
    for m in es_event.get("markets", []):
        outs = _jl(m.get("outcomes")); pr = _jl(m.get("outcomePrices"))
        if not (outs and pr and len(outs) == 2 and outs[0].lower() == "yes"):
            continue
        if float(pr[0]) > 0.99:                     # this exact-score line resolved YES
            q = m.get("question", "")
            if "Any Other Score" in q:
                return "other"
            mm = re.search(r":\s*.+?\s(\d+)\s*-\s*(\d+)\s", q)
            if mm:
                return int(mm.group(1)), int(mm.group(2))
    return None


def moneyline_winner(game_event: dict | None, home: str, away: str):
    """Winner of a resolved game from its MONEYLINE market. For knockouts we only need who
    ADVANCED, so this recovers a winner even when the exact score resolved to 'Any Other
    Score' (which blocks resolved_score). Returns the team name, 'draw' if the 90-minute
    draw market resolved YES (undecided by 90' — the ET/pens advancer still needs manual
    entry), or None if nothing resolved yet."""
    if not game_event:
        return None
    for m in game_event.get("markets", []):
        outs = _jl(m.get("outcomes")); pr = _jl(m.get("outcomePrices"))
        if not (outs and pr and len(outs) == 2 and outs[0].lower() == "yes"):
            continue
        if float(pr[0]) <= 0.99:                    # this line hasn't resolved YES
            continue
        q = m.get("question", "")
        if q.startswith(f"Will {home} win"):
            return home
        if q.startswith(f"Will {away} win"):
            return away
        if "end in a draw" in q:
            return "draw"
    return None


def apply_result(teams: dict, home: str, away: str, hg: int, ag: int):
    for t in (home, away):
        teams[t]["pld"] += 1
    teams[home]["gf"] += hg; teams[home]["ga"] += ag
    teams[away]["gf"] += ag; teams[away]["ga"] += hg
    teams[home]["gd"] = teams[home]["gf"] - teams[home]["ga"]
    teams[away]["gd"] = teams[away]["gf"] - teams[away]["ga"]
    if hg > ag:
        teams[home]["pts"] += 3
    elif hg < ag:
        teams[away]["pts"] += 3
    else:
        teams[home]["pts"] += 1; teams[away]["pts"] += 1


def fold_knockouts(teams: dict, events: dict, dry_run: bool = False):
    """Record winners of finished KNOCKOUT games so the bracket fills in round by round.

    Knockout games are the cross-group fixtures (R32 pairs teams from different groups).
    We match each resolved cross-group game to its bracket slot by the pair of teams,
    read the winner from the resolved exact-score market, and store it in
    `knockout_results.json`. A 90-minute draw (extra time / penalties) or a game with no
    exact-score market can't reveal who advanced -> flagged for MANUAL entry, exactly like
    the group-stage fallback. Folds iteratively (R32 -> R16 -> ... -> Final) so each round's
    winners unlock the next round's pairings. CI re-derives this each run; the committed
    file persists any manual entries.
    """
    ko = _load(KO_LEDGER, {})
    # resolved cross-group (knockout) games, indexed by their unordered team pair
    resolved = {}
    for slug, e in events.items():
        if not GAME_RE.match(slug) or not e.get("closed"):
            continue
        parts = re.split(r"\s+vs\.?\s+", e.get("title", ""), maxsplit=1)
        if len(parts) != 2:
            continue
        home, away = parts[0].strip(), parts[1].strip()
        if home not in teams or away not in teams:
            continue
        if teams[home]["group"] == teams[away]["group"]:
            continue                                       # same group = group game, skip
        resolved[frozenset((home, away))] = (home, away, slug)

    applied, manual = [], []
    for _ in range(5):                                     # R32, R16, QF, SF, Final
        changed = False
        for r in build_bracket(teams, ko)["rounds"]:
            for mt in r["matches"]:
                m = mt["m"]
                if str(m) in ko:
                    continue
                a, b = mt["slots"][0].get("team"), mt["slots"][1].get("team")
                if not (a and b):
                    continue                               # both sides not known yet
                got = resolved.get(frozenset((a, b)))
                if not got:
                    continue                               # not finished (or names mismatch)
                home, away, slug = got
                score = resolved_score(events.get(slug + "-exact-score"))
                if score is not None and score != "other" and score[0] != score[1]:
                    hg, ag = score                         # clean decisive scoreline
                    ko[str(m)] = {"winner": home if hg > ag else away, "score": [hg, ag]}
                    applied.append((m, ko[str(m)]["winner"]))
                    changed = True
                    continue
                # no clean exact score (high/uncommon scoreline lumped into "Any Other
                # Score", or a draw) -> take the ADVANCER from the moneyline market.
                win = moneyline_winner(events.get(slug), home, away)
                if win and win != "draw":
                    ko[str(m)] = {"winner": win}           # winner known, exact score isn't
                    applied.append((m, win))
                    changed = True
                else:
                    manual.append((a, b, slug))            # 90-min draw -> ET/pens, needs manual
        if not changed:
            break

    if not dry_run:
        json.dump(ko, open(KO_LEDGER, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return applied, manual


def main(dry_run: bool = False):
    teams = _load(TEAMS, None)
    if teams is None:
        raise SystemExit(f"missing {TEAMS}")
    ledger = _load(LEDGER, {"applied": {}})
    events = fetch_fifwc_events()

    applied, manual, unmatched, pending = [], [], [], 0
    for slug, e in events.items():
        if not GAME_RE.match(slug) or slug[-10:] < SNAPSHOT_FROM_DATE:
            continue
        if slug in ledger["applied"]:
            continue
        parts = re.split(r"\s+vs\.?\s+", e.get("title", ""), maxsplit=1)
        if len(parts) != 2:
            continue
        home, away = parts[0].strip(), parts[1].strip()
        if home not in teams or away not in teams:
            if e.get("closed"):                      # finished but names don't match
                unmatched.append((home, away, slug))
            continue
        if teams[home]["group"] != teams[away]["group"]:
            continue                                  # cross-group = knockout, skip
        if not e.get("closed"):                      # game not finished yet
            pending += 1
            continue
        score = resolved_score(events.get(slug + "-exact-score"))
        if score is None or score == "other":
            manual.append((home, away, slug))
            continue
        hg, ag = score
        if not dry_run:
            apply_result(teams, home, away, hg, ag)
            ledger["applied"][slug] = {"home": home, "away": away, "score": [hg, ag]}
        applied.append((home, hg, ag, away))

    if not dry_run:
        json.dump(teams, open(TEAMS, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        json.dump(ledger, open(LEDGER, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    tag = "[DRY-RUN] " if dry_run else ""
    print(f"{tag}Applied {len(applied)} new result(s); {pending} game(s) still upcoming.")
    for home, hg, ag, away in applied:
        print(f"  + {home} {hg}-{ag} {away}")
    if manual:
        print("  ! Could not auto-resolve exact score (enter manually in teams.json):")
        for home, away, slug in manual:
            print(f"      {home} vs {away}  ({slug})")
    if unmatched:
        print("  ! Finished game whose team names don't match teams.json (check spelling):")
        for home, away, slug in unmatched:
            print(f"      {home} vs {away}  ({slug})")
    # show affected group tables
    affected = {teams[h]["group"] for h, _, _, _ in applied}
    for g in sorted(affected):
        gt = sorted([t for t in teams if teams[t]["group"] == g],
                    key=lambda t: (teams[t]["pts"], teams[t]["gd"], teams[t]["gf"]),
                    reverse=True)
        print(f"\n  Group {g}:")
        for t in gt:
            i = teams[t]
            print(f"    {t:24} P{i['pld']} {i['pts']:2}pts  {i['gf']}-{i['ga']} ({i['gd']:+d})")

    # knockout stage: fold finished knockout games so the bracket advances
    ko_applied, ko_manual = fold_knockouts(teams, events, dry_run=dry_run)
    if ko_applied:
        print(f"\n{tag}Bracket: advanced {len(ko_applied)} winner(s):")
        for m, w in ko_applied:
            print(f"  Match {m}: {w} advances")
    if ko_manual:
        print("  ! Knockout game needs a manual winner (extra time / penalties / no exact "
              "score) — add it to knockout_results.json:")
        for a, b, slug in ko_manual:
            print(f"      {a} vs {b}  ({slug})")

    if not (applied or manual or ko_applied or ko_manual):
        print("  nothing to update — standings already current.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change without writing teams.json")
    main(dry_run=ap.parse_args().dry_run)
