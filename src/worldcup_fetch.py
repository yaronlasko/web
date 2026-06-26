"""Refresh the Polymarket market snapshot for World Cup 2026 group games.

Pulls live moneyline + exact-score markets from the Polymarket Gamma API and writes
data/worldcup/market.json (consumed by src.worldcup_predict). Odds move right up to
kickoff, so re-run this shortly before you lock in picks.

    python -m src.worldcup_fetch            # refresh all upcoming fifwc group games
    python -m src.worldcup_predict          # then regenerate predictions

Group labels are looked up from data/worldcup/teams.json by the home team, so this keeps
working as new fixtures appear (just keep teams.json's group assignments current).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import requests

from config import GAMMA_BASE, USER_AGENT

ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "data" / "worldcup"
GAME_SLUG = re.compile(r"^fifwc-[a-z]{3}-[a-z]{3}-2026-\d\d-\d\d$")  # main game (no suffix)
H = {"User-Agent": USER_AGENT}


def _jl(x):
    try:
        return json.loads(x) if isinstance(x, str) else x
    except Exception:
        return x


def fetch_events(max_offset: int = 2000) -> dict:
    """Return {slug: event} for all fifwc game events (main + -exact-score)."""
    keep = {}
    for off in range(0, max_offset, 100):
        r = requests.get(f"{GAMMA_BASE}/events",
                         params={"closed": "false", "limit": 100, "offset": off,
                                 "order": "volume24hr", "ascending": "false"},
                         headers=H, timeout=25)
        evs = r.json()
        if not evs:
            break
        for e in evs:
            s = e.get("slug", "")
            if s.startswith("fifwc-") and (GAME_SLUG.match(s) or s.endswith("-exact-score")):
                keep[s] = e
    return keep


def parse(keep: dict, teams: dict) -> dict:
    matches = {}
    for s, e in keep.items():
        if not GAME_SLUG.match(s):
            continue
        home, away = (p.strip() for p in re.split(r"\s+vs\.?\s+", e["title"], maxsplit=1))
        group = teams.get(home, {}).get("group", "?")
        ml = {}
        for m in e.get("markets", []):
            q = m.get("question", ""); pr = _jl(m.get("outcomePrices")); outs = _jl(m.get("outcomes"))
            if not (outs and pr and len(outs) == 2 and outs[0].lower() == "yes"):
                continue
            yes = float(pr[0])
            if "end in a draw" in q:
                ml["draw"] = yes
            elif q.startswith(f"Will {home} win"):
                ml["home"] = yes
            elif q.startswith(f"Will {away} win"):
                ml["away"] = yes
        tot = sum(ml.values())
        ml_norm = {k: round(v / tot, 4) for k, v in ml.items()} if tot else {}

        es = keep.get(s + "-exact-score")
        dist = None
        if es:
            dist = {}
            for m in es.get("markets", []):
                q = m.get("question", ""); pr = _jl(m.get("outcomePrices")); outs = _jl(m.get("outcomes"))
                if not (outs and pr and len(outs) == 2 and outs[0].lower() == "yes"):
                    continue
                yes = float(pr[0])
                mm = re.search(r":\s*.+?\s(\d+)\s*-\s*(\d+)\s", q)
                if "Any Other Score" in q:
                    dist["other"] = yes
                elif mm:
                    dist[f"{mm.group(1)}-{mm.group(2)}"] = yes
            t = sum(dist.values())
            dist = {k: round(v / t, 5) for k, v in dist.items()} if t else {}

        matches[s] = {"slug": s, "group": group, "home": home, "away": away,
                      "date": s[-10:], "kickoff": e.get("startTime") or e.get("endDate"),
                      "moneyline_raw": ml, "moneyline": ml_norm, "exact_score": dist}
    return matches


def main():
    teams = json.load(open(WC / "teams.json", encoding="utf-8"))
    keep = fetch_events()
    matches = parse(keep, teams)
    WC.mkdir(parents=True, exist_ok=True)
    json.dump(matches, open(WC / "market.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"Saved {len(matches)} games to {WC/'market.json'}")
    for m in sorted(matches.values(), key=lambda x: (x["date"], x["group"])):
        ml = m["moneyline"]
        nes = len(m["exact_score"]) if m["exact_score"] else 0
        print(f"  [{m['group']}] {m['home']} vs {m['away']} ({m['date']})  "
              f"H/D/A={ml.get('home')}/{ml.get('draw')}/{ml.get('away')}  exact={nes}")


if __name__ == "__main__":
    main()
