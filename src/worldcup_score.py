"""Office-pool scorecard: grade the forward-logged predictions against actual results.

`src.worldcup_predict` snapshots every game's pick at the LAST refresh before kickoff into
`pred_log.json` (leakage-free — only games whose kickoff is still in the future are logged,
so the stored prediction always predates the match). This script grades each logged game
that has since resolved and reports:

  * realized office points: EV-max pick vs the safe / most-likely pick (did chasing high
    scores actually pay off in OUR pool?)
  * exact-score hit rate and outcome (W/D/L) accuracy
  * calibration: Brier + log-loss of the blended W/D/L vs market-only vs model-only
    (does the 30% model help or hurt on top of the market?)

Actual scores come from `applied_results.json` (group games, incl. manual fixes) and,
best-effort, the resolved Polymarket exact-score markets (covers knockouts and their
90-minute scores). Games with no clean recorded score are listed as ungraded.

NOTE: forward-logging only captures games played AFTER it was deployed, so the group stage
(already finished) won't appear — the scorecard fills in over the remaining knockout games.

    python -m src.worldcup_score              # grade + write models/worldcup_scorecard.md
    python -m src.worldcup_score --no-fetch   # offline: grade only what's in applied_results
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "data" / "worldcup"
PRED_LOG = WC / "pred_log.json"
LEDGER = WC / "applied_results.json"
OUT_MD = ROOT / "models" / "worldcup_scorecard.md"
PAGES_PRED_LOG_URL = "https://yaronlasko.github.io/web/pred_log.json"


def _load(path: Path, default):
    return json.load(open(path, encoding="utf-8")) if path.exists() else default


def _fetch_remote_log(url: str) -> dict:
    import urllib.request
    try:
        import time
        req = urllib.request.Request(f"{url}?cb={int(time.time())}",
                                     headers={"User-Agent": "wc-score/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _merge_logs(*logs: dict) -> dict:
    out: dict = {}
    for log in logs:
        for slug, e in (log or {}).items():
            if slug not in out or e.get("logged_at", "") > out[slug].get("logged_at", ""):
                out[slug] = e
    return out


def cls(h: int, a: int) -> int:
    """Outcome index: 0 home win, 1 draw, 2 away win — matches the [H, D, A] wdl order."""
    return 0 if h > a else (1 if h == a else 2)


def office_points(pick, actual) -> int:
    """Office-game points: exact score = max(total goals, 3); right outcome only = 1; else 0."""
    (ph, pa), (ah, aa) = pick, actual
    if ph == ah and pa == aa:
        return max(ah + aa, 3)
    return 1 if cls(ph, pa) == cls(ah, aa) else 0


def brier(wdl, idx: int) -> float:
    return sum((wdl[i] - (1.0 if i == idx else 0.0)) ** 2 for i in range(3))


def logloss(wdl, idx: int) -> float:
    return -math.log(max(min(wdl[idx], 1 - 1e-12), 1e-12))


def load_actuals(fetch: bool = True) -> dict:
    """slug -> (home_goals, away_goals) for resolved games (90-minute / full-time score)."""
    actuals: dict[str, tuple[int, int]] = {}
    if fetch:                                            # live: covers knockouts too
        try:
            from src.worldcup_standings import fetch_fifwc_events, resolved_score
            evs = fetch_fifwc_events()
            for slug, e in evs.items():
                if not e.get("closed"):
                    continue
                sc = resolved_score(evs.get(slug + "-exact-score"))
                if isinstance(sc, tuple):
                    actuals[slug] = sc
        except Exception as ex:
            print(f"  (live result fetch skipped: {ex})")
    # applied_results.json wins — it carries the manual fixes for 'Any Other Score' games
    for slug, rec in _load(LEDGER, {}).get("applied", {}).items():
        sc = rec.get("score")
        if sc:
            actuals[slug] = (int(sc[0]), int(sc[1]))
    return actuals


def grade(fetch: bool = True):
    log = _merge_logs(_load(PRED_LOG, {}), _fetch_remote_log(PAGES_PRED_LOG_URL))
    actuals = load_actuals(fetch=fetch)

    rows, ungraded = [], []
    for slug, p in log.items():
        a = actuals.get(slug)
        if not a or not p.get("pick") or not p.get("modal"):
            ungraded.append(p)
            continue
        ah, aa = a
        aidx = cls(ah, aa)
        pick, modal = tuple(p["pick"]), tuple(p["modal"])
        rows.append({
            "slug": slug, "home": p["home"], "away": p["away"],
            "stage": p.get("stage") or (f"Grp {p.get('group')}" if p.get("group") else ""),
            "actual": (ah, aa), "pick": pick, "modal": modal,
            "ev_pts": office_points(pick, a), "safe_pts": office_points(modal, a),
            "exact_ev": pick == a, "exact_safe": modal == a,
            "out_ev": cls(*pick) == aidx, "out_safe": cls(*modal) == aidx,
            "brier_blend": brier(p["blend_wdl"], aidx), "ll_blend": logloss(p["blend_wdl"], aidx),
            "brier_mkt": brier(p["market_wdl"], aidx), "ll_mkt": logloss(p["market_wdl"], aidx),
            "brier_mod": brier(p["model_wdl"], aidx), "ll_mod": logloss(p["model_wdl"], aidx),
        })
    rows.sort(key=lambda r: r["slug"])
    return rows, ungraded


def _avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def report(rows, ungraded):
    n = len(rows)
    L = []
    P = L.append
    P("# World Cup 2026 — office-pool scorecard")
    P("")
    if not n:
        P("_No logged games have resolved yet. Forward-logging captures games from when it "
          "was deployed onward, so this fills in as the remaining knockout games are played._")
        out = "\n".join(L)
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text(out, encoding="utf-8")
        print(out)
        print(f"\n[saved] {OUT_MD}")
        return

    ev_tot = sum(r["ev_pts"] for r in rows)
    safe_tot = sum(r["safe_pts"] for r in rows)
    P(f"_{n} graded game(s). Predictions snapshotted before kickoff (leakage-free)._")
    P("")
    P("## Headline")
    P("")
    P(f"- **EV-max picks:** {ev_tot} pts total ({ev_tot/n:.2f}/game) · "
      f"exact-hit {100*_avg([r['exact_ev'] for r in rows]):.0f}% · "
      f"outcome right {100*_avg([r['out_ev'] for r in rows]):.0f}%")
    P(f"- **Safe (most-likely) picks:** {safe_tot} pts total ({safe_tot/n:.2f}/game) · "
      f"exact-hit {100*_avg([r['exact_safe'] for r in rows]):.0f}% · "
      f"outcome right {100*_avg([r['out_safe'] for r in rows]):.0f}%")
    diff = ev_tot - safe_tot
    verdict = ("EV-max ahead" if diff > 0 else "safe ahead" if diff < 0 else "tied")
    P(f"- **Difference:** {diff:+d} pts in favour of **{verdict}** "
      f"(this is the whole point of the EV-max strategy — is chasing high scores paying off?).")
    P("")
    P("## Calibration (lower is better) — is the 30% model helping?")
    P("")
    P("| Source | Brier | Log-loss |")
    P("|---|---|---|")
    for lab, bk, lk in (("Blend (what we use)", "brier_blend", "ll_blend"),
                        ("Market only", "brier_mkt", "ll_mkt"),
                        ("Model only", "brier_mod", "ll_mod")):
        P(f"| {lab} | {_avg([r[bk] for r in rows]):.4f} | {_avg([r[lk] for r in rows]):.4f} |")
    P("")
    P("_If 'market only' beats the blend, the model is dragging it down — lower `W_MARKET`'s "
      "complement (raise the market weight). If the blend wins, the model is earning its 30%._")
    P("")
    P("## Per-game")
    P("")
    P("| Game | Actual | EV pick (pts) | Safe pick (pts) |")
    P("|---|---|---|---|")
    for r in rows:
        def mark(score, exact):
            return f"{score[0]}-{score[1]}" + ("✓" if exact else "")
        P(f"| {r['stage']}: {r['home']} v {r['away']} | {r['actual'][0]}-{r['actual'][1]} | "
          f"{mark(r['pick'], r['exact_ev'])} ({r['ev_pts']}) | "
          f"{mark(r['modal'], r['exact_safe'])} ({r['safe_pts']}) |")
    P("")
    if ungraded:
        upcoming = [p for p in ungraded if p.get("kickoff")]
        P(f"_{len(ungraded)} logged game(s) not yet graded (unplayed, or resolved to "
          f"'Any Other Score' with no recorded exact score)._")
    out = "\n".join(L)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(out, encoding="utf-8")
    print(out)
    print(f"\n[saved] {OUT_MD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-fetch", action="store_true",
                    help="don't hit the API for results; grade only applied_results.json")
    args = ap.parse_args()
    rows, ungraded = grade(fetch=not args.no_fetch)
    report(rows, ungraded)


if __name__ == "__main__":
    main()
