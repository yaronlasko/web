"""World Cup 2026 group-stage match predictor.

Mission (extends the Polymarket-accuracy project's headline finding that SPORTS
markets are the best-calibrated): predict every remaining group-stage match with a
*point-optimal* exact-score guess for the office prediction game, plus W/D/L
confidence and a Monte-Carlo qualification projection.

Office game scoring (the optimisation target):
    exact score correct  -> max(total goals in match, 3) points
    right winner/draw only -> 1 point
    wrong outcome          -> 0 points
So the recommended score is the one MAXIMISING EXPECTED POINTS, not the single most
likely score.

Method = MODEL + MARKET BLEND
    model  : Elo (eloratings.net, current) + this-WC form -> expected goals ->
             Dixon-Coles bivariate Poisson -> full score matrix.
    market : Polymarket de-vigged moneyline + exact-score markets (the well-calibrated
             signal). Built into a score matrix and raked to the moneyline marginals.
    blend  : P = w_market * market + (1-w_market) * model   (market-led; see W_MARKET).

Data (snapshot 2026-06-25, gathered live):
    data/worldcup/teams.json   - Elo + current standings (pts, gf, ga) for all 48 teams
    data/worldcup/market.json  - Polymarket moneyline + exact-score per remaining match

Run:  python -m src.worldcup_predict
"""
from __future__ import annotations
import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

from src.worldcup_bracket import build_bracket

ROOT = Path(__file__).resolve().parent.parent
WC = ROOT / "data" / "worldcup"
WEB_DIR = ROOT / "web"
OUT_JSON = WC / "predictions.json"
OUT_MD = ROOT / "models" / "worldcup_predictions.md"
OUT_MD_SAFE = ROOT / "models" / "worldcup_predictions_safe.md"

# ---- model / blend hyperparameters (global, domain-set; NOT fit per match) ------
MAXG = 7                 # goals grid 0..MAXG-1 ... we use 0..6 for picks, 0..7 internal
GRID = 8                 # internal score grid size (0..7)
ELO_PER_GOAL = 175.0     # Elo points of supremacy worth ~1 goal of expected diff
BASE_TOTAL = 2.55        # baseline expected total goals (WC group avg ~2.5-2.7)
MISMATCH_TOTAL = 0.0009  # extra total goals per Elo of |gap| (blowouts score more)
MISMATCH_CAP = 0.8
HOST_HA_ELO = 60.0       # home-advantage Elo bump for a host nation (only USA remains)
FORM_COEF = 8.0          # Elo nudge per (goal-difference-per-game) so far this WC
DC_RHO = 0.06            # Dixon-Coles low-score dependence (mild draw inflation)
W_MARKET = 0.70          # blend weight on the (well-calibrated) market
HOSTS = {"United States", "Mexico", "Canada"}
N_SIMS = 40000


# ----------------------------------------------------------------------------
# Poisson / Dixon-Coles
# ----------------------------------------------------------------------------
def _pois(lmbda: float, n: int = GRID) -> np.ndarray:
    k = np.arange(n)
    return np.exp(-lmbda) * lmbda ** k / np.array([math.factorial(i) for i in k])


def dc_matrix(lh: float, la: float, rho: float = DC_RHO) -> np.ndarray:
    """Dixon-Coles score matrix: independent Poisson with a low-score correction."""
    ph, pa = _pois(lh), _pois(la)
    m = np.outer(ph, pa)
    m[0, 0] *= 1.0 - lh * la * rho
    m[0, 1] *= 1.0 + lh * rho
    m[1, 0] *= 1.0 + la * rho
    m[1, 1] *= 1.0 - rho
    return m / m.sum()


def wdl(mat: np.ndarray) -> tuple[float, float, float]:
    """(P home win, P draw, P away win) from a score matrix [home, away]."""
    home = np.tril(mat, -1).sum()   # home goals > away goals
    draw = np.trace(mat)
    away = np.triu(mat, 1).sum()
    return home, draw, away


# ----------------------------------------------------------------------------
# Model side: Elo + form -> expected goals -> score matrix
# ----------------------------------------------------------------------------
def model_matrix(home: dict, away: dict, home_team: str) -> np.ndarray:
    eh = home["elo"] + FORM_COEF * (home["gd"] / max(home["pld"], 1))
    ea = away["elo"] + FORM_COEF * (away["gd"] / max(away["pld"], 1))
    if home_team in HOSTS:
        eh += HOST_HA_ELO
    diff = eh - ea
    sup = diff / ELO_PER_GOAL
    total = BASE_TOTAL + min(MISMATCH_TOTAL * abs(diff), MISMATCH_CAP)
    lh = max(0.18, (total + sup) / 2.0)
    la = max(0.18, (total - sup) / 2.0)
    return dc_matrix(lh, la), lh, la


# ----------------------------------------------------------------------------
# Market side: exact-score + moneyline -> score matrix raked to moneyline
# ----------------------------------------------------------------------------
def market_matrix(match: dict, model_mat: np.ndarray) -> np.ndarray | None:
    """Build a score matrix from the Polymarket exact-score market, spreading the
    'other' bucket over un-enumerated cells in proportion to the model, then rake
    the three outcome regions to match the (more liquid) de-vigged moneyline."""
    ml = match.get("moneyline") or {}
    if not ml:
        return None
    es = match.get("exact_score")
    K = np.zeros((GRID, GRID))
    if es:
        enumerated = 0.0
        for key, p in es.items():
            if key == "other":
                continue
            h, a = key.split("-")
            h, a = int(h), int(a)
            if h < GRID and a < GRID:
                K[h, a] += p
                enumerated += p
        remaining = max(0.0, 1.0 - enumerated)             # 'other' + clipped tail
        mask = K == 0
        tail = model_mat * mask
        if tail.sum() > 0:
            K += remaining * tail / tail.sum()
    else:
        K = model_mat.copy()                                # no exact-score market
    K /= K.sum()
    # rake the three outcome regions so marginals equal the moneyline
    regions = {
        "home": np.tril(np.ones((GRID, GRID)), -1).astype(bool),
        "draw": np.eye(GRID, dtype=bool),
        "away": np.triu(np.ones((GRID, GRID)), 1).astype(bool),
    }
    for r, m in regions.items():
        cur = K[m].sum()
        if cur > 0:
            K[m] *= ml[r] / cur
    return K / K.sum()


# ----------------------------------------------------------------------------
# Expected-points optimiser for the office game
# ----------------------------------------------------------------------------
def best_pick(mat: np.ndarray, maxg: int = 6, eps: float = 0.005):
    """Score (a,b) maximising expected office-game points under matrix `mat`.

    Tie-break: among scorelines within `eps` of the best EV (common for blowouts where
    0-2/0-3/0-4 are near-tied), prefer the one MOST LIKELY to actually hit, then the
    lower-scoring one. Same expected value, lower variance / higher exact-hit rate.
    Returns (pick, ev_of_pick, p_exact_of_pick).
    """
    pH, pD, pA = wdl(mat)
    cands = []
    for a in range(maxg + 1):
        for b in range(maxg + 1):
            p_exact = mat[a, b]
            p_outcome = pH if a > b else (pD if a == b else pA)
            ev = p_exact * max(a + b, 3) + (p_outcome - p_exact) * 1.0
            cands.append((ev, p_exact, (a, b)))
    best_ev = max(c[0] for c in cands)
    near = [c for c in cands if c[0] >= best_ev - eps]
    near.sort(key=lambda c: (c[1], -(c[2][0] + c[2][1])), reverse=True)
    ev, p_exact, pick = near[0]
    return pick, ev, p_exact


def modal_score(mat: np.ndarray) -> tuple[tuple[int, int], float]:
    idx = np.unravel_index(np.argmax(mat), mat.shape)
    return (int(idx[0]), int(idx[1])), float(mat[idx])


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def build_predictions():
    teams = json.load(open(WC / "teams.json", encoding="utf-8"))
    market = json.load(open(WC / "market.json", encoding="utf-8"))
    # group-stage games only: both teams must exist and be in the SAME group
    # (knockout fixtures, e.g. a Round-of-32 match, are cross-group and are skipped).
    games = [m for m in market.values()
             if m["home"] in teams and m["away"] in teams
             and teams[m["home"]]["group"] == teams[m["away"]]["group"]]
    order = sorted(games, key=lambda m: (m["date"], m["group"]))
    preds = []
    for m in order:
        h, a = m["home"], m["away"]
        mm, lh, la = model_matrix(teams[h], teams[a], h)
        km = market_matrix(m, mm)
        if km is None:
            km = mm
        blend = W_MARKET * km + (1 - W_MARKET) * mm
        blend /= blend.sum()

        pick, ev, pick_p = best_pick(blend)
        modal, modal_p = modal_score(blend)
        pH, pD, pA = wdl(blend)
        mH, mD, mA = wdl(mm)               # model-only
        kH, kD, kA = wdl(km)               # market-only
        def stats_for(a, b):
            c = "home" if a > b else ("draw" if a == b else "away")
            conf = {"home": pH, "draw": pD, "away": pA}[c]
            p_exact = blend[a, b]
            ev_pts = p_exact * max(a + b, 3) + (conf - p_exact)
            return c, conf, ev_pts
        pick_cls, pick_conf, _ = stats_for(*pick)
        modal_cls, modal_conf, modal_ev = stats_for(*modal)
        preds.append({
            "group": m["group"], "home": h, "away": a, "date": m["date"],
            "lambda_home": round(lh, 2), "lambda_away": round(la, 2),
            "blend_wdl": [round(pH, 3), round(pD, 3), round(pA, 3)],
            "model_wdl": [round(mH, 3), round(mD, 3), round(mA, 3)],
            "market_wdl": [round(kH, 3), round(kD, 3), round(kA, 3)],
            "pick": list(pick), "pick_ev": round(ev, 3), "pick_class": pick_cls,
            "pick_hit": round(pick_p, 3), "confidence": round(pick_conf, 3),
            "modal": list(modal), "modal_p": round(modal_p, 3),
            "modal_ev": round(modal_ev, 3), "modal_class": modal_cls,
            "modal_conf": round(modal_conf, 3),
            "matrix": blend,            # kept in memory for the simulation
        })
    return teams, preds


# ----------------------------------------------------------------------------
# Monte-Carlo qualification (all remaining matches simulated jointly)
# ----------------------------------------------------------------------------
def simulate(teams: dict, preds: list, n: int = N_SIMS, seed: int = 7):
    rng = np.random.default_rng(seed)
    # pre-sample scorelines for each remaining match
    samples = {}
    for p in preds:
        flat = p["matrix"].flatten()
        flat = flat / flat.sum()
        draws = rng.choice(len(flat), size=n, p=flat)
        samples[(p["home"], p["away"])] = (draws // GRID, draws % GRID)  # (hg, ag)

    groups: dict[str, list[str]] = {}
    for t, info in teams.items():
        groups.setdefault(info["group"], []).append(t)

    active = sorted({p["group"] for p in preds})
    rem_by_group = {g: [(p["home"], p["away"]) for p in preds if p["group"] == g] for g in active}

    # advancement tallies
    adv = {t: 0 for t in teams}
    win = {t: 0 for t in teams}
    pos = {t: [0, 0, 0, 0] for t in teams}

    # fixed thirds from already-completed groups
    fixed_groups = [g for g in groups if g not in active]

    def rank(group_teams, pts, gf, ga):
        # FIFA-style: points, goal diff, goals for, then random (H2H not modelled)
        return sorted(group_teams,
                      key=lambda t: (pts[t], gf[t] - ga[t], gf[t], rng.random()),
                      reverse=True)

    for s in range(n):
        thirds = []          # (pts, gd, gf, team)
        for g in groups:
            gt = groups[g]
            pts = {t: teams[t]["pts"] for t in gt}
            gf = {t: teams[t]["gf"] for t in gt}
            ga = {t: teams[t]["ga"] for t in gt}
            for (hh, aa) in rem_by_group.get(g, []):
                hg = int(samples[(hh, aa)][0][s]); ag = int(samples[(hh, aa)][1][s])
                gf[hh] += hg; ga[hh] += ag; gf[aa] += ag; ga[aa] += hg
                if hg > ag:
                    pts[hh] += 3
                elif hg < ag:
                    pts[aa] += 3
                else:
                    pts[hh] += 1; pts[aa] += 1
            ordered = rank(gt, pts, gf, ga)
            for i, t in enumerate(ordered):
                pos[t][i] += 1
            adv[ordered[0]] += 1; adv[ordered[1]] += 1
            win[ordered[0]] += 1
            t3 = ordered[2]
            thirds.append((pts[t3], gf[t3] - ga[t3], gf[t3], t3))
        # 8 best third-placed across all 12 groups advance
        thirds.sort(key=lambda x: (x[0], x[1], x[2], rng.random()), reverse=True)
        for _, _, _, t in thirds[:8]:
            adv[t] += 1

    for d in (adv, win):
        for t in d:
            d[t] /= n
    for t in pos:
        pos[t] = [x / n for x in pos[t]]
    return adv, win, pos


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def fmt_pct(x):
    return f"{100*x:4.0f}%"


def main(safe: bool = False):
    teams, preds = build_predictions()
    adv, win, pos = simulate(teams, preds)
    out_md = OUT_MD_SAFE if safe else OUT_MD

    lines = []
    P = lines.append
    mode_name = "SAFE / most-likely" if safe else "MAX-EXPECTED-POINTS"
    P(f"# World Cup 2026 — remaining group-stage predictions ({mode_name})")
    P("")
    P(f"_Snapshot 2026-06-25. Blend = {int(100*W_MARKET)}% Polymarket market + "
      f"{int(100*(1-W_MARKET))}% Elo/form Dixon-Coles model. "
      + ("Score pick = single MOST LIKELY scoreline (highest exact-hit rate)._"
         if safe else
         "Score pick MAXIMISES EXPECTED POINTS for the office game "
         "(exact = max(total goals,3); right winner = 1)._"))
    P("")
    P("## How to read this (office-game strategy)")
    P("")
    if safe:
        P("- **Score pick** = the single highest-probability scoreline — the play that "
          "lands the exact result most often. Lower variance than chasing EV, but it "
          "leaves a little expected value on the table on lop-sided games.")
        P("- **Max-EV alt** = the expected-points-optimal score (run without `--safe`); "
          "for heavy favourites it chases a higher score like 0-4.")
    else:
        P("- **Score pick** = the scoreline with the highest EXPECTED POINTS, not the "
          "most likely score. For heavy favourites it deliberately chases a high score "
          "(e.g. 0-4) because exact 4-goal games pay `max(4,3)=4` and you still bank 1 "
          "pt for the right winner — that genuinely beats a 'safe' 0-2 in EV.")
        P("- **Most likely alt** = the single highest-probability score, if you'd rather "
          "lock in exacts than chase EV (run with `--safe`).")
    P("- **EV pts** = expected office-game points from the pick. **Hit%** = chance the "
      "exact score lands (your variance). **Conf** = chance the predicted winner/draw is "
      "right (your floor — the +1 outcome point).")
    P("- **Blend / Market / Model W/D/W** shows when the model and the (well-calibrated) "
      "market disagree.")
    P("")
    if safe:
        sure = sorted(preds, key=lambda p: p["confidence"], reverse=True)
        P("**Surest outcomes (highest confidence):** " + ", ".join(
            f"{p['home']} {p['modal'][0]}-{p['modal'][1]} {p['away']} "
            f"({fmt_pct(p['confidence']).strip()})" for p in sure[:5]))
    else:
        by_ev = sorted(preds, key=lambda p: p["pick_ev"], reverse=True)
        P("**Highest-EV games (bank these first):** " + ", ".join(
            f"{p['home']} {p['pick'][0]}-{p['pick'][1]} {p['away']} ({p['pick_ev']:.2f})"
            for p in by_ev[:5]))
    P("")
    coin = sorted(preds, key=lambda p: p["confidence"])
    P("**Coin-flips (lowest confidence — don't overthink these):** "
      + ", ".join(f"{p['home']} v {p['away']}" for p in coin[:4]))
    P("")
    P("## Per-match picks")
    P("")
    alt_hdr = "Max-EV alt" if safe else "Most likely alt"
    P(f"| Grp | Date | Match | **Score pick** | EV pts | Hit% | Conf | "
      f"{alt_hdr} | Blend W/D/W | Market W/D/W | Model W/D/W |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in preds:
        h, a = p["home"], p["away"]
        if safe:
            score, ev_pts, hit, conf = p["modal"], p["modal_ev"], p["modal_p"], p["modal_conf"]
            alt = f"{p['pick'][0]}-{p['pick'][1]} (EV {p['pick_ev']:.2f})"
        else:
            score, ev_pts, hit, conf = p["pick"], p["pick_ev"], p["pick_hit"], p["confidence"]
            alt = f"{p['modal'][0]}-{p['modal'][1]} ({fmt_pct(p['modal_p']).strip()})"
        pk = f"**{h} {score[0]}-{score[1]} {a}**"
        wdl_b = "/".join(fmt_pct(x).strip() for x in p["blend_wdl"])
        wdl_k = "/".join(fmt_pct(x).strip() for x in p["market_wdl"])
        wdl_m = "/".join(fmt_pct(x).strip() for x in p["model_wdl"])
        P(f"| {p['group']} | {p['date'][5:]} | {h} vs {a} | {pk} | "
          f"{ev_pts:.2f} | {fmt_pct(hit).strip()} | {fmt_pct(conf).strip()} | {alt} | "
          f"{wdl_b} | {wdl_k} | {wdl_m} |")

    # divergences: where model disagrees most with market on the favourite prob
    P("")
    P("## Where the model most disagrees with the market")
    P("")
    div = []
    for p in preds:
        d = max(abs(p["blend_wdl"][i] - p["model_wdl"][i]) for i in range(3))
        fav_mkt = max(p["market_wdl"]); fav_mod = max(p["model_wdl"])
        div.append((abs(fav_mkt - fav_mod), p))
    div.sort(reverse=True, key=lambda x: x[0])
    for d, p in div[:5]:
        P(f"- **{p['home']} vs {p['away']}**: market favourite "
          f"{fmt_pct(max(p['market_wdl'])).strip()} vs model "
          f"{fmt_pct(max(p['model_wdl'])).strip()} (Δ{fmt_pct(d).strip()})")

    # qualification projection
    P("")
    P("## Projected qualification (Monte-Carlo, "
      f"{N_SIMS:,} sims of all remaining matches)")
    P("")
    groups = {}
    for t, info in teams.items():
        groups.setdefault(info["group"], []).append(t)
    for g in sorted(groups):
        P(f"### Group {g}")
        P("")
        P("| Team | Pts | GD | P(win grp) | P(advance) |")
        P("|---|---|---|---|---|")
        gt = sorted(groups[g], key=lambda t: adv[t], reverse=True)
        for t in gt:
            P(f"| {t} | {teams[t]['pts']} | {teams[t]['gd']:+d} | "
              f"{fmt_pct(win[t]).strip()} | {fmt_pct(adv[t]).strip()} |")
        P("")

    P("## Data, method & caveats")
    P("")
    P("- **Market** (de-vigged): Polymarket moneyline + exact-score markets for each "
      "game (Gamma API, snapshot 2026-06-25). Polymarket sports markets are the "
      "best-calibrated signal found in the parent project, so they carry "
      f"{int(100*W_MARKET)}% of the blend.")
    P("- **Model** (30%): current Elo (eloratings.net via Wikipedia/worldcupelo) + a "
      "small this-WC form nudge -> expected goals -> Dixon-Coles bivariate Poisson. "
      "Host edge (+60 Elo) applied to the USA only.")
    P("- **Qualification**: joint Monte-Carlo of all 18 remaining matches; top-2 per "
      "group + 8 best third-placed advance. Tie-breaks use points, GD, goals-for "
      "(head-to-head is NOT modelled — a minor simplification in tight groups).")
    P("- **Caveats**: lineups/injuries/suspensions are already priced into the market, "
      "so they enter via the market side rather than as separate features. The model "
      "over-rates Elo-strong-but-underperforming sides (see the disagreement list) — "
      "that's exactly why the market is weighted higher. Numbers move as markets "
      "update; re-pull before kickoff for the freshest odds.")
    report = "\n".join(lines)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")

    # machine-readable predictions (mode-agnostic: carries BOTH picks; drop numpy matrix)
    dump = []
    for p in preds:
        q = {k: v for k, v in p.items() if k != "matrix"}
        q["advance"] = {p["home"]: round(adv[p["home"]], 3),
                        p["away"]: round(adv[p["away"]], 3)}
        dump.append(q)
    json.dump(dump, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---- website data bundle (single source for the static site) -------------
    # `groups` (built above for the markdown qualification section) is still in
    # scope; reuse it so the site carries the full per-team standings, not just
    # the two teams per match that predictions.json keeps.
    web = {
        "meta": {
            "snapshot": "2026-06-25",
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "w_market": W_MARKET,
            "n_sims": N_SIMS,
        },
        "matches": dump,
        "bracket": build_bracket(teams),
        "groups": [
            {
                "group": g,
                "teams": [
                    {
                        "team": t,
                        "pts": teams[t]["pts"],
                        "gd": teams[t]["gd"],
                        "win_group": round(win[t], 3),
                        "advance": round(adv[t], 3),
                    }
                    for t in sorted(groups[g], key=lambda t: adv[t], reverse=True)
                ],
            }
            for g in sorted(groups)
        ],
    }
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(web, ensure_ascii=False, indent=2)
    # data.js (works when opened via file:// too) + data.json (for fetch-based hosts)
    (WEB_DIR / "data.js").write_text(f"window.WC_DATA = {blob};\n", encoding="utf-8")
    (WEB_DIR / "data.json").write_text(blob, encoding="utf-8")

    print(report)
    print(f"\n[saved] {out_md}\n[saved] {OUT_JSON}"
          f"\n[saved] {WEB_DIR / 'data.js'}\n[saved] {WEB_DIR / 'data.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--safe", action="store_true",
                    help="recommend the single most-likely score (highest exact-hit "
                         "rate) instead of the expected-points-maximising score")
    main(safe=ap.parse_args().safe)
