"""Single source of truth for the office-pool point system.

This is BOTH the optimisation target (src.worldcup_predict picks the score that
maximises expected points under this rule) AND the grading rule
(src.worldcup_score grades resolved games with it). Keeping them in one module is
the whole point: the EV optimiser and the scorecard can never disagree.

Point system (from the league's official "חוקי הליגה" rules screen, confirmed by the
user 2026-06-29):

  Base, per match (predicted score `pick` vs actual score):
    * exact score correct        -> 3
        + goals bonus            -> +1 for EACH goal OVER 3, ONLY when the exact score
                                    is correct. So exact 3-1 (4 goals) = 3+1 = 4;
                                    exact 3-2 (5 goals) = 3+2 = 5; exact 1-1 = 3 (no bonus).
    * right winner/draw only     -> 1   (does NOT stack on an exact hit — exact replaces it)
    * exact REVERSED scoreline   -> -1  (predict 1-2 and it ends EXACTLY 2-1; the single
                                    mirror cell only — any OTHER opposite-winner score is 0,
                                    and a drawn pick can never be reversed)
    * anything else              -> 0

  Knockout multiplier — "הניקוד מוכפל ככל שמתקדמים בטורניר" (points multiply as you advance):
    Round of 32 x2 · Round of 16 x2 · Quarter-finals x3 · Semi-finals x3 ·
    Third-place x3 · Final x3 · group stage x1.
    Applied to the EARNED points. The -1 penalty is NOT multiplied (undocumented; the
    user believes it stays flat -> PENALTY_SCALES = False, flip if the pool says otherwise).
    A uniform positive multiplier doesn't change WHICH scoreline is optimal within a game;
    it scales the stakes, so KO games rightly outrank group games in the EV rankings.

  Season-long side bets (flat, NOT multiplied): top scorer = 10, champion = 10.
"""
from __future__ import annotations

import numpy as np

# Stage name (as emitted by src.worldcup_bracket) -> points multiplier.
KO_MULTIPLIER: dict[str, int] = {
    "Round of 32": 2,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 3,
    "Third-place": 3,
    "Third-place playoff": 3,
    "Final": 3,
}
PENALTY_SCALES = False   # does the KO multiplier also scale the -1 penalty? (undocumented -> no)


def stage_mult(stage: str | None) -> int:
    """Points multiplier for a stage name; 1 for the group stage / unknown."""
    return KO_MULTIPLIER.get(stage or "", 1)


def _cls(h: int, a: int) -> int:
    """Outcome index: 0 home win, 1 draw, 2 away win (matches the [H, D, A] wdl order)."""
    return 0 if h > a else (1 if h == a else 2)


def office_points(pick, actual, mult: int = 1) -> int:
    """Realized office points for a `pick` vs the `actual` scoreline at KO multiplier `mult`.

    Implements the rule documented at the top of this module. Returns an int.
    """
    (ph, pa), (ah, aa) = pick, actual
    if ph == ah and pa == aa:                                  # exact score
        return (3 + max(0, ah + aa - 3)) * mult                # +1 per goal over 3
    if _cls(ph, pa) == _cls(ah, aa):                           # right winner/draw, wrong score
        return 1 * mult
    if ph != pa and ph == aa and pa == ah:                     # exact REVERSED scoreline -> penalty
        return -1 * (mult if PENALTY_SCALES else 1)
    return 0


def ev_of_pick(mat: np.ndarray, a: int, b: int, mult: int = 1):
    """Expected office points of predicting scoreline (a, b) under score matrix `mat`
    (indexed [home_goals, away_goals]) at KO multiplier `mult`.

    Returns (ev, p_exact, p_outcome) where p_outcome is the probability the predicted
    winner/draw is right (the +1 'floor'). The expected points come from three disjoint
    buckets: the exact cell (3 + goals bonus), the rest of the same-direction mass (+1
    each), and the single mirror cell (b, a) (-1), which has the opposite direction so it
    never overlaps the others.
    """
    p_exact = float(mat[a, b])
    if a > b:
        p_outcome = float(np.tril(mat, -1).sum())              # home goals > away goals
    elif a == b:
        p_outcome = float(np.trace(mat))                       # draw mass
    else:
        p_outcome = float(np.triu(mat, 1).sum())               # away goals > home goals

    bonus = max(0, a + b - 3)                                   # realized only on the exact cell
    earned = p_exact * (3 + bonus) + (p_outcome - p_exact) * 1.0
    ev = earned * mult
    if a != b:                                                 # a drawn pick can't be reversed
        pen = float(mat[b, a])                                  # exact-reversed (mirror) cell
        ev -= pen * (mult if PENALTY_SCALES else 1)
    return ev, p_exact, p_outcome
