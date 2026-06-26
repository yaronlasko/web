"""World Cup 2026 knockout bracket: fixed structure + progressive fill.

The Round-of-32 .. Final structure is FIXED by the official FIFA match schedule
(verified against Wikipedia's knockout-stage bracket + Sky Sports / FIFA, 2026-06).
Teams are filled in progressively, so the bracket is useful before it's fully known:

  * group winners / runners-up are PROJECTED from the current group table, and
    become CONFIRMED once that group has finished all 3 games;
  * the 8 best third-placed teams are OPEN placeholders until the whole group stage
    is complete, then each is assigned to its Round-of-32 slot by that slot's
    eligible-group set (a perfect matching of the 8 qualifying thirds to the 8
    third-slots). NOTE: this follows the eligible-group rule; verify against FIFA's
    official R32 draw once published — rare combinations can admit another matching;
  * later rounds (R16+) stay as "Winner of Match N" placeholders until those games
    are played (knockout results aren't folded in yet).

Slot tuples:
    ("W",  "A")                       winner of group A
    ("RU", "A")                       runner-up of group A
    ("3RD", ("A","B","C","D","F"))    best 3rd from one of these groups
"""
from __future__ import annotations

# --- Round of 32 (matches 73-88): each match is two slots ----------------------
R32 = [
    (73, ("RU", "A"), ("RU", "B")),
    (74, ("W", "E"),  ("3RD", ("A", "B", "C", "D", "F"))),
    (75, ("W", "F"),  ("RU", "C")),
    (76, ("W", "C"),  ("RU", "F")),
    (77, ("W", "I"),  ("3RD", ("C", "D", "F", "G", "H"))),
    (78, ("RU", "E"), ("RU", "I")),
    (79, ("W", "A"),  ("3RD", ("C", "E", "F", "H", "I"))),
    (80, ("W", "L"),  ("3RD", ("E", "H", "I", "J", "K"))),
    (81, ("W", "D"),  ("3RD", ("B", "E", "F", "I", "J"))),
    (82, ("W", "G"),  ("3RD", ("A", "E", "H", "I", "J"))),
    (83, ("RU", "K"), ("RU", "L")),
    (84, ("W", "H"),  ("RU", "J")),
    (85, ("W", "B"),  ("3RD", ("E", "F", "G", "I", "J"))),
    (86, ("W", "J"),  ("RU", "H")),
    (87, ("W", "K"),  ("3RD", ("D", "E", "I", "J", "L"))),
    (88, ("RU", "D"), ("RU", "G")),
]

# --- later rounds: (match, feeder_match_x, feeder_match_y) ----------------------
R16   = [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
         (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)]
QF    = [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)]
SF    = [(101, 97, 98), (102, 99, 100)]
FINAL = [(104, 101, 102)]


def _group_order(teams: dict, g: str) -> list[str]:
    """Teams of group g, ranked by points, then goal-difference, then goals-for.
    (Head-to-head is not modelled — a minor simplification in tight groups.)"""
    gt = [t for t in teams if teams[t]["group"] == g]
    return sorted(gt, key=lambda t: (teams[t]["pts"], teams[t]["gd"], teams[t]["gf"]),
                  reverse=True)


def _group_done(teams: dict, g: str) -> bool:
    return all(teams[t]["pld"] >= 3 for t in teams if teams[t]["group"] == g)


def _match_thirds(groups: set[str], slots: list[tuple[int, set]]) -> dict[str, int]:
    """Perfect matching: assign each qualifying third's group to a third-slot whose
    eligible set contains it. Returns {group: match_no}. Backtracking (8x8 = trivial)."""
    gs = sorted(groups)
    used = [False] * len(slots)
    out: dict[str, int] = {}

    def bt(i: int) -> bool:
        if i == len(gs):
            return True
        g = gs[i]
        for si, (m, elig) in enumerate(slots):
            if not used[si] and g in elig:
                used[si] = True
                out[g] = m
                if bt(i + 1):
                    return True
                used[si] = False
                del out[g]
        return False

    bt(0)
    return dict(out)


def build_bracket(teams: dict) -> dict:
    """Build the progressively-filled bracket from the current standings (teams.json)."""
    groups = sorted({teams[t]["group"] for t in teams})
    order = {g: _group_order(teams, g) for g in groups}
    done = {g: _group_done(teams, g) for g in groups}
    gs_complete = all(done.values())

    # --- third-placed assignment (only once the entire group stage is complete) ---
    third_team_for_match: dict[int, str] = {}
    qualified_thirds: list[str] = []
    if gs_complete:
        thirds = {g: order[g][2] for g in groups}                 # 3rd-placed team / group
        ranked = sorted(groups, reverse=True,
                        key=lambda g: (teams[thirds[g]]["pts"], teams[thirds[g]]["gd"],
                                       teams[thirds[g]]["gf"]))
        qualified = ranked[:8]                                    # 8 best thirds advance
        qualified_thirds = [thirds[g] for g in qualified]
        third_slots = [(m, set(b[1])) for (m, a, b) in R32 if b[0] == "3RD"]
        for grp, m in _match_thirds(set(qualified), third_slots).items():
            third_team_for_match[m] = thirds[grp]

    def fill(slot, match_no) -> dict:
        kind = slot[0]
        if kind in ("W", "RU"):
            g = slot[1]
            team = order[g][0] if kind == "W" else order[g][1]
            label = ("Winner " if kind == "W" else "Runner-up ") + g
            return {"label": label, "team": team,
                    "status": "confirmed" if done[g] else "projected"}
        # third-place slot
        label = "3rd " + "/".join(slot[1])
        if match_no in third_team_for_match:
            return {"label": label, "team": third_team_for_match[match_no],
                    "status": "confirmed"}
        return {"label": label, "team": None, "status": "open"}

    rounds = [{
        "name": "Round of 32",
        "matches": [{"m": m, "slots": [fill(a, m), fill(b, m)]} for (m, a, b) in R32],
    }]
    for name, spec in [("Round of 16", R16), ("Quarter-finals", QF),
                       ("Semi-finals", SF), ("Final", FINAL)]:
        rounds.append({
            "name": name,
            "matches": [{"m": m, "slots": [
                {"label": f"Winner M{x}", "team": None, "status": "open"},
                {"label": f"Winner M{y}", "team": None, "status": "open"},
            ]} for (m, x, y) in spec],
        })

    return {
        "group_stage_complete": gs_complete,
        "qualified_thirds": qualified_thirds,
        "rounds": rounds,
    }
