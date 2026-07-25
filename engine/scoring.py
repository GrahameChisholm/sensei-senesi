"""Single source of truth for FPL scoring rules, 2026/27 season — the season this engine targets.

No model module should hardcode a point value, threshold, or BPS rule directly; import it from
here instead. See planning/BUILD_PLAN.md Phase 0 and section 2.5/2.6 for the sourcing rationale
behind the defensive-contribution and BPS-rework figures below.
"""

from types import MappingProxyType

GK, DEF, MID, FWD = "GK", "DEF", "MID", "FWD"
POSITIONS = (GK, DEF, MID, FWD)

# Minutes played -> appearance points. Unchanged for 2026/27.
APPEARANCE_POINTS = MappingProxyType(
    {
        "0": 0,
        "1-59": 1,
        "60+": 2,
    }
)

# Points per goal scored, by position. Confirmed unchanged for 2026/27.
GOAL_POINTS = MappingProxyType({GK: 10, DEF: 6, MID: 5, FWD: 4})

# Points per assist. Confirmed unchanged for 2026/27, same for every position.
ASSIST_POINTS = 3

# Points for a clean sheet, by position. Requires 60+ minutes played. Confirmed unchanged.
CLEAN_SHEET_POINTS = MappingProxyType({GK: 4, DEF: 4, MID: 1, FWD: 0})
CLEAN_SHEET_MIN_MINUTES = 60

# Goals conceded penalty: -1 for every 2 goals conceded, GK/DEF only. Confirmed unchanged.
GOALS_CONCEDED_PER_PENALTY = 2
GOALS_CONCEDED_PENALTY = -1
GOALS_CONCEDED_POSITIONS = (GK, DEF)

# Defensive contribution: reach the action threshold in a match -> flat points. New in 2025/26,
# confirmed unchanged into 2026/27. "Actions" = tackles + interceptions + blocks + clearances
# (+ recoveries for MID/FWD) per engine/models/defensive_contribution.py (2.5). Not for GK.
DEFENSIVE_CONTRIBUTION_THRESHOLD = MappingProxyType({DEF: 10, MID: 12, FWD: 12})
DEFENSIVE_CONTRIBUTION_POINTS = 2

# Saves (GK): 1 point per 3 saves, plus a flat bonus for a penalty save.
SAVES_PER_POINT = 3
PENALTY_SAVE_POINTS = 5

# Penalty miss: flat deduction, any position that takes one.
PENALTY_MISS_POINTS = -2

# Cards.
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3

# Own goal.
OWN_GOAL_POINTS = -2

# Bonus: top 3 BPS scorers in a match get 3/2/1 respectively (engine/models/bonus.py, 2.6).
BONUS_POINTS_BY_RANK = MappingProxyType({1: 3, 2: 2, 3: 1})

# --- BPS (Bonus Points System) formula, reworked for 2026/27 -----------------------------------
# The full underlying event-to-BPS point table is NOT reproduced here yet — engine/models/bonus.py
# (2.6) recomputes historical bonus from raw match-event data using the *current* formula rather
# than trusting any pre-2026/27 "actual bonus received" column, so the exact numeric table must be
# sourced from the official 2026/27 rules before that module is built. What's confirmed so far,
# recorded here so no downstream constant silently assumes the old (pre-2026/27) formula:
BPS_REWORK_NOTES_2026_27 = (
    "Tackles no longer carry the old -1 BPS penalty.",
    "Defenders earn BPS per 3 CBIs (clearances/blocks/interceptions) instead of per 2.",
    "Goalkeeper save BPS is restructured by shot category rather than a flat per-save rate.",
)
