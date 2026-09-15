"""One horizon gameweek's squad plan, the team-selection page's endpoints read and write -- 0 to 15
players, never locked in, no confirm step. ``api.state.get_squad_state``/``set_squad_state`` add
the per-gameweek dimension (a gameweek with no plan of its own live-follows the nearest earlier
gameweek that has one); this type itself is just one gameweek's snapshot. Kept separate from
``features.team_state.MyTeamState``
because it must represent every size from empty to a complete 15, whereas ``MyTeamState`` only
ever represents a complete, legal 15/11 squad; a bare ``SquadState`` is promoted to a real
``MyTeamState`` (via ``features.squad_rules.assemble_team_state``/``build_team_state``) only where
a full squad is actually required, e.g. a points preview.

A separate module from ``api.state`` (not folded into ``AppState``) so ``api.persistence`` can
import this type without a circular import back through ``api.state``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from features.squad_rules import INITIAL_BUDGET
from features.team_state import SquadPlayer

__all__ = ["SquadState"]


@dataclass
class SquadState:
    squad: tuple[SquadPlayer, ...] = ()
    starting_xi: tuple[int, ...] = ()
    bench_order: tuple[int, ...] = ()
    captain_id: int | None = None
    vice_captain_id: int | None = None
    mini_league_ids: tuple[int, ...] = field(default=())
    # Personal budget ceiling checked by every add/remove/optimize call -- the classic £100m by
    # default, or a higher figure computed at import time from the real squad's current player
    # prices plus the entry's bank. Reset back to INITIAL_BUDGET when the squad is cleared.
    budget_ceiling: int = INITIAL_BUDGET
    # FPL's own `last_deadline_total_transfers` for the imported entry -- how many transfers that
    # real manager has made this season. Account-level, not part of this app's own per-gameweek
    # plan transfer count (see `api.state.get_squad_transfers_made`), and only ever set by import.
    season_transfers_made: int = 0
