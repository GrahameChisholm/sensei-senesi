"""The one live sandbox squad the team-selection page's endpoints read and write — 0 to 15
players, never locked in, no confirm step. Kept separate from ``features.team_state.MyTeamState``
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
    # default, or a higher figure recorded at import time for a real squad whose current value
    # exceeds it. Reset back to INITIAL_BUDGET when the squad is cleared.
    budget_ceiling: int = INITIAL_BUDGET
