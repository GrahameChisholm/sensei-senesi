"""Top-level entry point: player -> full projection (distribution + component breakdown).

Ties together the deterministic aggregation (2.7) and the distributional simulation (2.9) into
one object per player per gameweek, and rolls several gameweeks up into a planning-horizon view
(BUILD_PLAN 2.7's "5 gameweeks by default"). The per-component breakdown stays attached to the
output — the web app's "detail on click" view and debugging both depend on being able to see what
drove a number (BUILD_PLAN 2.7).

**Definition of done for Phase 2** (BUILD_PLAN): the engine produces, for every player, a
per-gameweek and multi-gameweek projection with a full distribution and an attached component
breakdown, and it runs end-to-end on a historical snapshot without leakage. It is *not yet
validated* — that's Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.aggregate import DEFAULT_PLANNING_HORIZON, ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.simulate import PlayerSimulationSummary

__all__ = [
    "DEFAULT_PLANNING_HORIZON",
    "PlayerGameweekProjection",
    "PlayerHorizonProjection",
    "project_player_gameweek",
    "project_player_horizon",
]


@dataclass(frozen=True)
class PlayerGameweekProjection:
    """One player's full projection for one gameweek: the minutes distribution that gated every
    component, the deterministic per-component breakdown (2.7), and — when a full-fixture Monte
    Carlo run (2.9) was performed for this gameweek — the resulting outcome distribution."""

    player_id: int
    position: str
    gameweek: int
    minutes: MinutesDistribution
    breakdown: ComponentBreakdown
    simulation: PlayerSimulationSummary | None = None
    # ENGINE_IMPROVEMENTS_5.md Tier 2.1: E[points | plays 60+], the same component chain re-run
    # under "they definitely start" (``engine.pipeline._plays_60_counterfactual``). ``None`` when
    # the producing path didn't compute it, e.g. a cold-start baseline or a hand-built test
    # projection, matching how ``simulation`` is optional for the same reason.
    conditional_expected_points: float | None = None

    @property
    def expected_points(self) -> float:
        return self.breakdown.total

    @property
    def points_if_they_play(self) -> float:
        """``conditional_expected_points`` when it was computed, otherwise ``expected_points``.

        Ranking a shortlist on this rather than on ``expected_points`` removes the availability
        confound: ``expected_points`` is E[points | plays] multiplied by P(plays), so it scores a
        merely-adequate certain starter above an excellent rotation risk, which is not the
        comparison a manager is making once they have already decided to field someone. Falling
        back rather than raising keeps a mixed pool (engine projections alongside cold-start
        baselines) sortable on one key.
        """
        if self.conditional_expected_points is None:
            return self.breakdown.total
        return self.conditional_expected_points


def project_player_gameweek(
    player_id: int,
    position: str,
    gameweek: int,
    minutes: MinutesDistribution,
    breakdown: ComponentBreakdown,
    simulation: PlayerSimulationSummary | None = None,
    conditional_expected_points: float | None = None,
) -> PlayerGameweekProjection:
    """Combine one gameweek's minutes distribution and component breakdown (already computed by
    ``engine.aggregate.aggregate_gameweek``) into one player-facing projection, optionally
    attaching that gameweek's simulated outcome distribution and its conditional
    ``E[points | plays]`` (Tier 2.1)."""
    if simulation is not None and simulation.player_id != player_id:
        raise ValueError(
            f"simulation.player_id ({simulation.player_id}) does not match player_id ({player_id})"
        )
    return PlayerGameweekProjection(
        player_id=player_id,
        position=position,
        gameweek=gameweek,
        minutes=minutes,
        breakdown=breakdown,
        simulation=simulation,
        conditional_expected_points=conditional_expected_points,
    )


@dataclass(frozen=True)
class PlayerHorizonProjection:
    """A player's projection rolled up over the planning horizon — one
    :class:`PlayerGameweekProjection` per gameweek, keyed by gameweek number."""

    player_id: int
    position: str
    gameweeks: dict[int, PlayerGameweekProjection]

    def __post_init__(self) -> None:
        if not self.gameweeks:
            raise ValueError("gameweeks must not be empty")
        for gw_number, projection in self.gameweeks.items():
            if projection.player_id != self.player_id:
                raise ValueError(
                    f"gameweek {gw_number}: projection.player_id ({projection.player_id}) does "
                    f"not match player_id ({self.player_id})"
                )
            if projection.position != self.position:
                raise ValueError(
                    f"gameweek {gw_number}: projection.position ({projection.position!r}) does "
                    f"not match position ({self.position!r})"
                )

    @property
    def per_gameweek_points(self) -> dict[int, float]:
        return {gw_number: p.expected_points for gw_number, p in self.gameweeks.items()}

    @property
    def horizon_total_points(self) -> float:
        return sum(p.expected_points for p in self.gameweeks.values())


def project_player_horizon(
    player_id: int, position: str, gameweeks: dict[int, PlayerGameweekProjection]
) -> PlayerHorizonProjection:
    """Roll several gameweeks' projections up into one planning-horizon view (BUILD_PLAN 2.7 —
    :data:`DEFAULT_PLANNING_HORIZON` gameweeks by default, though the caller decides how many
    gameweek projections to actually pass in)."""
    return PlayerHorizonProjection(
        player_id=player_id, position=position, gameweeks=dict(gameweeks)
    )
