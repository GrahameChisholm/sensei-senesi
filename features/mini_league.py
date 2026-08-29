"""Mini-league math (MINI_LEAGUE_PLAN Phase 3): every number the Mini League page and the
Differentials page's league-ownership lens need, computed purely over an already-fetched
:class:`~engine.data.league_state_builder.LeagueSnapshot` and already-resolved engine projections.
No I/O, no HTTP, no FPL rule logic beyond what a plain arithmetic read of a snapshot requires --
matching every other module in this package.

**Every function here takes a gameweek-resolved projection map**
(``Mapping[int, PlayerGameweekProjection]``), not the multi-gameweek horizon, mirroring
``features.captaincy.rank_captaincy_pool``'s own convention: the caller resolves "which gameweek"
once, and every function below is then just a lookup, never a second gameweek parameter to
thread through.

**Two approximations, both load-bearing for the head-to-head math (M9), stated here so they are
never silently assumed correct elsewhere:**

1. **Independence across players.** A rival's differential gap is treated as a sum of independent
   player outcomes. This understates the true variance whenever a manager's differential and a
   rival's differential are in the same match (their outcomes are then genuinely correlated) -- a
   joint simulation would fix it and would require re-running the engine at request time, which
   this module deliberately does not do.
2. **Normality of the gap.** ``p_outscore``/``p_finish_ahead`` apply a normal CDF to a sum of a
   handful of skewed per-player point distributions. The central estimate is sound; the tails are
   optimistic. Both approximations live entirely inside this module (:func:`_normal_cdf` and the
   variance sums below), so the natural upgrade -- sampling the gap from persisted per-player
   quantiles instead of assuming normality -- is a swap of these functions' internals with no
   change to their signatures, the API layer, or the UI.

**Effective ownership excludes the caller's own entry (M6).** ``compute_league_ownership`` takes an
``exclude_entry_id`` precisely so "the league" means "the field you're being measured against," not
"the league including yourself."

**Multiplier is read as-is (M3).** Every function below trusts
:class:`~engine.data.league_state_builder.LeagueEntry.picks`' multiplier values completely --
FPL has already applied every chip effect to them, so nothing here re-derives a captaincy or
bench-boost effect from ``active_chip``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from engine.data.league_state_builder import LeagueEntry, LeagueSnapshot
from engine.projections import PlayerGameweekProjection
from features.squad_points import CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN, CHIPS
from features.team_state import MyTeamState

__all__ = [
    "KNOWN_CHIPS",
    "PlayerOwnership",
    "PlayerExposure",
    "DifferentialPick",
    "HeadToHead",
    "CaptainOption",
    "RivalChipState",
    "RivalPosture",
    "compute_league_ownership",
    "compute_exposures",
    "prospective_swing",
    "compute_head_to_head",
    "rank_captain_options",
    "compute_chip_states",
    "compute_posture",
    "league_template_xi",
    "compute_coverage",
]

# FPL's own internal chip name strings, carried through verbatim rather than mapped onto a fixed
# enum (M11) -- an unrecognised name (a future rules change adding or renaming a chip) is still
# displayed via RivalChipState.used_chip_names, just never counted as "remaining" for a chip this
# constant doesn't know about. Distinct from features.squad_points.CHIPS, which is the much
# smaller "what a manager can toggle in this sandbox" set -- rivals play the real game and can hold
# any of these four.
KNOWN_CHIPS = ("wildcard", "freehit", "bboost", "3xc")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _your_multipliers(team_state: MyTeamState, chip: str | None = None) -> dict[int, float]:
    """Every player in ``team_state``'s own multiplier under ``chip`` -- captain doubled (or
    tripled under Triple Captain), bench at 0 unless Bench Boost is being previewed, mirroring
    ``features.squad_points.projected_points``'s own captain/bench handling exactly, since this is
    the same rule applied to one player at a time instead of summed into a total."""
    if chip is not None and chip not in CHIPS:
        raise ValueError(f"unknown chip: {chip!r}")
    captain_multiplier = 3.0 if chip == CHIP_TRIPLE_CAPTAIN else 2.0
    include_bench = chip == CHIP_BENCH_BOOST

    multipliers: dict[int, float] = {}
    for player_id in team_state.starting_xi:
        multipliers[player_id] = captain_multiplier if player_id == team_state.captain_id else 1.0
    for player_id in team_state.bench_order:
        multipliers[player_id] = 1.0 if include_bench else 0.0
    return multipliers


def _expected_points(
    player_id: int, projections: Mapping[int, PlayerGameweekProjection]
) -> float | None:
    projection = projections.get(player_id)
    return None if projection is None else projection.expected_points


def _std(player_id: int, projections: Mapping[int, PlayerGameweekProjection]) -> float:
    """0.0 when no simulation was run for this player this gameweek (a cold-start baseline, or a
    projection built without the Monte Carlo pass) -- treating an un-simulated player as
    deterministic degrades the head-to-head variance gracefully rather than raising, matching this
    module's "a real answer, never a crash" stance on missing data."""
    projection = projections.get(player_id)
    if projection is None or projection.simulation is None or projection.simulation.std is None:
        return 0.0
    return projection.simulation.std


@dataclass(frozen=True)
class PlayerOwnership:
    """One player's ownership across the league, excluding whichever entry the caller passed as
    ``exclude_entry_id`` to :func:`compute_league_ownership` (M6). Three independent numbers, never
    blended into one: ``raw_ownership_percent`` says who is exposed to a price rise,
    ``eo_multiplier``/``eo_percent`` say who is exposed to a haul, ``captain_share_percent`` is
    itself an input to :func:`rank_captain_options`. ``owner_count`` is the plain integer
    ``raw_ownership_percent`` is a percentage of -- kept as its own exact field (rather than making
    a caller reconstruct it via ``round(raw_ownership_percent / 100 * n_rivals)``) since the
    Differentials page's league lens (MINI_LEAGUE_PLAN M28) needs an exact "owned by N of M
    rivals" count, not a percentage.
    """

    player_id: int
    raw_ownership_percent: float
    owner_count: int
    eo_multiplier: float
    eo_percent: float
    captain_share_percent: float
    owner_names: tuple[str, ...]


_ZERO_OWNERSHIP_TEMPLATE = PlayerOwnership(
    player_id=-1,
    raw_ownership_percent=0.0,
    owner_count=0,
    eo_multiplier=0.0,
    eo_percent=0.0,
    captain_share_percent=0.0,
    owner_names=(),
)


def _ownership_or_zero(
    player_id: int, ownership_by_player: Mapping[int, PlayerOwnership]
) -> PlayerOwnership:
    """A player nobody in the league owns simply has no entry in the ownership map -- not an
    error, just "zero across the board" (M6's per-player dataclasses are only ever built for
    players at least one rival owns, per :func:`compute_league_ownership`)."""
    ownership = ownership_by_player.get(player_id)
    if ownership is not None:
        return ownership
    return replace(_ZERO_OWNERSHIP_TEMPLATE, player_id=player_id)


def compute_league_ownership(
    snapshot: LeagueSnapshot, exclude_entry_id: int | None = None
) -> dict[int, PlayerOwnership]:
    """One :class:`PlayerOwnership` for every player owned by at least one rival (M6).
    ``exclude_entry_id`` is normally the caller's own entry, so the returned map answers "what does
    the field look like," not "what does the field including me look like." Returns an empty dict
    when there are no rivals at all (a league of one, or an entirely empty snapshot) -- there is no
    field to measure against, matching :func:`~features.fixture_swing.build_fixture_swing_rows`'s
    own "no baseline, no rows" handling of the equivalent preseason case.
    """
    rivals = tuple(entry for entry in snapshot.entries if entry.entry_id != exclude_entry_id)
    if not rivals:
        return {}

    n_rivals = float(len(rivals))
    player_ids = {player_id for rival in rivals for player_id in rival.picks}

    ownership: dict[int, PlayerOwnership] = {}
    for player_id in player_ids:
        owners = [rival for rival in rivals if player_id in rival.picks]
        eo_multiplier = sum(rival.picks.get(player_id, 0) for rival in rivals) / n_rivals
        captains = sum(1 for rival in owners if rival.picks[player_id] >= 2)
        ownership[player_id] = PlayerOwnership(
            player_id=player_id,
            raw_ownership_percent=100.0 * len(owners) / n_rivals,
            owner_count=len(owners),
            eo_multiplier=eo_multiplier,
            eo_percent=eo_multiplier * 100.0,
            captain_share_percent=100.0 * captains / n_rivals,
            owner_names=tuple(rival.manager_name for rival in owners),
        )
    return ownership


@dataclass(frozen=True)
class PlayerExposure:
    """One player's exposure (M7): how much your own points multiplier differs from the league's
    effective ownership. ``expected_points``/``expected_swing`` are ``None`` when this player has
    no projection for the resolved gameweek (a blank gameweek, or a cold-start gap) -- never a
    fabricated zero, matching ``features.squad_points.projected_points``'s own missing-projection
    discipline."""

    player_id: int
    your_multiplier: float
    ownership: PlayerOwnership
    expected_points: float | None
    exposure: float
    expected_swing: float | None


def compute_exposures(
    player_ids: Iterable[int],
    team_state: MyTeamState,
    ownership_by_player: Mapping[int, PlayerOwnership],
    projections: Mapping[int, PlayerGameweekProjection],
    chip: str | None = None,
) -> list[PlayerExposure]:
    """One :class:`PlayerExposure` per ``player_id`` in ``player_ids``, unsorted (matching this
    package's universal "the frontend sorts" convention). ``player_ids`` is supplied by the caller
    rather than inferred here, so the same function serves both the Mini League page's own exposure
    table (your squad plus every league-owned player above some threshold) and the Differentials
    page's league lens (Phase 7) without this module needing to know which caller it's serving.
    """
    your_multipliers = _your_multipliers(team_state, chip)
    results: list[PlayerExposure] = []
    for player_id in player_ids:
        your_multiplier = your_multipliers.get(player_id, 0.0)
        ownership = _ownership_or_zero(player_id, ownership_by_player)
        expected_points = _expected_points(player_id, projections)
        exposure = your_multiplier - ownership.eo_multiplier
        expected_swing = None if expected_points is None else exposure * expected_points
        results.append(
            PlayerExposure(
                player_id=player_id,
                your_multiplier=your_multiplier,
                ownership=ownership,
                expected_points=expected_points,
                exposure=exposure,
                expected_swing=expected_swing,
            )
        )
    return results


def prospective_swing(eo_multiplier: float, expected_points: float) -> float:
    """What a player *would* be worth to you, in the same expected-swing units as
    :class:`PlayerExposure`, if you brought him into your starting XI (multiplier 1) -- the
    Differentials page's league lens (MINI_LEAGUE_PLAN M28) uses this for a player not currently
    owned, since ``compute_exposures``' own ``exposure`` answers "how is my *current* squad doing,"
    not "is this differential worth buying." Deliberately a separate, simpler function rather than
    a mode flag on :func:`compute_exposures`: the two questions have different callers and
    different meanings for "your multiplier."""
    return (1.0 - eo_multiplier) * expected_points


@dataclass(frozen=True)
class DifferentialPick:
    """One player where a head-to-head opponent's pick differs from your own (M8).
    ``expected_gap_contribution`` is ``0.0`` (not ``None``) when ``expected_points`` is missing --
    a genuinely unquantifiable player is still shown (you own him, or they do, and that fact is
    real), it simply cannot move the computed gap."""

    player_id: int
    your_multiplier: float
    rival_multiplier: float
    expected_points: float | None
    expected_gap_contribution: float


@dataclass(frozen=True)
class HeadToHead:
    """One rival's full head-to-head decomposition for one gameweek (M8/M9). ``shared_count`` is
    the number of players where both sides hold the exact same multiplier -- shown only as a count,
    since by construction it contributes nothing to ``expected_gap``. ``p_outscore`` is this
    gameweek's probability of a strictly positive gap under the module docstring's two stated
    approximations (independence, normality) -- render it banded in the UI, not to a decimal place
    it hasn't earned."""

    rival_entry_id: int
    shared_count: int
    differentials: tuple[DifferentialPick, ...]
    expected_gap: float
    gap_std: float
    p_outscore: float


def compute_head_to_head(
    team_state: MyTeamState,
    rival: LeagueEntry,
    projections: Mapping[int, PlayerGameweekProjection],
    chip: str | None = None,
) -> HeadToHead:
    """Partition the union of your squad and ``rival``'s 15 into shared picks (same multiplier on
    both sides, contributing nothing) and differentials, then sum the differentials' contributions
    and, independently, their variance (M9's stated independence approximation) into one gap
    distribution."""
    your_multipliers = _your_multipliers(team_state, chip)
    player_ids = set(your_multipliers) | set(rival.picks)

    shared_count = 0
    differentials: list[DifferentialPick] = []
    expected_gap = 0.0
    gap_variance = 0.0

    for player_id in player_ids:
        your_multiplier = your_multipliers.get(player_id, 0.0)
        rival_multiplier = float(rival.picks.get(player_id, 0))
        if your_multiplier == rival_multiplier:
            shared_count += 1
            continue

        expected_points = _expected_points(player_id, projections)
        contribution = (
            0.0
            if expected_points is None
            else (your_multiplier - rival_multiplier) * expected_points
        )
        expected_gap += contribution
        gap_variance += ((your_multiplier - rival_multiplier) ** 2) * (
            _std(player_id, projections) ** 2
        )
        differentials.append(
            DifferentialPick(
                player_id=player_id,
                your_multiplier=your_multiplier,
                rival_multiplier=rival_multiplier,
                expected_points=expected_points,
                expected_gap_contribution=contribution,
            )
        )

    gap_std = math.sqrt(gap_variance)
    if gap_std > 0:
        p_outscore = _normal_cdf(expected_gap / gap_std)
    else:
        p_outscore = 1.0 if expected_gap > 0 else (0.0 if expected_gap < 0 else 0.5)

    return HeadToHead(
        rival_entry_id=rival.entry_id,
        shared_count=shared_count,
        differentials=tuple(differentials),
        expected_gap=expected_gap,
        gap_std=gap_std,
        p_outscore=p_outscore,
    )


@dataclass(frozen=True)
class CaptainOption:
    """One captaincy candidate re-ranked by gain on the field, not raw expected points (M10).
    ``net_captain_ev``/``net_captain_std`` are ``None`` when this player has no projection this
    gameweek, matching every other optional-projection field in this module."""

    player_id: int
    expected_points: float | None
    captain_share_percent: float
    eo_multiplier: float
    net_captain_ev: float | None
    net_captain_std: float | None


def rank_captain_options(
    player_ids: Iterable[int],
    ownership_by_player: Mapping[int, PlayerOwnership],
    projections: Mapping[int, PlayerGameweekProjection],
) -> list[CaptainOption]:
    """One :class:`CaptainOption` per ``player_id``, unsorted (the ``rank_`` prefix names what the
    caller does with this, not a promise this function sorts -- matching
    ``features.fixture_swing.rank_team_swings``'s own "unsorted" precedent for the same naming).

    ``net_captain_ev(p) = (2 - eo_multiplier(p)) * xP(p)``: captaining a player the league already
    captains heavily is close to a null move (at ``eo_multiplier`` above 2, it's a net loss, since
    rivals tripling him outpaces you doubling him), while a low-EO captain nets close to his full
    xP again on top of the field. This is the number the existing Team page has no equivalent of.
    """
    results: list[CaptainOption] = []
    for player_id in player_ids:
        ownership = _ownership_or_zero(player_id, ownership_by_player)
        expected_points = _expected_points(player_id, projections)
        net_multiplier = 2.0 - ownership.eo_multiplier
        net_ev = None if expected_points is None else net_multiplier * expected_points
        net_std = None if expected_points is None else net_multiplier * _std(player_id, projections)
        results.append(
            CaptainOption(
                player_id=player_id,
                expected_points=expected_points,
                captain_share_percent=ownership.captain_share_percent,
                eo_multiplier=ownership.eo_multiplier,
                net_captain_ev=net_ev,
                net_captain_std=net_std,
            )
        )
    return results


@dataclass(frozen=True)
class RivalChipState:
    """One rival's chip usage against the known roster (M11). ``used_chip_names``/
    ``remaining_chip_names`` are drawn from :data:`KNOWN_CHIPS` -- a chip this constant doesn't
    recognise is still counted in ``used_chip_names`` (nothing from the snapshot is ever dropped)
    but never appears in ``remaining_chip_names``, since "remaining" can only be reasoned about
    against a known roster."""

    entry_id: int
    used_chip_names: tuple[str, ...]
    remaining_chip_names: tuple[str, ...]


def compute_chip_states(
    entries: Iterable[LeagueEntry], known_chips: Sequence[str] = KNOWN_CHIPS
) -> list[RivalChipState]:
    states: list[RivalChipState] = []
    for entry in entries:
        used = tuple(chip.name for chip in entry.chips)
        remaining = tuple(name for name in known_chips if name not in used)
        states.append(
            RivalChipState(
                entry_id=entry.entry_id, used_chip_names=used, remaining_chip_names=remaining
            )
        )
    return states


@dataclass(frozen=True)
class RivalPosture:
    """Whether variance helps or hurts against one rival, and by how much (M12). ``sensitivity``
    is ``|∂p_finish_ahead/∂gap_std|`` at the current gap_std/gameweeks_remaining -- a caller can use
    it to grey out rivals who are mathematically settled (very small sensitivity, extreme
    ``p_finish_ahead``) rather than competing for attention with the two or three rivals still
    genuinely live."""

    rival_entry_id: int
    projected_final_gap: float
    p_finish_ahead: float
    variance_preference: str  # "increase" | "decrease" | "neutral"
    sensitivity: float


def compute_posture(
    your_total_points: int,
    rival: LeagueEntry,
    head_to_head: HeadToHead,
    gameweeks_remaining: int,
) -> RivalPosture:
    """``projected_final_gap = -deficit + gameweeks_remaining * expected_gap``, where ``deficit``
    is how many points behind this rival you currently are. The variance recommendation falls
    straight out of the sign of ``projected_final_gap`` (see the module docstring's derivative
    argument): projected to finish behind means increasing variance helps, projected to finish
    ahead means it hurts, and there is nothing to tune beyond that sign.
    """
    deficit = rival.total_points - your_total_points
    projected_final_gap = -deficit + gameweeks_remaining * head_to_head.expected_gap

    spread = (
        head_to_head.gap_std * math.sqrt(gameweeks_remaining) if gameweeks_remaining > 0 else 0.0
    )
    if spread > 0:
        z = projected_final_gap / spread
        p_finish_ahead = _normal_cdf(z)
        sensitivity = _normal_pdf(z) * abs(projected_final_gap) / (spread * spread)
    else:
        p_finish_ahead = (
            1.0 if projected_final_gap > 0 else (0.0 if projected_final_gap < 0 else 0.5)
        )
        sensitivity = 0.0

    if projected_final_gap < 0:
        variance_preference = "increase"
    elif projected_final_gap > 0:
        variance_preference = "decrease"
    else:
        variance_preference = "neutral"

    return RivalPosture(
        rival_entry_id=rival.entry_id,
        projected_final_gap=projected_final_gap,
        p_finish_ahead=p_finish_ahead,
        variance_preference=variance_preference,
        sensitivity=sensitivity,
    )


def league_template_xi(
    ownership_by_player: Mapping[int, PlayerOwnership], n: int = 11
) -> tuple[int, ...]:
    """The ``n`` highest-``eo_multiplier`` players in the league (M13) -- ties broken by
    ``player_id`` for a deterministic result, since an arbitrary dict-iteration-order tiebreak
    would make this flicker between otherwise-identical calls."""
    ranked = sorted(
        ownership_by_player.values(),
        key=lambda ownership: (-ownership.eo_multiplier, ownership.player_id),
    )
    return tuple(ownership.player_id for ownership in ranked[:n])


def compute_coverage(
    team_state: MyTeamState,
    ownership_by_player: Mapping[int, PlayerOwnership],
    chip: str | None = None,
) -> float:
    """``Σ min(your_mult, eo_mult) / Σ eo_mult`` over every player either you or the league holds
    (M13) -- near 1.0 means you are running the league's template and your rank is close to frozen;
    near 0.0 means you are fully decorrelated. Returns ``0.0`` when the league has no ownership at
    all (a league of one, or before the first deadline), since there is then no template to be
    covering."""
    your_multipliers = _your_multipliers(team_state, chip)
    player_ids = set(your_multipliers) | set(ownership_by_player)

    numerator = 0.0
    denominator = 0.0
    for player_id in player_ids:
        your_multiplier = your_multipliers.get(player_id, 0.0)
        eo_multiplier = _ownership_or_zero(player_id, ownership_by_player).eo_multiplier
        numerator += min(your_multiplier, eo_multiplier)
        denominator += eo_multiplier

    return numerator / denominator if denominator > 0 else 0.0
