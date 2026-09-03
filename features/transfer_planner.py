"""Transfer suggestion for the Team page's banner (TRANSFER_BANNER): given the current squad, a
budget, and a manager-chosen number of transfers, which players to sell and buy.

**Why expected swing is not the ranking metric, despite being the thing a manager asks for.**
Your expected gap against the league is ``sum((your_multiplier - eo_multiplier) * xP)`` over every
player either side holds (``features.mini_league.compute_exposures``). A transfer changes your own
multipliers and nothing else: the field's effective ownership is a property of the rivals' squads,
which your transfer cannot touch. So every ``eo`` term cancels out of the difference, and

    delta expected swing = m_in * xP_in - m_out * xP_out = delta expected points

exactly. Ranking transfers by expected swing and ranking them by expected points are the same
ranking, and any UI presenting them as two separate criteria is showing one number twice.

Where a mini-league genuinely changes the answer is **variance**, not expectation. Two transfers
with identical expected points gain differ in how wide they leave the distribution of your gap
against each rival: buying a player your rivals already own narrows it, buying one they do not
widens it. A manager projected to finish behind wants it wide, one projected to finish ahead wants
it narrow, and that preference is what makes the league-aware suggestion differ from the plain
points-maximising one. :func:`~features.mini_league.compute_posture` already computes exactly
this per rival, so this module composes it rather than deriving anything new.

**The headline metric is therefore expected final league rank**, ``1 + sum over rivals of
P(rival finishes ahead of you)``, computed per rival against that rival's real 15 picks rather
than against an averaged field, so no fractional synthetic squad is ever invented. It trades
expected points against variance in whichever direction the manager's actual league position
rewards, and it is measured in a unit ("you finish 4.2nd instead of 5.1st") a manager can read
directly.

**Two stages, because the exact metric is not linear.** Expected final rank is a sum of normal
CDFs and cannot go into an integer linear program. So stage one asks
:func:`~features.squad_optimizer.optimise_squad` for the top ``n_plans`` distinct squads by
expected points within the transfer and budget limits (repeated solves with no-good cuts), and
stage two rescores each of them exactly on expected final rank and ranks by that. This is a
search over the points-best candidates, not a proof of rank-optimality: a plan that is rank-better
but outside the points top ``n_plans`` will not be found, and raising ``n_plans`` widens the
search at a linear cost in solve time. Stated here rather than implied, because the distinction
matters to anyone reading a suggestion as if it were optimal.

**Inherited approximations.** Everything :mod:`features.mini_league`'s docstring states about
independence across players and normality of the gap applies unchanged here, since this module
computes no distribution of its own. The central estimates are sound and the tails are optimistic.

Pure functions over already-resolved inputs, like every other module in this package: no I/O, no
gameweek resolution, no league fetching. ``api.transfer_panel`` does all of that once and hands
the results in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engine.data.league_state_builder import LeagueEntry
from engine.projections import PlayerGameweekProjection, PlayerHorizonProjection
from features.mini_league import (
    PlayerOwnership,
    compute_exposures,
    compute_head_to_head,
    compute_posture,
)
from features.squad_optimizer import (
    OptimizedSquad,
    PlayerCandidate,
    SquadOptimizerError,
    optimise_squad,
)
from features.squad_points import projected_points
from features.squad_rules import INITIAL_BUDGET
from features.team_state import MyTeamState, SquadPlayer

__all__ = [
    "DEFAULT_PLAN_COUNT",
    "RANK_TOLERANCE",
    "TransferMove",
    "TransferPlan",
    "TransferSuggestion",
    "pair_moves",
    "plan_transfers",
]

# How many distinct squads stage one enumerates before stage two rescores them. Five is enough
# for the rank-best plan to be among them in practice (transfers that are far apart on expected
# points are rarely close on rank), and each extra plan is one more CBC solve on the banner's
# critical path.
DEFAULT_PLAN_COUNT = 5

# How close two plans' expected final ranks must be before the ranking treats them as level and
# falls back to expected points. One hundredth of a league place is both the precision the banner
# displays and a floor well below this estimate's own error, so anything finer is noise being
# allowed to overrule a visible points difference. See :func:`_plan_key`.
RANK_TOLERANCE = 0.01


@dataclass(frozen=True)
class TransferMove:
    """One sell paired with one buy, at the same position. The pairing is presentational only:
    the solver chooses a set of players to drop and a set to add, and any same position pairing
    of them produces the identical final squad. Ordered by ``price_delta`` across a plan so that
    the moves which free money come first, which is also the order
    :func:`~features.squad_rules.transfer` could be applied in without a transient budget
    breach."""

    out_player_id: int
    in_player_id: int
    position: str
    price_delta: int


@dataclass(frozen=True)
class TransferPlan:
    """One candidate set of transfers, already rescored on the league metric.

    ``expected_points_delta`` is over the caller's planning gameweeks, captain and chip applied,
    straight from :func:`~features.squad_points.projected_points`, so it is the same number the
    Team page header would show after making these moves. Every league field
    (``expected_gap_delta``, ``gap_std_delta``, ``expected_final_rank``) is for the single
    league gameweek the caller resolved, matching :mod:`features.mini_league`'s own convention,
    and is ``0.0``/``1.0`` when there are no rivals to measure against.

    ``expected_final_rank_delta`` is new minus current, so **negative is an improvement**. It is
    the field this module ranks on.
    """

    out_player_ids: tuple[int, ...]
    in_player_ids: tuple[int, ...]
    moves: tuple[TransferMove, ...]
    n_transfers: int
    squad: tuple[SquadPlayer, ...]
    starting_xi: tuple[int, ...]
    bench_order: tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    expected_points: float
    expected_points_delta: float
    expected_gap: float
    expected_gap_delta: float
    gap_std: float
    gap_std_delta: float
    expected_final_rank: float
    expected_final_rank_delta: float
    spend_delta: int
    budget_remaining: int


@dataclass(frozen=True)
class TransferSuggestion:
    """Everything the banner renders in one object.

    ``plans`` is ranked best first at the requested transfer count.
    ``best_by_transfer_count`` holds the single best plan for each of 1, 2, ... transfers, so the
    banner can show what each additional move actually buys rather than only the total: this
    sandbox has no free-transfer count and charges no hit (see ``features.team_state``), so
    whether a second transfer is worth its real world cost is a judgement the manager makes from
    the marginal gain, not one this module makes for them.

    ``variance_preference`` is the aggregate of every rival's own posture, weighted by how
    sensitive each is: ``"increase"`` when widening your gap distribution lowers your expected
    final rank on net, ``"decrease"`` when narrowing it does, ``"neutral"`` when there are no
    rivals or the two sides cancel.
    """

    plans: tuple[TransferPlan, ...]
    best_by_transfer_count: tuple[TransferPlan, ...]
    max_transfers: int
    current_expected_points: float
    current_expected_gap: float
    current_gap_std: float
    current_expected_final_rank: float
    variance_preference: str
    n_rivals: int
    gameweeks: tuple[int, ...]
    league_gameweek: int


def pair_moves(
    out_players: Sequence[SquadPlayer], in_players: Sequence[SquadPlayer]
) -> tuple[TransferMove, ...]:
    """Pair each sold player with a bought player at the same position, most expensive with most
    expensive, then order the resulting moves by ``price_delta`` ascending.

    Position counts always match, since both squads satisfy the same fixed quota, so no player is
    ever left unpaired. Raises ``ValueError`` if they somehow do not, rather than silently
    dropping a move and reporting a plan that does not describe the squad it came with.
    """
    out_by_position: dict[str, list[SquadPlayer]] = {}
    in_by_position: dict[str, list[SquadPlayer]] = {}
    for player in out_players:
        out_by_position.setdefault(player.position, []).append(player)
    for player in in_players:
        in_by_position.setdefault(player.position, []).append(player)

    if {p: len(v) for p, v in out_by_position.items()} != {
        p: len(v) for p, v in in_by_position.items()
    }:
        raise ValueError("out_players and in_players do not match position for position")

    moves: list[TransferMove] = []
    for position, outgoing in out_by_position.items():
        incoming = in_by_position[position]
        outgoing = sorted(outgoing, key=lambda p: (-p.price, p.player_id))
        incoming = sorted(incoming, key=lambda p: (-p.price, p.player_id))
        for out_player, in_player in zip(outgoing, incoming, strict=True):
            moves.append(
                TransferMove(
                    out_player_id=out_player.player_id,
                    in_player_id=in_player.player_id,
                    position=position,
                    price_delta=in_player.price - out_player.price,
                )
            )

    return tuple(sorted(moves, key=lambda m: (m.price_delta, m.out_player_id)))


@dataclass(frozen=True)
class _LeagueScore:
    expected_gap: float
    gap_std: float
    expected_final_rank: float
    variance_signal: float


def _score_against_league(
    state: MyTeamState,
    ownership_by_player: Mapping[int, PlayerOwnership],
    rivals: Sequence[LeagueEntry],
    gameweek_projections: Mapping[int, PlayerGameweekProjection],
    my_total_points: int,
    gameweeks_remaining: int,
    chip: str | None,
) -> _LeagueScore:
    """One squad's whole league picture for the resolved gameweek.

    ``expected_gap`` is against the field's effective ownership (the sum of
    :func:`~features.mini_league.compute_exposures`' own per player swings), which is the number a
    manager recognises as "how far ahead of the league am I set up to be this week."
    ``gap_std``/``expected_final_rank`` come from a per rival head to head instead, since a
    rival's real 15 picks give an exact variance where an averaged field would only give a
    synthetic one.

    ``variance_signal`` is the net of every rival's sensitivity, signed by whether more spread
    helps or hurts against that rival: positive means widening the distribution lowers expected
    final rank on net.
    """
    exposure_ids = set(state.player_ids) | set(ownership_by_player)
    expected_gap = sum(
        exposure.expected_swing or 0.0
        for exposure in compute_exposures(
            exposure_ids, state, ownership_by_player, gameweek_projections, chip=chip
        )
    )

    if not rivals:
        return _LeagueScore(
            expected_gap=expected_gap,
            gap_std=0.0,
            expected_final_rank=1.0,
            variance_signal=0.0,
        )

    total_std = 0.0
    expected_final_rank = 1.0
    variance_signal = 0.0
    for rival in rivals:
        head_to_head = compute_head_to_head(state, rival, gameweek_projections, chip=chip)
        posture = compute_posture(my_total_points, rival, head_to_head, gameweeks_remaining)
        total_std += head_to_head.gap_std
        expected_final_rank += 1.0 - posture.p_finish_ahead
        if posture.projected_final_gap < 0:
            variance_signal += posture.sensitivity
        elif posture.projected_final_gap > 0:
            variance_signal -= posture.sensitivity

    return _LeagueScore(
        expected_gap=expected_gap,
        gap_std=total_std / len(rivals),
        expected_final_rank=expected_final_rank,
        variance_signal=variance_signal,
    )


def _plan_key(plan: TransferPlan, rank_tolerance: float = RANK_TOLERANCE) -> tuple[float, float]:
    """Expected final rank ascending, expected points descending as the tie break, with the rank
    first rounded to ``rank_tolerance``.

    The rounding is the load-bearing part. A manager comfortably clear of the field (or hopelessly
    behind it) has an expected final rank that barely moves whatever they do, so an unrounded
    comparison lets a rank difference of a ten-thousandth of a place, far below both the displayed
    precision and this estimate's own error, outrank a points difference the manager can plainly
    see. Rounding first means the league only overrides points when it has something real to say.

    With no rivals every plan carries the same rank of 1.0, so the tie break becomes the whole
    ranking and this degrades cleanly to "best expected points" for a manager with no league.
    """
    bucket = round(plan.expected_final_rank / rank_tolerance) if rank_tolerance > 0 else 0.0
    return (bucket, -plan.expected_points_delta)


def plan_transfers(
    team_state: MyTeamState,
    candidates: Sequence[PlayerCandidate],
    horizon_projections: Mapping[int, PlayerHorizonProjection],
    gameweeks: Sequence[int],
    gameweek_projections: Mapping[int, PlayerGameweekProjection],
    ownership_by_player: Mapping[int, PlayerOwnership],
    rivals: Sequence[LeagueEntry],
    league_gameweek: int = 0,
    my_total_points: int = 0,
    gameweeks_remaining: int = 0,
    budget: int = INITIAL_BUDGET,
    max_transfers: int = 1,
    chip: str | None = None,
    n_plans: int = DEFAULT_PLAN_COUNT,
    objective: str = "starting_xi",
) -> TransferSuggestion:
    """Rank transfer plans of up to ``max_transfers`` moves for ``team_state``.

    ``candidates`` is the pool the solver may buy from, each already carrying his expected points
    summed over ``gameweeks``, which is what the integer program maximizes.
    ``gameweek_projections`` and ``ownership_by_player`` are for the single gameweek the league
    math is resolved to, and ``rivals`` excludes the manager's own entry.

    Returns a suggestion whose ``plans`` may be empty, never raising, when nothing legal exists at
    this budget and transfer count: a banner that shows "no move improves this" is a real answer,
    and an exception here would take down a page that works perfectly well without a suggestion.
    """
    if max_transfers < 1:
        raise ValueError(f"max_transfers must be at least 1, got {max_transfers}")

    current_ids = frozenset(team_state.player_ids)
    price_by_player = {player.player_id: player.price for player in team_state.squad}
    current_spend = sum(price_by_player.values())

    current_points = projected_points(team_state, horizon_projections, gameweeks, chip=chip).total
    current_league = _score_against_league(
        team_state,
        ownership_by_player,
        rivals,
        gameweek_projections,
        my_total_points,
        gameweeks_remaining,
        chip,
    )

    def build(n: int, count: int) -> list[TransferPlan]:
        plans: list[TransferPlan] = []
        excluded: list[frozenset[int]] = []
        # One extra attempt, since the very first solve can legitimately come back as the current
        # squad itself (no transfer beats every transfer available), which is not a plan but also
        # must not cost the caller one of the plans they asked for.
        for _ in range(count + 1):
            if len(plans) == count:
                break
            try:
                result = optimise_squad(
                    candidates,
                    objective=objective,
                    budget=budget,
                    current_squad_ids=current_ids,
                    max_transfers=n,
                    excluded_squads=excluded,
                )
            except SquadOptimizerError:
                break
            new_ids = frozenset(player.player_id for player in result.squad)
            excluded.append(new_ids)
            if new_ids == current_ids:
                # The optimum at this transfer count is to make no transfer at all. Nothing to
                # suggest, but the next solve is still worth trying: the no-good cut just added
                # forbids exactly the current squad, so it returns the best real move instead.
                continue
            plan = _build_plan(
                result,
                team_state,
                current_ids,
                price_by_player,
                current_spend,
                budget,
                horizon_projections,
                gameweeks,
                gameweek_projections,
                ownership_by_player,
                rivals,
                my_total_points,
                gameweeks_remaining,
                chip,
                current_points,
                current_league,
            )
            plans.append(plan)
        return sorted(plans, key=_plan_key)

    plans = build(max_transfers, n_plans)
    best_by_count: list[TransferPlan] = []
    for n in range(1, max_transfers + 1):
        at_n = build(n, 1) if n != max_transfers else plans[:1]
        if at_n:
            best_by_count.append(at_n[0])

    if current_league.variance_signal > 0:
        variance_preference = "increase"
    elif current_league.variance_signal < 0:
        variance_preference = "decrease"
    else:
        variance_preference = "neutral"

    return TransferSuggestion(
        plans=tuple(plans),
        best_by_transfer_count=tuple(best_by_count),
        max_transfers=max_transfers,
        current_expected_points=current_points,
        current_expected_gap=current_league.expected_gap,
        current_gap_std=current_league.gap_std,
        current_expected_final_rank=current_league.expected_final_rank,
        variance_preference=variance_preference,
        n_rivals=len(rivals),
        gameweeks=tuple(gameweeks),
        league_gameweek=league_gameweek,
    )


def _build_plan(
    result: OptimizedSquad,
    team_state: MyTeamState,
    current_ids: frozenset[int],
    price_by_player: Mapping[int, int],
    current_spend: int,
    budget: int,
    horizon_projections: Mapping[int, PlayerHorizonProjection],
    gameweeks: Sequence[int],
    gameweek_projections: Mapping[int, PlayerGameweekProjection],
    ownership_by_player: Mapping[int, PlayerOwnership],
    rivals: Sequence[LeagueEntry],
    my_total_points: int,
    gameweeks_remaining: int,
    chip: str | None,
    current_points: float,
    current_league: _LeagueScore,
) -> TransferPlan:
    """Turn one solver result into a fully scored :class:`TransferPlan`. Split out from
    :func:`plan_transfers` only so the enumeration loop there reads as the search it is."""
    new_ids = frozenset(player.player_id for player in result.squad)
    out_players = tuple(player for player in team_state.squad if player.player_id not in new_ids)
    in_players = tuple(player for player in result.squad if player.player_id not in current_ids)

    new_state = MyTeamState(
        squad=result.squad,
        starting_xi=result.starting_xi,
        bench_order=result.bench_order,
        captain_id=result.captain_id,
        vice_captain_id=result.vice_captain_id,
        mini_league_ids=team_state.mini_league_ids,
    )
    points = projected_points(new_state, horizon_projections, gameweeks, chip=chip).total
    league = _score_against_league(
        new_state,
        ownership_by_player,
        rivals,
        gameweek_projections,
        my_total_points,
        gameweeks_remaining,
        chip,
    )

    new_spend = sum(player.price for player in result.squad)
    return TransferPlan(
        out_player_ids=tuple(player.player_id for player in out_players),
        in_player_ids=tuple(player.player_id for player in in_players),
        moves=pair_moves(out_players, in_players),
        n_transfers=len(out_players),
        squad=result.squad,
        starting_xi=result.starting_xi,
        bench_order=result.bench_order,
        captain_id=result.captain_id,
        vice_captain_id=result.vice_captain_id,
        expected_points=points,
        expected_points_delta=points - current_points,
        expected_gap=league.expected_gap,
        expected_gap_delta=league.expected_gap - current_league.expected_gap,
        gap_std=league.gap_std,
        gap_std_delta=league.gap_std - current_league.gap_std,
        expected_final_rank=league.expected_final_rank,
        expected_final_rank_delta=league.expected_final_rank - current_league.expected_final_rank,
        spend_delta=new_spend - current_spend,
        budget_remaining=budget - new_spend,
    )
