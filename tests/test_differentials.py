"""Tests for features/differentials.py (DIFFERENTIALS_PLAN Phase 2) -- window resolution, price
bracket baselines, shrinkage-driven ranking, and archetype classification. No network, no disk.
"""

from __future__ import annotations

from engine.data.player_history import PlayerGameweekActual
from engine.scoring import GK, MID
from features.differentials import (
    GLOBAL_LENS,
    LEAGUE_LENS,
    Archetype,
    Confidence,
    DifferentialWindow,
    OwnershipLens,
    build_differentials,
    compute_player_differential,
    fit_price_bracket_baselines,
    resolve_window,
)


def _global_lens(percent: dict[int, float | None]) -> OwnershipLens:
    return OwnershipLens(
        source=GLOBAL_LENS,
        n_rivals=None,
        percent=percent,
        owner_count={},
        eo_multiplier={},
        owner_names={},
    )


def _league_lens(
    owner_count: dict[int, int], n_rivals: int, eo_multiplier: dict[int, float] | None = None
) -> OwnershipLens:
    eo_multiplier = eo_multiplier or {}
    return OwnershipLens(
        source=LEAGUE_LENS,
        n_rivals=n_rivals,
        percent={pid: mult * 100.0 for pid, mult in eo_multiplier.items()},
        owner_count=owner_count,
        eo_multiplier=eo_multiplier,
        owner_names={},
    )


def _actual(gameweek: int, **overrides) -> PlayerGameweekActual:
    base = dict(
        gameweek=gameweek,
        minutes=90,
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        saves=0,
        yellow_cards=0,
        red_cards=0,
        bonus=0,
        defensive_contribution=0,
        total_points=2,
        expected_goals=0.0,
        expected_assists=0.0,
        expected_goal_involvements=0.0,
        expected_goals_conceded=0.0,
        selected=None,
        starts=1,
        value=None,
        transfers_in=None,
        transfers_out=None,
        bps=None,
    )
    base.update(overrides)
    return PlayerGameweekActual(**base)


# --- resolve_window -------------------------------------------------------------------------


def test_resolve_window_normal_case():
    window = resolve_window(latest_played_gameweek=10, requested_gameweeks=6)
    assert window == DifferentialWindow(gameweek_from=5, gameweek_to=10, requested_gameweeks=6)


def test_resolve_window_clamps_when_fewer_gameweeks_played_than_requested():
    window = resolve_window(latest_played_gameweek=2, requested_gameweeks=6)
    assert window.gameweek_from == 1
    assert window.gameweek_to == 2


def test_resolve_window_preseason_is_empty_not_an_error():
    window = resolve_window(latest_played_gameweek=0, requested_gameweeks=6)
    assert window.gameweek_to < window.gameweek_from


def test_resolve_window_rejects_non_positive_requested_gameweeks():
    import pytest

    with pytest.raises(ValueError):
        resolve_window(latest_played_gameweek=5, requested_gameweeks=0)


# --- fit_price_bracket_baselines -------------------------------------------------------------


def test_bucket_median_uses_players_in_that_bucket():
    # Three MID players at the same £5.5m bucket (bucket width 5, in tenths -> bucket 55),
    # points/90 of 2, 4, 6 -> median 4.
    windowed = {
        1: [_actual(1, total_points=2, minutes=90)],
        2: [_actual(1, total_points=4, minutes=90)],
        3: [_actual(1, total_points=6, minutes=90)],
    }
    positions = {1: MID, 2: MID, 3: MID}
    prices = {1: 55, 2: 56, 3: 59}

    baselines = fit_price_bracket_baselines(windowed, positions, prices)

    baseline = baselines.median_for(MID, 55)
    assert baseline.median_points_per_90 == 4.0
    assert baseline.n_players == 3


def test_thin_bucket_falls_back_to_position_median():
    # Only 2 players in the £5.5m MID bucket (below PRICE_BRACKET_MIN_PLAYERS=3), but a third MID
    # player exists in a different bucket -- the thin bucket must fall back to the position median.
    windowed = {
        1: [_actual(1, total_points=2, minutes=90)],
        2: [_actual(1, total_points=4, minutes=90)],
        3: [_actual(1, total_points=9, minutes=90)],
    }
    positions = {1: MID, 2: MID, 3: MID}
    prices = {1: 55, 2: 56, 3: 100}

    baselines = fit_price_bracket_baselines(windowed, positions, prices)

    assert (MID, 55) not in baselines.by_bucket
    baseline = baselines.median_for(MID, 55)
    assert baseline is baselines.by_position[MID]
    assert baseline.median_points_per_90 == 4.0  # median of [2, 4, 9]


def test_players_with_zero_minutes_do_not_pollute_the_baseline():
    windowed = {
        1: [_actual(1, total_points=6, minutes=90)],
        2: [_actual(1, total_points=0, minutes=0)],
    }
    positions = {1: MID, 2: MID}
    prices = {1: 55, 2: 55}

    baselines = fit_price_bracket_baselines(windowed, positions, prices)

    assert baselines.by_position[MID].n_players == 1


# --- compute_player_differential: exclusions and edge cases -----------------------------------


def test_zero_minutes_is_excluded_the_only_hard_gate():
    baselines = fit_price_bracket_baselines(
        {2: [_actual(1, total_points=4, minutes=90)]}, {2: MID}, {2: 55}
    )
    records = [_actual(1, minutes=0, total_points=0)]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result is None


def test_no_baseline_at_all_degrades_to_none_rather_than_raising():
    baselines = fit_price_bracket_baselines({}, {}, {})
    records = [_actual(1, minutes=90, total_points=4)]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result is None


# --- The core D6 properties ---------------------------------------------------------------------


def _peer_pool_baseline(median_points_per_90: float = 4.0, n_peers: int = 10, position: str = MID):
    windowed = {
        i: [_actual(1, total_points=int(median_points_per_90), minutes=90)] for i in range(n_peers)
    }
    positions = {i: position for i in windowed}
    prices = {i: 55 for i in windowed}
    return fit_price_bracket_baselines(windowed, positions, prices)


def test_one_spectacular_gameweek_does_not_outrank_a_sustained_good_run():
    """D6's core claim: a single huge haul must not beat a consistently-good run, because the
    former is heavily shrunk toward the bracket median while the latter barely is."""
    baselines = _peer_pool_baseline(median_points_per_90=4.0)

    one_huge_game = [_actual(1, total_points=10, minutes=90)]
    six_good_games = [_actual(gw, total_points=9, minutes=90) for gw in range(1, 7)]

    spectacular = compute_player_differential(1, MID, 55, one_huge_game, baselines)
    sustained = compute_player_differential(2, MID, 55, six_good_games, baselines)

    assert sustained.surplus_vs_bracket > spectacular.surplus_vs_bracket


def test_surplus_grows_monotonically_as_identical_good_gameweeks_accumulate():
    """The "gets more accurate with time" property as an executable assertion: repeating the same
    per-gameweek performance should only ever pull the shrunk rate closer to the true observed
    rate, never further from it, as more evidence of the identical performance accumulates."""
    baselines = _peer_pool_baseline(median_points_per_90=4.0)

    surpluses = []
    for n_games in (1, 2, 4, 8):
        records = [_actual(gw, total_points=8, minutes=90) for gw in range(1, n_games + 1)]
        result = compute_player_differential(1, MID, 55, records, baselines)
        surpluses.append(result.surplus_vs_bracket)

    assert surpluses == sorted(surpluses)
    # And it must be approaching, never exceeding, the true unshrunk surplus (8 - 4 = 4.0/90).
    true_surplus = 8.0 - 4.0
    assert all(s <= true_surplus + 1e-9 for s in surpluses)


def test_confidence_increases_with_effective_sample_minutes():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)

    one_game = compute_player_differential(
        1, MID, 55, [_actual(1, total_points=4, minutes=90)], baselines
    )
    many_games = compute_player_differential(
        2,
        MID,
        55,
        [_actual(gw, total_points=4, minutes=90) for gw in range(1, 9)],
        baselines,
    )

    assert one_game.confidence == Confidence.LOW
    assert many_games.confidence == Confidence.HIGH


# --- Archetype classification --------------------------------------------------------------


def test_proven_requires_strong_surplus_and_underlying_support():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)
    # Strong, sustained surplus with actual G+A matching xGI closely (supported).
    records = [
        _actual(
            gw,
            total_points=10,
            minutes=90,
            goals_scored=1,
            expected_goal_involvements=1.0,
            starts=1,
        )
        for gw in range(1, 7)
    ]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result.archetype == Archetype.PROVEN


def test_riding_luck_flags_strong_returns_unsupported_by_xgi():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)
    # Strong surplus, but goals massively outstrip xGI -- finishing luck, not a real signal.
    records = [
        _actual(
            gw,
            total_points=10,
            minutes=90,
            goals_scored=1,
            expected_goal_involvements=0.05,
            starts=1,
        )
        for gw in range(1, 7)
    ]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result.archetype == Archetype.RIDING_LUCK


def test_emerging_flags_strong_xgi_and_secure_minutes_without_points_yet():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)
    # Points at the bracket median (no Proven surplus yet) and no actual returns, but xGI/90
    # clears EMERGING_XGI_THRESHOLD and starts are secure -- the underlying process says the
    # returns are coming, they just have not landed yet, which is precisely not "riding luck"
    # (luck_gap is negative here: xGI is ahead of actual output, not behind it).
    records = [
        _actual(
            gw,
            total_points=4,
            minutes=90,
            expected_goal_involvements=0.4,
            starts=1,
        )
        for gw in range(1, 4)
    ]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result.archetype == Archetype.EMERGING


def test_goalkeepers_are_never_classified_as_emerging():
    baselines = _peer_pool_baseline(median_points_per_90=4.0, position=GK)
    records = [
        _actual(gw, total_points=4, minutes=90, expected_goal_involvements=0.9, starts=1)
        for gw in range(1, 4)
    ]

    result = compute_player_differential(1, GK, 55, records, baselines)

    assert result.archetype != Archetype.EMERGING


def test_no_archetype_when_nothing_clears_either_bar():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)
    records = [_actual(1, total_points=4, minutes=90, starts=1)]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result.archetype == Archetype.NONE


# --- Optional-field propagation (Phase 1's None-not-zero contract) ---------------------------


def test_ownership_trend_is_none_when_selected_is_unknown():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)
    records = [_actual(gw, total_points=4, minutes=90, selected=None) for gw in range(1, 4)]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result.ownership_trend_pct_per_gw is None


def test_ownership_trend_computes_when_selected_is_known():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)
    records = [
        _actual(1, total_points=4, minutes=90, selected=110_000),
        _actual(2, total_points=4, minutes=90, selected=220_000),
    ]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result.ownership_trend_pct_per_gw is not None
    assert result.ownership_trend_pct_per_gw > 0


def test_bps_per_90_is_none_when_any_row_is_missing_it():
    baselines = _peer_pool_baseline(median_points_per_90=4.0)
    records = [
        _actual(1, total_points=4, minutes=90, bps=20),
        _actual(2, total_points=4, minutes=90, bps=None),
    ]

    result = compute_player_differential(1, MID, 55, records, baselines)

    assert result.bps_per_90 is None


# --- build_differentials: end to end -----------------------------------------------------------


def test_build_differentials_filters_by_max_ownership():
    history = {
        1: [_actual(1, total_points=8, minutes=90)],
        2: [_actual(1, total_points=8, minutes=90)],
    }
    positions = {1: MID, 2: MID}
    prices = {1: 55, 2: 55}
    ownership = _global_lens({1: 3.0, 2: 15.0})

    _, results = build_differentials(
        history,
        positions,
        prices,
        latest_played_gameweek=1,
        ownership=ownership,
        max_ownership_percent=10.0,
    )

    assert {d.player_id for d in results} == {1}


def test_build_differentials_keeps_players_with_unknown_ownership():
    history = {1: [_actual(1, total_points=8, minutes=90)]}
    positions = {1: MID}
    prices = {1: 55}

    _, results = build_differentials(
        history,
        positions,
        prices,
        latest_played_gameweek=1,
        ownership=_global_lens({}),
        max_ownership_percent=10.0,
    )

    assert {d.player_id for d in results} == {1}


def test_build_differentials_defaults_to_an_empty_global_lens_when_none_is_given():
    """No ``ownership`` argument at all must behave exactly like the pre-league-lens default --
    every player kept, no ownership columns populated."""
    history = {1: [_actual(1, total_points=8, minutes=90)]}
    positions = {1: MID}
    prices = {1: 55}

    _, results = build_differentials(history, positions, prices, latest_played_gameweek=1)

    [result] = results
    assert result.current_ownership_percent is None
    assert result.league_owner_count is None


class TestLeagueOwnershipLens:
    def test_filters_by_max_league_owners_not_percentage(self):
        history = {
            1: [_actual(1, total_points=8, minutes=90)],
            2: [_actual(1, total_points=8, minutes=90)],
        }
        positions = {1: MID, 2: MID}
        prices = {1: 55, 2: 55}
        ownership = _league_lens(owner_count={1: 0, 2: 5}, n_rivals=11)

        _, results = build_differentials(
            history,
            positions,
            prices,
            latest_played_gameweek=1,
            ownership=ownership,
            max_league_owners=1,
        )

        assert {d.player_id for d in results} == {1}

    def test_a_player_missing_from_owner_count_defaults_to_zero_not_unknown(self):
        """Unlike the global lens's percentage (D1's original "unknown, so keep" rule), the league
        lens always has a definite owner count -- 0 when no rival owns the player at all -- so
        there is no ambiguous case for the filter to preserve."""
        history = {1: [_actual(1, total_points=8, minutes=90)]}
        positions = {1: MID}
        prices = {1: 55}
        ownership = _league_lens(owner_count={}, n_rivals=11)

        _, results = build_differentials(
            history,
            positions,
            prices,
            latest_played_gameweek=1,
            ownership=ownership,
            max_league_owners=0,
        )

        assert {d.player_id for d in results} == {1}
        assert results[0].league_owner_count == 0

    def test_league_columns_are_populated_from_the_lens(self):
        history = {1: [_actual(1, total_points=8, minutes=90)]}
        positions = {1: MID}
        prices = {1: 55}
        ownership = OwnershipLens(
            source=LEAGUE_LENS,
            n_rivals=11,
            percent={1: 18.2},
            owner_count={1: 2},
            eo_multiplier={1: 0.4},
            owner_names={1: ("Dave", "Priya")},
        )

        _, results = build_differentials(
            history, positions, prices, latest_played_gameweek=1, ownership=ownership
        )

        [result] = results
        assert result.current_ownership_percent == 18.2
        assert result.league_owner_count == 2
        assert result.league_eo_multiplier == 0.4
        assert result.league_owner_names == ("Dave", "Priya")

    def test_global_lens_never_populates_league_columns(self):
        history = {1: [_actual(1, total_points=8, minutes=90)]}
        positions = {1: MID}
        prices = {1: 55}

        _, results = build_differentials(
            history,
            positions,
            prices,
            latest_played_gameweek=1,
            ownership=_global_lens({1: 5.0}),
        )

        [result] = results
        assert result.league_owner_count is None
        assert result.league_eo_multiplier is None
        assert result.league_owner_names == ()


def test_build_differentials_preseason_returns_no_rows_without_raising():
    history = {1: [_actual(1, total_points=8, minutes=90)]}

    window, results = build_differentials(history, {1: MID}, {1: 55}, latest_played_gameweek=0)

    assert results == []
    assert window.gameweek_to < window.gameweek_from
