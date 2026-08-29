"""Tests for engine/simulate.py — correlated Monte Carlo match simulation (2.9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.bonus import BonusModel, build_features
from engine.models.minutes import MinutesDistribution
from engine.simulate import (
    FixtureSimulationResult,
    PlayerMatchInputs,
    TeamMatchInputs,
    simulate_fixture,
)


def _nailed_on(**overrides) -> MinutesDistribution:
    defaults = dict(
        p_zero=0.02,
        p_1_to_59=0.08,
        p_60_plus=0.9,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=88.0,
    )
    defaults.update(overrides)
    return MinutesDistribution(**defaults)


def _fitted_bonus_model() -> BonusModel:
    rng = np.random.default_rng(0)
    rows, bonus = [], []
    for _ in range(150):
        position = rng.choice(["GK", "DEF", "MID", "FWD"])
        eg, ea, cs, dc = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 15)
        em = rng.uniform(0, 90)
        rows.append(build_features(eg, ea, cs, dc, position, expected_minutes=em))
        bonus.append(float(np.clip(1.5 * eg + ea + cs, 0, 3)))
    return BonusModel().fit(pd.DataFrame(rows), pd.Series(bonus))


def _make_squad(n_players: int, position: str, prefix: int) -> list[PlayerMatchInputs]:
    return [
        PlayerMatchInputs(
            player_id=prefix + i,
            position=position,
            minutes_distribution=_nailed_on(),
            adjusted_goal_rate_per_90=0.3 if position != "GK" else 0.0,
            adjusted_assist_rate_per_90=0.2 if position != "GK" else 0.0,
            adjusted_defensive_action_rate_per_90=8.0 if position not in ("GK",) else 0.0,
            yellow_card_rate_per_90=0.1,
            red_card_rate_per_90=0.01,
            expected_saves_full_match=3.0 if position == "GK" else 0.0,
            expected_penalties_faced_full_match=0.05 if position == "GK" else 0.0,
            penalty_save_rate=0.2 if position == "GK" else 0.0,
        )
        for i in range(n_players)
    ]


def _simple_teams() -> tuple[TeamMatchInputs, TeamMatchInputs]:
    home_players = (
        _make_squad(1, "GK", 100)
        + _make_squad(4, "DEF", 200)
        + _make_squad(4, "MID", 300)
        + _make_squad(2, "FWD", 400)
    )
    away_players = (
        _make_squad(1, "GK", 500)
        + _make_squad(4, "DEF", 600)
        + _make_squad(4, "MID", 700)
        + _make_squad(2, "FWD", 800)
    )
    home = TeamMatchInputs(players=home_players, team_expected_penalties=0.1)
    away = TeamMatchInputs(players=away_players, team_expected_penalties=0.1)
    return home, away


def test_simulate_fixture_returns_summary_for_every_player():
    home, away = _simple_teams()
    result = simulate_fixture(home, away, _fitted_bonus_model(), n_runs=300, seed=42)
    assert isinstance(result, FixtureSimulationResult)
    all_ids = {p.player_id for p in home.players} | {p.player_id for p in away.players}
    assert set(result.player_summaries) == all_ids


def test_simulate_fixture_summary_fields_are_well_formed():
    home, away = _simple_teams()
    result = simulate_fixture(home, away, _fitted_bonus_model(), n_runs=300, seed=42)
    for summary in result.player_summaries.values():
        assert summary.raw_points.shape == (300,)
        assert summary.floor <= summary.median <= summary.ceiling
        assert 0.0 <= summary.prob_big_haul <= 1.0
        assert np.isfinite(summary.mean)
        assert summary.std is not None and summary.std >= 0.0


def test_simulate_fixture_is_reproducible_with_same_seed():
    home, away = _simple_teams()
    result_a = simulate_fixture(home, away, _fitted_bonus_model(), n_runs=200, seed=7)
    result_b = simulate_fixture(home, away, _fitted_bonus_model(), n_runs=200, seed=7)
    for player_id in result_a.player_summaries:
        np.testing.assert_array_equal(
            result_a.player_summaries[player_id].raw_points,
            result_b.player_summaries[player_id].raw_points,
        )


def test_simulate_fixture_higher_goal_rate_scores_more_on_average():
    home_players = _make_squad(1, "GK", 100) + [
        PlayerMatchInputs(
            player_id=201,
            position="FWD",
            minutes_distribution=_nailed_on(),
            adjusted_goal_rate_per_90=1.2,
            adjusted_assist_rate_per_90=0.1,
        ),
        PlayerMatchInputs(
            player_id=202,
            position="FWD",
            minutes_distribution=_nailed_on(),
            adjusted_goal_rate_per_90=0.05,
            adjusted_assist_rate_per_90=0.1,
        ),
    ]
    away_players = _make_squad(1, "GK", 500) + _make_squad(4, "DEF", 600)
    home = TeamMatchInputs(players=home_players)
    away = TeamMatchInputs(players=away_players)

    result = simulate_fixture(home, away, _fitted_bonus_model(), n_runs=2000, seed=1)
    prolific = result.player_summaries[201]
    quiet = result.player_summaries[202]
    assert prolific.mean > quiet.mean


def test_simulate_fixture_gk_gets_saves_points_outfield_does_not():
    home, away = _simple_teams()
    result = simulate_fixture(home, away, _fitted_bonus_model(), n_runs=500, seed=3)
    gk_summary = result.player_summaries[100]
    def_summary = result.player_summaries[200]
    # GK's mean should reflect a real saves contribution (strictly positive expected saves).
    assert gk_summary.mean != def_summary.mean


def test_simulate_fixture_clean_sheet_team_scores_higher_defensively():
    # A team facing a very low-scoring opponent should have defenders with a healthy points floor
    # thanks to frequent clean sheets -- give the away team an essentially toothless attack.
    home_players = _make_squad(1, "GK", 100) + _make_squad(4, "DEF", 200)
    weak_away_players = [
        PlayerMatchInputs(
            player_id=500,
            position="GK",
            minutes_distribution=_nailed_on(),
            adjusted_goal_rate_per_90=0.0,
            adjusted_assist_rate_per_90=0.0,
        )
    ] + [
        PlayerMatchInputs(
            player_id=600 + i,
            position="FWD",
            minutes_distribution=_nailed_on(),
            adjusted_goal_rate_per_90=0.01,
            adjusted_assist_rate_per_90=0.01,
        )
        for i in range(2)
    ]
    home = TeamMatchInputs(players=home_players)
    away = TeamMatchInputs(players=weak_away_players)

    result = simulate_fixture(home, away, _fitted_bonus_model(), n_runs=1500, seed=5)
    home_def = result.player_summaries[200]
    assert home_def.mean > 2.0  # appearance + a very likely clean sheet


def test_team_match_inputs_rejects_empty_players():
    with pytest.raises(ValueError):
        TeamMatchInputs(players=[])
