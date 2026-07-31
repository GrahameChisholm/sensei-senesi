"""Integration test for simulator/run_simulation.py's full season loop -- a small synthetic
season (no network), sized so a legal 15-man squad (2 GK/5 DEF/5 MID/3 FWD, max 3 per club) can
actually be built, proving the whole pipeline (fit -> horizon projections -> chip/transfer/
formation/captaincy decisions -> reveal -> score -> baseline comparison) wires together correctly
end to end. Component-level correctness (chip triggers, greedy transfer comparator, autosub rule,
etc.) is already covered by each module's own focused unit tests
(``tests/test_simulator_chip_calendar.py`` etc.) -- this test only proves the orchestration
doesn't crash and produces a sane report.
"""

from __future__ import annotations

import pandas as pd

from backtest.run_season import engineer_features
from engine.scoring import DEF, FWD, GK, MID
from simulator.run_simulation import run_season_simulation

N_TEAMS = 22
N_GAMEWEEKS = 9
_POSITIONS = [GK] * 3 + [DEF] * 7 + [MID] * 7 + [FWD] * 5


def _build_synthetic_season() -> pd.DataFrame:
    """22 one-player "teams" (so ``simulator.initial_squad.build_squad``'s max-3-per-club
    constraint is trivially satisfiable -- every club only ever offers one candidate), fixed
    pairings across 9 gameweeks -- enough for the harness's 3-gameweek training minimum plus
    several real decision gameweeks with a horizon."""
    team_names = [f"Team{n}" for n in range(1, N_TEAMS + 1)]
    teams = pd.DataFrame({"id": list(range(1, N_TEAMS + 1)), "name": team_names})
    pairs = [(i, i + 11) for i in range(11)]  # 0-indexed: team i vs team i+11, fixed all season

    kickoffs = pd.date_range("2025-08-16", periods=N_GAMEWEEKS, freq="7D", tz="UTC")
    rows = []
    for gw_idx, kickoff in enumerate(kickoffs):
        gw_num = gw_idx + 1
        for a_idx, b_idx in pairs:
            a_team_id, b_team_id = a_idx + 1, b_idx + 1
            a_home = gw_idx % 2 == 0
            a_score, b_score = (2, 1) if a_home else (1, 2)
            for team_idx, opponent_id, is_home, score_for, score_against in (
                (a_idx, b_team_id, a_home, a_score, b_score),
                (b_idx, a_team_id, not a_home, b_score, a_score),
            ):
                player_id = team_idx + 1
                position = _POSITIONS[team_idx]
                rows.append(
                    {
                        "element": player_id,
                        "name": f"Player {player_id}",
                        "position": position,
                        "team": team_names[team_idx],
                        "GW": gw_num,
                        "kickoff_time": kickoff.isoformat(),
                        "minutes": 90,
                        "starts": 1,
                        "was_home": is_home,
                        "opponent_team": opponent_id,
                        "total_points": 2 + (player_id % 4),
                        "bonus": player_id % 3,
                        "goals_scored": 0,
                        "assists": 0,
                        "value": 40 + (player_id % 10),
                        "selected": 1000 * player_id,
                        "transfers_in": 100,
                        "transfers_out": 50,
                        "transfers_balance": 50,
                        "clean_sheets": 1 if score_against == 0 else 0,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "own_goals": 0,
                        "saves": 2 if position == GK else 0,
                        "bps": 10 + player_id,
                        "defensive_contribution": 8 if position == DEF else 5,
                        "penalties_missed": 0,
                        "team_h_score": score_for if is_home else score_against,
                        "team_a_score": score_against if is_home else score_for,
                    }
                )
    merged_gw = pd.DataFrame(rows)

    def _team_history(is_home_seq: list[bool]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": kickoffs,
                "xG": [1.4 if h else 0.9 for h in is_home_seq],
                "xGA": [0.8 if h else 1.3 for h in is_home_seq],
                "minutes": 90.0,
                "is_home": is_home_seq,
            }
        )

    team_histories = {}
    for team_idx in range(N_TEAMS):
        is_home_seq = [
            (gw_idx % 2 == 0) if team_idx < 11 else (gw_idx % 2 != 0)
            for gw_idx in range(N_GAMEWEEKS)
        ]
        team_histories[team_names[team_idx]] = _team_history(is_home_seq)

    def _player_history(base_npxg: float, base_xa: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": kickoffs,
                "npxG": [base_npxg] * N_GAMEWEEKS,
                "xA": [base_xa] * N_GAMEWEEKS,
                "goals": [0] * N_GAMEWEEKS,
                "npg": [0] * N_GAMEWEEKS,
                "time": [90] * N_GAMEWEEKS,
                "season": ["2025"] * N_GAMEWEEKS,
            }
        )

    player_histories = {}
    for team_idx in range(N_TEAMS):
        position = _POSITIONS[team_idx]
        base_npxg = 0.3 if position == FWD else (0.12 if position == MID else 0.02)
        base_xa = 0.15 if position in (MID, FWD) else 0.02
        player_histories[team_idx + 1] = _player_history(base_npxg, base_xa)

    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    team_name_to_id = dict(zip(teams["name"], teams["id"], strict=True))
    engineered["team_id"] = engineered["team"].map(team_name_to_id)
    return engineered


def test_run_season_simulation_end_to_end_on_synthetic_data():
    engineered = _build_synthetic_season()
    report = run_season_simulation(
        season_start_year=2025,
        engineered=engineered,
        horizon_length=3,
        min_training_gameweeks=3,
        n_simulation_runs=20,
        seed=42,
    )

    assert report.season_start_year == 2025
    assert report.gameweek_log  # at least one real decision gameweek ran
    assert report.engine_total_points >= 0
    assert report.baseline_total_points >= 0
    # Gameweeks before min_training_gameweeks worth of history exist must be skipped, not scored.
    # Gameweek 1 itself is dropped entirely by `engineer_features` (no prior history for anyone),
    # so the universe to reconstruct is whatever gameweeks actually survived that, not 1..N.
    all_engineered_gameweeks = set(int(gw) for gw in engineered["gameweek"].unique())
    decided = {record.gameweek for record in report.gameweek_log}
    assert decided | set(report.skipped_gameweeks) == all_engineered_gameweeks
    assert not (decided & set(report.skipped_gameweeks))

    for record in report.gameweek_log:
        assert len(record.starting_xi) == 11
        assert record.captain_id in record.starting_xi
        assert record.vice_captain_id in record.starting_xi
        assert record.captain_id != record.vice_captain_id
        assert record.hit_cost in (0, 4)

    # Running totals are monotonically non-decreasing and match the cumulative per-gameweek sum.
    running = 0.0
    baseline_running = 0.0
    for record in report.gameweek_log:
        running += record.points_scored
        baseline_running += record.baseline_points_scored
        assert record.running_total == running
        assert record.baseline_running_total == baseline_running
