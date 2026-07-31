"""Fetch + engineer one real historical season's data for the simulator — the same wiring
``backtest.run_season``'s own (private) ``_prepare_season_backtest_data`` uses to build its
walk-forward input, minus the walk-forward/scoring/coverage parts the simulator doesn't need (it
runs its own gameweek-by-gameweek loop instead of a single ``run_walk_forward`` pass, since it
needs a fixed fitted state reused across a multi-gameweek horizon — see ``simulator/horizon.py``).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

from backtest.run_season import (
    DEFAULT_CACHE_DIR,
    build_season_crosswalk,
    build_team_rate_histories,
    engineer_features,
    fetch_understat_league_data_raw,
    fetch_understat_multi_season_league_data,
    fetch_understat_player_histories,
    fetch_vaastav_merged_gw,
    fetch_vaastav_teams,
)
from engine.data.understat_client import UnderstatClient, league_data_to_dataframes

__all__ = ["prepare_season_data"]


def prepare_season_data(
    season_start_year: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Real vaastav ground truth + Understat team/player rates, joined into one point-in-time
    ``engineered`` frame for ``season_start_year`` — exactly what ``backtest.run_season.fit_fn``/
    ``make_predict_fn`` expect, fetched from (or cached under) ``cache_dir``.
    """
    with httpx.Client(timeout=30.0) as http_client, UnderstatClient() as understat:
        merged_gw = fetch_vaastav_merged_gw(season_start_year, cache_dir, http_client, refresh)
        teams = fetch_vaastav_teams(season_start_year, cache_dir, http_client, refresh)
        league_data = fetch_understat_league_data_raw(
            season_start_year, cache_dir, understat, refresh
        )
        multi_season_league_data = fetch_understat_multi_season_league_data(
            season_start_year, cache_dir, understat, refresh
        )
        multi_season_teams_history = pd.concat(
            [
                league_data_to_dataframes(data)["teams_history"]
                for data in multi_season_league_data.values()
            ],
            ignore_index=True,
        )
        team_histories = build_team_rate_histories(multi_season_teams_history)
        crosswalk = build_season_crosswalk(
            season_start_year, league_data, cache_dir, http_client, refresh
        )
        player_histories = fetch_understat_player_histories(
            crosswalk, season_start_year, cache_dir, understat, refresh
        )

    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    # `engineer_features` maps the numeric `opponent_team` -> `opponent_team_name`, but never
    # attaches the player's *own* team's numeric id (only its name, in `team`) -- the simulator
    # needs a numeric id to build `features.fixtures.TeamFixture` and `team_id_by_player` for the
    # Free Hit blank-exposure evaluator, so derive it here from the same season-specific `teams`
    # frame (never the live bootstrap-static id mapping, which drifts season to season).
    team_name_to_id = dict(zip(teams["name"], teams["id"], strict=True))
    engineered["team_id"] = engineered["team"].map(team_name_to_id)
    return engineered
