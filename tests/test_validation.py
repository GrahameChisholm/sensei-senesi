"""Tests for engine.data.validation — sanity checks and the fail-same-as-fetch-failure path."""

import pandas as pd

from engine.data.validation import (
    check_null_rate,
    check_required_columns,
    check_row_count,
    check_row_count_collapse,
    make_validator,
    row_counts,
    validate_fpl_element_summaries,
    validate_fpl_tables,
    validate_understat_player_histories,
    validate_understat_tables,
)

REQUIRED_FPL_COLUMNS = [
    "id",
    "web_name",
    "team",
    "element_type",
    "status",
    "now_cost",
    "minutes",
    "defensive_contribution",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
]


def _good_fpl_tables(n_elements: int = 450, n_teams: int = 20) -> dict[str, pd.DataFrame]:
    elements = pd.DataFrame({col: list(range(n_elements)) for col in REQUIRED_FPL_COLUMNS})
    elements["status"] = "a"
    teams = pd.DataFrame(
        {"id": list(range(n_teams)), "name": [f"Team {i}" for i in range(n_teams)]}
    )
    return {"elements": elements, "teams": teams}


def _good_understat_tables(n_players: int = 350) -> dict[str, pd.DataFrame]:
    players = pd.DataFrame(
        {
            "id": list(range(n_players)),
            "player_name": [f"Player {i}" for i in range(n_players)],
            "team_title": ["Team"] * n_players,
            "xG": [1.0] * n_players,
            "xA": [1.0] * n_players,
            "npxG": [1.0] * n_players,
            "time": [90] * n_players,
        }
    )
    return {"players": players}


def test_check_row_count_pass_and_fail():
    df = pd.DataFrame({"a": range(10)})
    assert check_row_count(df, "t", minimum=5).ok
    outcome = check_row_count(df, "t", minimum=20)
    assert not outcome.ok
    assert "t" in outcome.reason


def test_check_required_columns_reports_missing():
    df = pd.DataFrame({"a": [1]})
    outcome = check_required_columns(df, "t", ["a", "b"])
    assert not outcome.ok
    assert "b" in outcome.reason


def test_check_null_rate_flags_high_null_fraction():
    df = pd.DataFrame({"a": [1, None, None, None]})
    outcome = check_null_rate(df, "t", ["a"], max_null_rate=0.1)
    assert not outcome.ok


def test_check_null_rate_ignores_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    assert check_null_rate(df, "t", ["nonexistent"]).ok


def test_check_row_count_collapse_flags_large_drop():
    outcome = check_row_count_collapse("t", current_count=40, previous_count=100)
    assert not outcome.ok
    outcome_ok = check_row_count_collapse("t", current_count=90, previous_count=100)
    assert outcome_ok.ok


def test_check_row_count_collapse_zero_previous_is_ok():
    assert check_row_count_collapse("t", current_count=0, previous_count=0).ok


def test_validate_fpl_tables_passes_for_healthy_data():
    assert validate_fpl_tables(_good_fpl_tables()).ok


def test_validate_fpl_tables_fails_on_too_few_elements():
    tables = _good_fpl_tables(n_elements=10)
    outcome = validate_fpl_tables(tables)
    assert not outcome.ok
    assert "elements" in outcome.reason


def test_validate_fpl_tables_fails_on_wrong_team_count():
    tables = _good_fpl_tables(n_teams=15)
    outcome = validate_fpl_tables(tables)
    assert not outcome.ok
    assert "teams" in outcome.reason


def test_validate_fpl_tables_detects_row_collapse_vs_previous():
    tables = _good_fpl_tables(n_elements=450)
    previous = row_counts(_good_fpl_tables(n_elements=500))
    outcome = validate_fpl_tables(tables, previous_row_counts=previous)
    assert outcome.ok  # 450/500 is only a 10% drop, not a collapse

    tables_collapsed = _good_fpl_tables(n_elements=450)
    previous_large = row_counts(_good_fpl_tables(n_elements=2000))
    outcome_collapsed = validate_fpl_tables(tables_collapsed, previous_row_counts=previous_large)
    assert not outcome_collapsed.ok


def test_validate_understat_tables_passes_for_healthy_data():
    assert validate_understat_tables(_good_understat_tables()).ok


def test_validate_understat_tables_fails_on_missing_columns():
    tables = _good_understat_tables()
    tables["players"] = tables["players"].drop(columns=["xG"])
    outcome = validate_understat_tables(tables)
    assert not outcome.ok
    assert "xG" in outcome.reason


def test_make_validator_binds_previous_row_counts():
    previous = {"elements": 1000}
    validator = make_validator(validate_fpl_tables, previous)
    tables = _good_fpl_tables(n_elements=450)  # 55% drop vs. 1000 -> collapse
    outcome = validator(tables)
    assert not outcome.ok


def _good_element_summary_histories(n_rows: int = 450) -> dict[str, pd.DataFrame]:
    return {
        "histories": pd.DataFrame(
            {
                "element": list(range(n_rows)),
                "round": [1] * n_rows,
                "minutes": [90] * n_rows,
                "total_points": [6] * n_rows,
            }
        )
    }


def test_validate_fpl_element_summaries_passes_for_healthy_data():
    assert validate_fpl_element_summaries(_good_element_summary_histories()).ok


def test_validate_fpl_element_summaries_fails_on_too_few_rows():
    outcome = validate_fpl_element_summaries(_good_element_summary_histories(n_rows=10))
    assert not outcome.ok
    assert "histories" in outcome.reason


def test_validate_fpl_element_summaries_fails_on_missing_columns():
    tables = _good_element_summary_histories()
    tables["histories"] = tables["histories"].drop(columns=["round"])
    outcome = validate_fpl_element_summaries(tables)
    assert not outcome.ok
    assert "round" in outcome.reason


def test_validate_fpl_element_summaries_detects_row_collapse_vs_previous():
    tables = _good_element_summary_histories(n_rows=450)
    previous = row_counts(_good_element_summary_histories(n_rows=2000))
    outcome = validate_fpl_element_summaries(tables, previous_row_counts=previous)
    assert not outcome.ok


def _good_understat_player_histories(n_rows: int = 400) -> dict[str, pd.DataFrame]:
    return {
        "histories": pd.DataFrame(
            {
                "fpl_id": list(range(n_rows)),
                "date": ["2025-08-16"] * n_rows,
                "xG": [0.3] * n_rows,
                "xA": [0.1] * n_rows,
            }
        )
    }


def test_validate_understat_player_histories_passes_for_healthy_data():
    assert validate_understat_player_histories(_good_understat_player_histories()).ok


def test_validate_understat_player_histories_fails_on_too_few_rows():
    outcome = validate_understat_player_histories(_good_understat_player_histories(n_rows=10))
    assert not outcome.ok
    assert "histories" in outcome.reason


def test_validate_understat_player_histories_fails_on_missing_columns():
    tables = _good_understat_player_histories()
    tables["histories"] = tables["histories"].drop(columns=["xA"])
    outcome = validate_understat_player_histories(tables)
    assert not outcome.ok
    assert "xA" in outcome.reason


def test_validate_understat_player_histories_does_not_flag_a_thin_crosswalk_as_a_collapse():
    # Deliberately no row-count-collapse check for this source -- a thinner crosswalk match
    # share (e.g. a new signing not yet in the manual overlay) is not itself a broken pull.
    tables = _good_understat_player_histories(n_rows=400)
    previous = row_counts(_good_understat_player_histories(n_rows=2000))
    outcome = validate_understat_player_histories(tables, previous_row_counts=previous)
    assert outcome.ok


def test_row_counts_helper():
    tables = {"a": pd.DataFrame({"x": [1, 2, 3]}), "b": pd.DataFrame({"x": [1]})}
    assert row_counts(tables) == {"a": 3, "b": 1}
