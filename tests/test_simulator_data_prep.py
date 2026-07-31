"""Tests for simulator/data_prep.py -- runs entirely against the real 2025/26 season cache already
populated under data_store/season_cache (no network call is made when every cache file already
exists and refresh=False). Skipped if that cache isn't present, since it isn't something every
environment running this suite is guaranteed to have pre-populated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backtest.run_season import DEFAULT_CACHE_DIR
from simulator.data_prep import prepare_season_data

_CACHE_PRESENT = (DEFAULT_CACHE_DIR / "vaastav" / "2025-26" / "merged_gw.parquet").exists()


@pytest.mark.skipif(not _CACHE_PRESENT, reason="real 2025/26 season cache not present locally")
def test_prepare_season_data_attaches_numeric_team_id():
    engineered = prepare_season_data(2025, cache_dir=Path(DEFAULT_CACHE_DIR), refresh=False)

    assert "team_id" in engineered.columns
    assert engineered["team_id"].notna().all()
    assert engineered["team_id"].map(float).map(float.is_integer).all()
    # Unlike a from-scratch synthetic season, real GW1 rows are *not* dropped here: multi-season
    # carry-forward (prior seasons' Understat history) already gives most players a real
    # `.shift(1)`-safe prior rate by the real season's very first gameweek.
    assert engineered["gameweek"].min() == 1
    assert engineered["gameweek"].max() == 38
