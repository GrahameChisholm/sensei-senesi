"""Phase 0 smoke test: proves the package skeleton imports and pytest/CI wiring works end to end."""

from engine import scoring


def test_dummy_module_imports_cleanly():
    import backtest  # noqa: F401
    import engine  # noqa: F401
    import features  # noqa: F401
    import market_overlay  # noqa: F401


def test_scoring_constants_are_internally_consistent():
    assert set(scoring.GOAL_POINTS) == set(scoring.POSITIONS)
    assert set(scoring.CLEAN_SHEET_POINTS) == set(scoring.POSITIONS)
    assert scoring.CLEAN_SHEET_POINTS[scoring.FWD] == 0
    assert scoring.GOAL_POINTS[scoring.GK] > scoring.GOAL_POINTS[scoring.FWD]
