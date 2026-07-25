"""Phase 0 smoke test: proves the package skeleton imports and pytest/CI wiring works end to end."""

import pytest

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


@pytest.mark.xfail(
    reason="Phase 0 placeholder — no engine logic yet; replace with real coverage in Phase 2",
    strict=False,
)
def test_hello_placeholder_for_real_engine_tests():
    raise AssertionError("replace me once the minutes model (2.1) lands")
