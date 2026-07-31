"""Model Performance data source (BUILD_PLAN 5.2) — the stored season-backtest headline report,
serving what `backtest.run_season`'s CLI wrote to `<report-path>.json` (see
`SeasonReport.headline_summary`'s own docstring for exactly what's in it and why it's a curated
subset, not a full dump).

**Live accuracy is intentionally not attempted here.** BUILD_PLAN 5.2 also wants the screen to
show live accuracy "once enough live gameweeks have accumulated" — that needs predictions from
`backtest.prediction_log` joined against real per-gameweek outcomes, which in turn needs the same
kind of live per-gameweek-history read `engine.data.live_adapter` already does, applied to
finished (not upcoming) gameweeks. No real weekly refresh has run yet in this environment (Track
A6), so there is no logged-prediction history to join against and nothing to build this against
without guessing — building it now would be exactly the kind of unverified, untestable-against-
anything-real code this project's own convention avoids. `has_live_accuracy` in the response is
deliberately always `False` until that exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPORT_PATH = Path("backtest/reports/2025-26.json")


@dataclass(frozen=True)
class ModelPerformanceData:
    headline: dict | None
    has_live_accuracy: bool


def load_stored_report(report_path: Path = DEFAULT_REPORT_PATH) -> ModelPerformanceData:
    """Read the stored season-backtest headline report, or ``None`` if no backtest has ever been
    run and stored — a missing report is a real, reportable state (BUILD_PLAN 5.2's own gate is
    "not yet trusted" until this exists), not an error.
    """
    if not report_path.exists():
        return ModelPerformanceData(headline=None, has_live_accuracy=False)
    return ModelPerformanceData(
        headline=json.loads(report_path.read_text()), has_live_accuracy=False
    )


def get_default_model_performance() -> ModelPerformanceData:
    """FastAPI dependency wrapping :func:`load_stored_report` at its default path with **no**
    parameters of its own — depending on ``load_stored_report`` directly would expose its
    ``report_path`` argument as a public, client-controlled query parameter, letting a request
    read an arbitrary file off the server filesystem and return its contents as JSON. Tests
    override this via ``app.dependency_overrides[get_default_model_performance]`` to inject a
    different stored report.
    """
    return load_stored_report()
