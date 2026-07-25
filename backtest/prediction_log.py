"""Immutable, timestamped prediction logging tagged with model version (3.4).

Every prediction is written before the deadline and never touched again. This is the only honest
way to judge whether the engine works: it prevents unconsciously judging a model against outcomes
already known, and — because every log is tagged with the model version (git tag/hash) that
produced it — it lets accuracy changes be attributed to a specific engine version rather than
hindsight-adjusted guessing about whether a "tweak" actually helped (BUILD_PLAN 3.4).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DEFAULT_LOG_DIR = Path("logs/predictions")

__all__ = [
    "DEFAULT_LOG_DIR",
    "current_model_version",
    "PredictionLogEntry",
    "log_predictions",
    "load_logged_predictions",
]


def current_model_version(repo_root: Path | None = None) -> str:
    """The model version a prediction run is tagged with — the git tag/hash of the code that
    produced it (BUILD_PLAN's version-control strategy: "tag every engine version... and record
    that tag with each logged prediction, so accuracy is always attributable to a specific model").
    Falls back to ``"unknown"`` outside a git repo (e.g. a packaged deployment with no ``.git``)
    rather than raising — a missing version tag shouldn't block logging a prediction.
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass(frozen=True)
class PredictionLogEntry:
    path: Path
    gameweek: int
    model_version: str
    logged_at: datetime


def log_predictions(
    predictions: pd.DataFrame,
    gameweek: int,
    model_version: str,
    logged_at: datetime | None = None,
    log_dir: Path = DEFAULT_LOG_DIR,
) -> PredictionLogEntry:
    """Write ``predictions`` immutably to ``log_dir``, stamped with ``gameweek``, ``model_version``,
    and ``logged_at`` (defaults to now, UTC). Refuses to overwrite an existing log for the same
    (gameweek, model_version, timestamp) — a prediction log that could be silently replaced is
    worthless as an unbiased accuracy record.
    """
    logged_at = logged_at or datetime.now(UTC)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = logged_at.strftime("%Y%m%dT%H%M%SZ")
    safe_version = model_version.replace("/", "_")
    path = log_dir / f"gw{gameweek:02d}_{safe_version}_{timestamp}.parquet"
    if path.exists():
        raise FileExistsError(f"prediction log already exists and is immutable: {path}")

    to_write = predictions.copy()
    to_write["gameweek"] = gameweek
    to_write["model_version"] = model_version
    to_write["logged_at"] = logged_at.isoformat()
    to_write.to_parquet(path, index=False)
    return PredictionLogEntry(
        path=path, gameweek=gameweek, model_version=model_version, logged_at=logged_at
    )


def load_logged_predictions(
    log_dir: Path = DEFAULT_LOG_DIR,
    gameweek: int | None = None,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Read back logged predictions, optionally filtered by gameweek and/or model version. Returns
    an empty frame (not an error) when nothing has been logged yet."""
    if not log_dir.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in sorted(log_dir.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if gameweek is not None:
        combined = combined[combined["gameweek"] == gameweek]
    if model_version is not None:
        combined = combined[combined["model_version"] == model_version]
    return combined.reset_index(drop=True)
