"""Shared constants for the test suite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# Every API test's AppState stands for "this gameweek is still open for decisions". The API works
# that out at read time now, comparing the deadline against the clock rather than trusting the
# `deadline_passed` flag frozen into a cache at build time, so a fixed literal date would quietly
# come to mean the opposite once it slipped into the past. Anchoring to the clock keeps the
# fixtures saying what they were written to say.
UPCOMING_DEADLINE = datetime.now(UTC) + timedelta(days=1)
