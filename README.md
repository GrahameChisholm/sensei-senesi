# FPL Assistant

A Fantasy Premier League decision tool, built as a versioned codebase: a stats-only prediction
engine, backtested and tuned in isolation.

See [`planning/BUILD_PLAN.md`](planning/BUILD_PLAN.md) for the original phased plan and
architecture. Note that its Phase 5 (web app) and Phase 6 (in-season operation) are stale — the
API and web frontend they describe have been removed and are being rebuilt from scratch.

## Status

Phase 0 (project setup), Phase 1 (data foundation), Phase 2 (the prediction engine), and Phase 3
(backtesting & validation) are built out. Phase 4 (feature logic — `features/fixtures.py`,
`captaincy.py`, `transfers.py`, `chips.py`, plus the shared `MyTeamState`), Phase 4b (the live
market overlay — `market_overlay/odds_client.py`, `divergence.py`), and the season simulator
(`simulator/`) are also implemented and tested.

## Development

```bash
uv sync --extra dev
uv run pytest
```

Copy `.env.example` to `.env` and fill in secrets locally (never commit `.env`).
