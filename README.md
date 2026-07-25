# FPL Assistant

A Fantasy Premier League decision tool, built as a versioned codebase: a stats-only prediction
engine (backtested and tuned in isolation) underneath a web app presentation layer.

See [`planning/BUILD_PLAN.md`](planning/BUILD_PLAN.md) for the full phased plan, architecture,
and definitions of done — it is the reference spec for this repo.

## Status

Phase 0 (project setup), Phase 1 (data foundation — FPL/Understat clients, ID crosswalk,
point-in-time snapshots, storage schema, validation), and Phase 2 (the prediction engine —
minutes, goals, assists, clean sheets, defensive contribution, saves/bonus/cards, aggregation,
per-position regression, and Monte Carlo simulation) are complete. The engine runs end-to-end
producing a per-gameweek projection with a full outcome distribution and component breakdown, but
is **not yet validated** — Phase 3 (backtesting) is next.

## Development

```bash
uv sync --extra dev
uv run pytest
```

Copy `.env.example` to `.env` and fill in secrets locally (never commit `.env`).
