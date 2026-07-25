# FPL Assistant

A Fantasy Premier League decision tool, built as a versioned codebase: a stats-only prediction
engine (backtested and tuned in isolation) underneath a web app presentation layer.

See [`planning/BUILD_PLAN.md`](planning/BUILD_PLAN.md) for the full phased plan, architecture,
and definitions of done — it is the reference spec for this repo.

## Status

Phase 0 (project setup) is in progress.

## Development

```bash
uv sync --extra dev
uv run pytest
```

Copy `.env.example` to `.env` and fill in secrets locally (never commit `.env`).
