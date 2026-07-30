# FPL Assistant

A Fantasy Premier League decision tool, built as a versioned codebase: a stats-only prediction
engine (backtested and tuned in isolation) underneath a web app presentation layer.

See [`planning/BUILD_PLAN.md`](planning/BUILD_PLAN.md) for the full phased plan, architecture,
and definitions of done — it is the reference spec for this repo.

## Status

Phase 0 (project setup), Phase 1 (data foundation), Phase 2 (the prediction engine), and Phase 3
(backtesting & validation) are built out. Phase 4 (feature logic — `features/fixtures.py`,
`captaincy.py`, `transfers.py`, `chips.py`, plus the shared `MyTeamState`), Phase 4b (the live
market overlay — `market_overlay/odds_client.py`, `divergence.py`), and Phase 5 (the web app —
a FastAPI backend in `api/` and a React/Vite/TypeScript/Tailwind frontend in `web/`) are also
implemented and tested. Phase 6 (in-season operation) has a real, tested orchestration scaffold
(`scripts/weekly_refresh.py`) wiring snapshot capture → projection → prediction logging → odds
pull → API state refresh — see that module's docstring for the one piece (turning a snapshot into
a not-yet-played gameweek's projections) still needing dedicated review before going live.

The API currently serves a deterministic synthetic dataset (`api/demo_data.py`) rather than a
real live snapshot, since Phase 1's snapshot store is empty in this environment — swap in a real
`AppState` (e.g. via `scripts/weekly_refresh.py`) for live use.

## Development

Engine, features, backtest, and API (Python):

```bash
uv sync --extra dev --extra web
uv run pytest
uv run uvicorn api.main:app --reload   # serves the demo dataset on :8000
```

Web frontend (React):

```bash
cd web
npm install
npm run dev   # expects the API above running on :8000, or set VITE_API_BASE_URL
```

Copy `.env.example` to `.env` and fill in secrets locally (never commit `.env`).
