# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Fantasy Premier League (FPL) decision tool for the 2026/27 season, built in two strictly separated
layers: a stats-only prediction engine (backtested and validated in isolation, no market data), and a
web app on top of it (FastAPI + React) for building a squad, picking captains, and planning transfers
and chips. `planning/BUILD_PLAN.md` is the original phased spec (Phases 0-6) and is still the reference
for the engine's design rationale; `planning/TEAM_PAGE_PLAN.md` is the detailed build plan for the
current web app (the "Team Selection" page), including every locked product decision (D1-D24). The
`planning/ENGINE_IMPROVEMENTS*.md` files are dated logs of real backtest findings and the fixes that
followed, most recent last.

Docstrings throughout the codebase are unusually dense and cite these planning docs by section number
(e.g. "BUILD_PLAN 2.7", "D16/G8") and by decision letter. Read the module docstring before the code. It
usually explains *why* the module exists and what it deliberately does not do, which saves re-deriving
a design decision that was already made and argued through.

## Commands

Python (managed with `uv`; Python 3.11+):
```bash
uv sync --extra dev          # install deps (+ pytest, ruff, black)
uv run pytest                # run the full test suite
uv run pytest tests/test_squad_rules.py                  # one file
uv run pytest tests/test_squad_rules.py::test_name -v    # one test
uv run ruff check .          # lint
uv run black .               # format
```

Web (`web/`, React + Vite + TypeScript):
```bash
cd web
npm install
npm run dev        # Vite dev server, http://localhost:5173
npm run build      # tsc -b && vite build
```

Running the API locally:
```bash
uv run uvicorn api.main:app --reload   # http://localhost:8000
```
The API serves precomputed data only; it never fetches or computes projections on request (D7). Run
`build_projections.py` first to populate `data_store/projections/` from a live snapshot, e.g.:
```bash
uv run python scripts/build_projections.py --season 2026-27 --gameweek 1 \
  --understat-season-start-year 2026 --prior-season-start-year 2025
```
`--season`, `--gameweek`, `--understat-season-start-year`, and `--prior-season-start-year` are all
required, there is no bare no-argument form. To point a running API process at a historical replay
season instead of live data, set `FPL_REPLAY_SEASON=2025-26` before starting uvicorn (see
`api/state.py`).

Copy `.env.example` to `.env` for local secrets (only needed for the market overlay's odds API key;
the core engine has no external-key dependency). Never commit `.env`.

## Architecture

### The engine/app split (the single most important structural fact)

The repo is built engine-first: `engine/`, `features/`, `market_overlay/`, and `backtest/` are a
pure, stats-only prediction engine with zero UI dependency, validated entirely from the command line
and by walk-forward backtesting before any web code touches it. `api/` and `web/` are a thin
presentation layer that imports the engine directly and reimplements none of its logic. This ordering
is enforced by `backtest/gate.py` ("Engine Definition of Done") in principle, and by convention
everywhere else: if you find yourself writing FPL scoring or squad-legality logic inside `api/`, that
logic belongs in `features/` instead.

### Prediction engine (`engine/`)

Player points are modelled as a sum of independently-fit components, each gated by a minutes model,
per BUILD_PLAN Phase 2:

- `engine/models/minutes.py` — the foundation. Two-stage: start/no-start, then a withdrawal-timing
  distribution conditional on starting. Everything else is scaled by this.
- `engine/models/goals.py`, `assists.py` — xG/xA-based multiplicative rate models
  (`player_rate90 × opponent_adjustment × minutes/90`), Poisson at simulation time.
- `engine/models/clean_sheets.py` — team-level xG/xGA, Dixon-Coles-correlated scorelines (not
  independent per-team Poisson draws).
- `engine/models/defensive_contribution.py` — the 2025/26 scoring addition; opponent-*possession*
  adjusted (the one component where a stronger opponent means *more* opportunities, not fewer).
- `engine/models/saves.py`, `bonus.py`, `cards.py` — smaller components. `bonus.py` is a regression
  proxy against a player's own expected stats, not a joint simulation of all 22 players on the pitch;
  it is trained on bonus *recomputed* under the current (2026/27-reworked) BPS formula, not the stale
  as-recorded column from earlier seasons.
- `engine/aggregate.py` — sums components into a `ComponentBreakdown` per player per gameweek, keeping
  the per-component detail attached (this is what the UI's "detail on click" and any debugging reads).
- `engine/regression.py` — fits each component's rate model per position on real outcomes. One
  regression per component per position, never one blended points regression across components or
  positions — see the module docstring / BUILD_PLAN 2.8 for why that distinction matters for
  calibration checking.
- `engine/simulate.py` — Monte Carlo simulation for outcome distributions (median/floor/ceiling,
  P(10+)). Correlates components that move together within one simulated match (a match "script":
  minutes → correlated team scorelines → individual goals/assists apportioned from the team total →
  clean sheets read off the same scoreline → independently-drawn defensive contribution/cards →
  bonus computed from that run's realized stats) rather than sampling each component independently.
- `engine/projections.py` — the top-level entry point (`project_player_gameweek`,
  `project_player_horizon`), tying aggregation and simulation into one object and rolling multiple
  gameweeks into a horizon view.
- `engine/pipeline.py` — batch orchestration: wires the full per-player chain across an entire player
  pool for one gameweek, given already-prepared feature rows. Does not fetch or prepare data itself.
- `engine/data/` — ingestion (`fpl_client.py`, `understat_client.py`), the player/team ID crosswalk
  between FPL and Understat (`crosswalk.py`), point-in-time snapshots (`snapshots.py` — the
  anti-leakage mechanism: backtesting must only ever see data as it stood before the gameweek being
  predicted), cold-start priors for historyless players (`cold_start.py`), cross-season history
  stitching for the true-GW1 problem (`cross_season.py`), and the live adapters
  (`live_adapter.py`, `live_horizon.py`) that turn a raw snapshot into engine-ready feature rows.
- `engine/scoring.py` — the single source of truth for 2026/27 FPL point values. Never hardcode a
  point value elsewhere.

The engine deliberately has **no dependency on betting-market data**. `market_overlay/` (odds client,
stats-vs-market divergence detection) is a separate, live-only module, pulled once per gameweek at
decision time, never used in backtesting, and never imported by `engine/`.

### Backtesting (`backtest/`)

`backtest/harness.py` runs walk-forward validation (expanding window, refit every gameweek — never a
random train/test split, which would leak the future). `backtest/metrics.py` scores accuracy,
bias, and per-component calibration. `backtest/baselines.py` compares the engine against template
captain / naive form / pure-xG baselines using paired statistical tests (bootstrap/permutation), not
raw point-estimate comparisons. `backtest/gate.py` evaluates the full Definition of Done. `backtest/
run_season.py` is the large end-to-end season-backtest driver (data fetch → feature engineering →
fit → score) that most of the above is exercised through. `backtest/prediction_log.py` writes
immutable, model-version-tagged predictions to `logs/predictions/`.

### Feature/decision logic (`features/`)

Pure functions over engine projections: `(state, projections) → recommendation`. No FPL rule logic
lives in `api/` — it all lives here.

- `features/team_state.py` — `MyTeamState`/`SquadPlayer`, the canonical shared squad representation
  every decision feature reads. Always a complete, currently-valid 15/11/4 squad.
- `features/squad_rules.py` — FPL legality: quota (2/5/5/3), budget, max-3-per-club, transfers and
  sell-price handling (only half the profit on a risen player, rounded down), transfer-hit cost.
  Raises `SquadRuleError`/`RuleViolation` for any illegal mutation — the API turns these into 400s,
  never 500s.
- `features/squad_draft.py` — the preview-then-confirm draft/commit state machine (TEAM_PAGE_PLAN
  D16-D24): nothing a manager does in the UI is real until an explicit Confirm. Separate from
  `squad_rules` because it adds time/sequencing/chip concerns `squad_rules` has no concept of. Two
  distinct lifecycles: building the first squad from empty (`confirm_initial_squad`) vs. every
  subsequent edit (`open_draft`/`apply_*_to_draft`/`confirm_draft`), because the first has no existing
  committed squad to diff against.
- `features/chip_calendar.py` — 2026/27 chip rules specifically: all four chips (Wildcard, Free Hit,
  Bench Boost, Triple Captain), one full set per half-season. The older 2025/26 ruleset (2 Wildcards,
  one each of the others, no halves) no longer has any code path in this repo.
- `features/captaincy.py`, `transfers.py`, `chips.py`, `fixtures.py` — the four planning features,
  each ranking/evaluating over the full player pool or planning horizon, not just the user's own 15.
- `features/squad_points.py` — projects points for a squad/XI, chip-aware.

### Web app (`api/`, `web/`)

`api/main.py` is FastAPI with thin endpoints that call straight into `features/`; it never computes
projections itself and never touches squad legality directly (see its module docstring). `api/state.py`
holds in-memory app state (the projection cache plus committed/pending squad state), persisted via
`api/persistence.py` (SQLite, via `engine/data/storage.py`). `api/schemas.py` has the Pydantic
request/response models.

`web/src/` is React + Vite + TypeScript. `web/src/api.ts` is a typed fetch client with one function per
backend endpoint and no FPL rule logic of its own — the server is always the source of truth, and the
client just re-renders whatever it returns. `web/src/pages/TeamSelection.tsx` is the current (and so
far only) page; `web/src/components/` holds the pitch view, squad builder, chip bar, and per-player
breakdown popover.

## Working conventions

- **No dashes in any copy you write** — not em dashes, en dashes, or `--`. This applies to code
  comments, docstrings, UI strings, commit messages, and PR descriptions. Use commas, periods, or
  restructure the sentence. (Repo-enforced rule; see `.claude/rules/no-dashes.md`.)
- Data snapshots, the SQLite store, and secrets are gitignored — never commit anything under
  `data_store/` or a real `.env`.
- `notebooks/` is exploration only; anything that needs to be relied on gets promoted into a tested
  module under `engine/`, `features/`, etc.
- Prefer editing/extending the interpretable regression models in `engine/models/` over ad hoc
  heuristics; `xgboost` is present only as a benchmark to check whether a flexible model materially
  beats the interpretable one, not a default choice.
