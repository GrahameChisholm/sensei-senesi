# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Fantasy Premier League (FPL) decision tool for the 2026/27 season, built in two strictly separated
layers: a stats-only prediction engine (backtested and validated in isolation, no market data), and a
web app on top of it (FastAPI + React) for building a squad, picking captains, and comparing players.

Docstrings throughout the codebase are unusually dense and cite planning documents by name and
section (e.g. "BUILD_PLAN 2.7", "DIFFERENTIALS_PLAN D6", "PLAYER_STATS_PLAN D2") and by decision
letter. Those planning documents are not checked into this repo (`planning/` is empty), so the
docstrings are the only surviving record of the design rationale. Read the module docstring before
the code; it usually explains why the module exists and what it deliberately does not do, which
saves re-deriving a design decision that was already made and argued through. Where a docstring's
description of another module's behaviour conflicts with what the code actually does (a stale
reference to a since-removed module, for instance), trust the code.

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
The API serves precomputed data only; it never fetches or computes projections on request. Run
`build_projections.py` first to populate `data_store/projections/` from a live snapshot, e.g.:
```bash
uv run python scripts/build_projections.py --season 2026-27 --gameweek 1 \
  --understat-season-start-year 2026 --prior-season-start-year 2025
```
`--season`, `--gameweek`, `--understat-season-start-year`, and `--prior-season-start-year` are all
required, there is no bare no-argument form. To point a running API process at a historical replay
season instead of live data, set `FPL_REPLAY_SEASON=2025-26` before starting uvicorn (see
`api/state.py`).

`build_projections.py` also appends one availability observation batch per run to
`data_store/availability/observations.parquet` (see `engine/data/availability_log.py`), but a full
build is a several-minute operation, so how a player's listed availability moves through the week
is in practice only captured whenever a build happens to run. To capture a batch cheaply, without
a full build:
```bash
uv run python scripts/capture_availability.py --season 2026-27
```
Auto-detects the gameweek to tag from FPL's own `is_next` event; pass `--gameweek` to override.

Copy `.env.example` to `.env` for local secrets (only needed for the market overlay's odds API key,
the core engine has no external-key dependency). Never commit `.env`.

## Architecture

### The engine/app split (the single most important structural fact)

The repo is built engine-first: `engine/`, `features/`, `market_overlay/`, and `backtest/` are a
pure, stats-only prediction engine with zero UI dependency, validated entirely from the command line
and by walk-forward backtesting before any web code touches it. `api/` and `web/` are a thin
presentation layer that imports the engine directly and reimplements none of its logic. This ordering
is enforced by `backtest/gate.py` ("Engine Definition of Done") in principle, and by convention
everywhere else. If you find yourself writing FPL scoring or squad-legality logic inside `api/`, that
logic belongs in `features/` instead.

### Prediction engine (`engine/`)

Player points are modelled as a sum of independently-fit components, each gated by a minutes model:

- `engine/models/minutes.py`, the foundation. Two-stage: start/no-start, then a withdrawal-timing
  distribution conditional on starting. Everything else is scaled by this.
- `engine/models/goals.py`, `assists.py`, xG/xA-based multiplicative rate models
  (`player_rate90 × opponent_adjustment × minutes/90`), Poisson at simulation time.
- `engine/models/clean_sheets.py`, team-level xG/xGA, Dixon-Coles-correlated scorelines (not
  independent per-team Poisson draws).
- `engine/models/defensive_contribution.py`, opponent-*possession*-adjusted, the one component
  where a stronger opponent means *more* opportunities, not fewer.
- `engine/models/saves.py`, `bonus.py`, `cards.py`, smaller components. `bonus.py` is a regression
  proxy against a player's own expected stats, not a joint simulation of all 22 players on the pitch.
  It is trained on bonus *recomputed* under the current (2026/27-reworked) BPS formula, not the stale
  as-recorded column from earlier seasons.
- `engine/rates.py`, shared per-90 rate-stat utility (EWMA over the numerator and denominator
  separately, not over the per-match rate) that goals, assists, clean sheets, defensive contribution,
  and `features/differentials.py` all go through for "what is this player's/team's current rate."
- `engine/aggregate.py`, sums components into a `ComponentBreakdown` per player per gameweek, keeping
  the per-component detail attached. This is what the UI's "detail on click" and any debugging reads.
- `engine/regression.py`, fits each component's rate model per position on real outcomes. One
  regression per component per position, never one blended points regression across components or
  positions.
- `engine/simulate.py`, Monte Carlo simulation for outcome distributions (median/floor/ceiling,
  P(10+)). Correlates components that move together within one simulated match: a match "script" of
  minutes, then correlated team scorelines, then individual goals/assists apportioned from the team
  total, clean sheets read off the same scoreline, independently-drawn defensive contribution/cards,
  and bonus computed from that run's realized stats, rather than sampling each component independently.
- `engine/projections.py`, the top-level entry point (`project_player_gameweek`,
  `project_player_horizon`), tying aggregation and simulation into one object.
- `engine/horizon.py`, builds a multi-gameweek planning horizon by reusing one fixed, already-fitted
  engine state across every horizon gameweek (fit strictly on history before the real "current"
  decision gameweek). Only the fixture-dependent feature row per player changes gameweek to gameweek.
- `engine/pipeline.py`, batch orchestration: wires the full per-player chain across an entire player
  pool for one gameweek, given already-prepared feature rows. Does not fetch or prepare data itself.
- `engine/data/`, ingestion (`fpl_client.py`, `understat_client.py`, and `ingest.py` which ties the
  sources together into one on-demand "produce a clean snapshot" operation), the player/team ID
  crosswalk between FPL and Understat (`crosswalk.py`), data sanity checks and freshness tracking
  (`validation.py`), point-in-time snapshots (`snapshots.py`, the anti-leakage mechanism ensuring
  backtesting only ever sees data as it stood before the gameweek being predicted), cold-start
  priors for historyless players (`cold_start.py`), cross-season history stitching for the true-GW1
  problem (`cross_season.py`), an accumulating log of live availability signals and what they turned
  out to mean (`availability_log.py`, availability discrimination is the single largest lever in the
  engine's accuracy by a wide margin), per-player per-gameweek actual performance for the Player Stats
  page (`player_history.py`, live-only, sourced from FPL's own element-summary history rather than a
  new client), building a real manager's squad from FPL's entry API (`team_state_builder.py`; in the
  sandbox model this only ever needs current picks and today's prices, never a purchase price or
  chip/transfer history), and the live adapters (`live_adapter.py`, `live_horizon.py`) that turn a raw
  snapshot into engine-ready feature rows.
- `engine/scoring.py`, the single source of truth for 2026/27 FPL point values. Never hardcode a
  point value elsewhere.

The engine deliberately has **no dependency on betting-market data**. `market_overlay/` (odds client,
stats-vs-market divergence detection) is a separate, live-only module, pulled once per gameweek at
decision time, never used in backtesting, and never imported by `engine/`.

### Backtesting (`backtest/`)

`backtest/harness.py` runs walk-forward validation (expanding window, refit every gameweek, never a
random train/test split, which would leak the future). `backtest/metrics.py` scores accuracy,
bias, and per-component calibration. `backtest/baselines.py` compares the engine against template
captain / naive form / pure-xG baselines using paired statistical tests (bootstrap/permutation), not
raw point-estimate comparisons. `backtest/gate.py` evaluates the full Definition of Done. `backtest/
run_season.py` is the large end-to-end season-backtest driver (data fetch, feature engineering,
fit, score) that most of the above is exercised through. `backtest/diagnostics.py` is the versioned
per-component regression/VIF/xgboost-benchmark reporting pass, promoted out of ad hoc notebook
scripts (production diagnostics logic always lives in a real, testable module, never a notebook).
`backtest/prediction_log.py` writes immutable, model-version-tagged predictions to `logs/predictions/`.

### Feature/decision logic (`features/`)

Pure functions over engine projections, and over already-loaded app state: `(state, projections) →
result`. No FPL rule logic lives in `api/`, it all lives here.

- `features/team_state.py`, `MyTeamState`/`SquadPlayer`, the canonical shared squad representation
  captaincy/squad_rules/squad_optimizer all read from: current squad, starting XI/bench order, and
  captain/vice. The squad is a **permanent sandbox**: no confirm step, no transfer economy. A
  player's price is always just their current price, and there is no purchase-price/sell-price
  distinction, free-transfer count, or chip-usage tracking on this state at all.
- `features/squad_rules.py`, FPL legality: quota (2/5/5/3), max-3-per-club, budget vs. a spend
  ceiling (ordinarily the classic £100m, but caller-supplied higher for an imported squad whose real
  value has grown past that), a position-legal starting XI, and the add/remove/captaincy/
  bench-order/optimise-XI mutations a manager actually performs. Every mutation is pure and total.
  It returns a new, frozen `MyTeamState` (or a bare squad tuple for the partial-squad functions) or
  raises `SquadRuleError`, never mutating its arguments.
- `features/squad_optimizer.py`, best-possible-squad solver: given a candidate player pool and
  whichever players are already picked, finds the legal 15-man squad that maximizes projected
  points via an integer linear program (PuLP + bundled CBC). Filling every slot from empty and
  filling only a few just-vacated slots are the same problem, since already-picked players are
  simply constrained to stay in the result (`locked_player_ids`). `objective="starting_xi"`
  maximizes only the XI (right when Bench Boost isn't active); `objective="full_squad"` maximizes
  all 15 (for when it is). Captain/vice are chosen post-hoc from the two highest-EV starting-XI
  players, not baked into the ILP objective.
- `features/formation.py`, given a squad and each player's expected points, picks the highest-EV
  legal starting XI (1 GK plus a valid DEF/MID/FWD split) and orders the rest as the bench.
- `features/squad_points.py`, the one function behind the predicted-points number, pitch-card
  values, and chip previews: a pure read over a squad plus its projections. `chip` is a stateless,
  always-available toggle (`None`, `"bench_boost"`, or `"triple_captain"`, the only two chips left
  in the sandbox model, with no scarcity or usage tracking). Previewing a chip needs no separate
  step, it's just a different `chip` argument to the same read-only call.
- `features/captaincy.py`, ranks the full player pool by EV/floor/ceiling, highlighting
  owned/eligible (started, not just owned) options. No single headline pick; surfaces the top EV,
  floor, and ceiling options side by side.
- `features/fixtures.py`, custom per-team difficulty rating built from the engine's own inputs
  (opponent expected goals conceded, opponent attacking strength, home/away) rather than FPL's
  arbitrary 1-5 scale. Keeps a separate `attack_rating` and `defense_rating` rather than collapsing
  them into one number, since a fixture can be easy for goals/assists and hard for a clean sheet, or
  vice versa.
- `features/players.py`, player search/comparison, keyed one gameweek at a time. A player's full
  horizon is never collapsed into one "expected points" number.
- `features/player_stats.py`, actual-performance summarization for the Player Stats page: sums a
  player's per-gameweek actual counts and points (`engine.data.player_history`) over a UI-selected
  gameweek range. Every other filter on that page (search/team/position/price) is client-side over
  one bulk fetch.
- `features/differentials.py`, players sustainedly outperforming their own position and price
  bracket, among low-owned players. The headline metric is shrunk (never raw) toward the position/
  price-bucket median via `engine.rates.shrink_toward_prior`, which is what lets it show something
  meaningful from GW1 onward without a minimum-sample gate. Current season only, never cross-season
  history, since a summer transfer or lost starting place makes prior-season evidence misleading for
  a feature whose whole claim is "this is verified."

### Web app (`api/`, `web/`)

`api/main.py` is FastAPI with thin endpoints that call straight into `features/`. A
`SquadRuleError` (or any other `ValueError`) becomes a 400, never a 500, since these are all
caller-input problems. `api/state.py` holds in-memory app state (the projection cache plus the
squad's one live sandbox state), persisted via `api/persistence.py` (JSON round-trip through
`engine/data/storage.py`, a single row since this is a single-user local tool). `api/squad_state.py`
is the live sandbox squad type itself (0 to 15 players, never locked in), kept separate from
`features.team_state.MyTeamState` (which only ever represents a complete, legal 15/11 squad) so
`api/persistence.py` can import it without a circular import back through `api/state.py`. It's
promoted to a real `MyTeamState` only where a full squad is actually required (e.g. a points
preview). `api/schemas.py` has the Pydantic request/response models. `api/panel.py`,
`api/player_stats_panel.py`, `api/differentials_panel.py`, and `api/fixtures_view.py` are row
assembly for the Players, Player Stats, Differentials, and Fixtures pages respectively, each
enriching with a per-gameweek fixture cell (opponent, venue, expected points) across the app's
3-gameweek horizon.

`web/src/` is React + Vite + TypeScript, routed in `App.tsx` across four pages:
`pages/TeamSelection.tsx` (squad building, captaincy, chips), `pages/PlayerStats.tsx`,
`pages/Differentials.tsx`, and `pages/FixturesPage.tsx`. `web/src/api.ts` is a typed fetch client
with one function per backend endpoint and no FPL rule logic of its own. The server is always the
source of truth and the client just re-renders whatever it returns. `web/src/components/` holds the
pitch view, player panel/card, breakdown popover, chip bar, and the per-page filter/table components.

## Working conventions

- **No dashes in any copy you write.** Not em dashes, en dashes, or `--` used as punctuation. This
  applies to code comments, docstrings, UI strings, commit messages, and PR descriptions. Use
  commas, periods, or restructure the sentence instead. (Repo-enforced rule, see
  `.claude/rules/no-dashes.md`.)
- Data snapshots, the SQLite store, and secrets are gitignored. Never commit anything under
  `data_store/` or a real `.env`.
- `notebooks/` is exploration only; anything that needs to be relied on gets promoted into a tested
  module under `engine/`, `features/`, etc.
- Prefer editing/extending the interpretable regression models in `engine/models/` over ad hoc
  heuristics. `xgboost` is present only as a benchmark to check whether a flexible model materially
  beats the interpretable one, not a default choice.
