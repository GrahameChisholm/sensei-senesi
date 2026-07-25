# FPL Assistant — Build Plan

A phased plan for building a Fantasy Premier League decision tool as a versioned codebase. The prediction engine is built, tested, backtested and tuned in isolation **first**; the web application is only started once the engine has cleared an explicit quality gate. This document is the reference spec — keep it in the repo root and update it as decisions change.

---

## Guiding principles

1. **Engine before app.** The web application is a presentation layer over a trusted engine. If the engine isn't good, a beautiful UI just makes wrong answers look convincing. Nothing in the web phase starts until the engine passes its Definition of Done.
2. **Everything is testable without a UI.** Projections and every downstream decision (captaincy, transfers, chips) are pure functions of data in → recommendation out. They get unit-tested and backtested from the command line long before any screen exists.
3. **No leakage, ever.** Every backtest uses only data that was available *before* the gameweek being predicted. This single constraint shapes the entire data layer.
4. **Log predictions before outcomes exist.** The engine records what it predicted, timestamped and tied to a model version, before each deadline. This is the only honest way to judge whether it works and whether a "tweak" actually improved anything.
5. **Explainable over clever.** Prefer interpretable models (regression with inspectable coefficients) so that "which stats matter" is a measurable output, not a black box — and so a suspicious recommendation can always be traced back to its cause.
6. **The engine is stats-only; the market is a live overlay, not a foundation.** Free historical player-prop odds essentially don't exist at any depth, while free player and team xG (Understat, since 2014/15) and the FPL API's own history run over a decade deep. Building the core engine entirely on FPL API + Understat means **every component backtests on the same footing, with no proxy layers or shorter-history asterisks.** Live pre-match odds are pulled in only at decision time each gameweek, as a separate, explicitly-tracked overlay — never baked into the backtested core.

---

## Recommended stack

You can swap any of this, but this combination minimises friction and — crucially — lets the web backend reuse the engine directly rather than reimplementing it.

**Engine (Phases 1–4)**
- Python 3.11+
- `pandas` / `numpy` — data wrangling and vectorised simulation
- `httpx` or `requests` — API clients
- `scikit-learn` — regression, probability calibration
- `statsmodels` — interpretable regression with diagnostics (coefficients, p-values, residuals)
- `xgboost` — optional, only as a benchmark to check whether a flexible model beats the interpretable one
- `pytest` — testing
- `SQLite` (via SQLAlchemy) for structured data, plus `parquet` files for immutable point-in-time snapshots
- Jupyter — for exploration only; production logic always lives in versioned modules, never in notebooks

**Web app (Phase 5)**
- `FastAPI` backend — Python, so it imports the engine directly with zero reimplementation
- `React` + `Vite` + `TypeScript` frontend
- `Tailwind CSS` for styling
- Alternative fast path: `Streamlit` if you'd accept a less polished internal tool for much less frontend work. Recommended path is FastAPI + React given you want a proper application.

**Tooling**
- Git for version control (branching + tagging strategy below)
- `ruff` + `black` for linting/formatting
- `.gitignore` all data snapshots and secrets; never commit API keys or large data files

---

## Proposed repository structure

```
fpl-app/
├── engine/
│   ├── data/
│   │   ├── fpl_client.py            # official FPL API
│   │   ├── understat_client.py      # xG / xA — core engine's only external stats source
│   │   └── snapshots.py             # point-in-time capture (no leakage)
│   ├── models/
│   │   ├── minutes.py               # THE foundation — build first
│   │   ├── goals.py
│   │   ├── assists.py
│   │   ├── clean_sheets.py          # + goals conceded
│   │   ├── defensive_contribution.py
│   │   ├── saves.py
│   │   ├── bonus.py
│   │   └── cards.py
│   ├── aggregate.py                 # sum components, gated by minutes, per position
│   ├── regression.py                # fit weights per position; feature importance
│   ├── simulate.py                  # Monte Carlo → distributions
│   └── projections.py               # top-level: player → full projection
├── features/                        # decision logic — pure functions over projections
│   ├── captaincy.py
│   ├── transfers.py
│   ├── chips.py
│   └── fixtures.py
├── market_overlay/                  # LIVE-ONLY, never used in backtesting — see Phase 4b
│   ├── odds_client.py               # pre-match odds, pulled at decision time only
│   └── divergence.py                # compares stats projection vs market-implied probability
├── backtest/
│   ├── harness.py                   # walk-forward engine
│   ├── metrics.py                   # MAE, calibration, captaincy hit-rate
│   └── baselines.py                 # template captain, form model, pure xG model
├── data_store/                      # SQLite + parquet snapshots (gitignored)
├── logs/predictions/                # immutable pre-deadline prediction logs
├── tests/
├── notebooks/                       # exploration only
├── api/                             # Phase 5 — FastAPI backend
├── web/                             # Phase 5 — React frontend
├── pyproject.toml
└── README.md  /  BUILD_PLAN.md
```

The `engine/` and `features/` directories are the entire product minus its face. If they're solid, the web app is comparatively easy.

---

# PHASE 0 — Project setup

**Goal:** a clean, reproducible, version-controlled foundation.

- Initialise the git repo; set up branch strategy (see "Version control strategy" below).
- Create the Python environment (`venv` or `conda`), `pyproject.toml`, pin dependencies.
- Lay down the directory skeleton above with empty modules and a failing "hello" test so CI/test wiring is proven from day one.
- Add `.gitignore` for `data_store/`, secrets, `__pycache__`, notebooks' checkpoints.
- Write a `.env.example` and a secrets-loading convention for the odds API key used by the Phase 4b live overlay (never hardcoded, never committed). Not needed for Phases 1–3 — the backtestable engine has no dependency on it at all.
- Decide and document the scoring rules as a single source of truth (a `scoring.py` constants module encoding the **2026/27** point values — the season this engine targets — including the reworked BPS formula and the confirmed-unchanged goal/clean-sheet/defensive-contribution values), so no magic numbers leak into individual models.

**Definition of done:** repo builds, tests run, a dummy module imports cleanly, no secrets in git history.

---

# PHASE 1 — Data foundation

**Goal:** reliable, point-in-time-correct data flowing into local storage. This is unglamorous plumbing and it is where a surprising amount of the total effort lives. Do it properly now or pay for it repeatedly later.

### 1.1 Ingestion clients
- **FPL API client** (`fpl_client.py`) — the primary source. Pull: player master data, prices, ownership, positions, ICT components, the defensive-contribution stats, fixtures, team news/injury flags, and full gameweek-by-gameweek history. No key required, no rate limits of concern.
- **Understat client** (`understat_client.py`) — xG, xA, npxG at **player and team** level, per match, back to 2014/15. This is the core engine's other pillar alongside the FPL API — team-level xG/xGA covers clean-sheet and goals-conceded modelling just as well as player-level xG covers goals and assists, so the entire engine runs on these two free, deep sources with no market dependency.

These two clients are the **only** data sources the backtestable core (Phases 1–3) depends on. The odds client lives separately in `market_overlay/` (see Phase 4b) and is deliberately kept out of the engine's data layer — it's pulled live at decision time each gameweek, not during backtesting, since free historical player-prop odds don't exist at any real depth (confirmed: match-level odds archives run back to 2000/01, but player-prop archives are paid and typically only from ~2019 onward).

**Player/team ID crosswalk — a real gap, not a nice-to-have.** FPL and Understat share no common ID scheme, and Understat player names don't always match FPL's transliterations (accents, nicknames, Portuguese/Brazilian name forms especially) — every component depends on correctly joining the two sources, and a silent mismatch doesn't crash anything, it just quietly attributes one player's xG to another. Build this by starting from an existing open-source crosswalk (e.g. `vaastav/Fantasy-Premier-League` on GitHub, which the wider community already hand-verifies), overlaid with a small hand-maintained table for the current season's new signings/transfers each window. Any Understat player that can't be matched to an FPL ID **fails loudly at ingestion** — a missing match is far safer than a wrong one.

**Rolling rate stats.** Every component in Phase 2 references a player's "current" per-90 rate (non-penalty xG/90, xA/90, team xGA/90, etc.) — computed as an **exponentially-weighted moving average** over match history, not a fixed trailing-N window or a flat season-to-date average. EWMA avoids the arbitrary window-size cliff of trailing-N, reacts to genuine changes in role/form the way season-to-date can't, and its single decay parameter is something backtesting can actually tune and justify. It also solves the cold-start problem for free — early in a new season, a well-chosen decay means prior-season history still carries real weight and fades gradually as current-season matches accumulate, rather than needing a separate hand-coded blend rule for the first few gameweeks.

### 1.2 Point-in-time snapshots (`snapshots.py`) — critical
This is the anti-leakage mechanism and the most important design decision in the whole data layer. Capture the full state of every data source *as it stood at that moment* and freeze it as an immutable parquet snapshot keyed by `(season, gameweek, captured_at)`. Backtesting later replays these snapshots. **Never** backtest against current data that has since been revised (prices move, injuries resolve, xG gets recalculated) — that silently leaks the future into the past and flatters the model.

**Cadence: daily, not just once per deadline.** A single deadline-moment capture is the cleanest match to the literal invariant needed, but it's also a single point of failure — this system has to run unattended every week for a full season (Phase 6), and if the one scheduled job doesn't fire exactly when intended, that gameweek's snapshot is lost with no fallback. Capture daily instead, with retry-on-failure for each day's attempt; "the pre-deadline snapshot" (what backtesting replays, what a live decision reads) is simply the most recent capture strictly before the deadline cutoff. If the final capture attempt before a deadline is exhausted, fall back to the previous day's snapshot rather than losing the gameweek.

### 1.3 Storage schema
- SQLite for the structured relational data (players, teams, fixtures, gameweek results).
- Parquet snapshots for the point-in-time captures.
- A results table recording what *actually* happened each gameweek (actual points, minutes, goals, defensive contributions, bonus) — this is the ground truth every backtest scores against.

### 1.4 Data validation & freshness
- Sanity checks on every pull (row counts, null rates, obvious anomalies).
- A "last updated" record per source so a future dashboard can show data freshness.
- **Failure mode: a failed sanity check is treated the same as a failed fetch** — reject the capture, fall back to the last known-good snapshot, and alert. A capture that fetches successfully but looks wrong (a sudden null-rate spike, a missing team, a row count that collapsed overnight) is a worse failure mode than being one day stale, since a silently-corrupted snapshot would feed wrong inputs into every downstream component for a week. This reuses the same fallback path as the retry/cadence mechanism above rather than needing a separate "degraded confidence" mechanism.

**Definition of done:** you can, on demand, produce a clean snapshot for the current gameweek, and you have (or can reconstruct) snapshots for a meaningful stretch of historical gameweeks to backtest against. Data validation catches obvious breakage automatically.

> **Note on history:** Because every component now sources from the FPL API and Understat only, **every component except one backtests on the same 2014/15-onward depth** — goals, assists, clean sheets, minutes all have over a decade of matched (data-at-the-time, actual-outcome) pairs. The one exception is **defensive contribution**, introduced in 2025/26 with no prior-season history under this exact scoring. Since this engine targets the 2026/27 season, one full season of real DC history (2025/26) exists by the time it's backtested — much shallower than the decade-plus depth everything else gets, but not zero. See Phase 3.6 for how the DoD gate handles that shallower depth.

---

# PHASE 2 — The prediction engine (the core — most detail here)

**Goal:** for any player, in any gameweek, produce an expected-points projection *and* a full outcome distribution (median, floor, ceiling, probability of a big haul), built up component by component, weighted per position, with the stat-importance weights *learned* from history rather than guessed.

The engine models FPL points as a **sum of scoring components**, because each component is best predicted from a different source and carries a different point value per position. Rather than predicting "points" directly, predict each component and sum them:

This additive decomposition is how the *expected value* is built up. It does **not** mean components are drawn independently at simulation time (2.9) — real hauls come from correlated events within the same match (a big win brings more minutes, more goals, more assists, and more bonus together), and modelling components as independent coin-flips would understate the ceiling/floor spread that captaincy and chip decisions specifically depend on. See 2.9 for how the correlation is wired in.

```
Expected points =
      appearance
    + goals            (position-weighted)
    + assists
    + clean sheet      (position-weighted)
    − goals conceded   (GK/DEF only)
    + defensive contribution
    + saves            (GK only)
    + bonus
    − cards
    − penalty misses
```

Build the components in the order below, because each depends on the one before it. **Build, unit-test, and sanity-check each component in isolation before moving on.**

### 2.1 Minutes model (`minutes.py`) — the foundation, build first
The single highest-value and hardest part of the engine. A brilliant attacker on the bench scores zero, so every other component is *gated* and *scaled* by minutes.

**Output — two linked parts, not one:**
1. **Bucket probabilities:** probability of playing 0 minutes, 1–59 minutes, and 60+ minutes — these three thresholds map directly onto FPL's own appearance-point and clean-sheet-eligibility cutoffs, so they gate every downstream component's points conversion.
2. **Conditional expected minutes:** given each bucket (especially 60+), the expected number of minutes actually played. Two players can share identical bucket probabilities while one reliably plays the full 90 once they start and the other is a near-automatic 65th-minute substitution — that difference matters for scaling every per-90 rate stat (goals, assists, defensive contribution) downstream, so the bucket probabilities alone aren't enough.

**Fit as a two-stage model** rather than one flat three-way classifier: first a start/no-start decision, then, conditional on starting, a withdrawal-timing distribution. This mirrors the actual sequence of real-world decisions (team-sheet, then in-game substitution) and keeps each stage interpretable, per the "explainable over clever" principle.

**Inputs:**

| Input | Role |
|-------|------|
| Recent starts / minutes trend (per-player rolling history) | Model input |
| Fixture congestion — games played in the last *N* days, derived from fixture dates already held | Model input. Fully quantifiable, not a fuzzy signal, and one of the more evidence-backed rotation drivers in football (European-competition sides measurably rotate domestic lineups midweek; fixture pile-ups precede rest for key players ahead of run-ins) |
| `chance_of_playing_next_round` (FPL API, 0/25/50/75/100) | Model input — clean structured fitness signal |
| `status` (FPL API, `a`/`d`/`i`/`s`/`u`) | Model input — clean structured availability signal |
| `news` free text (FPL API, e.g. *"Knee injury - Expected back 01 Jan"*) | **Display flag only, not a model input.** Turning free text into a clean numeric feature needs real NLP work, and the two structured fields above already carry the same information as a clean number — building text-parsing infrastructure to duplicate them isn't worth it at this stage |

**Considered and cut:**
- **Position** — not a feature here, because the regression layer (2.8) already fits a separate model per position; within a single-position model every row shares the same position value, so it carries zero information. A structural redundancy, not a weak signal.
- **Squad depth** — not a feature here. No clean free data source exists for it (it would require a hand-built, quality-weighted depth chart — exactly the manual, subjective labelling this approach is designed to avoid), and it's largely redundant with the recent-minutes trend above, which already implicitly reflects rotation pressure from competition for a starting shirt. Cut for cost/benefit and redundancy, not because the underlying idea is unsound.

Expect this to be the component you revisit most often — invest here.

**Why it matters more than the fancy stuff:** most of the real predictive skill in FPL modelling is calling who starts and finishes, not modelling xG to three decimal places. Get minutes wrong and nothing downstream can save the projection.

### 2.2 Goals (`goals.py`) — stats-led (xG-based)
Derive a scoring probability from the player's own **non-penalty xG per 90** (Understat), adjusted for the specific fixture using the opponent's defensive xG-against trend, and scaled by expected minutes. This is deeply backtestable — over a decade of matched (xG-at-the-time, actual-goals) pairs — which is exactly the sort of validated relationship the regression layer (2.8) is meant to fit and confirm, rather than assume. Convert to points using the **position-specific** goal values:

**Functional form:** a multiplicative rate, in the same family as standard football rate models (Dixon-Coles-style attack/defence strength):

```
expected_goal_rate = player_npxG90 × (opponent_xGA90 / league_avg_xGA90) × (expected_minutes / 90)
```

This produces a rate parameter (λ), not just a point estimate — needed because the simulation layer (2.9) draws discrete open-play goal counts per run as `goals ~ Poisson(λ)`.

**Penalty sub-model — a genuine addition, not covered by non-penalty xG.** Because this component is deliberately built on *non-penalty* xG, and the top-level formula above also lists a `− penalty misses` line with no model behind it, both penalty goals and penalty misses need their own small sub-model: (a) identify each team's primary penalty taker (Understat/FPL historical penalty-taking data), (b) estimate the team's expected penalties won per game, opponent-adjusted the same way as open-play goals, (c) apply the taker's historical conversion rate to split that into expected penalty goals (added into this component) vs expected penalty misses (its own deduction line). This matters disproportionately for premium captaincy picks, who are often exactly the players on penalties.

**Inputs:**

| Input | Role |
|-------|------|
| Player non-penalty xG per 90 (Understat) | Model input — scoring rate |
| Opponent team xGA per 90 (Understat) | Model input — **this is the fixture-difficulty adjustment.** It's expressed as the raw stat rather than a bucketed 1–5 rating deliberately: a composite difficulty rating (the kind the Fixtures page shows) is a lossy simplification built for human eyes, whereas the raw opponent xGA number is more precise and is what the regression should actually see |
| Expected minutes (from 2.1) | Scaling factor |
| Team's penalty-taker identity + penalty win rate + taker's conversion rate | Model input — penalty sub-model, covers penalty goals and penalty misses |

| Position | Points per goal |
|----------|-----------------|
| GK       | 10              |
| DEF      | 6               |
| MID      | 5               |
| FWD      | 4               |

Note this deliberately leaves out bookmaker odds entirely — that's a considered choice (see the stats-only guiding principle above), not an oversight. The live market comparison happens later, at decision time, in the Phase 4b overlay — never inside the backtested core.

### 2.3 Assists (`assists.py`) — stats-led
Bookmakers barely price assists, so use stats. Kept structurally symmetric with the goals model (2.2) rather than blending in team xG as a standing factor — xA is already, by construction, a rate stat built from that player's actual key passes in actual matches for that actual team, so a team-xG blend on top risks double-counting the same team-attacking-quality signal rather than adding something new (the same redundancy logic used to cut squad depth from 2.1).

**Functional form**, same family as goals:

```
expected_assist_rate = player_xA90 × (opponent_xGA90 / league_avg_xGA90) × (expected_minutes / 90)
```
with `assists ~ Poisson(λ)` per simulated run.

**Inputs:**

| Input | Role |
|-------|------|
| Player xA per 90 (Understat) | Model input — chance-creation rate |
| Opponent team xGA per 90 (Understat) | Model input — fixture-difficulty adjustment. Creating a chance is harder against a well-organised low block than a leaky defence, same logic as the goals model in 2.2 |
| Team xG per 90 (Understat) | **Not a standing multiplicative factor** — reserved only as a shrinkage prior for low-sample players (a new signing or a player back from injury with only a few games of individual xA), where team-level data fills in for a thin individual history. Not applied to every player as an ongoing input |
| Expected minutes (from 2.1) | Scaling factor |

### 2.4 Clean sheets & goals conceded (`clean_sheets.py`) — stats-led (team xG/xGA-based)
Derive the probability a team keeps a clean sheet from **team-level expected goals against** (Understat), adjusted for the specific opponent's attacking xG, and converted through a Poisson goals model to a clean-sheet probability. Understat carries team xG *and* xGA per match back to 2014/15, so this is exactly as deeply backtestable as the goals component — no market data required to model defensive strength well.

**Home/away split:** team xG/xGA rates are split into home and away components rather than a single season-blended average — home defence is measurably stronger than away defence on average, the same reasoning already applied to the saves model (2.6).

**Correlated, not independent, scorelines:** rather than treating each team's Poisson goal draw as independent of the opponent's, apply a Dixon-Coles-style low-score correlation adjustment. Low-scoring outcomes (0-0, 1-0, 1-1) are measurably more common in football than two independent Poisson draws would predict — a well-documented correction (Dixon & Coles, 1997), concentrated exactly at the score-lines that determine clean sheets. Given clean sheets are worth real points (4 for GK/DEF), this is worth doing rather than assuming independence. This same correlated scoreline is the "match script" the simulation layer (2.9) draws once per run and reads goals/clean-sheets off consistently — not two separate independent per-team calls.

**Internal consistency requirement:** the "opponent's attacking xG" used here must be the *same number* as the sum of that opponent's on-pitch players' individual goal rates from 2.2 — i.e. team-level Λ = Σ λ_i over the players actually on the pitch that run, not a separately-fit team-level attack parameter. Otherwise the clean-sheet model and the goals model would carry two independently-estimated, potentially divergent views of the same team's attacking strength, and the match script in 2.9 would have no single consistent number to draw from.

Apply position-specific clean-sheet values:

| Position | Clean sheet points | Requires |
|----------|--------------------|----------|
| GK / DEF | 4                  | 60+ min  |
| MID      | 1                  | 60+ min  |
| FWD      | 0                  | —        |

The same expected-goals-conceded distribution drives the **−1 per 2 goals conceded** penalty for GK/DEF.

### 2.5 Defensive contribution (`defensive_contribution.py`) — stats-led, your edge
The 2025/26 scoring addition, and bookmakers offer **no market for it** — so most tools underweight it. Model it from each player's historical rate of qualifying defensive actions per 90 (tackles, interceptions, blocks, clearances; plus recoveries for MID/FWD), turned into the probability of clearing the threshold and banking 2 points.

**Opponent-possession adjustment — the one fixture adjustment this component was missing, and it points the opposite way from goals/assists/clean sheets.** A player makes more defensive actions when their team *doesn't* have the ball, so facing a possession-dominant opponent means more tackling/intercepting opportunities, not fewer — the reverse of how a strong opponent is bad news for goals or clean sheets. Scale the per-90 rate up against possession-dominant opponents and down against low-possession ones, using opponent possession share (or pass-volume) as the adjustment input.

**Distributional form:** don't assume a plain Poisson by default — defensive actions tend to cluster (scrappy, backs-to-the-wall games spike hard), so a Negative Binomial (allows variance to exceed the mean) is likely the better fit. Check this empirically against the data once it's available rather than committing to one now.

| Position | Actions needed | Points |
|----------|----------------|--------|
| DEF      | 10             | 2      |
| MID      | 12             | 2      |
| FWD      | 12             | 2      |

Some defenders and defensive midfielders clear the threshold almost every game — a near-guaranteed points floor invisible to the market. **Caveat:** limited history under the new rule, so lean on per-90 action rates early and let the regression re-weight as real data accumulates.

### 2.6 Saves, bonus, cards (`saves.py`, `bonus.py`, `cards.py`) — the smaller components
- **Saves (GK):** expected opponent shots on target → expected saves → 1 point per 3 saves. Add penalty-save expectation (5 points, low probability).

  **Inputs:**

  | Input | Role |
  |-------|------|
  | Opponent shots on target per 90 | Model input — direct shot-volume driver of save opportunities |
  | Team xGA | Model input — defensive-strength signal; a weak defence facing lots of shots drives keeper workload |
  | Home/away | Model input (added) — away teams face somewhat more pressure and shots on average, so expected save count genuinely shifts by venue |
- **Bonus:** structurally different from every other component — FPL awards 3/2/1 bonus to the top three BPS scorers **within that specific match**, a relative ranking across ~22 players, not an absolute per-player threshold. Full accuracy would mean simulating every player's BPS-contributing events for both full lineups jointly each run and ranking them — a real engineering step up, since it needs both squads modelled together rather than just the player being projected. Given bonus is explicitly one of the smaller components, use a **regression proxy** instead: regress a player's expected bonus directly against their own expected BPS-relevant stats (own goals, assists, clean sheet, defensive actions, position), letting the regression implicitly absorb "how much competition for bonus typically exists" without explicitly simulating the other 21 players. Revisit with full joint simulation only if backtesting (Phase 3) shows bonus calibration is a material error source.

  **The BPS formula itself was reworked for 2026/27** (tackles no longer carry the old −1 BPS penalty; defenders earn BPS per 3 CBIs instead of per 2; goalkeeper save BPS restructured by shot category) — specifically to reduce overlap with defensive contribution. This means **"actual bonus received" from any pre-2026/27 season reflects a superseded formula**, the same "new rule, no native history" problem defensive contribution already has (2.5, 3.6). Don't regress directly against that stale column. Instead, **recompute historical bonus under the current BPS formula from raw match-event data** (tackle counts, CBI counts, saves by category) — the formula is fully public and mechanical, and most of the needed raw event data is already being pulled for the defensive-contribution and saves components anyway. This gives the regression much deeper effective history than waiting for real 2026/27 gameweeks to accumulate.
- **Cards:** historical yellow/red rates → expected −1 / −3. Small but real, especially for aggressive defenders and midfielders. Referee assignment is a genuine, quantifiable, knowable-in-advance factor (referees vary measurably in card strictness, and appointments are announced days ahead) — deliberately **not** modelled for v1, since cards are one of the smallest components in the engine and the added data-sourcing/maintenance cost isn't worth the marginal accuracy here. Revisit only if backtesting shows cards materially affecting captaincy or differential calls.

### 2.7 Aggregation (`aggregate.py`)
Sum all components, each already gated/scaled by the minutes distribution, to produce a single expected-points number per player per gameweek — and a rolled-up projection over the planning horizon (5 gameweeks by default). Keep the per-component breakdown attached to the output: the web app's "detail on click" view and your own debugging both depend on being able to see *what* drove a number.

### 2.8 Regression layer (`regression.py`) — this answers "which stats matter"
Instead of hand-guessing how much weight xG, form, fixture difficulty, and defensive-action rate should each carry, **fit those relationships on historical data**. Regress actual outcomes (real goals, real defensive-contribution hits, real bonus) against the candidate stats-only inputs, **fitting a separate model per position** because value concentrates so differently by position (a goal is worth 10 for a keeper and 4 for a forward; clean sheets are worthless to forwards; defensive thresholds differ). A single blended model would average these distinctions away and get every position slightly wrong.

**Locked-in: this is a per-component regression, never a unified points regression.** Each component (2.2–2.6) keeps its own independently-fit probability/rate model, refined by regressing its own real-world outcome (real goals, real defensive-contribution hits, etc.) against its own candidate inputs per position — it is *not* a single regression per position that takes every feature and predicts total FPL points directly. Two reasons this matters: (1) Phase 3.2's calibration check verifies each component *separately* ("when the model says 40% clean sheet, does it happen ~40% of the time"), which only makes sense if each component has its own independently-fit probability; (2) the point *conversion* rates (10/6/5/4 per goal, thresholds per position, etc.) are already fixed exactly by the game's rules and encoded in `scoring.py` (Phase 0) — there's nothing to learn there, only the underlying *rates* (probability of a goal, probability of clearing the DC threshold) are genuinely uncertain and worth fitting from data.

The resulting coefficients **are** the measured answer to your original question — which free stats are the biggest indicators of points, per position, no longer a guess. And because the whole regression is fit purely on FPL-API-and-Understat data, it's fully reproducible from historical snapshots alone with zero live-data dependency — a meaningfully simpler validation story than a model with a market-blend term.

Start with interpretable linear/logistic regression. Only reach for `xgboost` as a *benchmark* — if it materially beats the interpretable model in the backtest, consider adopting it, accepting the loss of interpretability; if it doesn't, you've confirmed the simple model is enough. Some position/component combinations will have thin samples (e.g. forwards rarely clearing the defensive-contribution threshold) — check for overfitting/multicollinearity empirically once fitting on real data, rather than assuming it away up front.

### 2.9 Simulation layer (`simulate.py`) — for ceiling/floor
A single expected number hides the *shape* of the outcome, and captaincy and chips need the shape. Simulate the gameweek thousands of times (vectorised in numpy), seeded for reproducibility. The spread across runs yields the full distribution: **median, floor (safe baseline), ceiling (haul potential), and P(10+ points)**.

Each run follows one coherent generative story, correlating components that genuinely move together in a real match rather than drawing each independently (see the note in the Phase 2 intro):

1. **Minutes** — for every player, draw from their bucket probabilities (2.1), then draw actual minutes from the conditional expected-minutes distribution for whichever bucket landed.
2. **Match script** — for every fixture, draw a correlated scoreline pair for the two teams (the Dixon-Coles-adjusted bivariate Poisson from 2.4, home/away split), rather than drawing each team's goals independently. Each team's Λ here is the sum of that team's on-pitch players' individual goal rates from 2.2 (not a separately-fit team-level parameter) — the internal-consistency requirement noted in 2.4, so the clean-sheet model and the individual goals model are always reading from the same number.
3. **Individual goals/assists** — apportion each team's drawn goal total across its players who are on the pitch that run, weighted by each player's share of the team's total non-penalty xG/xA (a multinomial draw conditional on the team total from step 2), plus the penalty sub-model (2.2) layered on top for the designated taker. This keeps individual outputs consistent with the team's actual scoreline in that run, rather than each player rolling goals independently of what their team just did.
4. **Clean sheet / goals conceded** — read directly off the same scoreline from step 2 (0 goals against = clean sheet), so it's automatically consistent with the goals a player just got credited with, not a separately-drawn number.
5. **Defensive contribution** — drawn independently per player (Negative Binomial or Poisson per 2.5, opponent-possession-adjusted) — this one genuinely doesn't hinge on the scoreline, so it doesn't share the match script.
6. **Bonus** — computed via the regression proxy (2.6) applied to each player's *realized* stats for that run, not drawn independently — so a player who happened to get a goal+assist in this particular run also gets a bonus bump in this run, not just on average.
7. **Cards** — drawn independently per player, pure historical rate (2.6).
8. Sum everything for that run → one full points outcome. Repeat thousands of times to build the distribution.

This is what lets captaincy distinguish "highest average" from "highest ceiling" from "safest floor" — three genuinely different recommendations one number can't express.

**Definition of done for Phase 2:** the engine produces, for every player, a per-gameweek and multi-gameweek projection with a full distribution and an attached component breakdown, and it runs end-to-end on a historical snapshot without leakage. It is *not yet validated* — that's Phase 3.

---

# PHASE 3 — Backtesting & validation (the quality gate)

**Goal:** prove the engine is actually good, find where it's wrong, improve it, and repeat — until it clears an explicit bar. This phase is what separates a trustworthy tool from a plausible-sounding one.

### 3.1 Walk-forward harness (`harness.py`)
Because this is time-series data, **never** use random train/test splits — that leaks the future. Use walk-forward validation: fit on gameweeks 1…N, predict N+1 using only the N+1 pre-deadline snapshot, record the prediction, roll forward, repeat across the season(s). This mirrors exactly how the engine will be used in real life.

**Refit every gameweek, expanding window.** Refit the regression layer (2.8) after every gameweek rather than periodically — cheap enough for interpretable linear/logistic regression that there's no real trade-off, and it matches exactly what real deployment will do anyway. Use an **expanding window** (train on everything from gameweek 1 to N, growing every step) rather than a sliding window that discards older data — the EWMA decay already built into the rate-stat inputs (Phase 1) gives recency its due weight; a sliding window on top would be a second, redundant recency mechanism stacked on the first.

### 3.2 Metrics (`metrics.py`)
Track at several levels, because a single "is it right" number hides which part is failing:
- **Player-level accuracy:** mean absolute error and RMSE of predicted vs actual points, overall and **per position**.
- **Bias detection:** is the model systematically over/under-rating any group (premium forwards, budget defenders, differentials)? Residual analysis, not just average error.
- **Component calibration:** separately check each component — e.g. when the model says "40% clean sheet," do clean sheets actually happen ~40% of the time? This tells you *which component* to trust and which to treat cautiously.
- **Captaincy hit-rate:** the single most decision-relevant metric — how often was the recommended captain actually the highest scorer among the eligible options? Track this specifically.
  - **Eligible options = your actual starting XI that gameweek**, not the full 15 — bench players were never really in contention for the armband, so this is the decision-realistic slice.
  - Track **two versions side by side**, not one: the raw hit-rate (post-deadline misfortune — an unexpected injury, a red card — still counts as a miss, because a real captaincy tool has to live with that risk every week regardless of fault), and a secondary **"played-as-expected" hit-rate** restricted to gameweeks where the captain actually played their expected minutes. This separates "the model picked wrong" from "the pick was right but football happened," rather than letting one blended number hide which failure mode is actually occurring. Report both on the Model Performance page.

### 3.3 Baselines (`baselines.py`)
An accuracy number is meaningless without something to beat. Compare the engine against:
- **Template captain** — captaining the highest-owned/highest-price player each week.
- **Naive form model** — projecting last-few-weeks' points forward.
- **Pure xG model** — a version using only raw xG/xA per 90 with no minutes-model gating or regression weighting, to prove the fuller engine earns its extra complexity over the simplest possible stats approach.

If the engine can't beat these over a meaningful sample, it isn't earning its complexity yet. (The market-based comparison — whether the live odds overlay adds anything beyond the stats-only engine — is a separate, ongoing comparison run in Phase 6, once real live odds have accumulated; see Phase 4b.)

**"Beats the baselines" is a statistical test, not a point-estimate comparison.** This is the literal gate to the entire web app (3.6), and football's match-to-match variance is large enough that a small, genuine edge and a small, lucky edge can look identical over a modest sample — exactly the "variance masquerading as skill" risk flagged in the risk register below, which needs an actual mechanism here rather than just a general warning. Use a paired test on the per-gameweek difference between the engine and each baseline (e.g. paired bootstrap or paired t-test on the MAE difference; a binomial/permutation test on the captaincy hit-rate difference), and require the confidence interval to exclude zero before counting a baseline as "beaten." A raw "our number is better" comparison isn't sufficient to cross this gate.

### 3.4 Prediction logging (`logs/predictions/`)
Every prediction is written immutably, timestamped, before the deadline, tagged with the **model version** (git tag/hash). This prevents unconsciously judging the model against outcomes you now know, and it lets you attribute accuracy changes to specific engine versions — so you can tell whether a "tweak" genuinely helped or just got lucky.

### 3.5 The iteration loop
Backtest → find where errors cluster → form a hypothesis (e.g. "defensive-contribution predictions are systematically low for full-backs") → adjust that component or its weighting → re-backtest → confirm the change actually improved out-of-sample accuracy (not just in-sample fit). Repeat. Most of your modelling time lives here.

### 3.6 Engine Definition of Done — the gate to Phase 5
Do **not** start the web app until all of these hold over a meaningful sample (ideally a full season; a good 5-gameweek run tells you almost nothing given football's variance):
- [ ] Beats all three baselines on player-level accuracy and captaincy hit-rate, **statistically** (3.3) — not just on the raw point estimate.
- [ ] No severe, uncorrected systematic bias in any position or price tier.
- [ ] Each component is reasonably calibrated, or its miscalibration is understood and documented.
- [ ] Predictions are being logged immutably and tied to model versions.
- [ ] You personally trust it enough to act on it — the honest final check.

**No carve-out for defensive contribution or bonus.** Both are held to exactly the same bar as every other component, not a relaxed one. Defensive contribution enters this gate with only **one season of native history** (2025/26, the rule's first year) versus a decade-plus for goals/assists/clean sheets, since 2026/27 is the season this engine targets. Bonus has a related but distinct issue: the BPS formula itself was reworked for 2026/27 (2.6), so pre-2026/27 "actual bonus received" reflects a superseded formula — its regression is instead trained on bonus **recomputed under the current formula from raw match-event data** (2.6), which restores real depth rather than leaving it native-history-thin like DC. In both cases the statistical test in 3.3 accounts for whatever sample thinness remains correctly on its own: a thinner sample produces a wider confidence interval, so a genuine signal should still clear the bar despite less native history, and a signal that doesn't clear it is real information (not ready to trust yet) rather than something to special-case around.

---

# PHASE 4 — Feature logic (still pre-web, pure functions)

**Goal:** build the six features' *decision logic* as testable modules over engine output — no UI yet. This is a deliberate insight: the features are just different questions asked of the same projections, so they can be built and unit-tested from the command line before any screen exists.

- **Fixtures (`fixtures.py`):** a custom difficulty rating per team per fixture, built from the engine's own inputs (opponent expected goals conceded, opponent attacking strength, home/away) rather than FPL's arbitrary colour scale. Feeds the other features *and* surfaces as its own view later.
- **Captaincy (`captaincy.py`):** rank players by expected points, with the simulation distributions exposing a "safe" pick (highest floor) vs a "punt" pick (highest ceiling). **Ranks the full player pool, not just your 15** — your squad's eligible options are highlighted/filterable within that full ranking, since seeing how your best captain option stacks up against the entire league is useful context for gauging differentials and spotting transfer targets, even though the armband itself can only ever go to a player you actually own (an FPL structural rule, not a design choice). Attach the reasoning for each. **No single headline pick** — deliberately kept simple rather than mini-league-rank-aware (which the shared "My Team" state's mini-league ID would make possible): show the highest-EV, floor, and ceiling picks side by side in the app, and let you choose, rather than the engine picking one "the" recommendation based on your league position.
- **Transfers (`transfers.py`):** the multi-gameweek planner. Compare summed projected points over the horizon for sell vs buy candidates, net of the −4 hit, with correct **sell-price** handling (FPL's rule: you only get back half the profit on a risen player, rounded down). Distinguish forced transfers (injury/suspension) from optional upgrades, and reason about sequencing (bank a transfer, use it on a fixture swing). **v1 scope cut:** a greedy one-swap-at-a-time comparator, not a full multi-transfer combinatorial search — it evaluates each sell/buy pair in isolation rather than searching over which *combination* of several simultaneous transfers (e.g. spending multiple banked transfers, or taking a −4 for a double swap) maximizes total expected points across the horizon. This mirrors the "don't over-invest in the smaller pieces up front" logic already applied to bonus and cards in Phase 2 — revisit with real multi-transfer search only if real use shows the greedy version is missing genuinely valuable multi-move plans.
- **Chips (`chips.py`):** per-chip "value now vs waiting" evaluators sharing the same forward-looking horizon as transfers (don't duplicate blank/double-gameweek logic). Bench Boost and Triple Captain compare this week's projection against the **best week available anywhere in the planning horizon** (not a historical rolling average) — the actual question a use-it-once chip needs answered is "is this the peak week within reach, or is patience worth more," which a rolling-average comparison against your own history doesn't answer. A rolling average is a reasonable secondary sanity check (is this a good week at all, historically), but not the primary now-vs-waiting comparison. Free Hit = exposure to blanks/doubles; Wildcard = squad value drifting below optimal — both of these already read across the horizon correctly.

Each is a pure function: `(my_team_state, projections) → recommendation + reasoning`. Unit-test each against fixed inputs with known expected outputs.

**Shared state — "My Team":** define one canonical object all three decision features read from: current 15 (with purchase and current price), starting XI + bench order, captain/vice, bank, **sell price per player**, free transfers available (banked up to 5), chips used/remaining, and mini-league ID(s). Most of this the FPL API returns from your team ID; sell price is the one thing needing careful transaction-history tracking.

**2026/27 chip allowance, confirmed:** eight chips total — Wildcard, Free Hit, Triple Captain, and Bench Boost, one full set per half of the season. The first-half set must be used before the Gameweek 19 deadline (13:30 GMT, 2 January 2027) and does **not** carry over into the second half; the second half's set then refreshes. Only one chip playable per gameweek. Free Hit specifically can't be played in Gameweek 1, and if used in Gameweek 19, the second Free Hit can't be played in Gameweek 20.

**Definition of done:** all six features return correct, reasoned recommendations from the command line, covered by unit tests, running on real engine projections.

---

# PHASE 4b — Live market overlay (separate from the core engine)

**Goal:** bring bookmaker odds in as a live, decision-time signal that sits *alongside* the stats-only engine rather than inside it — giving you the market's speed advantage on breaking news without compromising the engine's clean backtestability.

This module is deliberately kept out of `engine/` and out of every backtest. It has no historical-depth requirement to satisfy, because it's never asked to reproduce the past — only to react in the moment.

### 4b.1 Odds client (`market_overlay/odds_client.py`)
Pulls pre-match odds (anytime scorer, clean sheet / match result, over-under goals) **once per gameweek**, shortly before the deadline. One snapshot a week is all this needs, so free-tier request limits are ample — build on one established, well-documented provider.

### 4b.2 Divergence detection (`market_overlay/divergence.py`)
For each player, convert the odds to an implied probability (removing the bookmaker's margin) and compare it against the stats engine's own projection for the same outcome. Surface the gap, don't silently resolve it:
- **Small divergence:** no action — the two views broadly agree.
- **Large divergence:** flag it. This usually means the market has priced something recent the stats model can't see yet — a tactical change, a fitness boost, a confirmed nailed-on run of starts — since the market reacts to breaking news faster than a model retrained on historical patterns can. (Note the FPL API's own injury/suspension flags already cover the *blunter* version of this for the minutes model — the overlay specifically catches the subtler, more speculative signals odds embed that team-news flags don't.)

### 4b.3 Whether to let it adjust the number
Don't decide this up front — decide it empirically, using the same discipline as Phase 3. Once the Phase 6 prediction log has accumulated real gameweeks, run **stats-only** and **stats+market-adjusted** as two separately tracked and logged variants, and let the season tell you whether the overlay earns its keep versus just being a transparency flag the reasoning view surfaces.

**Definition of done:** the overlay runs weekly, independently of the engine, flags meaningful stats-vs-market divergence per player, and both variants (with/without the adjustment) are logged separately from day one of live use so the comparison in Phase 6 is honest.

---

# PHASE 5 — Web application

**Goal:** a presentation layer over the (now trusted) engine and feature logic. Because the backend is Python, it imports `engine/` and `features/` directly — no logic is reimplemented.

### 5.1 Backend (`api/`, FastAPI)
Thin endpoints that call existing functions: projections, captaincy, transfers, chips, fixtures, player search, and the accuracy metrics from Phase 3. A weekly job refreshes data, regenerates projections, and logs predictions. The API mostly serves precomputed results.

### 5.2 Frontend (`web/`, React)
Implements the agreed sitemap and interaction pattern:

- **Dashboard (home):** a row of compact, equal-weight cards — Captain, Transfers, Chips, plus a lighter "urgent" card (e.g. injury flags). Each shows only its headline verdict. This is the "compact summary of everything" you asked for.
- **Interaction pattern (applies everywhere):** *verdict first, reasoning on click.* A card shows the answer; one click reveals what's informing it (component breakdown, driving stats, ceiling/floor, and — where relevant — the Phase 4b market-divergence flag if the odds and the engine disagree meaningfully); a further "view full breakdown" navigates to the dedicated page. Dashboard cards are previews, so the home screen stays light rather than turning into four expanding panels.
- **Core screens:** My Team (the shared-state view), Captaincy, Transfers (the multi-GW planner), Chips (season dashboard, per-chip value signal).
- **Supporting screens:** Fixtures (sortable difficulty ticker), Player Search/Comparison (any player's full projection + component breakdown), **Model Performance** (surfaces the Phase 3 accuracy metrics live — prediction vs actual, captaincy hit-rate, component calibration, **plus the stats-only vs stats+market comparison from Phase 4b once enough live gameweeks have accumulated**), Settings (FPL team ID, mini-league ID, planning horizon default).

**Definition of done:** you can open the app, see every recommendation at a glance, click into any of them to see exactly what drove it, and watch the accuracy page confirm (or challenge) that the engine is helping.

---

# PHASE 6 — In-season operation

**Goal:** keep it running and honest across the season.
- Weekly: refresh data, capture the pre-deadline snapshot, regenerate projections, pull the Phase 4b odds snapshot, log **both** the stats-only and stats+market-adjusted predictions.
- Periodically: re-fit the regression as this season's data accumulates (especially the defensive-contribution component); watch for drift.
- Periodically: review whether the market overlay is actually improving accuracy (Phase 4b.3) — once enough live gameweeks exist, this stops being a hunch and becomes a measured comparison on the Model Performance page.
- Continuously: track live accuracy vs the baselines and vs your own past-season performance. Decide your personal "worth trusting over my gut" threshold and hold the tool to it.

---

## Version control strategy

- **Trunk-based with short feature branches.** `main` always runs.
- **Tag every engine version** (`engine-v0.3`, etc.) and record that tag with each logged prediction, so accuracy is always attributable to a specific model. This is how you prove a tweak helped rather than got lucky.
- **Never commit** data snapshots, secrets, or the SQLite store — gitignore them. If you want versioned data, use DVC or git-lfs rather than raw git.
- Keep `notebooks/` out of the critical path — exploration only; anything real gets promoted into a tested module.

---

## Risk register (know these going in)

| Risk | Mitigation |
|------|------------|
| Minutes prediction is hard and dominates error | Invest most modelling effort here; treat it as the first-class problem it is |
| Defensive contribution has only one season of history (new-for-2025/26 rule) | Proxy from per-90 action rates; the statistical baseline test (3.3) naturally accounts for the shallower sample via a wider confidence interval, no separate DoD carve-out needed (3.6) |
| BPS formula reworked for 2026/27, making pre-2026/27 "actual bonus received" reflect a superseded formula | Recompute historical bonus under the current formula from raw match-event data (2.6) rather than regressing on the stale column; no separate DoD carve-out needed (3.6) |
| Market overlay (Phase 4b) can only be evaluated on this season's live data | Kept fully separate from the backtested core; log both variants from day one so the comparison is honest as soon as enough gameweeks exist |
| Cold-start at season open | Lean on prior-season priors early; widen uncertainty in the first few gameweeks |
| Overfitting in backtests | Walk-forward validation only; judge on out-of-sample, never in-sample fit |
| Data leakage flattering the model | Point-in-time snapshots are mandatory, not optional |
| Variance masquerading as skill (or failure) | Judge over a full season, not a hot/cold streak |

---

## Build order at a glance

```
Phase 0  Setup ─────────────────────────────────┐
Phase 1  Data foundation (FPL API + Understat    │  ENGINE
         only — no odds dependency) ─────────────┤  (stats-only,
Phase 2  Engine: minutes → components →          │   fully backtestable,
         aggregate → regression → simulation      │   prove in isolation)
Phase 3  Backtest, tune, iterate ────────────────┤
         ▼ GATE: Engine Definition of Done ───────┘
Phase 4  Feature logic (captaincy/transfers/      ┐
         chips/fixtures) — pure functions          │
Phase 4b Live market overlay — odds pulled at      │  APP
         decision time only, tracked separately    │  (build on trust)
Phase 5  Web app (FastAPI + React)                 │
Phase 6  In-season operation, incl. ongoing        │
         stats-only vs stats+market comparison ────┘
```

The gate between Phase 3 and Phase 4 is the heart of this plan: everything above it earns the right to build everything below it. Note Phase 4b sits entirely on the app side of that gate — the odds it uses never touch the backtested engine.
