# Prediction Engine — Accuracy Improvement Plan

Findings from the first **real-data** walk-forward backtest of the Phase 2 engine, and the
prioritised changes that follow from them. This is a companion to `BUILD_PLAN.md`: the plan says
*what* to build, this says *what the evidence now says to fix*. Every claim below is backed by a
number from the run described in "Provenance", and the full diagnostic output is reproduced in the
appendix so the reasoning can be audited without re-running anything.

**Status:** written after commit `af6a86e` ("Phase 3: backtesting & validation"). Execution-order
steps 1–5 have since been implemented and re-measured — see "Implementation status /
re-measurement" immediately below. Tier 3.1–3.4 and Tier 4 remain unimplemented, as scoped.

> **Superseded in part.** `ENGINE_IMPROVEMENTS_2.md` (written after commit `8f6d633`) shows that
> three numbers in the re-measurement table below are not comparable to the run they are compared
> against — the scored sample shrank 19%, double gameweeks are no longer collapsed, and the reported
> Spearman is the starters-restricted variant. See its Corrections 3–5 before quoting the table, and
> its Tier A for the fixes. Everything else here — including Corrections 1–2 and the Tier 3/4
> conclusions — stands.

**Headline:** the engine's accuracy is bottlenecked almost entirely by the **minutes model** (2.1),
exactly as `BUILD_PLAN.md` predicted it would be. Misallocated points on players who never appeared
cost **0.438 MAE — about 6× the engine's entire current edge over the naive baseline.** Two
conclusions from the first pass at reading the backtest were also wrong, and are corrected below.

---

## Implementation status / re-measurement (2026-07-26)

Execution-order steps 1–5 — Tier 2 (2.1–2.4, the gate/metrics fixes), promoting the driver to the
versioned `backtest/run_season.py`, Tier 1.1 (minutes features), Tier 1.3 (real Understat npxG +
penalty sub-model), and Tier 1.2 (wiring the regression/fitting layer for the five previously-inert
constants) — have been implemented and re-measured against a real, cached 2025/26 run:

```
python -m backtest.run_season --season 2025 --report-path backtest/reports/2025-26.txt
```

| Metric | Before | Target | After | Hit target? |
|---|---|---|---|---|
| Overall MAE | 1.6683 | < 1.45 | **1.5170** | No — improved 9.1%, short of target |
| Overall RMSE | 2.5620 | < 2.45 | **2.4419** | **Yes** |
| AUC "played at all" | 0.8107 | > 0.88 | **0.8653** | No — improved substantially, short of target |
| Predicted-points mass on zero-minute players | 6,618 | < 3,500 | **4,007** | No — a 39.5% reduction, short of target |
| Mean actual points of top-10 picks | 4.947 | > 5.5 | **4.653** | No — regressed vs. the original run |
| Captaincy raw hit-rate | 0.265 (n=34) | > 0.30, significant | not re-measured this pass | — |
| Clean-sheet MACE (gated, correct comparison) | 0.0157 | hold < 0.03 | **0.0187** | **Yes** (held, though a touch higher) |
| Beats constant-median baseline (MAE) | +2.5% | > +10% | **≈+10.6%** (paired-bootstrap mean diff −0.179 vs. a ≈1.70 baseline MAE) | **Yes** (narrowly) |

Gate verdict this run (`backtest.gate.evaluate_definition_of_done`):

```
[x] Beats all baselines, statistically (3.3)
[x] No severe systematic bias in any position/price tier (3.2)
[x] Each component reasonably calibrated (3.2)
[x] Predictions logged immutably, tagged by model version (3.4)
[ ] Personally trusted enough to act on (3.6)

NOT YET — gate to Phase 5
```

**Reading this honestly:**

- Every metric this pass targeted moved in the right direction — MAE, RMSE, AUC, zero-minute mass,
  and the baseline margin all improved, several by a wide margin — but three explicit numeric
  targets (MAE < 1.45, AUC > 0.88, zero-minute mass < 3,500) were not fully reached. The gap
  narrowed substantially rather than closed. Tier 1.1's ceiling estimate (26% MAE improvement from
  perfect minutes knowledge) was never claimed to be reachable in one pass — recoverable retrospectively
  was only ever "some," since two of the three originally-defaulted features
  (`chance_of_playing_next_round`, `status`) are genuinely live-only and still can't be reconstructed
  from historical data (see "Why the model is under-performing" below) — meaning the engine should,
  per that section's own note, perform better live than this backtest indicates.
- The bias and calibration gates now both pass **for real reasons**: the Tier 2.3 effect-size floor
  is doing its job (no position is flagged purely on sample size), and clean-sheet MACE stayed
  comfortably under 0.03 — not because a broken measurement flagged the wrong thing, as Corrections
  1–2 below describe for the original run.
- **Top-10 mean actual points regressed** (4.947 → 4.653) rather than improving, the one targeted
  metric that moved the wrong way on a run where everything else improved. Not diagnosed further in
  this pass; plausible causes worth checking before trusting this version for captaincy: the new
  minutes features shifting who ranks at the very top of the pool in ways that don't track "who
  actually hauls," a harder 34-gameweek sample of top-10 picks this time round, or an interaction
  between the newly-fitted `FittedConstants` and the goals/clean-sheet components at the extreme
  high end of the distribution.
- Captaincy hit-rate was not re-measured — `backtest.run_season.score_season` doesn't currently wire
  in `backtest.metrics.captaincy_hit_rate` (it needs a squad-selection rule, same caveat as the
  original run's stand-in squad); a gap in the scoring harness, not the engine itself.
- The five Tier 1.2 constants are now refitting every gameweek instead of sitting on untuned
  defaults — except the two saves constants (`save_conversion_rate`, `away_shot_multiplier`), which
  still fall back to their defaults every refit since goalkeepers remain out of scope (no shots-on-
  target data, Tier 3.2). The penalty conversion rate has an inherently tiny per-season sample, as
  flagged as a risk during design.
- Tier 3.1 (multi-season), 3.2 (GK/saves), 3.3 (bonus BPS recompute — still blocked on the missing
  2026/27 BPS numeric table), 3.4 (DC opponent-possession) and Tier 4 (EWMA halflife) remain
  **unimplemented**, exactly as scoped going in.

Cached raw pulls live under `data_store/season_cache/` (gitignored); re-running without `--refresh`
reuses them.

---

## Provenance — how this was measured

| Item | Value |
|------|-------|
| Data source | `vaastav/Fantasy-Premier-League` community archive, `data/2025-26/gws/merged_gw.csv` |
| Season | 2025/26, complete (38 gameweeks) |
| Raw outfield rows | 26,330 (744 outfield players, 20 teams) |
| After collapsing double gameweeks | 25,955 `(player, gameweek)` rows |
| After dropping rows with no prior history | 16,053 |
| Gameweeks predicted | 34 (GW 2–4 skipped: `min_training_gameweeks=3`) |
| Player-gameweek predictions scored | 15,097 |
| Model version tag on the logged predictions | `ca2995c-dirty` |
| Harness | `backtest/harness.py` — expanding window, refit every gameweek |
| Point-in-time features | `engine/rates.py` EWMA, `shift(1)` so a match never informs its own prediction |

Actual points in the scored sample: **mean 1.871, median 1.0, std 2.813** — heavily zero-inflated
and right-skewed. That distribution shape matters for interpreting every accuracy number below.

### Known limitations of *this specific run*

These are data-availability and wiring gaps in the backtest driver, not defects in the engine
modules. They bound how much the run can tell us, and several are themselves the improvements
recommended later.

1. **Goalkeepers excluded entirely** (3,427 rows, ~13% of the pool). The archive carries no
   opponent shots-on-target, so the saves component (2.6) had nothing real to run on. The `saves`
   points line summed to exactly `0.0000`.
2. **`opponent_possession_share` was a constant `0.5`** for every fixture, so the
   defensive-contribution opponent adjustment (2.5) was a no-op — the component ran on raw per-90
   action rates only.
3. **Three of the minutes model's five features were hardcoded constants:**
   `fixture_congestion=0.0`, `chance_of_playing_next_round=100.0`, `status_score=available("a")`.
   See "The dominant finding" — one of the three is trivially recoverable and should never have been
   defaulted.
4. **The regression layer (2.8) was never wired in.** `fit_fn` fit only `MinutesModel` and
   `BonusModel`. Every other component ran on hand-coded functional forms with untuned literals.
5. **The "npxG" input was FPL's `expected_goals`, which includes penalty xG.** The goals model
   (2.2) expects *non-penalty* xG plus a separate penalty sub-model; the sub-model was passed
   `team_expected_penalties=0` and contributed nothing.
6. **Bonus was regressed against the raw `bonus` column** — the stale pre-2026/27 BPS formula that
   `BUILD_PLAN.md` 2.6 explicitly warns against using directly.
7. **Bonus training features used each row's realised minutes**, while prediction uses modelled
   expected minutes — a train/serve skew.
8. **Double gameweeks** were summed for ground truth; opponent-adjustment features took whichever
   fixture appeared first that gameweek (~1.6% of rows affected).

### Reproduction scripts

The driver and diagnostics currently live in the session scratchpad, which is **ephemeral**:

- `real_backtest.py` — data prep, walk-forward run, metrics, baselines, logging, gate
- `diagnose.py` — hypotheses H1–H6
- `diagnose2.py` — hypotheses H2c, H7–H10

They saved these artefacts: `/tmp/real_ground_truth.parquet`,
`/tmp/real_engine_predictions.parquet`, `/tmp/real_naive_form_baseline.parquet`,
`/tmp/real_template_baseline.parquet`.

**Recommended:** promote the driver to a versioned module (e.g. `backtest/run_season.py`) so this
is repeatable and reviewable, per the plan's "production logic always lives in versioned modules,
never in notebooks" principle. Right now the only real backtest we have is not reproducible from
the repo.

---

## Corrections to the record

Two conclusions from the first reading of the backtest output were wrong. Both mattered, and both
were caused by the *measurement*, not the model.

### Correction 1 — the clean-sheet model is well calibrated, not badly miscalibrated

The first pass reported "mean absolute calibration error 0.16, overpredicts clean sheets almost
everywhere." That compared the engine's `clean_sheet_probability` against each player's actual
`clean_sheets` column. But `engine/models/clean_sheets.py:123-125` documents that probability as
explicitly **team-level and not gated by any individual's minutes**, whereas FPL's `clean_sheets`
flag requires the player to have played **60+ minutes**. Two different quantities.

Comparing like-for-like — team probability × P(60+ min), which is what the points path in
`aggregate.py:111-113` already does correctly — the error nearly vanishes:

| Comparison | Mean predicted | Mean actual | MACE |
|---|---|---|---|
| As the backtest ran it (team prob vs 60-min-gated actual) | 0.2752 | 0.1150 | **0.1602** |
| Apples-to-apples (team prob × P(60+) vs same actual) | 0.1153 | 0.1150 | **0.0157** |

And at pure team level, against the true 2025/26 clean-sheet base rate computed from actual
scorelines:

| Quantity | Value |
|---|---|
| Real team-match clean-sheet rate, 2025/26 (n=760 team-matches) | 0.2553 |
| Engine mean predicted team clean-sheet probability | 0.2739 |
| Relative over-prediction | +7.3% |

**Conclusion:** the clean-sheet component is one of the engine's *better* parts. A mild ~7%
optimism at team level is worth a look (candidate cause: the Dixon-Coles `rho = -0.1` default, or
the absence of the home/away rate split that 2.4 specifies but the backtest didn't supply), but
this is a refinement, not a defect.

> **Process note.** A first attempt at the team-level check was also broken: it derived "team kept a
> clean sheet" as `max(goals_conceded == 0)` across the team's players, which is ~always true
> because a late substitute who came on at 3–0 has `goals_conceded = 0` for their own window. It
> returned an absurd 0.885 base rate. The corrected check reads the actual fixture scoreline. Flagged
> here because it is exactly the class of silent-but-wrong measurement bug that
> `BUILD_PLAN.md` 1.1 warns about for the ID crosswalk, and it caught us twice in one component.

### Correction 2 — the "severe MID bias" is a false positive, and the gate missed the larger real effect

`backtest/gate.py` flags bias when a one-sample t-test on per-group residuals rejects "mean residual
is zero" at p < 0.01. On real data:

| Position | Mean residual (pts) | As % of that position's mean actual | n | t | p | Flagged "severe"? |
|---|---|---|---|---|---|---|
| DEF | +0.0034 | 0.2% | 5,662 | 0.095 | 0.924 | No |
| FWD | **−0.1357** | **6.8%** | 1,912 | −2.206 | 0.028 | **No** |
| MID | −0.0728 | 4.0% | 7,523 | −2.589 | 0.0096 | **Yes** |

The gate flags midfielders (4.0% effect) and clears forwards (6.8% effect) — purely because
midfielders have 4× the sample. This is significance-versus-practical-importance confusion, and it
gets structurally worse as data accumulates: with a full season of multiple years, *every* group's
tiny non-zero bias becomes significant and the Definition-of-Done gate becomes unpassable.

**Conclusion:** the bias criterion needs an **effect-size floor AND significance**, not significance
alone. Suggested: flag only when `|mean residual| > 0.25 points` (or > 10% of the group's mean
actual) *and* p < 0.01. Neither MID nor FWD would flag under that rule; a genuine 1-point systematic
error would.

### Not a correction — a worry that checked out fine

`paired_bootstrap_test` resamples **individual rows** i.i.d. Players in the same match share
shocks (a red card, a 5–0 drubbing, a manager resting the squad), so the effective sample is
nearer the number of fixtures than the number of rows, and an i.i.d. bootstrap should understate
uncertainty. Re-running with block bootstraps that respect that clustering:

| Bootstrap scheme | Blocks | 95% CI on paired MAE difference | Excludes zero? |
|---|---|---|---|
| i.i.d. by row (as shipped) | — | [−0.0585, −0.0184] | Yes |
| Block by fixture | 335 | [−0.0619, −0.0146] | Yes |
| Block by gameweek | 34 | [−0.0652, −0.0128] | Yes |

The interval widens as expected but the conclusion holds. **The shipped MAE significance result is
robust to clustering.** Adding a `block_by` option to `paired_bootstrap_test` is still worth doing
for rigour (and it will matter more for captaincy, where n is small), but nothing needs revisiting.

---

## The dominant finding — the minutes model is the bottleneck

`BUILD_PLAN.md` 2.1 calls the minutes model "the single highest-value and hardest part of the
engine" and 2.1's closing line says "get minutes wrong and nothing downstream can save the
projection." The real backtest confirms this quantitatively, and it is not close.

### The evidence

| Metric | Value |
|---|---|
| Rows where the player actually played **zero** minutes | **38.1%** (5,752 of 15,097) |
| Mean predicted `expected_minutes` on those zero-minute rows | 24.51 (ideal: ~0) |
| Mean predicted `expected_points` on those zero-minute rows | **1.151** (actual: 0.0) |
| Total predicted-points mass assigned to players who never appeared | **6,618 points** |
| MAE of `expected_minutes` vs actual, all rows | 25.32 minutes |
| MAE of `expected_minutes` vs actual, rows that did play | 25.82 minutes |
| Spearman(`expected_minutes`, actual minutes) | 0.6376 |
| **AUC for "played at all" using `expected_minutes`** | **0.8107** |

### Why this dominates everything else

Those 6,618 misallocated points are `6618 / 15097 = 0.438` of pure MAE contribution. Since the error
on a zero-minute row is just the prediction itself, perfect minutes knowledge would take MAE from
**1.668 → 1.230, a 26.3% improvement.** For scale:

| Quantity | MAE impact |
|---|---|
| Engine's entire current edge over the naive-form baseline | 0.071 |
| Cost of misallocating points to players who didn't play | **0.438** |
| Ratio | **6.2×** |

The component breakdown says the same thing from another angle — the largest single line in the
average projection is simply "will this player turn up":

| Component | Mean predicted points | Share of total |
|---|---|---|
| **appearance** | **+1.0610** | **58.3%** |
| goals | +0.3058 | 16.8% |
| clean_sheet | +0.2405 | 13.2% |
| bonus | +0.1427 | 7.8% |
| assists | +0.1169 | 6.4% |
| defensive_contribution | +0.1098 | 6.0% |
| cards | −0.0914 | — |
| goals_conceded | −0.0669 | — |
| saves | +0.0000 | (GK excluded) |
| penalty_misses | +0.0000 | (sub-model inert) |
| **Total predicted** | **+1.8184** | vs actual mean **1.8706** |

Overall points calibration is good (2.8% under), which is reassuring — but it is an *average* over
a population where the engine is simultaneously over-predicting non-players and under-predicting
starters. Aggregate calibration hides the problem that per-row accuracy exposes.

### Why the model is under-performing, and what is actually fixable

Only two of the five features in `engine/models/minutes.py:FEATURE_COLUMNS` carried signal
(`recent_start_rate`, `recent_minutes_ewma`). Of the three constants:

| Feature | Why it was constant | Recoverable? |
|---|---|---|
| `fixture_congestion` | Defaulted to `0.0` in the driver | **Yes — trivially.** `kickoff_time` is column 20 of the source data. Games-in-last-N-days is fully derivable. This was defaulted for no good reason. |
| `chance_of_playing_next_round` | Not retained by the archive (live-only FPL field) | Not retrospectively. Phase 1.2 snapshots capture it going forward. |
| `status` → `status_score` | Not retained by the archive (live-only FPL field) | Not retrospectively. Same as above. |

**An important nuance for expectations:** because two of those three are genuinely live-only fields
that the engine *will* have at real decision time, the engine should perform **better live than it
backtests**. The backtest is a pessimistic bound on the minutes component, not an unbiased estimate.
That is an argument for getting Phase 1.2's daily snapshot capture running as early as possible —
every day it isn't running is a day of unrecoverable training data for the highest-leverage model in
the engine.

---

## Tier 1 — highest-value model changes

### 1.1 Feed the minutes model properly

**Target:** AUC for "played at all" from **0.811 → 0.88+**; recover a meaningful share of the 0.438
MAE that is currently misallocated.

Restore the one wrongly-defaulted feature and add derivable ones the plan didn't originally
enumerate but which cost nothing:

- **`fixture_congestion` from `kickoff_time`** — games played in the last *N* days per team.
  Already specified in 2.1's input table; just needs wiring.
- **Days since the player's last appearance.** A player who hasn't featured in 4 weeks is
  categorically different from one rested for one match. This is the strongest available *proxy* for
  the missing injury/availability flags in historical data.
- **Consecutive zero-minute streak length.** Directly separates "deep squad / unavailable" from
  "rotation risk", which is precisely the discrimination the model is failing at.
- **Start rate at multiple horizons** (e.g. last 3, 6, 15 matches) rather than one EWMA level. A
  single smoothed level cannot express "recently nailed-on but historically fringe" versus the
  reverse, and rotation decisions are driven by the trend.
- **Team-level rotation propensity** — some managers rotate systematically. A team-level random
  effect or simple per-team start-rate dispersion is cheap and captures real signal.

Consider also promoting P(0 minutes) to its own well-calibrated classifier and checking it with
`component_calibration`, since it is now demonstrably the highest-value probability in the engine
and is not currently calibration-checked at all.

### 1.2 Wire the regression layer in — it was never used

`engine/regression.py` (`PerPositionRegression`, `variance_inflation_factors`,
`benchmark_against_xgboost`) is built, tested, and **entirely unexercised by the real backtest**.
Only `MinutesModel` and `BonusModel` were fit. Every other component ran on hand-coded literals:

| Constant | Location | Current value | Ever fitted? |
|---|---|---|---|
| Dixon-Coles `rho` | `clean_sheets.py:42` | −0.1 | No |
| Save conversion rate | `saves.py:21` | 0.67 | No |
| Away shot multiplier | `saves.py:17` | 1.1 | No |
| DC overdispersion `alpha` | `defensive_contribution.py` | default | No |
| Penalty conversion rate | `goals.py` | default | No |
| EWMA halflife | `rates.py:41` | 10.0 | No (see Tier 4) |

This is the single largest **designed-but-unexploited** capability in the codebase. `BUILD_PLAN.md`
2.8 is explicit that these underlying rates should be *fitted per position per component* against
real outcomes, refit every gameweek by the walk-forward harness — and that the resulting
coefficients *are* the measured answer to "which stats matter", which was one of the original goals
of the project and currently remains unanswered.

Note the plan's own guardrail applies: 2.8 locks this to **per-component** regressions, never a
unified points regression, because 3.2's calibration check only makes sense if each component owns
its own fitted probability.

### 1.3 Use true non-penalty xG, and activate the penalty sub-model

Two coupled defects:

- The `npxg_per_90` input was FPL's `expected_goals`, which **includes penalty xG** (~0.79 per
  penalty awarded). So penalty takers' "non-penalty" open-play rates are systematically inflated.
- The explicit penalty sub-model received `team_expected_penalties=0`, so it contributed nothing —
  the `penalty_misses` line summed to **exactly 0.0 across all 15,097 rows**, against 15 real
  penalty misses in the season.

Fix: source real npxG from Understat via the already-built `engine/data/understat_client.py`, and
supply the penalty sub-model its three real inputs (primary taker identity, team expected penalties
per game, taker conversion rate). `BUILD_PLAN.md` 2.2 notes this "matters disproportionately for
premium captaincy picks, who are often exactly the players on penalties" — i.e. it lands on the
decisions that matter most.

This also means **the engine has never actually run on Understat data.** The whole backtest ran on
FPL's own xG columns. Given the plan's guiding principle #6 builds the engine's entire history story
on "FPL API + Understat", validating that the Understat path works end-to-end is load-bearing.

---

## Tier 2 — fix what the gate measures

These change *what you optimise*, so they arguably belong before Tier 1 in execution order: right
now the gate is capable of rewarding the wrong thing.

### 2.1 The engine is worse at global ranking but much better at the top — and the gate cannot see it

| Metric (mean across 34 gameweeks) | Engine | Naive form | Winner |
|---|---|---|---|
| Spearman rank correlation with actual points | 0.5219 | **0.5458** | Baseline |
| Gameweeks with better rank correlation | 9 / 34 | 25 / 34 | Baseline |
| **Mean actual points of its own top-10 picks** | **4.947** | 3.803 | **Engine (+30.1%)** |

This is coherent, not contradictory. The naive-form baseline's entire signal is "did you score
points recently", which trivially separates starters from non-starters and therefore scores well on
whole-pool rank correlation. The engine adds genuine skill at the *top* of the distribution, where
xG, fixture difficulty and minutes interact — and loses on the low-minutes mass where its weak
minutes discrimination (Tier 1.1) hurts it.

**Every FPL decision — captaincy, transfers, chips — only ever reads the top of the ranking.** So
the engine is decision-better while being rank-worse, and neither MAE nor global Spearman can
express that. Add to `backtest/metrics.py`:

- **Top-N precision / mean actual points of the top-N** at N = 1, 5, 10, 20 (the metric that
  actually revealed the engine's skill).
- **Per-position and per-price-tier rank correlation**, since a single pooled Spearman is dominated
  by the will-they-play axis.
- **Spearman restricted to players who started**, to separate "ranks footballers well" from "ranks
  availability well".
- **Distributional metrics** the simulation layer (2.9) already supports but nothing scores: is
  `prob_big_haul` calibrated? Is the `floor`/`ceiling` spread honest? These gate the captaincy
  "safe vs punt" distinction that 4's captaincy feature depends on.

### 2.2 Add a constant-predictor baseline — the current bar is too low

| Predictor | MAE |
|---|---|
| **Engine** | **1.6683** |
| Predict the median (1.0) every time | 1.7105 |
| Naive form (last-4 mean) | 1.7390 |
| Predict per-position mean | 1.9543 |
| Predict the sample mean (1.871) | 1.9546 |
| Predict a flat 2.0 | 1.9909 |

The engine beats a **constant median predictor by 2.5%**, and the naive-form baseline is *worse
than a constant*. So "beats all three baselines statistically" — the gate criterion that passed — is
a much lower bar than it sounds. `BUILD_PLAN.md` 3.3 says "an accuracy number is meaningless without
something to beat"; the corollary is that it is also meaningless if the thing you beat is trivial.
Add a constant/median baseline as a hard floor in `backtest/baselines.py`.

Related: **MAE is a poor headline metric for this target.** On a zero-inflated, right-skewed
distribution (median 1.0, mean 1.871, std 2.813), MAE rewards hugging the median. RMSE already shows
a larger relative edge (2.562 vs 2.796, **8.4%** versus MAE's 4.1%) because it takes the tail
seriously. Report both, and treat the decision metrics in 2.1 as primary.

### 2.3 Fix the bias criterion

Per Correction 2: require `|mean residual| > 0.25 pts` (or > 10% of the group's mean actual) **and**
p < 0.01. Without an effect-size floor the gate is unpassable at scale and currently mis-ranks which
position is actually worst.

Also worth doing: cluster the residual standard errors by fixture or gameweek, for the same reason
the block bootstrap was checked — the t-test currently treats ~15k correlated rows as independent,
which overstates its own confidence.

### 2.4 Make the calibration comparison hard to get wrong

`engine/pipeline.py` emits `clean_sheet_probability` (team-level) but **not** `p_60_plus`, which is
what invited Correction 1's error. Change `project_gameweek_pool` to emit the full minutes
distribution (`p_zero`, `p_1_to_59`, `p_60_plus`) plus an explicit
`player_clean_sheet_probability = clean_sheet_probability × p_60_plus`, so the gated and ungated
quantities are separately named and the wrong one cannot be silently picked up.

Then **run 3.2's calibration check on every component, not just clean sheets.** The plan asks for
per-component calibration; only one component was ever checked, and that check was mis-specified.
Goals, assists, DC-threshold and bonus are all unvalidated.

---

## Tier 3 — coverage and statistical power

### 3.1 Multi-season backtest

Captaincy hit-rate is where the engine most plausibly earns its keep, and one season cannot prove
it:

| Metric | Engine | Template captain |
|---|---|---|
| Raw hit-rate | **0.265** (n=34) | 0.189 (n=37) |
| "Played as expected" hit-rate | 0.273 | — |
| Permutation test on the difference | observed +0.088, **p = 0.374** | not significant |

An 8.8pp edge on 34 observations cannot separate skill from luck — precisely the "variance
masquerading as skill" risk in the plan's own risk register. The archive runs back to 2016/17 and
Understat to 2014/15; ~10 seasons is ~380 gameweeks, which brings real power.

Per the plan's note in 1.1, split the depth by component: goals, assists, clean sheets and minutes
backtest on the full decade; **defensive contribution stays on 2025/26 only** (the rule's first
year), and 3.6 already handles that correctly via wider confidence intervals rather than a
carve-out.

**Caveat to handle:** the captaincy figures above used a *stand-in* squad (the 13 highest-minutes
outfield players, chosen independently of any model output to avoid circularity), because no real
historical "my team" exists. Real captaincy evaluation needs either a real team's history via the
FPL API or an explicitly documented squad-selection rule.

### 3.2 Include goalkeepers and validate the saves component

3,427 GK rows (~13% of the pool) were excluded and `saves` contributed exactly `0.0000`. The
component is entirely unvalidated. Needs opponent shots-on-target — Understat carries team-level
shot data. Note GKs are also where the goals-conceded penalty and clean-sheet points concentrate
most heavily, so this is not a peripheral 13%.

### 3.3 Recompute the bonus target from raw BPS events

Two defects, both already anticipated by `BUILD_PLAN.md` 2.6:

- The regression was fit against the raw `bonus` column, which reflects the **superseded
  pre-2026/27 BPS formula**. The plan says in terms: "Don't regress directly against that stale
  column."
- Training features used realised minutes; prediction uses modelled expected minutes — a
  **train/serve skew** that biases the fitted coefficients.

The source data carries `tackles`, `clearances_blocks_interceptions`, `recoveries` and `bps`, which
is most of what is needed to recompute bonus under the current formula. Bonus is +0.1427 mean
predicted points (7.8% of the projection), so this is material rather than cosmetic.

Blocker to note: `engine/scoring.py:BPS_REWORK_NOTES_2026_27` records that the **exact 2026/27 BPS
numeric table has not yet been sourced** — only three qualitative changes are documented. That table
is a prerequisite for this work.

### 3.4 Supply the DC opponent-possession adjustment

`opponent_possession_share` was constant at 0.5, making 2.5's opponent adjustment inert. This
adjustment is interesting because it points the *opposite* way to every other fixture adjustment
(facing a possession-dominant opponent means *more* defensive actions), so it is a genuinely
uncorrelated signal — and the plan calls DC "your edge" precisely because no market prices it.
Needs real possession or pass-volume data per fixture.

Encouraging sign that the underlying rate is strong: DC per-90 rates predict actual DC action counts
at **Spearman 0.757**, far higher than npxG→goals (0.230) or xA→assists (0.155). The component has
real signal; it is just missing its fixture adjustment.

---

## Tier 4 — verified low value, deprioritise

Recorded so the effort isn't spent, and so the conclusion can be revisited if the data changes.

### 4.1 EWMA halflife tuning — measured, marginal

`BUILD_PLAN.md` 1.1 says the decay parameter "is something backtesting can actually tune and
justify". It was tested. It barely matters:

| Halflife (matches) | npxG→goals MAE | xA→assists MAE | DC→actions MAE |
|---|---|---|---|
| 2 | 0.1630 | 0.1380 | 2.3950 |
| 4 | 0.1624 | 0.1372 | 2.3001 |
| 6 | 0.1620 | 0.1367 | 2.2760 |
| **10 (current default)** | **0.1616** | **0.1363** | **2.2635** |
| 15 | 0.1613 | 0.1360 | 2.2599 |
| 25 | **0.1611** | **0.1358** | **2.2585** |

(n = 10,107 player-gameweeks with minutes > 0; Spearman moves in the same direction and is equally
flat.)

Three observations:

1. **Gains are negligible** — moving 10 → 25 improves npxG MAE by 0.31%. Against a 26% opportunity
   in the minutes model, this is noise.
2. **The direction is "longer is better", monotonically** — within-season recency weighting is
   barely earning its keep, and a near-season-long average performs about as well. Mildly
   surprising, and worth remembering when reasoning about form.
3. **The optimum is outside the tested grid.** Every stat prefers the boundary value (25), so the
   true optimum is unidentified. If revisited, extend to 40/60/∞ and test across multiple seasons —
   the "longer is better" signal may simply reflect that one season is too short for decay to help,
   in which case multi-season data could reverse it.

### 4.2 Dixon-Coles `rho` and other clean-sheet constants

Untuned, but the component already calibrates at MACE 0.0157 with only ~7% team-level optimism, so
headroom is small. Revisit after Tier 1, and prefer fixing the **missing home/away rate split**
(specified in 2.4, not supplied by the backtest) before tuning `rho`, since the former is a
structural omission and the latter a second-order parameter.

---

## Suggested execution order

Sequenced so that measurement is trustworthy *before* modelling effort is spent against it —
otherwise Tier 1 work gets judged by metrics that demonstrably mis-rank quality.

1. **Tier 2 first (2.1–2.4).** Cheap, and it fixes what "better" means. Without 2.1's top-N metrics,
   a real Tier 1 improvement could look like a regression on global Spearman.
2. **Promote the backtest driver to a versioned module.** Everything downstream depends on being
   able to re-run this reliably.
3. **Tier 1.1 (minutes features).** Largest quantified gain (26% ceiling), no new data dependency.
4. **Tier 1.3 (Understat npxG + penalties).** Validates the Understat path the plan's whole history
   story rests on.
5. **Tier 1.2 (wire the regression layer).** Highest ceiling of the three but most work; benefits
   from 1.1/1.3 being in place first so it is fitting on clean inputs.
6. **Tier 3.1 (multi-season).** The only route to statistical significance on captaincy.
7. **Tier 3.2–3.4** as data sources allow.

## Re-measurement checklist

After each change, re-run and record against these baselines from this run, so improvements are
attributable (and tagged by model version per 3.4):

| Metric | Current | Target |
|---|---|---|
| Overall MAE | 1.6683 | < 1.45 |
| Overall RMSE | 2.5620 | < 2.45 |
| AUC "played at all" | 0.8107 | > 0.88 |
| Predicted points mass on zero-minute players | 6,618 | < 3,500 |
| Mean actual points of top-10 picks | 4.947 | > 5.5 |
| Captaincy raw hit-rate | 0.265 (n=34) | > 0.30, significant on multi-season |
| Clean-sheet MACE (gated, correct comparison) | 0.0157 | hold < 0.03 |
| Beats constant-median baseline (MAE 1.7105) | +2.5% | > +10% |

---

## Current Definition-of-Done status

For reference, the gate verdict from this run (`backtest/gate.py`):

```
[ ] Beats all baselines, statistically (3.3)
[ ] No severe systematic bias in any position/price tier (3.2)
[ ] Each component reasonably calibrated (3.2)
[x] Predictions logged immutably, tagged by model version (3.4)
[ ] Personally trusted enough to act on (3.6)

NOT YET — gate to Phase 5
```

Two of those four failures are now known to be **measurement artefacts rather than engine
defects**: the calibration failure was Correction 1, and the bias failure was Correction 2. The
baseline criterion failed only on captaincy significance, which is a sample-size problem (Tier 3.1)
rather than a modelling one. That means the engine is in meaningfully better shape than the gate
currently reports — but the honest reading is that **the gate itself needs fixing before its verdict
means anything**, which is why Tier 2 leads the execution order.

---

## Appendix — raw diagnostic output

### Player-level accuracy by position

```
position      mae     rmse    n
     DEF 1.827230 2.672510 5662
     FWD 1.688522 2.692672 1912
     MID 1.543524 2.439626 7523
```

Engine overall MAE 1.668, RMSE 2.562. Naive-form baseline overall MAE 1.739, RMSE 2.796.

### Clean-sheet calibration, corrected (gated, apples-to-apples)

```
          bin  predicted_mean  actual_rate    n
(-0.001, 0.1]        0.042655     0.048598 7202
   (0.1, 0.2]        0.145147     0.161682 3544
   (0.2, 0.3]        0.243867     0.207174 1617
   (0.3, 0.4]        0.340306     0.297386  612
   (0.4, 0.5]        0.440628     0.373563  174
   (0.5, 0.6]        0.546912     0.281250   32
   (0.6, 0.7]        0.617469     0.500000    4
MACE = 0.0157
```

Well calibrated where the mass is (bins up to 0.4 hold 12,975 of 13,185 rows); the mild optimism in
the sparse high-probability bins is where the ~7% team-level over-prediction shows up.

### Clean-sheet calibration, as originally (mis-)measured

```
          bin  predicted_mean  actual_rate    n
(-0.001, 0.1]        0.072392     0.029740  269
   (0.1, 0.2]        0.163413     0.087153 4429
   (0.2, 0.3]        0.244487     0.108207 4935
   (0.3, 0.4]        0.346859     0.122592 3271
   (0.4, 0.5]        0.437206     0.177632 1520
   (0.5, 0.6]        0.540715     0.199170  482
   (0.6, 0.7]        0.645523     0.124260  169
   (0.7, 0.8]        0.726794     0.409091   22
MACE = 0.1602
```

The near-constant ~2.4× ratio between predicted and actual across every bin is the signature of a
missing multiplicative gate (mean P(60+) = 0.416), not of a model that misunderstands football.

### Dropped-row composition

```
Total outfield (player,GW) rows = 25,955   kept = 16,053   dropped = 9,902
Of dropped rows: 95.0% played ZERO minutes; mean minutes 2.4
Of KEPT rows:    37.0% played zero minutes
```

The drop is mostly deep-squad players with no prior minutes history at all, so it is less biasing
than the 38% headline suggests — but note the retained sample is *still* 37% zero-minute rows, which
is why the minutes model dominates the error.

### Rate-stat predictive strength (Spearman, halflife = 10)

```
npxG/90 -> goals            0.2290
xA/90   -> assists          0.1543
DC/90   -> DC actions       0.7560
```

Defensive contribution is by far the most predictable component from its own rate — consistent with
2.5's claim that it offers a reliable points floor, and an argument for prioritising its missing
fixture adjustment (Tier 3.4).
