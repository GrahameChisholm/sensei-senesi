---
description: Rebuild all predictions/data from a fresh live snapshot and launch a local instance of the app on it
allowed-tools: Bash(git status:*), Bash(date:*), Bash(uv run python scripts/build_projections.py:*), Bash(ls:*), Bash(mv:*), Bash(curl:*), Bash(python3:*), Bash(uv run uvicorn:*), Bash(cd:*), Bash(npm install:*), Bash(npm run dev:*), Bash(lsof:*), Bash(kill:*)
---

Rebuild the projection cache from a brand new live snapshot for whatever gameweek is genuinely
current right now, retire any leftover manual gameweek-pin artifacts (now obsolete, see below), then
start a local instance of the app on that fresh data.

This differs from `/run-local` in one deliberate way: `/run-local` only builds a cache if one is
missing for the target gameweek, since it exists to just get something running. This command always
rebuilds, even if a cache already exists, since the whole point of running it is to pull fresh live
data (updated prices, injury news, minutes, confirmed bonus points) for whichever gameweek is
current.

Steps:
1. Run `git status` so you know whether there are uncommitted changes (report this, don't block on
   it, and don't touch them).
2. Work out the season string: today's date, month July or later means this calendar year is the
   season's start year, otherwise last calendar year is (a March 2027 run is still the 2026-27
   season). Format as `"<start>-<end % 100>"`, e.g. `"2026-27"`. `--understat-season-start-year` is
   that same start year; `--prior-season-start-year` is one less.
3. Run:
   ```
   uv run python scripts/build_projections.py --season <season> \
     --understat-season-start-year <year> --prior-season-start-year <year - 1>
   ```
   Deliberately no `--gameweek`: it auto-resolves (`scripts/build_projections.py`'s
   `resolve_build_gameweek`) to the earliest gameweek FPL hasn't yet flagged `data_checked`, i.e.
   the one still being decided or still being played out, so this always targets the right
   gameweek without you having to work that out by hand. This is a live network pull plus a full
   refit, several minutes, not instant. Report the gameweek number the build's own summary output
   says it wrote, and the summary's headline diagnostics (player counts, cold-start count, training
   rows).
4. Retire any manual gameweek-pin leftovers: `ls data_store/projections/<season>/` and look for
   any `gwNN.json.<suffix>` file that is not a plain `gwNN.json` (e.g. a
   `gw04.json.pending-until-gw3-complete` left over from a one-off manual pin, see
   `api/state.py`'s `decision_gameweek`, which is deadline-only and does not need or use these
   files). If the gameweek this build just wrote is greater than or equal to the number in such a
   filename, that file is now superseded by this run's own fresh cache for that gameweek: `mv` it
   to end in `.retired` instead (don't delete outright) and tell the user it was cleaned up and
   why. If this build's gameweek is still lower than the number in such a filename, leave it alone
   and don't treat it as an error, that just means FPL hasn't confirmed the pinned gameweek's own
   matches as finished yet, which has no bearing on what `decision_gameweek` itself reports.
5. Now launch the app. Follow `.claude/commands/run-local.md`'s own process from its port check
   onward (read that file for the exact steps if you need them): check ports 8000 and 5173 aren't
   already in use, handle an already-running instance the same cautious way it describes (ask
   before killing anything), then start the API (`uv run uvicorn api.main:app --reload`) and the
   web dev server (`npm run dev` from `web/`, running `npm install` first only if
   `web/node_modules` is missing) in the background.
6. Confirm both processes came up cleanly (no immediate crash in their output), then tell the user:
   - the API is at http://localhost:8000
   - the web app is at http://localhost:5173
   - which gameweek/season the fresh build produced and the API is now serving
   - any pin file retired in step 4
   - that both processes are running in the background and how to stop them (or offer to stop them)
