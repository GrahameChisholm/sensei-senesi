---
description: Start a locally hosted copy of the app (API + web) on the current branch
allowed-tools: Bash(git status:*), Bash(git rev-parse:*), Bash(ls:*), Bash(curl:*), Bash(python3:*), Bash(uv run uvicorn:*), Bash(uv run python scripts/build_projections.py:*), Bash(cd:*), Bash(npm install:*), Bash(npm run dev:*), Bash(lsof:*), Bash(kill:*)
---

Start a local instance of the app as it exists on whatever branch is currently checked out. Do not
switch branches or stash/commit anything, just run what's here.

Steps:
1. Run `git status` and `git rev-parse --abbrev-ref HEAD` so you know the branch and whether there
   are uncommitted changes (report this, don't block on it).
2. Work out which gameweek the app should actually be showing, then make sure a projection cache
   exists for exactly that gameweek, since a stale or wrong-gameweek cache is worse than an empty
   one, it looks correct while quietly showing last gameweek's picks.
   - Skip this whole step if the user is asking for a historical replay season (see below) — the
     replay path has its own fixed, already-built cache and isn't derived from live data.
   - Otherwise, find the live target gameweek: `curl -s
     https://fantasy.premierleague.com/api/bootstrap-static/` and take the first event whose
     `deadline_time` is still in the future. That is the gameweek a manager is actually setting a
     squad for right now, not necessarily the one FPL flags `is_current` (that stays true while
     last gameweek's matches are still being played, even after its deadline has passed). A one-off
     `python3` snippet piping the curl output through `json.load` is the simplest way to pick this
     out and print `gameweek=<n> season=<yyyy-yy> understat_start=<yyyy> prior_start=<yyyy>` in one
     go: derive the season from that event's `deadline_time` year, using the year itself if the
     month is July or later, otherwise the year before (e.g. a March 2027 deadline is still the
     2026-27 season), and format as `"<start>-<end % 100>"`.
   - Check whether `data_store/projections/<season>/gw<NN>.json` (two-digit gameweek, e.g. `gw02`)
     already exists for that exact gameweek number. Don't stop at "the season directory has some
     files in it" the way an emptiness check would, an existing file for gw01 does not mean gw02 is
     covered.
   - If that exact file is missing, run `uv run python scripts/build_projections.py --season
     <season> --gameweek <n> --understat-season-start-year <year> --prior-season-start-year
     <year - 1>` before doing anything else, since the API only serves precomputed data and never
     computes projections on request. All four flags are required, there is no bare no-argument
     form.
3. Check ports 8000 and 5173 aren't already in use (`lsof -i :8000` / `lsof -i :5173`).
   - If nothing is listening, proceed straight to starting both processes (steps 4-5).
   - If something is already listening on 8000: the API loads its projection cache into memory
     once, on first request, and keeps serving that same snapshot for the rest of the process's
     life, so an already-running instance started before step 2 built or rebuilt the cache is very
     likely still serving the old one. Tell the user this plainly and ask whether to restart it
     (`kill` the existing uvicorn PIDs, then start fresh) rather than either leaving stale
     projections live or silently restarting something they didn't ask you to touch. Apply the same
     logic to 5173, though the web dev server itself has no projection data to go stale.
4. Start the API in the background: `uv run uvicorn api.main:app --reload` from the repo root.
5. Start the web dev server in the background: `npm run dev` from `web/` (run `npm install` first
   only if `web/node_modules` is missing).
6. Confirm both processes came up cleanly (no immediate crash in their output), then tell the user:
   - the API is at http://localhost:8000
   - the web app is at http://localhost:5173
   - which branch this is running from, and which gameweek/season the API is now serving
   - that both processes are running in the background and how to stop them (or offer to stop them)

If the user wants a historical replay season instead of live data, they'll say so explicitly, in
which case set `FPL_REPLAY_SEASON=<season>` (e.g. `2025-26`) in the environment before starting
uvicorn in step 4, rather than the live default, and skip the live-gameweek lookup in step 2
entirely (check the existing cache for that season is non-empty instead, the same shallow check
this command used before).
