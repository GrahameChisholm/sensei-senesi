---
description: Start a locally hosted copy of the app (API + web) on the current branch
allowed-tools: Bash(git status:*), Bash(git rev-parse:*), Bash(ls:*), Bash(uv run uvicorn:*), Bash(cd:*), Bash(npm install:*), Bash(npm run dev:*), Bash(lsof:*)
---

Start a local instance of the app as it exists on whatever branch is currently checked out. Do not
switch branches or stash/commit anything, just run what's here.

Steps:
1. Run `git status` and `git rev-parse --abbrev-ref HEAD` so you know the branch and whether there
   are uncommitted changes (report this, don't block on it).
2. Check `data_store/projections/` has content for the relevant season. If it's empty, run
   `uv run python scripts/build_projections.py` first, since the API only serves precomputed data
   and never computes projections on request.
3. Check ports 8000 and 5173 aren't already in use (`lsof -i :8000` / `lsof -i :5173`). If something
   is already listening there, tell the user instead of starting a second instance on top of it.
4. Start the API in the background: `uv run uvicorn api.main:app --reload` from the repo root.
5. Start the web dev server in the background: `npm run dev` from `web/` (run `npm install` first
   only if `web/node_modules` is missing).
6. Confirm both processes came up cleanly (no immediate crash in their output), then tell the user:
   - the API is at http://localhost:8000
   - the web app is at http://localhost:5173
   - which branch this is running from
   - that both processes are running in the background and how to stop them (or offer to stop them)

If the user wants a historical replay season instead of live data, they'll say so explicitly, in
which case set `FPL_REPLAY_SEASON=<season>` (e.g. `2025-26`) in the environment before starting
uvicorn in step 4, rather than the live default.
