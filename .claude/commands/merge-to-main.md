---
description: Commit all current changes in sensible logical groups, push them, then merge this branch into main and push main too
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(git rev-parse:*), Bash(git branch:*), Bash(git fetch:*), Bash(git checkout:*), Bash(git merge:*)
---

Commit all current changes (staged, unstaged, and untracked) and push them, then bring this
branch's work into `main` and push that too.

## Part 1: commit and push the current branch

1. Run `git status` and `git diff` (and `git diff --staged` if anything is already staged) to see
   everything that's changed.
2. Group the changes into sensible, logically related commits, e.g. by feature/component, not one
   giant commit and not one commit per file. Use judgment: a component and the screen that uses it
   can be one commit; unrelated areas of the app should be separate commits.
3. Check `git log -5 --oneline` to match this repo's commit message style (short, plain, no
   prefixes/scopes, imperative or brief noun phrase).
4. For each group: `git add` the specific files (never `git add -A`/`.`), then commit with a
   **brief** message (one short line, no body, no "Co-Authored-By" footer unless that's already
   this repo's convention for local commits) that ending with:
   Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
5. Determine the current branch (`git rev-parse --abbrev-ref HEAD`) and check whether it has an
   upstream (`git rev-parse --abbrev-ref --symbolic-full-name @{u}`, ignoring failure). Push with
   `git push` if an upstream exists, or `git push -u origin <branch>` if not.
6. Run `git status` to confirm a clean working tree and that the push succeeded.

## Part 2: merge into main

7. If the branch from step 5 is already `main`, stop here, there is nothing to merge. Report the
   commits from Part 1 and finish.
8. Otherwise, exit the current worktree with ExitWorktree (`action: "keep"`, so this worktree and
   branch stay on disk exactly as they are) so the session moves back to the main worktree.
9. Run `git fetch origin main` so local `main` reflects the true latest remote commit, then
   `git checkout main` and `git merge origin/main` (fast-forward, since nothing local should be
   ahead of origin on `main`) so `main` is current before merging the feature branch in.
10. Run `git merge <branch>` (the branch from step 5) into `main`. If it merges cleanly (including
    "Already up to date"), continue to step 11. If it conflicts, stop, leave the merge for the user
    to resolve, and report which files conflict. Don't try to resolve it yourself.
11. Push the merge: `git push origin main`.
12. Run `git status` and `git log -5 --oneline` on `main` to confirm the merge landed and the push
    succeeded.

Rules:
- Never use `git add -A` or `git add .`, always add specific files/paths so nothing accidental
  (e.g. `.env`, credentials) gets swept in.
- Never use `git commit --amend`, `git push --force`, or skip hooks (`--no-verify`).
- Never force-push `main`, and never resolve a merge conflict unilaterally. Stop and hand it back
  to the user.
- If a file looks like it might contain secrets, flag it and skip it rather than committing it.
- If pre-commit hooks fail, fix the issue and create a new commit, don't bypass the hook.
- Don't remove or otherwise touch the feature branch/worktree after merging. Leave that decision
  to the user.
- Report back: the list of commits made on the feature branch (hash and message) with confirmation
  that push succeeded, then whether the merge into `main` was a no-op, fast-forward, or a real
  merge, plus confirmation that `main` was pushed.
