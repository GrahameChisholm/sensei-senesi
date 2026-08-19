---
description: Commit all current changes in sensible logical groups with brief messages, then push to the current branch's remote
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(git rev-parse:*), Bash(git branch:*)
---

Commit all current changes (staged, unstaged, and untracked) and push them.

Steps:
1. Run `git status` and `git diff` (and `git diff --staged` if anything is already staged) to see everything that's changed.
2. Group the changes into sensible, logically related commits — e.g. by feature/component, not one giant commit and not one commit per file. Use judgment: a component and the screen that uses it can be one commit; unrelated areas of the app should be separate commits.
3. Check `git log -5 --oneline` to match this repo's commit message style (short, plain, no prefixes/scopes, imperative or brief noun phrase).
4. For each group: `git add` the specific files (never `git add -A`/`.`), then commit with a **brief** message (one short line, no body, no "Co-Authored-By" footer unless that's already this repo's convention for local commits) that ending with:
   Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
5. After all groups are committed, determine the current branch (`git rev-parse --abbrev-ref HEAD`) and check whether it has an upstream (`git rev-parse --abbrev-ref --symbolic-full-name @{u}` — ignore failure). Push with `git push` if an upstream exists, or `git push -u origin <branch>` if not.
6. Run `git status` after pushing to confirm a clean working tree and that the push succeeded.

Rules:
- Never use `git add -A` or `git add .` — always add specific files/paths so nothing accidental (e.g. `.env`, credentials) gets swept in.
- Never use `git commit --amend`, `git push --force`, or skip hooks (`--no-verify`).
- If a file looks like it might contain secrets, flag it and skip it rather than committing it.
- If pre-commit hooks fail, fix the issue and create a new commit — don't bypass the hook.
- Report back the list of commits made (hash + message) and confirm the push succeeded, with the remote/branch pushed to.
