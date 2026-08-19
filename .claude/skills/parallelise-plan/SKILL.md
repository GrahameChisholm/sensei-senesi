---
name: parallelize-plan
description: Converts an already-decided plan (e.g. a finished plan-mode plan) into an implementation plan organized into dependency-ordered waves of parallelizable tasks, so independent work can be handed to concurrent subagents. Use when the user asks to "turn this plan into an implementation plan", "break this down for parallel work", or wants a task breakdown grouped by dependency.
---

# Parallelize Plan

Takes a plan whose product/architecture decisions are already settled (this skill does not re-decide scope) and restructures it into an execution-ready implementation plan: atomic tasks grouped into waves, where every task in a wave can run concurrently because none of them depend on each other, and each wave depends only on waves before it.

## Steps

1. **Find the source plan.** Use the path given by the user, or the most recently modified file in `~/.claude/plans/`. Ask if ambiguous.
2. **Decompose into atomic tasks.** Each task must be self-contained enough that a subagent with zero conversation context could execute it from the task text alone: what to change, in which files, and why (one sentence). Split by file/module boundary, not by feature area — e.g. "add the migration" and "add the endpoint that uses it" are separate tasks even if the source plan describes them under one heading.
3. **Find real dependencies only.** Task A depends on Task B only if A needs something B produces to actually exist (a column, a function, an exported type) — not because they're part of the same feature. A contract already fixed in the source plan (an API request/response shape, a function signature) is not a dependency: both sides can be built against it in parallel and only need to agree at integration/test time.
4. **Assign waves.** Wave 1 = tasks with no dependencies. Wave N = tasks whose dependencies are all satisfied by waves < N. Within a wave, no two tasks may touch the same file — if two otherwise-parallel tasks can't avoid a shared file, keep them in the same wave but flag them as needing serialization or worktree isolation. Tasks that verify or test the whole feature end-to-end depend on everything else and form the final wave. If the plan has no real parallelism, say so plainly rather than forcing artificial groups.
5. **Write the implementation plan** to `<source-plan-name>-implementation.md`, next to the source plan. Use the output shape below. End with a one-line summary: wave count and the largest wave's size.
6. **Stop there.** Don't execute the plan or spawn agents — this skill only produces the document.

## Output shape

```markdown
# Implementation plan: <feature>

## Wave 1 (parallel, no dependencies)
### T1 — <title>
Files: ...
Depends on: none
Must not touch: ...
<self-contained description>

### T2 — <title>
...

## Wave 2 (depends on Wave 1)
### T3 — <title>
Depends on: T1
...
```
