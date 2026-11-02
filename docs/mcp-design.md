# MCP server design

The PM spike showed muse-spark can plan using in-process tools (`runtime/pm_tools.py`).
This document covers the real version: one MCP server that is the only way any agent
reaches the orchestrator, and the runner that drives agents through an accepted plan.

## What already exists (no changes needed)

- `TaskOrchestrator`: every agent action is already a method (`claim`, `complete`,
  `record_review`, `request_human`, `record_usage`, `submit_plan`, ...), one transaction each.
- Identity: `claim()` returns a one-time **run token**; `resolve_run_token()` turns it into
  `{run_id, task_id, project_id, agent, kind}`. The token dies when the task leaves RUNNING.
- `ToolCall` table with an idempotency key (`task_id:model_call_id`), for replay-safe tools.
- `Artifact` table; `scope.task_can_write()` for write checks.
- Gate and approval tasks are never run by agents (`UNASSIGNED_KINDS`).

## Shape

```
orch run "<objective>"
 └─ Runner (one process)
     ├─ TaskOrchestrator + SQLite (.orch/)
     ├─ MCP server  — Streamable HTTP on 127.0.0.1:<random port>, same process
     └─ loop: get_runnable → claim → start agent as an asyncio task (pydantic-ai Agent +
              MCP client, header  Authorization: Bearer <run_token>) → record_usage
              up to N agents at once (default 3); a finished agent frees a slot
              gate tasks: the runner runs pytest/ruff/policy itself → record_gate_results
              approval / questions: wait for `orch approve` / `orch answer`
```

**Why one HTTP server in the runner process** (not stdio per agent): one shared
orchestrator and event log, no per-agent subprocesses, and the token header gives identity
for free. Bound to localhost only.

**Library:** the official `mcp` Python SDK (`FastMCP`) for the server; pydantic-ai's
`MCPServerStreamableHTTP` as the client (already a dependency family).

## Identity and permissions

Every call goes through one wrapper:

1. Read the bearer token → `resolve_run_token`. Unknown or ended → tool error
   `"run is not active"`. The agent never passes its own name or task id.
2. **Tool allowlist:** the agent's `tools:` list in the team YAML. `list_tools` returns only
   those; calling anything else is refused (defense in depth, since a model can guess names).
3. **Kind check:** e.g. `submit_review` only from a review/security_review task,
   `submit_plan` only from a plan task (the orchestrator already raises `InvalidOperation`).
4. Write `ToolCall` (status=started) → act → update with result. A repeated
   idempotency key returns the stored result instead of acting twice.
5. `OrchestratorError` → MCP tool error with the message (the model sees it and retries),
   same as `ModelRetry` in the spike.

## Tools

| Tool | Who | Calls |
|---|---|---|
| `list_agents` | PM | `orch.list_agents` |
| `validate_plan`, `submit_plan` | PM | same as spike, same return shapes |
| `get_task_status`, `get_task_tree` | PM | `get_task`, `task_tree` |
| `get_remaining_budget` | PM | project limits − spent |
| `request_human_input` | PM (any later) | `request_human`; task pauses |
| `get_task` *(new, all)* | all | my task: kind, criteria, `paths`, prior review findings |
| `read_file`, `list_files`, `search_code` | workers, reviewers | read-only in the worktree |
| `write_file` | test_engineer, developers | `task_can_write(agent paths, task paths)` |
| `run_tests` | test_engineer, developers | pytest in worktree, timeout, output tail |
| `submit_artifact` | workers | file bytes stored by sha256 → `Artifact` row |
| `complete_task` | workers | `orch.complete` (→ VERIFYING or DONE) |
| `submit_review` | reviewer, security_reviewer | `orch.record_review` with the diff of the target |

Note: the PM's tools stay plain functions in `pm_tools.py` reused by the MCP layer, so the
spike and the server can't drift. The spike gains `--mcp` to run the same trials over the
real server and compare.

## Workspace

- The target app's source lives in this repo at `examples/minibank/` (a small
  standard-library Python service with a `payments/` package). `orch run` copies it into
  its own git repo, `.orch/repos/minibank`, so agent branches never touch orch's history
  and the team's paths (`minibank/**`, `tests/**`) are relative to that repo's root.
- Each objective gets a branch, `orch/<root task id>`, made from `main`.
- **Parallel from day one:** each claimed test/implement task gets its **own git worktree**
  under `.orch/worktrees/<task id>`, on a branch made from the objective branch's current
  head. The runner records that base commit.
- Reviewers and gates run in the work task's worktree and see `git diff <base>`.
- When a work task becomes DONE, the runner merges its branch into the objective branch.
  A merge conflict stops the run with the conflict and the folder to resolve it in
  (a DONE work task can't be reopened by the state machine; PAR-001 makes this rare).
- Conflicts are mostly prevented up front: every node already declares `paths`. New
  validator rule **PAR-001**: two work nodes that can run at the same time (no `blocks`
  path between them) must not have overlapping `paths`. The PM gets the error and adds an
  edge or splits the paths.
- Concurrency limit: 3 agents (contributor model allows 150 requests/min per team;
  a PM run is ~6 requests over ~50 s, so 3 is far below it). `--parallel N` to change.
- SQLite with several writers: WAL mode + short transactions (already one per method);
  the claim is an atomic `UPDATE ... WHERE status = READY`, so two runners can't double-claim.

## CLI added

- `orch run "<objective>" [--team ...] [--max-usd 1.00]` — create project + root task, run
  until done, failed, or waiting on a human; prints the live event log.
- `orch inbox`, `orch answer <request> <response>`, `orch approve <request>` — human side.
- `orch evidence <root task>` — the audit pack: plan, reviews, gate results, approvals,
  hash-chain check (`verify_events`).

## Build order

1. `orch/mcp/server.py`: MCP SDK low-level server, auth wrapper, PM tools only. Test: spike `--mcp
   --offline` passes the same as in-process.
2. Runner loop + `orch run` with the PM only (stops at WAITING).
3. Worktree per task + merge back + PAR-001; worker tools (`read/list/search/write_file`,
   `run_tests`, `complete_task`); N agents at once.
4. Gates run by the runner; `submit_review`; reopen-on-changes path.
5. Human CLI (`inbox/answer/approve`) and `evidence`.
6. `minibank` target repo + the demo objective end to end, live on muse-spark.

Status (Nov 2026): all six steps are built and tested offline with a scripted team
(`orch run --offline`, `tests/test_runtime.py`).

## Decisions

1. Target app source lives in this repo at `examples/minibank/`; name kept.
2. Parallel work from the start (worktree per task, PAR-001).
3. Every agent uses the cheapest model, `muse-spark-1.3-contributor` (already the default).
4. The policy gate checks that every changed file is inside the task's planned paths and
   the agent's write_paths, and that no secrets (keys, passwords) were added.

## Not built yet

- `submit_artifact` is on the team's allow-lists but not served (agents never see it).
- Two efficiency cleanups from the step 1 review: `get_task_status` loads the whole tree,
  and each tool call makes four database round trips.
