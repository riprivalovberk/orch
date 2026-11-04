# orch (working name)

**Your agents write their own workflow. We check it like a compiler before anything runs.**

A team of AI agents, defined in one YAML file, ships code changes under a bank's change-control rules.
A PM agent plans each objective as a task graph. A validator rejects plans that break the rules
(segregation of duties, required test gates, criteria coverage, security review for payments code)
before any work starts. Agents talk to the orchestrator only through MCP.

Built for the Meta Global AI Developer Hackathon (Indie track) on `muse-spark-1.3`.

## Quickstart

```bash
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run pytest
uv run orch agents
uv run orch plan-check examples/plan_transfers.yaml   # clears
uv run orch plan-check examples/plan_bad.yaml         # rejected, with node-level errors
uv run orch run --offline --repo <minibank copy>      # whole flow, scripted team, no network
```

## Running an objective

```bash
uv run orch run --repo ~/code/minibank \
  "Add an idempotent POST /transfers endpoint in minibank/api/transfers.py" \
  --ac "A repeated Idempotency-Key never double-posts" --ac "Amounts are Decimal, never float"
```

`--ac` is optional. Without it the PM reads the repo (code and any sample data), writes
numbered criteria itself and asks you to confirm them before it plans: reply to approve them
or say what to change.

The PM plans, the validator checks the plan, then the team works in parallel (`--parallel 3`):
each test/implement task in its own git worktree, gates (pytest, ruff, policy) run by the
runner, reviews by an independent agent, and a human approval before the change merges into
the target repo's `main`. Questions and approvals are asked in the terminal; with
`--no-interactive` the run stops and waits:

```bash
uv run orch inbox                      # what is waiting on a person
uv run orch approve <request id>       # or: orch answer <request id> <response>
uv run orch run --resume <objective>   # carry on
uv run orch evidence <objective>       # audit pack: plans, gates, reviews, approvals, chain check
```

`--repo` is the target git repo (it needs a `main` branch); approved changes merge into its
`main`, and `--resume` always continues in the objective's own repo. The sample app the demos
use, minibank (a tiny bank service with a protected `payments/` package), is its own repo
next to orch, not part of it; `--offline` only works on a copy of it.
Each repo gets one project whose budget covers all its objectives: $2.00 to start,
changed with `--budget`. Every agent uses `muse-spark-1.3-contributor` unless `--model`
says otherwise. Workers can hand in non-code files (test reports, notes) with
`submit_artifact`; they are stored by hash under `.orch/artifacts/` and listed in the evidence.

Everything orch creates (databases, spike results, agent transcripts) goes in one folder,
`.orch/` (gitignored; override with `ORCH_DATA_DIR`). Delete it any time with `uv run orch nuke` (or one part: `orch nuke db`, `orch nuke spike`,
`orch nuke repos`).

## Web UI

```bash
uv run orch ui                         # http://127.0.0.1:8765
```

Each project opens as a chat with the PM. Ask it about the project (where code is, what an
objective changed, why a check failed: it reads main and every objective's branch), or say
what to build: it starts an objective, drafts acceptance criteria with you, then posts its
plan and the team's progress in the thread. Its questions, the final approval and any escalation arrive as messages you
answer in place (answering resumes the run). A collapsible Files panel on the right lists the
repo's files, previews them and uploads new ones (committed to `main`, so every new
objective's agents can read them).

A project points at a git repo you already have (type the path or Browse…), or at a new or
empty folder, which starts as an empty Python project. The offline demo needs a project whose
repo is a copy of minibank. Everything else is on the project's details page and the top bar:
budget, objectives with their tasks, gates, reviews, diffs and evidence pack, the filtered
event log, the event-chain check, the team, plan check, clean-up and the model API key
(Admin; saved to `.env`). Runs started in the UI
go in background threads of the UI process; events are read from the database, so
`orch run` runs show too.

## Layout

```
src/orch/
  domain/        enums + Pydantic schemas (team, plan, verdicts)
  core/          state machine, team registry, plan validator, event log, orchestrator
  policies/      policy packs (fintech_sdlc: SOD, GATE, COV, APR, SCOPE, PAY rules)
  db/            SQLAlchemy models (12 tables) + SQLite session
  mcp/           the agent MCP server (run-token identity, tool allow-lists) + worker tools
  runtime/       runner, git workspace, gates, Meta Model API wiring, PM tools, spike,
                 scripted team (--offline)
  web/           the web UI (orch ui): Starlette JSON API + one HTML page
  paths.py       the one data folder (.orch/)
teams/           the agent team (YAML)
examples/        sample plans (Plan check) and sample data
docs/            design notes (mcp-design.md)
```
