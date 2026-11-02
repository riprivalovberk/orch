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
uv run orch run --offline                             # whole flow, scripted team, no network
```

## Running an objective

```bash
uv run orch run "Add an idempotent POST /transfers endpoint in minibank/api/transfers.py" \
  --ac "A repeated Idempotency-Key never double-posts" --ac "Amounts are Decimal, never float"
```

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

The target repo defaults to `.orch/repos/minibank`, made on first run from
`examples/minibank/` (a tiny bank service with a protected `payments/` package); pass
`--repo` for another git repo. Every agent uses `muse-spark-1.3-contributor` unless
`--model` says otherwise.

Everything orch creates (databases, spike results, agent transcripts) goes in one folder,
`.orch/` (gitignored; override with `ORCH_DATA_DIR`). Delete it any time with `uv run orch nuke` (or one part: `orch nuke db`, `orch nuke spike`,
`orch nuke repos`).

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
  paths.py       the one data folder (.orch/)
teams/           the agent team (YAML)
examples/        sample plans; minibank/ (the target app template)
docs/            design notes (mcp-design.md)
```
