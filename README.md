# FinOrch

**Your agents write their own workflow. We check it like a compiler before anything runs.**

A team of AI agents, defined in one YAML file, ships code changes under a bank's change-control rules.
A PM agent plans each objective as a task graph. A validator rejects plans that break the rules
(segregation of duties, required test gates, criteria coverage, security review for payments code)
before any work starts. Agents talk to the orchestrator only through MCP.

Built for the Meta Global AI Developer Hackathon (Indie track) on `muse-spark-1.3`.

## Quickstart

None of this needs a model API key.

```bash
brew install uv            # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run pytest
uv run orch agents
uv run orch plan-check examples/plan_transfers.yaml   # clears
uv run orch plan-check examples/plan_bad.yaml         # rejected, with node-level errors
uv run orch ui                                        # http://127.0.0.1:8765
```

`uv run pytest` runs everything offline. The live tests (they call the real model and cost a
little credit) are skipped unless you set `ORCH_LIVE=1` and `MODEL_API_KEY` (your Meta Model
API key). To run every test, live ones included:

```bash
ORCH_LIVE=1 MODEL_API_KEY=<your Meta Model API key> uv run pytest
```

## The minibank demo

The demos build on minibank, a tiny bank service with a protected `payments/` package. Its
source ships in `examples/minibank/`; the agents never work on that folder itself, but on a
git repo made from it:

```bash
uv run orch minibank                   # makes .orch/repos/minibank (or: orch minibank <folder>)
```

### Offline: the whole flow, no key, no network

A scripted team stands in for the model and builds the transfers feature for real: plan, plan
check, tests, code, gates, review.

```bash
uv run orch run --offline --no-interactive --repo .orch/repos/minibank
uv run orch inbox                      # the approval it stopped at
uv run orch approve <request id>
uv run orch run --resume <objective>   # merges into the repo's main
uv run orch evidence <objective>       # audit pack: plans, gates, reviews, approvals, chain check
```

Without `--no-interactive` the approval is asked in the terminal and the run goes straight
through. `--resume` keeps the team the objective started with, so an offline objective stays
offline.

### Live: your own objective

Put your key in `.env` (`MODEL_API_KEY=...`, gitignored) or set it in the web UI's Admin page.
Then, on a fresh minibank repo:

```bash
uv run orch minibank ~/demo/minibank
uv run orch run --repo ~/demo/minibank \
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
`--no-interactive` the run stops and waits for `orch inbox` / `orch approve` (or
`orch answer <request id> <response>`) and `orch run --resume`, as above.

## Running on your own repo

`--repo` can be any git repo with a `main` branch; approved changes merge into its `main`,
and `--resume` always continues in the objective's own repo. The offline team only knows the
minibank transfers change, so other repos need a key.

Each repo gets one project whose budget covers all its objectives: $2.00 to start, changed
with `--budget`. Every agent uses `muse-spark-1.3` unless `--model` (or the UI's Run
settings) says otherwise; `muse-spark-1.3-contributor` is cheaper, but Meta may train on
what it is sent. A resumed run keeps the model and parallel cap it started with. Workers can
hand in non-code files (test reports, notes) with `submit_artifact`; they are stored by hash
under `.orch/artifacts/` and listed in the evidence.

Everything FinOrch creates (databases, demo repos, worktrees, agent transcripts) goes in one
folder, `.orch/` (gitignored; override with `ORCH_DATA_DIR`). Delete it any time with
`uv run orch nuke`, or one part: `orch nuke db`, `orch nuke spike`, `orch nuke repos` (this
also deletes the minibank repo `orch minibank` made there).

## Web UI

```bash
uv run orch ui                         # http://127.0.0.1:8765
```

Make a project with **+ New**. It points at a git repo you already have (type the path or
Browse…), or at a new or empty folder, which starts as an empty Python project or as the
**minibank sample**. For the offline demo, pick the minibank sample and tick *Offline
scripted demo* under the chat's *Run settings*: sending a message then starts the transfers
objective.

Each project opens as a chat with the PM. Ask it about the project (where code is, what an
objective changed, why a check failed: it reads main and every objective's branch), or say
what to build: it starts an objective, drafts acceptance criteria with you, then posts its
plan and the team's progress in the thread. Its questions, the final approval and any
escalation arrive as messages you answer in place (answering resumes the run). A collapsible
Files panel on the right lists the repo's files, previews them and uploads new ones
(committed to `main`, so every new objective's agents can read them).

Everything else is on the project's details page and the top bar: budget, objectives with
their tasks, gates, reviews, diffs and evidence pack, the objective's event log, the
event-chain check, the team, clean-up and the model API key (Admin; saved to `.env`). Runs
started in the UI go in background threads of the UI process; events are read from the
database, so `orch run` runs show too.

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
  web/           the web UI (`orch ui`): Starlette JSON API + one HTML page
  paths.py       the one data folder (.orch/)
teams/           the agent team (YAML)
examples/        sample plans (plan-check), sample data, and minibank (the demo app's source)
docs/            design notes (mcp-design.md)
```
