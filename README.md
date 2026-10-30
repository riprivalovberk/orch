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
```

Everything orch creates (databases, spike results, agent transcripts) goes in one folder,
`.orch/` (gitignored; override with `ORCH_DATA_DIR`). Delete it any time with `uv run orch nuke` (or one part: `orch nuke db`, `orch nuke spike`).

## Layout

```
src/orch/
  domain/        enums + Pydantic schemas (team, plan, verdicts)
  core/          state machine, team registry, plan validator, event log, orchestrator
  policies/      policy packs (fintech_sdlc: SOD, GATE, COV, APR, PAY rules)
  db/            SQLAlchemy models (12 tables) + SQLite session
  runtime/       Meta Model API wiring, the PM's tools, the PM spike
  paths.py       the one data folder (.orch/)
teams/           the agent team (YAML)
examples/        sample plans
```
