"""Command line: run an objective, answer the human inbox, export evidence."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import typer
import yaml

from orch.core.orchestrator import OrchestratorError, TaskOrchestrator
from orch.core.plan import PlanValidator, ValidationContext
from orch.core.team import TeamRegistry
from orch.db.models import Event
from orch.db.session import open_db
from orch.domain.schemas import Plan, ProjectLimits
from orch.paths import data_dir, db_path

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_TEAM = Path("teams/fintech_sdlc.yaml")
DEFAULT_BUDGET = 2.0


@app.command()
def agents(team: Path = typer.Option(DEFAULT_TEAM, help="Team YAML file")) -> None:
    """List the agents the PM can plan with."""
    reg = TeamRegistry.load(team)
    for a in reg.describe():
        pm = " (PM)" if a["name"] == reg.pm else ""
        typer.echo(f"{a['name']}{pm}  [{', '.join(a['capabilities'])}]")  # type: ignore[arg-type]
        typer.echo(f"    {a['description']}")


@app.command("plan-check")
def plan_check(
    plan: Path = typer.Argument(..., help="Plan YAML or JSON file"),
    team: Path = typer.Option(DEFAULT_TEAM, help="Team YAML file"),
    pack: str = typer.Option("fintech_sdlc", help="Policy pack"),
) -> None:
    """Validate a plan file against the structural rules and a policy pack."""
    spec = Plan.model_validate(yaml.safe_load(plan.read_text()))
    ctx = ValidationContext.from_team(TeamRegistry.load(team))
    errors = PlanValidator.for_pack(pack).validate(spec, ctx)
    if not errors:
        typer.secho(f"plan cleared: {len(spec.nodes)} nodes, {len(spec.edges)} edges", fg="green")
        raise typer.Exit(0)
    typer.secho(f"plan rejected: {len(errors)} error(s)", fg="red", bold=True)
    for e in errors:
        flag = "  [needs a human]" if e.escalate else ""
        typer.echo(f"  {e}{flag}")
        if e.hint:
            typer.echo(f"      hint: {e.hint}")
    raise typer.Exit(1)


# ------------------------------------------------------------------------- run


def _orch() -> TaskOrchestrator:
    data_dir().mkdir(parents=True, exist_ok=True)
    return TaskOrchestrator(open_db(db_path()))


def _criteria(items: list[str]) -> dict[str, str]:
    """--ac "text" or --ac "AC-7=text" -> {"AC-1": "text", ...}"""
    out: dict[str, str] = {}
    for i, item in enumerate(items, 1):
        key, sep, text = item.partition("=")
        if sep and key.strip().upper().startswith("AC-"):
            out[key.strip().upper()] = text.strip()
        else:
            out[f"AC-{i}"] = item.strip()
    return out


class EventPrinter:
    """One line per event that a person watching the run cares about."""

    def __init__(self, orch: TaskOrchestrator) -> None:
        self.orch = orch
        self.names: dict[str, str] = {}

    def name(self, task_id: str | None) -> str:
        if not task_id:
            return "-"
        if task_id not in self.names:
            t = self.orch.get_task(task_id)
            self.names[task_id] = t["node"] or "objective"
        return self.names[task_id]

    def __call__(self, e: Event) -> None:
        p: dict[str, Any] = e.payload
        who = f"{self.name(e.task_id):<20} {e.actor:<18}"
        if e.type == "task.status":
            line = f"{p['from']:>14} -> {p['to']:<14} {p.get('reason', '')}"
        elif e.type == "plan.rejected":
            codes = sorted({err["code"] for err in p["errors"]})
            line = f"plan v{p['version']} rejected: {', '.join(codes)}"
        elif e.type == "plan.accepted":
            line = f"plan v{p['version']} accepted: {len(p['tasks'])} tasks"
        elif e.type == "human.requested":
            line = f"asks a human: {p['prompt'][:90]}"
        elif e.type.startswith("workspace."):
            line = e.type.removeprefix("workspace.").replace("_", " ")
        elif e.type == "artifact.added":
            line = f"artifact {p['name']} ({p['kind']}, {p['bytes']:,} bytes)"
        elif e.type == "run.usage":
            line = f"usage {p['tokens']:,} tokens ${p['usd']:.4f}"
        else:
            return
        typer.echo(f"  {who} {line}")


def _ask(req: dict[str, Any]) -> str | None:
    typer.echo("")
    typer.secho(f"  [{req['kind']}] {req['prompt']}", fg="yellow", bold=True)
    if req["options"]:
        while True:
            answer = typer.prompt(f"  answer ({' / '.join(req['options'])}, or 'later')")
            if answer == "later":
                return None
            if answer in req["options"]:
                return answer
    answer = typer.prompt("  answer (or 'later')")
    return None if answer == "later" else answer


@app.command()
def run(
    objective: str = typer.Argument("", help="What to build (omit with --resume)"),
    ac: list[str] = typer.Option([], "--ac", help="Acceptance criterion; repeat for each"),
    resume: str = typer.Option("", help="Continue an objective: its root task id"),
    team: Path = typer.Option(DEFAULT_TEAM, help="Team YAML file"),
    repo: Path | None = typer.Option(
        None,
        help="Target git repo with a main branch (on --resume, the objective's own repo)",
    ),
    parallel: int = typer.Option(3, min=1, help="Agents running at once"),
    model_name: str = typer.Option("", "--model", help="Model for every agent"),
    budget: float | None = typer.Option(
        None,
        min=0.01,
        help=f"Total USD budget for the repo's project, shared by all its objectives "
        f"(new project: ${DEFAULT_BUDGET:.2f}; given again, it replaces the old budget)",
    ),
    offline: bool = typer.Option(False, help="Scripted team, no network (transfers demo)"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Answer human requests in the terminal"
    ),
) -> None:
    """Run an objective end to end: plan, build in parallel, verify, approve, merge."""
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    from orch.runtime.model import DEFAULT_MODEL, PRICES, meta_model
    from orch.runtime.runner import Runner, last_model
    from orch.runtime.scripted import (
        SCRIPTED_MODEL,
        TRANSFERS_CRITERIA,
        TRANSFERS_OBJECTIVE,
        scripted_team,
    )
    from orch.runtime.workspace import GitError, Workspace

    model_name = model_name or DEFAULT_MODEL
    if model_name not in PRICES:
        raise typer.BadParameter(f"no price for {model_name}; add it to model.PRICES")
    orch = _orch()
    reg = TeamRegistry.load(team)
    repo = repo.resolve() if repo else None

    if resume:
        try:
            pid = orch.get_task(resume)["project_id"]
        except OrchestratorError as e:
            raise typer.BadParameter(f"no objective {resume}") from e
        saved = orch.get_project(pid)["repo_path"]
        if repo and saved and repo != Path(saved):
            raise typer.BadParameter(f"{resume} runs in {saved}, not {repo}")
        repo = Path(saved) if saved else repo
        root = resume
        if not offline and last_model(orch, root) == SCRIPTED_MODEL:
            offline = True  # the scripted team started it; a live model can't pick up its work
            typer.echo("  started offline: resuming with the scripted team")
    else:
        if offline and not objective:
            objective, ac = TRANSFERS_OBJECTIVE, [f"{k}={v}" for k, v in TRANSFERS_CRITERIA.items()]
        if not objective:
            raise typer.BadParameter("give an objective, or --resume <root task id>")
        if repo is None:
            hint = "; make a demo repo with: orch minibank" if offline else ""
            raise typer.BadParameter(f"give --repo, the target git repo the agents work in{hint}")
        pid = orch.project_for_repo(str(repo))
        if pid is None:
            limits = ProjectLimits(budget_usd=budget or DEFAULT_BUDGET)
            pid = orch.create_project(repo.name, repo_path=str(repo), limits=limits)
            budget = None  # already set
        root = ""
    if repo is None:
        raise typer.BadParameter(f"{resume} has no repo; give --repo")
    try:
        ws = Workspace.create(repo, data_dir() / "worktrees")
    except GitError as e:
        raise typer.BadParameter(str(e)) from e
    orch.register_team(pid, reg)  # on resume too: picks up team edits
    if budget is not None:
        orch.set_budget(pid, budget)
    if not root:
        root = orch.create_root_task(pid, objective, acceptance_criteria=_criteria(ac))
    orch.recover()  # anything RUNNING was left by a runner that died
    money = orch.get_project(pid)
    typer.secho(f"objective {root} in {repo}", bold=True)
    typer.echo(f"  budget ${money['budget_usd']:.2f}, spent ${money['spent_usd']:.4f} so far")
    orch.events.subscribe(EventPrinter(orch))

    model = scripted_team() if offline else meta_model(model_name)
    runner = Runner(orch, reg, ws, model=model, price_as=model_name, parallel=parallel)
    ask = _ask if interactive and sys.stdin.isatty() else None
    report = asyncio.run(runner.run(root, ask=ask))

    typer.echo("")
    color = "green" if report.status == "DONE" else "yellow"
    typer.secho(f"{report.status}: {report.note or 'objective finished'}", fg=color, bold=True)
    if report.merged_into_main:
        typer.echo(f"  merged into main of {repo} at {report.merged_into_main[:10]}")
    for r in report.waiting_on:
        typer.echo(f"  waiting on {r['id']}: {r['prompt'][:100]}")
    if report.waiting_on:
        typer.echo(f"  answer with: orch answer <id> <response>, then orch run --resume {root}")
    typer.echo(f"  evidence: orch evidence {root}")


@app.command()
def minibank(
    folder: Path | None = typer.Argument(
        None, help="New folder for the repo (default: <data dir>/repos/minibank)"
    ),
) -> None:
    """Make a fresh git repo of minibank, the sample bank app, to run the demos on."""
    from orch.paths import default_minibank_repo
    from orch.runtime.workspace import GitError, make_minibank

    try:
        repo = make_minibank((folder or default_minibank_repo()).expanduser().resolve())
    except GitError as e:
        raise typer.BadParameter(str(e)) from e
    typer.secho(f"minibank repo ready: {repo}", fg="green")
    typer.echo(f"  offline demo:  orch run --offline --repo {repo}")
    typer.echo(f"  web UI:        orch ui, then + New project with repo folder {repo}")


# ------------------------------------------------------------------------- humans


@app.command()
def inbox() -> None:
    """Questions, approvals and escalations waiting on a person."""
    orch = _orch()
    reqs = [r for pid in orch.project_ids() for r in orch.open_human_requests(pid)]
    if not reqs:
        typer.echo("inbox empty")
        return
    for r in reqs:
        root = orch.get_task(r["task_id"])["root_id"]
        opts = f"  [{' / '.join(r['options'])}]" if r["options"] else ""
        typer.secho(f"{r['id']}  {r['kind']}  (objective {root})", bold=True)
        typer.echo(f"  {r['prompt']}{opts}")


@app.command()
def answer(
    request_id: str = typer.Argument(..., help="Request id from orch inbox"),
    response: str = typer.Argument(..., help="One of the options, or free text"),
) -> None:
    """Answer a human request. Then continue with orch run --resume <objective>."""
    orch = _orch()
    try:
        status = orch.resolve_human(request_id, response)
    except OrchestratorError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e
    typer.echo(f"answered; task is now {status}")


@app.command()
def approve(request_id: str = typer.Argument(..., help="Approval request id")) -> None:
    """Approve a change for merge (same as: orch answer <id> approve)."""
    answer(request_id, "approve")


# ------------------------------------------------------------------------- evidence


def evidence_markdown(ev: dict[str, Any], chain_ok: bool, events: int) -> str:
    root = ev["root"]
    lines = [
        f"# Change evidence: {ev['objective']}",
        "",
        f"- Objective task: `{root['id']}`, status **{root['status']}**"
        + (f" ({root['summary']})" if root["summary"] else ""),
        f"- Model cost: ${ev['cost_usd']:.4f}",
        f"- Event log: {events} events, hash chain "
        + ("**intact**" if chain_ok else "**BROKEN: tampered**"),
        "",
        "## Acceptance criteria",
        "",
        *[f"- **{k}**: {v}" for k, v in ev["criteria"].items()],
        "",
        "## Plans",
        "",
    ]
    for p in ev["plans"]:
        verdict = "accepted" if p["valid"] else "rejected"
        lines.append(f"- v{p['version']}: {verdict}")
        lines += [f"  - {e}" for e in p["errors"]]
    lines += ["", "## Tasks", "", "| node | kind | assignee | status | outcome | rounds |"]
    lines.append("|---|---|---|---|---|---|")
    for t in ev["tasks"][1:]:
        lines.append(
            f"| {t['node']} | {t['kind']} | {t['assignee'] or '-'} | {t['status']} "
            f"| {t['outcome'] or '-'} | {t['revision_round']} |"
        )
    lines += ["", "## Gates", "", "| target | round | gate | passed |", "|---|---|---|---|"]
    lines += [
        f"| {g['target']} | {g['round']} | {g['gate']} | {g['passed']} |" for g in ev["gates"]
    ]
    lines += ["", "## Reviews", ""]
    for r in ev["reviews"]:
        lines.append(f"- {r['target']} round {r['round']}: **{r['verdict']}** by {r['reviewer']}")
        if r["summary"]:
            lines.append(f"  - {r['summary']}")
        for f in r["findings"]:
            cite = f" ({f['criterion_id']})" if f.get("criterion_id") else ""
            lines.append(f"  - [{f['severity']}] {f['issue']}{cite}")
    if ev["artifacts"]:
        lines += ["", "## Artifacts", "", "| task | name | kind | sha256 |", "|---|---|---|---|"]
        lines += [
            f"| {a['task']} | {a['name']} | {a['kind']} | `{a['sha256'][:12]}` |"
            for a in ev["artifacts"]
        ]
    lines += ["", "## Human decisions", ""]
    for d in ev["decisions"]:
        lines.append(f"- {d['task']} ({d['kind']}): {d['response'] or d['status']}")
        lines.append(f"  - {d['prompt']}")
    if ev["plans"] and (spec := ev["plans"][-1]["spec"]):
        lines += ["", "## Accepted plan", "", "```json", json.dumps(spec, indent=2), "```"]
    return "\n".join(lines) + "\n"


@app.command()
def evidence(
    root_id: str = typer.Argument(..., help="Objective (root task) id"),
    out: Path = typer.Option(None, help="Write here (default: <data dir>/evidence/<id>.md)"),
) -> None:
    """The audit pack for one objective: plans, verdicts, gates, approvals, chain check."""
    orch = _orch()
    ev = orch.evidence(root_id)
    chain = orch.verify_events(ev["project_id"])
    out = out or data_dir() / "evidence" / f"{root_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(evidence_markdown(ev, chain.ok, chain.count))
    typer.echo(f"wrote {out}")
    if not chain.ok:
        typer.secho(f"event chain broken at seq {chain.first_bad_seq}: {chain.reason}", fg="red")
        raise typer.Exit(1)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Address to listen on"),
    port: int = typer.Option(8765, help="Port"),
    team: Path = typer.Option(DEFAULT_TEAM, help="Team YAML file"),
) -> None:
    """The web UI: projects, a chat with each project's PM, runs, approvals, evidence."""
    import uvicorn

    from orch.web.server import create_app

    web = create_app(_orch(), team_path=team.resolve(), default_budget=DEFAULT_BUDGET)
    typer.secho(f"FinOrch UI on http://{host}:{port}  (data in {data_dir()})", bold=True)
    uvicorn.run(web, host=host, port=port, log_level="warning")


NUKE_SECTIONS = ("all", "db", "spike", "repos")


@app.command()
def nuke(
    section: str = typer.Argument(
        "all", help="all | db (orch.db) | spike (spike runs) | repos (target repos, worktrees)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation"),
) -> None:
    """Delete what FinOrch has created: everything, or one section of the data folder."""
    if section not in NUKE_SECTIONS:
        raise typer.BadParameter(f"pick one of {list(NUKE_SECTIONS)}")
    d = data_dir()
    targets = {
        "all": [d],
        "db": sorted(d.glob("orch.db*")),  # db + its -wal/-shm files
        "spike": [d / "spike"],
        "repos": [d / "repos", d / "worktrees"],
    }[section]
    targets = [t for t in targets if t.exists()]
    if not targets:
        typer.echo(f"nothing to delete in {d}")
        return
    if not yes:
        names = ", ".join(str(t) for t in targets)
        typer.confirm(f"Delete {names}?", abort=True)
    for t in targets:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
        typer.echo(f"deleted {t}")


if __name__ == "__main__":
    app()
