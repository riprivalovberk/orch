"""Command line. More commands (run, approve, evidence, baseline) land with the runtime."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml

from orch.core.plan import PlanValidator, ValidationContext
from orch.core.team import TeamRegistry
from orch.domain.schemas import Plan
from orch.paths import data_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_TEAM = Path("teams/fintech_sdlc.yaml")


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


NUKE_SECTIONS = ("all", "db", "spike")


@app.command()
def nuke(
    section: str = typer.Argument("all", help="all | db (orch.db) | spike (spike runs)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation"),
) -> None:
    """Delete what orch has created: everything, or one section of the data folder."""
    if section not in NUKE_SECTIONS:
        raise typer.BadParameter(f"pick one of {list(NUKE_SECTIONS)}")
    d = data_dir()
    targets = {
        "all": [d],
        "db": sorted(d.glob("orch.db*")),  # db + its -wal/-shm files
        "spike": [d / "spike"],
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
