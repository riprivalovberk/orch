"""Pydantic models: team config, plans, validation errors, review verdicts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from orch.core.scope import is_safe_relative
from orch.domain.enums import AGENT_CAPABILITIES, EdgeKind, GateName, TaskKind, Verdict

SLUG = r"^[a-z][a-z0-9_]{0,63}$"

KNOWN_TOOLS = frozenset(
    {
        "list_agents",
        "validate_plan",
        "submit_plan",
        "create_subtask",
        "handoff_task",
        "get_task_status",
        "get_task_tree",
        "request_human_input",
        "read_file",
        "list_files",
        "search_code",
        "write_file",
        "run_tests",
        "submit_artifact",
        "complete_task",
        "submit_review",
        "get_remaining_budget",
    }
)


class ProjectLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tasks: int = 40
    max_depth: int = 3
    max_children: int = 12
    max_revision_rounds: int = 2
    max_plan_attempts: int = 3
    max_attempts: int = 3
    budget_usd: float = 5.0
    est_cost_per_task_usd: float = 0.05


# --------------------------------------------------------------------------- team


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=SLUG)
    description: str = Field(min_length=10)
    capabilities: list[str] = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)
    write_paths: list[str] = Field(default_factory=list)  # globs; see core/scope.py
    model: str = "muse-spark-1.3"
    prompt: str = ""

    @field_validator("capabilities")
    @classmethod
    def _known_capabilities(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - AGENT_CAPABILITIES)
        if unknown:
            raise ValueError(
                f"unknown capabilities {unknown}; allowed: {sorted(AGENT_CAPABILITIES)}"
            )
        return v

    @field_validator("tools")
    @classmethod
    def _known_tools(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - KNOWN_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools {unknown}")
        return v

    @field_validator("write_paths")
    @classmethod
    def _relative_paths(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if not is_safe_relative(p)]
        if bad:
            raise ValueError(f"write_paths must be repo-relative without '..': {bad}")
        return v

    @model_validator(mode="after")
    def _writer_has_scope(self) -> AgentConfig:
        if "write_file" in self.tools and not self.write_paths:
            raise ValueError(f"{self.name} has write_file but no write_paths")
        return self


class TeamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team: str
    pm: str
    agents: list[AgentConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> TeamConfig:
        names = [a.name for a in self.agents]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate agent names {dupes}")
        by_name = {a.name: a for a in self.agents}
        if self.pm not in by_name:
            raise ValueError(f"pm '{self.pm}' is not defined in agents")
        if TaskKind.PLAN.value not in by_name[self.pm].capabilities:
            raise ValueError(f"pm '{self.pm}' must have the 'plan' capability")
        return self


# --------------------------------------------------------------------------- plans


class PlanNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SLUG)
    kind: TaskKind
    title: str | None = None
    objective: str = ""
    assignee: str | None = None
    covers: list[str] = Field(default_factory=list)  # acceptance criterion ids
    gates: list[GateName] = Field(default_factory=list)  # gate nodes only


class PlanEdge(BaseModel):
    """``source -> target``. Also accepts the short form ``[source, target, kind]``."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: EdgeKind = EdgeKind.BLOCKS

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, data: Any) -> Any:
        if isinstance(data, list | tuple):
            if len(data) not in (2, 3):
                raise ValueError("an edge is [source, target] or [source, target, kind]")
            out: dict[str, Any] = {"source": data[0], "target": data[1]}
            if len(data) == 3:
                out["kind"] = data[2]
            return out
        return data


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str
    acceptance_criteria: dict[str, str] = Field(default_factory=dict)
    nodes: list[PlanNode]
    edges: list[PlanEdge] = Field(default_factory=list)


class PlanError(BaseModel):
    """One validation failure. ``escalate`` means the PM cannot fix it by re-planning."""

    code: str
    node: str | None = None
    message: str
    hint: str = ""
    escalate: bool = False

    def __str__(self) -> str:
        where = f" [{self.node}]" if self.node else ""
        return f"{self.code}{where}: {self.message}"


# --------------------------------------------------------------------------- verification


class Finding(BaseModel):
    severity: Literal["blocker", "major", "minor"]
    issue: str
    file: str | None = None
    line: int | None = None
    criterion_id: str | None = None


class ReviewVerdict(BaseModel):
    verdict: Verdict
    findings: list[Finding] = Field(default_factory=list)
    summary: str = ""


class GateResultIn(BaseModel):
    gate: GateName
    passed: bool
    output: str = ""
    duration_ms: int = 0
