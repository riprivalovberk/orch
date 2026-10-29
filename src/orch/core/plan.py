"""PlanValidator: checks a PM-written task graph before anything runs.

Rules are plain functions ``(PlanGraph, ValidationContext) -> Iterable[PlanError]``.
Structural rules live here; domain rules come from a policy pack (``orch.policies``).
No I/O and no LLM calls, so every rule is deterministic and unit-testable.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter

from orch.core.team import TeamRegistry
from orch.domain.enums import (
    UNASSIGNED_KINDS,
    VERIFIER_KINDS,
    EdgeKind,
    TaskKind,
)
from orch.domain.schemas import Plan, PlanEdge, PlanError, PlanNode, ProjectLimits


@dataclass(frozen=True)
class AgentCard:
    name: str
    capabilities: frozenset[str]
    description: str = ""
    write_paths: tuple[str, ...] = ()


@dataclass
class ValidationContext:
    agents: dict[str, AgentCard]
    pm: str
    limits: ProjectLimits = field(default_factory=ProjectLimits)
    existing_task_count: int = 0
    remaining_budget_usd: float = float("inf")

    @classmethod
    def from_team(cls, team: TeamRegistry, **kwargs: object) -> ValidationContext:
        agents = {
            a.name: AgentCard(
                a.name, frozenset(a.capabilities), a.description, tuple(a.write_paths)
            )
            for a in team.config.agents
        }
        return cls(agents=agents, pm=team.pm, **kwargs)  # type: ignore[arg-type]

    def agents_with(self, capability: str) -> list[str]:
        return sorted(n for n, a in self.agents.items() if capability in a.capabilities)


class PlanGraph:
    """Adjacency view of a plan. Edges with unknown endpoints are dropped (STR-002 reports them)."""

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.nodes: dict[str, PlanNode] = {}
        for n in plan.nodes:
            self.nodes.setdefault(n.id, n)
        self.out: dict[str, list[PlanEdge]] = defaultdict(list)
        self.inc: dict[str, list[PlanEdge]] = defaultdict(list)
        for e in plan.edges:
            if e.source in self.nodes and e.target in self.nodes:
                self.out[e.source].append(e)
                self.inc[e.target].append(e)

    def of_kind(self, *kinds: TaskKind) -> list[PlanNode]:
        return [n for n in self.nodes.values() if n.kind in kinds]

    def verifiers_of(self, node_id: str) -> list[PlanNode]:
        return [self.nodes[e.target] for e in self.out[node_id] if e.kind == EdgeKind.VERIFIES]

    def verified_by(self, node_id: str) -> list[PlanNode]:
        """The work nodes that ``node_id`` verifies."""
        return [self.nodes[e.source] for e in self.inc[node_id] if e.kind == EdgeKind.VERIFIES]

    def reachable_from(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        queue = deque([node_id])
        while queue:
            for e in self.out[queue.popleft()]:
                if e.target not in seen:
                    seen.add(e.target)
                    queue.append(e.target)
        return seen

    def find_cycle(self) -> list[str] | None:
        """One cycle in edge order, e.g. ``["a", "b", "a"]``, or None if the plan is a DAG."""
        deps = {n: [e.source for e in self.inc[n]] for n in self.nodes}
        try:
            TopologicalSorter(deps).prepare()
        except CycleError as exc:
            return list(exc.args[1])
        return None


Rule = Callable[[PlanGraph, ValidationContext], Iterable[PlanError]]


def err(code: str, node: str | None, message: str, hint: str = "", *, escalate: bool = False):
    return PlanError(code=code, node=node, message=message, hint=hint, escalate=escalate)


# --------------------------------------------------------------------------- structural rules


def str_002_integrity(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    seen: set[str] = set()
    for n in g.plan.nodes:
        if n.id in seen:
            yield err("STR-002", n.id, f"duplicate node id '{n.id}'", "give every node a unique id")
        seen.add(n.id)
    for e in g.plan.edges:
        for end in (e.source, e.target):
            if end not in seen:
                yield err(
                    "STR-002",
                    end,
                    f"edge {e.source} -> {e.target} references unknown node '{end}'",
                    "add the node or fix the edge",
                )
        if e.source == e.target:
            yield err("STR-002", e.source, f"node '{e.source}' has an edge to itself")
    for n in g.plan.nodes:
        for ac in n.covers:
            if ac not in g.plan.acceptance_criteria:
                yield err(
                    "STR-002",
                    n.id,
                    f"covers unknown acceptance criterion '{ac}'",
                    f"use one of {sorted(g.plan.acceptance_criteria)}",
                )


def str_001_acyclic(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    cycle = g.find_cycle()
    if cycle:
        yield err(
            "STR-001",
            cycle[0],
            "the plan has a cycle: " + " -> ".join(cycle),
            "remove one edge so work flows in one direction",
        )


def str_003_assignees(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    for n in g.nodes.values():
        if n.kind in UNASSIGNED_KINDS:
            if n.assignee:
                yield err(
                    "STR-003",
                    n.id,
                    f"{n.kind} nodes are run by the system or a human and take no assignee",
                    "remove the assignee",
                )
            continue
        candidates = ctx.agents_with(n.kind)
        options = ", ".join(candidates) or "none in the team"
        if not n.assignee:
            yield err(
                "STR-003", n.id, f"{n.kind} node has no assignee", f"assign one of: {options}"
            )
        elif n.assignee not in ctx.agents:
            yield err(
                "STR-003",
                n.id,
                f"unknown agent '{n.assignee}'",
                f"call list_agents; agents with {n.kind}: {options}",
            )
        elif n.kind not in ctx.agents[n.assignee].capabilities:
            yield err(
                "STR-003",
                n.id,
                f"agent '{n.assignee}' does not have capability '{n.kind}'",
                f"agents with {n.kind}: {options}",
                escalate=not candidates,
            )


def str_004_limits(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    n = len(g.plan.nodes)
    lim = ctx.limits
    if n == 0:
        yield err("STR-004", None, "the plan has no nodes")
    if n > lim.max_children:
        yield err(
            "STR-004",
            None,
            f"{n} nodes exceeds the limit of {lim.max_children} per plan",
            "merge small steps",
        )
    if ctx.existing_task_count + n > lim.max_tasks:
        yield err(
            "STR-004",
            None,
            f"project would have {ctx.existing_task_count + n} tasks; limit is {lim.max_tasks}",
        )
    est = n * lim.est_cost_per_task_usd
    if est > ctx.remaining_budget_usd:
        yield err(
            "STR-004",
            None,
            f"estimated cost ${est:.2f} exceeds remaining budget ${ctx.remaining_budget_usd:.2f}",
            escalate=True,
        )


def str_005_shapes(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    for n in g.nodes.values():
        if n.kind == TaskKind.PLAN:
            yield err("STR-005", n.id, "plan nodes cannot appear inside a plan")
        if n.kind == TaskKind.GATE and not n.gates:
            yield err(
                "STR-005", n.id, "gate node lists no gates", "set gates: [pytest, ruff, policy]"
            )
        if n.kind != TaskKind.GATE and n.gates:
            yield err("STR-005", n.id, "only gate nodes take a gates list")
        if n.kind in VERIFIER_KINDS and g.out[n.id]:
            targets = ", ".join(e.target for e in g.out[n.id])
            yield err(
                "STR-005",
                n.id,
                f"{n.kind} node {n.id} cannot have outgoing edges (to {targets})",
                f"remove them; point later nodes at the work {n.id} verifies, which only "
                "finishes once every verifier passed",
            )
        if n.kind in VERIFIER_KINDS and not g.verified_by(n.id):
            yield err(
                "STR-005",
                n.id,
                f"{n.kind} node does not verify anything",
                f"add a verifies edge from the work node it checks to {n.id}",
            )
    for e in g.plan.edges:
        if e.kind != EdgeKind.VERIFIES or e.source not in g.nodes or e.target not in g.nodes:
            continue
        src, dst = g.nodes[e.source], g.nodes[e.target]
        if dst.kind not in VERIFIER_KINDS:
            yield err(
                "STR-005",
                dst.id,
                f"verifies edge {src.id} -> {dst.id} must point at a gate, review or "
                "security_review node",
                "use a blocks edge for ordinary ordering",
            )
        if src.kind in VERIFIER_KINDS or src.kind == TaskKind.APPROVAL:
            yield err(
                "STR-005",
                src.id,
                f"{src.kind} nodes cannot be verified (edge {src.id} -> {dst.id})",
            )


STRUCTURAL_RULES: tuple[Rule, ...] = (
    str_001_acyclic,
    str_003_assignees,
    str_004_limits,
    str_005_shapes,
)


class PlanValidator:
    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = list(rules)

    @classmethod
    def for_pack(cls, pack: str) -> PlanValidator:
        from orch.policies import get_pack

        return cls([*STRUCTURAL_RULES, *get_pack(pack)])

    def validate(self, plan: Plan, ctx: ValidationContext) -> list[PlanError]:
        g = PlanGraph(plan)
        errors = list(str_002_integrity(g, ctx))
        if errors:  # later rules assume a well-formed graph
            return errors
        for rule in self.rules:
            errors.extend(rule(g, ctx))
        unique: dict[tuple[str, str | None, str], PlanError] = {}
        for e in errors:
            unique.setdefault((e.code, e.node, e.message), e)
        return list(unique.values())
