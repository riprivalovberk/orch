"""Change-control rules for software changes at a regulated financial institution.

SOD  segregation of duties     GATE  deterministic checks     COV  test coverage
APR  human approval            PAY   payments code needs a security review
SCOPE  every test/implement node says what it writes, and its assignee may write it
"""

from __future__ import annotations

from collections.abc import Iterable

from orch.core.plan import PlanGraph, Rule, ValidationContext, err
from orch.core.scope import can_write, protected_grants, touches
from orch.domain.enums import REVIEW_KINDS, GateName, TaskKind
from orch.domain.schemas import PlanError

REQUIRED_GATES = frozenset({GateName.PYTEST, GateName.RUFF, GateName.POLICY})
SENSITIVE_DIRS = frozenset({"payments"})
BUILD_KINDS = frozenset({TaskKind.IMPLEMENT, TaskKind.TEST, *REVIEW_KINDS})
WRITING_KINDS = (TaskKind.TEST, TaskKind.IMPLEMENT)


def sod_001_independent_review(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    """Every implement node is reviewed by someone other than its author."""
    for impl in g.of_kind(TaskKind.IMPLEMENT):
        reviews = [v for v in g.verifiers_of(impl.id) if v.kind in REVIEW_KINDS]
        others = [a for a in ctx.agents_with(TaskKind.REVIEW) if a != impl.assignee]
        for v in reviews:
            if v.assignee and v.assignee == impl.assignee:
                yield err(
                    "SOD-001",
                    v.id,
                    f"{v.id} is assigned to {impl.assignee}, who also implements {impl.id}; "
                    "authors cannot review their own work",
                    f"assign {v.id} to {others[0]}" if others else "",
                )
        independent = [
            v
            for v in reviews
            if v.kind == TaskKind.REVIEW and v.assignee and v.assignee != impl.assignee
        ]
        if not independent:
            yield err(
                "SOD-001",
                impl.id,
                f"{impl.id} has no independent review",
                f"add a review node assigned to {others[0]} with a verifies edge from {impl.id}"
                if others
                else "no agent other than the author can review; a human must add a reviewer",
                escalate=not others,
            )


def sod_002_pm_does_not_build(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    """The PM plans; it never writes, tests or reviews code."""
    for n in g.nodes.values():
        if n.assignee == ctx.pm and n.kind in BUILD_KINDS:
            yield err(
                "SOD-002",
                n.id,
                f"the PM ({ctx.pm}) cannot be assigned {n.kind} work",
                "assign this node to a specialist",
            )


def gate_001_required_gates(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    """Every implement node is verified by pytest, ruff and the policy scan."""
    for impl in g.of_kind(TaskKind.IMPLEMENT):
        covered: set[GateName] = set()
        for v in g.verifiers_of(impl.id):
            if v.kind == TaskKind.GATE:
                covered.update(v.gates)
        missing = sorted(REQUIRED_GATES - covered)
        if missing:
            yield err(
                "GATE-001",
                impl.id,
                f"{impl.id} is not verified by gate(s) {', '.join(missing)}",
                "add a gate node with gates: [pytest, ruff, policy] and a verifies edge "
                f"from {impl.id}",
            )


def cov_001_criteria_tested(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    """Every acceptance criterion is covered by at least one test node."""
    if not g.plan.acceptance_criteria:
        yield err(
            "COV-001",
            None,
            "the plan has no acceptance criteria",
            "list the criteria agreed with the user (AC-1, AC-2, ...)",
        )
        return
    covered = {ac for n in g.of_kind(TaskKind.TEST) for ac in n.covers}
    for ac in g.plan.acceptance_criteria:
        if ac not in covered:
            yield err(
                "COV-001",
                None,
                f"criterion {ac} has no test node",
                f"add {ac} to the covers list of a test node",
            )


def apr_001_single_approval(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    """Exactly one human approval node, last, reachable from every test and implement node.

    Verifiers need no path: their work node only finishes once they pass.
    """
    approvals = g.of_kind(TaskKind.APPROVAL)
    if len(approvals) != 1:
        yield err(
            "APR-001",
            None,
            f"the plan needs exactly one approval node, found {len(approvals)}",
            "end the plan with one approval node",
        )
        return
    apr = approvals[0]
    if g.out[apr.id]:
        yield err("APR-001", apr.id, "the approval node must be last", "remove its outgoing edges")
    for n in g.of_kind(TaskKind.TEST, TaskKind.IMPLEMENT):
        if apr.id not in g.reachable_from(n.id):
            yield err(
                "APR-001",
                n.id,
                f"{n.id} does not lead to the approval node {apr.id}",
                f"add a blocks edge from {n.id} to {apr.id}",
            )


def scope_001_paths(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    """Test and implement nodes list the paths they write, inside the assignee's write_paths.

    Without this, PAY-001 could only guess from the assignee, and a PM could dodge it by
    giving payments work to an agent that can't write payments code.
    """
    for n in g.nodes.values():
        if n.kind not in WRITING_KINDS:
            if n.paths:
                yield err("SCOPE-001", n.id, "only test and implement nodes take paths")
            continue
        card = ctx.agents.get(n.assignee or "")
        if card is None:  # STR-003 reports it
            continue
        if not n.paths:
            yield err(
                "SCOPE-001",
                n.id,
                f"{n.id} does not say which files it writes",
                f"set paths to the files or globs it will write, within {card.name}'s "
                f"write_paths {list(card.write_paths)}",
            )
        for path in n.paths:
            if can_write(card.write_paths, path, SENSITIVE_DIRS):
                continue
            able = [
                a
                for a in ctx.agents_with(n.kind)
                if can_write(ctx.agents[a].write_paths, path, SENSITIVE_DIRS)
            ]
            protected = sorted(touches([path], SENSITIVE_DIRS))
            why = (
                f" ({protected[0]}/ is only writable by agents whose scope names it)"
                if (protected)
                else ""
            )
            yield err(
                "SCOPE-001",
                n.id,
                f"{card.name} cannot write {path}{why}",
                f"assign {n.id} to {' or '.join(able)}"
                if able
                else f"no agent with {n.kind} can write {path}; change the path",
            )


def pay_001_security_review(g: PlanGraph, ctx: ValidationContext) -> Iterable[PlanError]:
    """Work that writes payments/ code needs a security review, whoever is assigned."""
    for n in g.of_kind(*WRITING_KINDS):
        card = ctx.agents.get(n.assignee or "")
        if hits := sorted(touches(n.paths, SENSITIVE_DIRS)):
            what = f"{n.id} writes {hits[0]}/ code"
        elif card and (hits := sorted(protected_grants(card.write_paths, SENSITIVE_DIRS))):
            what = f"{n.id} is assigned to {n.assignee}, who can write {hits[0]}/ code"
        else:
            continue
        if any(v.kind == TaskKind.SECURITY_REVIEW for v in g.verifiers_of(n.id)):
            continue
        specialists = [a for a in ctx.agents_with(TaskKind.SECURITY_REVIEW) if a != n.assignee]
        if specialists:
            yield err(
                "PAY-001",
                n.id,
                f"{what}, so it needs a security review",
                f"add a security_review node assigned to {specialists[0]} with a verifies "
                f"edge from {n.id}",
            )
        else:
            yield err(
                "PAY-001",
                n.id,
                f"{what}, which requires a security review, but no agent in the team has "
                "the security_review capability",
                "a human must add a security reviewer to the team file; re-planning cannot "
                "fix this",
                escalate=True,
            )


RULES: tuple[Rule, ...] = (
    sod_001_independent_review,
    sod_002_pm_does_not_build,
    gate_001_required_gates,
    cov_001_criteria_tested,
    apr_001_single_approval,
    scope_001_paths,
    pay_001_security_review,
)
