from __future__ import annotations

import itertools

from orch.core.plan import PlanGraph, PlanValidator, ValidationContext
from orch.domain.schemas import Plan, PlanEdge, PlanNode, ProjectLimits

V = PlanValidator.for_pack("fintech_sdlc")


def codes(errors):
    return sorted(e.code for e in errors)


def check(plan: Plan, team, **ctx):
    return V.validate(plan, ValidationContext.from_team(team, **ctx))


def edit(plan: Plan, *, drop_nodes=(), drop_edges=(), add_nodes=(), add_edges=()) -> Plan:
    p = plan.model_copy(deep=True)
    p.nodes = [n for n in p.nodes if n.id not in drop_nodes] + [PlanNode(**n) for n in add_nodes]
    p.edges = [
        e
        for e in p.edges
        if e.source not in drop_nodes
        and e.target not in drop_nodes
        and (e.source, e.target) not in drop_edges
    ] + [PlanEdge.model_validate(e) for e in add_edges]
    return p


def test_good_plan_clears(good_plan, team):
    assert check(good_plan, team) == []


def test_missing_review_is_sod_001(good_plan, team):
    errors = check(edit(good_plan, drop_nodes=["review_transfer"]), team)
    assert "SOD-001" in codes(errors)
    sod = next(e for e in errors if e.code == "SOD-001")
    assert sod.node == "implement_transfer" and "code_reviewer" in sod.hint


def test_self_review_is_sod_001(good_plan, team, team_with_security):
    # make the developer also a reviewer, then assign the review to them
    cfg = team.config.model_copy(deep=True)
    next(a for a in cfg.agents if a.name == "developer").capabilities.append("review")
    from orch.core.team import TeamRegistry

    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "review_transfer").assignee = "developer"
    errors = check(p, TeamRegistry(cfg))
    assert codes(errors).count("SOD-001") == 2  # self review + no independent review


def test_pm_cannot_build(good_plan, team):
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "write_tests").assignee = "project_manager"
    assert "SOD-002" in codes(check(p, team))


def test_uncovered_criterion_is_cov_001(good_plan, team):
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "write_tests").covers = ["AC-1", "AC-2", "AC-4"]
    errors = [e for e in check(p, team) if e.code == "COV-001"]
    assert len(errors) == 1 and "AC-3" in errors[0].message


def test_missing_gate_is_gate_001(good_plan, team):
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "gates").gates = ["pytest"]
    errors = [e for e in check(p, team) if e.code == "GATE-001"]
    assert errors and "policy" in errors[0].message and "ruff" in errors[0].message


def test_cycle_is_str_001(good_plan, team):
    p = edit(good_plan, add_edges=[["implement_transfer", "write_tests"]])
    assert "STR-001" in codes(check(p, team))


def test_gate_verifying_tests_and_impl_is_str_006(good_plan, team):
    # the positions-rec deadlock: write_tests waits for gates, gates waits for
    # implement_transfer, implement_transfer waits for write_tests
    p = edit(good_plan, add_edges=[["write_tests", "gates", "verifies"]])
    errors = [e for e in check(p, team) if e.code == "STR-006"]
    assert len(errors) == 1 and errors[0].node == "gates"
    assert {"write_tests", "implement_transfer"} <= set(
        errors[0].message.split(": ")[1].split(", ")
    )
    assert "STR-001" not in codes(check(p, team))  # the edges alone have no cycle


SECOND_IMPL = {
    "id": "implement_api",
    "kind": "implement",
    "assignee": "developer",
    "paths": ["minibank/api.py"],
}


def test_gate_before_later_implementation_is_gate_002(good_plan, team):
    # positions-rec-5: the gate on the first step ran the tests for the second step's code
    p = edit(
        good_plan,
        add_nodes=[SECOND_IMPL],
        add_edges=[["implement_transfer", "implement_api"], ["implement_api", "approve"]],
    )
    errors = [e for e in check(p, team) if e.code == "GATE-002"]
    assert len(errors) == 1 and errors[0].node == "gates" and "implement_api" in errors[0].message


def test_one_gate_over_parallel_implementations_clears_gate_002(good_plan, team):
    p = edit(
        good_plan,
        add_nodes=[
            SECOND_IMPL,
            {"id": "review_api", "kind": "review", "assignee": "code_reviewer"},
        ],
        add_edges=[
            ["write_tests", "implement_api"],
            ["implement_api", "gates", "verifies"],
            ["implement_api", "review_api", "verifies"],
            ["implement_api", "approve"],
        ],
    )
    assert check(p, team) == []


def test_dangling_edge_short_circuits(good_plan, team):
    errors = check(edit(good_plan, add_edges=[["ghost", "approve"]]), team)
    assert codes(errors) == ["STR-002"]


def test_unknown_agent_is_str_003(good_plan, team):
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "review_transfer").assignee = "nobody"
    assert "STR-003" in codes(check(p, team))


def test_two_approvals_is_apr_001(good_plan, team):
    p = edit(good_plan, add_nodes=[{"id": "approve2", "kind": "approval"}])
    assert "APR-001" in codes(check(p, team))


def test_unreachable_approval_is_apr_001(good_plan, team):
    p = edit(good_plan, drop_edges=[("implement_transfer", "approve")])
    errors = [e for e in check(p, team) if e.code == "APR-001"]
    assert {e.node for e in errors} == {"write_tests", "implement_transfer"}


def test_verifier_with_outgoing_edge_is_str_005(good_plan, team):
    p = edit(good_plan, add_edges=[["review_transfer", "approve"]])
    errors = [e for e in check(p, team) if e.code == "STR-005"]
    assert errors and errors[0].node == "review_transfer"


def test_payments_without_specialist_escalates(good_plan, team):
    p = good_plan.model_copy(deep=True)
    impl = next(n for n in p.nodes if n.id == "implement_transfer")
    impl.assignee, impl.paths = "payments_developer", ["minibank/payments/transfers.py"]
    errors = [e for e in check(p, team) if e.code == "PAY-001"]
    assert errors and errors[0].escalate and "security_review" in errors[0].message


def test_payments_with_specialist_needs_node_then_clears(good_plan, team_with_security):
    p = good_plan.model_copy(deep=True)
    impl = next(n for n in p.nodes if n.id == "implement_transfer")
    impl.assignee, impl.paths = "payments_developer", ["minibank/payments/transfers.py"]
    errors = check(p, team_with_security)
    assert codes(errors) == ["PAY-001"] and not errors[0].escalate
    assert "security_reviewer" in errors[0].hint

    fixed = edit(
        p,
        add_nodes=[
            {"id": "security_check", "kind": "security_review", "assignee": "security_reviewer"}
        ],
        add_edges=[["implement_transfer", "security_check", "verifies"]],
    )
    assert check(fixed, team_with_security) == []


def test_http_criterion_needs_security_review_then_clears(good_plan, team_with_security):
    p = good_plan.model_copy(deep=True)
    p.acceptance_criteria["AC-1"] = "GET /api/breaks returns HTTP 200 with JSON"
    errors = check(p, team_with_security)
    assert codes(errors) == ["SEC-001"] and "AC-1" in errors[0].message
    assert "security_reviewer" in errors[0].hint

    fixed = edit(
        p,
        add_nodes=[
            {"id": "security_check", "kind": "security_review", "assignee": "security_reviewer"}
        ],
        add_edges=[["implement_transfer", "security_check", "verifies"]],
    )
    assert check(fixed, team_with_security) == []


def test_http_criterion_without_specialist_escalates(good_plan, team):
    p = good_plan.model_copy(deep=True)
    p.acceptance_criteria["AC-1"] = "an HTTP trigger returns the breaks"
    errors = check(p, team)
    assert codes(errors) == ["SEC-001"] and errors[0].escalate


def test_payments_work_given_to_developer_is_scope_001(good_plan, team):
    """The dodge from the first live spike: move payments work to an agent without review."""
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "implement_transfer").paths = [
        "minibank/payments/refunds.py"
    ]
    errors = check(p, team)
    assert codes(errors) == ["PAY-001", "SCOPE-001"]  # declared payments path: review needed too
    scope = next(e for e in errors if e.code == "SCOPE-001")
    assert "developer cannot write minibank/payments/refunds.py" in scope.message
    assert "payments_developer" in scope.hint


def test_writing_node_without_paths_is_scope_001(good_plan, team):
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "write_tests").paths = []
    errors = check(p, team)
    assert codes(errors) == ["SCOPE-001"] and "tests/**" in errors[0].hint


def test_only_writing_nodes_take_paths(good_plan, team):
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "review_transfer").paths = ["minibank/**"]
    assert codes(check(p, team)) == ["SCOPE-001"]


def test_limits_str_004(good_plan, team):
    errors = check(good_plan, team, limits=ProjectLimits(max_children=3))
    assert "STR-004" in codes(errors)


def test_bad_example_reports_everything(bad_plan, team):
    assert set(codes(check(bad_plan, team))) >= {"SOD-001", "COV-001", "PAY-001"}


def test_cycle_is_reported_in_edge_order():
    nodes = [{"id": i, "kind": "implement", "assignee": "developer"} for i in "abcd"]
    edges = [["d", "a"], ["a", "b"], ["b", "c"], ["c", "a"]]
    plan = Plan.model_validate({"objective": "x", "nodes": nodes, "edges": edges})
    cycle = PlanGraph(plan).find_cycle()
    assert cycle is not None and cycle[0] == cycle[-1]
    pairs = set(itertools.pairwise(cycle))
    assert pairs == {("a", "b"), ("b", "c"), ("c", "a")}


def test_iface001_code_needs_an_interface_first(team, good_plan):
    p = good_plan.model_copy(update={"interface": ""})
    errors = check(p, team)
    assert [e.code for e in errors] == ["IFACE-001"] and "exact names" in errors[0].hint
    assert check(good_plan, team) == []
