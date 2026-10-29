from __future__ import annotations

import pytest
from sqlalchemy import text

from orch.core.orchestrator import InvalidOperation, TaskOrchestrator
from orch.domain.enums import GateName
from orch.domain.enums import TaskStatus as S
from orch.domain.schemas import Finding, GateResultIn, ProjectLimits, ReviewVerdict

ALL_GATES = [GateResultIn(gate=g, passed=True) for g in GateName]
APPROVE = ReviewVerdict(verdict="approve", summary="meets every criterion")
CHANGES = ReviewVerdict(
    verdict="request_changes",
    findings=[Finding(severity="blocker", issue="amount uses float", criterion_id="AC-2")],
)


def setup(orch: TaskOrchestrator, team, limits=None):
    pid = orch.create_project("minibank", limits=limits)
    orch.register_team(pid, team)
    root = orch.create_root_task(pid, "Add idempotent POST /transfers")
    return pid, root


def st(orch, task_id):
    return orch.get_task(task_id)["status"]


def start(orch, task_id):
    claim = orch.claim(task_id)
    assert claim is not None, f"{task_id} not claimable ({st(orch, task_id)})"
    return claim


def planned(orch, team, plan, limits=None):
    pid, root = setup(orch, team, limits)
    start(orch, root)
    result = orch.submit_plan(root, plan)
    assert result.valid, result.errors
    return pid, root, result.task_ids


def test_rejected_then_accepted_plan(orch, team, good_plan, bad_plan):
    pid, root = setup(orch, team)
    assert st(orch, root) == S.READY
    start(orch, root)
    good_only_review_missing = good_plan.model_copy(deep=True)
    good_only_review_missing.nodes = [n for n in good_plan.nodes if n.id != "review_transfer"]
    good_only_review_missing.edges = [
        e for e in good_plan.edges if "review_transfer" not in (e.source, e.target)
    ]
    r1 = orch.submit_plan(root, good_only_review_missing)
    assert not r1.valid and r1.attempts_left == 2 and not r1.escalated
    assert st(orch, root) == S.RUNNING

    r2 = orch.submit_plan(root, good_plan)
    assert r2.valid and r2.version == 2
    ids = r2.task_ids
    assert st(orch, ids["write_tests"]) == S.READY
    assert st(orch, ids["implement_transfer"]) == S.PENDING
    assert st(orch, ids["approve"]) == S.PENDING
    with pytest.raises(InvalidOperation):
        orch.submit_plan(root, good_plan)


def test_full_lifecycle_with_one_revision(orch, team, good_plan):
    pid, root, ids = planned(orch, team, good_plan)
    assert st(orch, root) == S.WAITING

    start(orch, ids["write_tests"])
    assert orch.complete(ids["write_tests"], "4 tests") == S.DONE
    impl = ids["implement_transfer"]
    assert st(orch, impl) == S.READY

    # round 0: gates pass, reviewer requests changes
    start(orch, impl)
    assert orch.complete(impl, "first cut") == S.VERIFYING
    assert st(orch, ids["gates"]) == S.READY and st(orch, ids["review_transfer"]) == S.READY
    start(orch, ids["gates"])
    orch.record_gate_results(ids["gates"], ALL_GATES)
    assert st(orch, impl) == S.VERIFYING  # still waiting for the review
    assert st(orch, ids["approve"]) == S.PENDING
    start(orch, ids["review_transfer"])
    orch.record_review(ids["review_transfer"], CHANGES)
    assert st(orch, impl) == S.READY
    assert orch.get_task(impl)["revision_round"] == 1
    assert st(orch, ids["gates"]) == S.PENDING and st(orch, ids["review_transfer"]) == S.PENDING

    # round 1: both pass
    start(orch, impl)
    orch.complete(impl, "uses Decimal now")
    start(orch, ids["gates"])
    orch.record_gate_results(ids["gates"], ALL_GATES)
    start(orch, ids["review_transfer"])
    orch.record_review(ids["review_transfer"], APPROVE)
    assert st(orch, impl) == S.DONE
    assert st(orch, ids["approve"]) == S.AWAITING_HUMAN

    [req] = orch.open_human_requests(pid)
    assert req["kind"] == "approval"
    orch.resolve_human(req["id"], "approve")
    assert st(orch, ids["approve"]) == S.DONE
    # the orchestrator closes the plan task itself once every planned task passed
    assert st(orch, root) == S.DONE
    assert orch.get_task(root)["outcome"] == "pass"
    assert orch.verify_events(pid).ok


def test_review_must_cite_a_criterion(orch, team, good_plan):
    _, _, ids = planned(orch, team, good_plan)
    start(orch, ids["write_tests"])
    orch.complete(ids["write_tests"], "ok")
    start(orch, ids["implement_transfer"])
    orch.complete(ids["implement_transfer"], "ok")
    start(orch, ids["review_transfer"])
    uncited = ReviewVerdict(
        verdict="request_changes", findings=[Finding(severity="blocker", issue="vibes")]
    )
    with pytest.raises(InvalidOperation, match="cite an acceptance criterion"):
        orch.record_review(ids["review_transfer"], uncited)


def test_revision_limit_escalates_and_override(orch, team, good_plan):
    pid, root, ids = planned(orch, team, good_plan, ProjectLimits(max_revision_rounds=1))
    impl = ids["implement_transfer"]
    start(orch, ids["write_tests"])
    orch.complete(ids["write_tests"], "ok")
    for _ in range(2):
        start(orch, impl)
        orch.complete(impl, "try")
        start(orch, ids["gates"])
        orch.record_gate_results(ids["gates"], ALL_GATES)
        start(orch, ids["review_transfer"])
        orch.record_review(ids["review_transfer"], CHANGES)
    assert st(orch, impl) == S.AWAITING_HUMAN
    [req] = orch.open_human_requests(pid)
    assert req["kind"] == "escalation"
    orch.resolve_human(req["id"], "override_approve")
    assert st(orch, impl) == S.DONE
    assert st(orch, ids["approve"]) == S.AWAITING_HUMAN  # waived review unblocks approval


def test_recovery_requeues_then_fails_and_cancels_downstream(orch, team, good_plan):
    pid, root, ids = planned(orch, team, good_plan, ProjectLimits(max_attempts=2))
    tests = ids["write_tests"]
    start(orch, tests)
    assert orch.recover() == [tests]  # runner restarted mid-task
    assert st(orch, tests) == S.READY

    start(orch, tests)
    orch.recover()
    assert st(orch, tests) == S.FAILED
    for node in ("implement_transfer", "gates", "review_transfer", "approve"):
        assert st(orch, ids[node]) == S.CANCELED, node
    assert st(orch, root) == S.FAILED  # plan task closes as failed
    assert "write_tests (FAILED)" in orch.get_task(root)["summary"]


def run_to_approval(orch, ids):
    for node in ("write_tests", "implement_transfer"):
        start(orch, ids[node])
        orch.complete(ids[node], "ok")
    start(orch, ids["gates"])
    orch.record_gate_results(ids["gates"], ALL_GATES)
    start(orch, ids["review_transfer"])
    orch.record_review(ids["review_transfer"], APPROVE)


def test_rejected_approval_fails_the_plan(orch, team, good_plan):
    pid, root, ids = planned(orch, team, good_plan)
    run_to_approval(orch, ids)
    [req] = orch.open_human_requests(pid)
    orch.resolve_human(req["id"], "reject")
    assert st(orch, root) == S.FAILED
    assert orch.get_task(root)["outcome"] == "fail"


def test_plan_task_cannot_complete_itself(orch, team):
    _, root = setup(orch, team)
    start(orch, root)
    with pytest.raises(InvalidOperation, match="submit_plan"):
        orch.complete(root, "nothing to do")


def test_retry_after_timeout_kills_the_token(orch, team):
    _, root = setup(orch, team)
    claim = start(orch, root)
    assert orch.retry(root, "timed out") == S.READY
    assert orch.resolve_run_token(claim.run_token) is None
    assert start(orch, root).attempt == 2


def test_usage_is_charged_after_the_run_ends(orch, team, good_plan):
    from orch.db.models import Project, Run

    pid, root = setup(orch, team)
    claim = start(orch, root)
    orch.submit_plan(root, good_plan)  # accepted: plan task leaves RUNNING, run closes
    orch.record_usage(claim.run_id, tokens_in=1000, tokens_out=200, cost_usd=0.25)
    with orch.tx() as s:
        run = s.get(Run, claim.run_id)
        assert run.ended_at is not None and run.end_reason.startswith("waiting on")
        assert s.get(Project, pid).spent_usd == 0.25


def test_double_claim_is_refused(orch, team):
    _, root = setup(orch, team)
    assert orch.claim(root) is not None
    assert orch.claim(root) is None


def test_run_token_resolves_only_while_running(orch, team):
    _, root = setup(orch, team)
    claim = start(orch, root)
    who = orch.resolve_run_token(claim.run_token)
    assert who and who["task_id"] == root and who["agent"] == "project_manager"
    assert orch.resolve_run_token("not-a-token") is None


def test_missing_specialist_escalates_to_human(orch, team, good_plan):
    pid, root = setup(orch, team)
    start(orch, root)
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "implement_transfer").assignee = "payments_developer"
    result = orch.submit_plan(root, p)
    assert result.escalated and not result.valid
    assert st(orch, root) == S.AWAITING_HUMAN
    [req] = orch.open_human_requests(pid)
    assert "security_review" in req["prompt"]


def test_adding_an_agent_lets_the_plan_clear(orch, team, team_with_security, good_plan):
    pid, root = setup(orch, team)
    start(orch, root)
    p = good_plan.model_copy(deep=True)
    next(n for n in p.nodes if n.id == "implement_transfer").assignee = "payments_developer"
    orch.submit_plan(root, p)
    [req] = orch.open_human_requests(pid)

    change = orch.register_team(pid, team_with_security)
    assert change["added"] == ["security_reviewer"]
    orch.resolve_human(req["id"], "retry_planning")
    start(orch, root)

    from orch.domain.schemas import PlanEdge, PlanNode

    p.nodes.append(
        PlanNode(id="security_check", kind="security_review", assignee="security_reviewer")
    )
    p.edges.append(PlanEdge(source="implement_transfer", target="security_check", kind="verifies"))
    assert orch.submit_plan(root, p).valid


def test_tampering_breaks_the_chain(orch, team, good_plan):
    pid, _, _ = planned(orch, team, good_plan)
    assert orch.verify_events(pid).ok
    with orch.tx() as s:
        s.execute(text("UPDATE events SET actor = 'mallory' WHERE seq = 3"))
    result = orch.verify_events(pid)
    assert not result.ok and result.first_bad_seq == 3


def test_complete_requires_running(orch, team):
    _, root = setup(orch, team)
    with pytest.raises(InvalidOperation):
        orch.complete(root, "nope")


def test_json_columns_round_trip_as_models(orch, team, good_plan):
    from orch.db.models import PlanRecord, Project
    from orch.domain.schemas import Plan, PlanError

    pid, root = setup(orch, team, ProjectLimits(max_children=9))
    start(orch, root)
    bad = good_plan.model_copy(deep=True)
    bad.nodes = [n for n in bad.nodes if n.id != "review_transfer"]
    bad.edges = [e for e in bad.edges if "review_transfer" not in (e.source, e.target)]
    orch.submit_plan(root, bad)
    with orch.tx() as s:
        p = s.get(Project, pid)
        assert isinstance(p.limits, ProjectLimits) and p.limits.max_children == 9
        rec = s.query(PlanRecord).one()
        assert isinstance(rec.spec, Plan) and rec.spec.nodes[0].id == "write_tests"
        assert rec.errors and isinstance(rec.errors[0], PlanError)
