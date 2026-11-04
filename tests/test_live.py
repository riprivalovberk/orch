"""Live runs on muse-spark: the real model plans, writes, reviews. Costs a few cents.

    ORCH_LIVE=1 uv run pytest tests/test_live.py -s

Skipped otherwise. Model output varies, so these check outcomes (status, merges, the policy
held), not exact plans. Each test prints a short report for the run.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from orch.core.orchestrator import TaskOrchestrator
from orch.core.team import TeamRegistry
from orch.domain.enums import TaskStatus
from orch.domain.schemas import AgentConfig, ProjectLimits
from orch.runtime import scripted
from orch.runtime.model import DEFAULT_MODEL, ModelEnv, meta_model
from orch.runtime.runner import Runner, RunReport
from orch.runtime.workspace import Workspace

from .conftest import minibank_repo

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("ORCH_LIVE") != "1" or ModelEnv().model_api_key is None,
        reason="live model run: set ORCH_LIVE=1 and MODEL_API_KEY",
    ),
]

BUDGET = 0.50

PAYMENTS_OBJECTIVE = (
    "Add refunds to the card processor in minibank/payments/cards.py: "
    "CardProcessor.refund(capture_id, amount) refunds part or all of an earlier capture."
)
PAYMENTS_CRITERIA = {
    "AC-1": "refund(capture_id, amount) with a Decimal amount up to the captured amount "
    "records a refund and returns it",
    "AC-2": "Refunding more than what is left of the capture raises ValueError",
    "AC-3": "Refunding an unknown capture_id raises KeyError",
}
SECURITY_REVIEWER = AgentConfig(
    name="security_reviewer",
    description="Reviews payment code for PCI and PII exposure, idempotency and "
    "authorization gaps.",
    capabilities=["security_review"],
    tools=["read_file", "list_files", "search_code", "submit_review"],
    prompt="Review the diff for card or account data exposure, missing idempotency, and "
    "authorization gaps. Cite the criterion or the policy for every blocker.",
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace.create(minibank_repo(tmp_path / "repo"), tmp_path / "wt")


def live_run(
    orch: TaskOrchestrator,
    team: TeamRegistry,
    ws: Workspace,
    objective: str,
    criteria: dict[str, str],
) -> tuple[str, RunReport, list[dict[str, Any]]]:
    pid = orch.create_project("minibank", limits=ProjectLimits(budget_usd=BUDGET))
    orch.register_team(pid, team)
    root = orch.create_root_task(pid, objective, acceptance_criteria=criteria)
    asked: list[dict[str, Any]] = []

    def ask(req: dict[str, Any]) -> str | None:
        asked.append(req)
        return "approve" if req["kind"] == "approval" else None  # anything else: stop

    runner = Runner(orch, team, ws, model=meta_model(DEFAULT_MODEL), parallel=3)
    rep = asyncio.run(runner.run(root, ask=ask))
    report(orch, root, rep, asked)
    return root, rep, asked


def report(orch: TaskOrchestrator, root: str, rep: RunReport, asked: list[dict[str, Any]]):
    ev = orch.evidence(root)
    print(f"\n=== {ev['objective'][:70]}")
    print(f"status {rep.status}  cost ${ev['cost_usd']:.4f}  {rep.note}")
    for p in ev["plans"]:
        codes = sorted({e.split(":")[0] for e in p["errors"]})
        print(f"  plan v{p['version']}: {'accepted' if p['valid'] else 'rejected'} {codes or ''}")
    for t in ev["tasks"][1:]:
        print(
            f"  {t['node']:<22} {t['kind']:<16} {t['assignee'] or '-':<20} "
            f"{t['status']:<15} rounds {t['revision_round']}"
        )
    for g in ev["gates"]:
        print(f"  gate {g['gate']:<7} round {g['round']} passed={g['passed']}")
    for r in ev["reviews"]:
        print(f"  review by {r['reviewer']} round {r['round']}: {r['verdict']}")
    for a in ev["artifacts"]:
        print(f"  artifact {a['name']} ({a['kind']}) from {a['task']}")
    for q in asked:
        print(f"  asked a human ({q['kind']}): {q['prompt'][:100]}")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()


def repo_tests_pass(repo: Path) -> bool:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    return subprocess.run(cmd, cwd=repo, capture_output=True).returncode == 0


def test_live_transfers_end_to_end(orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace):
    root, rep, asked = live_run(
        orch, team, ws, scripted.TRANSFERS_OBJECTIVE, scripted.TRANSFERS_CRITERIA
    )
    assert rep.status == TaskStatus.DONE, rep
    assert rep.merged_into_main
    assert [q["kind"] for q in asked] == ["approval"]
    assert (ws.repo / "minibank" / "api" / "transfers.py").exists()
    assert repo_tests_pass(ws.repo)
    ev = orch.evidence(root)
    assert orch.verify_events(ev["project_id"]).ok
    assert ev["cost_usd"] < BUDGET
    assert any(a["name"] == "test-plan.md" for a in ev["artifacts"]), "no test plan artifact"


def test_live_payments_without_security_reviewer_escalates(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace
):
    root, rep, asked = live_run(orch, team, ws, PAYMENTS_OBJECTIVE, PAYMENTS_CRITERIA)
    ev = orch.evidence(root)
    assert rep.status != TaskStatus.DONE, rep
    # The PM either submits and is rejected, or sees PAY-001 in validate_plan and asks first.
    rejected = any("PAY-001" in e for p in ev["plans"] for e in p["errors"])
    assert rejected or any("PAY-001" in q["prompt"] for q in asked), (ev["plans"], asked)
    assert asked and asked[0]["kind"] != "approval", asked
    # The policy held: nothing touched payments code.
    assert not any(t["kind"] == "implement" and t["status"] == "DONE" for t in ev["tasks"])
    first = git(ws.repo, "rev-list", "--max-parents=0", "main")
    assert git(ws.repo, "diff", "--stat", first, "main", "--", "minibank/payments") == ""


def test_live_payments_with_security_reviewer_runs_to_main(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace
):
    cfg = team.config.model_copy(deep=True)
    cfg.agents.append(SECURITY_REVIEWER)
    secured = TeamRegistry(cfg)
    root, rep, _ = live_run(orch, secured, ws, PAYMENTS_OBJECTIVE, PAYMENTS_CRITERIA)
    assert rep.status == TaskStatus.DONE, rep
    ev = orch.evidence(root)
    sec = [t for t in ev["tasks"] if t["kind"] == "security_review"]
    assert sec and all(t["status"] == "DONE" for t in sec)
    assert any(r["reviewer"] == "security_reviewer" for r in ev["reviews"])
    impl = [t for t in ev["tasks"] if t["kind"] == "implement"]
    assert impl and all(t["assignee"] == "payments_developer" for t in impl)
    assert "def refund" in (ws.repo / "minibank" / "payments" / "cards.py").read_text()
    assert repo_tests_pass(ws.repo)
