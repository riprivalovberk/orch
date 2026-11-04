"""Steps 2-6 of the runtime: PAR-001, the git workspace, gates, worker tools, the runner."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from typer.testing import CliRunner

from orch.cli import app
from orch.core.orchestrator import TaskOrchestrator
from orch.core.plan import PlanValidator, ValidationContext
from orch.core.team import TeamRegistry
from orch.domain.enums import TaskStatus
from orch.domain.schemas import AgentConfig, Plan
from orch.mcp.server import AgentMCP
from orch.policies.fintech_sdlc import PROTECTED_DIRS
from orch.runtime import gates, scripted
from orch.runtime.runner import Runner, merge_into_main
from orch.runtime.workspace import Workspace, git

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples" / "minibank"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace.create(tmp_path / "repo", tmp_path / "wt", template=TEMPLATE)


def start(orch: TaskOrchestrator, team: TeamRegistry) -> str:
    pid = orch.create_project("minibank")
    orch.register_team(pid, team)
    return orch.create_root_task(
        pid, scripted.TRANSFERS_OBJECTIVE, acceptance_criteria=scripted.TRANSFERS_CRITERIA
    )


# ------------------------------------------------------------------------- PAR-001


def test_par001_parallel_writers_must_not_share_files(team: TeamRegistry, good_plan: Plan):
    ctx = ValidationContext.from_team(team)
    v = PlanValidator.for_pack("fintech_sdlc")
    assert v.validate(good_plan, ctx) == []  # write_tests blocks implement: ordered, fine
    spec = good_plan.model_dump()
    spec["edges"] = [e for e in spec["edges"] if e["source"] != "write_tests"]
    spec["nodes"][0]["paths"] = ["minibank/api/transfers.py", "tests/test_transfers.py"]
    spec["nodes"][0]["assignee"] = "developer"
    spec["nodes"][0]["kind"] = "implement"
    codes = {e.code for e in v.validate(Plan.model_validate(spec), ctx)}
    assert "PAR-001" in codes


# ------------------------------------------------------------------------- workspace


def test_task_worktrees_merge_into_the_objective(ws: Workspace):
    a, b = ws.open_task("root", "t1"), ws.open_task("root", "t2")
    (a / "minibank" / "a.py").write_text("A = 1\n")
    (b / "minibank" / "b.py").write_text("B = 2\n")
    assert ws.changed_files("t1") == ["minibank/a.py"]
    assert ws.commit("t1", "a", author="developer") and ws.commit("t2", "b", author="developer")
    assert ws.commit("t1", "nothing new", author="developer") is None
    assert ws.merge_task("root", "t1").ok and ws.merge_task("root", "t2").ok
    assert ws.is_merged("root", "t1") and ws.is_merged("root", "t2")
    obj = ws.objective_dir("root")
    assert (obj / "minibank" / "a.py").exists() and (obj / "minibank" / "b.py").exists()
    assert "A = 1" in ws.diff("t1")


def test_conflicting_merge_is_aborted(ws: Workspace):
    for tid, text in [("t1", "x = 1\n"), ("t2", "x = 2\n")]:
        d = ws.open_task("root", tid)
        (d / "minibank" / "same.py").write_text(text)
        ws.commit(tid, tid, author="developer")
    assert ws.merge_task("root", "t1").ok
    m = ws.merge_task("root", "t2")
    assert not m.ok and "conflict" in m.detail.lower()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ws.objective_dir("root"),
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""  # aborted cleanly


def test_main_moving_reruns_the_tests_before_merging(ws: Workspace):
    d = ws.open_task("root", "t1")
    (d / "minibank" / "a.py").write_text("A = 1\n")
    ws.commit("t1", "a", author="developer")
    assert ws.merge_task("root", "t1").ok
    ran: list[Path] = []
    passing = lambda cwd: (ran.append(cwd), (True, "", 1))[1]  # noqa: E731
    # another objective merged first: main moved after this one was built
    (ws.repo / "minibank" / "b.py").write_text("B = 2\n")
    git(ws.repo, "add", "-A")
    git(ws.repo, "commit", "-q", "-m", "another objective")
    before = ws.head(ws.repo)

    kind, detail = merge_into_main(ws, "root", tests=lambda cwd: (False, "2 failed", 1))
    assert (kind, detail) == ("workspace.merge_blocked", "2 failed")
    assert ws.head(ws.repo) == before  # main untouched
    assert not (ws.objective_dir("root") / "minibank" / "b.py").exists()  # branch as it was

    kind, _ = merge_into_main(ws, "root", tests=passing)
    assert kind == "workspace.merged_to_main" and ran == [ws.objective_dir("root")]
    assert (ws.repo / "minibank" / "a.py").exists() and (ws.repo / "minibank" / "b.py").exists()


def test_merge_skips_the_rerun_when_main_did_not_move(ws: Workspace):
    d = ws.open_task("root", "t1")
    (d / "minibank" / "a.py").write_text("A = 1\n")
    ws.commit("t1", "a", author="developer")
    ws.merge_task("root", "t1")
    kind, _ = merge_into_main(ws, "root", tests=lambda cwd: pytest.fail("no rerun needed"))
    assert kind == "workspace.merged_to_main"


# ------------------------------------------------------------------------- gates


def test_policy_gate(tmp_path: Path):
    (tmp_path / "ok.py").write_text("x = 1\n")
    (tmp_path / "leak.py").write_text('password = "hunter22"\n')
    common: dict[str, Any] = {
        "agent_paths": ["**"],
        "task_paths": ["ok.py", "leak.py"],
        "protected": frozenset({"payments"}),
    }
    assert gates.policy(tmp_path, ["ok.py"], **common)[0]
    ok, out, _ = gates.policy(tmp_path, ["leak.py"], **common)
    assert not ok and "hard-coded password" in out
    ok, out, _ = gates.policy(tmp_path, ["other.py"], **common)
    assert not ok and "outside the task's paths" in out
    assert not gates.policy(tmp_path, [], **common)[0]


def test_pytest_and_ruff_gates_on_the_template(ws: Workspace):
    assert gates.pytest(ws.repo)[0]
    assert gates.ruff(ws.repo)[0]
    (ws.repo / "tests" / "test_broken.py").write_text("def test_x():\n    assert False\n")
    assert not gates.pytest(ws.repo)[0]


# ------------------------------------------------------------------------- worker tools


def worker(orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan):
    root = start(orch, team)
    claim = orch.claim(root)
    assert claim is not None
    plan = good_plan.model_dump(mode="json")
    tasks = orch.submit_plan(root, Plan.model_validate(plan), run_id=claim.run_id).task_ids
    c = orch.claim(tasks["write_tests"])
    assert c is not None
    ws.open_task(root, c.task_id)
    srv = AgentMCP(orch, ws)
    return srv, run(srv.caller(c.run_token)), tasks


# A test file naming a test for each criterion the good plan's test node covers.
COVERED = "\n\n".join(f"def test_ac{i}_holds():\n    pass\n" for i in range(1, 5))


def test_write_file_is_held_to_the_task_scope(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    ok = run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": COVERED}))
    assert not ok.is_error
    wrong = run(srv.call(c, "write_file", {"path": "tests/test_other.py", "content": ""}))
    assert wrong.is_error and "outside this task's scope" in wrong.content[0].text  # type: ignore[union-attr]
    escape = run(srv.call(c, "read_file", {"path": "../../etc/passwd"}))
    assert escape.is_error
    listed = run(srv.call(c, "list_files", {"glob": "tests/*"}))
    assert "tests/test_transfers.py" in listed.content[0].text  # type: ignore[union-attr]
    done = run(srv.call(c, "complete_task", {"summary": "tests"}))
    assert json.loads(done.content[0].text)["commit"]  # type: ignore[union-attr]
    assert orch.get_task(c.task_id)["status"] == TaskStatus.DONE


# ------------------------------------------------------------------------- runner


def runner(orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, model: Any) -> Runner:
    return Runner(orch, team, ws, model=model, parallel=3)


def test_scripted_objective_runs_to_main(orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace):
    root = start(orch, team)
    asked: list[str] = []
    rep = run(
        runner(orch, team, ws, scripted.scripted_team()).run(
            root, ask=lambda r: asked.append(r["kind"]) or "approve"
        )
    )
    assert rep.status == TaskStatus.DONE and rep.merged_into_main, rep
    assert asked == ["approval"]
    assert (ws.repo / "minibank" / "api" / "transfers.py").exists()
    ev = orch.evidence(root)
    assert {g["gate"] for g in ev["gates"] if g["passed"]} == {"pytest", "ruff", "policy"}
    assert orch.verify_events(ev["project_id"]).ok


def test_without_a_human_the_run_stops_at_approval(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace
):
    root = start(orch, team)
    rep = run(runner(orch, team, ws, scripted.scripted_team()).run(root, ask=None))
    assert rep.status == TaskStatus.WAITING and rep.waiting_on[0]["kind"] == "approval"
    assert not (ws.repo / "minibank" / "api" / "transfers.py").exists()  # main untouched
    orch.resolve_human(rep.waiting_on[0]["id"], "approve")
    rep = run(runner(orch, team, ws, scripted.scripted_team()).run(root, ask=None))
    assert rep.status == TaskStatus.DONE and rep.merged_into_main


def revising_team(first_impl: str) -> FunctionModel:
    """Scripted team whose developer's first try is `first_impl` and whose reviewer asks for
    changes once; the second round uses the real implementation."""
    base = scripted.scripted_team()
    reviews = {"n": 0}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools = {t.name for t in info.function_tools}
        prompt = scripted._prompt(messages)
        done = scripted._returned(messages)
        if "submit_review" in tools and not done:
            reviews["n"] += 1
            if reviews["n"] == 1:
                return ModelResponse(
                    [
                        ToolCallPart(
                            "submit_review",
                            {
                                "verdict": "request_changes",
                                "summary": "Amounts are parsed as float.",
                                "findings": [
                                    {
                                        "severity": "blocker",
                                        "issue": "float amounts",
                                        "criterion_id": "AC-2",
                                    }
                                ],
                            },
                        )
                    ]
                )
        if "(implement)" in prompt and "revision round" not in prompt and not done:
            return ModelResponse(
                [
                    ToolCallPart(
                        "write_file", {"path": "minibank/api/transfers.py", "content": first_impl}
                    )
                ]
            )
        if "(implement)" in prompt and "revision round" in prompt:
            assert "float amounts" in prompt  # the finding reached the developer
        return base.function(messages, info)  # type: ignore[misc]

    return FunctionModel(respond, model_name="revising-team")


def test_review_sends_work_back_and_the_revision_passes(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace
):
    root = start(orch, team)
    rep = run(
        runner(orch, team, ws, revising_team(scripted.TRANSFERS + "# first draft\n")).run(
            root, ask=lambda r: "approve"
        )
    )
    assert rep.status == TaskStatus.DONE, rep
    impl = next(t for t in orch.task_tree(root) if t["node"] == "implement_transfer")
    assert impl["revision_round"] == 1
    verdicts = [(r["round"], r["verdict"]) for r in orch.evidence(root)["reviews"]]
    assert verdicts == [(0, "request_changes"), (1, "approve")]


def test_failing_gate_sends_work_back(orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace):
    root = start(orch, team)
    broken = scripted.TRANSFERS.replace('return error(422, "insufficient funds")', "pass")
    rep = run(runner(orch, team, ws, revising_team(broken)).run(root, ask=lambda r: "approve"))
    assert rep.status == TaskStatus.DONE, rep
    gate_rounds = [(g["round"], g["gate"], g["passed"]) for g in orch.evidence(root)["gates"]]
    assert (0, "pytest", False) in gate_rounds and (1, "pytest", True) in gate_rounds


def test_cli_offline_run_approve_resume_evidence(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path / "data"))
    cli = CliRunner()
    r = cli.invoke(app, ["run", "--offline", "--no-interactive"])
    assert r.exit_code == 0, r.output
    assert "waiting on hum_" in r.output
    inbox = cli.invoke(app, ["inbox"]).output
    req = inbox.split()[0]
    root = inbox.split("objective ")[1].split(")")[0]
    assert cli.invoke(app, ["approve", req]).exit_code == 0
    r = cli.invoke(app, ["run", "--resume", root, "--no-interactive"])
    assert "DONE" in r.output and "merged into main" in r.output, r.output
    r = cli.invoke(app, ["evidence", root])
    assert r.exit_code == 0
    md = (tmp_path / "data" / "evidence" / f"{root}.md").read_text()
    assert "hash chain **intact**" in md and "| implement_transfer | 0 | pytest | True |" in md


def test_edit_file_replaces_one_exact_match_within_scope(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    path = "tests/test_transfers.py"
    run(srv.call(c, "write_file", {"path": path, "content": "A = 1\nB = 1\n"}))

    def edit(old: str, new: str, file: str = path) -> Any:
        return run(srv.call(c, "edit_file", {"path": file, "old_text": old, "new_text": new}))

    assert not edit("A = 1\n", "A = 2\n").is_error
    assert (ws.task_dir(c.task_id) / path).read_text() == "A = 2\nB = 1\n"
    twice = edit(" = ", " == ")
    assert twice.is_error and "occurs 2 times" in twice.content[0].text  # type: ignore[union-attr]
    assert edit("C = 1", "").is_error  # not in the file
    assert edit("x", "y", "minibank/api/transfers.py").is_error  # outside the task's paths
    assert (ws.task_dir(c.task_id) / path).read_text() == "A = 2\nB = 1\n"


def test_edit_file_forgives_indentation_and_line_numbers(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    path = "tests/test_transfers.py"
    src = "def f(x):\n    if x:\n        return 1\n    return 2\n"
    run(srv.call(c, "write_file", {"path": path, "content": src}))

    def edit(old: str, new: str) -> Any:
        return run(srv.call(c, "edit_file", {"path": path, "old_text": old, "new_text": new}))

    # Copied from read_file with its line numbers and two spaces too many of indentation.
    r = edit("    2        if x:\n    3            return 1", "      if x > 0:\n          return 3")
    assert not r.is_error, r
    assert (ws.task_dir(c.task_id) / path).read_text() == (
        "def f(x):\n    if x > 0:\n        return 3\n    return 2\n"
    )
    miss = edit("if y:", "")
    assert miss.is_error and "closest line is 2" in miss.content[0].text  # type: ignore[union-attr]


def test_complete_task_refuses_lint_errors_and_empty_revisions(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    bad = "import os\n"  # unused import: ruff F401
    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": bad}))
    r = run(srv.call(c, "complete_task", {"summary": "tests"}))
    assert r.is_error and "ruff" in r.content[0].text  # type: ignore[union-attr]
    assert orch.get_task(c.task_id)["status"] == TaskStatus.RUNNING


def test_complete_task_formats_and_sorts_imports_itself(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    messy = "import sys\nimport os\nPATHS=[os.sep,sys.argv[0]]\n\n\n" + COVERED
    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": messy}))
    r = run(srv.call(c, "complete_task", {"summary": "tests"}))
    assert not r.is_error, r.content[0].text  # type: ignore[union-attr]
    assert "tests/test_transfers.py" in r.content[0].text  # type: ignore[union-attr]
    tidy = (ws.task_dir(c.task_id) / "tests/test_transfers.py").read_text()
    assert tidy == "import os\nimport sys\n\nPATHS = [os.sep, sys.argv[0]]\n\n\n" + COVERED
    assert orch.get_task(c.task_id)["status"] != TaskStatus.RUNNING


def test_complete_task_removes_files_outside_the_task_paths(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    # positions-rec-6: code run at test time copied deploy files to the repo root
    srv, c, _ = worker(orch, team, ws, good_plan)
    d = ws.task_dir(c.task_id)
    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": COVERED}))
    readme = (d / "README.md").read_text()
    (d / "host.json").write_text("{}\n")
    (d / "README.md").write_text("changed\n")
    ws.commit(c.task_id, "earlier round", author="developer")
    r = run(srv.call(c, "complete_task", {"summary": "tests"}))
    text = r.content[0].text  # type: ignore[union-attr]
    assert r.is_error and "README.md, host.json are outside" in text
    assert not (d / "host.json").exists() and (d / "README.md").read_text() == readme
    assert (d / "tests/test_transfers.py").exists()
    done = run(srv.call(c, "complete_task", {"summary": "tests"}))
    assert not done.is_error and ws.changed_files(c.task_id) == ["tests/test_transfers.py"]


def test_complete_task_wants_a_test_named_for_each_covered_criterion(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    three = COVERED.replace("test_ac4_holds", "test_amounts_hold")
    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": three}))
    r = run(srv.call(c, "complete_task", {"summary": "tests"}))
    assert r.is_error and "covers AC-4, but no test is named" in r.content[0].text  # type: ignore[union-attr]
    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": COVERED}))
    assert not run(srv.call(c, "complete_task", {"summary": "tests"})).is_error


def test_policy_gate_fails_network_code_without_a_security_review(tmp_path: Path):
    (tmp_path / "app.py").write_text("import azure.functions as func\n\napp = func.FunctionApp()\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_app.py").write_text("@client.get('/x')\ndef test_x(): ...\n")
    changed, scope = ["app.py", "tests/test_app.py"], {"agent_paths": ["**", "tests/**"]}
    args = {**scope, "task_paths": ["**", "tests/**"], "protected": PROTECTED_DIRS}
    ok, out, _ = gates.policy(tmp_path, changed, **args)
    assert not ok and "app.py:3: serves the network (FunctionApp(" in out
    assert "tests/test_app.py" not in out  # tests may call routes
    assert gates.policy(tmp_path, changed, **args, security_reviewed=True)[0]


def test_revision_without_changes_is_refused(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan, monkeypatch
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": COVERED}))
    ws.commit(c.task_id, "round 0", author="developer")
    real = orch.get_task
    monkeypatch.setattr(orch, "get_task", lambda tid: {**real(tid), "revision_round": 1})
    r = run(srv.call(c, "complete_task", {"summary": "no changes"}))
    assert r.is_error and "changed nothing" in r.content[0].text  # type: ignore[union-attr]


def test_submit_artifact_stores_by_hash_and_shows_in_evidence(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, _ = worker(orch, team, ws, good_plan)
    note = {"name": "test-plan.md", "kind": "notes", "content": "# AC-1 covered\n"}
    r = run(srv.call(c, "submit_artifact", note))
    assert not r.is_error, r
    sha = json.loads(r.content[0].text)["sha256"]  # type: ignore[union-attr]
    assert (ws.artifacts / sha[:2] / sha).read_text() == "# AC-1 covered\n"

    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": "X = 1\n"}))
    copy = {"name": "tests.py", "kind": "other", "path": "tests/test_transfers.py"}
    assert not run(srv.call(c, "submit_artifact", copy)).is_error
    assert ws.changed_files(c.task_id) == ["tests/test_transfers.py"]  # repo untouched

    both = {"name": "x.md", "kind": "notes", "content": "a", "path": "tests/test_transfers.py"}
    assert run(srv.call(c, "submit_artifact", both)).is_error
    assert run(srv.call(c, "submit_artifact", {"name": "x.md", "kind": "notes"})).is_error
    bad_name = {"name": "../x.md", "kind": "notes", "content": "a"}
    assert run(srv.call(c, "submit_artifact", bad_name)).is_error

    root = orch.get_task(c.task_id)["root_id"]
    arts = orch.evidence(root)["artifacts"]
    assert [(a["task"], a["name"]) for a in arts] == [
        ("write_tests", "test-plan.md"),
        ("write_tests", "tests.py"),
    ]


def test_cli_resume_uses_the_objectives_repo_and_budget_updates(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path / "data"))
    cli = CliRunner()
    repo = tmp_path / "bank"
    r = cli.invoke(app, ["run", "--offline", "--no-interactive", "--repo", str(repo)])
    assert r.exit_code == 0 and "budget $2.00" in r.output, r.output
    inbox = cli.invoke(app, ["inbox"]).output
    req, root = inbox.split()[0], inbox.split("objective ")[1].split(")")[0]
    assert cli.invoke(app, ["approve", req]).exit_code == 0

    other = cli.invoke(app, ["run", "--resume", root, "--repo", str(tmp_path / "x")])
    assert other.exit_code != 0 and "runs in" in other.output, other.output
    r = cli.invoke(app, ["run", "--resume", root, "--no-interactive", "--budget", "5"])
    assert f"merged into main of {repo.resolve()}" in r.output, r.output
    assert "budget $5.00" in r.output
    assert (repo / "minibank" / "api" / "transfers.py").exists()
    assert not (tmp_path / "data" / "repos" / "minibank").exists()

    assert cli.invoke(app, ["run", "--offline", "--parallel", "0"]).exit_code != 0


def test_pm_reads_the_repo_and_is_asked_for_criteria_when_none_given(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace
):
    pid = orch.create_project("minibank")
    orch.register_team(pid, team)
    root = orch.create_root_task(pid, "Reconcile two position files")  # no criteria
    claim = orch.claim(root)
    assert claim is not None
    ws.start_objective(root)
    srv = AgentMCP(orch, ws)
    c = run(srv.caller(claim.run_token))
    listed = run(srv.call(c, "list_files", {"glob": "minibank/*.py"}))
    assert "minibank/ledger.py" in listed.content[0].text  # type: ignore[union-attr]
    assert run(srv.call(c, "read_file", {"path": "minibank/ledger.py"})).is_error is False
    assert run(srv.call(c, "write_file", {"path": "x.py", "content": ""})).is_error

    brief = runner(orch, team, ws, scripted.scripted_team()).brief(orch.get_task(root))
    assert "No acceptance criteria were given" in brief


def test_briefs_carry_the_interface_and_the_standards(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    _, c, tasks = worker(orch, team, ws, good_plan)
    r = runner(orch, team, ws, scripted.scripted_team())
    tests_brief = r.brief(orch.get_task(tasks["write_tests"]))
    assert "class Transfers(ledger: Ledger)" in tests_brief
    assert "STD-001" not in tests_brief  # standards go to the developer and reviewers
    impl_brief = r.brief(orch.get_task(tasks["implement_transfer"]))
    assert "class Transfers(ledger: Ledger)" in impl_brief and "STD-003" in impl_brief


def test_a_retried_task_carries_on_from_what_it_wrote(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    srv, c, tasks = worker(orch, team, ws, good_plan)
    r = runner(orch, team, ws, scripted.scripted_team())
    tid = tasks["write_tests"]
    assert "previous run" not in r.brief(orch.get_task(tid))
    run(srv.call(c, "write_file", {"path": "tests/test_transfers.py", "content": COVERED}))
    orch.retry(tid, "TimeoutError")
    assert orch.claim(tid) is not None
    brief = r.brief(orch.get_task(tid))
    assert "previous run" in brief and "tests/test_transfers.py" in brief


def test_briefs_name_the_work_other_steps_build(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    plan = good_plan.model_dump(mode="json")
    plan["nodes"] += [
        {
            "id": "implement_api",
            "kind": "implement",
            "title": "HTTP endpoint",
            "assignee": "developer",
            "paths": ["minibank/api.py"],
        },
        {"id": "review_api", "kind": "review", "assignee": "code_reviewer"},
    ]
    plan["edges"] += [
        ["write_tests", "implement_api", "blocks"],
        ["implement_api", "gates", "verifies"],
        ["implement_api", "review_api", "verifies"],
        ["implement_api", "approve", "blocks"],
    ]
    _, _, tasks = worker(orch, team, ws, Plan.model_validate(plan))
    r = runner(orch, team, ws, scripted.scripted_team())
    impl_brief = r.brief(orch.get_task(tasks["implement_transfer"]))
    assert "implement_api: HTTP endpoint (writes minibank/api.py)" in impl_brief
    # a reviewer's own step is the work it checks, so only the other one is listed
    assert r._verified_nodes(orch.get_task(tasks["review_transfer"])) == ["implement_transfer"]
    tests_brief = r.brief(orch.get_task(tasks["write_tests"]))
    assert "Other steps" not in tests_brief
    # the test engineer learns where the code lands, so tests don't look elsewhere
    assert "implement_api: writes minibank/api.py" in tests_brief


def test_each_role_sets_its_reasoning_effort(team_with_security: TeamRegistry):
    assert team_with_security.get("developer").effort == "low"
    assert team_with_security.get("code_reviewer").effort == "high"
    bad = team_with_security.get("developer").model_dump() | {"effort": "max"}
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(bad)


def test_writing_briefs_show_csv_heads_and_implement_briefs_show_tests(
    orch: TaskOrchestrator, team: TeamRegistry, ws: Workspace, good_plan: Plan
):
    _, c, tasks = worker(orch, team, ws, good_plan)
    impl = orch.get_task(tasks["implement_transfer"])
    ws.open_task(impl["root_id"], impl["id"])
    for tid in (c.task_id, impl["id"]):
        d = ws.task_dir(tid)
        (d / "data").mkdir(exist_ok=True)
        (d / "data" / "pos.csv").write_text("id,qty\n" + "".join(f"{i},1\n" for i in range(9)))
        (d / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    r = runner(orch, team, ws, scripted.scripted_team())
    impl_brief = r.brief(orch.get_task(impl["id"]))
    assert "data/pos.csv (first lines):\nid,qty\n0,1\n1,1\n2,1" in impl_brief
    assert "3,1" not in impl_brief
    assert "tests/test_x.py:\n```python\ndef test_x():" in impl_brief
    tests_brief = r.brief(orch.get_task(c.task_id))
    assert "data/pos.csv (first lines)" in tests_brief and "tests/test_x.py:" not in tests_brief
