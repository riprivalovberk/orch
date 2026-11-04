"""The web UI's API: projects, an offline run through approval, events, plan check."""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from orch.core.orchestrator import TaskOrchestrator
from orch.db.session import open_db
from orch.web.server import RunInfo, RunOptions, _track, create_app

from .conftest import ROOT, minibank_repo


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path))
    orch = TaskOrchestrator(open_db(tmp_path / "orch.db"))
    app = create_app(
        orch,
        team_path=ROOT / "teams" / "fintech_sdlc.yaml",
    )
    return TestClient(app)


def runs(c: TestClient):
    """The app's run registry; ``TestClient.app`` is typed as a bare ASGI app."""
    return cast(Starlette, c.app).state.runs


def wait_for_run(c: TestClient, root: str, timeout: float = 60) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        ob = c.get(f"/api/objectives/{root}").json()
        if ob["run"]["state"] != "running":
            return ob
        time.sleep(0.2)
    raise AssertionError(f"run of {root} did not stop")


def test_offline_objective_runs_through_approval_to_main(client: TestClient, tmp_path: Path):
    c = client
    assert c.get("/").status_code == 200
    bank = str(minibank_repo(tmp_path / "bank"))
    p = c.post("/api/projects", json={"name": "Mini Bank", "budget": 3, "repo": bank}).json()
    assert p["repo_path"] == bank
    assert c.post("/api/projects", json={"name": "again", "repo": bank}).status_code == 409

    r = c.post(f"/api/projects/{p['id']}/objectives", json={"offline": True}).json()
    ob = wait_for_run(c, r["root_id"])
    assert ob["run"]["report"]["note"] == "waiting on a human"
    [req] = ob["requests"]
    assert req["kind"] == "approval"
    chat = c.get(f"/api/projects/{p['id']}/chat").json()
    roles = [m["role"] for m in chat["messages"] if m["role"] != "activity"]
    assert roles == ["user", "pm", "pm"]  # objective, the plan, the approval request
    first, plan, ask = [m for m in chat["messages"] if m["role"] != "activity"]
    assert first["starts_objective"] and first["root_id"] == r["root_id"]
    assert [s["node"] for s in plan["plan"]][:2] == ["write_tests", "implement_transfer"]
    assert plan["criteria"]["AC-1"].startswith("A repeated Idempotency-Key")
    assert ask["request"] == {
        "id": req["id"],
        "kind": "approval",
        "options": req["options"],
        "open": True,
    }
    assert chat["open_requests"][0]["id"] == req["id"] and chat["running"] == []
    assert c.get("/api/inbox").status_code == 404  # no inbox: decisions are answered in the chat

    a = c.post(f"/api/requests/{req['id']}/answer", json={"response": "approve"}).json()
    assert a["resumed"]
    ob = wait_for_run(c, r["root_id"])
    assert ob["root"]["status"] == "DONE", ob["run"]
    assert ob["run"]["report"]["merged_into_main"]
    assert c.get(f"/api/projects/{p['id']}/chat").json()["open_requests"] == []
    chat = c.get(f"/api/projects/{p['id']}/chat").json()["messages"]
    assert not chat[-1]["text"].startswith("I had to stop")
    # the plan message is a live workflow: planning, then every planned task, all timed
    steps = next(m for m in chat if m.get("steps"))["steps"]
    assert [s["node"] for s in steps][:3] == ["plan", "write_tests", "implement_transfer"]
    assert all(s["state"] == "done" for s in steps)
    assert all(s["started"] and s["ended"] for s in steps)
    assert all(s["since"] is None for s in steps)
    assert next(s for s in steps if s["node"] == "approve")["busy_ms"] == 0  # your wait
    # ...and a timeline of everything that happened, in order
    tl = next(m for m in chat if m.get("steps"))["timeline"]
    assert [x["ts"] for x in tl] == sorted(x["ts"] for x in tl)
    said = [x["text"] for x in tl]
    assert any(t.startswith("plan v1 passed the checks") for t in said)
    assert "FinOrch asked you to approve the change for merge" in said
    assert "you approved the change for merge" in said and "merged into main" in said
    gate = next(s for s in steps if s["node"] == "gates")
    assert gate["note"] == "all checks passed" and gate["log"]
    assert not any("started" in m["text"] for m in chat if m["role"] == "activity")
    said = [m["text"] for m in chat if m["role"] in ("user", "pm")]
    assert "approve" in said and any(t.startswith("Merged into main") for t in said)

    events = c.get("/api/events", params={"root": r["root_id"]}).json()
    assert events and all(e["task_id"] for e in events)
    later = c.get("/api/events", params={"root": r["root_id"], "after": events[-5]["seq"]}).json()
    assert len(later) == 4
    assert c.get(f"/api/projects/{p['id']}/verify").json()["ok"]
    ev = c.get(f"/api/objectives/{r['root_id']}/evidence").json()
    assert ev["chain_ok"] and "# Change evidence" in ev["markdown"]
    impl = next(t for t in ob["tasks"] if t["kind"] == "implement")
    assert "transfers" in c.get(f"/api/tasks/{impl['id']}").json()["diff"]


def test_live_objective_without_a_key_is_refused(client: TestClient, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.chdir(ROOT / "examples")  # no .env here
    p = client.post("/api/projects", json={"name": "x"}).json()
    r = client.post(f"/api/projects/{p['id']}/objectives", json={"objective": "build it"})
    assert r.status_code == 400 and "MODEL_API_KEY" in r.json()["error"]


def test_bad_repo_paths_get_a_clear_error(client: TestClient, tmp_path: Path):
    def create(repo: str) -> dict:
        r = client.post("/api/projects", json={"name": "x", "repo": repo})
        assert r.status_code == 400, r.text
        return r.json()["error"]

    assert "not writable" in create("/positions-quantity-rec")  # / is read-only on macOS
    full = tmp_path / "notes"
    full.mkdir()
    (full / "todo.txt").write_text("keep me")
    assert "not a git repo" in create(str(full))
    assert sorted(p.name for p in full.iterdir()) == ["todo.txt"]  # template not copied in
    made = client.post("/api/projects", json={"name": "a", "repo": str(tmp_path / "a")}).json()
    assert "inside the git repo" in create(str(Path(made["repo_path"]) / "tests"))
    assert client.get("/api/projects").json()[0]["id"] == made["id"]  # only the good one


def test_existing_git_repo_is_used_as_is(client: TestClient, tmp_path: Path):
    repo = tmp_path / "svc"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "app.py").write_text("x = 1\n")
    r = client.post("/api/projects", json={"name": "svc", "repo": str(repo)})
    assert r.status_code == 400 and "main branch" in r.json()["error"]
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "add", "-A"], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=repo, check=True)
    p = client.post("/api/projects", json={"name": "svc", "repo": str(repo)}).json()
    assert p["repo_path"] == str(repo.resolve()) and not (repo / "minibank").exists()


def test_upload_csvs_commits_them_to_main(client: TestClient, tmp_path: Path):
    p = client.post("/api/projects", json={"name": "rec"}).json()
    csvs = {
        "internal_positions.csv": b"position_id,quantity\nPOS-1,10\n",
        "vendor_reported_positions.csv": b"VendorRecordID,HeldQty\nV-1,10\n",
    }
    files = [("files", (name, data, "text/csv")) for name, data in csvs.items()]
    r = client.post(f"/api/projects/{p['id']}/files", data={"folder": "data"}, files=files)
    assert r.status_code == 200, r.text
    up = r.json()
    assert up["files"] == ["data/internal_positions.csv", "data/vendor_reported_positions.csv"]
    repo = Path(p["repo_path"])
    log = subprocess.run(
        ["git", "log", "-1", "--name-only", "--format=%s", "main"],
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout
    assert "data/internal_positions.csv" in log and up["commit"]
    listed = {f["path"]: f["bytes"] for f in client.get(f"/api/projects/{p['id']}/files").json()}
    assert listed["data/internal_positions.csv"] == len(csvs["internal_positions.csv"])
    events = client.get("/api/events", params={"project": p["id"]}).json()
    assert events[-1]["type"] == "repo.files_added"

    again = client.post(f"/api/projects/{p['id']}/files", data={"folder": "data"}, files=files)
    assert again.json()["commit"] == ""  # same content: nothing new to commit


def test_upload_stays_inside_the_repo(client: TestClient):
    p = client.post("/api/projects", json={"name": "rec"}).json()
    one = [("files", ("a.csv", b"x,y\n1,2\n", "text/csv"))]
    url = f"/api/projects/{p['id']}/files"
    for folder in ("../escape", "data/../../x", ".git/hooks"):
        r = client.post(url, data={"folder": folder}, files=one)
        assert r.status_code == 400 and "inside the repo" in r.json()["error"]
    sneaky = [("files", ("../../evil.csv", b"x", "text/csv"))]
    r = client.post(url, data={"folder": "data"}, files=sneaky)
    assert r.json()["files"] == ["data/evil.csv"]  # only the base name is kept
    assert client.post(url, data={"folder": "data"}).status_code == 400  # no files


def test_upload_waits_for_runs_of_the_project(client: TestClient):
    p = client.post("/api/projects", json={"name": "rec"}).json()
    body = {"objective": "reconcile", "start": False}
    r = client.post(f"/api/projects/{p['id']}/objectives", json=body).json()
    runs(client).runs[r["root_id"]] = RunInfo(r["root_id"], RunOptions(), "now")
    one = [("files", ("a.csv", b"x\n", "text/csv"))]
    res = client.post(f"/api/projects/{p['id']}/files", data={"folder": "data"}, files=one)
    assert res.status_code == 409


def test_new_projects_start_empty(client: TestClient):
    p = client.post("/api/projects", json={"name": "Positions Rec"}).json()
    files = [f["path"] for f in client.get(f"/api/projects/{p['id']}/files").json()]
    assert files == [".gitignore", "README.md", "pyproject.toml", "tests/__init__.py"]
    toml = (Path(p["repo_path"]) / "pyproject.toml").read_text()
    assert 'name = "positions-rec"' in toml and 'known-first-party = ["positions_rec"]' in toml
    r = client.post(f"/api/projects/{p['id']}/objectives", json={"offline": True})
    assert r.status_code == 400 and "minibank" in r.json()["error"]


def test_minibank_starter_runs_offline_even_after_the_ui_restarts(client: TestClient):
    c = client
    p = c.post("/api/projects", json={"name": "Demo", "starter": "minibank"}).json()
    files = [f["path"] for f in c.get(f"/api/projects/{p['id']}/files").json()]
    assert "minibank/payments/cards.py" in files and "tests/test_ledger.py" in files
    r = c.post(f"/api/projects/{p['id']}/objectives", json={"offline": True}).json()
    [req] = wait_for_run(c, r["root_id"])["requests"]
    runs(c).runs.clear()  # a restarted UI no longer knows the run was offline
    c.post(f"/api/requests/{req['id']}/answer", json={"response": "approve"})
    ob = wait_for_run(c, r["root_id"])
    assert ob["run"]["options"]["offline"] and ob["run"]["report"]["status"] == "DONE", ob["run"]
    bad = c.post("/api/projects", json={"name": "x", "starter": "nope"})
    assert bad.status_code == 400 and "starter" in bad.json()["error"]


def test_file_preview_reads_main(client: TestClient):
    p = client.post("/api/projects", json={"name": "rec"}).json()
    one = [("files", ("a.csv", b"x,y\n1,2\n", "text/csv"))]
    client.post(f"/api/projects/{p['id']}/files", data={"folder": "data"}, files=one)
    url = f"/api/projects/{p['id']}/files/content"
    f = client.get(url, params={"path": "data/a.csv"}).json()
    assert f == {"path": "data/a.csv", "bytes": 8, "text": "x,y\n1,2\n", "truncated": False}
    assert client.get(url, params={"path": "../../etc/passwd"}).status_code == 404
    chat = client.get(f"/api/projects/{p['id']}/chat").json()["messages"]
    assert [m["text"] for m in chat] == ["uploaded data/a.csv"]


# ------------------------------------------------------------------ PM chat and settings


def scripted_pm(calls: list[tuple[str, dict]], reply: Callable[[list[str]], str]):
    """A model that makes these tool calls in order, then replies with reply(tool outputs)."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    def respond(messages, info):
        outputs = [
            str(p.content)
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart) or type(p).__name__ == "RetryPromptPart"
        ]
        if len(outputs) < len(calls):
            name, args = calls[len(outputs)]
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart(reply(outputs))])

    return lambda _name: FunctionModel(respond)


def make_client(tmp_path: Path, monkeypatch, model=None, env_text: str | None = None):
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    env = tmp_path / ".env"
    if env_text is not None:
        env.write_text(env_text)
    orch = TaskOrchestrator(open_db(tmp_path / "orch.db"))
    app = create_app(
        orch,
        team_path=ROOT / "teams" / "fintech_sdlc.yaml",
        env_file=env,
        chat_model=model,
    )
    return TestClient(app), env


def wait_for_reply(c: TestClient, pid: str, said: str, timeout: float = 30) -> list[dict]:
    """The chat once the PM has replied to our message `said`."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        msgs = c.get(f"/api/projects/{pid}/chat").json()["messages"]
        i = max((n for n, m in enumerate(msgs) if m["text"] == said), default=None)
        after = msgs[i + 1 :] if i is not None else []
        if any(m["role"] == "pm" and not m.get("starts_objective") for m in after):
            return msgs
        time.sleep(0.1)
    raise AssertionError("the PM did not reply")


def test_pm_answers_where_the_code_is(tmp_path: Path, monkeypatch):
    seen: list[str] = []

    def reply(outputs: list[str]) -> str:
        seen.extend(outputs)
        return "It is on the objective branch, not merged yet."

    c, _ = make_client(tmp_path, monkeypatch)
    bank = str(minibank_repo(tmp_path / "bank"))
    p = c.post("/api/projects", json={"name": "bank", "repo": bank}).json()
    r = c.post(f"/api/projects/{p['id']}/objectives", json={"offline": True}).json()
    wait_for_run(c, r["root_id"])  # stops at the approval: code is on obj/<root> only
    calls = [
        ("project_status", {}),
        ("read_file", {"path": "minibank/api/transfers.py", "where": r["root_id"]}),
        ("read_file", {"path": "minibank/api/transfers.py", "where": "main"}),
    ]
    c2, _ = make_client(tmp_path, monkeypatch, model=scripted_pm(calls, reply))
    assert c2.post(f"/api/projects/{p['id']}/say", json={"text": "where is the code?"}).json()
    msgs = wait_for_reply(c2, p["id"], "where is the code?")
    status, on_branch, on_main = seen
    assert f"obj/{r['root_id']}" in status and "minibank/api/transfers.py" in status
    assert "'merged_into_main': False" in status
    assert "class Transfers" in on_branch and "is not on main" in on_main
    said = [(m["role"], m["text"]) for m in msgs if m["role"] in ("user", "pm")]
    assert said[-2:] == [
        ("user", "where is the code?"),
        ("pm", "It is on the objective branch, not merged yet."),
    ]
    assert len(c2.get(f"/api/projects/{p['id']}").json()["objectives"]) == 1  # no new plan
    events = c2.get("/api/events", params={"project": p["id"]}).json()
    assert any(e["type"] == "chat.usage" for e in events)


def test_pm_starts_an_objective_when_asked_for_work(tmp_path: Path, monkeypatch):
    calls = [("start_objective", {"objective": "Reconcile data/a.csv against data/b.csv"})]
    c, _ = make_client(tmp_path, monkeypatch, model=scripted_pm(calls, lambda o: "Started."))
    started: list[str] = []
    runs(c).start = lambda root, opts=None: started.append(root)  # no real run
    p = c.post("/api/projects", json={"name": "rec"}).json()
    c.post(f"/api/projects/{p['id']}/say", json={"text": "build me a reconciliation"})
    msgs = wait_for_reply(c, p["id"], "build me a reconciliation")
    [o] = c.get(f"/api/projects/{p['id']}").json()["objectives"]
    assert started == [o["id"]] and o["objective"].startswith("Reconcile data/a.csv")
    pm = [m for m in msgs if m["role"] == "pm"]
    assert pm[0]["starts_objective"] and pm[0]["text"].startswith("New objective:")
    assert pm[-1]["text"] == "Started." and pm[-1]["root_id"] == o["id"]


def test_pm_will_not_start_a_second_objective_while_one_runs(tmp_path: Path, monkeypatch):
    calls = [("start_objective", {"objective": "more work"})]
    c, _ = make_client(tmp_path, monkeypatch, model=scripted_pm(calls, lambda o: o[0]))
    p = c.post("/api/projects", json={"name": "rec"}).json()
    body = {"objective": "first", "start": False}
    r = c.post(f"/api/projects/{p['id']}/objectives", json=body).json()
    runs(c).runs[r["root_id"]] = RunInfo(r["root_id"], RunOptions(), "now")
    c.post(f"/api/projects/{p['id']}/say", json={"text": "also do more work"})
    msgs = wait_for_reply(c, p["id"], "also do more work")
    assert "already running" in msgs[-1]["text"]
    assert len(c.get(f"/api/projects/{p['id']}").json()["objectives"]) == 1


def test_say_needs_a_key(tmp_path: Path, monkeypatch):
    c, _ = make_client(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)  # no .env for meta_model to find
    p = c.post("/api/projects", json={"name": "rec"}).json()
    r = c.post(f"/api/projects/{p['id']}/say", json={"text": "hi"})
    assert r.status_code == 400 and "MODEL_API_KEY" in r.json()["error"]
    assert c.get(f"/api/projects/{p['id']}/chat").json()["messages"] == []


def test_api_key_is_saved_to_env_and_never_shown(tmp_path: Path, monkeypatch):
    c, env = make_client(tmp_path, monkeypatch, env_text="OTHER=1\nMODEL_API_KEY=old-key-0000\n")
    assert c.get("/api/settings").json()["api_key"]["hint"] == "…0000"
    r = c.post("/api/settings/api-key", json={"key": "test-key-abcd1234"})
    assert r.json()["api_key"] == {
        "set": True,
        "source": ".env",
        "hint": "…1234",
        "env_file": str(env),
        "env_overrides_file": False,
    }
    assert "test-key-abcd1234" not in r.text
    assert env.read_text() == "OTHER=1\nMODEL_API_KEY=test-key-abcd1234\n"
    assert env.stat().st_mode & 0o777 == 0o600
    assert c.get("/api/overview").json()["api_key"] is True
    bad = c.post("/api/settings/api-key", json={"key": "two words"})
    assert bad.status_code == 400
    gone = c.post("/api/settings/api-key", json={"remove": True}).json()["api_key"]
    assert not gone["set"] and env.read_text() == "OTHER=1\n"


def test_rejecting_the_change_says_what_to_do_next(client: TestClient, tmp_path: Path):
    c = client
    bank = str(minibank_repo(tmp_path / "bank"))
    p = c.post("/api/projects", json={"name": "Rej", "repo": bank}).json()
    r = c.post(f"/api/projects/{p['id']}/objectives", json={"offline": True}).json()
    [req] = wait_for_run(c, r["root_id"])["requests"]
    c.post(f"/api/requests/{req['id']}/answer", json={"response": "reject"})
    wait_for_run(c, r["root_id"])
    chat = c.get(f"/api/projects/{p['id']}/chat").json()["messages"]
    assert chat[-1]["text"].startswith("You rejected the change, so nothing was merged")
    tl = next(m for m in chat if m.get("steps"))["timeline"]
    assert "you rejected the change for merge" in [x["text"] for x in tl]
    head = subprocess.run(
        ["git", "log", "--oneline"], cwd=p["repo_path"], capture_output=True, text=True
    ).stdout
    assert "Merge obj/" not in head  # main untouched


def test_a_short_chat_reply_is_the_decision(client: TestClient, tmp_path: Path):
    c = client
    bank = str(minibank_repo(tmp_path / "bank"))
    p = c.post("/api/projects", json={"name": "Say", "repo": bank}).json()
    r = c.post(f"/api/projects/{p['id']}/objectives", json={"offline": True}).json()
    wait_for_run(c, r["root_id"])
    said = c.post(f"/api/projects/{p['id']}/say", json={"text": "ok approve"}).json()
    assert said["status"] and said["resumed"]  # answered like the button; no PM call
    ob = wait_for_run(c, r["root_id"])
    assert ob["run"]["state"] != "running" and not runs(c).orch.open_human_requests(p["id"])
    head = subprocess.run(
        ["git", "log", "--oneline"], cwd=bank, capture_output=True, text=True
    ).stdout
    assert "Merge obj/" in head


def test_short_decision_only_takes_a_bare_yes_or_no():
    from orch.web.server import short_decision

    both = ["approve", "reject"]
    assert [short_decision(t, both) for t in ["Ok, approve!", "lgtm", "no, reject it"]] == [
        "approve",
        "approve",
        "reject",
    ]
    assert short_decision("approve but rename it", both) is None
    assert short_decision("did you use numpy", both) is None
    assert short_decision("approve", ["a", "b"]) is None


def test_step_time_counts_only_working_time():
    # a gate that ran twice, two minutes apart, because a reviewer sent the code back
    d: dict = {"started": None, "ended": None}
    for frm, to, ts in [
        ("READY", "RUNNING", "2026-11-04T02:43:22.000+00:00"),
        ("RUNNING", "DONE", "2026-11-04T02:43:22.600+00:00"),
        ("DONE", "PENDING", "2026-11-04T02:44:46.000+00:00"),
        ("PENDING", "READY", "2026-11-04T02:45:16.000+00:00"),
        ("READY", "RUNNING", "2026-11-04T02:45:18.000+00:00"),
        ("RUNNING", "DONE", "2026-11-04T02:45:18.500+00:00"),
    ]:
        _track(d, {"from": frm, "to": to}, ts)
    assert d["busy_ms"] == 1100 and d["since"] is None
    assert (d["started"], d["ended"]) == (
        "2026-11-04T02:43:22.000+00:00",
        "2026-11-04T02:45:18.500+00:00",
    )


def test_time_waiting_on_a_person_is_kept_apart():
    # the PM asks for sign-off on the criteria, the person answers 20 minutes later
    d: dict = {"started": None, "ended": None}
    for frm, to, ts in [
        ("READY", "RUNNING", "2026-11-04T15:28:00.000+00:00"),
        ("RUNNING", "AWAITING_HUMAN", "2026-11-04T15:28:40.000+00:00"),
        ("AWAITING_HUMAN", "READY", "2026-11-04T15:48:40.000+00:00"),
        ("READY", "RUNNING", "2026-11-04T15:48:41.000+00:00"),
    ]:
        _track(d, {"from": frm, "to": to}, ts)
    assert d["waited_ms"] == 20 * 60 * 1000 and d["waiting_since"] is None
    assert d["busy_ms"] == 40_000 and d["since"] == "2026-11-04T15:48:41.000+00:00"
    _track(d, {"from": "RUNNING", "to": "AWAITING_HUMAN"}, "2026-11-04T15:49:00.000+00:00")
    assert d["waiting_since"] == "2026-11-04T15:49:00.000+00:00"


def test_timeline_says_what_a_person_answered_and_why_a_draft_failed():
    from orch.web.server import _answer_line, _plan_errors

    assert _answer_line("criteria", "approve") == "you approved the acceptance criteria"
    assert _answer_line("approval", "reject") == "you rejected the change for merge"
    assert _answer_line("question", "use EUR") == "you answered: use EUR"
    errors = [
        {"code": "COV-001", "message": f"criterion AC-{i} has no test node"} for i in (1, 2, 3)
    ] + [{"code": "GATE-001", "message": "implement is not verified by gate(s) ruff"}]
    assert _plan_errors(errors) == (
        "COV-001: criterion AC-1 has no test node, and 2 more like it; "
        "GATE-001: implement is not verified by gate(s) ruff"
    )


def test_choose_folder_returns_the_dialogs_pick_or_null(client: TestClient, monkeypatch):
    from orch.web import server

    seen = []

    def fake_run(cmd, **_):
        seen.append(cmd[0])
        out = "/Users/me/code/svc/\n" if len(seen) == 1 else ""
        return subprocess.CompletedProcess(cmd, 0 if out else 1, out, "User canceled.")

    monkeypatch.setattr(server.sys, "platform", "darwin")
    monkeypatch.setattr(server.subprocess, "run", fake_run)
    assert client.post("/api/choose-folder").json() == {"path": "/Users/me/code/svc"}
    assert client.post("/api/choose-folder").json() == {"path": None}  # cancelled
    assert seen == ["osascript", "osascript"]

    monkeypatch.setattr(server.sys, "platform", "win32")
    client.post("/api/choose-folder")
    assert seen[-1] == "powershell"

    monkeypatch.setattr(server.sys, "platform", "linux")
    assert client.post("/api/choose-folder").status_code == 400


def test_only_one_folder_dialog_opens_at_a_time(client: TestClient, monkeypatch):
    import threading

    from orch.web import server

    opened, close = threading.Event(), threading.Event()

    def slow_dialog(cmd, **_):
        opened.set()
        close.wait(10)
        return subprocess.CompletedProcess(cmd, 0, "/picked\n", "")

    monkeypatch.setattr(server.sys, "platform", "darwin")
    monkeypatch.setattr(server.subprocess, "run", slow_dialog)
    first: dict = {}
    t = threading.Thread(target=lambda: first.update(client.post("/api/choose-folder").json()))
    t.start()
    assert opened.wait(10)
    second = client.post("/api/choose-folder")
    assert second.status_code == 409  # the first dialog is still open
    close.set()
    t.join(10)
    assert first == {"path": "/picked"}
    assert client.post("/api/choose-folder").status_code == 200  # free again


def test_project_uses_the_picked_folder_or_the_default(client: TestClient, tmp_path: Path):
    picked = tmp_path / "picked" / "svc"
    p = client.post("/api/projects", json={"name": "Svc", "repo": str(picked)}).json()
    assert p["repo_path"] == str(picked.resolve())
    assert (picked / ".git").exists()

    d = client.post("/api/projects", json={"name": "Dflt", "repo": ""}).json()
    assert d["repo_path"] == str((tmp_path / "repos" / "dflt").resolve())
