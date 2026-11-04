"""Web UI: projects, objectives, live events, the PM chat and evidence, on one page.

    uv run orch ui            # http://127.0.0.1:8765

Runs started here go in background threads of this process, one per objective, with no
terminal to ask in: a question or approval stops the run, the page answers it and starts
the run again. Events are read from the database, so runs started with `orch run` show too.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from orch.core.orchestrator import OrchestratorError, TaskOrchestrator
from orch.core.team import TeamRegistry
from orch.domain.schemas import ProjectLimits
from orch.ids import now_iso
from orch.paths import data_dir

INDEX = Path(__file__).with_name("index.html")
log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class RunOptions:
    model: str = ""
    parallel: int = 3
    offline: bool = False


@dataclass
class RunInfo:
    root_id: str
    options: RunOptions
    started_at: str
    state: str = "running"  # running | stopped | error
    report: dict[str, Any] | None = None
    error: str = ""


@dataclass
class Runs:
    """Background runs, one thread (and event loop) per objective."""

    orch: TaskOrchestrator
    team_path: Path
    runs: dict[str, RunInfo] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    chatting: set[str] = field(default_factory=set)  # projects whose PM is writing a reply

    def running(self) -> list[str]:
        return [r for r, info in self.runs.items() if info.state == "running"]

    def start(self, root_id: str, options: RunOptions | None = None) -> RunInfo:
        os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
        from orch.runtime.model import DEFAULT_MODEL, PRICES, meta_model
        from orch.runtime.runner import Runner, last_model
        from orch.runtime.scripted import SCRIPTED_MODEL, scripted_team
        from orch.runtime.workspace import Workspace

        prev = self.runs.get(root_id)
        if options is None and prev is None:  # e.g. the UI restarted since it last ran
            options = RunOptions(offline=last_model(self.orch, root_id) == SCRIPTED_MODEL)
        opts = options or (prev.options if prev else RunOptions())
        opts.model = opts.model or DEFAULT_MODEL
        if opts.model not in PRICES:
            raise ApiError(f"no price for {opts.model}; add it to model.PRICES")
        project = self.orch.get_project(self.orch.get_task(root_id)["project_id"])
        model = scripted_team() if opts.offline else _live_model(meta_model, opts.model)
        team = TeamRegistry.load(self.team_path)
        ws = Workspace.create(Path(project["repo_path"]), data_dir() / "worktrees")
        with self.lock:
            if root_id in self.running():
                raise ApiError(f"{root_id} is already running", 409)
            self.orch.register_team(project["id"], team)  # picks up team edits
            if not self.running():
                # recover() retries every RUNNING task; with another run live that task is ours.
                self.orch.recover()
            info = RunInfo(root_id, opts, now_iso())
            self.runs[root_id] = info
        runner = Runner(
            self.orch, team, ws, model=model, price_as=opts.model, parallel=opts.parallel
        )

        def go() -> None:
            try:
                info.report = asdict(asyncio.run(runner.run(root_id, ask=None)))
                info.state = "stopped"
            except Exception:
                info.error = traceback.format_exc()
                info.state = "error"

        threading.Thread(target=go, name=f"run-{root_id}", daemon=True).start()
        return info


def _live_model(meta_model: Any, name: str) -> Any:
    try:
        return meta_model(name)
    except RuntimeError as e:  # no MODEL_API_KEY
        raise ApiError(f"{e}; tick 'offline' for the scripted demo team") from e


def check_repo_path(repo: Path) -> None:
    """A target repo is an existing git repo's top folder, or a new or empty folder that gets
    an empty Python project. Never write into a folder that already has other files."""
    from orch.runtime.workspace import git

    if (repo / ".git").exists():
        if git(repo, "rev-parse", "--verify", "-q", "refs/heads/main", check=False).returncode:
            raise ApiError(f"{repo} needs a main branch with at least one commit")
        return
    if repo.exists() and not repo.is_dir():
        raise ApiError(f"{repo} is a file, not a folder")
    if repo.is_dir() and any(repo.iterdir()):
        top = git(repo, "rev-parse", "--show-toplevel", check=False).stdout.strip()
        if top:
            raise ApiError(f"{repo} is inside the git repo {top}; give {top} itself")
        raise ApiError(f"{repo} has files but is not a git repo; run git init there first")
    parent = next((p for p in repo.parents if p.exists()), None)
    if parent is None or not os.access(parent, os.W_OK):
        raise ApiError(
            f"can't create {repo}: {parent} is not writable. Give a full path such as "
            "~/code/my-service, or leave the repo empty to make one under the data folder"
        )


FOLDER_PROMPT = "Choose the target repo folder"

# The OS's own folder dialog, run as a child process so it never touches the server's threads.
_MAC_PICK = [
    "osascript",
    "-e", "activate",  # bring the dialog in front of the browser
    "-e", f'POSIX path of (choose folder with prompt "{FOLDER_PROMPT}")',
]  # fmt: skip
_WIN_PICK = [
    "powershell", "-NoProfile", "-STA", "-Command",
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8;"
    "Add-Type -AssemblyName System.Windows.Forms;"
    "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
    f"$d.Description = '{FOLDER_PROMPT}';"
    "$top = New-Object System.Windows.Forms.Form -Property @{TopMost = $true};"
    "if ($d.ShowDialog($top) -eq 'OK') { $d.SelectedPath }",
]  # fmt: skip
_picking = threading.Lock()  # one dialog at a time, even from several tabs


def pick_folder() -> str | None:
    """Open Finder's / Explorer's folder dialog on this machine; the chosen path, or None if
    the user cancelled. Works because the UI server runs on the user's own machine."""
    if sys.platform == "darwin":
        cmd = _MAC_PICK
    elif sys.platform == "win32":
        cmd = _WIN_PICK
    else:
        raise ApiError("the folder dialog works on macOS and Windows; type the path instead")
    if not _picking.acquire(blocking=False):
        raise ApiError("a folder dialog is already open", 409)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    finally:
        _picking.release()
    path = p.stdout.strip()  # cancelling exits non-zero (macOS) or prints nothing (Windows)
    return str(Path(path)) if p.returncode == 0 and path else None


STARTERS = ("empty", "minibank")


def write_empty_repo(repo: Path, name: str) -> None:
    """A new git repo with just enough for the gates: pytest, ruff, a README. No app code."""
    from orch.runtime.workspace import git

    pkg = package_name(name)
    files = {
        "README.md": (
            f"# {name}\n\nBuilt by FinOrch's agents. The code lives in the `{pkg}` package; "
            "tests run with `python -m pytest`.\n"
        ),
        "pyproject.toml": (
            f'[project]\nname = "{_slug(name)}"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
            '\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = "-q"\n'
            '\n[tool.ruff]\nline-length = 100\ntarget-version = "py312"\n'
            '\n[tool.ruff.lint]\nselect = ["E", "F", "I", "B", "UP"]\n'
            # Tests import the package before it exists (test-first); without this, ruff sorts
            # it as third-party then, first-party once written, and the gate fails a file the
            # developer may not touch.
            f'\n[tool.ruff.lint.isort]\nknown-first-party = ["{pkg}"]\n'
        ),
        ".gitignore": "__pycache__/\n*.pyc\n.pytest_cache/\n.ruff_cache/\n",
        "tests/__init__.py": "",
    }
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Initial import")


KEY_VAR = "MODEL_API_KEY"
KEY_LINE = re.compile(rf"^\s*(export\s+)?{KEY_VAR}\s*=\s*(.*)$")


def key_status(env_file: Path) -> dict[str, Any]:
    """Whether the model API key is set and where, with only its last 4 characters."""
    value, source = os.environ.get(KEY_VAR, ""), "environment"
    if not value and env_file.is_file():
        for line in env_file.read_text().splitlines():
            if m := KEY_LINE.match(line):
                value, source = m.group(2).strip().strip("'\""), ".env"
    return {
        "set": bool(value),
        "source": source if value else None,
        "hint": f"…{value[-4:]}" if len(value) >= 8 else ("set" if value else ""),
        "env_file": str(env_file),
        # A key exported in the shell wins over .env (pydantic-settings), so .env edits
        # don't apply until the UI is started without it.
        "env_overrides_file": bool(os.environ.get(KEY_VAR)),
    }


def write_key(env_file: Path, key: str | None) -> None:
    """Set (or with None, remove) MODEL_API_KEY in .env, keeping its other lines."""
    lines = env_file.read_text().splitlines() if env_file.is_file() else []
    lines = [line for line in lines if not KEY_LINE.match(line)]
    if key is not None:
        lines.append(f"{KEY_VAR}={key}")
    tmp = env_file.with_name(env_file.name + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    tmp.chmod(0o600)  # a secret: readable by this user only
    tmp.replace(env_file)


MAX_FILES = 20
MAX_FILE_BYTES = 25_000_000


def upload_folder(folder: str) -> str:
    """A folder inside the repo, as a clean relative path ("" for the top)."""
    parts = [p for p in folder.replace("\\", "/").split("/") if p not in ("", ".")]
    if any(p == ".." or p.startswith(".") for p in parts):
        raise ApiError(f"folder {folder!r} must stay inside the repo (no .. or hidden folders)")
    return "/".join(parts)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"


def package_name(name: str) -> str:
    """The project's Python package: its name as an identifier (positions-rec -> positions_rec)."""
    pkg = _slug(name).replace("-", "_")
    return pkg if pkg[0].isalpha() else f"p_{pkg}"


WORK_KINDS = ("test", "implement", "review", "security_review")
QUIET_NOTES = ("waiting on a human", "")


def _msg(e: dict[str, Any], role: str, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "seq": e["seq"],
        "ts": e["ts"],
        "role": role,
        "actor": e["actor"],
        "text": text,
        **extra,
    }


def chat_messages(orch: TaskOrchestrator, runs: Runs, pid: str) -> dict[str, Any]:
    """The project's history as a conversation with the PM, built from its event log.

    role is "user" (what the person said), "pm" (the PM speaking: questions, plans, results)
    or "activity" (one-line progress from the rest of the team).
    """
    objectives = {o["id"]: o for o in orch.objectives(pid)}
    tasks = {t["id"]: t for root in objectives for t in orch.task_tree(root)}
    open_reqs = {r["id"]: {**r, "criteria": _is_criteria(r)} for r in orch.open_human_requests(pid)}
    out: list[dict[str, Any]] = []
    # Each objective's plan message carries live steps; their updates fold in under them.
    plan_msgs: dict[str, dict[str, Any]] = {}
    timing: dict[str, dict[str, str | None]] = {}
    logs: dict[str, list[dict[str, str]]] = {}
    notes: dict[str, str] = {}
    # Each objective's timeline: everything that happened to it, in order, handoffs included.
    timeline: dict[str, list[dict[str, str]]] = {}

    def mark(root: str | None, e: dict[str, Any], text: str, kind: str = "step") -> None:
        if root:
            timeline.setdefault(root, []).append({"ts": e["ts"], "text": text, "kind": kind})

    asked: dict[str, str] = {}  # request id -> what was asked: "criteria", "approval" or "question"
    drafts: dict[str, int] = {}  # objective -> plan drafts the PM has checked so far
    rejected: set[str] = set()  # objectives whose change the person rejected at approval
    events = orch.list_events(project_id=pid, limit=20_000)
    from_chat = {e["payload"].get("objective_id") for e in events if e["type"] == "chat.reply"} - {
        None
    }
    for e in events:
        p, tid, kind = e["payload"], e["task_id"], e["type"]
        t = tasks.get(tid or "")
        node = e["node"]
        root = t["root_id"] if t else None
        if kind == "task.created" and tid in objectives:
            o = objectives[tid]
            if tid in from_chat:  # the PM started it from the chat: it says so in its reply
                text = f"New objective: {o['objective']}"
                out.append(_msg(e, "pm", text, root_id=tid, starts_objective=True))
            else:
                out.append(_msg(e, "user", o["objective"], root_id=tid, starts_objective=True))
        elif kind == "chat.message":
            out.append(_msg(e, "user", p["text"]))
        elif kind == "chat.reply":
            out.append(_msg(e, "pm", p["text"], root_id=p.get("objective_id")))
        elif kind == "human.requested":
            rid = p["request_id"]
            req = {
                "id": rid,
                "kind": p["kind"],
                "options": p["options"],
                "open": rid in open_reqs,
                "criteria": _is_criteria(p),
            }
            out.append(_msg(e, "pm", p["prompt"], root_id=root, request=req))
            if p["kind"] == "approval":
                asked[rid] = "approval"
                mark(root, e, "FinOrch asked you to approve the change for merge", "human")
            elif req["criteria"]:
                asked[rid] = "criteria"
                mark(root, e, "the PM sent you the acceptance criteria to approve", "human")
            else:
                asked[rid] = "question"
                mark(root, e, f"the PM asked you: {p['prompt'].splitlines()[0][:80]}", "human")
        elif kind == "human.responded":
            out.append(_msg(e, "user", p["response"], root_id=root))
            mark(root, e, _answer_line(asked.get(p["request_id"], ""), p["response"]), "human")
            if asked.get(p["request_id"]) == "approval" and p["response"] == "reject" and root:
                rejected.add(root)
        elif kind == "plan.checked":
            n = drafts[root or ""] = drafts.get(root or "", 0) + 1
            if not p["valid"]:
                which = {1: "first", 2: "second", 3: "third"}.get(n, f"#{n}")
                why = _plan_errors(p["errors"])
                mark(
                    root,
                    e,
                    f"the PM's {which} draft of the plan failed checks ({why}); revising",
                    "fail",
                )
        elif kind == "plan.rejected":
            codes = ", ".join(sorted({err["code"] for err in p["errors"]}))
            out.append(
                _msg(e, "activity", f"plan v{p['version']} failed checks ({codes}); revising")
            )
            mark(
                root, e, f"plan v{p['version']} was rejected ({_plan_errors(p['errors'])})", "fail"
            )
        elif kind == "plan.accepted":
            # Task ids are time-ordered, so sorting them gives the order the PM wrote the plan.
            ids = sorted(p["tasks"].values()) if isinstance(p["tasks"], dict) else []
            steps = [
                {k: tasks[i][k] for k in ("node", "kind", "assignee")} for i in ids if i in tasks
            ]
            crit = objectives[root]["criteria"] if root in objectives else {}
            text = f"Here is the plan (v{p['version']}), checked against the rules. Starting now."
            out.append(_msg(e, "pm", text, root_id=root, plan=steps, criteria=crit))
            if root:
                plan_msgs[root] = out[-1]
            mark(root, e, f"plan v{p['version']} passed the checks: {len(steps)} steps")
        elif kind == "task.status" and t:
            _track(timing.setdefault(tid, {"started": None, "ended": None}), p, e["ts"])
            line = _status_line(t, p, tid in objectives)
            if tid in rejected and p["to"] == "FAILED":
                line = (
                    "You rejected the change, so nothing was merged into main. Tell me what to "
                    "change and the team will start again."
                )
            if p.get("reason") == "changes requested":  # a handoff back to the agent
                by = [tasks[v]["assignee"] for v in p.get("failed_verifiers", []) if v in tasks]
                mark(
                    root,
                    e,
                    f"{' and '.join(filter(None, by)) or 'review'} sent {t['node']} back to "
                    f"{t['assignee']} for changes (round {p.get('round', '?')})",
                    "back",
                )
            elif line:
                failed = p["to"] == "FAILED" or "FAILED" in line or "request_changes" in line
                label = line.removeprefix("checks: ") if t["kind"] == "gate" else line
                mark(root, e, label, "fail" if failed else "done" if tid in objectives else "step")
            if tid not in objectives:  # a planned step: shown in its row on the plan
                if line:
                    logs.setdefault(tid, []).append({"ts": e["ts"], "text": line})
                note = _step_note(t, p)
                if note:
                    notes[tid] = note
                elif p["to"] == "READY":  # reopened: the old result no longer applies
                    notes.pop(tid, None)
            elif line:
                role = "pm" if tid in objectives and p["to"] in ("DONE", "FAILED") else "activity"
                out.append(_msg(e, role, line, root_id=root))
        elif kind == "workspace.merged_to_main":
            out.append(_msg(e, "pm", f"Merged into main ({p['detail'][:10]}).", root_id=root))
            mark(root, e, "merged into main", "done")
        elif kind == "workspace.merge_conflict":
            out.append(_msg(e, "activity", f"merge conflict on {p['branch']}", root_id=root))
            mark(root, e, f"merge conflict on {p['branch']}", "fail")
        elif kind == "workspace.merge_blocked":
            text = (
                "Not merged: main changed since this was built, and with those changes the "
                "tests fail. main is unchanged. Ask me to make the change again and the team "
                "will build it on the current main."
            )
            out.append(_msg(e, "pm", text, root_id=root))
            mark(root, e, "not merged: the tests fail together with the latest main", "fail")
        elif kind == "artifact.added" and t and tid not in objectives:
            logs.setdefault(tid, []).append({"ts": e["ts"], "text": f"attached {p['name']}"})
        elif kind == "artifact.added":
            out.append(_msg(e, "activity", f"{node} attached {p['name']}", root_id=root))
        elif kind == "repo.files_added":
            out.append(_msg(e, "activity", f"uploaded {', '.join(p['files'])}"))
    for root, m in plan_msgs.items():
        m["steps"] = _steps(root, tasks, timing, logs, notes)
        m["timeline"] = timeline.get(root, [])
    working: list[str] = ["the PM is writing a reply"] if pid in runs.chatting else []
    for rid in runs.running():
        for t in orch.task_tree(rid) if rid in objectives else []:
            if t["status"] == "RUNNING" and t["kind"] != "gate":
                who = t["assignee"] or "system"
                working.append(f"{who} is working on {t['node'] or 'the plan'}")
    for rid, info in runs.runs.items():
        if rid not in objectives or info.state == "running":
            continue
        note = (info.report or {}).get("note", "")
        if info.state == "error":
            out.append(
                {
                    "role": "activity",
                    "text": "the runner crashed; see project details",
                    "root_id": rid,
                    "seq": None,
                    "ts": "",
                    "actor": "system",
                }
            )
        elif note not in QUIET_NOTES and objectives[rid]["status"] not in ("DONE", "FAILED"):
            out.append(
                {
                    "role": "pm",
                    "text": f"I had to stop: {note}",
                    "root_id": rid,
                    "seq": None,
                    "ts": "",
                    "actor": "system",
                }
            )
    running = [r for r in runs.running() if r in objectives]
    return {
        "messages": out,
        "working": working,
        "running": running,
        "open_requests": list(open_reqs.values()),
    }


STEP_STATE = {
    "RUNNING": "running",
    "VERIFYING": "checking",
    "AWAITING_HUMAN": "needs_you",
    "READY": "queued",
    "PENDING": "waiting",
    "FAILED": "failed",
    "CANCELED": "canceled",
}


def _track(d: dict[str, Any], p: dict[str, Any], ts: str) -> None:
    """A step's clocks. started/ended span the whole step (cleared if reopened); busy_ms adds up
    only the time spent RUNNING, so waits between review rounds and for a person don't count;
    waited_ms adds up the time spent waiting on a person, which the run's total leaves out."""
    to = p["to"]
    if p["from"] == "RUNNING" and d.get("since"):
        d["busy_ms"] = d.get("busy_ms", 0) + _ms(ts) - _ms(d["since"])
        d["since"] = None
    if p["from"] == "AWAITING_HUMAN" and d.get("waiting_since"):
        d["waited_ms"] = d.get("waited_ms", 0) + _ms(ts) - _ms(d["waiting_since"])
        d["waiting_since"] = None
    if to == "RUNNING":
        d["since"] = ts
    elif to == "AWAITING_HUMAN":
        d["waiting_since"] = ts
    if to in ("RUNNING", "AWAITING_HUMAN") and not d["started"]:
        d["started"] = ts
    if to in ("DONE", "FAILED", "CANCELED", "WAITING"):
        d["ended"] = d["ended"] if p["from"] == "WAITING" else ts  # root: planning time only
    elif to in ("READY", "RUNNING"):
        d["ended"] = None


def _is_criteria(req: dict[str, Any]) -> bool:
    """The PM's acceptance criteria sent for approval: an open question listing AC-1, AC-2..."""
    return req["kind"] != "approval" and not req["options"] and req["prompt"].startswith("AC-1")


def _answer_line(asked: str, response: str) -> str:
    """The timeline's line for a person's answer, saying what they answered."""
    if asked == "criteria" and response == "approve":
        return "you approved the acceptance criteria"
    if asked == "approval" and response == "approve":
        return "you approved the change for merge"
    if asked == "approval" and response == "reject":
        return "you rejected the change for merge"
    if asked == "approval" and response == "override_approve":
        return "you approved the change for merge, overriding a failed check"
    return f"you answered: {response.splitlines()[0][:80]}"


def _plan_errors(errors: list[dict[str, Any]]) -> str:
    """Why a draft plan failed, briefly: each rule broken with its first message and a count."""
    by_code: dict[str, list[str]] = {}
    for err in errors:
        by_code.setdefault(err["code"], []).append(err["message"])
    parts = [
        f"{code}: {msgs[0]}" + (f", and {len(msgs) - 1} more like it" if len(msgs) > 1 else "")
        for code, msgs in by_code.items()
    ]
    return textwrap.shorten("; ".join(parts), 220, placeholder="…")


def _ms(ts: str) -> int:
    return int(datetime.fromisoformat(ts).timestamp() * 1000)


def _step_note(t: dict[str, Any], p: dict[str, Any]) -> str:
    """The latest result worth showing next to a step, or ""."""
    if t["kind"] == "gate" and p["to"] == "DONE":
        failed = [g for g, ok in (p.get("results") or {}).items() if not ok]
        return f"{', '.join(failed)} failed" if failed else "all checks passed"
    if p.get("reason", "").startswith("review:"):
        return str(p.get("verdict", p["reason"].removeprefix("review: "))).replace("_", " ")
    if p["to"] == "FAILED":
        return p.get("reason", "")[:120]
    return ""


def _steps(
    root: str,
    tasks: dict[str, dict[str, Any]],
    timing: dict[str, dict[str, str | None]],
    logs: dict[str, list[dict[str, str]]],
    notes: dict[str, str],
) -> list[dict[str, Any]]:
    """The objective as a workflow: planning, then each planned task in the order written."""
    r, clock = tasks[root], timing.get(root, {})
    plan_state = "done" if clock.get("ended") else STEP_STATE.get(r["status"], "running")
    steps = [
        {
            "id": root,
            "node": "plan",
            "assignee": "project_manager",
            "state": plan_state,
            "note": "",
            "started": clock.get("started"),
            "ended": clock.get("ended"),
            "busy_ms": clock.get("busy_ms", 0),
            "since": clock.get("since") if plan_state == "running" else None,
            "waited_ms": clock.get("waited_ms", 0),
            "waiting_since": clock.get("waiting_since") if plan_state == "needs_you" else None,
            "log": [],
        }
    ]
    for tid in sorted(t for t, x in tasks.items() if x["root_id"] == root and t != root):
        t, c = tasks[tid], timing.get(tid, {})
        state = STEP_STATE.get(t["status"], "running")
        if t["status"] == "DONE":
            state = "done" if t["outcome"] in ("pass", "waived") else "failed"
        note = notes.get(tid, "")
        if state in ("waiting", "queued") and c.get("started"):  # ran before, reopened
            state, note = "rerun", f"last run: {note}" if note else ""
        if t["outcome"] == "waived":
            note = "overridden by you" + (f" · {note}" if note else "")
        if t["revision_round"]:
            note = f"revision {t['revision_round']}" + (f" · {note}" if note else "")
        steps.append(
            {
                "id": tid,
                "node": t["node"],
                "assignee": t["assignee"] or ("you" if t["kind"] == "approval" else t["kind"]),
                "state": state,
                "note": note,
                "started": c.get("started"),
                "ended": c.get("ended") if state not in ("running", "checking") else None,
                "busy_ms": c.get("busy_ms", 0),
                "since": c.get("since") if state == "running" else None,
                "waited_ms": c.get("waited_ms", 0),
                "waiting_since": c.get("waiting_since") if state == "needs_you" else None,
                "log": logs.get(tid, []),
            }
        )
    return steps


def _status_line(t: dict[str, Any], p: dict[str, Any], is_root: bool) -> str:
    """One chat line for a status change worth telling a person about, or ""."""
    to, reason, node = p["to"], p.get("reason", ""), t["node"] or "objective"
    if is_root:
        if to == "DONE":
            return "All tasks passed. The change is approved and done."
        if to == "FAILED":
            return f"This objective failed: {reason}."
        if to == "CANCELED":
            return f"Canceled: {reason}."
        return ""
    if t["kind"] == "gate" and to == "DONE":
        res = p.get("results") or {}
        return "checks: " + ", ".join(
            f"{g} {'passed' if ok else 'FAILED'}" for g, ok in res.items()
        )
    if t["kind"] not in WORK_KINDS and t["kind"] != "approval":
        return ""
    who = t["assignee"] or "you"
    if reason == "claimed" and t["kind"] in WORK_KINDS:
        return f"{who} started {node}"
    if reason.startswith("review:"):
        return f"{who} reviewed: {p.get('verdict', reason.removeprefix('review: '))}"
    if reason == "changes requested":
        return f"{node} sent back for changes (round {p.get('round', '?')})"
    if to == "DONE" and t["kind"] in ("test", "implement"):
        return f"{who} finished {node}"
    if to == "VERIFYING":
        return f"{who} handed in {node} for checks and review"
    if to == "FAILED":
        return f"{node} failed: {reason}"
    return ""


def create_app(
    orch: TaskOrchestrator,
    *,
    team_path: Path,
    default_budget: float = 2.0,
    env_file: Path | None = None,
    chat_model: Any = None,  # model name -> Model for the PM chat; tests pass a scripted one
) -> Starlette:
    from orch.cli import _criteria, evidence_markdown

    env = (env_file or Path(".env")).resolve()  # where ModelEnv reads the key from
    runs = Runs(orch, team_path)

    def team() -> TeamRegistry:
        return TeamRegistry.load(team_path)

    def project_view(pid: str) -> dict[str, Any]:
        objectives = orch.objectives(pid)
        for o in objectives:
            info = runs.runs.get(o["id"])
            o["run"] = asdict(info) if info else None
        return {**orch.get_project(pid), "objectives": objectives}

    def inbox(pid: str | None = None) -> list[dict[str, Any]]:
        out = []
        for p in [pid] if pid else orch.project_ids():
            for r in orch.open_human_requests(p):
                out.append(
                    {**r, "project_id": p, "root_id": orch.get_task(r["task_id"])["root_id"]}
                )
        return out

    # ------------------------------------------------------------------ handlers

    async def index(_: Request) -> Response:
        return FileResponse(INDEX, headers={"cache-control": "no-cache"})

    async def overview(_: Request) -> dict[str, Any]:
        from orch.runtime.model import DEFAULT_MODEL, PRICES

        reg = team()
        return {
            "data_dir": str(data_dir()),
            "team": {"path": str(team_path), "pm": reg.pm, "agents": reg.describe()},
            "models": list(PRICES),
            "default_model": DEFAULT_MODEL,
            "api_key": key_status(env)["set"],
            "default_budget": default_budget,
            "running": runs.running(),
        }

    async def projects(_: Request) -> list[dict[str, Any]]:
        return [project_view(pid) for pid in reversed(orch.project_ids())]

    async def create_project(req: Request) -> dict[str, Any]:
        from orch.runtime.workspace import GitError, Workspace, make_minibank

        body = await req.json()
        name = (body.get("name") or "").strip()
        if not name:
            raise ApiError("give the project a name")
        given = (body.get("repo") or "").strip()
        repo = Path(given or data_dir() / "repos" / _slug(name)).expanduser().resolve()
        budget = float(body.get("budget") or default_budget)
        if orch.project_for_repo(str(repo)):
            raise ApiError(f"{repo} already has a project; pick another repo path", 409)
        starter = body.get("starter") or "empty"
        if starter not in STARTERS:
            raise ApiError(f"starter must be one of {list(STARTERS)}")
        check_repo_path(repo)
        try:
            if (repo / ".git").exists():
                pass  # an existing repo is used as it is
            elif starter == "minibank":
                make_minibank(repo)
            else:
                write_empty_repo(repo, name)
            Workspace.create(repo, data_dir() / "worktrees")
        except (GitError, OSError) as e:
            raise ApiError(f"could not set up a repo at {repo}: {e}") from e
        pid = orch.create_project(
            name, repo_path=str(repo), limits=ProjectLimits(budget_usd=budget)
        )
        orch.register_team(pid, team())
        return project_view(pid)

    async def choose_folder(_: Request) -> dict[str, Any]:
        return {"path": await asyncio.to_thread(pick_folder)}

    async def project(req: Request) -> dict[str, Any]:
        return project_view(req.path_params["pid"])

    async def set_budget(req: Request) -> dict[str, Any]:
        pid = req.path_params["pid"]
        orch.set_budget(pid, float((await req.json())["budget"]))
        return project_view(pid)

    def project_repo(pid: str) -> Path:
        repo = orch.get_project(pid)["repo_path"]
        if not repo or not (Path(repo) / ".git").exists():
            raise ApiError(f"project {pid} has no git repo")
        return Path(repo)

    async def list_files(req: Request) -> list[dict[str, Any]]:
        from orch.runtime.workspace import git

        repo = project_repo(req.path_params["pid"])
        # What is committed on main: what an objective's worktree starts from.
        names = git(repo, "ls-tree", "-r", "-z", "--name-only", "main").stdout.split("\0")
        return [
            {"path": n, "bytes": (repo / n).stat().st_size if (repo / n).is_file() else None}
            for n in sorted(filter(None, names))
        ]

    async def upload_files(req: Request) -> dict[str, Any]:
        """Write the uploaded files into the repo's main branch and commit them, so every
        objective's worktree (cut from main) has them."""
        from orch.runtime.workspace import git

        pid = req.path_params["pid"]
        repo = project_repo(pid)
        busy = [r for r in runs.running() if orch.get_task(r)["project_id"] == pid]
        if busy:
            raise ApiError("wait for this project's runs to stop before adding files", 409)
        form = await req.form(max_files=MAX_FILES, max_part_size=MAX_FILE_BYTES)
        folder = upload_folder(str(form.get("folder") or ""))
        uploads = [f for f in form.getlist("files") if isinstance(f, UploadFile) and f.filename]
        if not uploads:
            raise ApiError("choose at least one file")
        if git(repo, "symbolic-ref", "--short", "HEAD", check=False).stdout.strip() != "main":
            raise ApiError(f"{repo} must have main checked out to add files")
        written: list[str] = []
        for f in uploads:
            name = Path(f.filename or "").name
            if not name or name.startswith("."):
                raise ApiError(f"bad file name {f.filename!r}")
            data = await f.read()
            if len(data) > MAX_FILE_BYTES:
                raise ApiError(f"{name} is over {MAX_FILE_BYTES // 1_000_000} MB")
            rel = (Path(folder) / name).as_posix() if folder else name
            dest = repo / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            written.append(rel)
        git(repo, "add", "--", *written)
        commit = ""
        if git(repo, "diff", "--cached", "--quiet", "--", *written, check=False).returncode:
            msg = f"Add {', '.join(written)}" if len(written) <= 3 else f"Add {len(written)} files"
            git(repo, "commit", "-q", "-m", msg, "--", *written)
            commit = git(repo, "rev-parse", "HEAD").stdout.strip()
        orch.log_project_event(pid, "repo.files_added", {"files": written, "commit": commit})
        return {"files": written, "commit": commit}

    async def verify(req: Request) -> dict[str, Any]:
        return asdict(orch.verify_events(req.path_params["pid"]))

    async def agents(req: Request) -> list[dict[str, Any]]:
        return orch.list_agents(req.path_params["pid"])

    async def create_objective(req: Request) -> dict[str, Any]:
        return await create_objective_from(req.path_params["pid"], await req.json())

    async def create_objective_from(pid: str, body: dict[str, Any]) -> dict[str, Any]:
        from orch.runtime.scripted import TRANSFERS_CRITERIA, TRANSFERS_OBJECTIVE

        opts = run_options(body)
        objective = (body.get("objective") or "").strip()
        criteria = [c for c in body.get("criteria") or [] if c.strip()]
        repo_path = orch.get_project(pid)["repo_path"]
        if opts.offline and not (Path(repo_path or "") / "minibank").is_dir():
            raise ApiError(
                "the offline scripted team only builds its transfers demo, which needs a "
                "minibank repo; make a new project that starts as the minibank sample"
            )
        if opts.offline and not objective:
            objective, criteria = (
                TRANSFERS_OBJECTIVE,
                [f"{k}={v}" for k, v in TRANSFERS_CRITERIA.items()],
            )
        if not objective:
            raise ApiError("say what to build")
        if body.get("budget"):
            orch.set_budget(pid, float(body["budget"]))
        orch.register_team(pid, team())
        root = orch.create_root_task(pid, objective, acceptance_criteria=_criteria(criteria))
        info = runs.start(root, opts) if body.get("start", True) else None
        return {"root_id": root, "run": asdict(info) if info else None}

    def run_options(body: dict[str, Any]) -> RunOptions:
        return RunOptions(
            model=body.get("model") or "",
            parallel=max(int(body.get("parallel") or 3), 1),
            offline=bool(body.get("offline")),
        )

    async def say(req: Request) -> dict[str, Any]:
        """A chat message to the PM. It answers in the background (the page polls); if the
        message asks for work, it starts an objective."""
        from orch.runtime.model import DEFAULT_MODEL, meta_model
        from orch.runtime.pm_chat import ChatDeps

        pid = req.path_params["pid"]
        body = await req.json()
        text = (body.get("text") or "").strip()
        if not text:
            raise ApiError("say something")
        decisions = [r for r in orch.open_human_requests(pid) if r["options"]]
        if len(decisions) == 1 and (said := short_decision(text, decisions[0]["options"])):
            return answer_request(decisions[0]["id"], said)
        opts = run_options(body)
        if opts.offline:  # the scripted team has no chat: the message is the objective
            return await create_objective_from(pid, {**body, "objective": text})
        if pid in runs.chatting:
            raise ApiError("the PM is still answering your last message", 409)
        if orch.remaining_budget(pid)["remaining_usd"] <= 0:
            raise ApiError("the project budget is used up; raise it in project details")
        model_name = opts.model or DEFAULT_MODEL
        model = chat_model(model_name) if chat_model else _live_model(meta_model, model_name)
        history = [
            (m["role"], m["text"])
            for m in chat_messages(orch, runs, pid)["messages"]
            if m["role"] in ("user", "pm")
        ]
        orch.log_project_event(pid, "chat.message", {"text": text})

        def start(objective: str) -> str:
            mine = {o["id"] for o in orch.objectives(pid)}
            if any(r in mine for r in runs.running()):
                raise ValueError("an objective is already running in this project; wait for it")
            orch.register_team(pid, team())
            root = orch.create_root_task(pid, objective)
            runs.start(root, opts)
            return root

        deps = ChatDeps(orch, pid, project_repo(pid), start)
        runs.chatting.add(pid)
        # Own thread and loop, like runs: the reply outlives this request.
        threading.Thread(
            target=lambda: asyncio.run(pm_reply(deps, history, text, model, model_name)),
            name=f"chat-{pid}",
            daemon=True,
        ).start()
        return {"ok": True}

    async def pm_reply(
        deps: Any, history: list[tuple[str, str]], text: str, model: Any, model_name: str
    ) -> None:
        from orch.runtime.model import cost_usd
        from orch.runtime.pm_chat import answer

        pid = deps.project_id
        pm = orch.get_project_pm(pid)
        try:
            reply = await answer(deps, history, text, model=model)
            orch.record_chat_usage(
                pid, actor=pm, tokens=reply.tokens, cost_usd=cost_usd(reply.usage, model_name)
            )
            payload: dict[str, Any] = {"text": reply.text}
            if reply.started:
                payload["objective_id"] = reply.started
            orch.log_project_event(pid, "chat.reply", payload, actor=pm)
        except Exception as e:  # tell the person in the thread, not just the server log
            log.exception("PM chat failed")
            msg = f"Sorry, I couldn't answer that ({type(e).__name__}: {e})."
            orch.log_project_event(pid, "chat.reply", {"text": msg}, actor=pm)
        finally:
            runs.chatting.discard(pid)

    async def objective(req: Request) -> dict[str, Any]:
        root = req.path_params["root"]
        ev = orch.evidence(root)
        info = runs.runs.get(root)
        tree = orch.task_tree(root)
        ids = {t["id"] for t in tree}
        return {
            **ev,
            "requests": [r for r in inbox(ev["project_id"]) if r["task_id"] in ids],
            "run": asdict(info) if info else None,
        }

    async def resume(req: Request) -> dict[str, Any]:
        body = await req.json() if await req.body() else {}
        opts = None
        if body:
            opts = RunOptions(
                model=body.get("model") or "",
                parallel=max(int(body.get("parallel") or 3), 1),
                offline=bool(body.get("offline")),
            )
        return asdict(runs.start(req.path_params["root"], opts))

    async def evidence(req: Request) -> Response:
        root = req.path_params["root"]
        ev = orch.evidence(root)
        chain = orch.verify_events(ev["project_id"])
        md = evidence_markdown(ev, chain.ok, chain.count)
        out = data_dir() / "evidence" / f"{root}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        return JSONResponse({"path": str(out), "markdown": md, "chain_ok": chain.ok})

    async def events(req: Request) -> list[dict[str, Any]]:
        q = req.query_params
        return orch.list_events(
            project_id=q.get("project") or None,
            root_id=q.get("root") or None,
            after=int(q.get("after") or 0),
            limit=min(int(q.get("limit") or 500), 2000),
        )

    def answer_request(rid: str, response: str, resume: bool = True) -> dict[str, Any]:
        status = orch.resolve_human(rid, response)
        out: dict[str, Any] = {"status": status, "resumed": False}
        if resume:
            root = _root_of_request(orch, rid)
            if root and root not in runs.running():
                runs.start(root)
                out["resumed"] = True
        return out

    async def answer(req: Request) -> dict[str, Any]:
        body = await req.json()
        return answer_request(
            req.path_params["rid"], str(body["response"]), bool(body.get("resume", True))
        )

    async def task(req: Request) -> dict[str, Any]:
        from orch.runtime.workspace import GitError, Workspace

        t = orch.get_task(req.path_params["tid"])
        repo = orch.get_project(t["project_id"])["repo_path"]
        diff = ""
        if t["kind"] in ("test", "implement") and repo:
            ws = Workspace(Path(repo), data_dir() / "worktrees")
            try:
                diff = ws.diff(t["id"]) if ws.task_dir(t["id"]).exists() else ""
            except GitError as e:
                diff = f"(no diff: {e})"
        return {
            **t,
            "feedback": orch.feedback(t["id"]),
            "answers": orch.human_answers(t["id"]),
            "diff": diff,
        }

    async def cancel(req: Request) -> dict[str, Any]:
        body = await req.json() if await req.body() else {}
        reason = body.get("reason") or "canceled from the web UI"
        return {"status": orch.cancel(req.path_params["tid"], reason)}

    async def nuke(req: Request) -> dict[str, Any]:
        """Spike results or target repos. The db stays: this process holds it open."""
        section = (await req.json()).get("section")
        d = data_dir()
        targets = {"spike": [d / "spike"], "repos": [d / "repos", d / "worktrees"]}
        if section not in targets:
            raise ApiError("pick spike or repos (for the db, stop the UI and run orch nuke db)")
        if runs.running():
            raise ApiError("wait for the running objectives to stop", 409)
        deleted = [str(t) for t in targets[section] if t.exists()]
        for t in deleted:
            shutil.rmtree(t)
        return {"deleted": deleted}

    async def get_settings(_: Request) -> dict[str, Any]:
        return {"api_key": key_status(env)}

    async def set_api_key(req: Request) -> dict[str, Any]:
        body = await req.json()
        if body.get("remove"):
            write_key(env, None)
        else:
            key = str(body.get("key") or "").strip()
            if not key or any(c.isspace() for c in key) or len(key) > 500:
                raise ApiError("paste the key as one line, with no spaces")
            write_key(env, key)
        return {"api_key": key_status(env)}

    async def chat(req: Request) -> dict[str, Any]:
        return chat_messages(orch, runs, req.path_params["pid"])

    async def file_content(req: Request) -> dict[str, Any]:
        from orch.runtime.workspace import git

        repo = project_repo(req.path_params["pid"])
        path = req.query_params.get("path", "")
        names = git(repo, "ls-tree", "-r", "-z", "--name-only", "main").stdout.split("\0")
        if path not in names:
            raise ApiError(f"{path} is not in main", 404)
        blob = subprocess.run(
            ["git", "show", f"main:{path}"], cwd=repo, capture_output=True, timeout=30
        ).stdout
        limit = 200_000
        return {
            "path": path,
            "bytes": len(blob),
            "text": blob[:limit].decode(errors="replace"),
            "truncated": len(blob) > limit,
        }

    # ------------------------------------------------------------------ wiring

    def api(fn: Any) -> Any:
        async def handler(req: Request) -> Response:
            try:
                out = await fn(req)
            except ApiError as e:
                return JSONResponse({"error": str(e)}, status_code=e.status)
            except OrchestratorError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
            except (KeyError, ValueError) as e:
                return JSONResponse({"error": f"bad request: {e}"}, status_code=400)
            except Exception as e:  # say what broke instead of a bare 500
                log.exception("%s %s failed", req.method, req.url.path)
                return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
            return out if isinstance(out, Response) else JSONResponse(out)

        return handler

    routes = [
        Route("/", index),
        Route("/api/overview", api(overview)),
        Route("/api/projects", api(projects)),
        Route("/api/projects", api(create_project), methods=["POST"]),
        Route("/api/choose-folder", api(choose_folder), methods=["POST"]),
        Route("/api/projects/{pid}", api(project)),
        Route("/api/projects/{pid}/budget", api(set_budget), methods=["POST"]),
        Route("/api/projects/{pid}/verify", api(verify)),
        Route("/api/projects/{pid}/agents", api(agents)),
        Route("/api/projects/{pid}/files", api(list_files)),
        Route("/api/projects/{pid}/files", api(upload_files), methods=["POST"]),
        Route("/api/projects/{pid}/files/content", api(file_content)),
        Route("/api/projects/{pid}/chat", api(chat)),
        Route("/api/projects/{pid}/say", api(say), methods=["POST"]),
        Route("/api/projects/{pid}/objectives", api(create_objective), methods=["POST"]),
        Route("/api/objectives/{root}", api(objective)),
        Route("/api/objectives/{root}/resume", api(resume), methods=["POST"]),
        Route("/api/objectives/{root}/evidence", api(evidence)),
        Route("/api/tasks/{tid}", api(task)),
        Route("/api/tasks/{tid}/cancel", api(cancel), methods=["POST"]),
        Route("/api/events", api(events)),
        Route("/api/requests/{rid}/answer", api(answer), methods=["POST"]),
        Route("/api/nuke", api(nuke), methods=["POST"]),
        Route("/api/settings", api(get_settings)),
        Route("/api/settings/api-key", api(set_api_key), methods=["POST"]),
    ]
    app = Starlette(routes=routes)
    app.state.runs = runs
    return app


# A short chat reply to an open Approve / Reject decision is that decision, as if clicked.
_SAID = {
    "approve": re.compile(
        r"(?:(?:ok(?:ay)?|yes|yep|sure)\W*)?(?:i\s+)?"
        r"(?:approved?(?:\s+it)?|lgtm|merge(?:\s+it)?|ship\s+it|go\s+ahead|yes|ok(?:ay)?)"
    ),
    "reject": re.compile(
        r"(?:no\W*)?(?:i\s+)?(?:reject(?:ed)?(?:\s+it)?|don'?t\s+merge(?:\s+it)?|no)"
    ),
}


def short_decision(text: str, options: list[str]) -> str | None:
    """'ok approve' -> 'approve' when the request offers it; None for anything longer."""
    said = re.sub(r"[\s.!]+$", "", text.strip().lower())
    for option, pattern in _SAID.items():
        if option in options and pattern.fullmatch(said):
            return option
    return None


def _root_of_request(orch: TaskOrchestrator, request_id: str) -> str | None:
    with orch.tx() as s:
        from orch.db.models import HumanRequest, Task

        hr = s.get(HumanRequest, request_id)
        t = s.get(Task, hr.task_id) if hr else None
        return t.root_id if t else None
