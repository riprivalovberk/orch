"""Tools for the agents that do the work: test engineers, developers and reviewers.

Each tool acts on one directory, chosen by the caller's task, never by the arguments:
a test/implement task gets its own worktree (read and write); a review gets the worktree
of the work it checks (read only). Writes must match both the agent's write_paths and the
task's planned paths (scope.task_can_write).

The docstrings are the tool descriptions agents see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai import ModelRetry, RunContext

from orch.core.orchestrator import OrchestratorError, TaskOrchestrator
from orch.core.scope import is_safe_relative, task_can_write
from orch.domain.enums import REVIEW_KINDS, TaskKind, Verdict
from orch.domain.schemas import Finding, ReviewVerdict
from orch.policies.fintech_sdlc import PROTECTED_DIRS
from orch.runtime import gates
from orch.runtime.workspace import Workspace

if TYPE_CHECKING:
    from orch.mcp.server import Caller

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
MAX_FILES = 500
MAX_HITS = 100
MAX_ARTIFACT_BYTES = 1_000_000
ARTIFACT_NAME = re.compile(r"^[\w][\w.-]{0,99}$")
ArtifactKind = Literal["report", "notes", "design", "log", "other"]
WRITING = (TaskKind.TEST, TaskKind.IMPLEMENT)


@dataclass
class WorkerDeps:
    orch: TaskOrchestrator
    caller: Caller
    ws: Workspace

    @property
    def writes(self) -> bool:
        return self.caller.kind in WRITING

    def root(self) -> Path:
        c = self.caller
        if c.kind in WRITING:
            return self.ws.task_dir(c.task_id)
        if c.kind == TaskKind.PLAN:  # the PM reads the objective's branch to write criteria
            return self.ws.objective_dir(self.orch.get_task(c.task_id)["root_id"])
        if c.kind in REVIEW_KINDS:
            targets = self.orch.verified_targets(c.task_id)
            if targets:
                return self.ws.task_dir(targets[0])
        raise ModelRetry(f"{c.kind} tasks have no working directory")

    def resolve(self, path: str) -> Path:
        if not is_safe_relative(path):
            raise ModelRetry(f"{path!r}: use a repo-relative path without '..'")
        if ".git" in Path(path).parts:
            raise ModelRetry(".git is off limits")
        return self.root() / path


def _walk(root: Path, pattern: str) -> list[str]:
    out = []
    for p in sorted(root.glob(pattern)):
        rel = p.relative_to(root)
        if p.is_file() and not SKIP_DIRS & set(rel.parts):
            out.append(rel.as_posix())
    return out


def _call(fn: Any, *args: Any, **kw: Any) -> Any:
    try:
        return fn(*args, **kw)
    except OrchestratorError as e:
        raise ModelRetry(str(e)) from e


def read_file(
    ctx: RunContext[WorkerDeps], path: str, start_line: int = 1, max_lines: int = 400
) -> str:
    """Read a text file from the repo, with line numbers.

    Args:
        path: repo-relative path, e.g. minibank/api/transfers.py
        start_line: first line to return (1-based)
        max_lines: at most this many lines
    """
    p = ctx.deps.resolve(path)
    if not p.is_file():
        raise ModelRetry(f"{path} does not exist; use list_files")
    lines = p.read_text(errors="replace").splitlines()
    start = max(start_line, 1)
    chunk = lines[start - 1 : start - 1 + max_lines]
    body = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start))
    more = len(lines) - (start - 1 + len(chunk))
    return body + (f"\n... {more} more line(s)" if more > 0 else "")


def list_files(ctx: RunContext[WorkerDeps], glob: str = "**/*") -> list[str]:
    """List repo files matching a glob (default: every file).

    Args:
        glob: e.g. "minibank/**/*.py" or "tests/*"
    """
    files = _walk(ctx.deps.root(), glob)
    return files[:MAX_FILES] + (
        [f"... {len(files) - MAX_FILES} more"] if len(files) > MAX_FILES else []
    )


def search_code(ctx: RunContext[WorkerDeps], pattern: str, glob: str = "**/*.py") -> list[str]:
    """Search files for a regular expression. Returns "path:line: text" hits.

    Args:
        pattern: Python regular expression
        glob: which files to search
    """
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ModelRetry(f"bad regular expression: {e}") from e
    root = ctx.deps.root()
    hits: list[str] = []
    for rel in _walk(root, glob):
        for n, line in enumerate((root / rel).read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:200]}")
                if len(hits) >= MAX_HITS:
                    return [*hits, "... more hits, narrow the search"]
    return hits


def write_file(ctx: RunContext[WorkerDeps], path: str, content: str) -> dict[str, Any]:
    """Create or overwrite a file with the full new content.

    Only paths inside both your write_paths and this task's planned paths are allowed.

    Args:
        path: repo-relative path
        content: the whole file
    """
    p = _writable(ctx.deps, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"written": path, "bytes": len(content.encode())}


def edit_file(
    ctx: RunContext[WorkerDeps], path: str, old_text: str, new_text: str
) -> dict[str, Any]:
    """Change part of an existing file: replace old_text, which must occur exactly once,
    with new_text. Use it for fixes instead of rewriting the whole file with write_file.

    Args:
        path: repo-relative path
        old_text: text copied exactly from the file, with enough lines around it to be unique
        new_text: what replaces it (empty deletes it)
    """
    p = _writable(ctx.deps, path)
    if not p.is_file():
        raise ModelRetry(f"{path} does not exist; create it with write_file")
    text = p.read_text()
    found = text.count(old_text) if old_text else 0
    if found != 1:
        raise ModelRetry(
            f"old_text occurs {found} times in {path}; copy it exactly from read_file, "
            "with enough surrounding lines to make it unique"
        )
    p.write_text(text.replace(old_text, new_text))
    return {"edited": path, "bytes": len(p.read_bytes())}


def _writable(d: WorkerDeps, path: str) -> Path:
    """The file to write, once the task and agent scope allow it."""
    c = d.caller
    if not d.writes:
        raise ModelRetry(f"{c.kind} tasks are read-only")
    if not task_can_write(c.write_paths, c.paths, path, PROTECTED_DIRS):
        raise ModelRetry(
            f"{path} is outside this task's scope: your write_paths are {list(c.write_paths)} "
            f"and this task may write {list(c.paths)}"
        )
    return d.resolve(path)


def run_tests(ctx: RunContext[WorkerDeps], target: str = "") -> dict[str, Any]:
    """Run pytest in the repo (or on one test file or node id) and return the result.

    Args:
        target: e.g. tests/test_transfers.py, or empty for the whole suite
    """
    if target and not is_safe_relative(target.split("::")[0]):
        raise ModelRetry("target must be a repo-relative test path")
    ok, out, ms = gates.pytest(ctx.deps.root(), *([target] if target else []), timeout=180)
    return {"passed": ok, "seconds": round(ms / 1000, 1), "output": out}


def submit_artifact(
    ctx: RunContext[WorkerDeps],
    name: str,
    kind: ArtifactKind,
    content: str = "",
    path: str = "",
) -> dict[str, Any]:
    """Hand in a file that is not code, e.g. a test report or design notes. It is kept with
    the task's evidence; it does not change the repo. Give content, or the path of a file
    in the repo, not both.

    Args:
        name: file name to keep it under, e.g. test-report.md
        kind: report | notes | design | log | other
        content: the text to keep
        path: repo-relative file to keep a copy of instead
    """
    d, c = ctx.deps, ctx.deps.caller
    if not ARTIFACT_NAME.match(name):
        raise ModelRetry("name: a plain file name like test-report.md (letters, digits, . _ -)")
    if bool(content) == bool(path):
        raise ModelRetry("give either content or path, not both and not neither")
    if path:
        p = d.resolve(path)
        if not p.is_file():
            raise ModelRetry(f"{path} does not exist; use list_files")
        data = p.read_bytes()
    else:
        data = content.encode()
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ModelRetry(f"too big: {len(data):,} bytes, limit {MAX_ARTIFACT_BYTES:,}")
    sha, stored = d.ws.store_artifact(data)
    art = _call(
        d.orch.add_artifact,
        c.task_id,
        name=name,
        kind=kind,
        sha256=sha,
        path=str(stored),
        size=len(data),
        run_id=c.run_id,
        actor=c.agent,
    )
    return {"artifact": art, "name": name, "sha256": sha, "bytes": len(data)}


def complete_task(ctx: RunContext[WorkerDeps], summary: str) -> dict[str, Any]:
    """Finish your task: your changes are committed and sent to the verifiers.

    Args:
        summary: what you changed and why, in a few sentences
    """
    d, c = ctx.deps, ctx.deps.caller
    sha = None
    tidied: list[str] = []
    if d.writes:
        # Cheap checks the gates would fail anyway: catch them here, while the agent can fix
        # them, instead of spending a review round on them.
        changed = d.ws.changed_files(c.task_id)
        stray = [
            f for f in changed if not task_can_write(c.write_paths, c.paths, f, PROTECTED_DIRS)
        ]
        if stray:
            # Something other than write_file made them, e.g. code that runs on import or
            # in a test. The policy gate would fail every round, so undo them now.
            d.ws.revert_files(c.task_id, stray)
            raise ModelRetry(
                f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} outside this task's "
                f"paths ({', '.join(c.paths)}), so they were removed. Code you wrote created "
                "or changed them (on import, at test time, or in a script): remove that code, "
                "run the tests, then complete_task. If a test needs a file you may not write, "
                "say so in your summary instead."
            )
        py = [f for f in changed if f.endswith(".py") and (d.root() / f).exists()]
        tidied = gates.ruff_tidy(d.root(), py)
        if py:
            ok, out, _ = gates.ruff(d.root(), *py)
            if not ok:
                raise ModelRetry(
                    "ruff fails on your files; fix each spot with edit_file, then "
                    f"complete_task:\n{out}"
                )
        sha = d.ws.commit(c.task_id, f"{c.agent}: {summary[:72]}\n\n{summary}", author=c.agent)
        rnd = d.orch.get_task(c.task_id)["revision_round"]
        if sha is None and rnd > 0:
            raise ModelRetry(
                f"this is revision round {rnd} and you changed nothing since the last round; "
                "address the feedback in your brief, then complete_task"
            )
    status = _call(d.orch.complete, c.task_id, summary, actor=c.agent)
    return {
        "status": status,
        "commit": sha,
        "formatted": tidied,
        "next": "Done. Reply with one line and make no more calls.",
    }


def submit_review(
    ctx: RunContext[WorkerDeps],
    verdict: Verdict,
    summary: str,
    findings: list[Finding] | None = None,
) -> dict[str, Any]:
    """Record your review of the change.

    request_changes needs at least one blocker or major finding whose criterion_id cites
    an acceptance criterion (AC-1, ...), an engineering standard (STD-001, ...) or GAP.
    Say in the issue exactly what to change. approve when every criterion and standard is
    met and nothing important is missing.

    GAP is for what the criteria leave out but the work needs (bad input, errors, edge
    cases) or where they drift from the objective. Give it an assumption: the behavior
    the work should have, e.g. "a missing required column returns HTTP 400 naming it".
    Blocking GAP findings go back to the developer; every GAP is shown to the human at
    final approval, so state your assumptions plainly.

    Args:
        verdict: approve or request_changes
        summary: one paragraph on what you checked
        findings: issues found; severity blocker | major | minor
    """
    d, c = ctx.deps, ctx.deps.caller
    v = ReviewVerdict(verdict=verdict, findings=findings or [], summary=summary)
    status = _call(d.orch.record_review, c.task_id, v, actor=c.agent)
    return {"status": status, "next": "Review recorded. Reply with one line and stop."}


WORKER_TOOLS = [
    read_file,
    list_files,
    search_code,
    write_file,
    edit_file,
    run_tests,
    submit_artifact,
    complete_task,
    submit_review,
]
