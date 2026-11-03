"""Gate checks the runner runs itself, never an agent: pytest, ruff and the policy scan.

Each runs in the worktree of the work task being verified and returns a GateResultIn.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from orch.core.scope import task_can_write
from orch.domain.enums import GateName
from orch.domain.schemas import GateResultIn

TIMEOUT_S = 300
TAIL = 6000

# Things that must never land in a bank's repo, whatever the task.
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "hard-coded password": re.compile(r"(?i)\bpassword\s*=\s*['\"][^'\"]{4,}['\"]"),
}


def _run(cmd: list[str], cwd: Path, *, timeout: float = TIMEOUT_S) -> tuple[int, str, int]:
    t0 = time.monotonic()
    try:
        # No __pycache__ in the worktree: the policy gate would count it as a changed file.
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        code, out = p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        code, out = -1, f"timed out after {timeout:.0f}s\n{e.stdout or ''}"
    return code, out[-TAIL:], int((time.monotonic() - t0) * 1000)


def pytest(cwd: Path, *args: str, timeout: float = TIMEOUT_S) -> tuple[bool, str, int]:
    code, out, ms = _run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd,
        timeout=timeout,
    )
    if code == 5:  # pytest: no tests collected
        out += "\nno tests were collected"
    return code == 0, out, ms


def ruff_tidy(cwd: Path, paths: Iterable[str]) -> list[str]:
    """Format and sort imports in these files; return the ones that changed.

    Only layout changes, never behaviour: an agent's code is formatted the same way whatever
    order the tests and the package were written in. Anything else ruff finds stays the
    agent's to fix.
    """
    paths = list(paths)
    if not paths:
        return []
    before = {p: (cwd / p).read_bytes() for p in paths}
    ruff = [sys.executable, "-m", "ruff"]
    _run([*ruff, "check", "--no-cache", "--select", "I", "--fix", "--exit-zero", *paths], cwd)
    _run([*ruff, "format", "--no-cache", *paths], cwd)
    return [p for p in paths if (cwd / p).read_bytes() != before[p]]


def ruff(cwd: Path, *paths: str) -> tuple[bool, str, int]:
    code, out, ms = _run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", *(paths or ["."])], cwd
    )
    return code == 0, out, ms


def policy(
    cwd: Path,
    changed: Iterable[str],
    *,
    agent_paths: Iterable[str],
    task_paths: Iterable[str],
    protected: frozenset[str],
) -> tuple[bool, str, int]:
    """Every changed file is inside the task's declared scope; no secrets in what changed."""
    t0 = time.monotonic()
    agent_paths, task_paths = list(agent_paths), list(task_paths)
    problems: list[str] = []
    files = sorted(changed)
    for f in files:
        if not task_can_write(agent_paths, task_paths, f, protected):
            problems.append(f"{f}: outside the task's paths {task_paths}")
        p = cwd / f
        if not p.is_file():
            continue
        text = p.read_text(errors="replace")
        for what, rx in SECRET_PATTERNS.items():
            if m := rx.search(text):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{f}:{line}: {what}")
    if not files:
        problems.append("the task changed no files")
    out = "\n".join(problems) or f"{len(files)} changed file(s) in scope, no secrets found"
    return not problems, out, int((time.monotonic() - t0) * 1000)


def run_gates(
    gates: Iterable[str],
    cwd: Path,
    *,
    changed: Iterable[str],
    agent_paths: Iterable[str],
    task_paths: Iterable[str],
    protected: frozenset[str],
) -> list[GateResultIn]:
    changed = list(changed)
    results = []
    for gate in gates:
        if gate == GateName.PYTEST:
            ok, out, ms = pytest(cwd)
        elif gate == GateName.RUFF:
            ok, out, ms = ruff(cwd)
        elif gate == GateName.POLICY:
            ok, out, ms = policy(
                cwd,
                changed,
                agent_paths=agent_paths,
                task_paths=task_paths,
                protected=protected,
            )
        else:
            ok, out, ms = False, f"unknown gate {gate}", 0
        results.append(GateResultIn(gate=GateName(gate), passed=ok, output=out, duration_ms=ms))
    return results
