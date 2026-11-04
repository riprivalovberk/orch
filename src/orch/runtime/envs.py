"""Each project's tests run with the project's own dependencies.

A project that declares dependencies (requirements.txt, or [project].dependencies in
pyproject.toml) gets one virtualenv, named after it (.orch/envs/positions-rec-10), holding
exactly those plus pytest. When the list changes, uv syncs the packages in place: the
environment is never deleted, so tests running on it keep their interpreter. Code that
imports a package it did not declare fails its tests, as it would on deploy, and nothing
has to stand in for a package the project needs. A project that declares nothing runs on
orch's Python.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from orch.paths import data_dir

SHARED = "ORCH_SHARED_PYTHON"  # "1": always use orch's Python (orch's own tests, offline)
INSTALL_TIMEOUT_S = 600


# uv's words when the package index cannot be reached, as opposed to a package that is not on it.
UNREACHABLE = re.compile(
    r"(?i)error sending request|dns error|failed to lookup address|name resolution|"
    r"connection (refused|reset|closed)|network is unreachable|operation timed out|"
    r"tls handshake|certificate"
)


class EnvError(RuntimeError):
    """The project's dependencies could not be installed: fix what it declares."""


class IndexUnreachable(EnvError):
    """The package index could not be reached; nothing the project declares is wrong."""


def declared(repo: Path) -> list[str]:
    """The packages the project says it needs, from requirements.txt and pyproject.toml."""
    deps: set[str] = set()
    req = repo / "requirements.txt"
    if req.is_file():
        for line in req.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line and not line.startswith("-"):  # -r, -e and other pip options
                deps.add(line)
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        # A broken pyproject.toml fails the tests themselves, with a clearer message.
        with contextlib.suppress(tomllib.TOMLDecodeError):
            project = tomllib.loads(pyproject.read_text()).get("project", {})
            deps.update(project.get("dependencies", []))
    return sorted(deps)


def python_for(repo: Path) -> str:
    """The Python to run this project's tests with, building its environment if needed."""
    deps = declared(repo)
    uv = shutil.which("uv")
    if not deps or os.environ.get(SHARED) == "1" or uv is None:
        return sys.executable
    spec = [*deps, "pytest"]
    want = "\n".join([sys.version, *spec]) + "\n"
    env = data_dir() / "envs" / project_name(repo)
    python = env / "bin" / "python"
    built = env / "spec.txt"  # written last: what the environment holds once it is synced
    if _reads(built) == want:
        return str(python)
    env.parent.mkdir(parents=True, exist_ok=True)
    # Tasks verified in parallel may ask for the same environment: sync it once.
    with (env.parent / f"{env.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _reads(built) != want:
            built.unlink(missing_ok=True)  # not ready until the sync below finishes
            _build(uv, env, spec)
            built.write_text(want)
    return str(python)


def project_name(repo: Path) -> str:
    """The project a checkout belongs to: a task's worktree shares its repo's .git."""
    p = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    common = Path(p.stdout.strip()) if p.returncode == 0 else repo / ".git"
    return common.parent.name if common.name == ".git" else common.name


def _reads(path: Path) -> str | None:
    return path.read_text() if path.is_file() else None


def _build(uv: str, env: Path, spec: list[str]) -> None:
    """Create the environment if it is missing, then make its packages exactly spec."""
    python = env / "bin" / "python"
    if not python.exists():
        _uv([uv, "venv", "--quiet", "--python", sys.executable, str(env)], spec)
    wanted, locked = env / "requirements.in", env / "requirements.lock"
    wanted.write_text("\n".join(spec) + "\n")
    # compile resolves the dependencies; sync installs those and removes everything else.
    _uv(
        [uv, "pip", "compile", "--quiet", "--python", str(python), str(wanted), "-o", str(locked)],
        spec,
    )
    _uv([uv, "pip", "sync", "--quiet", "--python", str(python), str(locked)], spec)


def _uv(cmd: list[str], spec: list[str]) -> None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise IndexUnreachable(f"installing {', '.join(spec)} timed out") from e
    if p.returncode != 0:
        detail = (p.stderr or p.stdout)[-3000:]
        if UNREACHABLE.search(detail):
            raise IndexUnreachable(f"could not reach the package index:\n{detail}")
        raise EnvError(f"could not install {', '.join(spec)}:\n{detail}")
