"""Each project's tests run with the project's own dependencies.

A project that declares dependencies (requirements.txt, or [project].dependencies in
pyproject.toml) gets a virtualenv named after it (.orch/envs/positions-rec-10) with exactly
those plus pytest, built with uv and rebuilt whenever the list changes. Code that imports a
package it did not declare fails its tests, as it would on deploy, and nothing has to stand
in for a package the project needs. A project that declares nothing runs on orch's Python.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from orch.paths import data_dir

SHARED = "ORCH_SHARED_PYTHON"  # "1": always use orch's Python (orch's own tests, offline)
INSTALL_TIMEOUT_S = 600


class EnvError(RuntimeError):
    """The project's dependencies could not be installed."""


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
    built = env / "spec.txt"  # written last: what the environment holds once it is whole
    if _reads(built) == want:
        return str(python)
    env.parent.mkdir(parents=True, exist_ok=True)
    # Tasks verified in parallel may ask for the same environment: build it once.
    with (env.parent / f"{env.name}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _reads(built) != want:
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
    shutil.rmtree(env, ignore_errors=True)  # an old package list, or a build that failed
    steps = [
        [uv, "venv", "--quiet", "--python", sys.executable, str(env)],
        [uv, "pip", "install", "--quiet", "--python", str(env / "bin" / "python"), *spec],
    ]
    for cmd in steps:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            raise EnvError(f"installing {', '.join(spec)} timed out") from e
        if p.returncode != 0:
            shutil.rmtree(env, ignore_errors=True)
            raise EnvError(
                f"could not install {', '.join(spec)}:\n{(p.stderr or p.stdout)[-3000:]}"
            )
