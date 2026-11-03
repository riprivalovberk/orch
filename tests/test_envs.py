from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from orch.runtime import envs, gates


def repo(tmp_path: Path, *, requirements: str = "", dependencies: str = "") -> Path:
    r = tmp_path / "repo"
    (r / "tests").mkdir(parents=True)
    (r / "pyproject.toml").write_text(f'[project]\nname = "x"\ndependencies = [{dependencies}]\n')
    if requirements:
        (r / "requirements.txt").write_text(requirements)
    return r


def test_declared_reads_requirements_and_pyproject(tmp_path: Path):
    r = repo(
        tmp_path,
        requirements="azure-functions\n# a comment\n-r other.txt\npandas>=2  # data\n",
        dependencies='"pandas>=2", "numpy"',
    )
    assert envs.declared(r) == ["azure-functions", "numpy", "pandas>=2"]


def test_nothing_declared_runs_on_orchs_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(envs.SHARED)
    assert envs.python_for(repo(tmp_path)) == sys.executable


def test_each_project_gets_its_own_env_rebuilt_when_its_packages_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(envs.SHARED)
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path / "data"))
    built: list[list[str]] = []
    monkeypatch.setattr(
        envs, "_build", lambda uv, env, spec: built.append(spec) or env.mkdir(exist_ok=True)
    )
    r = repo(tmp_path, requirements="azure-functions\n")
    first = envs.python_for(r)
    assert envs.python_for(r) == first
    assert first == str(tmp_path / "data" / "envs" / "repo" / "bin" / "python")
    assert built == [["azure-functions", "pytest"]]  # once, with pytest added
    (r / "requirements.txt").write_text("azure-functions\npandas\n")
    envs.python_for(r)
    assert built[-1] == ["azure-functions", "pandas", "pytest"]


def test_a_task_worktree_shares_its_projects_env_name(tmp_path: Path):
    r = repo(tmp_path)
    run = {"cwd": r, "check": True, "capture_output": True}
    subprocess.run(["git", "init", "-q", "-b", "main"], **run)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        **run,
    )
    wt = tmp_path / "wt" / "tsk_123"
    subprocess.run(["git", "worktree", "add", "-q", str(wt)], **run)
    assert envs.project_name(r) == envs.project_name(wt) == "repo"


def test_a_bad_package_fails_the_pytest_gate_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def fail(uv, env, spec):
        raise envs.EnvError("could not install no-such-pkg")

    monkeypatch.delenv(envs.SHARED)
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(envs, "_build", fail)
    ok, out, _ = gates.pytest(repo(tmp_path, requirements="no-such-pkg\n"))
    assert not ok and "no-such-pkg" in out and "requirements.txt" in out


def test_tests_run_with_the_declared_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A real uv build, offline from uv's cache: iniconfig ships with pytest, so it is there."""
    monkeypatch.delenv(envs.SHARED)
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("UV_OFFLINE", "1")
    r = repo(tmp_path, requirements="iniconfig\n")
    (r / "tests" / "test_env.py").write_text(
        "import sys\n\n\ndef test_own_env():\n    assert '/envs/' in sys.executable\n"
    )
    try:
        python = envs.python_for(r)
    except envs.EnvError as e:
        pytest.skip(f"uv cannot build offline here: {e}")
    out = subprocess.run([python, "-c", "import pandas"], capture_output=True, env=os.environ)
    assert out.returncode != 0  # only what the project declared, not orch's packages
    ok, text, _ = gates.pytest(r)
    assert ok, text
