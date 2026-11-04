import subprocess

from typer.testing import CliRunner

from orch.cli import app

from .conftest import ROOT

runner = CliRunner()


def test_plan_check_good():
    r = runner.invoke(
        app,
        [
            "plan-check",
            str(ROOT / "examples/plan_transfers.yaml"),
            "--team",
            str(ROOT / "teams/fintech_sdlc.yaml"),
        ],
    )
    assert r.exit_code == 0 and "cleared" in r.output


def test_plan_check_bad():
    r = runner.invoke(
        app,
        [
            "plan-check",
            str(ROOT / "examples/plan_bad.yaml"),
            "--team",
            str(ROOT / "teams/fintech_sdlc.yaml"),
        ],
    )
    assert r.exit_code == 1 and "PAY-001" in r.output and "security_reviewer" in r.output


def test_nuke_deletes_the_data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "spike" / "run1").mkdir(parents=True)
    (data / "orch.db").write_text("x")
    monkeypatch.setenv("ORCH_DATA_DIR", str(data))
    r = runner.invoke(app, ["nuke", "--yes"])
    assert r.exit_code == 0 and not data.exists()
    assert "nothing to delete" in runner.invoke(app, ["nuke", "--yes"]).output


def test_nuke_one_section(tmp_path, monkeypatch):
    (tmp_path / "spike" / "run1").mkdir(parents=True)
    for name in ("orch.db", "orch.db-wal"):
        (tmp_path / name).write_text("x")
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path))
    assert runner.invoke(app, ["nuke", "db", "-y"]).exit_code == 0
    assert not list(tmp_path.glob("orch.db*")) and (tmp_path / "spike").exists()
    assert runner.invoke(app, ["nuke", "spike", "-y"]).exit_code == 0
    assert not (tmp_path / "spike").exists()
    assert runner.invoke(app, ["nuke", "bogus", "-y"]).exit_code != 0


def test_minibank_makes_a_fresh_demo_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_DATA_DIR", str(tmp_path / "data"))
    r = runner.invoke(app, ["minibank"])
    repo = tmp_path / "data" / "repos" / "minibank"
    assert r.exit_code == 0 and f"--repo {repo}" in r.output, r.output
    assert (repo / "minibank" / "payments" / "cards.py").exists()
    assert not (repo / ".ruff_cache").exists() and not (repo / "__pycache__").exists()
    log = subprocess.run(
        ["git", "log", "--oneline", "main"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert log.count("\n") == 1 and "Initial import" in log
    again = runner.invoke(app, ["minibank", str(repo)])
    assert again.exit_code != 0 and "not empty" in again.output
