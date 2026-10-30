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
    assert r.exit_code == 1 and "PAY-001" in r.output and "needs a human" in r.output


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
