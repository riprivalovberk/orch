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
