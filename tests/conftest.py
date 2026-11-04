from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orch.core.orchestrator import TaskOrchestrator
from orch.core.team import TeamRegistry
from orch.db.session import open_db
from orch.domain.schemas import Plan
from orch.runtime.envs import SHARED

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def shared_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Target-repo tests run on orch's Python: no per-project installs (or network) here."""
    monkeypatch.setenv(SHARED, "1")


@pytest.fixture
def team(team_with_security: TeamRegistry) -> TeamRegistry:
    """The team without its security specialist, as before one was hired."""
    cfg = team_with_security.config.model_copy(deep=True)
    cfg.agents = [a for a in cfg.agents if "security_review" not in a.capabilities]
    return TeamRegistry(cfg)


@pytest.fixture
def team_with_security() -> TeamRegistry:
    return TeamRegistry.load(ROOT / "teams" / "fintech_sdlc.yaml")


@pytest.fixture
def good_plan() -> Plan:
    return Plan.model_validate(
        yaml.safe_load((ROOT / "examples" / "plan_transfers.yaml").read_text())
    )


@pytest.fixture
def bad_plan() -> Plan:
    return Plan.model_validate(yaml.safe_load((ROOT / "examples" / "plan_bad.yaml").read_text()))


@pytest.fixture
def orch(tmp_path: Path) -> TaskOrchestrator:
    return TaskOrchestrator(open_db(tmp_path / "orch.db"))
