from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orch.core.orchestrator import TaskOrchestrator
from orch.core.team import TeamRegistry
from orch.db.session import open_db
from orch.domain.schemas import AgentConfig, Plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def team() -> TeamRegistry:
    return TeamRegistry.load(ROOT / "teams" / "fintech_sdlc.yaml")


@pytest.fixture
def team_with_security(team: TeamRegistry) -> TeamRegistry:
    cfg = team.config.model_copy(deep=True)
    cfg.agents.append(
        AgentConfig(
            name="security_reviewer",
            description="Reviews payment code for PCI and PII exposure.",
            capabilities=["security_review"],
            tools=["read_file", "list_files", "search_code", "submit_review"],
        )
    )
    return TeamRegistry(cfg)


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
