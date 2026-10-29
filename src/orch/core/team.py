"""Loads the YAML team file. Agents are configuration, not processes."""

from __future__ import annotations

from pathlib import Path

import yaml

from orch.domain.schemas import AgentConfig, TeamConfig


class TeamRegistry:
    def __init__(self, config: TeamConfig) -> None:
        self.config = config
        self._by_name = {a.name: a for a in config.agents}

    @classmethod
    def load(cls, path: str | Path) -> TeamRegistry:
        data = yaml.safe_load(Path(path).read_text())
        return cls(TeamConfig.model_validate(data))

    @property
    def pm(self) -> str:
        return self.config.pm

    def get(self, name: str) -> AgentConfig:
        return self._by_name[name]

    def names(self) -> list[str]:
        return list(self._by_name)

    def agents_with(self, capability: str) -> list[str]:
        return [a.name for a in self.config.agents if capability in a.capabilities]

    def describe(self) -> list[dict[str, object]]:
        """What the PM sees through list_agents()."""
        return [
            {
                "name": a.name,
                "description": a.description,
                "capabilities": a.capabilities,
                "write_paths": a.write_paths,
            }
            for a in self.config.agents
        ]
