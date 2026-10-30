"""Where an agent may write. Enforced by the MCP ``write_file`` tool and read by PAY-001.

``write_paths`` are repo-relative globs (``minibank/**``). A protected directory such as
``payments`` is never reached through a broad glob: the agent needs a pattern that names it
(``minibank/payments/**``). So ``developer: [minibank/**]`` cannot touch payments code, and
the validator can tell from the team file alone who can.
"""

from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import PurePosixPath


def is_safe_relative(path: str) -> bool:
    p = PurePosixPath(path)
    return bool(path) and not p.is_absolute() and ".." not in p.parts


def protected_grants(write_paths: Iterable[str], protected: frozenset[str]) -> set[str]:
    """The protected directories these patterns name explicitly."""
    return {part for g in write_paths for part in PurePosixPath(g).parts if part in protected}


def can_write(write_paths: Iterable[str], path: str, protected: frozenset[str]) -> bool:
    if not is_safe_relative(path):
        return False
    granted = [g for g in write_paths if fnmatchcase(path, g)]
    if not granted:
        return False
    hit = protected & set(PurePosixPath(path).parts[:-1])
    return not hit or hit <= protected_grants(granted, protected)


def task_can_write(
    agent_paths: Iterable[str], task_paths: Iterable[str], path: str, protected: frozenset[str]
) -> bool:
    """write_file's check: inside the agent's scope AND inside what the plan said this task
    writes. A PM that leaves payments paths off a node can't reach payments code at runtime."""
    return can_write(agent_paths, path, protected) and can_write(task_paths, path, protected)


def touches(paths: Iterable[str], protected: frozenset[str]) -> set[str]:
    """Protected directories any of these paths or globs names."""
    return {part for p in paths for part in PurePosixPath(p).parts if part in protected}
