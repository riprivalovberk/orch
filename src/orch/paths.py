"""Where FinOrch keeps everything it creates: one folder, safe to delete at any time.

Databases, spike results and transcripts all live under ``data_dir()``. Default ``.orch/``
in the current directory (gitignored); override with ORCH_DATA_DIR. Wipe it with ``orch nuke``.
"""

from __future__ import annotations

import os
from pathlib import Path

# The sample bank app the demos build on (a tiny service with a protected payments/ package).
# It ships with FinOrch as a template: `orch minibank <folder>` makes a fresh git repo from it.
MINIBANK = Path(__file__).resolve().parents[2] / "examples" / "minibank"


def data_dir() -> Path:
    return Path(os.environ.get("ORCH_DATA_DIR", ".orch")).resolve()


def db_path() -> Path:
    """The main orchestrator database (runner, MCP server, CLI)."""
    return data_dir() / "orch.db"


def default_minibank_repo() -> Path:
    """Where `orch minibank` puts the demo repo when no folder is given."""
    return data_dir() / "repos" / "minibank"
