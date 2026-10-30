"""Where orch keeps everything it creates: one folder, safe to delete at any time.

Databases, spike results and transcripts all live under ``data_dir()``. Default ``.orch/``
in the current directory (gitignored); override with ORCH_DATA_DIR. Wipe it with ``orch nuke``.
"""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("ORCH_DATA_DIR", ".orch")).resolve()


def db_path() -> Path:
    """The main orchestrator database (runner, MCP server, CLI)."""
    return data_dir() / "orch.db"
