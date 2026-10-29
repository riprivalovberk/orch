"""SQLite engine (WAL, foreign keys on) and session factory."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from orch.db.models import Base


def make_engine(db_path: str | Path) -> Engine:
    """A SQLite engine that the MCP server, runner and CLI can share safely."""
    # check_same_thread=False: let a connection be used from a thread other than the one
    # that opened it (the async MCP server and worker threads share the pool).
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    # SQLite settings are per connection, so set them every time the pool opens one.
    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        # Write-ahead log: readers don't block the writer and vice versa.
        cur.execute("PRAGMA journal_mode=WAL")
        # SQLite ignores ForeignKey(...) constraints unless this is on.
        cur.execute("PRAGMA foreign_keys=ON")
        # If another connection holds the write lock, wait up to 5s instead of
        # failing at once with "database is locked".
        cur.execute("PRAGMA busy_timeout=5000")
        # Fewer fsyncs, faster writes. With WAL a power loss can drop the last few
        # commits but never corrupts the file.
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


def open_db(db_path: str | Path) -> sessionmaker[Session]:
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
