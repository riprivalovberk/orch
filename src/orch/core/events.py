"""EventLog: append-only, hash-chained history. Feeds the live UI and the evidence pack."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from orch.db.models import Event
from orch.ids import now_iso

GENESIS = "0" * 64
_LAST = "orch_last_hash"
_PENDING = "orch_pending_events"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def event_hash(prev_hash: str, fields: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(fields)).encode()).hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    count: int
    first_bad_seq: int | None = None
    reason: str = ""


Listener = Callable[[Event], None]


class EventLog:
    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    def subscribe(self, fn: Listener) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn)

    def append(
        self,
        s: Session,
        *,
        project_id: str,
        type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        task_id: str | None = None,
        run_id: str | None = None,
    ) -> Event:
        last: dict[str, str] = s.info.setdefault(_LAST, {})
        prev = last.get(project_id)
        if prev is None:
            prev = (
                s.scalar(
                    select(Event.hash)
                    .where(Event.project_id == project_id)
                    .order_by(Event.seq.desc())
                    .limit(1)
                )
                or GENESIS
            )
        fields = {
            "project_id": project_id,
            "task_id": task_id,
            "run_id": run_id,
            "actor": actor,
            "type": type,
            # Round-trip through JSON so the hashed form equals the stored form.
            "payload": json.loads(canonical_json(payload or {})),
            "ts": now_iso(),
        }
        h = event_hash(prev, fields)
        ev = Event(**fields, prev_hash=prev, hash=h)
        s.add(ev)
        last[project_id] = h
        s.info.setdefault(_PENDING, []).append(ev)
        return ev

    def publish(self, events: Iterable[Event]) -> None:
        """Called by the orchestrator after a transaction commits."""
        for ev in events:
            for fn in list(self._listeners):
                fn(ev)

    @staticmethod
    def verify_chain(s: Session, project_id: str) -> VerifyResult:
        prev = GENESIS
        count = 0
        rows = s.scalars(select(Event).where(Event.project_id == project_id).order_by(Event.seq))
        for ev in rows:
            count += 1
            if ev.prev_hash != prev:
                return VerifyResult(
                    False, count, ev.seq, "prev_hash does not match the previous event"
                )
            if event_hash(prev, ev.hashed_fields()) != ev.hash:
                return VerifyResult(False, count, ev.seq, "event contents do not match their hash")
            prev = ev.hash
        return VerifyResult(True, count)
