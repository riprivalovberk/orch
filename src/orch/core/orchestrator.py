"""TaskOrchestrator: the only writer of task state.

Agents never call this directly; the MCP server does, on their behalf. Every public method is
one transaction. Every status change goes through ``_transition`` (state-machine check,
hash-chained event, then ``_after_change`` to wake whatever the change unblocks).
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from orch.core.events import EventLog, VerifyResult
from orch.core.plan import AgentCard, PlanValidator, ValidationContext
from orch.core.state import IllegalTransition, assert_transition
from orch.core.team import TeamRegistry
from orch.db.models import (
    Agent,
    GateResult,
    HumanRequest,
    PlanRecord,
    Project,
    Review,
    Run,
    Task,
    TaskDep,
)
from orch.domain.enums import (
    PASSING_OUTCOMES,
    REVIEW_KINDS,
    TERMINAL_STATUSES,
    VERIFIER_KINDS,
    EdgeKind,
    HumanRequestKind,
    Outcome,
    TaskKind,
    TaskStatus,
    Verdict,
)
from orch.domain.schemas import GateResultIn, Plan, PlanError, ProjectLimits, ReviewVerdict
from orch.ids import new_id, now_iso
from orch.policies import get_pack

S = TaskStatus
SYSTEM = "system"
HUMAN = "human"


class OrchestratorError(Exception):
    """Base class for errors the MCP layer turns into a tool error for the agent."""


class NotFound(OrchestratorError):
    """An id that doesn't exist (project, task, run or human request)."""


class InvalidOperation(OrchestratorError):
    """The id exists, but the action isn't allowed in the task's current state."""


@dataclass
class PlanResult:
    """What submit_plan returns to the PM: accepted, or the errors and attempts left."""

    plan_id: str
    version: int
    valid: bool
    errors: list[PlanError] = field(default_factory=list)
    task_ids: dict[str, str] = field(default_factory=dict)  # plan node id -> task id
    escalated: bool = False
    attempts_left: int = 0


@dataclass
class Claim:
    """A claimed task: the run it opened and the one-time token the agent uses over MCP."""

    task_id: str
    run_id: str
    run_token: str  # shown once; only its hash is stored
    attempt: int


def hash_token(token: str) -> str:
    """Tokens are stored hashed, so a database leak doesn't leak live credentials."""
    return hashlib.sha256(token.encode()).hexdigest()


class TaskOrchestrator:
    """The only writer of task state. See the module docstring for the rules."""

    def __init__(self, session_factory: sessionmaker[Session], events: EventLog | None = None):
        """Wire up the DB session factory and the event log (shared with the live UI)."""
        self._sf = session_factory
        self.events = events or EventLog()
        self._validators: dict[str, PlanValidator] = {}

    @contextmanager
    def tx(self) -> Generator[Session, None, None]:
        """One transaction: commit at the end, roll back on error, then publish its events."""
        with self._sf() as s:
            with s.begin():
                yield s
            pending = s.info.pop("orch_pending_events", [])
        self.events.publish(pending)

    # ------------------------------------------------------------------ projects and teams

    def create_project(
        self,
        name: str,
        *,
        repo_path: str | None = None,
        policy_pack: str = "fintech_sdlc",
        limits: ProjectLimits | None = None,
    ) -> str:
        """Create a project (repo + policy pack + limits) and log project.created."""
        get_pack(policy_pack)  # fail fast on a typo
        limits = limits or ProjectLimits()
        with self.tx() as s:
            p = Project(
                id=new_id("prj"),
                name=name,
                repo_path=repo_path,
                policy_pack=policy_pack,
                limits=limits,
            )
            s.add(p)
            self.events.append(
                s,
                project_id=p.id,
                type="project.created",
                actor=HUMAN,
                payload={"name": name, "policy_pack": policy_pack},
            )
            return p.id

    def register_team(self, project_id: str, team: TeamRegistry) -> dict[str, list[str]]:
        """Upsert agents from the team file. Re-running after editing the YAML adds new agents."""
        with self.tx() as s:
            p = self._project(s, project_id)
            existing = {a.name: a for a in s.scalars(select(Agent).where(Agent.project_id == p.id))}
            added, updated = [], []
            for cfg in team.config.agents:
                fields = {
                    "description": cfg.description,
                    "capabilities": list(cfg.capabilities),
                    "tools": list(cfg.tools),
                    "write_paths": list(cfg.write_paths),
                    "model": cfg.model,
                    "prompt": cfg.prompt,
                }
                row = existing.pop(cfg.name, None)
                if row is None:
                    s.add(Agent(id=new_id("agt"), project_id=p.id, name=cfg.name, **fields))
                    added.append(cfg.name)
                elif any(getattr(row, k) != v for k, v in fields.items()):
                    for k, v in fields.items():
                        setattr(row, k, v)
                    updated.append(cfg.name)
            removed = sorted(existing)
            for row in existing.values():
                s.delete(row)
            p.pm_agent = team.pm
            change = {"added": added, "updated": updated, "removed": removed}
            self.events.append(
                s,
                project_id=p.id,
                type="team.registered",
                actor=HUMAN,
                payload={"team": team.config.team, "pm": team.pm, **change},
            )
            return change

    def list_agents(self, project_id: str) -> list[dict[str, Any]]:
        """The team as the PM sees it: name, description, capabilities and write scope."""
        with self.tx() as s:
            rows = s.scalars(
                select(Agent).where(Agent.project_id == project_id).order_by(Agent.name)
            )
            return [
                {
                    "name": a.name,
                    "description": a.description,
                    "capabilities": a.capabilities,
                    "write_paths": a.write_paths,
                }
                for a in rows
            ]

    # ------------------------------------------------------------------ root task and planning

    def create_root_task(
        self,
        project_id: str,
        objective: str,
        *,
        title: str | None = None,
        acceptance_criteria: dict[str, str] | None = None,
    ) -> str:
        """Start a new objective: a plan task for the PM, READY to be claimed."""
        with self.tx() as s:
            p = self._project(s, project_id)
            if not p.pm_agent:
                raise InvalidOperation("register a team before creating tasks")
            t = Task(
                id=new_id("tsk"),
                project_id=p.id,
                kind=TaskKind.PLAN,
                title=title or objective[:200],
                objective=objective,
                acceptance_criteria=acceptance_criteria or {},
                assignee=p.pm_agent,
                status=S.PENDING,
                depth=0,
            )
            t.root_id = t.id
            s.add(t)
            s.flush()
            self._event(s, t, "task.created", {"kind": t.kind, "title": t.title}, actor=HUMAN)
            self._try_ready(s, t)
            return t.id

    def validation_context(self, s: Session, plan_task: Task) -> ValidationContext:
        """Everything the validator needs about this project: team, limits, budget."""
        p = self._project(s, plan_task.project_id)
        limits = p.limits
        agents = {
            a.name: AgentCard(
                a.name, frozenset(a.capabilities), a.description, tuple(a.write_paths)
            )
            for a in s.scalars(select(Agent).where(Agent.project_id == p.id))
        }
        existing = s.scalar(select(func.count()).select_from(Task).where(Task.project_id == p.id))
        return ValidationContext(
            agents=agents,
            pm=p.pm_agent or "",
            limits=limits,
            existing_task_count=existing or 0,
            remaining_budget_usd=max(limits.budget_usd - p.spent_usd, 0.0),
        )

    def validate_plan(self, plan_task_id: str, plan: Plan, *, actor: str | None = None):
        """Dry run. Logged, so the evidence shows every draft the PM checked."""
        with self.tx() as s:
            t = self._plan_task(s, plan_task_id)
            errors = self._validator(s, t).validate(plan, self.validation_context(s, t))
            self._event(
                s,
                t,
                "plan.checked",
                {"valid": not errors, "errors": [e.model_dump() for e in errors]},
                actor=actor or t.assignee or SYSTEM,
            )
            return errors

    def submit_plan(
        self, plan_task_id: str, plan: Plan, *, run_id: str | None = None, actor: str | None = None
    ) -> PlanResult:
        """Validate and record a plan attempt. Accepted: create its tasks and park the PM.

        Rejected: return node-level errors. After max attempts, or an error re-planning
        can't fix, ask a human.
        """
        with self.tx() as s:
            t = self._plan_task(s, plan_task_id)
            if t.status != S.RUNNING:
                raise InvalidOperation(
                    f"plan task {t.id} must be RUNNING to submit (is {t.status})"
                )
            accepted = s.scalar(
                select(PlanRecord.id).where(
                    PlanRecord.root_task_id == t.id, PlanRecord.valid.is_(True)
                )
            )
            if accepted:
                raise InvalidOperation("this task already has an accepted plan")
            actor = actor or t.assignee or SYSTEM
            ctx = self.validation_context(s, t)
            errors = self._validator(s, t).validate(plan, ctx)
            version = (
                s.scalar(
                    select(func.max(PlanRecord.version)).where(PlanRecord.root_task_id == t.id)
                )
                or 0
            ) + 1
            rec = PlanRecord(
                id=new_id("pln"),
                project_id=t.project_id,
                root_task_id=t.id,
                version=version,
                spec=plan,
                valid=not errors,
                errors=errors,
            )
            s.add(rec)

            if errors:
                t.plan_attempts += 1
                left = max(ctx.limits.max_plan_attempts - t.plan_attempts, 0)
                blocking = [e for e in errors if e.escalate]
                self._event(
                    s,
                    t,
                    "plan.rejected",
                    {
                        "plan_id": rec.id,
                        "version": version,
                        "errors": [e.model_dump() for e in errors],
                        "attempts_left": left,
                    },
                    actor=actor,
                    run_id=run_id,
                )
                escalated = bool(blocking) or left == 0
                if escalated:
                    prompt = (
                        "The team cannot satisfy the plan rules: "
                        + "; ".join(f"{e} ({e.hint})" for e in blocking)
                        if blocking
                        else f"No valid plan after {t.plan_attempts} attempts. Last errors: "
                        + "; ".join(str(e) for e in errors)
                    )
                    self._open_human(
                        s, t, HumanRequestKind.QUESTION, prompt, ["retry_planning", "cancel"]
                    )
                return PlanResult(
                    rec.id, version, False, errors, escalated=escalated, attempts_left=left
                )

            t.acceptance_criteria = dict(plan.acceptance_criteria)
            ids = self._materialize(s, t, rec, plan)
            self._event(
                s,
                t,
                "plan.accepted",
                {
                    "plan_id": rec.id,
                    "version": version,
                    "tasks": ids,
                    "edges": [e.model_dump(mode="json") for e in plan.edges],
                },
                actor=actor,
                run_id=run_id,
            )
            # The PM is done; _settle_plan closes this task when the last planned task ends.
            self._transition(s, t, S.WAITING, f"waiting on {len(ids)} planned tasks", actor=actor)
            return PlanResult(rec.id, version, True, task_ids=ids)

    def _materialize(self, s: Session, parent: Task, rec: PlanRecord, plan: Plan) -> dict[str, str]:
        """Turn an accepted plan into Task rows (one per node) and TaskDep rows (per edge)."""
        ids = {n.id: new_id("tsk") for n in plan.nodes}
        children: list[Task] = []
        for n in plan.nodes:
            child = Task(
                id=ids[n.id],
                project_id=parent.project_id,
                plan_node_id=n.id,
                parent_id=parent.id,
                root_id=parent.root_id,
                kind=n.kind,
                title=n.title or n.id,
                objective=n.objective,
                gates=[g.value for g in n.gates],
                paths=list(n.paths),
                assignee=n.assignee,
                status=S.PENDING,
                depth=parent.depth + 1,
            )
            s.add(child)
            children.append(child)
        s.flush()
        for e in plan.edges:
            s.add(TaskDep(task_id=ids[e.target], depends_on_id=ids[e.source], kind=e.kind))
        s.flush()
        for child in children:
            self._event(
                s,
                child,
                "task.created",
                {"kind": child.kind, "node": child.plan_node_id, "assignee": child.assignee},
                actor=parent.assignee or SYSTEM,
            )
        for child in children:
            self._try_ready(s, child)
        return ids

    # ------------------------------------------------------------------ claims and runs

    def get_runnable(
        self,
        project_id: str | None = None,
        *,
        kinds: Sequence[TaskKind] | None = None,
        limit: int = 10,
    ) -> list[str]:
        """READY task ids, oldest first, for the runner to claim."""
        with self.tx() as s:
            q = select(Task.id).where(Task.status == S.READY)
            if project_id:
                q = q.where(Task.project_id == project_id)
            if kinds:
                q = q.where(Task.kind.in_([k.value for k in kinds]))
            q = q.order_by(Task.created_at, Task.id).limit(limit)
            return list(s.scalars(q))

    def claim(self, task_id: str) -> Claim | None:
        """Atomically READY -> RUNNING and open a run. None if the task was not READY."""
        with self.tx() as s:
            t = self._task(s, task_id)
            claimed = s.scalar(
                update(Task)
                .where(Task.id == t.id, Task.status == S.READY)
                .values(status=S.RUNNING, attempt=Task.attempt + 1)
                .returning(Task.id)
                .execution_options(synchronize_session=False)
            )
            if claimed is None:  # someone else claimed it first, or it wasn't READY
                return None
            s.refresh(t)
            token = secrets.token_urlsafe(32)
            run = Run(id=new_id("run"), task_id=t.id, run_token_hash=hash_token(token))
            s.add(run)
            self._event(
                s,
                t,
                "task.status",
                {"from": S.READY, "to": S.RUNNING, "reason": "claimed", "attempt": t.attempt},
                run_id=run.id,
            )
            return Claim(t.id, run.id, token, t.attempt)

    def retry(self, task_id: str, reason: str) -> str:
        """Give up on the current run (crash, timeout): READY again, or FAILED when out of tries."""
        with self.tx() as s:
            t = self._running(s, task_id)
            self._retry_or_fail(s, t, reason)
            return t.status

    def recover(self) -> list[str]:
        """Runner startup. Only the runner runs tasks, so anything still RUNNING was orphaned."""
        with self.tx() as s:
            orphans = s.scalars(select(Task).where(Task.status == S.RUNNING)).all()
            handled = []
            for t in orphans:
                if t.status == S.RUNNING:  # an earlier failure may have canceled it
                    self._retry_or_fail(s, t, "runner restarted")
                    handled.append(t.id)
            return handled

    def resolve_run_token(self, token: str) -> dict[str, Any] | None:
        """MCP identity: a live run token -> who is calling and on which task."""
        with self.tx() as s:
            run = s.scalar(select(Run).where(Run.run_token_hash == hash_token(token)))
            if run is None or run.ended_at is not None:
                return None
            t = self._task(s, run.task_id)
            return {
                "run_id": run.id,
                "task_id": t.id,
                "project_id": t.project_id,
                "agent": t.assignee,
                "kind": t.kind,
            }

    def record_usage(
        self, run_id: str, *, tokens_in: int = 0, tokens_out: int = 0, cost_usd: float = 0.0
    ) -> None:
        """Model usage for a run, charged to the project budget. Fine after the run has ended."""
        with self.tx() as s:
            run = s.get(Run, run_id)
            if run is None:
                raise NotFound(run_id)
            run.tokens_in += tokens_in
            run.tokens_out += tokens_out
            run.cost_usd += cost_usd
            t = self._task(s, run.task_id)
            p = self._project(s, t.project_id)
            p.spent_usd += cost_usd
            p.spent_tokens += tokens_in + tokens_out
            self._event(
                s,
                t,
                "run.usage",
                {"tokens": tokens_in + tokens_out, "usd": cost_usd},
                run_id=run.id,
                actor=t.assignee or SYSTEM,
            )

    # ------------------------------------------------------------------ agent actions

    def complete(self, task_id: str, summary: str, *, actor: str | None = None) -> str:
        """An agent finished: send the task to its verifiers, or mark it DONE if it has none."""
        with self.tx() as s:
            t = self._running(s, task_id)
            if t.kind in VERIFIER_KINDS:
                raise InvalidOperation(
                    f"{t.kind} tasks finish with submit_review or gate results, not complete_task"
                )
            if t.kind == TaskKind.PLAN:
                raise InvalidOperation(
                    "plan tasks finish with submit_plan; they close when every planned task ends"
                )
            t.result_summary = summary
            who = actor or t.assignee or SYSTEM
            if self._verifiers(s, t.id):
                self._transition(
                    s,
                    t,
                    S.VERIFYING,
                    "submitted for verification",
                    actor=who,
                    round=t.revision_round,
                )
            else:
                t.outcome = Outcome.PASS
                self._transition(s, t, S.DONE, "completed", actor=who)
            return t.status

    def request_human(
        self,
        task_id: str,
        prompt: str,
        options: Sequence[str] | None = None,
        *,
        actor: str | None = None,
    ) -> str:
        """An agent asks a person something; the task pauses until they answer."""
        with self.tx() as s:
            t = self._running(s, task_id)
            hr = self._open_human(
                s,
                t,
                HumanRequestKind.QUESTION,
                prompt,
                list(options or []),
                actor=actor or t.assignee or SYSTEM,
            )
            return hr.id

    def record_review(
        self, review_task_id: str, verdict: ReviewVerdict, *, actor: str | None = None
    ) -> str:
        """A reviewer's verdict. Blocking findings must cite an acceptance criterion."""
        with self.tx() as s:
            v = self._running(s, review_task_id)
            if v.kind not in REVIEW_KINDS:
                raise InvalidOperation(f"{v.id} is a {v.kind} task, not a review")
            criteria = self._root(s, v).acceptance_criteria or {}
            if verdict.verdict == Verdict.REQUEST_CHANGES:
                blocking = [f for f in verdict.findings if f.severity in ("blocker", "major")]
                if not blocking:
                    raise InvalidOperation(
                        "request_changes needs at least one blocker or major finding"
                    )
                uncited = [f.issue for f in blocking if f.criterion_id not in criteria]
                if uncited:
                    raise InvalidOperation(
                        "every blocker or major finding must cite an acceptance criterion "
                        f"({sorted(criteria)}); missing on: {uncited}"
                    )
            reviewer = actor or v.assignee or SYSTEM
            for target in self._verified_targets(s, v.id):
                s.add(
                    Review(
                        id=new_id("rev"),
                        review_task_id=v.id,
                        target_task_id=target.id,
                        reviewer=reviewer,
                        round=target.revision_round,
                        verdict=verdict.verdict,
                        findings=list(verdict.findings),
                        summary=verdict.summary,
                    )
                )
            v.outcome = Outcome.PASS if verdict.verdict == Verdict.APPROVE else Outcome.FAIL
            v.result_summary = verdict.summary
            self._transition(
                s,
                v,
                S.DONE,
                f"review: {verdict.verdict}",
                actor=reviewer,
                verdict=verdict.verdict,
                findings=[f.model_dump() for f in verdict.findings],
            )
            return v.status

    def record_gate_results(self, gate_task_id: str, results: Sequence[GateResultIn]) -> str:
        """Results of a gate's checks (pytest, ruff, policy); any failure fails the gate."""
        with self.tx() as s:
            g = self._running(s, gate_task_id)
            if g.kind != TaskKind.GATE:
                raise InvalidOperation(f"{g.id} is a {g.kind} task, not a gate")
            missing = sorted(set(g.gates) - {r.gate for r in results})
            if missing:
                raise InvalidOperation(f"missing results for gate(s) {missing}")
            for target in self._verified_targets(s, g.id):
                for r in results:
                    s.add(
                        GateResult(
                            id=new_id("gat"),
                            gate_task_id=g.id,
                            target_task_id=target.id,
                            round=target.revision_round,
                            gate=r.gate,
                            passed=r.passed,
                            output=r.output[-20_000:],
                            duration_ms=r.duration_ms,
                        )
                    )
            passed = all(r.passed for r in results)
            g.outcome = Outcome.PASS if passed else Outcome.FAIL
            self._transition(
                s,
                g,
                S.DONE,
                "gates passed" if passed else "gates failed",
                results={r.gate: r.passed for r in results},
            )
            return g.status

    # ------------------------------------------------------------------ humans

    def open_human_requests(self, project_id: str) -> list[dict[str, Any]]:
        """The human inbox: unanswered questions, approvals and escalations, oldest first."""
        with self.tx() as s:
            rows = s.scalars(
                select(HumanRequest)
                .where(HumanRequest.project_id == project_id, HumanRequest.status == "open")
                .order_by(HumanRequest.created_at)
            )
            return [
                {
                    "id": r.id,
                    "task_id": r.task_id,
                    "kind": r.kind,
                    "prompt": r.prompt,
                    "options": r.options,
                }
                for r in rows
            ]

    def resolve_human(self, request_id: str, response: str, *, responder: str = HUMAN) -> str:
        """Apply a person's answer: approve, reject, override, retry planning, cancel or reply."""
        with self.tx() as s:
            hr = s.get(HumanRequest, request_id)
            if hr is None:
                raise NotFound(request_id)
            if hr.status != "open":
                raise InvalidOperation(f"request {hr.id} was already answered")
            if hr.options and response not in hr.options:
                raise InvalidOperation(f"response must be one of {hr.options}")
            hr.response = response
            hr.status = "answered"
            t = self._task(s, hr.task_id)
            self._event(
                s,
                t,
                "human.responded",
                {"request_id": hr.id, "response": response},
                actor=responder,
            )
            if hr.kind == HumanRequestKind.APPROVAL:
                if response == "approve":
                    t.outcome = Outcome.PASS
                    self._transition(s, t, S.DONE, "approved", actor=responder)
                else:
                    t.outcome = Outcome.FAIL
                    self._transition(s, t, S.FAILED, "rejected at approval", actor=responder)
            elif hr.kind == HumanRequestKind.ESCALATION:
                if response == "override_approve":
                    for v in self._verifiers(s, t.id):
                        if v.outcome == Outcome.FAIL:
                            v.outcome = Outcome.WAIVED
                            self._event(
                                s, v, "verification.waived", {"target": t.id}, actor=responder
                            )
                    t.outcome = Outcome.WAIVED
                    self._transition(s, t, S.DONE, "human override", actor=responder)
                else:
                    t.outcome = Outcome.FAIL
                    self._transition(s, t, S.FAILED, "failed by human", actor=responder)
            elif response == "cancel":
                self._cancel(s, t, "canceled by human", actor=responder)
            else:
                if t.kind == TaskKind.PLAN:
                    t.plan_attempts = 0
                self._transition(s, t, S.READY, "human answered", actor=responder)
            return t.status

    # ------------------------------------------------------------------ admin

    def cancel(self, task_id: str, reason: str, *, actor: str = HUMAN) -> str:
        """Admin: cancel a task (the cascade cancels everything that depends on it)."""
        with self.tx() as s:
            t = self._task(s, task_id)
            self._cancel(s, t, reason, actor=actor)
            return t.status

    def fail(self, task_id: str, reason: str, *, actor: str = SYSTEM) -> str:
        """Admin: mark a task FAILED by hand."""
        with self.tx() as s:
            t = self._task(s, task_id)
            t.outcome = Outcome.FAIL
            self._transition(s, t, S.FAILED, reason, actor=actor)
            return t.status

    # ------------------------------------------------------------------ queries

    def get_task(self, task_id: str) -> dict[str, Any]:
        """One task as a plain dict (status, outcome, deps, summary) for tools and the UI."""
        with self.tx() as s:
            return self._snapshot(s, self._task(s, task_id))

    def task_tree(self, root_id: str) -> list[dict[str, Any]]:
        """The root task and every task under it, as dicts."""
        with self.tx() as s:
            rows = s.scalars(
                select(Task).where(Task.root_id == root_id).order_by(Task.depth, Task.id)
            )
            return [self._snapshot(s, t) for t in rows]

    def verify_events(self, project_id: str) -> VerifyResult:
        """Re-hash this project's event chain; reports the first tampered event, if any."""
        with self.tx() as s:
            return self.events.verify_chain(s, project_id)

    # ================================================================== internals

    def _project(self, s: Session, project_id: str) -> Project:
        """Load a project or raise NotFound."""
        p = s.get(Project, project_id)
        if p is None:
            raise NotFound(f"project {project_id}")
        return p

    def _task(self, s: Session, task_id: str) -> Task:
        """Load a task or raise NotFound."""
        t = s.get(Task, task_id)
        if t is None:
            raise NotFound(f"task {task_id}")
        return t

    def _root(self, s: Session, t: Task) -> Task:
        """The task's root: the original objective, which holds the acceptance criteria."""
        return self._task(s, t.root_id)

    def _plan_task(self, s: Session, task_id: str) -> Task:
        """Load a task and check it's a plan task."""
        t = self._task(s, task_id)
        if t.kind != TaskKind.PLAN:
            raise InvalidOperation(f"{t.id} is a {t.kind} task; only plan tasks take plans")
        return t

    def _running(self, s: Session, task_id: str) -> Task:
        """Load a task and check it's RUNNING (most agent actions require that)."""
        t = self._task(s, task_id)
        if t.status != S.RUNNING:
            raise InvalidOperation(f"task {t.id} is {t.status}, not RUNNING")
        return t

    def _validator(self, s: Session, t: Task) -> PlanValidator:
        """The plan validator for this project's policy pack, built once and cached."""
        pack = self._project(s, t.project_id).policy_pack
        if pack not in self._validators:
            self._validators[pack] = PlanValidator.for_pack(pack)
        return self._validators[pack]

    def _event(
        self,
        s: Session,
        t: Task,
        type: str,
        payload: dict[str, Any],
        *,
        actor: str = SYSTEM,
        run_id: str | None = None,
    ) -> None:
        """Append a hash-chained event about task t."""
        self.events.append(
            s,
            project_id=t.project_id,
            task_id=t.id,
            run_id=run_id,
            type=type,
            actor=actor,
            payload=payload,
        )

    def _snapshot(self, s: Session, t: Task) -> dict[str, Any]:
        """Serialize a task for get_task / task_tree."""
        deps = [
            {"on": d.depends_on_id, "kind": d.kind}
            for d in s.scalars(select(TaskDep).where(TaskDep.task_id == t.id))
        ]
        return {
            "id": t.id,
            "node": t.plan_node_id,
            "kind": t.kind,
            "title": t.title,
            "status": t.status,
            "outcome": t.outcome,
            "assignee": t.assignee,
            "paths": t.paths,
            "parent_id": t.parent_id,
            "attempt": t.attempt,
            "revision_round": t.revision_round,
            "depends_on": deps,
            "summary": t.result_summary,
        }

    # -- graph helpers ---------------------------------------------------------------

    def _upstream(self, s: Session, task_id: str) -> list[tuple[TaskDep, Task]]:
        """What this task depends on, as (edge, task) pairs."""
        rows = s.execute(
            select(TaskDep, Task)
            .join(Task, Task.id == TaskDep.depends_on_id)
            .where(TaskDep.task_id == task_id)
        ).all()
        return [(d, t) for d, t in rows]

    def _downstream(self, s: Session, task_id: str) -> list[tuple[TaskDep, Task]]:
        """What depends on this task, as (edge, task) pairs."""
        rows = s.execute(
            select(TaskDep, Task)
            .join(Task, Task.id == TaskDep.task_id)
            .where(TaskDep.depends_on_id == task_id)
        ).all()
        return [(d, t) for d, t in rows]

    def _verifiers(self, s: Session, target_id: str) -> list[Task]:
        """The verifier tasks (gate, review, security_review) that check this task."""
        return [t for d, t in self._downstream(s, target_id) if d.kind == EdgeKind.VERIFIES]

    def _verified_targets(self, s: Session, verifier_id: str) -> list[Task]:
        """The work task(s) this verifier checks."""
        return [t for d, t in self._upstream(s, verifier_id) if d.kind == EdgeKind.VERIFIES]

    @staticmethod
    def _dep_satisfied(dep: TaskDep, src: Task) -> bool:
        """One dependency met? verifies: source VERIFYING. blocks: source DONE and passing."""
        if dep.kind == EdgeKind.VERIFIES:
            return src.status == S.VERIFYING
        return src.status == S.DONE and src.outcome in PASSING_OUTCOMES

    # -- state changes ---------------------------------------------------------------

    def _transition(
        self, s: Session, t: Task, to: TaskStatus, reason: str, *, actor: str = SYSTEM, **extra: Any
    ) -> None:
        """The only way a status changes: check the state machine, close the run, log, react."""
        frm = TaskStatus(t.status)
        assert_transition(frm, to)
        if frm == S.DONE and t.kind not in VERIFIER_KINDS:
            raise IllegalTransition(f"only verifier tasks can be reopened ({t.id} is {t.kind})")
        t.status = to
        if frm == S.RUNNING:
            self._close_run(s, t, reason)
        self._event(
            s, t, "task.status", {"from": frm, "to": to, "reason": reason, **extra}, actor=actor
        )
        self._after_change(s, t)

    def _after_change(self, s: Session, t: Task) -> None:
        """React to a status change: settle verification, wake/cancel downstream, settle plan."""
        to = t.status
        # 1. A verifier finished: settle the work it checks before anything downstream.
        if t.kind in VERIFIER_KINDS and to in TERMINAL_STATUSES:
            for target in self._verified_targets(s, t.id):
                self._evaluate_verification(s, target)
        # 2. Wake or cancel downstream tasks. Verifiers are leaves (STR-005), so this only
        #    ever follows work -> work and work -> verifier edges.
        if to in (S.VERIFYING, S.DONE):
            for _, child in self._downstream(s, t.id):
                self._try_ready(s, child)
        elif to in (S.CANCELED, S.FAILED):
            for _, child in self._downstream(s, t.id):
                if child.status not in TERMINAL_STATUSES:
                    self._cancel(s, child, f"upstream task {t.id} is {to}")
        # 3. The last planned task to finish closes the plan task.
        if to in TERMINAL_STATUSES and t.parent_id:
            self._settle_plan(s, t.parent_id)

    def _try_ready(self, s: Session, t: Task) -> None:
        """PENDING -> READY once all dependencies are met; approval tasks then ask the human."""
        if t.status != S.PENDING:
            return
        if all(self._dep_satisfied(d, src) for d, src in self._upstream(s, t.id)):
            self._transition(s, t, S.READY, "dependencies satisfied")
            if t.kind == TaskKind.APPROVAL:
                root = self._root(s, t)
                self._open_human(
                    s,
                    t,
                    HumanRequestKind.APPROVAL,
                    f"All checks passed for: {root.objective}. Approve the change for merge?",
                    ["approve", "reject"],
                )

    def _evaluate_verification(self, s: Session, target: Task) -> None:
        """Once every verifier is done: pass the work, send it back for revision, or escalate."""
        if target.status != S.VERIFYING:
            return
        verifiers = self._verifiers(s, target.id)
        if any(v.status not in TERMINAL_STATUSES for v in verifiers):
            return  # wait for every verdict, so one revision addresses all findings
        broken = [v.id for v in verifiers if v.status != S.DONE]
        failed = [v for v in verifiers if v.outcome == Outcome.FAIL]
        if broken:
            self._open_human(
                s,
                target,
                HumanRequestKind.ESCALATION,
                f"Verifier task(s) {broken} could not finish for {target.title}.",
                ["fail"],
            )
        elif not failed:
            target.outcome = Outcome.PASS
            self._transition(s, target, S.DONE, "all verifiers passed", round=target.revision_round)
        elif target.revision_round < self._project(s, target.project_id).limits.max_revision_rounds:
            target.revision_round += 1
            self._transition(
                s,
                target,
                S.READY,
                "changes requested",
                round=target.revision_round,
                failed_verifiers=[v.id for v in failed],
            )
            for v in verifiers:
                v.outcome = None
                self._transition(s, v, S.PENDING, f"re-verify {target.id} after revision")
        else:
            self._open_human(
                s,
                target,
                HumanRequestKind.ESCALATION,
                f"{target.title} still fails verification after {target.revision_round} "
                "revision round(s).",
                ["override_approve", "fail"],
            )

    def _settle_plan(self, s: Session, plan_task_id: str) -> None:
        """Close a WAITING plan task once every planned task has ended. PASS only if all passed."""
        plan = self._task(s, plan_task_id)
        if plan.status != S.WAITING:
            return
        children = s.scalars(select(Task).where(Task.parent_id == plan.id)).all()
        if any(c.status not in TERMINAL_STATUSES for c in children):
            return
        bad = [c for c in children if c.status != S.DONE or c.outcome not in PASSING_OUTCOMES]
        if bad:
            summary = "not passed: " + ", ".join(f"{c.plan_node_id} ({c.status})" for c in bad)
            plan.outcome, plan.result_summary = Outcome.FAIL, summary
            self._transition(s, plan, S.FAILED, summary)
        else:
            summary = f"all {len(children)} planned tasks passed"
            plan.outcome, plan.result_summary = Outcome.PASS, summary
            self._transition(s, plan, S.DONE, summary)

    def _retry_or_fail(self, s: Session, t: Task, reason: str) -> None:
        """Requeue a task whose run died, or fail it once it has used all its attempts."""
        max_attempts = self._project(s, t.project_id).limits.max_attempts
        if t.attempt >= max_attempts:
            t.outcome = Outcome.FAIL
            self._transition(s, t, S.FAILED, f"{reason}; {t.attempt}/{max_attempts} attempts used")
        else:
            self._transition(s, t, S.READY, f"{reason}; will retry", attempt=t.attempt)

    def _close_run(self, s: Session, t: Task, reason: str) -> None:
        """A run lives exactly as long as its task is RUNNING; closing it kills the token."""
        for run in s.scalars(select(Run).where(Run.task_id == t.id, Run.ended_at.is_(None))):
            run.ended_at = now_iso()
            run.end_reason = reason

    def _cancel(self, s: Session, t: Task, reason: str, *, actor: str = SYSTEM) -> None:
        """Cancel a task and withdraw any open human requests on it."""
        if t.status in TERMINAL_STATUSES:
            return
        for hr in s.scalars(
            select(HumanRequest).where(HumanRequest.task_id == t.id, HumanRequest.status == "open")
        ):
            hr.status = "withdrawn"
        self._transition(s, t, S.CANCELED, reason, actor=actor)

    def _open_human(
        self,
        s: Session,
        t: Task,
        kind: HumanRequestKind,
        prompt: str,
        options: list[str],
        *,
        actor: str = SYSTEM,
    ) -> HumanRequest:
        """Create a human request and pause the task (AWAITING_HUMAN) until it's answered."""
        hr = HumanRequest(
            id=new_id("hum"),
            project_id=t.project_id,
            task_id=t.id,
            kind=kind,
            prompt=prompt,
            options=options,
        )
        s.add(hr)
        self._event(
            s,
            t,
            "human.requested",
            {"request_id": hr.id, "kind": kind, "prompt": prompt, "options": options},
            actor=actor,
        )
        self._transition(s, t, S.AWAITING_HUMAN, f"{kind} requested", actor=actor)
        return hr
