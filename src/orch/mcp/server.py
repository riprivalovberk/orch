"""The agent MCP server: the only way an agent reaches the orchestrator.

One Streamable HTTP server on 127.0.0.1, run inside the runner process. Every request
carries ``Authorization: Bearer <run token>`` (handed out by ``claim``); the token says who
is calling and on which task, so no tool takes an agent name or a task id for "me".

Each call: resolve the token -> check the agent's tool allow-list (team YAML ``tools``)
-> log it in ``tool_calls`` (a repeated idempotency key returns the stored result instead
of acting twice) -> run the tool -> store the result. Orchestrator errors come back to the
model as tool errors it can fix and retry.
"""

from __future__ import annotations

import json
import logging
import socket
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
import uvicorn
from mcp import types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from pydantic import ValidationError
from pydantic_ai import ModelRetry, Tool
from pydantic_core import SchemaValidator
from starlette.applications import Starlette
from starlette.requests import Request

from orch.core.orchestrator import OrchestratorError, TaskOrchestrator
from orch.ids import new_id
from orch.mcp.worker_tools import WORKER_TOOLS, WorkerDeps
from orch.runtime import pm_tools as pm
from orch.runtime.pm_tools import PMDeps
from orch.runtime.workspace import Workspace

NAME = "orch-agent-mcp"
#: Clients put the model's tool-call id here (request ``_meta``) to make a call idempotent.
CALL_ID_META = "orch/tool_call_id"
SHUTDOWN_GRACE_S = 3

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Caller:
    """Who is calling, from the run token. Never from the tool arguments."""

    run_id: str
    task_id: str
    project_id: str
    agent: str
    kind: str
    tools: frozenset[str]
    write_paths: tuple[str, ...] = ()  # the agent's scope (team file)
    paths: tuple[str, ...] = ()  # what the plan says this task writes


@dataclass(frozen=True)
class Call:
    """One tool call's context: the caller and what the server can reach for it."""

    orch: TaskOrchestrator
    caller: Caller
    ws: Workspace | None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    validator: SchemaValidator  # the same argument validation Pydantic AI does in-process
    impl: Callable[..., Any]  # impl(deps, **validated args)
    deps: Callable[[Call], Any]  # the deps impl takes, built for this call


class Denied(Exception):
    """The caller may not make this call (bad token, tool not on its allow-list)."""


def pm_deps(call: Call) -> PMDeps:
    c = call.caller
    return PMDeps(call.orch, c.project_id, c.task_id, c.run_id)


def worker_deps(call: Call) -> WorkerDeps:
    if call.ws is None:
        raise ModelRetry("this server has no workspace; file tools are unavailable")
    return WorkerDeps(call.orch, call.caller, call.ws)


@dataclass(frozen=True)
class _Ctx:
    """Stands in for Pydantic AI's RunContext: the tools only read ``ctx.deps``."""

    deps: Any


def _spec(
    fn: Callable[..., Any], deps: Callable[[Call], Any], impl: Callable[..., Any] | None = None
) -> ToolSpec:
    """Name, description, schema and validator from the Pydantic AI tool, so MCP matches
    what an in-process agent would see. Without ``impl`` the tool function itself runs."""
    t = Tool(fn, takes_ctx=True)
    d = t.tool_def

    def run_fn(dep: Any, **kw: Any) -> Any:
        return fn(_Ctx(dep), **kw)

    return ToolSpec(
        d.name,
        d.description or "",
        d.parameters_json_schema,
        t.function_schema.validator,
        impl or run_fn,
        deps,
    )


TOOLS: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        _spec(pm.list_agents, pm_deps, pm.do_list_agents),
        _spec(pm.validate_plan, pm_deps, pm.do_validate_plan),
        _spec(pm.submit_plan, pm_deps, pm.do_submit_plan),
        _spec(pm.get_task_status, pm_deps, pm.do_get_task_status),
        _spec(pm.get_task_tree, pm_deps, pm.do_get_task_tree),
        _spec(pm.get_remaining_budget, pm_deps, pm.do_get_remaining_budget),
        _spec(pm.request_human_input, pm_deps, pm.do_request_human_input),
        *(_spec(fn, worker_deps) for fn in WORKER_TOOLS),
    ]
}


def _token(ctx: ServerRequestContext[Any, Any]) -> str:
    req = ctx.request
    auth = req.headers.get("authorization", "") if isinstance(req, Request) else ""
    scheme, _, token = auth.partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _call_id(params: types.CallToolRequestParams) -> str | None:
    meta: Mapping[str, Any] | None = params.meta  # type: ignore[assignment]
    value = meta.get(CALL_ID_META) if meta else None
    return str(value) if value else None


def _result(payload: Any, *, error: bool = False) -> types.CallToolResult:
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=payload if isinstance(payload, dict) and not error else None,
        is_error=error,
    )


class AgentMCP:
    """Turns MCP requests from agents into orchestrator calls for the caller's run."""

    def __init__(
        self,
        orch: TaskOrchestrator,
        ws: Workspace | None = None,
        tools: Mapping[str, ToolSpec] = TOOLS,
    ):
        self.orch = orch
        self.ws = ws
        self.tools = dict(tools)
        self.server: Server[Any] = Server(
            NAME,
            version="0.1.0",
            instructions="Tools for one task. Your identity comes from your run token.",
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )

    # ------------------------------------------------------------------ identity

    async def caller(self, token: str) -> Caller:
        if not token:
            raise Denied("missing run token (Authorization: Bearer <token>)")
        who = await anyio.to_thread.run_sync(self.orch.resolve_run_token, token)
        if who is None:
            raise Denied("run is not active: the token is unknown or its task left RUNNING")
        return Caller(
            run_id=who["run_id"],
            task_id=who["task_id"],
            project_id=who["project_id"],
            agent=who["agent"],
            kind=who["kind"],
            tools=frozenset(who["tools"]),
            write_paths=tuple(who["write_paths"]),
            paths=tuple(who["paths"]),
        )

    def allowed(self, c: Caller) -> list[ToolSpec]:
        return [s for name, s in self.tools.items() if name in c.tools]

    # ------------------------------------------------------------------ handlers

    async def _list_tools(
        self, ctx: ServerRequestContext[Any, Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        try:
            c = await self.caller(_token(ctx))
        except Denied:
            return types.ListToolsResult(tools=[])
        return types.ListToolsResult(
            tools=[
                types.Tool(name=s.name, description=s.description, input_schema=s.input_schema)
                for s in self.allowed(c)
            ]
        )

    async def _call_tool(
        self, ctx: ServerRequestContext[Any, Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        try:
            c = await self.caller(_token(ctx))
        except Denied as e:
            return _result(str(e), error=True)
        return await self.call(c, params.name, dict(params.arguments or {}), _call_id(params))

    async def call(
        self, c: Caller, name: str, args: dict[str, Any], call_id: str | None = None
    ) -> types.CallToolResult:
        spec = self.tools.get(name)
        if spec is None or name not in c.tools:
            return _result(f"{c.agent} may not call {name}", error=True)
        key = f"{c.task_id}:{call_id or new_id('call')}"
        try:
            prev = await anyio.to_thread.run_sync(
                self.orch.begin_tool_call, c.run_id, c.task_id, key, name, args
            )
        except OrchestratorError as e:
            return _result(str(e), error=True)
        if prev is not None:  # a retry of a call that already ran: same answer, no new effect
            return _result(prev["result"], error=prev["status"] == "error")
        # Run and log the result in one worker thread: the thread finishes even if this
        # request is cancelled (client gone), so the log never stays at "started".
        status, out = await anyio.to_thread.run_sync(self._execute, c, spec, key, args)
        return _result(out, error=status != "ok")

    def _execute(
        self, c: Caller, spec: ToolSpec, key: str, args: dict[str, Any]
    ) -> tuple[str, Any]:
        try:
            kwargs = spec.validator.validate_python(args)
            out = spec.impl(spec.deps(Call(self.orch, c, self.ws)), **kwargs)
            status = "ok"
        except (ModelRetry, OrchestratorError, ValidationError) as e:
            status, out = "error", e.message if isinstance(e, ModelRetry) else str(e)
        except Exception as e:  # e.g. "database is locked": nothing committed, safe to rerun
            log.exception("tool %s failed for %s", spec.name, c.agent)
            status, out = "failed", f"internal error ({type(e).__name__}); call the tool again"
        try:
            self.orch.end_tool_call(key, status, out)
        except Exception:
            log.exception("could not store the result of tool call %s", key)
        return status, out

    # ------------------------------------------------------------------ HTTP

    def app(self) -> Starlette:
        return self.server.streamable_http_app(host="127.0.0.1")


@asynccontextmanager
async def serve(
    orch: TaskOrchestrator, ws: Workspace | None = None, *, port: int = 0
) -> AsyncIterator[str]:
    """Run the agent MCP server on 127.0.0.1 for the duration of the block; yields its URL.

    port=0 picks a free port, so parallel runs (and tests) never collide.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:  # closed on any exit
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
        config = uvicorn.Config(
            AgentMCP(orch, ws).app(),
            log_level="warning",
            lifespan="on",
            # An agent that never closes its stream (killed mid-run) must not hold up shutdown.
            timeout_graceful_shutdown=SHUTDOWN_GRACE_S,
        )
        server = uvicorn.Server(config)
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: server.serve(sockets=[sock]))
            try:
                with anyio.fail_after(10):  # uvicorn returns without starting on a startup error
                    while not server.started:
                        await anyio.sleep(0.01)
                yield url
            finally:
                server.should_exit = True
