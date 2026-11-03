"""The agent MCP server: run-token identity, tool allow-lists, idempotent calls."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mcp import types
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openai import OpenAIChatModelSettings

import orch.core.orchestrator as mod
from orch.core.orchestrator import TaskOrchestrator
from orch.core.team import TeamRegistry
from orch.domain.enums import TaskStatus
from orch.domain.schemas import Plan
from orch.mcp.server import AgentMCP, Caller, serve
from orch.runtime.spike import SCENARIOS, run_trial, scripted_pm

PM_TOOLS = {
    "list_agents",
    "validate_plan",
    "submit_plan",
    "get_task_status",
    "get_task_tree",
    "get_remaining_budget",
    "read_file",
    "list_files",
    "search_code",
    "request_human_input",
}


def start(orch: TaskOrchestrator, team: TeamRegistry):
    pid = orch.create_project("p")
    orch.register_team(pid, team)
    root = orch.create_root_task(pid, "Add transfers", acceptance_criteria={"AC-1": "x"})
    claim = orch.claim(root)
    assert claim is not None
    return root, claim


def run(coro):
    return asyncio.run(coro)


def text(result: types.CallToolResult) -> str:
    part = result.content[0]
    assert isinstance(part, types.TextContent)
    return part.text


def body(result: types.CallToolResult) -> Any:
    return json.loads(text(result))


def test_token_identifies_the_caller(orch: TaskOrchestrator, team: TeamRegistry):
    root, claim = start(orch, team)
    c = run(AgentMCP(orch).caller(claim.run_token))
    assert (c.task_id, c.agent, c.kind) == (root, "project_manager", "plan")
    assert c.tools >= PM_TOOLS


def test_unknown_or_missing_token_is_denied(orch: TaskOrchestrator, team: TeamRegistry):
    start(orch, team)
    srv = AgentMCP(orch)
    for token in ["", "not-a-token"]:
        try:
            run(srv.caller(token))
        except Exception as e:
            assert "token" in str(e)
        else:
            raise AssertionError("denied token was accepted")


def test_tool_not_on_the_allow_list_is_refused(orch: TaskOrchestrator, team: TeamRegistry):
    _, claim = start(orch, team)
    srv = AgentMCP(orch)
    c = run(srv.caller(claim.run_token))
    narrowed = Caller(**{**c.__dict__, "tools": frozenset({"list_agents"})})
    r = run(srv.call(narrowed, "get_task_tree", {}))
    assert r.is_error and "may not call get_task_tree" in text(r)
    r = run(srv.call(c, "write_file", {"path": "x"}))  # not a tool this server has (yet)
    assert r.is_error


def test_repeated_call_id_submits_once(orch: TaskOrchestrator, team: TeamRegistry, good_plan: Plan):
    root, claim = start(orch, team)
    srv = AgentMCP(orch)
    c = run(srv.caller(claim.run_token))
    args = good_plan.model_dump(mode="json")
    first = run(srv.call(c, "submit_plan", args, call_id="call-1"))
    again = run(srv.call(c, "submit_plan", args, call_id="call-1"))
    assert body(first)["accepted"] and body(again) == body(first)
    assert orch.get_task(root)["status"] == TaskStatus.WAITING
    assert len(orch.task_tree(root)) == 1 + len(good_plan.nodes)  # materialized once


def test_bad_arguments_come_back_as_a_tool_error(orch: TaskOrchestrator, team: TeamRegistry):
    _, claim = start(orch, team)
    srv = AgentMCP(orch)
    c = run(srv.caller(claim.run_token))
    r = run(srv.call(c, "validate_plan", {"nodes": "nope"}))
    assert r.is_error
    r = run(srv.call(c, "get_task_status", {"task_id": "tsk_elsewhere"}))
    assert r.is_error and "not a task in your objective" in text(r)


def test_over_http_a_finished_run_loses_its_tools(
    orch: TaskOrchestrator, team: TeamRegistry, good_plan: Plan
):
    _, claim = start(orch, team)

    async def go():
        async with serve(orch) as url:
            async with MCPToolset(url, auth=claim.run_token) as ts:
                assert {t.name for t in await ts.client.list_tools()} == PM_TOOLS
                r = await ts.client.call_tool("submit_plan", good_plan.model_dump(mode="json"))
                assert r.structured_content and r.structured_content["accepted"]
            # The plan task left RUNNING, so its token is dead.
            async with MCPToolset(url, auth=claim.run_token) as ts:
                assert await ts.client.list_tools() == []

    run(go())


def test_spike_over_mcp_matches_in_process(tmp_path: Path):
    def trial(sub: str, use_mcp: bool):
        (tmp_path / sub).mkdir()
        return run(
            run_trial(
                SCENARIOS["transfers"],
                0,
                model=scripted_pm(),
                workdir=tmp_path / sub,
                use_mcp=use_mcp,
                with_validate=True,
                settings=OpenAIChatModelSettings(),
                timeout_s=30,
                price_as="muse-spark-1.3",
            )
        )

    local, remote = trial("local", False), trial("mcp", True)
    assert remote.outcome == local.outcome == "accepted"
    assert remote.tool_calls == local.tool_calls
    assert remote.rule_codes == local.rule_codes == ["SOD-001"]
    assert remote.retries == local.retries == 0


def test_arguments_are_validated_like_in_process(orch: TaskOrchestrator, team: TeamRegistry):
    root, claim = start(orch, team)
    srv = AgentMCP(orch)
    c = run(srv.caller(claim.run_token))
    r = run(srv.call(c, "request_human_input", {"prompt": "Ship?", "options": "approve"}))
    assert r.is_error and "options" in text(r)
    assert orch.get_task(root)["status"] == TaskStatus.RUNNING  # nothing was asked


def test_internal_error_is_logged_and_can_be_retried(
    orch: TaskOrchestrator, team: TeamRegistry, monkeypatch
):
    _, claim = start(orch, team)
    srv = AgentMCP(orch)
    c = run(srv.caller(claim.run_token))
    real = orch.remaining_budget
    monkeypatch.setattr(orch, "remaining_budget", lambda pid: (_ for _ in ()).throw(OSError()))
    r = run(srv.call(c, "get_remaining_budget", {}, call_id="call-1"))
    assert r.is_error and "internal error (OSError)" in text(r)
    monkeypatch.setattr(orch, "remaining_budget", real)
    r = run(srv.call(c, "get_remaining_budget", {}, call_id="call-1"))  # not stuck "started"
    assert not r.is_error and body(r)["budget_usd"] == 5.0


def test_call_id_reused_for_another_call_is_refused(orch: TaskOrchestrator, team: TeamRegistry):
    _, claim = start(orch, team)
    srv = AgentMCP(orch)
    c = run(srv.caller(claim.run_token))
    assert not run(srv.call(c, "list_agents", {}, call_id="call-1")).is_error
    r = run(srv.call(c, "get_task_tree", {}, call_id="call-1"))
    assert r.is_error and "different call" in text(r)


def test_same_key_inserted_concurrently_is_reported_as_running(
    orch: TaskOrchestrator, team: TeamRegistry, monkeypatch
):
    _, claim = start(orch, team)
    srv = AgentMCP(orch)
    c = run(srv.caller(claim.run_token))
    key = f"{c.task_id}:call-1"
    orch.begin_tool_call(c.run_id, c.task_id, key, "list_agents", {})
    # Another request's insert landed between our lookup and our insert.
    real_select = mod.select
    monkeypatch.setattr(mod, "select", lambda *a: real_select(*a).where(mod.ToolCall.id == "none"))
    try:
        orch.begin_tool_call(c.run_id, c.task_id, key, "list_agents", {})
    except Exception as e:
        assert "still running" in str(e)
    else:
        raise AssertionError("duplicate insert was not caught")
