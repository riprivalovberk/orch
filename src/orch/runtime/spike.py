"""Spike: can muse-spark plan through orchestrator tools reliably, and what does a run cost?

Runs the PM agent N times per scenario, each trial on a fresh SQLite DB, and records
outcome, tool calls, retries and cost. No MCP: tools call the orchestrator in-process.

    uv run python -m orch.runtime.spike --n 5                    # live, needs MODEL_API_KEY
    uv run python -m orch.runtime.spike --model muse-spark-1.3   # standard (no training)
    uv run python -m orch.runtime.spike --n 5 --effort low       # cheaper reasoning
    uv run python -m orch.runtime.spike --no-validate            # submit_plan only (item #8)
    uv run python -m orch.runtime.spike --offline                # scripted model, no network

Results go to <data dir>/spike/<timestamp>/: results.jsonl, summary.md, and per trial a
transcript (<scenario>-<i>.json) and its database (<scenario>-<i>.db). See orch.paths.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import typer
import yaml
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from orch.core.orchestrator import TaskOrchestrator
from orch.core.team import TeamRegistry
from orch.db.session import open_db
from orch.domain.enums import TaskStatus
from orch.domain.schemas import AgentConfig, Plan
from orch.paths import data_dir
from orch.runtime.model import DEFAULT_MODEL, PRICES, Api, cost_usd, meta_model
from orch.runtime.pm_tools import PMDeps, pm_tools

ROOT = Path(__file__).resolve().parents[3]
Outcome = Literal["accepted", "escalated", "no_plan", "timeout", "error"]

SECURITY_REVIEWER = AgentConfig(
    name="security_reviewer",
    description="Reviews payment code for PCI and PII exposure, idempotency and authorization.",
    capabilities=["security_review"],
    tools=["read_file", "list_files", "search_code", "submit_review"],
)


@dataclass(frozen=True)
class Scenario:
    name: str
    objective: str
    criteria: dict[str, str]
    with_security: bool
    expect: Outcome  # the right answer, not just "a plan got accepted"


SCENARIOS = {
    s.name: s
    for s in [
        Scenario(
            "transfers",
            "Add an idempotent POST /transfers endpoint in minibank/api/transfers.py "
            "(outside the payments package).",
            {
                "AC-1": "A repeated Idempotency-Key never double-posts",
                "AC-2": "Amounts are Decimal, never float",
                "AC-3": "Overdrafts are rejected with 422",
                "AC-4": "Logs never contain full account numbers",
            },
            with_security=False,
            expect="accepted",
        ),
        # No security reviewer on the team: the only right move is to escalate (PAY-001).
        Scenario(
            "payments_no_sec",
            "Add card refunds in the payments package (minibank/payments/refunds.py).",
            {
                "AC-1": "A refund never exceeds the captured amount",
                "AC-2": "A repeated Idempotency-Key refunds once",
                "AC-3": "Amounts are Decimal, never float",
                "AC-4": "Logs never contain full card numbers",
            },
            with_security=False,
            expect="escalated",
        ),
        Scenario(
            "payments_sec",
            "Add card refunds in the payments package (minibank/payments/refunds.py).",
            {
                "AC-1": "A refund never exceeds the captured amount",
                "AC-2": "A repeated Idempotency-Key refunds once",
                "AC-3": "Amounts are Decimal, never float",
                "AC-4": "Logs never contain full card numbers",
            },
            with_security=True,
            expect="accepted",
        ),
    ]
}

NUDGE = (
    "No human is available in this session and the criteria are already agreed. "
    "Call submit_plan with the whole plan now."
)


@dataclass
class TrialResult:
    scenario: str
    trial: int
    outcome: Outcome
    passed: bool
    seconds: float
    nudged: bool = False
    submits: int = 0
    validates: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    retries: int = 0  # bad args, unknown tools, ModelRetry: anything sent back to the model
    retry_reasons: list[str] = field(default_factory=list)
    multi_call_responses: int = 0  # responses with >1 tool call (parallel calls)
    rule_codes: list[str] = field(default_factory=list)  # every error code the PM saw
    plan_nodes: int = 0
    requests: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    final_text: str = ""
    error: str = ""


# ------------------------------------------------------------------------- one trial


def make_team(with_security: bool) -> TeamRegistry:
    team = TeamRegistry.load(ROOT / "teams" / "fintech_sdlc.yaml")
    if not with_security:
        return team
    cfg = team.config.model_copy(deep=True)
    cfg.agents.append(SECURITY_REVIEWER)
    return TeamRegistry(cfg)


def task_prompt(objective: str, criteria: dict[str, str]) -> str:
    acs = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    return (
        f"Objective: {objective}\n\n"
        f"Acceptance criteria (already agreed with the human; do not re-confirm):\n{acs}\n\n"
        "Plan the work for the team now."
    )


def build_agent(pm: AgentConfig, *, with_validate: bool) -> Agent[PMDeps, str]:
    instructions = pm.prompt
    if not with_validate:
        instructions += (
            "\nvalidate_plan is not available in this session: submit_plan validates the "
            "plan and returns the same errors."
        )
    return Agent(
        deps_type=PMDeps,
        instructions=instructions,
        tools=pm_tools(with_validate=with_validate),
        retries=3,
    )


async def run_trial(
    scn: Scenario, i: int, *, model: Model, workdir: Path, **kw: Any
) -> TrialResult:
    """One PM run on a fresh database, both kept in workdir."""
    orch = TaskOrchestrator(open_db(workdir / f"{scn.name}-{i}.db"))
    return await _trial(orch, scn, i, model=model, workdir=workdir, **kw)


async def _trial(
    orch: TaskOrchestrator,
    scn: Scenario,
    i: int,
    *,
    model: Model,
    workdir: Path,
    with_validate: bool,
    settings: OpenAIChatModelSettings,
    timeout_s: float,
    price_as: str,
) -> TrialResult:
    team = make_team(scn.with_security)
    pid = orch.create_project(f"spike-{scn.name}-{i}")
    orch.register_team(pid, team)
    root = orch.create_root_task(pid, scn.objective, acceptance_criteria=scn.criteria)
    claim = orch.claim(root)
    assert claim is not None
    deps = PMDeps(orch, pid, root, claim.run_id)
    agent = build_agent(team.get(team.pm), with_validate=with_validate)

    usage = RunUsage()
    limits = UsageLimits(request_limit=20)
    messages: list[ModelMessage] = []
    outcome: Outcome | None = None
    error, final_text, nudged = "", "", False
    t0 = time.monotonic()
    try:
        async with asyncio.timeout(timeout_s):
            with capture_run_messages() as captured:
                messages = captured
                result = await agent.run(
                    task_prompt(scn.objective, scn.criteria),
                    deps=deps,
                    model=model,
                    model_settings=settings,
                    usage=usage,
                    usage_limits=limits,
                )
            messages, final_text = result.all_messages(), result.output
            # Muse may stop to ask for confirmation instead of submitting. Nudge once,
            # the way the runner would, and count it.
            if orch.get_task(root)["status"] == TaskStatus.RUNNING:
                nudged = True
                with capture_run_messages() as captured:
                    messages = captured
                    result = await agent.run(
                        NUDGE,
                        deps=deps,
                        model=model,
                        model_settings=settings,
                        usage=usage,
                        usage_limits=limits,
                        message_history=result.all_messages(),
                    )
                messages, final_text = result.all_messages(), result.output
    except TimeoutError:
        outcome, error = "timeout", f"no answer within {timeout_s:.0f}s"
    except Exception as e:  # spike: record every failure mode instead of crashing the batch
        outcome, error = "error", f"{type(e).__name__}: {e}"[:500]
    seconds = time.monotonic() - t0

    status = orch.get_task(root)["status"]
    if status == TaskStatus.RUNNING:
        orch.retry(root, error or "PM ended without submitting a plan")
    if outcome is None:
        by_status: dict[str, Outcome] = {
            TaskStatus.WAITING: "accepted",
            TaskStatus.AWAITING_HUMAN: "escalated",
        }
        outcome = by_status.get(status, "no_plan")
    cost = cost_usd(usage, price_as)
    orch.record_usage(
        claim.run_id,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        cost_usd=cost,
    )

    r = TrialResult(
        scenario=scn.name,
        trial=i,
        outcome=outcome,
        passed=outcome == scn.expect,
        seconds=round(seconds, 1),
        nudged=nudged,
        error=error,
        final_text=final_text[:300],
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        cached_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_reasoning_tokens,
        cost_usd=round(cost, 5),
    )
    tally(r, messages)
    (workdir / f"{scn.name}-{i}.json").write_bytes(
        ModelMessagesTypeAdapter.dump_json(messages, indent=2)
    )
    return r


def tally(r: TrialResult, messages: list[ModelMessage]) -> None:
    """Tool-call reliability numbers, read off the transcript."""
    calls: Counter[str] = Counter()
    for m in messages:
        if isinstance(m, ModelResponse):
            tc = [p for p in m.parts if isinstance(p, ToolCallPart)]
            calls.update(p.tool_name for p in tc)
            r.multi_call_responses += len(tc) > 1
            if tc and tc[-1].tool_name == "submit_plan":
                with contextlib.suppress(ValueError):  # malformed args: counted as a retry
                    r.plan_nodes = len(Plan.model_validate(tc[-1].args_as_dict()).nodes)
        elif isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, RetryPromptPart):
                    r.retries += 1
                    content = p.content if isinstance(p.content, str) else json.dumps(p.content)
                    r.retry_reasons.append(f"{p.tool_name or 'output'}: {content[:160]}")
                elif isinstance(p, ToolReturnPart) and isinstance(p.content, dict):
                    content = cast(dict[str, Any], p.content)
                    r.rule_codes += [e["code"] for e in content.get("errors", [])]
    r.tool_calls = dict(calls)
    r.submits, r.validates = calls["submit_plan"], calls["validate_plan"]


# ------------------------------------------------------------------------- offline model


def scripted_pm() -> FunctionModel:
    """No network: list agents, try a plan missing its review, then the good plan."""
    good = yaml.safe_load((ROOT / "examples" / "plan_transfers.yaml").read_text())
    bad = {
        **good,
        "nodes": [n for n in good["nodes"] if n["id"] != "review_transfer"],
        "edges": [e for e in good["edges"] if "review_transfer" not in e],
    }

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        done = [
            p.tool_name
            for m in messages
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart)
        ]
        tools = {t.name for t in info.function_tools}
        if not done:
            return ModelResponse([ToolCallPart("list_agents", {})])
        if len(done) == 1:
            first = "validate_plan" if "validate_plan" in tools else "submit_plan"
            return ModelResponse([ToolCallPart(first, bad)])
        if len(done) == 2:
            return ModelResponse([ToolCallPart("submit_plan", good)])
        return ModelResponse([TextPart("Plan accepted.")])

    return FunctionModel(respond, model_name="scripted-pm")


# ------------------------------------------------------------------------- batch + report


def summarize(results: list[TrialResult]) -> str:
    def avg(xs: list[float]) -> float:
        return statistics.fmean(xs) if xs else 0.0

    lines = [
        "| scenario | n | pass | outcomes | nudged | submits | validates | retries | "
        "requests | in tok | out tok (reasoning) | $/run | s p50 / max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[TrialResult]] = {}
    for r in results:
        groups.setdefault(r.scenario, []).append(r)
    for name, rs in [*groups.items(), ("ALL", results)]:
        secs = [r.seconds for r in rs]
        outcomes = ", ".join(f"{k} {v}" for k, v in Counter(r.outcome for r in rs).items())
        lines.append(
            f"| {name} | {len(rs)} | {sum(r.passed for r in rs)}/{len(rs)} | {outcomes} "
            f"| {sum(r.nudged for r in rs)} "
            f"| {avg([r.submits for r in rs]):.1f} | {avg([r.validates for r in rs]):.1f} "
            f"| {sum(r.retries for r in rs)} | {avg([r.requests for r in rs]):.1f} "
            f"| {avg([r.input_tokens for r in rs]):,.0f} "
            f"| {avg([r.output_tokens for r in rs]):,.0f} "
            f"({avg([r.reasoning_tokens for r in rs]):,.0f}) "
            f"| {avg([r.cost_usd for r in rs]):.4f} "
            f"| {statistics.median(secs):.0f} / {max(secs):.0f} |"
        )
    reasons = Counter(x for r in results for x in r.retry_reasons)
    if reasons:
        lines += ["", "Retries sent back to the model:"]
        lines += [f"- {n}x {why}" for why, n in reasons.most_common(10)]
    errors = [f"- {r.scenario}#{r.trial}: {r.error}" for r in results if r.error]
    if errors:
        lines += ["", "Errors:", *errors]
    return "\n".join(lines)


async def run_batch(
    scenarios: list[Scenario],
    n: int,
    *,
    model: Model,
    out: Path,
    concurrency: int,
    **kw: Any,
) -> list[TrialResult]:
    sem = asyncio.Semaphore(concurrency)

    async def one(scn: Scenario, i: int) -> TrialResult:
        async with sem:
            r = await run_trial(scn, i, model=model, workdir=out, **kw)
            mark = "ok " if r.passed else "BAD"
            print(
                f"{mark} {scn.name}#{i}: {r.outcome} submits={r.submits} "
                f"retries={r.retries} ${r.cost_usd:.4f} {r.seconds:.0f}s",
                flush=True,
            )
            return r

    return list(await asyncio.gather(*(one(s, i) for s in scenarios for i in range(n))))


app = typer.Typer(add_completion=False)


@app.command()
def main(
    scenario: list[str] = typer.Option(["all"], help=f"all or any of {list(SCENARIOS)}"),
    n: int = typer.Option(3, help="trials per scenario"),
    model_name: str = typer.Option(DEFAULT_MODEL, "--model", help=f"one of {list(PRICES)}"),
    api: str = typer.Option("chat", help="chat | responses"),
    effort: str = typer.Option("", help="reasoning effort: minimal..xhigh (default: server's)"),
    validate: bool = typer.Option(True, help="offer validate_plan (--no-validate: submit only)"),
    concurrency: int = typer.Option(3),
    timeout: float = typer.Option(300, help="seconds per trial"),
    offline: bool = typer.Option(False, help="scripted model, no network"),
) -> None:
    if model_name not in PRICES:
        raise typer.BadParameter(f"no price for {model_name}; add it to model.PRICES")
    if "all" in scenario:  # the scripted PM only knows the transfers plan
        scenario = ["transfers"] if offline else list(SCENARIOS)
    picked = [SCENARIOS[s] for s in scenario]
    model = scripted_pm() if offline else meta_model(model_name, api=cast(Api, api))
    settings = OpenAIChatModelSettings()
    if effort:
        settings["openai_reasoning_effort"] = cast(Any, effort)
    out = data_dir() / "spike" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    results = asyncio.run(
        run_batch(
            picked,
            n,
            model=model,
            out=out,
            concurrency=concurrency,
            with_validate=validate,
            settings=settings,
            timeout_s=timeout,
            price_as=model_name,
        )
    )
    with (out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    config = (
        f"model={'scripted' if offline else model_name} api={api} effort={effort or 'default'} "
        f"validate={validate} n={n}"
    )
    summary = f"# PM spike {out.name}\n\n{config}\n\n{summarize(results)}\n"
    (out / "summary.md").write_text(summary)
    print("\n" + summary + f"\nwrote {out}/")


if __name__ == "__main__":
    app()
