"""The PM spike's plumbing, with a scripted model instead of muse-spark."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.usage import RunUsage

from orch.runtime.model import cost_usd
from orch.runtime.spike import SCENARIOS, run_trial, scripted_pm


def trial(tmp_path: Path, *, with_validate: bool):
    return asyncio.run(
        run_trial(
            SCENARIOS["transfers"],
            0,
            model=scripted_pm(),
            workdir=tmp_path,
            with_validate=with_validate,
            settings=OpenAIChatModelSettings(),
            timeout_s=30,
            price_as="muse-spark-1.3",
        )
    )


def test_rejected_draft_then_accepted_plan(tmp_path: Path):
    r = trial(tmp_path, with_validate=False)
    assert r.outcome == "accepted" and r.passed
    assert r.submits == 2 and r.validates == 0
    assert "SOD-001" in r.rule_codes  # the draft without a review was rejected
    assert r.plan_nodes == 5 and r.retries == 0
    assert r.cost_usd > 0
    assert (tmp_path / "transfers-0.json").exists()


def test_validate_then_submit(tmp_path: Path):
    r = trial(tmp_path, with_validate=True)
    assert r.outcome == "accepted"
    assert (r.validates, r.submits) == (1, 1)


def test_cost_counts_cached_input_at_the_cached_price():
    u = RunUsage(input_tokens=1_000_000, cache_read_tokens=400_000, output_tokens=100_000)
    assert round(cost_usd(u, "muse-spark-1.3"), 4) == round(0.6 * 1.25 + 0.4 * 0.15 + 0.1 * 4.25, 4)
    assert round(cost_usd(u), 4) == round(0.6 * 0.10 + 0.4 * 0.002 + 0.1 * 0.20, 4)
