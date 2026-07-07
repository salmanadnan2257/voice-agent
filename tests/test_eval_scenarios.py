"""Runs the full offline evaluation harness (20+ scripted scenarios) as part
of the regular pytest suite. Each scenario is also its own parametrized test
so a failure names the exact scenario, not just an aggregate count.
"""
from __future__ import annotations

import pytest

from eval.scenarios import SCENARIOS, run_scenario


def test_at_least_twenty_scenarios_registered():
    assert len(SCENARIOS) >= 20, f"only {len(SCENARIOS)} scenarios registered, brief requires 20+"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_scenario_passes(scenario):
    passed, detail, _ = run_scenario(scenario)
    assert passed, f"{scenario.name} ({scenario.category}) failed: {detail}"
