from __future__ import annotations

import pytest

from src.common.types import ProductLifecycleState
from src.payload.information_sufficiency import (
    delta_information_utility,
    product_information_utility,
    residual_information_utility,
)
from src.simulator.step_runner import StepRunner


def _runner() -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = {"progressive_payload": {"roi_context_after_roi": {"enabled": True}}}
    return runner


def test_roi_closes_verification():
    runner = _runner()
    state = ProductLifecycleState("v", product_requirement="verification", delivered_products={"roi"})
    assert runner._integrated_branch_closure_satisfied(state)
    assert residual_information_utility({"roi"}, "verification") > 0.0


def test_roi_does_not_close_evidence():
    runner = _runner()
    state = ProductLifecycleState("e", product_requirement="evidence", delivered_products={"roi"})
    assert not runner._integrated_branch_closure_satisfied(state)
    assert residual_information_utility({"roi"}, "evidence") > 0.0


def test_roi_context_closes_context_or_evidence():
    runner = _runner()
    context = ProductLifecycleState("c", product_requirement="context", delivered_products={"roi", "roi_context"})
    evidence = ProductLifecycleState("e", product_requirement="evidence", delivered_products={"roi", "roi_context"})
    assert runner._integrated_branch_closure_satisfied(context)
    assert runner._integrated_branch_closure_satisfied(evidence)


def test_roi_context_after_roi_uses_marginal_delta_not_total_utility():
    before = product_information_utility({"roi"}, "context")
    after = product_information_utility({"roi", "roi_context"}, "context")
    marginal = delta_information_utility("roi_context", {"roi"}, "context")
    total = product_information_utility({"roi_context"}, "context")
    assert marginal == pytest.approx(after - before)
    assert marginal < total


def test_duplicate_roi_context_delta_zero():
    assert delta_information_utility("roi_context", {"roi_context"}, "context") == pytest.approx(0.0)
    assert delta_information_utility("roi_context", {"roi", "roi_context"}, "evidence") == pytest.approx(0.0)


def test_legacy_default_unchanged():
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = {}
    assert runner._roi_context_after_roi_cfg()["enabled"] is False
