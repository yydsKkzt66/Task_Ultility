from __future__ import annotations

import pytest

from scripts.run_myopic_e2e_utility_greedy import (
    DEFAULT_METHODS,
    EXPECTED_FROZEN_V25_COUNTS,
    EXPECTED_FROZEN_V25_TASK_HASH,
    METHOD_NAMES,
    build_shared_scenario,
    configure_target_capacity_multiplier,
    configure_target_scenario_profile,
    task_effective_value,
)
from scripts.run_v25_cap050_five_method_multiseed import (
    ALL_REPORTED_METRICS,
    EXPECTED_METHODS,
    PRIMARY_METRICS,
    SCENARIO_PROFILE,
    _default_output_for_mode,
    _parse_seeds,
    _seeds_for_mode,
    build_metric_summary,
)


def test_default_five_method_identity_is_exact():
    assert set(METHOD_NAMES[key] for key in DEFAULT_METHODS) == set(
        EXPECTED_METHODS
    )
    assert (
        METHOD_NAMES["MYOPIC"]
        == "MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN"
    )


def test_seed_parser_rejects_empty_and_duplicate_values():
    assert _parse_seeds("2026, 2027") == (2026, 2027)
    with pytest.raises(ValueError):
        _parse_seeds("")
    with pytest.raises(ValueError):
        _parse_seeds("2026,2026")


def test_one_and_five_seed_switches_are_explicit_and_use_distinct_outputs():
    assert _seeds_for_mode("one_seed") == (2026,)
    assert _seeds_for_mode("five_seed") == (2026, 2027, 2028, 2029, 2030)
    assert _default_output_for_mode("one_seed") != _default_output_for_mode(
        "five_seed"
    )
    with pytest.raises(ValueError):
        _seeds_for_mode("unsupported")


def test_controlled_v25_keeps_exact_anchor_and_value_identity():
    configure_target_capacity_multiplier(0.50)
    configure_target_scenario_profile(SCENARIO_PROFILE)
    _, anchor_timeline, anchor_identity = build_shared_scenario(2026, "main")
    _, varied_timeline, varied_identity = build_shared_scenario(2027, "main")
    for timeline in (anchor_timeline, varied_timeline):
        tasks = [step.task for step in timeline if step.task is not None]
        assert len(tasks) == 268
        assert {
            level: sum(task.value_level == level for task in tasks)
            for level in EXPECTED_FROZEN_V25_COUNTS
        } == EXPECTED_FROZEN_V25_COUNTS
        assert all(
            task_effective_value(task)
            == pytest.approx(
                0.17447916666666666
                * {"low": 1.0, "medium": 3.0, "high": 10.0, "urgent": 25.0}[
                    task.value_level
                ]
            )
            for task in tasks
        )
    assert (
        anchor_identity["task_manifest_hash_v2"]
        == EXPECTED_FROZEN_V25_TASK_HASH
    )
    assert (
        varied_identity["task_manifest_hash_v2"]
        != anchor_identity["task_manifest_hash_v2"]
    )


def test_multiseed_summary_uses_per_method_seed_samples():
    rows = []
    for seed, offset in ((2026, 0.0), (2027, 1.0)):
        for method in EXPECTED_METHODS:
            row = {"seed": str(seed), "method": method}
            row.update(
                {
                    metric: str(index + offset)
                    for index, metric in enumerate(ALL_REPORTED_METRICS, start=1)
                }
            )
            rows.append(row)
    summary = build_metric_summary(rows)
    assert len(summary) == len(EXPECTED_METHODS) * len(ALL_REPORTED_METRICS)
    first = next(
        row
        for row in summary
        if row["method"] == EXPECTED_METHODS[0]
        and row["metric"] == PRIMARY_METRICS[0]
    )
    assert first["seed_count"] == 2
    assert first["mean"] == pytest.approx(1.5)
    assert first["sample_std"] == pytest.approx(2 ** -0.5)
    assert first["normal_ci95_low"] < first["mean"]
    assert first["normal_ci95_high"] > first["mean"]
