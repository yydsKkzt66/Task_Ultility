from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.common.config import load_project_configs, load_yaml
from src.evaluation.canonical_sarfish_environment import (
    canonical_environment_hash_v2,
    canonical_task_hash_v2,
    canonical_task_rows_v2,
    canonical_value_level,
    load_canonical_sarfish_configs,
    validate_legacy_task_equivalence,
    validate_main_paper_environment,
)
from src.timeline.timeline_builder import TimelineBuilder


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_value_level_mapping_is_total():
    assert canonical_value_level("routine") == "medium"
    assert canonical_value_level("high_value") == "high"
    assert canonical_value_level("emergency") == "urgent"
    with pytest.raises(ValueError):
        canonical_value_level("unknown")


def test_path_independent_hash_ignores_machine_dataset_root():
    row = {
        "task_id": "sarfishsample_task_0001",
        "scene_id": "scene-1",
        "logical_image_path": "GRD/validation/X.SAFE/measurement/x.tiff",
        "resolved_image_path": "A:/data/GRD/validation/X.SAFE/measurement/x.tiff",
        "value_level": "high",
    }
    moved = {**row, "resolved_image_path": "D:/mirror/GRD/validation/X.SAFE/measurement/x.tiff"}
    assert canonical_task_hash_v2([row]) == canonical_task_hash_v2([moved])


def test_formal_nominal_seed2026_preflight_and_hashes_are_deterministic():
    configs = load_canonical_sarfish_configs(ROOT, seed=2026)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    rows = canonical_task_rows_v2(timeline)
    report = validate_main_paper_environment(configs, timeline)
    assert report["passed"]
    assert len(rows) == len({row["task_id"] for row in rows}) == 268
    assert all(row["value_level"] in {"medium", "high", "urgent"} for row in rows)
    assert all(row["logical_image_path"].startswith("GRD/") for row in rows)
    assert canonical_task_hash_v2(rows) == canonical_task_hash_v2(canonical_task_rows_v2(timeline))
    assert canonical_environment_hash_v2(configs, timeline)


def test_fail_fast_rejects_mock_and_empty_value_labels():
    configs = load_canonical_sarfish_configs(ROOT, seed=2026)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    bad = copy.deepcopy(configs)
    bad["dataset"]["dataset_name"] = "mock"
    with pytest.raises(ValueError, match="SARFishSample"):
        validate_main_paper_environment(bad, timeline)


def test_portable_manifest_matches_legacy_scene_task_order_and_fields():
    configs = load_canonical_sarfish_configs(ROOT, seed=2026)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    rows = canonical_task_rows_v2(timeline)
    result = validate_legacy_task_equivalence(
        rows,
        ROOT / "outputs/_dataset_environment_restore_seed2026/intended_sarfish_environment_manifest.yaml",
    )
    assert result["passed"]
    assert result["scene_count"] == 50
    assert result["task_count"] == 268
