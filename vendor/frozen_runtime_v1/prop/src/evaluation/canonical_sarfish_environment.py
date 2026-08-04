"""Canonical SARFishSample main-paper environment and path-independent hashes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.baselines.b1_enhanced import contact_schedule_hash, resource_trace_hash
from src.common.config import load_project_configs, load_yaml
import yaml


TASK_ID_RE = re.compile(r"^sarfishsample_task_[0-9]{4}$")
VALUE_LEVEL_BY_CLASS = {"routine": "medium", "high_value": "high", "emergency": "urgent"}
LEGACY_AUDIT_MANIFEST = Path("outputs/_dataset_environment_restore_seed2026/intended_sarfish_environment_manifest.yaml")


def canonical_value_level(task_class: str) -> str:
    try:
        return VALUE_LEVEL_BY_CLASS[str(task_class)]
    except KeyError as exc:
        raise ValueError(f"unsupported SARFish task class: {task_class}") from exc


def load_canonical_sarfish_configs(root: Path, *, seed: int = 2026) -> dict[str, dict[str, Any]]:
    root = Path(root)
    configs = load_project_configs(root)
    configs["dataset"] = load_yaml(root / "configs/dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(root / "configs/task_stream_sarfish_formal_nominal_bupt1_deadline_minutes.yaml")
    configs["timeline"] = load_yaml(root / "configs/timeline_sarfish_formal_nominal_bupt1_deadline_minutes_20s.yaml")
    configs["simulator"] = load_yaml(root / "configs/simulator_sarfish_formal_nominal_bupt1_deadline_minutes_1day_20s.yaml")
    configs["resource"] = load_yaml(root / "configs/resource_sarfish_formal_nominal_bupt1.yaml")
    configs["dataset"]["seed"] = int(seed)
    configs["task_stream"]["seed"] = int(seed)
    configs["contact"] = copy.deepcopy(configs["timeline"]["contact"])
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    profiles = load_yaml(root / "configs/model_profiles_yolo_inspired_calibrated.yaml")
    configs["model_pool"] = copy.deepcopy(configs["model_pool"])
    configs["model_pool"]["profile_overrides"] = dict(profiles["profile_overrides"])
    configs["payload_profiles"] = load_yaml(root / "configs/payload_profiles_roi_compression_calibrated.yaml")
    return configs


def canonical_task_rows_v2(timeline) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, step in enumerate(timeline):
        task = step.task
        if task is None:
            continue
        logical_image = str((task.metadata or {}).get("logical_image_path", "")).replace("\\", "/")
        logical_label = str((task.metadata or {}).get("logical_label_path", "GRD/validation/GRD_validation.csv")).replace("\\", "/")
        rows.append({
            "order": len(rows),
            "timeline_slot": int(step.slot_index),
            "task_id": str(task.task_id),
            "scene_id": str(task.scene_id),
            "product_identifier": str((task.metadata or {}).get("product_identifier", "")),
            "logical_image_path": logical_image,
            "logical_label_path": logical_label,
            "arrival_slot": int(task.arrival_slot),
            "arrival_time_s": float(task.arrival_slot * step.slot_seconds),
            "capture_time_s": float(task.arrival_slot * step.slot_seconds),
            "observation_window_id": str(task.observation_window_id),
            "task_class": str(task.task_class),
            "value_level": str(task.value_level),
            "task_value": float(task.task_value),
            "deadline_s": float(task.deadline_s),
            "quality_requirement": float(task.quality_requirement),
            "information_requirement": float(task.information_requirement),
            "product_requirement": str(task.product_requirement),
            "product_requirement_source": str(task.product_requirement_source),
            "difficulty": float(task.difficulty),
            "uncertainty": float(task.uncertainty),
            "high_value_flag": bool(task.high_value_flag),
            "emergency_flag": bool(task.emergency_flag),
        })
    return rows


def _stable_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_task_hash_v2(rows: list[dict[str, Any]]) -> str:
    excluded = {"resolved_image_path", "resolved_label_path", "image_path", "label_path", "dataset_root", "source_image_root"}
    stable_rows = [{key: value for key, value in row.items() if key not in excluded} for row in rows]
    return _stable_hash(stable_rows)


def canonical_environment_hash_v2(configs: dict[str, Any], timeline) -> str:
    dataset = configs["dataset"]
    stable_dataset = {
        "dataset_name": str(dataset.get("dataset_name", "")).lower(),
        "product_type": str(dataset.get("product_type", "")).upper(),
        "split": str(dataset.get("split", "")).lower(),
        "scene_index_path": "data/indexes/sarfishsample_scene_index.csv",
        "task_state_profile_path": "data/indexes/sarfishsample_task_state_profile.json",
    }
    value = {
        "version": "path_independent_canonical_environment_v2",
        "dataset": stable_dataset,
        "seed": int(dataset.get("seed", 0)),
        "task_manifest_hash_v2": canonical_task_hash_v2(canonical_task_rows_v2(timeline)),
        "contact_trace_hash": contact_schedule_hash(timeline),
        "resource_trace_hash": resource_trace_hash(timeline),
        "deadline_inputs": {key: configs["task_stream"][key] for key in sorted(configs["task_stream"]) if "deadline" in key},
        "contact_inputs": configs["contact"],
        "resource_inputs": configs["resource"],
    }
    return _stable_hash(value)


def validate_main_paper_environment(
    configs: dict[str, Any],
    timeline,
    *,
    expected_task_hash_v2: str | None = None,
    expected_environment_hash_v2: str | None = None,
) -> dict[str, Any]:
    dataset = configs.get("dataset", {})
    if not (
        str(dataset.get("dataset_name", "")).lower() == "sarfishsample"
        and str(dataset.get("product_type", "")).upper() == "GRD"
        and str(dataset.get("split", "")).lower() == "validation"
    ):
        raise ValueError("main-paper resolved dataset must be SARFishSample/GRD/validation, never mock")
    rows = canonical_task_rows_v2(timeline)
    if len(rows) != 268 or len({row["task_id"] for row in rows}) != 268:
        raise ValueError(f"main-paper run requires exactly 268 unique tasks, got {len(rows)}")
    if not all(TASK_ID_RE.fullmatch(row["task_id"]) for row in rows):
        raise ValueError("main-paper task IDs must match sarfishsample_task_XXXX")
    for row in rows:
        expected = canonical_value_level(row["task_class"])
        if not row["value_level"] or row["value_level"] != expected:
            raise ValueError(f"runtime value label mismatch for {row['task_id']}: {row['value_level']} != {expected}")
        if not row["logical_image_path"].startswith("GRD/"):
            raise ValueError(f"non-portable logical image path for {row['task_id']}")
    task_hash = canonical_task_hash_v2(rows)
    env_hash = canonical_environment_hash_v2(configs, timeline)
    if expected_task_hash_v2 is not None and task_hash != expected_task_hash_v2:
        raise ValueError(f"task-manifest hash mismatch: {task_hash} != {expected_task_hash_v2}")
    if expected_environment_hash_v2 is not None and env_hash != expected_environment_hash_v2:
        raise ValueError(f"environment hash mismatch: {env_hash} != {expected_environment_hash_v2}")
    return {"passed": True, "task_count": 268, "task_manifest_hash_v2": task_hash, "environment_hash_v2": env_hash}


def validate_legacy_task_equivalence(rows: list[dict[str, Any]], legacy_manifest_path: Path) -> dict[str, Any]:
    legacy = yaml.safe_load(Path(legacy_manifest_path).read_text(encoding="utf-8"))
    legacy_rows = list(legacy.get("tasks", []))
    if len(rows) != len(legacy_rows) or len(rows) != 268:
        raise ValueError(f"legacy/current task count mismatch: {len(legacy_rows)} != {len(rows)}")
    stable_pairs = {
        "task_id": "task_id",
        "scene_id": "scene_id",
        "arrival_slot": "arrival_slot",
        "arrival_time_s": "arrival_time_s",
        "capture_time_s": "capture_time_s",
        "observation_window_id": "observation_window_id",
        "task_class": "task_class",
        "value_level": "value_level",
        "task_value": "task_value",
        "deadline_s": "deadline_s",
        "quality_requirement": "quality_requirement",
        "information_requirement": "information_requirement",
        "product_requirement": "product_requirement",
        "product_requirement_source": "product_requirement_source",
        "difficulty": "difficulty",
        "uncertainty": "uncertainty",
        "high_value_flag": "high_value_flag",
        "emergency_flag": "emergency_flag",
    }
    for index, (current, old) in enumerate(zip(rows, legacy_rows)):
        for current_key, old_key in stable_pairs.items():
            if current[current_key] != old[old_key]:
                raise ValueError(
                    f"legacy task field mismatch at order {index}, {current['task_id']}, {current_key}: "
                    f"{current[current_key]!r} != {old[old_key]!r}"
                )
    current_scenes = {row["scene_id"] for row in rows}
    legacy_scenes = {row["scene_id"] for row in legacy_rows}
    if current_scenes != legacy_scenes or len(current_scenes) != 50:
        raise ValueError("legacy/current 50-scene set mismatch")
    return {"passed": True, "task_count": len(rows), "scene_count": len(current_scenes)}
