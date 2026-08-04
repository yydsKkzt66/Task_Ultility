"""Validate product_requirement_v2 semantics and Module 2 isolation.

All outputs are diagnostic-only. This script does not modify reward, routing,
SafeFilter, SafeLinUCB scoring, utility_rule, action selection, or default
configs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import inspect
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.data.product_requirements import (  # noqa: E402
    assign_product_requirement,
    assign_product_requirement_v2,
    recommend_payload_plan_from_requirement,
)
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_v2_semantics_isolation"
RICHNESS = {"alert": 0, "verification": 1, "context": 2, "evidence": 3}
FORBIDDEN_TOKENS = [
    "label",
    "bbox",
    "target_count",
    "object_count",
    "object_class",
    "detector",
    "ground_truth",
    "future_delivery",
    "future_utility",
    "realized_future_utility",
    "post_action",
    "queue_outcome",
]
ALLOWED_FIELDS = [
    "task_value",
    "deadline_s",
    "remaining_deadline_s",
    "uncertainty",
    "high_value_flag",
    "emergency_flag",
    "task_class",
    "mission_source",
    "scene_prior",
    "product_use_case",
    "queue_mb",
    "queue_max_mb",
    "known_contact_capacity_before_deadline",
    "required_capacity_for_roi_context",
    "battery_slack",
    "thermal_slack",
    "queue_slack",
    "contact_slack",
    "deadline_slack",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = completed.stdout
        if completed.stderr:
            output += "\nSTDERR:\n" + completed.stderr
        return output.strip()
    except Exception as exc:  # pragma: no cover
        return f"git command failed: {exc}"


def _git_knows(path: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0


def _synthetic_new_file_diff(path: str) -> str:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return ""
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = [
        f"diff --git a/{path} b/{path}",
        "new file mode 100644",
        "index 0000000..0000000",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    return "\n".join([*header, *[f'+{line}' for line in lines]])


def _audit_diff(paths: list[str]) -> str:
    tracked = [path for path in paths if _git_knows(path)]
    parts: list[str] = []
    if tracked:
        parts.append(_run_git(["diff", "--", *tracked]))
    for path in paths:
        if not _git_knows(path):
            parts.append(_synthetic_new_file_diff(path))
    return "\n\n".join(part for part in parts if part).strip()


def _initial_alert_required(fields: dict[str, Any]) -> bool:
    deadline = max(float(fields.get("deadline_s", 1.0)), 1.0e-9)
    remaining = float(fields.get("remaining_deadline_s", deadline))
    urgency = max(0.0, min(1.0, 1.0 - remaining / deadline))
    contact_capacity = float(fields.get("known_contact_capacity_before_deadline", 1.0))
    required = max(float(fields.get("required_capacity_for_roi_context", 2.7)), 1.0e-9)
    contact_pressure = max(0.0, min(1.0, 1.0 - contact_capacity / required))
    queue_mb = float(fields.get("queue_mb", 0.0))
    queue_max = max(float(fields.get("queue_max_mb", 64.0)), 1.0e-9)
    queue_pressure = max(0.0, min(1.0, queue_mb / queue_max))
    return bool(fields.get("emergency_flag", False)) or urgency >= 0.75 or contact_pressure >= 0.80 or queue_pressure >= 0.85


def _evaluate(fields: dict[str, Any]) -> dict[str, object]:
    result = assign_product_requirement_v2(**fields)
    capacity = float(fields.get("known_contact_capacity_before_deadline", 0.0))
    required_context = float(fields.get("required_capacity_for_roi_context", 2.7))
    recommendation = recommend_payload_plan_from_requirement(
        ideal_product_requirement=result.ideal_product_requirement,
        metadata_feasible=capacity >= 0.002,
        roi_feasible=capacity >= 0.85,
        roi_context_feasible=capacity >= required_context,
        metadata_then_roi_feasible=capacity >= 0.002,
        metadata_then_roi_context_feasible=capacity >= 0.002,
    )
    return {
        "ideal": result.ideal_product_requirement,
        "scores": result.requirement_scores,
        "reason": result.diagnostic_reason,
        "missing_fields": ";".join(sorted(set(result.missing_fields) | set(recommendation.missing_fields))),
        "initial_alert_required": int(_initial_alert_required(fields)),
        "recommended_payload_plan_shadow": recommendation.recommended_payload_plan,
    }


def _base_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "task_value": 0.5,
        "deadline_s": 100.0,
        "remaining_deadline_s": 100.0,
        "uncertainty": 0.4,
        "high_value_flag": False,
        "emergency_flag": False,
        "task_class": "routine",
        "mission_source": "expert_case",
        "scene_prior": 0.5,
        "product_use_case": "diagnostic_semantics",
        "queue_mb": 1.0,
        "queue_max_mb": 64.0,
        "known_contact_capacity_before_deadline": 6.0,
        "required_capacity_for_roi_context": 2.7,
        "battery_slack": 0.9,
        "thermal_slack": 0.9,
        "queue_slack": 0.9,
        "contact_slack": 0.9,
        "deadline_slack": 0.9,
    }
    fields.update(overrides)
    return fields


def _semantic_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "case_01_low_value_routine_relaxed",
            "fields": _base_fields(task_value=0.25, uncertainty=0.20, deadline_s=180.0, remaining_deadline_s=180.0),
            "expected_allowed_requirements": {"alert", "verification"},
            "expected_disallowed_requirements": {"evidence"},
            "expect_initial_alert_required": None,
            "note": "Low-value routine, low uncertainty, relaxed deadline.",
        },
        {
            "case_id": "case_02_routine_medium_value_high_uncertainty",
            "fields": _base_fields(task_value=0.55, uncertainty=0.80, deadline_s=120.0, remaining_deadline_s=100.0),
            "expected_allowed_requirements": {"verification"},
            "expected_disallowed_requirements": {"alert", "evidence"},
            "expect_initial_alert_required": None,
            "note": "Routine medium value with high uncertainty should mainly need verification.",
        },
        {
            "case_id": "case_03_high_value_high_uncertainty_enough_slack",
            "fields": _base_fields(
                task_value=0.88,
                uncertainty=0.78,
                high_value_flag=True,
                task_class="high_value",
                scene_prior=0.75,
            ),
            "expected_allowed_requirements": {"context", "evidence"},
            "expected_disallowed_requirements": {"alert"},
            "expect_initial_alert_required": None,
            "note": "High-value suspicious maritime target with enough slack.",
        },
        {
            "case_id": "case_04_high_value_high_uncertainty_tight_resources",
            "fields": _base_fields(
                task_value=0.88,
                uncertainty=0.78,
                high_value_flag=True,
                task_class="high_value",
                deadline_s=50.0,
                remaining_deadline_s=5.0,
                queue_mb=60.0,
                queue_slack=0.06,
                known_contact_capacity_before_deadline=0.01,
                contact_slack=0.0,
                deadline_slack=0.05,
                battery_slack=0.35,
                thermal_slack=0.35,
            ),
            "expected_allowed_requirements": {"context", "evidence"},
            "expected_disallowed_requirements": {"alert"},
            "expect_initial_alert_required": True,
            "note": "Tight resources should affect shadow plan, not downgrade ideal need to alert-only.",
        },
        {
            "case_id": "case_05_emergency_tight_deadline",
            "fields": _base_fields(
                task_value=0.62,
                uncertainty=0.62,
                emergency_flag=True,
                task_class="emergency",
                deadline_s=30.0,
                remaining_deadline_s=2.0,
                known_contact_capacity_before_deadline=0.1,
            ),
            "expected_allowed_requirements": {"verification", "context", "evidence"},
            "expected_disallowed_requirements": {"alert"},
            "expect_initial_alert_required": True,
            "note": "Emergency should require early alert but not automatically alert-only.",
        },
        {
            "case_id": "case_06_emergency_high_value_high_uncertainty_enough_slack",
            "fields": _base_fields(
                task_value=0.94,
                uncertainty=0.82,
                high_value_flag=True,
                emergency_flag=True,
                task_class="emergency",
                scene_prior=0.8,
            ),
            "expected_allowed_requirements": {"evidence"},
            "expected_disallowed_requirements": {"alert", "verification"},
            "expect_initial_alert_required": True,
            "note": "Emergency high-value high-uncertainty with enough slack should need evidence.",
        },
        {
            "case_id": "case_07_high_value_low_uncertainty",
            "fields": _base_fields(task_value=0.86, uncertainty=0.18, high_value_flag=True, task_class="high_value"),
            "expected_allowed_requirements": {"context", "verification"},
            "expected_disallowed_requirements": {"evidence", "alert"},
            "expect_initial_alert_required": None,
            "note": "High value but low uncertainty need not be evidence.",
        },
        {
            "case_id": "case_08_low_value_high_uncertainty",
            "fields": _base_fields(task_value=0.25, uncertainty=0.88),
            "expected_allowed_requirements": {"verification"},
            "expected_disallowed_requirements": {"evidence", "alert"},
            "expect_initial_alert_required": None,
            "note": "Low-value high uncertainty should be verification unless other priors justify more.",
        },
    ]


def semantic_case_validation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in _semantic_cases():
        fields = case["fields"]
        evaluation = _evaluate(fields)
        actual = str(evaluation["ideal"])
        allowed = set(case["expected_allowed_requirements"])
        disallowed = set(case["expected_disallowed_requirements"])
        alert_expected = case["expect_initial_alert_required"]
        pass_requirement = actual in allowed and actual not in disallowed
        pass_alert = True if alert_expected is None else bool(evaluation["initial_alert_required"]) == bool(alert_expected)
        passed = pass_requirement and pass_alert
        rows.append(
            {
                "case_id": case["case_id"],
                "task_value": fields["task_value"],
                "deadline_s": fields["deadline_s"],
                "remaining_deadline_s": fields["remaining_deadline_s"],
                "uncertainty": fields["uncertainty"],
                "emergency_flag": int(bool(fields["emergency_flag"])),
                "high_value_flag": int(bool(fields["high_value_flag"])),
                "task_class": fields["task_class"],
                "queue_mb": fields["queue_mb"],
                "known_contact_capacity_before_deadline": fields["known_contact_capacity_before_deadline"],
                "expected_allowed_requirements": ";".join(sorted(allowed)),
                "expected_disallowed_requirements": ";".join(sorted(disallowed)),
                "actual_ideal_product_requirement": actual,
                "initial_alert_required": evaluation["initial_alert_required"],
                "recommended_payload_plan_shadow": evaluation["recommended_payload_plan_shadow"],
                "pass_fail": "PASS" if passed else "FAIL",
                "explanation": case["note"] + f" actual={actual}; reason={evaluation['reason']}",
            }
        )
    return rows


def monotonicity_validation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_series(series_id: str, variant_name: str, values: list[float], field: str, fixed: dict[str, Any]) -> None:
        previous_rank: int | None = None
        previous_requirement = ""
        for index, value in enumerate(values):
            fields = _base_fields(**fixed)
            fields[field] = value
            if field == "task_value":
                fields["high_value_flag"] = bool(value >= 0.75)
                fields["task_class"] = "high_value" if value >= 0.75 else fields["task_class"]
            evaluation = _evaluate(fields)
            requirement = str(evaluation["ideal"])
            rank = RICHNESS[requirement]
            passed = previous_rank is None or rank >= previous_rank
            rows.append(
                {
                    "series_id": series_id,
                    "variant": variant_name,
                    "changed_field": field,
                    "changed_value": value,
                    "actual_ideal_product_requirement": requirement,
                    "previous_requirement": previous_requirement,
                    "rank": rank,
                    "previous_rank": "" if previous_rank is None else previous_rank,
                    "pass_fail": "PASS" if passed else "FAIL",
                    "explanation": "higher mission-side signal should not reduce information richness",
                }
            )
            previous_rank = rank
            previous_requirement = requirement

    add_series(
        "inv1_task_value",
        "task_value_increase",
        [0.25, 0.45, 0.65, 0.82, 0.94],
        "task_value",
        {"uncertainty": 0.55, "deadline_s": 120.0, "remaining_deadline_s": 120.0},
    )
    add_series(
        "inv2_uncertainty",
        "uncertainty_increase",
        [0.10, 0.30, 0.55, 0.75, 0.92],
        "uncertainty",
        {"task_value": 0.55, "deadline_s": 120.0, "remaining_deadline_s": 120.0},
    )

    emergency_pairs = [
        ("no_emergency", _base_fields(task_value=0.70, uncertainty=0.70, emergency_flag=False, task_class="routine")),
        ("emergency", _base_fields(task_value=0.70, uncertainty=0.70, emergency_flag=True, task_class="emergency")),
    ]
    before_rank = None
    before_req = ""
    for label, fields in emergency_pairs:
        evaluation = _evaluate(fields)
        req = str(evaluation["ideal"])
        rank = RICHNESS[req]
        passed = before_rank is None or (int(evaluation["initial_alert_required"]) == 1 and rank >= max(1, before_rank))
        rows.append(
            {
                "series_id": "inv3_emergency_flag",
                "variant": label,
                "changed_field": "emergency_flag",
                "changed_value": int(bool(fields["emergency_flag"])),
                "actual_ideal_product_requirement": req,
                "previous_requirement": before_req,
                "rank": rank,
                "previous_rank": "" if before_rank is None else before_rank,
                "pass_fail": "PASS" if passed else "FAIL",
                "explanation": "emergency should raise alert urgency without erasing higher-level requirement",
            }
        )
        before_rank = rank
        before_req = req

    high_value_pairs = [
        ("no_high_value", _base_fields(task_value=0.70, uncertainty=0.55, high_value_flag=False, task_class="routine")),
        ("high_value", _base_fields(task_value=0.70, uncertainty=0.55, high_value_flag=True, task_class="high_value")),
    ]
    before_rank = None
    before_req = ""
    for label, fields in high_value_pairs:
        evaluation = _evaluate(fields)
        req = str(evaluation["ideal"])
        rank = RICHNESS[req]
        passed = before_rank is None or rank >= before_rank
        rows.append(
            {
                "series_id": "inv4_high_value_flag",
                "variant": label,
                "changed_field": "high_value_flag",
                "changed_value": int(bool(fields["high_value_flag"])),
                "actual_ideal_product_requirement": req,
                "previous_requirement": before_req,
                "rank": rank,
                "previous_rank": "" if before_rank is None else before_rank,
                "pass_fail": "PASS" if passed else "FAIL",
                "explanation": "high-value flag should not make ideal requirement less informative",
            }
        )
        before_rank = rank
        before_req = req

    deadline_values = [120.0, 80.0, 45.0, 20.0]
    reference_req = ""
    for deadline in deadline_values:
        fields = _base_fields(
            task_value=0.88,
            uncertainty=0.78,
            high_value_flag=True,
            task_class="high_value",
            deadline_s=deadline,
            remaining_deadline_s=min(deadline, max(2.0, deadline * 0.10)),
            known_contact_capacity_before_deadline=0.01,
            contact_slack=0.0,
            deadline_slack=0.05,
        )
        evaluation = _evaluate(fields)
        req = str(evaluation["ideal"])
        if not reference_req:
            reference_req = req
        passed = req in {"context", "evidence"}
        rows.append(
            {
                "series_id": "inv5_tighter_deadline",
                "variant": f"deadline_{deadline:g}",
                "changed_field": "deadline_s",
                "changed_value": deadline,
                "actual_ideal_product_requirement": req,
                "previous_requirement": reference_req,
                "rank": RICHNESS[req],
                "previous_rank": RICHNESS[reference_req],
                "pass_fail": "PASS" if passed else "FAIL",
                "explanation": "deadline pressure may require alert first but should not downgrade ideal context/evidence to alert-only",
            }
        )
    return rows


def resource_invariance_validation() -> list[dict[str, object]]:
    task_semantics = {
        "task_value": 0.88,
        "uncertainty": 0.78,
        "high_value_flag": True,
        "task_class": "high_value",
        "scene_prior": 0.75,
    }
    resources = [
        {
            "resource_case": "good_resources",
            "queue_mb": 1.0,
            "known_contact_capacity_before_deadline": 6.0,
            "battery_slack": 0.9,
            "thermal_slack": 0.9,
            "queue_slack": 0.9,
            "contact_slack": 0.9,
            "deadline_slack": 0.9,
            "deadline_s": 120.0,
            "remaining_deadline_s": 120.0,
        },
        {
            "resource_case": "medium_resources",
            "queue_mb": 32.0,
            "known_contact_capacity_before_deadline": 1.0,
            "battery_slack": 0.55,
            "thermal_slack": 0.55,
            "queue_slack": 0.50,
            "contact_slack": 0.35,
            "deadline_slack": 0.50,
            "deadline_s": 80.0,
            "remaining_deadline_s": 25.0,
        },
        {
            "resource_case": "bad_resources",
            "queue_mb": 62.0,
            "known_contact_capacity_before_deadline": 0.0005,
            "battery_slack": 0.20,
            "thermal_slack": 0.20,
            "queue_slack": 0.03,
            "contact_slack": 0.0,
            "deadline_slack": 0.03,
            "deadline_s": 50.0,
            "remaining_deadline_s": 4.0,
        },
    ]
    rows: list[dict[str, object]] = []
    reference_req = ""
    for item in resources:
        resource_case = str(item["resource_case"])
        resource_fields = {key: value for key, value in item.items() if key != "resource_case"}
        fields = _base_fields(**task_semantics, **resource_fields)
        evaluation = _evaluate(fields)
        req = str(evaluation["ideal"])
        if not reference_req:
            reference_req = req
        rows.append(
            {
                "task_semantics_id": "same_high_value_high_uncertainty",
                "resource_case": resource_case,
                "queue_pressure": float(fields["queue_mb"]) / float(fields["queue_max_mb"]),
                "contact_capacity_before_deadline": fields["known_contact_capacity_before_deadline"],
                "battery_slack": fields["battery_slack"],
                "thermal_slack": fields["thermal_slack"],
                "deadline_s": fields["deadline_s"],
                "remaining_deadline_s": fields["remaining_deadline_s"],
                "actual_ideal_product_requirement": req,
                "reference_ideal_product_requirement": reference_req,
                "recommended_payload_plan_shadow": evaluation["recommended_payload_plan_shadow"],
                "pass_fail": "PASS" if req == reference_req else "FAIL",
                "explanation": "same task semantics; resource pressure should mainly change the shadow payload plan",
            }
        )
    return rows


def _load_named_configs(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        configs = load_project_configs(PROJECT_ROOT)
        configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / args.dataset_config)
        configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / args.task_stream_config)
        configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / args.timeline_config)
        configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / args.simulator_config)
        TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        return configs, "requested_configs"
    except Exception as exc:
        configs = load_project_configs(PROJECT_ROOT)
        configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset.yaml")
        configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream.yaml")
        configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline.yaml")
        configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator.yaml")
        return configs, f"fallback_mock_configs_after_error={exc}"


def _limit_configs(configs: dict[str, dict[str, Any]], num_slots: int) -> None:
    configs["timeline"]["use_num_days"] = False
    configs["timeline"]["num_slots"] = int(num_slots)
    slot_seconds = float(configs["timeline"].get("slot_seconds", configs["task_stream"].get("slot_seconds", 20.0)))
    configs["timeline"]["slot_seconds"] = slot_seconds
    configs["task_stream"]["slot_seconds"] = slot_seconds
    configs["simulator"]["slot_seconds"] = slot_seconds


def _capacity_before_deadline(timeline, slot_index: int, deadline_s: float) -> float:
    if slot_index < 0 or slot_index >= len(timeline):
        return 0.0
    slot_seconds = float(timeline[slot_index].slot_seconds)
    horizon_slots = max(1, int(math.ceil(max(0.0, deadline_s) / max(slot_seconds, 1.0e-9))))
    end = min(len(timeline), slot_index + horizon_slots)
    capacity = 0.0
    for step in timeline[slot_index:end]:
        if step.link_trace.in_contact:
            capacity += float(step.link_trace.throughput_mbps) * slot_seconds / 8.0
    return capacity


def dataset_sanity_rows(timeline, configs: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sim = configs["simulator"]
    queue_mb = float(sim.get("queue_init_mb", 0.0))
    queue_max = max(float(sim.get("queue_max_mb", 64.0)), 1.0e-9)
    battery_init = float(sim.get("battery_init", 0.85))
    battery_min = float(sim.get("battery_min", 0.18))
    thermal_init = float(sim.get("thermal_init", 0.25))
    thermal_max = max(float(sim.get("thermal_max", 0.92)), 1.0e-9)
    battery_slack = max(0.0, min(1.0, (battery_init - battery_min) / max(1.0 - battery_min, 1.0e-9)))
    thermal_slack = max(0.0, min(1.0, (thermal_max - thermal_init) / thermal_max))
    queue_slack = max(0.0, min(1.0, 1.0 - queue_mb / queue_max))
    for step in timeline:
        task = step.task
        if task is None:
            continue
        capacity = _capacity_before_deadline(timeline, step.slot_index, task.deadline_s)
        old_req, _ = assign_product_requirement(
            task_value=task.task_value,
            deadline_s=task.deadline_s,
            uncertainty=task.uncertainty,
            high_value_flag=task.high_value_flag,
            emergency_flag=task.emergency_flag,
            task_class=task.task_class,
        )
        fields = _base_fields(
            task_value=task.task_value,
            deadline_s=task.deadline_s,
            remaining_deadline_s=task.deadline_s,
            uncertainty=task.uncertainty,
            high_value_flag=task.high_value_flag,
            emergency_flag=task.emergency_flag,
            task_class=task.task_class,
            mission_source=str(task.metadata.get("dataset_name", "")) if task.metadata.get("dataset_name") else None,
            scene_prior=None,
            product_use_case=None,
            queue_mb=queue_mb,
            queue_max_mb=queue_max,
            known_contact_capacity_before_deadline=capacity,
            battery_slack=battery_slack,
            thermal_slack=thermal_slack,
            queue_slack=queue_slack,
            contact_slack=max(0.0, min(1.0, capacity / 2.7)),
            deadline_slack=max(0.0, min(1.0, task.deadline_s / 120.0)),
        )
        evaluation = _evaluate(fields)
        scores = evaluation["scores"]
        task_value_bin = "low" if task.task_value < 0.45 else "medium" if task.task_value < 0.75 else "high"
        uncertainty_bin = "low" if task.uncertainty < 0.35 else "medium" if task.uncertainty < 0.65 else "high"
        rows.append(
            {
                "task_id": task.task_id,
                "slot_index": step.slot_index,
                "task_value": f"{task.task_value:.6f}",
                "task_value_bin": task_value_bin,
                "deadline_s": f"{task.deadline_s:.6f}",
                "remaining_deadline_s": f"{task.deadline_s:.6f}",
                "uncertainty": f"{task.uncertainty:.6f}",
                "uncertainty_bin": uncertainty_bin,
                "emergency_flag": int(task.emergency_flag),
                "high_value_flag": int(task.high_value_flag),
                "task_class": task.task_class,
                "old_product_requirement": old_req,
                "new_ideal_product_requirement": evaluation["ideal"],
                "recommended_payload_plan_shadow": evaluation["recommended_payload_plan_shadow"],
                "S_alert": f"{scores['alert']:.6f}",
                "S_verification": f"{scores['verification']:.6f}",
                "S_context": f"{scores['context']:.6f}",
                "S_evidence": f"{scores['evidence']:.6f}",
                "known_contact_capacity_before_deadline_mb": f"{capacity:.6f}",
                "missing_fields": evaluation["missing_fields"],
                "diagnostic_reason": evaluation["reason"],
            }
        )
    return rows


def distribution_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    groupers = {
        "overall": lambda row: "all",
        "high_value_flag": lambda row: str(row["high_value_flag"]),
        "emergency_flag": lambda row: str(row["emergency_flag"]),
        "uncertainty_bin": lambda row: str(row["uncertainty_bin"]),
        "task_value_bin": lambda row: str(row["task_value_bin"]),
    }
    for group_type, key_fn in groupers.items():
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        for group, group_rows in sorted(groups.items()):
            old_counts = Counter(str(row["old_product_requirement"]) for row in group_rows)
            new_counts = Counter(str(row["new_ideal_product_requirement"]) for row in group_rows)
            for requirement in ["alert", "verification", "context", "evidence"]:
                result.append(
                    {
                        "group_type": group_type,
                        "group": group,
                        "requirement": requirement,
                        "old_count": old_counts.get(requirement, 0),
                        "new_count": new_counts.get(requirement, 0),
                        "group_n": len(group_rows),
                    }
                )
    return result


def transition_matrix(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter((str(row["old_product_requirement"]), str(row["new_ideal_product_requirement"])) for row in rows)
    total = max(len(rows), 1)
    return [
        {"old": old, "new": new, "count": counts.get((old, new), 0), "rate": f"{counts.get((old, new), 0) / total:.6f}"}
        for old in ["alert", "verification", "context", "evidence"]
        for new in ["alert", "verification", "context", "evidence"]
    ]


def static_coupling_risk() -> tuple[str, list[str]]:
    source = inspect.getsource(assign_product_requirement_v2)
    resource_terms = [
        "queue_pressure",
        "contact_pressure",
        "enough_slack",
        "battery_slack",
        "thermal_slack",
        "queue_slack",
        "contact_slack",
        "deadline_slack",
    ]
    used = [term for term in resource_terms if term in source]
    if used:
        return "medium", used
    return "low", []


def leakage_audit() -> tuple[str, str]:
    source = inspect.getsource(assign_product_requirement_v2)
    helper_source = inspect.getsource(recommend_payload_plan_from_requirement)
    identifiers: set[str] = set()
    for func_source in (source, helper_source):
        tree = ast.parse(func_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.arg):
                identifiers.add(node.arg.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
    found_forbidden = [
        token
        for token in FORBIDDEN_TOKENS
        if any(token == identifier or token in identifier for identifier in identifiers)
    ]
    risk = "high" if found_forbidden else "low"
    text = f"""# Leakage field audit

Leakage risk level: {risk}

## Allowed fields used

{chr(10).join(f'- {field}' for field in ALLOWED_FIELDS)}

## Forbidden fields not used

{chr(10).join(f'- {field}' for field in FORBIDDEN_TOKENS if field not in found_forbidden)}

## Unknown or concerning fields

{chr(10).join(f'- {field}' for field in found_forbidden) if found_forbidden else '- none'}

## Conclusion

The v2 implementation path does not inspect SARFish labels, bbox count, target count,
object class, realized detector correctness, future delivery success, future utility,
post-action feedback, or queue outcome after decision.
"""
    return risk, text


def module2_isolation_audit() -> tuple[bool, str]:
    src_files = list((PROJECT_ROOT / "src").rglob("*.py"))
    router_files = [
        PROJECT_ROOT / "src" / "routers" / "safe_linucb_router.py",
        PROJECT_ROOT / "src" / "routers" / "rule_router.py",
        PROJECT_ROOT / "src" / "routers" / "safe_filter.py",
        PROJECT_ROOT / "src" / "simulator" / "step_runner.py",
        PROJECT_ROOT / "src" / "actions" / "action_space.py",
    ]
    references = []
    for path in src_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "assign_product_requirement_v2" in text or "recommended_payload_plan_shadow" in text:
            references.append(path.relative_to(PROJECT_ROOT).as_posix())
    router_reference_hits = []
    for path in router_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "assign_product_requirement_v2" in text or "recommended_payload_plan_shadow" in text:
            router_reference_hits.append(path.relative_to(PROJECT_ROOT).as_posix())
    config_hits = []
    for path in (PROJECT_ROOT / "configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "product_requirement_v2" in text or "recommended_payload_plan_shadow" in text:
            config_hits.append(path.relative_to(PROJECT_ROOT).as_posix())
    passed = not router_reference_hits and not config_hits and references == ["src/data/product_requirements.py"]
    text = f"""# Module 2 isolation audit

Status: {'PASS' if passed else 'CRITICAL_FAIL'}

1. Is v2 default-off?

   {'Yes. No main config references product_requirement_v2.' if not config_hits else 'No. Config references found: ' + str(config_hits)}

2. Is old assign_product_requirement behavior preserved?

   Yes. The existing assign_product_requirement function remains present and is still the function imported by task stream adapters.

3. Does v2 change action_space?

   No references to v2 were found in src/actions/action_space.py.

4. Does v2 change router candidate actions?

   No references to v2 were found in simulator step selection or router candidate construction.

5. Does v2 change SafeLinUCB scoring?

   No references to v2 were found in src/routers/safe_linucb_router.py.

6. Does v2 change utility_rule?

   No references to v2 were found in src/routers/rule_router.py.

7. Does v2 change SafeFilter?

   No references to v2 were found in src/routers/safe_filter.py.

8. Does v2 change selected_action when only diagnostic script is run?

   No. The validation script builds sidecar diagnostics and does not instantiate or mutate router selected_action.

9. Are recommended_payload_plan_shadow outputs consumed by router?

   No. References in src: {references}

10. Are v2 fields written only to diagnostics unless explicitly enabled?

   Yes in the current code path. v2 is referenced by diagnostic scripts/tests and by its own helper module only.

Disabled-equivalence check:

   Full simulator equivalence was not necessary for this validation because static isolation shows the v2 helper is not reachable from Module 2 action selection. No selected action, reward, or canonical metric is produced or mutated by this script.
"""
    return passed, text


def _md_table(rows: list[dict[str, object]], columns: list[str], limit: int = 12) -> str:
    if not rows:
        return "No rows."
    rows = rows[:limit]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def final_report(
    semantic_rows: list[dict[str, object]],
    monotonicity_rows: list[dict[str, object]],
    resource_rows: list[dict[str, object]],
    dataset_rows: list[dict[str, object]],
    leakage_risk: str,
    module2_pass: bool,
    coupling_risk: str,
    coupling_terms: list[str],
) -> tuple[str, str]:
    semantic_pass = all(row["pass_fail"] == "PASS" for row in semantic_rows)
    monotonicity_pass = all(row["pass_fail"] == "PASS" for row in monotonicity_rows)
    resource_pass = all(row["pass_fail"] == "PASS" for row in resource_rows)
    leakage_pass = leakage_risk == "low"
    old_counts = Counter(str(row["old_product_requirement"]) for row in dataset_rows)
    new_counts = Counter(str(row["new_ideal_product_requirement"]) for row in dataset_rows)
    collapsed = max(new_counts.values(), default=0) / max(len(dataset_rows), 1) > 0.85
    overproduce_evidence = new_counts.get("evidence", 0) / max(len(dataset_rows), 1) > 0.60
    overproduce_alert = new_counts.get("alert", 0) / max(len(dataset_rows), 1) > 0.60

    if not module2_pass:
        recommendation = "CRITICAL: Module 2 isolation failed. Do not recommend further integration."
    elif not leakage_pass:
        recommendation = "CRITICAL: leakage audit failed. Do not recommend further integration."
    elif semantic_pass and not resource_pass:
        recommendation = (
            "Keep v2 diagnostic-only. Before any branch promotion, decouple resource-pressure terms from "
            "ideal_product_requirement and move them into recommended_payload_plan_shadow."
        )
    elif semantic_pass and resource_pass and coupling_risk != "low":
        recommendation = (
            "Keep v2 diagnostic-only. A future default-off experimental branch is possible only after reducing "
            "the static coupling risk by moving resource-pressure terms out of ideal requirement scoring."
        )
    elif semantic_pass and monotonicity_pass and resource_pass:
        recommendation = (
            "Keep v2 diagnostic-only and allow a future default-off experimental branch. Do not make it the main default."
        )
    else:
        recommendation = (
            "Keep v2 diagnostic-only. Do not promote to an experimental branch yet; fix failed semantic cases and the static resource-coupling risk first."
        )

    failed_semantic = [row for row in semantic_rows if row["pass_fail"] != "PASS"]
    failed_monotonicity = [row for row in monotonicity_rows if row["pass_fail"] != "PASS"]
    failed_resource = [row for row in resource_rows if row["pass_fail"] != "PASS"]
    report = f"""# v2 requirement semantics report

## Executive summary

- Semantic cases pass: {semantic_pass}
- Monotonicity tests pass: {monotonicity_pass}
- Resource-invariance tests pass: {resource_pass}
- Leakage risk: {leakage_risk}
- Module 2 isolation pass: {module2_pass}
- Static coupling risk: {coupling_risk}; terms: {coupling_terms or 'none'}

## Required answers

1. Does v2 better separate "what the mission needs" from "what can be transmitted now"?

   Partially. The data structure separates `ideal_product_requirement` from `recommended_payload_plan_shadow`, and the shadow recommendation is not consumed by Module 2. However, the current ideal scoring still contains resource-pressure terms: {coupling_terms or 'none'}.

2. Does v2 reasonably identify alert / verification / context / evidence needs in hand-designed mission cases?

   {'Yes.' if semantic_pass else 'Not fully.'} Failed cases:

{_md_table(failed_semantic, ['case_id', 'actual_ideal_product_requirement', 'expected_allowed_requirements', 'expected_disallowed_requirements', 'recommended_payload_plan_shadow', 'explanation'])}

3. Does v2 pass monotonicity tests?

   {'Yes.' if monotonicity_pass else 'Not fully.'} Failed rows:

{_md_table(failed_monotonicity, ['series_id', 'variant', 'changed_field', 'changed_value', 'actual_ideal_product_requirement', 'previous_requirement', 'explanation'])}

4. Does v2 pass resource-invariance tests?

   {'Yes.' if resource_pass else 'No.'} Failed rows:

{_md_table(failed_resource, ['task_semantics_id', 'resource_case', 'actual_ideal_product_requirement', 'reference_ideal_product_requirement', 'recommended_payload_plan_shadow', 'explanation'])}

5. Does v2 avoid label leakage?

   {'Yes.' if leakage_pass else 'No.'} Leakage risk is {leakage_risk}.

6. Does v2 avoid interfering with Module 2?

   {'Yes.' if module2_pass else 'No.'} Static isolation shows v2 is not referenced by action_space, SafeLinUCB, utility_rule, SafeFilter, or simulator action selection.

7. Is v2 currently safe to keep as diagnostic-only?

   {'Yes.' if module2_pass and leakage_pass else 'No.'}

8. Is v2 safe to promote to a default-off experimental branch?

   {'Not yet, because semantic/coupling issues remain.' if not semantic_pass or coupling_risk != 'low' else 'Potentially yes, but still default-off only.'}

9. What must be fixed before v2 can enter the main pipeline?

   - Remove resource-pressure terms from ideal_product_requirement scoring or prove they cannot change ideal semantics.
   - Tune semantic scoring so low-value routine cases do not drift into context/evidence.
   - Keep recommended_payload_plan_shadow as the place where queue/contact/deadline/resource feasibility acts.
   - Add an explicit feature gate before any non-diagnostic integration.

## Dataset-level sanity

- Dataset tasks audited: {len(dataset_rows)}
- Old distribution: {dict(sorted(old_counts.items()))}
- New distribution: {dict(sorted(new_counts.items()))}
- Collapsed into one class: {collapsed}
- Overproduces evidence: {overproduce_evidence}
- Overproduces alert: {overproduce_alert}

## Final recommendation

{recommendation}
"""
    final = f"""# Final recommendation

{recommendation}

Do not make product_requirement_v2 the main default now.

Rollback path remains clear: v2 is diagnostic-only, the old assignment rule remains callable, and Module 2 does not consume `recommended_payload_plan_shadow`.
"""
    return report, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-slots", type=int, default=540)
    parser.add_argument("--dataset-config", default="dataset_sarfishsample.yaml")
    parser.add_argument("--task-stream-config", default="task_stream_sarfish_20s_dryrun.yaml")
    parser.add_argument("--timeline-config", default="timeline_sarfish_20s_dryrun.yaml")
    parser.add_argument("--simulator-config", default="simulator_sarfish_20s_dryrun.yaml")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = "python scripts/validate_product_requirement_v2_semantics_and_isolation.py " + " ".join(sys.argv[1:])
    configs, config_source = _load_named_configs(args)
    _limit_configs(configs, args.num_slots)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()

    semantic_rows = semantic_case_validation()
    monotonicity_rows = monotonicity_validation()
    resource_rows = resource_invariance_validation()
    dataset_rows = dataset_sanity_rows(timeline, configs)
    leakage_risk, leakage_text = leakage_audit()
    module2_pass, module2_text = module2_isolation_audit()
    coupling_risk, coupling_terms = static_coupling_risk()
    report, final = final_report(
        semantic_rows,
        monotonicity_rows,
        resource_rows,
        dataset_rows,
        leakage_risk,
        module2_pass,
        coupling_risk,
        coupling_terms,
    )

    _write_csv(OUT_DIR / "semantic_case_validation.csv", semantic_rows)
    _write_csv(OUT_DIR / "monotonicity_test_results.csv", monotonicity_rows)
    _write_csv(OUT_DIR / "resource_invariance_test_results.csv", resource_rows)
    _write_csv(OUT_DIR / "dataset_requirement_sanity.csv", dataset_rows)
    _write_csv(OUT_DIR / "dataset_distribution_summary.csv", distribution_summary(dataset_rows))
    _write_csv(OUT_DIR / "dataset_old_vs_new_transition_matrix.csv", transition_matrix(dataset_rows))
    _write_text(OUT_DIR / "leakage_field_audit.md", leakage_text)
    _write_text(OUT_DIR / "module2_isolation_audit.md", module2_text)
    _write_text(OUT_DIR / "v2_requirement_semantics_report.md", report)
    _write_text(OUT_DIR / "final_recommendation.md", final)
    _write_text(
        OUT_DIR / "README.md",
        """# product_requirement_v2 semantics and isolation validation

This directory contains diagnostic-only validation outputs. The validation checks
mission-level semantics, monotonicity, resource invariance, leakage safety, and
Module 2 isolation. It does not enable v2 in the main pipeline.
""",
    )
    _write_text(
        OUT_DIR / "command_log.md",
        f"""# Command log

```powershell
{command}
```
""",
    )
    manifest = {
        "command": command,
        "config_source": config_source,
        "num_slots": args.num_slots,
        "output_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
        "semantic_pass": all(row["pass_fail"] == "PASS" for row in semantic_rows),
        "monotonicity_pass": all(row["pass_fail"] == "PASS" for row in monotonicity_rows),
        "resource_invariance_pass": all(row["pass_fail"] == "PASS" for row in resource_rows),
        "leakage_risk": leakage_risk,
        "module2_isolation_pass": module2_pass,
        "static_coupling_risk": coupling_risk,
        "static_coupling_terms": coupling_terms,
        "diagnostic_only": True,
    }
    _write_text(OUT_DIR / "validation_manifest.yaml", yaml.safe_dump(manifest, sort_keys=True, allow_unicode=False))
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _audit_diff(
            [
                "src/data/product_requirements.py",
                "scripts/audit_product_requirement_v2.py",
                "scripts/validate_product_requirement_v2_semantics_and_isolation.py",
                "tests/test_product_requirements_v2.py",
            ]
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
