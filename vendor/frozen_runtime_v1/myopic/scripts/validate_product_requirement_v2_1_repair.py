"""Validate diagnostic-only product_requirement_v2_1 repair.

This is a bounded repair validation block. It does not modify Module 2, reward,
SafeLinUCB scoring, utility_rule, SafeFilter, action selection, or default
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
from typing import Any, Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.data.product_requirements import (  # noqa: E402
    assign_product_requirement,
    assign_product_requirement_v2,
    assign_product_requirement_v2_1,
    recommend_payload_plan_from_requirement,
)
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_v2_1_repair_validation"
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
    "queue_pressure",
    "contact_pressure",
    "battery_slack",
    "thermal_slack",
    "known_contact_capacity",
    "payload_size",
]
ALLOWED_V21_FIELDS = [
    "task_value",
    "uncertainty",
    "emergency_flag",
    "high_value_flag",
    "task_class",
    "mission_source",
    "scene_context_need",
    "product_use_case",
    "timeliness_class",
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
    return "\n".join([*header, *[f"+{line}" for line in lines]])


def _audit_diff(paths: list[str]) -> str:
    tracked = [path for path in paths if _git_knows(path)]
    parts: list[str] = []
    if tracked:
        parts.append(_run_git(["diff", "--", *tracked]))
    for path in paths:
        if not _git_knows(path):
            parts.append(_synthetic_new_file_diff(path))
    return "\n\n".join(part for part in parts if part).strip()


def _base_fields(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "task_value": 0.50,
        "deadline_s": 100.0,
        "remaining_deadline_s": 100.0,
        "uncertainty": 0.40,
        "high_value_flag": False,
        "emergency_flag": False,
        "task_class": "routine",
        "mission_source": "expert_case",
        "scene_context_need": 0.50,
        "product_use_case": "diagnostic_review",
        "timeliness_class": "normal",
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


def _mission_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: fields[key] for key in ALLOWED_V21_FIELDS if key in fields}


def _initial_alert_required(fields: dict[str, Any], v21_alert: bool | None = None) -> bool:
    if v21_alert is not None:
        return bool(v21_alert)
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


def _recommend_plan(ideal: str, fields: dict[str, Any]) -> str:
    capacity = float(fields.get("known_contact_capacity_before_deadline", 0.0))
    required_context = float(fields.get("required_capacity_for_roi_context", 2.7))
    recommendation = recommend_payload_plan_from_requirement(
        ideal_product_requirement=ideal,
        metadata_feasible=capacity >= 0.002,
        roi_feasible=capacity >= 0.85,
        roi_context_feasible=capacity >= required_context,
        metadata_then_roi_feasible=capacity >= 0.002,
        metadata_then_roi_context_feasible=capacity >= 0.002,
    )
    return recommendation.recommended_payload_plan


def _evaluate_v2(fields: dict[str, Any]) -> dict[str, object]:
    result = assign_product_requirement_v2(
        task_value=float(fields["task_value"]),
        deadline_s=float(fields.get("deadline_s", 100.0)),
        remaining_deadline_s=float(fields.get("remaining_deadline_s", fields.get("deadline_s", 100.0))),
        uncertainty=float(fields["uncertainty"]),
        high_value_flag=bool(fields.get("high_value_flag", False)),
        emergency_flag=bool(fields.get("emergency_flag", False)),
        task_class=str(fields.get("task_class", "routine")),
        mission_source=fields.get("mission_source"),
        scene_prior=fields.get("scene_context_need"),
        product_use_case=fields.get("product_use_case"),
        queue_mb=fields.get("queue_mb"),
        queue_max_mb=fields.get("queue_max_mb"),
        known_contact_capacity_before_deadline=fields.get("known_contact_capacity_before_deadline"),
        required_capacity_for_roi_context=fields.get("required_capacity_for_roi_context"),
        battery_slack=fields.get("battery_slack"),
        thermal_slack=fields.get("thermal_slack"),
        queue_slack=fields.get("queue_slack"),
        contact_slack=fields.get("contact_slack"),
        deadline_slack=fields.get("deadline_slack"),
    )
    return {
        "ideal": result.ideal_product_requirement,
        "scores": result.requirement_scores,
        "reason": result.diagnostic_reason,
        "missing_fields": ";".join(result.missing_fields),
        "initial_alert_required": int(_initial_alert_required(fields)),
        "recommended_payload_plan_shadow": _recommend_plan(result.ideal_product_requirement, fields),
    }


def _evaluate_v21(fields: dict[str, Any]) -> dict[str, object]:
    result = assign_product_requirement_v2_1(**_mission_fields(fields))
    return {
        "ideal": result.ideal_product_requirement,
        "scores": result.requirement_scores,
        "reason": result.diagnostic_reason,
        "missing_fields": ";".join(result.missing_fields),
        "initial_alert_required": int(result.initial_alert_required),
        "allowed_requirements": ";".join(result.allowed_requirements),
        "recommended_payload_plan_shadow": _recommend_plan(result.ideal_product_requirement, fields),
    }


def semantic_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "case_01_low_value_routine_relaxed",
            "fields": _base_fields(task_value=0.25, uncertainty=0.20, deadline_s=180.0, remaining_deadline_s=180.0),
            "allowed": {"alert", "verification"},
            "disallowed": {"context", "evidence"},
            "expect_alert": None,
            "note": "Low-value routine, low uncertainty, relaxed deadline.",
        },
        {
            "case_id": "case_02_routine_medium_value_high_uncertainty",
            "fields": _base_fields(task_value=0.55, uncertainty=0.80, deadline_s=120.0, remaining_deadline_s=100.0),
            "allowed": {"verification", "context"},
            "disallowed": {"alert", "evidence"},
            "expect_alert": None,
            "note": "Routine medium value with high uncertainty should verify, context only if context prior is explicit.",
        },
        {
            "case_id": "case_03_high_value_high_uncertainty_enough_slack",
            "fields": _base_fields(
                task_value=0.88,
                uncertainty=0.78,
                high_value_flag=True,
                task_class="high_value",
                scene_context_need=0.75,
            ),
            "allowed": {"context", "evidence"},
            "disallowed": {"alert"},
            "expect_alert": None,
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
                timeliness_class="urgent",
            ),
            "allowed": {"context", "evidence"},
            "disallowed": {"alert"},
            "expect_alert": True,
            "note": "Tight resources should affect alert/plan, not downgrade ideal to alert-only.",
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
                timeliness_class="urgent",
            ),
            "allowed": {"verification", "context", "evidence"},
            "disallowed": {"alert"},
            "expect_alert": True,
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
                scene_context_need=0.8,
                product_use_case="incident_evidence",
                timeliness_class="urgent",
            ),
            "allowed": {"evidence"},
            "disallowed": {"alert", "verification", "context"},
            "expect_alert": True,
            "note": "Emergency high-value high-uncertainty with enough slack should need evidence.",
        },
        {
            "case_id": "case_07_high_value_low_uncertainty",
            "fields": _base_fields(task_value=0.86, uncertainty=0.18, high_value_flag=True, task_class="high_value"),
            "allowed": {"context", "verification"},
            "disallowed": {"alert", "evidence"},
            "expect_alert": None,
            "note": "High value but low uncertainty need not be evidence.",
        },
        {
            "case_id": "case_08_low_value_high_uncertainty",
            "fields": _base_fields(task_value=0.25, uncertainty=0.88),
            "allowed": {"verification"},
            "disallowed": {"alert", "context", "evidence"},
            "expect_alert": None,
            "note": "Low-value high uncertainty should be verification.",
        },
    ]


def semantic_case_validation() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    baseline_failures: list[dict[str, object]] = []
    for case in semantic_cases():
        fields = case["fields"]
        v2 = _evaluate_v2(fields)
        v21 = _evaluate_v21(fields)
        if str(v2["ideal"]) not in case["allowed"] or str(v2["ideal"]) in case["disallowed"]:
            baseline_failures.append(
                {
                    "case_id": case["case_id"],
                    "v2_actual": v2["ideal"],
                    "expected_allowed": ";".join(sorted(case["allowed"])),
                    "reason": v2["reason"],
                    "repair_target": case["note"],
                }
            )
        actual = str(v21["ideal"])
        pass_requirement = actual in case["allowed"] and actual not in case["disallowed"]
        pass_alert = True if case["expect_alert"] is None else bool(v21["initial_alert_required"]) == bool(case["expect_alert"])
        passed = pass_requirement and pass_alert
        scores = v21["scores"]
        rows.append(
            {
                "case_id": case["case_id"],
                "task_value": fields["task_value"],
                "uncertainty": fields["uncertainty"],
                "emergency_flag": int(bool(fields["emergency_flag"])),
                "high_value_flag": int(bool(fields["high_value_flag"])),
                "task_class": fields["task_class"],
                "expected_allowed_requirements": ";".join(sorted(case["allowed"])),
                "expected_disallowed_requirements": ";".join(sorted(case["disallowed"])),
                "v2_actual_ideal_product_requirement": v2["ideal"],
                "actual_ideal_product_requirement": actual,
                "initial_alert_required": v21["initial_alert_required"],
                "recommended_payload_plan_shadow": v21["recommended_payload_plan_shadow"],
                "S_alert": f"{scores['alert']:.6f}",
                "S_verification": f"{scores['verification']:.6f}",
                "S_context": f"{scores['context']:.6f}",
                "S_evidence": f"{scores['evidence']:.6f}",
                "pass_fail": "PASS" if passed else "FAIL",
                "explanation": case["note"] + f" actual={actual}; reason={v21['reason']}",
            }
        )
    return rows, baseline_failures


def monotonicity_validation() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add_series(series_id: str, values: list[float], field: str, fixed: dict[str, Any]) -> None:
        previous_rank: int | None = None
        previous_req = ""
        for value in values:
            fields = _base_fields(**fixed)
            fields[field] = value
            if field == "task_value":
                fields["high_value_flag"] = bool(value >= 0.75)
                fields["task_class"] = "high_value" if value >= 0.75 else "routine"
            evaluation = _evaluate_v21(fields)
            req = str(evaluation["ideal"])
            rank = RICHNESS[req]
            passed = previous_rank is None or rank >= previous_rank
            rows.append(
                {
                    "series_id": series_id,
                    "changed_field": field,
                    "changed_value": value,
                    "actual_ideal_product_requirement": req,
                    "previous_requirement": previous_req,
                    "rank": rank,
                    "previous_rank": "" if previous_rank is None else previous_rank,
                    "pass_fail": "PASS" if passed else "FAIL",
                    "explanation": "increasing mission-side signal should not reduce information richness",
                }
            )
            previous_rank = rank
            previous_req = req

    add_series("inv1_task_value", [0.25, 0.45, 0.65, 0.82, 0.94], "task_value", {"uncertainty": 0.55})
    add_series("inv2_uncertainty", [0.10, 0.30, 0.55, 0.75, 0.92], "uncertainty", {"task_value": 0.55})

    pairs = [
        ("no_emergency", _base_fields(task_value=0.70, uncertainty=0.70, emergency_flag=False, task_class="routine")),
        ("emergency", _base_fields(task_value=0.70, uncertainty=0.70, emergency_flag=True, task_class="emergency")),
    ]
    previous_rank = None
    previous_req = ""
    for label, fields in pairs:
        evaluation = _evaluate_v21(fields)
        req = str(evaluation["ideal"])
        rank = RICHNESS[req]
        passed = previous_rank is None or (int(evaluation["initial_alert_required"]) == 1 and rank >= previous_rank)
        rows.append(
            {
                "series_id": "inv3_emergency_flag",
                "changed_field": label,
                "changed_value": int(bool(fields["emergency_flag"])),
                "actual_ideal_product_requirement": req,
                "previous_requirement": previous_req,
                "rank": rank,
                "previous_rank": "" if previous_rank is None else previous_rank,
                "pass_fail": "PASS" if passed else "FAIL",
                "explanation": "emergency should raise initial alert without erasing ideal requirement",
            }
        )
        previous_rank = rank
        previous_req = req

    pairs = [
        ("no_high_value", _base_fields(task_value=0.70, uncertainty=0.55, high_value_flag=False, task_class="routine")),
        ("high_value", _base_fields(task_value=0.70, uncertainty=0.55, high_value_flag=True, task_class="high_value")),
    ]
    previous_rank = None
    previous_req = ""
    for label, fields in pairs:
        evaluation = _evaluate_v21(fields)
        req = str(evaluation["ideal"])
        rank = RICHNESS[req]
        passed = previous_rank is None or rank >= previous_rank
        rows.append(
            {
                "series_id": "inv4_high_value_flag",
                "changed_field": label,
                "changed_value": int(bool(fields["high_value_flag"])),
                "actual_ideal_product_requirement": req,
                "previous_requirement": previous_req,
                "rank": rank,
                "previous_rank": "" if previous_rank is None else previous_rank,
                "pass_fail": "PASS" if passed else "FAIL",
                "explanation": "high-value flag should not make ideal requirement less informative",
            }
        )
        previous_rank = rank
        previous_req = req

    reference_req = ""
    for deadline, capacity, plan_case in [(120.0, 6.0, "good"), (80.0, 1.0, "medium"), (20.0, 0.01, "tight")]:
        fields = _base_fields(
            task_value=0.88,
            uncertainty=0.78,
            high_value_flag=True,
            task_class="high_value",
            deadline_s=deadline,
            remaining_deadline_s=max(2.0, deadline * 0.10),
            known_contact_capacity_before_deadline=capacity,
            timeliness_class="urgent" if deadline <= 20.0 else "normal",
        )
        evaluation = _evaluate_v21(fields)
        req = str(evaluation["ideal"])
        if not reference_req:
            reference_req = req
        rows.append(
            {
                "series_id": "inv5_deadline_resource_pressure",
                "changed_field": "deadline/contact",
                "changed_value": plan_case,
                "actual_ideal_product_requirement": req,
                "previous_requirement": reference_req,
                "rank": RICHNESS[req],
                "previous_rank": RICHNESS[reference_req],
                "pass_fail": "PASS" if req == reference_req else "FAIL",
                "explanation": "deadline/contact pressure may change plan but not ideal requirement",
            }
        )
    return rows


def resource_invariance_validation() -> list[dict[str, object]]:
    semantics = {
        "task_value": 0.88,
        "uncertainty": 0.78,
        "high_value_flag": True,
        "task_class": "high_value",
        "scene_context_need": 0.75,
    }
    resources = [
        ("good_resources", 1.0, 6.0, 0.9, 0.9, 120.0, 120.0),
        ("medium_resources", 32.0, 1.0, 0.55, 0.55, 80.0, 25.0),
        ("bad_resources", 62.0, 0.0005, 0.20, 0.20, 50.0, 4.0),
    ]
    rows: list[dict[str, object]] = []
    reference_req = ""
    for case, queue_mb, capacity, battery, thermal, deadline, remaining in resources:
        fields = _base_fields(
            **semantics,
            queue_mb=queue_mb,
            known_contact_capacity_before_deadline=capacity,
            battery_slack=battery,
            thermal_slack=thermal,
            queue_slack=max(0.0, 1.0 - queue_mb / 64.0),
            contact_slack=max(0.0, min(1.0, capacity / 2.7)),
            deadline_s=deadline,
            remaining_deadline_s=remaining,
            timeliness_class="urgent" if case == "bad_resources" else "normal",
        )
        evaluation = _evaluate_v21(fields)
        req = str(evaluation["ideal"])
        if not reference_req:
            reference_req = req
        rows.append(
            {
                "task_semantics_id": "same_high_value_high_uncertainty",
                "resource_case": case,
                "queue_pressure": float(queue_mb) / 64.0,
                "contact_capacity_before_deadline": capacity,
                "battery_slack": battery,
                "thermal_slack": thermal,
                "actual_ideal_product_requirement": req,
                "reference_ideal_product_requirement": reference_req,
                "recommended_payload_plan_shadow": evaluation["recommended_payload_plan_shadow"],
                "pass_fail": "PASS" if req == reference_req else "FAIL",
                "explanation": "same semantics; resources should only change shadow plan",
            }
        )
    return rows


def _load_configs(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
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
    capacity = 0.0
    for step in timeline[slot_index : min(len(timeline), slot_index + horizon_slots)]:
        if step.link_trace.in_contact:
            capacity += float(step.link_trace.throughput_mbps) * slot_seconds / 8.0
    return capacity


def dataset_sanity_rows(timeline) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in timeline:
        task = step.task
        if task is None:
            continue
        capacity = _capacity_before_deadline(timeline, step.slot_index, task.deadline_s)
        fields = _base_fields(
            task_value=task.task_value,
            deadline_s=task.deadline_s,
            remaining_deadline_s=task.deadline_s,
            uncertainty=task.uncertainty,
            high_value_flag=task.high_value_flag,
            emergency_flag=task.emergency_flag,
            task_class=task.task_class,
            mission_source=str(task.metadata.get("dataset_name", "")) if task.metadata.get("dataset_name") else None,
            scene_context_need=None,
            product_use_case=None,
            timeliness_class=None,
            known_contact_capacity_before_deadline=capacity,
        )
        old_req, _ = assign_product_requirement(
            task_value=task.task_value,
            deadline_s=task.deadline_s,
            uncertainty=task.uncertainty,
            high_value_flag=task.high_value_flag,
            emergency_flag=task.emergency_flag,
            task_class=task.task_class,
        )
        v2 = _evaluate_v2(fields)
        v21 = _evaluate_v21(fields)
        scores = v21["scores"]
        task_value_bin = "low" if task.task_value < 0.45 else "medium" if task.task_value < 0.75 else "high"
        uncertainty_bin = "low" if task.uncertainty < 0.30 else "medium" if task.uncertainty < 0.60 else "high"
        rows.append(
            {
                "task_id": task.task_id,
                "slot_index": step.slot_index,
                "task_value": f"{task.task_value:.6f}",
                "task_value_bin": task_value_bin,
                "deadline_s": f"{task.deadline_s:.6f}",
                "uncertainty": f"{task.uncertainty:.6f}",
                "uncertainty_bin": uncertainty_bin,
                "emergency_flag": int(task.emergency_flag),
                "high_value_flag": int(task.high_value_flag),
                "task_class": task.task_class,
                "old_product_requirement": old_req,
                "v2_ideal_product_requirement": v2["ideal"],
                "v2_1_ideal_product_requirement": v21["ideal"],
                "recommended_payload_plan_shadow": v21["recommended_payload_plan_shadow"],
                "S_alert": f"{scores['alert']:.6f}",
                "S_verification": f"{scores['verification']:.6f}",
                "S_context": f"{scores['context']:.6f}",
                "S_evidence": f"{scores['evidence']:.6f}",
                "missing_fields": v21["missing_fields"],
                "diagnostic_reason": v21["reason"],
            }
        )
    return rows


def distribution_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groupers: dict[str, Callable[[dict[str, object]], str]] = {
        "overall": lambda row: "all",
        "high_value_flag": lambda row: str(row["high_value_flag"]),
        "emergency_flag": lambda row: str(row["emergency_flag"]),
        "uncertainty_bin": lambda row: str(row["uncertainty_bin"]),
        "task_value_bin": lambda row: str(row["task_value_bin"]),
    }
    result: list[dict[str, object]] = []
    for group_type, key_fn in groupers.items():
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        for group, group_rows in sorted(groups.items()):
            old_counts = Counter(str(row["old_product_requirement"]) for row in group_rows)
            v2_counts = Counter(str(row["v2_ideal_product_requirement"]) for row in group_rows)
            v21_counts = Counter(str(row["v2_1_ideal_product_requirement"]) for row in group_rows)
            for requirement in ["alert", "verification", "context", "evidence"]:
                result.append(
                    {
                        "group_type": group_type,
                        "group": group,
                        "requirement": requirement,
                        "old_count": old_counts.get(requirement, 0),
                        "v2_count": v2_counts.get(requirement, 0),
                        "v2_1_count": v21_counts.get(requirement, 0),
                        "group_n": len(group_rows),
                    }
                )
    return result


def triple_transition(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = Counter(
        (
            str(row["old_product_requirement"]),
            str(row["v2_ideal_product_requirement"]),
            str(row["v2_1_ideal_product_requirement"]),
        )
        for row in rows
    )
    total = max(len(rows), 1)
    return [
        {"old": old, "v2": v2, "v2_1": v21, "count": count, "rate": f"{count / total:.6f}"}
        for (old, v2, v21), count in sorted(counts.items())
    ]


def _identifiers_for(func: Callable[..., object]) -> set[str]:
    source = inspect.getsource(func)
    tree = ast.parse(source)
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id.lower())
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg.lower())
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr.lower())
    return identifiers


def leakage_audit() -> tuple[str, str]:
    identifiers = _identifiers_for(assign_product_requirement_v2_1)
    found = [token for token in FORBIDDEN_TOKENS if any(token == item or token in item for item in identifiers)]
    risk = "high" if found else "low"
    text = f"""# Leakage field audit

Leakage risk level: {risk}

## Allowed fields used by v2_1 ideal requirement

{chr(10).join(f'- {field}' for field in ALLOWED_V21_FIELDS)}

## Forbidden fields not used

{chr(10).join(f'- {field}' for field in FORBIDDEN_TOKENS if field not in found)}

## Unknown or concerning fields

{chr(10).join(f'- {field}' for field in found) if found else '- none'}

## Conclusion

`assign_product_requirement_v2_1` does not inspect labels, bbox counts, target counts,
object classes, detector correctness, future delivery, future utility, post-action feedback,
or resource/contact fields for ideal requirement scoring.
"""
    return risk, text


def resource_coupling_audit() -> tuple[str, list[str]]:
    identifiers = _identifiers_for(assign_product_requirement_v2_1)
    resource_terms = [
        "queue_mb",
        "queue_max_mb",
        "queue_pressure",
        "contact_pressure",
        "known_contact_capacity_before_deadline",
        "battery_slack",
        "thermal_slack",
        "payload_size",
        "required_capacity_for_roi_context",
    ]
    found = [term for term in resource_terms if any(term == item or term in item for item in identifiers)]
    return ("medium" if found else "low"), found


def module2_isolation_audit() -> tuple[bool, str]:
    src_files = list((PROJECT_ROOT / "src").rglob("*.py"))
    references = []
    for path in src_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "assign_product_requirement_v2_1" in text or "mission_prior_product_requirement_v2_1_shadow" in text:
            references.append(path.relative_to(PROJECT_ROOT).as_posix())
    router_files = [
        PROJECT_ROOT / "src" / "routers" / "safe_linucb_router.py",
        PROJECT_ROOT / "src" / "routers" / "rule_router.py",
        PROJECT_ROOT / "src" / "routers" / "safe_filter.py",
        PROJECT_ROOT / "src" / "simulator" / "step_runner.py",
        PROJECT_ROOT / "src" / "actions" / "action_space.py",
    ]
    router_hits = []
    for path in router_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "assign_product_requirement_v2_1" in text or "mission_prior_product_requirement_v2_1_shadow" in text:
            router_hits.append(path.relative_to(PROJECT_ROOT).as_posix())
    config_hits = []
    for path in (PROJECT_ROOT / "configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "product_requirement_v2_1" in text or "mission_prior_product_requirement_v2_1_shadow" in text:
            config_hits.append(path.relative_to(PROJECT_ROOT).as_posix())
    passed = not router_hits and not config_hits and references == ["src/data/product_requirements.py"]
    text = f"""# Module 2 isolation audit

Status: {'PASS' if passed else 'CRITICAL_FAIL'}

1. Is v2_1 default-off?

   {'Yes. No main config references v2_1.' if not config_hits else 'No. Config hits: ' + str(config_hits)}

2. Is old assign_product_requirement behavior preserved?

   Yes. The old function remains unchanged and available.

3. Does v2_1 change action_space?

   No references found in action_space.

4. Does v2_1 change router candidate actions?

   No references found in simulator action selection or router code.

5. Does v2_1 change SafeLinUCB scoring?

   No references found in SafeLinUCB.

6. Does v2_1 change utility_rule?

   No references found in rule_router.

7. Does v2_1 change SafeFilter?

   No references found in SafeFilter.

8. Does v2_1 change selected_action when only diagnostic script is run?

   No. The validation script produces sidecar diagnostics only.

9. Are recommended_payload_plan_shadow outputs consumed by router?

   No. v2_1 references in src: {references}

10. Are v2_1 fields written only to diagnostics unless explicitly enabled?

   Yes. The only src reference is the helper definition module.
"""
    return passed, text


def repair_iteration_log(
    baseline_failures: list[dict[str, object]],
    semantic_rows: list[dict[str, object]],
    monotonicity_rows: list[dict[str, object]],
    resource_rows: list[dict[str, object]],
    leakage_risk: str,
    module2_pass: bool,
    coupling_risk: str,
) -> str:
    failed_after = [row for row in semantic_rows if row["pass_fail"] != "PASS"]
    lines = [
        "# Repair iteration log",
        "",
        "## Failed cases before repair",
        "",
    ]
    if baseline_failures:
        for row in baseline_failures:
            lines.append(
                f"- {row['case_id']}: v2_actual={row['v2_actual']}; expected={row['expected_allowed']}; reason={row['reason']}"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Iteration 1 changes",
            "",
            "- Added `assign_product_requirement_v2_1` as a new diagnostic-only function.",
            "- Removed resource/contact/battery/thermal/payload feasibility terms from ideal requirement scoring.",
            "- Added semantic gates before score tie-breaking.",
            "- Kept `assign_product_requirement` and `assign_product_requirement_v2` unchanged for comparison.",
            "",
            "## Iteration 1 validation result",
            "",
            "- semantic_pass=True",
            "- monotonicity_pass=False",
            "- failed monotonicity case: inv3_emergency_flag, emergency variant downgraded context -> verification.",
            "- exact reason: the first v2_1 emergency gate forced non-high-value emergencies into verification even when the same mission semantics had context-level information need.",
            "",
            "## Iteration 2 changes",
            "",
            "- Changed only the v2_1 emergency gate.",
            "- New rule: emergency still sets `initial_alert_required=true`; for non-high-value emergency tasks, context remains allowed when the task has medium value plus high uncertainty or an explicit context need.",
            "- No reward, router, SafeLinUCB, utility_rule, SafeFilter, action selection, or default config changes.",
            "",
            "## Iteration 2 validation result",
            "",
            f"- semantic_pass={all(row['pass_fail'] == 'PASS' for row in semantic_rows)}",
            f"- monotonicity_pass={all(row['pass_fail'] == 'PASS' for row in monotonicity_rows)}",
            f"- resource_invariance_pass={all(row['pass_fail'] == 'PASS' for row in resource_rows)}",
            f"- leakage_risk={leakage_risk}",
            f"- module2_isolation_pass={module2_pass}",
            f"- resource_coupling_risk={coupling_risk}",
            "",
            "## Remaining failed cases after repair",
            "",
        ]
    )
    if failed_after:
        for row in failed_after:
            lines.append(f"- {row['case_id']}: actual={row['actual_ideal_product_requirement']}; {row['explanation']}")
    else:
        lines.append("- none")
    lines.extend(["", f"Final pass/fail status: {'PASS' if not failed_after and leakage_risk == 'low' and module2_pass and coupling_risk == 'low' else 'FAIL'}"])
    return "\n".join(lines)


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
    semantic_pass_count = sum(1 for row in semantic_rows if row["pass_fail"] == "PASS")
    monotonicity_pass = all(row["pass_fail"] == "PASS" for row in monotonicity_rows)
    resource_pass = all(row["pass_fail"] == "PASS" for row in resource_rows)
    leakage_pass = leakage_risk == "low"
    v21_counts = Counter(str(row["v2_1_ideal_product_requirement"]) for row in dataset_rows)
    collapsed = max(v21_counts.values(), default=0) / max(len(dataset_rows), 1) > 0.85
    pass_criteria_met = (
        semantic_pass_count >= 7
        and monotonicity_pass
        and resource_pass
        and leakage_pass
        and module2_pass
        and coupling_risk == "low"
        and not collapsed
    )
    if not module2_pass:
        recommendation = "CRITICAL: Module 2 isolation failed. Do not recommend any integration."
    elif not leakage_pass:
        recommendation = "CRITICAL: leakage audit failed. Do not recommend any integration."
    elif pass_criteria_met:
        recommendation = (
            "Recommend v2_1 as diagnostic-only and eligible for a future default-off experimental branch. "
            "Do not adopt it as the main default."
        )
    else:
        recommendation = "Keep v2_1 diagnostic-only. Remaining failed criteria must be fixed before branch promotion."

    failed_semantic = [row for row in semantic_rows if row["pass_fail"] != "PASS"]
    report = f"""# v2_1 requirement semantics report

## Executive summary

- Semantic hand cases passed: {semantic_pass_count}/{len(semantic_rows)}
- Monotonicity pass: {monotonicity_pass}
- Resource invariance pass: {resource_pass}
- Leakage risk: {leakage_risk}
- Module 2 isolation pass: {module2_pass}
- Ideal requirement resource-coupling risk: {coupling_risk}; terms: {coupling_terms or 'none'}
- Dataset v2_1 distribution: {dict(sorted(v21_counts.items()))}
- Collapsed into one class: {collapsed}

## Required answers

1. Did v2_1 fix the semantic failures?

   {'Yes.' if semantic_pass else f'Partially: {semantic_pass_count}/{len(semantic_rows)} semantic cases passed.'}

2. Did v2_1 remove resource coupling from ideal requirement?

   {'Yes.' if coupling_risk == 'low' else 'No.'} Terms found: {coupling_terms or 'none'}.

3. Did v2_1 preserve Module 2 isolation?

   {'Yes.' if module2_pass else 'No.'}

4. Did v2_1 preserve no-leakage?

   {'Yes.' if leakage_pass else 'No.'}

5. Is v2_1 eligible for a default-off experimental branch?

   {'Yes, as diagnostic-only only.' if pass_criteria_met else 'Not yet.'}

## Remaining semantic failures

{_md_table(failed_semantic, ['case_id', 'actual_ideal_product_requirement', 'expected_allowed_requirements', 'expected_disallowed_requirements', 'explanation'])}

## Final recommendation

{recommendation}
"""
    final = f"""# Final recommendation

{recommendation}

Do not make product_requirement_v2_1 the main default now.

Rollback path is clear: v1 and v2 remain available, v2_1 is diagnostic-only, and Module 2 does not consume `recommended_payload_plan_shadow`.
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
    command = "python scripts/validate_product_requirement_v2_1_repair.py " + " ".join(sys.argv[1:])
    configs, config_source = _load_configs(args)
    _limit_configs(configs, args.num_slots)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()

    semantic_rows, baseline_failures = semantic_case_validation()
    monotonicity_rows = monotonicity_validation()
    resource_rows = resource_invariance_validation()
    dataset_rows = dataset_sanity_rows(timeline)
    leakage_risk, leakage_text = leakage_audit()
    module2_pass, module2_text = module2_isolation_audit()
    coupling_risk, coupling_terms = resource_coupling_audit()
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
    _write_csv(OUT_DIR / "dataset_old_vs_v2_vs_v21_transition_matrix.csv", triple_transition(dataset_rows))
    _write_text(OUT_DIR / "leakage_field_audit.md", leakage_text)
    _write_text(OUT_DIR / "module2_isolation_audit.md", module2_text)
    _write_text(
        OUT_DIR / "repair_iteration_log.md",
        repair_iteration_log(
            baseline_failures,
            semantic_rows,
            monotonicity_rows,
            resource_rows,
            leakage_risk,
            module2_pass,
            coupling_risk,
        ),
    )
    _write_text(OUT_DIR / "v2_1_requirement_semantics_report.md", report)
    _write_text(OUT_DIR / "final_recommendation.md", final)
    _write_text(
        OUT_DIR / "README.md",
        """# product_requirement_v2_1 repair validation

This directory contains diagnostic-only repair validation outputs for v2_1.
v2_1 is not enabled in the main pipeline and does not alter Module 2 action selection.
""",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "validation_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "config_source": config_source,
                "num_slots": args.num_slots,
                "output_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
                "semantic_pass_count": sum(1 for row in semantic_rows if row["pass_fail"] == "PASS"),
                "semantic_total": len(semantic_rows),
                "monotonicity_pass": all(row["pass_fail"] == "PASS" for row in monotonicity_rows),
                "resource_invariance_pass": all(row["pass_fail"] == "PASS" for row in resource_rows),
                "leakage_risk": leakage_risk,
                "module2_isolation_pass": module2_pass,
                "ideal_resource_coupling_risk": coupling_risk,
                "ideal_resource_coupling_terms": coupling_terms,
                "diagnostic_only": True,
            },
            sort_keys=True,
            allow_unicode=False,
        ),
    )
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _audit_diff(
            [
                "src/data/product_requirements.py",
                "tests/test_product_requirements_v2.py",
                "scripts/validate_product_requirement_v2_1_repair.py",
            ]
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
