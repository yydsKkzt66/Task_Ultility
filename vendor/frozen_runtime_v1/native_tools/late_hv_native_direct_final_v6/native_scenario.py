"""Primitive neutral-bundle to repository-native scenario reconstruction."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Callable

try:
    from .contracts import PRODUCT_STAGE_MAP
except ImportError:
    from contracts import PRODUCT_STAGE_MAP


def stage_for_product(product: str) -> str | None:
    return PRODUCT_STAGE_MAP.get(str(product or "").strip().lower())


def _f(value: Any, default: float = 0.0) -> float:
    return default if value in (None, "") else float(value)


def _i(value: Any, default: int = 0) -> int:
    return default if value in (None, "") else int(float(value))


def _b(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


@dataclass
class NativeScenario:
    configs: dict[str, Any]
    timeline: list[Any]
    canonical_tasks: tuple[dict[str, str], ...]
    task_hash: str
    environment_hash: str
    benchmark_fingerprint: str
    runtime_task_values: dict[str, float]
    config_value_multiplier: float
    field_mismatch_count: int = 0


def reconstruct_native_scenario(bundle: Any, task_type: Callable[..., Any], resource_type: Callable[..., Any],
                                link_type: Callable[..., Any], step_type: Callable[..., Any]) -> NativeScenario:
    tasks: dict[str, Any] = {}
    mismatches = 0
    for row in bundle.tasks:
        metadata = json.loads(row.get("metadata_json") or "{}")
        arrival_slot = _i(row["arrival_time_s"]) // _i(bundle.contract["slot_seconds"])
        task = task_type(
            task_id=row["task_id"], image_path=row["image_path"], label_path=row["label_path"],
            scene_id=row["scene_id"], virtual_time=arrival_slot,
            task_value=float(Decimal(row["effective_task_value"])),
            deadline_s=float(Decimal(row["original_deadline_slack_s"])),
            difficulty=_f(row["difficulty"]), uncertainty=_f(row["uncertainty"]), metadata=metadata,
            arrival_slot=arrival_slot, observation_window_id=row["observation_window_id"],
            task_class=row["task_class"], quality_requirement=_f(row["quality_requirement"]),
            information_requirement=_f(row["information_requirement"]),
            high_value_flag=_b(row["high_value_flag"]), emergency_flag=_b(row["emergency_flag"]),
            product_requirement=row["product_requirement"], product_requirement_source=row["product_requirement_source"],
            base_prior_score=_f(row["base_prior_score"]), value_level=row["value_level"], value_multiplier=1.0,
            value_skew_profile=row["value_skew_profile"], normalization_alpha=_f(row["normalization_alpha"], 1.0),
            calibrated_task_value=float(Decimal(row["effective_task_value"])),
            value_assignment_reason=row["value_assignment_reason"],
        )
        if Decimal.from_float(float(task.task_value)) != Decimal.from_float(float(Decimal(row["effective_task_value"]))):
            mismatches += 1
        if Decimal(str(task.deadline_s)) != Decimal(str(float(Decimal(row["original_deadline_slack_s"])))):
            mismatches += 1
        tasks[row["task_id"]] = task
    timeline = []
    for row in bundle.slots:
        resource = resource_type(
            slot_index=_i(row["resource__slot_index"]), solar_energy_j=_f(row["resource__solar_energy_j"]),
            base_power_w=_f(row["resource__base_power_w"]), temperature_reference_c=_f(row["resource__temperature_reference_c"]),
            is_sunlit=_b(row["resource__is_sunlit"]), is_eclipse=_b(row["resource__is_eclipse"]),
            orbit_index=_i(row["resource__orbit_index"]), day_index=_i(row["resource__day_index"]),
            orbit_phase=_f(row["resource__orbit_phase"]), solar_power_w=_f(row["resource__solar_power_w"]),
            wasted_solar_energy_j=_f(row["resource__wasted_solar_energy_j"]), battery_soc=_f(row["resource__battery_soc"]),
            dod=_f(row["resource__dod"]), dod_guard_violation=_i(row["resource__dod_guard_violation"]),
            compute_power_w=_f(row["resource__compute_power_w"]), comm_power_w=_f(row["resource__comm_power_w"]),
        )
        link = link_type(
            slot_index=_i(row["link__slot_index"]), in_contact=_b(row["link__in_contact"]),
            snr_db=_f(row["link__snr_db"]), throughput_mbps=_f(row["link__throughput_mbps"]),
            link_quality=_f(row["link__link_quality"]), remaining_window_s=_f(row["link__remaining_window_s"]),
        )
        timeline.append(step_type(slot_index=_i(row["slot_index"]), slot_seconds=_f(row["slot_seconds"]),
                                  task=tasks.get(row.get("task_id", "")), resource_trace=resource, link_trace=link))
    values = {task_id: float(value) for task_id, value in bundle.task_values.items()}
    return NativeScenario(copy.deepcopy(bundle.config), timeline, bundle.tasks, bundle.contract["task_hash"],
                          bundle.contract["environment_hash"], bundle.contract["benchmark_fingerprint"],
                          values, 1.0, mismatches)


def _canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def recompute_contract_identity(scenario: NativeScenario) -> dict[str, str]:
    task_rows = []
    for step in scenario.timeline:
        task = step.task
        if task is None:
            continue
        arrival = step.slot_index * step.slot_seconds
        task_rows.append({
            "task_id": task.task_id, "scene_id": task.scene_id,
            "arrival_slot": step.slot_index, "arrival_time_s": arrival,
            "deadline_s": task.deadline_s, "deadline_time_s": arrival + task.deadline_s,
            "task_class": task.task_class, "product_requirement": task.product_requirement,
            "base_prior_value": task.base_prior_score, "value_level": task.value_level,
            "level_multiplier_audit_only": task.metadata["level_multiplier_audit_only"],
            "effective_task_value": task.task_value, "value_multiplier_runtime": task.value_multiplier,
            "high_value_flag": task.high_value_flag, "urgent_flag": task.emergency_flag,
        })
    task_hash = _canonical_json_hash(sorted(task_rows, key=lambda row: row["task_id"]))
    traces = [{"resource": asdict(step.resource_trace), "link": asdict(step.link_trace)} for step in scenario.timeline]
    config = scenario.configs
    environment_hash = _canonical_json_hash({
        "task_hash": task_hash, "seed": 2026, "capacity": 0.35,
        "slot_seconds": 20.0, "horizon_s": 86400, "trace": traces,
        "battery": config["simulator"].get("battery", config["timeline"]["resource"].get("reference_platform", {})),
        "thermal": config["simulator"].get("thermal", {}),
    })
    return {"task_hash": task_hash, "environment_hash": environment_hash}
