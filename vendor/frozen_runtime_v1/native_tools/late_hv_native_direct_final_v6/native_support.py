"""Standard-library identity support shared by method-specific native bindings."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    from .contracts import build_native_manifest
    from .native_scenario import recompute_contract_identity, reconstruct_native_scenario, stage_for_product
    from .runtime import validate_runtime_scenario
    from .sentinel_adapter import ControlEventAudit, classify_native_event
except ImportError:
    from contracts import build_native_manifest
    from native_scenario import recompute_contract_identity, reconstruct_native_scenario, stage_for_product
    from runtime import validate_runtime_scenario
    from sentinel_adapter import ControlEventAudit, classify_native_event


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loaded_src_modules() -> dict[str, str]:
    return {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "src" or name.startswith("src.")) and getattr(module, "__file__", None)
    }


class IdentityBinding:
    method: str

    def __init__(self, method: str, source_root: Path) -> None:
        self.method = method
        self.source_root = Path(source_root).resolve()
        self.spec = build_native_manifest()["methods"][method]
        expected = Path(self.spec["source_root"]).resolve()
        if self.source_root != expected:
            raise RuntimeError(f"source root mismatch for {method}: {self.source_root} != {expected}")
        self.instance: Any = None
        self.imported_file = ""

    def loaded_src_modules(self) -> dict[str, str]:
        return loaded_src_modules()

    def _instantiate(self) -> Any:
        raise NotImplementedError

    def identity(self) -> dict[str, Any]:
        self.instance = self._instantiate()
        module = sys.modules[type(self.instance).__module__]
        self.imported_file = str(Path(module.__file__).resolve())
        hashes = {}
        for relative, expected in self.spec["hashes"].items():
            actual = file_hash(self.source_root / relative)
            hashes[relative] = {"expected": expected, "actual": actual, "match": actual == expected}
        modules = self.loaded_src_modules()
        all_local = bool(modules) and all(Path(path).is_relative_to(self.source_root) for path in modules.values())
        imported_local = Path(self.imported_file).is_relative_to(self.source_root)
        router_class = f"{type(self.instance).__module__}.{type(self.instance).__name__}"
        router_class_match = router_class == self.spec["router_class"]
        passed = all_local and imported_local and router_class_match and all(row["match"] for row in hashes.values())
        return {
            "method": self.method, "source_root": str(self.source_root),
            "entry_point": self.spec["entry_point"], "requested_identifier": self.spec["identifier"],
            "router_class": router_class, "router_class_match": router_class_match,
            "imported_router_module_file": self.imported_file,
            "source_config_hashes": hashes, "src_module_files": modules,
            "all_src_modules_within_source_root": all_local,
            "source_identity_passed": passed, "executable": sys.executable,
            "cwd": str(Path.cwd().resolve()), "pythonpath": os.environ.get("PYTHONPATH"),
            "sys_path": list(sys.path),
        }

    def build_scenario(self, bundle: Any) -> Any:
        from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
        return reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)

    def validate_scenario(self, scenario: Any, bundle: Any) -> dict[str, Any]:
        from src.evaluation.canonical_sarfish_environment import (
            canonical_environment_hash_v2, canonical_task_hash_v2, canonical_task_rows_v2,
        )
        common = validate_runtime_scenario(scenario, bundle)
        contract_identity = recompute_contract_identity(scenario)
        task_hash = contract_identity["task_hash"]
        environment_hash = contract_identity["environment_hash"]
        native_v2_task_hash = canonical_task_hash_v2(canonical_task_rows_v2(scenario.timeline))
        native_v2_environment_hash = canonical_environment_hash_v2(scenario.configs, scenario.timeline)
        fingerprint_path = self.source_root / "manifests/benchmark_environment_fingerprint_v1.json"
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))["benchmark_environment_fingerprint"]
        checks = {
            "recomputed_task_hash": task_hash,
            "recomputed_environment_hash": environment_hash,
            "recomputed_benchmark_fingerprint": fingerprint,
            "native_v2_task_hash_auxiliary": native_v2_task_hash,
            "native_v2_environment_hash_auxiliary": native_v2_environment_hash,
            "task_hash_match": task_hash == bundle.contract["task_hash"],
            "environment_hash_match": environment_hash == bundle.contract["environment_hash"],
            "benchmark_fingerprint_match": fingerprint == bundle.contract["benchmark_fingerprint"],
            "field_mismatch_count": scenario.field_mismatch_count,
            "value_level_counts": bundle.contract["value_level_counts"],
            "late_window": bundle.contract["late_window"],
            "swap_count": len(bundle.swaps),
        }
        if not all((checks["task_hash_match"], checks["environment_hash_match"],
                    checks["benchmark_fingerprint_match"], checks["field_mismatch_count"] == 0,
                    checks["swap_count"] == 27)):
            raise ValueError(f"native scenario reconstruction mismatch: {checks}")
        return {**common, **checks}

    def normalize_outputs(self, native: Any, bundle: Any) -> dict[str, Any]:
        return dict(native["normalized"])


def _eligible(product: str) -> tuple[str, ...]:
    product = str(product or "").lower()
    return {
        "metadata": ("metadata",), "roi": ("metadata", "roi"),
        "roi_context": ("metadata", "roi", "evidence"),
        "compressed_full_frame": ("metadata", "roi", "evidence", "full"),
        "full": ("metadata", "roi", "evidence", "full"),
    }.get(product, ())


def required_stage(requirement: str) -> str:
    return {"alert": "metadata", "verification": "roi", "context": "evidence", "evidence": "evidence"}.get(
        str(requirement or "verification").lower(), "roi")


def normalize_episode(method: str, scenario: Any, result: Any, runner: Any, slots: int) -> dict[str, Any]:
    from src.evaluation.task_outcome_adapter import proposed_delivery_events
    records = list(result.records)
    lifecycle = list(runner.last_payload_lifecycle_current_records)
    native_events = proposed_delivery_events(records, lifecycle, slot_seconds=float(scenario.configs["timeline"]["slot_seconds"]))
    events: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_task_by_id = {task["task_id"]: task for task in scenario.canonical_tasks}
    control_audit = ControlEventAudit()
    native_event_index = 0
    for task_id, rows in native_events.items():
        for index, row in enumerate(sorted(rows, key=lambda x: (float(x["ground_available_time_s"]), str(x["product_type"])))):
            product = str(row["product_type"])
            stage = stage_for_product(product)
            if stage is None:
                continue
            task = classify_native_event(
                task_id, row, native_event_index, canonical_task_by_id, control_audit)
            native_event_index += 1
            if task is None:
                continue
            event = {"method": method, "task_id": task_id, "stage": stage,
                     "ground_available_time": float(row["ground_available_time_s"]),
                     "deadline": float(task["deadline_time_s"]),
                     "valid_before_deadline": int(float(row["ground_available_time_s"]) <= float(task["deadline_time_s"])),
                     "event_source": f"{row.get('delivery_route','native')}:{product}:{index}", "duplicate_flag": 0}
            events.append(event)
            by_task[task_id].append({**event, "product": product})
    record_by_task = {str(r.task_id): r for r in records if getattr(r, "has_new_task", 0)}
    tasks = []
    for row in scenario.canonical_tasks:
        task_id = row["task_id"]; record = record_by_task.get(task_id)
        arrivals = sorted(by_task.get(task_id, []), key=lambda x: float(x["ground_available_time"]))
        required = required_stage(row["product_requirement"])
        closure = next((e for e in arrivals if required in _eligible(e["product"])), None)
        product = "+".join(dict.fromkeys(e["product"] for e in arrivals))
        tasks.append({
            "method": method, "task_id": task_id, "arrival_time": row["arrival_time_s"],
            "deadline": row["deadline_time_s"], "value_level": row["value_level"],
            "effective_task_value": row["effective_task_value"], "product_requirement": row["product_requirement"],
            "selected_action": getattr(record, "raw_action", "") if record else "",
            "executed_action": getattr(record, "executed_action", "") if record else "",
            "selected_model": getattr(record, "selected_model", "") if record else "",
            "delivered_product": product, "closure_success": int(bool(closure) and int(closure["valid_before_deadline"])),
            "closure_ground_time": closure["ground_available_time"] if closure else "",
            "failure_reason": getattr(record, "service_failure_reason", "not_arrived_in_phase") if record else "not_arrived_in_phase",
            "controllable_energy_j": (float(getattr(record, "compute_energy_j", 0.0)) + float(getattr(record, "comm_energy_j", 0.0))) if record else 0.0,
            "transmitted_bytes": float(getattr(record, "S_down_current_task_mb", 0.0)) * 1_000_000.0 if record else 0.0,
        })
    slot_rows = []
    battery_min = float(scenario.configs["simulator"].get("battery_min", 0.0))
    thermal_max = float(scenario.configs["simulator"].get("thermal_max", 1.0))
    for record in records:
        soc = float(getattr(record, "battery_next_soc", getattr(record, "battery_soc", 0.0)))
        thermal = float(getattr(record, "thermal_next", getattr(record, "Theta_t", 0.0)))
        slot_rows.append({"method": method, "slot": int(record.slot),
            "time_s": int(record.slot) * float(scenario.configs["timeline"]["slot_seconds"]),
            "battery_soc": soc, "thermal": thermal, "queue_mb": float(getattr(record, "queue_next_mb", 0.0)),
            "compute_energy_j": float(getattr(record, "compute_energy_j", 0.0)),
            "comm_energy_j": float(getattr(record, "comm_energy_j", 0.0)),
            "idle_energy_j": float(getattr(record, "idle_energy_j", 0.0)),
            "transmitted_mb": float(getattr(record, "S_down_total_mb", 0.0)),
            "in_contact": int(float(getattr(record, "C_t", 0.0)) > 0),
            "soc_violation": int(soc < battery_min), "thermal_violation": int(thermal > thermal_max)})
    energy = sum(r["compute_energy_j"] + r["comm_energy_j"] for r in slot_rows)
    control_fields = control_audit.as_dict()
    summary = {"method": method, "slot_count": len(slot_rows),
        "minimum_soc": min((r["battery_soc"] for r in slot_rows), default=None),
        "final_soc": slot_rows[-1]["battery_soc"] if slot_rows else None,
        "maximum_thermal": max((r["thermal"] for r in slot_rows), default=None),
        "soc_violation_count": sum(r["soc_violation"] for r in slot_rows),
        "thermal_violation_count": sum(r["thermal_violation"] for r in slot_rows),
        "controllable_energy_wh": energy / 3600.0,
        "transmitted_mb": sum(r["transmitted_mb"] for r in slot_rows),
        "native_summary": dict(result.summary),
        "native_adapter_event_count": native_event_index,
        "scientific_ground_delivery_event_count": len(events),
        "adapter_event_conservation_ok": native_event_index == len(events) + control_fields["control_event_count"],
        **control_fields}
    return {"per_task": tasks, "events": events, "slots": slot_rows, "resource_summary": summary}
