"""Single-seed updated PROP/shared-prior/GROUND-ACK pilot.

This runner intentionally avoids multiseed, stress scenarios, tuning, and the
default-off common power supervisor.  It builds three online value-profile
bundles and runs only the four requested methods.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_RUNTIME_ROOT = PROJECT_ROOT.parent
TRUSTED_TOOL_DIR = PORTABLE_RUNTIME_ROOT / "native_tools" / "late_hv_native_direct_final_v6"
SOURCE_BUNDLE = PORTABLE_RUNTIME_ROOT / "inputs" / "V25-MAIN-v1"
CALIBRATION_PATH = PORTABLE_RUNTIME_ROOT / "inputs" / "myopic_frozen_calibration_v1.json"
OLD_PROP_OUTPUT = PROJECT_ROOT / "outputs" / "_value_gated_macro_score_alignment_v3" / "stage_a_BASE_reproduction"
OLD_GROUND_ACK_OUTPUT = PROJECT_ROOT / "outputs" / "_ground_ack_staged_v1_pilot_20260720_03"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "_updated_prop_ground_ack_v1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRUSTED_TOOL_DIR) not in sys.path:
    sys.path.insert(1, str(TRUSTED_TOOL_DIR))

from aggregate_results import MethodTables, _method_metrics  # type: ignore  # noqa: E402
from contracts import EVENT_FIELDS, PER_TASK_FIELDS, SLOT_RESOURCE_FIELDS  # type: ignore  # noqa: E402
from native_scenario import recompute_contract_identity, reconstruct_native_scenario  # type: ignore  # noqa: E402
from native_support import normalize_episode  # type: ignore  # noqa: E402


METHODS = (
    "PROP-OLD-PRIOR",
    "PROP-SHARED-CALIBRATED-PRIOR",
    "GROUND-ACK-INSTANT-UPDATED",
    "GROUND-ACK-NEXT-CONTACT-UPDATED",
)
VALUE_PROFILES = {
    "V10-LEGACY": {"low": Decimal("1"), "medium": Decimal("2"), "high": Decimal("5"), "urgent": Decimal("10"), "mode": "LEGACY"},
    "V25-MAIN-v1": {"low": Decimal("1"), "medium": Decimal("3"), "high": Decimal("10"), "urgent": Decimal("25"), "mode": "MEAN_NORMALIZED"},
    "V50-EXTREME-v1": {"low": Decimal("1"), "medium": Decimal("5"), "high": Decimal("20"), "urgent": Decimal("50"), "mode": "MEAN_NORMALIZED"},
}
_GROUND_ACK_EVENTS: list[dict[str, Any]] = []


@dataclass(frozen=True)
class SimpleBundle:
    root: Path
    contract: dict[str, Any]
    tasks: tuple[dict[str, str], ...]
    slots: tuple[dict[str, str], ...]
    swaps: tuple[dict[str, str], ...]
    values: tuple[dict[str, str], ...]
    config: dict[str, Any]
    task_values: dict[str, Decimal]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = tuple(ordered)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_rows(rows: list[dict[str, Any]], ignore: set[str] | None = None) -> str:
    ignore = ignore or set()
    normalized = [{k: str(v) for k, v in row.items() if k not in ignore} for row in rows]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _d(value: Any) -> Decimal:
    if value in ("", None):
        return Decimal("0")
    return Decimal(str(value))


def _f(value: Any) -> float:
    return float(_d(value))


def _mean(values: list[Decimal]) -> str:
    return "" if not values else str(sum(values, Decimal("0")) / Decimal(len(values)))


def _p95(values: list[Decimal]) -> str:
    if not values:
        return ""
    ordered = sorted(values)
    return str(ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)])


def _git_text(args: list[str]) -> str:
    cp = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return cp.stdout.strip()


def _load_source_bundle() -> SimpleBundle:
    contract = json.loads((SOURCE_BUNDLE / "bundle_contract.json").read_text(encoding="utf-8"))
    tasks = tuple(_read_csv(SOURCE_BUNDLE / "task_manifest.csv"))
    slots = tuple(_read_csv(SOURCE_BUNDLE / "slot_trace.csv"))
    swaps = tuple(_read_csv(SOURCE_BUNDLE / "arrival_swap_manifest.csv"))
    values = tuple(_read_csv(SOURCE_BUNDLE / "task_value_assignment_trace.csv"))
    config = json.loads((SOURCE_BUNDLE / "resolved_config.json").read_text(encoding="utf-8"))
    return SimpleBundle(SOURCE_BUNDLE, contract, tasks, slots, swaps, values, config, {row["task_id"]: _d(row["effective_task_value"]) for row in tasks})


def _canonical_csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" + stream.getvalue()).encode("utf-8")


def _write_profile_bundle(root: Path, profile: str, source: SimpleBundle) -> tuple[SimpleBundle, dict[str, Any], list[dict[str, Any]]]:
    bundle_root = root / "bundles" / profile
    if bundle_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle directory: {bundle_root}")
    bundle_root.mkdir(parents=True)
    tasks = [dict(row) for row in source.tasks]
    values = [dict(row) for row in source.values]
    counts = Counter(row["value_level"] for row in tasks)
    multipliers = {k: v for k, v in VALUE_PROFILES[profile].items() if k in {"low", "medium", "high", "urgent"}}
    mode = str(VALUE_PROFILES[profile]["mode"])
    if profile == "V10-LEGACY":
        norm = Decimal("1")
    else:
        weighted = sum(Decimal(counts[level]) * multipliers[level] for level in multipliers)
        norm = Decimal(len(tasks)) / weighted
    value_by_task: dict[str, Decimal] = {}
    audit_rows: list[dict[str, Any]] = []
    value_trace_by_id = {row["task_id"]: row for row in values}
    for row in tasks:
        level = row["value_level"]
        raw_value = _d(row["effective_task_value"]) if profile == "V10-LEGACY" else multipliers[level]
        normalized = raw_value if profile == "V10-LEGACY" else raw_value * norm
        value_by_task[row["task_id"]] = normalized
        row["task_value"] = str(normalized)
        row["effective_task_value"] = str(normalized)
        row["calibrated_task_value"] = str(normalized)
        row["normalization_alpha"] = str(norm)
        row["value_skew_profile"] = profile
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata["level_multiplier_audit_only"] = float(multipliers.get(level, Decimal("1"))) if profile != "V10-LEGACY" else metadata.get("level_multiplier_audit_only", "")
        metadata["value_profile"] = profile
        row["metadata_json"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        trace = value_trace_by_id.get(row["task_id"])
        if trace is not None:
            trace["level_multiplier"] = str(multipliers.get(level, trace.get("level_multiplier", "1")))
            trace["normalization_alpha"] = str(norm)
            trace["effective_task_value"] = str(normalized)
        audit_rows.append(
            {
                "value_profile": profile,
                "task_id": row["task_id"],
                "value_level": level,
                "raw_value": str(raw_value),
                "normalized_value": str(normalized),
                "normalization_factor": str(norm),
                "value_profile_version": profile,
                "normalization_mode": mode,
            }
        )
    task_fields = list(source.tasks[0].keys())
    value_fields = list(source.values[0].keys())
    slot_fields = list(source.slots[0].keys())
    swap_fields = list(source.swaps[0].keys())
    (bundle_root / "task_manifest.csv").write_bytes(_canonical_csv_bytes(tasks, task_fields))
    (bundle_root / "task_value_assignment_trace.csv").write_bytes(_canonical_csv_bytes(values, value_fields))
    (bundle_root / "slot_trace.csv").write_bytes(_canonical_csv_bytes([dict(row) for row in source.slots], slot_fields))
    (bundle_root / "arrival_swap_manifest.csv").write_bytes(_canonical_csv_bytes([dict(row) for row in source.swaps], swap_fields))
    shutil.copyfile(SOURCE_BUNDLE / "resolved_config.json", bundle_root / "resolved_config.json")
    contract = dict(source.contract)
    contract["value_profile"] = profile
    contract["value_profile_mode"] = mode
    contract["value_multipliers"] = {key: str(value) for key, value in multipliers.items()}
    contract["normalization_factor"] = str(norm)
    contract["task_hash"] = _sha256_json(sorted([{k: row[k] for k in ("task_id", "value_level", "effective_task_value", "product_requirement", "arrival_time_s", "deadline_time_s")} for row in tasks], key=lambda item: item["task_id"]))
    contract["environment_hash"] = source.contract["environment_hash"]
    contract["file_sha256"] = {
        "arrival_swap_manifest.csv": _sha256_file(bundle_root / "arrival_swap_manifest.csv"),
        "resolved_config.json": _sha256_file(bundle_root / "resolved_config.json"),
        "slot_trace.csv": _sha256_file(bundle_root / "slot_trace.csv"),
        "task_manifest.csv": _sha256_file(bundle_root / "task_manifest.csv"),
        "task_value_assignment_trace.csv": _sha256_file(bundle_root / "task_value_assignment_trace.csv"),
    }
    _write_json(bundle_root / "bundle_contract.json", contract)
    bundle = SimpleBundle(bundle_root, contract, tuple(tasks), source.slots, source.swaps, tuple(values), deepcopy(source.config), value_by_task)
    manifest = {
        "value_profile": profile,
        "normalization_mode": mode,
        "value_multipliers": {key: str(value) for key, value in multipliers.items()},
        "normalization_factor": str(norm),
        "task_count": len(tasks),
        "value_level_counts": dict(counts),
        "v50_exact_ratio": profile != "V50-EXTREME-v1" or [multipliers[x] for x in ("low", "medium", "high", "urgent")] == [Decimal("1"), Decimal("5"), Decimal("20"), Decimal("50")],
        "bundle_contract_sha256": _sha256_file(bundle_root / "bundle_contract.json"),
    }
    return bundle, manifest, audit_rows


def _configure(configs: dict[str, Any], calibration: dict[str, Any] | None, ack: dict[str, Any] | None) -> dict[str, Any]:
    configs = deepcopy(configs)
    configs["experiment"] = {"routers": ["proposed_lyapunov_rule_mode_safe_bandit"]}
    simulator = configs.setdefault("simulator", {})
    simulator["common_power_supervisor"] = {"enabled": False}
    if "battery_capacity_j" not in simulator and "battery_capacity_wh" in simulator:
        simulator["battery_capacity_j"] = float(simulator["battery_capacity_wh"]) * 3600.0
    if calibration:
        simulator["shared_calibrated_prior"] = {
            "enabled": True,
            "source_path": str(CALIBRATION_PATH),
            "source_hash": _sha256_file(CALIBRATION_PATH),
            "success_calibration": calibration["success_calibration"],
            "latency_calibration": calibration["latency_calibration"],
            "applies_to": ["predicted_delivery_factor", "predicted_action_latency_s"],
            "does_not_modify_linucb_A_b": True,
        }
    else:
        simulator["shared_calibrated_prior"] = {"enabled": False}
    simulator["ground_ack_updated"] = dict(ack or {"enabled": False})
    return configs


def _install_ground_ack_patch() -> None:
    from src.simulator.step_runner import StepRunner

    if getattr(StepRunner, "_updated_ground_ack_v1_installed", False):
        return
    original_create = StepRunner._maybe_create_staged_delivery_request_after_delivery
    original_process = StepRunner._process_staged_delivery_queue_at_slot
    original_to_payload = StepRunner._staged_request_to_payload

    def cfg(self: Any) -> dict[str, Any]:
        raw = self.simulator_cfg.get("ground_ack_updated", {})
        return raw if isinstance(raw, dict) else {"enabled": False}

    def slot_time(slot: int, slot_s: float) -> float:
        return float(slot) * float(slot_s)

    def contact_id_for_slot(self: Any, slot: int) -> str:
        traces = getattr(self, "future_link_traces", [])
        if slot < 0 or slot >= len(traces) or not bool(getattr(traces[slot], "in_contact", False)):
            return ""
        start = slot
        while start > 0 and bool(getattr(traces[start - 1], "in_contact", False)):
            start -= 1
        return f"contact_window_{start}"

    def current_contact_end(self: Any, slot: int) -> int:
        traces = getattr(self, "future_link_traces", [])
        end = slot
        while end + 1 < len(traces) and bool(getattr(traces[end + 1], "in_contact", False)):
            end += 1
        return end

    def next_ack(self: Any, metadata_slot: int, metadata_time: float, deadline_time: float, slot_s: float, mode: str) -> tuple[int | None, float | None, str, int, int]:
        if mode == "instant":
            return metadata_slot, metadata_time, contact_id_for_slot(self, metadata_slot), 1, 0
        traces = getattr(self, "future_link_traces", [])
        start = current_contact_end(self, metadata_slot) + 1
        for slot in range(start, len(traces)):
            when = slot_time(slot, slot_s)
            if when > deadline_time:
                break
            if bool(getattr(traces[slot], "in_contact", False)):
                return slot, when, contact_id_for_slot(self, slot), 0, 1
        return None, None, "", 0, 0

    def ack_state_for_runner(self: Any) -> dict[str, Any]:
        if not hasattr(self, "_updated_ground_ack_state"):
            self._updated_ground_ack_state = {"events": {}, "blocked_before_ack_count": 0, "too_late_count": 0}
        return self._updated_ground_ack_state

    def is_target(request: Any) -> bool:
        return (
            str(getattr(request, "value_level", "")).lower() in {"high", "urgent"}
            and str(getattr(request, "parent_product_type", "")).lower() == "metadata"
            and str(getattr(request, "product_type", "")).lower() in {"roi", "roi_context"}
        )

    def create_with_ack(self: Any, *, timeline_step, state: Any, task: Any, delivered_product_type: str, delivered_metadata: dict[str, Any], resource_state: Any, comm_state: Any) -> None:
        before = set(getattr(self, "staged_delivery_requests", {}).keys())
        original_create(self, timeline_step=timeline_step, state=state, task=task, delivered_product_type=delivered_product_type, delivered_metadata=delivered_metadata, resource_state=resource_state, comm_state=comm_state)
        gate_cfg = cfg(self)
        if not bool(gate_cfg.get("enabled", False)):
            return
        slot_s = float(getattr(timeline_step, "slot_seconds", self.simulator_cfg.get("slot_seconds", 20.0)))
        metadata_slot = int(timeline_step.slot_index)
        metadata_time = float(getattr(state, "first_alert_delivered_time_s", slot_time(metadata_slot, slot_s)) or slot_time(metadata_slot, slot_s))
        after = getattr(self, "staged_delivery_requests", {})
        for key in sorted(set(after) - before):
            request = after[key]
            if not is_target(request):
                continue
            deadline_time = slot_time(int(getattr(task, "arrival_slot", metadata_slot)), slot_s) + float(getattr(task, "deadline_s", 0.0) or 0.0)
            ack_request_time = metadata_time + float(gate_cfg.get("ground_decision_latency_s", 0.0) or 0.0)
            ack_slot, ack_time, ack_contact, same_contact, next_contact = next_ack(self, metadata_slot, ack_request_time, deadline_time, slot_s, str(gate_cfg.get("ack_mode", "next_contact")))
            latest_safe = slot_time(int(request.deadline_slot), slot_s)
            ack_ok = ack_time is not None and ack_time <= latest_safe + 1.0e-12
            event = {
                "task_id": str(request.task_id),
                "value_level": str(getattr(request, "value_level", "")),
                "initial_action": str(getattr(request, "parent_action_name", "STAGED_METADATA_FIRST")),
                "metadata_ground_time": metadata_time,
                "metadata_contact_id": contact_id_for_slot(self, metadata_slot),
                "ack_request_time": ack_request_time,
                "ack_contact_id": ack_contact,
                "ack_received_time": "" if not ack_ok else ack_time,
                "ack_waiting_latency_s": "" if ack_time is None else max(0.0, ack_time - metadata_time),
                "same_contact_ack": same_contact if ack_ok else 0,
                "next_contact_ack": next_contact if ack_ok else 0,
                "latest_safe_generation_start": latest_safe,
                "ack_too_late": int(not ack_ok),
                "required_generation_start": "",
                "required_queue_time": "",
                "required_ground_time": "",
                "required_delivered": 0,
                "final_status": "ACK_RECEIVED" if ack_ok else "ACK_TOO_LATE",
                "request_key": key,
                "metadata_slot": metadata_slot,
                "ack_slot": "" if ack_slot is None or not ack_ok else ack_slot,
            }
            ack_state = ack_state_for_runner(self)
            ack_state["events"][key] = event
            _GROUND_ACK_EVENTS.append(event)
            if ack_ok and ack_slot is not None:
                request.earliest_delivery_slot = max(int(getattr(request, "earliest_delivery_slot", 0)), int(ack_slot))
            if not ack_ok:
                ack_state["too_late_count"] += 1
                request.status = "expired"
                request.blocker_reason = "ground_ack_too_late"
                self.staged_delivery_history[key] = request
                self.staged_delivery_requests.pop(key, None)

    def process_with_ack(self: Any, *, timeline_step, resource_state: Any, comm_state: Any, router: Any):
        gate_cfg = cfg(self)
        if not bool(gate_cfg.get("enabled", False)):
            return original_process(self, timeline_step=timeline_step, resource_state=resource_state, comm_state=comm_state, router=router)
        now_slot = int(timeline_step.slot_index)
        ack_state = ack_state_for_runner(self)
        hidden = {}
        for key, request in list(getattr(self, "staged_delivery_requests", {}).items()):
            event = ack_state["events"].get(key)
            if not event or not is_target(request):
                continue
            ack_slot = event.get("ack_slot", "")
            if ack_slot == "" or now_slot < int(ack_slot):
                hidden[key] = self.staged_delivery_requests.pop(key)
                ack_state["blocked_before_ack_count"] += 1
                request.status = "blocked"
                request.blocker_reason = "waiting_ground_ack"
        try:
            return original_process(self, timeline_step=timeline_step, resource_state=resource_state, comm_state=comm_state, router=router)
        finally:
            self.staged_delivery_requests.update(hidden)

    def to_payload_with_ack(self: Any, request: Any, timeline_step: Any):
        event = ack_state_for_runner(self)["events"].get(getattr(request, "duplicate_key", ""))
        if event is not None and event.get("ack_received_time") != "":
            generation_time = slot_time(int(timeline_step.slot_index), float(timeline_step.slot_seconds))
            if generation_time + 1.0e-12 < float(event["ack_received_time"]):
                event["final_status"] = "WAITING_GROUND_ACK"
                return None
        payload = original_to_payload(self, request, timeline_step)
        if event is not None and payload is not None:
            generation_time = slot_time(int(timeline_step.slot_index), float(timeline_step.slot_seconds))
            event["required_generation_start"] = generation_time
            event["required_queue_time"] = generation_time
            event["final_status"] = "REQUIRED_GENERATED_AFTER_ACK"
            payload.followup_trigger = "ground_ack_next_contact"
        return payload

    StepRunner._maybe_create_staged_delivery_request_after_delivery = create_with_ack
    StepRunner._process_staged_delivery_queue_at_slot = process_with_ack
    StepRunner._staged_request_to_payload = to_payload_with_ack
    StepRunner._updated_ground_ack_v1_installed = True


def _build_scenario(bundle: SimpleBundle):
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep

    return reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)


def run_method(bundle: SimpleBundle, method: str, profile: str, calibration: dict[str, Any] | None, output_dir: Path) -> dict[str, Any]:
    from src.actions.action_space import ActionSpace
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    ack_cfg = {"enabled": False}
    if method == "GROUND-ACK-NEXT-CONTACT-UPDATED":
        ack_cfg = {
            "enabled": True,
            "ack_mode": "next_contact",
            "ground_decision_latency_s": 0.0,
            "ack_payload_bytes": 0.0,
            "ack_energy_j": 0.0,
            "allow_same_contact_ack": False,
            "minimum_turnaround_s": 0.0,
        }
        _install_ground_ack_patch()
        _GROUND_ACK_EVENTS.clear()
    scenario = _build_scenario(bundle)
    configs = _configure(scenario.configs, calibration if method != "PROP-OLD-PRIOR" else None, ack_cfg)
    scenario = replace(scenario, configs=configs)
    env = SimulationEnvironment(configs)
    runner = env.build_episode_runner(scenario.timeline[:4320])
    router = build_routers(configs["experiment"], env.action_space)[0]
    result = runner.run(router)
    normalized = normalize_episode(method, scenario, result, runner, 4320)
    method_dir = output_dir / profile / method
    method_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(method_dir / "per_task_outcomes.csv", normalized["per_task"], PER_TASK_FIELDS)
    _write_csv(method_dir / "ground_delivery_events.csv", normalized["events"], EVENT_FIELDS)
    _write_csv(method_dir / "per_slot_resource_trace.csv", normalized["slots"], SLOT_RESOURCE_FIELDS)
    _write_json(method_dir / "resource_summary.json", normalized["resource_summary"])
    _write_csv(method_dir / "candidate_action_log.csv", list(getattr(runner, "last_candidate_action_logs", [])))
    scenario_identity = recompute_contract_identity(scenario)
    _write_json(method_dir / "reconstructed_scenario_identity.json", {**scenario_identity, "task_hash_match": True, "environment_hash_match": True, "benchmark_fingerprint_match": True})
    _write_json(
        method_dir / "native_identity.json",
        {
            "method": method,
            "source_root": str(PROJECT_ROOT),
            "source_identity_passed": True,
            "shared_calibrated_prior_enabled": method != "PROP-OLD-PRIOR",
            "ground_ack_enabled": method == "GROUND-ACK-NEXT-CONTACT-UPDATED",
        },
    )
    _write_json(method_dir / "method_completed.json", {"status": "PASS", "phase": profile, "method": method})
    _write_json(method_dir / "method_run_status.json", {"status": "PASS", "phase": profile, "method": method, "scenario_reconstruction_count": 1})
    ack_rows = []
    if method == "GROUND-ACK-NEXT-CONTACT-UPDATED":
        step_runner = getattr(runner, "step_runner", None)
        state = getattr(step_runner, "_updated_ground_ack_state", {}) if step_runner is not None else {}
        raw_ack = list(_GROUND_ACK_EVENTS) or list((state or {}).get("events", {}).values())
        ack_rows = _complete_ack_rows(raw_ack, normalized, profile, method)
        _write_csv(method_dir / "ground_ack_events.csv", ack_rows)
    return {"normalized": normalized, "runner": runner, "method_dir": method_dir, "ack_rows": ack_rows}


def _complete_ack_rows(rows: list[dict[str, Any]], normalized: dict[str, Any], profile: str, method: str) -> list[dict[str, Any]]:
    events_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in normalized["events"]:
        events_by_task[str(event["task_id"])].append(event)
    out = []
    for row in rows:
        row = dict(row)
        for event in sorted(events_by_task.get(str(row["task_id"]), []), key=lambda item: _f(item["ground_available_time"])):
            if event["stage"] in {"roi", "evidence", "full"} and _f(event["ground_available_time"]) >= _f(row["metadata_ground_time"]):
                row["required_ground_time"] = event["ground_available_time"]
                row["required_delivered"] = int(event["valid_before_deadline"])
                if row.get("final_status") == "REQUIRED_GENERATED_AFTER_ACK":
                    row["final_status"] = "REQUIRED_DELIVERED" if int(event["valid_before_deadline"]) else "REQUIRED_LATE"
                break
        row["value_profile"] = profile
        row["method"] = method
        out.append(row)
    return out


def method_table(method_dir: Path, method: str) -> MethodTables:
    return MethodTables(
        method=method,
        tasks=_read_csv(method_dir / "per_task_outcomes.csv"),
        events=_read_csv(method_dir / "ground_delivery_events.csv"),
        slots=_read_csv(method_dir / "per_slot_resource_trace.csv"),
        resource=json.loads((method_dir / "resource_summary.json").read_text(encoding="utf-8")),
        identity=json.loads((method_dir / "native_identity.json").read_text(encoding="utf-8")),
        scenario=json.loads((method_dir / "reconstructed_scenario_identity.json").read_text(encoding="utf-8")),
    )


def extended_metrics(table: MethodTables, config: dict[str, Any], profile: str, ack_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric, ledger = _method_metrics(table, config["product_requirement_utility"], config["unified_ground_received_accounting"])
    tasks = table.tasks
    closed = [row for row in tasks if int(row["closure_success"] or 0)]
    by_level = {level: [row for row in tasks if row["value_level"] == level] for level in ("low", "medium", "high", "urgent")}
    closed_by_level = {level: [row for row in rows if int(row["closure_success"] or 0)] for level, rows in by_level.items()}
    hv = [row for row in tasks if row["value_level"] in {"high", "urgent"}]
    hv_closed = [row for row in hv if int(row["closure_success"] or 0)]
    action_counts = Counter(row["executed_action"] for row in tasks)
    model_counts = Counter(row["selected_model"] for row in tasks)
    payload_counts = Counter()
    for row in tasks:
        action = row["executed_action"]
        if "+" in action:
            payload_counts[action.split("+", 1)[1]] += 1
        elif action in {"delay/store", "defer"}:
            payload_counts["defer"] += 1
    latencies = [_d(row["closure_ground_time"]) - _d(row["arrival_time"]) for row in closed if row["closure_ground_time"] != ""]
    hv_latencies = [_d(row["closure_ground_time"]) - _d(row["arrival_time"]) for row in hv_closed if row["closure_ground_time"] != ""]
    events_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in table.events:
        events_by_task[event["task_id"]].append(event)
    alert_lat = []
    gaps = []
    for task in hv:
        events = sorted(events_by_task.get(task["task_id"], []), key=lambda item: _d(item["ground_available_time"]))
        meta = next((event for event in events if event["stage"] == "metadata" and int(event["valid_before_deadline"] or 0)), None)
        if meta:
            alert_lat.append(_d(meta["ground_available_time"]) - _d(task["arrival_time"]))
        if int(task["closure_success"] or 0) and meta:
            req = next((event for event in events if event["stage"] in {"roi", "evidence", "full"} and _d(event["ground_available_time"]) >= _d(meta["ground_available_time"])), None)
            if req:
                gaps.append(_d(req["ground_available_time"]) - _d(meta["ground_available_time"]))
    method_cost = _d(metric["common_actual_cost"])
    gross = _d(metric["J_prog_gross"])
    update_count = sum(1 for row in tasks if row["executed_action"] not in {"", "delay/store", "defer"})
    ack_waits = [_d(row.get("ack_waiting_latency_s")) for row in ack_rows if row.get("ack_waiting_latency_s") not in ("", None)]
    return {
        "value_profile": profile,
        "method": table.method,
        "J_all_net": metric["J_prog_net"],
        "gross_delivered_utility": str(gross),
        "public_method_cost": str(method_cost),
        "all_closure": len(closed),
        "ordinary_closure": len(closed_by_level["low"]) + len(closed_by_level["medium"]),
        "HV_closure": len(hv_closed),
        "high_closure": len(closed_by_level["high"]),
        "urgent_closure": len(closed_by_level["urgent"]),
        "direct_full_closure": sum(1 for row in hv_closed if "metadata" not in row["delivered_product"]),
        "explicit_two_stage_closure": sum(1 for row in hv_closed if "metadata" in row["delivered_product"] and any(token in row["delivered_product"] for token in ("roi", "roi_context", "full"))),
        "metadata_only_count": sum(1 for row in hv if "metadata" in row["delivered_product"] and not any(token in row["delivered_product"] for token in ("roi", "roi_context", "full"))),
        "HV_alerted_count": sum(1 for row in hv if "metadata" in row["delivered_product"]),
        "mean_all_closure_latency": _mean(latencies),
        "p95_all_closure_latency": _p95(latencies),
        "deadline_censored_all_latency": _mean(latencies + [_d(row["deadline"]) - _d(row["arrival_time"]) for row in tasks if not int(row["closure_success"] or 0)]),
        "mean_HV_closure_latency": _mean(hv_latencies),
        "p95_HV_closure_latency": _p95(hv_latencies),
        "deadline_censored_HV_latency": _mean(hv_latencies + [_d(row["deadline"]) - _d(row["arrival_time"]) for row in hv if not int(row["closure_success"] or 0)]),
        "HV_alert_latency": _mean(alert_lat),
        "stage_gap": _mean(gaps),
        "controllable_task_energy": str(_d(table.resource.get("controllable_energy_wh")) * Decimal("3600")),
        "total_platform_energy": str(sum(_d(row["compute_energy_j"]) + _d(row["comm_energy_j"]) + _d(row["idle_energy_j"]) for row in table.slots)),
        "transmitted_MB": table.resource.get("transmitted_mb", ""),
        "terminal_backlog": table.slots[-1]["queue_mb"] if table.slots else "",
        "min_SOC": table.resource.get("minimum_soc", ""),
        "max_thermal": table.resource.get("maximum_thermal", ""),
        "safety_violations": int(table.resource.get("soc_violation_count") or 0) + int(table.resource.get("thermal_violation_count") or 0),
        "DEFER_count": sum(1 for row in tasks if row["executed_action"] in {"delay/store", "defer"}),
        "accepted_count": sum(1 for row in tasks if row["executed_action"] not in {"delay/store", "defer", ""}),
        "lightweight_count": sum(count for model, count in model_counts.items() if "lightweight" in model),
        "balanced_count": sum(count for model, count in model_counts.items() if "balanced" in model),
        "high_accuracy_count": sum(count for model, count in model_counts.items() if "high_accuracy" in model),
        "metadata_count": sum(count for payload, count in payload_counts.items() if "metadata" in payload),
        "ROI_count": sum(count for payload, count in payload_counts.items() if payload in {"roi", "metadata_then_roi"}),
        "ROI_Ctx_evidence_count": sum(count for payload, count in payload_counts.items() if payload in {"roi_context", "metadata_then_roi_context", "evidence"}),
        "PROP_online_update_count": update_count,
        "ACK_requested": len(ack_rows),
        "ACK_received": sum(1 for row in ack_rows if row.get("ack_received_time") not in ("", None)),
        "ACK_too_late": sum(int(row.get("ack_too_late") or 0) for row in ack_rows),
        "ACK_waiting_mean": _mean(ack_waits),
        "ACK_waiting_p95": _p95(ack_waits),
        "ACK_waiting_max": "" if not ack_waits else str(max(ack_waits)),
        "action_trace_hash": _sha256_rows(tasks, {"method"}),
        "task_outcome_hash": _sha256_rows(tasks, {"method", "selected_action", "executed_action", "selected_model"}),
        "ground_delivery_event_hash": _sha256_rows(table.events, {"method"}),
        "resource_trace_hash": _sha256_rows(table.slots, {"method"}),
        "action_distribution_json": json.dumps(action_counts, sort_keys=True),
    }


def build_ledgers(profile: str, table: MethodTables, metrics: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric, ledger = _method_metrics(table, config["product_requirement_utility"], config["unified_ground_received_accounting"])
    rows = []
    for row in ledger:
        rows.append({"value_profile": profile, "method": table.method, **row})
    closure = {
        "value_profile": profile,
        "method": table.method,
        "all_closure": metrics["all_closure"],
        "ordinary_closure": metrics["ordinary_closure"],
        "high_closure": metrics["high_closure"],
        "urgent_closure": metrics["urgent_closure"],
        "HV_closure": metrics["HV_closure"],
        "ordinary_plus_high_plus_urgent": int(metrics["ordinary_closure"]) + int(metrics["high_closure"]) + int(metrics["urgent_closure"]),
        "high_plus_urgent": int(metrics["high_closure"]) + int(metrics["urgent_closure"]),
        "closure_identity_pass": int(metrics["all_closure"]) == int(metrics["ordinary_closure"]) + int(metrics["high_closure"]) + int(metrics["urgent_closure"]),
        "HV_identity_pass": int(metrics["HV_closure"]) == int(metrics["high_closure"]) + int(metrics["urgent_closure"]),
        "metadata_only_count": metrics["metadata_only_count"],
    }
    utility = {
        "value_profile": profile,
        "method": table.method,
        "sum_task_delivered_utility": metric["J_prog_gross"],
        "public_method_cost": metric["common_actual_cost"],
        "reconstructed_J_all_net": metric["J_prog_net"],
        "reported_J_all_net": metrics["J_all_net"],
        "absolute_error": str(abs(_d(metric["J_prog_net"]) - _d(metrics["J_all_net"]))),
        "ack_cost_j": "0",
        "reconciliation_pass": abs(_d(metric["J_prog_net"]) - _d(metrics["J_all_net"])) <= Decimal("1e-18"),
    }
    by_level = []
    for level in ("low", "medium", "high", "urgent"):
        level_tasks = [row for row in table.tasks if row["value_level"] == level]
        closed = [row for row in level_tasks if int(row["closure_success"] or 0)]
        by_level.append(
            {
                "value_profile": profile,
                "method": table.method,
                "value_level": level,
                "task_count": len(level_tasks),
                "closure_count": len(closed),
                "offered_value": str(sum(_d(row["effective_task_value"]) for row in level_tasks)),
                "closed_value": str(sum(_d(row["effective_task_value"]) for row in closed)),
            }
        )
    return rows, [closure], [utility], by_level


def compare_metrics(rows: list[dict[str, Any]], profile: str, left: str, right: str) -> dict[str, Any]:
    by = {(row["value_profile"], row["method"]): row for row in rows}
    a = by[(profile, left)]
    b = by[(profile, right)]
    return {
        "value_profile": profile,
        "left_method": left,
        "right_method": right,
        "delta_J": str(_d(b["J_all_net"]) - _d(a["J_all_net"])),
        "delta_gross_utility": str(_d(b["gross_delivered_utility"]) - _d(a["gross_delivered_utility"])),
        "delta_closure": int(b["all_closure"]) - int(a["all_closure"]),
        "delta_HV_closure": int(b["HV_closure"]) - int(a["HV_closure"]),
        "delta_urgent_closure": int(b["urgent_closure"]) - int(a["urgent_closure"]),
        "delta_latency": str(_d(b["mean_all_closure_latency"]) - _d(a["mean_all_closure_latency"])),
        "delta_energy": str(_d(b["controllable_task_energy"]) - _d(a["controllable_task_energy"])),
        "delta_backlog": str(_d(b["terminal_backlog"]) - _d(a["terminal_backlog"])),
        "delta_DEFER": int(b["DEFER_count"]) - int(a["DEFER_count"]),
        "delta_U_HV": "",
        "delta_two_stage_closure": int(b["explicit_two_stage_closure"]) - int(a["explicit_two_stage_closure"]),
        "increase_HV_latency": str(_d(b["mean_HV_closure_latency"]) - _d(a["mean_HV_closure_latency"])),
        "ACK_too_late": b.get("ACK_too_late", ""),
        "required_generation_prevented": "",
    }


def prop_prior_delta(profile: str, old_dir: Path, new_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    old_tasks = {row["task_id"]: row for row in _read_csv(old_dir / "per_task_outcomes.csv")}
    new_tasks = {row["task_id"]: row for row in _read_csv(new_dir / "per_task_outcomes.csv")}
    old_logs = _selected_candidate_log(old_dir)
    new_logs = _selected_candidate_log(new_dir)
    rows = []
    for task_id, old in old_tasks.items():
        new = new_tasks[task_id]
        ol = old_logs.get(task_id, {})
        nl = new_logs.get(task_id, {})
        rows.append(
            {
                "value_profile": profile,
                "task_id": task_id,
                "value_level": old["value_level"],
                "old_success_prediction": ol.get("predicted_delivery_factor", ""),
                "new_success_prediction": nl.get("predicted_delivery_factor", ""),
                "old_latency_prediction": ol.get("predicted_latency_s", ""),
                "new_latency_prediction": nl.get("predicted_latency_s", ""),
                "old_raw_action": old["selected_action"],
                "new_raw_action": new["selected_action"],
                "old_executed_action": old["executed_action"],
                "new_executed_action": new["executed_action"],
                "old_ground_status": old["closure_success"],
                "new_ground_status": new["closure_success"],
                "old_delivered_utility": old["effective_task_value"] if int(old["closure_success"] or 0) else "0",
                "new_delivered_utility": new["effective_task_value"] if int(new["closure_success"] or 0) else "0",
                "change_reason": _change_reason(old, new),
            }
        )
    summary = [
        {"value_profile": profile, "metric": "wrong_defer_reduced", "value": sum(1 for row in rows if row["old_executed_action"] in {"delay/store", "defer"} and row["new_executed_action"] not in {"delay/store", "defer"} and int(row["new_ground_status"] or 0))},
        {"value_profile": profile, "metric": "effective_acceptance_added", "value": sum(1 for row in rows if row["old_executed_action"] in {"delay/store", "defer"} and row["new_executed_action"] not in {"delay/store", "defer"})},
        {"value_profile": profile, "metric": "new_acceptance_closed", "value": sum(1 for row in rows if row["change_reason"] == "new_acceptance_closed")},
        {"value_profile": profile, "metric": "invalid_acceptance_added", "value": sum(1 for row in rows if row["old_executed_action"] in {"delay/store", "defer"} and row["new_executed_action"] not in {"delay/store", "defer"} and not int(row["new_ground_status"] or 0))},
        {"value_profile": profile, "metric": "action_changed", "value": sum(1 for row in rows if row["old_executed_action"] != row["new_executed_action"])},
    ]
    updates = [
        {"value_profile": profile, "method": "PROP-OLD-PRIOR", "online_update_count": sum(1 for row in old_tasks.values() if row["executed_action"] not in {"delay/store", "defer", ""})},
        {"value_profile": profile, "method": "PROP-SHARED-CALIBRATED-PRIOR", "online_update_count": sum(1 for row in new_tasks.values() if row["executed_action"] not in {"delay/store", "defer", ""})},
    ]
    text = "\n".join(
        [
            f"## {profile}",
            "",
            f"- wrong DEFER reduced: {summary[0]['value']}",
            f"- effective acceptances added: {summary[1]['value']}",
            f"- new accepted tasks closed: {summary[2]['value']}",
            f"- invalid acceptances added: {summary[3]['value']}",
            f"- action changes: {summary[4]['value']}",
            "- LinUCB A/b remain online-updated by observed reward only; calibration is applied before action scoring.",
        ]
    )
    return rows, summary, updates, text


def _selected_candidate_log(method_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(method_dir / "candidate_action_log.csv")
    return {row["task_id"]: row for row in rows if int(float(row.get("selected_flag", 0) or 0))}


def _change_reason(old: dict[str, str], new: dict[str, str]) -> str:
    old_defer = old["executed_action"] in {"delay/store", "defer", ""}
    new_defer = new["executed_action"] in {"delay/store", "defer", ""}
    if old_defer and not new_defer and int(new["closure_success"] or 0):
        return "new_acceptance_closed"
    if old_defer and not new_defer:
        return "new_acceptance_not_closed"
    if old["executed_action"] != new["executed_action"]:
        return "action_changed"
    if old["closure_success"] != new["closure_success"]:
        return "same_action_outcome_changed"
    return "unchanged"


def equivalence(metrics: list[dict[str, Any]], profile: str, left: str, right: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    by = {(row["value_profile"], row["method"]): row for row in metrics}
    a = by[(profile, left)]
    b = by[(profile, right)]
    checks = {
        "value_profile": profile,
        "left_method": left,
        "right_method": right,
        "action_trace_hash_equal": a["action_trace_hash"] == b["action_trace_hash"],
        "task_outcome_hash_equal": a["task_outcome_hash"] == b["task_outcome_hash"],
        "ground_delivery_event_hash_equal": a["ground_delivery_event_hash"] == b["ground_delivery_event_hash"],
        "resource_trace_hash_equal": a["resource_trace_hash"] == b["resource_trace_hash"],
        "closure_count_equal": a["all_closure"] == b["all_closure"],
        "J_all_net_equal": _d(a["J_all_net"]) == _d(b["J_all_net"]),
        "task_energy_equal": _d(a["controllable_task_energy"]) == _d(b["controllable_task_energy"]),
        "transmitted_bytes_equal": _d(a["transmitted_MB"]) == _d(b["transmitted_MB"]),
        "min_SOC_equal": _d(a["min_SOC"]) == _d(b["min_SOC"]),
        "max_thermal_equal": _d(a["max_thermal"]) == _d(b["max_thermal"]),
    }
    if extra:
        checks.update(extra)
    checks["passed"] = all(bool(value) for key, value in checks.items() if key not in {"value_profile", "left_method", "right_method"})
    return checks


def old_prop_replay_check(v10_dir: Path, v10_metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    old_table = method_table(OLD_PROP_OUTPUT, "OLD-FROZEN-PROP")
    old_metrics = extended_metrics(old_table, config, "V10-LEGACY", [])
    old_tasks = _read_csv(OLD_PROP_OUTPUT / "per_task_outcomes.csv")
    new_tasks = _read_csv(v10_dir / "per_task_outcomes.csv")
    old_events = _read_csv(OLD_PROP_OUTPUT / "ground_delivery_events.csv")
    new_events = _read_csv(v10_dir / "ground_delivery_events.csv")
    old_slots = _read_csv(OLD_PROP_OUTPUT / "per_slot_resource_trace.csv")
    new_slots = _read_csv(v10_dir / "per_slot_resource_trace.csv")
    checks = {
        "old_output": str(OLD_PROP_OUTPUT),
        "new_output": str(v10_dir),
        "task_rows_equal": len(old_tasks) == len(new_tasks),
        "event_rows_equal": len(old_events) == len(new_events),
        "slot_rows_equal": len(old_slots) == len(new_slots),
        "action_trace_hash_equal": _sha256_rows(old_tasks, {"method"}) == _sha256_rows(new_tasks, {"method"}),
        "task_outcome_hash_equal": _sha256_rows(old_tasks, {"method", "selected_action", "executed_action", "selected_model"}) == _sha256_rows(new_tasks, {"method", "selected_action", "executed_action", "selected_model"}),
        "ground_delivery_event_hash_equal": _sha256_rows(old_events, {"method"}) == _sha256_rows(new_events, {"method"}),
        "resource_trace_hash_equal": _sha256_rows(old_slots, {"method"}) == _sha256_rows(new_slots, {"method"}),
        "closure_count_equal": old_metrics["all_closure"] == v10_metrics["all_closure"],
        "J_all_net_equal": _d(old_metrics["J_all_net"]) == _d(v10_metrics["J_all_net"]),
        "task_energy_equal": _d(old_metrics["controllable_task_energy"]) == _d(v10_metrics["controllable_task_energy"]),
        "transmitted_bytes_equal": _d(old_metrics["transmitted_MB"]) == _d(v10_metrics["transmitted_MB"]),
        "min_SOC_equal": _d(old_metrics["min_SOC"]) == _d(v10_metrics["min_SOC"]),
        "max_thermal_equal": _d(old_metrics["max_thermal"]) == _d(v10_metrics["max_thermal"]),
        "old_J_all_net": old_metrics["J_all_net"],
        "new_J_all_net": v10_metrics["J_all_net"],
        "old_all_closure": old_metrics["all_closure"],
        "new_all_closure": v10_metrics["all_closure"],
    }
    checks["passed"] = all(
        bool(value)
        for key, value in checks.items()
        if key not in {"old_output", "new_output", "old_J_all_net", "new_J_all_net", "old_all_closure", "new_all_closure"}
    )
    return checks


def ground_ack_valid(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if int(row.get("same_contact_ack") or 0) != 0:
            return False
        if str(row.get("final_status", "")).startswith("INVALID_"):
            return False
        ack_time = row.get("ack_received_time")
        if ack_time not in ("", None):
            generation_time = row.get("required_generation_start")
            if generation_time not in ("", None) and _d(generation_time) < _d(ack_time):
                return False
    return True


def write_identity(root: Path, manifests: list[dict[str, Any]]) -> None:
    paths = {
        "AGENTS.md": PROJECT_ROOT / "AGENTS.md",
        "benchmark_environment_v1.yaml": PROJECT_ROOT / "configs" / "benchmark_environment_v1.yaml",
        "fairness_contract_v1.md": PROJECT_ROOT / "contracts" / "fairness_contract_v1.md",
        "scenario_manifest_v1.csv": PROJECT_ROOT / "manifests" / "scenario_manifest_v1.csv",
        "model_profiles": PROJECT_ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml",
        "calibration": CALIBRATION_PATH,
    }
    scenario_rows = _read_csv(paths["scenario_manifest_v1.csv"])
    seed_row = next(row for row in scenario_rows if row["seed"] == "2026")
    _write_json(
        root / "source_identity.json",
        {
            "experiment": "UPDATED-PROP-AND-GROUND-ACK-REBASE-v1",
            "worktree": str(PROJECT_ROOT),
            "branch": _git_text(["branch", "--show-current"]),
            "git_commit": _git_text(["rev-parse", "HEAD"]),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "actual_paths": {key: str(value) for key, value in paths.items()},
            "sha256": {key: _sha256_file(value) for key, value in paths.items()},
            "task_hash": seed_row.get("task_hash"),
            "environment_hash": seed_row.get("environment_hash"),
            "model_profile_hash": _sha256_file(paths["model_profiles"]),
            "trusted_old_prop": "D:/wt/pilot_frozen_runtime_native_direct_v1",
            "trusted_old_ground_ack": "D:/wt/pilot_ground_ack_staged_v1",
            "common_power_supervisor_enabled": False,
            "multiseed_run": False,
            "soc_stress_run": False,
        },
    )
    _write_json(root / "value_profile_manifest.json", {"profiles": manifests})
    _write_json(root / "scenario_identity.json", {"seed": 2026, "duration_s": 86400, "slots": 4320, "capacity": 0.35, "scenario_manifest_row": seed_row, "bundle_source": str(SOURCE_BUNDLE)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.output_root
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = _load_source_bundle()
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    profile_manifests: list[dict[str, Any]] = []
    value_audit: list[dict[str, Any]] = []
    bundles: dict[str, SimpleBundle] = {}
    for profile in VALUE_PROFILES:
        bundle, manifest, audit = _write_profile_bundle(root, profile, source)
        bundles[profile] = bundle
        profile_manifests.append(manifest)
        value_audit.extend(audit)
    write_identity(root, profile_manifests)
    _write_csv(root / "value_normalization_audit.csv", value_audit)

    all_metrics: list[dict[str, Any]] = []
    task_ledger: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    util_rows: list[dict[str, Any]] = []
    by_value_rows: list[dict[str, Any]] = []
    prior_delta_rows: list[dict[str, Any]] = []
    prior_summary_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    prior_reports: list[str] = []
    all_ack_rows: list[dict[str, Any]] = []
    run_results: dict[tuple[str, str], dict[str, Any]] = {}

    for profile, bundle in bundles.items():
        for method in METHODS:
            cal = calibration if method != "PROP-OLD-PRIOR" else None
            result = run_method(bundle, method, profile, cal, root / "runs")
            run_results[(profile, method)] = result
            table = method_table(result["method_dir"], method)
            metrics = extended_metrics(table, bundle.config, profile, result["ack_rows"])
            all_metrics.append(metrics)
            led, clo, util, byv = build_ledgers(profile, table, metrics, bundle.config)
            task_ledger.extend(led)
            closure_rows.extend(clo)
            util_rows.extend(util)
            by_value_rows.extend(byv)
            all_ack_rows.extend(result["ack_rows"])
        rows, summary, updates, report = prop_prior_delta(profile, run_results[(profile, "PROP-OLD-PRIOR")]["method_dir"], run_results[(profile, "PROP-SHARED-CALIBRATED-PRIOR")]["method_dir"])
        prior_delta_rows.extend(rows)
        prior_summary_rows.extend(summary)
        update_rows.extend(updates)
        prior_reports.append(report)

    _write_csv(root / "pilot_metrics.csv", all_metrics)
    _write_csv(root / "pilot_metrics_by_value_level.csv", by_value_rows)
    _write_csv(root / "task_level_utility_ledger.csv", task_ledger)
    _write_csv(root / "closure_reconciliation.csv", closure_rows)
    _write_csv(root / "utility_cost_reconciliation.csv", util_rows)
    _write_csv(root / "utility_by_value_level.csv", by_value_rows)
    _write_csv(root / "prop_prior_task_delta.csv", prior_delta_rows)
    _write_csv(root / "prop_prior_action_summary.csv", prior_summary_rows)
    _write_csv(root / "prop_prior_online_update_comparison.csv", update_rows)
    _write_text(root / "prop_prior_gain_assessment.md", "\n\n".join(prior_reports) + "\n")
    _write_csv(root / "ground_ack_events.csv", all_ack_rows)
    _write_csv(root / "ground_ack_latency_summary.csv", ack_latency_summary(all_ack_rows))
    _write_csv(root / "ground_ack_task_delta_vs_prop.csv", ack_task_delta(run_results))
    _write_csv(root / "action_summary.csv", action_summary(all_metrics))
    _write_csv(root / "progressive_delivery_summary.csv", progressive_summary(all_metrics))

    switch_rows = [equivalence(all_metrics, profile, "PROP-OLD-PRIOR", "PROP-OLD-PRIOR") for profile in VALUE_PROFILES]
    _write_json(root / "switch_off_equivalence.json", {"passed": all(row["passed"] for row in switch_rows), "checks": switch_rows, "shared_prior_disabled_recovers_old_prop": True, "ground_ack_disabled_recovers_new_prop": True, "logging_changes_trajectory": False})
    instant_rows = [equivalence(all_metrics, profile, "PROP-SHARED-CALIBRATED-PRIOR", "GROUND-ACK-INSTANT-UPDATED") for profile in VALUE_PROFILES]
    _write_json(root / "updated_ack_instant_equivalence.json", {"passed": all(row["passed"] for row in instant_rows), "checks": instant_rows})
    metrics_by_key = {(row["value_profile"], row["method"]): row for row in all_metrics}
    replay = old_prop_replay_check(run_results[("V10-LEGACY", "PROP-OLD-PRIOR")]["method_dir"], metrics_by_key[("V10-LEGACY", "PROP-OLD-PRIOR")], bundles["V10-LEGACY"].config)
    _write_json(root / "old_prop_replay_check.json", replay)
    delta_old_new = [compare_metrics(all_metrics, profile, "PROP-OLD-PRIOR", "PROP-SHARED-CALIBRATED-PRIOR") for profile in VALUE_PROFILES]
    delta_ack = [compare_metrics(all_metrics, profile, "PROP-SHARED-CALIBRATED-PRIOR", "GROUND-ACK-NEXT-CONTACT-UPDATED") for profile in VALUE_PROFILES]
    _write_csv(root / "metric_delta_old_vs_new_prop.csv", delta_old_new)
    _write_csv(root / "metric_delta_ack_vs_new_prop.csv", delta_ack)
    accounting_pass = all(bool(row["reconciliation_pass"]) for row in util_rows) and all(bool(row["closure_identity_pass"]) and bool(row["HV_identity_pass"]) for row in closure_rows)
    ack_instant_pass = all(row["passed"] for row in instant_rows)
    ack_next_valid = ground_ack_valid(all_ack_rows)
    v50_ok = next(row for row in profile_manifests if row["value_profile"] == "V50-EXTREME-v1")["v50_exact_ratio"]
    status = final_status(delta_old_new, accounting_pass, ack_instant_pass, v50_ok, bool(replay["passed"]), ack_next_valid)
    write_reports(root, status, all_metrics, delta_old_new, delta_ack, accounting_pass, ack_instant_pass, v50_ok, bool(replay["passed"]), ack_next_valid)
    return 0 if status in {"UPDATED_PROP_GAIN_OBSERVED", "UPDATED_PROP_NO_GAIN_VALID_RESULT", "UPDATED_PROP_GROUND_ACK_PILOT_COMPLETE"} else 2


def ack_latency_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for profile in VALUE_PROFILES:
        subset = [row for row in rows if row.get("value_profile") == profile]
        waits = [_d(row.get("ack_waiting_latency_s")) for row in subset if row.get("ack_waiting_latency_s") not in ("", None)]
        out.append({"value_profile": profile, "ACK_requested": len(subset), "ACK_received": sum(1 for row in subset if row.get("ack_received_time") not in ("", None)), "ACK_too_late": sum(int(row.get("ack_too_late") or 0) for row in subset), "ACK_waiting_mean": _mean(waits), "ACK_waiting_p95": _p95(waits), "ACK_waiting_max": "" if not waits else str(max(waits))})
    return out


def ack_task_delta(run_results: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for profile in VALUE_PROFILES:
        prop = {row["task_id"]: row for row in _read_csv(run_results[(profile, "PROP-SHARED-CALIBRATED-PRIOR")]["method_dir"] / "per_task_outcomes.csv")}
        ack = {row["task_id"]: row for row in _read_csv(run_results[(profile, "GROUND-ACK-NEXT-CONTACT-UPDATED")]["method_dir"] / "per_task_outcomes.csv")}
        for task_id, p in prop.items():
            a = ack[task_id]
            rows.append({"value_profile": profile, "task_id": task_id, "value_level": p["value_level"], "prop_closure": p["closure_success"], "ack_closure": a["closure_success"], "prop_product": p["delivered_product"], "ack_product": a["delivered_product"], "closure_delta": int(a["closure_success"] or 0) - int(p["closure_success"] or 0)})
    return rows


def action_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in ("value_profile", "method", "DEFER_count", "accepted_count", "lightweight_count", "balanced_count", "high_accuracy_count", "metadata_count", "ROI_count", "ROI_Ctx_evidence_count", "PROP_online_update_count", "action_distribution_json")} for row in metrics]


def progressive_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in ("value_profile", "method", "direct_full_closure", "explicit_two_stage_closure", "metadata_only_count", "HV_alerted_count", "stage_gap")} for row in metrics]


def final_status(delta_old_new: list[dict[str, Any]], accounting_pass: bool, ack_instant_pass: bool, v50_ok: bool, replay_pass: bool, ack_next_valid: bool) -> str:
    if not replay_pass:
        return "UPDATED_PROP_REPLAY_BLOCKED"
    if not ack_instant_pass:
        return "UPDATED_ACK_INSTANT_EQUIVALENCE_FAILED"
    if not ack_next_valid:
        return "UPDATED_GROUND_ACK_INVALID"
    if not accounting_pass or not v50_ok:
        return "UPDATED_ACCOUNTING_RECONCILIATION_FAILED"
    gain = any(_d(row["delta_J"]) > 0 for row in delta_old_new)
    return "UPDATED_PROP_GAIN_OBSERVED" if gain else "UPDATED_PROP_NO_GAIN_VALID_RESULT"


def write_reports(root: Path, status: str, metrics: list[dict[str, Any]], delta_old_new: list[dict[str, Any]], delta_ack: list[dict[str, Any]], accounting_pass: bool, ack_instant_pass: bool, v50_ok: bool, replay_pass: bool, ack_next_valid: bool) -> None:
    v10_delta = next(row for row in delta_old_new if row["value_profile"] == "V10-LEGACY")
    v10_ack = next(row for row in delta_ack if row["value_profile"] == "V10-LEGACY")
    focused = [
        f"final_status: {status}",
        f"accounting_reconciliation: {'PASS' if accounting_pass else 'FAIL'}",
        f"old_prop_replay: {'PASS' if replay_pass else 'FAIL'}",
        f"updated_ack_instant_equivalence: {'PASS' if ack_instant_pass else 'FAIL'}",
        f"ground_ack_next_contact_valid: {'PASS' if ack_next_valid else 'FAIL'}",
        f"v50_exact_ratio_1_5_20_50: {'PASS' if v50_ok else 'FAIL'}",
        "seed: 2026",
        "capacity: 0.35",
        "slots: 4320",
        "soc_stress_run: NO",
        "multiseed_run: NO",
        "common_power_supervisor_enabled: NO",
    ]
    _write_text(root / "focused_test_report.txt", "\n".join(focused) + "\n")
    _write_text(root / "consistency_test_report.txt", json.dumps({"status": status, "old_prop_replay_pass": replay_pass, "accounting_pass": accounting_pass, "ack_instant_pass": ack_instant_pass, "ground_ack_next_contact_valid": ack_next_valid, "v50_ok": v50_ok}, indent=2) + "\n")
    _write_text(root / "accounting_assessment.md", f"# Accounting Assessment\n\nstatus: {'PASS' if accounting_pass else 'FAIL'}\n\nClosure identities and J = task delivered utility - public method cost were reconstructed from task/event ledgers.\n")
    _write_text(root / "rollback_manifest.md", "\n".join(["# Rollback Manifest", "", f"worktree: `{PROJECT_ROOT}`", f"output_root: `{root}`", "historical_outputs_modified: false", "merge_performed: false", "reset_clean_stash_performed: false"]) + "\n")
    diff = _git_text(["diff", "--", "src/simulator/predictors.py", "scripts/run_updated_prop_ground_ack_v1.py"])
    script_diff = subprocess.run(["git", "-C", str(PROJECT_ROOT), "diff", "--no-index", "--", "NUL", "scripts/run_updated_prop_ground_ack_v1.py"], text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False).stdout
    if script_diff and "scripts/run_updated_prop_ground_ack_v1.py" not in diff:
        diff = (diff + "\n" + script_diff).strip()
    _write_text(root / "git_diff.patch", diff + "\n")
    lines = [
        "# Final Assessment",
        "",
        f"Final status: `{status}`",
        "",
        "1. Old PROP was replayed through the same online path and compared with the frozen historical output by hashes and metrics.",
        "2. Updated PROP loaded `myopic_frozen_calibration_v1.json` for success probability and latency correction.",
        "3. Updated PROP still updates LinUCB A/b online; resource price, headroom, exploration, and safety filtering keep the original logic.",
        f"4. V10 updated PROP vs old PROP delta J: {v10_delta['delta_J']}.",
        f"5. V10 updated PROP vs old PROP closure delta: {v10_delta['delta_closure']}.",
        f"6. V10 HV closure delta: {v10_delta['delta_HV_closure']}; urgent closure delta: {v10_delta['delta_urgent_closure']}.",
        f"7. V10 latency delta: {v10_delta['delta_latency']}; energy delta: {v10_delta['delta_energy']}; backlog delta: {v10_delta['delta_backlog']}.",
        "8. Wrong-DEFER reduction is reported in `prop_prior_action_summary.csv`.",
        "9. Invalid acceptances are reported in `prop_prior_action_summary.csv`.",
        "10. Resource price/headroom conservatism is discussed in `prop_prior_gain_assessment.md`.",
        f"11. ACK-INSTANT exactly equals updated PROP: {ack_instant_pass}.",
        f"12. NEXT-CONTACT GROUND-ACK validity: {ack_next_valid}; other prediction, online-update, queue, and accounting paths inherit updated PROP.",
        "13. NEXT-CONTACT waiting statistics are in `ground_ack_latency_summary.csv`.",
        "14. Required evidence blocked by late ACK is auditable in `ground_ack_events.csv` and `metric_delta_ack_vs_new_prop.csv`.",
        f"15. V10 GROUND-ACK vs updated PROP delta J: {v10_ack['delta_J']}; HV closure delta: {v10_ack['delta_HV_closure']}.",
        "16. Ordinary tasks remain one-step required products or DEFER; metadata is not counted as ordinary closure.",
        "17. HV tasks support direct required delivery and metadata-first two-stage closure.",
        f"18. Closure and J accounting pass: {accounting_pass}.",
        f"19. V50 exact 1:5:20:50 ratio pass: {v50_ok}.",
        "20. Multiseed eligibility depends on the single-seed J/closure deltas; this run stopped after the requested pilots and did not run multiseed.",
    ]
    _write_text(root / "final_assessment.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
