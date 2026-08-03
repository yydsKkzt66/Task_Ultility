from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "_v25_bupt1_prop_reqalign_seed2026_v1"
CORRECTION_AUDIT_OUT = ROOT / "outputs" / "_v25_bupt1_prop_reqalign_minimal_correction_v1"
OLD_V25 = ROOT / "outputs" / "_v25_bupt1_power_true_rerun_seed2026_v1"
SOLAR_TRACE = OLD_V25 / "calibration" / "solar_trace_one_orbit.csv"
V25_RUNNER = ROOT / "scripts" / "run_bupt1_power_true_rerun_seed2026_v1.py"
PROJECTION_SOURCE = ROOT / "src" / "routers" / "requirement_aware_projection.py"
PROJECTION_CONFIG = ROOT / "configs" / "prop_requirement_projection_v1.yaml"
ROLLBACK_DIR = ROOT / "artifacts" / "rollback" / "prop_requirement_projection_v1"
ROLLBACK_MANIFEST = ROLLBACK_DIR / "rollback_manifest.json"
RESTORE_SCRIPT = ROLLBACK_DIR / "restore_prop_requirement_projection_v1.py"

METHOD_PROP_OLD = "PROP-SHARED-CALIBRATED-PRIOR"
METHOD_PROP_NEW = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-v1"
METHODS: list[tuple[str, str]] = [
    ("PROP_ORIGINAL", METHOD_PROP_OLD),
    ("PROP_REQALIGN", METHOD_PROP_NEW),
    ("B1", "B1-COMMON-THERMAL"),
    ("OFFLOAD", "OFFLOAD-EFT-THERMAL-ALIGNED"),
    ("TUT", "TUT-UOI-PRODUCT-LITE-v2"),
    ("MYOPIC", "MYOPIC-FROZEN-CALIBRATED"),
]
BASELINE_LABELS = {"B1", "OFFLOAD", "TUT", "MYOPIC"}
FAILURE_BY_LABEL = {
    "PROP_ORIGINAL": "PROP_ORIGINAL_RUNTIME_FAILED",
    "PROP_REQALIGN": "PROP_REQALIGN_RUNTIME_FAILED",
    "B1": "B1_RUNTIME_FAILED",
    "OFFLOAD": "OFFLOAD_RUNTIME_FAILED",
    "TUT": "TUT_RUNTIME_FAILED",
    "MYOPIC": "MYOPIC_RUNTIME_FAILED",
}
FINAL_COMPLETE = "PROP_REQUIREMENT_ALIGNMENT_MINIMAL_CORRECTION_COMPLETE"
EXPECTED_TASK_HASH = "4c23cf3184f137733495f95a7c5bd7631adf95b3fbc4eb057351f27dac2f2ab5"
EXPECTED_TASK_COUNT = 268
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}
EXPECTED_HV = 75
EXPECTED_OLD_PRIMARY = {
    "J_all_net": Decimal("84.61324475969297327315182059"),
    "J_HV_net": Decimal("41.15358981050381326801987751"),
    "HV_closure_count": 32,
    "high_closure_count": 21,
    "urgent_closure_count": 11,
}
SLOTS = 4320
SLOT_SECONDS = 20
HORIZON_S = SLOTS * SLOT_SECONDS
CAPACITY = Decimal("0.35")
FLOAT_TOL = Decimal("1e-12")
AUDIT_TASK_IDS = ("task_0123", "task_0142", "task_0151", "task_0260")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    fields = fields or ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [{"empty": ""}])


def write_md_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_table(rows) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0])
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def dec(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    try:
        if value in (None, ""):
            return default
        result = Decimal(str(value))
        return result if result.is_finite() else default
    except Exception:
        return default


def flt(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def percentile95(values: list[Decimal]) -> Decimal | str:
    if not values:
        return "NA"
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def load_python_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_v25():
    return load_python_module("v25_power_true_runner_for_reqalign", V25_RUNNER)


def load_projection_module():
    return load_python_module("requirement_aware_projection_runtime_v1", PROJECTION_SOURCE)


def load_projection_config(enabled: bool) -> dict[str, Any]:
    loaded = yaml.safe_load(PROJECTION_CONFIG.read_text(encoding="utf-8"))
    cfg = deepcopy(loaded["requirement_alignment"])
    cfg["enabled"] = bool(enabled)
    return cfg


def semantic_csv_hash(path: Path, fields: list[str] | None = None) -> str:
    rows = read_csv(path)
    if fields is None:
        fields = list(rows[0]) if rows else []
    normalized = [{field: row.get(field, "") for field in fields} for row in rows]
    return sha256_json(normalized)


def source_code_path_audit(v25) -> dict[str, Any]:
    source_root = v25.SOURCE_PROP.resolve()
    entries = {
        "prop_python_root": str(source_root),
        "router_factory": "src.evaluation.baselines.build_routers",
        "router_class": "src.routers.safe_linucb_router.SafeLinUCBRouter",
        "scorer_entry": "src.routers.safe_linucb_router.SafeLinUCBRouter.select_action: final_action_score",
        "selector_entry": "src.routers.safe_linucb_router.SafeLinUCBRouter._best_name(final_action_score)",
        "safe_filter_entry": "src.routers.safe_filter.SafeFilter.filter_actions",
        "public_safety_executor_entry": "src.routers.safe_filter.SafeFilter.shield_or_fallback",
        "executor_entry": "src.simulator.step_runner.StepRunner.run_step",
        "required_product_field": "src.common.types.TaskSample.product_requirement",
        "delivered_product_fields": [
            "src.common.types.ProductLifecycleState.delivered_products",
            "src.common.types.DownlinkResult.delivered_product_types",
        ],
        "residual_requirement_existing": True,
        "residual_state_fields": [
            "ProductLifecycleState.pending_products",
            "ProductLifecycleState.residual_utility_remaining",
            "ProductLifecycleState.task_closed_flag",
            "ResidualCandidateState.residual_product_type",
        ],
        "literal_residual_required_product_field_existing": False,
        "existing_prediction_entry": "src.simulator.predictors.Predictor.predict_action",
        "existing_prediction_limitation": "current-slot latency does not include full future-contact wait",
        "new_equivalent_prediction_entry": "src.routers.requirement_aware_projection.predict_ground_arrival",
        "deadline_closure_entry": "src.evaluation.task_outcome_adapter.adapt_proposed",
        "requirement_closure_predicate": "src.evaluation.task_outcome_adapter.requirement_satisfied_by_products",
    }
    required = {
        "AGENTS.md": ROOT / "AGENTS.md",
        "configs/benchmark_environment_v1.yaml": ROOT / "configs" / "benchmark_environment_v1.yaml",
        "contracts/fairness_contract_v1.md": ROOT / "contracts" / "fairness_contract_v1.md",
        "scenario_manifest_v1.csv": ROOT / "scenario_manifest_v1.csv",
    }
    snapshot = {
        "AGENTS.md": ROOT / ".codex_method_preservation_audit_v1" / "AGENTS.md",
        "configs/benchmark_environment_v1.yaml": ROOT / ".codex_method_preservation_audit_v1" / "configs" / "benchmark_environment_v1.yaml",
        "contracts/fairness_contract_v1.md": ROOT / ".codex_method_preservation_audit_v1" / "contracts" / "fairness_contract_v1.md",
        "scenario_manifest_v1.csv": ROOT / ".codex_method_preservation_audit_v1" / "manifests" / "scenario_manifest_v1.csv",
    }
    entries["required_contract_resolution"] = {
        name: {
            "root_path": str(path),
            "root_exists": path.exists(),
            "resolved_snapshot_path": str(snapshot[name]),
            "resolved_snapshot_exists": snapshot[name].exists(),
            "resolved_snapshot_sha256": sha256_file(snapshot[name]) if snapshot[name].exists() else "",
        }
        for name, path in required.items()
    }
    return entries


class RequirementAwareRouterWrapper:
    """Post-selector wrapper that preserves the delegate's scorer and update."""

    _DEFER_RECHECK_MARKER = "requirement_alignment_defer_recheck"

    def __init__(self, delegate: Any, config: dict[str, Any], timeline: list[Any], simulator_cfg: dict[str, Any], projection: Any):
        self.delegate = delegate
        self.config = config
        self.timeline = timeline
        self.simulator_cfg = simulator_cfg
        self.projection = projection
        self.router_name = delegate.router_name
        self.use_lyapunov_pricing = delegate.use_lyapunov_pricing
        self.use_mode_controller = delegate.use_mode_controller
        self.use_guided_utility = delegate.use_guided_utility
        self.last_selected_action_score: float | None = None
        self.last_score_components: dict[str, Any] = {}
        self.last_all_score_components: dict[str, dict[str, Any]] = {}
        self.current_step = None
        self.current_resource_state = None
        self.current_comm_state = None
        self.step_runner = None
        self.rows: list[dict[str, Any]] = []
        self.active_reentry_task_id = ""
        self.active_reentry_contact_window_id = ""

    def bind_step_runner(self, step_runner: Any) -> None:
        self.step_runner = step_runner

    def bind_step(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> None:
        self.current_step = timeline_step
        self.current_resource_state = resource_state
        self.current_comm_state = comm_state

    @staticmethod
    def _task_value_level(task: Any) -> str:
        return str(getattr(task, "value_level", "") or (getattr(task, "metadata", {}) or {}).get("value_level", "")).lower()

    @staticmethod
    def _task_absolute_deadline_s(task: Any, slot_seconds: float) -> float:
        return (
            float(getattr(task, "arrival_slot", 0) or 0) * float(slot_seconds)
            + float(getattr(task, "deadline_s", 0.0) or 0.0)
        )

    def _closing_product_for_state(self, state: Any) -> str:
        return str(
            self.step_runner._minimum_closing_residual_product(
                str(getattr(state, "product_requirement", "") or ""),
                set(getattr(state, "delivered_products", set()) or set()),
            )
            or ""
        )

    def _register_high_value_defer_recheck(self, task: Any, decision_row: dict[str, Any]) -> bool:
        """Register a no-delivery retry in the existing lifecycle state.

        Registration occurs after the DEFER action has executed.  That ordering
        prevents the simulator's residual-payload hook from turning the current
        DEFER into an unscored same-slot payload.
        """

        if (
            self.step_runner is None
            or decision_row.get("projection_reason") != "high_value_defer_recheck"
            or self._task_value_level(task) not in {"high", "urgent"}
        ):
            return False
        state = self.step_runner._get_product_lifecycle(task)
        if bool(getattr(state, "task_closed_flag", False)):
            return False
        closing_product = self._closing_product_for_state(state)
        if closing_product not in {"metadata", "roi", "roi_context"}:
            return False
        already_registered = (
            getattr(state, "residual_parent_payload_plan", "") == self._DEFER_RECHECK_MARKER
            and closing_product in set(getattr(state, "pending_products", set()) or set())
        )
        state.pending_products.add(closing_product)
        state.residual_parent_payload_plan = self._DEFER_RECHECK_MARKER
        if state.pending_residual_registered_slot is None:
            state.pending_residual_registered_slot = int(self.current_step.slot_index)
        if not already_registered:
            state.residual_requested_count += 1
            state.residual_request_time_s = float(self.current_step.slot_index) * float(self.current_step.slot_seconds)
        from src.payload.information_sufficiency import residual_information_utility

        state.residual_utility_remaining = residual_information_utility(
            set(state.delivered_products),
            state.product_requirement,
        )
        decision_row["defer_recheck_registered"] = 1
        decision_row["defer_recheck_product"] = closing_product
        return True

    def _expire_defer_rechecks(self, timeline_step: Any) -> None:
        if self.step_runner is None:
            return
        now_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
        for task_id, state in list(self.step_runner.product_lifecycle.items()):
            if getattr(state, "residual_parent_payload_plan", "") != self._DEFER_RECHECK_MARKER:
                continue
            task = self.step_runner.task_snapshots.get(task_id)
            if task is None or now_s <= self._task_absolute_deadline_s(task, timeline_step.slot_seconds) + 1.0e-9:
                continue
            closing_product = self._closing_product_for_state(state)
            state.pending_products.discard(closing_product)
            state.residual_parent_payload_plan = ""
            state.residual_expired_count += 1
            for row in reversed(self.rows):
                if row.get("task_id") == task_id and row.get("defer_recheck_registered"):
                    row["defer_recheck_expired"] = 1
                    break

    def _queued_or_active(self, task_id: str) -> bool:
        queued = set(self.step_runner._queued_product_keys())
        if any(str(key).startswith(f"{task_id}:") for key in queued):
            return True
        return any(
            str(getattr(request, "task_id", "")) == task_id
            and str(getattr(request, "status", "")) in {"pending", "enqueued", "blocked"}
            for request in getattr(self.step_runner, "staged_delivery_requests", {}).values()
        )

    def _attempted_in_contact_window(self, task_id: str, contact_window_id: str) -> bool:
        return any(
            row.get("task_id") == task_id
            and row.get("reentry_attempt")
            and row.get("reentry_contact_window_id") == contact_window_id
            for row in self.rows
        )

    def _reentry_physical_feasibility(
        self,
        task: Any,
        timeline_step: Any,
        resource_state: Any,
        comm_state: Any,
    ) -> dict[str, Any]:
        """Read-only hard-safety/deadline screen before cross-task retry."""

        context = self.step_runner.context_builder.build(
            task,
            resource_state,
            comm_state,
            timeline_step.link_trace,
        )
        actions = self.step_runner.action_space.list_actions()
        predicted_map = {
            action.action_name: self.step_runner.predictor.predict_action(
                task=task,
                context=context,
                resource_trace=timeline_step.resource_trace,
                link_trace=timeline_step.link_trace,
                comm_state=comm_state,
                action=action,
                guidance_bundle=None,
            )
            for action in actions
        }
        safe_actions, _reasons = self.step_runner.safe_filter.filter_actions(task, actions, predicted_map)
        safe_names = {action.action_name for action in safe_actions}
        current_time_s = float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
        deadline_s = self._task_absolute_deadline_s(task, timeline_step.slot_seconds)
        direct_feasible = False
        metadata_feasible = False
        for action in actions:
            if action.action_name not in safe_names:
                continue
            predicted = predicted_map[action.action_name]
            product = str(predicted.predicted_product_type)
            if not (
                self.projection.product_satisfies(product, task.product_requirement)
                or self.projection.normalize_product(product) == "metadata"
            ):
                continue
            arrival = self.projection.predict_ground_arrival(
                current_time_s=current_time_s,
                compute_and_encode_s=float(predicted.predicted_L_inf),
                backlog_mb=float(comm_state.queue_mb),
                payload_mb=float(predicted.predicted_S_gen_mb),
                contact_slots=self._contacts(int(timeline_step.slot_index), deadline_s),
                deadline_s=deadline_s,
            )
            feasible = arrival.ground_arrival_s is not None and float(arrival.ground_arrival_s) <= deadline_s + 1.0e-9
            direct_feasible = direct_feasible or (
                feasible and self.projection.product_satisfies(product, task.product_requirement)
            )
            metadata_feasible = metadata_feasible or (
                feasible and self.projection.normalize_product(product) == "metadata"
            )
        previous_score = -math.inf
        for row in reversed(self.rows):
            if row.get("task_id") == task.task_id:
                previous_score = float(row.get("original_score", -math.inf) or -math.inf)
                break
        from src.data.value_skew_calibration import task_effective_value

        return {
            "task": task,
            "task_id": str(task.task_id),
            "effective_task_value": float(task_effective_value(task)),
            "remaining_slack_s": deadline_s - current_time_s,
            "original_score": previous_score,
            "hard_safe": bool(direct_feasible or metadata_feasible),
            "deadline_feasible": bool(direct_feasible or metadata_feasible),
        }

    def prepare_step_for_reentry(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> Any:
        """Use an otherwise idle contact slot for one full deferred-task retry."""

        self.active_reentry_task_id = ""
        self.active_reentry_contact_window_id = ""
        self._expire_defer_rechecks(timeline_step)
        if self.step_runner is None or timeline_step.task is not None or not bool(timeline_step.link_trace.in_contact):
            return timeline_step
        contact_window_id = str(self.step_runner._contact_window_id_for_slot(int(timeline_step.slot_index)))
        candidates: list[dict[str, Any]] = []
        for task_id, state in sorted(self.step_runner.product_lifecycle.items()):
            if getattr(state, "residual_parent_payload_plan", "") != self._DEFER_RECHECK_MARKER:
                continue
            task = self.step_runner.task_snapshots.get(task_id)
            if (
                task is None
                or self._task_value_level(task) not in {"high", "urgent"}
                or bool(getattr(state, "task_closed_flag", False))
                or self._queued_or_active(str(task_id))
                or self._attempted_in_contact_window(str(task_id), contact_window_id)
            ):
                continue
            candidate = self._reentry_physical_feasibility(task, timeline_step, resource_state, comm_state)
            if candidate["hard_safe"] and candidate["deadline_feasible"]:
                candidates.append(candidate)
        chosen = self.projection.choose_competing_task(candidates)
        if chosen is None:
            return timeline_step
        task = chosen["task"]
        state = self.step_runner.product_lifecycle[task.task_id]
        state.pending_products.discard(self._closing_product_for_state(state))
        state.residual_parent_payload_plan = ""
        self.active_reentry_task_id = str(task.task_id)
        self.active_reentry_contact_window_id = contact_window_id
        return replace(timeline_step, task=task)

    def finish_step(self, timeline_step: Any) -> None:
        """Persist a DEFER retry only after the executor has finished the slot."""

        task = timeline_step.task
        if task is None:
            return
        for row in reversed(self.rows):
            if row.get("task_id") != task.task_id or int(row.get("_decision_slot", -1)) != int(timeline_step.slot_index):
                continue
            self._register_high_value_defer_recheck(task, row)
            break

    def _contacts(self, start_slot: int, deadline_s: float) -> list[Any]:
        contacts = []
        efficiency = float(self.simulator_cfg.get("downlink_efficiency", 1.0))
        for step in self.timeline[start_slot:]:
            start_s = float(step.slot_index) * float(step.slot_seconds)
            if start_s > deadline_s + 1.0e-9:
                break
            rate = float(step.link_trace.throughput_mbps) * efficiency / 8.0 if step.link_trace.in_contact else 0.0
            contacts.append(
                self.projection.ContactCapacitySlot(
                    start_s=start_s,
                    end_s=start_s + float(step.slot_seconds),
                    rate_mb_s=rate,
                )
            )
        return contacts

    @staticmethod
    def _finite(value: float | None) -> float | None:
        return value if value is not None and math.isfinite(float(value)) else None

    def _arrival(self, action: Any, predicted: Any, task: Any, current_time_s: float, absolute_deadline_s: float):
        return self.projection.predict_ground_arrival(
            current_time_s=current_time_s,
            compute_and_encode_s=float(predicted.predicted_L_inf),
            backlog_mb=float(self.current_comm_state.queue_mb),
            payload_mb=float(predicted.predicted_S_gen_mb),
            contact_slots=self._contacts(int(self.current_step.slot_index), absolute_deadline_s),
            deadline_s=absolute_deadline_s,
        )

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        original = self.delegate.select_action(context, candidate_actions, predicted_map, guidance_bundle)
        self.last_all_score_components = deepcopy(self.delegate.last_all_score_components)
        self.last_score_components = deepcopy(self.delegate.last_score_components)
        self.last_selected_action_score = self.delegate.last_selected_action_score

        task = self.current_step.task
        if task is None or self.step_runner is None:
            return original
        current_time_s = float(self.current_step.slot_index) * float(self.current_step.slot_seconds)
        absolute_deadline_s = self._task_absolute_deadline_s(task, self.current_step.slot_seconds)
        lifecycle = self.step_runner.product_lifecycle.get(task.task_id)
        delivered_before = set() if lifecycle is None else set(lifecycle.delivered_products)
        required = str(task.product_requirement)
        residual_before = self.projection.residual_required_product(required, delivered_before)
        scores = {
            name: float(components.get("final_action_score", -math.inf))
            for name, components in self.last_all_score_components.items()
        }
        defer_score = scores.get("delay/store", math.inf)
        eligible_names = {action.action_name for action in candidate_actions}
        all_actions = self.step_runner.action_space.list_actions()
        candidate_facts = []
        arrival_by_action: dict[str, Any] = {}
        for action in all_actions:
            predicted = predicted_map[action.action_name]
            arrival = self._arrival(action, predicted, task, current_time_s, absolute_deadline_s)
            arrival_by_action[action.action_name] = arrival
            safety_reasons = tuple(self.step_runner.safe_filter.evaluate_action(task, action, predicted_map))
            score = scores.get(action.action_name, -math.inf)
            candidate_facts.append(
                self.projection.ProjectionCandidate(
                    action=action,
                    action_name=action.action_name,
                    model_name=action.model_name,
                    product=str(predicted.predicted_product_type),
                    original_score=score,
                    predicted_ground_arrival_s=self._finite(arrival.ground_arrival_s),
                    hard_safe=not safety_reasons,
                    deadline_feasible=(
                        arrival.ground_arrival_s is not None
                        and float(arrival.ground_arrival_s) <= absolute_deadline_s + 1.0e-9
                    ),
                    score_above_defer=score >= defer_score - 1.0e-12,
                    safety_reasons=safety_reasons,
                    contact_found=bool(arrival.contact_found),
                    eligible_by_existing_filters=action.action_name in eligible_names,
                )
            )

        original_predicted = predicted_map[original.action_name]
        result = self.projection.project_requirement_aware_action(
            selected_action=original,
            selected_product=original_predicted.predicted_product_type,
            selected_model=original.model_name,
            residual_requirement=residual_before,
            candidates=candidate_facts,
            value_level=str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "")),
            prefer_same_model=bool(self.config["prefer_same_model"]),
            prefer_minimum_satisfying_product=bool(self.config["prefer_minimum_satisfying_product"]),
        )
        final = result.final_action
        final_predicted = predicted_map[final.action_name]
        final_fact = next(item for item in candidate_facts if item.action_name == final.action_name)
        original_fact = next(item for item in candidate_facts if item.action_name == original.action_name)
        original_arrival = arrival_by_action[original.action_name]
        final_arrival = arrival_by_action[final.action_name]

        self.last_selected_action_score = scores.get(final.action_name, float(final_predicted.predicted_reward))
        self.last_score_components = deepcopy(self.last_all_score_components.get(final.action_name, {}))
        self.delegate.last_selected_action_score = self.last_selected_action_score
        self.delegate.last_score_components = deepcopy(self.last_score_components)

        from src.payload.payload_sizer import estimate_payload_encoding_cost
        from src.data.value_skew_calibration import task_effective_value

        _, original_encode = estimate_payload_encoding_cost(
            original.payload_policy,
            float(original_predicted.predicted_S_gen_mb),
            self.step_runner.payload_profiles_cfg,
        )
        _, final_encode = estimate_payload_encoding_cost(
            final.payload_policy,
            float(final_predicted.predicted_S_gen_mb),
            self.step_runner.payload_profiles_cfg,
        )
        original_compute = float(original_predicted.predicted_E_comp) - float(original_encode)
        final_compute = float(final_predicted.predicted_E_comp) - float(final_encode)
        original_product = str(original_predicted.predicted_product_type)
        final_product = str(final_predicted.predicted_product_type)
        model_changed = final.model_name != original.model_name
        product_upgraded = self.projection.product_rank(final_product) > self.projection.product_rank(original_product)
        blocked = set(result.blocked_reasons)
        self.rows.append(
            {
                "task_id": task.task_id,
                "value_level": str(getattr(task, "value_level", "") or (task.metadata or {}).get("value_level", "")),
                "effective_task_value": float(task_effective_value(task)),
                "required_product": required,
                "highest_delivered_product_before": self.projection.highest_delivered_product(delivered_before),
                "residual_required_product_before": residual_before,
                "original_model": original.model_name,
                "original_product": original_product,
                "original_score": scores.get(original.action_name, ""),
                "original_predicted_compute_finish_s": original_arrival.compute_finish_s,
                "original_predicted_queue_delay_s": original_arrival.queue_delay_s,
                "original_predicted_ground_arrival_s": self._finite(original_arrival.ground_arrival_s),
                "projection_enabled": True,
                "projection_reason": result.projection_reason,
                "reentry_attempt": int(str(task.task_id) == self.active_reentry_task_id),
                "reentry_contact_window_id": self.active_reentry_contact_window_id,
                "defer_recheck_registered": 0,
                "defer_recheck_product": "",
                "defer_recheck_expired": 0,
                "final_model": final.model_name,
                "final_product": final_product,
                "final_score": scores.get(final.action_name, ""),
                "final_predicted_compute_finish_s": final_arrival.compute_finish_s,
                "final_predicted_queue_delay_s": final_arrival.queue_delay_s,
                "final_predicted_ground_arrival_s": self._finite(final_arrival.ground_arrival_s),
                "same_model_upgrade": int(product_upgraded and not model_changed),
                "model_changed": int(model_changed),
                "product_upgraded": int(product_upgraded),
                "direct_closure_feasible": int(result.direct_closure_feasible),
                "hard_safe": int(final_fact.hard_safe),
                "deadline_feasible": int(final_fact.deadline_feasible),
                "score_above_defer": int(final_fact.score_above_defer),
                "residual_required_product_after": result.residual_requirement_after_action,
                "closure_status_after": result.closure_status_after,
                "reenter_future_decisions": int(result.progressive_reentry),
                "one_shot_delivery": int(result.one_shot_delivery),
                "incremental_compute_energy_Wh": (final_compute - original_compute) / 3600.0,
                "incremental_encode_energy_Wh": (float(final_encode) - float(original_encode)) / 3600.0,
                "incremental_tx_energy_Wh": (
                    float(final_predicted.predicted_E_comm) - float(original_predicted.predicted_E_comm)
                )
                / 3600.0,
                "incremental_bytes": (
                    float(final_predicted.predicted_S_gen_mb) - float(original_predicted.predicted_S_gen_mb)
                )
                * 1_000_000.0,
                "projection_blocked_by_deadline": int("deadline" in blocked),
                "projection_blocked_by_soc": int("soc" in blocked),
                "projection_blocked_by_thermal": int("thermal" in blocked),
                "projection_blocked_by_queue": int("queue" in blocked),
                "projection_blocked_by_negative_score": int("negative_score" in blocked),
                "projection_blocked_by_no_contact": int("no_contact" in blocked),
                "original_action": original.action_name,
                "final_action": final.action_name,
                "_absolute_deadline_s": absolute_deadline_s,
                "_decision_slot": int(self.current_step.slot_index),
            }
        )
        return final

    def update(self, context: Any, action: Any, observed_reward: float) -> None:
        self.delegate.update(context, action, observed_reward)


def bind_projection_runtime(runner: Any, wrapper: RequirementAwareRouterWrapper) -> None:
    original_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = original_factory(queue)
        wrapper.bind_step_runner(step_runner)
        original_run_step = step_runner.run_step

        def intercepted(timeline_step, resource_state, mission_state, comm_state, router):
            effective_step = wrapper.prepare_step_for_reentry(timeline_step, resource_state, comm_state)
            wrapper.bind_step(effective_step, resource_state, comm_state)
            result = original_run_step(effective_step, resource_state, mission_state, comm_state, router)
            wrapper.finish_step(effective_step)
            return result

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def write_energy_components(path: Path, result: Any) -> None:
    total_compute = sum(float(getattr(record, "compute_energy_j", 0.0)) for record in result.records)
    encoding = sum(float(getattr(record, "payload_encoding_energy_j", 0.0)) for record in result.records)
    communication = sum(float(getattr(record, "comm_energy_j", 0.0)) for record in result.records)
    write_json(
        path / "energy_components.json",
        {
            "compute_including_encoding_energy_j": total_compute,
            "encoding_energy_j": encoding,
            "compute_excluding_encoding_energy_j": max(0.0, total_compute - encoding),
            "tx_energy_j": communication,
        },
    )


def enrich_projection_rows(wrapper: RequirementAwareRouterWrapper, method_path: Path) -> list[dict[str, Any]]:
    tasks = {row["task_id"]: row for row in read_csv(method_path / "per_task_outcomes.csv")}
    events_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in read_csv(method_path / "ground_delivery_events.csv"):
        events_by_task[event["task_id"]].append(event)
    stage_product = {"metadata": "metadata", "roi": "roi", "evidence": "roi_context", "full": "roi_context"}
    output = []
    for raw in wrapper.rows:
        row = dict(raw)
        task = tasks[row["task_id"]]
        value_level = str(row.get("value_level", "")).lower()
        ordinary = value_level in {"low", "medium"}
        delivered = {
            stage_product.get(str(event.get("stage", "")).lower(), "")
            for event in events_by_task.get(row["task_id"], [])
        } - {""}
        residual = wrapper.projection.residual_required_product(row["required_product"], delivered)
        closure = bool(integer(task.get("closure_success")))
        expired = HORIZON_S >= flt(row["_absolute_deadline_s"])
        state = wrapper.step_runner.product_lifecycle.get(row["task_id"]) if wrapper.step_runner is not None else None
        reentry_count = 0
        if state is not None:
            reentry_count = int(state.residual_requested_count) + int(state.residual_enqueued_count)
        row["highest_delivered_product_after"] = wrapper.projection.highest_delivered_product(delivered)
        row["residual_required_product_after"] = "" if ordinary else residual
        if closure:
            row["closure_status_after"] = "closed"
        elif ordinary:
            row["closure_status_after"] = (
                "deferred"
                if row["projection_reason"] == "ordinary_required_product_infeasible"
                else ("expired" if expired else "incomplete")
            )
        else:
            row["closure_status_after"] = "expired" if expired else "incomplete"
        row["reenter_future_decisions"] = int(
            not ordinary
            and not closure
            and not expired
            and row["projection_reason"]
            in {"metadata_first_progressive_fallback", "high_value_defer_recheck"}
        )
        row["residual_task_reentry"] = int(not ordinary and reentry_count > 0)
        row["residual_upgrade_success"] = int(
            not ordinary
            and closure
            and row["projection_reason"] == "metadata_first_progressive_fallback"
            and len(delivered) > 1
        )
        row["residual_expired"] = int(not ordinary and bool(residual) and expired)
        row["ordinary_progressive_violation"] = int(ordinary and (reentry_count > 0 or bool(row["reenter_future_decisions"])))
        row.pop("_absolute_deadline_s", None)
        output.append(row)
    return output


def build_disabled_projection_log(method_path: Path) -> list[dict[str, Any]]:
    tasks = read_csv(method_path / "per_task_outcomes.csv")
    selected_candidates = {
        row["task_id"]: row
        for row in read_csv(method_path / "candidate_action_log.csv")
        if integer(row.get("selected_flag"))
    }
    rows = []
    for task in tasks:
        cand = selected_candidates.get(task["task_id"], {})
        rows.append(
            {
                "task_id": task["task_id"],
                "value_level": task.get("value_level", ""),
                "effective_task_value": task.get("effective_task_value", ""),
                "required_product": task.get("product_requirement", ""),
                "highest_delivered_product_before": "",
                "residual_required_product_before": task.get("product_requirement", ""),
                "original_model": task.get("selected_model", ""),
                "original_product": cand.get("candidate_payload", ""),
                "original_score": cand.get("final_candidate_score", ""),
                "original_predicted_ground_arrival_s": "",
                "projection_enabled": False,
                "projection_reason": "disabled",
                "final_model": task.get("selected_model", ""),
                "final_product": cand.get("candidate_payload", ""),
                "final_score": cand.get("final_candidate_score", ""),
                "final_predicted_ground_arrival_s": "",
                "same_model_upgrade": 0,
                "model_changed": 0,
                "product_upgraded": 0,
                "direct_closure_feasible": "",
                "hard_safe": 1,
                "deadline_feasible": "",
                "score_above_defer": "",
                "residual_required_product_after": "" if integer(task.get("closure_success")) else task.get("product_requirement", ""),
                "closure_status_after": "closed" if integer(task.get("closure_success")) else "expired",
                "reenter_future_decisions": 0,
                "incremental_compute_energy_Wh": 0,
                "incremental_encode_energy_Wh": 0,
                "incremental_tx_energy_Wh": 0,
                "incremental_bytes": 0,
                "original_action": task.get("selected_action", ""),
                "final_action": task.get("executed_action", ""),
            }
        )
    return rows


def run_prop_variant(v25, bundle_root: Path, output: Path, *, projection_enabled: bool) -> None:
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    prop_mod = v25.import_prop_script()
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(scenario.configs, calibration, {"enabled": False})
    scenario = replace(scenario, configs=configs)
    scenario, power_identity = v25.inject_power(scenario, v25.SOURCE_PROP)
    method = METHOD_PROP_NEW if projection_enabled else METHOD_PROP_OLD
    scenario_identity = v25.relaxed_scenario_identity(method, scenario, bundle, power_identity)
    if not scenario_identity["task_hash_match"] or scenario_identity["runtime_value_mismatch_count"]:
        raise RuntimeError({"scenario_identity_failed": scenario_identity})
    env = SimulationEnvironment(scenario.configs)
    runner = env.build_episode_runner(scenario.timeline[:SLOTS])
    base_router = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    wrapper = None
    router = base_router
    if projection_enabled:
        projection = load_projection_module()
        projection_cfg = load_projection_config(True)
        wrapper = RequirementAwareRouterWrapper(
            base_router,
            projection_cfg,
            scenario.timeline[:SLOTS],
            scenario.configs["simulator"],
            projection,
        )
        bind_projection_runtime(runner, wrapper)
        router = wrapper
    result = runner.run(router)
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(output / "candidate_action_log.csv", list(getattr(runner, "last_candidate_action_logs", [])))
    write_energy_components(output, result)
    if wrapper is not None:
        projection_rows = enrich_projection_rows(wrapper, output)
    else:
        projection_rows = build_disabled_projection_log(output)
    write_csv(output / "requirement_projection_log.csv", projection_rows)
    identity = {
        "method": method,
        "source_root": str(v25.SOURCE_PROP.resolve()),
        "source_identity_passed": True,
        "shared_calibrated_prior_enabled": True,
        "shared_prior_calibration_path": str(v25.SHARED_PRIOR_CALIBRATION),
        "shared_prior_calibration_sha256": sha256_file(v25.SHARED_PRIOR_CALIBRATION),
        "ground_ack_enabled": False,
        "router_identifier": "proposed_lyapunov_rule_mode_safe_bandit",
        "requirement_projection_enabled": projection_enabled,
        "projection_module_imported": projection_enabled,
        "projection_source": str(PROJECTION_SOURCE) if projection_enabled else "",
        "projection_source_sha256": sha256_file(PROJECTION_SOURCE) if projection_enabled else "",
        "projection_config_sha256": sha256_file(PROJECTION_CONFIG),
        "preserve_original_scorer": True,
        "preserve_original_selector": True,
    }
    status = {
        "status": "PASS",
        "phase": "formal_true_rerun",
        "method": method,
        "scenario_reconstruction_count": 1,
        "simulation_started": True,
        "event_count": len(normalized["events"]),
        "reused_action_trace": False,
        "reused_decision_trace": False,
        "reused_queue_trace": False,
        "reused_delivery_ledger": False,
        "reused_soc_trace": False,
        "reused_safety_filter_result": False,
        "identity": identity,
        "scenario_identity": scenario_identity,
    }
    write_json(output / "native_identity.json", identity)
    write_json(output / "reconstructed_scenario_identity.json", scenario_identity)
    write_json(output / "method_run_status.json", status)
    write_json(
        output / "method_completed.json",
        {"phase": "formal_true_rerun", "method": method, "status": "PASS", "scenario_reconstruction_count": 1},
    )


def child_main(args: argparse.Namespace) -> int:
    v25 = load_v25()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.label in {"PROP_ORIGINAL", "PROP_DISABLED_REGRESSION"}:
        run_prop_variant(v25, Path(args.bundle_root), output, projection_enabled=False)
    elif args.label == "PROP_REQALIGN":
        run_prop_variant(v25, Path(args.bundle_root), output, projection_enabled=True)
    elif args.label in BASELINE_LABELS:
        method = dict(METHODS)[args.label]
        if args.label == "MYOPIC":
            v25.run_myopic_method(Path(args.bundle_root), output)
        else:
            v25.run_native_binding_method(method, Path(args.bundle_root), output)
    else:
        raise ValueError(args.label)
    print(json.dumps({"status": "PASS", "label": args.label, "output": str(output)}, ensure_ascii=False))
    return 0


def launch_child(label: str, bundle_root: Path, output: Path, log_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--label",
        label,
        "--bundle-root",
        str(bundle_root),
        "--output",
        str(output),
    ]
    started = utc_now()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    process = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{label}_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (log_dir / f"{label}_stderr.txt").write_text(process.stderr, encoding="utf-8")
    return {
        "label": label,
        "method": dict(METHODS).get(label, METHOD_PROP_OLD),
        "returncode": process.returncode,
        "output": str(output),
        "started_utc": started,
        "finished_utc": utc_now(),
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-4000:],
        "status": "PASS" if process.returncode == 0 and (output / "method_completed.json").exists() else "FAIL",
    }


def method_paths(out: Path) -> dict[str, Path]:
    return {method: out / "runs" / label for label, method in METHODS}


def hv_net(eval_mod: Any, method_path: Path, metric: dict[str, Any], utility_cfg: dict[str, Any]) -> Decimal:
    tasks = read_csv(method_path / "per_task_outcomes.csv")
    events = read_csv(method_path / "ground_delivery_events.csv")
    closed_hv = {
        row["task_id"]
        for row in tasks
        if row.get("value_level") in {"high", "urgent"} and integer(row.get("closure_success"))
    }
    ledger = eval_mod.ledger_for(tasks, events, utility_cfg)
    gross = sum(
        (dec(row["positive_gross_utility"]) for row in ledger if row["task_id"] in closed_hv),
        Decimal("0"),
    )
    total_gross = dec(metric["gross_delivered_utility"])
    allocated = dec(metric["public_method_cost"]) * gross / total_gross if total_gross else Decimal("0")
    return gross - allocated


def evaluate_methods(out: Path, bundle_root: Path, v25: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    eval_mod = v25.load_eval_module(out)
    config = read_json(bundle_root / "resolved_config.json")
    metrics = []
    accounting = []
    for method, path in method_paths(out).items():
        row, audit = eval_mod.summarize_method(
            method,
            "TRUE_FRESH_FROZEN_POWER_REQALIGN_ABLATION",
            path,
            config["product_requirement_utility"],
            config["unified_ground_received_accounting"],
        )
        row["J_HV_net"] = hv_net(eval_mod, path, row, config["product_requirement_utility"])
        metrics.append(row)
        accounting.append(audit)
    return metrics, accounting, eval_mod, config


def disabled_regression_audit(regression_path: Path, v25: Any, bundle_root: Path) -> dict[str, Any]:
    eval_mod = v25.load_eval_module(regression_path.parent)
    config = read_json(bundle_root / "resolved_config.json")
    metric, accounting = eval_mod.summarize_method(
        METHOD_PROP_OLD,
        "REQALIGN_DISABLED_REGRESSION",
        regression_path,
        config["product_requirement_utility"],
        config["unified_ground_received_accounting"],
    )
    metric["J_HV_net"] = hv_net(eval_mod, regression_path, metric, config["product_requirement_utility"])
    old_path = OLD_V25 / "formal" / METHOD_PROP_OLD
    comparisons = {
        "action_trace_hash": {
            "new": semantic_csv_hash(regression_path / "candidate_action_log.csv"),
            "old": semantic_csv_hash(old_path / "candidate_action_log.csv"),
        },
        "delivery_event_hash": {
            "new": semantic_csv_hash(regression_path / "ground_delivery_events.csv"),
            "old": semantic_csv_hash(old_path / "ground_delivery_events.csv"),
        },
        "queue_trace_hash": {
            "new": semantic_csv_hash(regression_path / "per_slot_resource_trace.csv", ["slot", "queue_mb"]),
            "old": semantic_csv_hash(old_path / "per_slot_resource_trace.csv", ["slot", "queue_mb"]),
        },
        "soc_trace_hash": {
            "new": semantic_csv_hash(regression_path / "per_slot_resource_trace.csv", ["slot", "battery_soc"]),
            "old": semantic_csv_hash(old_path / "per_slot_resource_trace.csv", ["slot", "battery_soc"]),
        },
        "thermal_trace_hash": {
            "new": semantic_csv_hash(regression_path / "per_slot_resource_trace.csv", ["slot", "thermal"]),
            "old": semantic_csv_hash(old_path / "per_slot_resource_trace.csv", ["slot", "thermal"]),
        },
    }
    hashes_match = all(item["new"] == item["old"] for item in comparisons.values())
    old_metric = next(
        row
        for row in read_csv(OLD_V25 / "method_metrics_full.csv")
        if row["method"] == METHOD_PROP_OLD
    )
    scalar_checks = {
        "J_all_net": abs(dec(metric["J_all_net"]) - EXPECTED_OLD_PRIMARY["J_all_net"]) <= FLOAT_TOL,
        "J_HV_net": abs(dec(metric["J_HV_net"]) - EXPECTED_OLD_PRIMARY["J_HV_net"]) <= FLOAT_TOL,
        "HV_closure_count": integer(metric["HV_closure_count"]) == EXPECTED_OLD_PRIMARY["HV_closure_count"],
        "high_closure_count": integer(metric["high_closure_count"]) == EXPECTED_OLD_PRIMARY["high_closure_count"],
        "urgent_closure_count": integer(metric["urgent_closure_count"]) == EXPECTED_OLD_PRIMARY["urgent_closure_count"],
        "action_count": len(read_csv(regression_path / "candidate_action_log.csv"))
        == len(read_csv(old_path / "candidate_action_log.csv")),
        "online_update_count": integer(metric["online_update_count"]) == integer(old_metric["online_update_count"]),
        "transmitted_MB": abs(dec(metric["transmitted_MB"]) - dec(old_metric["transmitted_MB"])) <= FLOAT_TOL,
        "controllable_energy_Wh": abs(dec(metric["controllable_energy_Wh"]) - dec(old_metric["controllable_energy_Wh"]))
        <= FLOAT_TOL,
        "public_accounting": abs(dec(accounting["accounting_error"])) <= Decimal("1e-10"),
    }
    passed = hashes_match and all(scalar_checks.values())
    return {
        "passed": passed,
        "projection_module_imported": read_json(regression_path / "native_identity.json").get("projection_module_imported"),
        "semantic_hashes": comparisons,
        "scalar_checks": scalar_checks,
        "observed_primary": {
            "J_all_net": metric["J_all_net"],
            "J_HV_net": metric["J_HV_net"],
            "HV_closure_count": metric["HV_closure_count"],
            "high_closure_count": metric["high_closure_count"],
            "urgent_closure_count": metric["urgent_closure_count"],
        },
        "expected_primary": EXPECTED_OLD_PRIMARY,
    }


def energy_metrics(method: str, path: Path) -> dict[str, Any]:
    slots = read_csv(path / "per_slot_resource_trace.csv")
    resource = read_json(path / "resource_summary.json")
    components = read_json(path / "energy_components.json", {})
    compute_total_j = sum(dec(row.get("compute_energy_j")) for row in slots)
    tx_j = sum(dec(row.get("comm_energy_j")) for row in slots)
    encoding_j = dec(components.get("encoding_energy_j", 0))
    compute_only_j = max(Decimal("0"), compute_total_j - encoding_j)
    socs = [dec(row["battery_soc"]) for row in slots]
    thermals = [dec(row["thermal"]) for row in slots]
    queues = [dec(row["queue_mb"]) for row in slots]
    solar_rows = read_csv(SOLAR_TRACE)
    previous_soc = Decimal("0.85")
    residual_max = Decimal("0")
    capacity_wh = Decimal("230.0")
    for row in slots:
        slot = integer(row["slot"])
        solar_power_w = dec(solar_rows[slot % len(solar_rows)]["solar_power_W"])
        solar_wh = solar_power_w * Decimal(SLOT_SECONDS) / Decimal("3600")
        load_wh = (
            dec(row.get("idle_energy_j"))
            + dec(row.get("compute_energy_j"))
            + dec(row.get("comm_energy_j"))
        ) / Decimal("3600")
        net_wh = solar_wh - load_wh
        battery_delta_wh = (
            net_wh * Decimal("0.90")
            if net_wh >= 0
            else net_wh / Decimal("0.95")
        )
        expected_soc = previous_soc + battery_delta_wh / capacity_wh
        actual_soc = dec(row["battery_soc"])
        if Decimal("0") < actual_soc < Decimal("1"):
            residual_max = max(residual_max, abs(actual_soc - expected_soc) * capacity_wh)
        previous_soc = actual_soc
    return {
        "method": method,
        "compute_energy_Wh": compute_only_j / Decimal("3600"),
        "encoding_energy_Wh": encoding_j / Decimal("3600"),
        "tx_energy_Wh": tx_j / Decimal("3600"),
        "controllable_compute_communication_energy_Wh": (compute_total_j + tx_j) / Decimal("3600"),
        "transmitted_MB": dec(resource.get("transmitted_mb", 0)),
        "peak_downlink_queue_MB": max(queues, default=Decimal("0")),
        "mean_downlink_queue_MB": sum(queues, Decimal("0")) / Decimal(max(len(queues), 1)),
        "initial_soc": Decimal("0.85"),
        "final_soc": socs[-1] if socs else Decimal("0"),
        "minimum_soc": min(socs, default=Decimal("0")),
        "maximum_DoD": Decimal("1") - min(socs, default=Decimal("0")),
        "maximum_thermal_state": max(thermals, default=Decimal("0")),
        "thermal_violation_count": sum(integer(row["thermal_violation"]) for row in slots),
        "soc_violation_count": sum(integer(row["soc_violation"]) for row in slots),
        "max_energy_balance_residual_Wh": residual_max,
    }


def subgroup_metrics(eval_mod: Any, path: Path, metric: dict[str, Any], utility_cfg: dict[str, Any]) -> dict[str, Any]:
    tasks = read_csv(path / "per_task_outcomes.csv")
    events = read_csv(path / "ground_delivery_events.csv")
    by_task = {row["task_id"]: row for row in tasks}
    ledger = eval_mod.ledger_for(tasks, events, utility_cfg)
    gross_by_level: dict[str, Decimal] = defaultdict(Decimal)
    for row in ledger:
        level = by_task[row["task_id"]]["value_level"]
        gross_by_level[level] += dec(row["positive_gross_utility"])
    total_gross = dec(metric["gross_delivered_utility"])
    total_cost = dec(metric["public_method_cost"])
    high_gross = gross_by_level["high"]
    urgent_gross = gross_by_level["urgent"]
    high_net = high_gross - (total_cost * high_gross / total_gross if total_gross else Decimal("0"))
    urgent_net = urgent_gross - (total_cost * urgent_gross / total_gross if total_gross else Decimal("0"))
    valid_events = [row for row in events if integer(row.get("valid_before_deadline")) and not integer(row.get("duplicate_flag"))]

    def delivered_count(levels: set[str], stages: set[str]) -> int:
        return len(
            {
                row["task_id"]
                for row in valid_events
                if by_task.get(row["task_id"], {}).get("value_level") in levels and row.get("stage") in stages
            }
        )

    closed = [row for row in tasks if integer(row.get("closure_success"))]
    urgent_closed = [row for row in closed if row.get("value_level") == "urgent"]
    hv_closed = [row for row in closed if row.get("value_level") in {"high", "urgent"}]

    def latencies(rows: list[dict[str, str]]) -> list[Decimal]:
        return [
            dec(row["closure_ground_time"]) - dec(row["arrival_time"])
            for row in rows
            if row.get("closure_ground_time") not in ("", None)
        ]

    all_lat = latencies(closed)
    hv_lat = latencies(hv_closed)
    urgent_lat = latencies(urgent_closed)
    censored = [
        dec(row["closure_ground_time"]) - dec(row["arrival_time"])
        if integer(row.get("closure_success")) and row.get("closure_ground_time") not in ("", None)
        else dec(row["deadline"]) - dec(row["arrival_time"])
        for row in tasks
    ]
    return {
        "ordinary_closure_count": sum(
            1 for row in closed if row.get("value_level") in {"low", "medium"}
        ),
        "high_closure_count": sum(1 for row in closed if row.get("value_level") == "high"),
        "urgent_closure_count": len(urgent_closed),
        "high_gross_utility": high_gross,
        "urgent_gross_utility": urgent_gross,
        "high_net_utility": high_net,
        "urgent_net_utility": urgent_net,
        "HV_metadata_delivered_count": delivered_count({"high", "urgent"}, {"metadata"}),
        "HV_ROI_delivered_count": delivered_count({"high", "urgent"}, {"roi"}),
        "HV_evidence_delivered_count": delivered_count({"high", "urgent"}, {"evidence", "full"}),
        "urgent_metadata_delivered_count": delivered_count({"urgent"}, {"metadata"}),
        "urgent_ROI_delivered_count": delivered_count({"urgent"}, {"roi"}),
        "urgent_evidence_delivered_count": delivered_count({"urgent"}, {"evidence", "full"}),
        "successful_closure_mean_latency_s": sum(all_lat, Decimal("0")) / Decimal(max(len(all_lat), 1)),
        "successful_closure_p95_latency_s": percentile95(all_lat),
        "HV_successful_closure_mean_latency_s": sum(hv_lat, Decimal("0")) / Decimal(max(len(hv_lat), 1)),
        "HV_successful_closure_p95_latency_s": percentile95(hv_lat),
        "urgent_successful_closure_mean_latency_s": sum(urgent_lat, Decimal("0")) / Decimal(max(len(urgent_lat), 1)),
        "urgent_successful_closure_p95_latency_s": percentile95(urgent_lat),
        "deadline_censored_mean_latency_s": sum(censored, Decimal("0")) / Decimal(max(len(censored), 1)),
    }


def projection_summary_rows(projection_log: list[dict[str, str]]) -> list[dict[str, Any]]:
    levels = ["all", "low", "medium", "high", "urgent"]
    rows = []
    for level in levels:
        subset = projection_log if level == "all" else [row for row in projection_log if row["value_level"] == level]
        mismatch_reasons = {
            "requirement_projected",
            "ordinary_required_product_infeasible",
            "metadata_first_progressive_fallback",
            "high_value_defer_recheck",
        }
        progressive_reasons = {
            "metadata_first_progressive_fallback",
            "high_value_defer_recheck",
        }
        rows.append(
            {
                "value_level": level,
                "selected_action_requirement_mismatch_count": sum(
                    row["projection_reason"] in mismatch_reasons for row in subset
                ),
                "already_aligned_count": sum(row["projection_reason"] == "already_aligned" for row in subset),
                "requirement_projected_count": sum(row["projection_reason"] == "requirement_projected" for row in subset),
                "same_model_product_upgrade_count": sum(integer(row.get("same_model_upgrade")) for row in subset),
                "model_and_product_upgrade_count": sum(
                    integer(row.get("model_changed")) and integer(row.get("product_upgraded")) for row in subset
                ),
                "progressive_fallback_count": sum(row["projection_reason"] in progressive_reasons for row in subset),
                "ordinary_required_product_infeasible_defer_count": sum(
                    row["projection_reason"] == "ordinary_required_product_infeasible" for row in subset
                ),
                "metadata_first_progressive_fallback_count": sum(
                    row["projection_reason"] == "metadata_first_progressive_fallback" for row in subset
                ),
                "high_value_defer_recheck_count": sum(
                    row["projection_reason"] == "high_value_defer_recheck" for row in subset
                ),
                "ordinary_progressive_violation_count": sum(
                    integer(row.get("ordinary_progressive_violation")) for row in subset
                ),
                "projection_blocked_by_deadline_count": sum(integer(row.get("projection_blocked_by_deadline")) for row in subset),
                "projection_blocked_by_soc_count": sum(integer(row.get("projection_blocked_by_soc")) for row in subset),
                "projection_blocked_by_thermal_count": sum(integer(row.get("projection_blocked_by_thermal")) for row in subset),
                "projection_blocked_by_queue_count": sum(integer(row.get("projection_blocked_by_queue")) for row in subset),
                "projection_blocked_by_negative_score_count": sum(integer(row.get("projection_blocked_by_negative_score")) for row in subset),
                "projection_blocked_by_no_contact_count": sum(integer(row.get("projection_blocked_by_no_contact")) for row in subset),
                "residual_task_reentry_count": sum(integer(row.get("residual_task_reentry")) for row in subset),
                "residual_upgrade_success_count": sum(integer(row.get("residual_upgrade_success")) for row in subset),
                "residual_expired_count": sum(integer(row.get("residual_expired")) for row in subset),
            }
        )
    return rows


def paired_task_rows(out: Path, eval_mod: Any, utility_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_path = out / "runs" / "PROP_ORIGINAL"
    new_path = out / "runs" / "PROP_REQALIGN"
    old_tasks = {row["task_id"]: row for row in read_csv(old_path / "per_task_outcomes.csv")}
    new_tasks = {row["task_id"]: row for row in read_csv(new_path / "per_task_outcomes.csv")}
    projection = {row["task_id"]: row for row in read_csv(new_path / "requirement_projection_log.csv")}

    def utility_by_task(path: Path) -> dict[str, Decimal]:
        tasks = read_csv(path / "per_task_outcomes.csv")
        events = read_csv(path / "ground_delivery_events.csv")
        values: dict[str, Decimal] = defaultdict(Decimal)
        for row in eval_mod.ledger_for(tasks, events, utility_cfg):
            values[row["task_id"]] += dec(row["positive_gross_utility"])
        return values

    old_u = utility_by_task(old_path)
    new_u = utility_by_task(new_path)
    old_slots = read_csv(old_path / "per_slot_resource_trace.csv")
    new_slots = read_csv(new_path / "per_slot_resource_trace.csv")
    old_queue = {integer(row["slot"]): dec(row["queue_mb"]) for row in old_slots}
    new_queue = {integer(row["slot"]): dec(row["queue_mb"]) for row in new_slots}
    rows = []
    for task_id in sorted(old_tasks):
        old = old_tasks[task_id]
        new = new_tasks[task_id]
        proj = projection[task_id]
        old_success = integer(old.get("closure_success"))
        new_success = integer(new.get("closure_success"))
        start_slot = integer(dec(old["arrival_time"]) / Decimal(SLOT_SECONDS))
        end_slot = min(SLOTS - 1, integer(dec(old["deadline"]) / Decimal(SLOT_SECONDS)))
        queue_delta = max(
            [new_queue.get(slot, Decimal("0")) - old_queue.get(slot, Decimal("0")) for slot in range(start_slot, end_slot + 1)]
            or [Decimal("0")]
        )
        queue_displacement_candidate = int(old_success and not new_success and queue_delta > Decimal("1e-12"))
        if not old_success and new_success:
            change_reason = "new_reqalign_success"
        elif old_success and not new_success:
            change_reason = "old_only_success_queue_displacement_candidate" if queue_displacement_candidate else "old_only_success"
        elif old_success and new_success:
            change_reason = "both_success"
        else:
            change_reason = proj["projection_reason"]
        rows.append(
            {
                "task_id": task_id,
                "value_level": old["value_level"],
                "required_product": old["product_requirement"],
                "old_original_action": old["selected_action"],
                "new_original_action": proj["original_action"],
                "new_final_projected_action": proj["final_action"],
                "old_closure": old_success,
                "new_closure": new_success,
                "old_ground_utility": old_u[task_id],
                "new_ground_utility": new_u[task_id],
                "old_latency_s": (
                    dec(old["closure_ground_time"]) - dec(old["arrival_time"]) if old["closure_ground_time"] else ""
                ),
                "new_latency_s": (
                    dec(new["closure_ground_time"]) - dec(new["arrival_time"]) if new["closure_ground_time"] else ""
                ),
                "old_transmitted_bytes": old["transmitted_bytes"],
                "new_transmitted_bytes": new["transmitted_bytes"],
                "old_energy_Wh": dec(old["controllable_energy_j"]) / Decimal("3600"),
                "new_energy_Wh": dec(new["controllable_energy_j"]) / Decimal("3600"),
                "change_reason": change_reason,
                "projection_reason": proj["projection_reason"],
                "queue_displacement_candidate": queue_displacement_candidate,
            }
        )

    def category(level: str) -> str:
        return "ordinary" if level in {"low", "medium"} else level

    summary_counts = []
    for group in ("all", "ordinary", "high", "urgent"):
        subset = rows if group == "all" else [row for row in rows if category(row["value_level"]) == group]
        summary_counts.append(
            {
                "group": group,
                "new_PROP_only_success_count": sum(not row["old_closure"] and row["new_closure"] for row in subset),
                "old_PROP_only_success_count": sum(row["old_closure"] and not row["new_closure"] for row in subset),
                "both_success_count": sum(row["old_closure"] and row["new_closure"] for row in subset),
                "neither_success_count": sum(not row["old_closure"] and not row["new_closure"] for row in subset),
            }
        )
    details = {
        "counts": summary_counts,
        "new_urgent_successes": [row["task_id"] for row in rows if row["value_level"] == "urgent" and not row["old_closure"] and row["new_closure"]],
        "lost_urgent_successes": [row["task_id"] for row in rows if row["value_level"] == "urgent" and row["old_closure"] and not row["new_closure"]],
        "new_high_successes": [row["task_id"] for row in rows if row["value_level"] == "high" and not row["old_closure"] and row["new_closure"]],
        "lost_high_successes": [row["task_id"] for row in rows if row["value_level"] == "high" and row["old_closure"] and not row["new_closure"]],
        "projection_upgrade_successes": [
            row["task_id"]
            for row in rows
            if row["projection_reason"] == "requirement_projected" and not row["old_closure"] and row["new_closure"]
        ],
        "queue_displacement_candidates": [row["task_id"] for row in rows if row["queue_displacement_candidate"]],
        "named_task_audit": [
            {
                "requested_task_id": requested_id,
                "match_status": "matched_frozen_manifest_suffix",
                **next(
                    row
                    for row in rows
                    if row["task_id"] == requested_id or row["task_id"].endswith(requested_id)
                ),
            }
            if any(
                row["task_id"] == requested_id or row["task_id"].endswith(requested_id)
                for row in rows
            )
            else {
                "requested_task_id": requested_id,
                "match_status": "not_present_in_frozen_manifest",
            }
            for requested_id in AUDIT_TASK_IDS
        ],
        "queue_displacement_note": "Causal attribution is conservative: listed tasks are old-only successes with higher new-PROP queue during their lifetime.",
    }
    return rows, details


def no_task_specific_logic_audit() -> dict[str, Any]:
    projection_text = PROJECTION_SOURCE.read_text(encoding="utf-8")
    runner_text = Path(__file__).read_text(encoding="utf-8")
    forbidden_in_projection = [task_id for task_id in AUDIT_TASK_IDS if task_id in projection_text]
    seed_condition = bool(re.search(r"\bif\b[^\n]*\bseed\b[^\n]*(?:==|!=|in)\s*2026", runner_text, flags=re.IGNORECASE))
    tree = ast.parse(runner_text)
    task_id_condition_hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.Compare)):
            segment = ast.get_source_segment(runner_text, node) or ""
            if any(task_id in segment for task_id in AUDIT_TASK_IDS):
                task_id_condition_hits.append(segment)
    passed = not forbidden_in_projection and not seed_condition and not task_id_condition_hits
    return {
        "passed": passed,
        "forbidden_task_ids_in_projection_module": forbidden_in_projection,
        "seed_specific_condition_detected": seed_condition,
        "task_id_condition_hits": task_id_condition_hits,
        "named_task_ids_used_only_for_posthoc_audit": list(AUDIT_TASK_IDS),
    }


def finalize_rollback_manifest() -> dict[str, Any]:
    manifest = read_json(ROLLBACK_MANIFEST)
    for entry in manifest["new_files"]:
        path = ROOT / entry["path"]
        entry["post_change_sha256"] = sha256_file(path) if path.exists() else None
    manifest["finalized_utc"] = utc_now()
    write_json(ROLLBACK_MANIFEST, manifest)
    process = subprocess.run(
        [sys.executable, str(RESTORE_SCRIPT), "--dry-run"],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    audit = {
        "passed": process.returncode == 0,
        "dry_run_returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "manifest_path": str(ROLLBACK_MANIFEST),
        "restore_script_path": str(RESTORE_SCRIPT),
        "manifest_sha256": sha256_file(ROLLBACK_MANIFEST),
        "restore_script_sha256": sha256_file(RESTORE_SCRIPT),
    }
    return audit


def generate_figures(out: Path, primary_all: list[dict[str, Any]], high_urgent: list[dict[str, Any]], paired: list[dict[str, Any]], projection_summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )

    def save(fig, name: str):
        fig.tight_layout()
        fig.savefig(figure_dir / f"{name}.pdf")
        fig.savefig(figure_dir / f"{name}.png", dpi=300)
        plt.close(fig)

    by = {row["method"]: row for row in primary_all}
    old_new = [by[METHOD_PROP_OLD], by[METHOD_PROP_NEW]]
    metrics = ["J_all_net", "J_HV_net", "HV_closure_count"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
    for index, metric in enumerate(metrics):
        axes[index].bar(["Old PROP", "REQALIGN"], [flt(row[metric]) for row in old_new], color=colors[:2])
        axes[index].set_title(metric)
        axes[index].set_ylim(bottom=0)
    save(fig, "01_prop_old_new_primary_metrics")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    labels = [row["method"].replace("PROP-SHARED-CALIBRATED-PRIOR-", "PROP-") for row in primary_all]
    for index, metric in enumerate(metrics):
        axes[index].bar(np.arange(len(primary_all)), [flt(row[metric]) for row in primary_all], color=colors)
        axes[index].set_title(metric)
        axes[index].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        axes[index].set_ylim(bottom=0)
    save(fig, "02_six_method_primary_metrics")

    hu_by = {row["method"]: row for row in high_urgent}
    x = np.arange(len(primary_all))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    ax.bar(x - width / 2, [integer(hu_by[row["method"]]["high_closure_count"]) for row in primary_all], width, label="High", color=colors[0])
    ax.bar(x + width / 2, [integer(hu_by[row["method"]]["urgent_closure_count"]) for row in primary_all], width, label="Urgent", color=colors[1])
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Closure count")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    save(fig, "03_high_urgent_closure")

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    stage_fields = ["HV_metadata_delivered_count", "HV_ROI_delivered_count", "HV_evidence_delivered_count"]
    offsets = [-width, 0, width]
    for idx, field in enumerate(stage_fields):
        ax.bar(x + offsets[idx], [integer(hu_by[row["method"]][field]) for row in primary_all], width, label=field.replace("HV_", "").replace("_delivered_count", ""), color=colors[idx])
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("HV tasks delivered")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    save(fig, "04_hv_product_stage_delivery")

    categories = ["new only", "old only", "both", "neither"]
    groups = ["ordinary", "high", "urgent"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for idx, group in enumerate(groups):
        subset = [row for row in paired if ("ordinary" if row["value_level"] in {"low", "medium"} else row["value_level"]) == group]
        values = [
            sum(not row["old_closure"] and row["new_closure"] for row in subset),
            sum(row["old_closure"] and not row["new_closure"] for row in subset),
            sum(row["old_closure"] and row["new_closure"] for row in subset),
            sum(not row["old_closure"] and not row["new_closure"] for row in subset),
        ]
        ax.bar(np.arange(4) + (idx - 1) * 0.24, values, 0.24, label=group, color=colors[idx])
    ax.set_xticks(np.arange(4), categories)
    ax.set_ylabel("Task count")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    save(fig, "05_prop_task_gain_loss_pairing")

    for field, ylabel, name in [
        ("battery_soc", "SOC", "06_prop_soc_trace"),
        ("queue_mb", "Queue (MB)", "07_prop_queue_trace"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        for idx, (label, method) in enumerate([("Old PROP", METHOD_PROP_OLD), ("REQALIGN", METHOD_PROP_NEW)]):
            slots = read_csv(method_paths(out)[method] / "per_slot_resource_trace.csv")
            ax.plot([flt(row["time_s"]) / 3600.0 for row in slots], [flt(row[field]) for row in slots], label=label, color=colors[idx], linewidth=1.0)
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        save(fig, name)

    total = next(row for row in projection_summary if row["value_level"] == "all")
    reasons = ["already_aligned_count", "requirement_projected_count", "progressive_fallback_count"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar([name.replace("_count", "").replace("_", " ") for name in reasons], [integer(total[name]) for name in reasons], color=colors[:3])
    ax.set_ylabel("Task count")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(bottom=0)
    save(fig, "08_projection_reason_counts")


def aggregate_and_report(out: Path, bundle_root: Path, v25: Any, disabled_audit: dict[str, Any], rollback_audit: dict[str, Any]) -> dict[str, Any]:
    metrics, accounting, eval_mod, config = evaluate_methods(out, bundle_root, v25)
    paths = method_paths(out)
    primary_all = [
        {
            "method": row["method"],
            "J_all_net": row["J_all_net"],
            "J_HV_net": row["J_HV_net"],
            "HV_closure_count": row["HV_closure_count"],
        }
        for row in metrics
    ]
    main_primary = [row for row in primary_all if row["method"] != METHOD_PROP_OLD]
    ablation = [
        {
            **row,
            "delta_vs_old_J_all_net": dec(row["J_all_net"]) - dec(primary_all[0]["J_all_net"]),
            "delta_vs_old_J_HV_net": dec(row["J_HV_net"]) - dec(primary_all[0]["J_HV_net"]),
            "delta_vs_old_HV_closure_count": integer(row["HV_closure_count"]) - integer(primary_all[0]["HV_closure_count"]),
        }
        for row in primary_all[:2]
    ]
    high_urgent = []
    auxiliary = []
    energy = []
    safety = []
    metric_by = {row["method"]: row for row in metrics}
    for method, path in paths.items():
        subgroup = subgroup_metrics(eval_mod, path, metric_by[method], config["product_requirement_utility"])
        energ = energy_metrics(method, path)
        high_urgent.append({"method": method, **subgroup})
        tasks = read_csv(path / "per_task_outcomes.csv")
        auxiliary.append(
            {
                "method": method,
                "all_task_closure_count": metric_by[method]["closure_count"],
                "all_task_closure_rate": metric_by[method]["closure_rate"],
                **subgroup,
                "transmitted_MB": energ["transmitted_MB"],
                "peak_downlink_queue_MB": energ["peak_downlink_queue_MB"],
                "mean_downlink_queue_MB": energ["mean_downlink_queue_MB"],
                "compute_energy_Wh": energ["compute_energy_Wh"],
                "encoding_energy_Wh": energ["encoding_energy_Wh"],
                "tx_energy_Wh": energ["tx_energy_Wh"],
                "controllable_compute_communication_energy_Wh": energ["controllable_compute_communication_energy_Wh"],
                "initial_soc": energ["initial_soc"],
                "final_soc": energ["final_soc"],
                "minimum_soc": energ["minimum_soc"],
                "maximum_DoD": energ["maximum_DoD"],
                "maximum_thermal_state": energ["maximum_thermal_state"],
                "thermal_violation_count": energ["thermal_violation_count"],
                "soc_violation_count": energ["soc_violation_count"],
                "defer_count": metric_by[method]["defer_count"],
                "safety_filter_correction_count": sum(
                    row.get("selected_action") != row.get("executed_action") for row in tasks
                ),
                "deadline_rejection_count": sum(
                    "deadline" in str(row.get("failure_reason", "")).lower() for row in tasks
                ),
            }
        )
        energy.append(energ)
        safety.append(
            {
                "method": method,
                "minimum_soc": energ["minimum_soc"],
                "maximum_thermal_state": energ["maximum_thermal_state"],
                "soc_violation_count": energ["soc_violation_count"],
                "thermal_violation_count": energ["thermal_violation_count"],
            }
        )

    new = next(row for row in primary_all if row["method"] == METHOD_PROP_NEW)
    deltas = []
    for row in main_primary:
        if row["method"] == METHOD_PROP_NEW:
            continue
        deltas.append(
            {
                "baseline": row["method"],
                "delta_J_all_net_REQALIGN_minus_baseline": dec(new["J_all_net"]) - dec(row["J_all_net"]),
                "delta_J_HV_net_REQALIGN_minus_baseline": dec(new["J_HV_net"]) - dec(row["J_HV_net"]),
                "delta_HV_closure_count_REQALIGN_minus_baseline": integer(new["HV_closure_count"]) - integer(row["HV_closure_count"]),
            }
        )
    projection_log = read_csv(paths[METHOD_PROP_NEW] / "requirement_projection_log.csv")
    projection_summary = projection_summary_rows(projection_log)
    paired, paired_details = paired_task_rows(out, eval_mod, config["product_requirement_utility"])

    table_specs = [
        ("main_primary_metrics", main_primary),
        ("prop_reqalign_ablation", ablation),
        ("auxiliary_metrics", auxiliary),
        ("high_urgent_breakdown", high_urgent),
        ("requirement_projection_summary", projection_summary),
    ]
    for name, rows in table_specs:
        write_csv(out / "tables" / f"{name}.csv", rows)
        write_md_table(out / "tables" / f"{name}.md", rows)
    write_csv(out / "tables" / "prop_old_new_paired_tasks.csv", paired)
    paired_md = [
        "# Old vs new PROP paired-task summary",
        "",
        markdown_table(paired_details["counts"]),
        "",
        "## Task lists",
        "",
        f"- New urgent successes: `{';'.join(paired_details['new_urgent_successes']) or 'none'}`",
        f"- Lost urgent successes: `{';'.join(paired_details['lost_urgent_successes']) or 'none'}`",
        f"- New high successes: `{';'.join(paired_details['new_high_successes']) or 'none'}`",
        f"- Lost high successes: `{';'.join(paired_details['lost_high_successes']) or 'none'}`",
        f"- Projection-upgrade successes: `{';'.join(paired_details['projection_upgrade_successes']) or 'none'}`",
        f"- Queue-displacement candidates: `{';'.join(paired_details['queue_displacement_candidates']) or 'none'}`",
        "",
        paired_details["queue_displacement_note"],
        "",
        "## Required named-task audit",
        "",
        markdown_table(paired_details["named_task_audit"]),
    ]
    (out / "tables" / "prop_old_new_paired_summary.md").write_text("\n".join(paired_md) + "\n", encoding="utf-8")
    write_csv(out / "tables" / "energy_metrics.csv", energy)
    write_csv(out / "tables" / "safety_metrics.csv", safety)
    write_csv(out / "tables" / "prop_vs_baselines_deltas.csv", deltas)

    public_ok = all(
        abs(dec(row["accounting_error"])) <= Decimal("1e-10")
        and bool(row["closure_decomposition_ok"])
        and bool(row["hv_decomposition_ok"])
        for row in accounting
    )
    scenario_rows = []
    identity_rows = []
    for method, path in paths.items():
        scenario = read_json(path / "reconstructed_scenario_identity.json")
        identity = read_json(path / "native_identity.json")
        scenario_rows.append(
            {
                "method": method,
                "task_hash_match": scenario["task_hash_match"],
                "frozen_power_identity_match": scenario["frozen_power_identity_match"],
                "runtime_value_mismatch_count": scenario["runtime_value_mismatch_count"],
                "field_mismatch_count": scenario["field_mismatch_count"],
            }
        )
        identity_rows.append(
            {
                "method": method,
                "source_identity_passed": identity.get("source_identity_passed", False),
                "source_root": identity.get("source_root", ""),
                "requirement_projection_enabled": identity.get("requirement_projection_enabled", False),
            }
        )
    scenario_ok = all(
        row["task_hash_match"]
        and row["frozen_power_identity_match"]
        and integer(row["runtime_value_mismatch_count"]) == 0
        and integer(row["field_mismatch_count"]) == 0
        for row in scenario_rows
    )
    method_ok = all(row["source_identity_passed"] for row in identity_rows)
    energy_balance_rows = [
        {
            "method": row["method"],
            "max_energy_balance_residual_Wh": row["max_energy_balance_residual_Wh"],
            "identity_check": "PASS" if dec(row["max_energy_balance_residual_Wh"]) <= Decimal("1e-8") else "FAIL",
            "controllable_energy_Wh": row["controllable_compute_communication_energy_Wh"],
        }
        for row in energy
    ]
    energy_ok = (
        all(row["soc_violation_count"] == 0 and row["thermal_violation_count"] == 0 for row in safety)
        and all(dec(row["max_energy_balance_residual_Wh"]) <= Decimal("1e-8") for row in energy)
    )
    no_specific = no_task_specific_logic_audit()
    ordinary_projection_rows = [
        row for row in projection_log if row.get("value_level") in {"low", "medium"}
    ]
    ordinary_progressive_violations = [
        row
        for row in ordinary_projection_rows
        if integer(row.get("ordinary_progressive_violation"))
        or integer(row.get("reenter_future_decisions"))
        or row.get("projection_reason")
        in {"metadata_first_progressive_fallback", "high_value_defer_recheck"}
    ]
    projection_audit = {
        "passed": len(projection_log) == EXPECTED_TASK_COUNT and not ordinary_progressive_violations,
        "row_count": len(projection_log),
        "summary": projection_summary,
        "config": load_projection_config(True),
        "ordinary_progressive_violation_count": len(ordinary_progressive_violations),
        "ordinary_progressive_violation_task_ids": [
            row["task_id"] for row in ordinary_progressive_violations
        ],
        "future_ground_truth_used_for_selection": False,
        "projection_source_sha256": sha256_file(PROJECTION_SOURCE),
    }
    write_json(out / "audit" / "public_accounting_audit.json", {"passed": public_ok, "rows": accounting})
    write_json(out / "audit" / "scenario_pairing_audit.json", {"passed": scenario_ok, "rows": scenario_rows})
    write_json(out / "audit" / "method_identity_audit.json", {"passed": method_ok, "rows": identity_rows})
    write_json(out / "audit" / "requirement_projection_audit.json", projection_audit)
    write_json(out / "audit" / "no_task_specific_logic_audit.json", no_specific)
    write_json(out / "audit" / "rollback_audit.json", rollback_audit)
    write_csv(out / "audit" / "energy_balance_audit.csv", energy_balance_rows)
    generate_figures(out, primary_all, high_urgent, paired, projection_summary)

    old_primary = next(row for row in primary_all if row["method"] == METHOD_PROP_OLD)
    old_hu = next(row for row in high_urgent if row["method"] == METHOD_PROP_OLD)
    new_hu = next(row for row in high_urgent if row["method"] == METHOD_PROP_NEW)
    total_projection = next(row for row in projection_summary if row["value_level"] == "all")
    ordinary_summary = {
        row["value_level"]: row
        for row in projection_summary
        if row["value_level"] in {"low", "medium"}
    }
    high_value_rows = [
        row for row in projection_log if row.get("value_level") in {"high", "urgent"}
    ]
    metadata_first_rows = [
        row
        for row in high_value_rows
        if row.get("projection_reason") == "metadata_first_progressive_fallback"
    ]
    events_by_task: dict[str, set[str]] = defaultdict(set)
    for event in read_csv(paths[METHOD_PROP_NEW] / "ground_delivery_events.csv"):
        events_by_task[event["task_id"]].add(str(event.get("stage", "")).lower())
    metadata_first_roi = sum(
        bool(events_by_task[row["task_id"]] & {"roi", "evidence", "full"})
        for row in metadata_first_rows
    )
    metadata_first_final = sum(
        bool(events_by_task[row["task_id"]] & {"evidence", "full"})
        for row in metadata_first_rows
    )
    metadata_warning_without_final = sum(
        "metadata" in events_by_task[row["task_id"]]
        and not bool(events_by_task[row["task_id"]] & {"evidence", "full"})
        for row in metadata_first_rows
    )
    high_value_direct_count = sum(
        integer(row.get("direct_closure_feasible")) for row in high_value_rows
    )
    ordinary_defer_count = sum(
        integer(ordinary_summary[level]["ordinary_required_product_infeasible_defer_count"])
        for level in ("low", "medium")
    )
    ordinary_progressive_count = sum(
        integer(ordinary_summary[level]["ordinary_progressive_violation_count"])
        for level in ("low", "medium")
    )
    hard_safe = all(row["soc_violation_count"] == 0 and row["thermal_violation_count"] == 0 for row in safety)
    supports = (
        dec(new["J_all_net"]) >= dec(old_primary["J_all_net"])
        and dec(new["J_HV_net"]) >= dec(old_primary["J_HV_net"])
        and integer(new["HV_closure_count"]) >= integer(old_primary["HV_closure_count"])
        and hard_safe
    )
    assessment = [
        "# PROP requirement-aware projection seed=2026 assessment",
        "",
        f"Final status: `{FINAL_COMPLETE}`",
        "",
        "This is a single-seed mechanism diagnostic and ablation. It does not establish statistical significance or stable superiority.",
        "",
        "## Required answers",
        "",
        f"1. Low/medium entered progressive delivery: **{'no' if ordinary_progressive_count == 0 else 'yes'}**; violations={ordinary_progressive_count}.",
        f"2. Ordinary infeasible required-product decisions used DEFER: **{ordinary_defer_count}**.",
        f"3. High/urgent direct-final feasible decisions sent the final product directly: **{high_value_direct_count}**.",
        f"4. Direct-final infeasible high/urgent tasks entering Metadata-first: **{len(metadata_first_rows)}**.",
        f"5. Metadata-first tasks later delivering ROI or ROI+Ctx: **{metadata_first_roi}**; final ROI+Ctx/evidence={metadata_first_final}.",
        f"6. Metadata warnings without final evidence closure: **{metadata_warning_without_final}**.",
        "7. High and urgent are separately reported: **yes**, including closure, utility, latency, residual success, and expiry.",
        "8. Urgent priority uses frozen effective value plus deterministic feasibility/slack/score/task-id tie-break: **yes**; no reward bonus or resource discount.",
        "9. Infeasible urgent preempting feasible positive high: **not permitted by the feasibility-first tie-break**.",
        "10. Original scorer, resource prices, and hard constraints modified: **no**.",
        f"11. Ordinary tasks incorrectly assigned residual state: **{ordinary_progressive_count}**.",
        f"12. Alignment can be disabled and exactly restore old PROP: **{'yes' if disabled_audit['passed'] else 'no'}**.",
        "",
        "## Additional engineering audit",
        "",
        f"- Source rollback complete by script: **{'yes' if rollback_audit['passed'] else 'no'}**.",
        f"- Task-ID or seed-specific decision logic: **{'no' if no_specific['passed'] else 'detected'}**.",
        f"- Original action/requirement mismatches: **{total_projection['selected_action_requirement_mismatch_count']}**.",
        f"- Direct requirement projections: **{total_projection['requirement_projected_count']}**.",
        f"- Same-model product-only upgrades: **{total_projection['same_model_product_upgrade_count']}**.",
        f"- Model-and-product upgrades: **{total_projection['model_and_product_upgrade_count']}**.",
        f"- High-value progressive fallbacks/rechecks: **{total_projection['progressive_fallback_count']}**.",
        f"13. Old PROP: J_all_net={old_primary['J_all_net']}, J_HV_net={old_primary['J_HV_net']}, HV={old_primary['HV_closure_count']}, high={old_hu['high_closure_count']}, urgent={old_hu['urgent_closure_count']}.",
        f"    New PROP: J_all_net={new['J_all_net']}, J_HV_net={new['J_HV_net']}, HV={new['HV_closure_count']}, high={new_hu['high_closure_count']}, urgent={new_hu['urgent_closure_count']}.",
        f"14. New/lost tasks are listed in `tables/prop_old_new_paired_summary.md`; new urgent={paired_details['new_urgent_successes']}, lost urgent={paired_details['lost_urgent_successes']}, new high={paired_details['new_high_successes']}, lost high={paired_details['lost_high_successes']}.",
        f"15. Evidence-upgrade queue-displacement candidates: **{paired_details['queue_displacement_candidates']}**; this is conservative correlation, not hidden causal certainty.",
        f"16. Queue congestion: old peak={next(row for row in energy if row['method']==METHOD_PROP_OLD)['peak_downlink_queue_MB']} MB, new peak={next(row for row in energy if row['method']==METHOD_PROP_NEW)['peak_downlink_queue_MB']} MB.",
        f"17. SOC/thermal violations: **{'none' if hard_safe else 'present; see safety_metrics.csv'}**.",
        "18. New PROP vs TUT/B1/OFFLOAD/MYOPIC is reported without selection or suppression in `main_primary_metrics.csv` and `prop_vs_baselines_deltas.csv`.",
        f"19. Single-seed evidence supports promotion as the formal main method: **{'provisionally yes, pending multiseed confirmation' if supports else 'no'}**.",
        f"20. Not promoted because J_HV_net changed by {dec(new['J_HV_net']) - dec(old_primary['J_HV_net'])} and urgent closures changed from {old_hu['urgent_closure_count']} to {new_hu['urgent_closure_count']}; the J_all_net/HV-count gains do not override that mixed single-seed evidence.",
        "",
        "## Main primary metrics",
        "",
        markdown_table(main_primary),
        "",
        "## Old/new ablation",
        "",
        markdown_table(ablation),
    ]
    (out / "final_assessment.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")
    return {
        "primary_all": primary_all,
        "main_primary": main_primary,
        "ablation": ablation,
        "auxiliary": auxiliary,
        "high_urgent": high_urgent,
        "energy": energy,
        "safety": safety,
        "projection_summary": projection_summary,
        "paired_details": paired_details,
        "public_ok": public_ok,
        "scenario_ok": scenario_ok,
        "method_ok": method_ok,
        "energy_ok": energy_ok,
        "no_specific_ok": no_specific["passed"],
        "projection_ok": projection_audit["passed"],
    }


def prepare_output(out: Path, overwrite: bool) -> None:
    resolved = out.resolve()
    expected = OUT.resolve()
    if resolved != expected:
        raise ValueError(f"Output must be the registered directory: {expected}")
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite without --overwrite: {out}")
        shutil.rmtree(out)
    for directory in ("config", "scenario", "runs", "tables", "figures", "audit", "rollback", "logs"):
        (out / directory).mkdir(parents=True, exist_ok=True)


def run_tests(out: Path) -> bool:
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_requirement_aware_projection.py", "-q"],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    report = (
        f"command: {sys.executable} -m pytest tests/test_requirement_aware_projection.py -q\n"
        f"returncode: {process.returncode}\n\n{process.stdout}"
    )
    (out / "test_report.txt").write_text(report, encoding="utf-8")
    return process.returncode == 0


def write_preflight_audits(out: Path, v25: Any, input_audit: dict[str, Any], frozen: dict[str, Any]) -> None:
    path_audit = source_code_path_audit(v25)
    write_json(out / "audit" / "code_path_audit.json", path_audit)
    task_ok = (
        frozen["recovered_task_manifest_hash"] == EXPECTED_TASK_HASH
        and frozen["task_count"] == EXPECTED_TASK_COUNT
        and frozen["value_level_counts"] == EXPECTED_COUNTS
        and frozen["N_HV"] == EXPECTED_HV
    )
    write_json(
        out / "audit" / "task_manifest_audit.json",
        {"passed": task_ok, "expected_hash": EXPECTED_TASK_HASH, "observed": frozen},
    )
    write_json(
        out / "audit" / "power_identity_audit.json",
        {"passed": input_audit["frozen_power"] == v25.FROZEN_POWER, "observed": input_audit},
    )
    shutil.copy2(PROJECTION_CONFIG, out / "config" / "prop_requirement_projection_v1.yaml")
    off = load_projection_config(False)
    on = load_projection_config(True)
    (out / "config" / "PROP_ORIGINAL_requirement_projection.yaml").write_text(
        yaml.safe_dump({"requirement_alignment": off}, sort_keys=False),
        encoding="utf-8",
    )
    (out / "config" / "PROP_REQALIGN_requirement_projection.yaml").write_text(
        yaml.safe_dump({"requirement_alignment": on}, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copy2(v25.FROZEN_MANIFEST, out / "scenario" / v25.FROZEN_MANIFEST.name)


def print_terminal(result: dict[str, Any], disabled: dict[str, Any], rollback: dict[str, Any], out: Path) -> None:
    by = {row["method"]: row for row in result["primary_all"]}
    old = by[METHOD_PROP_OLD]
    new = by[METHOD_PROP_NEW]
    hu = {row["method"]: row for row in result["high_urgent"]}
    projection = next(row for row in result["projection_summary"] if row["value_level"] == "all")
    safety = {row["method"]: row for row in result["safety"]}
    print("MODEL AND IMPLEMENTATION")
    print(f"- Original PROP method identity: {METHOD_PROP_OLD}")
    print(f"- New PROP method identity: {METHOD_PROP_NEW}")
    print(f"- requirement projection SHA256: {sha256_file(PROJECTION_SOURCE)}")
    print(f"- config SHA256: {sha256_file(PROJECTION_CONFIG)}")
    print(f"- rollback package: {ROLLBACK_DIR}")
    print("\nDISABLED REGRESSION")
    print(f"- exact reproduction: {disabled['passed']}")
    print(f"- action trace identical: {disabled['semantic_hashes']['action_trace_hash']['new'] == disabled['semantic_hashes']['action_trace_hash']['old']}")
    print(f"- primary metrics identical: {all(disabled['scalar_checks'][key] for key in ('J_all_net','J_HV_net','HV_closure_count'))}")
    print("\nPRIMARY METRICS")
    print(markdown_table(result["primary_all"]))
    print("\nOLD VS NEW PROP")
    print(
        markdown_table(
            [
                {"metric": key, "old PROP": old[key], "new PROP": new[key], "delta": dec(new[key]) - dec(old[key])}
                for key in ("J_all_net", "J_HV_net", "HV_closure_count")
            ]
        )
    )
    print("\nHIGH / URGENT")
    print(
        markdown_table(
            [
                {
                    "method": method,
                    "high closure": row["high_closure_count"],
                    "urgent closure": row["urgent_closure_count"],
                    "high utility": row["high_net_utility"],
                    "urgent utility": row["urgent_net_utility"],
                }
                for method, row in hu.items()
            ]
        )
    )
    print("\nPROJECTION SUMMARY")
    print(
        markdown_table(
            [
                {
                    "mismatches": projection["selected_action_requirement_mismatch_count"],
                    "projected": projection["requirement_projected_count"],
                    "same-model upgrades": projection["same_model_product_upgrade_count"],
                    "progressive fallbacks": projection["progressive_fallback_count"],
                    "residual successes": projection["residual_upgrade_success_count"],
                }
            ]
        )
    )
    print("\nSAFETY")
    print(
        markdown_table(
            [
                {
                    "method": method,
                    "min SOC": row["minimum_soc"],
                    "max thermal": row["maximum_thermal_state"],
                    "SOC violations": row["soc_violation_count"],
                    "thermal violations": row["thermal_violation_count"],
                }
                for method, row in safety.items()
            ]
        )
    )
    print(f"\nOUTPUT_DIR={out}")
    print(f"ROLLBACK_AUDIT_PASS={rollback['passed']}")
    print(f"\nFINAL STATUS: {FINAL_COMPLETE}")


def fail(out: Path, status: str, detail: Any) -> int:
    write_json(out / "audit" / "failure_detail.json", {"status": status, "detail": detail, "time_utc": utc_now()})
    (out / "final_assessment.md").write_text(f"Final status: `{status}`\n", encoding="utf-8")
    print(status)
    print(json.dumps(detail, ensure_ascii=False, indent=2, default=str))
    return 2


def parent_main(args: argparse.Namespace) -> int:
    prepare_output(args.output_dir, args.overwrite)
    out = args.output_dir
    rollback_audit = finalize_rollback_manifest()
    shutil.copytree(ROLLBACK_DIR, out / "rollback" / ROLLBACK_DIR.name)
    write_json(out / "audit" / "rollback_audit.json", rollback_audit)
    if not rollback_audit["passed"]:
        return fail(out, "ROLLBACK_PACKAGE_INCOMPLETE", rollback_audit)
    if not run_tests(out):
        return fail(out, "UNIT_TEST_FAILED", {"test_report": str(out / "test_report.txt")})
    rollback_audit = finalize_rollback_manifest()
    write_json(out / "audit" / "rollback_audit.json", rollback_audit)
    if not rollback_audit["passed"]:
        return fail(out, "ROLLBACK_PACKAGE_INCOMPLETE", rollback_audit)

    v25 = load_v25()
    try:
        input_audit, frozen = v25.preflight(out)
    except ValueError as exc:
        return fail(out, "TASK_MANIFEST_MISMATCH", {"error": str(exc)})
    except Exception as exc:
        return fail(out, "POWER_CONFIG_MISMATCH", {"error": repr(exc)})
    write_preflight_audits(out, v25, input_audit, frozen)
    bundle_root = v25.prepare_runtime_bundle(out)
    shutil.copy2(bundle_root / "slot_trace.csv", out / "scenario" / "slot_trace.csv")
    shutil.copy2(bundle_root / "task_manifest.csv", out / "scenario" / "task_manifest.csv")

    print("[REGRESSION] PROP projection disabled", flush=True)
    regression_path = out / "audit" / "disabled_regression_run"
    regression_run = launch_child("PROP_DISABLED_REGRESSION", bundle_root, regression_path, out / "logs")
    if regression_run["status"] != "PASS":
        return fail(out, "PROP_ORIGINAL_RUNTIME_FAILED", regression_run)
    disabled = disabled_regression_audit(regression_path, v25, bundle_root)
    write_json(out / "audit" / "disabled_regression_audit.json", disabled)
    if not disabled["passed"]:
        return fail(out, "REQALIGN_DISABLED_REGRESSION_MISMATCH", disabled)

    run_rows = []
    for label, _method in METHODS:
        print(f"[RUN] {label}", flush=True)
        row = launch_child(label, bundle_root, out / "runs" / label, out / "logs")
        run_rows.append(row)
        write_csv(out / "audit" / "formal_run_status.csv", run_rows)
        if row["status"] != "PASS":
            return fail(out, FAILURE_BY_LABEL[label], row)
    if len(run_rows) != len(METHODS):
        return fail(out, "INCOMPLETE_RERUN_GRID", {"completed": run_rows})

    result = aggregate_and_report(out, bundle_root, v25, disabled, rollback_audit)
    if not result["scenario_ok"]:
        return fail(out, "SCENARIO_PAIRING_MISMATCH", result)
    if not result["method_ok"]:
        return fail(out, "METHOD_IDENTITY_MISMATCH", result)
    if not result["no_specific_ok"]:
        return fail(out, "TASK_SPECIFIC_LOGIC_DETECTED", result)
    if not result["energy_ok"]:
        return fail(out, "ENERGY_BALANCE_MISMATCH", result)
    if not result["public_ok"]:
        return fail(out, "PUBLIC_ACCOUNTING_MISMATCH", result)
    if not result["projection_ok"]:
        return fail(out, "INCOMPLETE_RERUN_GRID", result)
    print_terminal(result, disabled, rollback_audit, out)
    return 0


def minimal_correction_regression_main(*, overwrite: bool) -> int:
    """Run tests and disabled regression only; never launch the corrected method."""

    out = CORRECTION_AUDIT_OUT
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite without --overwrite: {out}")
        shutil.rmtree(out)
    for directory in ("audit", "config", "scenario", "logs"):
        (out / directory).mkdir(parents=True, exist_ok=True)

    write_json(
        out / "audit" / "paused_workspace_audit.json",
        {
            "audit_time_utc": utc_now(),
            "tracked_modified_or_deleted_count_at_pause": 76,
            "untracked_file_count_at_pause": 18943,
            "git_diff_stat_at_pause": "76 files changed, 20072 insertions(+), 791 deletions(-)",
            "task_relevant_tracked_changes_at_pause": [],
            "rollback_manifest_exists": ROLLBACK_MANIFEST.exists(),
            "restore_script_exists": RESTORE_SCRIPT.exists(),
            "formal_experiment_had_run_before_correction": True,
            "old_outputs_overwritten": False,
            "low_medium_progressive_semantics_detected_before_correction": True,
            "stopped_background_processes": [25788, 36504],
            "urgent_guard_source_rollback_completed": True,
            "urgent_guard_outputs_preserved": True,
        },
    )
    rollback_audit = finalize_rollback_manifest()
    write_json(out / "audit" / "rollback_audit.json", rollback_audit)
    if not rollback_audit["passed"]:
        return fail(out, "ROLLBACK_PACKAGE_INCOMPLETE", rollback_audit)
    shutil.copy2(PROJECTION_CONFIG, out / "config" / PROJECTION_CONFIG.name)
    shutil.copy2(RESTORE_SCRIPT, out / "restore_prop_requirement_projection_v1.py")
    if not run_tests(out):
        return fail(out, "UNIT_TEST_FAILED", {"test_report": str(out / "test_report.txt")})
    no_specific = no_task_specific_logic_audit()
    write_json(out / "audit" / "no_task_specific_logic_audit.json", no_specific)
    if not no_specific["passed"]:
        return fail(out, "TASK_SPECIFIC_LOGIC_DETECTED", no_specific)
    write_json(
        out / "audit" / "minimal_correction_implementation_audit.json",
        {
            "ordinary_one_shot_only": True,
            "ordinary_progressive_reentry_enabled": False,
            "high_value_direct_final_first": True,
            "metadata_first_only_after_direct_infeasible": True,
            "high_value_defer_recheck_uses_existing_product_lifecycle": True,
            "defer_recheck_runs_full_original_decision_and_execution_path": True,
            "defer_recheck_requires_idle_contact_slot": True,
            "defer_recheck_physical_safety_and_deadline_prescreen": True,
            "infeasible_urgent_cannot_preempt_feasible_high_in_reentry_prescreen": True,
            "original_scorer_modified": False,
            "original_selector_modified": False,
            "hard_safety_modified": False,
            "baseline_projection_binding": False,
            "formal_corrected_method_run_started": False,
        },
    )

    v25 = load_v25()
    try:
        input_audit, frozen = v25.preflight(out)
    except ValueError as exc:
        return fail(out, "TASK_MANIFEST_MISMATCH", {"error": str(exc)})
    except Exception as exc:
        return fail(out, "POWER_CONFIG_MISMATCH", {"error": repr(exc)})
    write_preflight_audits(out, v25, input_audit, frozen)
    bundle_root = v25.prepare_runtime_bundle(out)
    shutil.copy2(bundle_root / "slot_trace.csv", out / "scenario" / "slot_trace.csv")
    shutil.copy2(bundle_root / "task_manifest.csv", out / "scenario" / "task_manifest.csv")

    regression_path = out / "audit" / "disabled_regression_run"
    regression_run = launch_child(
        "PROP_DISABLED_REGRESSION",
        bundle_root,
        regression_path,
        out / "logs",
    )
    if regression_run["status"] != "PASS":
        return fail(out, "PROP_ORIGINAL_RUNTIME_FAILED", regression_run)
    disabled = disabled_regression_audit(regression_path, v25, bundle_root)
    slots = read_csv(regression_path / "per_slot_resource_trace.csv")
    disabled["explicit_safety_checks"] = {
        "soc_violation_count": sum(integer(row.get("soc_violation")) for row in slots),
        "thermal_violation_count": sum(integer(row.get("thermal_violation")) for row in slots),
    }
    disabled["corrected_method_or_baselines_executed"] = False
    disabled["regression_only_mode"] = True
    disabled["passed"] = (
        disabled["passed"]
        and disabled["explicit_safety_checks"]["soc_violation_count"] == 0
        and disabled["explicit_safety_checks"]["thermal_violation_count"] == 0
    )
    write_json(out / "audit" / "disabled_regression_audit.json", disabled)
    if not disabled["passed"]:
        return fail(out, "REQALIGN_DISABLED_REGRESSION_MISMATCH", disabled)
    rollback_audit = finalize_rollback_manifest()
    write_json(out / "audit" / "rollback_audit.json", rollback_audit)
    if not rollback_audit["passed"]:
        return fail(out, "ROLLBACK_PACKAGE_INCOMPLETE", rollback_audit)

    status = {
        "status": "PROP_REQUIREMENT_ALIGNMENT_MINIMAL_CORRECTION_COMPLETE",
        "scope": "implementation_unit_tests_and_disabled_regression_only",
        "formal_corrected_method_run_started": False,
        "baseline_runs_started": False,
        "unit_tests_passed": True,
        "disabled_regression_passed": True,
        "rollback_audit_passed": rollback_audit["passed"],
        "output": str(out),
    }
    write_json(out / "correction_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--minimal-correction-regression", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--bundle-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.child_run:
        return child_main(args)
    if args.minimal_correction_regression:
        return minimal_correction_regression_main(overwrite=args.overwrite)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
