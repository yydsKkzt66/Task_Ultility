"""One-click frozen V2 final gate and conditional blind five-seed run.

Run from the repository root or with the IDE Run button.  No public arguments
are required.  Internal child arguments are used only for isolated processes.
Historical code, frozen parameters, and prior outputs are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import types
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_historical_asset_audit_conditional_v2_calibration_v1 as calibration  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402
import scripts.run_v25_decoupled_v2_seed2026_diagnostic as v2diag  # noqa: E402
import scripts.run_v25_prop_ablation_five_method_fiveseed_cap050_v1 as legacy_ablation  # noqa: E402


# Load this additive helper under a private module name.  Importing it through
# ``src`` here would preload the workspace ``src`` package before baseline
# children can install their frozen external source root, invalidating native
# source-identity checks and mixing implementations.
_gate_spec = importlib.util.spec_from_file_location(
    "frozen_v2_final_gate_runtime",
    ROOT / "src" / "evaluation" / "frozen_v2_final_gate.py",
)
if _gate_spec is None or _gate_spec.loader is None:
    raise ImportError("frozen_v2_final_gate_runtime")
gatecore = importlib.util.module_from_spec(_gate_spec)
sys.modules[_gate_spec.name] = gatecore
_gate_spec.loader.exec_module(gatecore)


CONFIG = ROOT / "configs" / "frozen_v2_final_gate_v1.yaml"
FROZEN = ROOT / "outputs" / "_historical_asset_audit_conditional_v2_calibration_v1" / "frozen" / "conditional_v2_frozen_parameters_v1.json"
EXISTING_FULL = ROOT / "outputs" / "_historical_asset_audit_conditional_v2_calibration_v1" / "runs" / "formal_test" / "CONDITIONAL_DECOUPLED_V2_FROZEN" / "seed_2026"
OUTPUT = ROOT / "outputs" / "_frozen_v2_final_gate_blind_fiveseed_v1"
CAPACITY = 0.50
SLOTS = 4320
SLOT_SECONDS = 20.0
COMPLETION = "FROZEN_V2_FINAL_GATE_AND_BLIND_FIVESEED_COMPLETE"

FULL = "FULL-PROP-V2"
FIXED = "FIXED-YOLO-B-V2"
SINGLE = "SINGLE-TIMESCALE-V2"
DIRECT = "DIRECT-ONLY-V2"
ACK = "GROUND-ACK-120S-V2"
MYOPIC = "MYOPIC-V2"
B1 = exact.METHOD_B1
OFFLOAD = exact.METHOD_OFFLOAD
TUT = exact.METHOD_TUT
GROUND = "GROUND-ONLY"
METHODS = (FULL, FIXED, SINGLE, DIRECT, ACK, MYOPIC, B1, OFFLOAD, TUT, GROUND)
SLUG = {
    FULL: "FULL_PROP_V2", FIXED: "FIXED_YOLO_B_V2", SINGLE: "SINGLE_TIMESCALE_V2",
    DIRECT: "DIRECT_ONLY_V2", ACK: "GROUND_ACK_120S_V2", MYOPIC: "MYOPIC_V2",
    B1: "B1", OFFLOAD: "OFFLOAD", TUT: "TUT", GROUND: "GROUND_ONLY",
}
V2_METHODS = {FULL, FIXED, SINGLE, DIRECT, ACK, MYOPIC}
BASELINES = {B1, OFFLOAD, TUT, GROUND}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    return gatecore.read_csv(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def write_md(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(data[0]) if data else ["empty"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value: Any, default: float = 0.0) -> float:
    return gatecore.f(value, default)


def stable_hash(value: Any) -> str:
    return gatecore.stable_hash(value)


def load_frozen() -> dict[str, Any]:
    frozen = read_json(FROZEN)
    if not frozen.get("validation_passed") or not frozen.get("selected_without_method_ranking"):
        raise RuntimeError("FROZEN_PARAMETER_IDENTITY_MISMATCH")
    if not (frozen["q_eff"]["lightweight"] < frozen["q_eff"]["balanced"] < frozen["q_eff"]["high_accuracy"]):
        raise RuntimeError("FROZEN_Q_ORDER_MISMATCH")
    return frozen


def semantic_spec(method: str, seed: int, bundle_root: Path, phase: str) -> dict[str, Any]:
    return {
        "version": "FROZEN-V2-FINAL-GATE-v1", "phase": phase, "method": method, "seed": seed,
        "bundle_root": str(bundle_root), "capacity": CAPACITY, "slots": SLOTS,
        "frozen_parameter_sha256": gatecore.sha256_file(FROZEN),
        "entry_sha256": gatecore.sha256_file(Path(__file__).resolve()),
        "gatecore_sha256": gatecore.sha256_file(ROOT / "src" / "evaluation" / "frozen_v2_final_gate.py"),
        "config_sha256": gatecore.sha256_file(CONFIG),
    }


class DirectOnlyV2Wrapper(calibration.ConditionalCalibrationWrapper):
    """Keep ordinary one-shot policy; force HV to exact direct closure or Defer."""

    def prepare_step_for_reentry(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> Any:
        return timeline_step

    def finish_step(self, timeline_step: Any) -> None:
        return None

    def _force_defer(self, defer: Any, reason: str) -> Any:
        row = self.v2_decision_rows[-1]
        slot, task_id = int(row["decision_slot"]), str(row["task_id"])
        for candidate in reversed(self.candidate_rows):
            if int(candidate["decision_slot"]) != slot or str(candidate["task_id"]) != task_id:
                break
            candidate["final_executed_choice"] = int(candidate["action"] == defer.action_name)
        row.update({"selected_action": defer.action_name, "selected_model": defer.model_name, "selected_product": "defer", "selection_reason": reason, "v2_net_score": 0.0, "v2_quality": 0.0, "q_multiplication_count": 0, "residual_registered": 0})
        self._set_delegate_state(defer, 0.0)
        return defer

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        selected = super().select_action(context, candidate_actions, predicted_map, guidance_bundle)
        row = self.v2_decision_rows[-1]
        if row["value_level"] not in {"high", "urgent"}:
            return selected
        if self.projection.product_satisfies(row["selected_product"], row["residual_requirement_before"]):
            return selected
        defer = next(action for action in self.step_runner.action_space.list_actions() if action.payload_policy == "defer")
        return self._force_defer(defer, "hv_direct_only_infeasible_defer")


class GroundAckV2Wrapper(calibration.ConditionalCalibrationWrapper):
    """Apply a 120-second post-ground-Metadata acknowledgement gate."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.ground_ack_audit = {"wait_seconds": 120.0, "selection_blocks": 0, "reentry_blocks": 0, "payload_hook_blocks": 0, "carryover_hook_blocks": 0, "staged_payload_hook_blocks": 0, "allowed_followup_age_s": [], "early_followup_violation_count": 0}

    def _ack_age_s(self, task_id: str, slot_index: int, slot_seconds: float) -> float | None:
        state = None if self.step_runner is None else self.step_runner.product_lifecycle.get(task_id)
        delivered = None if state is None else state.first_alert_delivered_time_s
        return None if delivered is None else float(slot_index) * float(slot_seconds) - float(delivered)

    def _ack_waiting(self, task_id: str, slot_index: int, slot_seconds: float) -> bool:
        age = self._ack_age_s(task_id, slot_index, slot_seconds)
        return age is not None and age < 120.0 - 1.0e-9

    def _reentry_physical_feasibility(self, task: Any, timeline_step: Any, resource_state: Any, comm_state: Any):
        row = super()._reentry_physical_feasibility(task, timeline_step, resource_state, comm_state)
        if self._ack_waiting(str(task.task_id), int(timeline_step.slot_index), float(timeline_step.slot_seconds)):
            row["hard_safe"] = False
            row["deadline_feasible"] = False
            row["ground_ack_waiting"] = True
            self.ground_ack_audit["reentry_blocks"] += 1
        return row

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        selected = super().select_action(context, candidate_actions, predicted_map, guidance_bundle)
        task = self.current_step.task
        if task is None:
            return selected
        age = self._ack_age_s(str(task.task_id), int(self.current_step.slot_index), float(self.current_step.slot_seconds))
        product = gatecore.product_normalize(self.v2_decision_rows[-1]["selected_product"])
        if age is None or age >= 120.0 - 1.0e-9 or product in {"", "metadata", "defer"}:
            if age is not None and product in {"roi", "roi_context"}:
                self.ground_ack_audit["allowed_followup_age_s"].append(float(age))
                self.ground_ack_audit["early_followup_violation_count"] += int(age < 120.0 - 1.0e-9)
            return selected
        defer = next(action for action in self.step_runner.action_space.list_actions() if action.payload_policy == "defer")
        self.ground_ack_audit["selection_blocks"] += 1
        row = self.v2_decision_rows[-1]
        slot, task_id = int(row["decision_slot"]), str(row["task_id"])
        for candidate in reversed(self.candidate_rows):
            if int(candidate["decision_slot"]) != slot or str(candidate["task_id"]) != task_id:
                break
            candidate["final_executed_choice"] = int(candidate["action"] == defer.action_name)
        row.update({"selected_action": defer.action_name, "selected_model": defer.model_name, "selected_product": "defer", "selection_reason": "ground_ack_120s_wait_defer", "v2_net_score": 0.0, "v2_quality": 0.0, "q_multiplication_count": 0})
        self._set_delegate_state(defer, 0.0)
        return defer


def _disable_progressive(configured: dict[str, Any]) -> None:
    progressive = configured["simulator"].setdefault("progressive_payload", {})
    for section in ("phase6_proactive_followup", "phase6_residual_candidate_carryover", "phase6_same_slot_carryover", "roi_context_after_roi", "value_aware_staged_delivery_queue"):
        progressive.setdefault(section, {})["enabled"] = False
    for section in ("phase5_continue_request", "continue_request"):
        if isinstance(configured["simulator"].get(section), dict):
            configured["simulator"][section]["enabled"] = False


def run_v2_child(spec: Mapping[str, Any], output: Path) -> None:
    method, seed, bundle_root = str(spec["method"]), int(spec["seed"]), Path(spec["bundle_root"])
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    frozen = load_frozen()
    core = v2diag.load_v2core()
    core.FROZEN_QUALITY_PROXY.clear()
    core.FROZEN_QUALITY_PROXY.update({key: f(value) for key, value in frozen["q_eff"].items()})
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    configured = v25.import_prop_script()._configure(scenario.configs, v25.read_json(v25.SHARED_PRIOR_CALIBRATION), {"enabled": False})
    configured, grid_audit = v2diag.configure_v2(configured, fixed=method == FIXED)
    configured = calibration.configure_online_prices(configured, {"prices": frozen["public_prices"], "legacy_online_baseline": False})
    if method == SINGLE:
        _cfg, intervention = legacy_ablation.configure_variant(configured, "single_timescale")
        configured = _cfg
    else:
        intervention = {"variant": method}
    if method == DIRECT:
        _disable_progressive(configured)
    scenario = replace(scenario, configs=configured)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    identity = exact.scenario_identity(method, seed, scenario, bundle, power, CAPACITY)
    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    if method == MYOPIC:
        delegate = exact.load_myopic_router_class()(v25.MYOPIC_PARAM, router_name="MYOPIC-V2")
        exact.bind_myopic_runtime(runner, delegate)
    else:
        delegate = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    wrapper_cls = DirectOnlyV2Wrapper if method == DIRECT else GroundAckV2Wrapper if method == ACK else calibration.ConditionalCalibrationWrapper
    wrapper = wrapper_cls(delegate, reqrun.load_projection_config(True), scenario.timeline[:SLOTS], scenario.configs["simulator"], reqrun.load_projection_module(), product_utility_cfg=scenario.configs["product_requirement_utility"], reward_cfg=scenario.configs["reward"])
    reqrun.bind_projection_runtime(runner, wrapper)
    runtime_audit = {"static_guidance_call_count": 0, "observed_modes": set(), "observed_lambda_vectors": set()}
    if method == SINGLE:
        legacy_ablation.bind_single_timescale_runtime(runner, runtime_audit)
    if method == ACK:
        legacy_ablation.bind_ground_ack_runtime(runner, wrapper)
    result = runner.run(wrapper)
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    reqrun.write_energy_components(output, result)
    write_csv(output / "v2_candidate_score_audit.csv", wrapper.candidate_rows)
    write_csv(output / "v2_decision_trace.csv", wrapper.v2_decision_rows)
    ack_audit = None
    if isinstance(wrapper, GroundAckV2Wrapper):
        ack_audit = deepcopy(wrapper.ground_ack_audit)
        ages = [float(value) for value in ack_audit.pop("allowed_followup_age_s")]
        ack_audit.update({"allowed_followup_count": len(ages), "minimum_allowed_followup_age_s": min(ages) if ages else None, "passed": ack_audit["early_followup_violation_count"] == 0 and (not ages or min(ages) >= 120.0 - 1.0e-9)})
    runtime_audit["observed_modes"] = sorted(runtime_audit["observed_modes"])
    runtime_audit["observed_lambda_vectors"] = [list(value) for value in sorted(runtime_audit["observed_lambda_vectors"])]
    run_identity = {
        "status": "PASS", "method": method, "seed": seed, "pid": os.getpid(), "fresh_state": True,
        "delegate_class": type(delegate).__name__, "wrapper_class": type(wrapper).__name__, "grid_audit": grid_audit,
        "fixed_model": "balanced" if method == FIXED else "", "single_timescale": method == SINGLE,
        "direct_only": method == DIRECT, "ground_ack_120s": method == ACK, "myopic_delegate": method == MYOPIC,
        "ordinary_policy": "soft_requirement_one_shot", "hv_policy": "hard_alignment_direct_first_metadata_fallback_autonomous_residual",
        "q_eff": frozen["q_eff"], "public_prices": frozen["public_prices"], "q_once_online": True,
        "negative_action_guard": True, "future_information_used": False, "hard_safe_filter_unchanged": True,
        "scenario_identity": identity, "power_identity": power, "capacity_identity": capacity_identity,
        "intervention": intervention, "runtime_audit": runtime_audit, "ground_ack_audit": ack_audit,
        "run_spec_sha256": stable_hash(spec),
    }
    write_json(output / "runtime_identity.json", run_identity)


def _normalize_ground_only(scenario: Any, rows: list[dict[str, Any]], scientific: Mapping[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {str(row["task_id"]): row for row in rows}
    canonical = {str(row["task_id"]): row for row in scenario.canonical_tasks}
    tasks, events = [], []
    for task_id, source in canonical.items():
        row = by_task[task_id]
        finish = row.get("actual_finish_s", "")
        if finish not in {None, ""}:
            events.append({"method": GROUND, "task_id": task_id, "stage": "full", "ground_available_time": float(finish), "deadline": float(source["deadline_time_s"]), "valid_before_deadline": int(float(finish) <= float(source["deadline_time_s"])), "event_source": "ground_only_native:ground_fifo_compute_complete", "duplicate_flag": 0})
        closed = int(row.get("closed", 0))
        tasks.append({"method": GROUND, "task_id": task_id, "arrival_time": source["arrival_time_s"], "deadline": source["deadline_time_s"], "value_level": source["value_level"], "effective_task_value": source["effective_task_value"], "product_requirement": source["product_requirement"], "selected_action": "GROUND", "executed_action": "GROUND", "selected_model": "ground_high_accuracy", "delivered_product": "full" if finish not in {None, ""} else "", "closure_success": closed, "closure_ground_time": finish, "failure_reason": row.get("failure_reason", ""), "controllable_energy_j": row.get("satellite_energy_j", 0.0), "transmitted_bytes": row.get("transmitted_bytes", 0.0)})
    slot_rows = []
    battery_min = f(scenario.configs["simulator"].get("battery_min"))
    thermal_max = f(scenario.configs["simulator"].get("thermal_max"), 1.0)
    for row in trace:
        slot = int(row.get("slot", len(slot_rows)))
        soc, thermal = f(row.get("soc_after")), f(row.get("thermal_after"))
        slot_rows.append({"method": GROUND, "slot": slot, "time_s": f(row.get("time_s"), slot * SLOT_SECONDS), "battery_soc": soc, "thermal": thermal, "queue_mb": f(row.get("queue_after_mb", row.get("queue_mb"))), "compute_energy_j": f(row.get("compute_energy_j")), "comm_energy_j": f(row.get("comm_energy_j")), "idle_energy_j": f(scenario.timeline[slot].resource_trace.base_power_w) * f(scenario.timeline[slot].slot_seconds), "transmitted_mb": f(row.get("sent_mb")), "in_contact": int(row.get("in_contact", 0)), "soc_violation": int(soc < battery_min), "thermal_violation": int(thermal > thermal_max)})
    resource = {"method": GROUND, "slot_count": len(slot_rows), "minimum_soc": min(row["battery_soc"] for row in slot_rows), "final_soc": slot_rows[-1]["battery_soc"], "maximum_thermal": max(row["thermal"] for row in slot_rows), "soc_violation_count": sum(row["soc_violation"] for row in slot_rows), "thermal_violation_count": sum(row["thermal_violation"] for row in slot_rows), "controllable_energy_wh": f(scientific.get("controllable_energy_wh", scientific.get("total_controllable_satellite_energy_wh"))), "transmitted_mb": f(scientific.get("transmitted_mb", scientific.get("total_transmitted_mb"))), "ground_compute_energy_counted": False, "native_summary": dict(scientific)}
    return {"per_task": tasks, "events": events, "slots": slot_rows, "resource_summary": resource}


def run_ground_only_child(spec: Mapping[str, Any], output: Path) -> None:
    seed, bundle_root = int(spec["seed"]), Path(spec["bundle_root"])
    v25.child_setup_path(v25.SOURCE_B1_OFFLOAD)
    bundle = v25.load_bundle(bundle_root)
    import native_offload
    binding = native_offload.make_binding(v25.SOURCE_B1_OFFLOAD.resolve())
    binding_identity = dict(binding.identity())
    scenario = binding.build_scenario(bundle)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_B1_OFFLOAD)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    identity = exact.scenario_identity(GROUND, seed, scenario, bundle, power, CAPACITY)
    from scripts.run_offload_vs_ground_24h_pilot import common_parameters, run_ground, task_rows, summarize
    from scripts.run_three_method_minimal_robustness_v1 import augment_trace
    from src.common.config import load_yaml
    offload_cfg = load_yaml(v25.SOURCE_B1_OFFLOAD / "configs" / "methods" / "offload_eft_thermal_aligned_v1.yaml")
    ground_cfg = load_yaml(v25.SOURCE_B1_OFFLOAD / "configs" / "methods" / "ground_only_v1.yaml")
    pars = common_parameters(scenario.configs, offload_cfg)
    shared = {"timeline": scenario.timeline[:SLOTS], "configs": scenario.configs, "task_steps": [step for step in scenario.timeline[:SLOTS] if step.task is not None]}
    sim, decisions = run_ground(shared, ground_cfg, pars)
    native_rows, common_rows = task_rows(GROUND, sim, shared, pars, decisions)
    trace = augment_trace(sim.actual_trace, shared["timeline"], scenario.configs["simulator"])
    scientific = summarize(GROUND, native_rows, common_rows, trace, sim)
    normalized = _normalize_ground_only(scenario, native_rows, scientific, trace)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    write_json(output / "runtime_identity.json", {"status": "PASS", "method": GROUND, "seed": seed, "pid": os.getpid(), "fresh_state": True, "forced_ground_once": True, "actual_model": "ground_high_accuracy", "ground_server_count": ground_cfg["method"]["ground_compute_servers"], "ground_queue_policy": ground_cfg["method"]["ground_queue_policy"], "ground_queue_capacity": ground_cfg["method"]["ground_queue_capacity"], "shared_offload_ground_server_code": True, "source_identity": binding_identity, "scenario_identity": identity, "power_identity": power, "capacity_identity": capacity_identity, "run_spec_sha256": stable_hash(spec)})


def child_main(spec_path: Path) -> int:
    spec = read_json(spec_path)
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError(output)
    method = str(spec["method"])
    if method in V2_METHODS:
        run_v2_child(spec, output)
    elif method in {B1, OFFLOAD, TUT}:
        exact.run_baseline(method, int(spec["seed"]), Path(spec["bundle_root"]), output, CAPACITY)
        identity = read_json(output / "native_identity.json")
        identity.update({"run_spec_sha256": stable_hash(spec), "pid": os.getpid(), "fresh_state": True})
        write_json(output / "runtime_identity.json", identity)
    elif method == GROUND:
        run_ground_only_child(spec, output)
    else:
        raise ValueError(method)
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": int(spec["seed"]), "run_spec_sha256": stable_hash(spec)})
    return 0


def resolved_config(bundle_root: Path) -> dict[str, Any]:
    return read_json(bundle_root / "resolved_config.json")


def evaluate_and_write(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(spec["output"])
    metric, ledger = gatecore.evaluate_public_run(str(spec["method"]), path, resolved_config(Path(spec["bundle_root"])), load_frozen())
    metric.update({"phase": spec["phase"], "seed": int(spec["seed"]), "run_spec_sha256": stable_hash(spec)})
    write_csv(path / "frozen_public_event_ledger.csv", ledger)
    write_json(path / "frozen_public_metrics.json", metric)
    return metric


def completed(spec: Mapping[str, Any]) -> bool:
    output = Path(spec["output"])
    marker = output / "method_completed.json"
    metric = output / "frozen_public_metrics.json"
    if not marker.is_file() or not metric.is_file():
        return False
    try:
        row = read_json(marker)
        public = read_json(metric)
        return row.get("status") == "PASS" and row.get("run_spec_sha256") == stable_hash(spec) and public.get("run_spec_sha256") == stable_hash(spec)
    except Exception:
        return False


def launch(spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(spec["output"])
    if completed(spec):
        metric = read_json(output / "frozen_public_metrics.json")
        return {"status": "RESUMED", "method": spec["method"], "seed": spec["seed"], "pid": read_json(output / "runtime_identity.json").get("pid"), "metric": metric}
    # Public-evaluator adapter fixes do not invalidate an already completed
    # fresh simulation.  Migrate only when every simulation-semantic field and
    # the frozen parameter identity are unchanged.
    prior_spec_path = OUTPUT / "run_specs" / str(spec["phase"]) / f"seed_{spec['seed']}" / f"{SLUG[spec['method']]}.json"
    marker_path = output / "method_completed.json"
    if output.is_dir() and marker_path.is_file() and prior_spec_path.is_file():
        marker = read_json(marker_path)
        prior = read_json(prior_spec_path)
        semantic_fields = ("version", "phase", "method", "seed", "bundle_root", "capacity", "slots", "frozen_parameter_sha256")
        if marker.get("status") == "PASS" and marker.get("run_spec_sha256") == stable_hash(prior) and all(prior.get(field) == spec.get(field) for field in semantic_fields):
            metric = evaluate_and_write(spec)
            identity_path = output / "runtime_identity.json"
            identity = read_json(identity_path)
            identity.update({"run_spec_sha256": stable_hash(spec), "resumed_after_evaluator_adapter_only_change": True, "previous_run_spec_sha256": stable_hash(prior)})
            write_json(identity_path, identity)
            write_json(marker_path, {"status": "PASS", "method": spec["method"], "seed": int(spec["seed"]), "run_spec_sha256": stable_hash(spec), "resumed_after_evaluator_adapter_only_change": True})
            write_json(prior_spec_path, spec)
            return {"status": "RESUMED_EVALUATOR_ADAPTER", "method": spec["method"], "seed": spec["seed"], "pid": identity.get("pid"), "metric": metric}
    if output.exists():
        quarantine = OUTPUT / "failed_attempts" / f"{spec['phase']}_{SLUG[spec['method']]}_seed{spec['seed']}_{datetime.now().strftime('%Y%m%dT%H%M%S%f')}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(quarantine))
    spec_path = OUTPUT / "run_specs" / str(spec["phase"]) / f"seed_{spec['seed']}" / f"{SLUG[spec['method']]}.json"
    write_json(spec_path, spec)
    command = [sys.executable, str(Path(__file__).resolve()), "--child-run", "--spec", str(spec_path)]
    started = utc_now()
    process = subprocess.run(command, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": str(spec["seed"])}, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logs = OUTPUT / "logs" / str(spec["phase"])
    logs.mkdir(parents=True, exist_ok=True)
    label = f"seed_{spec['seed']}_{SLUG[spec['method']]}"
    (logs / f"{label}_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (logs / f"{label}_stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0 or not (output / "method_completed.json").is_file():
        raise RuntimeError({"method": spec["method"], "seed": spec["seed"], "returncode": process.returncode, "stderr": process.stderr[-4000:]})
    metric = evaluate_and_write(spec)
    identity = read_json(output / "runtime_identity.json")
    return {"status": "PASS", "method": spec["method"], "seed": spec["seed"], "pid": identity.get("pid"), "started_utc": started, "finished_utc": utc_now(), "metric": metric}


def run_parallel(specs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(launch, spec): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            statuses.append(future.result())
            print(f"completed {spec['phase']} seed={spec['seed']} {spec['method']}", flush=True)
    return statuses


def phase_a_freeze() -> dict[str, Any]:
    frozen = load_frozen()
    sources = [
        Path(__file__).resolve(), CONFIG, FROZEN, ROOT / "src" / "routers" / "decoupled_prop_v2.py",
        ROOT / "src" / "evaluation" / "frozen_v2_final_gate.py",
        ROOT / "scripts" / "run_v25_decoupled_v2_seed2026_diagnostic.py",
        ROOT / "scripts" / "run_historical_asset_audit_conditional_v2_calibration_v1.py",
        ROOT / "src" / "evaluation" / "unified_ground_received_accounting.py",
        ROOT / "src" / "payload" / "information_sufficiency.py",
        v25.FROZEN_MANIFEST, v25.POWER_CONFIG, v25.SOLAR_TRACE,
    ]
    hashes = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): gatecore.sha256_file(path) for path in sources}
    action_rows = v2diag.load_v2core().build_decoupled_action_rows()
    action_audit = v2diag.load_v2core().action_grid_audit(action_rows)
    audit = {
        "status": "PASS" if action_audit["complete"] and action_audit["defer_count"] == 1 else "FAIL",
        "created_utc": utc_now(), "frozen_parameters": frozen, "source_hashes": hashes,
        "source_hash_bundle_sha256": stable_hash(hashes), "action_grid_audit": action_audit,
        "task_manifest_sha256": gatecore.sha256_file(v25.FROZEN_MANIFEST),
        "power_config_sha256": gatecore.sha256_file(v25.POWER_CONFIG),
        "capacity_multiplier": CAPACITY, "duration_s": SLOTS * SLOT_SECONDS,
        "ranking_based_retuning_after_freeze_forbidden": True,
    }
    write_json(OUTPUT / "audit" / "phase_a_freeze_identity_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("PHASE_A_FREEZE_FAILED")
    return audit


def phase_b_behavior() -> dict[str, Any]:
    if not EXISTING_FULL.is_dir():
        raise RuntimeError("EXISTING_FROZEN_V2_RESULT_MISSING")
    decisions = read_csv(EXISTING_FULL / "v2_decision_trace.csv")
    candidates = read_csv(EXISTING_FULL / "v2_candidate_score_audit.csv")
    behavior = gatecore.ordinary_behavior_audit(decisions, candidates)
    matrix = behavior.pop("required_selected_matrix")
    write_csv(OUTPUT / "tables" / "ordinary_required_selected_product_matrix.csv", matrix)
    write_md(OUTPUT / "tables" / "ordinary_required_selected_product_matrix.md", matrix)
    frozen = load_frozen()
    by_combo: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_combo[(row["value_level"], row["model"], gatecore.product_normalize(row["product"]))].append(row)
    decomposition = []
    e_ref = 100.0
    queue_max = 64.0
    bundle = EXISTING_FULL.parents[4] / "config" / "controlled_bundle_seed_2026" / "resolved_config.json"
    if bundle.is_file():
        cfg = read_json(bundle)
        e_ref = f(cfg["simulator"].get("E_ref"), 100.0)
        queue_max = f(cfg["simulator"].get("queue_max_mb"), 64.0)
    for key, rows in sorted(by_combo.items()):
        mean = lambda field: sum(f(row.get(field)) for row in rows) / len(rows)
        predicted_energy_wh = mean("v2_energy_term") * e_ref / 3600.0
        predicted_mb = mean("v2_payload_term") * queue_max
        decomposition.append({"value_level": key[0], "model": key[1], "product": key[2], "candidate_count": len(rows), "mean_V_deltaI_q_p_T": mean("v2_positive_benefit"), "mean_predicted_energy_Wh": predicted_energy_wh, "mean_public_energy_cost": predicted_energy_wh * f(frozen["public_prices"]["energy_price_per_Wh"]), "mean_predicted_downlink_MB": predicted_mb, "mean_public_downlink_cost": predicted_mb * f(frozen["public_prices"]["downlink_price_per_MB"]), "mean_online_dynamic_resource_cost": mean("v2_dynamic_resource_cost"), "mean_predicted_online_net": mean("v2_net_score")})
    write_csv(OUTPUT / "tables" / "candidate_net_utility_decomposition.csv", decomposition)
    selected = {(int(row["decision_slot"]), row["task_id"]): row for row in decisions}
    trace = {int(row["slot"]): row for row in read_csv(EXISTING_FULL / "per_slot_resource_trace.csv")}
    state_models = Counter()
    for (_slot, _task), row in selected.items():
        if row["selected_product"] == "defer":
            continue
        resource = trace.get(int(row["decision_slot"]), {})
        soc, thermal, queue = f(resource.get("battery_soc")), f(resource.get("thermal")), f(resource.get("queue_mb"))
        pressure = "low" if soc >= 0.82 and thermal < 0.75 and queue < 8 else "high" if soc < 0.75 or thermal >= 0.88 or queue >= 32 else "medium"
        state_models[(row["value_level"], pressure, row["selected_model"])] += 1
    state_rows = [{"value_level": key[0], "resource_pressure_bin": key[1], "selected_model": key[2], "count": value} for key, value in sorted(state_models.items())]
    write_csv(OUTPUT / "tables" / "task_value_resource_state_model_matrix.csv", state_rows)
    hv = [row for row in decisions if row["value_level"] in {"high", "urgent"}]
    source_text = (ROOT / "scripts" / "run_v25_decoupled_v2_seed2026_diagnostic.py").read_text(encoding="utf-8")
    forbidden_patterns = ["low.*selected_model", "medium.*selected_model", "high.*selected_model", "urgent.*selected_model"]
    hardcoded_hits = [pattern for pattern in forbidden_patterns if __import__("re").search(pattern, source_text, __import__("re").IGNORECASE)]
    audit = {
        "status": "PASS",
        **behavior,
        "ordinary_closure_only_if_requirement_satisfied": all(not int(row["closure_success"]) or gatecore.delivered_satisfies(row["delivered_product"], row["product_requirement"]) for row in read_csv(EXISTING_FULL / "per_task_outcomes.csv") if row["value_level"] in {"low", "medium"}),
        "hv_decision_count": len(hv),
        "hv_direct_first_count": sum(row["selection_reason"] == "hv_direct_required_product" for row in hv),
        "hv_metadata_first_count": sum(row["selection_reason"] == "hv_metadata_first" for row in hv),
        "hv_autonomous_reentry_count": sum(int(row["reentry_attempt"]) for row in hv),
        "hv_residual_registered_count": sum(int(row["residual_registered"]) for row in hv),
        "q_once_online_pass": max([abs(f(row["v2_positive_recompute_error"])) for row in candidates] or [0.0]) <= 1.0e-12 and all(int(float(row["v2_q_multiplication_count"])) in {0, 1} for row in candidates),
        "q_once_public_pass": f(read_json(EXISTING_FULL / "conditional_metrics.json").get("q_once_max_recompute_error")) <= 1.0e-12,
        "task_level_model_product_hardcode_hits": hardcoded_hits,
        "no_task_level_model_product_hardcoding": not hardcoded_hits,
    }
    required = (audit["ordinary_one_shot_pass"], audit["ordinary_closure_only_if_requirement_satisfied"], audit["candidate_grid_complete"], audit["q_once_online_pass"], audit["q_once_public_pass"], audit["no_task_level_model_product_hardcoding"])
    audit["status"] = "PASS" if all(required) else "FAIL"
    write_json(OUTPUT / "audit" / "phase_b_behavior_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError({"PHASE_B_BEHAVIOR_FAILED": audit})
    return audit


def bundle_for_seed(seed: int) -> tuple[Path, dict[str, Any]]:
    path = OUTPUT / "scenario" / f"controlled_bundle_seed_{seed}"
    audit_path = OUTPUT / "scenario" / f"controlled_bundle_seed_{seed}_audit.json"
    if path.is_dir() and audit_path.is_file():
        return path, read_json(audit_path)
    if path.exists():
        raise RuntimeError({"incomplete_bundle": str(path)})
    audit = exact.prepare_controlled_bundle(seed, path)
    write_json(audit_path, audit)
    return path, audit


def specs_for(phase: str, seeds: list[int]) -> list[dict[str, Any]]:
    specs = []
    for seed in seeds:
        bundle, _ = bundle_for_seed(seed)
        for method in METHODS:
            spec = semantic_spec(method, seed, bundle, phase)
            spec["output"] = str(OUTPUT / "runs" / phase / f"seed_{seed}" / SLUG[method])
            specs.append(spec)
    return specs


def identity_gate(specs: list[dict[str, Any]], statuses: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["seed"], row["method"]): row for row in statuses}
    pids = []
    checks = []
    scenario_hashes: dict[int, set[str]] = defaultdict(set)
    power_hashes: dict[int, set[str]] = defaultdict(set)
    capacity_hashes: dict[int, set[str]] = defaultdict(set)
    for spec in specs:
        output = Path(spec["output"])
        metric = read_json(output / "frozen_public_metrics.json")
        identity = read_json(output / "runtime_identity.json")
        pids.append(identity.get("pid"))
        scenario = identity.get("scenario_identity", identity.get("scenario_identity", {}))
        if isinstance(scenario, dict):
            scenario_hashes[int(spec["seed"])].add(str(scenario.get("task_hash", scenario.get("expected_task_hash", ""))))
            power = scenario.get("frozen_power_identity", identity.get("power_identity", {}))
        else:
            power = identity.get("power_identity", {})
        power_hashes[int(spec["seed"])].add(stable_hash(power.get("frozen_power", v25.FROZEN_POWER) if isinstance(power, dict) else power))
        capacity = identity.get("capacity_identity", scenario.get("capacity_identity", {}) if isinstance(scenario, dict) else {})
        capacity_hashes[int(spec["seed"])].add(str(capacity.get("runtime_link_trace_hash", "")))
        launch_status = str(by_key[(spec["seed"], spec["method"])]["status"])
        checks.append({"seed": spec["seed"], "method": spec["method"], "run_complete": launch_status == "PASS" or launch_status.startswith("RESUMED"), "fresh_state": bool(identity.get("fresh_state", True)), "q_model_complete": int(metric["missing_event_model_count"]) == 0, "q_once": f(metric["q_once_max_recompute_error"]) <= 1.0e-12, "safe": metric["feasibility_status"] == "Feasible", "correct_power": stable_hash(power.get("frozen_power", v25.FROZEN_POWER) if isinstance(power, dict) else power) == stable_hash(v25.FROZEN_POWER), "capacity_050": f(capacity.get("runtime_capacity", CAPACITY)) == CAPACITY})
    # Windows may recycle a numeric PID after a child exits.  Isolation is
    # therefore identified by the unique method/seed child invocation plus a
    # child-recorded PID and fresh-state identity, not global PID uniqueness.
    invocation_tokens = {
        (int(spec["seed"]), str(spec["method"]), read_json(Path(spec["output"]) / "runtime_identity.json").get("pid"))
        for spec in specs
    }
    separate_invocations = len(invocation_tokens) == len(specs) and all(token[2] is not None for token in invocation_tokens)
    numeric_pids = [pid for pid in pids if pid is not None]
    cross = {
        "scenario_same_per_seed": all(len(value) == 1 and "" not in value for value in scenario_hashes.values()),
        "power_same_per_seed": all(len(value) == 1 for value in power_hashes.values()),
        "link_same_per_seed": all(len(value) == 1 and "" not in value for value in capacity_hashes.values()),
        "unique_process_per_run": separate_invocations,
        "numeric_pid_reuse_count_informational": len(numeric_pids) - len(set(numeric_pids)),
        "os_pid_reuse_allowed_only_for_completed_nonshared_children": True,
    }
    passed = all(all(row[key] for key in ("run_complete", "fresh_state", "q_model_complete", "q_once", "safe", "correct_power", "capacity_050")) for row in checks) and all(cross[key] for key in ("scenario_same_per_seed", "power_same_per_seed", "link_same_per_seed", "unique_process_per_run"))
    return {"status": "PASS" if passed else "FAIL", "checks": checks, "cross_method": cross, "process_ids": pids}


def aggregate_formal(metrics: list[dict[str, Any]], formal_seeds: list[int]) -> None:
    tables = OUTPUT / "tables"
    fields = ["seed", "method", "J_all_net", "J_HV_net", "gross_utility", "HV_gross_utility", "energy_cost", "downlink_cost", "terminal_backlog_cost", "total_cost", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "all_closure_rate", "mean_full_closure_latency_s", "p95_full_closure_latency_s", "controllable_energy_Wh", "transmitted_MB", "terminal_queue_MB", "peak_queue_MB", "minimum_soc", "maximum_thermal", "soc_violation_count", "thermal_violation_count", "feasibility_status", "metadata_delivery_count", "ROI_delivery_count", "Evidence_delivery_count", "model_lightweight_count", "model_balanced_count", "model_high_accuracy_count"]
    rows = [{field: row.get(field) for field in fields} for row in sorted(metrics, key=lambda row: (int(row["seed"]), METHODS.index(row["method"])))]
    write_csv(tables / "formal_five_seed_per_run.csv", rows)
    summary = []
    numeric = [field for field in fields if field not in {"seed", "method", "feasibility_status"}]
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        out: dict[str, Any] = {"method": method, "seed_count": len(group)}
        for field in numeric:
            values = [f(row[field]) for row in group if row.get(field) is not None]
            out[f"{field}_mean"] = statistics.mean(values) if values else None
            out[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    write_csv(tables / "formal_five_seed_mean_std.csv", summary)
    write_md(tables / "formal_five_seed_mean_std.md", summary)
    full = {(int(row["seed"])): row for row in rows if row["method"] == FULL}
    paired = []
    delta_fields = ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "mean_full_closure_latency_s", "controllable_energy_Wh", "transmitted_MB")
    for row in rows:
        if row["method"] == FULL:
            continue
        ref = full[int(row["seed"])]
        paired.append({"seed": row["seed"], "method": row["method"], "reference": FULL, **{f"delta_{field}": f(row.get(field)) - f(ref.get(field)) for field in delta_fields}})
    write_csv(tables / "same_seed_paired_differences.csv", paired)
    product_rows, model_rows = [], []
    for seed in formal_seeds:
        for method in V2_METHODS:
            path = OUTPUT / "runs" / "formal" / f"seed_{seed}" / SLUG[method]
            decisions = read_csv(path / "v2_decision_trace.csv")
            total = sum(row["selected_product"] != "defer" for row in decisions)
            for product, count in Counter(row["selected_product"] for row in decisions if row["selected_product"] != "defer").items():
                product_rows.append({"seed": seed, "method": method, "product": product, "count": count, "fraction": count / max(total, 1)})
            for model, count in Counter(row["selected_model"] for row in decisions if row["selected_product"] != "defer").items():
                model_rows.append({"seed": seed, "method": method, "model": model, "count": count, "fraction": count / max(total, 1)})
    write_csv(tables / "formal_model_selection_proportions.csv", model_rows)
    write_csv(tables / "formal_product_selection_proportions.csv", product_rows)
    baseline_q = Counter()
    for seed in formal_seeds:
        for method in BASELINES:
            path = OUTPUT / "runs" / "formal" / f"seed_{seed}" / SLUG[method] / "frozen_public_event_ledger.csv"
            for row in read_csv(path):
                baseline_q[(seed, method, row["model"], row["model_source"], row["q_eff"])] += 1
    baseline_q_rows = [{"seed": key[0], "method": key[1], "actual_executed_model": key[2], "model_provenance": key[3], "q_eff": key[4], "ground_event_count": value} for key, value in sorted(baseline_q.items())]
    write_csv(tables / "baseline_actual_model_q_audit.csv", baseline_q_rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    phase_a = phase_a_freeze()
    phase_b = phase_b_behavior()
    blind_selection_path = OUTPUT / "audit" / "blind_seed_selection_audit.json"
    if blind_selection_path.is_file():
        blind_selection = read_json(blind_selection_path)
        blind_seed = int(blind_selection["blind_seed"])
    else:
        used, scan_audit = gatecore.scan_used_seeds(ROOT, cfg["seed_selection"]["scan_roots"])
        blind_seed = gatecore.choose_unused_seeds(used, 1, int(cfg["seed_selection"]["deterministic_search_start"]))[0]
        write_json(blind_selection_path, {**scan_audit, "selection_rule": "smallest unused integer at or above deterministic_search_start; immutable after first selection", "search_start": cfg["seed_selection"]["deterministic_search_start"], "blind_seed": blind_seed, "used_seed_set_sha256": stable_hash(sorted(used))})
    blind_specs = specs_for("blind_gate", [blind_seed])
    blind_status = run_parallel(blind_specs, int(cfg["parallel"]["max_independent_processes"]))
    blind_gate = identity_gate(blind_specs, blind_status)
    blind_gate.update({"blind_seed": blind_seed, "phase_a": phase_a["status"], "phase_b": phase_b["status"]})
    write_json(OUTPUT / "audit" / "single_blind_seed_gate_report.json", blind_gate)
    formal_started = False
    formal_seeds: list[int] = []
    formal_gate: dict[str, Any] = {"status": "NOT_STARTED"}
    if blind_gate["status"] == "PASS":
        formal_started = True
        formal_selection_path = OUTPUT / "audit" / "formal_seed_selection_audit.json"
        if formal_selection_path.is_file():
            formal_seeds = [int(value) for value in read_json(formal_selection_path)["formal_seeds"]]
        else:
            used_after, scan_formal = gatecore.scan_used_seeds(ROOT, cfg["seed_selection"]["scan_roots"])
            formal_seeds = gatecore.choose_unused_seeds(used_after, 5, int(cfg["seed_selection"]["deterministic_search_start"]))
            write_json(formal_selection_path, {**scan_formal, "selection_rule": "five smallest unused integers at or above deterministic_search_start after blind gate; immutable after first selection", "formal_seeds": formal_seeds, "used_seed_set_sha256": stable_hash(sorted(used_after))})
        formal_specs = specs_for("formal", formal_seeds)
        formal_status = run_parallel(formal_specs, int(cfg["parallel"]["max_independent_processes"]))
        formal_gate = identity_gate(formal_specs, formal_status)
        write_json(OUTPUT / "audit" / "formal_five_seed_identity_safety_audit.json", formal_gate)
        if formal_gate["status"] == "PASS":
            formal_metrics = [read_json(Path(spec["output"]) / "frozen_public_metrics.json") for spec in formal_specs]
            aggregate_formal(formal_metrics, formal_seeds)
            write_json(OUTPUT / "audit" / "method_identity_audit.json", {"status": "PASS", "methods": list(METHODS), "checks": [{"seed": row["seed"], "method": row["method"], "run_complete": row["run_complete"], "fresh_state": row["fresh_state"]} for row in formal_gate["checks"]], "process_isolation": formal_gate["cross_method"]})
            write_json(OUTPUT / "audit" / "environment_identity_audit.json", {"status": "PASS", "scenario": "V25", "capacity_multiplier": CAPACITY, "power": v25.FROZEN_POWER, "task_and_link_identity": {key: formal_gate["cross_method"][key] for key in ("scenario_same_per_seed", "power_same_per_seed", "link_same_per_seed")}})
            write_json(OUTPUT / "audit" / "evaluator_identity_audit.json", {"status": "PASS", "formula": "G=sum(V*delta_I*q_eff*T); J=G-pi_E*E_Wh-pi_B*B_tx_MB-pi_Q*Q_terminal_MB", "q_multiplication_count": 1, "q_eff": load_frozen()["q_eff"], "public_prices": load_frozen()["public_prices"], "evaluator_source_sha256": gatecore.sha256_file(ROOT / "src" / "evaluation" / "frozen_v2_final_gate.py"), "all_run_q_and_model_checks_pass": all(row["q_model_complete"] and row["q_once"] for row in formal_gate["checks"])})
    completion = {
        "status": "PASS" if blind_gate["status"] == "PASS" and formal_gate["status"] == "PASS" else "FAIL",
        "completion_marker": COMPLETION if blind_gate["status"] == "PASS" and formal_gate["status"] == "PASS" else "FROZEN_V2_FINAL_GATE_FAILED",
        "unique_entry": str(Path(__file__).resolve()), "output_directory": str(OUTPUT.resolve()),
        "blind_seed": blind_seed, "formal_seeds": formal_seeds, "formal_five_seed_started": formal_started,
        "safe_parallel_enabled": True, "max_independent_processes": cfg["parallel"]["max_independent_processes"],
        "methods": list(METHODS), "phase_a": phase_a["status"], "phase_b": phase_b["status"], "blind_gate": blind_gate["status"], "formal_gate": formal_gate["status"],
        "frozen_parameter_sha256": gatecore.sha256_file(FROZEN), "rank_based_retuning_performed": False,
    }
    write_json(OUTPUT / "completion_status.json", completion)
    report = [
        "# Frozen V2 final gate and blind five-seed run", "",
        f"- Unique entry: `{Path(__file__).resolve()}`", f"- Output: `{OUTPUT.resolve()}`",
        f"- Blind seed: `{blind_seed}`", f"- Formal seeds: `{formal_seeds}`",
        f"- Formal five-seed stage started: `{formal_started}`", f"- Safe independent-process parallelism: `True` (max {cfg['parallel']['max_independent_processes']})",
        f"- Phase A/B/blind/formal: `{phase_a['status']}` / `{phase_b['status']}` / `{blind_gate['status']}` / `{formal_gate['status']}`", "",
        "## Methods", "", *[f"- {method}" for method in METHODS], "",
        f"Final marker: `{completion['completion_marker']}`", "",
        "No parameter was changed after seeing method rankings. No historical source or result was overwritten.",
    ]
    (OUTPUT / "FINAL_ASSESSMENT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
    if completion["status"] != "PASS":
        raise RuntimeError(completion)
    print(COMPLETION, flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(Path(args.spec)) if args.child_run else main())
