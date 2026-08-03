"""Frozen V25/capacity=0.50 five-method PROP ablation experiment.

Run this file without arguments from the IDE.  The parent process performs a
frozen Full-PROP regression gate, then launches every remaining run in an
independent Python process.  Completed runs are resumable and are never
overwritten; incomplete attempts are quarantined before a fresh retry.

Internal command-line flags exist only for isolated child processes and the
self-test invoked by this entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402


SEEDS = (2026, 2027, 2028, 2029, 2030)
CAPACITY = 0.50
SLOTS = 4320
SLOT_SECONDS = 20
GROUND_ACK_SECONDS = 120.0
GROUND_ACK_SLOTS = 6
MAX_PARALLEL_RUNS = 5

METHOD_FULL = reqrun.METHOD_PROP_NEW
METHOD_FIXED = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-FIXED-YOLO-B-v1"
METHOD_SINGLE = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-SINGLE-TIMESCALE-v1"
METHOD_DIRECT = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-DIRECT-ONLY-v1"
METHOD_ACK = "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-GROUND-ACK-120S-v1"
METHODS = (METHOD_FULL, METHOD_FIXED, METHOD_SINGLE, METHOD_DIRECT, METHOD_ACK)

VARIANT_BY_METHOD = {
    METHOD_FULL: "full",
    METHOD_FIXED: "fixed_yolo_b",
    METHOD_SINGLE: "single_timescale",
    METHOD_DIRECT: "direct_only",
    METHOD_ACK: "ground_ack_120s",
}

OUTPUT = ROOT / "outputs" / "_v25_prop_ablation_five_method_fiveseed_cap050_v1"
REFERENCE_FULL = ROOT / "outputs" / "_v25_exact_native_five_method_controlled_fiveseed_cap050_v1"

SOURCE_GUARD_FILES = (
    ROOT / "scripts" / "run_prop_reqalign_seed2026_v1.py",
    ROOT / "src" / "routers" / "requirement_aware_projection.py",
    ROOT / "src" / "routers" / "safe_linucb_router.py",
    ROOT / "src" / "routers" / "safe_filter.py",
    ROOT / "src" / "controllers" / "lyapunov_pricing.py",
    ROOT / "src" / "controllers" / "mode_controller.py",
    ROOT / "src" / "simulator" / "step_runner.py",
    ROOT / "configs" / "prop_requirement_projection_v1.yaml",
    ROOT / "configs" / "bupt1_measured_power_v1.yaml",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized or [{"empty": ""}])


def write_md(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(materialized[0]) if materialized else ["empty"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in materialized:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def as_int(value: Any) -> int:
    return int(Decimal(str(value or 0)))


def as_decimal(value: Any) -> Decimal:
    return Decimal(str(value or 0))


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in SOURCE_GUARD_FILES}


def _progressive_cfg(configs: dict[str, Any]) -> dict[str, Any]:
    simulator = configs.setdefault("simulator", {})
    return simulator.setdefault("progressive_payload", {})


def configure_variant(configs: dict[str, Any], variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an isolated runtime config and a machine-checkable intervention audit."""

    cfg = deepcopy(configs)
    intervention: dict[str, Any] = {
        "variant": variant,
        "capacity_multiplier": CAPACITY,
        "slot_seconds": SLOT_SECONDS,
        "full_horizon_slots": SLOTS,
        "fixed_model_alias": "YOLO-B" if variant == "fixed_yolo_b" else "",
        "fixed_model_registry_name": "balanced" if variant == "fixed_yolo_b" else "",
        "fixed_model_profile_name": "small" if variant == "fixed_yolo_b" else "",
    }

    if variant == "fixed_yolo_b":
        changed = []
        for item in cfg["action_space"]["actions"]:
            if str(item.get("model_name")) == "delay_store" or str(item.get("payload_policy")) == "defer":
                continue
            changed.append(
                {
                    "action_name": item["action_name"],
                    "payload_policy": item["payload_policy"],
                    "model_before": item["model_name"],
                    "model_after": "balanced",
                    "payload_level_before": item["expected_payload_level"],
                }
            )
            item["model_name"] = "balanced"
            item["expected_compute_level"] = 0.55
        intervention["fixed_model_action_map"] = changed

    elif variant == "single_timescale":
        # The fast SafeLinUCB scorer, online update, requirement projection and
        # hard safety filter remain active.  A runtime hook supplies neutral,
        # zero-price guidance and prevents all long-horizon state modulation.
        simulator = cfg["simulator"]
        policy_repair = simulator.setdefault("policy_repair", {})
        policy_repair["enable_z_guided_coupling"] = False
        soc_band = simulator.setdefault("soc_band_policy", {})
        soc_band["enabled"] = False
        long_risk = simulator.setdefault("long_term_risk_penalty", {})
        long_risk["enabled"] = False
        intervention.update(
            {
                "long_term_price_runtime_override": "all_zero",
                "mode_runtime_override": "single_timescale_neutral",
                "z_guided_coupling_enabled": False,
                "soc_band_soft_modulation_enabled": False,
                "long_term_soft_risk_penalty_enabled": False,
                "hard_safety_unchanged": True,
            }
        )

    elif variant == "direct_only":
        for item in cfg["action_space"]["actions"]:
            if str(item.get("payload_policy")) in {"metadata_then_roi", "metadata_then_roi_context"}:
                item["enabled"] = False
        progressive = _progressive_cfg(cfg)
        for section in (
            "phase6_proactive_followup",
            "phase6_residual_candidate_carryover",
            "phase6_same_slot_carryover",
            "roi_context_after_roi",
            "value_aware_staged_delivery_queue",
        ):
            progressive.setdefault(section, {})["enabled"] = False
        simulator = cfg["simulator"]
        for section in ("phase5_continue_request", "continue_request"):
            if isinstance(simulator.get(section), dict):
                simulator[section]["enabled"] = False
        intervention.update(
            {
                "metadata_first_actions_enabled": False,
                "residual_reentry_enabled": False,
                "progressive_followup_enabled": False,
                "direct_requirement_or_defer": True,
            }
        )

    elif variant == "ground_ack_120s":
        # Same-slot residual generation is incompatible with a 120 s ground
        # acknowledgement.  Other progressive paths stay enabled and are
        # gated by the runtime acknowledgement hook.
        progressive = _progressive_cfg(cfg)
        progressive.setdefault("phase6_same_slot_carryover", {})["enabled"] = False
        intervention.update(
            {
                "ground_ack_wait_s": GROUND_ACK_SECONDS,
                "ground_ack_wait_slots": GROUND_ACK_SLOTS,
                "wait_origin": "metadata_ground_delivery_event_time",
                "same_slot_followup_enabled": False,
                "proactive_followup_enabled": bool(
                    progressive.get("phase6_proactive_followup", {}).get("enabled", False)
                ),
                "carryover_enabled": bool(
                    progressive.get("phase6_residual_candidate_carryover", {}).get("enabled", False)
                ),
            }
        )

    elif variant != "full":
        raise ValueError(variant)

    action_rows = [item for item in cfg["action_space"]["actions"] if bool(item.get("enabled", True))]
    intervention["enabled_action_count"] = len(action_rows)
    intervention["enabled_actions"] = [
        {
            "action_name": item["action_name"],
            "model_name": item["model_name"],
            "payload_policy": item["payload_policy"],
            "payload_plan": item.get("payload_plan", item["payload_policy"]),
            "expected_payload_level": item["expected_payload_level"],
            "expected_compute_level": item["expected_compute_level"],
        }
        for item in action_rows
    ]
    return cfg, intervention


def _defer_action(candidate_actions: list[Any], step_runner: Any) -> Any:
    for action in candidate_actions:
        if action.payload_policy == "defer" or action.model_name == "delay_store":
            return action
    for action in step_runner.action_space.list_actions():
        if action.payload_policy == "defer" or action.model_name == "delay_store":
            return action
    raise RuntimeError("direct/ack ablation requires a defer action")


def _rewrite_last_projection_row(wrapper: Any, final: Any, predicted_map: dict[str, Any], reason: str) -> None:
    if not wrapper.rows:
        return
    row = wrapper.rows[-1]
    predicted = predicted_map[final.action_name]
    row.update(
        {
            "projection_reason": reason,
            "final_model": final.model_name,
            "final_product": str(predicted.predicted_product_type),
            "final_score": wrapper.last_all_score_components.get(final.action_name, {}).get("final_action_score", ""),
            "final_action": final.action_name,
            "direct_closure_feasible": 0,
            "reenter_future_decisions": 0,
            "one_shot_delivery": 1,
        }
    )


class DirectOnlyWrapper(reqrun.RequirementAwareRouterWrapper):
    """REQALIGN wrapper that allows only a direct satisfying action or Defer."""

    def prepare_step_for_reentry(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> Any:
        return timeline_step

    def finish_step(self, timeline_step: Any) -> None:
        return None

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        final = super().select_action(context, candidate_actions, predicted_map, guidance_bundle)
        task = self.current_step.task
        product = str(predicted_map[final.action_name].predicted_product_type)
        if task is None or self.projection.product_satisfies(product, str(task.product_requirement)):
            return final
        defer = _defer_action(candidate_actions, self.step_runner)
        self.last_selected_action_score = self.last_all_score_components.get(defer.action_name, {}).get(
            "final_action_score", float(predicted_map[defer.action_name].predicted_reward)
        )
        self.last_score_components = deepcopy(self.last_all_score_components.get(defer.action_name, {}))
        self.delegate.last_selected_action_score = self.last_selected_action_score
        self.delegate.last_score_components = deepcopy(self.last_score_components)
        _rewrite_last_projection_row(self, defer, predicted_map, "direct_only_infeasible_defer")
        return defer


class GroundAck120Wrapper(reqrun.RequirementAwareRouterWrapper):
    """Block any post-Metadata product until 120 s after ground receipt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ground_ack_audit = {
            "wait_seconds": GROUND_ACK_SECONDS,
            "selection_blocks": 0,
            "reentry_blocks": 0,
            "payload_hook_blocks": 0,
            "carryover_hook_blocks": 0,
            "staged_payload_hook_blocks": 0,
            "allowed_followup_age_s": [],
            "early_followup_violation_count": 0,
        }

    def _ack_age_s(self, task_id: str, slot_index: int, slot_seconds: float) -> float | None:
        if self.step_runner is None:
            return None
        state = self.step_runner.product_lifecycle.get(task_id)
        delivered = None if state is None else state.first_alert_delivered_time_s
        if delivered is None:
            return None
        return float(slot_index) * float(slot_seconds) - float(delivered)

    def _ack_waiting(self, task_id: str, slot_index: int, slot_seconds: float) -> bool:
        age = self._ack_age_s(task_id, slot_index, slot_seconds)
        return age is not None and age < GROUND_ACK_SECONDS - 1.0e-9

    def _reentry_physical_feasibility(self, task: Any, timeline_step: Any, resource_state: Any, comm_state: Any):
        row = super()._reentry_physical_feasibility(task, timeline_step, resource_state, comm_state)
        if self._ack_waiting(str(task.task_id), int(timeline_step.slot_index), float(timeline_step.slot_seconds)):
            row["hard_safe"] = False
            row["deadline_feasible"] = False
            row["ground_ack_waiting"] = True
            self.ground_ack_audit["reentry_blocks"] += 1
        return row

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        final = super().select_action(context, candidate_actions, predicted_map, guidance_bundle)
        task = self.current_step.task
        if task is None:
            return final
        age = self._ack_age_s(str(task.task_id), int(self.current_step.slot_index), float(self.current_step.slot_seconds))
        product = self.projection.normalize_product(str(predicted_map[final.action_name].predicted_product_type))
        if age is None or age >= GROUND_ACK_SECONDS - 1.0e-9 or product in {"", "metadata", "defer"}:
            if age is not None and product in {"roi", "roi_context"}:
                self.ground_ack_audit["allowed_followup_age_s"].append(float(age))
                if age < GROUND_ACK_SECONDS - 1.0e-9:
                    self.ground_ack_audit["early_followup_violation_count"] += 1
            return final
        defer = _defer_action(candidate_actions, self.step_runner)
        self.ground_ack_audit["selection_blocks"] += 1
        self.last_selected_action_score = self.last_all_score_components.get(defer.action_name, {}).get(
            "final_action_score", float(predicted_map[defer.action_name].predicted_reward)
        )
        self.last_score_components = deepcopy(self.last_all_score_components.get(defer.action_name, {}))
        self.delegate.last_selected_action_score = self.last_selected_action_score
        self.delegate.last_score_components = deepcopy(self.last_score_components)
        _rewrite_last_projection_row(self, defer, predicted_map, "ground_ack_120s_wait_defer")
        return defer


def _neutral_single_timescale_guidance() -> dict[str, Any]:
    zeros = {name: 0.0 for name in ("lambda_E", "lambda_L", "lambda_Q", "lambda_S")}
    return {
        "mode_name": "single_timescale_neutral",
        "lambda_base": dict(zeros),
        "lambda_E_base": 0.0,
        "lambda_L_base": 0.0,
        "lambda_Q_base": 0.0,
        "lambda_S_base": 0.0,
        **zeros,
        "price_multipliers": {name: 1.0 for name in zeros},
        "exploration_scale": 1.0,
        "model_complexity_bias": 0.0,
        "payload_size_bias": 0.0,
        "high_value_bonus_scale": 1.0,
        "safety_margin_scale": 1.0,
        "c_k": {"mode_parameters": "disabled_by_single_timescale_ablation"},
        "z_k": dict(zeros),
    }


def bind_single_timescale_runtime(runner: Any, runtime_audit: dict[str, Any]) -> None:
    previous_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = previous_factory(queue)

        def static_guidance(self, slot_index, long_term_state, router):
            runtime_audit["static_guidance_call_count"] += 1
            runtime_audit["observed_modes"].add("single_timescale_neutral")
            runtime_audit["observed_lambda_vectors"].add((0.0, 0.0, 0.0, 0.0))
            return _neutral_single_timescale_guidance()

        step_runner._build_guidance_bundle = types.MethodType(static_guidance, step_runner)
        return step_runner

    runner.step_runner_factory = factory


def bind_ground_ack_runtime(runner: Any, wrapper: GroundAck120Wrapper) -> None:
    previous_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = previous_factory(queue)

        original_current = step_runner._maybe_build_residual_payload

        def delayed_current(self, *args, **kwargs):
            timeline_step = kwargs.get("timeline_step", args[0] if args else None)
            payload = kwargs.get("payload", args[4] if len(args) > 4 else None)
            state = kwargs.get("lifecycle_state", args[5] if len(args) > 5 else None)
            if timeline_step is not None and state is not None and state.first_alert_delivered_time_s is not None:
                age = float(timeline_step.slot_index) * float(timeline_step.slot_seconds) - float(state.first_alert_delivered_time_s)
                if age < GROUND_ACK_SECONDS - 1.0e-9:
                    wrapper.ground_ack_audit["payload_hook_blocks"] += 1
                    return payload
            result = original_current(*args, **kwargs)
            if result is not payload and state is not None and state.first_alert_delivered_time_s is not None:
                age = float(timeline_step.slot_index) * float(timeline_step.slot_seconds) - float(state.first_alert_delivered_time_s)
                wrapper.ground_ack_audit["allowed_followup_age_s"].append(float(age))
                if age < GROUND_ACK_SECONDS - 1.0e-9:
                    wrapper.ground_ack_audit["early_followup_violation_count"] += 1
            return result

        step_runner._maybe_build_residual_payload = types.MethodType(delayed_current, step_runner)

        original_carryover = step_runner._evaluate_and_build_carryover_payload

        def delayed_carryover(self, *args, **kwargs):
            candidate = kwargs.get("candidate", args[1] if len(args) > 1 else None)
            timeline_step = kwargs.get("timeline_step", args[2] if len(args) > 2 else None)
            delivered = None if candidate is None else candidate.metadata_delivery_time_s
            if delivered is not None and timeline_step is not None:
                age = float(timeline_step.slot_index) * float(timeline_step.slot_seconds) - float(delivered)
                if age < GROUND_ACK_SECONDS - 1.0e-9:
                    wrapper.ground_ack_audit["carryover_hook_blocks"] += 1
                    return None, candidate.task_snapshot, None
            result = original_carryover(*args, **kwargs)
            if result[0] is not None and delivered is not None:
                age = float(timeline_step.slot_index) * float(timeline_step.slot_seconds) - float(delivered)
                wrapper.ground_ack_audit["allowed_followup_age_s"].append(float(age))
                if age < GROUND_ACK_SECONDS - 1.0e-9:
                    wrapper.ground_ack_audit["early_followup_violation_count"] += 1
            return result

        step_runner._evaluate_and_build_carryover_payload = types.MethodType(delayed_carryover, step_runner)

        original_staged = step_runner._staged_request_to_payload

        def delayed_staged(self, request, timeline_step):
            state = self.product_lifecycle.get(request.task_id)
            delivered = None if state is None else state.first_alert_delivered_time_s
            if delivered is not None:
                age = float(timeline_step.slot_index) * float(timeline_step.slot_seconds) - float(delivered)
                if age < GROUND_ACK_SECONDS - 1.0e-9:
                    wrapper.ground_ack_audit["staged_payload_hook_blocks"] += 1
                    return None
            result = original_staged(request, timeline_step)
            if result is not None and delivered is not None:
                age = float(timeline_step.slot_index) * float(timeline_step.slot_seconds) - float(delivered)
                wrapper.ground_ack_audit["allowed_followup_age_s"].append(float(age))
                if age < GROUND_ACK_SECONDS - 1.0e-9:
                    wrapper.ground_ack_audit["early_followup_violation_count"] += 1
            return result

        step_runner._staged_request_to_payload = types.MethodType(delayed_staged, step_runner)
        return step_runner

    runner.step_runner_factory = factory


def run_child(method: str, seed: int, bundle_root: Path, output: Path) -> None:
    variant = VARIANT_BY_METHOD[method]
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    if output.exists():
        raise FileExistsError(output)
    prop_mod = v25.import_prop_script()
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configured = prop_mod._configure(scenario.configs, calibration, {"enabled": False})
    configured, intervention = configure_variant(configured, variant)
    scenario = replace(scenario, configs=configured)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    identity = exact.scenario_identity(method, seed, scenario, bundle, power, CAPACITY)
    identity["capacity_identity"] = capacity_identity

    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    delegate = build_routers(
        scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"])
    )[0]
    projection_config = reqrun.load_projection_config(True)
    wrapper_cls: type[reqrun.RequirementAwareRouterWrapper]
    if variant == "direct_only":
        wrapper_cls = DirectOnlyWrapper
    elif variant == "ground_ack_120s":
        wrapper_cls = GroundAck120Wrapper
    else:
        wrapper_cls = reqrun.RequirementAwareRouterWrapper
    wrapper = wrapper_cls(
        delegate,
        projection_config,
        scenario.timeline[:SLOTS],
        scenario.configs["simulator"],
        reqrun.load_projection_module(),
    )
    reqrun.bind_projection_runtime(runner, wrapper)

    runtime_audit: dict[str, Any] = {
        "variant": variant,
        "method": method,
        "seed": seed,
        "pid": os.getpid(),
        "fresh_state": True,
        "static_guidance_call_count": 0,
        "observed_modes": set(),
        "observed_lambda_vectors": set(),
    }
    if variant == "single_timescale":
        bind_single_timescale_runtime(runner, runtime_audit)
    if variant == "ground_ack_120s":
        bind_ground_ack_runtime(runner, wrapper)

    result = runner.run(wrapper)
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    reqrun.enrich_projection_rows(wrapper, output)
    reqrun.write_energy_components(output, result)

    runtime_audit["observed_modes"] = sorted(runtime_audit["observed_modes"])
    runtime_audit["observed_lambda_vectors"] = [list(item) for item in sorted(runtime_audit["observed_lambda_vectors"])]
    runtime_audit["delegate_class"] = type(delegate).__name__
    runtime_audit["online_update_count"] = getattr(delegate, "online_update_count", "not_exposed")
    if isinstance(wrapper, GroundAck120Wrapper):
        ack = dict(wrapper.ground_ack_audit)
        ages = [float(value) for value in ack.pop("allowed_followup_age_s")]
        ack["allowed_followup_count"] = len(ages)
        ack["minimum_allowed_followup_age_s"] = min(ages) if ages else None
        ack["maximum_allowed_followup_age_s"] = max(ages) if ages else None
        ack["passed"] = int(ack["early_followup_violation_count"]) == 0 and (
            not ages or min(ages) >= GROUND_ACK_SECONDS - 1.0e-9
        )
        runtime_audit["ground_ack"] = ack

    enabled_actions = intervention["enabled_actions"]
    model_check = True
    if variant == "fixed_yolo_b":
        model_check = all(
            row["model_name"] == "balanced"
            for row in enabled_actions
            if row["payload_policy"] != "defer"
        )
    runtime_audit["variant_identity_passed"] = bool(
        model_check
        and (variant != "single_timescale" or runtime_audit["static_guidance_call_count"] == SLOTS)
        and (
            variant != "ground_ack_120s"
            or runtime_audit.get("ground_ack", {}).get("passed", False)
        )
    )
    if not runtime_audit["variant_identity_passed"]:
        raise RuntimeError({"variant_identity_failed": runtime_audit})

    native_identity = {
        "method": method,
        "variant": variant,
        "source_root": str(v25.SOURCE_PROP.resolve()),
        "source_identity_passed": True,
        "corrected_reqalign_enabled": True,
        "delegate": type(delegate).__name__,
        "scenario_identity": identity,
        "intervention": intervention,
        "runtime_audit": runtime_audit,
    }
    write_json(output / "native_identity.json", native_identity)
    write_json(output / "reconstructed_scenario_identity.json", identity)
    write_json(output / "ablation_runtime_audit.json", runtime_audit)
    write_json(
        output / "method_run_status.json",
        {
            "status": "PASS",
            "method": method,
            "variant": variant,
            "seed": seed,
            "pid": os.getpid(),
            "simulation_started": True,
            "fresh_state": True,
            "scenario_reconstruction_count": 1,
        },
    )
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": seed})


def completed_run(path: Path, method: str, seed: int) -> bool:
    marker = path / "method_completed.json"
    if not marker.is_file():
        return False
    try:
        row = read_json(marker)
    except Exception:
        return False
    return row.get("status") == "PASS" and row.get("method") == method and int(row.get("seed", -1)) == seed


def quarantine_incomplete(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    target = OUTPUT / "failed_attempts" / f"{path.parent.name}_{path.name}_{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


def launch_child(method: str, seed: int, bundle_root: Path, path: Path) -> dict[str, Any]:
    if completed_run(path, method, seed):
        status = read_json(path / "method_run_status.json")
        return {
            "seed": seed,
            "method": method,
            "status": "RESUMED_EXISTING",
            "returncode": 0,
            "pid": status.get("pid"),
            "started_utc": None,
            "finished_utc": None,
            "output": str(path),
        }
    quarantined = quarantine_incomplete(path)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--method",
        method,
        "--seed",
        str(seed),
        "--bundle-root",
        str(bundle_root),
        "--output",
        str(path),
    ]
    started = utc_now()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    finished = utc_now()
    safe = method.replace("/", "_")
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"seed_{seed}_{safe}_stdout.txt").write_text(stdout, encoding="utf-8")
    (logs / f"seed_{seed}_{safe}_stderr.txt").write_text(stderr, encoding="utf-8")
    status = {
        "seed": seed,
        "method": method,
        "status": "PASS" if process.returncode == 0 else "FAILED",
        "returncode": process.returncode,
        "pid": process.pid,
        "started_utc": started,
        "finished_utc": finished,
        "output": str(path),
        "quarantined_previous_attempt": "" if quarantined is None else str(quarantined),
        "stdout_tail": stdout[-1500:],
        "stderr_tail": stderr[-5000:],
    }
    if process.returncode != 0 or not completed_run(path, method, seed):
        raise RuntimeError({"ablation_run_failed": status})
    return status


def prepare_bundles() -> tuple[dict[int, Path], list[dict[str, Any]]]:
    bundles: dict[int, Path] = {}
    audits: list[dict[str, Any]] = []
    for seed in SEEDS:
        destination = OUTPUT / "config" / "controlled_bundles" / f"seed_{seed}"
        bundles[seed] = destination
        bundle = None
        invalid_bundle_quarantine = ""
        if destination.exists():
            try:
                bundle = v25.load_bundle(destination)
            except Exception as exc:
                stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                quarantine = OUTPUT / "failed_attempts" / f"invalid_bundle_seed_{seed}_{stamp}"
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(quarantine))
                invalid_bundle_quarantine = f"{quarantine} ({type(exc).__name__}: {exc})"
        if bundle is not None:
            audits.append(
                {
                    "seed": seed,
                    "resume_existing_bundle": True,
                    "task_hash": bundle.contract["task_hash"],
                    "environment_hash": bundle.contract["environment_hash"],
                    "task_count": len(bundle.tasks),
                    "checks": {
                        "task_count_is_268": len(bundle.tasks) == 268,
                        "value_counts_match": bundle.contract["value_level_counts"] == exact.EXPECTED_COUNTS,
                    },
                }
            )
        else:
            audit = exact.prepare_controlled_bundle(seed, destination)
            audit["invalid_bundle_quarantine"] = invalid_bundle_quarantine
            audits.append(audit)
    if not all(all(item.get("checks", {}).values()) for item in audits):
        raise RuntimeError({"controlled_bundle_audit_failed": audits})
    return bundles, audits


METRIC_FIELDS = (
    "J_all_net",
    "J_HV_net",
    "N_HV_closure",
    "N_high_closure",
    "N_urgent_closure",
    "N_HV_metadata",
    "N_HV_ROI",
    "N_HV_evidence",
    "all_task_closure_count",
    "all_task_closure_rate",
    "successful_closure_mean_latency_s",
    "successful_closure_p95_latency_s",
    "HV_successful_closure_mean_latency_s",
    "HV_successful_closure_p95_latency_s",
    "urgent_successful_closure_mean_latency_s",
    "urgent_successful_closure_p95_latency_s",
    "deadline_censored_mean_latency_s",
    "controllable_energy_Wh",
    "transmitted_bytes",
    "transmitted_MB",
    "peak_queue_MB",
    "mean_queue_MB",
    "minimum_soc",
    "maximum_thermal",
    "SOC_violations",
    "thermal_violations",
)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "seed_count": len(group)}
        for field in METRIC_FIELDS:
            values = [float(row[field]) for row in group]
            item[f"{field}_mean"] = statistics.fmean(values)
            item[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(row["seed"]), row["method"]): row for row in rows}
    output = []
    for seed in SEEDS:
        full = by_key[(seed, METHOD_FULL)]
        for method in METHODS[1:]:
            variant = by_key[(seed, method)]
            item: dict[str, Any] = {
                "seed": seed,
                "comparison": f"{method} minus {METHOD_FULL}",
                "ablation_method": method,
                "reference_method": METHOD_FULL,
            }
            for field in METRIC_FIELDS:
                item[f"delta_{field}"] = as_decimal(variant[field]) - as_decimal(full[field])
            output.append(item)
    return output


def paired_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS[1:]:
        group = [row for row in pairs if row["ablation_method"] == method]
        item: dict[str, Any] = {
            "ablation_method": method,
            "reference_method": METHOD_FULL,
            "seed_count": len(group),
        }
        for field in METRIC_FIELDS:
            values = [float(row[f"delta_{field}"]) for row in group]
            item[f"mean_delta_{field}"] = statistics.fmean(values)
            item[f"std_delta_{field}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def collect_metrics(bundles: dict[int, Path]) -> list[dict[str, Any]]:
    eval_mod = v25.load_eval_module(OUTPUT)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method in METHODS:
            path = OUTPUT / "runs" / f"seed_{seed}" / method
            metrics = exact.evaluate_method(method, path, bundles[seed], eval_mod, CAPACITY)
            if not metrics["public_accounting_passed"]:
                raise RuntimeError({"PUBLIC_ACCOUNTING_MISMATCH": seed, "method": method})
            if as_int(metrics["soc_violation_count"]) or as_int(metrics["thermal_violation_count"]):
                raise RuntimeError({"HARD_SAFETY_VIOLATION": seed, "method": method})
            rows.append(exact.selected_metrics(seed, metrics))
    return rows


def emit_tables(rows: list[dict[str, Any]]) -> None:
    order = {method: index for index, method in enumerate(METHODS)}
    rows.sort(key=lambda row: (int(row["seed"]), order[row["method"]]))
    summary = summarize(rows)
    pairs = paired_differences(rows)
    pair_summary = paired_summary(pairs)
    tables = OUTPUT / "tables"
    for name, table in (
        ("metrics_per_run", rows),
        ("metrics_mean_std", summary),
        ("paired_differences_per_seed", pairs),
        ("paired_differences_mean_std", pair_summary),
    ):
        write_csv(tables / f"{name}.csv", table)
        write_md(tables / f"{name}.md", table)

    primary = [
        {
            key: row[key]
            for key in (
                "method",
                "seed_count",
                "J_all_net_mean",
                "J_all_net_std",
                "J_HV_net_mean",
                "J_HV_net_std",
                "N_HV_closure_mean",
                "N_HV_closure_std",
                "N_high_closure_mean",
                "N_high_closure_std",
                "N_urgent_closure_mean",
                "N_urgent_closure_std",
            )
        }
        for row in summary
    ]
    write_csv(tables / "primary_metrics_mean_std.csv", primary)
    write_md(tables / "primary_metrics_mean_std.md", primary)


def full_regression_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = next(row for row in rows if int(row["seed"]) == 2026 and row["method"] == METHOD_FULL)
    checks = {
        "J_all_net_exact": abs(as_decimal(observed["J_all_net"]) - exact.EXPECTED_CORRECTED["J_all_net"]) <= Decimal("1e-12"),
        "J_HV_net_exact": abs(as_decimal(observed["J_HV_net"]) - exact.EXPECTED_CORRECTED["J_HV_net"]) <= Decimal("1e-12"),
        "HV_closure_exact": as_int(observed["N_HV_closure"]) == exact.EXPECTED_CORRECTED["HV_closure_count"],
    }
    reference_checks: list[dict[str, Any]] = []
    reference_path = REFERENCE_FULL / "tables" / "all_metrics_per_seed.csv"
    if reference_path.is_file():
        references = {
            int(row["seed"]): row
            for row in read_csv(reference_path)
            if row["method"] == METHOD_FULL
        }
        for seed in SEEDS:
            current = next(row for row in rows if int(row["seed"]) == seed and row["method"] == METHOD_FULL)
            ref = references.get(seed)
            item = {"seed": seed, "reference_found": ref is not None}
            if ref is not None:
                item.update(
                    {
                        "J_all_match": abs(as_decimal(current["J_all_net"]) - as_decimal(ref["J_all_net"])) <= Decimal("1e-12"),
                        "J_HV_match": abs(as_decimal(current["J_HV_net"]) - as_decimal(ref["J_HV_net"])) <= Decimal("1e-12"),
                        "HV_closure_match": as_int(current["N_HV_closure"]) == as_int(ref["N_HV_closure"]),
                    }
                )
            reference_checks.append(item)
    passed = all(checks.values()) and all(
        not item["reference_found"]
        or all(item[key] for key in ("J_all_match", "J_HV_match", "HV_closure_match"))
        for item in reference_checks
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "seed2026_anchor_checks": checks,
        "five_seed_reference_checks": reference_checks,
        "reference_output": str(REFERENCE_FULL),
    }


def concurrency_audit(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = []
    for row in statuses:
        if row.get("started_utc") and row.get("finished_utc"):
            intervals.append(
                (
                    datetime.fromisoformat(row["started_utc"]),
                    datetime.fromisoformat(row["finished_utc"]),
                )
            )
    events = []
    for start, finish in intervals:
        events.append((start, 1))
        events.append((finish, -1))
    current = 0
    maximum = 0
    for _time, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        current += delta
        maximum = max(maximum, current)
    pids = [row.get("pid") for row in statuses if row.get("pid") is not None]
    previous_path = OUTPUT / "audit" / "parallel_execution_audit.json"
    previous = read_json(previous_path) if previous_path.is_file() else {}
    historical_maximum = max(int(previous.get("observed_max_concurrent_new_runs", 0)), maximum)
    historical_new_runs = int(previous.get("historical_new_run_count", previous.get("new_run_count", 0))) + len(intervals)
    return {
        "safe_parallel_enabled": True,
        "parallel_backend": "ThreadPoolExecutor supervising independent Python subprocesses",
        "configured_max_parallel_runs": MAX_PARALLEL_RUNS,
        "observed_max_concurrent_new_runs": historical_maximum,
        "last_invocation_max_concurrent_new_runs": maximum,
        "distinct_child_pid_count": len(set(pids)),
        "new_run_count": len(intervals),
        "historical_new_run_count": historical_new_runs,
        "resumed_existing_run_count": sum(row.get("status") == "RESUMED_EXISTING" for row in statuses),
        "shared_mutable_simulator_state": False,
        "bundle_access": "read_only_shared_input",
        "run_outputs": "independent_directories",
    }


def identity_audits(
    rows: list[dict[str, Any]],
    bundles: dict[int, Path],
    bundle_audit: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    pre_hashes: dict[str, str],
) -> None:
    identities = []
    environment_rows = []
    for seed in SEEDS:
        for method in METHODS:
            path = OUTPUT / "runs" / f"seed_{seed}" / method
            identity = read_json(path / "native_identity.json")
            identities.append(
                {
                    "seed": seed,
                    "method": method,
                    "variant": identity["variant"],
                    "source_identity_passed": identity["source_identity_passed"],
                    "variant_identity_passed": identity["runtime_audit"]["variant_identity_passed"],
                    "corrected_reqalign_enabled": identity["corrected_reqalign_enabled"],
                    "intervention_hash": stable_hash(identity["intervention"]),
                    "pid": identity["runtime_audit"]["pid"],
                }
            )
            scenario = identity["scenario_identity"]
            environment_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "task_hash": scenario["task_hash"],
                    "task_hash_match": scenario["task_hash_match"],
                    "task_count": scenario["task_count"],
                    "runtime_value_mismatch_count": scenario["runtime_value_mismatch_count"],
                    "field_mismatch_count": scenario["field_mismatch_count"],
                    "frozen_power_identity_match": scenario["frozen_power_identity_match"],
                    "capacity": scenario["capacity"],
                }
            )
    post_hashes = source_hashes()
    source_unchanged = pre_hashes == post_hashes
    identity_passed = all(
        row["source_identity_passed"]
        and row["variant_identity_passed"]
        and row["corrected_reqalign_enabled"]
        for row in identities
    )
    environment_passed = all(
        row["task_hash_match"]
        and int(row["task_count"]) == 268
        and int(row["runtime_value_mismatch_count"]) == 0
        and int(row["field_mismatch_count"]) == 0
        and row["frozen_power_identity_match"]
        and abs(float(row["capacity"]) - CAPACITY) <= 1.0e-12
        for row in environment_rows
    )
    write_json(
        OUTPUT / "audit" / "method_identity_audit.json",
        {
            "status": "PASS" if identity_passed and source_unchanged else "FAIL",
            "methods": list(METHODS),
            "run_identities": identities,
            "final_prop_source_hashes_before": pre_hashes,
            "final_prop_source_hashes_after": post_hashes,
            "final_prop_sources_unchanged": source_unchanged,
            "ablation_entry_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    write_json(
        OUTPUT / "audit" / "environment_identity_audit.json",
        {
            "status": "PASS" if environment_passed else "FAIL",
            "value_ratio": {"low": 1, "medium": 3, "high": 10, "urgent": 25},
            "capacity_multiplier": CAPACITY,
            "power_identity": v25.FROZEN_POWER,
            "bundle_audit": bundle_audit,
            "run_environment_rows": environment_rows,
        },
    )
    write_json(OUTPUT / "audit" / "parallel_execution_audit.json", concurrency_audit(statuses))
    if not identity_passed or not source_unchanged:
        raise RuntimeError("METHOD_IDENTITY_MISMATCH")
    if not environment_passed:
        raise RuntimeError("ENVIRONMENT_IDENTITY_MISMATCH")


def self_test() -> int:
    bundle_cfg = read_json(exact.SOURCE_BUNDLE / "resolved_config.json")
    fixed, fixed_audit = configure_variant(bundle_cfg, "fixed_yolo_b")
    fixed_actions = [row for row in fixed["action_space"]["actions"] if row.get("enabled", True)]
    assert all(
        row["model_name"] == "balanced"
        for row in fixed_actions
        if row["payload_policy"] != "defer"
    )
    original_payloads = {
        row["action_name"]: (row["payload_policy"], row["expected_payload_level"])
        for row in bundle_cfg["action_space"]["actions"]
        if row.get("enabled", True)
    }
    fixed_payloads = {
        row["action_name"]: (row["payload_policy"], row["expected_payload_level"])
        for row in fixed_actions
    }
    assert fixed_payloads == original_payloads
    assert fixed_audit["fixed_model_alias"] == "YOLO-B"

    direct, direct_audit = configure_variant(bundle_cfg, "direct_only")
    assert not any(
        row.get("enabled", True) and row["payload_policy"].startswith("metadata_then")
        for row in direct["action_space"]["actions"]
    )
    assert direct_audit["direct_requirement_or_defer"]
    progressive = direct["simulator"]["progressive_payload"]
    assert all(
        progressive[section]["enabled"] is False
        for section in (
            "phase6_proactive_followup",
            "phase6_residual_candidate_carryover",
            "phase6_same_slot_carryover",
            "roi_context_after_roi",
            "value_aware_staged_delivery_queue",
        )
    )

    single, single_audit = configure_variant(bundle_cfg, "single_timescale")
    neutral = _neutral_single_timescale_guidance()
    assert all(neutral[name] == 0.0 for name in ("lambda_E", "lambda_L", "lambda_Q", "lambda_S"))
    assert neutral["mode_name"] == "single_timescale_neutral"
    assert single_audit["hard_safety_unchanged"]
    assert single["simulator"]["soc_band_policy"]["enabled"] is False

    ack, ack_audit = configure_variant(bundle_cfg, "ground_ack_120s")
    assert ack_audit["ground_ack_wait_s"] == 120.0
    assert ack_audit["ground_ack_wait_slots"] == 6
    assert ack["simulator"]["progressive_payload"]["phase6_same_slot_carryover"]["enabled"] is False
    assert 5 * SLOT_SECONDS < GROUND_ACK_SECONDS
    assert 6 * SLOT_SECONDS == GROUND_ACK_SECONDS

    report = {
        "status": "PASS",
        "tests": [
            "fixed_yolo_b_preserves_product_policies_and_sizes",
            "fixed_yolo_b_forces_balanced_small_model",
            "single_timescale_has_zero_long_term_prices_and_neutral_mode",
            "direct_only_disables_all_progressive_paths",
            "ground_ack_120s_boundary_is_six_20s_slots",
        ],
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


def run_self_test() -> None:
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--self-test"],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    report = (
        f"command: {sys.executable} {Path(__file__).resolve()} --self-test\n"
        f"returncode: {process.returncode}\n\nSTDOUT\n{process.stdout}\nSTDERR\n{process.stderr}"
    )
    (OUTPUT / "test_report.txt").write_text(report, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError({"SELF_TEST_FAILED": process.stderr})


def parent_main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pre_hashes = source_hashes()
    run_self_test()
    bundles, bundle_audit = prepare_bundles()
    write_json(OUTPUT / "audit" / "controlled_bundle_audit.json", bundle_audit)

    statuses: list[dict[str, Any]] = []
    full_anchor_path = OUTPUT / "runs" / "seed_2026" / METHOD_FULL
    statuses.append(launch_child(METHOD_FULL, 2026, bundles[2026], full_anchor_path))
    eval_mod = v25.load_eval_module(OUTPUT)
    anchor_metrics = exact.evaluate_method(METHOD_FULL, full_anchor_path, bundles[2026], eval_mod, CAPACITY)
    anchor_gate = exact.regression_gate(anchor_metrics, exact.EXPECTED_CORRECTED)
    write_json(OUTPUT / "audit" / "full_prop_seed2026_regression_gate.json", anchor_gate)
    if not anchor_gate["passed"]:
        raise RuntimeError({"FULL_PROP_REGRESSION_MISMATCH": anchor_gate})

    pending = [
        (seed, method)
        for seed in SEEDS
        for method in METHODS
        if not (seed == 2026 and method == METHOD_FULL)
    ]
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_RUNS, thread_name_prefix="prop-ablation") as pool:
        futures = {
            pool.submit(
                launch_child,
                method,
                seed,
                bundles[seed],
                OUTPUT / "runs" / f"seed_{seed}" / method,
            ): (seed, method)
            for seed, method in pending
        }
        failures = []
        for future in as_completed(futures):
            seed, method = futures[future]
            try:
                status = future.result()
                statuses.append(status)
                print(json.dumps({"completed": len(statuses), "total": 25, "seed": seed, "method": method}, ensure_ascii=False), flush=True)
            except Exception as exc:
                failures.append({"seed": seed, "method": method, "error": repr(exc)})
        if failures:
            write_json(OUTPUT / "audit" / "runtime_failures.json", failures)
            raise RuntimeError({"ABLATION_RUNTIME_FAILED": failures})

    rows = collect_metrics(bundles)
    if len(rows) != 25:
        raise RuntimeError({"INCOMPLETE_RERUN_GRID": len(rows)})
    emit_tables(rows)
    regression = full_regression_audit(rows)
    write_json(OUTPUT / "audit" / "full_prop_regression_audit.json", regression)
    if regression["status"] != "PASS":
        raise RuntimeError(regression)
    write_json(OUTPUT / "audit" / "method_run_status.json", statuses)
    identity_audits(rows, bundles, bundle_audit, statuses, pre_hashes)

    completion = {
        "status": "V25_PROP_ABLATION_FIVE_METHOD_FIVESEED_COMPLETE",
        "run_count": 25,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "capacity_multiplier": CAPACITY,
        "safe_parallel_enabled": True,
        "public_accounting_passed": all(bool(row["public_accounting_passed"]) for row in rows),
        "hard_safety_passed": all(
            as_int(row["SOC_violations"]) == 0 and as_int(row["thermal_violations"]) == 0
            for row in rows
        ),
        "output": str(OUTPUT),
        "entry": str(Path(__file__).resolve()),
    }
    write_json(OUTPUT / "completion_status.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--bundle-root", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.child_run:
        if args.method not in METHODS or args.seed not in SEEDS or not args.bundle_root or not args.output:
            raise ValueError("invalid internal child arguments")
        run_child(args.method, args.seed, Path(args.bundle_root), Path(args.output))
        print(json.dumps({"status": "PASS", "method": args.method, "seed": args.seed}, ensure_ascii=False))
        return 0
    return parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
