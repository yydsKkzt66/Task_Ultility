"""One-click controlled safe/metadata/thermal exploration (seed 2026 only).

The public entry takes no arguments.  A private ``--child-spec`` argument is
used solely to run each variant in an isolated Python process.  Existing source,
configuration and result assets are treated as read-only; every invocation
creates a new numbered attempt directory.
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
from collections import Counter, defaultdict, deque
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
import scripts.run_frozen_v2_final_gate_blind_fiveseed_v1 as finalgate  # noqa: E402
import scripts.run_historical_asset_audit_conditional_v2_calibration_v1 as calibration  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_normal_scenario_tut_proxy_ground_recovery_v1 as normal  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402
import scripts.run_v25_decoupled_v2_seed2026_diagnostic as v2diag  # noqa: E402


def _private_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exphelper = _private_module(
    "safe_metadata_thermal_exploration_helper",
    ROOT / "src" / "experiments" / "safe_metadata_thermal_exploration.py",
)
gatecore = finalgate.gatecore

VERSION = "SAFE-METADATA-THERMAL-EXPLORATION-v1"
COMPLETION = "SINGLE_SEED_CONTROLLED_SAFE_METADATA_THERMAL_EXPLORATION_COMPLETE"
SEED = 2026
SLOTS = 4320
CAPACITY = 0.50
EXPECTED_TASK_HASH = "c31d50bf0633e0d7395f78a785768c2ac492eb5f707c02a9e5be93b1226b37fa"
CONFIG = ROOT / "configs" / "safe_metadata_thermal_exploration_v1.yaml"
FROZEN = finalgate.FROZEN
TRUSTED_SCENARIO = normal.TRUSTED_SCENARIO
OUTPUT_BASE = ROOT / "outputs" / "_single_seed_controlled_safe_metadata_thermal_exploration_v1"
REFERENCE_RUN = ROOT / "outputs" / "_hv_balanced_to_high_single_seed_counterfactual_audit_v1" / "runs" / "FULL_PROP_V2"
REFERENCE_METRICS = ROOT / "outputs" / "_hv_balanced_to_high_single_seed_counterfactual_audit_v1" / "audit" / "source_run_public_metrics.json"
ROOT_CAUSES = ROOT / "outputs" / "_hv_balanced_to_high_single_seed_counterfactual_audit_v1" / "tables" / "unclosed_hv_evidence_root_causes.csv"
BALANCED_HIGH_CF = ROOT / "outputs" / "_hv_balanced_to_high_single_seed_counterfactual_audit_v1" / "tables" / "hv_balanced_to_high_task_counterfactual.csv"

V0 = "V0-CURRENT-BASELINE"
V1 = "V1-SAFE-SET-CONSISTENT-FALLBACK"
V2 = "V2-HV-METADATA-RESIDUAL-REPAIR"
V3 = "V3-THERMAL-PRICE-AWARE"
V4 = "V4-COMBINED-ABC"
METHODS = (V0, V1, V2, V3, V4)
SLUG = {method: method.replace("-", "_") for method in METHODS}
FLAGS = {
    V0: (False, False, False),
    V1: (True, False, False),
    V2: (False, True, False),
    V3: (False, False, True),
    V4: (True, True, True),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def f(value: Any, default: float = 0.0) -> float:
    return gatecore.f(value, default)


def i(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    return gatecore.stable_hash(value)


def current_rows(wrapper: Any) -> list[dict[str, Any]]:
    if not wrapper.v2_decision_rows:
        return []
    decision = wrapper.v2_decision_rows[-1]
    slot, task_id = i(decision["decision_slot"]), str(decision["task_id"])
    return [row for row in wrapper.candidate_rows if i(row["decision_slot"]) == slot and str(row["task_id"]) == task_id]


class ControlledExplorationWrapper(calibration.ConditionalCalibrationWrapper):
    """Process-local adapter containing only the three registered mechanisms."""

    def __init__(self, *args: Any, safe_consistent: bool, metadata_repair: bool, thermal_price: bool, thermal_alpha: float, mode_update_slots: int, ordinary_hard_requirement: bool = False, difficulty_by_task: Mapping[str, Mapping[str, str]] | None = None, conditional_quality_table: Mapping[str, Mapping[str, float]] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.safe_consistent = bool(safe_consistent)
        self.metadata_repair = bool(metadata_repair)
        self.thermal_price_enabled = bool(thermal_price)
        self.ordinary_hard_requirement = bool(ordinary_hard_requirement)
        self.difficulty_by_task = {str(key): dict(value) for key, value in (difficulty_by_task or {}).items()}
        self.conditional_quality_table = {str(model): {str(level): float(q) for level, q in values.items()} for model, values in (conditional_quality_table or {}).items()}
        self.difficulty_rows: list[dict[str, Any]] = []
        self.thermal_alpha = float(thermal_alpha)
        self.intervention_rows: list[dict[str, Any]] = []
        self.thermal_rows: list[dict[str, Any]] = []
        self.guard_rows: list[dict[str, Any]] = []
        self.recent_compute = deque(maxlen=max(1, int(mode_update_slots)))

    def _replace_choice(self, action: Any, predicted_map: Mapping[str, Any], reason: str, extra: Mapping[str, Any] | None = None) -> Any:
        decision = self.v2_decision_rows[-1]
        rows = current_rows(self)
        chosen = next((row for row in rows if row["action"] == action.action_name), None)
        score = 0.0 if action.payload_policy == "defer" else f(chosen.get("v2_net_score") if chosen else 0.0)
        quality = 0.0 if action.payload_policy == "defer" else f(chosen.get("v2_quality") if chosen else 0.0)
        product = str(predicted_map[action.action_name].predicted_product_type)
        previous = str(decision["selected_action"])
        for row in rows:
            row["final_executed_choice"] = int(row["action"] == action.action_name)
        decision.update({
            "selected_action": action.action_name,
            "selected_model": action.model_name,
            "selected_product": product,
            "selection_reason": reason,
            "v2_net_score": score,
            "v2_quality": quality,
            "q_multiplication_count": 0 if action.payload_policy == "defer" else 1,
        })
        self._set_delegate_state(action, score)
        row = {
            "decision_slot": decision["decision_slot"], "task_id": decision["task_id"],
            "value_level": decision["value_level"], "mechanism_reason": reason,
            "action_before": previous, "action_after": action.action_name,
            "changed": int(previous != action.action_name),
        }
        if extra:
            row.update(dict(extra))
        self.intervention_rows.append(row)
        return action

    def _arrival_guard(self, action: Any, predicted_map: Mapping[str, Any]) -> tuple[bool, list[str], Any]:
        task = self.current_step.task
        hard = list(self.step_runner.safe_filter.evaluate_action(task, action, predicted_map))
        if action.payload_policy == "defer":
            return not hard, hard, None
        current_s = f(self.current_step.slot_index) * f(self.current_step.slot_seconds)
        deadline_s = self._task_absolute_deadline_s(task, self.current_step.slot_seconds)
        arrival = self._arrival(action, predicted_map[action.action_name], task, current_s, deadline_s)
        if not arrival.contact_found:
            hard.append("no contact before deadline")
        if arrival.ground_arrival_s is None or f(arrival.ground_arrival_s) > deadline_s + 1.0e-9:
            hard.append("deadline infeasible")
        return not hard, list(dict.fromkeys(hard)), arrival

    def _safe_prefilter(self, candidate_actions: list[Any], predicted_map: Mapping[str, Any]) -> list[Any]:
        if not self.safe_consistent:
            return candidate_actions
        facts = []
        for action in candidate_actions:
            passed, reasons, _ = self._arrival_guard(action, predicted_map)
            facts.append((action, passed, reasons))
        admitted = [action for action, passed, _ in facts if passed]
        if not admitted:
            defer = next(action for action in self.step_runner.action_space.list_actions() if action.payload_policy == "defer")
            admitted = [defer]
        self.guard_rows.extend({
            "decision_slot": i(self.current_step.slot_index), "task_id": self.current_step.task.task_id,
            "phase": "pre_filter", "action": action.action_name, "guard_pass": int(passed),
            "guard_reasons": ";".join(reasons), "admitted_to_selector": int(action in admitted),
        } for action, passed, reasons in facts)
        return admitted

    def _repair_metadata(self, selected: Any, predicted_map: Mapping[str, Any]) -> Any:
        if not self.metadata_repair:
            return selected
        decision = self.v2_decision_rows[-1]
        if decision["value_level"] not in {"high", "urgent"}:
            return selected
        residual = self.projection.normalize_product(str(decision["residual_requirement_before"]))
        delivered = set(filter(None, str(decision.get("delivered_products_before", "")).split(";")))
        if not residual or "metadata" in delivered:
            return selected
        rows = current_rows(self)
        direct = [row for row in rows if self.projection.normalize_product(str(row["product"])) == residual and i(row["safe_filter_pass"]) and i(row["deadline_feasible"]) and f(row["v2_net_score"]) > 1.0e-12]
        if direct:
            return selected
        metadata = [row for row in rows if row["product"] == "metadata" and i(row["safe_filter_pass"]) and i(row["deadline_feasible"]) and f(row["v2_net_score"]) > 1.0e-12]
        if not metadata:
            self.intervention_rows.append({
                "decision_slot": decision["decision_slot"], "task_id": decision["task_id"], "value_level": decision["value_level"],
                "mechanism_reason": "hv_metadata_repair_no_positive_feasible_metadata", "action_before": selected.action_name,
                "action_after": selected.action_name, "changed": 0, "required_positive_direct_count": 0, "positive_metadata_count": 0,
            })
            return selected
        chosen = max(metadata, key=lambda row: (f(row["v2_net_score"]), str(row["action"])))
        action = next(action for action in self.step_runner.action_space.list_actions() if action.action_name == chosen["action"])
        return self._replace_choice(action, predicted_map, "hv_metadata_residual_repair", {
            "required_positive_direct_count": 0, "positive_metadata_count": len(metadata), "metadata_net_score": f(chosen["v2_net_score"]),
        })

    def _pending_hv_residual(self) -> bool:
        for task_id, state in self.step_runner.product_lifecycle.items():
            task = self.step_runner.task_snapshots.get(task_id)
            if task is None or self._task_value_level(task) not in {"high", "urgent"}:
                continue
            if bool(getattr(state, "task_closed_flag", False)):
                continue
            if getattr(state, "residual_parent_payload_plan", "") == self._DEFER_RECHECK_MARKER or getattr(state, "pending_products", set()):
                return True
        return False

    def _apply_thermal_price(self, selected: Any, predicted_map: Mapping[str, Any]) -> Any:
        if not self.thermal_price_enabled:
            return selected
        decision = self.v2_decision_rows[-1]
        rows = current_rows(self)
        defer_action = next(action for action in self.step_runner.action_space.list_actions() if action.payload_policy == "defer")
        defer_theta = f(predicted_map[defer_action.action_name].predicted_Theta_next)
        theta = f(self.current_resource_state.thermal_pressure)
        theta0 = f(self.simulator_cfg.get("thermal_init", 0.52), 0.52)
        theta_max = f(self.simulator_cfg["thermal_max"], 0.92)
        alpha = self.thermal_alpha
        snapshot = exphelper.thermal_price_snapshot(
            current_thermal=theta, defer_theta_next=defer_theta, thermal_initial=theta0, thermal_max=theta_max,
            recent_compute_pressure=statistics.fmean(self.recent_compute) if self.recent_compute else 0.0,
            has_pending_hv_residual=self._pending_hv_residual(), alpha_thermal=alpha,
        )
        all_actions = {action.action_name: action for action in self.step_runner.action_space.list_actions()}
        ranked = []
        for row in rows:
            action = all_actions[row["action"]]
            increment = exphelper.normalized_thermal_increment(
                action_theta_next=f(predicted_map[action.action_name].predicted_Theta_next), defer_theta_next=defer_theta,
                thermal_initial=theta0, thermal_max=theta_max,
            )
            adjusted = f(row["v2_net_score"]) - snapshot.thermal_price * increment
            row.update({"thermal_price": snapshot.thermal_price, "normalized_predicted_thermal_increment": increment, "thermal_adjusted_score": adjusted})
            if i(row["safe_filter_pass"]) and (action.payload_policy == "defer" or i(row["deadline_feasible"])):
                ranked.append((adjusted, f(row["v2_net_score"]), row["action"], action, increment))
            self.thermal_rows.append({
                "decision_slot": decision["decision_slot"], "decision_time_s": decision["decision_time_s"],
                "task_id": decision["task_id"], "value_level": decision["value_level"], "action": row["action"],
                **snapshot.as_dict(), "predicted_theta_next": f(predicted_map[action.action_name].predicted_Theta_next),
                "normalized_predicted_thermal_increment": increment, "original_score": f(row["v2_net_score"]), "thermal_adjusted_score": adjusted,
                "applied_to_decision": int(decision["value_level"] in {"low", "medium"}),
            })
        if decision["value_level"] not in {"low", "medium"}:
            return selected
        chosen = max(ranked, default=(0.0, 0.0, defer_action.action_name, defer_action, 0.0), key=lambda item: (item[0], item[1], item[2]))
        action = chosen[3] if chosen[0] > 1.0e-12 else defer_action
        if action.action_name == selected.action_name:
            return selected
        old_rank = {"delay_store": 0, "lightweight": 1, "balanced": 2, "high_accuracy": 3}.get(selected.model_name, 0)
        new_rank = {"delay_store": 0, "lightweight": 1, "balanced": 2, "high_accuracy": 3}.get(action.model_name, 0)
        return self._replace_choice(action, predicted_map, "ordinary_thermal_price_aware", {
            "thermal_price": snapshot.thermal_price, "thermal_adjusted_score": chosen[0], "thermal_model_downgrade": int(new_rank < old_rank),
        })

    def _final_consistency(self, selected: Any, predicted_map: Mapping[str, Any]) -> Any:
        if not self.safe_consistent:
            return selected
        passed, reasons, _ = self._arrival_guard(selected, predicted_map)
        decision = self.v2_decision_rows[-1]
        if passed:
            self.guard_rows.append({
                "decision_slot": decision["decision_slot"], "task_id": decision["task_id"], "phase": "final_guard",
                "action": selected.action_name, "guard_pass": 1, "guard_reasons": "", "fallback_used": 0,
            })
            return selected
        rows = current_rows(self)
        for row in rows:
            action = next(action for action in self.step_runner.action_space.list_actions() if action.action_name == row["action"])
            ok, why, _ = self._arrival_guard(action, predicted_map)
            row["unified_guard_pass"] = int(ok)
            row["unified_guard_reasons"] = ";".join(why)
        chosen = exphelper.choose_ranked_fallback(rows, original_product=str(decision["selected_product"]))
        if chosen is None:
            action = next(action for action in self.step_runner.action_space.list_actions() if action.payload_policy == "defer")
            reason = "consistent_guard_terminal_defer"
        else:
            action = next(action for action in self.step_runner.action_space.list_actions() if action.action_name == chosen["action"])
            reason = "consistent_guard_safe_fallback"
        self.guard_rows.append({
            "decision_slot": decision["decision_slot"], "task_id": decision["task_id"], "phase": "final_guard",
            "action": selected.action_name, "guard_pass": 0, "guard_reasons": ";".join(reasons),
            "fallback_used": int(action.payload_policy != "defer"), "fallback_action": action.action_name,
        })
        return self._replace_choice(action, predicted_map, reason, {"rejected_primary": selected.action_name, "guard_reasons": ";".join(reasons)})

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        task = self.current_step.task
        difficulty = self.difficulty_by_task.get(str(task.task_id), {}) if task is not None else {}
        observed_difficulty = str(difficulty.get("difficulty_observed", ""))
        if observed_difficulty and self.conditional_quality_table:
            core = v2diag.load_v2core()
            core.FROZEN_QUALITY_PROXY.clear()
            core.FROZEN_QUALITY_PROXY.update({model: float(values[observed_difficulty]) for model, values in self.conditional_quality_table.items()})
        if task is not None and self.ordinary_hard_requirement and self._task_value_level(task) in {"low", "medium"}:
            required = str(task.product_requirement)
            constrained = [
                action for action in candidate_actions
                if action.payload_policy == "defer"
                or self.projection.product_satisfies(str(predicted_map[action.action_name].predicted_product_type), required)
            ]
            if constrained:
                candidate_actions = constrained
        filtered = self._safe_prefilter(candidate_actions, predicted_map)
        selected = super().select_action(context, filtered, predicted_map, guidance_bundle)
        selected = self._repair_metadata(selected, predicted_map)
        selected = self._apply_thermal_price(selected, predicted_map)
        selected = self._final_consistency(selected, predicted_map)
        non_defer = [f(item.predicted_E_comp) for name, item in predicted_map.items() if name != "delay/store"]
        denom = max(non_defer, default=1.0)
        self.recent_compute.append(min(1.0, f(predicted_map[selected.action_name].predicted_E_comp) / max(denom, 1.0e-12)))
        if task is not None and difficulty and self.conditional_quality_table:
            selected_product = str(predicted_map[selected.action_name].predicted_product_type)
            balanced = next((action for action in self.step_runner.action_space.list_actions() if action.model_name == "balanced" and str(predicted_map[action.action_name].predicted_product_type) == selected_product), None)
            self.difficulty_rows.append({
                "decision_slot": int(self.current_step.slot_index), "task_id": str(task.task_id),
                "value_level": self._task_value_level(task), "required_product": str(task.product_requirement),
                "difficulty_observed": observed_difficulty,
                "selected_action": selected.action_name, "selected_model": selected.model_name, "selected_product": selected_product,
                "selected_q_observed": 0.0 if selected.payload_policy == "defer" else self.conditional_quality_table.get(selected.model_name, {}).get(observed_difficulty, 0.0),
                "balanced_same_product_action": "" if balanced is None else balanced.action_name,
                "balanced_same_product_hard_safe": "" if balanced is None else int(not self.step_runner.safe_filter.evaluate_action(task, balanced, predicted_map)),
                "selected_predicted_energy_j": f(predicted_map[selected.action_name].predicted_E_comp) + f(predicted_map[selected.action_name].predicted_E_comm),
                "balanced_predicted_energy_j": "" if balanced is None else f(predicted_map[balanced.action_name].predicted_E_comp) + f(predicted_map[balanced.action_name].predicted_E_comm),
                "selected_predicted_theta_next": f(predicted_map[selected.action_name].predicted_Theta_next),
                "balanced_predicted_theta_next": "" if balanced is None else f(predicted_map[balanced.action_name].predicted_Theta_next),
            })
        return selected


class DirectOnlyControlledExplorationWrapper(ControlledExplorationWrapper):
    """A1 adapter: current safe-consistency logic, but no HV staged delivery."""

    def prepare_step_for_reentry(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> Any:
        return timeline_step

    def finish_step(self, timeline_step: Any) -> None:
        return None

    def _repair_metadata(self, selected: Any, predicted_map: Mapping[str, Any]) -> Any:
        return selected

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        selected = super().select_action(context, candidate_actions, predicted_map, guidance_bundle)
        row = self.v2_decision_rows[-1]
        if row["value_level"] not in {"high", "urgent"}:
            return selected
        if self.projection.product_satisfies(row["selected_product"], row["residual_requirement_before"]):
            return selected
        defer = next(action for action in self.step_runner.action_space.list_actions() if action.payload_policy == "defer")
        return self._replace_choice(defer, predicted_map, "A1_direct_only_required_infeasible_defer", {"progressive_delivery_disabled": 1})


def install_safe_greedy_delegate(delegate: Any, audit: dict[str, Any]) -> None:
    """Replace online/UCB behavior by deterministic predicted-net greedy choice."""

    def select_action(self: Any, context: Any, candidate_actions: list[Any], predicted_map: Mapping[str, Any], guidance_bundle=None):
        audit["greedy_select_call_count"] += 1
        rows = {}
        for action in candidate_actions:
            score = f(predicted_map[action.action_name].predicted_reward)
            rows[action.action_name] = {
                "predicted_reward": score, "final_action_score": score, "ucb_bonus": 0.0,
                "raw_ucb_bonus": 0.0, "online_bandit_disabled": 1,
            }
        selected = max(candidate_actions, key=lambda action: (f(predicted_map[action.action_name].predicted_reward), action.action_name))
        self.last_all_score_components = rows
        self.last_score_components = deepcopy(rows[selected.action_name])
        self.last_selected_action_score = f(predicted_map[selected.action_name].predicted_reward)
        return selected

    def update(self: Any, context: Any, action: Any, observed_reward: float) -> None:
        audit["blocked_online_update_count"] += 1

    delegate.select_action = types.MethodType(select_action, delegate)
    delegate.update = types.MethodType(update, delegate)


def configure_and_run_child(spec: Mapping[str, Any], output: Path) -> None:
    method = str(spec["method"])
    safe_consistent, metadata_repair, thermal_price = tuple(spec.get("flags", FLAGS.get(method, (False, False, False))))
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    frozen = finalgate.load_frozen()
    core = v2diag.load_v2core()
    core.FROZEN_QUALITY_PROXY.clear()
    core.FROZEN_QUALITY_PROXY.update({key: f(value) for key, value in frozen["q_eff"].items()})
    bundle = normal.load_neutral_bundle(Path(spec["bundle_root"]))
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    configured = v25.import_prop_script()._configure(scenario.configs, v25.read_json(v25.SHARED_PRIOR_CALIBRATION), {"enabled": False})
    fixed_model_name = str(spec.get("fixed_model_name", "balanced" if spec.get("fixed_model", False) else ""))
    fixed_model = bool(fixed_model_name)
    if fixed_model_name in {"", "balanced"}:
        configured, grid_audit = v2diag.configure_v2(configured, fixed=fixed_model)
    elif fixed_model_name in {"high_accuracy", "lightweight"}:
        configured = deepcopy(configured)
        configured["action_space"] = {"actions": core.build_decoupled_action_rows(fixed_model=fixed_model_name)}
        configured["simulator"]["safety_terminal_fallback_candidates"] = ["delay/store", f"v2:{fixed_model_name}+metadata"]
        grid_audit = core.action_grid_audit(configured["action_space"]["actions"], fixed_model=fixed_model_name)
        if not grid_audit["complete"] or grid_audit["defer_count"] != 1:
            raise RuntimeError({"V2_FIXED_MODEL_ACTION_GRID_INVALID": fixed_model_name, "audit": grid_audit})
    else:
        raise ValueError({"unsupported_fixed_model_name": fixed_model_name})
    configured = calibration.configure_online_prices(configured, {"prices": frozen["public_prices"], "legacy_online_baseline": False})
    single_timescale = bool(spec.get("single_timescale", False))
    direct_only = bool(spec.get("direct_only", False))
    greedy_no_bandit = bool(spec.get("greedy_no_bandit", False))
    intervention: dict[str, Any] = {"variant": method}
    if single_timescale:
        configured, intervention = finalgate.legacy_ablation.configure_variant(configured, "single_timescale")
    if direct_only:
        finalgate._disable_progressive(configured)
    scenario = replace(scenario, configs=configured)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, CAPACITY)
    identity = exact.scenario_identity(method, SEED, scenario, bundle, power, CAPACITY)
    if identity.get("task_hash") != EXPECTED_TASK_HASH:
        raise RuntimeError({"TASK_HASH_MISMATCH": identity.get("task_hash")})
    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    delegate = build_routers(scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"]))[0]
    greedy_audit = {"enabled": greedy_no_bandit, "greedy_select_call_count": 0, "blocked_online_update_count": 0}
    if greedy_no_bandit:
        install_safe_greedy_delegate(delegate, greedy_audit)
    wrapper_cls = DirectOnlyControlledExplorationWrapper if direct_only else ControlledExplorationWrapper
    wrapper = wrapper_cls(
        delegate, reqrun.load_projection_config(True), scenario.timeline[:SLOTS], scenario.configs["simulator"],
        reqrun.load_projection_module(), product_utility_cfg=scenario.configs["product_requirement_utility"],
        reward_cfg=scenario.configs["reward"], safe_consistent=safe_consistent,
        metadata_repair=metadata_repair, thermal_price=thermal_price,
        thermal_alpha=f(scenario.configs["lyapunov_pricing"]["alpha_thermal"]),
        mode_update_slots=i(scenario.configs["mode_controller"]["update_interval_slots"]),
        ordinary_hard_requirement=bool(spec.get("ordinary_hard_requirement", False)),
        difficulty_by_task=spec.get("difficulty_by_task", {}),
        conditional_quality_table=spec.get("conditional_quality_table", {}),
    )
    reqrun.bind_projection_runtime(runner, wrapper)
    slow_runtime_audit = {"static_guidance_call_count": 0, "observed_modes": set(), "observed_lambda_vectors": set()}
    if single_timescale:
        finalgate.legacy_ablation.bind_single_timescale_runtime(runner, slow_runtime_audit)
    result = runner.run(wrapper)
    slow_runtime_audit["observed_modes"] = sorted(slow_runtime_audit["observed_modes"])
    slow_runtime_audit["observed_lambda_vectors"] = [list(value) for value in sorted(slow_runtime_audit["observed_lambda_vectors"])]
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    reqrun.write_energy_components(output, result)
    write_csv(output / "v2_candidate_score_audit.csv", wrapper.candidate_rows)
    write_csv(output / "v2_decision_trace.csv", wrapper.v2_decision_rows)
    write_csv(output / "mechanism_intervention_trace.csv", wrapper.intervention_rows)
    write_csv(output / "unified_guard_trace.csv", wrapper.guard_rows)
    write_csv(output / "thermal_price_action_trace.csv", wrapper.thermal_rows)
    write_csv(output / "difficulty_decision_trace.csv", wrapper.difficulty_rows)
    run_identity = {
        "status": "PASS", "version": VERSION, "method": method, "seed": SEED, "pid": os.getpid(),
        "fresh_state": True, "independent_rng_queue_soc_thermal_bandit_residual": True,
        "safe_set_consistent_fallback": safe_consistent, "hv_metadata_residual_repair": metadata_repair,
        "thermal_price_aware": thermal_price, "thermal_price_status": "experimental_no_new_tuning" if thermal_price else "disabled",
        "wrapper_class": type(wrapper).__name__, "delegate_class": type(delegate).__name__, "grid_audit": grid_audit,
        "scenario_identity": identity, "power_identity": power, "capacity_identity": capacity_identity,
        "q_eff": frozen["q_eff"], "public_prices": frozen["public_prices"], "q_once_online": True,
        "fixed_model": fixed_model_name, "direct_only": direct_only,
        "ordinary_hard_requirement": bool(spec.get("ordinary_hard_requirement", False)),
        "difficulty_aware_quality_enabled": bool(spec.get("difficulty_by_task")) and bool(spec.get("conditional_quality_table")),
        "difficulty_policy_observes_true": False,
        "single_timescale": single_timescale, "greedy_no_online_bandit": greedy_no_bandit,
        "greedy_audit": greedy_audit, "ablation_intervention": intervention, "slow_runtime_audit": slow_runtime_audit,
        "hard_safe_filter_unchanged": True, "public_evaluator_unchanged": True, "future_information_used": False,
        "spec_sha256": stable_hash(spec),
    }
    write_json(output / "runtime_identity.json", run_identity)
    write_json(output / "resolved_config.json", scenario.configs)
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": SEED, "spec_sha256": stable_hash(spec)})


def child_main(spec_path: Path) -> int:
    spec = read_json(spec_path)
    output = Path(spec["output"])
    if output.exists():
        raise FileExistsError(output)
    configure_and_run_child(spec, output)
    return 0


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    index = 1
    while (OUTPUT_BASE / f"attempt_{index:03d}").exists():
        index += 1
    attempt = OUTPUT_BASE / f"attempt_{index:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    for name in ("audit", "runs", "run_specs", "tables", "scenario"):
        (attempt / name).mkdir()
    return attempt


def source_files() -> list[Path]:
    return [
        Path(__file__).resolve(), CONFIG, ROOT / "src" / "experiments" / "safe_metadata_thermal_exploration.py",
        ROOT / "src" / "evaluation" / "frozen_v2_final_gate.py", FROZEN, v25.POWER_CONFIG,
        v25.SOURCE_PROP / "src" / "routers" / "safe_filter.py",
        v25.SOURCE_PROP / "src" / "simulator" / "step_runner.py",
        v25.SOURCE_PROP / "src" / "energy" / "thermal_update.py",
        ROOT / "scripts" / "run_v25_decoupled_v2_seed2026_diagnostic.py",
        ROOT / "scripts" / "run_historical_asset_audit_conditional_v2_calibration_v1.py",
        TRUSTED_SCENARIO / "bundle_contract.json", TRUSTED_SCENARIO / "task_manifest.csv", TRUSTED_SCENARIO / "slot_trace.csv",
    ]


def hash_snapshot() -> dict[str, str]:
    return {str(path): sha256_file(path) for path in source_files() if path.is_file()}


def make_spec(method: str, bundle_root: Path, attempt: Path) -> dict[str, Any]:
    return {
        "version": VERSION, "method": method, "seed": SEED, "slots": SLOTS, "capacity": CAPACITY,
        "bundle_root": str(bundle_root), "output": str(attempt / "runs" / SLUG[method]),
        "frozen_sha256": sha256_file(FROZEN), "config_sha256": sha256_file(CONFIG),
        "entry_sha256": sha256_file(Path(__file__).resolve()), "flags": FLAGS[method],
    }


def launch(spec: Mapping[str, Any], attempt: Path) -> dict[str, Any]:
    spec_path = attempt / "run_specs" / f"{SLUG[str(spec['method'])]}.json"
    write_json(spec_path, spec)
    log = attempt / "audit" / f"{SLUG[str(spec['method'])]}.stdout.log"
    err = attempt / "audit" / f"{SLUG[str(spec['method'])]}.stderr.log"
    with log.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)],
            cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError({"VARIANT_RUNTIME_FAILED": spec["method"], "returncode": completed.returncode, "stderr": str(err)})
    identity = read_json(Path(spec["output"]) / "runtime_identity.json")
    return {"method": spec["method"], "pid": identity["pid"], "output": spec["output"]}


def evaluate_run(method: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = finalgate.load_frozen()
    metric, ledger = gatecore.evaluate_public_run(method, path, read_json(path / "resolved_config.json"), frozen)
    decisions = read_csv(path / "v2_decision_trace.csv")
    candidates = read_csv(path / "v2_candidate_score_audit.csv")
    slots = read_csv(path / "per_slot_resource_trace.csv")
    outcomes = read_csv(path / "per_task_outcomes.csv")
    events = read_csv(path / "ground_delivery_events.csv")
    interventions = read_csv(path / "mechanism_intervention_trace.csv")
    native_candidates = read_csv(path / "candidate_action_log.csv")
    task_closed = {row["task_id"]: i(row["closure_success"]) for row in outcomes}
    metadata_tasks = {row["task_id"] for row in ledger if row["product"] == "metadata"}
    hv_metadata_decisions = {row["task_id"] for row in decisions if row["value_level"] in {"high", "urgent"} and row["selected_product"] == "metadata"}
    residual_created = {row["task_id"] for row in decisions if i(row.get("residual_registered"))}
    residual_upgraded = {task_id for task_id in residual_created if task_closed.get(task_id, 0)}
    mismatch_pairs = defaultdict(lambda: {"selected": "", "executed": ""})
    thermal_rejected = set()
    for row in native_candidates:
        key = (row["slot"], row["task_id"])
        if i(row.get("selected_flag")):
            mismatch_pairs[key]["selected"] = row["candidate_action"]
        if i(row.get("executed_flag")):
            mismatch_pairs[key]["executed"] = row["candidate_action"]
        if "thermal" in str(row.get("safety_rejection_reason", "")).lower():
            thermal_rejected.add((row["slot"], row["task_id"], row["candidate_action"]))
    mismatches = [value for value in mismatch_pairs.values() if value["selected"] and value["executed"] and value["selected"] != value["executed"]]
    partial_unclosed_hv = sum(
        f(row["gross"]) for row in ledger
        if row["value_level"] in {"high", "urgent"} and not task_closed.get(row["task_id"], 0)
    )
    model_total = metric["model_lightweight_count"] + metric["model_balanced_count"] + metric["model_high_accuracy_count"]
    queue_values = [f(row.get("queue_mb")) for row in slots]
    thermal_values = [f(row.get("thermal")) for row in slots]
    thermal_max = f(read_json(path / "resolved_config.json")["simulator"]["thermal_max"], 0.92)
    near_limit_threshold = f(read_json(path / "resolved_config.json")["mode_controller"]["thresholds"]["thermal_high"], 0.8)
    thermal_before_slot = {index: (thermal_values[index - 1] if index > 0 else f(read_json(path / "resolved_config.json")["simulator"].get("thermal_init"), 0.52)) for index in range(len(thermal_values))}
    hv_arrival_slots = {i(f(row["arrival_time"]) / 20.0) for row in outcomes if row["value_level"] in {"high", "urgent"}}
    hv_arrival_thermal = [thermal_before_slot[slot] for slot in sorted(hv_arrival_slots) if slot in thermal_before_slot]
    fallback_success = sum(row.get("mechanism_reason") == "consistent_guard_safe_fallback" and i(row.get("changed")) for row in interventions)
    metric.update({
        "mean_queue_MB": statistics.fmean(queue_values) if queue_values else 0.0,
        "HV_metadata_first_task_count": len(hv_metadata_decisions), "residual_created_task_count": len(residual_created),
        "residual_upgrade_success_task_count": len(residual_upgraded),
        "residual_expired_task_count": sum(row["value_level"] in {"high", "urgent"} and not i(row["closure_success"]) and f(row["deadline"]) <= 86400 for row in outcomes),
        "HV_partial_gross_utility_unclosed": partial_unclosed_hv,
        "selector_final_guard_mismatch_count": len(mismatches), "fallback_success_count": fallback_success,
        "final_defer_decision_count": sum(row["selected_product"] == "defer" for row in decisions),
        "thermal_hard_rejection_count": len(thermal_rejected),
        "model_lightweight_ratio": metric["model_lightweight_count"] / max(model_total, 1),
        "model_balanced_ratio": metric["model_balanced_count"] / max(model_total, 1),
        "model_high_accuracy_ratio": metric["model_high_accuracy_count"] / max(model_total, 1),
        "metadata_alert_task_count": len(metadata_tasks),
        "safe_guard_fallback_intervention_count": sum(row.get("mechanism_reason") in {"consistent_guard_safe_fallback", "consistent_guard_terminal_defer"} for row in interventions),
        "hv_metadata_repair_intervention_count": sum(row.get("mechanism_reason") == "hv_metadata_residual_repair" and i(row.get("changed")) for row in interventions),
        "ordinary_thermal_action_change_count": sum(row.get("mechanism_reason") == "ordinary_thermal_price_aware" and i(row.get("changed")) for row in interventions),
        "ordinary_thermal_model_downgrade_count": sum(row.get("mechanism_reason") == "ordinary_thermal_price_aware" and i(row.get("thermal_model_downgrade")) for row in interventions),
        "HV_arrival_mean_thermal": statistics.fmean(hv_arrival_thermal) if hv_arrival_thermal else None,
        "HV_arrival_mean_thermal_headroom": statistics.fmean(thermal_max - value for value in hv_arrival_thermal) if hv_arrival_thermal else None,
        "thermal_near_limit_threshold": near_limit_threshold,
        "thermal_near_limit_slot_count": sum(value >= near_limit_threshold for value in thermal_values),
        "thermal_near_limit_duration_s": 20.0 * sum(value >= near_limit_threshold for value in thermal_values),
    })
    write_csv(path / "frozen_public_event_ledger.csv", ledger)
    write_json(path / "frozen_public_metrics.json", metric)
    return metric, ledger


def decision_semantic_hash(path: Path) -> str:
    rows = read_csv(path / "v2_decision_trace.csv")
    semantic = [{key: row.get(key, "") for key in (
        "decision_slot", "task_id", "value_level", "required_product", "delivered_products_before",
        "residual_requirement_before", "selected_action", "selected_model", "selected_product", "selection_reason",
        "predicted_ground_arrival_s", "v2_net_score", "v2_quality", "q_multiplication_count", "reentry_attempt", "residual_registered",
    )} for row in rows]
    return stable_hash(semantic)


def v0_regression(metric: Mapping[str, Any], path: Path) -> dict[str, Any]:
    reference = read_json(REFERENCE_METRICS)
    keys = [
        "J_all_net", "J_HV_net", "gross_utility", "HV_gross_utility", "all_closure_count", "HV_closure_count",
        "high_closure_count", "urgent_closure_count", "controllable_energy_Wh", "transmitted_MB", "terminal_queue_MB",
        "peak_queue_MB", "minimum_soc", "maximum_thermal", "soc_violation_count", "thermal_violation_count",
    ]
    differences = {key: f(metric.get(key)) - f(reference.get(key)) for key in keys}
    metric_pass = all(abs(value) <= 1.0e-10 for value in differences.values())
    reference_trace_hash = decision_semantic_hash(REFERENCE_RUN)
    new_trace_hash = decision_semantic_hash(path)
    audit = {
        "status": "PASS" if metric_pass and reference_trace_hash == new_trace_hash else "FAIL",
        "metric_tolerance": 1.0e-10, "metric_differences": differences,
        "reference_decision_semantic_hash": reference_trace_hash, "new_decision_semantic_hash": new_trace_hash,
        "decision_trace_exact": reference_trace_hash == new_trace_hash,
    }
    return audit


def write_thermal_audit(path: Path) -> None:
    resolved = read_json(TRUSTED_SCENARIO / "resolved_config.json")
    lines = [
        "# Current thermal mechanism audit",
        "",
        "This is a read-only audit of the frozen V0 runtime before the experimental V3 adapter.",
        "",
        "- Mode Controller: **present and enabled**. It switches every 10 slots; `thermal_high=0.8` selects an energy-saving mode.",
        "- Existing Lyapunov thermal coupling: **present but indirect**. `alpha_thermal=0.8` raises the dynamic energy price as normalized thermal pressure rises.",
        "- Independent thermal virtual queue / action-level thermal price: **absent** in V0.",
        "- SafeFilter: thermal is a hard rejection whenever `PredictedMetrics.predicted_Theta_next > thermal_max`.",
        "- Physical predictor: `src/energy/thermal_update.py::update_thermal_pressure`; state and increments are dimensionless normalized thermal pressure, while compute/communication inputs are joules.",
        "- Hard limit: `thermal_max=0.92`; V3 does not change this value or the SafeFilter.",
        "- V3 status: **experimental_no_new_tuning**. It applies only to low/medium actions, averages five normalized monotone pressure signals, and multiplies that mean by the frozen `alpha_thermal=0.8`.",
        "- V3 score: `V2 q-once predicted net score - thermal_price * normalized(action Theta_next - Defer Theta_next)`.",
        "",
        "## Frozen values",
        "",
        f"- thermal_init: {resolved['simulator']['thermal_init']}",
        f"- thermal_max: {resolved['simulator']['thermal_max']}",
        f"- cooling_per_slot: {resolved['simulator']['cooling_per_slot']}",
        f"- passive_cooling: {resolved['simulator']['thermal_passive_cooling']}",
        f"- sunlight_gain: {resolved['simulator']['thermal_sunlight_gain']}",
        f"- eclipse_cooling_bonus: {resolved['simulator']['thermal_eclipse_cooling_bonus']}",
        f"- alpha_thermal: {resolved['lyapunov_pricing']['alpha_thermal']}",
        "",
        "No threshold, q value, resource price, hard margin, or model/product profile was fitted from the experiment outcome.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_safe_mismatch_trace(attempt: Path) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        run = attempt / "runs" / SLUG[method]
        grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"selected": "", "executed": "", "reasons": ""})
        for row in read_csv(run / "candidate_action_log.csv"):
            key = (row["slot"], row["task_id"])
            if i(row.get("selected_flag")):
                grouped[key].update({"selected": row["candidate_action"], "reasons": row.get("safety_rejection_reason", "")})
            if i(row.get("executed_flag")):
                grouped[key]["executed"] = row["candidate_action"]
        for (slot, task_id), row in grouped.items():
            if row["selected"] != row["executed"]:
                output.append({"method": method, "slot": slot, "task_id": task_id, "selector_action": row["selected"], "final_guard_action": row["executed"], "mismatch": 1, "rejection_reasons": row["reasons"]})
        for row in read_csv(run / "unified_guard_trace.csv"):
            if row.get("phase") == "final_guard" and not i(row.get("guard_pass")):
                final_action = row.get("fallback_action", "")
                output.append({"method": method, "slot": row["decision_slot"], "task_id": row["task_id"], "selector_action": row["action"], "final_guard_action": final_action, "mismatch": int(row["action"] != final_action), "rejection_reasons": row.get("guard_reasons", ""), "adapter_reselection": int(row["action"] != final_action), "fallback_success": row.get("fallback_used", 0), "no_fully_safe_action_terminal_defer": int(row["action"] == "delay/store" and final_action == "delay/store")})
    return output


def build_residual_trace(attempt: Path) -> list[dict[str, Any]]:
    causes = {row["task_id"]: row for row in read_csv(ROOT_CAUSES)} if ROOT_CAUSES.is_file() else {}
    output = []
    for method in (V0, V2, V4):
        run = attempt / "runs" / SLUG[method]
        decisions = read_csv(run / "v2_decision_trace.csv")
        candidates = read_csv(run / "v2_candidate_score_audit.csv")
        outcomes = {row["task_id"]: row for row in read_csv(run / "per_task_outcomes.csv")}
        events = read_csv(run / "ground_delivery_events.csv")
        by_decision: dict[str, list[dict[str, str]]] = defaultdict(list)
        by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in decisions:
            by_decision[row["task_id"]].append(row)
        for row in candidates:
            by_candidate[row["task_id"]].append(row)
        delivered_metadata = {row["task_id"] for row in events if row.get("stage", "").lower() == "metadata" and i(row.get("valid_before_deadline"))}
        for task_id, cause in causes.items():
            drows = by_decision.get(task_id, [])
            crows = by_candidate.get(task_id, [])
            metadata_feasible = [row for row in crows if row.get("product") == "metadata" and i(row.get("safe_filter_pass")) and i(row.get("deadline_feasible")) and f(row.get("v2_net_score")) > 1.0e-12]
            metadata_selected = [row for row in drows if row.get("selected_product") == "metadata"]
            outcome = outcomes.get(task_id, {})
            output.append({
                "method": method, "task_id": task_id, "value_level": cause.get("value_level"), "required_product": cause.get("required_product"),
                "V0_primary_failure_reason": cause.get("primary_failure_reason"), "metadata_positive_safe_timely_candidate_count": len(metadata_feasible),
                "metadata_selected_count": len(metadata_selected), "metadata_delivered_before_deadline": int(task_id in delivered_metadata),
                "residual_registered_count": sum(i(row.get("residual_registered")) for row in drows),
                "residual_reentry_decision_count": sum(i(row.get("reentry_attempt")) for row in drows),
                "later_upgrade_selected_count": sum(row.get("selected_product") in {"roi", "roi_context"} and i(row.get("reentry_attempt")) for row in drows),
                "final_complete_closure": i(outcome.get("closure_success")), "final_failure_reason": outcome.get("failure_reason", ""),
            })
    return output


def build_changed_tasks(attempt: Path, ledgers: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    root_causes = {row["task_id"]: row for row in read_csv(ROOT_CAUSES)} if ROOT_CAUSES.is_file() else {}
    cf_thermal = {row["task_id"] for row in read_csv(BALANCED_HIGH_CF) if not i(row.get("high_safe_filter_pass"))} if BALANCED_HIGH_CF.is_file() else set()
    base_run = attempt / "runs" / SLUG[V0]
    base_decisions = read_csv(base_run / "v2_decision_trace.csv")
    base_sequence: dict[str, list[str]] = defaultdict(list)
    for row in base_decisions:
        base_sequence[row["task_id"]].append(row["selected_action"])
    base_outcomes = {row["task_id"]: row for row in read_csv(base_run / "per_task_outcomes.csv")}
    v0_native = read_csv(base_run / "candidate_action_log.csv")
    v0_selected: dict[tuple[str, str], str] = {}
    v0_executed: dict[tuple[str, str], str] = {}
    for row in v0_native:
        key = (row["slot"], row["task_id"])
        if i(row.get("selected_flag")):
            v0_selected[key] = row["candidate_action"]
        if i(row.get("executed_flag")):
            v0_executed[key] = row["candidate_action"]
    v0_mismatch_tasks = {task_id for (slot, task_id), action in v0_selected.items() if v0_executed.get((slot, task_id), action) != action}
    output = []
    frozen = finalgate.load_frozen()
    prices = frozen["public_prices"]
    for method in (V1, V2, V3, V4):
        run = attempt / "runs" / SLUG[method]
        decisions = read_csv(run / "v2_decision_trace.csv")
        candidates = read_csv(run / "v2_candidate_score_audit.csv")
        outcomes = {row["task_id"]: row for row in read_csv(run / "per_task_outcomes.csv")}
        slot_trace = read_csv(run / "per_slot_resource_trace.csv")
        theta_by_slot = {i(row["slot"]): f(row["thermal"]) for row in slot_trace}
        thermal_max = f(read_json(run / "resolved_config.json")["simulator"]["thermal_max"], 0.92)
        sequence: dict[str, list[str]] = defaultdict(list)
        by_candidate: dict[str, list[dict[str, str]]] = defaultdict(list)
        reasons: dict[str, list[str]] = defaultdict(list)
        for row in decisions:
            sequence[row["task_id"]].append(row["selected_action"])
            reasons[row["task_id"]].append(row["selection_reason"])
        for row in candidates:
            by_candidate[row["task_id"]].append(row)
        ledger_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in ledgers[method]:
            ledger_by_task[row["task_id"]].append(row)
        for task_id in sorted(set(base_sequence) | set(sequence)):
            if base_sequence.get(task_id, []) == sequence.get(task_id, []):
                continue
            outcome = outcomes.get(task_id, {})
            base = base_outcomes.get(task_id, {})
            crows = by_candidate.get(task_id, [])
            selected_rows = [row for row in crows if i(row.get("final_executed_choice"))]
            predicted_net = sum(f(row.get("v2_net_score")) for row in selected_rows)
            gross = sum(f(row["gross"]) for row in ledger_by_task.get(task_id, []))
            energy_wh = f(outcome.get("controllable_energy_j")) / 3600.0
            tx_mb = f(outcome.get("transmitted_bytes")) / 1_000_000.0
            actual_net = gross - f(prices["energy_price_per_Wh"]) * energy_wh - f(prices["downlink_price_per_MB"]) * tx_mb
            required = outcome.get("product_requirement", base.get("product_requirement", ""))
            required_norm = gatecore.product_normalize(required)
            required_reachable = any(gatecore.product_normalize(row.get("product")) == required_norm and i(row.get("safe_filter_pass")) and i(row.get("deadline_feasible")) and f(row.get("v2_net_score")) > 0 for row in crows)
            metadata_reachable = any(row.get("product") == "metadata" and i(row.get("safe_filter_pass")) and i(row.get("deadline_feasible")) and f(row.get("v2_net_score")) > 0 for row in crows)
            partial = gross if not i(outcome.get("closure_success")) else 0.0
            cause = root_causes.get(task_id, {})
            changed_slots = [i(row["decision_slot"]) for row in decisions if row["task_id"] == task_id]
            first_changed_slot = min(changed_slots) if changed_slots else 0
            theta = theta_by_slot.get(max(0, first_changed_slot - 1), f(read_json(run / "resolved_config.json")["simulator"].get("thermal_init"), 0.52))
            output.append({
                "method": method, "task_id": task_id, "value_level": outcome.get("value_level", base.get("value_level", "")),
                "requirement": required, "V0_action_sequence": ";".join(base_sequence.get(task_id, [])),
                "experimental_action_sequence": ";".join(sequence.get(task_id, [])), "action_change_reason": ";".join(dict.fromkeys(reasons.get(task_id, []))),
                "required_product_positive_safe_timely_reachable": int(required_reachable), "metadata_positive_safe_timely_reachable": int(metadata_reachable),
                "predicted_net_utility_selected_sum": predicted_net, "actual_task_local_public_net_diagnostic": actual_net,
                "V0_complete_closure": i(base.get("closure_success")), "experimental_complete_closure": i(outcome.get("closure_success")),
                "partial_utility_without_closure": partial, "affects_later_tasks": "potential_resource_displacement_not_causally_identified",
                "current_thermal_at_first_experimental_decision": theta, "thermal_headroom_at_first_experimental_decision": thermal_max - theta,
                "V0_root_cause_group": cause.get("primary_failure_reason", ""),
                "special_4_final_guard_case": int(task_id in v0_mismatch_tasks),
                "special_29_unclosed_hv_evidence": int(task_id in root_causes), "special_23_balanced_to_high_thermal_rejection": int(task_id in cf_thermal),
            })
    return output


def classify(delta_j: float, delta_hv: float, delta_closure: int, delta_partial: float, *, aligned: bool = False) -> str:
    if aligned:
        return "IMPLEMENTATION_ALREADY_ALIGNED"
    if delta_hv > 1.0e-9 and delta_closure > 0 and delta_j >= -1.0e-9:
        return "CLEAR_IMPROVEMENT"
    if delta_closure < 0:
        return "TRADEOFF_ONLY"
    if delta_hv > 1.0e-9 or delta_closure > 0 or delta_partial > 1.0e-9:
        return "MODEST_IMPROVEMENT" if delta_j >= -0.05 else "TRADEOFF_ONLY"
    if abs(delta_j) <= 1.0e-9 and abs(delta_hv) <= 1.0e-9 and delta_closure == 0:
        return "NO_MEANINGFUL_CHANGE"
    return "REGRESSION" if delta_j < -1.0e-9 and delta_hv <= 1.0e-9 else "TRADEOFF_ONLY"


def write_reports(attempt: Path, metrics: Mapping[str, Mapping[str, Any]], ledgers: Mapping[str, list[dict[str, Any]]], regression: Mapping[str, Any], process_rows: list[dict[str, Any]]) -> None:
    base = metrics[V0]
    summary = []
    for method in METHODS:
        row = dict(metrics[method])
        row.update({
            "delta_J_all_net_vs_V0": f(row["J_all_net"]) - f(base["J_all_net"]),
            "delta_J_HV_net_vs_V0": f(row["J_HV_net"]) - f(base["J_HV_net"]),
            "delta_HV_closure_vs_V0": i(row["HV_closure_count"]) - i(base["HV_closure_count"]),
            "delta_HV_partial_unclosed_vs_V0": f(row["HV_partial_gross_utility_unclosed"]) - f(base["HV_partial_gross_utility_unclosed"]),
        })
        aligned = method == V1 and i(row["selector_final_guard_mismatch_count"]) == 0 and i(row["fallback_success_count"]) == 0 and abs(row["delta_J_all_net_vs_V0"]) <= 1.0e-9
        row["classification"] = "BASELINE" if method == V0 else classify(row["delta_J_all_net_vs_V0"], row["delta_J_HV_net_vs_V0"], row["delta_HV_closure_vs_V0"], row["delta_HV_partial_unclosed_vs_V0"], aligned=aligned)
        summary.append(row)
    write_csv(attempt / "five_variant_summary.csv", summary)
    write_csv(attempt / "tables" / "five_variant_summary.csv", summary)
    changed = build_changed_tasks(attempt, ledgers)
    safe = build_safe_mismatch_trace(attempt)
    residual = build_residual_trace(attempt)
    thermal = []
    for method in (V3, V4):
        for row in read_csv(attempt / "runs" / SLUG[method] / "thermal_price_action_trace.csv"):
            thermal.append({"method": method, **row})
    write_csv(attempt / "changed_task_trace.csv", changed)
    write_csv(attempt / "safe_guard_mismatch_trace.csv", safe)
    write_csv(attempt / "hv_metadata_residual_trace.csv", residual)
    write_csv(attempt / "thermal_price_timeseries.csv", thermal)
    write_json(attempt / "audit" / "V0_disabled_regression.json", regression)
    write_json(attempt / "audit" / "process_isolation.json", {
        "status": "PASS" if len({row["pid"] for row in process_rows}) == 5 else "FAIL", "processes": process_rows,
        "V0_gate_then_parallel_V1_to_V4": True, "unique_pid_count": len({row["pid"] for row in process_rows}),
    })
    lines = ["# Metric decomposition", "", "All deltas are paired on the identical seed-2026 frozen ordinary scenario.", ""]
    for row in summary:
        lines.extend([
            f"## {row['method']}", "",
            f"- J_all_net: {f(row['J_all_net']):.12f} (delta {f(row['delta_J_all_net_vs_V0']):+.12f})",
            f"- J_HV_net: {f(row['J_HV_net']):.12f} (delta {f(row['delta_J_HV_net_vs_V0']):+.12f})",
            f"- gross / energy / downlink / backlog costs: {f(row['gross_utility']):.12f} / {f(row['energy_cost']):.12f} / {f(row['downlink_cost']):.12f} / {f(row['terminal_backlog_cost']):.12f}",
            f"- all / HV / high / urgent closure: {i(row['all_closure_count'])} / {i(row['HV_closure_count'])} / {i(row['high_closure_count'])} / {i(row['urgent_closure_count'])}",
            f"- unclosed-HV partial gross utility (diagnostic only): {f(row['HV_partial_gross_utility_unclosed']):.12f}",
            f"- HV-arrival mean thermal headroom / near-limit duration: {f(row['HV_arrival_mean_thermal_headroom']):.12f} / {f(row['thermal_near_limit_duration_s']):.1f} s",
            f"- classification: {row['classification']}", "",
        ])
    lines.extend([
        "The frozen public evaluator includes HV event utility in J_HV only when that task ultimately closes. Therefore Metadata partial utility from an unclosed HV task is disclosed separately and is not silently inserted into J_HV.",
        "",
    ])
    (attempt / "metric_decomposition.md").write_text("\n".join(lines), encoding="utf-8")
    by_method = {row["method"]: row for row in summary}
    v1 = by_method[V1]
    v2 = by_method[V2]
    v3 = by_method[V3]
    v4 = by_method[V4]
    assessment = [
        "# Final assessment", "",
        "## What was already implemented", "",
        "- A: Partly implemented. Pre-filter and final shield share the hard `evaluate_action` predicate, but the empty-safe-set path exposed all actions to the selector and caused four select-then-Defer corrections. V1 closes that consistency gap without changing hard limits.",
        "- B: Partly implemented. Autonomous residual re-entry existed, but a positive safe Metadata alternative was not independently reconsidered after the conditional negative-net guard rejected a required-product action.",
        "- C: Partly implemented. Thermal state already affected Mode Controller, dynamic energy price, physical prediction and hard SafeFilter; no independent action-level predicted-thermal-increment price existed.",
        "",
        "## Paired single-seed findings", "",
        f"- V1: delta J_all={f(v1['delta_J_all_net_vs_V0']):+.12f}, delta J_HV={f(v1['delta_J_HV_net_vs_V0']):+.12f}, delta HV closure={i(v1['delta_HV_closure_vs_V0']):+d}; final mismatches={i(v1['selector_final_guard_mismatch_count'])}, safe fallback successes={i(v1['fallback_success_count'])}. Classification: **{v1['classification']}**.",
        f"- V2: delta J_all={f(v2['delta_J_all_net_vs_V0']):+.12f}, delta J_HV={f(v2['delta_J_HV_net_vs_V0']):+.12f}, delta HV closure={i(v2['delta_HV_closure_vs_V0']):+d}, delta unclosed-HV partial gross={f(v2['delta_HV_partial_unclosed_vs_V0']):+.12f}. Classification: **{v2['classification']}**.",
        f"- V3: delta J_all={f(v3['delta_J_all_net_vs_V0']):+.12f}, delta J_HV={f(v3['delta_J_HV_net_vs_V0']):+.12f}, delta HV closure={i(v3['delta_HV_closure_vs_V0']):+d}, ordinary thermal changes={i(v3['ordinary_thermal_action_change_count'])}, HV-arrival headroom delta={f(v3['HV_arrival_mean_thermal_headroom']) - f(base['HV_arrival_mean_thermal_headroom']):+.12f}, thermal hard rejection delta={i(v3['thermal_hard_rejection_count']) - i(base['thermal_hard_rejection_count']):+d}. Classification: **{v3['classification']}**.",
        f"- V4: delta J_all={f(v4['delta_J_all_net_vs_V0']):+.12f}, delta J_HV={f(v4['delta_J_HV_net_vs_V0']):+.12f}, delta HV closure={i(v4['delta_HV_closure_vs_V0']):+d}; SOC/thermal violations={i(v4['soc_violation_count'])}/{i(v4['thermal_violation_count'])}. Classification: **{v4['classification']}**.",
        "",
        "## Design decision", "",
        "This is a single-seed mechanism exploration, not evidence of statistical superiority. No experimental adapter is promoted into formal PROP automatically. Whether the action-level thermal-price mechanism deserves a later V3 design should be judged from its paired HV headroom/rejection benefit versus ordinary-task utility loss, not from ranking alone.",
        "",
        f"V0 exact regression gate: **{regression['status']}**; decision trace exact: **{regression['decision_trace_exact']}**.",
        "",
        COMPLETION,
    ]
    (attempt / "final_assessment.md").write_text("\n".join(assessment) + "\n", encoding="utf-8")


def write_git_patch(path: Path) -> None:
    import difflib
    chunks = []
    for file in (ROOT / "src" / "experiments" / "safe_metadata_thermal_exploration.py", CONFIG, Path(__file__).resolve()):
        lines = file.read_text(encoding="utf-8").splitlines(keepends=True)
        chunks.extend(difflib.unified_diff([], lines, fromfile="/dev/null", tofile=str(file.relative_to(ROOT)).replace("\\", "/")))
    path.write_text("".join(chunks), encoding="utf-8")


def synthetic_tests(scenario_audit: Mapping[str, Any]) -> list[tuple[str, bool, str]]:
    tests = []
    low = exphelper.thermal_price_snapshot(current_thermal=0.6, defer_theta_next=0.58, thermal_initial=0.52, thermal_max=0.92, recent_compute_pressure=0.1, has_pending_hv_residual=False, alpha_thermal=0.8)
    high = exphelper.thermal_price_snapshot(current_thermal=0.85, defer_theta_next=0.86, thermal_initial=0.52, thermal_max=0.92, recent_compute_pressure=0.8, has_pending_hv_residual=True, alpha_thermal=0.8)
    tests.append(("thermal_price_monotone_pressure", high.thermal_price > low.thermal_price, f"low={low.thermal_price}, high={high.thermal_price}"))
    tests.append(("thermal_increment_zero_for_defer", exphelper.normalized_thermal_increment(action_theta_next=0.7, defer_theta_next=0.7, thermal_initial=0.52, thermal_max=0.92) == 0.0, ""))
    rows = [
        {"model": "lightweight", "product": "roi", "action": "L", "v2_net_score": 2, "unified_guard_pass": 1},
        {"model": "balanced", "product": "roi", "action": "B", "v2_net_score": 1, "unified_guard_pass": 1},
        {"model": "high_accuracy", "product": "metadata", "action": "H-M", "v2_net_score": 9, "unified_guard_pass": 1},
    ]
    chosen = exphelper.choose_ranked_fallback(rows, original_product="roi")
    tests.append(("fallback_tier_same_product_balanced_first", chosen is not None and chosen["action"] == "B", str(chosen)))
    tests.append(("trusted_ordinary_scenario_gate", scenario_audit.get("status") == "PASS" and scenario_audit.get("task_hash") == EXPECTED_TASK_HASH, str(scenario_audit.get("checks"))))
    return tests


def main() -> int:
    attempt = next_attempt()
    pre_hash = hash_snapshot()
    write_json(attempt / "audit" / "pre_run_hashes.json", pre_hash)
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    (attempt / "audit" / "pre_run_git_status.txt").write_text(git_status, encoding="utf-8")
    scenario_audit = normal.scenario_gate(TRUSTED_SCENARIO, attempt / "scenario" / "ordinary_random_seed2026", attempt / "audit")
    write_json(attempt / "scenario_identity_report.json", scenario_audit)
    write_thermal_audit(attempt / "current_thermal_mechanism_audit.md")
    tests = synthetic_tests(scenario_audit)
    if not all(passed for _, passed, _ in tests):
        raise RuntimeError({"SYNTHETIC_TEST_FAILED": tests})
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt) for method in METHODS}

    process_rows = [launch(specs[V0], attempt)]
    v0_metric, v0_ledger = evaluate_run(V0, Path(specs[V0]["output"]))
    regression = v0_regression(v0_metric, Path(specs[V0]["output"]))
    if regression["status"] != "PASS":
        write_json(attempt / "audit" / "V0_disabled_regression.json", regression)
        raise RuntimeError("V0_BASELINE_REGRESSION_MISMATCH")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in (V1, V2, V3, V4)}
        for future in as_completed(futures):
            process_rows.append(future.result())
    metrics: dict[str, dict[str, Any]] = {V0: v0_metric}
    ledgers: dict[str, list[dict[str, Any]]] = {V0: v0_ledger}
    for method in (V1, V2, V3, V4):
        metrics[method], ledgers[method] = evaluate_run(method, Path(specs[method]["output"]))
    post_hash = hash_snapshot()
    hash_audit = {"status": "PASS" if pre_hash == post_hash else "FAIL", "pre": pre_hash, "post": post_hash, "changed": {key: [pre_hash.get(key), post_hash.get(key)] for key in set(pre_hash) | set(post_hash) if pre_hash.get(key) != post_hash.get(key)}}
    write_json(attempt / "audit" / "source_immutability_audit.json", hash_audit)
    if hash_audit["status"] != "PASS":
        raise RuntimeError("SOURCE_IMMUTABILITY_MISMATCH")
    write_reports(attempt, metrics, ledgers, regression, process_rows)
    runtime_tests = [
        ("V0_exact_regression", regression["status"] == "PASS", str(regression)),
        ("five_unique_processes", len({row["pid"] for row in process_rows}) == 5, str(process_rows)),
        ("all_variants_safe", all(i(metrics[m]["soc_violation_count"]) == 0 and i(metrics[m]["thermal_violation_count"]) == 0 for m in METHODS), ""),
        ("source_hashes_unchanged", hash_audit["status"] == "PASS", str(hash_audit["changed"])),
        ("q_once_public", all(f(metrics[m]["q_once_max_recompute_error"]) <= 1.0e-12 for m in METHODS), ""),
    ]
    tests.extend(runtime_tests)
    (attempt / "test_report.txt").write_text("\n".join(f"{'PASS' if passed else 'FAIL'}\t{name}\t{detail}" for name, passed, detail in tests) + "\n", encoding="utf-8")
    if not all(passed for _, passed, _ in tests):
        raise RuntimeError("CONTROLLED_EXPLORATION_TEST_FAILED")
    write_git_patch(attempt / "git_diff.patch")
    write_json(attempt / "completion.json", {"status": "PASS", "marker": COMPLETION, "completed_at": utc_now(), "output": str(attempt)})
    print(f"Unique entry: {Path(__file__).resolve()}")
    print(f"Output: {attempt}")
    print("Execution: V0 gate sequential, then V1-V4 in four independent parallel processes")
    print(COMPLETION)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--child-spec", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
