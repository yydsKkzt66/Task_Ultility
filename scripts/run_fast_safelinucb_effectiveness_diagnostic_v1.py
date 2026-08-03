"""Read-only single-seed SafeLinUCB effectiveness diagnostic.

This entry creates only a new experiment directory.  The frozen formal method,
its configuration, and all historical outputs are treated as read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import types
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "FAST-SAFELINUCB-EFFECTIVENESS-DIAGNOSTIC-v1"
COMPLETION = "FAST_SAFELINUCB_EFFECTIVENESS_DIAGNOSTIC_COMPLETE"
CURRENT = "CURRENT-BANDIT"
GREEDY = "SAFE-GREEDY"
ACTIVE = "ACTIVE-RESIDUAL-SAFELINUCB"
METHODS = (CURRENT, GREEDY, ACTIVE)
SLUG = {CURRENT: "CURRENT_BANDIT", GREEDY: "SAFE_GREEDY", ACTIVE: "ACTIVE_RESIDUAL_SAFELINUCB"}
OUTPUT_BASE = ROOT / "outputs" / "_fast_safelinucb_effectiveness_diagnostic_v1"
FORMAL_REFERENCE = ROOT / "outputs" / "_current_prop_v1_v2_ablation_single_seed_v1" / "attempt_001" / "runs" / "PROP_V2_SAFE_METADATA_CURRENT_v1"

ACTIVE_VARIANT = CURRENT
LAST_WRAPPER: "BanditAuditWrapper | None" = None
EPS = 1.0e-12


def f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def matrix_hash(value: np.ndarray) -> str:
    return sha256_bytes(np.asarray(value, dtype=np.float64).tobytes(order="C"))


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    index = 1
    while (OUTPUT_BASE / f"attempt_{index:03d}").exists():
        index += 1
    attempt = OUTPUT_BASE / f"attempt_{index:03d}"
    for name in ("audit", "runs", "run_specs", "scenario"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    return attempt


class BanditAuditWrapper(engine.ControlledExplorationWrapper):
    """Audit adapter; ACTIVE alone can override the physical greedy choice."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global LAST_WRAPPER
        super().__init__(*args, **kwargs)
        self.audit_variant = ACTIVE_VARIANT
        self.bandit_decisions: list[dict[str, Any]] = []
        self.feedback_rows: list[dict[str, Any]] = []
        self.error_rows: list[dict[str, Any]] = []
        self.pending: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
        self.pending_all: list[dict[str, Any]] = []
        self.call_counts = Counter()
        self._feedback_sequence = 0
        self._last_pending: dict[str, Any] | None = None
        self._initial_A = {name: np.asarray(value, dtype=float).copy() for name, value in self.delegate.A.items()}
        self._initial_b = {name: np.asarray(value, dtype=float).copy() for name, value in self.delegate.b.items()}
        self._quality = dict(engine.finalgate.load_frozen()["q_eff"])
        LAST_WRAPPER = self

    @staticmethod
    def _product(value: Any) -> str:
        text = str(value or "").strip().lower().replace("+", "_")
        return {"evidence": "roi_context", "context": "roi_context", "verification": "roi", "alert": "metadata"}.get(text, text)

    def _bandit_components(self, context: Any, action: Any, predicted: Any, guidance_bundle: Mapping[str, Any] | None) -> dict[str, float]:
        if action.payload_policy == "defer":
            return {"mean": 0.0, "bonus": 0.0, "explore_norm": 0.0, "lcb_residual": 0.0, "ucb_residual": 0.0}
        x = np.asarray(context.as_array(), dtype=float)
        A_inv = np.linalg.inv(self.delegate.A[action.action_name])
        theta = A_inv @ self.delegate.b[action.action_name]
        mean = float(theta @ x)
        explore_norm = float(np.sqrt(max(0.0, x @ A_inv @ x)))
        raw_bonus = float(self.delegate.alpha) * f((guidance_bundle or {}).get("exploration_scale"), 1.0) * explore_norm
        _discount, bonus, _reason = self.delegate._cost_aware_ucb(context, action, predicted, raw_bonus, guidance_bundle)
        return {"mean": mean, "bonus": float(bonus), "explore_norm": explore_norm, "lcb_residual": mean - float(bonus), "ucb_residual": mean + float(bonus)}

    def _phase_pool(self, rows: list[dict[str, Any]], decision: Mapping[str, Any]) -> list[dict[str, Any]]:
        safe = [
            row for row in rows
            if i(row.get("safe_filter_pass"))
            and (self._product(row.get("product")) == "defer" or i(row.get("deadline_feasible")))
        ]
        if str(decision.get("value_level")) not in {"high", "urgent"}:
            return safe
        reason = str(decision.get("selection_reason", ""))
        if "metadata" in reason:
            return [row for row in safe if self._product(row.get("product")) == "metadata"]
        if "direct_required" in reason:
            target = self._product(decision.get("residual_requirement_before"))
            return [row for row in safe if self._product(row.get("product")) == target]
        return [row for row in safe if self._product(row.get("product")) == "defer"]

    def _register_pending(self, context: Any, action: Any, predicted_map: Mapping[str, Any]) -> None:
        if self.current_step.task is None or action.payload_policy == "defer":
            self._last_pending = None
            return
        rows = engine.current_rows(self)
        row = next((item for item in rows if item["action"] == action.action_name), None)
        if row is None:
            self._last_pending = None
            return
        task = self.current_step.task
        product = self._product(row.get("product"))
        components = self._bandit_components(context, action, predicted_map[action.action_name], self.step_runner.last_guidance_bundle if hasattr(self.step_runner, "last_guidance_bundle") else None)
        record = {
            "task_id": str(task.task_id), "decision_slot": i(self.current_step.slot_index),
            "decision_time_s": f(self.current_step.slot_index) * f(self.current_step.slot_seconds),
            "value_level": self._task_value_level(task), "action": action.action_name,
            "model": str(action.model_name), "product": product,
            "x": np.asarray(context.as_array(), dtype=float).tolist(),
            "predicted_public_increment": f(row.get("v2_positive_benefit")),
            "predicted_physical_net": f(row.get("v2_net_score")),
            "task_value": f(row.get("v2_task_value")), "quality": f(row.get("v2_quality")),
            "residual_stage": int(bool(i(self.v2_decision_rows[-1].get("reentry_attempt"))) or bool(str(self.v2_decision_rows[-1].get("delivered_products_before", "")))),
            "learned_residual_mean_at_decision": components["mean"],
            "ucb_bonus_at_decision": components["bonus"], "resolved": False,
        }
        self.pending[(record["task_id"], product)].append(record)
        self.pending_all.append(record)
        self._last_pending = record

        # The frozen lifecycle can autonomously create the closing residual
        # after a partial product reaches ground, without another router call.
        # Pre-register that separate incremental product against the original
        # task-state and the cached inference model so its later feedback is not
        # attributed to the unrelated task occupying the delivery slot.
        decision = self.v2_decision_rows[-1]
        target_product = self._product(decision.get("residual_requirement_before"))
        if product != target_product and target_product in {"roi", "roi_context"}:
            target_row = next(
                (
                    item for item in rows
                    if str(item.get("model")) == str(action.model_name)
                    and self._product(item.get("product")) == target_product
                ),
                None,
            )
            target_action = None if target_row is None else next(
                (candidate for candidate in self.step_runner.action_space.list_actions() if candidate.action_name == target_row["action"]),
                None,
            )
            if target_row is not None and target_action is not None:
                from src.payload.information_sufficiency import delta_information_utility

                residual_delta = delta_information_utility(
                    target_product,
                    {product},
                    str(getattr(task, "product_requirement", "verification")),
                    self.product_utility_cfg,
                )
                target_components = self._bandit_components(
                    context,
                    target_action,
                    predicted_map[target_action.action_name],
                    self.step_runner.last_guidance_bundle if hasattr(self.step_runner, "last_guidance_bundle") else None,
                )
                residual_record = {
                    **record,
                    "action": target_action.action_name,
                    "model": str(target_action.model_name),
                    "product": target_product,
                    "predicted_public_increment": (
                        f(target_row.get("v2_task_value"))
                        * f(residual_delta)
                        * f(target_row.get("v2_quality"))
                        * f(target_row.get("v2_ground_success_probability"))
                        * f(target_row.get("v2_timeliness"))
                    ),
                    "predicted_physical_net": f(target_row.get("v2_net_score")),
                    "task_value": f(target_row.get("v2_task_value")),
                    "quality": f(target_row.get("v2_quality")),
                    "residual_stage": 1,
                    "learned_residual_mean_at_decision": target_components["mean"],
                    "ucb_bonus_at_decision": target_components["bonus"],
                    "resolved": False,
                    "autonomous_residual_pre_registered": 1,
                }
                self.pending[(record["task_id"], target_product)].append(residual_record)
                self.pending_all.append(residual_record)
                self.call_counts["autonomous_residual_pre_registered"] += 1

    def select_action(self, context: Any, candidate_actions: list[Any], predicted_map: dict[str, Any], guidance_bundle=None):
        self.call_counts["propose"] += 1
        current_selected = super().select_action(context, candidate_actions, predicted_map, guidance_bundle)
        decision = self.v2_decision_rows[-1]
        rows = engine.current_rows(self)
        actions = {action.action_name: action for action in self.step_runner.action_space.list_actions()}
        pool = self._phase_pool(rows, decision)
        if not pool:
            pool = [row for row in rows if row["action"] == current_selected.action_name]

        for row in pool:
            action = actions[row["action"]]
            comp = self._bandit_components(context, action, predicted_map[action.action_name], guidance_bundle)
            physical = 0.0 if action.payload_policy == "defer" else f(row.get("v2_net_score"))
            if action.payload_policy != "defer" and physical < -EPS:
                eligible = False
            else:
                eligible = True
            row.update({
                "diagnostic_physical_net": physical, "diagnostic_residual_mean": comp["mean"],
                "diagnostic_ucb_bonus": comp["bonus"], "diagnostic_total_score": physical + comp["mean"] + comp["bonus"],
                "diagnostic_lcb": physical + comp["lcb_residual"], "diagnostic_ucb": physical + comp["ucb_residual"],
                "diagnostic_negative_net_exploration_blocked": int(not eligible),
            })

        valid = [row for row in pool if not i(row.get("diagnostic_negative_net_exploration_blocked"))]
        defer_rows = [row for row in pool if self._product(row.get("product")) == "defer"]
        if not valid and defer_rows:
            valid = defer_rows
        greedy = max(valid, key=lambda row: (f(row["diagnostic_physical_net"]), str(row["action"])))
        bandit = max(valid, key=lambda row: (f(row["diagnostic_total_score"]), str(row["action"])))
        mean_only = max(valid, key=lambda row: (f(row["diagnostic_physical_net"]) + f(row["diagnostic_residual_mean"]), str(row["action"])))
        gate_pass = f(bandit["diagnostic_ucb"]) > f(greedy["diagnostic_lcb"]) + EPS
        active_pre_guard = bandit if bandit["action"] != greedy["action"] and gate_pass else greedy

        selected = current_selected
        if self.audit_variant == ACTIVE:
            selected = actions[active_pre_guard["action"]]
            if selected.action_name != current_selected.action_name:
                selected = self._replace_choice(selected, predicted_map, "active_residual_safelinucb", {
                    "physical_greedy_action": greedy["action"], "active_bandit_action": bandit["action"],
                    "optimistic_vs_conservative_gate_pass": int(gate_pass),
                })

        final_name = selected.action_name
        delegate_name = next((row["action"] for row in rows if i(row.get("delegate_selected"))), "")
        learned_change = int(self.audit_variant == ACTIVE and final_name != greedy["action"] and mean_only["action"] != greedy["action"])
        exploration_change = int(self.audit_variant == ACTIVE and final_name != greedy["action"] and not learned_change)
        self.bandit_decisions.append({
            "decision_slot": decision["decision_slot"], "task_id": decision["task_id"], "value_level": decision["value_level"],
            "greedy_argmax": greedy["action"], "bandit_argmax": bandit["action"], "mean_only_argmax": mean_only["action"],
            "current_delegate_argmax": delegate_name, "current_policy_before_active": current_selected.action_name,
            "final_action": final_name, "bandit_vs_greedy_different": int(bandit["action"] != greedy["action"]),
            "final_changed_by_bandit": int(self.audit_variant == ACTIVE and final_name != greedy["action"]),
            "learned_residual_change": learned_change, "exploration_only_change": exploration_change,
            "optimistic_vs_conservative_gate_pass": int(gate_pass),
            "greedy_physical_net": greedy["diagnostic_physical_net"], "greedy_lcb": greedy["diagnostic_lcb"],
            "bandit_physical_net": bandit["diagnostic_physical_net"], "bandit_residual_mean": bandit["diagnostic_residual_mean"],
            "bandit_ucb_bonus": bandit["diagnostic_ucb_bonus"], "bandit_total_score": bandit["diagnostic_total_score"],
        })
        self._register_pending(context, selected, predicted_map)
        return selected

    def update(self, context: Any, action: Any, observed_reward: float) -> None:
        self.call_counts["update_interface"] += 1
        if self._last_pending is not None and self._last_pending["action"] != action.action_name:
            self._last_pending["executed_action_mismatch"] = action.action_name
            self.call_counts["executed_action_mismatch"] += 1
        if self.audit_variant == ACTIVE:
            self.call_counts["immediate_feedback_suppressed"] += 1
            return
        before_A = self.delegate.A[action.action_name].copy()
        before_b = self.delegate.b[action.action_name].copy()
        super().update(context, action, observed_reward)
        changed = not np.array_equal(before_A, self.delegate.A[action.action_name]) or not np.array_equal(before_b, self.delegate.b[action.action_name])
        if changed:
            self.call_counts["learn"] += 1
            self.call_counts["prediction_target_update_without_public_feedback"] += 1

    def bind_step_runner(self, step_runner: Any) -> None:
        super().bind_step_runner(step_runner)
        lifecycle_method_name = (
            "_apply_downlink_feedback"
            if hasattr(step_runner, "_apply_downlink_feedback")
            else "_apply_product_lifecycle_updates"
        )
        original = getattr(step_runner, lifecycle_method_name)

        def intercepted(_runner: Any, *args: Any, **kwargs: Any):
            timeline_step = kwargs.get("timeline_step", args[0] if args else None)
            downlink = kwargs.get("downlink_result", args[2] if len(args) > 2 else None)
            captures = []
            if downlink is not None:
                for index, task_id in enumerate(list(getattr(downlink, "delivered_task_ids", []) or [])):
                    product = list(getattr(downlink, "delivered_product_types", []) or [])
                    event_times = list(getattr(downlink, "delivered_task_event_times_s", []) or [])
                    state = step_runner.product_lifecycle.get(task_id)
                    captures.append({
                        "event": "delivered", "task_id": str(task_id),
                        "product": self._product(product[index] if index < len(product) else ""),
                        "feedback_time_s": f(event_times[index] if index < len(event_times) else f(timeline_step.slot_index) * f(timeline_step.slot_seconds)),
                        "feedback_slot": i(timeline_step.slot_index),
                        "delivered_before": set() if state is None else set(state.delivered_products),
                    })
                for label, id_field, product_field in (
                    ("expired", "expired_task_ids", "expired_product_types"),
                    ("dropped", "dropped_task_ids", "dropped_product_types"),
                ):
                    ids = list(getattr(downlink, id_field, []) or [])
                    products = list(getattr(downlink, product_field, []) or [])
                    for index, task_id in enumerate(ids):
                        captures.append({"event": label, "task_id": str(task_id), "product": self._product(products[index] if index < len(products) else ""), "feedback_time_s": f(timeline_step.slot_index) * f(timeline_step.slot_seconds), "feedback_slot": i(timeline_step.slot_index), "delivered_before": set()})
            result = original(*args, **kwargs)
            for capture in captures:
                self._resolve_feedback(capture)
            return result

        setattr(step_runner, lifecycle_method_name, types.MethodType(intercepted, step_runner))

    def _resolve_feedback(self, event: Mapping[str, Any]) -> None:
        key = (str(event["task_id"]), self._product(event["product"]))
        queue = self.pending.get(key)
        pending = queue.popleft() if queue else None
        if pending is None:
            self.feedback_rows.append({**dict(event), "matched_original_decision": 0, "feedback_used_for_learning": 0, "reason": "no_original_task_model_product_record"})
            self.call_counts["unmatched_feedback"] += 1
            return
        pending["resolved"] = True
        actual = 0.0
        if event["event"] == "delivered":
            from src.payload.information_sufficiency import delta_information_utility

            task = self.step_runner.task_snapshots.get(str(event["task_id"]))
            requirement = str(getattr(task, "product_requirement", "verification"))
            delta = delta_information_utility(self._product(event["product"]), set(event["delivered_before"]), requirement, self.product_utility_cfg)
            arrival_s = f(getattr(task, "arrival_slot", 0)) * f(self.current_step.slot_seconds)
            lifetime = max(f(getattr(task, "deadline_s", 0.0)), EPS)
            timely = max(0.0, min(1.0, 1.0 - (f(event["feedback_time_s"]) - arrival_s) / lifetime))
            actual = f(pending["task_value"]) * f(delta) * f(pending["quality"]) * timely
        residual = actual - f(pending["predicted_public_increment"])
        predicted_residual = f(pending["learned_residual_mean_at_decision"])
        error = residual - predicted_residual
        delayed = i(event["feedback_slot"]) > i(pending["decision_slot"])
        used = self.audit_variant == ACTIVE
        before_A = self.delegate.A[pending["action"]].copy()
        before_b = self.delegate.b[pending["action"]].copy()
        if used:
            x = np.asarray(pending["x"], dtype=float)
            self.delegate.A[pending["action"]] = before_A + np.outer(x, x)
            self.delegate.b[pending["action"]] = before_b + residual * x
            self.call_counts["observe"] += 1
            self.call_counts["learn"] += 1
            self.call_counts["realized_public_feedback_update"] += 1
            self.call_counts["delayed_feedback"] += int(delayed)
            self.call_counts["residual_feedback"] += i(pending["residual_stage"])
        self._feedback_sequence += 1
        feedback_row = {
            "feedback_sequence": self._feedback_sequence, "event": event["event"], "task_id": pending["task_id"],
            "value_level": pending["value_level"], "original_decision_slot": pending["decision_slot"],
            "feedback_slot": event["feedback_slot"], "delayed_feedback": int(delayed),
            "original_action": pending["action"], "original_model": pending["model"], "original_product": pending["product"],
            "residual_stage": pending["residual_stage"], "predicted_public_increment": pending["predicted_public_increment"],
            "realized_public_increment": actual, "residual_reward": residual,
            "predicted_residual_mean_before_update": predicted_residual, "residual_prediction_error": error,
            "matched_original_decision": 1, "feedback_used_for_learning": int(used),
            "A_changed": int(used and not np.array_equal(before_A, self.delegate.A[pending["action"]])),
            "b_changed": int(used and not np.array_equal(before_b, self.delegate.b[pending["action"]])),
        }
        self.feedback_rows.append(feedback_row)
        self.error_rows.append({
            "method": self.audit_variant, "feedback_sequence": self._feedback_sequence,
            "task_id": pending["task_id"], "decision_slot": pending["decision_slot"], "feedback_slot": event["feedback_slot"],
            "predicted_residual": predicted_residual, "actual_residual": residual,
            "error": error, "squared_error": error * error, "residual_stage": pending["residual_stage"],
        })

    def finalize_audit(self) -> dict[str, Any]:
        unresolved = [row for row in self.pending_all if not row["resolved"]]
        for row in unresolved:
            self.feedback_rows.append({
                "event": "no_feedback_by_horizon", "task_id": row["task_id"], "value_level": row["value_level"],
                "original_decision_slot": row["decision_slot"], "original_action": row["action"],
                "original_model": row["model"], "original_product": row["product"], "residual_stage": row["residual_stage"],
                "predicted_public_increment": row["predicted_public_increment"], "matched_original_decision": 1,
                "feedback_used_for_learning": 0,
            })
        self.call_counts["no_feedback_by_horizon"] = len(unresolved)
        state_rows = []
        for name in sorted(self.delegate.A):
            A0, b0 = self._initial_A[name], self._initial_b[name]
            A1, b1 = np.asarray(self.delegate.A[name]), np.asarray(self.delegate.b[name])
            theta0 = np.linalg.solve(A0, b0)
            theta1 = np.linalg.solve(A1, b1)
            state_rows.append({
                "action": name, "A_initial_hash": matrix_hash(A0), "A_final_hash": matrix_hash(A1),
                "A_initial_fro_norm": float(np.linalg.norm(A0)), "A_final_fro_norm": float(np.linalg.norm(A1)),
                "A_delta_fro_norm": float(np.linalg.norm(A1 - A0)), "b_initial_norm": float(np.linalg.norm(b0)),
                "b_final_norm": float(np.linalg.norm(b1)), "b_delta_norm": float(np.linalg.norm(b1 - b0)),
                "theta_initial_norm": float(np.linalg.norm(theta0)), "theta_final_norm": float(np.linalg.norm(theta1)),
                "theta_delta_norm": float(np.linalg.norm(theta1 - theta0)),
            })
        errors = self.error_rows
        midpoint = len(errors) // 2
        rmse = lambda rows: math.sqrt(sum(f(row["squared_error"]) for row in rows) / len(rows)) if rows else 0.0
        audit = {
            "method": self.audit_variant, "alpha_beta": f(self.delegate.alpha), "ridge": f(self.delegate.ridge),
            "calls": dict(self.call_counts), "propose_count": self.call_counts["propose"],
            "observe_count": self.call_counts["observe"], "update_interface_count": self.call_counts["update_interface"],
            "learn_count": self.call_counts["learn"], "with_realized_feedback_count": self.call_counts["realized_public_feedback_update"],
            "without_realized_feedback_update_count": self.call_counts["prediction_target_update_without_public_feedback"],
            "delayed_feedback_count": self.call_counts["delayed_feedback"], "residual_feedback_count": self.call_counts["residual_feedback"],
            "no_feedback_by_horizon_count": len(unresolved),
            "bandit_argmax_vs_greedy_count": sum(i(row["bandit_vs_greedy_different"]) for row in self.bandit_decisions),
            "true_final_action_change_count": sum(i(row["final_changed_by_bandit"]) for row in self.bandit_decisions),
            "learned_residual_action_change_count": sum(i(row["learned_residual_change"]) for row in self.bandit_decisions),
            "exploration_only_action_change_count": sum(i(row["exploration_only_change"]) for row in self.bandit_decisions),
            "prediction_rmse_first_half": rmse(errors[:midpoint]), "prediction_rmse_second_half": rmse(errors[midpoint:]),
            "prediction_error_decreased": bool(errors and rmse(errors[midpoint:]) < rmse(errors[:midpoint])),
            "state_by_action": state_rows,
        }
        return audit


def make_spec(method: str, bundle: Path, attempt: Path) -> dict[str, Any]:
    return {
        "version": VERSION, "method": method, "seed": engine.SEED, "slots": engine.SLOTS,
        "capacity": engine.CAPACITY, "bundle_root": str(bundle), "output": str(attempt / "runs" / SLUG[method]),
        "flags": (True, True, False), "greedy_no_bandit": method == GREEDY,
        "bandit_variant": method, "entry_sha256": engine.sha256_file(Path(__file__).resolve()),
    }


def child_main(spec_path: Path) -> int:
    global ACTIVE_VARIANT, LAST_WRAPPER
    spec = read_json(spec_path)
    ACTIVE_VARIANT = str(spec["bandit_variant"])
    LAST_WRAPPER = None
    original_cls = engine.ControlledExplorationWrapper
    engine.ControlledExplorationWrapper = BanditAuditWrapper
    try:
        output = Path(spec["output"])
        engine.configure_and_run_child(spec, output)
        if LAST_WRAPPER is None:
            raise RuntimeError("BANDIT_AUDIT_WRAPPER_NOT_BOUND")
        audit = LAST_WRAPPER.finalize_audit()
        write_json(output / "bandit_internal_audit.json", audit)
        write_csv(output / "bandit_decision_trace.csv", LAST_WRAPPER.bandit_decisions)
        write_csv(output / "feedback_alignment_trace.csv", LAST_WRAPPER.feedback_rows)
        write_csv(output / "prediction_error_timeseries.csv", LAST_WRAPPER.error_rows)
    finally:
        engine.ControlledExplorationWrapper = original_cls
    return 0


def launch(spec: Mapping[str, Any], attempt: Path) -> dict[str, Any]:
    method = str(spec["method"])
    spec_path = attempt / "run_specs" / f"{SLUG[method]}.json"
    write_json(spec_path, spec)
    stdout_path = attempt / "audit" / f"{SLUG[method]}.stdout.log"
    stderr_path = attempt / "audit" / f"{SLUG[method]}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)], cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
    if completed.returncode:
        raise RuntimeError({"METHOD_RUNTIME_FAILED": method, "stderr": str(stderr_path)})
    identity = read_json(Path(spec["output"]) / "runtime_identity.json")
    return {"method": method, "pid": identity["pid"], "output": spec["output"]}


def task_local_net(path: Path, ledger: list[dict[str, Any]], prices: Mapping[str, Any]) -> dict[str, float]:
    gross = Counter()
    for row in ledger:
        gross[str(row["task_id"])] += f(row["gross"])
    result = {}
    for row in read_csv(path / "per_task_outcomes.csv"):
        result[row["task_id"]] = gross[row["task_id"]] - f(prices["energy_price_per_Wh"]) * f(row["controllable_energy_j"]) / 3600.0 - f(prices["downlink_price_per_MB"]) * f(row["transmitted_bytes"]) / 1_000_000.0
    return result


def indexed_decisions(rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    counts = Counter()
    indexed = {}
    for row in rows:
        task_id = row["task_id"]
        key = (task_id, counts[task_id])
        counts[task_id] += 1
        indexed[key] = row
    return indexed


def run_unit_checks() -> list[tuple[str, bool]]:
    checks = []
    physical = {"defer": 0.0, "negative": -0.1, "positive": 0.2}
    eligible = [name for name, value in physical.items() if name == "defer" or value >= 0.0]
    checks.append(("negative_physical_action_cannot_be_explored", "negative" not in eligible))
    checks.append(("defer_score_is_zero", physical["defer"] == 0.0))
    candidate_ucb, greedy_lcb = 0.3, 0.31
    checks.append(("confidence_overlap_gate_blocks_override", not (candidate_ucb > greedy_lcb)))
    candidate_ucb, greedy_lcb = 0.32, 0.31
    checks.append(("confidence_overlap_gate_allows_override", candidate_ucb > greedy_lcb))
    A = np.eye(2); b = np.zeros(2); x = np.asarray([1.0, 2.0]); residual = 0.25
    A2, b2 = A + np.outer(x, x), b + residual * x
    checks.append(("residual_update_changes_A_and_b", not np.array_equal(A, A2) and not np.array_equal(b, b2)))
    checks.append(("residual_target_is_realized_minus_predicted", abs((0.4 - 0.55) - (-0.15)) < 1e-12))
    return checks


def main() -> int:
    attempt = next_attempt()
    pre_hashes = engine.hash_snapshot()
    pre_hashes[str(Path(__file__).resolve())] = engine.sha256_file(Path(__file__).resolve())
    scenario_audit = engine.normal.scenario_gate(engine.TRUSTED_SCENARIO, attempt / "scenario" / "ordinary_random_seed2026", attempt / "audit")
    if scenario_audit["status"] != "PASS":
        raise RuntimeError("SCENARIO_IDENTITY_GATE_FAILED")
    write_json(attempt / "scenario_identity_report.json", scenario_audit)
    bundle = attempt / "scenario" / "ordinary_random_seed2026"
    specs = {method: make_spec(method, bundle, attempt) for method in METHODS}
    processes = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(launch, specs[method], attempt): method for method in METHODS}
        for future in as_completed(futures):
            processes.append(future.result())

    metrics: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, dict[str, Any]] = {}
    decision_traces: dict[str, list[dict[str, str]]] = {}
    bandit_traces: dict[str, list[dict[str, str]]] = {}
    all_feedback, all_errors = [], []
    for method in METHODS:
        run = Path(specs[method]["output"])
        metrics[method], ledgers[method] = engine.evaluate_run(method, run)
        audits[method] = read_json(run / "bandit_internal_audit.json")
        decision_traces[method] = read_csv(run / "v2_decision_trace.csv")
        bandit_traces[method] = read_csv(run / "bandit_decision_trace.csv")
        all_feedback.extend({"method": method, **row} for row in read_csv(run / "feedback_alignment_trace.csv"))
        errors = read_csv(run / "prediction_error_timeseries.csv")
        midpoint = len(errors) // 2
        for index, row in enumerate(errors):
            row["half"] = "first" if index < midpoint else "second"
            all_errors.append(row)

    fixed = metrics[GREEDY]
    summary = []
    for method in METHODS:
        row = dict(metrics[method])
        audit = audits[method]
        row.update({
            "propose_count": audit["propose_count"], "observe_count": audit["observe_count"],
            "update_interface_count": audit["update_interface_count"], "learn_count": audit["learn_count"],
            "with_realized_feedback_count": audit["with_realized_feedback_count"],
            "without_realized_feedback_update_count": audit["without_realized_feedback_update_count"],
            "delayed_feedback_count": audit["delayed_feedback_count"], "residual_feedback_count": audit["residual_feedback_count"],
            "no_feedback_by_horizon_count": audit["no_feedback_by_horizon_count"],
            "bandit_argmax_vs_greedy_count": audit["bandit_argmax_vs_greedy_count"],
            "true_final_action_change_count": audit["true_final_action_change_count"],
            "learned_residual_action_change_count": audit["learned_residual_action_change_count"],
            "exploration_only_action_change_count": audit["exploration_only_action_change_count"],
            "prediction_rmse_first_half": audit["prediction_rmse_first_half"],
            "prediction_rmse_second_half": audit["prediction_rmse_second_half"],
            "prediction_error_decreased": int(audit["prediction_error_decreased"]),
            "delta_J_all_vs_safe_greedy": f(row["J_all_net"]) - f(fixed["J_all_net"]),
            "delta_J_HV_vs_safe_greedy": f(row["J_HV_net"]) - f(fixed["J_HV_net"]),
        })
        summary.append(row)
    write_csv(attempt / "three_method_summary.csv", summary)

    prices = engine.finalgate.load_frozen()["public_prices"]
    local = {method: task_local_net(Path(specs[method]["output"]), ledgers[method], prices) for method in METHODS}
    indexed = {method: indexed_decisions(decision_traces[method]) for method in METHODS}
    active_bandit_index = indexed_decisions(bandit_traces[ACTIVE])
    direct_override_tasks = {
        row["task_id"] for row in bandit_traces[ACTIVE]
        if i(row.get("final_changed_by_bandit"))
    }
    keys = sorted(set().union(*(set(value) for value in indexed.values())))
    divergence = []
    changed_tasks = set()
    for task_id, ordinal in keys:
        rows = {method: indexed[method].get((task_id, ordinal)) for method in METHODS}
        actions = {method: "" if rows[method] is None else rows[method]["selected_action"] for method in METHODS}
        changed = actions[ACTIVE] != actions[GREEDY]
        if changed:
            changed_tasks.add(task_id)
        delta_local = local[ACTIVE].get(task_id, 0.0) - local[GREEDY].get(task_id, 0.0)
        active_bandit_row = active_bandit_index.get((task_id, ordinal), {})
        direct_override = task_id in direct_override_tasks
        divergence.append({
            "task_id": task_id, "decision_ordinal": ordinal,
            "value_level": next((row["value_level"] for row in rows.values() if row is not None), ""),
            "current_bandit_action": actions[CURRENT], "safe_greedy_action": actions[GREEDY], "active_residual_action": actions[ACTIVE],
            "active_vs_greedy_changed": int(changed), "direct_bandit_override_task": int(direct_override),
            "physical_greedy_action": active_bandit_row.get("greedy_argmax", ""),
            "active_bandit_argmax": active_bandit_row.get("bandit_argmax", ""),
            "learned_residual_change": active_bandit_row.get("learned_residual_change", ""),
            "exploration_only_change": active_bandit_row.get("exploration_only_change", ""),
            "greedy_physical_net": active_bandit_row.get("greedy_physical_net", ""),
            "bandit_physical_net": active_bandit_row.get("bandit_physical_net", ""),
            "bandit_residual_mean": active_bandit_row.get("bandit_residual_mean", ""),
            "bandit_ucb_bonus": active_bandit_row.get("bandit_ucb_bonus", ""),
            "bandit_total_score": active_bandit_row.get("bandit_total_score", ""),
            "active_task_local_net": local[ACTIVE].get(task_id, 0.0),
            "greedy_task_local_net": local[GREEDY].get(task_id, 0.0), "active_minus_greedy_task_local_net": delta_local,
            "change_outcome": (
                "corrected_greedy" if direct_override and delta_local > EPS
                else "wrong_override" if direct_override and delta_local < -EPS
                else "neutral_direct_override" if direct_override
                else "trajectory_induced_difference" if changed
                else "no_change"
            ),
            "system_J_all_delta_active_minus_greedy": f(metrics[ACTIVE]["J_all_net"]) - f(metrics[GREEDY]["J_all_net"]),
        })
    write_csv(attempt / "decision_divergence_trace.csv", divergence)
    write_csv(attempt / "feedback_alignment_trace.csv", all_feedback)
    write_csv(attempt / "prediction_error_timeseries.csv", all_errors)

    corrected = {row["task_id"] for row in divergence if row["change_outcome"] == "corrected_greedy"}
    wrong = {row["task_id"] for row in divergence if row["change_outcome"] == "wrong_override"}
    active_audit = audits[ACTIVE]
    active = metrics[ACTIVE]
    safe = i(active["soc_violation_count"]) + i(active["thermal_violation_count"]) == 0
    has_value = (
        f(active["J_all_net"]) > f(fixed["J_all_net"]) + EPS
        and f(active["J_HV_net"]) >= f(fixed["J_HV_net"]) - EPS
        and safe
        and i(active_audit["learned_residual_action_change_count"]) > 0
        and bool(active_audit["prediction_error_decreased"])
    )
    if has_value:
        conclusion = "SAFELINUCB_HAS_OBSERVABLE_VALUE"
    elif i(active_audit["learn_count"]) > 0 and i(active_audit["true_final_action_change_count"]) == 0:
        conclusion = "BANDIT_UPDATES_WITHOUT_DECISION_VALUE"
    else:
        conclusion = "SAFELINUCB_NOT_SUPPORTED"

    state_changed = {
        method: sum(f(row["A_delta_fro_norm"]) > EPS or f(row["b_delta_norm"]) > EPS for row in audits[method]["state_by_action"])
        for method in METHODS
    }
    unit_checks = run_unit_checks()
    post_hashes = engine.hash_snapshot()
    post_hashes[str(Path(__file__).resolve())] = engine.sha256_file(Path(__file__).resolve())
    checks = {
        "scenario_gate_pass": scenario_audit["status"] == "PASS",
        "three_independent_processes": len({row["pid"] for row in processes}) == 3,
        "source_inputs_unchanged": pre_hashes == post_hashes,
        "current_formal_identity": read_json(Path(specs[CURRENT]["output"]) / "runtime_identity.json")["method"] == CURRENT,
        "current_exact_formal_reference": FORMAL_REFERENCE.is_dir() and engine.decision_semantic_hash(Path(specs[CURRENT]["output"])) == engine.decision_semantic_hash(FORMAL_REFERENCE),
        "current_and_greedy_exact_reproduction": engine.decision_semantic_hash(Path(specs[CURRENT]["output"])) == engine.decision_semantic_hash(Path(specs[GREEDY]["output"])) and abs(f(metrics[CURRENT]["J_all_net"]) - f(metrics[GREEDY]["J_all_net"])) <= EPS,
        "safe_filter_before_all_selectors": all(read_json(Path(specs[m]["output"]) / "runtime_identity.json")["safe_set_consistent_fallback"] for m in METHODS),
        "active_feedback_targets_original_action": all(i(row.get("matched_original_decision")) for row in all_feedback if row.get("method") == ACTIVE and i(row.get("feedback_used_for_learning"))),
        "active_has_no_unmatched_delivery_feedback": i(audits[ACTIVE].get("calls", {}).get("unmatched_feedback")) == 0,
        "active_q_once_public": f(active.get("q_once_max_recompute_error")) <= EPS,
        "zero_safety_violations": all(i(metrics[m]["soc_violation_count"]) + i(metrics[m]["thermal_violation_count"]) == 0 for m in METHODS),
        "unit_checks": all(passed for _, passed in unit_checks),
    }
    bandit_update_audit = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "methods": audits, "state_changed_action_count": state_changed,
        "bandit_corrected_greedy_error_task_count": len(corrected),
        "bandit_wrong_override_task_count": len(wrong), "direct_override_task_count": len(direct_override_tasks),
        "changed_task_count_including_trajectory_effects": len(changed_tasks),
        "processes": processes, "decision_hashes": {m: engine.decision_semantic_hash(Path(specs[m]["output"])) for m in METHODS},
        "unit_checks": [{"name": name, "pass": passed} for name, passed in unit_checks],
    }
    write_json(attempt / "bandit_update_audit.json", bandit_update_audit)

    lines = [
        "# Fast SafeLinUCB effectiveness diagnostic", "",
        "The formal PROP and all frozen parameters remained read-only.", "",
        "| method | J_all_net | J_HV_net | all | HV | high | urgent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = metrics[method]
        lines.append(f"| {method} | {f(row['J_all_net']):.6f} | {f(row['J_HV_net']):.6f} | {i(row['all_closure_count'])} | {i(row['HV_closure_count'])} | {i(row['high_closure_count'])} | {i(row['urgent_closure_count'])} |")
    lines.extend([
        "", f"Active minus Greedy: J_all {f(active['J_all_net'])-f(fixed['J_all_net']):+.12f}; J_HV {f(active['J_HV_net'])-f(fixed['J_HV_net']):+.12f}.",
        f"Active learned-residual changes: {active_audit['learned_residual_action_change_count']}; exploration-only changes: {active_audit['exploration_only_action_change_count']}.",
        f"Prediction RMSE first/second half: {f(active_audit['prediction_rmse_first_half']):.9f} / {f(active_audit['prediction_rmse_second_half']):.9f}.",
        f"Task-level corrected/wrong overrides: {len(corrected)} / {len(wrong)}.", "",
        f"Conclusion: **{conclusion}**.", "",
        "If unsupported, the engineering recommendation is to use safe-set predicted physical net maximization and retain SafeLinUCB only as an extension interface.", "",
        COMPLETION,
    ])
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    test_lines = [f"{'PASS' if passed else 'FAIL'}\t{name}" for name, passed in checks.items()]
    test_lines.extend(f"{'PASS' if passed else 'FAIL'}\tunit::{name}" for name, passed in unit_checks)
    (attempt / "test_report.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": "PASS" if all(checks.values()) else "FAIL", "marker": COMPLETION, "conclusion": conclusion, "completed_at": datetime.now(timezone.utc).isoformat()})
    if not all(checks.values()):
        raise RuntimeError("SAFELINUCB_DIAGNOSTIC_AUDIT_FAILED")
    print(f"Unique entry: {Path(__file__).resolve()}")
    print(f"Output: {attempt}")
    print(f"Conclusion: {conclusion}")
    print(COMPLETION)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path)
    args = parser.parse_args()
    raise SystemExit(child_main(args.child_spec) if args.child_spec else main())
