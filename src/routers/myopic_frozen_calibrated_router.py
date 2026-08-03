"""Frozen offline-calibrated MYOPIC selector used behind corrected REQALIGN.

The scoring and selection rules are preserved from the audited
``MYOPIC-FROZEN-CALIBRATED`` baseline.  ``bind_step`` is the only integration
adapter: it exposes the current task/time to the frozen selector without
changing its score, action set, safety filter, or update behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.data.value_skew_calibration import task_effective_value
from src.routers.base_router import BaseRouter


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_logit(p: float) -> float:
    p = max(1.0e-6, min(1.0 - 1.0e-6, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class MyopicFrozenCalibratedRouter(BaseRouter):
    """Immediate predicted net utility with frozen success/latency calibration."""

    def __init__(
        self,
        parameter_file: str | Path,
        *,
        router_name: str = "myopic_frozen_calibrated",
    ) -> None:
        super().__init__(router_name)
        self.parameter_file = Path(parameter_file)
        self.params: dict[str, Any] = json.loads(
            self.parameter_file.read_text(encoding="utf-8")
        )
        if bool(self.params.get("allow_online_update", False)):
            raise ValueError("frozen calibrated MYOPIC forbids online updates")
        self.current_task = None
        self.current_time_s = 0.0
        self.use_lyapunov_pricing = False
        self.use_mode_controller = False
        self.use_guided_utility = False
        self.online_update_count = 0

    def bind_step(
        self,
        timeline_step: Any,
        resource_state: Any,
        comm_state: Any,
    ) -> None:
        """Bind observable runtime context for the shared corrected backend."""

        self.current_task = timeline_step.task
        self.current_time_s = (
            float(timeline_step.slot_index) * float(timeline_step.slot_seconds)
        )
        _ = (resource_state, comm_state)

    def prepare_decision(
        self,
        *,
        timeline_step: Any,
        resource_state: Any,
        comm_state: Any,
        context: Any,
        predicted_map: Any,
    ) -> None:
        """Retain compatibility with the original frozen runtime."""

        self.bind_step(timeline_step, resource_state, comm_state)
        _ = (context, predicted_map)

    def _p_success(self, predicted: PredictedMetrics) -> float:
        raw = _clip01(
            float(
                getattr(
                    predicted,
                    "delivery_success_proxy",
                    predicted.predicted_delivery_factor,
                )
            )
        )
        p = _sigmoid(
            float(self.params["success_calibration"]["a"]) * _safe_logit(raw)
            + float(self.params["success_calibration"]["b"])
        )
        floor = float(self.params["success_calibration"].get("floor", 0.0))
        ceil = float(self.params["success_calibration"].get("ceil", 1.0))
        return max(floor, min(ceil, p))

    def _latency(self, predicted: PredictedMetrics) -> float:
        raw = float(
            predicted.predicted_L_inf
            + predicted.predicted_L_queue
            + predicted.predicted_L_comm
        )
        alpha = float(self.params["latency_calibration"]["alpha"])
        beta = float(self.params["latency_calibration"]["beta"])
        return max(0.0, alpha * raw + beta)

    def _public_immediate_cost(self, predicted: PredictedMetrics) -> float:
        cost_cfg = self.params.get("public_cost", {})
        norm_wh = float(cost_cfg.get("controllable_mission_energy_wh", 10.0))
        lam = float(cost_cfg.get("lambda_E_common", 1.0))
        energy_wh = float(
            predicted.predicted_E_comp + predicted.predicted_E_comm
        ) / 3600.0
        return lam * energy_wh / max(norm_wh, 1.0e-12)

    def _score_components(
        self,
        action: JointAction,
        predicted: PredictedMetrics,
    ) -> dict[str, object]:
        task = self.current_task
        if task is None or action.action_name == ACTION_FALLBACK_NAME:
            return {
                "scoring_mode": "myopic_frozen_calibrated",
                "p_success_calibrated": 0.0,
                "T_closure_calibrated_s": 0.0,
                "predicted_immediate_net": 0.0,
                "public_immediate_controllable_cost": 0.0,
                "final_action_score": 0.0,
                "defer_score_fixed": 0.0,
            }
        p_success = self._p_success(predicted)
        latency = self._latency(predicted)
        remaining = max(float(task.deadline_s), 1.0e-9)
        timeliness = _clip01(1.0 - latency / remaining)
        value = float(task_effective_value(task))
        information = max(0.0, float(predicted.effective_info_gain))
        quality = _clip01(float(predicted.predicted_detection_quality))
        gross = p_success * value * information * quality * timeliness
        cost = self._public_immediate_cost(predicted)
        net = gross - cost
        return {
            "scoring_mode": "myopic_frozen_calibrated",
            "raw_success_proxy": float(
                getattr(
                    predicted,
                    "delivery_success_proxy",
                    predicted.predicted_delivery_factor,
                )
            ),
            "p_success_calibrated": p_success,
            "raw_latency_s": float(
                predicted.predicted_L_inf
                + predicted.predicted_L_queue
                + predicted.predicted_L_comm
            ),
            "T_closure_calibrated_s": latency,
            "effective_task_value": value,
            "information_sufficiency": information,
            "profile_result_quality": quality,
            "timeliness_discount": timeliness,
            "predicted_gross_utility": gross,
            "public_immediate_controllable_cost": cost,
            "predicted_immediate_net": net,
            "final_action_score": net,
            "exploration_bonus": 0.0,
            "long_term_headroom": "NOT_USED_BY_CURRENT_SCORE",
            "future_task_displacement": "NOT_USED_BY_CURRENT_SCORE",
        }

    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        _ = (context, guidance_bundle)
        if not candidate_actions:
            raise ValueError("candidate_actions must not be empty")
        self.last_all_score_components = {}
        best_action = None
        best_score = 0.0
        defer_action = candidate_actions[0]
        for action in candidate_actions:
            predicted = predicted_map[action.action_name]
            components = self._score_components(action, predicted)
            score = float(components["final_action_score"])
            self.last_all_score_components[action.action_name] = components
            if action.action_name == ACTION_FALLBACK_NAME:
                defer_action = action
                continue
            if best_action is None or (
                score,
                -float(predicted.predicted_S_gen_mb),
                -float(predicted.predicted_E_comp + predicted.predicted_E_comm),
                action.action_name,
            ) > (
                best_score,
                -float(
                    predicted_map[best_action.action_name].predicted_S_gen_mb
                )
                if best_action
                else 0.0,
                -float(
                    predicted_map[best_action.action_name].predicted_E_comp
                    + predicted_map[best_action.action_name].predicted_E_comm
                )
                if best_action
                else 0.0,
                best_action.action_name if best_action else "",
            ):
                best_action = action
                best_score = score
        winner = (
            best_action
            if best_action is not None and best_score > 0.0
            else defer_action
        )
        self.last_selected_action_score = float(
            self.last_all_score_components[winner.action_name][
                "final_action_score"
            ]
        )
        self.last_score_components = dict(
            self.last_all_score_components[winner.action_name]
        )
        return winner

    def update(
        self,
        context: ContextVector,
        action: JointAction,
        observed_reward: float,
    ) -> None:
        _ = (context, action, observed_reward)
        self.online_update_count = 0
