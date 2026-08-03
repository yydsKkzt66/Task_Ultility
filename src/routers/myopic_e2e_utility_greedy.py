"""Fair current-task end-to-end utility greedy baseline.

The router changes only the action-ranking rule.  The caller still owns the
candidate action set, safety filter, inference, payload generation, queue,
contact-aware downlink, and ground accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.common.constants import ACTION_FALLBACK_NAME
from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.data.value_skew_calibration import task_effective_value
from src.payload.information_sufficiency import (
    compute_information_sufficiency,
    product_information_utility,
)
from src.reward.utility_calibration import calibrate_value, get_utility_calibration
from src.routers.base_router import BaseRouter


@dataclass(frozen=True)
class _RankedAction:
    action: JointAction
    score: float
    tie_key: tuple[float, float, float, str]


class MyopicE2EUtilityGreedyRouter(BaseRouter):
    """Greedily maximize frozen one-step predicted net utility.

    No learned state, online update, exploration bonus, experience bonus,
    seed-specific rule, or future-task value is used.
    """

    SCORE_TOLERANCE = 1.0e-12

    def __init__(
        self,
        reward_cfg: Mapping[str, Any],
        simulator_cfg: Mapping[str, Any],
        product_utility_cfg: Mapping[str, Any],
        *,
        router_name: str = "myopic_e2e_utility_greedy",
        task_value_reference: float = 1.0,
        score_mode: str = "proxy_e2e",
    ) -> None:
        super().__init__(router_name)
        self.reward_cfg = dict(reward_cfg)
        self.simulator_cfg = dict(simulator_cfg)
        self.product_utility_cfg = dict(product_utility_cfg)
        self.task_value_reference = float(task_value_reference)
        if self.task_value_reference <= 0.0:
            raise ValueError("task_value_reference must be positive")
        if score_mode not in {"proxy_e2e", "common_predicted_reward"}:
            raise ValueError(f"unsupported score_mode: {score_mode}")
        self.score_mode = score_mode
        self.current_step: Any | None = None
        self.current_resource_state: Any | None = None
        self.current_comm_state: Any | None = None

        # This is an independent one-step baseline.  Shared physics remains in
        # the runner, but long-horizon prices/modes and online learning are not
        # part of either its score or its lifecycle semantics.
        self.use_lyapunov_pricing = False
        self.use_mode_controller = False
        self.use_guided_utility = False
        self.online_update_count = 0

    def bind_step(self, timeline_step: Any, resource_state: Any, comm_state: Any) -> None:
        """Bind current observable state without changing the shared runner."""

        self.current_step = timeline_step
        self.current_resource_state = resource_state
        self.current_comm_state = comm_state

    def prepare_decision(
        self,
        *,
        timeline_step: Any,
        resource_state: Any,
        comm_state: Any,
        context: Any,
        predicted_map: Any,
    ) -> None:
        """Shared-runner hook; binds only currently observable state."""

        self.bind_step(timeline_step, resource_state, comm_state)
        _ = (context, predicted_map)

    def _require_step(self) -> Any:
        if self.current_step is None or self.current_step.task is None:
            raise RuntimeError("bind_step must provide a current task before select_action")
        return self.current_step

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _is_defer(action: JointAction) -> bool:
        return (
            action.action_name == ACTION_FALLBACK_NAME
            or action.payload_policy == "defer"
            or action.model_name == "delay_store"
        )

    def _normalized_task_value(self) -> float:
        task = self._require_step().task
        calibration = get_utility_calibration(self.reward_cfg)
        value = calibrate_value(task_effective_value(task), calibration)
        if not calibration.enabled:
            value /= self.task_value_reference
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "MYOPIC requires the shared public value calibration to map "
                f"task values into [0, 1], got {value!r}"
            )
        return float(value)

    def _remaining_lifetime_s(self) -> float:
        step = self._require_step()
        slot_seconds = float(step.slot_seconds)
        arrival_time_s = float(getattr(step.task, "arrival_slot", step.slot_index)) * slot_seconds
        deadline_time_s = arrival_time_s + float(step.task.deadline_s)
        current_time_s = float(step.slot_index) * slot_seconds
        return max(0.0, deadline_time_s - current_time_s)

    def _defer_components(self, action: JointAction) -> dict[str, object]:
        raw_store_cost = self.simulator_cfg.get(
            "store_cost",
            self.simulator_cfg.get("defer_cost"),
        )
        store_cost = 0.0 if raw_store_cost is None else float(raw_store_cost)
        return {
            "scoring_mode": self.router_name,
            "action_name": action.action_name,
            "normalized_task_value": self._normalized_task_value(),
            "current_achievable_product_type": "empty",
            "current_product_feasible": 0,
            "information_sufficiency": 0.0,
            "deadline_timeliness_factor": 0.0,
            "current_delivery_feasibility_factor": 0.0,
            "delivery_utility": 0.0,
            "energy_cost": 0.0,
            "latency_cost": 0.0,
            "queue_cost": 0.0,
            "requirement_fit": 0.0,
            "requirement_gap": 0.0,
            "store_cost": store_cost,
            "predicted_e2e_latency": 0.0,
            "predicted_total_energy": 0.0,
            "predicted_payload_mb": 0.0,
            "final_action_score": -store_cost,
            "exploration_bonus": 0.0,
            "experience_bonus": 0.0,
            "tie_break_reason": "",
        }

    def _score_components(
        self,
        context: ContextVector,
        action: JointAction,
        predicted: PredictedMetrics,
    ) -> dict[str, object]:
        if self.score_mode == "common_predicted_reward":
            score = float(predicted.predicted_reward)
            return {
                "scoring_mode": "common_predicted_reward",
                "action_name": action.action_name,
                "normalized_task_value": self._normalized_task_value(),
                "current_achievable_product_type": str(predicted.predicted_product_type),
                "current_product_feasible": int(not self._is_defer(action)),
                "information_sufficiency": float(predicted.effective_info_gain),
                "deadline_timeliness_factor": 0.0,
                "current_delivery_feasibility_factor": float(predicted.predicted_delivery_factor),
                "delivery_utility": score,
                "energy_cost": 0.0,
                "latency_cost": 0.0,
                "queue_cost": 0.0,
                "requirement_fit": 0.0,
                "requirement_gap": 0.0,
                "store_cost": 0.0,
                "predicted_e2e_latency": float(
                    predicted.predicted_L_inf
                    + predicted.predicted_L_queue
                    + predicted.predicted_L_comm
                ),
                "predicted_total_energy": float(predicted.predicted_E_comp + predicted.predicted_E_comm),
                "predicted_payload_mb": float(predicted.predicted_S_gen_mb),
                "final_action_score": score,
                "exploration_bonus": 0.0,
                "experience_bonus": 0.0,
                "tie_break_reason": "",
            }
        if self._is_defer(action):
            return self._defer_components(action)

        step = self._require_step()
        task = step.task
        epsilon = max(float(self.simulator_cfg.get("epsilon", 1.0e-9)), 1.0e-12)
        remaining = max(self._remaining_lifetime_s(), epsilon)
        latency = float(
            predicted.predicted_L_inf
            + predicted.predicted_L_queue
            + predicted.predicted_L_comm
        )
        total_energy = float(predicted.predicted_E_comp + predicted.predicted_E_comm)
        delivery_factor = self._clip01(predicted.predicted_delivery_factor)
        feasible = bool(
            predicted.predicted_status in {"delivered", "queued"}
            and latency <= self._remaining_lifetime_s()
            and delivery_factor > 0.0
        )
        product_type = str(predicted.predicted_product_type) if feasible else "empty"
        information = float(predicted.effective_info_gain) if feasible else 0.0
        information_sufficiency = self._clip01(
            compute_information_sufficiency(context, information, self.reward_cfg)
        )
        timeliness = self._clip01(1.0 - latency / remaining)
        requirement_fit = self._clip01(
            product_information_utility(
                {product_type} if product_type != "empty" else set(),
                str(task.product_requirement),
                self.product_utility_cfg,
            )
        )
        normalized_value = self._normalized_task_value()
        predicted_quality = self._clip01(predicted.predicted_detection_quality)
        delivery_utility = (
            normalized_value
            * predicted_quality
            * information_sufficiency
            * timeliness
            * delivery_factor
        )
        energy_cost = (
            float(self.reward_cfg.get("energy_penalty", 0.0))
            * total_energy
            / max(float(self.simulator_cfg["E_ref"]), epsilon)
        )
        latency_cost = float(self.reward_cfg.get("latency_penalty", 0.0)) * latency / remaining
        queue_cost = (
            float(self.reward_cfg.get("queue_penalty", 0.0))
            * float(predicted.predicted_Q_next_mb)
            / max(float(self.simulator_cfg["queue_max_mb"]), epsilon)
        )
        requirement_gap = (
            float(self.reward_cfg.get("information_gap_penalty", 0.0))
            * (1.0 - requirement_fit)
        )
        final_score = delivery_utility - energy_cost - latency_cost - queue_cost - requirement_gap
        return {
            "scoring_mode": self.router_name,
            "action_name": action.action_name,
            "normalized_task_value": normalized_value,
            "current_achievable_product_type": product_type,
            "current_product_feasible": int(feasible),
            "information_sufficiency": information_sufficiency,
            "deadline_timeliness_factor": timeliness,
            "current_delivery_feasibility_factor": delivery_factor,
            "delivery_utility": delivery_utility,
            "energy_cost": energy_cost,
            "latency_cost": latency_cost,
            "queue_cost": queue_cost,
            "requirement_fit": requirement_fit,
            "requirement_gap": requirement_gap,
            "store_cost": 0.0,
            "predicted_e2e_latency": latency,
            "predicted_total_energy": total_energy,
            "predicted_payload_mb": float(predicted.predicted_S_gen_mb),
            "final_action_score": final_score,
            "exploration_bonus": 0.0,
            "experience_bonus": 0.0,
            "tie_break_reason": "",
        }

    @classmethod
    def _better(cls, candidate: _RankedAction, incumbent: _RankedAction) -> bool:
        delta = candidate.score - incumbent.score
        if delta > cls.SCORE_TOLERANCE:
            return True
        if delta < -cls.SCORE_TOLERANCE:
            return False
        return candidate.tie_key < incumbent.tie_key

    @classmethod
    def _tie_reason(
        cls,
        winner: _RankedAction,
        runner_up: _RankedAction | None,
    ) -> str:
        if runner_up is None:
            return "only_candidate"
        if winner.score - runner_up.score > cls.SCORE_TOLERANCE:
            return "higher_myopic_utility"
        labels = ("earlier_ground_availability", "fewer_bytes", "lower_energy", "action_name")
        for label, left, right in zip(labels, winner.tie_key, runner_up.tie_key):
            if left != right:
                return label
        return "stable_equal_key"

    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        _ = guidance_bundle
        self._require_step()
        if not candidate_actions:
            raise ValueError("candidate_actions must not be empty")

        ranked: list[_RankedAction] = []
        self.last_all_score_components = {}
        for action in candidate_actions:
            components = self._score_components(
                context,
                action,
                predicted_map[action.action_name],
            )
            self.last_all_score_components[action.action_name] = components
            ranked.append(
                _RankedAction(
                    action=action,
                    score=float(components["final_action_score"]),
                    tie_key=(
                        float(components["predicted_e2e_latency"]),
                        float(components["predicted_payload_mb"]),
                        float(components["predicted_total_energy"]),
                        action.action_name,
                    ),
                )
            )

        winner = ranked[0]
        for candidate in ranked[1:]:
            if self._better(candidate, winner):
                winner = candidate

        alternatives = [item for item in ranked if item.action.action_name != winner.action.action_name]
        runner_up = alternatives[0] if alternatives else None
        for candidate in alternatives[1:]:
            if runner_up is not None and self._better(candidate, runner_up):
                runner_up = candidate

        reason = self._tie_reason(winner, runner_up)
        self.last_all_score_components[winner.action.action_name]["tie_break_reason"] = reason
        self.last_selected_action_score = winner.score
        self.last_score_components = dict(self.last_all_score_components[winner.action.action_name])
        return winner.action

    def update(
        self,
        context: ContextVector,
        action: JointAction,
        observed_reward: float,
    ) -> None:
        _ = (context, action, observed_reward)
        self.online_update_count = 0
