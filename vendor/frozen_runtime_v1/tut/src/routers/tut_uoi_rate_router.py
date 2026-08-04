"""Deterministic TUT-inspired task-oriented semantic-rate router."""

from __future__ import annotations

from typing import Any

from src.common.types import ContextVector, GeneratedPayload, JointAction, PredictedMetrics, TaskSample
from src.payload.information_sufficiency import product_information_utility
from src.routers.base_router import BaseRouter


RATE_BY_POLICY = {
    "metadata": "LOW_RATE",
    "roi": "MEDIUM_RATE",
    "roi_context": "HIGH_RATE",
    "defer": "DEFER",
}


class TutUoiRateRouter(BaseRouter):
    """Fixed-model, one-shot semantic-rate selection with Lyapunov prices."""

    def __init__(
        self,
        *,
        router_name: str = "tut_uoi_rate_lite",
        product_utility_cfg: dict[str, object] | None = None,
        v_uoi: float = 1.0,
        energy_ref_j: float = 100.0,
        bytes_ref_mb: float = 2.70,
        delivery_latency_weight: float = 0.5,
        information_gap_weight: float = 0.5,
        slot_seconds: float = 20.0,
    ) -> None:
        super().__init__(router_name)
        self.use_lyapunov_pricing = True
        self.use_mode_controller = False
        self.use_guided_utility = True
        self.product_utility_cfg = dict(product_utility_cfg or {})
        self.v_uoi = float(v_uoi)
        self.energy_ref_j = max(float(energy_ref_j), 1.0e-12)
        self.bytes_ref_mb = max(float(bytes_ref_mb), 1.0e-12)
        raw_latency_weight = max(0.0, float(delivery_latency_weight))
        raw_information_weight = max(0.0, float(information_gap_weight))
        weight_sum = raw_latency_weight + raw_information_weight
        if weight_sum <= 1.0e-12:
            raise ValueError("At least one predicted-UoI weight must be positive")
        self.delivery_latency_weight = raw_latency_weight / weight_sum
        self.information_gap_weight = raw_information_weight / weight_sum
        self.slot_seconds = max(float(slot_seconds), 1.0e-12)
        self._task: TaskSample | None = None
        self._decision_time_s = 0.0
        self.selected_task_ids: set[str] = set()

    def set_task_context(self, task: TaskSample, decision_time_s: float) -> None:
        self._task = task
        self._decision_time_s = float(decision_time_s)

    @staticmethod
    def _latency(metrics: PredictedMetrics) -> float:
        return metrics.predicted_L_inf + metrics.predicted_L_queue + metrics.predicted_L_comm

    def _information_sufficiency(self, action: JointAction, task: TaskSample) -> float:
        if action.payload_policy == "defer":
            return 0.0
        return product_information_utility(
            {action.payload_policy},
            task.product_requirement,
            self.product_utility_cfg,
        )

    def _predicted_uoi(
        self,
        action: JointAction,
        metrics: PredictedMetrics,
        task: TaskSample,
    ) -> tuple[float, float, float]:
        sufficiency = self._information_sufficiency(action, task)
        latency_norm = (
            1.0
            if action.payload_policy == "defer"
            else max(0.0, min(1.0, self._latency(metrics) / max(float(task.deadline_s), 1.0e-12)))
        )
        predicted_uoi = max(0.0, float(task.task_value)) * (
            self.delivery_latency_weight * latency_norm
            + self.information_gap_weight * (1.0 - sufficiency)
        )
        return predicted_uoi, sufficiency, latency_norm

    def _score(
        self,
        action: JointAction,
        metrics: PredictedMetrics,
        task: TaskSample,
        guidance: dict[str, object],
    ) -> tuple[float, dict[str, Any]]:
        predicted_uoi, sufficiency, delivery_latency_norm = self._predicted_uoi(action, metrics, task)
        defer_uoi = max(0.0, float(task.task_value))
        delta_uoi = defer_uoi - predicted_uoi
        energy_j = metrics.predicted_E_comp + metrics.predicted_E_comm
        latency_s = self._latency(metrics)
        lambda_e = float(guidance.get("lambda_E", 0.0))
        lambda_q = float(guidance.get("lambda_Q", 0.0))
        lambda_l = float(guidance.get("lambda_L", 0.0))
        energy_norm = energy_j / self.energy_ref_j
        bytes_norm = metrics.predicted_S_gen_mb / self.bytes_ref_mb
        latency_norm = latency_s / max(float(task.deadline_s), 1.0e-12)
        energy_cost = lambda_e * energy_norm
        byte_cost = lambda_q * bytes_norm
        latency_cost = lambda_l * latency_norm
        score = -self.v_uoi * predicted_uoi - energy_cost - byte_cost - latency_cost
        return score, {
            "rate_action": RATE_BY_POLICY[action.payload_policy],
            "uoi": predicted_uoi,
            "predicted_uoi": predicted_uoi,
            "predicted_uoi_defer": defer_uoi,
            "incompleteness": 1.0,
            "predicted_information_sufficiency": sufficiency,
            "delta_uoi": delta_uoi,
            "delivery_latency_weight": self.delivery_latency_weight,
            "information_gap_weight": self.information_gap_weight,
            "V_uoi": self.v_uoi,
            "lambda_E": lambda_e,
            "lambda_Q": lambda_q,
            "lambda_L": lambda_l,
            "predicted_energy_j": energy_j,
            "predicted_energy_normalized": energy_norm,
            "normalized_energy_cost": energy_cost,
            "predicted_bytes_mb": metrics.predicted_S_gen_mb,
            "predicted_bytes_normalized": bytes_norm,
            "normalized_byte_cost": byte_cost,
            "predicted_latency_s": latency_s,
            "predicted_latency_normalized": latency_norm,
            "normalized_predicted_delivery_latency": delivery_latency_norm,
            "normalized_latency_cost": latency_cost,
            "score": score,
        }

    def prepare_action_audit(
        self,
        actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> None:
        if self._task is None:
            raise RuntimeError("set_task_context must be called before TUT-UOI-PRODUCT-LITE audit")
        guidance = guidance_bundle or {}
        self.last_all_score_components = {
            action.action_name: self._score(action, predicted_map[action.action_name], self._task, guidance)[1]
            for action in actions
        }

    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        if self._task is None:
            raise RuntimeError("set_task_context must be called before TUT-UOI-PRODUCT-LITE selection")
        defer = next((action for action in candidate_actions if action.payload_policy == "defer"), None)
        if defer is None:
            raise RuntimeError("TUT-UOI-PRODUCT-LITE requires an explicit DEFER candidate")
        if self._task.task_id in self.selected_task_ids:
            return defer
        guidance = guidance_bundle or {}
        remaining_s = max(
            0.0,
            float(self._task.deadline_s)
            - max(0.0, self._decision_time_s - float(self._task.arrival_slot) * self.slot_seconds),
        )
        feasible = [
            action
            for action in candidate_actions
            if action.payload_policy == "defer"
            or self._latency(predicted_map[action.action_name]) <= remaining_s + 1.0e-12
        ]
        scored: list[tuple[float, float, str, JointAction, dict[str, Any]]] = []
        for action in feasible:
            score, components = self._score(action, predicted_map[action.action_name], self._task, guidance)
            self.last_all_score_components[action.action_name] = components
            scored.append((score, -predicted_map[action.action_name].predicted_S_gen_mb, action.action_name, action, components))
        _, _, _, selected, selected_components = max(scored, key=lambda row: (row[0], row[1], row[2]))
        self.last_selected_action_score = float(selected_components["score"])
        self.last_score_components = dict(selected_components)
        self.selected_task_ids.add(self._task.task_id)
        return selected

    def decorate_payload(self, payload: GeneratedPayload, action: JointAction) -> GeneratedPayload:
        components = self.last_all_score_components.get(action.action_name, self.last_score_components)
        payload.metadata = {
            **dict(payload.metadata),
            "tut_uoi_priority_enabled": True,
            "tut_rate_action": RATE_BY_POLICY.get(action.payload_policy, "DEFER"),
            "tut_delta_uoi": float(components.get("delta_uoi", 0.0) or 0.0),
            "tut_uoi": float(components.get("uoi", 0.0) or 0.0),
            "tut_urgency": float(components.get("urgency", 0.0) or 0.0),
        }
        return payload
