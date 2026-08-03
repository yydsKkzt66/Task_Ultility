"""Independent decision helpers for the decoupled PROP-V2 diagnostic.

This module deliberately does not modify the legacy predictor, SafeFilter, or
SafeLinUCB implementation.  It supplies a q-once utility prior and a complete
model-product action grid to a V2-only runtime wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.data.value_skew_calibration import task_effective_value
from src.payload.information_sufficiency import delta_information_utility


MODELS = ("lightweight", "balanced", "high_accuracy")
PRODUCTS = ("metadata", "roi", "roi_context")
MODEL_LEVEL = {"lightweight": 0.20, "balanced": 0.55, "high_accuracy": 0.95}
PRODUCT_LEVEL = {"metadata": 0.20, "roi": 0.55, "roi_context": 0.95}
FROZEN_QUALITY_PROXY = {"lightweight": 0.74, "balanced": 0.87, "high_accuracy": 0.94}


def build_decoupled_action_rows(*, fixed_model: str | None = None) -> list[dict[str, Any]]:
    """Return the full 3x3 joint grid (or the balanced lock) plus Defer."""

    models = (fixed_model,) if fixed_model else MODELS
    rows = [
        {
            "action_name": f"v2:{model}+{product}",
            "model_name": model,
            "payload_policy": product,
            "payload_plan": product,
            "transmission_option": "contact_aware",
            "expected_payload_level": PRODUCT_LEVEL[product],
            "expected_compute_level": MODEL_LEVEL[model],
            "enabled": True,
        }
        for model in models
        for product in PRODUCTS
    ]
    rows.append(
        {
            "action_name": "delay/store",
            "model_name": "delay_store",
            "payload_policy": "defer",
            "payload_plan": "defer",
            "transmission_option": "store",
            "expected_payload_level": 0.0,
            "expected_compute_level": 0.0,
            "enabled": True,
        }
    )
    return rows


def action_grid_audit(rows: Iterable[Mapping[str, Any]], *, fixed_model: str | None = None) -> dict[str, Any]:
    enabled = [dict(row) for row in rows if bool(row.get("enabled", True))]
    pairs = sorted(
        (str(row.get("model_name")), str(row.get("payload_policy")))
        for row in enabled
        if str(row.get("payload_policy")) != "defer"
    )
    expected_models = (fixed_model,) if fixed_model else MODELS
    expected = sorted((model, product) for model in expected_models for product in PRODUCTS)
    return {
        "fixed_model": fixed_model or "",
        "enabled_action_count": len(enabled),
        "model_product_pairs": [list(pair) for pair in pairs],
        "expected_model_product_pairs": [list(pair) for pair in expected],
        "complete": pairs == expected,
        "defer_count": sum(str(row.get("payload_policy")) == "defer" for row in enabled),
    }


def timeliness_factor(*, task_arrival_s: float, deadline_duration_s: float, ground_arrival_s: float | None) -> float:
    if ground_arrival_s is None or deadline_duration_s <= 0.0:
        return 0.0
    latency = max(0.0, float(ground_arrival_s) - float(task_arrival_s))
    return max(0.0, min(1.0, 1.0 - latency / float(deadline_duration_s)))


@dataclass(frozen=True)
class V2Score:
    action_name: str
    task_value: float
    delta_information: float
    quality: float
    ground_success_probability: float
    timeliness: float
    positive_benefit: float
    energy_term: float
    latency_term: float
    queue_term: float
    payload_term: float
    dynamic_resource_cost: float
    bias_term: float
    safety_margin_penalty: float
    net_score: float
    q_multiplication_count: int


def score_action_q_once(
    *,
    task: Any,
    action: Any,
    predicted: Any,
    delivered_products: Iterable[str],
    product_utility_cfg: Mapping[str, Any] | None,
    guidance_bundle: Mapping[str, Any] | None,
    simulator_cfg: Mapping[str, Any],
    beta_resource_cost: float,
    beta_hard_penalty: float,
    ground_arrival_s: float | None,
    slot_seconds: float,
) -> V2Score:
    """Compute V*deltaI*q*p_ground*T minus the unchanged dynamic prices.

    ``deltaI`` is taken directly from the frozen product-requirement utility
    table and therefore contains no model-quality factor.  ``q`` appears in
    exactly one explicit multiplication in ``positive_benefit``.
    """

    product = str(getattr(predicted, "predicted_product_type", "") or "")
    if str(getattr(action, "payload_policy", "")) == "defer":
        delta_i = quality = p_ground = timely = positive = 0.0
    else:
        delta_i = delta_information_utility(
            product,
            set(delivered_products),
            str(getattr(task, "product_requirement", "verification")),
            product_utility_cfg,
        )
        quality = FROZEN_QUALITY_PROXY[str(getattr(action, "model_name", ""))]
        p_ground = max(0.0, min(1.0, float(getattr(predicted, "delivery_success_proxy", 0.0))))
        task_arrival_s = float(getattr(task, "arrival_slot", 0) or 0) * float(slot_seconds)
        timely = timeliness_factor(
            task_arrival_s=task_arrival_s,
            deadline_duration_s=float(getattr(task, "deadline_s", 0.0) or 0.0),
            ground_arrival_s=ground_arrival_s,
        )
        positive = max(0.0, float(task_effective_value(task))) * delta_i * quality * p_ground * timely

    guide = dict(guidance_bundle or {})
    epsilon = float(simulator_cfg.get("epsilon", 1.0e-9))
    energy_term = (
        float(getattr(predicted, "predicted_E_comp", 0.0))
        + float(getattr(predicted, "predicted_E_comm", 0.0))
    ) / max(float(simulator_cfg.get("E_ref", 1.0)), epsilon)
    latency_term = (
        float(getattr(predicted, "predicted_L_inf", 0.0))
        + float(getattr(predicted, "predicted_L_queue", 0.0))
        + float(getattr(predicted, "predicted_L_comm", 0.0))
    ) / max(float(getattr(task, "deadline_s", 0.0) or 0.0), epsilon)
    queue_max = max(float(simulator_cfg.get("queue_max_mb", 1.0)), epsilon)
    queue_term = float(getattr(predicted, "predicted_Q_next_mb", 0.0)) / queue_max
    payload_term = float(getattr(predicted, "predicted_S_gen_mb", 0.0)) / queue_max
    resource_cost_raw = (
        float(guide.get("lambda_E", 0.0)) * energy_term
        + float(guide.get("lambda_L", 0.0)) * latency_term
        + float(guide.get("lambda_Q", 0.0)) * queue_term
        + float(guide.get("lambda_S", 0.0)) * payload_term
    )
    dynamic_cost = float(beta_resource_cost) * resource_cost_raw
    bias = (
        float(guide.get("model_complexity_bias", 0.0)) * float(getattr(action, "expected_compute_level", 0.0))
        + float(guide.get("payload_size_bias", 0.0)) * float(getattr(action, "expected_payload_level", 0.0))
    )
    battery_margin = max(0.0, float(simulator_cfg.get("battery_min", 0.0)) - float(getattr(predicted, "predicted_B_next", 0.0)))
    thermal_margin = max(0.0, float(getattr(predicted, "predicted_Theta_next", 0.0)) - float(simulator_cfg.get("thermal_max", 1.0)))
    queue_margin = max(0.0, float(getattr(predicted, "predicted_Q_next_mb", 0.0)) - queue_max) / queue_max
    safety_margin = float(guide.get("safety_margin_scale", 1.0)) * (battery_margin + thermal_margin + queue_margin)
    safety_penalty = float(beta_hard_penalty) * safety_margin
    return V2Score(
        action_name=str(getattr(action, "action_name", "")),
        task_value=float(task_effective_value(task)),
        delta_information=delta_i,
        quality=quality,
        ground_success_probability=p_ground,
        timeliness=timely,
        positive_benefit=positive,
        energy_term=energy_term,
        latency_term=latency_term,
        queue_term=queue_term,
        payload_term=payload_term,
        dynamic_resource_cost=dynamic_cost,
        bias_term=bias,
        safety_margin_penalty=safety_penalty,
        net_score=positive - dynamic_cost + bias - safety_penalty,
        q_multiplication_count=0 if str(getattr(action, "payload_policy", "")) == "defer" else 1,
    )
