"""Bounded virtual-queue support for high-value staged evidence delivery.

The helpers in this module are decision-time only. They use task metadata,
current delivered-product state, contact/resource feasibility rows, and
configured queue debt. They do not inspect future reward, future closure, or
ground-truth labels.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.payload.information_sufficiency import delta_information_utility
from src.payload.staged_delivery_queue import StagedDeliveryRequest, compute_staged_priority


DEFAULT_GUARANTEE_CFG: dict[str, Any] = {
    "enabled": False,
    "mode": "dynamic_age_miss_debt",
    "full_method_aliases": ["proposed_lyapunov_rule_mode_safe_bandit"],
    "allow_utility_rule": False,
    "eligible_value_levels": ["high", "urgent"],
    "eligible_product_requirements": ["context", "evidence"],
    "lambda_H": 0.35,
    "lambda_H_grid": [0.0, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50],
    "z_age_max": 2.0,
    "z_miss_max": 4.0,
    "z_total_max": 6.0,
    "z_max": 6.0,
    "z_max_grid": [2.0, 4.0, 6.0],
    "gamma_age_high_per_slot": 0.01,
    "gamma_age_urgent_per_slot": 0.015,
    "gamma_miss_high": 1.0,
    "gamma_miss_urgent": 1.25,
    "min_stage_miss_weight": 1.0,
    "target_stage_miss_weight": 0.5,
    "debt_increment_high": 1.0,
    "debt_increment_urgent": 1.25,
    "progress_credit_min_stage": 1.5,
    "progress_credit_target_stage": 3.0,
    "reset_debt_when_min_stage_reached": False,
    "reset_debt_when_task_closed": True,
    "update_timing": "dynamic_age_miss",
    "missed_opportunity_only": False,
    "no_contact_slots_do_not_increase_debt": True,
    "minimum_stage_for_context": "ROI",
    "minimum_stage_for_evidence": "ROI",
    "target_stage_for_context": "ROI+Ctx",
    "target_stage_for_evidence": "ROI+Ctx",
    "target_stage_requires_positive_marginal_utility": True,
    "marginal_utility_floor_main": 0.0,
    "marginal_utility_floor_stress": -0.01,
    "guarantee_window_budget_fraction": 0.30,
    "beta_H": 0.30,
    "beta_H_grid": [0.15, 0.30, 0.45, 0.60],
    "guarantee_window_budget_fraction_grid": [0.15, 0.30, 0.45],
    "normalize_priority_components": True,
    "component_normalization": "window_candidate_minmax",
    "waiting_limit_windows_for_target_progress": 1,
    "starvation_window_threshold": 2,
    "update_age_debt_each_slot": True,
    "update_miss_debt_at_contact_window_end": True,
    "recompute_priority_each_scheduler_step": True,
    "carry_debt_across_windows": True,
    "require_future_contact_for_age_debt": True,
    "no_future_contact_no_age_growth": True,
    "no_contact_slots_do_not_increase_miss_debt": True,
    "hard_safety_veto": True,
    "low_medium_debt_forbidden": True,
    "roi_cannot_close_evidence": True,
    "w_density": 0.35,
    "w_abs": 0.25,
    "lambda_D": 0.10,
    "lambda_C": 0.10,
    "lambda_A": 0.10,
    "lambda_P": 0.05,
    "waiting_age_norm_slots": 100.0,
}

STAGE_ORDER = {"none": 0, "metadata": 1, "roi": 2, "roi_context": 3}
STAGE_LABEL = {0: "none", 1: "metadata", 2: "ROI", 3: "ROI+Ctx"}


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_set(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip().lower() for item in value.split(",") if item.strip()}
    try:
        return {str(item).strip().lower() for item in value if str(item).strip()}
    except TypeError:
        return set()


def guarantee_config(container_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    container_cfg = container_cfg or {}
    raw = container_cfg.get("guarantee") if isinstance(container_cfg, Mapping) else None
    progressive = container_cfg.get("progressive_payload") if isinstance(container_cfg, Mapping) else None
    if not isinstance(raw, Mapping) and isinstance(progressive, Mapping):
        raw = progressive.get("guarantee")

    cfg = dict(DEFAULT_GUARANTEE_CFG)
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            cfg[str(key)] = value
    return cfg


def guarantee_enabled_for_router(container_cfg: Mapping[str, Any] | None, router_name: str) -> bool:
    cfg = guarantee_config(container_cfg)
    if not _as_bool(cfg.get("enabled"), False):
        return False
    if str(cfg.get("mode", "dynamic_age_miss_debt")) not in {"bounded_virtual_queue", "dynamic_age_miss_debt"}:
        return False
    router = str(router_name or "")
    if router == "utility_rule" and not _as_bool(cfg.get("allow_utility_rule"), False):
        return False
    aliases_raw = cfg.get("full_method_aliases", ())
    aliases = {aliases_raw} if isinstance(aliases_raw, str) else {str(item) for item in aliases_raw or []}
    return not aliases or router in aliases


def guarantee_priority_active(container_cfg: Mapping[str, Any] | None, router_name: str = "") -> bool:
    cfg = guarantee_config(container_cfg)
    if router_name and not guarantee_enabled_for_router(container_cfg, router_name):
        return False
    return _as_bool(cfg.get("enabled"), False) and _as_float(cfg.get("lambda_H"), 0.0) > 0.0


def is_high_value_guarantee_eligible(value_level: str, product_requirement: str, cfg: Mapping[str, Any] | None = None) -> bool:
    cfg = cfg or DEFAULT_GUARANTEE_CFG
    return (
        str(value_level or "").lower() in _as_set(cfg.get("eligible_value_levels"))
        and str(product_requirement or "").lower() in _as_set(cfg.get("eligible_product_requirements"))
    )


def stage_from_delivered(delivered_products: Iterable[str] | None) -> str:
    delivered = {str(product).strip().lower() for product in (delivered_products or [])}
    if "roi_context" in delivered:
        return "ROI+Ctx"
    if "roi" in delivered:
        return "ROI"
    if "metadata" in delivered:
        return "metadata"
    return "none"


def stage_rank(stage: str) -> int:
    normalized = str(stage or "none").strip().lower().replace("+ctx", "_context")
    return STAGE_ORDER.get(normalized, 0)


def product_advances_stage(product_type: str, delivered_products: Iterable[str] | None) -> bool:
    before = stage_rank(stage_from_delivered(delivered_products))
    after = stage_rank(stage_from_delivered(set(delivered_products or []) | {str(product_type).lower()}))
    return after > before


def marginal_utility_floor(cfg: Mapping[str, Any] | None) -> float:
    cfg = cfg or DEFAULT_GUARANTEE_CFG
    if "marginal_utility_floor" in cfg:
        return _as_float(cfg.get("marginal_utility_floor"), 0.0)
    if _as_bool(cfg.get("use_stress_marginal_utility_floor"), False):
        return _as_float(cfg.get("marginal_utility_floor_stress"), -0.02)
    return _as_float(cfg.get("marginal_utility_floor_main"), 0.0)


def guarantee_progress_gate(
    request: StagedDeliveryRequest,
    *,
    delivered_products: Iterable[str] | None,
    config: Mapping[str, Any] | None,
) -> float:
    cfg = guarantee_config(config)
    if not is_high_value_guarantee_eligible(request.value_level, request.product_requirement, cfg):
        return 0.0
    if request.product_requirement in {"verification", "alert"} and request.product_type == "roi_context":
        return 0.0
    if not product_advances_stage(request.product_type, delivered_products):
        return 0.0

    delivered = {str(product).lower() for product in (delivered_products or [])}
    current_stage = stage_from_delivered(delivered)
    product = str(request.product_type or "").lower()
    if current_stage == "none":
        if product == "metadata":
            return 0.25
        if product == "roi":
            return 1.0
        return 0.0
    if current_stage == "metadata":
        if product == "roi":
            return 1.0
        return 0.0
    if current_stage == "ROI" and product == "roi_context":
        if _as_bool(cfg.get("target_stage_requires_positive_marginal_utility"), True):
            delta_u = delta_information_utility("roi_context", delivered, request.product_requirement)
            if delta_u < marginal_utility_floor(cfg):
                return 0.0
        return 0.80 if request.product_requirement == "evidence" else 0.60
    return 0.0


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo + 1.0e-12:
        return [1.0 if value > 0.0 else 0.0 for value in values]
    return [(value - lo) / (hi - lo) for value in values]


def rank_guarantee_aware_requests(
    requests: Sequence[StagedDeliveryRequest],
    *,
    current_slot: int,
    config: Mapping[str, Any] | None,
    debt_by_task: Mapping[str, float],
    delivered_products_by_task: Mapping[str, Iterable[str]],
    debt_components_by_task: Mapping[str, Mapping[str, float]] | None = None,
    waiting_age_slots_by_task: Mapping[str, int] | None = None,
    queue_pressure: float = 0.0,
) -> list[StagedDeliveryRequest]:
    cfg = guarantee_config(config)
    lambda_h = _as_float(cfg.get("lambda_H"), 0.0)
    active = _as_bool(cfg.get("enabled"), False) and lambda_h > 0.0
    debt_components_by_task = debt_components_by_task or {}
    waiting_age_slots_by_task = waiting_age_slots_by_task or {}

    base_rows: list[tuple[StagedDeliveryRequest, float, dict[str, float], float, float]] = []
    for request in requests:
        base_priority, base_components = compute_staged_priority(request, current_slot=current_slot, config=config)
        density = _as_float(base_components.get("value_density_priority"), 0.0)
        absolute_delta = (
            _as_float(base_components.get("V_eff"), request.calibrated_task_value)
            * _as_float(base_components.get("delta_u_I"), request.residual_delta_information_utility)
            * _as_float(base_components.get("detection_quality"), request.detection_quality)
            * _as_float(base_components.get("timeliness_factor"), request.timeliness_factor)
        )
        base_rows.append((request, base_priority, base_components, density, max(0.0, absolute_delta)))

    if not active:
        for request, base_priority, base_components, _density, _absolute in base_rows:
            request.priority_score = base_priority
            request.priority_components = dict(base_components)
        return list(requests)

    norm_density = _minmax([row[3] for row in base_rows])
    norm_abs = _minmax([row[4] for row in base_rows])
    for idx, (request, base_priority, base_components, density, absolute_delta) in enumerate(base_rows):
        delivered = delivered_products_by_task.get(request.task_id, set())
        components_for_task = debt_components_by_task.get(request.task_id, {})
        z_age = max(0.0, min(_as_float(cfg.get("z_age_max"), _as_float(cfg.get("z_max"), 4.0)), _as_float(components_for_task.get("Z_age"), 0.0)))
        z_miss = max(0.0, min(_as_float(cfg.get("z_miss_max"), _as_float(cfg.get("z_max"), 4.0)), _as_float(components_for_task.get("Z_miss"), 0.0)))
        z_fallback = _as_float(debt_by_task.get(request.task_id), 0.0)
        if z_age <= 0.0 and z_miss <= 0.0 and z_fallback > 0.0:
            z_miss = z_fallback
        z_i = max(0.0, min(_as_float(cfg.get("z_total_max"), _as_float(cfg.get("z_max"), 4.0)), z_age + z_miss))
        g_i_p = guarantee_progress_gate(request, delivered_products=delivered, config=config)
        guarantee_term = lambda_h * z_i * g_i_p
        deadline_component = _as_float(base_components.get("deadline_urgency"), 0.0)
        semantic_component = _as_float(base_components.get("product_requirement_weight"), 0.0)
        waiting_age_slots = max(0, int(_as_float(waiting_age_slots_by_task.get(request.task_id), 0.0)))
        waiting_age_component = min(
            1.0,
            waiting_age_slots / max(1.0, _as_float(cfg.get("waiting_age_norm_slots"), 100.0)),
        )
        queue_pressure_component = max(0.0, float(queue_pressure))
        priority_without_guarantee = (
            _as_float(cfg.get("w_density"), 0.35) * norm_density[idx]
            + _as_float(cfg.get("w_abs"), 0.25) * norm_abs[idx]
            + _as_float(cfg.get("lambda_D"), 0.10) * deadline_component
            + _as_float(cfg.get("lambda_C"), 0.10) * semantic_component
            + _as_float(cfg.get("lambda_A"), 0.10) * waiting_age_component
            - _as_float(cfg.get("lambda_P"), 0.05) * queue_pressure_component
        )
        priority = max(0.0, priority_without_guarantee + guarantee_term)
        components = dict(base_components)
        components.update(
            {
                "base_priority_score": base_priority,
                "priority_without_guarantee": priority_without_guarantee,
                "final_priority_without_guarantee": priority_without_guarantee,
                "priority_score": priority,
                "final_priority_with_guarantee": priority,
                "norm_density": norm_density[idx],
                "norm_abs_delta_utility": norm_abs[idx],
                "absolute_delta_utility": absolute_delta,
                "guarantee_debt_z": z_i,
                "Z_age": z_age,
                "Z_miss": z_miss,
                "Z_total": z_i,
                "guarantee_progress_gate": g_i_p,
                "G_i_p": g_i_p,
                "guarantee_term": guarantee_term,
                "guarantee_assisted_candidate": int(guarantee_term > 1.0e-12),
                "queue_pressure_component": queue_pressure_component,
                "waiting_age_slots": waiting_age_slots,
                "waiting_age_component": waiting_age_component,
                "closure_progress_bonus": semantic_component,
            }
        )
        request.priority_score = priority
        request.priority_components = components
    without_rank = {
        id(request): rank
        for rank, request in enumerate(
            sorted(requests, key=lambda req: (-float(req.priority_components.get("final_priority_without_guarantee", req.priority_score)), req.deadline_slot, req.payload_size_mb, req.request_id)),
            start=1,
        )
    }
    with_rank = {
        id(request): rank
        for rank, request in enumerate(
            sorted(requests, key=lambda req: (-float(req.priority_components.get("final_priority_with_guarantee", req.priority_score)), req.deadline_slot, req.payload_size_mb, req.request_id)),
            start=1,
        )
    }
    for request in requests:
        rank_without = int(without_rank.get(id(request), 0))
        rank_with = int(with_rank.get(id(request), 0))
        request.priority_components.update(
            {
                "rank_without_guarantee": rank_without,
                "rank_with_guarantee": rank_with,
                "rank_gain": rank_without - rank_with,
            }
        )
    return list(requests)


def debt_increment_for_value(value_level: str, cfg: Mapping[str, Any] | None) -> float:
    cfg = cfg or DEFAULT_GUARANTEE_CFG
    return _as_float(cfg.get("debt_increment_urgent"), 1.25) if str(value_level).lower() == "urgent" else _as_float(
        cfg.get("debt_increment_high"), 1.0
    )
