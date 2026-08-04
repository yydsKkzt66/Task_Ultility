"""Information sufficiency and product-aware incremental utility helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.common.types import ContextVector


PRODUCT_REQUIREMENTS = ("alert", "verification", "context", "evidence")
PRODUCT_TYPES = ("metadata", "roi", "roi_context")

DEFAULT_PRODUCT_REQUIREMENT_UTILITY_TABLE: dict[str, dict[str, float]] = {
    "empty": {"alert": 0.0, "verification": 0.0, "context": 0.0, "evidence": 0.0},
    "metadata": {"alert": 0.80, "verification": 0.35, "context": 0.20, "evidence": 0.15},
    "roi": {"alert": 0.90, "verification": 0.90, "context": 0.55, "evidence": 0.65},
    "roi_context": {"alert": 0.95, "verification": 0.96, "context": 0.92, "evidence": 0.95},
    "metadata+roi": {"alert": 0.90, "verification": 0.90, "context": 0.55, "evidence": 0.65},
    "metadata+roi_context": {"alert": 0.95, "verification": 0.96, "context": 0.92, "evidence": 0.95},
}


def compute_information_requirement(context: ContextVector, reward_cfg: dict[str, float]) -> float:
    """Compute the legacy scalar information requirement."""

    return (
        float(reward_cfg["info_base"])
        + float(reward_cfg["info_value_weight"]) * context.V_t
        + float(reward_cfg["info_difficulty_weight"]) * context.H_t
        + float(reward_cfg["info_uncertainty_weight"]) * context.U_t
        + float(reward_cfg["info_urgency_weight"]) * context.D_t
    )


def compute_information_sufficiency(
    context: ContextVector,
    payload_information: float,
    reward_cfg: dict[str, float],
) -> float:
    """Compute the legacy saturated scalar information sufficiency."""

    requirement = max(compute_information_requirement(context, reward_cfg), 1.0e-9)
    return min(1.0, payload_information / requirement)


def normalize_product_requirement(product_requirement: str | None) -> str:
    requirement = str(product_requirement or "verification").strip().lower()
    return requirement if requirement in PRODUCT_REQUIREMENTS else "verification"


def normalize_product_type(product_type: str | None) -> str:
    product = str(product_type or "").strip().lower()
    if product in PRODUCT_TYPES or product == "defer":
        return product
    return product


def normalize_delivered_product_set(delivered_products: Iterable[str] | None) -> str:
    products = {normalize_product_type(product) for product in (delivered_products or [])}
    products = {product for product in products if product in PRODUCT_TYPES}
    if not products:
        return "empty"
    if "roi_context" in products:
        return "metadata+roi_context" if "metadata" in products else "roi_context"
    if "roi" in products:
        return "metadata+roi" if "metadata" in products else "roi"
    if "metadata" in products:
        return "metadata"
    return "empty"


def _utility_table(
    product_utility_cfg: Mapping[str, object] | None = None,
) -> Mapping[str, Mapping[str, float]]:
    if not product_utility_cfg:
        return DEFAULT_PRODUCT_REQUIREMENT_UTILITY_TABLE
    table = product_utility_cfg.get("utility_table", product_utility_cfg)
    if not isinstance(table, Mapping):
        return DEFAULT_PRODUCT_REQUIREMENT_UTILITY_TABLE
    return table  # type: ignore[return-value]


def product_information_utility(
    delivered_products: Iterable[str] | None,
    product_requirement: str,
    product_utility_cfg: Mapping[str, object] | None = None,
) -> float:
    delivered_key = normalize_delivered_product_set(delivered_products)
    requirement = normalize_product_requirement(product_requirement)
    table = _utility_table(product_utility_cfg)
    values = table.get(delivered_key, table.get("empty", {}))
    return max(0.0, min(1.0, float(values.get(requirement, 0.0))))


def delta_information_utility(
    product_type: str,
    delivered_products: Iterable[str] | None,
    product_requirement: str,
    product_utility_cfg: Mapping[str, object] | None = None,
) -> float:
    product = normalize_product_type(product_type)
    if product not in PRODUCT_TYPES:
        return 0.0
    delivered = set(delivered_products or [])
    before = product_information_utility(delivered, product_requirement, product_utility_cfg)
    after = product_information_utility(delivered | {product}, product_requirement, product_utility_cfg)
    return max(0.0, after - before)


def residual_information_utility(
    delivered_products: Iterable[str] | None,
    product_requirement: str,
    product_utility_cfg: Mapping[str, object] | None = None,
) -> float:
    current = product_information_utility(delivered_products, product_requirement, product_utility_cfg)
    target = product_information_utility({"metadata", "roi_context"}, product_requirement, product_utility_cfg)
    return max(0.0, target - current)


def validate_product_utility_table(
    product_utility_cfg: Mapping[str, object] | None = None,
) -> None:
    table = _utility_table(product_utility_cfg)
    for requirement in PRODUCT_REQUIREMENTS:
        empty = float(table["empty"][requirement])
        metadata = float(table["metadata"][requirement])
        roi = float(table["roi"][requirement])
        roi_context = float(table["roi_context"][requirement])
        metadata_roi = float(table["metadata+roi"][requirement])
        metadata_roi_context = float(table["metadata+roi_context"][requirement])
        values = [empty, metadata, roi, roi_context, metadata_roi, metadata_roi_context]
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"Product utility values for {requirement} must be in [0, 1]")
        if metadata < empty or roi < empty or roi_context < roi:
            raise ValueError(f"Product utility table is not monotone for {requirement}")
        if metadata_roi < metadata or metadata_roi < roi:
            raise ValueError(f"metadata+roi is not monotone for {requirement}")
        if metadata_roi_context < metadata or metadata_roi_context < roi_context:
            raise ValueError(f"metadata+roi_context is not monotone for {requirement}")


def clamp_residual_usefulness_gate(rho_res: float | None = None) -> float:
    if rho_res is None:
        return 1.0
    return max(0.0, min(1.0, float(rho_res)))


def compute_gross_incremental_utility(
    task_value: float,
    detection_quality: float,
    delta_u_i: float,
    timeliness: float,
    rho_res: float | None = None,
) -> float:
    residual_usefulness_gate = clamp_residual_usefulness_gate(rho_res)
    return (
        max(0.0, float(task_value))
        * max(0.0, float(detection_quality))
        * max(0.0, float(delta_u_i))
        * max(0.0, float(timeliness))
        * residual_usefulness_gate
    )


def compute_net_incremental_utility(
    gross_incremental_utility: float,
    *,
    resource_cost: float = 0.0,
    energy_cost: float = 0.0,
    queue_cost: float = 0.0,
    latency_cost: float = 0.0,
    thermal_cost: float = 0.0,
    safety_penalty: float = 0.0,
) -> float:
    return (
        float(gross_incremental_utility)
        - float(resource_cost)
        - float(energy_cost)
        - float(queue_cost)
        - float(latency_cost)
        - float(thermal_cost)
        - float(safety_penalty)
    )
