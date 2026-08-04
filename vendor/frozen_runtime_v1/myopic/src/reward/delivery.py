"""交付因子计算。"""

from __future__ import annotations

from src.common.constants import (
    DELIVERY_STATUS_DELIVERED,
    DELIVERY_STATUS_DEFERRED,
    DELIVERY_STATUS_DROPPED,
    DELIVERY_STATUS_EXPIRED,
    DELIVERY_STATUS_QUEUED,
)


def compute_delivery_factor(
    delivery_status: str,
    delivery_beta: float,
    deferred_credit: float,
) -> float:
    """计算 current-task delivery factor。"""

    if delivery_status == DELIVERY_STATUS_DELIVERED:
        return 1.0
    if delivery_status == DELIVERY_STATUS_QUEUED:
        return delivery_beta
    if delivery_status == DELIVERY_STATUS_DEFERRED:
        return deferred_credit
    if delivery_status == DELIVERY_STATUS_DROPPED:
        return 0.0
    if delivery_status == DELIVERY_STATUS_EXPIRED:
        return 0.0
    return 0.0
