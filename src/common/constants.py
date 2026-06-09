"""项目级常量。

这里集中放置跨模块共享的名称、动作定义与默认映射，避免在各处散落魔法值。
"""

from __future__ import annotations

CONTEXT_NAMES = (
    "B_t",
    "E_t_AI",
    "Theta_t",
    "V_t",
    "D_t",
    "H_t",
    "U_t",
    "C_t",
    "Q_t",
)

ACTION_FALLBACK_NAME = "delay/store"

DEFAULT_ACTION_SPECS = (
    ("lightweight+metadata", "lightweight", "metadata", "contact_aware", 0.20, 0.20),
    ("balanced+metadata", "balanced", "metadata", "contact_aware", 0.20, 0.55),
    ("balanced+roi", "balanced", "roi", "contact_aware", 0.55, 0.55),
    ("high_accuracy+roi_context", "high_accuracy", "roi_context", "contact_aware", 0.95, 0.95),
    ("delay/store", "delay_store", "defer", "store", 0.00, 0.00),
)

PAYLOAD_INFORMATION_LEVEL = {
    "metadata": 0.35,
    "roi": 0.72,
    "roi_context": 1.00,
    "defer": 0.05,
}

PAYLOAD_BASE_SIZE_MB = {
    "metadata": 0.10,
    "roi": 0.80,
    "roi_context": 1.60,
    "defer": 0.00,
}

DELIVERY_STATUS_DELIVERED = "delivered"
DELIVERY_STATUS_QUEUED = "queued"
DELIVERY_STATUS_DROPPED = "dropped"
DELIVERY_STATUS_DEFERRED = "deferred"
DELIVERY_STATUS_EMPTY = "empty"

EPSILON = 1.0e-9
