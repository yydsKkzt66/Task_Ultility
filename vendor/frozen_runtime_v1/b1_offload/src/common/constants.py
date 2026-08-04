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

DELIVERY_STATUS_DELIVERED = "delivered"
DELIVERY_STATUS_QUEUED = "queued"
DELIVERY_STATUS_DROPPED = "dropped"
DELIVERY_STATUS_EXPIRED = "expired"
DELIVERY_STATUS_DEFERRED = "deferred"
DELIVERY_STATUS_EMPTY = "empty"

EPSILON = 1.0e-9
