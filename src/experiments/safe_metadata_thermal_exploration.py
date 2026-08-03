"""Pure helpers for the isolated safe/metadata/thermal exploration.

This module deliberately owns no simulator state.  The experiment runner loads
it under a private module name and supplies frozen runtime values explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ThermalPriceSnapshot:
    current_thermal: float
    thermal_headroom: float
    proximity_pressure: float
    cooling_weakness: float
    recent_compute_pressure: float
    hv_residual_pressure: float
    future_headroom_scarcity: float
    alpha_thermal: float
    thermal_price: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def normalized_thermal_increment(
    *,
    action_theta_next: float,
    defer_theta_next: float,
    thermal_initial: float,
    thermal_max: float,
) -> float:
    """Return the action-only predicted heat increment on a unitless scale."""

    span = max(float(thermal_max) - float(thermal_initial), 1.0e-12)
    return max(0.0, float(action_theta_next) - float(defer_theta_next)) / span


def thermal_price_snapshot(
    *,
    current_thermal: float,
    defer_theta_next: float,
    thermal_initial: float,
    thermal_max: float,
    recent_compute_pressure: float,
    has_pending_hv_residual: bool,
    alpha_thermal: float,
) -> ThermalPriceSnapshot:
    """Build the no-new-tuning experimental thermal price.

    All five pressure signals are normalized to [0, 1] and combined by their
    arithmetic mean.  The only multiplier is the already-frozen
    ``alpha_thermal``.  This avoids introducing fitted weights or thresholds.
    """

    theta = float(current_thermal)
    theta_idle = float(defer_theta_next)
    theta0 = float(thermal_initial)
    theta_max = float(thermal_max)
    span = max(theta_max - theta0, 1.0e-12)
    proximity = clamp01((theta - theta0) / span)
    cooling_capacity = clamp01((theta - theta_idle) / span)
    cooling_weakness = 1.0 - cooling_capacity
    recent = clamp01(recent_compute_pressure)
    residual = 1.0 if has_pending_hv_residual else 0.0
    projected_theta = max(theta, theta_idle)
    future_scarcity = 1.0 - clamp01((theta_max - projected_theta) / span)
    price = max(0.0, float(alpha_thermal)) * (
        proximity + cooling_weakness + recent + residual + future_scarcity
    ) / 5.0
    return ThermalPriceSnapshot(
        current_thermal=theta,
        thermal_headroom=theta_max - theta,
        proximity_pressure=proximity,
        cooling_weakness=cooling_weakness,
        recent_compute_pressure=recent,
        hv_residual_pressure=residual,
        future_headroom_scarcity=future_scarcity,
        alpha_thermal=float(alpha_thermal),
        thermal_price=price,
    )


def choose_ranked_fallback(
    rows: Iterable[Mapping[str, Any]],
    *,
    original_product: str,
) -> Mapping[str, Any] | None:
    """Choose the first non-empty deterministic fallback tier.

    Tier order is exactly: same-product Balanced, same-product Lightweight,
    any Metadata, then no action (caller uses Defer).  Within a tier the
    original V2 net score remains the sole utility ranking signal.
    """

    candidates = [row for row in rows if bool(int(row.get("unified_guard_pass", 0)))]
    tiers = (
        [row for row in candidates if row.get("product") == original_product and row.get("model") == "balanced"],
        [row for row in candidates if row.get("product") == original_product and row.get("model") == "lightweight"],
        [row for row in candidates if row.get("product") == "metadata"],
    )
    for tier in tiers:
        if tier:
            return max(tier, key=lambda row: (float(row.get("v2_net_score", float("-inf"))), str(row.get("action", ""))))
    return None
