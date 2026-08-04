"""Lyapunov-inspired base pricing."""

from __future__ import annotations

from src.common.types import LongTermState


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class LyapunovPricing:
    """Compute interpretable long-horizon base prices from pressure signals."""

    def __init__(self, pricing_cfg: dict[str, float | bool]) -> None:
        self.pricing_cfg = pricing_cfg

    def compute_base_prices(self, long_state_or_context: LongTermState) -> dict[str, float]:
        """Return lambda_base = {lambda_E, lambda_L, lambda_Q, lambda_S}."""

        long_state = long_state_or_context
        battery_safe = max(float(long_state.battery_safe), 1.0e-9)
        thermal_max = max(float(long_state.thermal_max), 1.0e-9)
        queue_max_mb = max(float(long_state.queue_max_mb), 1.0e-9)

        energy_pressure = _clip01((battery_safe - float(long_state.battery_soc)) / battery_safe)
        thermal_pressure_norm = _clip01(float(long_state.thermal_pressure) / thermal_max)
        queue_pressure = _clip01(float(long_state.queue_mb) / queue_max_mb)
        deadline_pressure = _clip01(float(long_state.recent_deadline_urgency))
        egress_pressure = _clip01(queue_pressure * float(long_state.low_contact_capacity_factor))

        lambda_E = float(self.pricing_cfg["base_lambda_E"]) * (
            1.0
            + float(self.pricing_cfg["alpha_energy"]) * energy_pressure
            + float(self.pricing_cfg["alpha_thermal"]) * thermal_pressure_norm
        )
        lambda_Q = float(self.pricing_cfg["base_lambda_Q"]) * (
            1.0 + float(self.pricing_cfg["alpha_queue"]) * queue_pressure
        )
        lambda_L = float(self.pricing_cfg["base_lambda_L"]) * (
            1.0 + float(self.pricing_cfg["alpha_deadline"]) * deadline_pressure
        )
        lambda_S = float(self.pricing_cfg["base_lambda_S"]) * (
            1.0 + float(self.pricing_cfg["alpha_egress"]) * egress_pressure
        )

        return {
            "lambda_E": lambda_E,
            "lambda_L": lambda_L,
            "lambda_Q": lambda_Q,
            "lambda_S": lambda_S,
        }
