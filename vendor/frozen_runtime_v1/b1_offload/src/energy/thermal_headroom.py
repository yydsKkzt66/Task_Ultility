"""Exact thermal-headroom capacity using the shared thermal transition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.energy.comm_energy import compute_comm_energy_j
from src.energy.thermal_update import update_thermal_pressure


@dataclass(frozen=True)
class ThermalCapacityDecision:
    nominal_capacity_mb: float
    safe_capacity_mb: float
    predicted_nominal_thermal: float
    predicted_safe_thermal: float
    predicted_compute_only_thermal: float
    nominal_comm_energy_j: float
    safe_comm_energy_j: float
    search_iterations: int
    zero_capacity_reason: str
    monotonic: bool


def thermal_headroom_config(simulator_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = simulator_cfg.get("downlink_safety_throttle", {})
    raw = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(raw.get("enable_thermal_headroom_capacity", False)),
        "hysteresis_enabled": bool(raw.get("enable_thermal_headroom_hysteresis", False)),
        "guard": float(raw.get("thermal_guard_threshold", 0.85)),
        "hard": float(raw.get("thermal_hard_threshold", simulator_cfg.get("thermal_max", 1.0))),
        "iterations": max(1, int(raw.get("thermal_headroom_search_iterations", 40))),
        "tolerance": max(0.0, float(raw.get("thermal_headroom_tolerance", 1.0e-10))),
    }


def maximum_safe_downlink_capacity(
    *,
    current_thermal: float,
    nominal_capacity_mb: float,
    rate_mb_s: float,
    link_quality: float,
    compute_energy_j: float,
    compute_heat_gain_scale: float,
    slot_seconds: float,
    temperature_reference_c: float,
    is_sunlit: bool,
    simulator_cfg: dict[str, Any],
    force_hard_pause: bool = False,
) -> ThermalCapacityDecision:
    cfg = thermal_headroom_config(simulator_cfg)
    nominal = max(0.0, float(nominal_capacity_mb))
    hard = cfg["hard"]
    tolerance = cfg["tolerance"]
    safe_target = hard - tolerance

    def comm_energy(mb: float) -> float:
        tx_s = 0.0 if rate_mb_s <= 1.0e-12 else max(0.0, mb) / rate_mb_s
        return compute_comm_energy_j(
            comm_power_w=float(simulator_cfg.get("tx_power_w", simulator_cfg.get("comm_power_w", 0.0))),
            tx_time_s=tx_s,
            link_quality=link_quality,
            alpha_gamma=float(simulator_cfg.get("comm_alpha_gamma", 0.0)),
        )

    def thermal(mb: float) -> float:
        return update_thermal_pressure(
            thermal_pressure=current_thermal,
            compute_energy_j=max(0.0, compute_energy_j),
            comm_energy_j=comm_energy(mb),
            cooling=float(simulator_cfg.get("cooling_per_slot", 0.0)),
            compute_heat_gain_scale=compute_heat_gain_scale,
            comm_heat_gain_scale=1.0,
            slot_seconds=slot_seconds,
            reference_slot_seconds=float(simulator_cfg.get("thermal_reference_slot_seconds", slot_seconds)),
            temperature_reference_c=temperature_reference_c,
            is_sunlit=is_sunlit,
            passive_cooling=float(simulator_cfg.get("thermal_passive_cooling", 0.12)),
            sunlight_gain=float(simulator_cfg.get("thermal_sunlight_gain", 0.015)),
            eclipse_cooling_bonus=float(simulator_cfg.get("thermal_eclipse_cooling_bonus", 0.03)),
            reference_min_c=float(simulator_cfg.get("thermal_reference_min_c", -10.0)),
            reference_max_c=float(simulator_cfg.get("thermal_reference_max_c", 30.0)),
        )

    compute_only = thermal(0.0)
    nominal_thermal = thermal(nominal)
    probe = thermal(nominal * 0.5)
    monotonic = compute_only <= probe + tolerance and probe <= nominal_thermal + tolerance
    reason = ""
    if nominal <= 0.0 or rate_mb_s <= 0.0:
        safe, reason, iterations = 0.0, "NO_NOMINAL_CAPACITY", 0
    elif force_hard_pause or current_thermal >= hard - tolerance:
        safe, reason, iterations = 0.0, "CURRENT_AT_OR_ABOVE_HARD_LIMIT", 0
    elif compute_only > safe_target:
        safe, reason, iterations = 0.0, "COMPUTE_ALONE_EXHAUSTS_HEADROOM", 0
    elif nominal_thermal <= safe_target:
        safe, iterations = nominal, 0
    else:
        low, high = 0.0, nominal
        iterations = cfg["iterations"]
        for _ in range(iterations):
            mid = (low + high) * 0.5
            if thermal(mid) <= safe_target:
                low = mid
            else:
                high = mid
        safe = low
        if safe <= max(1.0e-12, nominal * 1.0e-12):
            safe, reason = 0.0, "NO_POSITIVE_SAFE_DOWNLINK_CAPACITY"
    return ThermalCapacityDecision(
        nominal_capacity_mb=nominal,
        safe_capacity_mb=safe,
        predicted_nominal_thermal=nominal_thermal,
        predicted_safe_thermal=thermal(safe),
        predicted_compute_only_thermal=compute_only,
        nominal_comm_energy_j=comm_energy(nominal),
        safe_comm_energy_j=comm_energy(safe),
        search_iterations=iterations,
        zero_capacity_reason=reason,
        monotonic=monotonic,
    )
