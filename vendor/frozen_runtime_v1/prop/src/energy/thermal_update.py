"""Thermal pressure update."""

from __future__ import annotations


def update_thermal_pressure(
    thermal_pressure: float,
    compute_energy_j: float,
    comm_energy_j: float,
    cooling: float,
    compute_heat_gain_scale: float = 1.0,
    comm_heat_gain_scale: float = 1.0,
    slot_seconds: float = 60.0,
    reference_slot_seconds: float = 60.0,
    temperature_reference_c: float = 10.0,
    is_sunlit: bool = True,
    passive_cooling: float = 0.12,
    sunlight_gain: float = 0.015,
    eclipse_cooling_bonus: float = 0.03,
    reference_min_c: float = -10.0,
    reference_max_c: float = 30.0,
) -> float:
    """Update normalized thermal pressure with dt-aware environmental terms.

    Compute and communication heating are driven by Joules and are already
    physical-time aware. Cooling, passive relaxation, sunlight heating, and the
    eclipse bonus are configured per reference slot and scaled by dt.
    """

    dt_scale = max(float(slot_seconds), 0.0) / max(float(reference_slot_seconds), 1.0e-9)
    heat_from_compute = 0.0025 * compute_energy_j * max(float(compute_heat_gain_scale), 0.0)
    heat_from_comm = 0.0015 * comm_energy_j * max(float(comm_heat_gain_scale), 0.0)
    ambient_norm = (temperature_reference_c - reference_min_c) / max(reference_max_c - reference_min_c, 1.0e-9)
    ambient_norm = max(0.0, min(1.0, ambient_norm))
    relaxation = passive_cooling * dt_scale * (thermal_pressure - ambient_norm)
    solar_term = sunlight_gain * dt_scale if is_sunlit else 0.0
    eclipse_cooling = eclipse_cooling_bonus * dt_scale if not is_sunlit else 0.0

    next_value = thermal_pressure + heat_from_compute + heat_from_comm + solar_term - relaxation - cooling * dt_scale - eclipse_cooling
    return max(0.0, min(1.0, next_value))
