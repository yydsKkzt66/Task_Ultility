"""Isolated two-node thermal diagnostic primitives.

This module is deliberately independent of every frozen runtime ``src`` tree.
It contains the single accounting/integration implementation used by the
diagnostic predictor, safety probe and executor adapters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
import inspect
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TwoNodeParameters:
    r_chip_surface_k_per_w: float
    r_surface_env_k_per_w: float
    c_chip_j_per_k: float
    c_surface_j_per_k: float
    compute_to_chip_fraction: float = 1.0
    comm_to_surface_fraction: float = 1.0
    t_chip_max_c: float = 26.8
    t_surface_max_c: float = 26.8
    reference_min_c: float = -10.0
    reference_max_c: float = 30.0
    max_substep_s: float = 1.0

    def validate(self) -> None:
        if min(
            self.r_chip_surface_k_per_w,
            self.r_surface_env_k_per_w,
            self.c_chip_j_per_k,
            self.c_surface_j_per_k,
            self.max_substep_s,
        ) <= 0.0:
            raise ValueError("all R, C and substep parameters must be positive")
        if not 0.0 <= self.compute_to_chip_fraction <= 1.0:
            raise ValueError("invalid compute heat split")
        if not 0.0 <= self.comm_to_surface_fraction <= 1.0:
            raise ValueError("invalid communication heat split")

    def stable_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ThermalState:
    chip_c: float
    surface_c: float


@dataclass(frozen=True)
class LoadPhase:
    name: str
    duration_s: float
    compute_power_w: float = 0.0
    comm_power_w: float = 0.0


@dataclass(frozen=True)
class ActionAccounting:
    inference_energy_j: float
    encoding_energy_j: float
    communication_energy_j: float
    inference_duration_s: float
    encoding_duration_s: float
    communication_duration_s: float
    idle_duration_s: float

    @property
    def compute_energy_j(self) -> float:
        return self.inference_energy_j + self.encoding_energy_j

    @property
    def total_energy_j(self) -> float:
        return self.compute_energy_j + self.communication_energy_j


@dataclass(frozen=True)
class ThermalTransition:
    initial: ThermalState
    final: ThermalState
    peak_chip_c: float
    peak_surface_c: float
    delta_chip_c: float
    delta_surface_c: float
    accounting: ActionAccounting


MODEL_BY_SCALE = {
    1.0: ("lightweight", 10.8, 0.90),
    1.6: ("balanced", 21.6, 1.44),
    3.0: ("high_accuracy", 51.84, 2.79),
}


def model_energy_duration(scale: float, compute_energy_j: float) -> tuple[str, float, float]:
    if compute_energy_j <= 1.0e-12:
        return "idle", 0.0, 0.0
    key = min(MODEL_BY_SCALE, key=lambda item: abs(item - float(scale)))
    name, energy, duration = MODEL_BY_SCALE[key]
    if compute_energy_j + 1.0e-9 < energy:
        return "encoding_or_partial", 0.0, 0.0
    return name, energy, duration


def build_accounting(
    *,
    compute_energy_j: float,
    comm_energy_j: float,
    compute_heat_gain_scale: float,
    slot_seconds: float,
    encoding_power_w: float = 14.0,
    radio_power_w: float = 38.0,
) -> ActionAccounting:
    """Split already-accounted joules exactly once into timed load phases."""
    compute = max(0.0, float(compute_energy_j))
    comm = max(0.0, float(comm_energy_j))
    _name, inference_j, inference_s = model_energy_duration(compute_heat_gain_scale, compute)
    inference_j = min(inference_j, compute)
    encoding_j = max(0.0, compute - inference_j)
    encoding_s = 0.0 if encoding_j <= 0.0 else encoding_j / max(encoding_power_w, 1.0e-12)
    comm_s = 0.0 if comm <= 0.0 else comm / max(radio_power_w, 1.0e-12)
    active = inference_s + encoding_s + comm_s
    # A pathological schedule is serialized and clipped only in its final idle
    # phase; no energy is discarded when active work exceeds one slot.
    idle = max(0.0, float(slot_seconds) - active)
    return ActionAccounting(inference_j, encoding_j, comm, inference_s, encoding_s, comm_s, idle)


def phases_from_accounting(accounting: ActionAccounting) -> list[LoadPhase]:
    phases: list[LoadPhase] = []
    if accounting.inference_duration_s > 0:
        phases.append(LoadPhase("inference", accounting.inference_duration_s, accounting.inference_energy_j / accounting.inference_duration_s, 0.0))
    if accounting.encoding_duration_s > 0:
        phases.append(LoadPhase("encoding", accounting.encoding_duration_s, accounting.encoding_energy_j / accounting.encoding_duration_s, 0.0))
    if accounting.communication_duration_s > 0:
        phases.append(LoadPhase("communication", accounting.communication_duration_s, 0.0, accounting.communication_energy_j / accounting.communication_duration_s))
    if accounting.idle_duration_s > 0:
        phases.append(LoadPhase("idle", accounting.idle_duration_s, 0.0, 0.0))
    return phases


@lru_cache(maxsize=32)
def _linear_system(r1: float, r2: float, c1: float, c2: float):
    a = np.array([
        [-1.0 / (r1 * c1), 1.0 / (r1 * c1)],
        [1.0 / (r1 * c2), -(1.0 / r1 + 1.0 / r2) / c2],
    ])
    eigenvalues, eigenvectors = np.linalg.eig(a)
    return a, eigenvalues, eigenvectors, np.linalg.inv(eigenvectors)


def _exact_step(state: ThermalState, phase: LoadPhase, env_c: float, params: TwoNodeParameters) -> ThermalState:
    dt = max(0.0, float(phase.duration_s))
    if dt <= 0.0:
        return state
    r1, r2 = params.r_chip_surface_k_per_w, params.r_surface_env_k_per_w
    c1, c2 = params.c_chip_j_per_k, params.c_surface_j_per_k
    p_chip = params.compute_to_chip_fraction * phase.compute_power_w + (1.0 - params.comm_to_surface_fraction) * phase.comm_power_w
    p_surface = (1.0 - params.compute_to_chip_fraction) * phase.compute_power_w + params.comm_to_surface_fraction * phase.comm_power_w
    a, eigenvalues, eigenvectors, inverse_eigenvectors = _linear_system(r1, r2, c1, c2)
    b = np.array([p_chip / c1, p_surface / c2 + env_c / (r2 * c2)])
    equilibrium = -np.linalg.solve(a, b)
    coefficients = inverse_eigenvectors @ (np.array([state.chip_c, state.surface_c]) - equilibrium)
    out = equilibrium + eigenvectors @ (np.exp(eigenvalues * dt) * coefficients)
    return ThermalState(float(out[0]), float(out[1]))


def integrate(
    state: ThermalState,
    phases: Iterable[LoadPhase],
    env_c: float,
    params: TwoNodeParameters,
) -> tuple[ThermalState, float, float]:
    params.validate()
    current = state
    peak_chip, peak_surface = state.chip_c, state.surface_c
    for phase in phases:
        remaining = max(0.0, phase.duration_s)
        while remaining > 1.0e-12:
            dt = min(params.max_substep_s, remaining)
            current = _exact_step(current, LoadPhase(phase.name, dt, phase.compute_power_w, phase.comm_power_w), env_c, params)
            peak_chip = max(peak_chip, current.chip_c)
            peak_surface = max(peak_surface, current.surface_c)
            remaining -= dt
    return current, peak_chip, peak_surface


def transition(
    state: ThermalState,
    accounting: ActionAccounting,
    env_c: float,
    params: TwoNodeParameters,
) -> ThermalTransition:
    final, peak_chip, peak_surface = integrate(state, phases_from_accounting(accounting), env_c, params)
    return ThermalTransition(
        initial=state,
        final=final,
        peak_chip_c=peak_chip,
        peak_surface_c=peak_surface,
        delta_chip_c=final.chip_c - state.chip_c,
        delta_surface_c=final.surface_c - state.surface_c,
        accounting=accounting,
    )


def theta_from_state(state: ThermalState, params: TwoNodeParameters) -> float:
    span = params.reference_max_c - params.reference_min_c
    return max(state.chip_c, state.surface_c) / span - params.reference_min_c / span


def state_from_theta(theta: float, params: TwoNodeParameters) -> ThermalState:
    temp = params.reference_min_c + float(theta) * (params.reference_max_c - params.reference_min_c)
    return ThermalState(temp, temp)


def thermal_safe(result: ThermalTransition, params: TwoNodeParameters) -> bool:
    return result.peak_chip_c <= params.t_chip_max_c + 1.0e-12 and result.peak_surface_c <= params.t_surface_max_c + 1.0e-12


class LegacyScalarAdapter:
    """Process-local reversible state bridge for frozen scalar runtimes.

    Each scalar returned to the frozen runtime is registered with its two-node
    state. Candidate predictions do not mutate the physical state; the executor
    obtains the same transition for identical inputs.
    """

    def __init__(self, params: TwoNodeParameters):
        self.params = params
        self._states: dict[str, ThermalState] = {}
        self.rows: list[dict[str, float | str]] = []

    @staticmethod
    def _key(theta: float) -> str:
        return format(float(theta), ".15g")

    def state_for(self, theta: float) -> ThermalState:
        return self._states.get(self._key(theta), state_from_theta(theta, self.params))

    def update(
        self,
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
        del cooling, comm_heat_gain_scale, reference_slot_seconds, is_sunlit, passive_cooling, sunlight_gain, eclipse_cooling_bonus, reference_min_c, reference_max_c
        before = self.state_for(thermal_pressure)
        accounting = build_accounting(
            compute_energy_j=compute_energy_j,
            comm_energy_j=comm_energy_j,
            compute_heat_gain_scale=compute_heat_gain_scale,
            slot_seconds=slot_seconds,
        )
        result = transition(before, accounting, float(temperature_reference_c), self.params)
        # The frozen scalar interface has no separate peak field.  Return the
        # peak-normalized safety pressure while registering the true end state
        # for the next transition.  This makes SafeFilter conservative without
        # discarding the two-node memory.
        peak_state = ThermalState(result.peak_chip_c, result.peak_surface_c)
        theta = theta_from_state(peak_state, self.params)
        self._states[self._key(theta)] = result.final
        caller = inspect.currentframe().f_back
        caller_name = caller.f_code.co_name if caller is not None else "unknown"
        caller_file = caller.f_code.co_filename if caller is not None else "unknown"
        self.rows.append({
            "caller": caller_name,
            "caller_file": caller_file,
            "theta_before": thermal_pressure,
            "theta_after": theta,
            "chip_before_c": before.chip_c,
            "surface_before_c": before.surface_c,
            "chip_after_c": result.final.chip_c,
            "surface_after_c": result.final.surface_c,
            "peak_chip_c": result.peak_chip_c,
            "peak_surface_c": result.peak_surface_c,
            "compute_energy_j": compute_energy_j,
            "comm_energy_j": comm_energy_j,
            "inference_energy_j": accounting.inference_energy_j,
            "encoding_energy_j": accounting.encoding_energy_j,
            "communication_energy_j": accounting.communication_energy_j,
            "slot_seconds": slot_seconds,
            "environment_c": temperature_reference_c,
        })
        return theta


def time_constants_s(params: TwoNodeParameters) -> list[float]:
    r1, r2 = params.r_chip_surface_k_per_w, params.r_surface_env_k_per_w
    c1, c2 = params.c_chip_j_per_k, params.c_surface_j_per_k
    a = np.array([
        [-1.0 / (r1 * c1), 1.0 / (r1 * c1)],
        [1.0 / (r1 * c2), -(1.0 / r1 + 1.0 / r2) / c2],
    ])
    return sorted(float(-1.0 / value.real) for value in np.linalg.eigvals(a))
