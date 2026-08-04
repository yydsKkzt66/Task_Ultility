"""Method-neutral single-reconstruction wrapper state machine."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

try:
    from .atomic_io import atomic_write_json
    from .output_schema import validate_unified_outputs, write_unified_outputs
except ImportError:
    from atomic_io import atomic_write_json
    from output_schema import validate_unified_outputs, write_unified_outputs


class NativeBinding(Protocol):
    method: str
    def identity(self) -> dict[str, Any]: ...
    def loaded_src_modules(self) -> dict[str, str]: ...
    def build_scenario(self, bundle: Any) -> Any: ...
    def validate_scenario(self, scenario: Any, bundle: Any) -> dict[str, Any]: ...
    def simulate(self, simulation_input: Any, slots: int) -> Any: ...
    def normalize_outputs(self, native: Any, bundle: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RunRequest:
    bundle: Any
    output: Path
    slots: int
    phase: str = "smoke"


class ReconstructionGuard:
    def __init__(self) -> None:
        self.count = 0
        self.object_id: int | None = None

    def reconstruct(self, builder):
        if self.count != 0:
            raise RuntimeError("scenario must be reconstructed exactly once")
        scenario = builder()
        self.count = 1
        self.object_id = id(scenario)
        return scenario


def validate_runtime_scenario(scenario: Any, bundle: Any) -> dict[str, Any]:
    if getattr(scenario, "task_hash", None) != bundle.contract["task_hash"]:
        raise ValueError("runtime task hash mismatch")
    if getattr(scenario, "environment_hash", None) != bundle.contract["environment_hash"]:
        raise ValueError("runtime environment hash mismatch")
    if getattr(scenario, "benchmark_fingerprint", None) != bundle.contract["benchmark_fingerprint"]:
        raise ValueError("runtime benchmark fingerprint mismatch")
    if Decimal(str(getattr(scenario, "config_value_multiplier", "NaN"))) != Decimal("1"):
        raise ValueError("runtime config value_multiplier is not exactly one")
    values = dict(getattr(scenario, "runtime_task_values", {}))
    if set(values) != set(bundle.task_values):
        raise ValueError("runtime task ID coverage mismatch")
    mismatches = []
    for task_id, expected in bundle.task_values.items():
        actual = values[task_id]
        # Compare the exact IEEE-754 values implied by the canonical lexical token.
        if Decimal.from_float(float(actual)) != Decimal.from_float(float(expected)):
            mismatches.append(task_id)
    if mismatches:
        raise ValueError(f"runtime effective_task_value mismatch: {mismatches[:5]}")
    return {
        "task_hash": scenario.task_hash, "environment_hash": scenario.environment_hash,
        "benchmark_fingerprint": scenario.benchmark_fingerprint,
        "task_count": len(values), "runtime_value_mismatch_count": 0,
        "config_value_multiplier": "1",
    }


def execute_identity(binding: NativeBinding) -> dict[str, Any]:
    result = dict(binding.identity())
    result["scenario_reconstruction_count"] = 0
    result["src_module_files"] = dict(binding.loaded_src_modules())
    return result


def execute_run(binding: NativeBinding, request: RunRequest) -> dict[str, Any]:
    try:
        from .diagnostics import DiagnosticProbe
    except ImportError:
        from diagnostics import DiagnosticProbe
    output = Path(request.output)
    output.mkdir(parents=True, exist_ok=True)
    guard = ReconstructionGuard()
    scenario_object_id = None
    simulation_input_object_id = None
    simulation_started = False
    event_count = 0
    state = "CREATED"
    probe = DiagnosticProbe(output)
    binding.diagnostic_probe = probe
    try:
        identity = dict(binding.identity())
        if not identity.get("source_identity_passed", False):
            raise RuntimeError("native source identity failed")
        pre_modules = dict(binding.loaded_src_modules())
        state = "IDENTITY_VERIFIED"
        scenario = guard.reconstruct(lambda: binding.build_scenario(request.bundle))
        scenario_object_id = id(scenario)
        state = "SCENARIO_RECONSTRUCTED"
        scenario_identity = dict(binding.validate_scenario(scenario, request.bundle))
        state = "SCENARIO_VALIDATED"
        simulation_input = scenario
        if simulation_input is not scenario:
            raise RuntimeError("simulation input must be the reconstructed scenario object")
        simulation_input_object_id = id(simulation_input)
        if guard.count != 1 or simulation_input_object_id != scenario_object_id:
            raise RuntimeError("single reconstruction or object identity assertion failed")
        simulation_started = True
        state = "SIMULATION_STARTED"
        native = binding.simulate(simulation_input, request.slots)
        event_count = int((native.get("event_count", 0) if isinstance(native, dict) else getattr(native, "event_count", 0)) or 0)
        state = "SIMULATION_COMPLETED"
        normalized = dict(binding.normalize_outputs(native, request.bundle))
        post_modules = dict(binding.loaded_src_modules())
        source_root = Path(identity["source_root"]).resolve()
        if not post_modules or not all(Path(path).resolve().is_relative_to(source_root) for path in post_modules.values()):
            raise RuntimeError("src module isolation failed before process exit")
        validate_unified_outputs(normalized, expected_tasks=len(request.bundle.tasks), expected_slots=request.slots)
        write_unified_outputs(output, normalized)
        state = "OUTPUT_VALIDATED"
        result = {
            "status": "PASS", "phase": request.phase, "method": binding.method,
            "scenario_reconstruction_count": guard.count,
            "scenario_object_id": scenario_object_id,
            "simulation_input_object_id": simulation_input_object_id,
            "simulation_started": simulation_started, "event_count": event_count,
            "state": state, "identity": identity, "scenario_identity": scenario_identity,
            "src_modules_after_imports": pre_modules, "src_modules_before_exit": post_modules,
            "normalized": normalized,
        }
        atomic_write_json(output / "native_identity.json", identity)
        atomic_write_json(output / "reconstructed_scenario_identity.json", scenario_identity)
        atomic_write_json(output / "method_run_status.json", result)
        atomic_write_json(output / "method_completed.json", {
            "phase": request.phase, "method": binding.method, "status": "PASS",
            "scenario_reconstruction_count": guard.count,
        })
        return result
    except BaseException as error:
        failure = {
            "status": "FAIL", "phase": request.phase, "method": binding.method,
            "failure_stage": state, "error_type": type(error).__name__, "error": str(error),
            "scenario_reconstruction_count": guard.count,
            "scenario_object_id": scenario_object_id,
            "simulation_input_object_id": simulation_input_object_id,
            "simulation_started": simulation_started, "event_count": event_count,
        }
        atomic_write_json(output / "method_run_status.json", failure)
        raise
