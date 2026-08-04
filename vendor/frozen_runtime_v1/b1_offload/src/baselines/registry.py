"""Method-independent registration contract for future single-satellite baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

OUTPUT_FIELDS = (
    "selected_model", "selected_product", "generated_bytes", "transmitted_bytes",
    "compute_energy_j", "communication_energy_j", "minimum_soc", "final_soc",
    "maximum_thermal_state", "closure_outcome", "utility", "latency_s",
)

@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    execution_location: str
    model_policy: str
    product_policy: str
    uses_common_queue: bool = True
    uses_common_thermal: bool = True
    uses_common_battery: bool = True
    enabled: bool = False

REGISTRY = {
    "FIXED-LIGHT": BaselineSpec("FIXED-LIGHT", "static_policy", "onboard", "fixed_lightweight", "existing_dynamic_product"),
    "FIXED-HEAVY": BaselineSpec("FIXED-HEAVY", "static_policy", "onboard", "fixed_heavy", "existing_dynamic_product"),
    "GROUND-ONLY": BaselineSpec("GROUND-ONLY", "ground_only_runnable", "ground", "ground_inference", "b1_compressed_full_frame"),
    "FIXED-SPLIT": BaselineSpec("FIXED-SPLIT", "external", "split", "fixed_onboard", "fixed_stage"),
    "OFFLOAD-EFT": BaselineSpec("OFFLOAD-EFT", "binary_offloading", "local_or_ground", "fixed_high_accuracy", "fixed_roi_or_compressed_full"),
    "OFFLOAD-EFT-THERMAL-ALIGNED": BaselineSpec("OFFLOAD-EFT-THERMAL-ALIGNED", "binary_offloading", "local_or_ground", "fixed_high_accuracy", "fixed_roi_or_compressed_full"),
}

def get_baseline(name: str) -> BaselineSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown baseline: {name}") from exc

def validate_output(row: Mapping[str, Any]) -> None:
    missing = [field for field in OUTPUT_FIELDS if field not in row]
    if missing:
        raise ValueError(f"baseline output missing fields: {', '.join(missing)}")

def validate_fairness(spec: BaselineSpec) -> None:
    if not (spec.uses_common_queue and spec.uses_common_thermal and spec.uses_common_battery):
        raise ValueError(f"{spec.name} violates the common physical-model contract")

def smoke_run(name: str) -> dict[str, Any]:
    """Return a schema-valid zero-work record; this does not simulate a baseline."""
    spec = get_baseline(name);validate_fairness(spec)
    row = {field: 0 for field in OUTPUT_FIELDS}
    row.update({"selected_model": spec.model_policy, "selected_product": spec.product_policy, "closure_outcome": "SMOKE_ONLY"})
    validate_output(row);return row
