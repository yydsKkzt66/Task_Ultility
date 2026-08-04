"""Resolve a method/experiment config without allowing environment overrides."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_METHOD_KEYS = {
    "method_id", "model_selection_policy", "product_selection_policy",
    "scheduling_policy", "staged_delivery", "onboard_inference",
    "offloading_policy", "feature_flags", "notes",
    "decision_mode", "local_model", "local_result_product",
    "ground_observation_product", "ground_compute_servers",
    "ground_queue_policy", "ground_queue_capacity", "tie_tolerance_s",
    "tie_break_rule", "allow_migration", "allow_replication",
}
ALLOWED_EXPERIMENT_KEYS = {
    "experiment_id", "scenario_seeds", "capacities", "methods",
    "output_directory", "reporting_options", "mode",
}


class ForbiddenOverrideError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return data


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _reject_extra(section: str, payload: dict[str, Any], allowed: set[str]) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ForbiddenOverrideError(f"{section} contains forbidden environment override(s): {', '.join(extra)}")


def resolve(base_path: str | Path, method_path: str | Path, experiment_path: str | Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    base = load_yaml(base_path)
    method_doc = load_yaml(method_path)
    experiment_doc = load_yaml(experiment_path)
    if set(method_doc) != {"method"} or not isinstance(method_doc["method"], dict):
        raise ForbiddenOverrideError("method override may contain only the top-level 'method' mapping")
    if set(experiment_doc) != {"experiment"} or not isinstance(experiment_doc["experiment"], dict):
        raise ForbiddenOverrideError("experiment config may contain only the top-level 'experiment' mapping")
    _reject_extra("method", method_doc["method"], ALLOWED_METHOD_KEYS)
    _reject_extra("experiment", experiment_doc["experiment"], ALLOWED_EXPERIMENT_KEYS)
    resolved = {
        "benchmark_environment": base,
        "method": method_doc["method"],
        "experiment": experiment_doc["experiment"],
    }
    audit = [
        {"path": f"method.{key}", "action": "METHOD_DIFFERENCE_ACCEPTED"}
        for key in sorted(method_doc["method"])
    ] + [
        {"path": f"experiment.{key}", "action": "EXPERIMENT_SELECTION_ACCEPTED"}
        for key in sorted(experiment_doc["experiment"])
    ]
    return resolved, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    resolved, audit = resolve(args.base, args.method, args.experiment)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    digest = config_sha256(resolved)
    (out / "resolved_config.sha256").write_text(digest + "\n", encoding="ascii")
    (out / "override_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
