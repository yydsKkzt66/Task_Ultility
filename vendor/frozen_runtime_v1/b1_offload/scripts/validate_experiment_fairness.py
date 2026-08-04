"""Validate reusable benchmark configuration and optional existing results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from resolve_experiment_config import ForbiddenOverrideError, config_sha256, resolve


REQUIRED_PATHS = {
    "simulation.horizon_s", "simulation.slot_seconds", "simulation.slot_count",
    "dataset.name", "dataset.root", "contacts.trace_source",
    "battery.capacity_wh", "battery.initial_soc", "battery.hard_soc_floor",
    "thermal.initial_state", "thermal.guard_threshold", "thermal.hard_threshold",
    "queues.downlink.capacity_mb", "evaluator.config_source",
}


def walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from walk(child, path)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from walk(child, f"{prefix}[{i}]")
    else:
        yield prefix, value


def get_path(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        cur = cur[part]
    return cur


def validate(base_path: str | Path, method_path: str | Path, experiment_path: str | Path, result_dir: str | Path | None = None) -> tuple[str, list[dict[str, str]]]:
    checks: list[dict[str, str]] = []
    try:
        resolved, _ = resolve(base_path, method_path, experiment_path)
    except ForbiddenOverrideError as exc:
        return "BLOCKED_FORBIDDEN_OVERRIDE", [{"check": "forbidden_override", "status": "FAIL", "detail": str(exc)}]
    base = resolved["benchmark_environment"]
    for path, value in walk(resolved):
        if isinstance(value, float) and not math.isfinite(value):
            return "BLOCKED_CONFIGURATION_CONFLICT", [{"check": "finite_values", "status": "FAIL", "detail": path}]
    for path in sorted(REQUIRED_PATHS):
        try:
            item = get_path(base, path)
        except (KeyError, TypeError):
            return "BLOCKED_REQUIRED_UNKNOWN", [{"check": path, "status": "FAIL", "detail": "missing"}]
        if not isinstance(item, dict) or item.get("status") == "UNKNOWN" or item.get("value") is None:
            return "BLOCKED_REQUIRED_UNKNOWN", [{"check": path, "status": "FAIL", "detail": "required value unknown"}]
        checks.append({"check": path, "status": "PASS", "detail": str(item.get("value"))})
    statuses = [v for p, v in walk(base) if p.endswith(".status")]
    invalid = sorted(set(statuses) - {"ACTIVE", "PROXY", "UNKNOWN", "INACTIVE", "FUTURE"})
    if invalid:
        return "BLOCKED_CONFIGURATION_CONFLICT", [{"check": "status_vocabulary", "status": "FAIL", "detail": repr(invalid)}]
    if result_dir:
        manifest = Path(result_dir) / "scenario_manifest_v1.csv"
        if manifest.exists():
            rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
            if len(rows) != 5 or any(not row.get("task_hash") or not row.get("environment_hash") for row in rows):
                return "BLOCKED_HASH_MISMATCH", [{"check": "result_hashes", "status": "FAIL", "detail": "scenario manifest incomplete"}]
            checks.append({"check": "result_hashes", "status": "PASS", "detail": "five scenario hashes present"})
    proxy_count = sum(1 for status in statuses if status == "PROXY")
    unknown_count = sum(1 for status in statuses if status == "UNKNOWN")
    checks.extend([
        {"check": "canonical_config_sha256", "status": "PASS", "detail": config_sha256(base)},
        {"check": "proxy_count", "status": "INFO", "detail": str(proxy_count)},
        {"check": "unknown_count", "status": "INFO", "detail": str(unknown_count)},
    ])
    if proxy_count:
        return "PASS_WITH_PROXY_VALUES", checks
    if unknown_count:
        return "WARNING_NONCRITICAL_UNKNOWN", checks
    return "PASS", checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--result-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    label, checks = validate(args.base, args.method, args.experiment, args.result_dir)
    payload = {"status": label, "checks": checks}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(label)
    return 0 if label.startswith("PASS") or label.startswith("WARNING") else 2


if __name__ == "__main__":
    raise SystemExit(main())
