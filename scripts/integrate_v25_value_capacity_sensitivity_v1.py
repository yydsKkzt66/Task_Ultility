"""Integrate the completed V10/V25/V50 and V25 capacity sensitivities.

The resulting design is intentionally not presented as a full factorial grid:
V10 and V50 exist at capacity 0.50, while the capacity sweep uses V25.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
VALUE_SOURCE = ROOT / "outputs" / "_v10_v50_exact_native_five_method_controlled_fiveseed_cap050_v2"
CAPACITY_SOURCE = ROOT / "outputs" / "_v25_capacity_sensitivity_025_035_050_065_five_method_fiveseed_v1"
OUTPUT = ROOT / "outputs" / "_integrated_v25_value_capacity_sensitivity_v1"

METHODS = (
    "MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN",
    "B1-COMMON-THERMAL",
    "OFFLOAD-EFT-THERMAL-ALIGNED",
    "TUT-UOI-PRODUCT-LITE-v2",
    "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-v1",
)
SCENARIOS = (
    ("V10_CAP050", "V10", 0.50),
    ("V25_CAP025", "V25", 0.25),
    ("V25_CAP035", "V25", 0.35),
    ("V25_CAP050", "V25", 0.50),
    ("V25_CAP065", "V25", 0.65),
    ("V50_CAP050", "V50", 0.50),
)

PRIMARY_METRICS = (
    "J_all_net",
    "J_HV_net",
    "N_HV_closure",
    "N_high_closure",
    "N_urgent_closure",
)
AUXILIARY_METRICS = (
    "all_task_closure_count",
    "all_task_closure_rate",
    "N_HV_metadata",
    "N_HV_ROI",
    "N_HV_evidence",
    "successful_closure_mean_latency_s",
    "successful_closure_p95_latency_s",
    "HV_successful_closure_mean_latency_s",
    "HV_successful_closure_p95_latency_s",
    "urgent_successful_closure_mean_latency_s",
    "urgent_successful_closure_p95_latency_s",
    "deadline_censored_mean_latency_s",
    "controllable_energy_Wh",
    "transmitted_MB",
    "peak_queue_MB",
    "mean_queue_MB",
    "minimum_soc",
    "maximum_thermal",
    "SOC_violations",
    "thermal_violations",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized or [{"empty": ""}])


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalized_row(raw: Mapping[str, str], scenario_id: str, profile: str, capacity: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_id": scenario_id,
        "value_profile": profile,
        "capacity": capacity,
        "method": raw["method"],
        "seed_count": int(raw["seed_count"]),
    }
    for metric in PRIMARY_METRICS + AUXILIARY_METRICS:
        for suffix in ("mean", "std"):
            key = f"{metric}_{suffix}"
            if key not in raw:
                raise KeyError(f"missing common metric {key} for {scenario_id}/{raw['method']}")
            row[key] = float(raw[key])
    return row


def collect() -> list[dict[str, Any]]:
    value_rows = read_csv(VALUE_SOURCE / "tables" / "all_metrics_mean_std.csv")
    capacity_rows = read_csv(CAPACITY_SOURCE / "tables" / "all_metrics_mean_std.csv")
    rows: list[dict[str, Any]] = []
    for scenario_id, profile, capacity in SCENARIOS:
        if profile in {"V10", "V50"}:
            group = [row for row in value_rows if row["value_profile"] == profile]
        else:
            group = [row for row in capacity_rows if abs(float(row["capacity"]) - capacity) <= 1.0e-12]
        if {row["method"] for row in group} != set(METHODS) or len(group) != len(METHODS):
            raise RuntimeError({"METHOD_GRID_MISMATCH": scenario_id, "rows": group})
        rows.extend(normalized_row(row, scenario_id, profile, capacity) for row in group)

    scenario_order = {scenario[0]: index for index, scenario in enumerate(SCENARIOS)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    rows.sort(key=lambda row: (scenario_order[row["scenario_id"]], method_order[row["method"]]))
    if len(rows) != 30 or any(row["seed_count"] != 5 for row in rows):
        raise RuntimeError({"INCOMPLETE_INTEGRATED_GRID": len(rows)})
    return rows


def select_table(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    fields = ["scenario_id", "value_profile", "capacity", "method", "seed_count"]
    fields.extend(f"{metric}_{suffix}" for metric in metrics for suffix in ("mean", "std"))
    return [{field: row[field] for field in fields} for row in rows]


def readable_table(rows: list[dict[str, Any]], metrics: tuple[str, ...], digits: int = 3) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "scenario": row["scenario_id"],
            "method": row["method"],
        }
        for metric in metrics:
            item[metric] = f"{row[f'{metric}_mean']:.{digits}f} ± {row[f'{metric}_std']:.{digits}f}"
        output.append(item)
    return output


def combined_readable_views(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    closure_metrics = PRIMARY_METRICS + (
        "all_task_closure_count", "all_task_closure_rate",
        "N_HV_metadata", "N_HV_ROI", "N_HV_evidence",
    )
    latency_metrics = (
        "successful_closure_mean_latency_s", "successful_closure_p95_latency_s",
        "HV_successful_closure_mean_latency_s", "HV_successful_closure_p95_latency_s",
        "urgent_successful_closure_mean_latency_s", "urgent_successful_closure_p95_latency_s",
        "deadline_censored_mean_latency_s",
    )
    resource_metrics = (
        "controllable_energy_Wh", "transmitted_MB", "peak_queue_MB", "mean_queue_MB",
        "minimum_soc", "maximum_thermal", "SOC_violations", "thermal_violations",
    )
    resources = readable_table(rows, resource_metrics)
    for raw, readable in zip(rows, resources):
        if raw["method"] == "OFFLOAD-EFT-THERMAL-ALIGNED":
            readable["peak_queue_MB"] = "N/A (native trace not exposed)"
            readable["mean_queue_MB"] = "N/A (native trace not exposed)"
    return {
        "all_methods_primary_and_closure_metrics": readable_table(rows, closure_metrics),
        "all_methods_latency_metrics": readable_table(rows, latency_metrics),
        "all_methods_resource_safety_metrics": resources,
    }


def value_normalization_audit() -> dict[str, Any]:
    # Values in task_manifest.csv are finite decimal serializations of the
    # analytically normalized Decimal values.  The largest observed aggregate
    # serialization residue is 3.3e-24 Wh-equivalent utility units, so use a
    # deliberately tight tolerance that is still above that text-rounding
    # residue.  This is not a simulation/metric tolerance.
    serialized_sum_tolerance = Decimal("1e-22")
    manifests: dict[str, dict[int, Path]] = {
        "V10": {seed: VALUE_SOURCE / "config" / "V10" / f"seed_{seed}" / "task_manifest.csv" for seed in range(2026, 2031)},
        "V25": {seed: CAPACITY_SOURCE / "config" / "controlled_bundles" / f"seed_{seed}" / "task_manifest.csv" for seed in range(2026, 2031)},
        "V50": {seed: VALUE_SOURCE / "config" / "V50" / f"seed_{seed}" / "task_manifest.csv" for seed in range(2026, 2031)},
    }
    profiles: dict[str, Any] = {}
    for profile, paths in manifests.items():
        per_seed: dict[str, Any] = {}
        for seed, path in paths.items():
            rows = read_csv(path)
            total = sum((Decimal(row["effective_task_value"]) for row in rows), Decimal("0"))
            per_seed[str(seed)] = {
                "task_count": len(rows),
                "effective_task_value_sum": format(total, "f"),
                "effective_task_value_mean": format(total / Decimal(len(rows)), "f"),
                "absolute_sum_error_from_268": format(abs(total - Decimal("268")), "f"),
                "sum_matches_268_within_1e-22": abs(total - Decimal("268")) <= serialized_sum_tolerance,
            }
        profiles[profile] = per_seed
    passed = all(
        item["task_count"] == 268 and item["sum_matches_268_within_1e-22"]
        for profile in profiles.values()
        for item in profile.values()
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "normalization_contract": "analytical sum effective_task_value = task count = 268 for every profile and seed",
        "serialized_manifest_sum_tolerance": format(serialized_sum_tolerance, "f"),
        "profiles": profiles,
    }


def main(refresh: bool = False) -> int:
    if OUTPUT.exists() and not refresh:
        raise FileExistsError(f"fresh output required: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    primary = select_table(rows, PRIMARY_METRICS)
    auxiliary = select_table(rows, AUXILIARY_METRICS)
    proposed = [row for row in rows if row["method"] == METHODS[-1]]

    tables = OUTPUT / "tables"
    for name, table in (
        ("integrated_all_common_metrics", rows),
        ("integrated_primary_metrics", primary),
        ("integrated_auxiliary_metrics", auxiliary),
        ("integrated_primary_metrics_readable", readable_table(rows, PRIMARY_METRICS)),
        ("integrated_auxiliary_metrics_readable", readable_table(rows, AUXILIARY_METRICS)),
        ("corrected_reqalign_integrated_metrics", proposed),
    ):
        write_csv(tables / f"{name}.csv", table)
        write_md(tables / f"{name}.md", table)
    for name, table in combined_readable_views(rows).items():
        write_csv(tables / f"{name}.csv", table)
        write_md(tables / f"{name}.md", table)

    audit = {
        "status": "PASS",
        "design": "V10/V50 at capacity 0.50 plus V25 at capacities 0.25/0.35/0.50/0.65",
        "full_factorial": False,
        "scenario_count": len(SCENARIOS),
        "method_count": len(METHODS),
        "row_count": len(rows),
        "seed_count_per_row": 5,
        "value_source": str(VALUE_SOURCE),
        "capacity_source": str(CAPACITY_SOURCE),
        "common_primary_metrics": list(PRIMARY_METRICS),
        "common_auxiliary_metrics": list(AUXILIARY_METRICS),
    }
    (OUTPUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit" / "integration_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    normalization = value_normalization_audit()
    (OUTPUT / "audit" / "value_normalization_audit.json").write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if normalization["status"] != "PASS":
        raise RuntimeError(normalization)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    raise SystemExit(main(refresh=args.refresh))
