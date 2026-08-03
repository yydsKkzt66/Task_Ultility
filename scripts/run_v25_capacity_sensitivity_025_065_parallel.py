"""Run only the missing V25 capacity points (0.25 and 0.65) in parallel.

Completed exact-native capacity-0.35 and capacity-0.50 experiments are read as
immutable references for the final four-point report.  All 50 runs at 0.25 and
0.65 are fresh child simulations using the same controlled V25 bundles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402


NEW_CAPACITIES = (0.25, 0.65)
ALL_CAPACITIES = (0.25, 0.35, 0.50, 0.65)
REFERENCE_OUTPUTS = {
    0.35: ROOT / "outputs" / "_v25_exact_native_five_method_controlled_fiveseed_cap035_v1",
    0.50: ROOT / "outputs" / "_v25_exact_native_five_method_controlled_fiveseed_cap050_parallel_rerun_v1",
}
BUNDLE_REFERENCE = REFERENCE_OUTPUTS[0.50]
DEFAULT_OUTPUT = ROOT / "outputs" / "_v25_capacity_sensitivity_025_035_050_065_five_method_fiveseed_v1"


def capacity_tag(capacity: float) -> str:
    return f"CAP{int(round(capacity * 100)):03d}"


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized or [{"empty": ""}])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_path(output: Path, capacity: float, seed: int, method: str) -> Path:
    return output / "runs" / capacity_tag(capacity) / f"seed_{seed}" / method


def status_path(output: Path, capacity: float, seed: int, method: str) -> Path:
    return run_path(output, capacity, seed, method) / "method_run_status.json"


def is_complete(output: Path, capacity: float, seed: int, method: str) -> bool:
    path = run_path(output, capacity, seed, method)
    return (path / "method_completed.json").is_file() and status_path(output, capacity, seed, method).is_file()


def prepare(output: Path, resume: bool) -> dict[int, Path]:
    audit_path = output / "audit" / "capacity_pairing_audit.json"
    bundles: dict[int, Path] = {}
    if resume:
        if not audit_path.is_file():
            raise FileNotFoundError(f"resume requires {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise RuntimeError("resume pairing audit is not PASS")
        for seed in exact.SEEDS:
            bundle = output / "config" / "controlled_bundles" / f"seed_{seed}"
            if not bundle.is_dir():
                raise FileNotFoundError(bundle)
            bundles[seed] = bundle
        return bundles

    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    output.mkdir(parents=True, exist_ok=False)
    reference_completion = {}
    for capacity, reference in REFERENCE_OUTPUTS.items():
        completion = json.loads((reference / "completion_status.json").read_text(encoding="utf-8"))
        reference_completion[str(capacity)] = completion
        if (
            float(completion.get("capacity", -1)) != capacity
            or int(completion.get("run_count", 0)) != 25
            or not bool(completion.get("public_accounting_passed"))
            or not bool(completion.get("hard_safety_passed"))
            or tuple(completion.get("methods", ())) != tuple(exact.METHODS)
            or tuple(completion.get("seeds", ())) != tuple(exact.SEEDS)
        ):
            raise RuntimeError({"REFERENCE_IDENTITY_MISMATCH": str(reference)})

    checks: dict[str, Any] = {}
    for seed in exact.SEEDS:
        source = BUNDLE_REFERENCE / "config" / "controlled_bundles" / f"seed_{seed}"
        destination = output / "config" / "controlled_bundles" / f"seed_{seed}"
        shutil.copytree(source, destination)
        bundles[seed] = destination
        task_hashes = {
            str(capacity): sha256(reference / "config" / "controlled_bundles" / f"seed_{seed}" / "task_manifest.csv")
            for capacity, reference in REFERENCE_OUTPUTS.items()
        }
        slot_hashes = {
            str(capacity): sha256(reference / "config" / "controlled_bundles" / f"seed_{seed}" / "slot_trace.csv")
            for capacity, reference in REFERENCE_OUTPUTS.items()
        }
        checks[str(seed)] = {
            "task_manifest_hashes": task_hashes,
            "slot_trace_hashes": slot_hashes,
            "task_identity_match": len(set(task_hashes.values())) == 1,
            "slot_trace_match": len(set(slot_hashes.values())) == 1,
            "copied_task_hash_matches": sha256(destination / "task_manifest.csv") == next(iter(task_hashes.values())),
            "copied_slot_hash_matches": sha256(destination / "slot_trace.csv") == next(iter(slot_hashes.values())),
        }
    passed = all(
        item["task_identity_match"]
        and item["slot_trace_match"]
        and item["copied_task_hash_matches"]
        and item["copied_slot_hash_matches"]
        for item in checks.values()
    )
    audit = {
        "status": "PASS" if passed else "FAIL",
        "new_capacities": list(NEW_CAPACITIES),
        "reference_capacities": list(REFERENCE_OUTPUTS),
        "methods": list(exact.METHODS),
        "seeds": list(exact.SEEDS),
        "reference_completion": reference_completion,
        "seed_checks": checks,
    }
    exact.write_json(audit_path, audit)
    if not passed:
        raise RuntimeError({"CAPACITY_PAIRING_MISMATCH": audit})
    return bundles


def run_one(output: Path, bundles: dict[int, Path], capacity: float, seed: int, method: str) -> None:
    status = exact.launch(
        method,
        seed,
        bundles[seed],
        run_path(output, capacity, seed, method),
        output / "logs" / capacity_tag(capacity),
        capacity,
    )
    status["capacity"] = capacity
    exact.write_json(status_path(output, capacity, seed, method), status)
    print(
        json.dumps({"status": "RUN_COMPLETE", "capacity": capacity, "seed": seed, "method": method}, ensure_ascii=False),
        flush=True,
    )


def run_grid(output: Path, bundles: dict[int, Path], workers: int) -> None:
    jobs = [
        (capacity, seed, method)
        for seed in exact.SEEDS
        for capacity in NEW_CAPACITIES
        for method in exact.METHODS
    ]
    pending = [job for job in jobs if not is_complete(output, *job)]
    print(
        json.dumps(
            {
                "status": "PARALLEL_GRID",
                "total": len(jobs),
                "completed_before_start": len(jobs) - len(pending),
                "remaining": len(pending),
                "workers": workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(run_one, output, bundles, capacity, seed, method): (capacity, seed, method)
            for capacity, seed, method in pending
        }
        for future in as_completed(futures):
            capacity, seed, method = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(
                    {"CAPACITY_SENSITIVITY_RUNTIME_FAILED": {"capacity": capacity, "seed": seed, "method": method, "error": str(error)}}
                ) from error


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def collect_rows(output: Path, bundles: dict[int, Path]) -> list[dict[str, Any]]:
    eval_mod = exact.v25.load_eval_module(output)
    rows: list[dict[str, Any]] = []
    for capacity in NEW_CAPACITIES:
        for seed in exact.SEEDS:
            for method in exact.METHODS:
                if not is_complete(output, capacity, seed, method):
                    raise RuntimeError({"INCOMPLETE_NEW_CAPACITY_GRID": {"capacity": capacity, "seed": seed, "method": method}})
                metrics = exact.evaluate_method(
                    method,
                    run_path(output, capacity, seed, method),
                    bundles[seed],
                    eval_mod,
                    capacity,
                )
                row = exact.selected_metrics(seed, metrics)
                row = {"capacity": capacity, **row}
                if not bool(row["public_accounting_passed"]):
                    raise RuntimeError({"PUBLIC_ACCOUNTING_MISMATCH": {"capacity": capacity, "seed": seed, "method": method}})
                if int(row["SOC_violations"]) or int(row["thermal_violations"]):
                    raise RuntimeError({"HARD_SAFETY_VIOLATION": {"capacity": capacity, "seed": seed, "method": method}})
                rows.append(row)

    for capacity, reference in REFERENCE_OUTPUTS.items():
        for raw in read_csv(reference / "tables" / "all_metrics_per_seed.csv"):
            if not bool_value(raw["public_accounting_passed"]):
                raise RuntimeError({"REFERENCE_PUBLIC_ACCOUNTING_MISMATCH": {"capacity": capacity, "row": raw}})
            if int(float(raw["SOC_violations"])) or int(float(raw["thermal_violations"])):
                raise RuntimeError({"REFERENCE_HARD_SAFETY_VIOLATION": {"capacity": capacity, "row": raw}})
            rows.append({"capacity": capacity, **raw})
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = [
        "J_all_net", "J_HV_net", "N_HV_closure", "N_high_closure", "N_urgent_closure",
        "all_task_closure_count", "all_task_closure_rate", "N_HV_metadata", "N_HV_ROI", "N_HV_evidence",
        "successful_closure_mean_latency_s", "successful_closure_p95_latency_s",
        "HV_successful_closure_mean_latency_s", "HV_successful_closure_p95_latency_s",
        "urgent_successful_closure_mean_latency_s", "urgent_successful_closure_p95_latency_s",
        "deadline_censored_mean_latency_s", "controllable_energy_Wh", "transmitted_MB", "transmitted_bytes",
        "peak_queue_MB", "mean_queue_MB", "minimum_soc", "maximum_thermal", "SOC_violations", "thermal_violations",
    ]
    summary: list[dict[str, Any]] = []
    for capacity in ALL_CAPACITIES:
        for method in exact.METHODS:
            group = [row for row in rows if abs(float(row["capacity"]) - capacity) < 1e-12 and row["method"] == method]
            if len(group) != 5:
                raise RuntimeError({"INCOMPLETE_COMBINED_CAPACITY_GRID": {"capacity": capacity, "method": method, "count": len(group)}})
            item: dict[str, Any] = {"capacity": capacity, "method": method, "seed_count": len(group)}
            for metric in numeric:
                values = [float(row[metric]) for row in group]
                item[f"{metric}_mean"] = statistics.fmean(values)
                item[f"{metric}_std"] = statistics.stdev(values)
            summary.append(item)
    return summary


def emit(output: Path, rows: list[dict[str, Any]], workers: int) -> None:
    method_order = {method: index for index, method in enumerate(exact.METHODS)}
    rows.sort(key=lambda row: (float(row["capacity"]), int(row["seed"]), method_order[row["method"]]))
    summary = summarize(rows)
    primary_keys = [
        "capacity", "method", "seed_count", "J_all_net_mean", "J_all_net_std", "J_HV_net_mean", "J_HV_net_std",
        "N_HV_closure_mean", "N_HV_closure_std", "N_high_closure_mean", "N_high_closure_std",
        "N_urgent_closure_mean", "N_urgent_closure_std",
    ]
    auxiliary_keys = [
        "capacity", "method", "seed_count", "all_task_closure_count_mean", "all_task_closure_count_std",
        "all_task_closure_rate_mean", "all_task_closure_rate_std", "N_HV_metadata_mean", "N_HV_metadata_std",
        "N_HV_ROI_mean", "N_HV_ROI_std", "N_HV_evidence_mean", "N_HV_evidence_std",
        "successful_closure_mean_latency_s_mean", "successful_closure_mean_latency_s_std",
        "HV_successful_closure_mean_latency_s_mean", "HV_successful_closure_mean_latency_s_std",
        "urgent_successful_closure_mean_latency_s_mean", "urgent_successful_closure_mean_latency_s_std",
        "controllable_energy_Wh_mean", "controllable_energy_Wh_std", "transmitted_MB_mean", "transmitted_MB_std",
        "peak_queue_MB_mean", "peak_queue_MB_std", "mean_queue_MB_mean", "mean_queue_MB_std",
    ]
    safety_keys = [
        "capacity", "method", "seed_count", "minimum_soc_mean", "minimum_soc_std", "maximum_thermal_mean",
        "maximum_thermal_std", "SOC_violations_mean", "SOC_violations_std", "thermal_violations_mean", "thermal_violations_std",
    ]
    primary = [{key: row[key] for key in primary_keys} for row in summary]
    auxiliary = [{key: row[key] for key in auxiliary_keys} for row in summary]
    safety = [{key: row[key] for key in safety_keys} for row in summary]
    prop = [row for row in primary if row["method"] == exact.METHOD_CORRECTED]
    deltas: list[dict[str, Any]] = []
    for capacity in ALL_CAPACITIES:
        reference = next(row for row in summary if float(row["capacity"]) == capacity and row["method"] == exact.METHOD_CORRECTED)
        for competitor in exact.METHODS:
            if competitor == exact.METHOD_CORRECTED:
                continue
            other = next(row for row in summary if float(row["capacity"]) == capacity and row["method"] == competitor)
            deltas.append(
                {
                    "capacity": capacity,
                    "comparison_method": competitor,
                    "delta_J_all_net": reference["J_all_net_mean"] - other["J_all_net_mean"],
                    "delta_J_HV_net": reference["J_HV_net_mean"] - other["J_HV_net_mean"],
                    "delta_N_HV_closure": reference["N_HV_closure_mean"] - other["N_HV_closure_mean"],
                    "delta_N_high_closure": reference["N_high_closure_mean"] - other["N_high_closure_mean"],
                    "delta_N_urgent_closure": reference["N_urgent_closure_mean"] - other["N_urgent_closure_mean"],
                }
            )
    tables = output / "tables"
    for name, table in (
        ("all_metrics_per_seed", rows), ("all_metrics_mean_std", summary), ("primary_metrics_mean_std", primary),
        ("auxiliary_metrics_mean_std", auxiliary), ("safety_metrics_mean_std", safety),
        ("prop_capacity_sensitivity", prop), ("prop_vs_baselines_deltas", deltas),
    ):
        write_csv(tables / f"{name}.csv", table)
        write_md(tables / f"{name}.md", table)
    exact.write_json(
        output / "completion_status.json",
        {
            "status": "V25_CAPACITY_025_035_050_065_FIVE_METHOD_FIVESEED_SENSITIVITY_COMPLETE",
            "new_capacities": list(NEW_CAPACITIES),
            "reference_capacities": list(REFERENCE_OUTPUTS),
            "all_capacities": list(ALL_CAPACITIES),
            "new_run_count": 50,
            "combined_row_count": len(rows),
            "methods": list(exact.METHODS),
            "seeds": list(exact.SEEDS),
            "workers": workers,
            "public_accounting_passed": True,
            "hard_safety_passed": True,
            "output": str(output),
        },
    )


def parent_main(output: Path, workers: int, resume: bool, preflight_only: bool) -> int:
    bundles = prepare(output, resume)
    if preflight_only:
        print(json.dumps({"status": "PREFLIGHT_COMPLETE", "output": str(output)}, ensure_ascii=False))
        return 0
    run_grid(output, bundles, workers)
    rows = collect_rows(output, bundles)
    if len(rows) != 100:
        raise RuntimeError({"INCOMPLETE_COMBINED_CAPACITY_GRID": len(rows)})
    emit(output, rows, workers)
    print((output / "completion_status.json").read_text(encoding="utf-8"), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    return parent_main(args.output_dir, max(1, int(args.workers)), args.resume, args.preflight_only)


if __name__ == "__main__":
    raise SystemExit(main())
