"""Complete a paired five-seed comparison of current PROP and four baselines.

The trusted, already completed seed-2026 five-method run is reused only after
identity/hash checks.  Seeds 2027--2030 are executed fresh in independent
processes on the ordinary value-blind V25 bundles and the shared calibrated
two-node thermal model.  Existing source, configuration and result assets are
read-only.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_current_prop_four_independent_baselines_single_seed_v1 as single  # noqa: E402
import scripts.run_current_prop_five_ablation_multiseed_dual_thermal_v1 as scenarios  # noqa: E402
import scripts.run_dual_node_thermal_calibration_and_four_method_rerun_v1 as dual  # noqa: E402
import scripts.run_normal_scenario_tut_proxy_ground_recovery_v1 as normal  # noqa: E402
import scripts.run_pure_myopic_safe_greedy_offload_thermal_comparison_v1 as adapted  # noqa: E402
import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "CURRENT-PROP-FOUR-BASELINES-FIVESEED-DUAL-THERMAL-v1"
COMPLETION = "CURRENT_PROP_FOUR_BASELINES_FIVESEED_DUAL_THERMAL_COMPLETE"
SEEDS = (2026, 2027, 2028, 2029, 2030)
NEW_SEEDS = SEEDS[1:]
METHODS = single.METHODS
CURRENT = single.CURRENT
CAPACITY, SLOTS, MAX_PARALLEL = 0.5, 4320, 5
SLUG = single.SLUG
OUTPUT_BASE = ROOT / "outputs" / "_current_prop_four_baselines_fiveseed_dual_thermal_v1"
REFERENCE_2026 = ROOT / "outputs" / "_current_prop_four_independent_baselines_single_seed_v1" / "attempt_001"
SCENARIO_REFERENCE = ROOT / "outputs" / "_current_prop_five_ablation_multiseed_dual_thermal_v1" / "attempt_001" / "scenario"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in data for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(data or [{"empty": ""}])


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    return int(f(value))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    number = 1
    while (OUTPUT_BASE / f"attempt_{number:03d}").exists():
        number += 1
    attempt = OUTPUT_BASE / f"attempt_{number:03d}"
    for name in ("audit", "logs", "runs", "run_specs", "tables"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    return attempt


def bundle_for_seed(seed: int) -> Path:
    if seed == 2026:
        return dual.BUNDLE
    return SCENARIO_REFERENCE / f"ordinary_seed_{seed}"


def set_seed_context(seed: int, bundle: Path, task_hash: str) -> None:
    """Update process-local imported runner constants for a seed-specific child."""

    engine.SEED = seed
    engine.EXPECTED_TASK_HASH = task_hash
    single.SEED, single.BUNDLE, single.TASK_HASH = seed, bundle, task_hash
    adapted.SEED, adapted.BUNDLE, adapted.TASK_HASH = seed, bundle, task_hash
    dual.SEED, dual.BUNDLE, dual.TASK_HASH = seed, bundle, task_hash
    normal.SEED = seed


def child_run(spec: Mapping[str, Any]) -> int:
    seed = i(spec["seed"])
    bundle = Path(spec["bundle_root"])
    contract = read_json(bundle / "bundle_contract.json")
    if contract.get("task_hash") != spec.get("task_hash"):
        raise RuntimeError("TASK_MANIFEST_MISMATCH")
    set_seed_context(seed, bundle, str(contract["task_hash"]))
    return single.child_run(spec)


def make_specs(attempt: Path, thermal_model: Path) -> list[dict[str, Any]]:
    result = []
    for seed in NEW_SEEDS:
        bundle = bundle_for_seed(seed)
        contract = read_json(bundle / "bundle_contract.json")
        for method in METHODS:
            result.append({
                "version": VERSION,
                "seed": seed,
                "method": method,
                "capacity": CAPACITY,
                "slots": SLOTS,
                "bundle_root": str(bundle),
                "task_hash": contract["task_hash"],
                "thermal_model": str(thermal_model),
                "output": str(attempt / "runs" / f"seed_{seed}" / SLUG[method]),
            })
    return result


def launch_one(spec: Mapping[str, Any], attempt: Path) -> dict[str, Any]:
    seed, method = i(spec["seed"]), str(spec["method"])
    label = f"seed_{seed}_{SLUG[method]}"
    spec_path = attempt / "run_specs" / f"{label}.json"
    write_json(spec_path, spec)
    stdout_path = attempt / "logs" / f"{label}.stdout.log"
    stderr_path = attempt / "logs" / f"{label}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)],
            cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False,
        )
    return {
        "seed": seed,
        "method": method,
        "pid": None,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "output": spec["output"],
        "stderr": str(stderr_path),
    }


def reference_run(method: str) -> Path:
    candidates = []
    if method == single.GREEDY:
        candidates.append(REFERENCE_2026 / "runs" / single.GREEDY_CORRECTED_SUBDIR)
    if method == single.OFFLOAD:
        candidates.append(single.COMPATIBLE_OFFLOAD_RUN)
    candidates.append(REFERENCE_2026 / "runs" / SLUG[method])
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "method_completed.json").is_file():
            return candidate
    raise FileNotFoundError({"seed_2026_reference_missing": method, "candidates": [str(path) for path in candidates]})


def evaluate_one(seed: int, method: str, run: Path, bundle: Path) -> dict[str, Any]:
    contract = read_json(bundle / "bundle_contract.json")
    set_seed_context(seed, bundle, str(contract["task_hash"]))
    # The trusted seed-2026 bundle is strictly read-only.  Its public metrics
    # were already completed and audited by the single-seed runner.
    metric = read_json(run / "frozen_public_metrics.json") if seed == 2026 else single.evaluate(method, run)
    row = single.summary_row(method, run, metric)
    row = {"seed": seed, **row}
    row.update({
        "HV_gross_utility": metric.get("HV_gross_utility"),
        "total_cost": metric.get("total_cost", f(metric.get("gross_utility")) - f(metric.get("J_all_net"))),
        "model_lightweight_count": metric.get("model_lightweight_count", 0),
        "model_balanced_count": metric.get("model_balanced_count", 0),
        "model_high_accuracy_count": metric.get("model_high_accuracy_count", 0),
        "task_hash": contract["task_hash"],
        "scenario_environment_hash": contract.get("environment_hash"),
    })
    runtime = read_json(run / "dual_node_runtime_audit.json")
    row["pid"] = runtime.get("pid")
    row["thermal_parameter_hash"] = runtime.get("thermal_parameter_hash")
    identity_path = run / "runtime_identity.json"
    identity = read_json(identity_path) if identity_path.is_file() else read_json(run / "reconstructed_scenario_identity.json")
    scenario_identity = identity.get("scenario_identity", identity)
    capacity_identity = identity.get("capacity_identity", scenario_identity.get("capacity_identity", {}))
    power_identity = identity.get("power_identity", scenario_identity.get("power_identity", {}))
    row["runtime_task_hash"] = scenario_identity.get("task_hash", identity.get("task_hash"))
    row["link_trace_hash"] = capacity_identity.get("runtime_link_trace_hash")
    row["power_trace_hash"] = power_identity.get("power_trace_hash")
    return row


SUMMARY_KEYS = (
    "J_all_net", "J_HV_net", "gross_utility", "HV_gross_utility", "total_cost",
    "energy_cost", "downlink_cost", "terminal_backlog_cost",
    "all_closure_count", "closure_rate", "ordinary_closure_count", "HV_closure_count",
    "high_closure_count", "urgent_closure_count", "metadata_delivery_count", "ROI_delivery_count",
    "Evidence_delivery_count", "HV_metadata_delivered_count", "HV_ROI_delivered_count",
    "HV_evidence_delivered_count", "mean_closure_latency_s", "p95_closure_latency_s",
    "controllable_energy_Wh", "transmitted_MB", "peak_queue_MB", "mean_queue_MB",
    "terminal_queue_MB", "minimum_soc", "maximum_chip_c", "maximum_surface_c",
    "soc_violation_count", "thermal_violation_count", "model_lightweight_count",
    "model_balanced_count", "model_high_accuracy_count",
)


def aggregate(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        group = [row for row in per_run if row["method"] == method]
        result: dict[str, Any] = {"method": method, "seed_count": len(group)}
        for key in SUMMARY_KEYS:
            values = [f(row.get(key)) for row in group]
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(result)
    return rows


def paired_deltas(per_run: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "controllable_energy_Wh", "transmitted_MB", "mean_closure_latency_s")
    by = {(i(row["seed"]), row["method"]): row for row in per_run}
    detailed = []
    for seed in SEEDS:
        prop = by[(seed, CURRENT)]
        for method in METHODS[1:]:
            baseline = by[(seed, method)]
            item: dict[str, Any] = {"seed": seed, "baseline": method, "reference": CURRENT}
            for key in keys:
                item[f"delta_{key}"] = f(baseline.get(key)) - f(prop.get(key))
            detailed.append(item)
    summary = []
    for method in METHODS[1:]:
        group = [row for row in detailed if row["baseline"] == method]
        item: dict[str, Any] = {"baseline": method, "reference": CURRENT, "seed_count": len(group)}
        for key in keys:
            values = [f(row[f"delta_{key}"]) for row in group]
            item[f"delta_{key}_mean"] = statistics.fmean(values)
            item[f"delta_{key}_std"] = statistics.stdev(values)
        summary.append(item)
    return detailed, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_spec:
        return child_run(read_json(args.child_spec))

    attempt = next_attempt()
    thermal_model = adapted.reference_thermal_model()
    reference_completion = read_json(REFERENCE_2026 / "completion.json")
    if reference_completion.get("status") != "PASS":
        raise RuntimeError("SEED_2026_REFERENCE_INCOMPLETE")

    scenario_audit = []
    for seed in SEEDS:
        bundle = bundle_for_seed(seed)
        contract = read_json(bundle / "bundle_contract.json")
        tasks = read_csv(bundle / "task_manifest.csv")
        counts = {level: sum(row.get("value_level") == level for row in tasks) for level in ("low", "medium", "high", "urgent")}
        scenario_audit.append({
            "seed": seed, "bundle": str(bundle.resolve()), "task_hash": contract.get("task_hash"),
            "environment_hash": contract.get("environment_hash"), "arrival_profile": contract.get("arrival_profile"),
            "active_arrival_swap_count": contract.get("active_arrival_swap_count", 0), "value_counts": counts,
            "task_count": len(tasks), "ordinary_random": contract.get("active_arrival_swap_count", 0) == 0,
        })
    if not all(row["ordinary_random"] and row["task_count"] == 268 and row["value_counts"] == {"low": 54, "medium": 139, "high": 54, "urgent": 21} for row in scenario_audit):
        raise RuntimeError("ORDINARY_SCENARIO_IDENTITY_FAILED")
    write_json(attempt / "audit" / "scenario_identity.json", {"status": "PASS", "seeds": scenario_audit})

    specs = make_specs(attempt, thermal_model)
    statuses = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(launch_one, spec, attempt): spec for spec in specs}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(f"[{len(statuses):02d}/20] {status['status']} seed={status['seed']} {status['method']}", flush=True)
    write_json(attempt / "audit" / "process_status.json", statuses)
    failures = [row for row in statuses if row["status"] != "PASS"]
    if failures:
        raise RuntimeError({"FRESH_FOUR_SEED_RUNTIME_FAILED": failures})

    spec_by = {(i(spec["seed"]), str(spec["method"])): spec for spec in specs}
    per_run = []
    reuse_rows = []
    for seed in SEEDS:
        bundle = bundle_for_seed(seed)
        for method in METHODS:
            if seed == 2026:
                run = reference_run(method)
                reuse_rows.append({"seed": seed, "method": method, "run": str(run.resolve()), "reuse": "HASH_GATED_TRUSTED_COMPLETE"})
            else:
                run = Path(spec_by[(seed, method)]["output"])
            per_run.append(evaluate_one(seed, method, run, bundle))
    write_json(attempt / "audit" / "seed_2026_reuse.json", {"status": "PASS", "rows": reuse_rows})

    summary = aggregate(per_run)
    paired, paired_summary = paired_deltas(per_run)
    write_csv(attempt / "tables" / "per_run_metrics.csv", per_run)
    write_csv(attempt / "tables" / "mean_std_metrics.csv", summary)
    write_md(attempt / "tables" / "mean_std_metrics.md", summary)
    write_csv(attempt / "tables" / "paired_deltas_per_seed.csv", paired)
    write_csv(attempt / "tables" / "paired_deltas_mean_std.csv", paired_summary)
    hv_per_seed = [{
        "seed": row["seed"], "method": row["method"],
        "HV_metadata": row["HV_metadata_delivered_count"],
        "HV_ROI": row["HV_ROI_delivered_count"],
        "HV_ROI_context": row["HV_evidence_delivered_count"],
        "HV_closure_count": row["HV_closure_count"],
    } for row in per_run]
    hv_summary = [{
        "method": row["method"], "seed_count": row["seed_count"],
        "HV_metadata_mean": row["HV_metadata_delivered_count_mean"],
        "HV_metadata_std": row["HV_metadata_delivered_count_std"],
        "HV_ROI_mean": row["HV_ROI_delivered_count_mean"],
        "HV_ROI_std": row["HV_ROI_delivered_count_std"],
        "HV_ROI_context_mean": row["HV_evidence_delivered_count_mean"],
        "HV_ROI_context_std": row["HV_evidence_delivered_count_std"],
        "HV_closure_mean": row["HV_closure_count_mean"],
        "HV_closure_std": row["HV_closure_count_std"],
    } for row in summary]
    write_csv(attempt / "tables" / "hv_layered_delivery_per_seed.csv", hv_per_seed)
    write_csv(attempt / "tables" / "hv_layered_delivery_mean_std.csv", hv_summary)
    write_md(attempt / "tables" / "hv_layered_delivery_mean_std.md", hv_summary)

    fresh = [row for row in per_run if i(row["seed"]) in NEW_SEEDS]
    checks = {
        "five_methods_x_five_seeds": len(per_run) == 25 and all(sum(row["method"] == method for row in per_run) == 5 for method in METHODS),
        "seed_2026_trusted_complete_reused": len(reuse_rows) == 5,
        "twenty_new_runs_completed": len(statuses) == 20 and not failures,
        "twenty_new_unique_processes": len({i(row["pid"]) for row in fresh}) == 20,
        "same_task_hash_within_each_seed": all(len({row["runtime_task_hash"] for row in per_run if i(row["seed"]) == seed}) == 1 for seed in SEEDS),
        "runtime_task_hash_matches_bundle": all(row["runtime_task_hash"] == row["task_hash"] for row in per_run),
        "same_link_trace_all_runs": len({row["link_trace_hash"] for row in per_run}) == 1,
        "same_power_trace_all_runs": len({row["power_trace_hash"] for row in per_run}) == 1,
        "same_two_node_thermal_all_runs": len({row["thermal_parameter_hash"] for row in per_run}) == 1,
        "zero_soc_and_thermal_violations": all(i(row["soc_violation_count"]) == 0 and i(row["thermal_violation_count"]) == 0 for row in per_run),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    audit = {
        "status": status, "version": VERSION, "checks": checks, "methods": METHODS,
        "seeds": SEEDS, "fresh_seeds": NEW_SEEDS, "capacity": CAPACITY, "duration_s": 86400,
        "power_environment": "corrected 40 W", "max_parallel_processes": MAX_PARALLEL,
        "thermal_model": str(thermal_model.resolve()), "thermal_model_sha256": sha256_file(thermal_model),
        "thermal_parameter_hash": per_run[0]["thermal_parameter_hash"],
        "execution": "seed 2026 reused after identity gate; seeds 2027-2030 freshly executed in independent processes",
    }
    write_json(attempt / "audit" / "method_environment_thermal_audit.json", audit)
    (attempt / "test_report.txt").write_text(
        "\n".join(("PASS" if passed else "FAIL") + "\t" + name for name, passed in checks.items())
        + "\n" + ("PASS\t" if status == "PASS" else "FAIL\t") + COMPLETION + "\n", encoding="utf-8",
    )

    lines = [
        "# Current PROP vs four baselines: five-seed dual-node-thermal comparison", "",
        "V25 ordinary value-blind arrival order, capacity=0.5, corrected 40 W power environment, shared calibrated two-node thermal model.", "",
        "| method | J_all mean+-std | J_HV mean+-std | all/HV/high/urgent closure mean | energy Wh | tx MB | mean/p95 latency s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {f(row['J_all_net_mean']):.6f}+-{f(row['J_all_net_std']):.6f} | "
            f"{f(row['J_HV_net_mean']):.6f}+-{f(row['J_HV_net_std']):.6f} | "
            f"{f(row['all_closure_count_mean']):.2f}/{f(row['HV_closure_count_mean']):.2f}/"
            f"{f(row['high_closure_count_mean']):.2f}/{f(row['urgent_closure_count_mean']):.2f} | "
            f"{f(row['controllable_energy_Wh_mean']):.6f} | {f(row['transmitted_MB_mean']):.6f} | "
            f"{f(row['mean_closure_latency_s_mean']):.2f}/{f(row['p95_closure_latency_s_mean']):.2f} |"
        )
    lines.extend(["", f"Audit status: **{status}**.", "", COMPLETION if status == "PASS" else "CURRENT_PROP_FOUR_BASELINES_FIVESEED_DUAL_THERMAL_FAILED"])
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": status, "marker": COMPLETION, "completed_utc": datetime.now(timezone.utc).isoformat(), "output": str(attempt)})
    print("OUTPUT:", attempt)
    print("NEW RUNS: 20; MAX PARALLEL:", MAX_PARALLEL)
    print(COMPLETION if status == "PASS" else "CURRENT_PROP_FOUR_BASELINES_FIVESEED_DUAL_THERMAL_FAILED")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
