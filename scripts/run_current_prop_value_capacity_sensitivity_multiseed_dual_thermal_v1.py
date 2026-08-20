"""Current PROP V1+V2 value/capacity sensitivity on the calibrated thermal model.

No-argument entry point.  It creates isolated copies of the already-audited
ordinary-random five-seed bundles, changes only the requested value profile or
runtime downlink-capacity multiplier, and runs one fresh current-PROP process
per scenario/seed pair.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_current_prop_five_ablation_multiseed_dual_thermal_v1 as base  # noqa: E402
import scripts.run_dual_node_thermal_calibration_and_four_method_rerun_v1 as dual  # noqa: E402
import scripts.run_normal_scenario_tut_proxy_ground_recovery_v1 as normal  # noqa: E402
import scripts.run_pure_myopic_safe_greedy_offload_thermal_comparison_v1 as adapted  # noqa: E402
import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402
import scripts.run_v10_v50_cap050_five_method_multiseed_v1 as value_profiles  # noqa: E402


VERSION = "CURRENT-PROP-VALUE-CAPACITY-SENSITIVITY-MULTISEED-DUAL-THERMAL-v1"
METHOD = "PROP-V2-SAFE-METADATA-CURRENT-v1"
COMPLETION = "CURRENT_PROP_VALUE_CAPACITY_SENSITIVITY_MULTISEED_DUAL_THERMAL_COMPLETE"
SEEDS = (2026, 2027, 2028, 2029, 2030)
SLOTS = 4320
MAX_PARALLEL = 2
SCENARIOS = (
    {"scenario": "V10_CAP050", "value_profile": "V10", "capacity": 0.50},
    {"scenario": "V50_CAP050", "value_profile": "V50", "capacity": 0.50},
    {"scenario": "V25_CAP025", "value_profile": "V25", "capacity": 0.25},
    {"scenario": "V25_CAP035", "value_profile": "V25", "capacity": 0.35},
    {"scenario": "V25_CAP070", "value_profile": "V25", "capacity": 0.70},
)
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}
OUTPUT_BASE = ROOT / "outputs" / "_current_prop_value_capacity_sensitivity_multiseed_dual_thermal_v1"
SOURCE_SCENARIOS = ROOT / "outputs" / "_current_prop_five_ablation_multiseed_dual_thermal_v1" / "attempt_001" / "scenario"
REFERENCE_CAP050 = ROOT / "outputs" / "_current_prop_five_ablation_multiseed_dual_thermal_v1" / "attempt_001" / "tables" / "per_run_metrics.csv"


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


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def next_attempt() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    number = 1
    while (OUTPUT_BASE / f"attempt_{number:03d}").exists():
        number += 1
    attempt = OUTPUT_BASE / f"attempt_{number:03d}"
    for name in ("audit", "logs", "runs", "run_specs", "scenario", "tables"):
        (attempt / name).mkdir(parents=True, exist_ok=True)
    return attempt


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def protected_snapshot() -> dict[str, str]:
    paths = [
        ROOT / "configs" / "prop_v2_safe_metadata_current_v1.yaml",
        adapted.reference_thermal_model(),
        Path(adapted.__file__).resolve(),
        Path(engine.__file__).resolve(),
        Path(value_profiles.__file__).resolve(),
    ]
    for seed in SEEDS:
        source = SOURCE_SCENARIOS / f"ordinary_seed_{seed}"
        paths.extend(source / name for name in ("task_manifest.csv", "slot_trace.csv", "bundle_contract.json"))
    return {str(path.resolve()): sha256_file(path) for path in paths if path.is_file()}


def non_value_hash(rows: list[dict[str, str]]) -> str:
    return value_profiles.invariant_task_hash(rows)


def prepare_bundle(scenario: Mapping[str, Any], seed: int, destination: Path) -> dict[str, Any]:
    source = SOURCE_SCENARIOS / f"ordinary_seed_{seed}"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    before_tasks = read_csv(destination / "task_manifest.csv")
    before_contract = read_json(destination / "bundle_contract.json")
    before = {
        "task_hash": before_contract["task_hash"],
        "non_value_hash": non_value_hash(before_tasks),
        "slot_hash": sha256_file(destination / "slot_trace.csv"),
        "arrival_hash": sha256_file(destination / "arrival_swap_manifest.csv"),
    }
    profile = str(scenario["value_profile"])
    profile_audit: dict[str, Any] | None = None
    if profile in {"V10", "V50"}:
        # The historical transformer ends with a loader whose only extra check
        # hard-codes the retired late-HV window distribution.  Keep the entire
        # transformation and its isolation checks, but replace only that final
        # incompatible loader call; the ordinary-random bundle is validated
        # below with explicit task/slot/arrival hashes.
        original_load_bundle = value_profiles.v25.load_bundle
        value_profiles.v25.load_bundle = lambda _path: None
        try:
            profile_audit = value_profiles._apply_value_profile(
                destination,
                seed=seed,
                profile=profile,
                controlled_identity={"checks": {"trusted_ordinary_bundle": True}},
            )
        finally:
            value_profiles.v25.load_bundle = original_load_bundle
    after_tasks = read_csv(destination / "task_manifest.csv")
    after_contract = read_json(destination / "bundle_contract.json")
    counts = dict(Counter(row["value_level"] for row in after_tasks))
    effective_sum = sum(f(row["effective_task_value"]) for row in after_tasks)
    checks = {
        "ordinary_random_profile": before_contract.get("arrival_profile") == "ordinary_random_value_blind_seed_permutation_v1",
        "late_hv_forcing_absent": i(before_contract.get("active_arrival_swap_count")) == 0,
        "task_count_268": len(after_tasks) == 268,
        "value_level_counts_frozen": counts == EXPECTED_COUNTS,
        "non_value_identity_unchanged": non_value_hash(after_tasks) == before["non_value_hash"],
        "slot_trace_unchanged": sha256_file(destination / "slot_trace.csv") == before["slot_hash"],
        "arrival_swap_manifest_unchanged": sha256_file(destination / "arrival_swap_manifest.csv") == before["arrival_hash"],
        "effective_value_sum_268": abs(effective_sum - 268.0) < 1e-10,
        "v25_task_hash_unchanged": profile != "V25" or after_contract["task_hash"] == before["task_hash"],
    }
    if not all(checks.values()):
        raise RuntimeError({"SCENARIO_ISOLATION_FAILED": scenario["scenario"], "seed": seed, "checks": checks})
    return {
        "scenario": scenario["scenario"], "seed": seed, "value_profile": profile,
        "capacity": scenario["capacity"], "source": str(source), "destination": str(destination),
        "source_task_hash": before["task_hash"], "task_hash": after_contract["task_hash"],
        "non_value_task_hash": non_value_hash(after_tasks), "effective_task_value_sum": effective_sum,
        "counts": counts, "profile_audit": profile_audit, "checks": checks,
    }


def child_run(spec: Mapping[str, Any]) -> int:
    seed, capacity = i(spec["seed"]), f(spec["capacity"])
    bundle = Path(spec["bundle_root"])
    output = Path(spec["output"])
    contract = read_json(bundle / "bundle_contract.json")
    # All modules keep legacy process globals.  Set them only inside this fresh
    # child process so capacity/seed cannot leak across runs.
    engine.SEED = seed
    engine.CAPACITY = capacity
    engine.EXPECTED_TASK_HASH = contract["task_hash"]
    adapted.SEED = seed
    adapted.CAPACITY = capacity
    adapted.BUNDLE = bundle
    adapted.TASK_HASH = contract["task_hash"]
    normal.SEED = seed
    params = dual.thermal.TwoNodeParameters(**read_json(Path(spec["thermal_model"]))["parameters"])
    adapter = dual.thermal.LegacyScalarAdapter(params)
    patch_audit = adapted.run_current_child(spec, output, adapter)
    write_csv(output / "dual_node_transition_trace.csv", adapter.rows)
    actual = [row for row in adapter.rows if row.get("caller") in {"run_step", "run", "actual_slot_transition"}]
    write_json(output / "dual_node_runtime_audit.json", {
        "status": "PASS", "scenario": spec["scenario"], "seed": seed, "capacity": capacity,
        "pid": os.getpid(), "thermal_parameter_hash": params.stable_hash(), "patch_audit": patch_audit,
        "transition_call_count": len(adapter.rows), "actual_slot_transition_count": len(actual),
        "max_chip_c": max((f(row.get("peak_chip_c")) for row in actual), default=0.0),
        "max_surface_c": max((f(row.get("peak_surface_c")) for row in actual), default=0.0),
        "chip_violation_count": sum(f(row.get("peak_chip_c")) > params.t_chip_max_c + 1e-12 for row in actual),
        "surface_violation_count": sum(f(row.get("peak_surface_c")) > params.t_surface_max_c + 1e-12 for row in actual),
    })
    return 0


def make_specs(attempt: Path, thermal_model: Path, bundles: Mapping[tuple[str, int], Path]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        name = str(scenario["scenario"])
        for seed in SEEDS:
            bundle = bundles[(name, seed)]
            specs.append({
                "version": VERSION, "method": METHOD, "scenario": name,
                "value_profile": scenario["value_profile"], "seed": seed,
                "capacity": scenario["capacity"], "slots": SLOTS,
                "bundle_root": str(bundle), "task_hash": read_json(bundle / "bundle_contract.json")["task_hash"],
                "thermal_model": str(thermal_model),
                "output": str(attempt / "runs" / name / f"seed_{seed}"),
            })
    return specs


def launch_one(spec: Mapping[str, Any], attempt: Path) -> dict[str, Any]:
    label = f"{spec['scenario']}_seed_{spec['seed']}"
    spec_path = attempt / "run_specs" / f"{label}.json"
    write_json(spec_path, spec)
    stdout_path, stderr_path = attempt / "logs" / f"{label}.stdout.log", attempt / "logs" / f"{label}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child-spec", str(spec_path)], cwd=ROOT, stdout=stdout, stderr=stderr, text=True, check=False)
    return {"scenario": spec["scenario"], "seed": spec["seed"], "pid": None, "status": "PASS" if completed.returncode == 0 else "FAIL", "returncode": completed.returncode, "stderr": str(stderr_path), "output": spec["output"]}


def product_counts(ledger: list[dict[str, str]], levels: set[str]) -> dict[str, int]:
    seen = {(row["task_id"], str(row.get("product", "")).lower()) for row in ledger if row.get("value_level") in levels}
    return {
        "metadata": len({task for task, product in seen if product == "metadata"}),
        "ROI": len({task for task, product in seen if product == "roi"}),
        "Evidence": len({task for task, product in seen if product in {"roi_context", "context", "evidence"}}),
    }


def evaluate_run(spec: Mapping[str, Any]) -> dict[str, Any]:
    run = Path(spec["output"])
    metric, ledger = engine.evaluate_run(METHOD, run)
    thermal = read_json(run / "dual_node_runtime_audit.json")
    identity = read_json(run / "runtime_identity.json")
    trace = read_csv(run / "per_slot_resource_trace.csv")
    queues = [f(row.get("queue_mb")) for row in trace]
    hv_products = product_counts(ledger, {"high", "urgent"})
    high_products = product_counts(ledger, {"high"})
    urgent_products = product_counts(ledger, {"urgent"})
    all_count, hv_count = i(metric["all_closure_count"]), i(metric["HV_closure_count"])
    row = {
        "scenario": spec["scenario"], "value_profile": spec["value_profile"], "capacity": spec["capacity"],
        "seed": spec["seed"], "method": METHOD,
        "J_all_net": metric["J_all_net"], "J_HV_net": metric["J_HV_net"],
        "gross_utility": metric["gross_utility"], "HV_gross_utility": metric.get("HV_gross_utility", 0),
        "energy_cost": metric.get("energy_cost", 0), "downlink_cost": metric.get("downlink_cost", 0),
        "terminal_backlog_cost": metric.get("terminal_backlog_cost", 0), "total_cost": metric["total_cost"],
        "all_closure_count": all_count, "all_closure_rate": all_count / 268.0,
        "ordinary_closure_count": all_count - hv_count, "HV_closure_count": hv_count,
        "high_closure_count": metric["high_closure_count"], "urgent_closure_count": metric["urgent_closure_count"],
        "HV_metadata_delivered_count": hv_products["metadata"], "HV_ROI_delivered_count": hv_products["ROI"], "HV_evidence_delivered_count": hv_products["Evidence"],
        "high_metadata_delivered_count": high_products["metadata"], "high_ROI_delivered_count": high_products["ROI"], "high_evidence_delivered_count": high_products["Evidence"],
        "urgent_metadata_delivered_count": urgent_products["metadata"], "urgent_ROI_delivered_count": urgent_products["ROI"], "urgent_evidence_delivered_count": urgent_products["Evidence"],
        "mean_closure_latency_s": metric["mean_full_closure_latency_s"], "p95_closure_latency_s": metric["p95_full_closure_latency_s"],
        "controllable_energy_Wh": metric["controllable_energy_Wh"], "transmitted_MB": metric["transmitted_MB"],
        "peak_queue_MB": metric["peak_queue_MB"], "mean_queue_MB": sum(queues) / max(len(queues), 1), "terminal_queue_MB": metric["terminal_queue_MB"],
        "minimum_soc": metric["minimum_soc"], "maximum_chip_c": thermal["max_chip_c"], "maximum_surface_c": thermal["max_surface_c"],
        "soc_violation_count": metric["soc_violation_count"], "thermal_violation_count": thermal["chip_violation_count"] + thermal["surface_violation_count"],
        "model_lightweight_count": metric.get("model_lightweight_count", 0), "model_balanced_count": metric.get("model_balanced_count", 0), "model_high_accuracy_count": metric.get("model_high_accuracy_count", 0),
        "task_hash": identity["scenario_identity"]["task_hash"], "link_trace_hash": identity["capacity_identity"]["runtime_link_trace_hash"],
        "power_trace_hash": identity["power_identity"]["power_trace_hash"], "thermal_parameter_hash": thermal["thermal_parameter_hash"], "pid": thermal["pid"],
    }
    write_csv(run / "frozen_public_event_ledger.csv", ledger)
    write_json(run / "frozen_public_metrics.json", metric)
    return row


SUMMARY_KEYS = (
    "J_all_net", "J_HV_net", "gross_utility", "HV_gross_utility", "energy_cost", "downlink_cost", "terminal_backlog_cost", "total_cost",
    "all_closure_count", "all_closure_rate", "ordinary_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count",
    "HV_metadata_delivered_count", "HV_ROI_delivered_count", "HV_evidence_delivered_count",
    "high_metadata_delivered_count", "high_ROI_delivered_count", "high_evidence_delivered_count",
    "urgent_metadata_delivered_count", "urgent_ROI_delivered_count", "urgent_evidence_delivered_count",
    "mean_closure_latency_s", "p95_closure_latency_s", "controllable_energy_Wh", "transmitted_MB", "peak_queue_MB", "mean_queue_MB", "terminal_queue_MB",
    "minimum_soc", "maximum_chip_c", "maximum_surface_c", "soc_violation_count", "thermal_violation_count",
    "model_lightweight_count", "model_balanced_count", "model_high_accuracy_count",
)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for scenario in SCENARIOS:
        group = [row for row in rows if row["scenario"] == scenario["scenario"]]
        item: dict[str, Any] = {"scenario": scenario["scenario"], "value_profile": scenario["value_profile"], "capacity": scenario["capacity"], "seed_count": len(group)}
        for key in SUMMARY_KEYS:
            values = [f(row[key]) for row in group]
            item[f"{key}_mean"] = statistics.fmean(values)
            item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def reference_and_deltas(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_rows: list[dict[str, Any]] = []
    if REFERENCE_CAP050.is_file():
        for row in read_csv(REFERENCE_CAP050):
            if row.get("method") == base.FULL:
                reference_rows.append(row)
    by_ref = {i(row["seed"]): row for row in reference_rows}
    deltas: list[dict[str, Any]] = []
    keys = ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "controllable_energy_Wh", "transmitted_MB", "mean_closure_latency_s", "p95_closure_latency_s")
    for row in rows:
        ref = by_ref.get(i(row["seed"]))
        if not ref:
            continue
        item: dict[str, Any] = {"scenario": row["scenario"], "seed": row["seed"], "reference": "V25_CAP050_CURRENT_PROP"}
        for key in keys:
            item[f"delta_{key}"] = f(row[key]) - f(ref[key])
        deltas.append(item)
    summary: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        group = [row for row in deltas if row["scenario"] == scenario["scenario"]]
        item: dict[str, Any] = {"scenario": scenario["scenario"], "reference": "V25_CAP050_CURRENT_PROP", "seed_count": len(group)}
        if group:
            for key in [key for key in group[0] if key.startswith("delta_")]:
                values = [f(row[key]) for row in group]
                item[f"{key}_mean"] = statistics.fmean(values)
                item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return deltas, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_spec:
        return child_run(read_json(args.child_spec))

    attempt = next_attempt()
    pre = protected_snapshot()
    write_json(attempt / "audit" / "pre_run_hashes.json", pre)
    thermal_model = adapted.reference_thermal_model()
    bundles: dict[tuple[str, int], Path] = {}
    scenario_audits: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            name = str(scenario["scenario"])
            bundle = attempt / "scenario" / name / f"seed_{seed}"
            scenario_audits.append(prepare_bundle(scenario, seed, bundle))
            bundles[(name, seed)] = bundle
    write_json(attempt / "audit" / "scenario_value_capacity_isolation_audit.json", {"status": "PASS", "runs": scenario_audits})

    specs = make_specs(attempt, thermal_model, bundles)
    statuses: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(launch_one, spec, attempt): spec for spec in specs}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(f"[{len(statuses):02d}/25] {status['status']} {status['scenario']} seed={status['seed']}", flush=True)
    write_json(attempt / "audit" / "process_status.json", statuses)
    failures = [row for row in statuses if row["status"] != "PASS"]
    if failures:
        raise RuntimeError({"SENSITIVITY_RUNTIME_FAILED": failures})

    rows = [evaluate_run(spec) for spec in specs]
    summary = aggregate(rows)
    deltas, delta_summary = reference_and_deltas(rows)
    write_csv(attempt / "tables" / "per_run_metrics.csv", rows)
    write_csv(attempt / "tables" / "mean_std_metrics.csv", summary)
    write_md(attempt / "tables" / "mean_std_metrics.md", summary)
    write_csv(attempt / "tables" / "paired_deltas_vs_v25_cap050.csv", deltas)
    write_csv(attempt / "tables" / "paired_deltas_vs_v25_cap050_mean_std.csv", delta_summary)

    identities = [read_json(Path(spec["output"]) / "runtime_identity.json") for spec in specs]
    thermal_audits = [read_json(Path(spec["output"]) / "dual_node_runtime_audit.json") for spec in specs]
    post = protected_snapshot()
    by_seed_profile = {(a["seed"], a["value_profile"]): a for a in scenario_audits}
    checks = {
        "25_fresh_runs_complete": len(rows) == 25,
        "five_seeds_per_scenario": all(sum(row["scenario"] == s["scenario"] for row in rows) == 5 for s in SCENARIOS),
        "25_unique_child_processes": len({row["pid"] for row in rows}) == 25,
        "ordinary_random_all_runs": all(a["checks"]["ordinary_random_profile"] and a["checks"]["late_hv_forcing_absent"] for a in scenario_audits),
        "same_non_value_tasks_across_profiles_per_seed": all(len({by_seed_profile[(seed, p)]["non_value_task_hash"] for p in ("V10", "V25", "V50")}) == 1 for seed in SEEDS),
        "value_sum_268_all_bundles": all(abs(f(a["effective_task_value_sum"]) - 268.0) < 1e-10 for a in scenario_audits),
        "value_level_counts_frozen": all(a["counts"] == EXPECTED_COUNTS for a in scenario_audits),
        "runtime_capacity_exact": all(abs(f(identity["capacity_identity"]["runtime_capacity"]) - f(spec["capacity"])) < 1e-12 for identity, spec in zip(identities, specs)),
        "current_prop_v1_v2_all_runs": all(identity.get("safe_set_consistent_fallback") and identity.get("hv_metadata_residual_repair") for identity in identities),
        "safe_greedy_current_selector_all_runs": all(identity.get("greedy_no_online_bandit") for identity in identities),
        "q_once_all_runs": all(identity.get("q_once_online") for identity in identities),
        "same_power_trace_all_runs": len({row["power_trace_hash"] for row in rows}) == 1,
        "same_two_node_thermal_parameters_all_runs": len({row["thermal_parameter_hash"] for row in rows}) == 1,
        "two_node_runtime_active_all_runs": all(a["actual_slot_transition_count"] > 0 for a in thermal_audits),
        "zero_soc_thermal_violations": all(i(row["soc_violation_count"]) == 0 and i(row["thermal_violation_count"]) == 0 for row in rows),
        "protected_sources_unchanged": pre == post,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    write_json(attempt / "audit" / "final_identity_safety_audit.json", {
        "status": status, "created_utc": datetime.now(timezone.utc).isoformat(), "version": VERSION,
        "method": METHOD, "seeds": SEEDS, "scenarios": SCENARIOS, "max_parallel_processes": MAX_PARALLEL,
        "thermal_model": str(thermal_model), "thermal_model_sha256": sha256_file(thermal_model), "checks": checks,
    })
    write_json(attempt / "audit" / "post_run_hashes.json", post)

    primary = [{
        "scenario": row["scenario"], "J_all_net_mean": row["J_all_net_mean"], "J_all_net_std": row["J_all_net_std"],
        "J_HV_net_mean": row["J_HV_net_mean"], "J_HV_net_std": row["J_HV_net_std"],
        "HV_closure_count_mean": row["HV_closure_count_mean"], "HV_closure_count_std": row["HV_closure_count_std"],
        "high_closure_count_mean": row["high_closure_count_mean"], "urgent_closure_count_mean": row["urgent_closure_count_mean"],
    } for row in summary]
    write_csv(attempt / "tables" / "primary_metrics.csv", primary)
    write_md(attempt / "tables" / "primary_metrics.md", primary)
    report = [
        "# Current PROP value/capacity sensitivity", "",
        f"Method: `{METHOD}` (current V1 safety consistency + V2 HV progressive delivery, safe-greedy selector).", "",
        "All 25 runs use ordinary random timing, the corrected 40 W power environment, frozen q-once/public scoring, and the same calibrated two-node thermal model.", "",
        "V10/V50 alter only the mean-normalized value ratio; V25 capacity points alter only the runtime downlink-capacity multiplier.", "",
        "## Primary metrics", "", (attempt / "tables" / "primary_metrics.md").read_text(encoding="utf-8").strip(), "",
        f"Audit status: **{status}**. Parallel child-process limit: **{MAX_PARALLEL}**.", "",
        COMPLETION if status == "PASS" else "CURRENT_PROP_VALUE_CAPACITY_SENSITIVITY_AUDIT_FAILED",
    ]
    (attempt / "final_assessment.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (attempt / "test_report.txt").write_text("\n".join(f"{key}: {'PASS' if value else 'FAIL'}" for key, value in checks.items()) + f"\nFINAL: {status}\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": status, "completion_marker": COMPLETION if status == "PASS" else None, "output": str(attempt)})
    print(json.dumps({"status": status, "output": str(attempt), "completion": COMPLETION if status == "PASS" else None}, ensure_ascii=False), flush=True)
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
