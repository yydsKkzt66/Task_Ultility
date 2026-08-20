"""Five-method, five-seed current-PROP ablation on the two-node thermal model."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
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

import scripts.run_current_prop_slow_control_factorial_single_seed_v1 as slow  # noqa: E402
import scripts.run_dual_node_thermal_calibration_and_four_method_rerun_v1 as dual  # noqa: E402
import scripts.run_normal_scenario_tut_proxy_ground_recovery_v1 as normal  # noqa: E402
import scripts.run_pure_myopic_safe_greedy_offload_thermal_comparison_v1 as thermal_ref  # noqa: E402
import scripts.run_single_seed_controlled_safe_metadata_thermal_exploration_v1 as engine  # noqa: E402


VERSION = "CURRENT-PROP-FIVE-ABLATION-MULTISEED-DUAL-THERMAL-v1"
COMPLETION = "CURRENT_PROP_FIVE_ABLATION_MULTISEED_DUAL_THERMAL_COMPLETE"
SEEDS = (2026, 2027, 2028, 2029, 2030)
CAPACITY, SLOTS = 0.5, 4320
MAX_PARALLEL = 2

FULL = "FULL-V1-V2-DUAL-THERMAL"
NO_LYAPUNOV = "NO-LYAPUNOV-KEEP-MODE-DUAL-THERMAL"
NO_MODE = "NO-MODE-KEEP-LYAPUNOV-DUAL-THERMAL"
FIXED_BALANCED = "FIXED-BALANCED-V1-V2-DUAL-THERMAL"
DIRECT_ONLY = "DIRECT-ONLY-NO-PROGRESSIVE-DUAL-THERMAL"
METHODS = (FULL, NO_LYAPUNOV, NO_MODE, FIXED_BALANCED, DIRECT_ONLY)
SLUG = {method: method.replace("-", "_") for method in METHODS}
OUTPUT_BASE = ROOT / "outputs" / "_current_prop_five_ablation_multiseed_dual_thermal_v1"
SOURCE_BUNDLE = normal.TRUSTED_SCENARIO
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}


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
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


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


def update_seed_fields(value: Any, seed: int) -> Any:
    result = deepcopy(value)
    if isinstance(result, dict):
        for key, item in list(result.items()):
            if key == "seed" and isinstance(item, (int, float, str)):
                result[key] = seed
            else:
                result[key] = update_seed_fields(item, seed)
    elif isinstance(result, list):
        result = [update_seed_fields(item, seed) for item in result]
    return result


def prepare_ordinary_bundle(seed: int, destination: Path) -> dict[str, Any]:
    """Create a value-blind arrival permutation from the trusted ordinary bundle."""

    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(SOURCE_BUNDLE, destination)
    tasks = read_csv(destination / "task_manifest.csv")
    slots = read_csv(destination / "slot_trace.csv")
    source_slots = sorted(i(row["arrival_slot"]) for row in tasks)
    old_slots = {row["task_id"]: i(row["arrival_slot"]) for row in tasks}
    if seed == 2026:
        assignment = dict(old_slots)
        assignment_rule = "trusted_exact_ordinary_seed2026"
    else:
        ordered = sorted(
            tasks,
            key=lambda row: hashlib.sha256(
                f"ordinary-v25-value-blind-arrival-v1:{seed}:{row['task_id']}".encode("utf-8")
            ).digest(),
        )
        assignment = {row["task_id"]: slot for row, slot in zip(ordered, source_slots)}
        assignment_rule = "sha256(seed,task_id) over all tasks; value/requirement excluded"

    for row in tasks:
        if seed == 2026:
            continue
        old_arrival = f(row["arrival_time_s"])
        old_deadline = f(row["deadline_time_s"])
        slack = f(row.get("original_deadline_slack_s"), old_deadline - old_arrival)
        slot = assignment[row["task_id"]]
        arrival = slot * 20.0
        row["arrival_slot"] = str(slot)
        row["virtual_time"] = str(slot)
        row["arrival_time_s"] = str(arrival)
        row["deadline_time_s"] = str(arrival + slack)
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata.update({
            "ordinary_multiseed_v1": True,
            "ordinary_multiseed_seed": seed,
            "ordinary_original_arrival_slot": old_slots[row["task_id"]],
            "arrival_assignment_uses_value_or_requirement": False,
        })
        row["metadata_json"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    task_at_slot = {assignment[row["task_id"]]: row["task_id"] for row in tasks}
    for row in slots:
        row["task_id"] = task_at_slot.get(i(row["slot_index"]), "")
    write_csv(destination / "task_manifest.csv", tasks)
    write_csv(destination / "slot_trace.csv", slots)

    resolved_path = destination / "resolved_config.json"
    write_json(resolved_path, update_seed_fields(read_json(resolved_path), seed))
    contract_path = destination / "bundle_contract.json"
    contract = read_json(contract_path)
    contract.update({
        "seed": seed,
        "arrival_profile": "ordinary_random_value_blind_seed_permutation_v1",
        "active_arrival_swap_count": 0,
        "task_hash": engine.v25.v25_bundle_hash(tasks),
        "ordinary_multiseed_assignment_rule": assignment_rule,
        "ordinary_multiseed_assignment_sha256": stable_hash(sorted(assignment.items())),
    })
    contract["environment_hash"] = stable_hash({
        "source_environment_hash": contract.get("environment_hash"),
        "seed": seed,
        "task_hash": contract["task_hash"],
        "ordinary_assignment": contract["ordinary_multiseed_assignment_sha256"],
    })
    for name in contract["file_sha256"]:
        contract["file_sha256"][name] = sha256_file(destination / name)
    write_json(contract_path, contract)
    normal.load_neutral_bundle(destination)

    counts = dict(Counter(row["value_level"] for row in tasks))
    arrival = {level: [f(row["arrival_time_s"]) for row in tasks if row["value_level"] == level] for level in EXPECTED_COUNTS}
    stats = {
        level: {
            "count": len(times),
            "mean_s": statistics.fmean(times),
            "median_s": statistics.median(times),
            "q1_s": percentile(times, 0.25),
            "q3_s": percentile(times, 0.75),
            "first_12h": sum(time < 43200.0 for time in times),
            "second_12h": sum(time >= 43200.0 for time in times),
        }
        for level, times in arrival.items()
    }
    checks = {
        "task_count_268": len(tasks) == 268,
        "value_counts_exact": counts == EXPECTED_COUNTS,
        "unique_task_ids": len({row["task_id"] for row in tasks}) == 268,
        "unique_arrival_slots": len({i(row["arrival_slot"]) for row in tasks}) == 268,
        "arrival_slot_multiset_preserved": sorted(assignment.values()) == source_slots,
        "assignment_value_blind_by_construction": True,
        "late_hv_forcing_absent": contract["active_arrival_swap_count"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError({"ORDINARY_MULTISEED_SCENARIO_FAILED": seed, "checks": checks})
    return {
        "seed": seed,
        "task_hash": contract["task_hash"],
        "environment_hash": contract["environment_hash"],
        "assignment_rule": assignment_rule,
        "assignment_sha256": contract["ordinary_multiseed_assignment_sha256"],
        "moved_task_count": sum(old_slots[task_id] != slot for task_id, slot in assignment.items()),
        "counts": counts,
        "arrival_statistics": stats,
        "checks": checks,
    }


def install_thermal_and_greedy(adapter: Any, greedy_audit: dict[str, Any]) -> dict[str, Any]:
    patch_audit = dual.install_runtime_patches(adapter)
    import src.evaluation.baselines as baselines

    original_build = baselines.build_routers

    def build_routers(*args: Any, **kwargs: Any) -> Any:
        routers = original_build(*args, **kwargs)
        engine.install_safe_greedy_delegate(routers[0], greedy_audit)
        return routers

    baselines.build_routers = build_routers
    return patch_audit


def run_engine_variant(spec: Mapping[str, Any], output: Path, adapter: Any) -> dict[str, Any]:
    patch_audit: dict[str, Any] = {}
    original_setup = engine.v25.child_setup_path

    def setup(path: Path) -> None:
        nonlocal patch_audit
        original_setup(path)
        patch_audit = dual.install_runtime_patches(adapter)

    engine.v25.child_setup_path = setup
    local = dict(spec)
    local.update({"flags": [True, True, False], "greedy_no_bandit": True})
    if spec["method"] == FIXED_BALANCED:
        local.update({"fixed_model": True, "fixed_model_name": "balanced"})
    if spec["method"] == DIRECT_ONLY:
        local["direct_only"] = True
    engine.configure_and_run_child(local, output)
    return patch_audit


def run_slow_variant(spec: Mapping[str, Any], output: Path, adapter: Any) -> dict[str, Any]:
    patch_audit: dict[str, Any] = {}
    greedy_audit = {"enabled": True, "greedy_select_call_count": 0, "blocked_online_update_count": 0}
    original_setup = engine.v25.child_setup_path

    def setup(path: Path) -> None:
        nonlocal patch_audit
        original_setup(path)
        patch_audit = install_thermal_and_greedy(adapter, greedy_audit)

    engine.v25.child_setup_path = setup
    local = dict(spec)
    local["flags"] = [True, True, False]
    local["variant"] = "no_lyapunov" if spec["method"] == NO_LYAPUNOV else "no_mode"
    slow.configure_and_run_child(local, output)
    identity_path = output / "runtime_identity.json"
    identity = read_json(identity_path)
    identity.update({
        "greedy_no_online_bandit": True,
        "greedy_audit": greedy_audit,
        "current_selector_alignment": "same safe predicted-reward greedy used by current PROP thermal branch",
    })
    write_json(identity_path, identity)
    return patch_audit


def child_run(spec: Mapping[str, Any]) -> int:
    seed, method, output = i(spec["seed"]), str(spec["method"]), Path(spec["output"])
    contract = read_json(Path(spec["bundle_root"]) / "bundle_contract.json")
    engine.SEED = seed
    engine.EXPECTED_TASK_HASH = contract["task_hash"]
    params = dual.thermal.TwoNodeParameters(**read_json(Path(spec["thermal_model"]))["parameters"])
    adapter = dual.thermal.LegacyScalarAdapter(params)
    if method in {FULL, FIXED_BALANCED, DIRECT_ONLY}:
        patch_audit = run_engine_variant(spec, output, adapter)
    elif method in {NO_LYAPUNOV, NO_MODE}:
        patch_audit = run_slow_variant(spec, output, adapter)
    else:
        raise ValueError(method)

    write_csv(output / "dual_node_transition_trace.csv", adapter.rows)
    actual = [row for row in adapter.rows if row.get("caller") in {"run_step", "run", "actual_slot_transition"}]
    write_json(output / "dual_node_runtime_audit.json", {
        "method": method,
        "seed": seed,
        "pid": os.getpid(),
        "thermal_parameter_hash": params.stable_hash(),
        "patch_audit": patch_audit,
        "transition_call_count": len(adapter.rows),
        "actual_slot_transition_count": len(actual),
        "max_chip_c": max((f(row.get("peak_chip_c")) for row in actual), default=0.0),
        "max_surface_c": max((f(row.get("peak_surface_c")) for row in actual), default=0.0),
        "chip_violation_count": sum(f(row.get("peak_chip_c")) > params.t_chip_max_c + 1e-12 for row in actual),
        "surface_violation_count": sum(f(row.get("peak_surface_c")) > params.t_surface_max_c + 1e-12 for row in actual),
    })
    return 0


def make_specs(attempt: Path, model: Path, bundles: Mapping[int, Path]) -> list[dict[str, Any]]:
    specs = []
    for seed in SEEDS:
        task_hash = read_json(bundles[seed] / "bundle_contract.json")["task_hash"]
        for method in METHODS:
            specs.append({
                "version": VERSION,
                "seed": seed,
                "method": method,
                "capacity": CAPACITY,
                "slots": SLOTS,
                "bundle_root": str(bundles[seed]),
                "task_hash": task_hash,
                "thermal_model": str(model),
                "output": str(attempt / "runs" / f"seed_{seed}" / SLUG[method]),
            })
    return specs


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
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    return {
        "seed": seed,
        "method": method,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "output": spec["output"],
        "stderr": str(stderr_path),
    }


def evaluate_run(spec: Mapping[str, Any]) -> dict[str, Any]:
    method, seed, run = str(spec["method"]), i(spec["seed"]), Path(spec["output"])
    metric, ledger = engine.evaluate_run(method, run)
    thermal = read_json(run / "dual_node_runtime_audit.json")
    identity = read_json(run / "runtime_identity.json")
    trace = read_csv(run / "per_slot_resource_trace.csv")
    queue = [f(row.get("queue_mb")) for row in trace]
    all_count, hv_count = i(metric["all_closure_count"]), i(metric["HV_closure_count"])
    row = {
        "seed": seed,
        "method": method,
        "J_all_net": metric["J_all_net"],
        "J_HV_net": metric["J_HV_net"],
        "gross_utility": metric["gross_utility"],
        "total_cost": metric["total_cost"],
        "all_closure_count": all_count,
        "all_closure_rate": all_count / 268.0,
        "ordinary_closure_count": all_count - hv_count,
        "HV_closure_count": hv_count,
        "high_closure_count": metric["high_closure_count"],
        "urgent_closure_count": metric["urgent_closure_count"],
        "metadata_delivery_count": metric["metadata_delivery_count"],
        "ROI_delivery_count": metric["ROI_delivery_count"],
        "Evidence_delivery_count": metric["Evidence_delivery_count"],
        "mean_closure_latency_s": metric["mean_full_closure_latency_s"],
        "p95_closure_latency_s": metric["p95_full_closure_latency_s"],
        "controllable_energy_Wh": metric["controllable_energy_Wh"],
        "transmitted_MB": metric["transmitted_MB"],
        "peak_queue_MB": metric["peak_queue_MB"],
        "mean_queue_MB": sum(queue) / max(len(queue), 1),
        "terminal_queue_MB": metric["terminal_queue_MB"],
        "minimum_soc": metric["minimum_soc"],
        "maximum_chip_c": thermal["max_chip_c"],
        "maximum_surface_c": thermal["max_surface_c"],
        "soc_violation_count": metric["soc_violation_count"],
        "thermal_violation_count": thermal["chip_violation_count"] + thermal["surface_violation_count"],
        "task_hash": identity["scenario_identity"]["task_hash"],
        "link_trace_hash": identity["capacity_identity"]["runtime_link_trace_hash"],
        "power_trace_hash": identity["power_identity"]["power_trace_hash"],
        "thermal_parameter_hash": thermal["thermal_parameter_hash"],
        "pid": thermal["pid"],
    }
    write_csv(run / "frozen_public_event_ledger.csv", ledger)
    write_json(run / "frozen_public_metrics.json", metric)
    return row


SUMMARY_KEYS = (
    "J_all_net", "J_HV_net", "gross_utility", "total_cost",
    "all_closure_count", "all_closure_rate", "ordinary_closure_count",
    "HV_closure_count", "high_closure_count", "urgent_closure_count",
    "metadata_delivery_count", "ROI_delivery_count", "Evidence_delivery_count",
    "mean_closure_latency_s", "p95_closure_latency_s",
    "controllable_energy_Wh", "transmitted_MB", "peak_queue_MB", "mean_queue_MB",
    "terminal_queue_MB", "minimum_soc", "maximum_chip_c", "maximum_surface_c",
    "soc_violation_count", "thermal_violation_count",
)


def aggregate(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        group = [row for row in per_run if row["method"] == method]
        result: dict[str, Any] = {"method": method, "seed_count": len(group)}
        for key in SUMMARY_KEYS:
            values = [f(row[key]) for row in group]
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(result)
    return rows


def paired_deltas(per_run: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by = {(i(row["seed"]), row["method"]): row for row in per_run}
    detailed = []
    for seed in SEEDS:
        full = by[(seed, FULL)]
        for method in METHODS[1:]:
            other = by[(seed, method)]
            row: dict[str, Any] = {"seed": seed, "ablation_method": method, "reference_method": FULL}
            for key in ("J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "controllable_energy_Wh", "transmitted_MB"):
                row[f"delta_{key}"] = f(other[key]) - f(full[key])
            detailed.append(row)
    summary = []
    for method in METHODS[1:]:
        group = [row for row in detailed if row["ablation_method"] == method]
        result: dict[str, Any] = {"ablation_method": method, "reference_method": FULL, "seed_count": len(group)}
        for key in [key for key in group[0] if key.startswith("delta_")]:
            values = [f(row[key]) for row in group]
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_std"] = statistics.stdev(values)
        summary.append(result)
    return detailed, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child_spec:
        return child_run(read_json(args.child_spec))

    attempt = next_attempt()
    model = thermal_ref.reference_thermal_model()
    source_contract = read_json(SOURCE_BUNDLE / "bundle_contract.json")
    if source_contract.get("arrival_profile") != "ordinary_random_restored_pre_late_hv_swap_v1":
        raise RuntimeError("TRUSTED_ORDINARY_SOURCE_MISMATCH")
    scenario_audits, bundles = [], {}
    for seed in SEEDS:
        bundle = attempt / "scenario" / f"ordinary_seed_{seed}"
        scenario_audits.append(prepare_ordinary_bundle(seed, bundle))
        bundles[seed] = bundle
    write_json(attempt / "audit" / "ordinary_multiseed_scenario_audit.json", {"status": "PASS", "seeds": scenario_audits})

    specs = make_specs(attempt, model, bundles)
    statuses = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(launch_one, spec, attempt): spec for spec in specs}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(f"[{len(statuses):02d}/25] {status['status']} seed={status['seed']} {status['method']}", flush=True)
    write_json(attempt / "audit" / "process_status.json", statuses)
    failures = [row for row in statuses if row["status"] != "PASS"]
    if failures:
        raise RuntimeError({"MULTISEED_RUNTIME_FAILED": failures})

    per_run = [evaluate_run(spec) for spec in specs]
    summary = aggregate(per_run)
    paired, paired_summary = paired_deltas(per_run)
    write_csv(attempt / "tables" / "per_run_metrics.csv", per_run)
    write_csv(attempt / "tables" / "mean_std_metrics.csv", summary)
    write_md(attempt / "tables" / "mean_std_metrics.md", summary)
    write_csv(attempt / "tables" / "paired_deltas_per_seed.csv", paired)
    write_csv(attempt / "tables" / "paired_deltas_mean_std.csv", paired_summary)

    identities = [read_json(Path(spec["output"]) / "runtime_identity.json") for spec in specs]
    thermal_audits = [read_json(Path(spec["output"]) / "dual_node_runtime_audit.json") for spec in specs]
    checks = {
        "25_runs_complete": len(per_run) == 25,
        "five_seeds_each_method": all(sum(row["method"] == method for row in per_run) == 5 for method in METHODS),
        "all_unique_processes": len({row["pid"] for row in per_run}) == 25,
        "ordinary_value_blind_arrivals": all(audit["checks"]["assignment_value_blind_by_construction"] for audit in scenario_audits),
        "no_late_hv_forcing": all(audit["checks"]["late_hv_forcing_absent"] for audit in scenario_audits),
        "same_task_hash_within_seed": all(len({row["task_hash"] for row in per_run if i(row["seed"]) == seed}) == 1 for seed in SEEDS),
        "same_link_hash_all_runs": len({row["link_trace_hash"] for row in per_run}) == 1,
        "same_power_hash_all_runs": len({row["power_trace_hash"] for row in per_run}) == 1,
        "same_new_thermal_hash_all_runs": len({row["thermal_parameter_hash"] for row in per_run}) == 1,
        "new_two_node_runtime_trace_all_runs": all(audit["actual_slot_transition_count"] > 0 for audit in thermal_audits),
        "full_v1_v2_all_seeds": all(identity.get("safe_set_consistent_fallback") and identity.get("hv_metadata_residual_repair") for identity in identities if identity["method"] == FULL),
        "fixed_balanced_all_seeds": all(identity.get("fixed_model") == "balanced" for identity in identities if identity["method"] == FIXED_BALANCED),
        "direct_only_all_seeds": all(identity.get("direct_only") for identity in identities if identity["method"] == DIRECT_ONLY),
        "no_lyapunov_prices_zero": all(identity["guidance_runtime_audit"]["nonzero_lambda_call_count"] == 0 for identity in identities if identity["method"] == NO_LYAPUNOV),
        "no_lyapunov_keeps_mode": all(len(identity["guidance_runtime_audit"]["mode_counts"]) > 1 or "resource_saving" in identity["guidance_runtime_audit"]["mode_counts"] for identity in identities if identity["method"] == NO_LYAPUNOV),
        "no_mode_keeps_nonzero_lyapunov": all(identity["guidance_runtime_audit"]["nonzero_lambda_call_count"] > 0 for identity in identities if identity["method"] == NO_MODE),
        "no_mode_neutral_mode": all(set(identity["guidance_runtime_audit"]["mode_counts"]) == {"no_mode"} for identity in identities if identity["method"] == NO_MODE),
        "zero_soc_thermal_violations": all(i(row["soc_violation_count"]) == 0 and i(row["thermal_violation_count"]) == 0 for row in per_run),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    write_json(attempt / "audit" / "method_environment_thermal_audit.json", {
        "status": status,
        "checks": checks,
        "seed_list": SEEDS,
        "methods": METHODS,
        "capacity": CAPACITY,
        "power_environment": "40 W",
        "thermal_model": str(model),
        "thermal_model_sha256": sha256_file(model),
        "thermal_parameter_hash": per_run[0]["thermal_parameter_hash"],
        "max_parallel_processes": MAX_PARALLEL,
        "current_selector": "safe predicted-reward greedy; SafeLinUCB disabled consistently in all five methods",
    })
    (attempt / "test_report.txt").write_text(
        "\n".join(("PASS" if passed else "FAIL") + "\t" + name for name, passed in checks.items())
        + "\n" + ("PASS\t" if status == "PASS" else "FAIL\t") + COMPLETION + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Current PROP five-ablation multiseed dual-thermal assessment", "",
        "Ordinary value-blind V25 arrivals; seeds 2026-2030; capacity=0.5; 40 W; shared two-node thermal model.", "",
        "| method | J_all mean±std | J_HV mean±std | all/HV/high/urgent closure mean | energy Wh | tx MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {f(row['J_all_net_mean']):.6f}±{f(row['J_all_net_std']):.6f} | "
            f"{f(row['J_HV_net_mean']):.6f}±{f(row['J_HV_net_std']):.6f} | "
            f"{f(row['all_closure_count_mean']):.2f}/{f(row['HV_closure_count_mean']):.2f}/"
            f"{f(row['high_closure_count_mean']):.2f}/{f(row['urgent_closure_count_mean']):.2f} | "
            f"{f(row['controllable_energy_Wh_mean']):.6f} | {f(row['transmitted_MB_mean']):.6f} |"
        )
    lines.extend(["", f"Audit status: **{status}**.", "", COMPLETION if status == "PASS" else "CURRENT_PROP_FIVE_ABLATION_MULTISEED_DUAL_THERMAL_FAILED"])
    (attempt / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(attempt / "completion.json", {"status": status, "marker": COMPLETION, "completed_utc": datetime.now(timezone.utc).isoformat(), "output": str(attempt)})
    print("OUTPUT:", attempt)
    print("PARALLEL:", MAX_PARALLEL)
    print(COMPLETION if status == "PASS" else "CURRENT_PROP_FIVE_ABLATION_MULTISEED_DUAL_THERMAL_FAILED")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
