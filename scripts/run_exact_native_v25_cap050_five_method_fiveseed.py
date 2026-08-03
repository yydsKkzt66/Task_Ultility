"""Exact-native five-method controlled-V25 five-seed experiment.

Seed 2026 is the exact frozen V25 anchor.  Seeds 2027--2030 retain the same
268 tasks and permute arrival slots only within each value level.  This keeps
the frozen value counts, effective 1:3:10:25 values, required products,
deadline durations, late-window HV counts, power trace, contact trace and
requested capacity environment while creating distinct temporal orderings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402
import scripts.run_prop_reqalign_seed2026_v1 as reqrun  # noqa: E402


SEEDS = (2026, 2027, 2028, 2029, 2030)
DEFAULT_CAPACITY = 0.50
SLOTS = 4320
METHOD_CORRECTED = reqrun.METHOD_PROP_NEW
METHOD_MYOPIC = "MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN"
METHOD_B1 = "B1-COMMON-THERMAL"
METHOD_OFFLOAD = "OFFLOAD-EFT-THERMAL-ALIGNED"
METHOD_TUT = "TUT-UOI-PRODUCT-LITE-v2"
METHODS = (METHOD_MYOPIC, METHOD_B1, METHOD_OFFLOAD, METHOD_TUT, METHOD_CORRECTED)
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}
EXPECTED_ANCHOR_HASH = v25.EXPECTED_TASK_HASH
EXPECTED_CORRECTED = {
    "J_all_net": Decimal("84.81879696613005556024229349"),
    "J_HV_net": Decimal("60.83715905944463137675442229"),
    "HV_closure_count": 42,
}
EXPECTED_TUT = {
    "J_all_net": Decimal("85.95275654067541261781494951"),
    "J_HV_net": Decimal("46.93953837671579199341830073"),
    "HV_closure_count": 26,
}
EXPECTED_CAP035 = {
    METHOD_CORRECTED: {
        "J_all_net": Decimal("81.69411543416106761044753997"),
        "J_HV_net": Decimal("57.90294032533795185236522636"),
        "HV_closure_count": 40,
    },
    METHOD_TUT: {
        "J_all_net": Decimal("83.57721157955365551783965731"),
        "J_HV_net": Decimal("46.00843703058319872849859307"),
        "HV_closure_count": 26,
    },
}
DEFAULT_OUTPUT = ROOT / "outputs" / "_v25_exact_native_five_method_controlled_fiveseed_cap050_v1"
SOURCE_BUNDLE = (
    ROOT
    / "outputs"
    / "_v25_bupt1_hv_urgent_residual_guard_seed2026_cap050_v1"
    / "config"
    / "frozen_runtime_bundle"
)
MYOPIC_SOURCE = ROOT / "src" / "routers" / "myopic_frozen_calibrated_router.py"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
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
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dec(value: Any) -> Decimal:
    return Decimal(str(value or 0))


def integer(value: Any) -> int:
    return int(Decimal(str(value or 0)))


def _controlled_order(tasks: list[dict[str, str]], seed: int) -> dict[str, int]:
    if seed == 2026:
        return {row["task_id"]: int(float(row["arrival_slot"])) for row in tasks}
    assignment: dict[str, int] = {}
    for level in ("low", "medium", "high", "urgent"):
        group = [row for row in tasks if row["value_level"] == level]
        slots = sorted(int(float(row["arrival_slot"])) for row in group)
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"exact-native-controlled-v25-v1:{seed}:{level}:{row['task_id']}".encode("utf-8")
            ).digest(),
        )
        for slot, row in zip(slots, ordered):
            assignment[row["task_id"]] = slot
    return assignment


def prepare_controlled_bundle(seed: int, destination: Path) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(SOURCE_BUNDLE, destination)
    tasks = read_csv(destination / "task_manifest.csv")
    slots = read_csv(destination / "slot_trace.csv")
    if seed == 2026:
        contract = json.loads(
            (destination / "bundle_contract.json").read_text(encoding="utf-8")
        )
        level_counts = dict(Counter(row["value_level"] for row in tasks))
        late_counts = dict(
            Counter(
                row["value_level"]
                for row in tasks
                if Decimal("43200")
                <= Decimal(row["arrival_time_s"])
                < Decimal("64800")
            )
        )
        checks = {
            "task_count_is_268": len(tasks) == 268,
            "value_counts_match": level_counts == EXPECTED_COUNTS,
            "task_ids_unique": len({row["task_id"] for row in tasks}) == 268,
            "arrival_slots_unique": len({row["arrival_slot"] for row in tasks})
            == 268,
            "late_high_is_30": late_counts.get("high", 0) == 30,
            "late_urgent_is_15": late_counts.get("urgent", 0) == 15,
            "effective_values_unchanged": all(
                Decimal(row["effective_task_value"])
                == Decimal(row["normalization_alpha"])
                * Decimal(
                    {
                        "low": "1",
                        "medium": "3",
                        "high": "10",
                        "urgent": "25",
                    }[row["value_level"]]
                )
                for row in tasks
            ),
            "anchor_hash_exact": contract["task_hash"]
            == EXPECTED_ANCHOR_HASH,
        }
        if not all(checks.values()):
            raise AssertionError(
                {"controlled_bundle_failed": seed, "checks": checks}
            )
        v25.load_bundle(destination)
        return {
            "seed": seed,
            "task_hash": contract["task_hash"],
            "environment_hash": contract["environment_hash"],
            "level_counts": level_counts,
            "late_counts": late_counts,
            "moved_task_count": 0,
            "assignment_sha256": stable_hash(
                sorted(
                    (row["task_id"], int(float(row["arrival_slot"])))
                    for row in tasks
                )
            ),
            "checks": checks,
        }
    assignment = _controlled_order(tasks, seed)
    original = {row["task_id"]: int(float(row["arrival_slot"])) for row in tasks}
    slot_seconds = 20.0
    for row in tasks:
        new_slot = assignment[row["task_id"]]
        deadline_s = Decimal(row["original_deadline_slack_s"])
        arrival_s = Decimal(new_slot) * Decimal("20")
        row["arrival_slot"] = str(new_slot)
        row["virtual_time"] = str(new_slot)
        row["arrival_time_s"] = str(arrival_s)
        row["deadline_time_s"] = str(arrival_s + deadline_s)
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata.update(
            {
                "controlled_v25_multiseed": True,
                "controlled_v25_seed": seed,
                "controlled_v25_original_arrival_slot": original[row["task_id"]],
            }
        )
        row["metadata_json"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    task_at_slot = {assignment[row["task_id"]]: row["task_id"] for row in tasks}
    for row in slots:
        row["task_id"] = task_at_slot.get(int(float(row["slot_index"])), "")
    write_csv(destination / "task_manifest.csv", tasks)
    write_csv(destination / "slot_trace.csv", slots)

    contract_path = destination / "bundle_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["seed"] = seed
    contract["task_hash"] = v25.v25_bundle_hash(tasks)
    contract["environment_hash"] = stable_hash(
        {
            "base_environment_hash": contract["environment_hash"],
            "controlled_seed": seed,
            "task_hash": contract["task_hash"],
            "capacity": contract["capacity"],
        }
    )
    for name in contract["file_sha256"]:
        contract["file_sha256"][name] = sha256_file(destination / name)
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    level_counts = dict(Counter(row["value_level"] for row in tasks))
    late_counts = dict(
        Counter(
            row["value_level"]
            for row in tasks
            if Decimal("43200") <= Decimal(row["arrival_time_s"]) < Decimal("64800")
        )
    )
    checks = {
        "task_count_is_268": len(tasks) == 268,
        "value_counts_match": level_counts == EXPECTED_COUNTS,
        "task_ids_unique": len({row["task_id"] for row in tasks}) == 268,
        "arrival_slots_unique": len({row["arrival_slot"] for row in tasks}) == 268,
        "late_high_is_30": late_counts.get("high", 0) == 30,
        "late_urgent_is_15": late_counts.get("urgent", 0) == 15,
        "effective_values_unchanged": all(
            Decimal(row["effective_task_value"])
            == Decimal(row["normalization_alpha"])
            * Decimal({"low": "1", "medium": "3", "high": "10", "urgent": "25"}[row["value_level"]])
            for row in tasks
        ),
        "anchor_hash_exact": seed != 2026 or contract["task_hash"] == EXPECTED_ANCHOR_HASH,
    }
    if not all(checks.values()):
        raise AssertionError({"controlled_bundle_failed": seed, "checks": checks})
    # Contract-first load is the strongest schema/identity check.
    v25.load_bundle(destination)
    return {
        "seed": seed,
        "task_hash": contract["task_hash"],
        "environment_hash": contract["environment_hash"],
        "level_counts": level_counts,
        "late_counts": late_counts,
        "moved_task_count": sum(original[task_id] != slot for task_id, slot in assignment.items()),
        "assignment_sha256": stable_hash(sorted(assignment.items())),
        "checks": checks,
    }


def bind_myopic_runtime(runner: Any, router: Any) -> None:
    original_factory = runner.step_runner_factory

    def factory(queue):
        step_runner = original_factory(queue)
        original_run_step = step_runner.run_step

        def intercepted(timeline_step, resource_state, mission_state, comm_state, active_router):
            router.bind_step(timeline_step, resource_state, comm_state)
            return original_run_step(
                timeline_step, resource_state, mission_state, comm_state, active_router
            )

        step_runner.run_step = intercepted
        return step_runner

    runner.step_runner_factory = factory


def load_myopic_router_class() -> Any:
    spec = importlib.util.spec_from_file_location("exact_native_myopic_reqalign_v1", MYOPIC_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(MYOPIC_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MyopicFrozenCalibratedRouter


def scenario_identity(
    method: str,
    seed: int,
    scenario: Any,
    bundle: Any,
    power: dict[str, Any],
    capacity: float,
) -> dict[str, Any]:
    values = dict(getattr(scenario, "runtime_task_values", {}))
    mismatches = [
        task_id
        for task_id, expected in bundle.task_values.items()
        if Decimal.from_float(float(values.get(task_id, "nan")))
        != Decimal.from_float(float(expected))
    ]
    identity = {
        "method": method,
        "seed": seed,
        "task_hash": scenario.task_hash,
        "expected_task_hash": bundle.contract["task_hash"],
        "task_hash_match": scenario.task_hash == bundle.contract["task_hash"],
        "task_count": len(values),
        "runtime_value_mismatch_count": len(mismatches),
        "field_mismatch_count": int(getattr(scenario, "field_mismatch_count", 0)),
        "value_level_counts": bundle.contract["value_level_counts"],
        "frozen_power_identity_match": power["frozen_power"] == v25.FROZEN_POWER,
        "frozen_power_identity": power,
        "capacity": capacity,
    }
    if not (
        identity["task_hash_match"]
        and identity["task_count"] == 268
        and identity["runtime_value_mismatch_count"] == 0
        and identity["field_mismatch_count"] == 0
        and identity["frozen_power_identity_match"]
    ):
        raise RuntimeError({"scenario_identity_failed": identity})
    return identity


def run_prop_family(method: str, seed: int, bundle_root: Path, output: Path, capacity: float) -> None:
    v25.child_setup_path(v25.SOURCE_PROP)
    from native_scenario import reconstruct_native_scenario
    from native_support import normalize_episode
    from src.actions.action_space import ActionSpace
    from src.common.types import LinkTrace, ResourceTrace, TaskSample, TimelineStep
    from src.evaluation.baselines import build_routers
    from src.simulator.environment import SimulationEnvironment

    prop_mod = v25.import_prop_script()
    bundle = v25.load_bundle(bundle_root)
    scenario = reconstruct_native_scenario(bundle, TaskSample, ResourceTrace, LinkTrace, TimelineStep)
    calibration = v25.read_json(v25.SHARED_PRIOR_CALIBRATION)
    configs = prop_mod._configure(scenario.configs, calibration, {"enabled": False})
    scenario = replace(scenario, configs=configs)
    scenario, power = v25.inject_power(scenario, v25.SOURCE_PROP)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, capacity)
    identity = scenario_identity(method, seed, scenario, bundle, power, capacity)
    identity["capacity_identity"] = capacity_identity
    environment = SimulationEnvironment(scenario.configs)
    runner = environment.build_episode_runner(scenario.timeline[:SLOTS])
    if method == METHOD_CORRECTED:
        delegate = build_routers(
            scenario.configs["experiment"], ActionSpace(scenario.configs["action_space"])
        )[0]
    else:
        delegate = load_myopic_router_class()(
            v25.MYOPIC_PARAM, router_name="MYOPIC-FROZEN-CALIBRATED"
        )
        bind_myopic_runtime(runner, delegate)
    wrapper = reqrun.RequirementAwareRouterWrapper(
        delegate,
        reqrun.load_projection_config(True),
        scenario.timeline[:SLOTS],
        scenario.configs["simulator"],
        reqrun.load_projection_module(),
    )
    reqrun.bind_projection_runtime(runner, wrapper)
    result = runner.run(wrapper)
    normalized = normalize_episode(method, scenario, result, runner, SLOTS)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    v25.write_csv(output / "candidate_action_log.csv", list(runner.last_candidate_action_logs))
    reqrun.enrich_projection_rows(wrapper, output)
    reqrun.write_energy_components(output, result)
    native_identity = {
        "method": method,
        "source_root": str(v25.SOURCE_PROP.resolve()),
        "source_identity_passed": True,
        "corrected_reqalign_enabled": True,
        "delegate": type(delegate).__name__,
        "online_update_count": getattr(delegate, "online_update_count", "not_exposed"),
        "scenario_identity": identity,
    }
    write_json(output / "native_identity.json", native_identity)
    write_json(output / "reconstructed_scenario_identity.json", identity)
    write_json(
        output / "method_run_status.json",
        {
            "status": "PASS",
            "method": method,
            "seed": seed,
            "simulation_started": True,
            "fresh_state": True,
            "scenario_reconstruction_count": 1,
        },
    )
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": seed})


def run_baseline(method: str, seed: int, bundle_root: Path, output: Path, capacity: float) -> None:
    source_root = v25.SOURCE_TUT if method == METHOD_TUT else v25.SOURCE_B1_OFFLOAD
    v25.child_setup_path(source_root)
    bundle = v25.load_bundle(bundle_root)
    module = __import__(
        {METHOD_B1: "native_b1", METHOD_OFFLOAD: "native_offload", METHOD_TUT: "native_tut"}[method]
    )
    binding = module.make_binding(source_root.resolve())
    native_identity = dict(binding.identity())
    if not native_identity.get("source_identity_passed", False):
        raise RuntimeError({"native_source_identity_failed": method, "identity": native_identity})
    scenario = binding.build_scenario(bundle)
    scenario, power = v25.inject_power(scenario, source_root)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, capacity)
    identity = scenario_identity(method, seed, scenario, bundle, power, capacity)
    identity["capacity_identity"] = capacity_identity
    native = binding.simulate(scenario, SLOTS)
    normalized = dict(binding.normalize_outputs(native, bundle))
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    native_identity.update({"method": method, "scenario_identity": identity})
    write_json(output / "native_identity.json", native_identity)
    write_json(output / "reconstructed_scenario_identity.json", identity)
    write_json(
        output / "method_run_status.json",
        {
            "status": "PASS",
            "method": method,
            "seed": seed,
            "simulation_started": True,
            "fresh_state": True,
            "scenario_reconstruction_count": 1,
        },
    )
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": seed})


def child_main(args: argparse.Namespace) -> int:
    method = str(args.method)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if method in {METHOD_CORRECTED, METHOD_MYOPIC}:
        run_prop_family(method, int(args.seed), Path(args.bundle_root), output, float(args.capacity))
    elif method in {METHOD_B1, METHOD_OFFLOAD, METHOD_TUT}:
        run_baseline(method, int(args.seed), Path(args.bundle_root), output, float(args.capacity))
    else:
        raise ValueError(method)
    print(json.dumps({"status": "PASS", "seed": int(args.seed), "method": method}, ensure_ascii=False))
    return 0


def launch(
    method: str,
    seed: int,
    bundle_root: Path,
    output: Path,
    logs: Path,
    capacity: float,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--seed",
        str(seed),
        "--method",
        method,
        "--capacity",
        str(capacity),
        "--bundle-root",
        str(bundle_root),
        "--output",
        str(output),
    ]
    started = datetime.now(timezone.utc).isoformat()
    process = subprocess.run(
        command,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    finished = datetime.now(timezone.utc).isoformat()
    logs.mkdir(parents=True, exist_ok=True)
    safe = method.replace("/", "_")
    (logs / f"seed_{seed}_{safe}_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (logs / f"seed_{seed}_{safe}_stderr.txt").write_text(process.stderr, encoding="utf-8")
    status = {
        "seed": seed,
        "method": method,
        "returncode": process.returncode,
        "started_utc": started,
        "finished_utc": finished,
        "output": str(output),
        "stdout_tail": process.stdout[-1000:],
        "stderr_tail": process.stderr[-4000:],
    }
    if process.returncode != 0 or not (output / "method_completed.json").is_file():
        raise RuntimeError({"method_runtime_failed": status})
    return status


def evaluate_method(
    method: str,
    path: Path,
    bundle_root: Path,
    eval_mod: Any,
    capacity: float,
) -> dict[str, Any]:
    config = json.loads((bundle_root / "resolved_config.json").read_text(encoding="utf-8"))
    metric, accounting = eval_mod.summarize_method(
        method,
        f"EXACT_NATIVE_CONTROLLED_V25_CAP{int(round(capacity * 100)):03d}",
        path,
        config["product_requirement_utility"],
        config["unified_ground_received_accounting"],
    )
    metric["J_HV_net"] = reqrun.hv_net(
        eval_mod, path, metric, config["product_requirement_utility"]
    )
    metric.update(
        reqrun.subgroup_metrics(
            eval_mod, path, metric, config["product_requirement_utility"]
        )
    )
    energy = reqrun.energy_metrics(method, path)
    metric.update(energy)
    metric["total_transmitted_bytes"] = dec(energy["transmitted_MB"]) * Decimal("1000000")
    metric["public_accounting_passed"] = bool(accounting.get("passed", True))
    return metric


def selected_metrics(seed: int, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "method": row["method"],
        "task_count": 268,
        "HV_task_count": 75,
        "J_all_net": row["J_all_net"],
        "J_HV_net": row["J_HV_net"],
        "N_HV_closure": row["HV_closure_count"],
        "N_high_closure": row["high_closure_count"],
        "N_urgent_closure": row["urgent_closure_count"],
        "all_task_closure_count": row["closure_count"],
        "all_task_closure_rate": row["closure_rate"],
        "N_HV_metadata": row["HV_metadata_delivered_count"],
        "N_HV_ROI": row["HV_ROI_delivered_count"],
        "N_HV_evidence": row["HV_evidence_delivered_count"],
        "successful_closure_mean_latency_s": row["successful_closure_mean_latency_s"],
        "successful_closure_p95_latency_s": row["successful_closure_p95_latency_s"],
        "HV_successful_closure_mean_latency_s": row["HV_successful_closure_mean_latency_s"],
        "HV_successful_closure_p95_latency_s": row["HV_successful_closure_p95_latency_s"],
        "urgent_successful_closure_mean_latency_s": row["urgent_successful_closure_mean_latency_s"],
        "urgent_successful_closure_p95_latency_s": row["urgent_successful_closure_p95_latency_s"],
        "deadline_censored_mean_latency_s": row["deadline_censored_mean_latency_s"],
        "controllable_energy_Wh": row["controllable_compute_communication_energy_Wh"],
        "transmitted_MB": row["transmitted_MB"],
        "transmitted_bytes": row["total_transmitted_bytes"],
        "peak_queue_MB": row["peak_downlink_queue_MB"],
        "mean_queue_MB": row["mean_downlink_queue_MB"],
        "minimum_soc": row["minimum_soc"],
        "maximum_thermal": row["maximum_thermal_state"],
        "SOC_violations": row["soc_violation_count"],
        "thermal_violations": row["thermal_violation_count"],
        "public_accounting_passed": row["public_accounting_passed"],
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = [
        "J_all_net",
        "J_HV_net",
        "N_HV_closure",
        "N_high_closure",
        "N_urgent_closure",
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
        "transmitted_bytes",
        "peak_queue_MB",
        "mean_queue_MB",
        "minimum_soc",
        "maximum_thermal",
        "SOC_violations",
        "thermal_violations",
    ]
    output = []
    for method in METHODS:
        group = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "seed_count": len(group)}
        for metric in numeric:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def emit_report_tables(output: Path, metric_rows: list[dict[str, Any]]) -> None:
    """Write the complete report and compact user-facing metric subsets."""
    method_order = {method: index for index, method in enumerate(METHODS)}
    metric_rows.sort(key=lambda row: (int(row["seed"]), method_order[row["method"]]))
    summary = summarize(metric_rows)
    primary_fields = (
        "method", "seed_count", "J_all_net_mean", "J_all_net_std",
        "J_HV_net_mean", "J_HV_net_std", "N_HV_closure_mean",
        "N_HV_closure_std", "N_high_closure_mean", "N_high_closure_std",
        "N_urgent_closure_mean", "N_urgent_closure_std",
    )
    auxiliary_fields = (
        "method", "seed_count", "all_task_closure_count_mean",
        "all_task_closure_count_std", "all_task_closure_rate_mean",
        "all_task_closure_rate_std", "N_HV_metadata_mean", "N_HV_metadata_std",
        "N_HV_ROI_mean", "N_HV_ROI_std", "N_HV_evidence_mean",
        "N_HV_evidence_std", "successful_closure_mean_latency_s_mean",
        "successful_closure_mean_latency_s_std", "successful_closure_p95_latency_s_mean",
        "successful_closure_p95_latency_s_std", "HV_successful_closure_mean_latency_s_mean",
        "HV_successful_closure_mean_latency_s_std", "HV_successful_closure_p95_latency_s_mean",
        "HV_successful_closure_p95_latency_s_std", "urgent_successful_closure_mean_latency_s_mean",
        "urgent_successful_closure_mean_latency_s_std", "urgent_successful_closure_p95_latency_s_mean",
        "urgent_successful_closure_p95_latency_s_std", "deadline_censored_mean_latency_s_mean",
        "deadline_censored_mean_latency_s_std", "controllable_energy_Wh_mean",
        "controllable_energy_Wh_std", "transmitted_MB_mean", "transmitted_MB_std",
        "peak_queue_MB_mean", "peak_queue_MB_std", "mean_queue_MB_mean",
        "mean_queue_MB_std",
    )
    safety_fields = (
        "method", "seed_count", "minimum_soc_mean", "minimum_soc_std",
        "maximum_thermal_mean", "maximum_thermal_std", "SOC_violations_mean",
        "SOC_violations_std", "thermal_violations_mean", "thermal_violations_std",
    )
    primary_summary = [{key: row[key] for key in primary_fields} for row in summary]
    auxiliary_summary = [{key: row[key] for key in auxiliary_fields} for row in summary]
    safety_summary = [{key: row[key] for key in safety_fields} for row in summary]
    primary_per_seed = [
        {key: row[key] for key in (
            "seed", "method", "J_all_net", "J_HV_net", "N_HV_closure",
            "N_high_closure", "N_urgent_closure",
        )}
        for row in metric_rows
    ]
    corrected = next(row for row in summary if row["method"] == METHOD_CORRECTED)
    delta_metrics = (
        "J_all_net", "J_HV_net", "N_HV_closure", "N_high_closure",
        "N_urgent_closure", "all_task_closure_count", "all_task_closure_rate",
    )
    deltas = []
    for row in summary:
        if row["method"] == METHOD_CORRECTED:
            continue
        item: dict[str, Any] = {
            "comparison": f"{METHOD_CORRECTED} minus {row['method']}",
            "reference_method": METHOD_CORRECTED,
            "comparison_method": row["method"],
        }
        for metric in delta_metrics:
            item[f"delta_{metric}"] = corrected[f"{metric}_mean"] - row[f"{metric}_mean"]
        deltas.append(item)

    write_csv(output / "tables" / "all_metrics_per_seed.csv", metric_rows)
    write_csv(output / "tables" / "primary_metrics_per_seed.csv", primary_per_seed)
    write_csv(output / "tables" / "primary_metrics_mean_std.csv", primary_summary)
    write_csv(output / "tables" / "all_metrics_mean_std.csv", summary)
    write_csv(output / "tables" / "auxiliary_metrics_mean_std.csv", auxiliary_summary)
    write_csv(output / "tables" / "safety_metrics_mean_std.csv", safety_summary)
    write_csv(output / "tables" / "prop_vs_other_methods_deltas.csv", deltas)
    write_md(output / "tables" / "primary_metrics_per_seed.md", primary_per_seed)
    write_md(output / "tables" / "primary_metrics_mean_std.md", primary_summary)
    write_md(output / "tables" / "all_metrics_mean_std.md", summary)
    write_md(output / "tables" / "auxiliary_metrics_mean_std.md", auxiliary_summary)
    write_md(output / "tables" / "safety_metrics_mean_std.md", safety_summary)
    write_md(output / "tables" / "prop_vs_other_methods_deltas.md", deltas)


def regression_gate(metrics: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "J_all_net_exact": abs(dec(metrics["J_all_net"]) - dec(expected["J_all_net"])) <= Decimal("1e-12"),
        "J_HV_net_exact": abs(dec(metrics["J_HV_net"]) - dec(expected["J_HV_net"])) <= Decimal("1e-12"),
        "HV_closure_exact": integer(metrics["HV_closure_count"]) == integer(expected["HV_closure_count"]),
        "public_accounting_passed": bool(metrics["public_accounting_passed"]),
        "hard_safety_passed": integer(metrics["soc_violation_count"]) == 0
        and integer(metrics["thermal_violation_count"]) == 0,
    }
    return {"passed": all(checks.values()), "checks": checks, "observed": dict(metrics), "expected": dict(expected)}


def expected_seed2026(capacity: float) -> dict[str, Mapping[str, Any]]:
    if abs(capacity - 0.50) <= 1.0e-12:
        return {METHOD_CORRECTED: EXPECTED_CORRECTED, METHOD_TUT: EXPECTED_TUT}
    if abs(capacity - 0.35) <= 1.0e-12:
        return EXPECTED_CAP035
    raise ValueError(f"no frozen seed-2026 regression anchor for capacity={capacity}")


def capacity_tag(capacity: float) -> str:
    return f"CAP{int(round(capacity * 100)):03d}"


def parent_main(output: Path, preflight_only: bool, capacity: float) -> int:
    if output.exists():
        raise FileExistsError(f"fresh output required: {output}")
    output.mkdir(parents=True, exist_ok=False)
    bundle_identities = []
    bundles: dict[int, Path] = {}
    for seed in SEEDS:
        bundle_root = output / "config" / "controlled_bundles" / f"seed_{seed}"
        bundles[seed] = bundle_root
        bundle_identities.append(prepare_controlled_bundle(seed, bundle_root))
    write_json(output / "audit" / "controlled_bundle_audit.json", bundle_identities)
    if preflight_only:
        print(json.dumps({"status": "PASS", "mode": "preflight_only", "output": str(output)}, ensure_ascii=False))
        return 0

    eval_mod = v25.load_eval_module(output)
    statuses = []
    metric_rows: list[dict[str, Any]] = []

    # Seed-2026 regression gate is mandatory before the remaining grid.
    anchors = expected_seed2026(capacity)
    for method in (METHOD_CORRECTED, METHOD_TUT):
        expected = anchors[method]
        path = output / "runs" / "seed_2026" / method
        statuses.append(launch(method, 2026, bundles[2026], path, output / "logs", capacity))
        metrics = evaluate_method(method, path, bundles[2026], eval_mod, capacity)
        gate = regression_gate(metrics, expected)
        write_json(output / "audit" / f"seed2026_{method}_regression_gate.json", gate)
        if not gate["passed"]:
            raise RuntimeError({"EXACT_NATIVE_SEED2026_REGRESSION_MISMATCH": method, "gate": gate})
        metric_rows.append(selected_metrics(2026, metrics))

    completed = {(2026, METHOD_CORRECTED), (2026, METHOD_TUT)}
    for seed in SEEDS:
        for method in METHODS:
            if (seed, method) in completed:
                continue
            path = output / "runs" / f"seed_{seed}" / method
            statuses.append(launch(method, seed, bundles[seed], path, output / "logs", capacity))
            metrics = evaluate_method(method, path, bundles[seed], eval_mod, capacity)
            if not metrics["public_accounting_passed"]:
                raise RuntimeError({"PUBLIC_ACCOUNTING_MISMATCH": seed, "method": method})
            if integer(metrics["soc_violation_count"]) or integer(metrics["thermal_violation_count"]):
                raise RuntimeError({"HARD_SAFETY_VIOLATION": seed, "method": method})
            metric_rows.append(selected_metrics(seed, metrics))

    emit_report_tables(output, metric_rows)
    write_json(output / "audit" / "method_run_status.json", statuses)
    completion = {
        "status": f"EXACT_NATIVE_V25_{capacity_tag(capacity)}_FIVE_METHOD_FIVESEED_COMPLETE",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "run_count": len(metric_rows),
        "capacity": capacity,
        "seed2026_regression_passed": True,
        "public_accounting_passed": all(bool(row["public_accounting_passed"]) for row in metric_rows),
        "hard_safety_passed": all(integer(row["SOC_violations"]) == 0 and integer(row["thermal_violations"]) == 0 for row in metric_rows),
        "output": str(output),
    }
    write_json(output / "completion_status.json", completion)
    print(json.dumps(completion, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacity", type=float, default=DEFAULT_CAPACITY)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only-existing", action="store_true")
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method")
    parser.add_argument("--bundle-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.child_run:
        return child_main(args)
    if args.report_only_existing:
        rows = read_csv(args.output_dir / "tables" / "all_metrics_per_seed.csv")
        emit_report_tables(args.output_dir, rows)
        print(json.dumps({"status": "REPORT_TABLES_REFRESHED", "output": str(args.output_dir)}, ensure_ascii=False))
        return 0
    return parent_main(args.output_dir, args.preflight_only, float(args.capacity))


if __name__ == "__main__":
    raise SystemExit(main())
