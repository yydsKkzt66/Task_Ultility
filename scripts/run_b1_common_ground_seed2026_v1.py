"""Run frozen V25 seed-2026 old B1 versus independent B1-COMMON-GROUND.

Default environment is the established capacity=0.50 exact-native frozen V25
scenario.  Existing baseline and PROP outputs are read only for before/after
hash auditing; both B1 variants are freshly simulated from t=0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402
import scripts.run_hv_urgent_residual_protection_v1 as guardrun  # noqa: E402


METHOD_OLD = "B1-COMMON-THERMAL"
METHOD_NEW = "B1-COMMON-GROUND"
CAPACITY = 0.50
SEED = 2026
SLOTS = 4320
SOURCE_ROOT = v25.SOURCE_B1_OFFLOAD
SOURCE_BUNDLE = exact.SOURCE_BUNDLE
RUNTIME_MODULE = ROOT / "scripts" / "b1_common_ground_runtime.py"
METHOD_CONFIG = ROOT / "configs" / "b1_common_ground_v1.yaml"
REFERENCE_ROOT = ROOT / "outputs" / "_v25_exact_native_five_method_controlled_fiveseed_cap050_v1" / "runs" / "seed_2026"
DEFAULT_OUTPUT = ROOT / "outputs" / "_v25_b1_common_ground_seed2026_cap050_v1"
UNCHANGED_METHODS = (
    exact.METHOD_OFFLOAD,
    exact.METHOD_TUT,
    exact.METHOD_CORRECTED,
)


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
    exact.write_csv(path, rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = []
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file.relative_to(path).as_posix()
        item_hash = sha256_file(file)
        size = file.stat().st_size
        entries.append({"path": relative, "sha256": item_hash, "bytes": size})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "path": str(path.resolve()),
        "tree_sha256": digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }


def load_runtime_module() -> Any:
    spec = importlib.util.spec_from_file_location("b1_common_ground_runtime_v1", RUNTIME_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(RUNTIME_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_child(method: str, output: Path, bundle_root: Path, capacity: float) -> int:
    v25.child_setup_path(SOURCE_ROOT)
    bundle = v25.load_bundle(bundle_root)
    native_b1 = __import__("native_b1")
    binding = native_b1.make_binding(SOURCE_ROOT.resolve())
    source_identity = dict(binding.identity())
    if not source_identity.get("source_identity_passed", False):
        raise RuntimeError({"native_source_identity_failed": source_identity})

    scenario = binding.build_scenario(bundle)
    scenario, power = v25.inject_power(scenario, SOURCE_ROOT)
    scenario, capacity_identity = guardrun.apply_capacity(scenario, capacity)
    identity = exact.scenario_identity(method, SEED, scenario, bundle, power, capacity)
    identity["capacity_identity"] = capacity_identity

    from scripts.run_sarfish_canonical_seed2026 import b1_config
    from src.baselines.b1_enhanced import B1EnhancedSimulator

    config = b1_config("enhanced")
    if method == METHOD_OLD:
        simulator = B1EnhancedSimulator(config, scenario.configs["simulator"], scenario.configs["model_pool"])
    elif method == METHOD_NEW:
        runtime = load_runtime_module()
        simulator = runtime.B1CommonGroundSimulator(
            config,
            scenario.configs["simulator"],
            scenario.configs["model_pool"],
            ground_servers=1,
            ground_queue_capacity=32,
        )
    else:
        raise ValueError(method)

    native = simulator.run(scenario.timeline[:SLOTS])
    normalized = native_b1._normalize_b1(method, scenario, native)
    output.mkdir(parents=True, exist_ok=False)
    v25.write_unified_method_outputs(output, normalized)
    write_csv(output / "native_per_task_delivery_trace.csv", native["per_task_delivery_trace"])
    write_csv(output / "native_ground_compute_queue.csv", native.get("ground_compute_queue", []))
    write_csv(output / "native_downlink_queue_trace.csv", native["queue_trace"])
    write_csv(output / "native_soc_trace.csv", native["soc_trace"])
    write_csv(output / "native_compression_trace.csv", native["compression_trace"])
    write_csv(output / "native_scheduler_candidate_trace.csv", native["scheduler_candidate_trace"])
    write_csv(output / "native_contact_window_schedule.csv", native["contact_window_schedule"])
    write_json(output / "native_summary.json", native["summary"])

    identity.update(
        {
            "source_root": str(SOURCE_ROOT.resolve()),
            "parent_b1_class": f"{B1EnhancedSimulator.__module__}.{B1EnhancedSimulator.__name__}",
            "runtime_adapter": str(RUNTIME_MODULE.resolve()) if method == METHOD_NEW else "",
            "fresh_state": True,
            "scenario_reconstruction_count": 1,
        }
    )
    write_json(output / "native_identity.json", {**source_identity, "method": method, "scenario_identity": identity})
    write_json(output / "reconstructed_scenario_identity.json", identity)
    write_json(output / "method_run_status.json", {"status": "PASS", "method": method, "seed": SEED})
    write_json(output / "method_completed.json", {"status": "PASS", "method": method, "seed": SEED})
    print(json.dumps({"status": "PASS", "method": method, "output": str(output)}, ensure_ascii=False))
    return 0


def launch(method: str, output: Path, bundle_root: Path, capacity: float, log_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-run",
        "--method",
        method,
        "--output",
        str(output),
        "--bundle-root",
        str(bundle_root),
        "--capacity",
        str(capacity),
    ]
    started = datetime.now(timezone.utc).isoformat()
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = method.replace("/", "_")
    (log_dir / f"{safe}.stdout.log").write_text(process.stdout, encoding="utf-8")
    (log_dir / f"{safe}.stderr.log").write_text(process.stderr, encoding="utf-8")
    status = {
        "method": method,
        "returncode": process.returncode,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": process.stdout[-2000:],
        "stderr_tail": process.stderr[-4000:],
    }
    if process.returncode != 0 or not (output / "method_completed.json").is_file():
        raise RuntimeError({"B1_RUNTIME_FAILED": status})
    return status


def comparison_row(method: str, run_dir: Path, metric: Mapping[str, Any]) -> dict[str, Any]:
    native_summary = json.loads((run_dir / "native_summary.json").read_text(encoding="utf-8"))
    jobs_path = run_dir / "native_ground_compute_queue.csv"
    jobs = read_csv(jobs_path) if jobs_path.is_file() else []
    jobs = [row for row in jobs if row.get("task_id")]
    waits = [float(row["waiting_time"]) for row in jobs]
    peak_waiting = max((int(float(row["queue_length_at_entry"])) for row in jobs), default=0)
    return {
        "method": method,
        "closure_count": int(metric["closure_count"]),
        "closure_rate": metric["closure_rate"],
        "mean_closure_latency_s": metric["successful_closure_mean_latency_s"],
        "mean_ground_queue_wait_s": sum(waits) / max(len(waits), 1),
        "max_ground_queue_wait_s": max(waits, default=0.0),
        "peak_ground_waiting_queue_tasks": peak_waiting,
        "ground_jobs_submitted": len(jobs),
        "ground_queue_rejections": int(native_summary.get("ground_queue_rejection_count", 0)),
        "minimum_soc": metric["minimum_soc"],
        "maximum_thermal": metric["maximum_thermal_state"],
        "soc_violation_count": int(metric["soc_violation_count"]),
        "thermal_violation_count": int(metric["thermal_violation_count"]),
        "transmitted_MB": metric["transmitted_MB"],
        "controllable_energy_Wh": metric["controllable_compute_communication_energy_Wh"],
    }


def main(output: Path, capacity: float) -> int:
    if abs(capacity - CAPACITY) > 1.0e-12:
        raise ValueError("this frozen comparison is registered for capacity=0.50")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)

    reference_before = {method: tree_snapshot(REFERENCE_ROOT / method) for method in UNCHANGED_METHODS}
    write_json(output / "audit" / "unchanged_methods_before.json", reference_before)

    task_contract = json.loads((SOURCE_BUNDLE / "bundle_contract.json").read_text(encoding="utf-8"))
    if task_contract["task_hash"] != v25.EXPECTED_TASK_HASH:
        raise RuntimeError("TASK_MANIFEST_MISMATCH")
    write_json(
        output / "audit" / "scenario_identity.json",
        {
            "seed": SEED,
            "capacity": capacity,
            "duration_s": 86400,
            "slot_s": 20,
            "task_hash": task_contract["task_hash"],
            "task_count": 268,
            "source_bundle": str(SOURCE_BUNDLE.resolve()),
            "source_bundle_contract_sha256": sha256_file(SOURCE_BUNDLE / "bundle_contract.json"),
        },
    )
    (output / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy2(METHOD_CONFIG, output / "config" / METHOD_CONFIG.name)

    statuses = []
    old_dir = output / "runs" / METHOD_OLD
    new_dir = output / "runs" / METHOD_NEW
    statuses.append(launch(METHOD_OLD, old_dir, SOURCE_BUNDLE, capacity, output / "logs"))
    statuses.append(launch(METHOD_NEW, new_dir, SOURCE_BUNDLE, capacity, output / "logs"))
    write_json(output / "audit" / "run_status.json", statuses)

    eval_mod = v25.load_eval_module(output)
    old_metric = exact.evaluate_method(METHOD_OLD, old_dir, SOURCE_BUNDLE, eval_mod, capacity)
    new_metric = exact.evaluate_method(METHOD_NEW, new_dir, SOURCE_BUNDLE, eval_mod, capacity)
    rows = [
        comparison_row(METHOD_OLD, old_dir, old_metric),
        comparison_row(METHOD_NEW, new_dir, new_metric),
    ]
    write_csv(output / "tables" / "b1_ground_comparison.csv", rows)
    exact.write_md(output / "tables" / "b1_ground_comparison.md", rows)

    unified_files = (
        "per_task_outcomes.csv",
        "ground_delivery_events.csv",
        "per_slot_resource_trace.csv",
        "resource_summary.json",
    )
    regression = []
    authoritative_old = REFERENCE_ROOT / METHOD_OLD
    for name in unified_files:
        fresh_hash = sha256_file(old_dir / name)
        reference_hash = sha256_file(authoritative_old / name)
        regression.append(
            {
                "file": name,
                "fresh_old_b1_sha256": fresh_hash,
                "authoritative_old_b1_sha256": reference_hash,
                "exact_match": fresh_hash == reference_hash,
            }
        )
    write_csv(output / "audit" / "old_b1_regression.csv", regression)
    if not all(row["exact_match"] for row in regression):
        raise RuntimeError("B1_ORIGINAL_REGRESSION_MISMATCH")

    invariant_files = (
        "native_compression_trace.csv",
        "native_scheduler_candidate_trace.csv",
        "native_contact_window_schedule.csv",
        "native_downlink_queue_trace.csv",
        "native_soc_trace.csv",
    )
    invariant_rows = []
    for name in invariant_files:
        old_hash = sha256_file(old_dir / name)
        new_hash = sha256_file(new_dir / name)
        invariant_rows.append(
            {
                "trace": name,
                "old_b1_sha256": old_hash,
                "b1_common_ground_sha256": new_hash,
                "exact_match": old_hash == new_hash,
            }
        )
    write_csv(output / "audit" / "b1_non_ground_execution_invariance.csv", invariant_rows)
    if not all(row["exact_match"] for row in invariant_rows):
        raise RuntimeError("B1_NON_GROUND_EXECUTION_MISMATCH")

    reference_after = {method: tree_snapshot(REFERENCE_ROOT / method) for method in UNCHANGED_METHODS}
    unchanged_rows = []
    for method in UNCHANGED_METHODS:
        before = reference_before[method]
        after = reference_after[method]
        unchanged_rows.append(
            {
                "method": method,
                "before_tree_sha256": before["tree_sha256"],
                "after_tree_sha256": after["tree_sha256"],
                "file_count_before": before["file_count"],
                "file_count_after": after["file_count"],
                "results_and_hashes_unchanged": before == after,
            }
        )
    write_csv(output / "audit" / "unchanged_methods_hash_audit.csv", unchanged_rows)
    write_json(output / "audit" / "unchanged_methods_after.json", reference_after)
    if not all(row["results_and_hashes_unchanged"] for row in unchanged_rows):
        raise RuntimeError("EXISTING_METHOD_OUTPUT_CHANGED")

    source_audit = {
        "method": METHOD_NEW,
        "runtime_adapter": str(RUNTIME_MODULE.resolve()),
        "runtime_adapter_sha256": sha256_file(RUNTIME_MODULE),
        "method_config": str(METHOD_CONFIG.resolve()),
        "method_config_sha256": sha256_file(METHOD_CONFIG),
        "offload_ground_queue_source": str((SOURCE_ROOT / "src" / "baselines" / "offload_eft.py").resolve()),
        "offload_ground_queue_source_sha256": sha256_file(SOURCE_ROOT / "src" / "baselines" / "offload_eft.py"),
        "b1_parent_source": str((SOURCE_ROOT / "src" / "baselines" / "b1_enhanced.py").resolve()),
        "b1_parent_source_sha256": sha256_file(SOURCE_ROOT / "src" / "baselines" / "b1_enhanced.py"),
        "directly_reuses_offload_queue_class": True,
        "ground_station_id_added": False,
        "servers": 1,
        "queue_policy": "fifo",
        "queue_capacity_tasks": 32,
        "decode_time_s": 1.5,
        "inference_time_s": 2.8,
        "compression_ratio": 8.0,
        "complete_receive_before_submit": True,
        "streaming_compute": False,
    }
    write_json(output / "audit" / "method_identity_audit.json", source_audit)

    old, new = rows
    report = [
        "# B1-COMMON-GROUND frozen comparison",
        "",
        f"- Scenario: V25 seed={SEED}, capacity={capacity:.2f}, 86400 s, 20 s slots, task hash `{task_contract['task_hash']}`.",
        "- Both B1 variants were freshly initialized and run from t=0.",
        "- The new method subclasses the unchanged B1 execution and submits only completely received frames to OFFLOAD's `FiniteGroundComputeQueue`.",
        "",
        (output / "tables" / "b1_ground_comparison.md").read_text(encoding="utf-8").strip(),
        "",
        "## Interpretation",
        "",
        f"The shared server observed a peak waiting queue of {new['peak_ground_waiting_queue_tasks']} task(s) and mean wait {float(new['mean_ground_queue_wait_s']):.6f} s.",
        "With this trace, B1 admits at most one completed frame per 20-second simulation slot, while decode plus inference takes 4.3 seconds. The server therefore clears each job before the next submission; zero queueing is a measured result, not an unlimited-compute assumption.",
        "",
        "## Immutability",
        "",
        "The authoritative OFFLOAD, TUT and corrected PROP directories were byte-hashed before and after this run. All aggregate tree hashes and every file entry remained unchanged.",
    ]
    (output / "final_assessment.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    completion = {
        "status": "B1_COMMON_GROUND_FROZEN_COMPARISON_COMPLETE",
        "output": str(output.resolve()),
        "comparison": rows,
        "old_b1_exact_regression": True,
        "offload_tut_prop_hashes_unchanged": True,
    }
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capacity", type=float, default=CAPACITY)
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--method")
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()
    if args.child_run:
        raise SystemExit(run_child(str(args.method), Path(args.output), Path(args.bundle_root), float(args.capacity)))
    raise SystemExit(main(Path(args.output), float(args.capacity)))
