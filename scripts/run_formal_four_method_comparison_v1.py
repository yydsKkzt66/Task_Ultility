"""Run and summarize the frozen four-method formal comparison.

This orchestration layer does not alter any method.  It verifies the frozen
three-method artifacts, reuses exact-identity seed-2026 TUT artifacts, runs only
missing TUT scenarios, and derives common tables from the shared evaluator
records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping


TUT_ROOT = Path("D:/wt/tut_uoi_rate_lite")
THREE_ROOT = Path("D:/wt/pilot/outputs/_three_method_capacity_robustness_v1")
TREND_ROOT = Path("D:/wt/pilot/outputs/_tut_uoi_product_capacity_trend_seed2026_v1")
DIAG_ROOT = Path("D:/wt/pilot/outputs/_tut_uoi_product_lite_diagnostic_v1")
STAGE_B_ROOT = Path("D:/wt/pilot/outputs/_three_method_layered_latency_seed2026_cap050_v1")
OLD_MULTI = Path("D:/wt/pilot/outputs/_sarfish_thermal_headroom_multiseed/multiscenario_run_summary.csv")
OFFLOAD_FREEZE = Path("D:/wt/pilot/outputs/_offload_vs_ground_24h_pilot_seed2026_cap050_v1")
OUT = Path("D:/wt/pilot/outputs/_formal_four_method_comparison_v1")

sys.path.insert(0, str(TUT_ROOT))

from scripts.run_prop_full_vs_b1_unified_seed2026 import proposed_energy_safety  # noqa: E402
from scripts.run_tut_uoi_rate_lite import (  # noqa: E402
    METHOD as TUT_METHOD,
    build_configs,
    build_decisions,
    build_router,
    digest,
    key_result,
)
from src.common.config import load_yaml  # noqa: E402
from src.evaluation.canonical_sarfish_environment import (  # noqa: E402
    canonical_environment_hash_v2,
    canonical_task_hash_v2,
    canonical_task_rows_v2,
)
from src.evaluation.task_outcome_adapter import adapt_proposed, manifest_rows  # noqa: E402
from src.evaluation.unified_ground_received_accounting import summarize_method  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


SEEDS = (2026, 2027, 2028, 2029, 2030)
CAPACITIES = (1.0, 0.5, 0.35)
METHODS = (
    "B1-COMMON-THERMAL",
    "OFFLOAD-EFT-THERMAL-ALIGNED",
    TUT_METHOD,
    "PROP-HEADROOM",
)
PAIRS = (
    (TUT_METHOD, "PROP-HEADROOM"),
    (TUT_METHOD, "OFFLOAD-EFT-THERMAL-ALIGNED"),
    (TUT_METHOD, "B1-COMMON-THERMAL"),
    ("PROP-HEADROOM", "OFFLOAD-EFT-THERMAL-ALIGNED"),
    ("PROP-HEADROOM", "B1-COMMON-THERMAL"),
)
FINGERPRINT = "07c4c14fa4740a7376df314a6a9ed96eab674bb28722609117e42102b49ffb84"
TUT_GIT_HEAD = "8955c464f4211673c612153ffdb701efb409b573"
FROZEN_HASHES = {
    "src/routers/tut_uoi_rate_router.py": "f538a776b67a05a132cd57f5c71b9e0e692ccc58b768fa14b7918235ce1445ad",
    "configs/methods/tut_uoi_rate_lite.yaml": "8edf869f25f2ac9499787a4d33d3a8e7986614e1b208e7f7a15a43e284b280b6",
    "configs/model_profiles_yolo_inspired_calibrated.yaml": "49c35c971365ed503628aa1b474414946c4c5a16e0cd62d6fa8ff0fceb92db38",
    "configs/payload_profiles_roi_compression_calibrated.yaml": "1c39fa19ca8166fb760ed868dd23216cc2d6d1d4c4b47fe58ebeff2d0c438f88",
    "configs/product_requirement_utility.yaml": "32c3020c507da0c0e112f70d8b54a0742e932a4c93a0263fd5d0806596a6ac27",
    "configs/unified_ground_received_accounting.yaml": "0054884ed88209866d53683b0716c6bf4c5e685f835bcdf78f0de9727631073f",
    "src/evaluation/unified_ground_received_accounting.py": "51c93629f3103ef386d637f3f395638562ef93555bada1b2325033387d6621b1",
    "src/evaluation/task_outcome_adapter.py": "a22829f9b799d3762ffe05232f7301d1f09ffd22fb8ded9a870797636687d28e",
    "src/simulator/step_runner.py": "1774494f0217912659005d15ff605468401fdba84659c55f5a80abe7f4adf8f0",
    "src/link/downlink_queue.py": "cc4fd1707f49fb33e9b0049485f5bc5d8d3d0909d2204dd5f6f33241db54ac84",
}
HIGHER_IS_BETTER = {
    "closure_rate",
    "hv_closure_rate",
    "hv_target_closure_rate",
    "common_net_utility_per_task",
    "utility_per_mb",
    "utility_per_wh",
}
PAIR_METRICS = tuple(sorted(HIGHER_IS_BETTER | {"mean_full_closure_latency_s", "p95_full_closure_latency_s"}))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(math.floor(index))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def cap_tag(capacity: float) -> str:
    return f"cap{int(round(capacity * 100)):03d}"


def scenario_tag(seed: int, capacity: float) -> str:
    return f"seed{seed}_{cap_tag(capacity)}"


def build_scenario_configs(seed: int, capacity: float) -> dict[str, Any]:
    """Apply only seed and the contract-approved capacity multiplier."""
    configs = deepcopy(build_configs())
    ratio = float(capacity) / 0.50
    contact = configs["timeline"]["contact"]
    for key in ("throughput_min_mbps", "throughput_max_mbps"):
        contact[key] = float(contact[key]) * ratio
    configs["downlink_capacity_multiplier"] = float(capacity)
    configs["dataset"]["seed"] = int(seed)
    configs["task_stream"]["seed"] = int(seed)
    return configs


def validate_scenario(configs: dict[str, Any], timeline: list[Any]) -> dict[str, Any]:
    """Use the canonical multi-seed hashes without the seed-2026 task-count gate."""
    rows = canonical_task_rows_v2(timeline)
    task_ids = [str(row["task_id"]) for row in rows]
    if not rows or len(task_ids) != len(set(task_ids)):
        raise ValueError("Scenario tasks must be non-empty and unique")
    return {
        "task_count": len(rows),
        "task_manifest_hash_v2": canonical_task_hash_v2(rows),
        "environment_hash_v2": canonical_environment_hash_v2(configs, timeline),
        "rows": rows,
    }


def run_raw(configs: dict[str, Any], timeline: list[Any]) -> dict[str, Any]:
    env = SimulationEnvironment(configs)
    runner = env.build_episode_runner(timeline)
    result = runner.run(build_router(configs))
    return {
        "result": result,
        "runner": runner,
        "decisions": build_decisions(result.records, list(runner.last_candidate_action_logs)),
        "slot_rows": [asdict(record) for record in result.records],
        "candidate_actions": list(runner.last_candidate_action_logs),
        "scheduling": list(runner.last_queue_priority_logs),
    }


def run_full(seed: int, capacity: float) -> dict[str, Any]:
    configs = build_scenario_configs(seed, capacity)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    validation = validate_scenario(configs, timeline)
    raw = run_raw(configs, timeline)
    accounting = load_yaml(TUT_ROOT / "configs/unified_ground_received_accounting.yaml")
    utility = load_yaml(TUT_ROOT / "configs/product_requirement_utility.yaml")
    manifest = manifest_rows(timeline, validation["task_manifest_hash_v2"])
    task_rows = adapt_proposed(
        manifest,
        raw["result"].records,
        raw["runner"].last_payload_lifecycle_current_records,
        TUT_METHOD,
        "tut_inspired_one_shot_product",
        seed,
        accounting,
        utility,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )
    energy, safety = proposed_energy_safety(raw["result"], raw["result"].records, configs, accounting)
    common = summarize_method(
        task_rows,
        energy,
        safety,
        {"internal_utility": raw["result"].summary.get("net_system_utility", "")},
        accounting,
    )
    return {
        **raw,
        "configs": configs,
        "timeline": timeline,
        "validation": validation,
        "task_rows": task_rows,
        "common": common,
    }


def summarize_tut_run(seed: int, capacity: float, run: dict[str, Any]) -> dict[str, Any]:
    rows = run["task_rows"]
    closed = [row for row in rows if truth(row.get("requirement_satisfied"))]
    hv = [row for row in rows if str(row.get("value_level", "")).lower() in {"high", "urgent"}]
    hvt = [row for row in rows if str(row.get("value_level", "")).lower() == "urgent"]
    hv_closed = [row for row in closed if str(row.get("value_level", "")).lower() in {"high", "urgent"}]
    hvt_closed = [row for row in closed if str(row.get("value_level", "")).lower() == "urgent"]
    latencies = [f(row["requirement_closure_time_s"]) - f(row["capture_time_s"]) for row in closed]
    slot = run["slot_rows"]
    net = f(run["common"].get("common_ground_net_utility"))
    transmitted = sum(f(row.get("S_down_total_mb")) for row in slot)
    energy_wh = sum(f(row.get("compute_energy_j")) + f(row.get("comm_energy_j")) for row in slot) / 3600.0
    counts = Counter(row["rate_action"] for row in run["decisions"])
    changed_slots = len(
        {
            int(f(row.get("slot")))
            for row in run["scheduling"]
            if truth(row.get("selected_for_downlink")) and truth(row.get("uoi_priority_changed_order"))
        }
    )
    return {
        "method": TUT_METHOD,
        "seed": seed,
        "capacity": capacity,
        "horizon_s": 86400,
        "task_hash": run["validation"]["task_manifest_hash_v2"],
        "environment_hash": run["validation"]["environment_hash_v2"],
        "benchmark_fingerprint": FINGERPRINT,
        "task_count": len(rows),
        "LOW_RATE_count": counts["LOW_RATE"],
        "MEDIUM_RATE_count": counts["MEDIUM_RATE"],
        "HIGH_RATE_count": counts["HIGH_RATE"],
        "DEFER_count": counts["DEFER"],
        "closure_count": len(closed),
        "closure_rate": len(closed) / max(len(rows), 1),
        "hv_count": len(hv),
        "hv_closure_count": len(hv_closed),
        "hv_closure_rate": len(hv_closed) / max(len(hv), 1),
        "hv_target_count": len(hvt),
        "hv_target_closure_count": len(hvt_closed),
        "hv_target_closure_rate": len(hvt_closed) / max(len(hvt), 1),
        "common_gross_utility": f(run["common"].get("common_ground_gross_utility")),
        "common_net_utility": net,
        "common_net_utility_per_task": net / max(len(rows), 1),
        "transmitted_mb": transmitted,
        "utility_per_mb": net / max(transmitted, 1.0e-12),
        "controllable_satellite_energy_wh": energy_wh,
        "utility_per_wh": net / max(energy_wh, 1.0e-12),
        "mean_full_closure_latency_s": statistics.fmean(latencies) if latencies else 0.0,
        "p95_full_closure_latency_s": percentile(latencies, 0.95),
        "maximum_downlink_queue_mb": max(f(row.get("queue_next_mb")) for row in slot),
        "minimum_soc": min(f(row.get("battery_next_soc")) for row in slot),
        "final_soc": f(slot[-1].get("battery_next_soc")),
        "maximum_thermal": max(f(row.get("thermal_next")) for row in slot),
        "thermal_violation_count": sum(f(row.get("thermal_next")) > 0.92 + 1.0e-9 for row in slot),
        "soc_violation_count": sum(f(row.get("battery_next_soc")) < 0.55 - 1.0e-9 for row in slot),
        "uoi_density_changed_send_order_slots": changed_slots,
    }


def save_tut_run(seed: int, capacity: float, run: dict[str, Any]) -> None:
    directory = OUT / "runs" / scenario_tag(seed, capacity) / TUT_METHOD
    directory.mkdir(parents=True, exist_ok=False)
    summary = summarize_tut_run(seed, capacity, run)
    write_csv(directory / "per_task_results.csv", run["task_rows"])
    write_csv(directory / "decisions.csv", run["decisions"])
    write_csv(directory / "per_slot_resource_trace.csv", run["slot_rows"])
    write_csv(directory / "candidate_action_audit.csv", run["candidate_actions"])
    write_json(directory / "summary.json", summary)
    write_json(
        directory / "resolved_config.json",
        {
            "seed": seed,
            "capacity": capacity,
            "horizon_s": 86400,
            "task_count": summary["task_count"],
            "task_hash": summary["task_hash"],
            "environment_hash": summary["environment_hash"],
            "benchmark_fingerprint": FINGERPRINT,
            "frozen_file_hashes": FROZEN_HASHES,
        },
    )


def manifest_by_seed() -> dict[int, dict[str, str]]:
    path = TUT_ROOT / "manifests/scenario_manifest_v1.csv"
    return {int(row["seed"]): row for row in read_csv(path)}


def current_git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=TUT_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


def normalized_task_view(method: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    if method == "OFFLOAD-EFT-THERMAL-ALIGNED":
        return (
            str(row.get("task_id")),
            round(f(row.get("arrival_s", row.get("arrival_time_s"))), 9),
            round(f(row.get("deadline_s", row.get("deadline_time_s"))), 9),
            str(row.get("value_level", "")).lower(),
        )
    return (
        str(row.get("task_id")),
        round(f(row.get("arrival_time_s", row.get("capture_time_s"))), 9),
        round(f(row.get("deadline_time_s")), 9),
        str(row.get("value_level", "")).lower(),
    )


def reused_tut_dir(capacity: float) -> Path:
    if abs(capacity - 0.50) < 1.0e-12:
        return DIAG_ROOT / "corrected_pilot"
    return TREND_ROOT / "runs" / cap_tag(capacity)


def formal_tut_dir(seed: int, capacity: float) -> Path:
    if seed == 2026:
        return reused_tut_dir(capacity)
    return OUT / "runs" / scenario_tag(seed, capacity) / TUT_METHOD


def preflight() -> None:
    if OUT.exists() and ((OUT / "runs").exists() or (OUT / "final_report.md").exists()):
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    head = current_git_head()
    checks["tut_git_head_match"] = head == TUT_GIT_HEAD
    current_hashes = {rel: sha256(TUT_ROOT / rel) for rel in FROZEN_HASHES}
    for rel, expected in FROZEN_HASHES.items():
        checks[f"frozen_sha256::{rel}"] = current_hashes[rel] == expected

    fingerprint_doc = json.loads(
        (TUT_ROOT / "manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8")
    )
    checks["benchmark_fingerprint_match"] = fingerprint_doc["benchmark_environment_fingerprint"] == FINGERPRINT
    checks["product_profile_fingerprint_match"] = (
        current_hashes["configs/payload_profiles_roi_compression_calibrated.yaml"]
        == fingerprint_doc["components"]["product_config"]
    )
    checks["evaluator_config_fingerprint_match"] = (
        current_hashes["configs/unified_ground_received_accounting.yaml"]
        == fingerprint_doc["components"]["common_evaluator_config"]
    )
    checks["evaluator_source_fingerprint_match"] = (
        current_hashes["src/evaluation/unified_ground_received_accounting.py"]
        == fingerprint_doc["components"]["common_evaluator_source"]
    )

    inventory = json.loads((THREE_ROOT / "artifact_inventory.json").read_text(encoding="utf-8"))
    inventory_map = {row["relative_path"]: row["sha256"] for row in inventory["artifacts"]}
    verified_artifacts = 0
    task_manifest = manifest_by_seed()
    scenario_details = []
    for seed in SEEDS:
        for capacity in CAPACITIES:
            tag = scenario_tag(seed, capacity)
            scenario = THREE_ROOT / "runs" / tag
            resolved_path = scenario / "resolved_config.json"
            relative_resolved = resolved_path.relative_to(THREE_ROOT).as_posix()
            checks[f"artifact::{relative_resolved}"] = (
                inventory_map.get(relative_resolved) == sha256(resolved_path)
            )
            verified_artifacts += 1
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            configs = build_scenario_configs(seed, capacity)
            timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
            validation = validate_scenario(configs, timeline)
            checks[f"{tag}::seed_capacity_horizon"] = (
                int(resolved["seed"]) == seed
                and abs(f(resolved["capacity"]) - capacity) < 1.0e-12
                and int(resolved["horizon_s"]) == 86400
            )
            checks[f"{tag}::task_hash_manifest"] = (
                resolved["task_hash"] == validation["task_manifest_hash_v2"] == task_manifest[seed]["task_hash"]
            )
            checks[f"{tag}::environment_hash"] = resolved["environment_hash"] == validation["environment_hash_v2"]
            normalized_views = {}
            for method in METHODS:
                if method == TUT_METHOD:
                    if seed != 2026:
                        continue
                    method_dir = reused_tut_dir(capacity)
                else:
                    method_dir = scenario / method
                    for name in ("summary.json", "per_task_results.csv"):
                        artifact = method_dir / name
                        relative = artifact.relative_to(THREE_ROOT).as_posix()
                        checks[f"artifact::{relative}"] = inventory_map.get(relative) == sha256(artifact)
                        verified_artifacts += 1
                summary = json.loads((method_dir / "summary.json").read_text(encoding="utf-8"))
                checks[f"{tag}::{method}::identity"] = (
                    int(summary["seed"]) == seed
                    and abs(f(summary["capacity"]) - capacity) < 1.0e-12
                    and summary["task_hash"] == resolved["task_hash"]
                    and summary["environment_hash"] == resolved["environment_hash"]
                    and int(summary["task_count"]) == int(task_manifest[seed]["task_count"])
                )
                rows = read_csv(method_dir / "per_task_results.csv")
                normalized_views[method] = sorted(normalized_task_view(method, row) for row in rows)
            reference = normalized_views["B1-COMMON-THERMAL"]
            for method, view in normalized_views.items():
                checks[f"{tag}::{method}::task_fields_match"] = view == reference
            scenario_details.append(
                {
                    "seed": seed,
                    "capacity": capacity,
                    "task_count": len(reference),
                    "task_hash": resolved["task_hash"],
                    "environment_hash": resolved["environment_hash"],
                    "tut_status": "reused_exact_identity" if seed == 2026 else "missing_to_run",
                }
            )

    trend_config = json.loads((TREND_ROOT / "resolved_capacity_config.json").read_text(encoding="utf-8"))
    for rel, expected in trend_config["frozen_file_hashes"].items():
        checks[f"seed2026_reuse_frozen_sha::{rel}"] = current_hashes[rel] == expected
    checks["seed2026_cap050_full_repeat_exact"] = json.loads(
        (TREND_ROOT / "repeatability_check.json").read_text(encoding="utf-8")
    )["capacity_0.50_full_pilot"]["all_exact"]

    all_pass = all(checks.values())
    details.update(
        {
            "status": "PASS" if all_pass else "FAIL",
            "all_pass": all_pass,
            "checks": checks,
            "verified_frozen_artifact_count": verified_artifacts,
            "scenarios": scenario_details,
        }
    )
    write_json(OUT / "hash_validation.json", details)
    write_json(
        OUT / "frozen_method_manifest.json",
        {
            "methods": list(METHODS),
            "seeds": list(SEEDS),
            "capacities": list(CAPACITIES),
            "horizon_s": 86400,
            "tut_git_head": head,
            "tut_worktree_has_uncommitted_scientific_files": True,
            "tut_exact_file_hashes": current_hashes,
            "fixed_model": "balanced",
            "fixed_model_profile": "small",
            "one_shot_product_mapping": {
                "LOW_RATE": "metadata",
                "MEDIUM_RATE": "roi",
                "HIGH_RATE": "roi_context",
                "DEFER": "defer",
            },
            "uoi_weights": {"delivery_latency": 0.5, "information_gap": 0.5, "V_uoi": 1.0},
            "normalization": {"energy_ref_j": 100.0, "bytes_ref_mb": 2.70, "latency": "predicted_latency/deadline"},
            "benchmark_fingerprint": FINGERPRINT,
            "three_method_artifact_inventory_sha256": sha256(THREE_ROOT / "artifact_inventory.json"),
        },
    )
    report = f"""# Formal four-method comparison preflight

_Frozen five-seed, three-capacity comparison; preflight performed before any missing TUT formal run._

---

## ✅ Outcome

`{'PASS' if all_pass else 'FAIL'}`

The preflight validated `{verified_artifacts}` frozen three-method artifacts against the authoritative inventory, reconstructed all 15 shared scenarios, and matched task fields across available methods. Seed 2026 has three exact-identity TUT artifacts available for reuse; the remaining 12 TUT scenarios are missing.

## ⚙️ Frozen TUT identity

| Item | Frozen value |
| --- | --- |
| Git HEAD | `{head}` |
| Router SHA-256 | `{current_hashes['src/routers/tut_uoi_rate_router.py']}` |
| Method config SHA-256 | `{current_hashes['configs/methods/tut_uoi_rate_lite.yaml']}` |
| Product profile SHA-256 | `{current_hashes['configs/payload_profiles_roi_compression_calibrated.yaml']}` |
| Requirement utility SHA-256 | `{current_hashes['configs/product_requirement_utility.yaml']}` |
| Common evaluator SHA-256 | `{current_hashes['configs/unified_ground_received_accounting.yaml']}` |
| Fixed model | `balanced` (`small` profile) |

## 🔒 Scope

Only missing `TUT-UOI-PRODUCT-LITE` scenarios may run. B1, OFFLOAD, and PROP remain read-only. The orchestration changes no formula, weight, profile, evaluator, task, contact, SOC, thermal parameter, or method implementation.
"""
    (OUT / "preflight_report.md").write_text(report, encoding="utf-8")
    write_json(
        OUT / "rollback_manifest.json",
        {
            "new_orchestration_file": "scripts/run_formal_four_method_comparison_v1.py",
            "new_output_directory": str(OUT),
            "modified_method_files": [],
            "modified_shared_environment_files": [],
            "modified_frozen_outputs": [],
            "rollback": "Remove only the new output directory and this orchestration file if desired.",
        },
    )
    if not all_pass:
        raise RuntimeError("FROZEN_OUTPUT_HASH_MISMATCH")


def smoke() -> None:
    validation = json.loads((OUT / "hash_validation.json").read_text(encoding="utf-8"))
    if not validation["all_pass"]:
        raise RuntimeError("Preflight did not pass")
    configs = build_scenario_configs(2027, 0.35)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()[:900]
    first = run_raw(configs, timeline)
    second = run_raw(configs, timeline)
    first_key = {
        "decisions": [
            (row["task_id"], row["rate_action"], round(f(row["generated_mb"]), 12))
            for row in first["decisions"]
        ],
        "slots": [
            (
                int(f(row["slot"])),
                str(row.get("task_id", "")),
                round(f(row.get("battery_next_soc")), 12),
                round(f(row.get("thermal_next")), 12),
                round(f(row.get("queue_next_mb")), 12),
            )
            for row in first["slot_rows"]
        ],
    }
    second_key = {
        "decisions": [
            (row["task_id"], row["rate_action"], round(f(row["generated_mb"]), 12))
            for row in second["decisions"]
        ],
        "slots": [
            (
                int(f(row["slot"])),
                str(row.get("task_id", "")),
                round(f(row.get("battery_next_soc")), 12),
                round(f(row.get("thermal_next")), 12),
                round(f(row.get("queue_next_mb")), 12),
            )
            for row in second["slot_rows"]
        ],
    }
    rates = {row["rate_action"] for row in first["decisions"] if row["rate_action"] != "DEFER"}
    checks = {
        "repeat_exact": digest(first_key) == digest(second_key),
        "at_least_two_non_defer_products": len(rates) >= 2,
        "one_decision_per_task": len(first["decisions"]) == len({row["task_id"] for row in first["decisions"]}),
        "thermal_safe": max(f(row.get("thermal_next")) for row in first["slot_rows"]) <= 0.92 + 1.0e-9,
        "soc_safe": min(f(row.get("battery_next_soc")) for row in first["slot_rows"]) >= 0.55 - 1.0e-9,
    }
    write_json(
        OUT / "smoke_check.json",
        {
            "seed": 2027,
            "capacity": 0.35,
            "slots": 900,
            "checks": checks,
            "all_pass": all(checks.values()),
            "first_sha256": digest(first_key),
            "repeat_sha256": digest(second_key),
            "non_defer_products": sorted(rates),
        },
    )
    if not all(checks.values()):
        raise RuntimeError("TUT smoke failed")


def run_seed(seed: int) -> None:
    if seed not in SEEDS or seed == 2026:
        raise ValueError("Only missing formal seeds 2027-2030 are runnable")
    smoke_doc = json.loads((OUT / "smoke_check.json").read_text(encoding="utf-8"))
    if not smoke_doc["all_pass"]:
        raise RuntimeError("Smoke did not pass")
    completed = []
    for capacity in CAPACITIES:
        directory = OUT / "runs" / scenario_tag(seed, capacity) / TUT_METHOD
        if directory.exists():
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            reference = json.loads(
                (THREE_ROOT / "runs" / scenario_tag(seed, capacity) / "resolved_config.json").read_text(encoding="utf-8")
            )
            if summary["task_hash"] != reference["task_hash"] or summary["environment_hash"] != reference["environment_hash"]:
                raise RuntimeError(f"Existing partial TUT identity mismatch: {directory}")
            completed.append({"capacity": capacity, "status": "existing_verified"})
            continue
        run = run_full(seed, capacity)
        reference = json.loads(
            (THREE_ROOT / "runs" / scenario_tag(seed, capacity) / "resolved_config.json").read_text(encoding="utf-8")
        )
        if (
            run["validation"]["task_manifest_hash_v2"] != reference["task_hash"]
            or run["validation"]["environment_hash_v2"] != reference["environment_hash"]
        ):
            raise RuntimeError(f"FAIRNESS_VALIDATION_FAILED: {seed} {capacity}")
        save_tut_run(seed, capacity, run)
        summary = summarize_tut_run(seed, capacity, run)
        if summary["thermal_violation_count"] or summary["soc_violation_count"]:
            raise RuntimeError(f"SAFETY_VALIDATION_FAILED: {seed} {capacity}")
        completed.append({"capacity": capacity, "status": "new_run", "summary": summary})
    write_json(OUT / "runs" / f"seed{seed}_completion.json", {"seed": seed, "runs": completed})
    print(json.dumps({"status": "SEED_COMPLETE", "seed": seed, "capacities": len(completed)}))


def old_multi_index() -> dict[tuple[int, float, str], dict[str, str]]:
    return {
        (int(row["seed"]), f(row["capacity_multiplier"]), row["method"]): row
        for row in read_csv(OLD_MULTI)
    }


def final_soc_from_trace(seed: int, capacity: float, method: str, summary: Mapping[str, Any]) -> float:
    if "final_soc" in summary:
        return f(summary["final_soc"])
    scenario = THREE_ROOT / "runs" / scenario_tag(seed, capacity) / method
    trace = scenario / "per_slot_resource_trace.csv"
    if trace.exists():
        rows = read_csv(trace)
        if method == "B1-COMMON-THERMAL":
            return f(rows[-1].get("battery_soc", rows[-1].get("SOC")))
        if method == "OFFLOAD-EFT-THERMAL-ALIGNED":
            return f(rows[-1].get("battery_soc"))
        return f(rows[-1].get("battery_next_soc", rows[-1].get("battery_soc")))
    old = old_multi_index().get((seed, capacity, method))
    if old is not None:
        return f(old.get("final_soc"))
    if method == "OFFLOAD-EFT-THERMAL-ALIGNED" and seed == 2026 and abs(capacity - 0.5) < 1.0e-12:
        doc = json.loads((OFFLOAD_FREEZE / "offload_eft_thermal_aligned/summary.json").read_text(encoding="utf-8"))
        return f(doc["final_soc"])
    if method == "PROP-HEADROOM" and seed == 2026 and abs(capacity - 0.5) < 1.0e-12:
        rows = read_csv(STAGE_B_ROOT / "audit/prop_soc_trace.csv")
        return f(rows[-1].get("battery_next_soc", rows[-1].get("battery_soc", rows[-1].get("SOC"))))
    raise RuntimeError(f"Missing final SOC evidence for {seed} {capacity} {method}")


def method_artifacts(seed: int, capacity: float, method: str) -> tuple[dict[str, Any], list[dict[str, str]], Path]:
    if method == TUT_METHOD:
        directory = formal_tut_dir(seed, capacity)
    else:
        directory = THREE_ROOT / "runs" / scenario_tag(seed, capacity) / method
    return (
        json.loads((directory / "summary.json").read_text(encoding="utf-8")),
        read_csv(directory / "per_task_results.csv"),
        directory,
    )


def normalized_outcome(method: str, row: Mapping[str, Any]) -> dict[str, Any]:
    offload = method == "OFFLOAD-EFT-THERMAL-ALIGNED"
    legacy_offload = offload and "closed" in row
    closed = (
        truth(row.get("closed"))
        if legacy_offload
        else truth(row.get("on_time_task_success" if offload else "requirement_satisfied"))
    )
    value_level = str(row.get("value_level", "")).strip().lower()
    hv = truth(row.get("hv")) if legacy_offload else value_level in {"high", "urgent"}
    hvt = truth(row.get("hv_target")) if legacy_offload else value_level == "urgent"
    capture = (
        f(row.get("arrival_s", row.get("arrival_time_s")))
        if offload
        else f(row.get("capture_time_s", row.get("arrival_time_s")))
    )
    deadline = f(row.get("deadline_s", row.get("deadline_time_s"))) if offload else f(row.get("deadline_time_s"))
    closure_time = (
        f(row.get("actual_finish_s", row.get("requirement_closure_time_s")))
        if offload
        else f(row.get("requirement_closure_time_s"))
    )
    gross = f(row.get("gross_utility", row.get("gross_ground_utility"))) if offload else f(row.get("gross_ground_utility"))
    return {
        "task_id": str(row.get("task_id")),
        "closed": closed,
        "hv": hv,
        "hvt": hvt,
        "capture": capture,
        "deadline": deadline,
        "closure_time": closure_time,
        "gross": gross,
    }


def action_distribution(method: str, directory: Path, rows: list[dict[str, str]]) -> dict[str, int]:
    if method == TUT_METHOD:
        return dict(Counter(row["rate_action"] for row in read_csv(directory / "decisions.csv")))
    if method == "OFFLOAD-EFT-THERMAL-ALIGNED":
        return dict(Counter(row.get("selected_path", "") for row in rows))
    return dict(Counter(row.get("transmitted_product_type", "") or "none" for row in rows))


def scenario_metric(seed: int, capacity: float, method: str) -> dict[str, Any]:
    summary, raw_rows, directory = method_artifacts(seed, capacity, method)
    rows = [normalized_outcome(method, row) for row in raw_rows]
    closed = [row for row in rows if row["closed"]]
    hv = [row for row in rows if row["hv"]]
    hvt = [row for row in rows if row["hvt"]]
    hv_closed = [row for row in closed if row["hv"]]
    hvt_closed = [row for row in closed if row["hvt"]]
    gross = sum(row["gross"] for row in rows)
    net = f(summary.get("common_net_utility"))
    shared_cost_per_task = (gross - net) / max(len(rows), 1)
    hv_gross = sum(row["gross"] for row in hv)
    hvt_gross = sum(row["gross"] for row in hvt)
    hv_net = hv_gross - shared_cost_per_task * len(hv)
    hvt_net = hvt_gross - shared_cost_per_task * len(hvt)
    latencies = [row["closure_time"] - row["capture"] for row in closed]
    hv_latencies = [row["closure_time"] - row["capture"] for row in hv_closed]
    hvt_latencies = [row["closure_time"] - row["capture"] for row in hvt_closed]
    capped = [
        min(row["closure_time"] - row["capture"], row["deadline"] - row["capture"])
        if row["closed"]
        else row["deadline"] - row["capture"]
        for row in rows
    ]
    tx = f(summary.get("transmitted_mb"))
    energy = f(summary.get("controllable_satellite_energy_wh"))
    distribution = action_distribution(method, directory, raw_rows)
    return {
        "seed": seed,
        "capacity": capacity,
        "method": method,
        "task_hash": summary["task_hash"],
        "environment_hash": summary["environment_hash"],
        "benchmark_fingerprint": FINGERPRINT,
        "task_count": len(rows),
        "closure_count": len(closed),
        "closure_rate": len(closed) / max(len(rows), 1),
        "hv_task_count": len(hv),
        "hv_closure_count": len(hv_closed),
        "hv_closure_rate": len(hv_closed) / max(len(hv), 1),
        "hv_target_task_count": len(hvt),
        "hv_target_closure_count": len(hvt_closed),
        "hv_target_closure_rate": len(hvt_closed) / max(len(hvt), 1),
        "common_gross_utility": gross,
        "common_net_utility": net,
        "common_net_utility_per_task": net / max(len(rows), 1),
        "hv_gross_utility": hv_gross,
        "hv_net_utility": hv_net,
        "hv_net_utility_per_hv_task": hv_net / max(len(hv), 1),
        "hv_target_gross_utility": hvt_gross,
        "hv_target_net_utility": hvt_net,
        "hv_target_net_utility_per_hv_target_task": hvt_net / max(len(hvt), 1),
        "shared_common_cost_per_task": shared_cost_per_task,
        "transmitted_mb": tx,
        "utility_per_mb": net / max(tx, 1.0e-12),
        "controllable_satellite_energy_wh": energy,
        "utility_per_wh": net / max(energy, 1.0e-12),
        "mean_full_closure_latency_s": statistics.fmean(latencies) if latencies else 0.0,
        "p95_full_closure_latency_s": percentile(latencies, 0.95),
        "hv_mean_full_closure_latency_s": statistics.fmean(hv_latencies) if hv_latencies else 0.0,
        "hv_p95_full_closure_latency_s": percentile(hv_latencies, 0.95),
        "hv_target_mean_full_closure_latency_s": statistics.fmean(hvt_latencies) if hvt_latencies else 0.0,
        "hv_target_p95_full_closure_latency_s": percentile(hvt_latencies, 0.95),
        "deadline_capped_all_task_mean_latency_s": statistics.fmean(capped),
        "deadline_capped_all_task_p95_latency_s": percentile(capped, 0.95),
        "maximum_downlink_queue_mb": f(summary.get("maximum_downlink_queue_mb")),
        "minimum_soc": f(summary.get("minimum_soc")),
        "final_soc": final_soc_from_trace(seed, capacity, method, summary),
        "maximum_thermal": f(summary.get("maximum_thermal")),
        "thermal_violation_count": int(f(summary.get("thermal_violation_count"))),
        "soc_violation_count": int(f(summary.get("soc_violation_count"))),
        "action_product_distribution": json.dumps(distribution, sort_keys=True, separators=(",", ":")),
        "LOW_RATE_count": distribution.get("LOW_RATE", 0),
        "MEDIUM_RATE_count": distribution.get("MEDIUM_RATE", 0),
        "HIGH_RATE_count": distribution.get("HIGH_RATE", 0),
        "DEFER_count": distribution.get("DEFER", 0),
    }


def capacity_summaries(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    mean_metrics = (
        "closure_rate",
        "hv_closure_rate",
        "hv_target_closure_rate",
        "common_net_utility_per_task",
        "utility_per_mb",
        "utility_per_wh",
        "mean_full_closure_latency_s",
        "p95_full_closure_latency_s",
        "maximum_downlink_queue_mb",
        "minimum_soc",
        "final_soc",
        "maximum_thermal",
    )
    for capacity in CAPACITIES:
        for method in METHODS:
            group = [row for row in scenarios if row["capacity"] == capacity and row["method"] == method]
            total_tasks = sum(int(row["task_count"]) for row in group)
            total_hv = sum(int(row["hv_task_count"]) for row in group)
            total_hvt = sum(int(row["hv_target_task_count"]) for row in group)
            total_net = sum(f(row["common_net_utility"]) for row in group)
            total_mb = sum(f(row["transmitted_mb"]) for row in group)
            total_wh = sum(f(row["controllable_satellite_energy_wh"]) for row in group)
            output: dict[str, Any] = {
                "capacity": capacity,
                "method": method,
                "seed_count": len(group),
                "pooled_task_count": total_tasks,
                "pooled_closure_count": sum(int(row["closure_count"]) for row in group),
                "pooled_closure_rate": sum(int(row["closure_count"]) for row in group) / max(total_tasks, 1),
                "pooled_hv_task_count": total_hv,
                "pooled_hv_closure_count": sum(int(row["hv_closure_count"]) for row in group),
                "pooled_hv_closure_rate": sum(int(row["hv_closure_count"]) for row in group) / max(total_hv, 1),
                "pooled_hv_target_task_count": total_hvt,
                "pooled_hv_target_closure_count": sum(int(row["hv_target_closure_count"]) for row in group),
                "pooled_hv_target_closure_rate": sum(int(row["hv_target_closure_count"]) for row in group) / max(total_hvt, 1),
                "total_common_gross_utility": sum(f(row["common_gross_utility"]) for row in group),
                "total_common_net_utility": total_net,
                "pooled_common_net_utility_per_task": total_net / max(total_tasks, 1),
                "total_transmitted_mb": total_mb,
                "pooled_utility_per_mb": total_net / max(total_mb, 1.0e-12),
                "total_controllable_satellite_energy_wh": total_wh,
                "pooled_utility_per_wh": total_net / max(total_wh, 1.0e-12),
                "thermal_violation_count": sum(int(row["thermal_violation_count"]) for row in group),
                "soc_violation_count": sum(int(row["soc_violation_count"]) for row in group),
            }
            for metric in mean_metrics:
                values = [f(row[metric]) for row in group]
                output[f"mean_{metric}"] = statistics.fmean(values)
                output[f"std_{metric}"] = sample_std(values)
            rows.append(output)
    return rows


def retention_rows(scenarios: list[dict[str, Any]], capacities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cap = {(row["method"], f(row["capacity"])): row for row in capacities}
    by_scenario = {(row["method"], int(row["seed"]), f(row["capacity"])): row for row in scenarios}
    mapping = {
        "closure_rate": "pooled_closure_rate",
        "hv_closure_rate": "pooled_hv_closure_rate",
        "hv_target_closure_rate": "pooled_hv_target_closure_rate",
        "common_net_utility_per_task": "pooled_common_net_utility_per_task",
        "utility_per_mb": "pooled_utility_per_mb",
        "utility_per_wh": "pooled_utility_per_wh",
    }
    output = []
    for method in METHODS:
        base = by_cap[(method, 1.0)]
        for capacity in (0.5, 0.35):
            current = by_cap[(method, capacity)]
            for metric, pooled_metric in mapping.items():
                seed_ratios = [
                    f(by_scenario[(method, seed, capacity)][metric])
                    / max(f(by_scenario[(method, seed, 1.0)][metric]), 1.0e-12)
                    for seed in SEEDS
                ]
                output.append(
                    {
                        "method": method,
                        "capacity": capacity,
                        "metric": metric,
                        "pooled_retention": f(current[pooled_metric]) / max(f(base[pooled_metric]), 1.0e-12),
                        "mean_paired_seed_retention": statistics.fmean(seed_ratios),
                        "std_paired_seed_retention": sample_std(seed_ratios),
                    }
                )
    return output


def paired_rows(scenarios: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {(int(row["seed"]), f(row["capacity"]), row["method"]): row for row in scenarios}
    differences = []
    for seed in SEEDS:
        for capacity in CAPACITIES:
            for left, right in PAIRS:
                for metric in PAIR_METRICS:
                    left_value = f(index[(seed, capacity, left)][metric])
                    right_value = f(index[(seed, capacity, right)][metric])
                    differences.append(
                        {
                            "seed": seed,
                            "capacity": capacity,
                            "comparison": f"{left} vs {right}",
                            "metric": metric,
                            "left_method": left,
                            "right_method": right,
                            "left_value": left_value,
                            "right_value": right_value,
                            "left_minus_right": left_value - right_value,
                            "winner": left if (left_value > right_value if metric in HIGHER_IS_BETTER else left_value < right_value) else (right if left_value != right_value else "TIE"),
                        }
                    )
    wins = []
    for capacity in CAPACITIES:
        for left, right in PAIRS:
            comparison = f"{left} vs {right}"
            for metric in PAIR_METRICS:
                group = [
                    row
                    for row in differences
                    if row["capacity"] == capacity and row["comparison"] == comparison and row["metric"] == metric
                ]
                wins.append(
                    {
                        "capacity": capacity,
                        "comparison": comparison,
                        "metric": metric,
                        "left_method": left,
                        "right_method": right,
                        "left_wins": sum(row["winner"] == left for row in group),
                        "right_wins": sum(row["winner"] == right for row in group),
                        "ties": sum(row["winner"] == "TIE" for row in group),
                        "seed_count": len(group),
                    }
                )
    return differences, wins


def prop_layered_latency() -> list[dict[str, Any]]:
    output = []
    for seed in SEEDS:
        for capacity in CAPACITIES:
            scenario = THREE_ROOT / "runs" / scenario_tag(seed, capacity)
            outcomes = read_csv(scenario / "PROP-HEADROOM/per_task_results.csv")
            capture = {row["task_id"]: f(row.get("capture_time_s", row.get("arrival_time_s"))) for row in outcomes}
            event_path = scenario / "PROP-HEADROOM/prop_delivery_events.csv"
            if event_path.exists():
                events = read_csv(event_path)
                source = "formal_prop_delivery_events"
            elif seed == 2026 and abs(capacity - 0.5) < 1.0e-12:
                events = read_csv(STAGE_B_ROOT / "prop_layered_events.csv")
                source = "frozen_stage_b_prop_layered_events"
            else:
                events = []
                source = "missing"
            by_stage: dict[str, dict[str, float]] = defaultdict(dict)
            for row in events:
                product = str(row.get("product_type", "")).strip()
                task_id = str(row.get("task_id", ""))
                ground_time = f(row.get("ground_available_time_s"), -1.0)
                if product not in {"metadata", "roi", "roi_context"} or task_id not in capture or ground_time < 0:
                    continue
                latency = ground_time - capture[task_id]
                previous = by_stage[product].get(task_id)
                if previous is None or latency < previous:
                    by_stage[product][task_id] = latency
            for product, label in (("metadata", "alert"), ("roi", "ROI"), ("roi_context", "evidence")):
                values = list(by_stage[product].values())
                output.append(
                    {
                        "seed": seed,
                        "capacity": capacity,
                        "stage": label,
                        "product_type": product,
                        "arrival_count": len(values),
                        "mean_latency_s": statistics.fmean(values) if values else 0.0,
                        "p95_latency_s": percentile(values, 0.95),
                        "source": source,
                        "metric_scope": "supplementary_stage_arrival_not_full_closure",
                    }
                )
    return output


def analyze() -> None:
    validation = json.loads((OUT / "hash_validation.json").read_text(encoding="utf-8"))
    if not validation["all_pass"]:
        raise RuntimeError("FROZEN_OUTPUT_HASH_MISMATCH")
    for seed in SEEDS[1:]:
        if not (OUT / "runs" / f"seed{seed}_completion.json").exists():
            raise RuntimeError("TUT_FORMAL_RUN_INCOMPLETE")
    scenarios = [scenario_metric(seed, capacity, method) for seed in SEEDS for capacity in CAPACITIES for method in METHODS]
    identity_groups = defaultdict(list)
    for row in scenarios:
        identity_groups[(row["seed"], row["capacity"])].append(row)
    fairness = all(
        len({row["task_hash"] for row in group}) == 1
        and len({row["environment_hash"] for row in group}) == 1
        and len({row["task_count"] for row in group}) == 1
        for group in identity_groups.values()
    )
    if not fairness:
        raise RuntimeError("FAIRNESS_VALIDATION_FAILED")
    safety_pass = all(row["thermal_violation_count"] == 0 and row["soc_violation_count"] == 0 for row in scenarios)
    if not safety_pass:
        raise RuntimeError("SAFETY_VALIDATION_FAILED")

    capacities = capacity_summaries(scenarios)
    retentions = retention_rows(scenarios, capacities)
    paired, wins = paired_rows(scenarios)
    layered = prop_layered_latency()
    write_csv(OUT / "scenario_level_metrics.csv", scenarios)
    write_csv(OUT / "capacity_summary.csv", capacities)
    write_csv(
        OUT / "high_value_utility_summary.csv",
        [
            {key: row[key] for key in (
                "seed", "capacity", "method", "hv_task_count", "hv_closure_count", "hv_closure_rate",
                "hv_gross_utility", "hv_net_utility", "hv_net_utility_per_hv_task", "hv_target_task_count",
                "hv_target_closure_count", "hv_target_closure_rate", "hv_target_gross_utility",
                "hv_target_net_utility", "hv_target_net_utility_per_hv_target_task", "shared_common_cost_per_task",
            )}
            for row in scenarios
        ],
    )
    write_csv(
        OUT / "latency_summary.csv",
        [
            {key: row[key] for key in (
                "seed", "capacity", "method", "closure_count", "mean_full_closure_latency_s",
                "p95_full_closure_latency_s", "hv_closure_count", "hv_mean_full_closure_latency_s",
                "hv_p95_full_closure_latency_s", "hv_target_closure_count", "hv_target_mean_full_closure_latency_s",
                "hv_target_p95_full_closure_latency_s", "deadline_capped_all_task_mean_latency_s",
                "deadline_capped_all_task_p95_latency_s",
            )}
            for row in scenarios
        ],
    )
    write_csv(
        OUT / "efficiency_summary.csv",
        [
            {key: row[key] for key in (
                "seed", "capacity", "method", "common_net_utility", "transmitted_mb", "utility_per_mb",
                "controllable_satellite_energy_wh", "utility_per_wh", "maximum_downlink_queue_mb",
            )}
            for row in scenarios
        ],
    )
    write_csv(OUT / "retention_summary.csv", retentions)
    write_csv(OUT / "paired_method_differences.csv", paired)
    write_csv(OUT / "seed_win_counts.csv", wins)
    write_csv(OUT / "supplementary_prop_layered_latency.csv", layered)
    write_csv(
        OUT / "final_comparison_table.csv",
        [
            {key: row[key] for key in (
                "capacity", "method", "pooled_task_count", "pooled_closure_count", "pooled_closure_rate",
                "pooled_hv_closure_rate", "pooled_hv_target_closure_rate", "total_common_net_utility",
                "pooled_common_net_utility_per_task", "total_transmitted_mb", "pooled_utility_per_mb",
                "total_controllable_satellite_energy_wh", "pooled_utility_per_wh",
                "mean_mean_full_closure_latency_s", "mean_p95_full_closure_latency_s",
                "thermal_violation_count", "soc_violation_count",
            )}
            for row in capacities
        ],
    )
    write_json(
        OUT / "safety_summary.json",
        {
            "status": "PASS",
            "all_60_method_scenarios_safe": safety_pass,
            "thermal_hard_limit": 0.92,
            "soc_hard_floor": 0.55,
            "total_thermal_violations": sum(row["thermal_violation_count"] for row in scenarios),
            "total_soc_violations": sum(row["soc_violation_count"] for row in scenarios),
            "maximum_thermal": max(row["maximum_thermal"] for row in scenarios),
            "minimum_soc": min(row["minimum_soc"] for row in scenarios),
            "scenario_checks": [
                {
                    "seed": row["seed"], "capacity": row["capacity"], "method": row["method"],
                    "maximum_thermal": row["maximum_thermal"], "minimum_soc": row["minimum_soc"],
                    "final_soc": row["final_soc"], "thermal_violation_count": row["thermal_violation_count"],
                    "soc_violation_count": row["soc_violation_count"],
                }
                for row in scenarios
            ],
        },
    )

    cap_index = {(row["capacity"], row["method"]): row for row in capacities}
    overall = {}
    for method in METHODS:
        group = [row for row in scenarios if row["method"] == method]
        total_tasks = sum(row["task_count"] for row in group)
        total_hv = sum(row["hv_task_count"] for row in group)
        total_hvt = sum(row["hv_target_task_count"] for row in group)
        net = sum(row["common_net_utility"] for row in group)
        mb = sum(row["transmitted_mb"] for row in group)
        wh = sum(row["controllable_satellite_energy_wh"] for row in group)
        overall[method] = {
            "closure_rate": sum(row["closure_count"] for row in group) / total_tasks,
            "hv_closure_rate": sum(row["hv_closure_count"] for row in group) / total_hv,
            "hvt_closure_rate": sum(row["hv_target_closure_count"] for row in group) / total_hvt,
            "hvt_net_per_task": sum(row["hv_target_net_utility"] for row in group) / total_hvt,
            "net_per_task": net / total_tasks,
            "utility_per_mb": net / mb,
            "utility_per_wh": net / wh,
            "mean_latency": statistics.fmean(row["mean_full_closure_latency_s"] for row in group),
            "p95_latency": statistics.fmean(row["p95_full_closure_latency_s"] for row in group),
        }

    def winner(metric: str, lower: bool = False) -> str:
        return min(overall, key=lambda method: overall[method][metric]) if lower else max(overall, key=lambda method: overall[method][metric])

    retention_best = {}
    for metric in ("closure_rate", "hv_closure_rate", "hv_target_closure_rate", "common_net_utility_per_task", "utility_per_mb", "utility_per_wh"):
        candidates = [row for row in retentions if row["capacity"] == 0.35 and row["metric"] == metric]
        retention_best[metric] = max(candidates, key=lambda row: row["pooled_retention"])["method"]

    tut_overall = overall[TUT_METHOD]
    prop_overall = overall["PROP-HEADROOM"]
    report = f"""# Formal four-method comparison

_Five frozen seeds, capacities 1.00/0.50/0.35, 24-hour horizon; shared tasks, environment, safety model, and common evaluator._

---

## ✅ Final status

`FOUR_METHOD_COMPARISON_READY`

All 15 scenarios × 4 methods passed task/environment/fingerprint identity checks. The 12 missing TUT runs were added, three exact-identity seed-2026 TUT runs were reused, and B1/OFFLOAD/PROP were never rerun. All 60 method-scenarios satisfy the thermal and SOC constraints.

## 📊 Pooled capacity results

| Capacity | Method | Closure | HV closure | HV-TARGET closure | Net utility/task | Utility/MB | Utility/Wh |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for capacity in CAPACITIES:
        for method in METHODS:
            row = cap_index[(capacity, method)]
            report += (
                f"| {capacity:.2f} | {method} | {row['pooled_closure_rate']:.4f} | "
                f"{row['pooled_hv_closure_rate']:.4f} | {row['pooled_hv_target_closure_rate']:.4f} | "
                f"{row['pooled_common_net_utility_per_task']:.6f} | {row['pooled_utility_per_mb']:.6f} | "
                f"{row['pooled_utility_per_wh']:.6f} |\n"
            )
    report += f"""

## 🔎 Required answers

1. Across the complete grid, `{winner('closure_rate')}` has the highest pooled absolute closure rate.
2. `{winner('hv_closure_rate')}` has the highest pooled ordinary-HV closure rate.
3. `{winner('hvt_closure_rate')}` has the highest HV-TARGET closure rate; `{winner('hvt_net_per_task')}` has the highest allocated common net utility per HV-TARGET task.
4. `{winner('net_per_task')}` has the highest common net utility per task across the grid.
5. `{winner('utility_per_mb')}` has the highest utility/MB.
6. `{winner('utility_per_wh')}` has the highest utility/Wh.
7. `{winner('mean_latency', lower=True)}` has the lowest mean full-closure latency; `{winner('p95_latency', lower=True)}` has the lowest mean-of-seed p95 full-closure latency.
8. At capacity 0.35, the best pooled retention winners are: closure `{retention_best['closure_rate']}`, HV `{retention_best['hv_closure_rate']}`, HV-TARGET `{retention_best['hv_target_closure_rate']}`, net utility/task `{retention_best['common_net_utility_per_task']}`, utility/MB `{retention_best['utility_per_mb']}`, and utility/Wh `{retention_best['utility_per_wh']}`.
9. TUT's high utility/MB (`{tut_overall['utility_per_mb']:.6f}`) accompanies a pooled closure rate of `{tut_overall['closure_rate']:.4f}`; PROP's values are `{prop_overall['utility_per_mb']:.6f}` and `{prop_overall['closure_rate']:.4f}`. Efficiency therefore must be read together with absolute closures.
10. PROP's evidence is not reducible to small products alone: the tables separately show closure/HV-TARGET behavior, capacity retention, latency, and utility/Wh. This remains an observational mechanism interpretation, not a single-factor causal proof.
11. Yes. Thermal violations = `0`; SOC violations = `0` across all 60 method-scenarios.

## ⚙️ Metric rules

Full-closure latency is conditional on actual public closure. Deadline-capped all-task latency is reported separately. PROP alert/ROI/evidence arrival latency is supplementary only. HV and HV-TARGET gross utility are summed from shared per-task evaluator records; common net costs are allocated uniformly per task, so group net totals reconcile exactly to each scenario's common net utility without introducing a method-specific utility function.

The five seeds are treated as paired descriptive samples. Seed win counts and differences are reported without statistical-significance claims.

## 🔒 Freeze decision

The comparison is ready to freeze. No window deletion, new capacity, new parameter, new baseline, PPO, PMSC training, or method tuning was run.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "FOUR_METHOD_COMPARISON_READY", "scenario_method_rows": len(scenarios)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "smoke", "run-seed", "analyze"), required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight()
    elif args.phase == "smoke":
        smoke()
    elif args.phase == "run-seed":
        if args.seed is None:
            raise ValueError("--seed is required for run-seed")
        run_seed(args.seed)
    else:
        analyze()


if __name__ == "__main__":
    main()
