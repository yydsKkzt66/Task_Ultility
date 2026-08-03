"""Exact-native V10/V50 value-ratio sensitivity at capacity 0.50.

The script reuses the validated controlled-V25 five-method runtime.  For each
seed it first creates the frozen controlled bundle, then changes only the
declared value ratio and the mean-preserving effective task values derived
from it.  Task identity, level assignment, arrival schedule, requirements,
deadlines, physical traces, method implementations and evaluator stay fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_bupt1_power_true_rerun_seed2026_v1 as v25  # noqa: E402
import scripts.run_exact_native_v25_cap050_five_method_fiveseed as exact  # noqa: E402


getcontext().prec = 50

SEEDS = exact.SEEDS
METHODS = exact.METHODS
CAPACITY = 0.50
TASK_COUNT = 268
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}
LEVELS = ("low", "medium", "high", "urgent")
VALUE_PROFILES: dict[str, dict[str, Decimal | str]] = {
    "V10": {
        "name": "V10-SENS-1-2-5-10-v1",
        "low": Decimal("1"),
        "medium": Decimal("2"),
        "high": Decimal("5"),
        "urgent": Decimal("10"),
    },
    "V50": {
        "name": "V50-SENS-1-5-20-50-v1",
        "low": Decimal("1"),
        "medium": Decimal("5"),
        "high": Decimal("20"),
        "urgent": Decimal("50"),
    },
}
REFERENCE_V25 = ROOT / "outputs" / "_v25_exact_native_five_method_controlled_fiveseed_cap050_v1"
DEFAULT_OUTPUT = ROOT / "outputs" / "_v10_v50_exact_native_five_method_controlled_fiveseed_cap050_v1"


def canonical_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(dict.fromkeys(key for row in materialized for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized or [{"empty": ""}])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def profile_weights(profile: str) -> dict[str, Decimal]:
    spec = VALUE_PROFILES[profile]
    return {level: Decimal(spec[level]) for level in LEVELS}


def normalization_alpha(
    weights: Mapping[str, Decimal],
    counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> Decimal:
    raw_sum = sum(
        Decimal(int(counts[level])) * Decimal(weights[level]) for level in LEVELS
    )
    return Decimal(sum(int(counts[level]) for level in LEVELS)) / raw_sum


def invariant_task_hash(rows: list[dict[str, str]]) -> str:
    fields = (
        "task_id",
        "scene_id",
        "arrival_time_s",
        "deadline_time_s",
        "original_deadline_slack_s",
        "value_level",
        "product_requirement",
        "quality_requirement",
        "information_requirement",
        "high_value_flag",
        "emergency_flag",
    )
    payload = sorted(
        ({field: row.get(field, "") for field in fields} for row in rows),
        key=lambda row: row["task_id"],
    )
    return exact.stable_hash(payload)


def _apply_value_profile(
    bundle_root: Path,
    *,
    seed: int,
    profile: str,
    controlled_identity: Mapping[str, Any],
) -> dict[str, Any]:
    tasks_path = bundle_root / "task_manifest.csv"
    values_path = bundle_root / "task_value_assignment_trace.csv"
    config_path = bundle_root / "resolved_config.json"
    contract_path = bundle_root / "bundle_contract.json"
    tasks = read_csv(tasks_path)
    value_rows = read_csv(values_path)
    counts = Counter(row["value_level"] for row in tasks)
    weights = profile_weights(profile)
    alpha = normalization_alpha(weights, counts)
    before_invariant_hash = invariant_task_hash(tasks)
    before_slot_hash = exact.sha256_file(bundle_root / "slot_trace.csv")
    before_swap_hash = exact.sha256_file(bundle_root / "arrival_swap_manifest.csv")
    profile_name = str(VALUE_PROFILES[profile]["name"])

    value_by_task: dict[str, Decimal] = {}
    for row in tasks:
        level = row["value_level"]
        effective = alpha * weights[level]
        token = canonical_decimal(effective)
        value_by_task[row["task_id"]] = effective
        row["task_value"] = token
        row["normalization_alpha"] = canonical_decimal(alpha)
        row["calibrated_task_value"] = token
        row["effective_task_value"] = token
        row["value_skew_profile"] = profile_name
        row["value_assignment_reason"] = (
            "fixed_v25_level_assignment_mean_preserving_value_ratio_sensitivity"
        )
        metadata = json.loads(row.get("metadata_json") or "{}")
        metadata.update(
            {
                "level_multiplier_audit_only": float(weights[level]),
                "value_profile": profile_name,
                "value_ratio_sensitivity": profile,
                "value_ratio_reference": "V25-MAIN-v1",
            }
        )
        row["metadata_json"] = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    write_csv(tasks_path, tasks)

    for row in value_rows:
        task_id = row["task_id"]
        level = row["value_level"]
        row["level_multiplier"] = canonical_decimal(weights[level])
        row["normalization_alpha"] = canonical_decimal(alpha)
        row["effective_task_value"] = canonical_decimal(value_by_task[task_id])
    write_csv(values_path, value_rows)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["value_ratio_sensitivity"] = {
        "profile": profile_name,
        "profile_label": profile,
        "weights": {level: canonical_decimal(weights[level]) for level in LEVELS},
        "normalization": "mean_preserving",
        "normalization_alpha": canonical_decimal(alpha),
        "reference_environment": "V25 capacity=0.50 controlled five-seed",
        "only_allowed_change": "task value ratio and derived effective values",
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    base_task_hash = contract["task_hash"]
    base_environment_hash = contract["environment_hash"]
    contract.update(
        {
            "value_profile": profile_name,
            "value_profile_mode": "MEAN_NORMALIZED_VALUE_RATIO_SENSITIVITY",
            "value_multipliers": {
                level: canonical_decimal(weights[level]) for level in LEVELS
            },
            "normalization_factor": canonical_decimal(alpha),
            "sensitivity_reference_profile": "V25-MAIN-v1",
            "sensitivity_reference_task_hash": base_task_hash,
            "task_hash": v25.v25_bundle_hash(tasks),
        }
    )
    contract["environment_hash"] = exact.stable_hash(
        {
            "base_environment_hash": base_environment_hash,
            "controlled_seed": seed,
            "capacity": CAPACITY,
            "value_profile": profile_name,
            "task_hash": contract["task_hash"],
            "only_change": "mean_preserving_value_ratio",
        }
    )
    for name in contract["file_sha256"]:
        contract["file_sha256"][name] = exact.sha256_file(bundle_root / name)
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    after_tasks = read_csv(tasks_path)
    observed_values = [Decimal(row["effective_task_value"]) for row in after_tasks]
    checks = {
        "controlled_bundle_preflight_passed": all(
            bool(value) for value in controlled_identity["checks"].values()
        ),
        "task_count_is_268": len(after_tasks) == TASK_COUNT,
        "level_counts_unchanged": dict(Counter(row["value_level"] for row in after_tasks))
        == EXPECTED_COUNTS,
        "non_value_task_identity_unchanged": invariant_task_hash(after_tasks)
        == before_invariant_hash,
        "slot_trace_unchanged": exact.sha256_file(bundle_root / "slot_trace.csv")
        == before_slot_hash,
        "arrival_swap_manifest_unchanged": exact.sha256_file(
            bundle_root / "arrival_swap_manifest.csv"
        )
        == before_swap_hash,
        # V10/V50 normalization factors are repeating decimals.  The bundle
        # stores 50 significant decimal digits, so exact rational equality is
        # represented to a far tighter tolerance than the public evaluator.
        "mean_effective_value_one_within_1e_45": abs(
            sum(observed_values) - Decimal(TASK_COUNT)
        )
        <= Decimal("1e-45"),
        "effective_values_match_ratio": all(
            Decimal(row["effective_task_value"])
            == alpha * weights[row["value_level"]]
            for row in after_tasks
        ),
        "runtime_value_multiplier_stays_one": all(
            Decimal(row.get("value_multiplier") or "1") == Decimal("1")
            for row in after_tasks
        ),
        "profile_not_v25": contract["value_profile"] != "V25-MAIN-v1",
    }
    if not all(checks.values()):
        raise AssertionError(
            {"VALUE_RATIO_ISOLATION_PREFLIGHT_FAILED": profile, "seed": seed, "checks": checks}
        )
    v25.load_bundle(bundle_root)
    return {
        "profile": profile,
        "profile_name": profile_name,
        "seed": seed,
        "weights": {level: canonical_decimal(weights[level]) for level in LEVELS},
        "normalization_alpha": canonical_decimal(alpha),
        "effective_values": {
            level: canonical_decimal(alpha * weights[level]) for level in LEVELS
        },
        "mean_effective_value": canonical_decimal(
            sum(observed_values) / Decimal(len(observed_values))
        ),
        "task_hash": contract["task_hash"],
        "non_value_task_identity_hash": before_invariant_hash,
        "slot_trace_sha256": before_slot_hash,
        "checks": checks,
    }


def prepare_profile_bundle(
    profile: str,
    seed: int,
    destination: Path,
) -> dict[str, Any]:
    source = REFERENCE_V25 / "config" / "controlled_bundles" / f"seed_{seed}"
    if destination.exists():
        raise FileExistsError(destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    # Copy the exact bundle that produced the completed V25 reference rather
    # than rebuilding it from a mutable upstream source.
    shutil.copytree(source, destination)
    loaded = v25.load_bundle(destination)
    tasks = read_csv(destination / "task_manifest.csv")
    contract = json.loads(
        (destination / "bundle_contract.json").read_text(encoding="utf-8")
    )
    controlled = {
        "seed": seed,
        "task_hash": contract["task_hash"],
        "environment_hash": contract["environment_hash"],
        "checks": {
            "completed_v25_reference_exists": (
                REFERENCE_V25 / "completion_status.json"
            ).is_file(),
            "source_bundle_contract_loaded": len(loaded.tasks) == TASK_COUNT,
            "task_count_is_268": len(tasks) == TASK_COUNT,
            "value_counts_match": dict(Counter(row["value_level"] for row in tasks))
            == EXPECTED_COUNTS,
            "source_task_hash_matches_contract": v25.v25_bundle_hash(tasks)
            == contract["task_hash"],
        },
    }
    return _apply_value_profile(
        destination,
        seed=seed,
        profile=profile,
        controlled_identity=controlled,
    )


def run_one(
    profile: str,
    seed: int,
    method: str,
    bundle_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    path = output_root / "runs" / profile / f"seed_{seed}" / method
    status = exact.launch(
        method,
        seed,
        bundle_root,
        path,
        output_root / "logs" / profile,
        CAPACITY,
    )
    print(
        json.dumps(
            {"status": "RUN_COMPLETE", "profile": profile, "seed": seed, "method": method},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return status


def full_metric_row(
    profile: str,
    seed: int,
    method: str,
    path: Path,
    bundle_root: Path,
    eval_mod: Any,
) -> dict[str, Any]:
    raw = exact.evaluate_method(method, path, bundle_root, eval_mod, CAPACITY)
    selected = exact.selected_metrics(seed, raw)
    extra_fields = (
        "ordinary_closure_count",
        "high_gross_utility",
        "urgent_gross_utility",
        "high_net_utility",
        "urgent_net_utility",
        "urgent_metadata_delivered_count",
        "urgent_ROI_delivered_count",
        "urgent_evidence_delivered_count",
        "compute_energy_Wh",
        "encoding_energy_Wh",
        "tx_energy_Wh",
        "initial_soc",
        "final_soc",
        "maximum_DoD",
        "max_energy_balance_residual_Wh",
    )
    return {
        "value_profile": profile,
        **selected,
        **{field: raw[field] for field in extra_fields},
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity = {"value_profile", "seed", "method", "public_accounting_passed"}
    numeric = [
        key
        for key in rows[0]
        if key not in identity and key not in {"task_count", "HV_task_count"}
    ]
    output = []
    for profile in ("V10", "V25", "V50"):
        for method in METHODS:
            group = [
                row for row in rows if row["value_profile"] == profile and row["method"] == method
            ]
            if not group:
                continue
            item: dict[str, Any] = {
                "value_profile": profile,
                "method": method,
                "seed_count": len(group),
            }
            for key in numeric:
                values = [float(row[key]) for row in group]
                item[f"{key}_mean"] = statistics.fmean(values)
                item[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output.append(item)
    return output


def emit_tables(output: Path, rows: list[dict[str, Any]]) -> None:
    order = {method: index for index, method in enumerate(METHODS)}
    profile_order = {"V10": 0, "V25": 1, "V50": 2}
    rows.sort(key=lambda row: (profile_order[row["value_profile"]], int(row["seed"]), order[row["method"]]))
    summary = aggregate(rows)
    primary = (
        "J_all_net",
        "J_HV_net",
        "N_HV_closure",
        "N_high_closure",
        "N_urgent_closure",
    )
    auxiliary = (
        "all_task_closure_count",
        "all_task_closure_rate",
        "ordinary_closure_count",
        "N_HV_metadata",
        "N_HV_ROI",
        "N_HV_evidence",
        "successful_closure_mean_latency_s",
        "successful_closure_p95_latency_s",
        "HV_successful_closure_mean_latency_s",
        "HV_successful_closure_p95_latency_s",
        "urgent_successful_closure_mean_latency_s",
        "urgent_successful_closure_p95_latency_s",
        "high_gross_utility",
        "urgent_gross_utility",
        "high_net_utility",
        "urgent_net_utility",
        "controllable_energy_Wh",
        "transmitted_MB",
        "peak_queue_MB",
        "mean_queue_MB",
    )
    safety = (
        "minimum_soc",
        "final_soc",
        "maximum_thermal",
        "SOC_violations",
        "thermal_violations",
    )

    def subset(metrics: tuple[str, ...]) -> list[dict[str, Any]]:
        fields = ["value_profile", "method", "seed_count"]
        for metric in metrics:
            fields.extend((f"{metric}_mean", f"{metric}_std"))
        return [{field: row[field] for field in fields} for row in summary]

    primary_summary = subset(primary)
    auxiliary_summary = subset(auxiliary)
    safety_summary = subset(safety)
    primary_per_seed = [
        {
            key: row[key]
            for key in ("value_profile", "seed", "method", *primary)
        }
        for row in rows
    ]
    deltas = []
    for profile in ("V10", "V25", "V50"):
        group = {row["method"]: row for row in summary if row["value_profile"] == profile}
        proposed = group[exact.METHOD_CORRECTED]
        for method in METHODS:
            if method == exact.METHOD_CORRECTED:
                continue
            item = {
                "value_profile": profile,
                "proposed_method": exact.METHOD_CORRECTED,
                "baseline": method,
            }
            for metric in primary:
                item[f"delta_{metric}"] = (
                    proposed[f"{metric}_mean"] - group[method][f"{metric}_mean"]
                )
            deltas.append(item)
    prop_sensitivity = [
        row for row in primary_summary if row["method"] == exact.METHOD_CORRECTED
    ]
    write_csv(output / "tables" / "all_metrics_per_seed.csv", rows)
    write_csv(output / "tables" / "all_metrics_mean_std.csv", summary)
    write_csv(output / "tables" / "primary_metrics_per_seed.csv", primary_per_seed)
    write_csv(output / "tables" / "primary_metrics_mean_std.csv", primary_summary)
    write_csv(output / "tables" / "auxiliary_metrics_mean_std.csv", auxiliary_summary)
    write_csv(output / "tables" / "safety_metrics_mean_std.csv", safety_summary)
    write_csv(output / "tables" / "prop_vs_baselines_deltas.csv", deltas)
    write_csv(output / "tables" / "prop_value_sensitivity.csv", prop_sensitivity)
    write_md(output / "tables" / "primary_metrics_mean_std.md", primary_summary)
    write_md(output / "tables" / "auxiliary_metrics_mean_std.md", auxiliary_summary)
    write_md(output / "tables" / "safety_metrics_mean_std.md", safety_summary)
    write_md(output / "tables" / "prop_vs_baselines_deltas.md", deltas)
    write_md(output / "tables" / "prop_value_sensitivity.md", prop_sensitivity)

    by = {(row["value_profile"], row["method"]): row for row in summary}
    lines = [
        "# V10/V25/V50 value-ratio sensitivity",
        "",
        "This is a five-seed controlled sensitivity analysis. Only the mean-preserving value ratio changes.",
        "",
        "## Proposed method primary metrics",
        "",
    ]
    for profile in ("V10", "V25", "V50"):
        row = by[(profile, exact.METHOD_CORRECTED)]
        lines.append(
            f"- {profile}: J_all={row['J_all_net_mean']:.6f} +/- {row['J_all_net_std']:.6f}; "
            f"J_HV={row['J_HV_net_mean']:.6f} +/- {row['J_HV_net_std']:.6f}; "
            f"HV closures={row['N_HV_closure_mean']:.2f} +/- {row['N_HV_closure_std']:.2f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The results characterize sensitivity to the declared value ratio in this controlled five-seed environment. They do not establish significance outside these seeds or environments.",
        ]
    )
    (output / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_all(
    output: Path,
    bundles: Mapping[tuple[str, int], Path],
) -> list[dict[str, Any]]:
    eval_mod = v25.load_eval_module(output)
    rows: list[dict[str, Any]] = []
    for profile in ("V10", "V50"):
        for seed in SEEDS:
            for method in METHODS:
                path = output / "runs" / profile / f"seed_{seed}" / method
                row = full_metric_row(
                    profile, seed, method, path, bundles[(profile, seed)], eval_mod
                )
                rows.append(row)
    completion = json.loads((REFERENCE_V25 / "completion_status.json").read_text(encoding="utf-8"))
    if completion.get("run_count") != 25 or not completion.get("hard_safety_passed"):
        raise RuntimeError("V25_REFERENCE_INCOMPLETE_OR_UNSAFE")
    for seed in SEEDS:
        bundle = REFERENCE_V25 / "config" / "controlled_bundles" / f"seed_{seed}"
        for method in METHODS:
            path = REFERENCE_V25 / "runs" / f"seed_{seed}" / method
            rows.append(full_metric_row("V25", seed, method, path, bundle, eval_mod))
    return rows


def parent_main(
    output: Path,
    workers: int,
    preflight_only: bool,
    resume: bool,
) -> int:
    if output.exists() and not resume:
        raise FileExistsError(f"fresh output required: {output}")
    if not (REFERENCE_V25 / "completion_status.json").is_file():
        raise FileNotFoundError(f"missing completed V25 reference: {REFERENCE_V25}")
    bundles: dict[tuple[str, int], Path] = {}
    if resume:
        isolation_path = output / "audit" / "value_ratio_isolation_audit.json"
        if not isolation_path.is_file():
            raise FileNotFoundError(f"resume requires completed isolation audit: {isolation_path}")
        isolation = json.loads(isolation_path.read_text(encoding="utf-8"))
        if isolation.get("status") != "PASS":
            raise RuntimeError("resume isolation audit is not PASS")
        for profile in ("V10", "V50"):
            for seed in SEEDS:
                bundle = output / "config" / profile / f"seed_{seed}"
                v25.load_bundle(bundle)
                bundles[(profile, seed)] = bundle
    else:
        output.mkdir(parents=True, exist_ok=False)
        audits = []
        for profile in ("V10", "V50"):
            for seed in SEEDS:
                destination = output / "config" / profile / f"seed_{seed}"
                bundles[(profile, seed)] = destination
                audits.append(prepare_profile_bundle(profile, seed, destination))
        # The two profiles must share every non-value task identity per seed.
        cross_profile_checks = {
            str(seed): {
                "non_value_task_identity_match": (
                    next(row for row in audits if row["profile"] == "V10" and row["seed"] == seed)["non_value_task_identity_hash"]
                    == next(row for row in audits if row["profile"] == "V50" and row["seed"] == seed)["non_value_task_identity_hash"]
                ),
                "slot_trace_match": (
                    next(row for row in audits if row["profile"] == "V10" and row["seed"] == seed)["slot_trace_sha256"]
                    == next(row for row in audits if row["profile"] == "V50" and row["seed"] == seed)["slot_trace_sha256"]
                ),
            }
            for seed in SEEDS
        }
        if not all(all(check.values()) for check in cross_profile_checks.values()):
            raise RuntimeError({"CROSS_PROFILE_PAIRING_MISMATCH": cross_profile_checks})
        write_json(
            output / "audit" / "value_ratio_isolation_audit.json",
            {
                "status": "PASS",
                "capacity": CAPACITY,
                "seeds": list(SEEDS),
                "methods": list(METHODS),
                "profiles": audits,
                "cross_profile_checks": cross_profile_checks,
            },
        )
    if preflight_only:
        print(json.dumps({"status": "PREFLIGHT_COMPLETE", "output": str(output)}, ensure_ascii=False))
        return 0

    all_jobs = [
        (profile, seed, method)
        for profile in ("V10", "V50")
        for seed in SEEDS
        for method in METHODS
    ]
    jobs = [
        (profile, seed, method)
        for profile, seed, method in all_jobs
        if not (
            output
            / "runs"
            / profile
            / f"seed_{seed}"
            / method
            / "method_completed.json"
        ).is_file()
    ]
    print(
        json.dumps(
            {
                "status": "RESUME" if resume else "FRESH",
                "completed_before_start": len(all_jobs) - len(jobs),
                "remaining_runs": len(jobs),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    statuses = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                run_one,
                profile,
                seed,
                method,
                bundles[(profile, seed)],
                output,
            ): (profile, seed, method)
            for profile, seed, method in jobs
        }
        for future in as_completed(futures):
            profile, seed, method = futures[future]
            try:
                statuses.append(future.result())
            except Exception as error:
                raise RuntimeError(
                    {"SENSITIVITY_RUNTIME_FAILED": {"profile": profile, "seed": seed, "method": method, "error": str(error)}}
                ) from error
    # Rebuild the status audit from all completed run directories so a resumed
    # parent contains the complete 50-run evidence, not only the latest batch.
    all_statuses = []
    for profile, seed, method in all_jobs:
        status_path = (
            output / "runs" / profile / f"seed_{seed}" / method / "method_run_status.json"
        )
        if not status_path.is_file():
            raise RuntimeError({"INCOMPLETE_SENSITIVITY_GRID": str(status_path)})
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update({"value_profile": profile, "seed": seed, "method": method})
        all_statuses.append(status)
    write_json(output / "audit" / "method_run_status.json", all_statuses)
    rows = evaluate_all(output, bundles)
    if len(rows) != 75:
        raise RuntimeError({"INCOMPLETE_SENSITIVITY_GRID": len(rows)})
    if any(not bool(row["public_accounting_passed"]) for row in rows):
        raise RuntimeError("PUBLIC_ACCOUNTING_MISMATCH")
    if any(int(row["SOC_violations"]) or int(row["thermal_violations"]) for row in rows):
        raise RuntimeError("HARD_SAFETY_VIOLATION")
    emit_tables(output, rows)
    completion = {
        "status": "V10_V50_CAP050_FIVE_METHOD_FIVESEED_SENSITIVITY_COMPLETE",
        "new_run_count": 50,
        "combined_with_v25_row_count": 75,
        "capacity": CAPACITY,
        "profiles": ["V10", "V25_REFERENCE", "V50"],
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "public_accounting_passed": True,
        "hard_safety_passed": True,
        "output": str(output),
    }
    write_json(output / "completion_status.json", completion)
    print(json.dumps(completion, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return parent_main(args.output_dir, args.workers, args.preflight_only, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
