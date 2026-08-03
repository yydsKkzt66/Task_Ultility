"""One-click controlled-V25/capacity-0.50 five-method experiment.

Set ``RUN_MODE`` below to ``"one_seed"`` for the validation run or
``"five_seed"`` for the final experiment.  The CLI ``--run-mode`` switch
overrides that file-level setting.

1. MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN
2. B1-COMMON-THERMAL
3. OFFLOAD-EFT-THERMAL-ALIGNED
4. TUT-UOI-PRODUCT-LITE
5. PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-v1

Seed 2026 is the exact frozen V25 manifest.  Seeds 2027--2030 preserve the
same 268-task cohort, value-level counts, required products, effective
1:3:10:25 values, deadlines and physical traces, and vary only the task
temporal ordering with an audited deterministic permutation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_myopic_e2e_utility_greedy import (  # noqa: E402
    DEFAULT_METHODS,
    EXPECTED_FROZEN_V25_COUNTS,
    EXPECTED_FROZEN_V25_TASK_HASH,
    METHOD_NAMES,
    VALUE_PROFILES,
    _write_json,
    build_shared_scenario,
    configure_target_capacity_multiplier,
    configure_target_scenario_profile,
    generate_environment_audit,
    run,
    task_effective_value,
    validate_seed2026_environment_accounting,
)


# User-facing switch. Change only this line before clicking Run.
RUN_MODE = "one_seed"  # choices: "one_seed", "five_seed"
RUN_MODE_SEEDS = {
    "one_seed": (2026,),
    "five_seed": (2026, 2027, 2028, 2029, 2030),
}
VALUE_PROFILE = "main"
CAPACITY_MULTIPLIER = 0.50
SCENARIO_PROFILE = "controlled_v25_multiseed"
EXPECTED_METHODS = (
    "MYOPIC-FROZEN-CALIBRATED-ON-CORRECTED-REQALIGN",
    "B1-COMMON-THERMAL",
    "OFFLOAD-EFT-THERMAL-ALIGNED",
    "TUT-UOI-PRODUCT-LITE",
    "PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-v1",
)

PRIMARY_METRICS = (
    "J_all_net",
    "J_HV_net",
    "N_HV_closure",
)
AUXILIARY_METRICS = (
    "N_HV_metadata",
    "N_HV_ROI",
    "N_HV_evidence",
    "overall_closure_rate",
    "mean_closure_latency_s",
    "p95_closure_latency_s",
    "energy_Wh",
    "transmitted_bytes",
)
SAFETY_METRICS = (
    "thermal_violations",
    "SOC_violations",
)
ALL_REPORTED_METRICS = PRIMARY_METRICS + AUXILIARY_METRICS + SAFETY_METRICS


def _seeds_for_mode(mode: str) -> tuple[int, ...]:
    selected = str(mode).strip().lower()
    if selected not in RUN_MODE_SEEDS:
        raise ValueError(
            f"run mode must be one of {sorted(RUN_MODE_SEEDS)}, got {mode!r}"
        )
    return RUN_MODE_SEEDS[selected]


def _default_output_for_mode(mode: str) -> Path:
    suffix = "seed2026_validation" if mode == "one_seed" else "five_seed_final"
    return (
        ROOT
        / "outputs"
        / f"v25_cap050_five_method_controlled_{suffix}_v2"
    )


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds are forbidden: {seeds}")
    return seeds


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    fields = list(
        dict.fromkeys(key for row in materialized for key in row)
    ) or ["empty"]
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


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_cell(row.get(field, "")) for field in fields)
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table(root: Path, name: str, rows: list[dict[str, Any]]) -> None:
    _write_csv(root / "tables" / f"{name}.csv", rows)
    _write_markdown(root / "tables" / f"{name}.md", rows)


def build_preflight(seeds: tuple[int, ...]) -> dict[str, Any]:
    configure_target_capacity_multiplier(CAPACITY_MULTIPLIER)
    configure_target_scenario_profile(SCENARIO_PROFILE)
    rows = []
    for seed in seeds:
        configs, timeline, identity = build_shared_scenario(
            seed,
            value_profile=VALUE_PROFILE,
        )
        tasks = [step.task for step in timeline if step.task is not None]
        level_counts = {
            level: sum(task.value_level == level for task in tasks)
            for level in ("low", "medium", "high", "urgent")
        }
        extensions = configs["simulator"]["formal_method_extensions"]
        frozen_alpha = 0.17447916666666666
        expected_effective = {
            level: frozen_alpha * float(weight)
            for level, weight in VALUE_PROFILES["main"]["weights"].items()
        }
        checks = {
            "capacity_is_0_50": abs(
                float(
                    identity["target_environment"]["communication"][
                        "capacity_multiplier"
                    ]
                )
                - CAPACITY_MULTIPLIER
            )
            <= 1.0e-12,
            "solar_peak_is_40W": abs(
                float(
                    identity["target_environment"]["solar"][
                        "configured_peak_w"
                    ]
                )
                - 40.0
            )
            <= 1.0e-12,
            "slot_count_is_4320": len(timeline) == 4320,
            "slot_seconds_is_20": all(
                abs(float(step.slot_seconds) - 20.0) <= 1.0e-12
                for step in timeline
            ),
            "tasks_are_nonempty_and_unique": bool(tasks)
            and len({task.task_id for task in tasks}) == len(tasks),
            "task_count_is_frozen_268": len(tasks) == 268,
            "value_level_counts_are_frozen_v25": (
                level_counts == EXPECTED_FROZEN_V25_COUNTS
            ),
            "effective_values_are_exact_v25": all(
                abs(
                    task_effective_value(task)
                    - expected_effective[str(task.value_level)]
                )
                <= 1.0e-12
                for task in tasks
            ),
            "required_products_are_present": all(
                str(task.product_requirement).strip() for task in tasks
            ),
            "value_profile_is_V25_main": (
                identity["value_profile"] == VALUE_PROFILES["main"]["profile"]
                and identity["nominal_value_weights"]
                == VALUE_PROFILES["main"]["weights"]
            ),
            "guard_disabled": not bool(
                extensions["hv_urgent_guard_enabled"]
            ),
            "medium_reclaim_disabled": not bool(
                extensions["medium_slack_reclaim_enabled"]
            ),
            "frozen_v25_base_identity_present": (
                identity["frozen_v25_identity"] is not None
                and identity["frozen_v25_identity"].get(
                    "base_frozen_task_hash",
                    identity["frozen_v25_identity"].get("task_hash"),
                )
                == EXPECTED_FROZEN_V25_TASK_HASH
            ),
        }
        rows.append(
            {
                "seed": seed,
                "task_count": len(tasks),
                "task_manifest_hash_v2": identity["task_manifest_hash_v2"],
                "environment_hash_v2": identity["environment_hash_v2"],
                "value_level_counts": level_counts,
                "checks": checks,
            }
        )
    method_names = tuple(METHOD_NAMES[key] for key in DEFAULT_METHODS)
    supported_seed_grid = seeds in tuple(RUN_MODE_SEEDS.values())
    checks = {
        "five_method_identity_exact": set(method_names)
        == set(EXPECTED_METHODS),
        "seed_grid_matches_one_or_five_mode": supported_seed_grid,
        "seed_scenarios_pass": all(
            all(row["checks"].values()) for row in rows
        ),
        "seed2026_is_exact_frozen_anchor": any(
            row["seed"] == 2026
            and row["task_manifest_hash_v2"]
            == EXPECTED_FROZEN_V25_TASK_HASH
            for row in rows
        ),
        "controlled_task_hashes_are_distinct": (
            len(rows) == 1
            or len({row["task_manifest_hash_v2"] for row in rows})
            == len(rows)
        ),
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "formal_run_authorized": passed,
        "scenario_profile": SCENARIO_PROFILE,
        "capacity_multiplier": CAPACITY_MULTIPLIER,
        "value_profile": VALUE_PROFILES["main"]["profile"],
        "value_weights": VALUE_PROFILES["main"]["weights"],
        "seeds": list(seeds),
        "method_keys": list(DEFAULT_METHODS),
        "method_names": list(method_names),
        "checks": checks,
        "per_seed": rows,
    }


def _seed_level_tables(
    metric_rows: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    base = ("seed", "method")
    primary = [
        {key: row[key] for key in base + PRIMARY_METRICS}
        for row in metric_rows
    ]
    auxiliary = []
    for row in metric_rows:
        output = {key: row[key] for key in base + AUXILIARY_METRICS}
        output["transmitted_MB"] = float(row["transmitted_bytes"]) / 1_000_000.0
        auxiliary.append(output)
    safety = [
        {key: row[key] for key in base + SAFETY_METRICS}
        for row in metric_rows
    ]
    return primary, auxiliary, safety


def build_metric_summary(
    metric_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        for metric in ALL_REPORTED_METRICS:
            grouped[(row["method"], metric)].append(float(row[metric]))
    output: list[dict[str, Any]] = []
    method_order = {
        method: index for index, method in enumerate(EXPECTED_METHODS)
    }
    metric_order = {
        metric: index for index, metric in enumerate(ALL_REPORTED_METRICS)
    }
    for (method, metric), values in sorted(
        grouped.items(),
        key=lambda item: (
            method_order.get(item[0][0], 999),
            metric_order.get(item[0][1], 999),
        ),
    ):
        mean = statistics.fmean(values)
        sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
        half_width = (
            1.96 * sample_std / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        output.append(
            {
                "method": method,
                "metric": metric,
                "seed_count": len(values),
                "mean": mean,
                "sample_std": sample_std,
                "normal_ci95_low": mean - half_width,
                "normal_ci95_high": mean + half_width,
                "min": min(values),
                "max": max(values),
            }
        )
    return output


def _mean_std_wide(
    summary: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["method"]), str(row["metric"])): row for row in summary
    }
    output = []
    for method in EXPECTED_METHODS:
        row: dict[str, Any] = {"method": method}
        for metric in metrics:
            stats = lookup[(method, metric)]
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["sample_std"]
        output.append(row)
    return output


def _primary_ranking(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means = {
        (str(row["method"]), str(row["metric"])): float(row["mean"])
        for row in summary
    }
    rows = []
    for metric in PRIMARY_METRICS:
        ordered = sorted(
            EXPECTED_METHODS,
            key=lambda method: (-means[(method, metric)], method),
        )
        for rank, method in enumerate(ordered, start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "method": method,
                    "mean": means[(method, metric)],
                }
            )
    return rows


def write_metric_outputs(
    output_root: Path,
    seeds: tuple[int, ...],
) -> None:
    metric_rows = _read_csv(output_root / "seed_metrics.csv")
    expected_rows = len(seeds) * len(DEFAULT_METHODS)
    if len(metric_rows) != expected_rows:
        raise AssertionError(
            f"incomplete seed-method grid: {len(metric_rows)} != {expected_rows}"
        )
    primary, auxiliary, safety = _seed_level_tables(metric_rows)
    summary = build_metric_summary(metric_rows)
    _write_table(output_root, "primary_metrics_per_seed", primary)
    _write_table(output_root, "auxiliary_metrics_per_seed", auxiliary)
    _write_table(output_root, "safety_metrics_per_seed", safety)
    _write_table(output_root, "multiseed_metric_summary", summary)
    _write_table(
        output_root,
        "primary_metrics_mean_std",
        _mean_std_wide(summary, PRIMARY_METRICS),
    )
    _write_table(
        output_root,
        "auxiliary_metrics_mean_std",
        _mean_std_wide(summary, AUXILIARY_METRICS),
    )
    _write_table(
        output_root,
        "safety_metrics_mean_std",
        _mean_std_wide(summary, SAFETY_METRICS),
    )
    _write_table(output_root, "primary_metric_ranking", _primary_ranking(summary))

    report = f"""# Controlled V25 capacity-0.50 five-method results

- Seeds: `{", ".join(str(seed) for seed in seeds)}`
- Methods: `{len(DEFAULT_METHODS)}`
- Scenario: `{SCENARIO_PROFILE}`
- Frozen V25 anchor hash: `{EXPECTED_FROZEN_V25_TASK_HASH}`
- Value profile: `{VALUE_PROFILES["main"]["profile"]}`
- Value ratio low:medium:high:urgent: `1:3:10:25`
- Capacity multiplier: `{CAPACITY_MULTIPLIER:.2f}`
- Duration / slot: `86400 s / 20 s`

## Main result tables

- `tables/primary_metrics_per_seed.csv`
- `tables/primary_metrics_mean_std.csv`
- `tables/auxiliary_metrics_per_seed.csv`
- `tables/auxiliary_metrics_mean_std.csv`
- `tables/safety_metrics_per_seed.csv`
- `tables/safety_metrics_mean_std.csv`
- `tables/multiseed_metric_summary.csv`
- `tables/primary_metric_ranking.csv`

The reported 95% intervals are descriptive normal intervals across the
declared seeds. They are not a claim of statistical significance or stable
superiority.
"""
    (output_root / "README_RESULTS.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-click five-method controlled-V25 capacity-0.50 experiment."
        )
    )
    parser.add_argument(
        "--run-mode",
        choices=tuple(RUN_MODE_SEEDS),
        default=RUN_MODE,
        help=(
            "one_seed runs the exact frozen seed-2026 validation; five_seed "
            "runs the controlled five-seed final grid."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Fresh output directory; default is selected from --run-mode.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate all scenarios/method identities without running episodes.",
    )
    args = parser.parse_args()
    seeds = _seeds_for_mode(args.run_mode)
    output_root = args.output_dir or _default_output_for_mode(args.run_mode)
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; choose a fresh directory: {output_root}"
        )

    configure_target_capacity_multiplier(CAPACITY_MULTIPLIER)
    configure_target_scenario_profile(SCENARIO_PROFILE)
    preflight = build_preflight(seeds)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "audit" / "multiseed_preflight.json", preflight)
    if preflight["status"] != "PASS":
        raise AssertionError({"multiseed_preflight_failed": preflight["checks"]})
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "preflight_only",
                    "output": str(output_root),
                },
                ensure_ascii=False,
            )
        )
        return

    generate_environment_audit(
        output_root,
        seeds,
        DEFAULT_METHODS,
        value_profile=VALUE_PROFILE,
    )
    validate_seed2026_environment_accounting(
        output_root,
        value_profile=VALUE_PROFILE,
    )
    run(
        output_root,
        seeds,
        DEFAULT_METHODS,
        overwrite=False,
        value_profile=VALUE_PROFILE,
    )
    write_metric_outputs(output_root, seeds)
    completion = {
        "status": "V25_CAP050_FIVE_METHOD_CONTROLLED_COMPLETE",
        "output": str(output_root),
        "run_mode": args.run_mode,
        "seeds": list(seeds),
        "methods": list(EXPECTED_METHODS),
        "seed_method_run_count": len(seeds) * len(DEFAULT_METHODS),
    }
    _write_json(output_root / "completion_status.json", completion)
    print(json.dumps(completion, ensure_ascii=False))


if __name__ == "__main__":
    main()
