"""Formal multi-seed runner for the BUPT-1 deadline-minute experiment.

The default invocation is dry-run only. A real run must pass --execute.
Heavy candidate-action and queue-priority logs are disabled unless explicitly
requested. This script does not change router, reward, queue, task-value,
profile, deadline, resource, contact, or metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_payload_type_attribution import (  # noqa: E402
    _write_csv as _write_attribution_csv,
)
from scripts.analyze_payload_type_attribution import (  # noqa: E402
    build_attribution_rows,
    write_attribution_markdown,
)
from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from scripts.run_bupt1_deadline_minutes_smoke import (  # noqa: E402
    DELIVERED_FIELDS,
    SCENARIOS,
    _build_router,
    _compute_delivered_metrics,
    _deadline_diagnostics,
    _extend_summary,
    _fmt,
    _load_configs,
    _variants,
    _write_csv,
    _write_markdown_table,
)
from src.common.config import ensure_output_dirs  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


PREP_DIR = PROJECT_ROOT / "outputs" / "_formal_multiseed_prep"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "formal_multiseed"

MINIMUM_FORMAL_VARIANTS = ["PROP-COMB", "UTIL", "FIX-L", "FIX-H", "NO-MODE", "NO-LYA"]
EXTENDED_APPENDIX_VARIANTS = MINIMUM_FORMAL_VARIANTS + ["PROP-Z", "PROP-PAY", "PROP-UCB"]
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]
PRIMARY_METRICS = [
    "delivered_only_gross_utility",
    "high_value_delivered_utility",
    "net_system_utility_provisional",
    "lifetime_delivered_task_level",
    "lifetime_expired_task_level",
    "delivered_mb",
    "generated_payload_mb",
    "min_battery_soc",
    "max_thermal",
    "average_queue_mb",
    "max_queue_mb",
]


def _variant_maps() -> tuple[dict[str, str], dict[str, str]]:
    short_to_full: dict[str, str] = {}
    full_to_short: dict[str, str] = {}
    for row in _variants():
        short = row["short_name"]
        full = row["full_name"]
        short_to_full[short] = full
        short_to_full[short.lower()] = full
        short_to_full[full] = full
        full_to_short[full] = short
    return short_to_full, full_to_short


def _parse_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SEEDS)
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _resolve_variants(raw_variants: list[str]) -> list[str]:
    short_to_full, _full_to_short = _variant_maps()
    resolved: list[str] = []
    for variant in raw_variants:
        full = short_to_full.get(variant) or short_to_full.get(variant.upper()) or short_to_full.get(variant.lower())
        if full is None:
            raise ValueError(f"Unknown formal variant: {variant}")
        if full not in resolved:
            resolved.append(full)
    return resolved


def _short_name(variant: str) -> str:
    _short_to_full, full_to_short = _variant_maps()
    return full_to_short.get(variant, variant)


def _scenario_dir(scenario: str, seed: int) -> Path:
    return OUTPUT_ROOT / scenario / f"seed_{seed}"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _safe_relative_delta(value: float, baseline: float) -> float:
    if not _finite(value) or not _finite(baseline) or abs(baseline) <= 1.0e-12:
        return math.nan
    return (value - baseline) / abs(baseline)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _augment_logs(rows: list[dict[str, object]], scenario: str, seed: int, variant: str) -> list[dict[str, object]]:
    short = _short_name(variant)
    return [dict(row, scenario=scenario, seed=seed, variant_full=variant, variant_short=short) for row in rows]


def _write_optional_logs(
    output_dir: Path,
    scenario: str,
    seed: int,
    variant: str,
    candidate_logs: list[dict[str, object]],
    queue_logs: list[dict[str, object]],
    write_candidate_logs: bool,
    write_queue_logs: bool,
) -> None:
    if write_candidate_logs and candidate_logs:
        _write_rows(output_dir / f"candidate_action_score_log_{variant}.csv", _augment_logs(candidate_logs, scenario, seed, variant))
    if write_queue_logs and queue_logs:
        _write_rows(output_dir / f"queue_priority_component_log_{variant}.csv", _augment_logs(queue_logs, scenario, seed, variant))


def _build_configs_for_run(scenario: str, variant: str, seed: int) -> dict[str, dict[str, Any]]:
    configs = _load_configs(scenario, variant)
    configs["dataset"] = dict(configs["dataset"])
    configs["dataset"]["seed"] = int(seed)
    return configs


def _run_one_scenario_seed(
    scenario: str,
    seed: int,
    variants: list[str],
    allow_overwrite: bool,
    write_candidate_logs: bool,
    write_queue_logs: bool,
) -> list[dict[str, Any]]:
    output_dir = _scenario_dir(scenario, seed)
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    output_dirs = ensure_output_dirs(output_dir)
    logger = get_logger("run_formal_multiseed")
    results = []
    timeline_for_report = None

    for variant in variants:
        configs = _build_configs_for_run(scenario, variant, seed)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        if timeline_for_report is None:
            timeline_for_report = timeline
        environment = SimulationEnvironment(configs)
        router = _build_router(variant, environment)
        logger.info("running formal multiseed scenario=%s seed=%s variant=%s", scenario, seed, variant)
        runner = environment.build_episode_runner(timeline)
        result = runner.run(router)
        results.append(result)
        _write_optional_logs(
            output_dir=output_dir,
            scenario=scenario,
            seed=seed,
            variant=variant,
            candidate_logs=runner.last_candidate_action_logs,
            queue_logs=runner.last_queue_priority_logs,
            write_candidate_logs=write_candidate_logs,
            write_queue_logs=write_queue_logs,
        )

    write_episode_results(results, output_dirs)
    slot_seconds = float(timeline_for_report[0].slot_seconds if timeline_for_report else 20.0)
    generate_lifetime_outputs(output_dir, slot_seconds)

    deadline_diag = _deadline_diagnostics(scenario)
    rows = [row for row in _compute_delivered_metrics(scenario, output_dir, deadline_diag) if row["variant_full"] in variants]
    for row in rows:
        row["scenario"] = scenario
        row["seed"] = seed
        row["run_output_dir"] = _relative(output_dir)

    _write_csv(
        output_dir / "delivered_only_metrics.csv",
        rows,
        ["scenario", "seed", "run_output_dir"] + DELIVERED_FIELDS,
    )
    _write_markdown_table(output_dir / "delivered_only_metrics.md", rows, f"{scenario.title()} Seed {seed} Delivered-only Metrics")
    _extend_summary(output_dir, rows)

    attribution_notes: list[str] = []
    try:
        attribution_rows, _by_variant = build_attribution_rows(f"{scenario}_seed_{seed}", output_dir)
        _write_attribution_csv(output_dir / "payload_type_attribution.csv", attribution_rows)
        write_attribution_markdown(output_dir / "payload_type_attribution.md", f"{scenario} seed {seed}", attribution_rows)
        attribution_notes.append(f"payload_attribution_rows={len(attribution_rows)}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        attribution_notes.append(f"payload_attribution_failed={type(exc).__name__}: {exc}")

    _write_text(
        output_dir / "run_summary.md",
        [
            "# Formal Multi-seed Run Summary",
            "",
            f"- scenario: {scenario}",
            f"- seed: {seed}",
            f"- variants: {', '.join(_short_name(variant) for variant in variants)}",
            f"- candidate logs written: {write_candidate_logs}",
            f"- queue priority logs written: {write_queue_logs}",
            f"- delivered-only metrics: `{_relative(output_dir / 'delivered_only_metrics.csv')}`",
            f"- payload attribution: {'; '.join(attribution_notes) if attribution_notes else 'not attempted'}",
        ],
    )
    return rows


def _aggregate_rows(per_run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run_rows:
        grouped[(str(row["scenario"]), str(row["variant_short"]), str(row["variant_full"]))].append(row)

    aggregate: list[dict[str, Any]] = []
    for (scenario, short, full), rows in grouped.items():
        out: dict[str, Any] = {
            "scenario": scenario,
            "variant_short": short,
            "variant_full": full,
            "seed_count": len(rows),
        }
        for metric in PRIMARY_METRICS:
            values = [_to_float(row.get(metric)) for row in rows]
            values = [value for value in values if _finite(value)]
            out[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) >= 2 else 0.0 if values else math.nan
        aggregate.append(out)

    for scenario in sorted({row["scenario"] for row in aggregate}):
        scenario_rows = [row for row in aggregate if row["scenario"] == scenario]
        by_short = {row["variant_short"]: row for row in scenario_rows}
        no_mode = _to_float(by_short.get("NO-MODE", {}).get("delivered_only_gross_utility_mean"))
        no_lya = _to_float(by_short.get("NO-LYA", {}).get("delivered_only_gross_utility_mean"))
        util = _to_float(by_short.get("UTIL", {}).get("delivered_only_gross_utility_mean"))
        for row in scenario_rows:
            value = _to_float(row.get("delivered_only_gross_utility_mean"))
            row["relative_improvement_over_NO_MODE"] = _safe_relative_delta(value, no_mode)
            row["relative_improvement_over_NO_LYA"] = _safe_relative_delta(value, no_lya)
            row["relative_gap_to_UTIL"] = _safe_relative_delta(value, util)
        for rank, row in enumerate(
            sorted(scenario_rows, key=lambda item: _to_float(item.get("delivered_only_gross_utility_mean")), reverse=True),
            start=1,
        ):
            row["rank_by_primary_delivered_only_utility"] = rank
        for rank, row in enumerate(
            sorted(scenario_rows, key=lambda item: _to_float(item.get("high_value_delivered_utility_mean")), reverse=True),
            start=1,
        ):
            row["rank_by_high_value_delivered_utility"] = rank
        for rank, row in enumerate(
            sorted(scenario_rows, key=lambda item: _to_float(item.get("net_system_utility_provisional_mean")), reverse=True),
            start=1,
        ):
            row["rank_by_net_provisional_utility"] = rank
    return aggregate


def _write_aggregate_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Formal Multi-seed Aggregate Summary",
        "",
        "| scenario | rank delivered | rank high-value | rank net | variant | seeds | delivered-only mean | high-value mean | net provisional mean | min SOC mean | max thermal mean | rel vs NO-MODE | rel vs NO-LYA | gap to UTIL |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["scenario"], item.get("rank_by_primary_delivered_only_utility", 999))):
        lines.append(
            f"| {row['scenario']} | {row.get('rank_by_primary_delivered_only_utility', '')} | "
            f"{row.get('rank_by_high_value_delivered_utility', '')} | {row.get('rank_by_net_provisional_utility', '')} | "
            f"{row['variant_short']} | {row['seed_count']} | {_fmt(row.get('delivered_only_gross_utility_mean'))} | "
            f"{_fmt(row.get('high_value_delivered_utility_mean'))} | {_fmt(row.get('net_system_utility_provisional_mean'))} | "
            f"{_fmt(row.get('min_battery_soc_mean'))} | {_fmt(row.get('max_thermal_mean'))} | "
            f"{_fmt(row.get('relative_improvement_over_NO_MODE'))} | {_fmt(row.get('relative_improvement_over_NO_LYA'))} | "
            f"{_fmt(row.get('relative_gap_to_UTIL'))} |"
        )
    _write_text(path, lines)


def _write_leaderboard(path_csv: Path, path_md: Path, rows: list[dict[str, Any]]) -> None:
    leaderboard = sorted(
        rows,
        key=lambda row: (
            row["scenario"],
            int(row.get("rank_by_primary_delivered_only_utility", 999)),
            int(row.get("rank_by_high_value_delivered_utility", 999)),
        ),
    )
    _write_rows(path_csv, leaderboard)
    lines = [
        "# Formal Multi-seed Leaderboard",
        "",
        "| scenario | delivered rank | high-value rank | variant | delivered-only mean | high-value mean | net provisional mean | resource safety |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in leaderboard:
        min_soc = _to_float(row.get("min_battery_soc_mean"))
        max_thermal = _to_float(row.get("max_thermal_mean"))
        safety = "ok" if (_finite(min_soc) and min_soc > 0.0 and _finite(max_thermal) and max_thermal <= 0.92) else "check"
        lines.append(
            f"| {row['scenario']} | {row.get('rank_by_primary_delivered_only_utility', '')} | "
            f"{row.get('rank_by_high_value_delivered_utility', '')} | {row['variant_short']} | "
            f"{_fmt(row.get('delivered_only_gross_utility_mean'))} | {_fmt(row.get('high_value_delivered_utility_mean'))} | "
            f"{_fmt(row.get('net_system_utility_provisional_mean'))} | {safety} |"
        )
    _write_text(path_md, lines)


def _write_dry_run_report(
    scenarios: list[str],
    variants: list[str],
    seeds: list[int],
    write_candidate_logs: bool,
    write_queue_logs: bool,
) -> Path:
    paths = [_scenario_dir(scenario, seed) for scenario in scenarios for seed in seeds]
    lines = [
        "# Formal Multi-seed Dry-run Report",
        "",
        "## Status",
        "",
        "- dry-run passed: yes",
        "- full simulation executed: no",
        "- output overwrite attempted: no",
        f"- formal output root: `{_relative(OUTPUT_ROOT)}`",
        f"- scenarios: {', '.join(scenarios)}",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- variants: {', '.join(_short_name(variant) for variant in variants)}",
        f"- candidate action log CSV writing enabled: {write_candidate_logs}",
        f"- queue priority log CSV writing enabled: {write_queue_logs}",
        "- delivered-only post-processing planned: yes",
        "- payload-type attribution post-processing planned: yes",
        "",
        "## Planned Run Directories",
        "",
    ]
    for path in paths:
        exists = path.exists() and any(path.iterdir())
        lines.append(f"- `{_relative(path)}` existing_nonempty={exists}")
    lines.extend(
        [
            "",
            "## Planned Aggregate Outputs",
            "",
            "- `outputs/formal_multiseed/per_run_summary.csv`",
            "- `outputs/formal_multiseed/aggregate_summary.csv`",
            "- `outputs/formal_multiseed/aggregate_summary.md`",
            "- `outputs/formal_multiseed/formal_leaderboard.csv`",
            "- `outputs/formal_multiseed/formal_leaderboard.md`",
            "",
            "## Interpretation",
            "",
            "This dry-run validates argument parsing, variant/scenario/seed expansion, deterministic output paths, default-disabled heavy log writing, and non-overwrite checks. It intentionally does not build timelines or run episodes.",
        ]
    )
    path = PREP_DIR / "formal_multiseed_dry_run_report.md"
    _write_text(path, lines)
    return path


def _run_execute(args: argparse.Namespace, scenarios: list[str], variants: list[str], seeds: list[int]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            all_rows.extend(
                _run_one_scenario_seed(
                    scenario=scenario,
                    seed=seed,
                    variants=variants,
                    allow_overwrite=args.allow_overwrite,
                    write_candidate_logs=args.write_candidate_logs,
                    write_queue_logs=args.write_queue_logs,
                )
            )
    _write_rows(OUTPUT_ROOT / "per_run_summary.csv", all_rows)
    aggregate = _aggregate_rows(all_rows)
    _write_rows(OUTPUT_ROOT / "aggregate_summary.csv", aggregate)
    _write_aggregate_markdown(OUTPUT_ROOT / "aggregate_summary.md", aggregate)
    _write_leaderboard(OUTPUT_ROOT / "formal_leaderboard.csv", OUTPUT_ROOT / "formal_leaderboard.md", aggregate)


def main() -> int:
    global OUTPUT_ROOT

    parser = argparse.ArgumentParser(description="Prepare or execute formal BUPT-1 multi-seed runs.")
    parser.add_argument("--dry-run", action="store_true", help="validate formal plan without running simulations")
    parser.add_argument("--execute", action="store_true", help="execute simulations; required for non-dry run")
    parser.add_argument("--scenarios", default="nominal,stress", help="comma-separated scenario names")
    parser.add_argument(
        "--variants",
        default=",".join(MINIMUM_FORMAL_VARIANTS),
        help="comma-separated variant short or full names",
    )
    parser.add_argument("--variant-set", choices=["minimum", "extended"], default=None)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="comma-separated integer seeds")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT), help="formal output root")
    parser.add_argument("--allow-overwrite", action="store_true", help="allow writing into non-empty per-run directories")
    parser.add_argument("--write-candidate-logs", action="store_true", help="write heavy candidate-action logs")
    parser.add_argument("--write-queue-logs", action="store_true", help="write heavy queue-priority logs")
    args = parser.parse_args()

    OUTPUT_ROOT = Path(args.output_root)
    if not OUTPUT_ROOT.is_absolute():
        OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT

    scenarios = _parse_list(args.scenarios, ["nominal", "stress"])
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")

    if args.variant_set == "minimum":
        raw_variants = MINIMUM_FORMAL_VARIANTS
    elif args.variant_set == "extended":
        raw_variants = EXTENDED_APPENDIX_VARIANTS
    else:
        raw_variants = _parse_list(args.variants, MINIMUM_FORMAL_VARIANTS)
    variants = _resolve_variants(raw_variants)
    seeds = _parse_seeds(args.seeds)

    configure_logging()

    if args.dry_run or not args.execute:
        report = _write_dry_run_report(
            scenarios=scenarios,
            variants=variants,
            seeds=seeds,
            write_candidate_logs=args.write_candidate_logs,
            write_queue_logs=args.write_queue_logs,
        )
        print(f"dry_run_passed=true")
        print(f"dry_run_report={_relative(report)}")
        print(f"logging_candidate_csv_enabled={args.write_candidate_logs}")
        print(f"logging_queue_csv_enabled={args.write_queue_logs}")
        print(f"variants={','.join(_short_name(variant) for variant in variants)}")
        print(f"scenarios={','.join(scenarios)}")
        print(f"seeds={','.join(str(seed) for seed in seeds)}")
        return 0

    _run_execute(args, scenarios, variants, seeds)
    print(f"execute_completed=true")
    print(f"output_root={_relative(OUTPUT_ROOT)}")
    print(f"aggregate_summary={_relative(OUTPUT_ROOT / 'aggregate_summary.csv')}")
    print(f"formal_leaderboard={_relative(OUTPUT_ROOT / 'formal_leaderboard.csv')}")
    print(f"logging_candidate_csv_enabled={args.write_candidate_logs}")
    print(f"logging_queue_csv_enabled={args.write_queue_logs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
