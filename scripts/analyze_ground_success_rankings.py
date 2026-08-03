"""Build ground-success-only rankings from uneven-pressure delivery outputs.

This analysis reads existing experiment artifacts and ranks methods only by
tasks/products that actually reached the ground, with deadline and stage
success enforced at task level.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_uneven_pressure_ground_delivery_experiment as base  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "_uneven_pressure_ground_delivery_v1"

SUCCESS_COLUMNS = [
    "method",
    "method_type",
    "seed",
    "pressure_scale",
    "total_task_count",
    "ground_success_task_count",
    "ground_success_task_ratio",
    "target_stage_success_count",
    "target_stage_success_ratio",
    "high_value_task_count",
    "high_value_ground_success_count",
    "high_value_ground_success_ratio",
    "high_value_target_success_count",
    "high_value_target_success_ratio",
    "quality_weighted_ground_success_value",
    "quality_weighted_high_value_success_value",
    "ground_success_event_gross_utility",
    "ground_success_event_net_utility",
    "high_value_ground_success_event_gross_utility",
    "high_value_ground_success_event_net_utility",
    "avg_ground_success_quality",
    "avg_ground_success_latency_h",
    "avg_high_value_ground_success_quality",
    "avg_high_value_ground_success_latency_h",
    "direct_energy_per_ground_success_task_wh",
    "system_energy_per_ground_success_task_wh",
    "direct_energy_per_high_value_ground_success_task_wh",
    "system_energy_per_high_value_ground_success_task_wh",
    "min_SOC",
    "final_SOC",
    "final_backlog_count",
    "max_backlog_count",
    "hard_safety_violation_count",
]


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None, "NA"}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None, "NA"}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _fmt(value: object, digits: int = 4) -> str:
    return f"{_to_float(value):.{digits}f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _product_from_stage(stage: str) -> str:
    return {"metadata": "metadata", "ROI": "roi", "ROI+Ctx": "roi_context"}.get(stage, "metadata")


def _target_success_before_deadline(row: dict[str, str]) -> int:
    target_time = row.get("target_ground_receive_time_h", "")
    if target_time == "":
        return 0
    return int(_to_float(target_time) <= _to_float(row.get("deadline_h")) + 1.0e-12)


def _event_utility(event: dict[str, str], task: dict[str, str]) -> tuple[float, float]:
    product = _product_from_stage(str(event.get("delivered_stage", "")))
    method = str(event.get("method", ""))
    payload = _to_float(event.get("product_size_MB"))
    _compute, _comm, account_cost = base._product_costs(method, product, payload)
    gross = (
        _to_float(task.get("V_eff"))
        * _to_float(event.get("incremental_information_utility"))
        * _to_float(event.get("delivered_quality"))
    )
    return gross, gross - account_cost


def build_ground_success_metrics(input_dir: Path) -> list[dict[str, Any]]:
    tasks = _read_csv(input_dir / "ground_delivery_task_delivery_by_method_seed.csv")
    events = _read_csv(input_dir / "ground_delivery_event_log.csv")
    summaries = _read_csv(input_dir / "ground_delivery_summary_by_method_seed.csv")
    summary_by_key = {
        (row["method"], _to_int(row["seed"]), _to_float(row["pressure_scale"])): row
        for row in summaries
    }
    task_by_key = {
        (row["method"], _to_int(row["seed"]), _to_float(row["pressure_scale"]), row["task_id"]): row
        for row in tasks
    }
    events_by_case: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for event in events:
        events_by_case[(event["method"], _to_int(event["seed"]), _to_float(event["pressure_scale"]))].append(event)

    tasks_by_case: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        tasks_by_case[(task["method"], _to_int(task["seed"]), _to_float(task["pressure_scale"]))].append(task)

    rows: list[dict[str, Any]] = []
    for key, case_tasks in sorted(tasks_by_case.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        method, seed, pressure = key
        success = [task for task in case_tasks if _to_int(task["ground_received_before_deadline_flag"])]
        target_success = [task for task in case_tasks if _target_success_before_deadline(task)]
        high_tasks = [task for task in case_tasks if _to_int(task["is_high_value"])]
        high_success = [task for task in high_tasks if _to_int(task["ground_received_before_deadline_flag"])]
        high_target_success = [task for task in high_tasks if _target_success_before_deadline(task)]
        success_ids = {task["task_id"] for task in success}
        high_success_ids = {task["task_id"] for task in high_success}

        event_gross = 0.0
        event_net = 0.0
        high_event_gross = 0.0
        high_event_net = 0.0
        for event in events_by_case.get(key, []):
            if not _to_int(event.get("before_deadline_flag")):
                continue
            task = task_by_key.get((method, seed, pressure, event["task_id"]))
            if task is None or event["task_id"] not in success_ids:
                continue
            gross, net = _event_utility(event, task)
            event_gross += gross
            event_net += net
            if event["task_id"] in high_success_ids:
                high_event_gross += gross
                high_event_net += net

        summary = summary_by_key.get(key, {})
        row = {
            "method": method,
            "method_type": base.METHOD_TYPES.get(method, ""),
            "seed": seed,
            "pressure_scale": pressure,
            "total_task_count": len(case_tasks),
            "ground_success_task_count": len(success),
            "ground_success_task_ratio": _safe_div(len(success), len(case_tasks)),
            "target_stage_success_count": len(target_success),
            "target_stage_success_ratio": _safe_div(len(target_success), len(case_tasks)),
            "high_value_task_count": len(high_tasks),
            "high_value_ground_success_count": len(high_success),
            "high_value_ground_success_ratio": _safe_div(len(high_success), len(high_tasks)),
            "high_value_target_success_count": len(high_target_success),
            "high_value_target_success_ratio": _safe_div(len(high_target_success), len(high_tasks)),
            "quality_weighted_ground_success_value": sum(_to_float(task["V_eff"]) * _to_float(task["ground_received_quality"]) for task in success),
            "quality_weighted_high_value_success_value": sum(_to_float(task["V_eff"]) * _to_float(task["ground_received_quality"]) for task in high_success),
            "ground_success_event_gross_utility": event_gross,
            "ground_success_event_net_utility": event_net,
            "high_value_ground_success_event_gross_utility": high_event_gross,
            "high_value_ground_success_event_net_utility": high_event_net,
            "avg_ground_success_quality": statistics.mean([_to_float(task["ground_received_quality"]) for task in success]) if success else 0.0,
            "avg_ground_success_latency_h": statistics.mean([_to_float(task["capture_to_ground_latency_h"]) for task in success]) if success else 0.0,
            "avg_high_value_ground_success_quality": statistics.mean([_to_float(task["ground_received_quality"]) for task in high_success]) if high_success else 0.0,
            "avg_high_value_ground_success_latency_h": statistics.mean([_to_float(task["capture_to_ground_latency_h"]) for task in high_success]) if high_success else 0.0,
            "direct_energy_per_ground_success_task_wh": _safe_div(sum(_to_float(task["task_energy_direct_wh"]) for task in success), len(success)),
            "system_energy_per_ground_success_task_wh": _safe_div(sum(_to_float(task["task_energy_system_wh"]) for task in success), len(success)),
            "direct_energy_per_high_value_ground_success_task_wh": _safe_div(sum(_to_float(task["task_energy_direct_wh"]) for task in high_success), len(high_success)),
            "system_energy_per_high_value_ground_success_task_wh": _safe_div(sum(_to_float(task["task_energy_system_wh"]) for task in high_success), len(high_success)),
            "min_SOC": _to_float(summary.get("min_SOC")),
            "final_SOC": _to_float(summary.get("final_SOC")),
            "final_backlog_count": _to_int(summary.get("final_backlog_count")),
            "max_backlog_count": _to_int(summary.get("max_backlog_count")),
            "hard_safety_violation_count": _to_int(summary.get("hard_safety_violation_count")),
        }
        rows.append(row)
    return rows


def add_case_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        ("ground_success_task_ratio", True),
        ("target_stage_success_ratio", True),
        ("high_value_ground_success_ratio", True),
        ("high_value_target_success_ratio", True),
        ("quality_weighted_ground_success_value", True),
        ("quality_weighted_high_value_success_value", True),
        ("ground_success_event_net_utility", True),
        ("high_value_ground_success_event_net_utility", True),
        ("avg_ground_success_quality", True),
        ("avg_ground_success_latency_h", False),
        ("system_energy_per_ground_success_task_wh", False),
        ("min_SOC", True),
    ]
    by_case: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[(_to_int(row["seed"]), _to_float(row["pressure_scale"]))].append(row)
    for case_rows in by_case.values():
        for metric, higher_better in metrics:
            ordered = sorted(case_rows, key=lambda row: _to_float(row[metric]), reverse=higher_better)
            for rank, row in enumerate(ordered, start=1):
                row[f"rank_{metric}"] = rank
    return rows


def build_multi_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "ground_success_task_ratio",
        "target_stage_success_ratio",
        "high_value_ground_success_ratio",
        "high_value_target_success_ratio",
        "quality_weighted_ground_success_value",
        "quality_weighted_high_value_success_value",
        "ground_success_event_net_utility",
        "high_value_ground_success_event_net_utility",
        "avg_ground_success_quality",
        "avg_ground_success_latency_h",
        "system_energy_per_ground_success_task_wh",
        "min_SOC",
    ]
    by_group: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for metric in metrics:
            by_group[(row["method"], _to_float(row["pressure_scale"]), metric)].append(row)
    out: list[dict[str, Any]] = []
    for (method, pressure, metric), group in sorted(by_group.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        values = [_to_float(row[metric]) for row in group]
        ranks = [_to_float(row.get(f"rank_{metric}")) for row in group]
        out.append(
            {
                "method": method,
                "method_type": base.METHOD_TYPES.get(method, ""),
                "pressure_scale": pressure,
                "metric": metric,
                "mean": statistics.mean(values) if values else 0.0,
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "rank_mean": statistics.mean(ranks) if ranks else 0.0,
                "rank_std": statistics.pstdev(ranks) if len(ranks) > 1 else 0.0,
            }
        )
    return out


def write_report(input_dir: Path, rows: list[dict[str, Any]], multi: list[dict[str, Any]]) -> None:
    base_case = {
        row["method"]: row
        for row in rows
        if _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    }
    prop = base_case["proposed_guarantee_aware_full"]
    base_rank = sorted(base_case.values(), key=lambda row: _to_float(row["quality_weighted_high_value_success_value"]), reverse=True)
    prop_multi = {
        row["metric"]: row
        for row in multi
        if row["method"] == "proposed_guarantee_aware_full" and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    }
    lines = [
        "# Ground-success-only ranking report",
        "",
        "All rankings in this report count only tasks/products that actually reached the ground. Task success requires `ground_received_before_deadline_flag=1`; target success requires target-stage ground receive before deadline.",
        "",
        "## Base case proposed",
        "",
        f"- ground_success_task_ratio: {_fmt(prop['ground_success_task_ratio'])} ({prop['ground_success_task_count']}/{prop['total_task_count']})",
        f"- target_stage_success_ratio: {_fmt(prop['target_stage_success_ratio'])}",
        f"- high_value_ground_success_ratio: {_fmt(prop['high_value_ground_success_ratio'])} ({prop['high_value_ground_success_count']}/{prop['high_value_task_count']})",
        f"- high_value_target_success_ratio: {_fmt(prop['high_value_target_success_ratio'])}",
        f"- quality_weighted_ground_success_value: {_fmt(prop['quality_weighted_ground_success_value'])}",
        f"- quality_weighted_high_value_success_value: {_fmt(prop['quality_weighted_high_value_success_value'])}",
        f"- ground_success_event_net_utility: {_fmt(prop['ground_success_event_net_utility'])}",
        f"- high_value_ground_success_event_net_utility: {_fmt(prop['high_value_ground_success_event_net_utility'])}",
        f"- min_SOC: {_fmt(prop['min_SOC'])}, final_backlog_count: {prop['final_backlog_count']}",
        "",
        "## Base case ranking by high-value ground-success value",
        "",
    ]
    for rank, row in enumerate(base_rank, start=1):
        lines.append(
            f"{rank}. {row['method']}: high_value_success_value={_fmt(row['quality_weighted_high_value_success_value'])}, "
            f"high_value_success_ratio={_fmt(row['high_value_ground_success_ratio'])}, min_SOC={_fmt(row['min_SOC'])}"
        )
    lines.extend(
        [
            "",
            "## Multi-seed proposed at pressure_scale=1.0",
            "",
        ]
    )
    for metric in [
        "ground_success_task_ratio",
        "target_stage_success_ratio",
        "high_value_ground_success_ratio",
        "high_value_target_success_ratio",
        "quality_weighted_ground_success_value",
        "quality_weighted_high_value_success_value",
        "ground_success_event_net_utility",
        "high_value_ground_success_event_net_utility",
        "avg_ground_success_quality",
        "min_SOC",
    ]:
        row = prop_multi.get(metric, {})
        lines.append(
            f"- {metric}: mean={_fmt(row.get('mean'))}, std={_fmt(row.get('std'))}, rank_mean={_fmt(row.get('rank_mean'), 2)}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Under ground-success-only ranking, proposed remains strong on delivered quality and safety, but it is not top-ranked on high-value ground-success coverage. The gap is caused by conservative low-SOC gating and backlog growth during wave_2/wave_3, which prevent enough high-value tasks from reaching target stage on the ground.",
        ]
    )
    _write_text(input_dir / "ground_success_only_report.md", "\n".join(lines))


def run(input_dir: Path = DEFAULT_INPUT_DIR) -> dict[str, Any]:
    rows = add_case_ranks(build_ground_success_metrics(input_dir))
    multi = build_multi_seed(rows)
    _write_csv(input_dir / "ground_success_only_metrics_by_method_seed.csv", rows, SUCCESS_COLUMNS + [key for key in rows[0] if key.startswith("rank_")])
    _write_csv(input_dir / "ground_success_only_multi_seed_ranking.csv", multi)
    write_report(input_dir, rows, multi)
    return {"rows": rows, "multi": multi, "input_dir": input_dir}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args(argv)
    result = run(args.input_dir)
    base_prop = next(
        row
        for row in result["rows"]
        if row["method"] == "proposed_guarantee_aware_full" and _to_int(row["seed"]) == 0 and abs(_to_float(row["pressure_scale"]) - 1.0) < 1.0e-9
    )
    print(f"input_dir={args.input_dir}")
    print(f"proposed_ground_success_ratio={_fmt(base_prop['ground_success_task_ratio'])}")
    print(f"proposed_high_value_ground_success_ratio={_fmt(base_prop['high_value_ground_success_ratio'])}")
    print(f"proposed_quality_weighted_high_value_success_value={_fmt(base_prop['quality_weighted_high_value_success_value'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
