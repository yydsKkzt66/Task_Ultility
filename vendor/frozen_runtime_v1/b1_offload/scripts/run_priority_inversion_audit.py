"""Priority inversion audit for the value-aware staged delivery queue."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_value_skew_progressive_followup_ab as base_ab  # noqa: E402
from src.common.config import ensure_output_dirs  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_priority_inversion_audit"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
SCENARIOS = ["natural", "targeted_resource_scarcity"]
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030]
ARMS = [
    "utility_rule",
    "current_progressive",
    "persistent_priority_queue_before_fix",
    "persistent_priority_queue_after_priority_inversion_fix",
]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _staged_queue_cfg(*, priority_inversion_fix_enabled: bool) -> dict[str, Any]:
    return {
        "enabled": True,
        "full_method_aliases": [PROPOSED_ROUTER],
        "allow_utility_rule": False,
        "residual_utility_threshold": 0.01,
        "min_value_density": 0.0,
        "eriv_threshold": 0.0,
        "require_future_contact_before_deadline": True,
        "allow_pending_without_future_contact": False,
        "max_admissions_per_slot": 2,
        "max_admissions_per_contact_window": 5,
        "deadline_urgency_lambda": 0.15,
        "timeliness_floor": 0.05,
        "priority_inversion_fix_enabled": bool(priority_inversion_fix_enabled),
        "preemption_priority_floor": 1000.0,
    }


def _load_configs(seed: int, scenario: str, arm: str, num_slots: int) -> dict[str, dict[str, Any]]:
    if arm == "utility_rule":
        configs = base_ab._load_run_configs(seed, scenario, "legacy_proposed", num_slots)
        configs["experiment"] = {"routers": ["utility_rule"]}
        return configs
    configs = base_ab._load_run_configs(
        seed,
        scenario,
        "soft_moderate_value_skew_plus_gate_plus_progressive_followup",
        num_slots,
    )
    if arm in {
        "persistent_priority_queue_before_fix",
        "persistent_priority_queue_after_priority_inversion_fix",
    }:
        progressive = dict(configs["simulator"].get("progressive_payload", {}))
        progressive["value_aware_staged_delivery_queue"] = _staged_queue_cfg(
            priority_inversion_fix_enabled=arm == "persistent_priority_queue_after_priority_inversion_fix"
        )
        configs["simulator"]["progressive_payload"] = progressive
    return configs


def _run_one(seed: int, scenario: str, arm: str, num_slots: int) -> dict[str, Any]:
    configs = _load_configs(seed, scenario, arm, num_slots)
    run_dir = OUT_DIR / "runs" / scenario / f"s{seed}" / arm
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    rows = [asdict(record) for record in result.records]
    _write_csv(run_dir / "queue_priority_trace.csv", runner.last_queue_priority_logs)
    _write_csv(run_dir / "contact_window_candidate_rank_trace.csv", runner.last_contact_window_candidate_rank_trace)
    _write_csv(run_dir / "contact_window_delivery_order_trace.csv", runner.last_contact_window_delivery_order_trace)
    _write_csv(
        run_dir / "ordinary_queue_vs_staged_queue_capacity_trace.csv",
        runner.last_ordinary_queue_vs_staged_queue_capacity_trace,
    )
    _write_csv(run_dir / "staged_delivery_request_trace.csv", runner.last_staged_delivery_request_trace)
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "summary": dict(result.summary),
        "records": rows,
        "candidate": runner.last_contact_window_candidate_rank_trace,
        "delivery": runner.last_contact_window_delivery_order_trace,
        "capacity": runner.last_ordinary_queue_vs_staged_queue_capacity_trace,
        "requests": runner.last_staged_delivery_request_trace,
    }


def _augment(rows: list[dict[str, object]], run: dict[str, Any]) -> list[dict[str, object]]:
    return [dict(row, scenario=run["scenario"], seed=run["seed"], arm=run["arm"]) for row in rows]


def _key(row: dict[str, object]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("scenario", "")),
        str(row.get("seed", "")),
        str(row.get("arm", "")),
        str(row.get("contact_window_id", "")),
        str(row.get("request_id", "")),
    )


def _is_feasible_high(row: dict[str, object]) -> bool:
    return bool(_to_int(row.get("feasible"))) and bool(_to_int(row.get("high_value_context_residual_flag")))


def _is_delivered(row: dict[str, object]) -> bool:
    return bool(_to_int(row.get("delivered_this_slot")) or _to_int(row.get("selected_for_downlink")))


def _analyze(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, object]]]:
    candidates = [row for run in runs for row in _augment(run["candidate"], run)]
    deliveries = [row for run in runs for row in _augment(run["delivery"], run)]
    capacities = [row for run in runs for row in _augment(run["capacity"], run)]
    feasible_high: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    infeasible_high_keys: set[tuple[str, str, str, str, str]] = set()
    for row in candidates:
        if not _to_int(row.get("high_value_context_residual_flag")):
            continue
        key = _key(row)
        if _is_feasible_high(row):
            old = feasible_high.get(key)
            if old is None or _to_float(row.get("priority_score")) > _to_float(old.get("priority_score")):
                feasible_high[key] = row
        else:
            infeasible_high_keys.add(key)

    delivered_staged_keys: set[tuple[str, str, str, str, str]] = set()
    delivered_by_run_window: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in deliveries:
        if not _is_delivered(row):
            continue
        run_window = (
            str(row.get("scenario", "")),
            str(row.get("seed", "")),
            str(row.get("arm", "")),
            str(row.get("contact_window_id", "")),
        )
        delivered_by_run_window[run_window].append(row)
        request_id = str(row.get("staged_request_id", ""))
        if request_id:
            delivered_staged_keys.add((*run_window, request_id))

    feasible_rows: list[dict[str, object]] = []
    missed_rows: list[dict[str, object]] = []
    for key, row in feasible_high.items():
        delivered = key in delivered_staged_keys
        out = dict(row, delivered_in_window=int(delivered))
        feasible_rows.append(out)
        if not delivered:
            missed_rows.append(out)

    inversions: list[dict[str, object]] = []
    low_value_while_high: list[dict[str, object]] = []
    for key, high_row in feasible_high.items():
        scenario, seed, arm, window, request_id = key
        if key in delivered_staged_keys:
            continue
        high_priority = _to_float(high_row.get("priority_score"))
        for delivered in delivered_by_run_window.get((scenario, seed, arm, window), []):
            delivered_priority = _to_float(delivered.get("queue_priority_total"))
            if high_priority <= delivered_priority:
                continue
            is_staged = _to_int(delivered.get("is_staged_delivery_queue"))
            delivered_value_level = str(delivered.get("value_level", "")).lower()
            delivered_high = _to_int(delivered.get("high_value_flag"))
            row = {
                "scenario": scenario,
                "seed": seed,
                "arm": arm,
                "contact_window_id": window,
                "slot": delivered.get("slot", ""),
                "missed_request_id": request_id,
                "missed_task_id": high_row.get("task_id", ""),
                "missed_product_type": high_row.get("product_type", ""),
                "missed_priority_score": high_priority,
                "missed_payload_size_mb": high_row.get("payload_size_mb", ""),
                "delivered_task_id": delivered.get("task_id", ""),
                "delivered_payload_id": delivered.get("payload_id", ""),
                "delivered_payload_type": delivered.get("payload_type", ""),
                "delivered_value_level": delivered_value_level,
                "delivered_priority_score": delivered_priority,
                "delivered_is_staged": is_staged,
                "delivered_tx_mb_estimate": delivered.get("tx_mb_estimate", ""),
            }
            inversions.append(row)
            if (not is_staged) and (delivered_value_level in {"", "low", "medium"} or not delivered_high):
                low_value_while_high.append(row)

    summary_rows: list[dict[str, object]] = []
    by_run: dict[tuple[str, str, str], dict[str, object]] = defaultdict(
        lambda: {
            "feasible": 0,
            "delivered": 0,
            "missed": 0,
            "inversion": 0,
            "low_value": 0,
            "ordinary_before": 0.0,
            "staged_used": 0.0,
            "infeasible": 0,
            "roi_context_delivered": 0,
            "hard_safety": 0,
            "low_medium_accidental": 0,
        }
    )
    for row in feasible_rows:
        key3 = (str(row["scenario"]), str(row["seed"]), str(row["arm"]))
        by_run[key3]["feasible"] += 1
        by_run[key3]["delivered"] += _to_int(row.get("delivered_in_window"))
    for row in missed_rows:
        by_run[(str(row["scenario"]), str(row["seed"]), str(row["arm"]))]["missed"] += 1
    for row in inversions:
        by_run[(str(row["scenario"]), str(row["seed"]), str(row["arm"]))]["inversion"] += 1
    for row in low_value_while_high:
        by_run[(str(row["scenario"]), str(row["seed"]), str(row["arm"]))]["low_value"] += 1
    for row in capacities:
        key3 = (str(row["scenario"]), str(row["seed"]), str(row["arm"]))
        by_run[key3]["ordinary_before"] += _to_float(row.get("ordinary_queue_capacity_used_before_staged_queue_mb"))
        by_run[key3]["staged_used"] += _to_float(row.get("staged_queue_capacity_used_mb"))
    for key in infeasible_high_keys - set(feasible_high):
        by_run[(key[0], key[1], key[2])]["infeasible"] += 1
    for run in runs:
        key3 = (str(run["scenario"]), str(run["seed"]), str(run["arm"]))
        records = run["records"]
        by_run[key3]["roi_context_delivered"] = sum(
            str(record.get("delivered_product_types", "")).split(";").count("roi_context") for record in records
        )
        by_run[key3]["hard_safety"] = sum(_to_int(record.get("hard_safety_violation")) for record in records)
        by_run[key3]["low_medium_accidental"] = sum(
            1
            for record in records
            if str(record.get("value_level", "")).lower() in {"low", "medium"}
            and "roi_context" in str(record.get("delivered_product_types", ""))
        )
    for (scenario, seed, arm), values in sorted(by_run.items()):
        feasible = int(values["feasible"])
        delivered = int(values["delivered"])
        summary_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "arm": arm,
                "feasible_high_value_residual_count": feasible,
                "delivered_feasible_high_value_residual_count": delivered,
                "missed_feasible_high_value_residual_count": int(values["missed"]),
                "high_value_opportunity_capture_rate": delivered / feasible if feasible else 0.0,
                "low_value_delivered_while_high_value_feasible_count": int(values["low_value"]),
                "priority_inversion_count": int(values["inversion"]),
                "ordinary_queue_capacity_used_before_staged_queue_mb": float(values["ordinary_before"]),
                "staged_queue_capacity_used_mb": float(values["staged_used"]),
                "leftover_high_value_feasible_count": int(values["missed"]),
                "leftover_high_value_infeasible_count": int(values["infeasible"]),
                "roi_context_delivered_count": int(values["roi_context_delivered"]),
                "low_medium_accidental_roi_context_count": int(values["low_medium_accidental"]),
                "hard_safety_violation_count": int(values["hard_safety"]),
            }
        )
    return {
        "candidate": candidates,
        "delivery": deliveries,
        "capacity": capacities,
        "feasible": feasible_rows,
        "missed": missed_rows,
        "inversion": inversions,
        "low_value": low_value_while_high,
        "summary": summary_rows,
    }


def _mean(rows: list[dict[str, object]], arm: str, field: str) -> float:
    vals = [_to_float(row.get(field)) for row in rows if row.get("arm") == arm]
    return sum(vals) / len(vals) if vals else 0.0


def _write_recommendation(summary_rows: list[dict[str, object]]) -> None:
    before = "persistent_priority_queue_before_fix"
    after = "persistent_priority_queue_after_priority_inversion_fix"
    lines = [
        "# Priority Inversion Audit Recommendation",
        "",
        "Scope: reward, gate, closure, and hard-safety definitions were not changed.",
        "",
        "Mean metrics across the small-scale A/B:",
        f"- before_fix priority_inversion_count: {_mean(summary_rows, before, 'priority_inversion_count'):.3f}",
        f"- after_fix priority_inversion_count: {_mean(summary_rows, after, 'priority_inversion_count'):.3f}",
        f"- before_fix high_value_opportunity_capture_rate: {_mean(summary_rows, before, 'high_value_opportunity_capture_rate'):.3f}",
        f"- after_fix high_value_opportunity_capture_rate: {_mean(summary_rows, after, 'high_value_opportunity_capture_rate'):.3f}",
        f"- before_fix roi_context_delivered_count: {_mean(summary_rows, before, 'roi_context_delivered_count'):.3f}",
        f"- after_fix roi_context_delivered_count: {_mean(summary_rows, after, 'roi_context_delivered_count'):.3f}",
        f"- after_fix hard_safety_violation_count: {_mean(summary_rows, after, 'hard_safety_violation_count'):.3f}",
        f"- after_fix low_medium_accidental_roi_context_count: {_mean(summary_rows, after, 'low_medium_accidental_roi_context_count'):.3f}",
        "",
    ]
    after_inversion = _mean(summary_rows, after, "priority_inversion_count")
    after_safety = _mean(summary_rows, after, "hard_safety_violation_count")
    after_low = _mean(summary_rows, after, "low_medium_accidental_roi_context_count")
    before_inversion = _mean(summary_rows, before, "priority_inversion_count")
    if before_inversion <= 0.0 and after_inversion <= 0.0:
        lines.append(
            "Recommendation: no true priority inversion was found after duplicate/lifecycle feasibility is included. "
            "The low ROI+Ctx count is better explained by few feasible high-value residual opportunities and many "
            "duplicate/lifecycle/deadline-infeasible candidates, not ordinary low-value payload stealing capacity. "
            "The priority-inversion fix remains a default-off guardrail for the persistent staged queue arm, but it "
            "does not materially change this small-scale A/B."
        )
    elif after_inversion <= 0.0 and after_safety <= 0.0 and after_low <= 0.0:
        lines.append(
            "Recommendation: keep the priority-inversion fix enabled for the persistent staged queue arm. "
            "It is still default-off and unavailable to utility_rule/legacy unless explicitly configured."
        )
    else:
        lines.append(
            "Recommendation: do not promote this fix yet. Inspect missed_feasible_high_value_residual_trace.csv "
            "and low_value_delivered_while_high_value_feasible.csv before broader A/B."
        )
    _write_text(OUT_DIR / "final_recommendation.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--num-slots", type=int, default=80)
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for seed in seeds:
            for arm in ARMS:
                print(f"run scenario={scenario} seed={seed} arm={arm}", flush=True)
                runs.append(_run_one(seed, scenario, arm, args.num_slots))
    audit = _analyze(runs)
    _write_csv(OUT_DIR / "contact_window_candidate_rank_trace.csv", audit["candidate"])
    _write_csv(OUT_DIR / "contact_window_delivery_order_trace.csv", audit["delivery"])
    _write_csv(OUT_DIR / "feasible_high_value_residual_trace.csv", audit["feasible"])
    _write_csv(OUT_DIR / "missed_feasible_high_value_residual_trace.csv", audit["missed"])
    _write_csv(OUT_DIR / "priority_inversion_trace.csv", audit["inversion"])
    _write_csv(OUT_DIR / "low_value_delivered_while_high_value_feasible.csv", audit["low_value"])
    _write_csv(OUT_DIR / "ordinary_queue_vs_staged_queue_capacity_trace.csv", audit["capacity"])
    _write_csv(OUT_DIR / "priority_inversion_summary.csv", audit["summary"])
    _write_recommendation(audit["summary"])
    print(f"wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
