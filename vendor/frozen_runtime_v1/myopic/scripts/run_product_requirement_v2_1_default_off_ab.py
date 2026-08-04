"""Default-off A/B evaluation for product_requirement_v2_1.

The v2_1 arm only changes task.product_requirement through an explicit
task_stream feature flag. It does not consume recommended_payload_plan_shadow in
Module 2 and does not modify reward, scorer, router, utility_rule, SafeFilter,
or main default configs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.data.product_requirements import assign_product_requirement_v2_1, recommend_payload_plan_from_requirement  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.reporting.lifecycle_shadow_metrics import build_per_task_lifecycle_rows, summarize_lifecycle_rows  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_v2_1_default_off_ab"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
DEFAULT_SEEDS = list(range(2026, 2036))
ARM_SPECS = {
    "old_req_ug_wx050": {"rule": "legacy", "router": PROPOSED_ROUTER, "wx": 0.50},
    "v21_req_ug_wx050": {"rule": "v2_1", "router": PROPOSED_ROUTER, "wx": 0.50},
    "old_req_utility_rule": {"rule": "legacy", "router": "utility_rule", "wx": None},
    "v21_req_utility_rule": {"rule": "v2_1", "router": "utility_rule", "wx": None},
}
PAIRWISE = [
    ("v21_req_ug_wx050", "old_req_ug_wx050"),
    ("v21_req_utility_rule", "old_req_utility_rule"),
    ("v21_req_ug_wx050", "v21_req_utility_rule"),
    ("old_req_ug_wx050", "old_req_utility_rule"),
]
IMPORTANT_METRICS = [
    "net_system_utility",
    "delivered_utility",
    "effective_utility",
    "total_penalized_reward",
    "mission_utility",
    "completed_tasks",
    "delivered_items",
    "expired_items",
    "deadline_satisfaction_ratio",
    "utility_per_joule",
    "utility_per_downlink_bit",
    "lifecycle_shadow_utility_balanced",
    "lifecycle_shadow_utility_alert_heavy",
    "lifecycle_shadow_utility_evidence_heavy",
    "time_shadow_utility",
    "evidence_closure_rate",
    "context_evidence_satisfaction_rate",
    "high_value_expired_rate",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = completed.stdout
        if completed.stderr:
            output += "\nSTDERR:\n" + completed.stderr
        return output.strip()
    except Exception as exc:  # pragma: no cover
        return f"git command failed: {exc}"


def _git_knows(path: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0


def _synthetic_new_file_diff(path: str) -> str:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return ""
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = [
        f"diff --git a/{path} b/{path}",
        "new file mode 100644",
        "index 0000000..0000000",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    return "\n".join([*header, *[f"+{line}" for line in lines]])


def _audit_diff(paths: list[str]) -> str:
    tracked = [path for path in paths if _git_knows(path)]
    parts: list[str] = []
    if tracked:
        parts.append(_run_git(["diff", "--", *tracked]))
    for path in paths:
        if not _git_knows(path):
            parts.append(_synthetic_new_file_diff(path))
    return "\n\n".join(part for part in parts if part).strip()


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _policy_repair(wx: float) -> dict[str, Any]:
    cfg = copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "policy_repair_utility_guided_safe_linucb.yaml"))
    cfg["enable_utility_guided_scoring"] = True
    scoring = dict(cfg.get("utility_guided_scoring", {}))
    scoring.update(
        {
            "utility_prior_weight": 1.0,
            "learned_correction_weight": 0.20,
            "exploration_weight": float(wx),
            "score_component_normalization": "zscore_by_candidate_set",
            "fallback_if_zero_variance": True,
            "preserve_legacy_score_logging": True,
        }
    )
    cfg["utility_guided_scoring"] = scoring
    cfg["enable_staged_feasibility_filter"] = False
    return cfg


def _load_run_configs(seed: int, arm: str) -> dict[str, dict[str, Any]]:
    spec = ARM_SPECS[arm]
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_stress_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["dataset"]["seed"] = int(seed)
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["timeline"]["use_num_days"] = False
    configs["timeline"]["num_slots"] = 120
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    configs["task_stream"]["product_requirement_rule"] = {"mode": spec["rule"]}
    if spec["router"] == "utility_rule":
        configs["experiment"] = {"routers": ["utility_rule"]}
    else:
        configs["simulator"]["policy_repair"] = _policy_repair(float(spec["wx"]))
        configs["experiment"] = {"routers": [PROPOSED_ROUTER]}
    return configs


def _write_config_snapshots(seeds: list[int]) -> None:
    for arm in ARM_SPECS:
        payload = {
            "arm": arm,
            "representative_seed": seeds[0],
            "diagnostic_default_off_ab": True,
            "configs": _load_run_configs(seeds[0], arm),
        }
        _write_text(OUT_DIR / f"config_{arm}.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _trajectory_rows(result) -> list[dict[str, Any]]:
    return [asdict(record) for record in result.records]


def _task_maps(timeline) -> dict[str, Any]:
    return {step.task.task_id: step.task for step in timeline if step.task is not None}


def _mission_v21_for_task(task) -> dict[str, Any]:
    meta = dict(task.metadata or {})
    scene_context_need = meta.get("scene_context_need", meta.get("g_area_prior"))
    try:
        scene_context_need = None if scene_context_need in {None, ""} else float(scene_context_need)
    except (TypeError, ValueError):
        scene_context_need = None
    result = assign_product_requirement_v2_1(
        task_value=task.task_value,
        uncertainty=task.uncertainty,
        emergency_flag=task.emergency_flag,
        high_value_flag=task.high_value_flag,
        task_class=task.task_class,
        mission_source=str(meta.get("dataset_name", "")) if meta.get("dataset_name") else None,
        scene_context_need=scene_context_need,
        product_use_case=str(meta.get("product_use_case", "")) if meta.get("product_use_case") else None,
        timeliness_class=str(meta.get("timeliness_class", "")) if meta.get("timeliness_class") else None,
    )
    recommendation = recommend_payload_plan_from_requirement(
        ideal_product_requirement=result.ideal_product_requirement,
        metadata_feasible=None,
        roi_feasible=None,
        roi_context_feasible=None,
        metadata_then_roi_feasible=None,
        metadata_then_roi_context_feasible=None,
    )
    return {
        "ideal": result.ideal_product_requirement,
        "scores": ";".join(f"{key}:{value:.6f}" for key, value in sorted(result.requirement_scores.items())),
        "recommended_payload_plan_shadow": recommendation.recommended_payload_plan,
    }


def _stage_from_lifecycle(row: dict[str, Any]) -> str:
    if _to_int(row.get("evidence_achieved")):
        return "evidence"
    if _to_int(row.get("confirmation_achieved")):
        return "confirmation"
    if _to_int(row.get("alert_achieved")):
        return "alert"
    return "none"


def _run_arm(seed: int, arm: str) -> dict[str, Any]:
    configs = _load_run_configs(seed, arm)
    run_dir = OUT_DIR / "runs" / f"seed_{seed}" / arm
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    rows = _trajectory_rows(result)
    lifecycle_rows = build_per_task_lifecycle_rows(rows, seed=seed, variant=arm, router_name=router.router_name)
    lifecycle_summary = summarize_lifecycle_rows(lifecycle_rows)
    task_maps = _task_maps(timeline)
    return {
        "seed": seed,
        "arm": arm,
        "router": router.router_name,
        "configs": configs,
        "timeline": timeline,
        "task_maps": task_maps,
        "result": result,
        "rows": rows,
        "summary": dict(result.summary),
        "lifecycle_rows": lifecycle_rows,
        "lifecycle_summary": lifecycle_summary,
        "candidate_logs": list(getattr(runner, "last_candidate_action_logs", [])),
    }


def _summarize_records(seed: int, arm: str, router: str, rows: list[dict[str, Any]], summary: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, object]:
    task_rows = [row for row in rows if _to_int(row.get("has_task"))]
    high_value_rows = [row for row in task_rows if _to_int(row.get("high_value_flag"))]
    context_evidence_rows = [row for row in task_rows if str(row.get("product_requirement")) in {"context", "evidence"}]
    ce_delivered = [
        row
        for row in context_evidence_rows
        if str(row.get("delivered_products", "")).strip() or _to_float(row.get("delivered_only_incremental_utility")) > 0.0
    ]
    action_counts = Counter(str(row.get("selected_action", "")) for row in task_rows)
    payload_counts = Counter(str(row.get("selected_payload_plan", row.get("payload_policy", ""))) for row in task_rows)
    total_tasks = max(len(task_rows), 1)
    return {
        "seed": seed,
        "arm": arm,
        "router": router,
        "net_system_utility": _to_float(summary.get("net_system_utility")),
        "delivered_utility": _to_float(summary.get("delivered_only_incremental_utility")),
        "effective_utility": _to_float(summary.get("total_effective_utility")),
        "total_penalized_reward": _to_float(summary.get("total_penalized_reward")),
        "mission_utility": _to_float(summary.get("total_mission_utility")),
        "completed_tasks": _to_float(summary.get("completed_tasks")),
        "delivered_items": _to_float(summary.get("delivered_items")),
        "expired_items": _to_float(summary.get("expired_items")),
        "deadline_satisfaction_ratio": _to_float(summary.get("deadline_satisfaction_ratio")),
        "utility_per_joule": _to_float(summary.get("utility_per_joule")),
        "utility_per_downlink_bit": _to_float(summary.get("utility_per_downlink_bit")),
        "hard_safety_violation_count": _to_float(summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count"))),
        "true_hard_failure_count": _to_float(summary.get("final_resource_safety_violation_count", 0.0)),
        "battery_min": _to_float(summary.get("min_battery_soc")),
        "thermal_max": _to_float(summary.get("max_thermal")),
        "queue_overflow_count": _to_float(summary.get("queue_overflow_count")),
        "deadline_miss_count": _to_float(summary.get("deadline_missed_count")),
        "direct_roi_rate": _safe_div(payload_counts.get("roi", 0), total_tasks),
        "roi_context_rate": _safe_div(payload_counts.get("roi_context", 0), total_tasks),
        "metadata_rate": _safe_div(payload_counts.get("metadata", 0), total_tasks),
        "metadata_then_roi_rate": _safe_div(payload_counts.get("metadata_then_roi", 0), total_tasks),
        "metadata_then_roi_context_rate": _safe_div(payload_counts.get("metadata_then_roi_context", 0), total_tasks),
        "defer_rate": _safe_div(payload_counts.get("defer", 0), total_tasks),
        "staged_plan_rate": _to_float(summary.get("staged_plan_rate")),
        "residual_enqueue_count": _to_float(summary.get("residual_enqueue_count")),
        "residual_delivery_count": _to_float(summary.get("residual_delivery_count")),
        "residual_delivery_rate": _to_float(summary.get("residual_delivery_rate")),
        "metadata_to_roi_success_rate": _to_float(summary.get("metadata_to_roi_success_rate")),
        "metadata_to_roi_context_success_rate": _to_float(summary.get("metadata_to_roi_context_success_rate")),
        "context_evidence_satisfaction_rate": _safe_div(len(ce_delivered), len(context_evidence_rows)),
        "high_value_context_evidence_delivered_utility": sum(_to_float(row.get("context_evidence_delivered_utility")) for row in high_value_rows),
        "high_value_expired_rate": _safe_div(sum(_to_int(row.get("expired_flag")) for row in high_value_rows), len(high_value_rows)),
        "requirement_to_payload_alignment_rate": _alignment_rate(task_rows),
        "alert_rate": _to_float(lifecycle.get("alert_rate")),
        "confirmation_rate": _to_float(lifecycle.get("confirmation_rate")),
        "evidence_rate": _to_float(lifecycle.get("evidence_rate")),
        "first_actionable_delivered_rate": _to_float(lifecycle.get("first_actionable_delivered_rate")),
        "first_actionable_latency_p50": _to_float(lifecycle.get("first_actionable_latency_p50")),
        "first_actionable_latency_p90": _to_float(lifecycle.get("first_actionable_latency_p90")),
        "lifecycle_shadow_utility_balanced": _to_float(lifecycle.get("lifecycle_shadow_utility_balanced")),
        "lifecycle_shadow_utility_alert_heavy": _to_float(lifecycle.get("lifecycle_shadow_utility_alert_heavy")),
        "lifecycle_shadow_utility_evidence_heavy": _to_float(lifecycle.get("lifecycle_shadow_utility_evidence_heavy")),
        "time_shadow_utility": _to_float(lifecycle.get("time_shadow_utility")),
        "evidence_closure_count": _to_float(lifecycle.get("evidence_rate")) * _to_float(lifecycle.get("task_count")),
        "evidence_closure_rate": _to_float(lifecycle.get("evidence_closure_rate")),
        "selected_action_distribution": ";".join(f"{key}:{value}" for key, value in sorted(action_counts.items())),
        "payload_plan_distribution": ";".join(f"{key}:{value}" for key, value in sorted(payload_counts.items())),
    }


def _alignment_rate(task_rows: list[dict[str, Any]]) -> float:
    aligned = 0
    total = 0
    for row in task_rows:
        requirement = str(row.get("product_requirement", ""))
        plan = str(row.get("selected_payload_plan", row.get("payload_policy", "")))
        if not requirement:
            continue
        total += 1
        if requirement == "alert" and plan in {"metadata", "metadata_then_roi", "metadata_then_roi_context"}:
            aligned += 1
        elif requirement == "verification" and plan in {"roi", "metadata_then_roi", "metadata"}:
            aligned += 1
        elif requirement == "context" and plan in {"roi_context", "metadata_then_roi_context", "roi"}:
            aligned += 1
        elif requirement == "evidence" and plan in {"roi_context", "metadata_then_roi_context"}:
            aligned += 1
    return _safe_div(aligned, total)


def _per_task_trace(seed: int, arm: str, run_data: dict[str, Any], legacy_map: dict[str, Any], v21_map: dict[str, Any]) -> list[dict[str, object]]:
    lifecycle_by_task = {row["task_id"]: row for row in run_data["lifecycle_rows"]}
    out: list[dict[str, object]] = []
    for row in run_data["rows"]:
        if not _to_int(row.get("has_task")):
            continue
        task_id = str(row.get("task_id", ""))
        old_task = legacy_map.get(task_id)
        v21_task = v21_map.get(task_id)
        if old_task is None:
            continue
        v21_diag = _mission_v21_for_task(old_task)
        lifecycle_row = lifecycle_by_task.get(task_id, {})
        selected_action = str(row.get("selected_action", ""))
        selected_payload = str(row.get("selected_payload_plan", row.get("payload_policy", "")))
        out.append(
            {
                "seed": seed,
                "arm": arm,
                "task_id": task_id,
                "task_value": old_task.task_value,
                "uncertainty": old_task.uncertainty,
                "emergency_flag": int(old_task.emergency_flag),
                "high_value_flag": int(old_task.high_value_flag),
                "task_class": old_task.task_class,
                "old_product_requirement": old_task.product_requirement,
                "v21_ideal_product_requirement": v21_task.product_requirement if v21_task is not None else v21_diag["ideal"],
                "v21_requirement_scores": v21_diag["scores"],
                "recommended_payload_plan_shadow": v21_diag["recommended_payload_plan_shadow"],
                "selected_action": selected_action,
                "selected_model": row.get("selected_model", row.get("model_name", "")),
                "selected_payload_policy": selected_payload,
                "delivered_products": row.get("delivered_products", ""),
                "pending_products": row.get("pending_products", ""),
                "residual_utility_remaining": row.get("residual_utility_remaining", 0.0),
                "lifecycle_stage_achieved": _stage_from_lifecycle(lifecycle_row),
                "deadline_met_flag": int(_to_int(row.get("deadline_missed")) == 0 and str(row.get("delivery_status", "")) == "delivered"),
                "expired_flag": row.get("expired_flag", 0),
                "reward": row.get("penalized_reward", 0.0),
                "net_utility_contribution": row.get("net_utility", row.get("effective_utility", 0.0)),
            }
        )
    return out


def _requirement_transition(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[int, str]] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        key = (int(row["seed"]), str(row["task_id"]))
        if key in seen:
            continue
        seen.add(key)
        counts[(str(row["old_product_requirement"]), str(row["v21_ideal_product_requirement"]))] += 1
    total = max(sum(counts.values()), 1)
    return [
        {"old_product_requirement": old, "v21_ideal_product_requirement": new, "count": count, "rate": count / total}
        for (old, new), count in sorted(counts.items())
    ]


def _distribution_summary(rows: list[dict[str, Any]], field: str, output_name: str) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if not _to_int(row.get("has_task")):
            continue
        key = (str(row["seed"]), str(row["arm"]))
        counts[key][str(row.get(field, ""))] += 1
        totals[key] += 1
    out = []
    for (seed, arm), counter in sorted(counts.items()):
        for value, count in sorted(counter.items()):
            out.append({"seed": seed, "arm": arm, output_name: value, "count": count, "rate": _safe_div(count, totals[(seed, arm)])})
    return out


def _aggregate_summary(rows: list[dict[str, object]], metrics: list[str], group_field: str = "arm") -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_field])].append(row)
    out: list[dict[str, object]] = []
    for group, group_rows in sorted(groups.items()):
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in group_rows]
            out.append(
                {
                    group_field: group,
                    "metric": metric,
                    "mean": mean(values) if values else 0.0,
                    "median": median(values) if values else 0.0,
                    "std": stdev(values) if len(values) >= 2 else 0.0,
                    "min": min(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                    "n": len(values),
                }
            )
    return out


def _bootstrap_ci(values: list[float], iterations: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(2026)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _item in values]
        means.append(mean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _paired_delta_summary(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in per_seed}
    out = []
    for left, right in PAIRWISE:
        for metric in IMPORTANT_METRICS:
            deltas = []
            for seed in sorted({int(row["seed"]) for row in per_seed}):
                lrow = by_key.get((seed, left))
                rrow = by_key.get((seed, right))
                if lrow is None or rrow is None:
                    continue
                deltas.append(_to_float(lrow.get(metric)) - _to_float(rrow.get(metric)))
            low, high = _bootstrap_ci(deltas) if deltas else (0.0, 0.0)
            out.append(
                {
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "mean_delta": mean(deltas) if deltas else 0.0,
                    "median_delta": median(deltas) if deltas else 0.0,
                    "std_delta": stdev(deltas) if len(deltas) >= 2 else 0.0,
                    "win_rate": _safe_div(sum(1 for value in deltas if value > 0.0), len(deltas)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci_crosses_zero": int(low <= 0.0 <= high),
                    "n": len(deltas),
                }
            )
    return out


def _lifecycle_shadow_summary(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    return _aggregate_summary(
        per_seed,
        [
            "alert_rate",
            "confirmation_rate",
            "evidence_rate",
            "first_actionable_delivered_rate",
            "first_actionable_latency_p50",
            "first_actionable_latency_p90",
            "lifecycle_shadow_utility_balanced",
            "lifecycle_shadow_utility_alert_heavy",
            "lifecycle_shadow_utility_evidence_heavy",
            "time_shadow_utility",
            "evidence_closure_count",
            "evidence_closure_rate",
        ],
    )


def _staged_closure_summary(per_seed: list[dict[str, object]], all_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if _to_int(row.get("has_task")):
            by_arm[str(row["arm"])].append(row)
    out = []
    for arm, rows in sorted(by_arm.items()):
        staged = [row for row in rows if str(row.get("selected_payload_plan")) in {"metadata_then_roi", "metadata_then_roi_context"}]
        evidence_enqueued_not_delivered = [
            row
            for row in rows
            if str(row.get("selected_payload_plan")) in {"roi_context", "metadata_then_roi_context"}
            and "roi_context" not in str(row.get("delivered_products", ""))
        ]
        metadata_no_confirmation = [
            row for row in rows if "metadata" in str(row.get("delivered_products", "")) and "roi" not in str(row.get("delivered_products", ""))
        ]
        out.append(
            {
                "arm": arm,
                "confirmed_evidence_closed_count": sum(1 for row in rows if "roi_context" in str(row.get("delivered_products", ""))),
                "confirmed_evidence_failed_count": len(evidence_enqueued_not_delivered),
                "unknown_evidence_closure_count": 0,
                "evidence_closure_unknown_rate": 0.0,
                "metadata_delivered_but_no_confirmation_generated": len(metadata_no_confirmation),
                "confirmation_generated_but_not_delivered": sum(
                    1 for row in rows if str(row.get("selected_product_type")) == "roi" and "roi" not in str(row.get("delivered_products", ""))
                ),
                "evidence_enqueued_but_not_delivered": len(evidence_enqueued_not_delivered),
                "no_staged_action_selected": len(rows) - len(staged),
                "partial_staged_behavior_count": len(metadata_no_confirmation),
            }
        )
    return out


def _safety_summary(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    return _aggregate_summary(
        per_seed,
        [
            "hard_safety_violation_count",
            "true_hard_failure_count",
            "battery_min",
            "thermal_max",
            "queue_overflow_count",
            "deadline_miss_count",
            "expired_items",
            "defer_rate",
        ],
    )


def _utility_rule_comparison(paired: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = {
        "v21_req_ug_wx050 - v21_req_utility_rule",
        "old_req_ug_wx050 - old_req_utility_rule",
        "v21_req_utility_rule - old_req_utility_rule",
    }
    return [row for row in paired if str(row["comparison"]) in wanted]


def _action_change_summary(per_task_trace: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["seed"]), str(row["arm"]), str(row["task_id"])): row
        for row in per_task_trace
    }
    out = []
    for v21_arm, old_arm in [
        ("v21_req_ug_wx050", "old_req_ug_wx050"),
        ("v21_req_utility_rule", "old_req_utility_rule"),
    ]:
        total = 0
        action_changed = 0
        payload_changed = 0
        for (seed, arm, task_id), row in list(by_key.items()):
            if arm != v21_arm:
                continue
            old = by_key.get((seed, old_arm, task_id))
            if old is None:
                continue
            total += 1
            action_changed += int(row.get("selected_action") != old.get("selected_action"))
            payload_changed += int(row.get("selected_payload_policy") != old.get("selected_payload_policy"))
        out.append(
            {
                "comparison": f"{v21_arm} - {old_arm}",
                "paired_task_count": total,
                "selected_action_changed_count": action_changed,
                "selected_action_changed_rate": _safe_div(action_changed, total),
                "selected_payload_policy_changed_count": payload_changed,
                "selected_payload_policy_changed_rate": _safe_div(payload_changed, total),
            }
        )
    return out


def _module2_isolation_ok() -> bool:
    router_paths = [
        PROJECT_ROOT / "src" / "routers" / "safe_linucb_router.py",
        PROJECT_ROOT / "src" / "routers" / "rule_router.py",
        PROJECT_ROOT / "src" / "routers" / "safe_filter.py",
        PROJECT_ROOT / "src" / "simulator" / "step_runner.py",
        PROJECT_ROOT / "src" / "actions" / "action_space.py",
    ]
    forbidden = ("recommended_payload_plan_shadow", "assign_product_requirement_v2_1")
    return not any(any(token in path.read_text(encoding="utf-8", errors="replace") for token in forbidden) for path in router_paths)


def _critical_guardrail_status() -> dict[str, object]:
    def contains_forbidden_hook(path: Path) -> int:
        text = path.read_text(encoding="utf-8", errors="replace")
        return int("assign_product_requirement_v2_1" in text or "recommended_payload_plan_shadow" in text)

    config_hits = []
    for path in (PROJECT_ROOT / "configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "product_requirement_rule" in text or "product_requirement_v2_1" in text:
            config_hits.append(path.relative_to(PROJECT_ROOT).as_posix())
    status = {
        "module2_isolation_ok": int(_module2_isolation_ok()),
        "main_default_config_references_v21": int(bool(config_hits)),
        "reward_function_references_v21": contains_forbidden_hook(PROJECT_ROOT / "src" / "reward" / "reward_function.py"),
        "safe_linucb_references_v21": contains_forbidden_hook(PROJECT_ROOT / "src" / "routers" / "safe_linucb_router.py"),
        "utility_rule_references_v21": contains_forbidden_hook(PROJECT_ROOT / "src" / "routers" / "rule_router.py"),
        "safe_filter_references_v21": contains_forbidden_hook(PROJECT_ROOT / "src" / "routers" / "safe_filter.py"),
        "config_hits": ";".join(config_hits),
    }
    status["critical_failure"] = int(
        not status["module2_isolation_ok"]
        or status["main_default_config_references_v21"]
        or status["reward_function_references_v21"]
        or status["safe_linucb_references_v21"]
        or status["utility_rule_references_v21"]
        or status["safe_filter_references_v21"]
    )
    return status


def _metric_delta(paired_rows: list[dict[str, object]], comparison: str, metric: str) -> float:
    for row in paired_rows:
        if row["comparison"] == comparison and row["metric"] == metric:
            return _to_float(row.get("mean_delta"))
    return 0.0


def _write_report(
    per_seed: list[dict[str, object]],
    paired: list[dict[str, object]],
    transition: list[dict[str, object]],
    guardrails: dict[str, object],
    action_changes: list[dict[str, object]],
) -> None:
    ug_net_delta = _metric_delta(paired, "v21_req_ug_wx050 - old_req_ug_wx050", "net_system_utility")
    ug_delivered_delta = _metric_delta(paired, "v21_req_ug_wx050 - old_req_ug_wx050", "delivered_utility")
    ug_lifecycle_delta = _metric_delta(paired, "v21_req_ug_wx050 - old_req_ug_wx050", "lifecycle_shadow_utility_balanced")
    utility_net_delta = _metric_delta(paired, "v21_req_utility_rule - old_req_utility_rule", "net_system_utility")
    old_gap = _metric_delta(paired, "old_req_ug_wx050 - old_req_utility_rule", "net_system_utility")
    new_gap = _metric_delta(paired, "v21_req_ug_wx050 - v21_req_utility_rule", "net_system_utility")
    gap_narrowed = new_gap > old_gap
    safety_bad = any(_to_float(row.get("hard_safety_violation_count")) > 0.0 for row in per_seed)
    action_change_text = "; ".join(
        f"{row['comparison']}: action_change_rate={_to_float(row['selected_action_changed_rate']):.6f}, payload_change_rate={_to_float(row['selected_payload_policy_changed_rate']):.6f}"
        for row in action_changes
    )
    useful = (
        ug_net_delta > 0.0
        or ug_delivered_delta > 0.0
        or (ug_lifecycle_delta > 0.0 and ug_net_delta >= -0.05 * abs(_mean_for_arm(per_seed, "old_req_ug_wx050", "net_system_utility")))
        or gap_narrowed
    ) and not safety_bad
    critical = bool(guardrails.get("critical_failure"))
    if critical:
        recommendation = "CRITICAL guardrail failed. Do not recommend integration."
    elif useful and not (utility_net_delta > ug_net_delta and not gap_narrowed):
        recommendation = "Case A: recommend v2_1 as a default-off experimental branch; do not adopt as main default."
    elif utility_net_delta > ug_net_delta and not gap_narrowed:
        recommendation = "Case D: v2_1 helps utility_rule more than Proposed or widens the gap; investigate staged closure before integration."
    elif ug_net_delta < 0.0 and ug_lifecycle_delta <= 0.0:
        recommendation = "Case C: keep legacy requirement for experiments; retain v2_1 as design reference."
    else:
        recommendation = "Case B: keep v2_1 diagnostic-only; repair payload-level trace/residual delivery before integration."

    report = f"""# product_requirement_v2_1 default-off A/B report

## Guardrails

- Module 2 isolation ok: {bool(guardrails.get('module2_isolation_ok'))}
- Critical failure: {bool(guardrails.get('critical_failure'))}
- `recommended_payload_plan_shadow` consumed by router: {not bool(guardrails.get('module2_isolation_ok'))}

## Key paired deltas

- v21_req_ug_wx050 - old_req_ug_wx050 net_system_utility mean delta: {ug_net_delta:.6f}
- v21_req_ug_wx050 - old_req_ug_wx050 delivered_utility mean delta: {ug_delivered_delta:.6f}
- v21_req_ug_wx050 - old_req_ug_wx050 lifecycle_shadow_utility_balanced mean delta: {ug_lifecycle_delta:.6f}
- v21_req_utility_rule - old_req_utility_rule net_system_utility mean delta: {utility_net_delta:.6f}
- old_req_ug_wx050 - old_req_utility_rule net gap: {old_gap:.6f}
- v21_req_ug_wx050 - v21_req_utility_rule net gap: {new_gap:.6f}

## Required answers

1. Did v2_1 change requirement distribution in a reasonable way?

   Transition rows: {transition}. Distribution did not collapse if no single v2_1 requirement dominates the dataset summaries.

2. Did v2_1 change selected actions indirectly through mission requirement input?

   {action_change_text}. v2_1 is only an input requirement change; Module 2 still selects actions.

3. Did v2_1 improve ug_wx050 compared with old requirement?

   net delta={ug_net_delta:.6f}; delivered delta={ug_delivered_delta:.6f}; lifecycle balanced delta={ug_lifecycle_delta:.6f}.

4. Did v2_1 also improve utility_rule?

   utility_rule net delta={utility_net_delta:.6f}.

5. Does v2_1 narrow or widen the gap between ug_wx050 and utility_rule?

   old gap={old_gap:.6f}; v2_1 gap={new_gap:.6f}; narrowed={gap_narrowed}.

6. Does v2_1 increase ROI+Ctx / metadata_then_roi_context where it should?

   See `payload_plan_distribution_summary.csv` and `per_task_requirement_action_trace.csv`.

7. Does v2_1 improve context/evidence satisfaction?

   See `paired_delta_summary.csv` metric `context_evidence_satisfaction_rate`.

8. Does v2_1 improve lifecycle shadow metrics?

   balanced lifecycle delta for ug_wx050 is {ug_lifecycle_delta:.6f}.

9. Does evidence closure remain near zero?

   See `staged_closure_summary.csv` and lifecycle evidence closure metrics.

10. Does v2_1 cause more defer, expiry, queue pressure, or safety risk?

   See `safety_guardrail_summary.csv`, `canonical_metric_summary.csv`, and paired deltas for `expired_items`.

11. Should v2_1 remain diagnostic-only, move to default-off experimental branch, or be rejected?

   {recommendation}
"""
    final = f"""# Final recommendation

{recommendation}

Do not make v2_1 the main default. The rollback path is to keep `product_requirement_rule.mode=legacy`, which is the default and preserves existing behavior.
"""
    _write_text(OUT_DIR / "v21_ab_report.md", report)
    _write_text(OUT_DIR / "final_recommendation.md", final)


def _mean_for_arm(per_seed: list[dict[str, object]], arm: str, metric: str) -> float:
    values = [_to_float(row.get(metric)) for row in per_seed if row.get("arm") == arm]
    return mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = "python scripts/run_product_requirement_v2_1_default_off_ab.py --seeds " + ",".join(str(seed) for seed in seeds)
    _write_config_snapshots(seeds)

    per_seed: list[dict[str, object]] = []
    per_task_trace: list[dict[str, object]] = []
    all_record_rows: list[dict[str, Any]] = []
    for seed in seeds:
        legacy_timeline = TimelineBuilder(
            _load_run_configs(seed, "old_req_ug_wx050")["dataset"],
            _load_run_configs(seed, "old_req_ug_wx050")["timeline"],
            _load_run_configs(seed, "old_req_ug_wx050")["task_stream"],
        ).build()
        v21_timeline = TimelineBuilder(
            _load_run_configs(seed, "v21_req_ug_wx050")["dataset"],
            _load_run_configs(seed, "v21_req_ug_wx050")["timeline"],
            _load_run_configs(seed, "v21_req_ug_wx050")["task_stream"],
        ).build()
        legacy_map = _task_maps(legacy_timeline)
        v21_map = _task_maps(v21_timeline)
        for arm in ARM_SPECS:
            run_data = _run_arm(seed, arm)
            for row in run_data["rows"]:
                row["seed"] = seed
                row["arm"] = arm
            all_record_rows.extend(run_data["rows"])
            per_seed.append(
                _summarize_records(
                    seed,
                    arm,
                    str(run_data["router"]),
                    run_data["rows"],
                    run_data["summary"],
                    run_data["lifecycle_summary"],
                )
            )
            per_task_trace.extend(_per_task_trace(seed, arm, run_data, legacy_map, v21_map))

    transition = _requirement_transition(per_task_trace)
    paired = _paired_delta_summary(per_seed)
    guardrails = _critical_guardrail_status()
    _write_csv(OUT_DIR / "per_seed_summary.csv", per_seed)
    _write_csv(OUT_DIR / "per_task_requirement_action_trace.csv", per_task_trace)
    _write_csv(OUT_DIR / "old_vs_v21_requirement_transition.csv", transition)
    _write_csv(OUT_DIR / "action_distribution_summary.csv", _distribution_summary(all_record_rows, "selected_action", "selected_action"))
    _write_csv(OUT_DIR / "payload_plan_distribution_summary.csv", _distribution_summary(all_record_rows, "selected_payload_plan", "selected_payload_plan"))
    _write_csv(OUT_DIR / "lifecycle_shadow_summary.csv", _lifecycle_shadow_summary(per_seed))
    _write_csv(OUT_DIR / "staged_closure_summary.csv", _staged_closure_summary(per_seed, all_record_rows))
    _write_csv(OUT_DIR / "safety_guardrail_summary.csv", _safety_summary(per_seed))
    _write_csv(OUT_DIR / "canonical_metric_summary.csv", _aggregate_summary(per_seed, IMPORTANT_METRICS))
    _write_csv(OUT_DIR / "paired_delta_summary.csv", paired)
    _write_csv(OUT_DIR / "utility_rule_comparison_summary.csv", _utility_rule_comparison(paired))
    action_changes = _action_change_summary(per_task_trace)
    _write_csv(OUT_DIR / "action_change_summary.csv", action_changes)
    _write_text(
        OUT_DIR / "README.md",
        """# product_requirement_v2_1 default-off A/B

This output is a controlled paired A/B evaluation. v2_1 is enabled only through
`task_stream.product_requirement_rule.mode=v2_1` in copied run configs. The main
default configs are not modified.
""",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "arms": ARM_SPECS,
                "paired_comparisons": PAIRWISE,
                "guardrails": guardrails,
                "diagnostic_default_off": True,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _audit_diff(
            [
                "src/data/task_stream.py",
                "src/data/product_requirements.py",
                "tests/test_task_stream.py",
                "tests/test_product_requirements_v2.py",
                "scripts/run_product_requirement_v2_1_default_off_ab.py",
            ]
        ),
    )
    _write_report(per_seed, paired, transition, guardrails, action_changes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
