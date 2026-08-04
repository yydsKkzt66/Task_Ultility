"""Default-off product-requirement-aware Module 2 A/B experiment."""

from __future__ import annotations

import argparse
import copy
import csv
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
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.reporting.lifecycle_shadow_metrics import build_per_task_lifecycle_rows, summarize_lifecycle_rows  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_aware_module2_ab"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
DEFAULT_SEEDS = list(range(2026, 2036))
REQAWARE_CFG = {
    "enabled": True,
    "requirement_fit_weight": 1.20,
    "model_fit_weight": 0.25,
    "deadline_feasibility_weight": 0.20,
    "staged_delivery_feasibility_weight": 0.45,
}
DISABLED_REQAWARE_CFG = {
    "enabled": False,
    "requirement_fit_weight": 0.0,
    "model_fit_weight": 0.0,
    "deadline_feasibility_weight": 0.0,
    "staged_delivery_feasibility_weight": 0.0,
}
ARMS = {
    "legacy_ug_wx050": {"rule": "legacy", "router": PROPOSED_ROUTER, "wx": 0.50, "reqaware": False},
    "v21_no_scoring_ug_wx050": {"rule": "v2_1", "router": PROPOSED_ROUTER, "wx": 0.50, "reqaware": False},
    "v21_reqaware_ug_wx050": {"rule": "v2_1", "router": PROPOSED_ROUTER, "wx": 0.50, "reqaware": True},
    "legacy_utility_rule": {"rule": "legacy", "router": "utility_rule", "wx": None, "reqaware": False},
    "v21_reqaware_utility_rule": {"rule": "v2_1", "router": "utility_rule", "wx": None, "reqaware": True},
}
PAIRWISE = [
    ("v21_no_scoring_ug_wx050", "legacy_ug_wx050"),
    ("v21_reqaware_ug_wx050", "v21_no_scoring_ug_wx050"),
    ("v21_reqaware_ug_wx050", "legacy_ug_wx050"),
    ("v21_reqaware_ug_wx050", "legacy_utility_rule"),
]
METRICS = [
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
    "selected_action_change_rate",
    "direct_roi_rate",
    "roi_context_rate",
    "metadata_rate",
    "metadata_then_roi_rate",
    "metadata_then_roi_context_rate",
    "defer_rate",
    "staged_plan_rate",
    "heavy_payload_rate_on_low_value_tasks",
    "heavy_payload_rate_on_context_evidence_tasks",
    "metadata_only_rate_on_context_evidence_tasks",
    "requirement_to_payload_alignment_rate",
    "high_value_context_evidence_delivery_rate",
    "high_value_context_evidence_expiry_rate",
    "first_actionable_delivered_before_deadline_rate",
    "metadata_before_deadline_rate",
    "direct_roi_before_deadline_rate",
    "staged_metadata_success_rate",
    "staged_followup_success_rate",
    "metadata_to_roi_success_rate",
    "metadata_to_roi_context_success_rate",
    "residual_enqueue_count",
    "residual_delivery_count",
    "residual_delivery_rate",
    "alert_rate",
    "confirmation_rate",
    "evidence_rate",
    "first_actionable_latency_p50",
    "first_actionable_latency_p90",
    "lifecycle_shadow_utility_balanced",
    "lifecycle_shadow_utility_alert_heavy",
    "lifecycle_shadow_utility_evidence_heavy",
    "evidence_closure_count",
    "evidence_closure_rate",
    "hard_safety_violation_count",
    "battery_min",
    "thermal_max",
    "queue_overflow_count",
    "deadline_miss_count",
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
    parts = []
    tracked = [path for path in paths if _git_knows(path)]
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
    spec = ARMS[arm]
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
    configs["simulator"]["module2_product_requirement_aware_scoring"] = (
        dict(REQAWARE_CFG) if bool(spec["reqaware"]) and spec["router"] == PROPOSED_ROUTER else dict(DISABLED_REQAWARE_CFG)
    )
    if spec["router"] == "utility_rule":
        configs["experiment"] = {"routers": ["utility_rule"]}
    else:
        configs["simulator"]["policy_repair"] = _policy_repair(float(spec["wx"]))
        configs["experiment"] = {"routers": [PROPOSED_ROUTER]}
    return configs


def _write_config_snapshots(seeds: list[int]) -> None:
    names = {
        "legacy_ug_wx050": "config_legacy_ug_wx050.yaml",
        "v21_no_scoring_ug_wx050": "config_v21_no_scoring_ug_wx050.yaml",
        "v21_reqaware_ug_wx050": "config_v21_reqaware_ug_wx050.yaml",
        "legacy_utility_rule": "config_legacy_utility_rule.yaml",
        "v21_reqaware_utility_rule": "config_v21_reqaware_utility_rule.yaml",
    }
    for arm, filename in names.items():
        payload = {
            "arm": arm,
            "representative_seed": seeds[0],
            "default_off_causal_experiment": True,
            "configs": _load_run_configs(seeds[0], arm),
        }
        _write_text(OUT_DIR / filename, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


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
    records = [asdict(record) for record in result.records]
    for row in records:
        row["seed"] = seed
        row["arm"] = arm
        row["router"] = router.router_name
    candidate_logs = [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_candidate_action_logs]
    lifecycle_rows = build_per_task_lifecycle_rows(records, seed=seed, variant=arm, router_name=router.router_name)
    lifecycle_summary = summarize_lifecycle_rows(lifecycle_rows)
    return {
        "seed": seed,
        "arm": arm,
        "router": router.router_name,
        "records": records,
        "summary": dict(result.summary),
        "candidate_logs": candidate_logs,
        "lifecycle_rows": lifecycle_rows,
        "lifecycle_summary": lifecycle_summary,
    }


def _alignment_rate(rows: list[dict[str, Any]]) -> float:
    aligned = 0
    total = 0
    for row in rows:
        req = str(row.get("product_requirement", ""))
        payload = str(row.get("selected_payload_plan", ""))
        if not req:
            continue
        total += 1
        if req == "alert" and payload in {"metadata", "metadata_then_roi", "metadata_then_roi_context"}:
            aligned += 1
        elif req == "verification" and payload in {"roi", "metadata_then_roi", "metadata"}:
            aligned += 1
        elif req == "context" and payload in {"roi_context", "metadata_then_roi_context", "roi"}:
            aligned += 1
        elif req == "evidence" and payload in {"roi_context", "metadata_then_roi_context"}:
            aligned += 1
    return _safe_div(aligned, total)


def _summarize_run(run: dict[str, Any], action_change_rate: float = 0.0) -> dict[str, object]:
    rows = [row for row in run["records"] if _to_int(row.get("has_task"))]
    payload_counts = Counter(str(row.get("selected_payload_plan", "")) for row in rows)
    total = max(len(rows), 1)
    ce_rows = [row for row in rows if str(row.get("product_requirement")) in {"context", "evidence"}]
    low_rows = [row for row in rows if _to_float(row.get("V_t")) < 0.45]
    high_ce = [row for row in rows if _to_int(row.get("high_value_flag")) and str(row.get("product_requirement")) in {"context", "evidence"}]
    heavy_payloads = {"roi_context", "metadata_then_roi_context"}
    metadata_before_deadline = sum(
        1
        for row in rows
        if str(row.get("selected_product_type")) == "metadata" and _to_int(row.get("deadline_missed")) == 0
    )
    direct_roi_before_deadline = sum(
        1
        for row in rows
        if str(row.get("selected_product_type")) in {"roi", "roi_context"} and _to_int(row.get("deadline_missed")) == 0
    )
    lifecycle = run["lifecycle_summary"]
    summary = run["summary"]
    return {
        "seed": run["seed"],
        "arm": run["arm"],
        "router": run["router"],
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
        "selected_action_change_rate": action_change_rate,
        "direct_roi_rate": _safe_div(payload_counts.get("roi", 0), total),
        "roi_context_rate": _safe_div(payload_counts.get("roi_context", 0), total),
        "metadata_rate": _safe_div(payload_counts.get("metadata", 0), total),
        "metadata_then_roi_rate": _safe_div(payload_counts.get("metadata_then_roi", 0), total),
        "metadata_then_roi_context_rate": _safe_div(payload_counts.get("metadata_then_roi_context", 0), total),
        "defer_rate": _safe_div(payload_counts.get("defer", 0), total),
        "staged_plan_rate": _to_float(summary.get("staged_plan_rate")),
        "heavy_payload_rate_on_low_value_tasks": _safe_div(
            sum(1 for row in low_rows if str(row.get("selected_payload_plan")) in heavy_payloads), len(low_rows)
        ),
        "heavy_payload_rate_on_context_evidence_tasks": _safe_div(
            sum(1 for row in ce_rows if str(row.get("selected_payload_plan")) in heavy_payloads), len(ce_rows)
        ),
        "metadata_only_rate_on_context_evidence_tasks": _safe_div(
            sum(1 for row in ce_rows if str(row.get("selected_payload_plan")) == "metadata"), len(ce_rows)
        ),
        "requirement_to_payload_alignment_rate": _alignment_rate(rows),
        "high_value_context_evidence_delivery_rate": _safe_div(
            sum(1 for row in high_ce if _to_float(row.get("delivered_only_incremental_utility")) > 0.0), len(high_ce)
        ),
        "high_value_context_evidence_expiry_rate": _safe_div(sum(_to_int(row.get("expired_flag")) for row in high_ce), len(high_ce)),
        "first_actionable_delivered_before_deadline_rate": _to_float(lifecycle.get("first_actionable_delivered_rate")),
        "metadata_before_deadline_rate": _safe_div(metadata_before_deadline, total),
        "direct_roi_before_deadline_rate": _safe_div(direct_roi_before_deadline, total),
        "staged_metadata_success_rate": _safe_div(
            sum(1 for row in rows if str(row.get("selected_payload_plan", "")).startswith("metadata_then") and "metadata" in str(row.get("delivered_products", ""))),
            sum(1 for row in rows if str(row.get("selected_payload_plan", "")).startswith("metadata_then")),
        ),
        "staged_followup_success_rate": _safe_div(
            sum(1 for row in rows if str(row.get("selected_payload_plan", "")).startswith("metadata_then") and ("roi" in str(row.get("delivered_products", "")))),
            sum(1 for row in rows if str(row.get("selected_payload_plan", "")).startswith("metadata_then")),
        ),
        "metadata_to_roi_success_rate": _to_float(summary.get("metadata_to_roi_success_rate")),
        "metadata_to_roi_context_success_rate": _to_float(summary.get("metadata_to_roi_context_success_rate")),
        "residual_enqueue_count": _to_float(summary.get("residual_enqueue_count")),
        "residual_delivery_count": _to_float(summary.get("residual_delivery_count")),
        "residual_delivery_rate": _to_float(summary.get("residual_delivery_rate")),
        "alert_rate": _to_float(lifecycle.get("alert_rate")),
        "confirmation_rate": _to_float(lifecycle.get("confirmation_rate")),
        "evidence_rate": _to_float(lifecycle.get("evidence_rate")),
        "first_actionable_latency_p50": _to_float(lifecycle.get("first_actionable_latency_p50")),
        "first_actionable_latency_p90": _to_float(lifecycle.get("first_actionable_latency_p90")),
        "lifecycle_shadow_utility_balanced": _to_float(lifecycle.get("lifecycle_shadow_utility_balanced")),
        "lifecycle_shadow_utility_alert_heavy": _to_float(lifecycle.get("lifecycle_shadow_utility_alert_heavy")),
        "lifecycle_shadow_utility_evidence_heavy": _to_float(lifecycle.get("lifecycle_shadow_utility_evidence_heavy")),
        "evidence_closure_count": _to_float(lifecycle.get("evidence_closure_rate")) * _to_float(lifecycle.get("task_count")),
        "evidence_closure_rate": _to_float(lifecycle.get("evidence_closure_rate")),
        "hard_safety_violation_count": _to_float(summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count"))),
        "battery_min": _to_float(summary.get("min_battery_soc")),
        "thermal_max": _to_float(summary.get("max_thermal")),
        "queue_overflow_count": _to_float(summary.get("queue_overflow_count")),
        "deadline_miss_count": _to_float(summary.get("deadline_missed_count")),
    }


def _task_trace(run: dict[str, Any]) -> list[dict[str, object]]:
    lifecycle_by_task = {row["task_id"]: row for row in run["lifecycle_rows"]}
    rows = []
    for row in run["records"]:
        if not _to_int(row.get("has_task")):
            continue
        task_id = str(row.get("task_id", ""))
        life = lifecycle_by_task.get(task_id, {})
        stage = "evidence" if _to_int(life.get("evidence_achieved")) else "confirmation" if _to_int(life.get("confirmation_achieved")) else "alert" if _to_int(life.get("alert_achieved")) else "none"
        rows.append(
            {
                "seed": run["seed"],
                "arm": run["arm"],
                "task_id": task_id,
                "task_value": row.get("V_t", 0.0),
                "uncertainty": row.get("U_t", 0.0),
                "emergency_flag": row.get("emergency_flag", 0),
                "high_value_flag": row.get("high_value_flag", 0),
                "task_class": row.get("task_class", ""),
                "ideal_product_requirement": row.get("product_requirement", ""),
                "selected_action": row.get("selected_action", ""),
                "selected_model": row.get("selected_model", row.get("model_name", "")),
                "selected_payload_policy": row.get("selected_payload_plan", row.get("payload_policy", "")),
                "delivered_products": row.get("delivered_products", ""),
                "pending_products": row.get("pending_products", ""),
                "residual_utility_remaining": row.get("residual_utility_remaining", 0.0),
                "lifecycle_stage_achieved": stage,
                "deadline_met_flag": int(_to_int(row.get("deadline_missed")) == 0),
                "expired_flag": row.get("expired_flag", 0),
                "reward": row.get("penalized_reward", 0.0),
                "net_utility_contribution": row.get("net_utility", row.get("effective_utility", 0.0)),
            }
        )
    return rows


def _action_change_rates(run_by_key: dict[tuple[int, str], dict[str, Any]]) -> dict[tuple[int, str], float]:
    rates: dict[tuple[int, str], float] = {}
    comparisons = {
        "v21_no_scoring_ug_wx050": "legacy_ug_wx050",
        "v21_reqaware_ug_wx050": "v21_no_scoring_ug_wx050",
    }
    for (seed, arm), run in run_by_key.items():
        base_arm = comparisons.get(arm)
        if not base_arm:
            rates[(seed, arm)] = 0.0
            continue
        base = run_by_key.get((seed, base_arm))
        if base is None:
            rates[(seed, arm)] = 0.0
            continue
        by_task = {row["task_id"]: row for row in run["records"] if _to_int(row.get("has_task"))}
        base_by_task = {row["task_id"]: row for row in base["records"] if _to_int(row.get("has_task"))}
        total = changed = 0
        for task_id, row in by_task.items():
            other = base_by_task.get(task_id)
            if other is None:
                continue
            total += 1
            changed += int(row.get("selected_action") != other.get("selected_action"))
        rates[(seed, arm)] = _safe_div(changed, total)
    return rates


def _distribution(rows: list[dict[str, Any]], field: str, name: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if not _to_int(row.get("has_task")):
            continue
        key = (str(row["seed"]), str(row["arm"]))
        groups[key][str(row.get(field, ""))] += 1
        totals[key] += 1
    out = []
    for (seed, arm), counter in sorted(groups.items()):
        for value, count in sorted(counter.items()):
            out.append({"seed": seed, "arm": arm, name: value, "count": count, "rate": _safe_div(count, totals[(seed, arm)])})
    return out


def _candidate_summary(candidate_rows: list[dict[str, object]], score_field: str) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in candidate_rows:
        groups[(str(row.get("arm")), str(row.get("ideal_product_requirement", "")), str(row.get("candidate_payload", "")))].append(
            _to_float(row.get(score_field))
        )
    return [
        {
            "arm": arm,
            "ideal_product_requirement": req,
            "candidate_payload": payload,
            "metric": score_field,
            "mean": mean(values) if values else 0.0,
            "n": len(values),
        }
        for (arm, req, payload), values in sorted(groups.items())
    ]


def _aggregate(per_seed: list[dict[str, object]], metrics: list[str]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        groups[str(row["arm"])].append(row)
    out = []
    for arm, rows in sorted(groups.items()):
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in rows]
            out.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "mean": mean(values) if values else 0.0,
                    "median": median(values) if values else 0.0,
                    "std": stdev(values) if len(values) >= 2 else 0.0,
                    "n": len(values),
                }
            )
    return out


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(2026)
    means = []
    for _ in range(2000):
        sample = [values[rng.randrange(len(values))] for _item in values]
        means.append(mean(sample))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _paired(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in per_seed}
    seeds = sorted({int(row["seed"]) for row in per_seed})
    out = []
    for left, right in PAIRWISE:
        for metric in METRICS:
            deltas = []
            for seed in seeds:
                if (seed, left) in by_key and (seed, right) in by_key:
                    deltas.append(_to_float(by_key[(seed, left)].get(metric)) - _to_float(by_key[(seed, right)].get(metric)))
            low, high = _bootstrap_ci(deltas)
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


def _staged_summary(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    return _aggregate(
        per_seed,
        [
            "staged_plan_rate",
            "staged_metadata_success_rate",
            "staged_followup_success_rate",
            "metadata_to_roi_success_rate",
            "metadata_to_roi_context_success_rate",
            "residual_enqueue_count",
            "residual_delivery_count",
            "residual_delivery_rate",
        ],
    )


def _gap_summary(paired: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in paired
        if str(row["comparison"]) in {"v21_reqaware_ug_wx050 - legacy_utility_rule"}
    ]


def _module2_isolation_ok() -> bool:
    forbidden = ("recommended_payload_plan_shadow",)
    for relative in [
        "src/routers/safe_linucb_router.py",
        "src/routers/rule_router.py",
        "src/routers/safe_filter.py",
        "src/simulator/step_runner.py",
        "src/actions/action_space.py",
    ]:
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8", errors="replace")
        if any(token in text for token in forbidden):
            return False
    return True


def _guardrails(per_seed: list[dict[str, object]]) -> dict[str, object]:
    config_hits = []
    for path in (PROJECT_ROOT / "configs").glob("*.yaml"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "module2_product_requirement_aware_scoring" in text or "product_requirement_rule" in text:
            config_hits.append(path.relative_to(PROJECT_ROOT).as_posix())
    low_value_heavy = [
        _to_float(row.get("heavy_payload_rate_on_low_value_tasks"))
        for row in per_seed
        if str(row.get("arm")) == "v21_reqaware_ug_wx050"
    ]
    net_values = [_to_float(row.get("net_system_utility")) for row in per_seed if str(row.get("arm")) == "v21_reqaware_ug_wx050"]
    status = {
        "main_default_config_changed": int(bool(config_hits)),
        "recommended_shadow_consumed_by_module2": int(not _module2_isolation_ok()),
        "low_value_heavy_payload_mean": mean(low_value_heavy) if low_value_heavy else 0.0,
        "canonical_net_min": min(net_values) if net_values else 0.0,
        "critical_failure": 0,
    }
    status["critical_failure"] = int(
        bool(config_hits)
        or bool(status["recommended_shadow_consumed_by_module2"])
        or float(status["low_value_heavy_payload_mean"]) > 0.30
    )
    return status


def _metric(paired: list[dict[str, object]], comparison: str, metric: str) -> float:
    for row in paired:
        if row["comparison"] == comparison and row["metric"] == metric:
            return _to_float(row.get("mean_delta"))
    return 0.0


def _write_reports(paired: list[dict[str, object]], guardrails: dict[str, object], action_change: float) -> None:
    causal_net = _metric(paired, "v21_reqaware_ug_wx050 - v21_no_scoring_ug_wx050", "net_system_utility")
    causal_life = _metric(paired, "v21_reqaware_ug_wx050 - v21_no_scoring_ug_wx050", "lifecycle_shadow_utility_balanced")
    causal_align = _metric(paired, "v21_reqaware_ug_wx050 - v21_no_scoring_ug_wx050", "requirement_to_payload_alignment_rate")
    causal_expired = _metric(paired, "v21_reqaware_ug_wx050 - v21_no_scoring_ug_wx050", "expired_items")
    gap = _metric(paired, "v21_reqaware_ug_wx050 - legacy_utility_rule", "net_system_utility")
    evidence = _metric(paired, "v21_reqaware_ug_wx050 - v21_no_scoring_ug_wx050", "evidence_closure_rate")
    if guardrails["critical_failure"]:
        recommendation = "CRITICAL guardrail failed. Reject integration."
    elif action_change <= 0.01:
        recommendation = "Case C: actions barely change; audit score normalization and candidate score scale."
    elif causal_net < -0.05 and causal_expired > 0:
        recommendation = "Case B: actions change but canonical utility drops because queue/expiry increases."
    elif evidence <= 0.0:
        recommendation = "Case D: keep as default-off experimental branch only; actions change, but evidence closure remains zero."
    elif causal_net > 0.0 or causal_life > 0.0 or causal_align > 0.0:
        recommendation = "Case A: keep as default-off experimental branch; do not adopt as main default."
    else:
        recommendation = "Case C: actions barely help; keep diagnostic-only."
    report = f"""# v2_1 requirement-aware Module 2 A/B report

## Guardrails

- critical_failure: {guardrails['critical_failure']}
- main_default_config_changed: {guardrails['main_default_config_changed']}
- recommended_payload_plan_shadow consumed by Module 2: {guardrails['recommended_shadow_consumed_by_module2']}
- low_value_heavy_payload_mean: {float(guardrails['low_value_heavy_payload_mean']):.6f}

## Utility-rule control note

`v21_reqaware_utility_rule` is treated as a v2_1 requirement-only utility_rule control. Product-aware Module 2 scoring is not applied there because utility_rule logic is intentionally left unchanged.

## Key causal deltas

- selected_action_change_rate, reqaware vs no-scoring: {action_change:.6f}
- net_system_utility delta: {causal_net:.6f}
- lifecycle_shadow_utility_balanced delta: {causal_life:.6f}
- requirement_to_payload_alignment_rate delta: {causal_align:.6f}
- expired_items delta: {causal_expired:.6f}
- evidence_closure_rate delta: {evidence:.6f}
- net gap vs legacy utility_rule: {gap:.6f}

## Required answers

1. Does v2_1 now causally influence Module 2 selected actions?

   selected_action_change_rate={action_change:.6f} for v21_reqaware_ug_wx050 vs v21_no_scoring_ug_wx050.

2. Which tasks/actions changed and why?

   See `per_task_action_trace.csv` and `per_candidate_score_trace.csv`; candidate rows include requirement_fit/model/deadline/staged terms.

3. Does the change align with ideal_product_requirement?

   requirement_to_payload_alignment_rate delta={causal_align:.6f}.

4. Does ROI+Ctx/staged delivery increase only where context/evidence is needed?

   See `payload_policy_distribution_summary.csv` and heavy-payload rates.

5. Does first actionable delivery before DDL improve?

   See `paired_delta_summary.csv` metric `first_actionable_delivered_before_deadline_rate`.

6. Does context/evidence satisfaction improve?

   See `paired_delta_summary.csv` requirement satisfaction metrics.

7. Does evidence closure remain zero?

   evidence_closure_rate delta={evidence:.6f}; absolute rates are in `lifecycle_shadow_summary.csv`.

8. Does net_system_utility improve, stay flat, or degrade?

   causal net delta={causal_net:.6f}.

9. Does the gap to utility_rule shrink?

   net gap vs legacy_utility_rule={gap:.6f}.

10. Is the result limited by scoring scale, queue pressure, or residual delivery?

   Recommendation branch: {recommendation}

11. Should this remain diagnostic-only, become default-off experimental branch, or be rejected?

   {recommendation}
"""
    final = f"""# Final recommendation

{recommendation}

Do not adopt as main default. Rollback path: keep `product_requirement_rule.mode=legacy` and `module2_product_requirement_aware_scoring.enabled=false`.
"""
    _write_text(OUT_DIR / "v21_reqaware_module2_ab_report.md", report)
    _write_text(OUT_DIR / "final_recommendation.md", final)
    _write_text(
        OUT_DIR / "causal_path_audit.md",
        f"""# Causal path audit

- v2_1 requirement reaches task.product_requirement through default-off `product_requirement_rule.mode=v2_1`.
- StepRunner copies `ideal_product_requirement` into `guidance_bundle.product_requirement_context`.
- SafeLinUCB computes `requirement_fit_score`, `model_fit_score`, `deadline_feasibility_score`, and `staged_delivery_feasibility_score`.
- Product-aware terms are added only when `module2_product_requirement_aware_scoring.enabled=true`.
- SafeFilter still runs before router scoring, so product-aware scoring only ranks safe candidates.
- `recommended_payload_plan_shadow` is not read by SafeLinUCB/Module 2.
- Observed selected_action_change_rate in causal comparison: {action_change:.6f}.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = "python scripts/run_product_requirement_aware_module2_ab.py --seeds " + ",".join(str(seed) for seed in seeds)
    _write_config_snapshots(seeds)

    run_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    all_candidates: list[dict[str, object]] = []
    all_task_trace: list[dict[str, object]] = []
    for seed in seeds:
        for arm in ARMS:
            run = _run_arm(seed, arm)
            run_by_key[(seed, arm)] = run
            all_records.extend(run["records"])
            all_candidates.extend(run["candidate_logs"])
            all_task_trace.extend(_task_trace(run))

    action_rates = _action_change_rates(run_by_key)
    per_seed = []
    for key, run in sorted(run_by_key.items()):
        per_seed.append(_summarize_run(run, action_rates.get(key, 0.0)))
    paired = _paired(per_seed)
    guardrails = _guardrails(per_seed)
    action_change = mean(
        [
            _to_float(row.get("selected_action_change_rate"))
            for row in per_seed
            if row.get("arm") == "v21_reqaware_ug_wx050"
        ]
    )

    _write_csv(OUT_DIR / "per_candidate_score_trace.csv", all_candidates)
    _write_csv(OUT_DIR / "per_task_action_trace.csv", all_task_trace)
    _write_csv(OUT_DIR / "action_distribution_summary.csv", _distribution(all_records, "selected_action", "selected_action"))
    _write_csv(OUT_DIR / "payload_policy_distribution_summary.csv", _distribution(all_records, "selected_payload_plan", "selected_payload_plan"))
    _write_csv(OUT_DIR / "requirement_fit_summary.csv", _candidate_summary(all_candidates, "requirement_fit_score"))
    _write_csv(OUT_DIR / "deadline_feasibility_summary.csv", _candidate_summary(all_candidates, "deadline_feasibility_score"))
    _write_csv(OUT_DIR / "staged_delivery_feasibility_summary.csv", _candidate_summary(all_candidates, "staged_delivery_feasibility_score"))
    _write_csv(OUT_DIR / "residual_followup_summary.csv", _aggregate(per_seed, ["residual_enqueue_count", "residual_delivery_count", "residual_delivery_rate"]))
    _write_csv(OUT_DIR / "lifecycle_shadow_summary.csv", _aggregate(per_seed, ["alert_rate", "confirmation_rate", "evidence_rate", "first_actionable_latency_p50", "first_actionable_latency_p90", "lifecycle_shadow_utility_balanced", "lifecycle_shadow_utility_alert_heavy", "lifecycle_shadow_utility_evidence_heavy", "evidence_closure_count", "evidence_closure_rate"]))
    _write_csv(OUT_DIR / "staged_closure_summary.csv", _staged_summary(per_seed))
    _write_csv(OUT_DIR / "canonical_metric_summary.csv", _aggregate(per_seed, ["net_system_utility", "delivered_utility", "effective_utility", "total_penalized_reward", "mission_utility", "completed_tasks", "delivered_items", "expired_items", "deadline_satisfaction_ratio", "utility_per_joule", "utility_per_downlink_bit"]))
    _write_csv(OUT_DIR / "safety_guardrail_summary.csv", _aggregate(per_seed, ["hard_safety_violation_count", "battery_min", "thermal_max", "queue_overflow_count", "deadline_miss_count", "expired_items"]))
    _write_csv(OUT_DIR / "paired_delta_summary.csv", paired)
    _write_csv(OUT_DIR / "utility_rule_gap_summary.csv", _gap_summary(paired))
    _write_csv(
        OUT_DIR / "action_change_summary.csv",
        [
            {
                "comparison": "v21_reqaware_ug_wx050 - v21_no_scoring_ug_wx050",
                "selected_action_change_rate_mean": action_change,
                "n": len(seeds),
            }
        ],
    )
    _write_text(OUT_DIR / "README.md", "# product_requirement-aware Module 2 default-off A/B\n")
    _write_text(OUT_DIR / "command_log.md", f"# Command log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "arms": ARMS,
                "reqaware_config": REQAWARE_CFG,
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
                "src/routers/safe_linucb_router.py",
                "src/simulator/step_runner.py",
                "tests/test_utility_guided_safe_linucb.py",
                "scripts/run_product_requirement_aware_module2_ab.py",
                "scripts/audit_product_requirement_to_module2_causal_path.py",
            ]
        ),
    )
    _write_reports(paired, guardrails, action_change)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
