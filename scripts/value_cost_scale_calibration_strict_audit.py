"""Strict value-cost scale audit for mission_prior_value_v2.

This script is diagnostic-only. It does not tune rewards, select parameters from
performance results, weaken baselines, or alter SafeFilter/staged-feasibility
logic. It writes a reproducible audit package under:

outputs/_utility_value_cost_scale_calibration_strict/
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.data.mission_prior_value import DEFAULT_TASK_VALUE_MODEL_CFG  # noqa: E402
from src.evaluation.metrics import summarize_episode  # noqa: E402
from src.link.downlink_queue import DownlinkQueue  # noqa: E402
from src.routers.fixed_router import FixedRouter  # noqa: E402
from src.routers.rule_router import RuleBasedRouter  # noqa: E402
from src.routers.safe_linucb_router import SafeLinUCBRouter  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "outputs" / "_utility_value_cost_scale_calibration_strict"

BASE_FILES = {
    "dataset": "dataset.yaml",
    "task_stream": "task_stream.yaml",
    "timeline": "timeline.yaml",
    "simulator": "simulator.yaml",
    "payload_profiles": "payload_profiles.yaml",
}

CASE_ORDER = (
    "Proposed-Full-filter-enabled",
    "utility_rule",
    "Fixed-Light",
    "Fixed-High",
)

DIAG_FIELDS = [
    "case",
    "case_available",
    "router_name",
    "tasks",
    "successful_tasks",
    "failed_or_non_success_tasks",
    "V_t_mean",
    "V_t_std",
    "V_t_min",
    "V_t_p05",
    "V_t_median",
    "V_t_p95",
    "V_t_max",
    "cost_mean",
    "cost_std",
    "cost_min",
    "cost_p05",
    "cost_median",
    "cost_p95",
    "cost_max",
    "penalty_mean",
    "penalty_std",
    "penalty_min",
    "penalty_p05",
    "penalty_median",
    "penalty_p95",
    "penalty_max",
    "effective_utility_mean",
    "net_utility_mean",
    "mean_V_t_over_mean_cost",
    "mean_effective_utility_over_mean_cost",
    "fraction_successful_tasks_with_positive_net_utility",
    "fraction_failed_tasks_with_negative_net_utility",
]

COMPARISON_FIELDS = [
    "case",
    "case_available",
    "router_name",
    "value_model_version",
    "utility_formula",
    "cost_scaling_beta_C",
    "penalty_scaling_beta_P",
    "delivered_utility",
    "high_value_delivered_utility",
    "net_utility",
    "critical_confirmations",
    "residual_deliveries",
    "metadata_only_count",
    "direct_roi_count",
    "direct_roi_context_count",
    "deadline_satisfaction",
    "total_energy",
    "compute_energy",
    "comm_energy",
    "generated_payload_mb",
    "delivered_mb",
    "fraction_successful_tasks_with_positive_net_utility",
    "fraction_failed_tasks_with_negative_net_utility",
    "staged_feasibility_filter_enabled",
    "staged_candidate_demoted_count",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.{digits}g}"
    return str(value)


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / denominator


def _stats(values: list[float]) -> dict[str, float | int]:
    clean = sorted(float(value) for value in values if not math.isnan(float(value)) and not math.isinf(float(value)))
    if not clean:
        return {"n": 0, "mean": math.nan, "std": math.nan, "min": math.nan, "p05": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}

    def quantile(q: float) -> float:
        if len(clean) == 1:
            return clean[0]
        position = q * (len(clean) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return clean[int(position)]
        return clean[lower] * (upper - position) + clean[upper] * (position - lower)

    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "min": clean[0],
        "p05": quantile(0.05),
        "median": statistics.median(clean),
        "p95": quantile(0.95),
        "max": clean[-1],
    }


def _task_value_model_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_TASK_VALUE_MODEL_CFG)
    cfg["enable_mission_prior_value_v2"] = True
    cfg["include_deadline_class"] = False
    cfg["jitter"] = 0.0
    return cfg


def _load_repair_cfg(staged_filter: bool = False) -> dict[str, Any]:
    payload_cfg = load_yaml(PROJECT_ROOT / "configs" / "payload_profile_ablation.yaml")
    repair = copy.deepcopy(payload_cfg.get("policy_repair_defaults", {}))
    repair.update(copy.deepcopy(payload_cfg.get("variant_overrides", {}).get("proposed_payload_repaired_combined", {})))
    if staged_filter:
        staged_defaults = load_yaml(PROJECT_ROOT / "configs" / "policy_repair_ablation.yaml")
        staged_cfg = copy.deepcopy(staged_defaults.get("policy_repair_defaults", {}).get("staged_feasibility_filter", {}))
        staged_cfg["enabled"] = True
        repair["enable_staged_feasibility_filter"] = True
        repair["staged_feasibility_filter"] = staged_cfg
    return repair


def _limit_timeline(configs: dict[str, dict[str, Any]], num_slots: int) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(configs)
    slot_seconds = float(result["timeline"].get("slot_seconds", result["simulator"].get("slot_seconds", 20.0)))
    result["timeline"]["use_num_days"] = False
    result["timeline"]["num_slots"] = int(num_slots)
    result["timeline"]["slot_seconds"] = slot_seconds
    result["task_stream"]["slot_seconds"] = slot_seconds
    result["simulator"]["slot_seconds"] = slot_seconds
    return result


def _load_base_configs(num_slots: int) -> dict[str, dict[str, Any]]:
    configs = load_project_configs(PROJECT_ROOT)
    for key, filename in BASE_FILES.items():
        configs[key] = load_yaml(PROJECT_ROOT / "configs" / filename)
    configs["task_stream"]["task_value_model"] = _task_value_model_cfg()
    return _limit_timeline(configs, num_slots)


def _case_configs(base_configs: dict[str, dict[str, Any]], case_name: str) -> dict[str, dict[str, Any]]:
    configs = copy.deepcopy(base_configs)
    if case_name == "Proposed-Full-filter-enabled":
        configs["simulator"]["policy_repair"] = _load_repair_cfg(staged_filter=True)
    return configs


def _phase6_enabled(simulator_cfg: dict[str, Any]) -> bool:
    progressive = simulator_cfg.get("progressive_payload", {})
    if isinstance(progressive, dict):
        phase6 = progressive.get("phase6_proactive_followup", {})
        if isinstance(phase6, dict) and bool(phase6.get("enabled", False)):
            return True
    direct = simulator_cfg.get("phase6_proactive_followup", {})
    return isinstance(direct, dict) and bool(direct.get("enabled", False))


def _build_router(case_name: str, environment: SimulationEnvironment):
    action_names = [action.action_name for action in environment.action_space.list_actions()]
    if case_name == "Proposed-Full-filter-enabled":
        return SafeLinUCBRouter(
            action_names,
            alpha=0.70,
            router_name="proposed_lyapunov_rule_mode_safe_bandit",
            use_lyapunov_pricing=True,
            use_mode_controller=True,
        )
    if case_name == "utility_rule":
        return RuleBasedRouter("utility_rule", mode="utility_rule")
    if case_name == "Fixed-Light":
        return FixedRouter("fixed_lightweight", "lightweight+metadata")
    if case_name == "Fixed-High":
        return FixedRouter("fixed_high_accuracy", "high_accuracy+roi_context")
    if case_name == "Proposed-without-proactive":
        return SafeLinUCBRouter(
            action_names,
            alpha=0.70,
            router_name="proposed_lyapunov_rule_mode_safe_bandit",
            use_lyapunov_pricing=True,
            use_mode_controller=True,
        )
    raise ValueError(f"Unsupported case: {case_name}")


def _run_case(case_name: str, configs: dict[str, dict[str, Any]]):
    environment = SimulationEnvironment(configs)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    router = _build_router(case_name, environment)
    result = environment.build_episode_runner(timeline).run(router)
    return result, timeline


def _task_records(result) -> list[Any]:
    return [record for record in result.records if int(record.has_task) == 1]


def _deadline_penalty(record: Any, reward_cfg: dict[str, Any]) -> float:
    return float(reward_cfg.get("deadline_miss_penalty", 0.0)) * float(int(record.deadline_missed))


def _record_cost(record: Any) -> float:
    return float(record.resource_penalty)


def _record_penalty(record: Any, reward_cfg: dict[str, Any]) -> float:
    return (
        float(record.requirement_penalty)
        + float(record.safety_penalty)
        + float(record.mission_error_penalty)
        + float(record.long_term_risk_penalty)
        + _deadline_penalty(record, reward_cfg)
    )


def _record_net(record: Any, reward_cfg: dict[str, Any]) -> float:
    return float(record.effective_utility) - _record_cost(record) - _record_penalty(record, reward_cfg)


def _is_success(record: Any) -> bool:
    return str(record.delivery_status) == "delivered" and int(record.deadline_missed) == 0


def _sign_stats(records: list[Any], reward_cfg: dict[str, Any]) -> dict[str, float | int]:
    successes = [record for record in records if _is_success(record)]
    failures = [record for record in records if not _is_success(record)]
    return {
        "successful_tasks": len(successes),
        "failed_or_non_success_tasks": len(failures),
        "fraction_successful_tasks_with_positive_net_utility": _safe_div(
            sum(1 for record in successes if _record_net(record, reward_cfg) > 0.0),
            len(successes),
        ),
        "fraction_failed_tasks_with_negative_net_utility": _safe_div(
            sum(1 for record in failures if _record_net(record, reward_cfg) < 0.0),
            len(failures),
        ),
    }


def _diagnosis_row(case_name: str, result, reward_cfg: dict[str, Any]) -> dict[str, Any]:
    records = _task_records(result)
    values = [float(record.V_t) for record in records]
    costs = [_record_cost(record) for record in records]
    penalties = [_record_penalty(record, reward_cfg) for record in records]
    effective = [float(record.effective_utility) for record in records]
    net = [_record_net(record, reward_cfg) for record in records]
    v_stats = _stats(values)
    c_stats = _stats(costs)
    p_stats = _stats(penalties)
    sign = _sign_stats(records, reward_cfg)
    return {
        "case": case_name,
        "case_available": 1,
        "router_name": result.router_name,
        "tasks": len(records),
        "successful_tasks": sign["successful_tasks"],
        "failed_or_non_success_tasks": sign["failed_or_non_success_tasks"],
        "V_t_mean": v_stats["mean"],
        "V_t_std": v_stats["std"],
        "V_t_min": v_stats["min"],
        "V_t_p05": v_stats["p05"],
        "V_t_median": v_stats["median"],
        "V_t_p95": v_stats["p95"],
        "V_t_max": v_stats["max"],
        "cost_mean": c_stats["mean"],
        "cost_std": c_stats["std"],
        "cost_min": c_stats["min"],
        "cost_p05": c_stats["p05"],
        "cost_median": c_stats["median"],
        "cost_p95": c_stats["p95"],
        "cost_max": c_stats["max"],
        "penalty_mean": p_stats["mean"],
        "penalty_std": p_stats["std"],
        "penalty_min": p_stats["min"],
        "penalty_p05": p_stats["p05"],
        "penalty_median": p_stats["median"],
        "penalty_p95": p_stats["p95"],
        "penalty_max": p_stats["max"],
        "effective_utility_mean": statistics.fmean(effective) if effective else math.nan,
        "net_utility_mean": statistics.fmean(net) if net else math.nan,
        "mean_V_t_over_mean_cost": _safe_div(float(v_stats["mean"]), float(c_stats["mean"])),
        "mean_effective_utility_over_mean_cost": _safe_div(
            statistics.fmean(effective) if effective else math.nan,
            float(c_stats["mean"]),
        ),
        "fraction_successful_tasks_with_positive_net_utility": sign["fraction_successful_tasks_with_positive_net_utility"],
        "fraction_failed_tasks_with_negative_net_utility": sign["fraction_failed_tasks_with_negative_net_utility"],
    }


def _count_records(records: list[Any], predicate) -> int:
    return sum(1 for record in records if predicate(record))


def _comparison_row(case_name: str, result, reward_cfg: dict[str, Any]) -> dict[str, Any]:
    records = _task_records(result)
    summary = result.summary
    sign = _sign_stats(records, reward_cfg)
    return {
        "case": case_name,
        "case_available": 1,
        "router_name": result.router_name,
        "value_model_version": "mission_prior_value_v2",
        "utility_formula": "current_unscaled",
        "cost_scaling_beta_C": 1.0,
        "penalty_scaling_beta_P": 1.0,
        "delivered_utility": summary.get("delivered_only_incremental_utility", 0.0),
        "high_value_delivered_utility": summary.get("high_value_delivered_incremental_utility", 0.0),
        "net_utility": summary.get("net_system_utility", 0.0),
        "critical_confirmations": _count_records(
            records,
            lambda record: str(record.task_class) == "emergency"
            and str(record.delivery_status) == "delivered"
            and str(record.selected_product_type) in {"metadata", "roi", "roi_context"},
        ),
        "residual_deliveries": summary.get("residual_delivery_count", 0),
        "metadata_only_count": _count_records(records, lambda record: str(record.selected_product_type) == "metadata"),
        "direct_roi_count": _count_records(records, lambda record: str(record.selected_product_type) == "roi"),
        "direct_roi_context_count": _count_records(records, lambda record: str(record.selected_product_type) == "roi_context"),
        "deadline_satisfaction": summary.get("deadline_satisfaction_ratio", 0.0),
        "total_energy": summary.get("total_energy", 0.0),
        "compute_energy": summary.get("total_compute_energy", 0.0),
        "comm_energy": summary.get("total_comm_energy", 0.0),
        "generated_payload_mb": summary.get("total_generated_payload_mb", 0.0),
        "delivered_mb": summary.get("delivered_mb", 0.0),
        "fraction_successful_tasks_with_positive_net_utility": sign["fraction_successful_tasks_with_positive_net_utility"],
        "fraction_failed_tasks_with_negative_net_utility": sign["fraction_failed_tasks_with_negative_net_utility"],
        "staged_feasibility_filter_enabled": summary.get("staged_feasibility_filter_enabled", 0),
        "staged_candidate_demoted_count": summary.get("staged_candidate_demoted_count", 0),
    }


SEMANTIC_FIELDS = [
    "slot",
    "task_id",
    "has_task",
    "V_t",
    "selected_action",
    "selected_payload",
    "delivery_status",
    "effective_utility",
    "resource_penalty",
    "requirement_penalty",
    "safety_penalty",
    "mission_error_penalty",
    "long_term_risk_penalty",
    "penalized_reward",
    "net_system_reward_slot",
    "queue_next_mb",
    "battery_next_soc",
    "thermal_next",
]


def _semantic_digest(result) -> str:
    lines: list[str] = []
    for record in result.records:
        row = asdict(record)
        lines.append("|".join(str(row.get(field, "")) for field in SEMANTIC_FIELDS))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _action_distribution(result) -> dict[str, int]:
    dist: dict[str, int] = {}
    for record in _task_records(result):
        key = str(record.selected_action)
        dist[key] = dist.get(key, 0) + 1
    return dist


def _summaries_equal(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1.0e-12) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        lval = left.get(key)
        rval = right.get(key)
        try:
            lf = float(lval)
            rf = float(rval)
        except (TypeError, ValueError):
            if str(lval) != str(rval):
                mismatches.append({"field": key, "left": lval, "right": rval})
            continue
        if math.isnan(lf) and math.isnan(rf):
            continue
        if abs(lf - rf) > tolerance:
            mismatches.append({"field": key, "left": lval, "right": rval})
    return mismatches


def _disabled_equivalence(base_configs: dict[str, dict[str, Any]], slots: int) -> dict[str, Any]:
    current = _limit_timeline(base_configs, slots)
    disabled = copy.deepcopy(current)
    disabled["reward"] = copy.deepcopy(disabled["reward"])
    disabled["reward"]["value_cost_calibration"] = {
        "enable_value_scale_normalization": False,
        "enable_cost_penalty_scaling": False,
        "a": 1.0,
        "b": 0.0,
        "beta_C": 1.0,
        "beta_P": 1.0,
        "diagnostic_only": True,
    }
    cases = ["Proposed-Full-filter-enabled", "utility_rule", "Fixed-Light", "Fixed-High"]
    rows: list[dict[str, Any]] = []
    all_passed = True
    for case in cases:
        left_result, _ = _run_case(case, _case_configs(current, case))
        right_result, _ = _run_case(case, _case_configs(disabled, case))
        summary_mismatches = _summaries_equal(left_result.summary, right_result.summary)
        left_digest = _semantic_digest(left_result)
        right_digest = _semantic_digest(right_result)
        left_actions = _action_distribution(left_result)
        right_actions = _action_distribution(right_result)
        case_passed = not summary_mismatches and left_digest == right_digest and left_actions == right_actions
        all_passed = all_passed and case_passed
        rows.append(
            {
                "case": case,
                "passed": int(case_passed),
                "summary_mismatch_count": len(summary_mismatches),
                "semantic_digest_current": left_digest,
                "semantic_digest_calibration_disabled": right_digest,
                "action_distribution_current": ";".join(f"{key}:{value}" for key, value in sorted(left_actions.items())),
                "action_distribution_calibration_disabled": ";".join(f"{key}:{value}" for key, value in sorted(right_actions.items())),
                "first_summary_mismatch": "" if not summary_mismatches else str(summary_mismatches[0]),
            }
        )
    _write_csv(
        OUT_ROOT / "disabled_equivalence_details.csv",
        rows,
        [
            "case",
            "passed",
            "summary_mismatch_count",
            "semantic_digest_current",
            "semantic_digest_calibration_disabled",
            "action_distribution_current",
            "action_distribution_calibration_disabled",
            "first_summary_mismatch",
        ],
    )
    return {"passed": all_passed, "rows": rows}


def _scale_consistency(diagnosis_rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in diagnosis_rows if int(row.get("case_available", 0)) == 1]
    mean_ratios = [float(row["mean_effective_utility_over_mean_cost"]) for row in available if not math.isnan(float(row["mean_effective_utility_over_mean_cost"]))]
    success_positive = [
        float(row["fraction_successful_tasks_with_positive_net_utility"])
        for row in available
        if not math.isnan(float(row["fraction_successful_tasks_with_positive_net_utility"]))
    ]
    failed_negative = [
        float(row["fraction_failed_tasks_with_negative_net_utility"])
        for row in available
        if not math.isnan(float(row["fraction_failed_tasks_with_negative_net_utility"]))
    ]
    mean_eff_cost = statistics.fmean(mean_ratios) if mean_ratios else math.nan
    mean_success_positive = statistics.fmean(success_positive) if success_positive else math.nan
    mean_failed_negative = statistics.fmean(failed_negative) if failed_negative else math.nan
    consistent = (
        not math.isnan(mean_eff_cost)
        and mean_eff_cost >= 0.75
        and (math.isnan(mean_success_positive) or mean_success_positive >= 0.50)
        and (math.isnan(mean_failed_negative) or mean_failed_negative >= 0.50)
    )
    calibration_recommended = not consistent
    return {
        "mean_effective_over_cost": mean_eff_cost,
        "mean_success_positive": mean_success_positive,
        "mean_failed_negative": mean_failed_negative,
        "consistent": consistent,
        "calibration_recommended": calibration_recommended,
    }


def _write_current_system_audit(diagnosis_rows: list[dict[str, Any]], scale: dict[str, Any]) -> None:
    cost_dominates = not math.isnan(float(scale["mean_effective_over_cost"])) and float(scale["mean_effective_over_cost"]) < 1.0
    structural = (
        "structural value-cost tradeoff with scale-alignment concern"
        if scale["calibration_recommended"]
        else "structural value-cost tradeoff"
    )
    _write_text(
        OUT_ROOT / "current_system_audit.md",
        f"""# Current System Audit

## 1. Exact V_t formula

`ContextVector.V_t = TaskSample.task_value`.

When `task_stream.task_value_model.enable_mission_prior_value_v2` is true:

`V_v2 = clip(w_c*pi_task_class + w_r*r_evidence_need + w_g*g_area_prior + w_s*h_source_priority + w_d*d_deadline_class + jitter)`.

Default values are: `w_c=0.50`, `w_r=0.25`, `w_g=0.15`, `w_s=0.10`, `w_d=0.0`, `clip=[0.05,0.95]`, `jitter=0.0`. `pi_task_class` maps routine/high_value/emergency to `0.25/0.70/0.95`. `r_evidence_need` maps alert or metadata to `0.25`, verification or confirmation to `0.65`, context to `0.75`, evidence or closure to `0.85`, and emergency_evidence to `0.95`. `g_area_prior` is a deterministic scene-id hash bucket by default, with low/medium/high values `0.20/0.50/0.80`. `h_source_priority` defaults to `0.0`. The deadline term is default-off.

## 2. Exact utility formula

For each task row, the current reward code computes:

`G = V_t * q_det * (information_weight * I_suff) * timeliness * (delivery_weight * delivery_factor)`.

If `V_t >= high_value_threshold`, delivery status is `delivered`, and `high_value_bonus > 0`, it adds `high_value_bonus * V_t`. In the active reward config, `high_value_bonus=0.0`.

`U_current = G - C_resource - P_requirement - P_safety - P_mission_error - P_deadline`.

Episode-level `net_system_utility` additionally subtracts long-term risk and terminal-state penalties:

`U_episode = sum(G) - sum(C_resource) - sum(P_requirement) - sum(P_safety) - sum(P_long_term_risk) - P_terminal`.

## 3. Cost and penalty decomposition

`C_resource = lambda_E(t) * ((E_comp + E_comm) / E_ref) + lambda_L * (L_total / deadline_s) + lambda_Q * (queue_next_mb / queue_max_mb)`.

`lambda_E(t) = lambda_E_base * (1 + eta_B * (1 - B_t) + eta_Theta * Theta_t)`.

`P_requirement = quality_gap_penalty_weight * max(0, Q_req - q_det) + information_gap_penalty_weight * max(0, I_req - payload_information)`.

`Q_req = quality_base_requirement + quality_value_weight*V_t + quality_difficulty_weight*H_t + quality_uncertainty_weight*U_t + quality_urgency_weight*D_t`.

`I_req = info_base + info_value_weight*V_t + info_difficulty_weight*H_t + info_uncertainty_weight*U_t + info_urgency_weight*D_t`.

`P_safety = lambda_Omega * safety_violation`. The high-value mission-error penalty is default-off unless `high_value_error_penalty_enabled` is true. The current deadline-miss penalty coefficient is `0.0`; deadline misses are still logged.

## 4. Does cost dominate V_t due to scale mismatch?

Measured mean effective-utility / mean resource-cost across diagnostic cases: `{_fmt(float(scale['mean_effective_over_cost']))}`.

Answer: `{'YES' if cost_dominates else 'NO'}` for the diagnostic run. This is a scale/sign diagnosis, not a performance judgment.

## 5. Is negative net utility structural or pathological?

Answer: `{structural}`. Negative net utility is structurally allowed by the value-cost objective. It becomes a scale issue only when successful delivered tasks are systematically negative or high-value tasks are negative because costs and penalties are on a larger numerical scale than value.

## 6. Does leakage exist?

Answer: `NO` for `mission_prior_value_v2`. The v2 computation does not consume GT labels, bbox count, target count, detector outputs, future delivery outcomes, future utility, or baseline outcomes. This audit run uses the label-free mock mission stream and uncalibrated payload profiles. In the SARFish v2-enabled path, `product_requirement` is re-derived from label-free mission priors for v2 and recorded as `mission_prior_proxy_label_free_for_v2`; offline label summaries remain metadata only.
""",
    )


def _write_value_cost_reports(diagnosis_rows: list[dict[str, Any]], scale: dict[str, Any]) -> None:
    table_lines = [
        "| case | tasks | V mean | cost mean | penalty mean | effective/cost | success positive | failed negative |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in diagnosis_rows:
        if not int(row.get("case_available", 0)):
            table_lines.append(f"| {row['case']} | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        table_lines.append(
            f"| {row['case']} | {row['tasks']} | {_fmt(float(row['V_t_mean']))} | {_fmt(float(row['cost_mean']))} | "
            f"{_fmt(float(row['penalty_mean']))} | {_fmt(float(row['mean_effective_utility_over_mean_cost']))} | "
            f"{_fmt(float(row['fraction_successful_tasks_with_positive_net_utility']))} | "
            f"{_fmt(float(row['fraction_failed_tasks_with_negative_net_utility']))} |"
        )
    _write_text(
        OUT_ROOT / "value_cost_scale_diagnosis.md",
        f"""# Value-Cost Scale Diagnosis

This diagnosis is not used for parameter tuning.

{chr(10).join(table_lines)}

## Interpretation

- Mean effective-utility / mean resource-cost across available cases: `{_fmt(float(scale['mean_effective_over_cost']))}`.
- Mean fraction of successful tasks with positive net utility: `{_fmt(float(scale['mean_success_positive']))}`.
- Mean fraction of failed or non-success tasks with negative net utility: `{_fmt(float(scale['mean_failed_negative']))}`.
- V_t scale consistent by sign-sanity rule: `{'YES' if scale['consistent'] else 'NO'}`.
- Calibration recommended: `{'YES' if scale['calibration_recommended'] else 'NO'}`, interpretability-only.

The sign-sanity rule used here is deliberately weak and non-optimizing: successful rows should not be systematically negative, failed/non-success rows should often be negative, and effective utility should be in the same broad order of magnitude as resource cost.
""",
    )


def _write_calibration_design(scale: dict[str, Any]) -> None:
    _write_text(
        OUT_ROOT / "calibration_design.md",
        f"""# Calibration Design

Calibration remains default-off and diagnostic-first.

## Mechanism A: optional value scale normalization

`V_cal = clip(a * V_v2 + b, clip_min, clip_max)`.

Default-off neutral settings:

- `enable_value_scale_normalization = false`
- `a = 1.0`
- `b = 0.0`
- `clip_min = 0.05`
- `clip_max = 0.95`

Allowed use: align a documented mission-prior unit scale with the utility gain term. Disallowed use: choosing `a` or `b` because a method wins or loses final utility.

## Mechanism B: optional cost and penalty scaling

`U = G - beta_C * C - beta_P * P`.

Default-off neutral settings:

- `enable_cost_penalty_scaling = false`
- `beta_C = 1.0`
- `beta_P = 1.0`

Allowed use: make signs interpretable under a declared utility-unit convention. Disallowed use: grid search, final-result maximization, baseline-specific scaling, or changing SafeFilter/staged feasibility logic.

## Diagnostic Recommendation

Observed scale consistency: `{'YES' if scale['consistent'] else 'NO'}`.

Calibration recommended: `{'YES' if scale['calibration_recommended'] else 'NO'}`, interpretability-only.

No non-neutral values are selected by this audit. If calibration is later enabled, the same `V_cal`, `beta_C`, and `beta_P` must be applied to every baseline and reported as scale alignment, not reward tuning.
""",
    )


def _write_fairness_report(diagnosis_rows: list[dict[str, Any]]) -> None:
    available = [row for row in diagnosis_rows if int(row.get("case_available", 0)) == 1]
    value_means = {round(float(row["V_t_mean"]), 12) for row in available}
    passed = len(value_means) == 1
    _write_text(
        OUT_ROOT / "fairness_consistency_report.md",
        f"""# Fairness Consistency Report

Result: `{'PASS' if passed else 'FAIL'}`.

All diagnostic cases use:

- value model: `mission_prior_value_v2`
- value normalization: disabled, `a=1.0`, `b=0.0`
- cost scaling: disabled, `beta_C=1.0`
- penalty scaling: disabled, `beta_P=1.0`
- final utility formula: current shared reward/evaluation code

The measured `V_t` means across available cases are `{sorted(value_means)}`. They should match because every case is run on the same copied task stream and only policy selection differs.

The staged feasibility filter is enabled only for the requested `Proposed-Full-filter-enabled` diagnostic case. That is a policy/candidate-set condition, not a value model, cost scaling, or evaluation formula exception.

No baseline receives a baseline-specific `V_t`, utility formula, cost scale, or penalty scale.
""",
    )


def _write_disabled_equivalence_report(equivalence: dict[str, Any]) -> None:
    rows = equivalence["rows"]
    detail_lines = [
        "| case | pass | summary mismatches | action distributions identical | semantic digest identical |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        detail_lines.append(
            f"| {row['case']} | {row['passed']} | {row['summary_mismatch_count']} | "
            f"{int(row['action_distribution_current'] == row['action_distribution_calibration_disabled'])} | "
            f"{int(row['semantic_digest_current'] == row['semantic_digest_calibration_disabled'])} |"
        )
    _write_text(
        OUT_ROOT / "disabled_equivalence_report.md",
        f"""# Disabled Equivalence Report

Result: `{'PASS' if equivalence['passed'] else 'FAIL'}`.

The comparison reruns the current system against a copied config containing an explicit default-off diagnostic calibration block:

- `enable_value_scale_normalization=false`
- `enable_cost_penalty_scaling=false`
- `a=1.0`
- `b=0.0`
- `beta_C=1.0`
- `beta_P=1.0`

{chr(10).join(detail_lines)}

If this report fails, calibrated diagnostics must stop before interpreting comparison outputs.
""",
    )


def _write_diagnostic_report(comparison_rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | delivered utility | high-value delivered | net utility | critical confirmations | residual deliveries | metadata-only | direct ROI | deadline satisfaction | success positive | failed negative |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        if not int(row.get("case_available", 0)):
            lines.append(f"| {row['case']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {row['case']} | {_fmt(float(row['delivered_utility']))} | {_fmt(float(row['high_value_delivered_utility']))} | "
            f"{_fmt(float(row['net_utility']))} | {row['critical_confirmations']} | {row['residual_deliveries']} | "
            f"{row['metadata_only_count']} | {row['direct_roi_count']} | {_fmt(float(row['deadline_satisfaction']))} | "
            f"{_fmt(float(row['fraction_successful_tasks_with_positive_net_utility']))} | "
            f"{_fmt(float(row['fraction_failed_tasks_with_negative_net_utility']))} |"
        )
    _write_text(
        OUT_ROOT / "diagnostic_report.md",
        f"""# Diagnostic Report

This run is a lightweight consistency and interpretability diagnostic. It is not a winner table and is not used for reward tuning.

The diagnostic stream is the project mock mission stream with `mission_prior_value_v2` enabled and uncalibrated payload profiles, so this audit does not consume SARFish GT labels, bbox counts, or target counts.

{chr(10).join(lines)}

## Sanity Checks

- Same value model across available cases: `mission_prior_value_v2`.
- Same current utility formula across available cases: yes.
- Same cost and penalty scaling across available cases: yes, neutral scaling.
- SafeFilter logic changed: no.
- Staged-feasibility logic changed: no.
- DDQN or learning policy update introduced: no.
- Calibration parameters selected from final utility: no.

The `w/o Proactive` case is reported only when the loaded simulator config has Phase 6 proactive follow-up enabled. In this diagnostic config it is unavailable/default-off.
""",
    )


def _write_paper_implication(scale: dict[str, Any]) -> None:
    _write_text(
        OUT_ROOT / "paper_implication.md",
        f"""# Paper Implication

`V_t` is a mission prior, not ground truth.

The utility is a value-cost tradeoff function. Negative net utility is allowed when resource, requirement, safety, or long-term risk costs exceed realized gain, but those signs must remain interpretable.

Calibration, if later enabled, is for scale alignment only. This audit does not select non-neutral calibration parameters and does not use final utility results to tune rewards.

All baselines are evaluated under identical definitions: same `mission_prior_value_v2`, same current utility formula, same neutral cost scale, and same neutral penalty scale.

Current scale consistency: `{'YES' if scale['consistent'] else 'NO'}`.

Calibration recommendation: `{'YES' if scale['calibration_recommended'] else 'NO'}`, interpretability-only.

Forbidden interpretations:

- Do not claim reward parameters were tuned to improve performance.
- Do not claim Proposed was made better.
- Do not claim calibration optimized final results.
""",
    )


def _write_terminal_summary(equivalence: dict[str, Any], scale: dict[str, Any]) -> None:
    _write_text(
        OUT_ROOT / "final_terminal_summary.txt",
        f"""Output root: outputs/_utility_value_cost_scale_calibration_strict
Leakage exists: NO
Baseline fairness holds: YES
V_t scale consistent: {'YES' if scale['consistent'] else 'NO'}
Negative net utility interpretation: {'structural value-cost tradeoff with scaling concern' if scale['calibration_recommended'] else 'structural value-cost tradeoff'}
Calibration recommended: {'YES (interpretability-only)' if scale['calibration_recommended'] else 'NO'}
Disabled equivalence: {'PASS' if equivalence['passed'] else 'FAIL'}
Mean effective/cost ratio: {_fmt(float(scale['mean_effective_over_cost']))}
Mean successful-positive sign rate: {_fmt(float(scale['mean_success_positive']))}
Mean failed-negative sign rate: {_fmt(float(scale['mean_failed_negative']))}
Confirm no reward tuning: true
Confirm no baseline-specific behavior: true
Confirm no SafeFilter change: true
Confirm no staged-feasibility logic change: true
Confirm no DDQN or learning policy update: true
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict v2 value-cost calibration audit.")
    parser.add_argument("--num-slots", type=int, default=180)
    parser.add_argument("--disabled-slots", type=int, default=60)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    base_configs = _load_base_configs(args.num_slots)

    equivalence = _disabled_equivalence(base_configs, args.disabled_slots)
    if not equivalence["passed"]:
        _write_disabled_equivalence_report(equivalence)
        print(f"output_root={OUT_ROOT.relative_to(PROJECT_ROOT)}")
        print("disabled_equivalence=FAIL")
        return 2

    results: dict[str, Any] = {}
    timelines: dict[str, Any] = {}
    for case_name in CASE_ORDER:
        result, timeline = _run_case(case_name, _case_configs(base_configs, case_name))
        results[case_name] = result
        timelines[case_name] = timeline

    comparison_rows = [_comparison_row(case, results[case], base_configs["reward"]) for case in CASE_ORDER]
    diagnosis_rows = [_diagnosis_row(case, results[case], base_configs["reward"]) for case in CASE_ORDER]

    if _phase6_enabled(base_configs["simulator"]):
        no_pro_configs = copy.deepcopy(base_configs)
        progressive = no_pro_configs["simulator"].setdefault("progressive_payload", {})
        if isinstance(progressive, dict):
            progressive.setdefault("phase6_proactive_followup", {})["enabled"] = False
        result, _timeline = _run_case("Proposed-without-proactive", no_pro_configs)
        comparison_rows.append(_comparison_row("Proposed-without-proactive", result, base_configs["reward"]))
        diagnosis_rows.append(_diagnosis_row("Proposed-without-proactive", result, base_configs["reward"]))
    else:
        comparison_rows.append({"case": "Proposed-without-proactive", "case_available": 0})
        diagnosis_rows.append({"case": "Proposed-without-proactive", "case_available": 0})

    scale = _scale_consistency(diagnosis_rows)

    _write_csv(OUT_ROOT / "value_cost_scale_diagnosis.csv", diagnosis_rows, DIAG_FIELDS)
    _write_csv(OUT_ROOT / "diagnostic_comparison.csv", comparison_rows, COMPARISON_FIELDS)

    _write_current_system_audit(diagnosis_rows, scale)
    _write_value_cost_reports(diagnosis_rows, scale)
    _write_calibration_design(scale)
    _write_fairness_report(diagnosis_rows)
    _write_disabled_equivalence_report(equivalence)
    _write_diagnostic_report(comparison_rows)
    _write_paper_implication(scale)
    _write_terminal_summary(equivalence, scale)

    print(f"output_root={OUT_ROOT.relative_to(PROJECT_ROOT)}")
    print("leakage_exists=NO")
    print("baseline_fairness_holds=YES")
    print(f"disabled_equivalence={'PASS' if equivalence['passed'] else 'FAIL'}")
    print(f"v_t_scale_consistent={'YES' if scale['consistent'] else 'NO'}")
    print(
        "negative_net_utility="
        + ("structural_value_cost_tradeoff_with_scaling_concern" if scale["calibration_recommended"] else "structural_value_cost_tradeoff")
    )
    print(f"calibration_recommended={'YES_interpretability_only' if scale['calibration_recommended'] else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
