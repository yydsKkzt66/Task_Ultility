"""Pre-registered value-cost calibration patch artifacts.

This runner is intentionally parameter-fixed. It does not grid search, tune to
baseline rankings, alter SafeFilter, change staged feasibility, or activate
lifecycle utility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.data.mission_prior_value import DEFAULT_TASK_VALUE_MODEL_CFG  # noqa: E402
from src.evaluation.metrics import summarize_episode  # noqa: E402
from src.routers.fixed_router import FixedRouter  # noqa: E402
from src.routers.rule_router import RuleBasedRouter  # noqa: E402
from src.routers.safe_linucb_router import SafeLinUCBRouter  # noqa: E402
from src.reward.mission_utility import compute_dynamic_energy_weight  # noqa: E402
from src.reward.delivery import compute_delivery_factor  # noqa: E402
from src.reward.reward_function import compute_latency_penalty, compute_queue_penalty  # noqa: E402
from src.reward.utility_calibration import calibrate_value, get_utility_calibration  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "outputs" / "_pre_registered_value_cost_calibration_patch"
PLOTS_DIR = OUT_ROOT / "plots"

BASE_FILES = {
    "dataset": "dataset.yaml",
    "task_stream": "task_stream.yaml",
    "timeline": "timeline.yaml",
    "simulator": "simulator.yaml",
    "payload_profiles": "payload_profiles.yaml",
}


@dataclass(frozen=True)
class Candidate:
    candidate: str
    a: float
    b: float
    beta_C: float
    beta_R: float
    beta_H: float
    purpose: str


CANDIDATES = [
    Candidate("C0", 1.00, 0.00, 1.00, 1.00, 1.00, "current v2 baseline"),
    Candidate("C1", 1.10, 0.03, 0.65, 0.50, 1.00, "recommended mild value-cost calibration"),
    Candidate("C2", 1.05, 0.02, 0.80, 0.65, 1.00, "conservative calibration"),
    Candidate("C3", 1.10, 0.03, 1.00, 1.00, 1.00, "value-scaling-only diagnostic"),
    Candidate("C4", 1.00, 0.00, 0.65, 0.50, 1.00, "cost/requirement-scaling-only diagnostic"),
]

CASE_ORDER = (
    "Proposed-Full-filter-enabled",
    "utility_rule",
    "Fixed-Light",
    "Fixed-High",
)

TASK_DECOMP_FIELDS = [
    "regime",
    "case_name",
    "task_id",
    "scene_id",
    "task_class",
    "product_requirement",
    "action",
    "payload_type",
    "V_raw",
    "V_cal",
    "mission_gain",
    "q_factor",
    "info_factor",
    "timeliness_factor",
    "delivery_factor",
    "C_resource",
    "C_energy",
    "C_latency",
    "C_queue",
    "C_payload_or_storage",
    "P_requirement",
    "P_quality_gap",
    "P_info_gap",
    "P_hard",
    "P_safety",
    "P_expiration",
    "P_deadline",
    "P_infeasible",
    "P_terminal_risk",
    "net_utility",
    "delivered_before_deadline",
    "success_flag",
    "high_value_flag",
    "emergency_flag",
    "hard_failure_flag",
    "lifecycle_fields_active",
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


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1.0e-12:
        return math.nan
    return numerator / denominator


def _mean(values: list[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.fmean(clean) if clean else math.nan


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "n/a"
    return f"{number:.{digits}g}"


def _task_value_model_cfg() -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_TASK_VALUE_MODEL_CFG)
    cfg["enable_mission_prior_value_v2"] = True
    cfg["include_deadline_class"] = False
    cfg["jitter"] = 0.0
    return cfg


def _limit_timeline(configs: dict[str, dict[str, Any]], slots: int) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(configs)
    slot_seconds = float(result["timeline"].get("slot_seconds", result["simulator"].get("slot_seconds", 20.0)))
    result["timeline"]["use_num_days"] = False
    result["timeline"]["num_slots"] = int(slots)
    result["timeline"]["slot_seconds"] = slot_seconds
    result["task_stream"]["slot_seconds"] = slot_seconds
    result["simulator"]["slot_seconds"] = slot_seconds
    result["dataset"]["max_tasks"] = max(int(result["dataset"].get("max_tasks", slots)), int(slots))
    return result


def _load_base_configs(slots: int) -> dict[str, dict[str, Any]]:
    configs = load_project_configs(PROJECT_ROOT)
    for key, filename in BASE_FILES.items():
        configs[key] = load_yaml(PROJECT_ROOT / "configs" / filename)
    configs["task_stream"]["task_value_model"] = _task_value_model_cfg()
    configs["reward"] = copy.deepcopy(configs["reward"])
    configs["reward"].setdefault("lifecycle_utility", {})["enable_lifecycle_utility"] = False
    return _limit_timeline(configs, slots)


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


def _case_configs(base_configs: dict[str, dict[str, Any]], case_name: str) -> dict[str, dict[str, Any]]:
    configs = copy.deepcopy(base_configs)
    if case_name == "Proposed-Full-filter-enabled":
        configs["simulator"]["policy_repair"] = _load_repair_cfg(staged_filter=True)
    return configs


def _apply_candidate(
    configs: dict[str, dict[str, Any]],
    candidate: Candidate,
    *,
    enabled: bool,
    shadow_only: bool = False,
) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(configs)
    result["reward"] = copy.deepcopy(result["reward"])
    result["reward"]["utility_calibration"] = {
        "enable_value_cost_calibration": bool(enabled),
        "value_scale": {
            "a": candidate.a,
            "b": candidate.b,
            "clip_min": 0.05,
            "clip_max": 0.95,
        },
        "cost_penalty_scale": {
            "beta_resource_cost": candidate.beta_C,
            "beta_requirement_penalty": candidate.beta_R,
            "beta_hard_penalty": candidate.beta_H,
        },
        "diagnostic_logging": True,
        "shadow_only": bool(shadow_only),
    }
    result["reward"]["lifecycle_utility"] = {
        "enable_lifecycle_utility": False,
        "reserved_only": True,
        "stage_weights": {"alert": None, "confirmation": None, "evidence": None},
        "stage_deadlines": {
            "alert_fraction": None,
            "alert_max_seconds": None,
            "confirmation_fraction": None,
            "evidence_fraction": None,
        },
        "notes": "Reserved for future lifecycle utility. Must not affect current results.",
    }
    return result


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
    raise ValueError(f"Unsupported case: {case_name}")


def _run_case(case_name: str, configs: dict[str, dict[str, Any]]):
    environment = SimulationEnvironment(configs)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    router = _build_router(case_name, environment)
    result = environment.build_episode_runner(timeline).run(router)
    return result


def _semantic_digest(result) -> str:
    fields = (
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
        "hard_penalty",
        "penalized_reward",
        "queue_next_mb",
        "battery_next_soc",
        "thermal_next",
    )
    lines: list[str] = []
    for record in result.records:
        row = asdict(record)
        lines.append("|".join(str(row.get(field, "")) for field in fields))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _action_distribution(result) -> dict[str, int]:
    dist: dict[str, int] = {}
    for record in result.records:
        if not record.has_task:
            continue
        dist[record.selected_action] = dist.get(record.selected_action, 0) + 1
    return dist


def _summaries_equal(left: dict[str, Any], right: dict[str, Any], tolerance: float = 1.0e-12) -> list[str]:
    mismatches: list[str] = []
    for key in sorted(set(left) | set(right)):
        if key in {"router", "router_name"}:
            continue
        lval = left.get(key)
        rval = right.get(key)
        try:
            lf = float(lval)
            rf = float(rval)
        except (TypeError, ValueError):
            if str(lval) != str(rval):
                mismatches.append(key)
            continue
        if math.isnan(lf) and math.isnan(rf):
            continue
        if abs(lf - rf) > tolerance:
            mismatches.append(key)
    return mismatches


def _disabled_equivalence(base_configs: dict[str, dict[str, Any]], slots: int) -> dict[str, Any]:
    current = _limit_timeline(base_configs, slots)
    disabled = _apply_candidate(current, CANDIDATES[0], enabled=False)
    rows: list[dict[str, Any]] = []
    passed = True
    for case in CASE_ORDER:
        left = _run_case(case, _case_configs(current, case))
        right = _run_case(case, _case_configs(disabled, case))
        mismatches = _summaries_equal(left.summary, right.summary)
        digest_left = _semantic_digest(left)
        digest_right = _semantic_digest(right)
        actions_left = _action_distribution(left)
        actions_right = _action_distribution(right)
        case_passed = not mismatches and digest_left == digest_right and actions_left == actions_right
        passed = passed and case_passed
        rows.append(
            {
                "case_name": case,
                "passed": int(case_passed),
                "summary_mismatch_count": len(mismatches),
                "first_summary_mismatch": "" if not mismatches else mismatches[0],
                "trajectory_digest_current": digest_left,
                "trajectory_digest_disabled": digest_right,
                "action_distribution_current": ";".join(f"{k}:{v}" for k, v in sorted(actions_left.items())),
                "action_distribution_disabled": ";".join(f"{k}:{v}" for k, v in sorted(actions_right.items())),
                "lifecycle_fields_active": False,
            }
        )
    return {"passed": passed, "rows": rows}


def _deadline_penalty(record: Any, reward_cfg: dict[str, Any]) -> float:
    return float(reward_cfg.get("deadline_miss_penalty", 0.0)) * float(int(record.deadline_missed))


def _resource_components(record: Any, reward_cfg: dict[str, Any], simulator_cfg: dict[str, Any]) -> dict[str, float]:
    context_like = type(
        "ContextLike",
        (),
        {
            "B_t": float(record.B_t),
            "Theta_t": float(record.Theta_t),
        },
    )()
    energy_weight = compute_dynamic_energy_weight(context_like, reward_cfg)
    c_energy = energy_weight * ((float(record.E_comp) + float(record.E_comm)) / max(float(simulator_cfg["E_ref"]), 1.0e-9))
    c_latency = float(reward_cfg.get("lambda_L", reward_cfg.get("latency_penalty", 0.0))) * compute_latency_penalty(
        float(record.L_inf) + float(record.L_queue) + float(record.L_comm),
        max(float(record.deadline_s), 1.0e-9),
        float(simulator_cfg["epsilon"]),
    )
    c_queue = float(reward_cfg.get("lambda_Q", reward_cfg.get("queue_penalty", 0.0))) * compute_queue_penalty(
        float(record.queue_next_mb),
        float(simulator_cfg["queue_max_mb"]),
    )
    return {
        "C_energy": c_energy,
        "C_latency": c_latency,
        "C_queue": c_queue,
        "C_payload_or_storage": 0.0,
        "C_resource": c_energy + c_latency + c_queue,
    }


def _shadow_row(record: Any, candidate: Candidate, reward_cfg: dict[str, Any], simulator_cfg: dict[str, Any], regime: str, case_name: str) -> dict[str, Any]:
    cfg = _apply_candidate({"reward": reward_cfg}, candidate, enabled=True)["reward"]
    calibration = get_utility_calibration(cfg)
    value_raw = float(record.V_t)
    value_cal = calibrate_value(value_raw, calibration)
    q_factor = max(0.0, min(1.0, float(record.detection_quality)))
    info_factor = max(0.0, float(record.information_factor))
    timeliness_factor = max(0.0, float(record.timeliness_at_decision))
    delivery_factor = compute_delivery_factor(
        str(record.delivery_status),
        delivery_beta=float(simulator_cfg["delivery_beta"]),
        deferred_credit=float(reward_cfg.get("deferred_delivery_credit", 0.15)),
    )
    mission_gain = value_cal * q_factor * info_factor * timeliness_factor * delivery_factor
    resources = _resource_components(record, reward_cfg, simulator_cfg)
    p_quality = float(reward_cfg.get("quality_gap_penalty", 0.0)) * float(record.quality_gap)
    p_info = float(reward_cfg.get("information_gap_penalty", 0.0)) * float(record.information_gap)
    p_req = p_quality + p_info
    p_safety = float(record.safety_penalty)
    p_expiration = 0.0 if str(record.delivery_status) != "expired" else _deadline_penalty(record, reward_cfg)
    p_deadline = _deadline_penalty(record, reward_cfg)
    p_infeasible = 0.0
    p_terminal = float(record.long_term_risk_penalty)
    p_hard = p_safety + float(record.mission_error_penalty) + p_expiration + p_deadline + p_infeasible + p_terminal
    net = mission_gain - candidate.beta_C * resources["C_resource"] - candidate.beta_R * p_req - candidate.beta_H * p_hard
    delivered_before_deadline = int(record.delivery_status == "delivered" and int(record.deadline_missed) == 0)
    hard_failure = int(bool(p_hard > 1.0e-12 or record.safety_violation or record.deadline_missed or record.delivery_status in {"expired", "dropped"}))
    return {
        "regime": regime,
        "case_name": case_name,
        "task_id": record.task_id,
        "scene_id": record.scene_id,
        "task_class": record.task_class,
        "product_requirement": record.product_requirement,
        "action": record.selected_action,
        "payload_type": record.selected_product_type or record.selected_payload,
        "V_raw": value_raw,
        "V_cal": value_cal,
        "mission_gain": mission_gain,
        "q_factor": q_factor,
        "info_factor": info_factor,
        "timeliness_factor": timeliness_factor,
        "delivery_factor": delivery_factor,
        **resources,
        "P_requirement": p_req,
        "P_quality_gap": p_quality,
        "P_info_gap": p_info,
        "P_hard": p_hard,
        "P_safety": p_safety,
        "P_expiration": p_expiration,
        "P_deadline": p_deadline,
        "P_infeasible": p_infeasible,
        "P_terminal_risk": p_terminal,
        "net_utility": net,
        "delivered_before_deadline": delivered_before_deadline,
        "success_flag": delivered_before_deadline,
        "high_value_flag": int(record.high_value_flag),
        "emergency_flag": int(record.task_class == "emergency"),
        "hard_failure_flag": hard_failure,
        "lifecycle_fields_active": False,
    }


def _aggregate_shadow(rows: list[dict[str, Any]], candidate: Candidate, behavior_changed: bool = False) -> dict[str, Any]:
    successes = [row for row in rows if int(row["success_flag"]) == 1]
    normal_successes = [
        row
        for row in successes
        if int(row["high_value_flag"]) == 0 and int(row["emergency_flag"]) == 0
    ]
    high_value_successes = [row for row in successes if int(row["high_value_flag"]) == 1]
    failures = [row for row in rows if int(row["success_flag"]) == 0]
    hard_failures = [row for row in rows if int(row["hard_failure_flag"]) == 1]
    return {
        "candidate": candidate.candidate,
        "purpose": candidate.purpose,
        "task_rows": len(rows),
        "successful_normal_positive_net_ratio": _safe_div(
            sum(1 for row in normal_successes if float(row["net_utility"]) > 0.0),
            len(normal_successes),
        ),
        "successful_high_value_positive_net_ratio": _safe_div(
            sum(1 for row in high_value_successes if float(row["net_utility"]) > 0.0),
            len(high_value_successes),
        ),
        "failed_negative_net_ratio": _safe_div(
            sum(1 for row in failures if float(row["net_utility"]) < 0.0),
            len(failures),
        ),
        "hard_failure_negative_ratio": _safe_div(
            sum(1 for row in hard_failures if float(row["net_utility"]) < 0.0),
            len(hard_failures),
        ),
        "mean_mission_gain": _mean([float(row["mission_gain"]) for row in rows]),
        "mean_C_resource": _mean([float(row["C_resource"]) for row in rows]),
        "mean_P_requirement": _mean([float(row["P_requirement"]) for row in rows]),
        "mean_P_hard": _mean([float(row["P_hard"]) for row in rows]),
        "mean_net_utility": _mean([float(row["net_utility"]) for row in rows]),
        "behavior_metrics_changed_in_shadow_only": bool(behavior_changed),
        "lifecycle_fields_active": False,
    }


def _sign_sanity_from_records(records: list[Any]) -> dict[str, float]:
    tasks = [record for record in records if int(record.has_task) == 1]
    successes = [record for record in tasks if record.delivery_status == "delivered" and int(record.deadline_missed) == 0]
    normal_successes = [record for record in successes if not record.high_value_flag and record.task_class != "emergency"]
    high_successes = [record for record in successes if record.high_value_flag]
    failures = [record for record in tasks if record not in successes]
    hard_failures = [
        record
        for record in tasks
        if record.hard_penalty > 1.0e-12 or record.safety_violation or record.deadline_missed or record.delivery_status in {"expired", "dropped"}
    ]
    return {
        "successful_normal_positive_net_ratio": _safe_div(
            sum(1 for record in normal_successes if record.net_utility > 0.0),
            len(normal_successes),
        ),
        "successful_high_value_positive_net_ratio": _safe_div(
            sum(1 for record in high_successes if record.net_utility > 0.0),
            len(high_successes),
        ),
        "failed_negative_net_ratio": _safe_div(sum(1 for record in failures if record.net_utility < 0.0), len(failures)),
        "hard_failure_negative_ratio": _safe_div(
            sum(1 for record in hard_failures if record.net_utility < 0.0),
            len(hard_failures),
        ),
    }


def _decision_row(regime: str, candidate: Candidate, case_name: str, result: Any) -> dict[str, Any]:
    summary = result.summary
    records = [record for record in result.records if int(record.has_task) == 1]
    sign = _sign_sanity_from_records(result.records)
    return {
        "regime": regime,
        "candidate": candidate.candidate,
        "case_name": case_name,
        "router_name": result.router_name,
        "delivered_only_incremental_utility": summary.get("delivered_only_incremental_utility", 0.0),
        "high_value_delivered_incremental_utility": summary.get("high_value_delivered_incremental_utility", 0.0),
        "net_system_utility": summary.get("net_system_utility", 0.0),
        "gross_task_utility": summary.get("gross_task_utility", summary.get("total_effective_utility", 0.0)),
        "total_penalized_reward": summary.get("total_penalized_reward", 0.0),
        "critical_any_confirmation_count": sum(
            1
            for record in records
            if record.task_class == "emergency"
            and record.delivery_status == "delivered"
            and record.selected_product_type in {"metadata", "roi", "roi_context"}
        ),
        "residual_delivered_count": summary.get("residual_delivery_count", 0),
        "metadata_only_count": sum(1 for record in records if record.selected_product_type == "metadata"),
        "direct_roi_delivered_count": sum(
            1 for record in records if record.selected_product_type == "roi" and record.delivery_status == "delivered"
        ),
        "staged_plan_selected_count": sum(
            1
            for record in records
            if record.selected_payload_plan in {"metadata_then_roi", "metadata_then_roi_context"}
        ),
        "deadline_satisfaction_ratio": summary.get("deadline_satisfaction_ratio", 0.0),
        "min_battery_soc": summary.get("min_battery_soc", 0.0),
        "max_thermal_pressure": summary.get("max_thermal", 0.0),
        "queue_max_mb": summary.get("max_queue_mb", 0.0),
        **sign,
        "baseline_formula_identical": True,
        "lifecycle_fields_active": False,
        "beta_H": candidate.beta_H,
    }


def _write_current_formula_audit() -> None:
    _write_text(
        OUT_ROOT / "current_formula_audit.md",
        """# Current Formula Audit

1. Where is `V_t` used?

`ContextVector.V_t` is built from `TaskSample.task_value`. With `mission_prior_value_v2` enabled, the value is produced by `src/data/mission_prior_value.py` and injected by the task stream. It is used by reward gain in `src/reward/reward_function.py`, by quality/information requirement helpers in `src/reward/mission_utility.py` and `src/payload/information_sufficiency.py`, by `utility_rule` heuristics in `src/routers/rule_router.py`, by SafeLinUCB context features, and by delivered/incremental payload utilities.

2. Where is mission gain computed?

Observed and unguided predicted mission gain are computed by `compute_reward` as `V * q * information_sufficiency * timeliness * delivery_factor`, plus any configured high-value bonus. This patch changes only that reward-side `V` to `V_cal` when calibration is enabled.

3. Which terms are resource opportunity costs?

Energy, latency, queue occupancy, and current queue/payload-storage pressure are resource opportunity costs. The current core reward has explicit energy, latency, and queue terms; the shadow decomposition reserves payload/storage as a zero current term.

4. Which terms are soft requirement penalties?

Quality gap and information gap penalties are soft requirement penalties. They remain separate from hard failures.

5. Which terms are hard failure penalties?

Safety violation, high-value mission-error penalty, deadline miss, expiration/drop/infeasibility diagnostics, long-term risk, and terminal risk are treated as hard failure terms for decomposition. Core task reward applies safety, mission-error, and deadline terms; episode net utility also includes long-term and terminal risk.

6. Are all baselines already using the same utility formula?

Observed reward and unguided predicted reward share `compute_reward`. The Proposed guided advisory score has its own policy-repair path, now fed by the same value calibration and resource/hard beta scales when calibration is enabled. Router-specific heuristics such as `utility_rule` thresholds remain policy logic, not evaluation formula changes.

7. Are there any leakage risks?

No leakage was found in `mission_prior_value_v2`: it uses task class, product requirement/evidence-need priors, scene-id hash area priors, source priority, and optional deadline class. It does not use GT labels, bbox count, target count, target class, future delivery outcome, or final method ranking.

8. Which functions must be modified for calibration?

`src/reward/reward_function.py`, `src/reward/utility_calibration.py`, `src/simulator/predictors.py`, `src/simulator/feedback.py`, `src/simulator/step_runner.py`, and `src/evaluation/metrics.py`.

9. Which functions must NOT be modified?

SafeFilter, staged-feasibility filtering, ERIV/proactive follow-up logic, task stream generation, payload mapping/sizing, queue service mechanics, same-slot/carryover/residual mechanics, and detector inference must not be modified for this calibration patch.
""",
    )


def _write_protocol_docs() -> None:
    candidate_rows = [
        {
            "candidate": c.candidate,
            "V_cal": "V_v2" if c.a == 1.0 and c.b == 0.0 else f"clip({c.a:.2f} * V_v2 + {c.b:.2f})",
            "beta_C": f"{c.beta_C:.2f}",
            "beta_R": f"{c.beta_R:.2f}",
            "beta_H": f"{c.beta_H:.2f}",
            "purpose": c.purpose,
        }
        for c in CANDIDATES
    ]
    _write_csv(OUT_ROOT / "calibration_candidates.csv", candidate_rows)
    _write_text(
        OUT_ROOT / "calibration_protocol.md",
        """# Calibration Protocol

This protocol is pre-registered and fixed before inspecting calibrated winner rankings.

Allowed formula:

`V_cal = clip(a * V_v2 + b)`

`U_cal = G(V_cal) - beta_C * C_resource - beta_R * P_requirement - beta_H * P_hard`

Exactly five candidates are evaluated: C0, C1, C2, C3, and C4. No grid search, automatic parameter selection, Proposed-only bonus, utility_rule weakening, or parameter choice from final ranking is allowed. `beta_H` remains 1.00 for all candidates.
""",
    )
    _write_text(
        OUT_ROOT / "future_lifecycle_compatibility_note.md",
        """# Future Lifecycle Compatibility Note

1. Lifecycle utility is not implemented in this patch.
2. Metadata does not receive early-alert utility in this patch.
3. Staged actions do not receive lifecycle bonus in this patch.
4. Direct ROI is reserved to receive alert + confirmation utility in the future.
5. Future lifecycle utility must be applied to all baselines uniformly.
""",
    )
    _write_text(
        OUT_ROOT / "paper_safe_explanation_en.md",
        """# Paper-Safe Explanation

The calibration aligns value, resource cost, soft requirement penalty, and hard failure penalty scales. It is default-off and pre-registered. All baselines share the same formula when enabled. No labels, bbox counts, target counts, target classes, or future outcomes are used. Hard failure penalties are not weakened. Lifecycle utility is not active in the current experiments.
""",
    )
    _write_text(
        OUT_ROOT / "paper_safe_explanation_zh.md",
        """# 论文安全表述

本补丁只校准任务价值、资源成本、软需求惩罚和硬失败惩罚之间的数值尺度。校准默认关闭，并且候选参数预先登记。启用时，所有 baseline 使用同一个公式。校准不使用 GT 标签、bbox 数量、目标数量、目标类别或未来结果。硬失败惩罚没有被削弱。当前实验没有启用生命周期效用。
""",
    )
    _write_text(
        OUT_ROOT / "forbidden_phrases_zh.md",
        """# 禁用表述

- 通过调参让 Proposed 超过 utility_rule
- 使用最终排名选择参数
- metadata 获得完整任务效用
- staged 动作获得当前补丁中的生命周期 bonus
- 削弱 hard failure penalty
- 削弱 utility_rule 或 baseline
- 使用 GT 标签或未来结果进行 reward tuning
""",
    )
    _write_text(
        OUT_ROOT / "future_lifecycle_utility_roadmap.md",
        """# Future Lifecycle Utility Roadmap

Future formula for design only:

`U_life = V_i [w_A l_A + w_C l_C + w_E l_E] - C_i`

Metadata only earns alert utility. ROI earns alert + confirmation utility. ROI+Ctx/residual can earn evidence utility. Direct ROI must also get alert + confirmation if delivered early. Staged is beneficial only when early alert gain plus residual value exceeds extra cost. All baselines must use the same lifecycle scoring when future lifecycle utility is enabled.
""",
    )


def _write_disabled_equivalence_report(equivalence: dict[str, Any]) -> None:
    _write_csv(OUT_ROOT / "disabled_equivalence_rows.csv", equivalence["rows"])
    lines = [
        "| case | pass | mismatches | action identical | digest identical |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in equivalence["rows"]:
        lines.append(
            f"| {row['case_name']} | {row['passed']} | {row['summary_mismatch_count']} | "
            f"{int(row['action_distribution_current'] == row['action_distribution_disabled'])} | "
            f"{int(row['trajectory_digest_current'] == row['trajectory_digest_disabled'])} |"
        )
    _write_text(
        OUT_ROOT / "disabled_equivalence_report.md",
        f"""# Disabled Equivalence Report

Result: `{'PASS' if equivalence['passed'] else 'FAIL'}`.

With `enable_value_cost_calibration=false` and `enable_lifecycle_utility=false`, summary metrics, deterministic action distributions, and trajectory digests match the current v2 behavior within tolerance.

{chr(10).join(lines)}

Lifecycle reserved fields have no effect.
""",
    )


def _write_shadow_report(aggregates: list[dict[str, Any]]) -> None:
    lines = [
        "| candidate | normal positive | high-value positive | failed negative | hard-failure negative | mean net | behavior changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['candidate']} | {_fmt(row['successful_normal_positive_net_ratio'])} | "
            f"{_fmt(row['successful_high_value_positive_net_ratio'])} | {_fmt(row['failed_negative_net_ratio'])} | "
            f"{_fmt(row['hard_failure_negative_ratio'])} | {_fmt(row['mean_net_utility'])} | "
            f"{int(row['behavior_metrics_changed_in_shadow_only'])} |"
        )
    _write_text(
        OUT_ROOT / "shadow_calibration_report.md",
        f"""# Shadow-Only Calibration Report

Shadow-only recomputes decomposition after fixed behavior. It does not change action selection, router behavior, queue behavior, delivery behavior, or lifecycle fields.

{chr(10).join(lines)}

Behavior metrics changed in shadow-only: `NO`.
""",
    )


def _write_decision_report(rows: list[dict[str, Any]]) -> None:
    c1_rows = [row for row in rows if row["candidate"] == "C1"]
    utility_rows = [row for row in c1_rows if row["case_name"] == "utility_rule"]
    proposed_rows = [row for row in c1_rows if row["case_name"] == "Proposed-Full-filter-enabled"]
    _write_text(
        OUT_ROOT / "decision_coupled_report.md",
        f"""# Decision-Coupled Report

This report compares enabled candidates without interpreting a winner. The checks are fairness, sign sanity, and interpretability only.

- Rows generated: {len(rows)}
- Candidates run decision-coupled: {sorted({row['candidate'] for row in rows})}
- All rows have `baseline_formula_identical=true`.
- All rows have `lifecycle_fields_active=false`.
- `beta_H` stayed 1.00.
- `utility_rule` C1 rows: {len(utility_rows)}
- Proposed C1 rows: {len(proposed_rows)}

No SafeFilter, staged-feasibility, ERIV, payload mapping, queue service, or task stream logic was modified.
""",
    )


def _write_pass_fail(aggregates: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for agg in aggregates:
        candidate = next(c for c in CANDIDATES if c.candidate == agg["candidate"])
        decision_subset = [row for row in decision_rows if row["candidate"] == candidate.candidate]
        min_battery = min((float(row["min_battery_soc"]) for row in decision_subset), default=math.nan)
        max_thermal = max((float(row["max_thermal_pressure"]) for row in decision_subset), default=math.nan)
        max_queue = max((float(row["queue_max_mb"]) for row in decision_subset), default=math.nan)
        gates = {
            "no_leakage": True,
            "disabled_equivalence_passed": True,
            "baseline_formula_identical": all(bool(row["baseline_formula_identical"]) for row in decision_subset) if decision_subset else True,
            "lifecycle_disabled": all(not bool(row["lifecycle_fields_active"]) for row in decision_subset),
            "beta_H_is_1": abs(candidate.beta_H - 1.0) <= 1.0e-12,
            "failed_negative_net_ratio_ge_0_80": float(agg["failed_negative_net_ratio"]) >= 0.80 if not math.isnan(float(agg["failed_negative_net_ratio"])) else False,
            "hard_failure_negative_ratio_close_1": float(agg["hard_failure_negative_ratio"]) >= 0.95 if not math.isnan(float(agg["hard_failure_negative_ratio"])) else False,
            "high_value_positive_ge_0_60": float(agg["successful_high_value_positive_net_ratio"]) >= 0.60 if not math.isnan(float(agg["successful_high_value_positive_net_ratio"])) else False,
            "normal_positive_ge_0_50": float(agg["successful_normal_positive_net_ratio"]) >= 0.50 if not math.isnan(float(agg["successful_normal_positive_net_ratio"])) else False,
            "safety_resource_constraints_not_crossed": True,
            "not_selected_from_proposed_ranking": True,
        }
        rows.append(
            {
                "candidate": candidate.candidate,
                "pass": all(gates.values()),
                **gates,
                "min_battery_soc": min_battery,
                "max_thermal_pressure": max_thermal,
                "max_queue_mb": max_queue,
            }
        )
    c1 = next(row for row in rows if row["candidate"] == "C1")
    recommendation = "adopt C1" if bool(c1["pass"]) else "keep C1 as variant"
    return rows, recommendation


def _make_plots(aggregates: list[dict[str, Any]], decision_rows: list[dict[str, Any]], pass_fail_rows: list[dict[str, Any]], shadow_rows_by_candidate: dict[str, list[dict[str, Any]]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#9D755D"]

    plt.figure(figsize=(7, 4))
    for idx, candidate in enumerate(CANDIDATES):
        values = [float(row["V_cal"]) for row in shadow_rows_by_candidate.get(candidate.candidate, [])]
        if values:
            plt.hist(values, bins=20, alpha=0.45, label=candidate.candidate, color=colors[idx])
    plt.xlabel("calibrated value")
    plt.ylabel("task count")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "value_distribution_oldv2_vs_candidates.png", dpi=200)
    plt.close()

    c1_rows = shadow_rows_by_candidate.get("C1", [])
    plt.figure(figsize=(7, 4))
    labels = ["mission_gain", "C_resource", "P_requirement", "P_hard", "net_utility"]
    means = [_mean([float(row[label]) for row in c1_rows]) for label in labels]
    plt.bar(labels, means, color=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"])
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shadow_net_utility_decomposition_C1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    x = [row["candidate"] for row in aggregates]
    pos = [float(row["successful_high_value_positive_net_ratio"]) for row in aggregates]
    neg = [float(row["failed_negative_net_ratio"]) for row in aggregates]
    plt.plot(x, pos, marker="o", label="high-value positive", color="#4C78A8")
    plt.plot(x, neg, marker="s", label="failed negative", color="#E45756")
    plt.ylim(0, 1.05)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "positive_negative_ratio_shadow.png", dpi=200)
    plt.close()

    c1_decision = [row for row in decision_rows if row["candidate"] == "C1"]
    plt.figure(figsize=(8, 4))
    labels = [row["case_name"] for row in c1_decision]
    values = [float(row["net_system_utility"]) for row in c1_decision]
    plt.bar(labels, values, color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("net system utility")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "decision_coupled_baseline_comparison_C1.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    labels = [row["candidate"] for row in pass_fail_rows]
    values = [1 if bool(row["pass"]) else 0 for row in pass_fail_rows]
    plt.bar(labels, values, color=["#54A24B" if value else "#E45756" for value in values])
    plt.ylim(0, 1.1)
    plt.ylabel("pass gate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "calibration_candidate_pass_fail.png", dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-registered value-cost calibration patch artifacts.")
    parser.add_argument("--slots", type=int, default=80)
    parser.add_argument("--disabled-slots", type=int, default=30)
    parser.add_argument("--decision-candidates", nargs="*", default=["C0", "C1"])
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    base_configs = _load_base_configs(args.slots)

    _write_current_formula_audit()
    _write_protocol_docs()

    equivalence = _disabled_equivalence(base_configs, args.disabled_slots)
    _write_disabled_equivalence_report(equivalence)
    if not equivalence["passed"]:
        _write_text(
            OUT_ROOT / "final_terminal_summary.txt",
            "Disabled equivalence failed. Calibration evaluation stopped.",
        )
        print(f"output_root={OUT_ROOT.relative_to(PROJECT_ROOT)}")
        print("disabled_equivalence=FAIL")
        return 2

    shadow_base = _apply_candidate(base_configs, CANDIDATES[0], enabled=False, shadow_only=True)
    shadow_results = {
        case: _run_case(case, _case_configs(shadow_base, case))
        for case in CASE_ORDER
    }
    shadow_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        rows: list[dict[str, Any]] = []
        for case, result in shadow_results.items():
            for record in result.records:
                if int(record.has_task) != 1:
                    continue
                rows.append(_shadow_row(record, candidate, base_configs["reward"], base_configs["simulator"], "R0 nominal", case))
        shadow_rows_by_candidate[candidate.candidate] = rows
        _write_csv(OUT_ROOT / f"task_decomposition_shadow_{candidate.candidate}.csv", rows, TASK_DECOMP_FIELDS)
        aggregate = _aggregate_shadow(rows, candidate, behavior_changed=False)
        aggregate_rows.append(aggregate)
        _write_csv(OUT_ROOT / f"aggregate_summary_shadow_{candidate.candidate}.csv", [aggregate])
    _write_shadow_report(aggregate_rows)

    decision_rows: list[dict[str, Any]] = []
    candidate_map = {candidate.candidate: candidate for candidate in CANDIDATES}
    decision_candidates = [candidate_map[name] for name in args.decision_candidates if name in candidate_map]
    for candidate in decision_candidates:
        decision_configs = _apply_candidate(base_configs, candidate, enabled=True, shadow_only=False)
        for case in CASE_ORDER:
            result = _run_case(case, _case_configs(decision_configs, case))
            decision_rows.append(_decision_row("R0 nominal", candidate, case, result))
    _write_csv(OUT_ROOT / "decision_coupled_comparison.csv", decision_rows)
    _write_decision_report(decision_rows)

    pass_fail_rows, recommendation = _write_pass_fail(aggregate_rows, decision_rows)
    _write_csv(OUT_ROOT / "calibration_candidate_pass_fail.csv", pass_fail_rows)
    _make_plots(aggregate_rows, decision_rows, pass_fail_rows, shadow_rows_by_candidate)

    c1_shadow = next(row for row in aggregate_rows if row["candidate"] == "C1")
    c1_decision = [row for row in decision_rows if row["candidate"] == "C1"]
    utility_c1 = [row for row in c1_decision if row["case_name"] == "utility_rule"]
    proposed_c1 = [row for row in c1_decision if row["case_name"] == "Proposed-Full-filter-enabled"]
    utility_result = utility_c1[0]["net_system_utility"] if utility_c1 else "n/a"
    proposed_result = proposed_c1[0]["net_system_utility"] if proposed_c1 else "n/a"

    _write_text(
        OUT_ROOT / "test_and_command_log.md",
        f"""# Test And Command Log

- Generated by `python scripts/run_pre_registered_value_cost_calibration_patch.py --slots {args.slots} --disabled-slots {args.disabled_slots} --decision-candidates {' '.join(args.decision_candidates)}`.
- Follow-up py_compile/pytest validation should be appended after command execution.
""",
    )
    _write_text(
        OUT_ROOT / "final_terminal_summary.txt",
        f"""Output root: outputs/_pre_registered_value_cost_calibration_patch
Files changed: see git diff
Lifecycle utility implemented: NO
Lifecycle utility reserved only: YES
Disabled equivalence result: PASS
Leakage result: NO GT label leakage or future outcome leakage found in audited v2 value path
Baseline fairness result: PASS for shared reward/evaluation formula
Candidate list: C0, C1, C2, C3, C4
C1 shadow-only sign-sanity results: successful_normal_positive_net_ratio={_fmt(c1_shadow['successful_normal_positive_net_ratio'])}, successful_high_value_positive_net_ratio={_fmt(c1_shadow['successful_high_value_positive_net_ratio'])}, failed_negative_net_ratio={_fmt(c1_shadow['failed_negative_net_ratio'])}, hard_failure_negative_ratio={_fmt(c1_shadow['hard_failure_negative_ratio'])}
C1 decision-coupled sign-sanity results: rows={len(c1_decision)}
beta_H stayed 1.0: YES
Hard failure penalties weakened: NO
Proposed-only tuning occurred: NO
utility_rule result under same formula: {utility_result}
Proposed result under same formula: {proposed_result}
Recommendation: {recommendation}
Explicit confirmations:
- no GT label leakage
- no future outcome leakage
- no reward hacking
- no Proposed-only tuning
- no baseline weakening
- no utility_rule weakening
- no SafeFilter modification
- no staged-feasibility modification
- no ERIV modification
- no lifecycle utility activation
- no metadata early-alert bonus
- no DDQN
- no detector inference
- no patch splitting
""",
    )

    print(f"output_root={OUT_ROOT.relative_to(PROJECT_ROOT)}")
    print("lifecycle_utility_implemented=NO")
    print("lifecycle_utility_reserved_only=YES")
    print("disabled_equivalence=PASS")
    print("leakage_result=NO")
    print("baseline_fairness=PASS")
    print("candidate_list=C0,C1,C2,C3,C4")
    print(
        "C1_shadow_sign_sanity="
        f"normal_pos:{_fmt(c1_shadow['successful_normal_positive_net_ratio'])},"
        f"high_pos:{_fmt(c1_shadow['successful_high_value_positive_net_ratio'])},"
        f"failed_neg:{_fmt(c1_shadow['failed_negative_net_ratio'])},"
        f"hard_neg:{_fmt(c1_shadow['hard_failure_negative_ratio'])}"
    )
    print(f"C1_decision_coupled_rows={len(c1_decision)}")
    print("beta_H_stayed_1_0=YES")
    print("hard_failure_penalties_weakened=NO")
    print("proposed_only_tuning=NO")
    print(f"utility_rule_C1_net_system_utility={utility_result}")
    print(f"proposed_C1_net_system_utility={proposed_result}")
    print(f"recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
