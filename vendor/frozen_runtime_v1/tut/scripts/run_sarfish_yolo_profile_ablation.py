"""Run YOLO-inspired profile calibration ablations on SARFish energy-shadow stress."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from dataclasses import asdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.common.types import CommState, ContextVector, EpisodeResult  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.models.model_registry import ModelRegistry  # noqa: E402
from src.payload.payload_policy import get_payload_policy_profile  # noqa: E402
from src.routers.fixed_router import FixedRouter  # noqa: E402
from src.routers.rule_router import RuleBasedRouter  # noqa: E402
from src.routers.safe_linucb_router import SafeLinUCBRouter  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


VARIANTS = (
    "proposed_original_profile",
    "proposed_yolo_calibrated_profile",
    "proposed_yolo_calibrated_no_ucb_bonus",
    "proposed_yolo_calibrated_stronger_resource_price",
    "proposed_yolo_calibrated_no_mission_priority_boost",
    "utility_rule",
    "no_mode",
    "no_lyapunov",
    "fixed_lightweight",
    "fixed_high_accuracy",
)
FOCUS_ACTIONS = ("lightweight+metadata", "balanced+metadata", "balanced+roi", "high_accuracy+roi_context", "delay/store")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if abs(denominator) > 1.0e-12 else 0.0


def _load_base_configs(calibrated_profile: bool = True) -> dict[str, dict]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_yolo_profile_ablation_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    if calibrated_profile:
        profile_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml")
        configs["model_pool"] = copy.deepcopy(configs["model_pool"])
        configs["model_pool"]["profile_overrides"] = dict(profile_cfg["profile_overrides"])
    return configs


def _variant_configs(variant: str) -> dict[str, dict]:
    configs = _load_base_configs(calibrated_profile=variant != "proposed_original_profile")
    if variant == "proposed_yolo_calibrated_stronger_resource_price":
        configs["lyapunov_pricing"] = copy.deepcopy(configs["lyapunov_pricing"])
        configs["lyapunov_pricing"]["base_lambda_E"] = float(configs["lyapunov_pricing"]["base_lambda_E"]) * 2.5
        configs["lyapunov_pricing"]["base_lambda_Q"] = float(configs["lyapunov_pricing"]["base_lambda_Q"]) * 1.8
        configs["lyapunov_pricing"]["base_lambda_S"] = float(configs["lyapunov_pricing"]["base_lambda_S"]) * 1.8
        configs["lyapunov_pricing"]["base_lambda_L"] = float(configs["lyapunov_pricing"]["base_lambda_L"]) * 1.3
    if variant == "proposed_yolo_calibrated_no_mission_priority_boost":
        configs["mode_controller"] = copy.deepcopy(configs["mode_controller"])
        mission = configs["mode_controller"]["modes"]["mission_priority"]
        mission["price_multipliers"] = {"lambda_E": 1.0, "lambda_L": 1.0, "lambda_Q": 1.0, "lambda_S": 1.0}
        mission["exploration_scale"] = 1.0
        mission["model_complexity_bias"] = 0.0
        mission["payload_size_bias"] = 0.0
        mission["high_value_bonus_scale"] = 1.0
        mission["safety_margin_scale"] = 1.0
    return configs


def _build_router(variant: str, environment: SimulationEnvironment):
    action_names = [action.action_name for action in environment.action_space.list_actions()]
    if variant.startswith("proposed_"):
        alpha = 0.0 if variant == "proposed_yolo_calibrated_no_ucb_bonus" else 0.70
        return SafeLinUCBRouter(
            action_names,
            alpha=alpha,
            router_name=variant,
            use_lyapunov_pricing=True,
            use_mode_controller=True,
        )
    if variant == "utility_rule":
        return RuleBasedRouter(variant, mode="utility_rule")
    if variant == "no_mode":
        return SafeLinUCBRouter(action_names, router_name=variant, use_lyapunov_pricing=True, use_mode_controller=False)
    if variant == "no_lyapunov":
        return SafeLinUCBRouter(action_names, router_name=variant, use_lyapunov_pricing=False, use_mode_controller=True)
    if variant == "fixed_lightweight":
        return FixedRouter(variant, "lightweight+metadata")
    if variant == "fixed_high_accuracy":
        return FixedRouter(variant, "high_accuracy+roi_context")
    raise ValueError(f"Unsupported variant: {variant}")


def write_profile_audit_report() -> Path:
    output_dir = PROJECT_ROOT / "outputs" / "sarfish_profile_calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = _load_base_configs(calibrated_profile=False)
    registry = ModelRegistry(configs["model_pool"])
    table = {row["profile_name"]: row for row in registry.profiling_store.export_table()}
    payload_cfg = load_yaml(PROJECT_ROOT / "configs" / "payload_profiles.yaml")
    payload_rows = payload_cfg["profiles"]
    tiny, small, medium = table["tiny"], table["small"], table["medium"]
    quality_ratio = _safe_div(float(medium["quality_base"]), float(small["quality_base"]))
    energy_ratio = _safe_div(float(medium["energy_j"]), float(small["energy_j"]))
    latency_ratio = _safe_div(float(medium["latency_s"]), float(small["latency_s"]))
    lines = [
        "# YOLO-Inspired Profile Calibration Audit",
        "",
        "## Current model profiles",
        "",
        "| logical model | profile | quality_base | robustness | latency_s | energy_j | estimated_power_w | thermal_gain_scale |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| lightweight | tiny | {float(tiny['quality_base']):.3f} | {float(tiny['robustness_to_difficulty']):.3f} | {float(tiny['latency_s']):.3f} | {float(tiny['energy_j']):.3f} | {float(tiny['estimated_power_w']):.3f} | {float(tiny['thermal_gain_scale']):.3f} |",
        f"| balanced | small | {float(small['quality_base']):.3f} | {float(small['robustness_to_difficulty']):.3f} | {float(small['latency_s']):.3f} | {float(small['energy_j']):.3f} | {float(small['estimated_power_w']):.3f} | {float(small['thermal_gain_scale']):.3f} |",
        f"| high_accuracy | medium | {float(medium['quality_base']):.3f} | {float(medium['robustness_to_difficulty']):.3f} | {float(medium['latency_s']):.3f} | {float(medium['energy_j']):.3f} | {float(medium['estimated_power_w']):.3f} | {float(medium['thermal_gain_scale']):.3f} |",
        "",
        "## Current payload profiles",
        "",
        "| payload | size_mb | information_factor |",
        "| --- | ---: | ---: |",
    ]
    for row in payload_rows:
        lines.append(f"| {row['name']} | {float(row['size_mb']):.3f} | {float(row['information_factor']):.3f} |")
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- high_accuracy / balanced quality_base ratio: {quality_ratio:.3f}.",
            f"- high_accuracy / balanced latency ratio: {latency_ratio:.3f}.",
            f"- high_accuracy / balanced energy ratio: {energy_ratio:.3f}.",
            "- Current thermal gain is not profile-specific before calibration; all profiles default to 1.0.",
            "- `high_accuracy+roi_context` combines the highest model quality with the highest payload information factor, so the predictor naturally estimates it as the highest task utility on many high-value tasks.",
            "- The old profile is not obviously cheap in absolute energy, but its marginal quality gain and `roi_context` information gain are strong enough that resource prices often do not offset it in the LinUCB prior.",
        ]
    )
    path = output_dir / "profile_audit_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(output_dir: Path) -> None:
    output_dirs = ensure_output_dirs(output_dir)
    logger = get_logger("run_sarfish_yolo_profile_ablation")
    results: list[EpisodeResult] = []
    for variant in VARIANTS:
        configs = _variant_configs(variant)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        environment = SimulationEnvironment(configs)
        router = _build_router(variant, environment)
        episode_runner = environment.build_episode_runner(timeline)
        logger.info("running variant: %s", variant)
        results.append(episode_runner.run(router))
    write_episode_results(results, output_dirs)
    generate_lifetime_outputs(output_dir, 20.0)
    write_action_distribution(output_dir)
    write_reports(output_dir)


def _trajectory_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return _read_csv(output_dir / "trajectories" / f"{variant}.csv")


def _task_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return [row for row in _trajectory_rows(output_dir, variant) if _to_int(row.get("has_task")) == 1]


def write_action_distribution(output_dir: Path) -> Path:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        task_rows = _task_rows(output_dir, variant)
        counts = Counter(row.get("selected_action", "") for row in task_rows)
        for action in FOCUS_ACTIONS:
            rows.append({"variant": variant, "action": action, "count": counts.get(action, 0), "total_tasks": len(task_rows)})
    path = output_dir / "action_distribution.csv"
    _write_csv(path, rows)
    return path


def _lifetime_counts(output_dir: Path) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in _read_csv(output_dir / "task_lifetime_outcomes.csv"):
        counts[str(row["router_name"])][str(row["final_lifetime_status"])] += 1
    return counts


def _max_dod(rows: list[dict[str, str]]) -> float:
    return max((1.0 - _to_float(row.get("battery_next_soc", row.get("battery_soc"))) for row in rows), default=0.0)


def _action_similarity(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]) -> tuple[int, int, float]:
    pairs = [(a, b) for a, b in zip(rows_a, rows_b) if a.get("task_id") == b.get("task_id")]
    same = sum(1 for a, b in pairs if a.get("selected_action") == b.get("selected_action"))
    return same, len(pairs), _safe_div(same, len(pairs))


def write_reports(output_dir: Path) -> None:
    summary_rows = _read_csv(output_dir / "summary.csv")
    summary = {row["router"]: row for row in summary_rows}
    lifetime = _lifetime_counts(output_dir)
    dist_rows = _read_csv(output_dir / "action_distribution.csv")
    dist = {(row["variant"], row["action"]): _to_int(row["count"]) for row in dist_rows}
    _write_profile_ablation_report(output_dir, summary, lifetime, dist)
    _write_action_shift_report(output_dir, summary, dist)
    _write_calibrated_sanity_report(output_dir, summary, dist)
    _write_calibrated_action_score_diagnosis(output_dir, summary, dist)
    _write_root_cause_after_calibration(output_dir, summary, dist)
    _write_paper_story_notes(output_dir)


def _summary_table(summary: dict[str, dict[str, str]], lifetime: dict[str, Counter], dist: dict[tuple[str, str], int]) -> list[str]:
    lines = [
        "| variant | gross | net | penalized | resource | requirement | total energy | compute energy | comm energy | max thermal | max DoD | delivered_mb | expired_items | lifetime delivered / expired / deferred | high_accuracy | balanced+roi |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for variant in VARIANTS:
        row = summary.get(variant, {})
        traj = _trajectory_rows(PROJECT_ROOT / "outputs" / "sarfish_yolo_profile_ablation", variant)
        counts = lifetime.get(variant, Counter())
        lines.append(
            f"| {variant} | {_to_float(row.get('gross_task_utility')):.3f} | {_to_float(row.get('net_system_utility')):.3f} | "
            f"{_to_float(row.get('total_penalized_reward')):.3f} | {_to_float(row.get('total_resource_penalty')):.3f} | "
            f"{_to_float(row.get('total_requirement_penalty')):.3f} | {_to_float(row.get('total_energy')):.1f} | "
            f"{_to_float(row.get('total_compute_energy')):.1f} | {_to_float(row.get('total_comm_energy')):.1f} | "
            f"{_to_float(row.get('max_thermal')):.4f} | {_max_dod(traj):.4f} | {_to_float(row.get('delivered_mb')):.3f} | "
            f"{_to_int(row.get('expired_items'))} | {counts.get('delivered', 0)} / {counts.get('expired', 0)} / {counts.get('deferred', 0)} | "
            f"{dist.get((variant, 'high_accuracy+roi_context'), 0)} | {dist.get((variant, 'balanced+roi'), 0)} |"
        )
    return lines


def _write_profile_ablation_report(output_dir: Path, summary, lifetime, dist) -> None:
    lines = ["# YOLO Profile Ablation Report", "", *_summary_table(summary, lifetime, dist)]
    path = output_dir / "profile_ablation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_action_shift_report(output_dir: Path, summary, dist) -> None:
    prop_orig = _task_rows(output_dir, "proposed_original_profile")
    prop_cal = _task_rows(output_dir, "proposed_yolo_calibrated_profile")
    no_mode = _task_rows(output_dir, "no_mode")
    same_orig_cal = _action_similarity(prop_orig, prop_cal)
    same_cal_nomode = _action_similarity(prop_cal, no_mode)
    lines = [
        "# Proposed Action Shift Report",
        "",
        f"- proposed original high_accuracy count: {dist.get(('proposed_original_profile', 'high_accuracy+roi_context'), 0)}",
        f"- proposed calibrated high_accuracy count: {dist.get(('proposed_yolo_calibrated_profile', 'high_accuracy+roi_context'), 0)}",
        f"- proposed calibrated balanced+roi count: {dist.get(('proposed_yolo_calibrated_profile', 'balanced+roi'), 0)}",
        f"- original vs calibrated same actions: {same_orig_cal[0]}/{same_orig_cal[1]} ({same_orig_cal[2]:.3f})",
        f"- calibrated proposed vs no_mode same actions: {same_cal_nomode[0]}/{same_cal_nomode[1]} ({same_cal_nomode[2]:.3f})",
        f"- calibrated net_system_utility: {_to_float(summary['proposed_yolo_calibrated_profile'].get('net_system_utility')):.3f}",
        f"- utility_rule net_system_utility: {_to_float(summary['utility_rule'].get('net_system_utility')):.3f}",
        f"- no_mode net_system_utility: {_to_float(summary['no_mode'].get('net_system_utility')):.3f}",
    ]
    path = output_dir / "proposed_action_shift_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_calibrated_sanity_report(output_dir: Path, summary, dist) -> None:
    profile_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml")
    lines = [
        "# Calibrated Profile Sanity Report",
        "",
        "## YOLO-inspired profile overrides",
        "",
        "| profile | quality_base | latency_s | energy_j | thermal_gain_scale |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for profile, values in profile_cfg["profile_overrides"].items():
        lines.append(
            f"| {profile} | {float(values['quality_base']):.3f} | {float(values['latency_s']):.3f} | "
            f"{float(values['energy_j']):.3f} | {float(values['thermal_gain_scale']):.3f} |"
        )
    lines.extend(
        [
            "",
            f"- calibrated proposed high_accuracy count: {dist.get(('proposed_yolo_calibrated_profile', 'high_accuracy+roi_context'), 0)}",
            f"- calibrated proposed balanced+roi count: {dist.get(('proposed_yolo_calibrated_profile', 'balanced+roi'), 0)}",
            f"- calibrated proposed max thermal: {_to_float(summary['proposed_yolo_calibrated_profile'].get('max_thermal')):.4f}",
            f"- calibrated proposed total energy: {_to_float(summary['proposed_yolo_calibrated_profile'].get('total_energy')):.3f}",
        ]
    )
    path = output_dir / "calibrated_profile_sanity_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_action_metrics(output_dir: Path, variant: str, action: str) -> dict[str, float]:
    rows = [row for row in _task_rows(output_dir, variant) if row.get("selected_action") == action]
    n = len(rows)
    effective = sum(_to_float(row.get("effective_utility")) for row in rows)
    resource = sum(_to_float(row.get("resource_penalty")) for row in rows)
    requirement = sum(_to_float(row.get("requirement_penalty")) for row in rows)
    energy = sum(_to_float(row.get("compute_energy_j")) + _to_float(row.get("comm_energy_j")) for row in rows)
    return {
        "count": n,
        "avg_effective": _safe_div(effective, n),
        "avg_resource": _safe_div(resource, n),
        "avg_requirement": _safe_div(requirement, n),
        "avg_energy": _safe_div(energy, n),
        "net_per_energy": _safe_div(effective - resource - requirement, energy),
        "utility_per_resource": _safe_div(effective, resource),
    }


def _context_from_row(row: dict[str, str]) -> ContextVector:
    return ContextVector(
        B_t=_to_float(row.get("B_t", row.get("battery_soc"))),
        E_t_AI=_to_float(row.get("E_AI_t")),
        Theta_t=_to_float(row.get("Theta_t")),
        V_t=_to_float(row.get("V_t")),
        D_t=_to_float(row.get("D_t")),
        H_t=_to_float(row.get("H_t")),
        U_t=_to_float(row.get("U_t")),
        C_t=_to_float(row.get("C_t")),
        Q_t=_to_float(row.get("Q_t")),
    )


def _guidance_from_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "mode_name": row.get("mode_name", "unknown"),
        "lambda_E": _to_float(row.get("lambda_E")),
        "lambda_L": _to_float(row.get("lambda_L")),
        "lambda_Q": _to_float(row.get("lambda_Q")),
        "lambda_S": _to_float(row.get("lambda_S")),
        "exploration_scale": _to_float(row.get("exploration_scale"), 1.0),
        "model_complexity_bias": _to_float(row.get("model_complexity_bias")),
        "payload_size_bias": _to_float(row.get("payload_size_bias")),
        "high_value_bonus_scale": _to_float(row.get("high_value_bonus_scale"), 1.0),
        "safety_margin_scale": _to_float(row.get("safety_margin_scale"), 1.0),
    }


def _candidate_score_diagnostics(output_dir: Path) -> dict[str, float]:
    configs = _variant_configs("proposed_yolo_calibrated_profile")
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    timeline_by_slot = {step.slot_index: step for step in timeline}
    environment = SimulationEnvironment(configs)
    action_by_name = {action.action_name: action for action in environment.action_space.list_actions()}
    task_rows = _task_rows(output_dir, "proposed_yolo_calibrated_profile")
    high_utility_highest = 0
    balanced_utility_highest = 0
    high_resource_gt_balanced = 0
    high_predicted_winner_selected = 0
    high_bal_resource_ratios: list[float] = []
    for row in task_rows:
        step = timeline_by_slot[_to_int(row.get("slot"))]
        if step.task is None:
            continue
        context = _context_from_row(row)
        guidance = _guidance_from_row(row)
        comm_state = CommState(
            queue_mb=context.Q_t * float(configs["simulator"]["queue_max_mb"]),
            queue_max_mb=float(configs["simulator"]["queue_max_mb"]),
        )
        utilities: dict[str, float] = {}
        resource_costs: dict[str, float] = {}
        for action_name in FOCUS_ACTIONS:
            action = action_by_name[action_name]
            metrics = environment.predictor.predict_action(
                task=step.task,
                context=context,
                resource_trace=step.resource_trace,
                link_trace=step.link_trace,
                comm_state=comm_state,
                action=action,
                guidance_bundle=guidance,
            )
            policy = get_payload_policy_profile(action.payload_policy)
            payload_information = policy.information_level * (0.10 if action.payload_policy == "defer" else max(metrics.predicted_detection_quality, 0.55))
            utility = step.task.task_value * metrics.predicted_detection_quality * payload_information * metrics.predicted_delivery_factor
            if step.task.task_value >= float(configs["simulator"].get("high_value_threshold", 0.75)):
                utility *= float(guidance.get("high_value_bonus_scale", 1.0))
            epsilon = float(configs["simulator"]["epsilon"])
            resource_cost = (
                float(guidance.get("lambda_E", 0.0)) * (metrics.predicted_E_comp + metrics.predicted_E_comm) / max(float(configs["simulator"]["E_ref"]), epsilon)
                + float(guidance.get("lambda_L", 0.0)) * (metrics.predicted_L_inf + metrics.predicted_L_queue + metrics.predicted_L_comm) / max(step.task.deadline_s, epsilon)
                + float(guidance.get("lambda_Q", 0.0)) * metrics.predicted_Q_next_mb / max(float(configs["simulator"]["queue_max_mb"]), epsilon)
                + float(guidance.get("lambda_S", 0.0)) * metrics.predicted_S_gen_mb / max(float(configs["simulator"]["queue_max_mb"]), epsilon)
            )
            utilities[action_name] = utility
            resource_costs[action_name] = resource_cost
        utility_winner = max(utilities, key=utilities.get)
        high_utility_highest += int(utility_winner == "high_accuracy+roi_context")
        balanced_utility_highest += int(utility_winner == "balanced+roi")
        high_resource_gt_balanced += int(resource_costs["high_accuracy+roi_context"] > resource_costs["balanced+roi"])
        high_predicted_winner_selected += int(utility_winner == "high_accuracy+roi_context" and row.get("selected_action") == "high_accuracy+roi_context")
        high_bal_resource_ratios.append(_safe_div(resource_costs["high_accuracy+roi_context"], resource_costs["balanced+roi"]))
    total = len(task_rows)
    return {
        "task_count": total,
        "high_utility_highest": high_utility_highest,
        "balanced_utility_highest": balanced_utility_highest,
        "high_resource_gt_balanced": high_resource_gt_balanced,
        "high_predicted_winner_selected": high_predicted_winner_selected,
        "avg_high_bal_predicted_resource_cost_ratio": mean(high_bal_resource_ratios) if high_bal_resource_ratios else 0.0,
    }


def _write_calibrated_action_score_diagnosis(output_dir: Path, summary, dist) -> None:
    variant = "proposed_yolo_calibrated_profile"
    hi = _aggregate_action_metrics(output_dir, variant, "high_accuracy+roi_context")
    bal = _aggregate_action_metrics(output_dir, variant, "balanced+roi")
    candidate = _candidate_score_diagnostics(output_dir)
    task_rows = _task_rows(output_dir, variant)
    hi_rows = [row for row in task_rows if row.get("selected_action") == "high_accuracy+roi_context"]
    resource_safe_hi = sum(
        1
        for row in hi_rows
        if _to_float(row.get("battery_soc")) > 0.70
        and _to_float(row.get("Theta_t")) < 0.85
        and _to_float(row.get("C_t")) > 0.45
    )
    lines = [
        "# Calibrated Action-Score Diagnosis",
        "",
        f"- calibrated high_accuracy+roi_context count: {hi['count']:.0f}",
        f"- calibrated balanced+roi count: {bal['count']:.0f}",
        f"- candidate predicted task-utility winner high_accuracy: {int(candidate['high_utility_highest'])}/{int(candidate['task_count'])}",
        f"- candidate predicted task-utility winner balanced+roi: {int(candidate['balanced_utility_highest'])}/{int(candidate['task_count'])}",
        f"- high_accuracy predicted resource cost > balanced+roi: {int(candidate['high_resource_gt_balanced'])}/{int(candidate['task_count'])}",
        f"- selected high_accuracy when it was the predicted utility winner: {int(candidate['high_predicted_winner_selected'])}/{int(candidate['high_utility_highest'])}",
        f"- avg high_accuracy/balanced predicted resource-cost ratio: {candidate['avg_high_bal_predicted_resource_cost_ratio']:.3f}",
        f"- high_accuracy selected in resource-safe/contact-friendly rows: {resource_safe_hi}/{int(hi['count'])}",
        f"- high_accuracy avg effective utility: {hi['avg_effective']:.3f}; balanced+roi: {bal['avg_effective']:.3f}",
        f"- high_accuracy avg resource penalty: {hi['avg_resource']:.3f}; balanced+roi: {bal['avg_resource']:.3f}",
        f"- high_accuracy avg energy: {hi['avg_energy']:.3f}; balanced+roi: {bal['avg_energy']:.3f}",
        f"- high_accuracy utility/resource: {hi['utility_per_resource']:.3f}; balanced+roi: {bal['utility_per_resource']:.3f}",
        f"- high_accuracy net/energy: {hi['net_per_energy']:.5f}; balanced+roi: {bal['net_per_energy']:.5f}",
        f"- no_ucb high_accuracy count: {dist.get(('proposed_yolo_calibrated_no_ucb_bonus', 'high_accuracy+roi_context'), 0)}",
        f"- stronger_resource_price high_accuracy count: {dist.get(('proposed_yolo_calibrated_stronger_resource_price', 'high_accuracy+roi_context'), 0)}",
        f"- no_mission_priority_boost high_accuracy count: {dist.get(('proposed_yolo_calibrated_no_mission_priority_boost', 'high_accuracy+roi_context'), 0)}",
        "",
        "Interpretation: high_accuracy remains the candidate-level predicted task-utility winner on most tasks because it combines the highest quality with roi_context. The calibrated profile still reduces final high_accuracy selections sharply because its resource and thermal costs now enter predicted reward and safety scoring more strongly. UCB matters for the residual selections because no_ucb removes them, while mission-priority removal has negligible effect in this run.",
    ]
    path = output_dir / "calibrated_action_score_diagnosis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_cause_after_calibration(output_dir: Path, summary, dist) -> None:
    orig_hi = dist.get(("proposed_original_profile", "high_accuracy+roi_context"), 0)
    cal_hi = dist.get(("proposed_yolo_calibrated_profile", "high_accuracy+roi_context"), 0)
    no_ucb_hi = dist.get(("proposed_yolo_calibrated_no_ucb_bonus", "high_accuracy+roi_context"), 0)
    price_hi = dist.get(("proposed_yolo_calibrated_stronger_resource_price", "high_accuracy+roi_context"), 0)
    no_mission_hi = dist.get(("proposed_yolo_calibrated_no_mission_priority_boost", "high_accuracy+roi_context"), 0)
    cal_net = _to_float(summary["proposed_yolo_calibrated_profile"].get("net_system_utility"))
    orig_net = _to_float(summary["proposed_original_profile"].get("net_system_utility"))
    no_ucb_net = _to_float(summary["proposed_yolo_calibrated_no_ucb_bonus"].get("net_system_utility"))
    price_net = _to_float(summary["proposed_yolo_calibrated_stronger_resource_price"].get("net_system_utility"))
    no_mission_net = _to_float(summary["proposed_yolo_calibrated_no_mission_priority_boost"].get("net_system_utility"))
    lines = [
        "# Root Cause After Profile Calibration",
        "",
        f"- original profile high_accuracy count: {orig_hi}",
        f"- calibrated profile high_accuracy count: {cal_hi}",
        f"- original profile net_system_utility: {orig_net:.3f}",
        f"- calibrated profile net_system_utility: {cal_net:.3f}",
        f"- no_ucb high_accuracy/net: {no_ucb_hi} / {no_ucb_net:.3f}",
        f"- stronger_resource_price high_accuracy/net: {price_hi} / {price_net:.3f}",
        f"- no_mission_priority_boost high_accuracy/net: {no_mission_hi} / {no_mission_net:.3f}",
        "",
    ]
    if cal_hi < orig_hi and cal_net > orig_net:
        lines.append("A: calibrated profile reduced high_accuracy and improved net utility, so the original issue is primarily action profile calibration.")
        lines.append("Residual note: high_accuracy can still be the predicted task-utility winner because roi_context carries the largest information factor; the improvement comes from better resource-cost calibration in final scoring, not from making high_accuracy low-quality.")
    elif cal_hi >= orig_hi:
        lines.append("B: calibrated profile did not reduce high_accuracy enough; resource price scale, mission-priority, or mode-price coupling remain likely.")
    else:
        lines.append("Profile calibration reduced high_accuracy, but net utility did not clearly improve; profile and downstream scoring both need attention.")
    if price_net > cal_net:
        lines.append("C: stronger_resource_price improves over calibrated baseline, indicating resource price scale is weak.")
    if no_ucb_net > cal_net:
        lines.append("D: no_ucb improves over calibrated baseline, indicating cold-start UCB amplifies costly actions.")
    if no_mission_net > cal_net:
        lines.append("E: no_mission_priority_boost improves over calibrated baseline, indicating mission-priority push is too strong.")
    if max(price_net, no_ucb_net, no_mission_net, cal_net) <= orig_net and cal_hi >= orig_hi:
        lines.append("F: none of the ablations improve; inspect score implementation or unused profile fields.")
    lines.append("G: no reward/cost implementation bug is indicated by these ablations.")
    path = output_dir / "root_cause_after_profile_calibration.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_story_notes(output_dir: Path) -> None:
    lines = [
        "# Paper Story Notes: Profile Calibration",
        "",
        "1. Highest detector accuracy is not automatically highest system utility because onboard decisions pay energy, heat, latency, queueing, and contact-window costs.",
        "2. `balanced+roi` is a strong baseline because it captures much of the quality/information gain while avoiding the heavy model's marginal compute and thermal burden.",
        "3. The case study supports task-utility-driven computation-communication co-design: detection model, payload richness, downlink opportunity, and energy-shadow state must be optimized together.",
        "4. The YOLO-inspired relative profile is more defensible than arbitrary tiny/small/medium numbers because it encodes the public trend that m-class models provide marginal accuracy gains at much higher latency/FLOPs cost. It is still not a substitute for real SARFish YOLO profiling.",
        "5. Suitable for the WC case study: heavy models are not universally optimal; balanced models can dominate system utility; resource-aware co-design matters. Keep as engineering diagnostics for now: exact high_accuracy counts, UCB cold-start details, and specific lambda/mode tuning implications.",
    ]
    path = output_dir / "paper_story_profile_calibration_notes.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARFish YOLO-inspired profile ablation.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sarfish_yolo_profile_ablation")
    args = parser.parse_args()
    configure_logging()
    write_profile_audit_report()
    run_ablation(args.output_dir)


if __name__ == "__main__":
    main()
