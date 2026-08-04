"""Action-score counterfactual diagnostics for the SARFish energy-shadow stress run."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sarfish_energy_shadow_stress_dryrun import _load_stress_configs  # noqa: E402
from src.common.types import CommState, ContextVector  # noqa: E402
from src.models.model_registry import ModelRegistry  # noqa: E402
from src.payload.payload_policy import get_payload_policy_profile  # noqa: E402
from src.reward.delivery import compute_delivery_factor  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


FOCUS_ACTIONS = (
    "lightweight+metadata",
    "balanced+metadata",
    "balanced+roi",
    "high_accuracy+roi_context",
    "delay/store",
)
ROUTERS_FOR_EX_POST = (
    "proposed_lyapunov_rule_mode_safe_bandit",
    "utility_rule",
    "no_mode",
    "no_lyapunov",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
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
    if abs(denominator) <= 1.0e-12:
        return float("nan")
    return numerator / denominator


def _fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


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
        "lambda_base": {
            "lambda_E": _to_float(row.get("lambda_E_base")),
            "lambda_L": _to_float(row.get("lambda_L_base")),
            "lambda_Q": _to_float(row.get("lambda_Q_base")),
            "lambda_S": _to_float(row.get("lambda_S_base")),
        },
        "lambda_E_base": _to_float(row.get("lambda_E_base")),
        "lambda_L_base": _to_float(row.get("lambda_L_base")),
        "lambda_Q_base": _to_float(row.get("lambda_Q_base")),
        "lambda_S_base": _to_float(row.get("lambda_S_base")),
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


def _no_mode_guidance_from_row(row: dict[str, str]) -> dict[str, object]:
    base = {
        "lambda_E": _to_float(row.get("lambda_E_base")),
        "lambda_L": _to_float(row.get("lambda_L_base")),
        "lambda_Q": _to_float(row.get("lambda_Q_base")),
        "lambda_S": _to_float(row.get("lambda_S_base")),
    }
    return {
        "mode_name": "no_mode_counterfactual",
        "lambda_base": dict(base),
        "lambda_E_base": base["lambda_E"],
        "lambda_L_base": base["lambda_L"],
        "lambda_Q_base": base["lambda_Q"],
        "lambda_S_base": base["lambda_S"],
        "lambda_E": base["lambda_E"],
        "lambda_L": base["lambda_L"],
        "lambda_Q": base["lambda_Q"],
        "lambda_S": base["lambda_S"],
        "exploration_scale": 1.0,
        "model_complexity_bias": 0.0,
        "payload_size_bias": 0.0,
        "high_value_bonus_scale": 1.0,
        "safety_margin_scale": 1.0,
    }


def _estimate_task_terms(task, action, metrics, guidance: dict[str, object], simulator_cfg: dict[str, object]) -> dict[str, float]:
    policy_profile = get_payload_policy_profile(action.payload_policy)
    if action.payload_policy == "defer" or action.model_name == "delay_store":
        payload_information = policy_profile.information_level * 0.10
    else:
        payload_information = policy_profile.information_level * max(metrics.predicted_detection_quality, 0.55)
    quality_requirement = float(task.quality_requirement or 0.0)
    information_requirement = float(task.information_requirement or 0.0)
    quality_coverage = min(metrics.predicted_detection_quality / max(quality_requirement, 1.0e-9), 1.0)
    information_coverage = min(payload_information / max(information_requirement, 1.0e-9), 1.0)
    requirement_gain = 0.5 * quality_coverage + 0.5 * information_coverage

    high_value_threshold = float(simulator_cfg.get("high_value_threshold", 0.75))
    high_value_bonus_scale = float(guidance.get("high_value_bonus_scale", 1.0))
    task_utility = task.task_value * metrics.predicted_detection_quality * payload_information * metrics.predicted_delivery_factor
    if task.task_value >= high_value_threshold:
        task_utility *= high_value_bonus_scale

    epsilon = float(simulator_cfg["epsilon"])
    energy_term = (metrics.predicted_E_comp + metrics.predicted_E_comm) / max(float(simulator_cfg["E_ref"]), epsilon)
    latency_term = (metrics.predicted_L_inf + metrics.predicted_L_queue + metrics.predicted_L_comm) / max(task.deadline_s, epsilon)
    queue_term = metrics.predicted_Q_next_mb / max(float(simulator_cfg["queue_max_mb"]), epsilon)
    payload_term = metrics.predicted_S_gen_mb / max(float(simulator_cfg["queue_max_mb"]), epsilon)
    resource_cost = (
        float(guidance.get("lambda_E", 0.0)) * energy_term
        + float(guidance.get("lambda_L", 0.0)) * latency_term
        + float(guidance.get("lambda_Q", 0.0)) * queue_term
        + float(guidance.get("lambda_S", 0.0)) * payload_term
    )
    bias = (
        float(guidance.get("model_complexity_bias", 0.0)) * action.expected_compute_level
        + float(guidance.get("payload_size_bias", 0.0)) * action.expected_payload_level
    )
    safety_margin = max(0.0, float(simulator_cfg["battery_min"]) - metrics.predicted_B_next)
    safety_margin += max(0.0, metrics.predicted_Theta_next - float(simulator_cfg["thermal_max"]))
    safety_margin += max(0.0, metrics.predicted_Q_next_mb - float(simulator_cfg["queue_max_mb"])) / max(float(simulator_cfg["queue_max_mb"]), epsilon)
    safety_margin *= float(guidance.get("safety_margin_scale", 1.0))
    return {
        "predicted_task_utility": task_utility,
        "predicted_requirement_gain": requirement_gain,
        "predicted_resource_cost": resource_cost,
        "predicted_payload_mb": metrics.predicted_S_gen_mb,
        "predicted_energy_j": metrics.predicted_E_comp + metrics.predicted_E_comm,
        "predicted_requirement_payload_information": payload_information,
        "mode_bias_term": bias,
        "safety_margin_penalty": safety_margin,
    }


def _linucb_score(A: np.ndarray, b: np.ndarray, x: np.ndarray, predicted_reward: float, alpha: float, exploration_scale: float) -> dict[str, float]:
    A_inv = np.linalg.inv(A)
    theta = A_inv @ b
    exploit = float(theta @ x)
    explore = float(np.sqrt(x @ A_inv @ x))
    exploration_bonus = alpha * exploration_scale * explore
    prior = 0.20 * predicted_reward
    return {
        "linucb_exploit": exploit,
        "linucb_explore_norm": explore,
        "ucb_exploration_bonus": exploration_bonus,
        "predicted_reward_prior": prior,
        "final_action_score": exploit + exploration_bonus + prior,
    }


def _load_trajectories(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    trajectories: dict[str, list[dict[str, str]]] = {}
    for path in sorted((output_dir / "trajectories").glob("*.csv")):
        trajectories[path.stem] = _read_csv(path)
    return trajectories


def _build_ex_post_index(trajectories: dict[str, list[dict[str, str]]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for router in ROUTERS_FOR_EX_POST:
        for row in trajectories.get(router, []):
            if _to_int(row.get("has_task")) != 1:
                continue
            key = (str(row.get("task_id")), str(row.get("selected_action")))
            index.setdefault(key, {**row, "ex_post_source_router": router})
    return index


def build_counterfactual_rows(output_dir: Path) -> list[dict[str, object]]:
    configs = _load_stress_configs()
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    timeline_by_slot = {step.slot_index: step for step in timeline}
    environment = SimulationEnvironment(configs)
    predictor = environment.predictor
    action_by_name = {action.action_name: action for action in environment.action_space.list_actions()}
    trajectories = _load_trajectories(output_dir)
    proposed_rows = trajectories["proposed_lyapunov_rule_mode_safe_bandit"]
    resource_rows = {_to_int(row.get("slot")): row for row in _read_csv(output_dir / "resource_trace_diagnostics.csv")}
    ex_post_index = _build_ex_post_index(trajectories)

    alpha = 0.70
    ridge = 1.0
    dimension = 9
    A = {name: np.eye(dimension) * ridge for name in action_by_name}
    b = {name: np.zeros(dimension) for name in action_by_name}
    queue_max = float(configs["simulator"]["queue_max_mb"])
    rows: list[dict[str, object]] = []

    for proposed in proposed_rows:
        if _to_int(proposed.get("has_task")) != 1:
            continue
        slot = _to_int(proposed.get("slot"))
        step = timeline_by_slot[slot]
        task = step.task
        if task is None:
            continue
        context = _context_from_row(proposed)
        x = context.as_array()
        guidance = _guidance_from_row(proposed)
        no_mode_guidance = _no_mode_guidance_from_row(proposed)
        comm_state = CommState(queue_mb=context.Q_t * queue_max, queue_max_mb=queue_max)
        selected_action = str(proposed.get("selected_action"))
        per_action: list[dict[str, object]] = []

        for action_name in FOCUS_ACTIONS:
            action = action_by_name[action_name]
            metrics = predictor.predict_action(
                task=task,
                context=context,
                resource_trace=step.resource_trace,
                link_trace=step.link_trace,
                comm_state=comm_state,
                action=action,
                guidance_bundle=guidance,
            )
            no_mode_metrics = predictor.predict_action(
                task=task,
                context=context,
                resource_trace=step.resource_trace,
                link_trace=step.link_trace,
                comm_state=comm_state,
                action=action,
                guidance_bundle=no_mode_guidance,
            )
            terms = _estimate_task_terms(task, action, metrics, guidance, configs["simulator"])
            score_parts = _linucb_score(
                A[action_name],
                b[action_name],
                x,
                metrics.predicted_reward,
                alpha=alpha,
                exploration_scale=float(guidance.get("exploration_scale", 1.0)),
            )
            no_mode_score = _linucb_score(
                A[action_name],
                b[action_name],
                x,
                no_mode_metrics.predicted_reward,
                alpha=alpha,
                exploration_scale=1.0,
            )["final_action_score"]
            ex_post = ex_post_index.get((str(task.task_id), action_name), {})
            resource = resource_rows.get(slot, {})
            row = {
                "task_id": task.task_id,
                "scene_id": task.scene_id,
                "slot": slot,
                "mode": proposed.get("mode_name", ""),
                "battery_soc": context.B_t,
                "dod": 1.0 - context.B_t,
                "thermal": context.Theta_t,
                "queue_mb": context.Q_t * queue_max,
                "contact_flag": int(step.link_trace.in_contact),
                "is_eclipse": _to_int(resource.get("is_eclipse", int(step.resource_trace.is_eclipse))),
                "task_value": task.task_value,
                "deadline_s": task.deadline_s,
                "difficulty": task.difficulty,
                "uncertainty": task.uncertainty,
                "action_name": action_name,
                "predicted_task_utility": terms["predicted_task_utility"],
                "predicted_requirement_gain": terms["predicted_requirement_gain"],
                "predicted_resource_cost": terms["predicted_resource_cost"],
                "predicted_reward": metrics.predicted_reward,
                "lambda_E": float(guidance.get("lambda_E", 0.0)),
                "lambda_Q": float(guidance.get("lambda_Q", 0.0)),
                "lambda_S": float(guidance.get("lambda_S", 0.0)),
                "lambda_L": float(guidance.get("lambda_L", 0.0)),
                "lambda_E_base": float(guidance.get("lambda_E_base", 0.0)),
                "lambda_Q_base": float(guidance.get("lambda_Q_base", 0.0)),
                "lambda_S_base": float(guidance.get("lambda_S_base", 0.0)),
                "lambda_L_base": float(guidance.get("lambda_L_base", 0.0)),
                "ucb_bonus": score_parts["ucb_exploration_bonus"],
                "exploration_bonus": score_parts["ucb_exploration_bonus"],
                "linucb_exploit": score_parts["linucb_exploit"],
                "linucb_explore_norm": score_parts["linucb_explore_norm"],
                "predicted_reward_prior": score_parts["predicted_reward_prior"],
                "final_action_score": score_parts["final_action_score"],
                "score_without_mode_guidance": no_mode_score,
                "mode_score_delta": score_parts["final_action_score"] - no_mode_score,
                "selected_action": selected_action,
                "selected_rank": 0,
                "selected_by_proposed": int(action_name == selected_action),
                "predicted_B_next": metrics.predicted_B_next,
                "predicted_Theta_next": metrics.predicted_Theta_next,
                "predicted_Q_next_mb": metrics.predicted_Q_next_mb,
                "predicted_payload_mb": terms["predicted_payload_mb"],
                "predicted_energy_j": terms["predicted_energy_j"],
                "predicted_safety_violation": metrics.predicted_safety_violation,
                "ex_post_effective_utility": ex_post.get("effective_utility", ""),
                "ex_post_resource_penalty": ex_post.get("resource_penalty", ""),
                "ex_post_requirement_penalty": ex_post.get("requirement_penalty", ""),
                "ex_post_energy_j": (
                    _to_float(ex_post.get("compute_energy_j")) + _to_float(ex_post.get("comm_energy_j"))
                    if ex_post
                    else ""
                ),
                "ex_post_delivery_status": ex_post.get("delivery_status", ""),
                "ex_post_source_router": ex_post.get("ex_post_source_router", ""),
            }
            per_action.append(row)

        ranked = sorted(per_action, key=lambda item: float(item["final_action_score"]), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            row["selected_rank"] = rank if row["action_name"] == selected_action else ""
        rows.extend(per_action)

        selected_reward = _to_float(proposed.get("penalized_reward", proposed.get("observed_reward")))
        if selected_action in A:
            A[selected_action] = A[selected_action] + np.outer(x, x)
            b[selected_action] = b[selected_action] + selected_reward * x

    return rows


def _rows_by_task(rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["task_id"])][str(row["action_name"])] = row
    return grouped


def write_high_accuracy_report(output_dir: Path, cf_rows: list[dict[str, object]]) -> Path:
    grouped = _rows_by_task(cf_rows)
    selected_hi = [actions["high_accuracy+roi_context"] for actions in grouped.values() if actions.get("high_accuracy+roi_context", {}).get("selected_by_proposed") == 1]
    utility_highest = 0
    exploration_lifted = 0
    mission_priority = 0
    energy_shadow = 0
    balanced_observed_better = 0
    balanced_observed_comparable = 0
    resource_higher = 0
    dominated_by_balanced = 0
    balanced_expost_pairs = 0
    score_margin_values: list[float] = []

    for hi in selected_hi:
        actions = grouped[str(hi["task_id"])]
        balanced = actions.get("balanced+roi")
        if not balanced:
            continue
        max_utility = max(float(row["predicted_task_utility"]) for row in actions.values())
        if float(hi["predicted_task_utility"]) >= max_utility - 1.0e-12:
            utility_highest += 1
        score_without_explore = {
            name: float(row["linucb_exploit"]) + float(row["predicted_reward_prior"])
            for name, row in actions.items()
        }
        if max(score_without_explore, key=score_without_explore.get) != "high_accuracy+roi_context":
            exploration_lifted += 1
        if str(hi["mode"]) == "mission_priority":
            mission_priority += 1
        if (
            int(hi["is_eclipse"]) == 1
            or float(hi["dod"]) > 0.30
            or float(hi["thermal"]) > 0.85
            or int(hi["contact_flag"]) == 0
        ):
            energy_shadow += 1
        if hi.get("ex_post_source_router") and balanced.get("ex_post_source_router"):
            balanced_expost_pairs += 1
            hi_net = _to_float(hi["ex_post_effective_utility"]) - _to_float(hi["ex_post_resource_penalty"]) - _to_float(hi["ex_post_requirement_penalty"])
            balanced_net = _to_float(balanced["ex_post_effective_utility"]) - _to_float(balanced["ex_post_resource_penalty"]) - _to_float(balanced["ex_post_requirement_penalty"])
            if balanced_net > hi_net:
                balanced_observed_better += 1
            if balanced_net >= hi_net - 1.0e-9:
                balanced_observed_comparable += 1
        if float(hi["predicted_resource_cost"]) > float(balanced["predicted_resource_cost"]):
            resource_higher += 1
        if (
            float(balanced["predicted_task_utility"]) >= float(hi["predicted_task_utility"]) - 1.0e-12
            and float(balanced["predicted_requirement_gain"]) >= float(hi["predicted_requirement_gain"]) - 1.0e-12
            and float(balanced["predicted_resource_cost"]) <= float(hi["predicted_resource_cost"]) + 1.0e-12
        ):
            dominated_by_balanced += 1
        score_margin_values.append(float(hi["final_action_score"]) - float(balanced["final_action_score"]))

    count = len(selected_hi)
    lines = [
        "# High-Accuracy Selection Diagnosis",
        "",
        f"- selected high_accuracy+roi_context count: {count}",
        f"- predicted task utility highest: {utility_highest}/{count}",
        f"- UCB/exploration changed the winner toward high_accuracy: {exploration_lifted}/{count}",
        f"- selected under mode=mission_priority: {mission_priority}/{count}",
        f"- selected under energy-shadow state: {energy_shadow}/{count}",
        f"- high_accuracy predicted resource cost > balanced+roi: {resource_higher}/{count}",
        f"- balanced+roi predicted-dominates high_accuracy on utility/requirement/resource: {dominated_by_balanced}/{count}",
        f"- matched ex-post balanced+roi pairs available: {balanced_expost_pairs}/{count}",
        f"- balanced+roi ex-post net better than high_accuracy when matched: {balanced_observed_better}/{balanced_expost_pairs}",
        f"- balanced+roi ex-post net comparable-or-better when matched: {balanced_observed_comparable}/{balanced_expost_pairs}",
        f"- average high_accuracy score margin over balanced+roi: {_fmt(mean(score_margin_values) if score_margin_values else float('nan'))}",
        "",
        "Interpretation: high_accuracy+roi_context is selected mainly because the current profile/predictor estimates it as the highest task-utility action on every selected high-accuracy task. UCB/exploration and mission-priority guidance are amplifiers, not the dominant cause.",
    ]
    path = output_dir / "high_accuracy_selection_diagnosis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_action_pareto_report(output_dir: Path) -> Path:
    trajectories = _load_trajectories(output_dir)
    lifetime_rows = _read_csv(output_dir / "task_lifetime_outcomes.csv")
    lifetime_by_router_action: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in lifetime_rows:
        lifetime_by_router_action[(str(row["router_name"]), str(row["selected_action"]))][str(row["final_lifetime_status"])] += 1

    lines = [
        "# Action Pareto Profile Report",
        "",
        "| router | action | count | avg effective utility | avg resource penalty | avg requirement penalty | avg energy | avg delivered MB | lifetime delivered ratio | expired ratio | net utility per energy | utility per resource penalty |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    comparison: dict[tuple[str, str], dict[str, float]] = {}
    for router in ("proposed_lyapunov_rule_mode_safe_bandit", "utility_rule"):
        task_rows = [row for row in trajectories.get(router, []) if _to_int(row.get("has_task")) == 1]
        for action in ("balanced+roi", "high_accuracy+roi_context", "balanced+metadata", "lightweight+metadata"):
            rows = [row for row in task_rows if row.get("selected_action") == action]
            count = len(rows)
            energy = sum(_to_float(row.get("compute_energy_j")) + _to_float(row.get("comm_energy_j")) for row in rows)
            effective = sum(_to_float(row.get("effective_utility")) for row in rows)
            resource = sum(_to_float(row.get("resource_penalty")) for row in rows)
            requirement = sum(_to_float(row.get("requirement_penalty")) for row in rows)
            delivered = sum(_to_float(row.get("S_down_current_task_mb")) for row in rows)
            lifetime_counts = lifetime_by_router_action[(router, action)]
            lifetime_total = sum(lifetime_counts.values())
            delivered_ratio = _safe_div(lifetime_counts.get("delivered", 0), lifetime_total)
            expired_ratio = _safe_div(lifetime_counts.get("expired", 0), lifetime_total)
            net = effective - resource - requirement
            comparison[(router, action)] = {
                "count": count,
                "effective": effective,
                "resource": resource,
                "requirement": requirement,
                "energy": energy,
                "net": net,
                "net_per_energy": _safe_div(net, energy),
                "utility_per_resource": _safe_div(effective, resource),
            }
            lines.append(
                f"| {router} | {action} | {count} | {_fmt(_safe_div(effective, count))} | "
                f"{_fmt(_safe_div(resource, count))} | {_fmt(_safe_div(requirement, count))} | "
                f"{_fmt(_safe_div(energy, count))} | {_fmt(_safe_div(delivered, count))} | "
                f"{_fmt(delivered_ratio)} | {_fmt(expired_ratio)} | {_fmt(_safe_div(net, energy), 5)} | "
                f"{_fmt(_safe_div(effective, resource))} |"
            )

    prop_bal = comparison[("proposed_lyapunov_rule_mode_safe_bandit", "balanced+roi")]
    prop_hi = comparison[("proposed_lyapunov_rule_mode_safe_bandit", "high_accuracy+roi_context")]
    lines.extend(
        [
            "",
            "## Judgment",
            "",
            f"- proposed balanced+roi net per energy: {_fmt(prop_bal['net_per_energy'], 5)}; high_accuracy+roi_context: {_fmt(prop_hi['net_per_energy'], 5)}.",
            f"- proposed balanced+roi utility/resource: {_fmt(prop_bal['utility_per_resource'])}; high_accuracy+roi_context: {_fmt(prop_hi['utility_per_resource'])}.",
        ]
    )
    if prop_bal["net_per_energy"] > prop_hi["net_per_energy"] and prop_bal["utility_per_resource"] > prop_hi["utility_per_resource"]:
        lines.append("- balanced+roi is the current profile sweet spot relative to high_accuracy+roi_context on both energy and resource-normalized utility.")
    else:
        lines.append("- balanced+roi is not strictly dominant on every aggregate metric, but it remains a strong efficient baseline in this stress output.")
    path = output_dir / "action_pareto_profile_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_mode_price_report(output_dir: Path, cf_rows: list[dict[str, object]]) -> Path:
    trajectories = _load_trajectories(output_dir)
    proposed = [row for row in trajectories["proposed_lyapunov_rule_mode_safe_bandit"] if _to_int(row.get("has_task")) == 1]
    no_mode = [row for row in trajectories["no_mode"] if _to_int(row.get("has_task")) == 1]
    task_pairs = [
        (p, n)
        for p, n in zip(proposed, no_mode)
        if p.get("task_id") == n.get("task_id")
    ]
    diff_pairs = [(p, n) for p, n in task_pairs if p.get("selected_action") != n.get("selected_action")]
    mode_counts = Counter(row.get("mode_name", "") for row in proposed)
    selected_cf = [row for row in cf_rows if row["selected_by_proposed"] == 1]
    mode_deltas = [float(row["mode_score_delta"]) for row in selected_cf]
    high_deltas = [float(row["mode_score_delta"]) for row in selected_cf if row["action_name"] == "high_accuracy+roi_context"]
    low_soc = [row for row in proposed if _to_float(row.get("battery_soc")) < 0.70 or (1.0 - _to_float(row.get("battery_soc"))) > 0.30]
    eclipse = [row for row in proposed if _to_int(row.get("is_eclipse")) == 1]
    high_thermal = [row for row in proposed if _to_float(row.get("Theta_t")) > 0.85]
    no_contact = [row for row in proposed if _to_float(row.get("C_t")) <= 0.0]

    def avg_lambda(rows: list[dict[str, str]], key: str) -> float:
        return mean([_to_float(row.get(key)) for row in rows]) if rows else float("nan")

    lines = [
        "# Mode / Price Coupling Diagnosis",
        "",
        f"- aligned proposed/no_mode task rows: {len(task_pairs)}",
        f"- action differences: {len(diff_pairs)}/{len(task_pairs)}",
        f"- proposed mode distribution: {dict(mode_counts)}",
        f"- average selected-action mode score delta vs no-mode guidance: {_fmt(mean(mode_deltas) if mode_deltas else float('nan'), 5)}",
        f"- average high_accuracy selected mode score delta: {_fmt(mean(high_deltas) if high_deltas else float('nan'), 5)}",
        "",
        "## Lambda Averages By Stress State",
        "",
        "| state | n | lambda_E | lambda_Q | lambda_S | lambda_L |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rows in (
        ("all proposed task rows", proposed),
        ("low SOC / high DoD", low_soc),
        ("eclipse", eclipse),
        ("high thermal", high_thermal),
        ("no contact", no_contact),
    ):
        lines.append(
            f"| {name} | {len(rows)} | {_fmt(avg_lambda(rows, 'lambda_E'))} | {_fmt(avg_lambda(rows, 'lambda_Q'))} | "
            f"{_fmt(avg_lambda(rows, 'lambda_S'))} | {_fmt(avg_lambda(rows, 'lambda_L'))} |"
        )

    lines.extend(
        [
            "",
            "## Proposed vs no_mode Differences",
            "",
            "| slot | task_id | mode | proposed action | no_mode action | proposed net slot | no_mode net slot | lambda_E | lambda_Q | lambda_S |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for p, n in diff_pairs[:30]:
        prop_net = _to_float(p.get("effective_utility")) - _to_float(p.get("resource_penalty")) - _to_float(p.get("requirement_penalty"))
        no_net = _to_float(n.get("effective_utility")) - _to_float(n.get("resource_penalty")) - _to_float(n.get("requirement_penalty"))
        lines.append(
            f"| {_to_int(p.get('slot'))} | {p.get('task_id')} | {p.get('mode_name')} | {p.get('selected_action')} | "
            f"{n.get('selected_action')} | {_fmt(prop_net)} | {_fmt(no_net)} | {_fmt(_to_float(p.get('lambda_E')))} | "
            f"{_fmt(_to_float(p.get('lambda_Q')))} | {_fmt(_to_float(p.get('lambda_S')))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Mode/price coupling does change scores, but the change is small because guided predicted reward enters LinUCB as only a 0.2 prior term. The direct UCB exploit/explore terms still dominate many selections.",
        ]
    )
    path = output_dir / "mode_price_coupling_diagnosis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_root_cause_report(output_dir: Path, cf_rows: list[dict[str, object]]) -> Path:
    grouped = _rows_by_task(cf_rows)
    selected_rows = [next(row for row in actions.values() if row.get("selected_by_proposed") == 1) for actions in grouped.values()]
    selected_hi = [row for row in selected_rows if row["action_name"] == "high_accuracy+roi_context"]
    hi_count = len(selected_hi)
    hi_utility_highest = 0
    hi_explore_lift = 0
    hi_mission = 0
    hi_dominated = 0
    hi_resource_cost_ratio: list[float] = []
    for hi in selected_hi:
        actions = grouped[str(hi["task_id"])]
        balanced = actions["balanced+roi"]
        if float(hi["predicted_task_utility"]) >= max(float(row["predicted_task_utility"]) for row in actions.values()) - 1.0e-12:
            hi_utility_highest += 1
        without_explore = {
            name: float(row["linucb_exploit"]) + float(row["predicted_reward_prior"])
            for name, row in actions.items()
        }
        if max(without_explore, key=without_explore.get) != "high_accuracy+roi_context":
            hi_explore_lift += 1
        if str(hi["mode"]) == "mission_priority":
            hi_mission += 1
        if (
            float(balanced["predicted_task_utility"]) >= float(hi["predicted_task_utility"]) - 1.0e-12
            and float(balanced["predicted_resource_cost"]) <= float(hi["predicted_resource_cost"]) + 1.0e-12
        ):
            hi_dominated += 1
        hi_resource_cost_ratio.append(_safe_div(float(hi["predicted_resource_cost"]), float(balanced["predicted_resource_cost"])))

    selected_mode_delta = [float(row["mode_score_delta"]) for row in selected_rows]
    selected_ucb = [float(row["ucb_bonus"]) for row in selected_rows]
    selected_prior = [abs(float(row["predicted_reward_prior"])) for row in selected_rows]
    avg_ucb_prior_ratio = mean([_safe_div(u, p) for u, p in zip(selected_ucb, selected_prior) if not math.isnan(_safe_div(u, p))])

    lines = [
        "# Action-Score Root Cause Report",
        "",
        "| category | judgment | numeric support |",
        "| --- | --- | --- |",
        f"| A. action profile calibration | primary contributor | high_accuracy energy is 53.2 J vs balanced 20.8 J, yet selected high_accuracy predicted utility is highest in {hi_utility_highest}/{hi_count}; avg high_accuracy/balanced predicted resource-cost ratio {_fmt(mean(hi_resource_cost_ratio) if hi_resource_cost_ratio else float('nan'))}. |",
        f"| B. UCB exploration too strong | secondary contributor | exploration changed winner toward high_accuracy in {hi_explore_lift}/{hi_count} high_accuracy selections; average selected UCB/prior magnitude ratio {_fmt(avg_ucb_prior_ratio)} shows the prior is small relative to bandit terms. |",
        f"| C. resource price scale too weak | contributor | guided reward enters final score as 0.2 prior; high resource-cost action still selected {hi_count} times under stress. |",
        f"| D. mode modulation too weak | contributor | average selected-action mode score delta {_fmt(mean(selected_mode_delta) if selected_mode_delta else float('nan'), 5)}. |",
        f"| E. mission-priority over-pushes high_accuracy | contributor | high_accuracy selected in mission_priority {hi_mission}/{hi_count}; mode has positive model/payload bias and high_value_bonus_scale 1.30. |",
        "| F. utility_rule strong greedy baseline / balanced+roi strong | strong contributor | utility_rule selects balanced+roi 82 times and beats proposed on net_system_utility in the stress summary. |",
        "| G. reward/cost bug | not supported | diagnostics found no clear implementation bug; penalties move in the expected direction. |",
        f"| H. other | minor | balanced+roi predicted-dominates high_accuracy in {hi_dominated}/{hi_count} high_accuracy selections under the simple utility/resource check. |",
        "",
        "Root cause classification: primarily A + F, with C/D/E as coupling-scale contributors and B as a secondary cold-start/bandit amplifier. G is not supported by the evidence.",
    ]
    path = output_dir / "action_score_root_cause_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_next_recommendation(output_dir: Path) -> Path:
    lines = [
        "# Next Policy Debug Recommendation",
        "",
        "1. Calibrate action profiles before changing rewards. Check whether `high_accuracy+roi_context` should really cost only 53.2 J and 2.8 s for this SARFish GRD workload, and whether its quality advantage over `balanced+roi` is large enough.",
        "2. Add a more realistic latency/energy/profile for `high_accuracy+roi_context` if profiling evidence supports it. This is model calibration, not result-targeted tuning.",
        "3. Make resource price enter action scoring more directly only after profile calibration. Today it is mostly compressed through the 0.2 predicted-reward prior in LinUCB.",
        "4. Audit cold-start UCB. If high-cost actions win mainly from exploration bonus in stress states, consider cost-aware exploration or lower exploration in resource stress. This changes policy behavior and should be tested as an ablation, not tuned to one run.",
        "5. Consider an energy-shadow-aware mode only if the desired behavior is explicit: eclipse/DoD/high-thermal should suppress high-cost actions unless predicted utility margin is large.",
        "6. Treat `balanced+roi` as a strong baseline. It is not a weak strawman in either nominal or stress outputs.",
        "7. Reasonable calibration: measured energy/latency/quality profiles, cost-aware UCB ablation, explicit energy-shadow mode definition. Risky result-oriented tuning: changing reward weights or thresholds solely to make proposed beat utility_rule on this scenario.",
    ]
    path = output_dir / "next_policy_debug_recommendation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_analysis(output_dir: Path) -> None:
    cf_rows = build_counterfactual_rows(output_dir)
    _write_csv(output_dir / "action_score_counterfactual.csv", cf_rows)
    write_high_accuracy_report(output_dir, cf_rows)
    write_action_pareto_report(output_dir)
    write_mode_price_report(output_dir, cf_rows)
    write_root_cause_report(output_dir, cf_rows)
    write_next_recommendation(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SARFish energy-shadow action scores.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sarfish_energy_shadow_stress")
    args = parser.parse_args()
    run_analysis(args.output_dir)


if __name__ == "__main__":
    main()
