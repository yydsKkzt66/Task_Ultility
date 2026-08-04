"""Run SARFish policy-repair ablations with calibrated YOLO-inspired profiles."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.routers.fixed_router import FixedRouter  # noqa: E402
from src.routers.rule_router import RuleBasedRouter  # noqa: E402
from src.routers.safe_linucb_router import SafeLinUCBRouter  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


VARIANTS = (
    "proposed_yolo_calibrated_baseline",
    "proposed_cost_aware_ucb",
    "proposed_delivery_aware_info_gain",
    "proposed_z_guided_coupling",
    "proposed_ucb_plus_delivery",
    "proposed_repaired_combined",
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


def _load_repair_cfg(variant: str) -> dict[str, object]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "policy_repair_ablation.yaml")
    repair = copy.deepcopy(cfg["policy_repair_defaults"])
    repair.update(copy.deepcopy(cfg.get("variant_overrides", {}).get(variant, {})))
    return repair


def _load_configs(variant: str) -> dict[str, dict]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_policy_repair_ablation_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    profile_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml")
    configs["model_pool"] = copy.deepcopy(configs["model_pool"])
    configs["model_pool"]["profile_overrides"] = dict(profile_cfg["profile_overrides"])
    configs["simulator"]["policy_repair"] = _load_repair_cfg(variant)
    if variant not in {
        "proposed_cost_aware_ucb",
        "proposed_delivery_aware_info_gain",
        "proposed_z_guided_coupling",
        "proposed_ucb_plus_delivery",
        "proposed_repaired_combined",
    }:
        configs["simulator"]["policy_repair"] = _load_repair_cfg("proposed_yolo_calibrated_baseline")
    return configs


def _build_router(variant: str, environment: SimulationEnvironment):
    action_names = [action.action_name for action in environment.action_space.list_actions()]
    if variant.startswith("proposed_"):
        return SafeLinUCBRouter(
            action_names,
            alpha=0.70,
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


def run_ablation(output_dir: Path) -> None:
    output_dirs = ensure_output_dirs(output_dir)
    logger = get_logger("run_sarfish_policy_repair_ablation")
    results = []
    for variant in VARIANTS:
        configs = _load_configs(variant)
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        environment = SimulationEnvironment(configs)
        router = _build_router(variant, environment)
        episode_runner = environment.build_episode_runner(timeline)
        logger.info("running variant: %s", variant)
        results.append(episode_runner.run(router))
    write_episode_results(results, output_dirs)
    generate_lifetime_outputs(output_dir, 20.0)
    write_action_distribution(output_dir)
    write_score_component_diagnostics(output_dir)
    write_reports(output_dir)


def _trajectory_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return _read_csv(output_dir / "trajectories" / f"{variant}.csv")


def _task_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return [row for row in _trajectory_rows(output_dir, variant) if _to_int(row.get("has_task")) == 1]


def write_action_distribution(output_dir: Path) -> Path:
    rows = []
    for variant in VARIANTS:
        task_rows = _task_rows(output_dir, variant)
        action_counts = Counter(row.get("selected_action", "") for row in task_rows)
        payload_counts = Counter(row.get("payload_policy", "") for row in task_rows)
        for action in FOCUS_ACTIONS:
            rows.append({"variant": variant, "type": "action", "name": action, "count": action_counts.get(action, 0), "total_tasks": len(task_rows)})
        for payload in ("metadata", "roi", "roi_context", "defer"):
            rows.append({"variant": variant, "type": "payload", "name": payload, "count": payload_counts.get(payload, 0), "total_tasks": len(task_rows)})
    path = output_dir / "action_distribution.csv"
    _write_csv(path, rows)
    return path


def write_score_component_diagnostics(output_dir: Path) -> Path:
    fields = [
        "router_name",
        "slot",
        "task_id",
        "selected_action",
        "mode_name",
        "battery_soc",
        "Theta_t",
        "Q_t",
        "is_eclipse",
        "raw_ucb_bonus",
        "risk_discount",
        "cost_aware_ucb_bonus",
        "ucb_suppression_reason",
        "raw_info_gain",
        "delivery_success_proxy",
        "effective_info_gain",
        "payload_delivery_risk_reason",
        "lambda_E_before_z",
        "lambda_E_after_z",
        "lambda_Q_before_z",
        "lambda_Q_after_z",
        "lambda_S_before_z",
        "lambda_S_after_z",
        "beta_before_z",
        "beta_after_z",
        "z_guidance_reason",
        "selected_action_score",
    ]
    rows = []
    for variant in VARIANTS:
        for row in _task_rows(output_dir, variant):
            rows.append({field: row.get(field, "") for field in fields})
    path = output_dir / "score_component_diagnostics.csv"
    _write_csv(path, rows)
    return path


def _lifetime_counts(output_dir: Path) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in _read_csv(output_dir / "task_lifetime_outcomes.csv"):
        counts[str(row["router_name"])][str(row["final_lifetime_status"])] += 1
    return counts


def _dist(output_dir: Path) -> dict[tuple[str, str, str], int]:
    return {
        (row["variant"], row["type"], row["name"]): _to_int(row["count"])
        for row in _read_csv(output_dir / "action_distribution.csv")
    }


def _max_dod(rows: list[dict[str, str]]) -> float:
    return max((1.0 - _to_float(row.get("battery_next_soc", row.get("battery_soc"))) for row in rows), default=0.0)


def _similarity(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]) -> tuple[int, int, float]:
    pairs = [(a, b) for a, b in zip(rows_a, rows_b) if a.get("task_id") == b.get("task_id")]
    same = sum(1 for a, b in pairs if a.get("selected_action") == b.get("selected_action"))
    return same, len(pairs), _safe_div(same, len(pairs))


def _summary_table(output_dir: Path, summary, lifetime, dist) -> list[str]:
    lines = [
        "| variant | gross | net | penalized | resource | requirement | total energy | compute | comm | max thermal | max DoD | delivered_mb | expired | lifetime delivered / expired / deferred | high_acc | balanced+roi | metadata / roi / roi_context |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for variant in VARIANTS:
        row = summary.get(variant, {})
        counts = lifetime.get(variant, Counter())
        traj = _trajectory_rows(output_dir, variant)
        lines.append(
            f"| {variant} | {_to_float(row.get('gross_task_utility')):.3f} | {_to_float(row.get('net_system_utility')):.3f} | "
            f"{_to_float(row.get('total_penalized_reward')):.3f} | {_to_float(row.get('total_resource_penalty')):.3f} | "
            f"{_to_float(row.get('total_requirement_penalty')):.3f} | {_to_float(row.get('total_energy')):.1f} | "
            f"{_to_float(row.get('total_compute_energy')):.1f} | {_to_float(row.get('total_comm_energy')):.1f} | "
            f"{_to_float(row.get('max_thermal')):.4f} | {_max_dod(traj):.4f} | {_to_float(row.get('delivered_mb')):.3f} | "
            f"{_to_int(row.get('expired_items'))} | {counts.get('delivered', 0)} / {counts.get('expired', 0)} / {counts.get('deferred', 0)} | "
            f"{dist.get((variant, 'action', 'high_accuracy+roi_context'), 0)} | {dist.get((variant, 'action', 'balanced+roi'), 0)} | "
            f"{dist.get((variant, 'payload', 'metadata'), 0)} / {dist.get((variant, 'payload', 'roi'), 0)} / {dist.get((variant, 'payload', 'roi_context'), 0)} |"
        )
    return lines


def write_reports(output_dir: Path) -> None:
    summary = {row["router"]: row for row in _read_csv(output_dir / "summary.csv")}
    lifetime = _lifetime_counts(output_dir)
    dist = _dist(output_dir)
    lines = ["# Policy Repair Ablation Report", "", *_summary_table(output_dir, summary, lifetime, dist)]
    (output_dir / "policy_repair_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_no_mode_report(output_dir, summary)
    _write_component_reports(output_dir, summary, dist)
    _write_counterfactual_report(output_dir, summary, dist)
    _write_root_cause_report(output_dir, summary, dist)
    _write_paper_notes(output_dir)


def _write_no_mode_report(output_dir: Path, summary) -> None:
    lines = ["# Proposed vs no_mode Separation Report", ""]
    no_mode_rows = _task_rows(output_dir, "no_mode")
    for variant in [item for item in VARIANTS if item.startswith("proposed_")]:
        same, total, ratio = _similarity(_task_rows(output_dir, variant), no_mode_rows)
        lines.append(
            f"- {variant}: same actions {same}/{total} ({ratio:.3f}); net {float(summary[variant]['net_system_utility']):.3f} vs no_mode {float(summary['no_mode']['net_system_utility']):.3f}."
        )
    (output_dir / "proposed_vs_no_mode_separation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _avg(rows: list[dict[str, str]], field: str) -> float:
    return _safe_div(sum(_to_float(row.get(field)) for row in rows), len(rows))


def _write_component_reports(output_dir: Path, summary, dist) -> None:
    diagnostics = _read_csv(output_dir / "score_component_diagnostics.csv")
    combined = [row for row in diagnostics if row.get("router_name") == "proposed_repaired_combined"]
    cost = [row for row in diagnostics if row.get("router_name") == "proposed_cost_aware_ucb"]
    delivery = [row for row in diagnostics if row.get("router_name") == "proposed_delivery_aware_info_gain"]
    z_rows = [row for row in diagnostics if row.get("router_name") == "proposed_z_guided_coupling"]
    (output_dir / "high_accuracy_suppression_report.md").write_text(
        "\n".join(
            [
                "# High Accuracy Suppression Report",
                "",
                f"- baseline high_accuracy count: {dist.get(('proposed_yolo_calibrated_baseline', 'action', 'high_accuracy+roi_context'), 0)}",
                f"- cost-aware UCB high_accuracy count: {dist.get(('proposed_cost_aware_ucb', 'action', 'high_accuracy+roi_context'), 0)}",
                f"- delivery-aware high_accuracy count: {dist.get(('proposed_delivery_aware_info_gain', 'action', 'high_accuracy+roi_context'), 0)}",
                f"- z-guided high_accuracy count: {dist.get(('proposed_z_guided_coupling', 'action', 'high_accuracy+roi_context'), 0)}",
                f"- combined high_accuracy count: {dist.get(('proposed_repaired_combined', 'action', 'high_accuracy+roi_context'), 0)}",
                f"- combined balanced+roi count: {dist.get(('proposed_repaired_combined', 'action', 'balanced+roi'), 0)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "cost_aware_ucb_report.md").write_text(
        "\n".join(
            [
                "# Cost-Aware UCB Report",
                "",
                f"- avg raw_ucb_bonus: {_avg(cost, 'raw_ucb_bonus'):.4f}",
                f"- avg risk_discount: {_avg(cost, 'risk_discount'):.4f}",
                f"- avg cost_aware_ucb_bonus: {_avg(cost, 'cost_aware_ucb_bonus'):.4f}",
                f"- high_accuracy count: {dist.get(('proposed_cost_aware_ucb', 'action', 'high_accuracy+roi_context'), 0)}",
                f"- net_system_utility: {float(summary['proposed_cost_aware_ucb']['net_system_utility']):.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "delivery_aware_info_gain_report.md").write_text(
        "\n".join(
            [
                "# Delivery-Aware Information Gain Report",
                "",
                f"- avg raw_info_gain: {_avg(delivery, 'raw_info_gain'):.4f}",
                f"- avg delivery_success_proxy: {_avg(delivery, 'delivery_success_proxy'):.4f}",
                f"- avg effective_info_gain: {_avg(delivery, 'effective_info_gain'):.4f}",
                f"- roi_context count: {dist.get(('proposed_delivery_aware_info_gain', 'payload', 'roi_context'), 0)}",
                f"- net_system_utility: {float(summary['proposed_delivery_aware_info_gain']['net_system_utility']):.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "z_guided_coupling_report.md").write_text(
        "\n".join(
            [
                "# z_k-Guided Coupling Report",
                "",
                f"- avg lambda_E before/after: {_avg(z_rows, 'lambda_E_before_z'):.4f} / {_avg(z_rows, 'lambda_E_after_z'):.4f}",
                f"- avg lambda_S before/after: {_avg(z_rows, 'lambda_S_before_z'):.4f} / {_avg(z_rows, 'lambda_S_after_z'):.4f}",
                f"- avg beta before/after: {_avg(z_rows, 'beta_before_z'):.4f} / {_avg(z_rows, 'beta_after_z'):.4f}",
                f"- action similarity to no_mode is reported in proposed_vs_no_mode_separation_report.md.",
                f"- net_system_utility: {float(summary['proposed_z_guided_coupling']['net_system_utility']):.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_counterfactual_report(output_dir: Path, summary, dist) -> None:
    lines = [
        "# Score Counterfactual After Policy Repair",
        "",
        "- Candidate task-utility can still favor high_accuracy because roi_context carries the largest raw information factor.",
        "- Delivery-aware variants reduce effective_info_gain through delivery_success_proxy in no-contact, queue, deadline, and large-payload conditions.",
        f"- baseline high_accuracy count: {dist.get(('proposed_yolo_calibrated_baseline', 'action', 'high_accuracy+roi_context'), 0)}",
        f"- repaired combined high_accuracy count: {dist.get(('proposed_repaired_combined', 'action', 'high_accuracy+roi_context'), 0)}",
        f"- repaired combined balanced+roi count: {dist.get(('proposed_repaired_combined', 'action', 'balanced+roi'), 0)}",
        f"- repaired combined net_system_utility: {float(summary['proposed_repaired_combined']['net_system_utility']):.3f}",
        f"- utility_rule net_system_utility: {float(summary['utility_rule']['net_system_utility']):.3f}",
        "- Over-conservatism should be read from gross utility and deferred count in policy_repair_ablation_report.md.",
    ]
    (output_dir / "score_counterfactual_after_policy_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_cause_report(output_dir: Path, summary, dist) -> None:
    baseline_net = float(summary["proposed_yolo_calibrated_baseline"]["net_system_utility"])
    combined_net = float(summary["proposed_repaired_combined"]["net_system_utility"])
    cost_net = float(summary["proposed_cost_aware_ucb"]["net_system_utility"])
    delivery_net = float(summary["proposed_delivery_aware_info_gain"]["net_system_utility"])
    z_net = float(summary["proposed_z_guided_coupling"]["net_system_utility"])
    lines = [
        "# Root Cause After Policy Repair",
        "",
        f"- baseline net: {baseline_net:.3f}",
        f"- cost-aware UCB net: {cost_net:.3f}",
        f"- delivery-aware info gain net: {delivery_net:.3f}",
        f"- z-guided coupling net: {z_net:.3f}",
        f"- combined net: {combined_net:.3f}",
        f"- baseline vs combined high_accuracy: {dist.get(('proposed_yolo_calibrated_baseline', 'action', 'high_accuracy+roi_context'), 0)} -> {dist.get(('proposed_repaired_combined', 'action', 'high_accuracy+roi_context'), 0)}",
        "",
        f"- cost-aware UCB solves no_ucb symptom: {'yes' if cost_net > baseline_net else 'not in this configuration'}.",
        f"- delivery-aware info gain improves roi_context overestimate: {'yes' if delivery_net > baseline_net or dist.get(('proposed_delivery_aware_info_gain', 'action', 'high_accuracy+roi_context'), 0) < dist.get(('proposed_yolo_calibrated_baseline', 'action', 'high_accuracy+roi_context'), 0) else 'not clearly'}.",
        f"- z-guided coupling separates proposed/no_mode: see separation report; net change {'positive' if z_net > baseline_net else 'not positive'}.",
        "- No implementation bug is indicated unless all enabled diagnostics remain identical to baseline.",
        "- Reward weights still should not be tuned; remaining work belongs in action scoring/profile/resource-price coupling.",
    ]
    (output_dir / "root_cause_after_policy_repair.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_notes(output_dir: Path) -> None:
    lines = [
        "# Paper Story Notes: Policy Repair",
        "",
        "1. Highest precision is not sufficient for onboard value because heavy models consume scarce energy, heat budget, latency, and downlink opportunity.",
        "2. Delivery-aware information gain is central to computation-communication co-design: information that cannot be downlinked before deadline should not be scored as fully useful.",
        "3. Resource-aware exploration is better suited to LEO than ordinary UCB because eclipse, contact gaps, and queue pressure make exploration costs state dependent.",
        "4. Long-term modulation must affect short-term ranking through lambdas and exploration scale; otherwise proposed collapses toward no_mode.",
        "5. WC case-study safe claims: resource-aware scoring changes action mix; balanced+roi is a strong baseline; heavy models require delivery/resource context. Keep exact thresholds and beta/lambda ablation details as engineering debug unless validated on broader scenarios.",
    ]
    (output_dir / "paper_story_policy_repair_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARFish policy repair ablation.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sarfish_policy_repair")
    args = parser.parse_args()
    configure_logging()
    run_ablation(args.output_dir)


if __name__ == "__main__":
    main()
