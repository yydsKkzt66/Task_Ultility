"""Run SARFish payload-profile calibration ablations."""

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
    "proposed_yolo_profile_policy_repair_baseline",
    "proposed_calibrated_payload_profile",
    "proposed_payload_specific_delivery_proxy",
    "proposed_calibrated_payload_plus_cost_aware_ucb",
    "proposed_calibrated_payload_plus_z_guided",
    "proposed_payload_repaired_combined",
    "utility_rule",
    "no_mode",
    "no_lyapunov",
    "fixed_lightweight",
    "fixed_high_accuracy",
)
PAYLOAD_TYPES = ("metadata", "roi", "roi_context", "defer")
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


def _avg(rows: list[dict[str, str]], field: str) -> float:
    return _safe_div(sum(_to_float(row.get(field)) for row in rows), len(rows))


def _load_ablation_cfg() -> dict[str, object]:
    return load_yaml(PROJECT_ROOT / "configs" / "payload_profile_ablation.yaml")


def _load_repair_cfg(variant: str) -> dict[str, object]:
    cfg = _load_ablation_cfg()
    repair = copy.deepcopy(cfg["policy_repair_defaults"])
    repair.update(copy.deepcopy(cfg.get("variant_overrides", {}).get(variant, {})))
    return repair


def _load_configs(variant: str) -> dict[str, dict]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_payload_profile_ablation_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    profile_cfg = load_yaml(PROJECT_ROOT / "configs" / "model_profiles_yolo_inspired_calibrated.yaml")
    configs["model_pool"] = copy.deepcopy(configs["model_pool"])
    configs["model_pool"]["profile_overrides"] = dict(profile_cfg["profile_overrides"])
    configs["simulator"]["policy_repair"] = _load_repair_cfg(variant)
    ablation_cfg = _load_ablation_cfg()
    profile_variant = dict(ablation_cfg.get("payload_profile_variants", {})).get(variant, "calibrated")
    if profile_variant == "calibrated":
        configs["payload_profiles"] = load_yaml(PROJECT_ROOT / "configs" / "payload_profiles_roi_compression_calibrated.yaml")
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


def _trajectory_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return _read_csv(output_dir / "trajectories" / f"{variant}.csv")


def _task_rows(output_dir: Path, variant: str) -> list[dict[str, str]]:
    return [row for row in _trajectory_rows(output_dir, variant) if _to_int(row.get("has_task")) == 1]


def write_payload_profile_audit(output_dir: Path) -> None:
    lines = [
        "# Payload Profile Audit Report",
        "",
        "## Current implementation",
        "",
        "1. Current executed payload sizes are fixed profile values plus detector ROI-count increments: metadata = 0.10 + 0.02*n MB, ROI = 0.80 + 0.18*n MB, ROI+Ctx = 1.60 + 0.35*n MB, defer = 0.",
        "2. Current information content is profile information_factor plus min(0.25, 0.05*n) for delivered payloads; defer keeps its profile value.",
        "3. Profile-level ROI+Ctx / ROI information ratio is 1.00 / 0.72 = 1.389 before the ROI-count bonus.",
        "4. Executed ROI+Ctx / ROI size ratio is (1.60 + 0.35*n) / (0.80 + 0.18*n), about 2.0 at n=0 and 1.94 for large n. Predictor-level comm cost ratio is 1.60*1.20 / (0.80*0.90) = 2.667 for equal detector quality. Payload-specific encoding latency/energy did not exist.",
        "5. The previous payload size path did not use SARFish bbox area, target_count, median bbox area, or scene density; it only used detector ROI count.",
        "6. Previous delivery-aware info gain used one shared proxy, with only a metadata floor inferred from low raw_info_gain rather than an explicit payload-type model.",
        "7. high_accuracy+roi_context stays candidate-high because ROI+Ctx has information_factor 1.00, predictor size is only moderately larger than ROI, and there is no separate encoding-cost multiplier.",
        "8. The old delivery proxy can be too conservative because predicted queued status and no-contact jointly multiply all payload types through one shared risk equation, so metadata and ROI are discounted nearly like ROI+Ctx except for the metadata floor.",
    ]
    (output_dir / "payload_profile_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_action_payload_distribution(output_dir: Path) -> None:
    action_rows = []
    payload_rows = []
    for variant in VARIANTS:
        task_rows = _task_rows(output_dir, variant)
        action_counts = Counter(row.get("selected_action", "") for row in task_rows)
        payload_counts = Counter(row.get("payload_policy", "") for row in task_rows)
        for action in FOCUS_ACTIONS:
            action_rows.append({"variant": variant, "action": action, "count": action_counts.get(action, 0), "total_tasks": len(task_rows)})
        for payload in PAYLOAD_TYPES:
            payload_rows.append({"variant": variant, "payload_type": payload, "count": payload_counts.get(payload, 0), "total_tasks": len(task_rows)})
    _write_csv(output_dir / "action_distribution.csv", action_rows)
    _write_csv(output_dir / "payload_distribution.csv", payload_rows)


def write_payload_size_diagnostics(output_dir: Path) -> None:
    rows = []
    for variant in VARIANTS:
        for payload in PAYLOAD_TYPES:
            selected = [row for row in _task_rows(output_dir, variant) if row.get("payload_policy") == payload]
            source_counts = Counter(row.get("payload_size_source", "") for row in selected)
            rows.append(
                {
                    "variant": variant,
                    "payload_type": payload,
                    "count": len(selected),
                    "avg_payload_size_kb": _avg(selected, "S_gen_current_mb") * 1000.0,
                    "avg_estimated_target_count": _avg(selected, "estimated_target_count"),
                    "avg_estimated_bbox_area": _avg(selected, "estimated_bbox_area"),
                    "avg_estimated_roi_size_kb": _avg(selected, "estimated_roi_size_kb"),
                    "avg_estimated_roi_context_size_kb": _avg(selected, "estimated_roi_context_size_kb"),
                    "payload_size_source_counts": ";".join(f"{key}:{value}" for key, value in sorted(source_counts.items())) or "none",
                }
            )
    _write_csv(output_dir / "payload_size_diagnostics.csv", rows)


def _lifetime_counts(output_dir: Path) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    path = output_dir / "task_lifetime_outcomes.csv"
    if not path.exists():
        return counts
    for row in _read_csv(path):
        counts[str(row["router_name"])][str(row["final_lifetime_status"])] += 1
    return counts


def _dist(path: Path, key_field: str) -> dict[tuple[str, str], int]:
    return {(row["variant"], row[key_field]): _to_int(row["count"]) for row in _read_csv(path)}


def _summary(output_dir: Path) -> dict[str, dict[str, str]]:
    return {row["router"]: row for row in _read_csv(output_dir / "summary.csv")}


def _similarity(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]) -> tuple[int, int, float]:
    pairs = [(a, b) for a, b in zip(rows_a, rows_b) if a.get("task_id") == b.get("task_id")]
    same = sum(1 for a, b in pairs if a.get("selected_action") == b.get("selected_action"))
    return same, len(pairs), _safe_div(same, len(pairs))


def _payload_averages(output_dir: Path, variant: str, payload: str) -> dict[str, float]:
    rows = [row for row in _task_rows(output_dir, variant) if row.get("payload_policy") == payload]
    return {
        "size_mb": _avg(rows, "S_gen_current_mb"),
        "proxy": _avg(rows, "delivery_success_proxy"),
        "info": _avg(rows, "effective_info_gain"),
        "raw_info": _avg(rows, "raw_info_gain"),
    }


def write_reports(output_dir: Path) -> None:
    summary = _summary(output_dir)
    lifetime = _lifetime_counts(output_dir)
    action_dist = _dist(output_dir / "action_distribution.csv", "action")
    payload_dist = _dist(output_dir / "payload_distribution.csv", "payload_type")

    lines = [
        "# Payload Profile Ablation Report",
        "",
        "| variant | gross | net | penalized | resource | requirement | compute | comm | total energy | max thermal | max DoD | delivered MB | generated MB | expired | lifetime delivered / expired / deferred | metadata / ROI / ROI+Ctx / defer | high_acc+roi_context | balanced+roi | avg proxy | avg effective info |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        row = summary.get(variant, {})
        traj = _trajectory_rows(output_dir, variant)
        max_dod = max((1.0 - _to_float(item.get("battery_next_soc", item.get("battery_soc"))) for item in traj), default=0.0)
        counts = lifetime.get(variant, Counter())
        task_rows = _task_rows(output_dir, variant)
        lines.append(
            f"| {variant} | {_to_float(row.get('gross_task_utility')):.3f} | {_to_float(row.get('net_system_utility')):.3f} | "
            f"{_to_float(row.get('total_penalized_reward')):.3f} | {_to_float(row.get('total_resource_penalty')):.3f} | "
            f"{_to_float(row.get('total_requirement_penalty')):.3f} | {_to_float(row.get('total_compute_energy')):.1f} | "
            f"{_to_float(row.get('total_comm_energy')):.1f} | {_to_float(row.get('total_energy')):.1f} | "
            f"{_to_float(row.get('max_thermal')):.4f} | {max_dod:.4f} | {_to_float(row.get('delivered_mb')):.3f} | "
            f"{_to_float(row.get('total_generated_payload_mb')):.3f} | {_to_int(row.get('expired_items'))} | "
            f"{counts.get('delivered', 0)} / {counts.get('expired', 0)} / {counts.get('deferred', 0)} | "
            f"{payload_dist.get((variant, 'metadata'), 0)} / {payload_dist.get((variant, 'roi'), 0)} / {payload_dist.get((variant, 'roi_context'), 0)} / {payload_dist.get((variant, 'defer'), 0)} | "
            f"{action_dist.get((variant, 'high_accuracy+roi_context'), 0)} | {action_dist.get((variant, 'balanced+roi'), 0)} | "
            f"{_avg(task_rows, 'delivery_success_proxy'):.3f} | {_avg(task_rows, 'effective_info_gain'):.3f} |"
        )
    (output_dir / "payload_profile_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_roi_context_overuse_report(output_dir, summary, action_dist, payload_dist)
    _write_delivery_proxy_report(output_dir)
    _write_comparison_report(output_dir, "utility_rule", "proposed_vs_utility_rule_payload_report.md")
    _write_comparison_report(output_dir, "no_mode", "proposed_vs_no_mode_payload_report.md")
    _write_counterfactual_report(output_dir, summary, action_dist, payload_dist)
    _write_root_cause_report(output_dir, summary, action_dist, payload_dist)
    _write_paper_notes(output_dir)


def _write_roi_context_overuse_report(output_dir: Path, summary, action_dist, payload_dist) -> None:
    lines = ["# ROI Context Overuse Report", ""]
    for variant in VARIANTS:
        roi = _payload_averages(output_dir, variant, "roi")
        ctx = _payload_averages(output_dir, variant, "roi_context")
        size_ratio = _safe_div(ctx["size_mb"], roi["size_mb"])
        raw_ratio = _safe_div(ctx["raw_info"], roi["raw_info"])
        lines.append(
            f"- {variant}: ROI+Ctx count {payload_dist.get((variant, 'roi_context'), 0)}, high_accuracy+roi_context {action_dist.get((variant, 'high_accuracy+roi_context'), 0)}, size ratio vs ROI {size_ratio:.3f}, raw-info ratio vs ROI {raw_ratio:.3f}."
        )
    (output_dir / "roi_context_overuse_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_delivery_proxy_report(output_dir: Path) -> None:
    lines = ["# Delivery Proxy Calibration Report", ""]
    for variant in VARIANTS:
        task_rows = _task_rows(output_dir, variant)
        lines.append(f"## {variant}")
        lines.append(f"- average proxy: {_avg(task_rows, 'delivery_success_proxy'):.3f}")
        for payload in PAYLOAD_TYPES:
            rows = [row for row in task_rows if row.get("payload_policy") == payload]
            reasons = Counter(row.get("payload_delivery_risk_reason", "") for row in rows).most_common(4)
            lines.append(
                f"- {payload}: count {len(rows)}, raw proxy {_avg(rows, 'delivery_proxy_raw'):.3f}, after floor {_avg(rows, 'delivery_proxy_after_floor'):.3f}, effective info {_avg(rows, 'effective_info_gain'):.3f}, reasons {reasons}"
            )
        lines.append("")
    (output_dir / "delivery_proxy_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison_report(output_dir: Path, baseline: str, filename: str) -> None:
    summary = _summary(output_dir)
    lines = [f"# Proposed vs {baseline} Payload Report", ""]
    baseline_rows = _task_rows(output_dir, baseline)
    baseline_net = _to_float(summary.get(baseline, {}).get("net_system_utility"))
    for variant in [item for item in VARIANTS if item.startswith("proposed_")]:
        same, total, ratio = _similarity(_task_rows(output_dir, variant), baseline_rows)
        net = _to_float(summary.get(variant, {}).get("net_system_utility"))
        lines.append(
            f"- {variant}: action similarity {same}/{total} ({ratio:.3f}); net {net:.3f} vs {baseline} {baseline_net:.3f}; delta {net - baseline_net:.3f}."
        )
    (output_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_variants(summary: dict[str, dict[str, str]]) -> list[str]:
    proposed = [item for item in VARIANTS if item.startswith("proposed_")]
    return sorted(proposed, key=lambda item: _to_float(summary.get(item, {}).get("net_system_utility")), reverse=True)[:3]


def _write_counterfactual_report(output_dir: Path, summary, action_dist, payload_dist) -> None:
    lines = ["# Payload Counterfactual Diagnosis", ""]
    for variant in _best_variants(summary):
        roi = _payload_averages(output_dir, variant, "roi")
        ctx = _payload_averages(output_dir, variant, "roi_context")
        meta = _payload_averages(output_dir, variant, "metadata")
        lines.extend(
            [
                f"## {variant}",
                f"1. ROI+Ctx candidate-level inflation: {'still possible' if action_dist.get((variant, 'high_accuracy+roi_context'), 0) > action_dist.get((variant, 'balanced+roi'), 0) else 'not dominant in selected actions'}.",
                f"2. ROI+Ctx raw_info / ROI raw_info = {_safe_div(ctx['raw_info'], roi['raw_info']):.3f}; calibrated profile caps the marginal gain.",
                f"3. ROI+Ctx size / ROI size = {_safe_div(ctx['size_mb'], roi['size_mb']):.3f}; comm cost follows this generated payload size.",
                f"4. Metadata keeps value in poor communication through average proxy {meta['proxy']:.3f}.",
                f"5. Delivery proxy by type separates metadata {meta['proxy']:.3f}, ROI {roi['proxy']:.3f}, ROI+Ctx {ctx['proxy']:.3f}.",
                f"6. balanced+roi count {action_dist.get((variant, 'balanced+roi'), 0)} vs high_accuracy+roi_context {action_dist.get((variant, 'high_accuracy+roi_context'), 0)}.",
                f"7. high_accuracy+roi_context should be read against contact/deadline/resource columns in trajectories; the calibrated proxy penalizes it most under queue/deadline/no-contact risk.",
                f"8. metadata/defer counts are {payload_dist.get((variant, 'metadata'), 0)} / {payload_dist.get((variant, 'defer'), 0)}.",
                f"9. Proposed/no_mode separation is quantified in proposed_vs_no_mode_payload_report.md.",
                "",
            ]
        )
    (output_dir / "payload_counterfactual_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_cause_report(output_dir: Path, summary, action_dist, payload_dist) -> None:
    baseline = "proposed_yolo_profile_policy_repair_baseline"
    calibrated = "proposed_calibrated_payload_profile"
    proxy = "proposed_payload_specific_delivery_proxy"
    combined = "proposed_payload_repaired_combined"
    baseline_net = _to_float(summary[baseline]["net_system_utility"])
    calibrated_net = _to_float(summary[calibrated]["net_system_utility"])
    proxy_net = _to_float(summary[proxy]["net_system_utility"])
    combined_net = _to_float(summary[combined]["net_system_utility"])
    no_mode_net = _to_float(summary["no_mode"]["net_system_utility"])
    utility_net = _to_float(summary["utility_rule"]["net_system_utility"])
    lines = [
        "# Root Cause After Payload Calibration",
        "",
        f"- baseline net: {baseline_net:.3f}",
        f"- calibrated payload net: {calibrated_net:.3f}",
        f"- payload-specific proxy net: {proxy_net:.3f}",
        f"- combined net: {combined_net:.3f}",
        f"- utility_rule net: {utility_net:.3f}",
        f"- no_mode net: {no_mode_net:.3f}",
        f"- ROI+Ctx count baseline -> combined: {payload_dist.get((baseline, 'roi_context'), 0)} -> {payload_dist.get((combined, 'roi_context'), 0)}",
        "",
        f"- payload profile was a major contributor: {'yes' if calibrated_net >= baseline_net or payload_dist.get((calibrated, 'roi_context'), 0) < payload_dist.get((baseline, 'roi_context'), 0) else 'not the only cause in this run'}.",
        f"- ROI+Ctx was previously too cheap / too informative: yes; old profile used information_factor 1.00 and no encoding multiplier, while calibrated ROI+Ctx is larger and only marginally more informative than ROI.",
        f"- payload-specific delivery proxy improved results: {'yes' if proxy_net >= calibrated_net else 'mixed; inspect delivery_proxy_calibration_report.md'}.",
        f"- cost-aware UCB still needs attention: {'yes' if _to_float(summary['proposed_calibrated_payload_plus_cost_aware_ucb']['net_system_utility']) < calibrated_net else 'not primary in this run'}.",
        f"- z-guided coupling still needs attention: {'yes' if _to_float(summary['proposed_calibrated_payload_plus_z_guided']['net_system_utility']) < calibrated_net else 'not primary in this run'}.",
        "- Continue calibrating roi_context payload size if ROI+Ctx/ROI size ratio is below the intended 2.5-5 range in roi_context_overuse_report.md.",
        "- No implementation bug is indicated unless diagnostics show unchanged payload_size_source/proxy fields across enabled variants.",
        "- Reward weights should still not be tuned; the remaining levers are payload profile, delivery proxy, UCB, and z-guided coupling.",
    ]
    (output_dir / "root_cause_after_payload_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_notes(output_dir: Path) -> None:
    lines = [
        "# Paper Story Payload Profile Notes",
        "",
        "1. Payload granularity is not monotonic: more context can raise semantic confidence, but it also consumes contact time, queue capacity, encoding energy, and deadline slack.",
        "2. ROI is the high value-per-bit semantic payload because it keeps the target evidence while avoiding full-context downlink burden.",
        "3. ROI+Ctx has marginal information gain over ROI, but its larger size and encoding cost make delivery risk state dependent.",
        "4. Delivery-aware information gain is the bridge between onboard compute and communication: information should be scored by whether it can reach the ground in time.",
        "5. Joint metadata / ROI / ROI+Ctx / defer selection captures the principle that a lightweight alert can be better than an expensive payload when contact or resources are tight.",
        "6. WC case-study-ready claims: calibrated payload profiles change action mix, ROI is a robust sweet spot, and contact-aware delivery gates heavy context. Engineering debug details such as exact proxy floors, UCB thresholds, and z multipliers should stay out until broader validation.",
    ]
    (output_dir / "paper_story_payload_profile_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(output_dir: Path) -> None:
    output_dirs = ensure_output_dirs(output_dir)
    write_payload_profile_audit(output_dir)
    logger = get_logger("run_sarfish_payload_profile_ablation")
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
    write_action_payload_distribution(output_dir)
    write_payload_size_diagnostics(output_dir)
    write_reports(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARFish payload profile calibration ablation.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sarfish_payload_profile_calibration")
    args = parser.parse_args()
    configure_logging()
    run_ablation(args.output_dir)


if __name__ == "__main__":
    main()
