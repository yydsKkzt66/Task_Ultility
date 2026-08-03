"""Run SARFish GRD validation 20s dry-run diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.plots import make_figures  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


ROUTERS_OF_INTEREST = (
    "proposed_lyapunov_rule_mode_safe_bandit",
    "utility_rule",
    "fixed_lightweight",
    "fixed_high_accuracy",
    "no_mode",
    "no_lyapunov",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return f"n={len(values)}, min={min(values):.3f}, mean={mean(values):.3f}, max={max(values):.3f}"


def _load_dry_run_configs() -> dict[str, dict]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_20s_dryrun.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_20s_dryrun.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_20s_dryrun.yaml")
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    return configs


def _write_sanity_report(output_dir: Path, configs: dict[str, dict]) -> Path:
    summary_rows = _read_csv(output_dir / "summary.csv")
    summary_by_router = {row["router"]: row for row in summary_rows}
    proposed_rows = _read_csv(output_dir / "trajectories" / "proposed_lyapunov_rule_mode_safe_bandit.csv")
    task_rows = [row for row in proposed_rows if _to_int(row.get("has_task")) == 1]
    scene_ids = {row.get("scene_id", "") for row in task_rows if row.get("scene_id")}
    window_counts: dict[str, int] = defaultdict(int)
    for row in task_rows:
        window_counts[row.get("observation_window_id") or "(background/no_window)"] += 1

    action_counts = Counter(row.get("selected_action", "") for row in task_rows)
    model_counts = Counter(row.get("model_name", "") for row in task_rows)
    payload_counts = Counter(row.get("payload_policy", "") for row in task_rows)
    total_slots = int(configs["timeline"]["num_slots"])
    slot_seconds = float(configs["timeline"]["slot_seconds"])
    duration_s = total_slots * slot_seconds

    lines = [
        "# SARFish 20s Dry Run Sanity Report",
        "",
        f"- output_dir: `{output_dir.as_posix()}`",
        f"- slot_seconds: {slot_seconds:g}",
        f"- num_slots: {total_slots}",
        f"- total_physical_duration_s: {duration_s:g}",
        f"- total_physical_duration_h: {duration_s / 3600.0:.3f}",
        f"- total_generated_tasks: {_to_int(summary_by_router['proposed_lyapunov_rule_mode_safe_bandit'].get('total_generated_tasks'))}",
        f"- unique scene_id count: {len(scene_ids)}",
        f"- scene_coverage_ratio: {len(scene_ids) / 50.0:.3f}",
        "",
        "## Observation windows",
        "",
        "| observation_window_id | task_count |",
        "| --- | ---: |",
    ]
    for window_id, count in sorted(window_counts.items()):
        lines.append(f"| {window_id} | {count} |")

    lines.extend(
        [
            "",
            "## Has-task statistics",
            "",
            f"- task_value: {_stats([_to_float(row.get('V_t')) for row in task_rows])}",
            f"- deadline_s: {_stats([_to_float(row.get('deadline_s')) for row in task_rows])}",
            f"- difficulty: {_stats([_to_float(row.get('H_t')) for row in task_rows])}",
            f"- uncertainty: {_stats([_to_float(row.get('U_t')) for row in task_rows])}",
            "",
            "## Main summary metrics",
            "",
            "| router | gross_task_utility | total_effective_utility | net_system_utility | total_penalized_reward | completed_tasks | deadline_met_tasks | delivered_items | expired_items | delivered_mb | total_energy | min_battery_soc | max_thermal | avg_queue_mb | max_queue_mb | final_resource_safety_violation_count |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for router in ROUTERS_OF_INTEREST:
        row = summary_by_router.get(router, {})
        lines.append(
            f"| {router} | {_to_float(row.get('gross_task_utility')):.3f} | "
            f"{_to_float(row.get('total_effective_utility')):.3f} | "
            f"{_to_float(row.get('net_system_utility')):.3f} | "
            f"{_to_float(row.get('total_penalized_reward')):.3f} | "
            f"{_to_int(row.get('completed_tasks'))} | {_to_int(row.get('deadline_met_tasks'))} | "
            f"{_to_int(row.get('delivered_items'))} | {_to_int(row.get('expired_items'))} | "
            f"{_to_float(row.get('delivered_mb')):.3f} | {_to_float(row.get('total_energy')):.3f} | "
            f"{_to_float(row.get('min_battery_soc')):.4f} | {_to_float(row.get('max_thermal')):.4f} | "
            f"{_to_float(row.get('average_queue_mb')):.3f} | {_to_float(row.get('max_queue_mb')):.3f} | "
            f"{_to_int(row.get('final_resource_safety_violation_count'))} |"
        )

    proposed_summary = summary_by_router["proposed_lyapunov_rule_mode_safe_bandit"]
    lines.extend(
        [
            "",
            "## Proposed distributions",
            "",
            f"- action distribution: {dict(action_counts)}",
            f"- model distribution: {dict(model_counts)}",
            f"- payload distribution: {dict(payload_counts)}",
            "",
            "## Proposed queue/resource checks",
            "",
            f"- average_queue_mb: {_to_float(proposed_summary.get('average_queue_mb')):.3f}",
            f"- max_queue_mb: {_to_float(proposed_summary.get('max_queue_mb')):.3f}",
            f"- expired_items: {_to_int(proposed_summary.get('expired_items'))}",
            f"- delivered_items: {_to_int(proposed_summary.get('delivered_items'))}",
            f"- completed_tasks: {_to_int(proposed_summary.get('completed_tasks'))}",
            f"- min_battery_soc: {_to_float(proposed_summary.get('min_battery_soc')):.4f}",
            f"- max_thermal: {_to_float(proposed_summary.get('max_thermal')):.4f}",
            f"- raw_safety_violation_count: {_to_int(proposed_summary.get('raw_safety_violation_count'))}",
            f"- final_resource_safety_violation_count: {_to_int(proposed_summary.get('final_resource_safety_violation_count'))}",
            "",
            "## 60s short-run comparison",
            "",
        ]
    )
    sixty_summary_path = PROJECT_ROOT / "outputs" / "sarfish_grd_validation_short_sim" / "summary.csv"
    if sixty_summary_path.exists():
        sixty_rows = {row["router"]: row for row in _read_csv(sixty_summary_path)}
        sixty_prop = sixty_rows.get("proposed_lyapunov_rule_mode_safe_bandit", {})
        lines.extend(
            [
                f"- 60s proposed expired_items: {_to_int(sixty_prop.get('expired_items'))}; 20s proposed expired_items: {_to_int(proposed_summary.get('expired_items'))}.",
                f"- 60s proposed delivered_items: {_to_int(sixty_prop.get('delivered_items'))}; 20s proposed delivered_items: {_to_int(proposed_summary.get('delivered_items'))}.",
                f"- 60s proposed completed_tasks: {_to_int(sixty_prop.get('completed_tasks'))}; 20s proposed completed_tasks: {_to_int(proposed_summary.get('completed_tasks'))}.",
                f"- 60s proposed gross_task_utility: {_to_float(sixty_prop.get('gross_task_utility')):.3f}; 20s proposed gross_task_utility: {_to_float(proposed_summary.get('gross_task_utility')):.3f}.",
            ]
        )
    else:
        lines.append("- 60s short-run summary was not found, so direct comparison was skipped.")

    lines.extend(
        [
            "",
            "## Scale diagnostics",
            "",
            "- Downlink capacity uses `throughput_mbps * slot_seconds / 8`, so the capacity formula scales with 20s slots.",
            "- Base energy uses `base_power_w * slot_seconds`, so base energy scales with physical time.",
            "- Queue timestamps and expiration use `slot_index * slot_seconds`, so deadline checks remain in seconds.",
            "- Synthetic contact currently uses `slot_index % 18`; changing 60s to 20s changes the physical contact cadence from 1080s to 360s. This is a visible configuration-scale caveat for the dry run.",
            "- Thermal cooling terms include per-slot constants, so a 20s run is not thermally identical to a 60s run unless those terms are explicitly time-normalized. This run diagnoses that issue but does not tune it.",
        ]
    )

    report_path = output_dir / "dry_run_sanity_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_dry_run(output_dir: Path) -> None:
    configs = _load_dry_run_configs()
    output_dirs = ensure_output_dirs(output_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    routers = build_routers(configs["experiment"], environment.action_space)
    episode_runner = environment.build_episode_runner(timeline)
    logger = get_logger("run_sarfish_20s_dry_run")

    results = []
    for router in routers:
        logger.info("running router: %s", router.router_name)
        results.append(episode_runner.run(router))

    summary_path, trajectory_paths = write_episode_results(results, output_dirs)
    make_figures(summary_path, trajectory_paths, output_dirs["figures"])
    generate_lifetime_outputs(output_dir, float(configs["timeline"]["slot_seconds"]))
    _write_sanity_report(output_dir, configs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARFish GRD validation 20s dry run.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sarfish_20s_dry_run")
    args = parser.parse_args()
    configure_logging()
    run_dry_run(args.output_dir)


if __name__ == "__main__":
    main()
