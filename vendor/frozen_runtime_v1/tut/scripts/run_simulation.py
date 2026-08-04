"""运行完整 case study 闭环仿真的脚本入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import ensure_output_dirs, load_project_configs  # noqa: E402
from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.plots import make_figures  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


def _summaries_match(proposed_summary: dict, no_mode_summary: dict) -> bool:
    ignored = {"router", "router_name", "rank"}
    comparable_keys = sorted((set(proposed_summary) & set(no_mode_summary)) - ignored)
    return bool(comparable_keys) and all(str(proposed_summary[key]) == str(no_mode_summary[key]) for key in comparable_keys)


def _log_clean_eval_sanity_checks(results, logger) -> None:
    by_router = {result.router_name: result for result in results}
    proposed = by_router.get("proposed_lyapunov_rule_mode_safe_bandit")
    no_mode = by_router.get("no_mode")
    reference = proposed or next(iter(results), None)
    if reference is None:
        return

    logger.info("sanity total_generated_tasks=%s", reference.summary.get("total_generated_tasks"))
    logger.info("sanity emergency_task_count=%s", reference.summary.get("emergency_task_count"))
    if proposed is not None:
        logger.info(
            "sanity proposed battery min/avg/final=%.6f/%.6f/%.6f",
            float(proposed.summary.get("min_battery_soc", 0.0)),
            float(proposed.summary.get("average_battery_soc", 0.0)),
            float(proposed.summary.get("final_battery_soc", proposed.summary.get("final_battery", 0.0))),
        )
        logger.info("sanity proposed raw_safety_violation_count=%s", proposed.summary.get("raw_safety_violation_count"))
        logger.info(
            "sanity proposed final_resource_safety_violation_count=%s",
            proposed.summary.get("final_resource_safety_violation_count"),
        )
        logger.info("sanity proposed task_deferred_count=%s", proposed.summary.get("task_deferred_count"))
        logger.info("sanity proposed deadline_missed_count=%s", proposed.summary.get("deadline_missed_count"))
        logger.info("sanity proposed downlink_throttle_count=%s", proposed.summary.get("downlink_throttle_count"))
        logger.info(
            "sanity proposed downlink_throttle_thermal_count=%s",
            proposed.summary.get("downlink_throttle_thermal_count"),
        )
        logger.info("sanity proposed delivered_mb=%s", proposed.summary.get("delivered_mb"))
        corrected_but_still_resource_unsafe_count = sum(
            1
            for record in proposed.records
            if record.safety_corrected and record.final_resource_safety_violation
        )
        logger.info(
            "sanity proposed corrected_but_still_resource_unsafe_count=%s",
            corrected_but_still_resource_unsafe_count,
        )

    summaries_identical = False
    actions_identical = False
    action_difference_count = 0
    if proposed is not None and no_mode is not None:
        logger.info("sanity no_mode raw_safety_violation_count=%s", no_mode.summary.get("raw_safety_violation_count"))
        logger.info(
            "sanity no_mode final_resource_safety_violation_count=%s",
            no_mode.summary.get("final_resource_safety_violation_count"),
        )
        summaries_identical = _summaries_match(proposed.summary, no_mode.summary)
        proposed_actions = [record.executed_action for record in proposed.records]
        no_mode_actions = [record.executed_action for record in no_mode.records]
        actions_identical = proposed_actions == no_mode_actions
        action_difference_count = sum(
            proposed_action != no_mode_action for proposed_action, no_mode_action in zip(proposed_actions, no_mode_actions)
        )
    logger.info("sanity proposed_no_mode_summaries_identical=%s", summaries_identical)
    logger.info("sanity proposed_no_mode_selected_actions_identical=%s", actions_identical)
    logger.info("sanity proposed_no_mode_selected_action_difference_count=%s", action_difference_count)
    logger.info("sanity deadline_satisfaction_ratio denominator=total_generated_tasks")
    if summaries_identical or actions_identical:
        logger.warning("proposed and no_mode are identical in clean-eval sanity checks; inspect z_k/mode effects.")


def run_case_study(output_dir: Path, num_slots: int | None = None):
    """执行整套 mock case study。"""

    configs = load_project_configs(PROJECT_ROOT)
    if num_slots is not None:
        configs["timeline"]["use_num_days"] = False
        configs["timeline"]["num_slots"] = int(num_slots)
        configs["dataset"]["max_tasks"] = max(int(configs["dataset"]["max_tasks"]), int(num_slots))
    configs["simulator"]["slot_seconds"] = float(configs["timeline"].get("slot_seconds", configs["simulator"]["slot_seconds"]))

    output_dirs = ensure_output_dirs(output_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    routers = build_routers(configs["experiment"], environment.action_space)

    logger = get_logger("run_simulation")
    results = []
    episode_runner = environment.build_episode_runner(timeline)
    for router in routers:
        logger.info("开始运行 router: %s", router.router_name)
        results.append(episode_runner.run(router))

    summary_path, trajectory_paths = write_episode_results(results, output_dirs)
    make_figures(summary_path, trajectory_paths, output_dirs["figures"])
    _log_clean_eval_sanity_checks(results, logger)
    logger.info("仿真完成，输出目录: %s", output_dirs["root"])
    return summary_path, trajectory_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 LEO 星上算通协同 mock case study。")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--num-slots", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    run_case_study(output_dir=args.output_dir, num_slots=args.num_slots)


if __name__ == "__main__":
    main()
