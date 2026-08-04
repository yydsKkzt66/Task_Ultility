"""Run the default project demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import ensure_output_dirs, load_project_configs
from src.common.logging_utils import configure_logging, get_logger
from src.evaluation.baselines import build_routers
from src.evaluation.plots import make_figures
from src.evaluation.report import write_episode_results
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_demo(output_dir: Path, num_slots: int | None = None):
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

    logger = get_logger("run_demo")
    results = []
    episode_runner = environment.build_episode_runner(timeline)
    for router in routers:
        logger.info("Running router: %s", router.router_name)
        results.append(episode_runner.run(router))

    summary_path, trajectory_paths = write_episode_results(results, output_dirs)
    make_figures(summary_path, trajectory_paths, output_dirs["figures"])
    logger.info("Demo completed. Outputs: %s", output_dirs["root"])
    return summary_path, trajectory_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the default project demo.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "demo")
    parser.add_argument("--num-slots", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    run_demo(output_dir=args.output_dir, num_slots=args.num_slots)


if __name__ == "__main__":
    main()
