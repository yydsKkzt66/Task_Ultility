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


def run_case_study(output_dir: Path, num_slots: int | None = None):
    """执行整套 mock case study。"""

    configs = load_project_configs(PROJECT_ROOT)
    if num_slots is not None:
        configs["timeline"]["num_slots"] = int(num_slots)
        configs["dataset"]["max_tasks"] = max(int(configs["dataset"]["max_tasks"]), int(num_slots))

    output_dirs = ensure_output_dirs(output_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"]).build()
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
