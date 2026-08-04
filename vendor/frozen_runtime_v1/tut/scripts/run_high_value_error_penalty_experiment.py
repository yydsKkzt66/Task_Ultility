"""运行高价值任务错判重罚实验。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.high_value_error_penalty_experiment import run_high_value_error_penalty_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行独立的高价值任务错判重罚实验。")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "high_value_error_penalty")
    parser.add_argument("--num-slots", type=int, default=120)
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("run_high_value_error_penalty_experiment")
    logger.info("开始运行高价值任务错判重罚实验。")
    artifact_paths = run_high_value_error_penalty_experiment(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        num_slots=args.num_slots,
    )
    logger.info("实验完成，结果目录: %s", args.output_dir)
    for name, path in artifact_paths.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
