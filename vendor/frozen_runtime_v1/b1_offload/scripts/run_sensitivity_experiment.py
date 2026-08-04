"""运行小型参数敏感性实验。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging_utils import configure_logging, get_logger  # noqa: E402
from src.evaluation.sensitivity import run_sensitivity_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 reward 与 profile 组合的参数敏感性实验。")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sensitivity")
    parser.add_argument("--num-slots", type=int, default=120)
    args = parser.parse_args()

    configure_logging()
    logger = get_logger("run_sensitivity_experiment")
    logger.info("开始运行小型参数敏感性实验。")
    artifact_paths = run_sensitivity_experiment(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        num_slots=args.num_slots,
    )
    logger.info("敏感性实验完成，结果目录: %s", args.output_dir)
    for name, path in artifact_paths.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
