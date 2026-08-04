"""从已有 CSV 重新生成图表。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import make_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="根据已有输出重新生成 figures。")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    summary_path = args.output_dir / "summary.csv"
    trajectory_paths = sorted((args.output_dir / "trajectories").glob("*.csv"))
    make_figures(summary_path, trajectory_paths, args.output_dir / "figures")


if __name__ == "__main__":
    main()
