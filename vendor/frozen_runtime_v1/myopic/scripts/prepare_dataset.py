"""导出 mock 任务流预览。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs  # noqa: E402
from src.data.task_stream import TaskStreamBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="导出任务流预览。")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "task_stream_preview.csv")
    args = parser.parse_args()

    configs = load_project_configs(PROJECT_ROOT)
    tasks = TaskStreamBuilder(configs["dataset"]).build(int(configs["dataset"]["max_tasks"]))
    rows = [task.__dict__ for task in tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
