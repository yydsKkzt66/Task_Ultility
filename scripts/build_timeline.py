"""构建并预览 virtual timeline。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 synthetic virtual timeline。")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "timeline_preview.csv")
    args = parser.parse_args()

    configs = load_project_configs(PROJECT_ROOT)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"]).build()
    rows = [
        {
            "slot_index": step.slot_index,
            "task_id": step.task.task_id,
            "task_value": step.task.task_value,
            "deadline_s": step.task.deadline_s,
            "in_contact": step.link_trace.in_contact,
            "throughput_mbps": step.link_trace.throughput_mbps,
            "solar_energy_j": step.resource_trace.solar_energy_j,
        }
        for step in timeline
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
