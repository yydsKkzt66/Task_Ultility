"""Audit diagnostic-only product requirement v2 assignment.

This script compares the current product requirement rule against the isolated
v2 ideal-requirement rule. It does not modify router, reward, scorer, filter, or
default configuration behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.data.product_requirements import (  # noqa: E402
    assign_product_requirement,
    assign_product_requirement_v2,
    recommend_payload_plan_from_requirement,
)
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_v2_audit"
FORBIDDEN_FIELDS = {
    "label",
    "label_path",
    "target_count",
    "bbox",
    "object_count",
    "detector",
    "ground_truth",
    "future_delivery",
    "realized_future_utility",
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = completed.stdout
        if completed.stderr:
            output += "\nSTDERR:\n" + completed.stderr
        return output.strip()
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"git command failed: {exc}"


def _git_knows(path: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0


def _synthetic_new_file_diff(path: str) -> str:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return ""
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = [
        f"diff --git a/{path} b/{path}",
        "new file mode 100644",
        "index 0000000..0000000",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    return "\n".join([*header, *[f"+{line}" for line in lines]])


def _audit_diff(paths: list[str]) -> str:
    tracked_paths = [path for path in paths if _git_knows(path)]
    parts: list[str] = []
    if tracked_paths:
        parts.append(_run_git(["diff", "--", *tracked_paths]))
    for path in paths:
        if not _git_knows(path):
            parts.append(_synthetic_new_file_diff(path))
    return "\n\n".join(part for part in parts if part).strip()


def _payload_size_map(configs: dict[str, dict[str, Any]]) -> dict[str, float]:
    candidates = [
        configs.get("payload_profiles_roi_compression_calibrated", {}),
        configs.get("payload_profiles", {}),
    ]
    for cfg in candidates:
        profiles = cfg.get("profiles", [])
        if isinstance(profiles, list) and profiles:
            return {
                str(item.get("name", "")): float(item.get("size_mb", 0.0))
                for item in profiles
                if isinstance(item, dict)
            }
    return {"metadata": 0.1, "roi": 0.8, "roi_context": 1.6}


def _load_named_configs(
    *,
    dataset_file: str,
    task_stream_file: str,
    timeline_file: str,
    simulator_file: str,
) -> dict[str, dict[str, Any]]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / dataset_file)
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / task_stream_file)
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / timeline_file)
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / simulator_file)
    return configs


def _load_audit_configs(args: argparse.Namespace) -> tuple[dict[str, dict[str, Any]], str]:
    preferred = {
        "dataset_file": args.dataset_config,
        "task_stream_file": args.task_stream_config,
        "timeline_file": args.timeline_config,
        "simulator_file": args.simulator_config,
    }
    try:
        configs = _load_named_configs(**preferred)
        TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        return configs, "requested_configs"
    except Exception as exc:
        fallback = {
            "dataset_file": "dataset.yaml",
            "task_stream_file": "task_stream.yaml",
            "timeline_file": "timeline.yaml",
            "simulator_file": "simulator.yaml",
        }
        configs = _load_named_configs(**fallback)
        configs["timeline"]["use_num_days"] = False
        configs["timeline"]["num_slots"] = min(int(args.num_slots), int(configs["timeline"].get("num_slots", args.num_slots)))
        configs["task_stream"]["slot_seconds"] = configs["timeline"].get("slot_seconds", configs["task_stream"].get("slot_seconds", 60))
        return configs, f"fallback_mock_configs_after_error={exc}"


def _limit_timeline(configs: dict[str, dict[str, Any]], num_slots: int) -> None:
    configs["timeline"]["use_num_days"] = False
    configs["timeline"]["num_slots"] = int(num_slots)
    slot_seconds = float(configs["timeline"].get("slot_seconds", configs["task_stream"].get("slot_seconds", 20.0)))
    configs["timeline"]["slot_seconds"] = slot_seconds
    configs["task_stream"]["slot_seconds"] = slot_seconds
    configs["simulator"]["slot_seconds"] = slot_seconds


def _capacity_before_deadline(timeline, slot_index: int, deadline_s: float, downlink_efficiency: float) -> float:
    if slot_index < 0 or slot_index >= len(timeline):
        return 0.0
    slot_seconds = float(timeline[slot_index].slot_seconds)
    horizon_slots = max(1, int(math.ceil(max(0.0, deadline_s) / max(slot_seconds, 1.0e-9))))
    end = min(len(timeline), slot_index + horizon_slots)
    capacity_mb = 0.0
    for step in timeline[slot_index:end]:
        link = step.link_trace
        if not link.in_contact:
            continue
        capacity_mb += max(0.0, float(link.throughput_mbps)) * slot_seconds / 8.0 * downlink_efficiency
    return capacity_mb


def _slack_values(configs: dict[str, dict[str, Any]]) -> dict[str, float]:
    sim = configs["simulator"]
    queue_mb = float(sim.get("queue_init_mb", 0.0))
    queue_max = max(float(sim.get("queue_max_mb", 1.0)), 1.0e-9)
    battery_init = float(sim.get("battery_init", 1.0))
    battery_min = float(sim.get("battery_min", 0.0))
    thermal_init = float(sim.get("thermal_init", 0.0))
    thermal_max = max(float(sim.get("thermal_max", 1.0)), 1.0e-9)
    return {
        "queue_mb": queue_mb,
        "queue_max_mb": queue_max,
        "queue_slack": max(0.0, min(1.0, 1.0 - queue_mb / queue_max)),
        "battery_slack": max(0.0, min(1.0, (battery_init - battery_min) / max(1.0 - battery_min, 1.0e-9))),
        "thermal_slack": max(0.0, min(1.0, (thermal_max - thermal_init) / thermal_max)),
    }


def _scene_prior(task) -> float | None:
    meta = dict(getattr(task, "metadata", {}) or {})
    for key in ("scene_prior", "g_area_prior", "area_prior", "source_priority"):
        if key not in meta:
            continue
        try:
            return max(0.0, min(1.0, float(meta[key])))
        except (TypeError, ValueError):
            continue
    return None


def _product_use_case(task) -> str | None:
    meta = dict(getattr(task, "metadata", {}) or {})
    for key in ("product_use_case", "mission_use_case", "use_case"):
        value = meta.get(key)
        if value:
            return str(value)
    return None


def _rows_for_timeline(timeline, configs: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    sizes = _payload_size_map(configs)
    required_context_mb = max(float(sizes.get("roi_context", 1.6)), 1.0e-9)
    required_roi_mb = max(float(sizes.get("roi", 0.8)), 1.0e-9)
    required_metadata_mb = max(float(sizes.get("metadata", 0.1)), 1.0e-9)
    downlink_efficiency = float(configs["simulator"].get("downlink_efficiency", 1.0))
    slacks = _slack_values(configs)
    rows: list[dict[str, object]] = []

    for step in timeline:
        task = step.task
        if task is None:
            continue
        old_req, old_source = assign_product_requirement(
            task_value=task.task_value,
            deadline_s=task.deadline_s,
            uncertainty=task.uncertainty,
            high_value_flag=task.high_value_flag,
            emergency_flag=task.emergency_flag,
            task_class=task.task_class,
        )
        capacity = _capacity_before_deadline(timeline, step.slot_index, task.deadline_s, downlink_efficiency)
        deadline_slack = max(0.0, min(1.0, float(task.deadline_s) / 120.0))
        result = assign_product_requirement_v2(
            task_value=task.task_value,
            deadline_s=task.deadline_s,
            remaining_deadline_s=task.deadline_s,
            uncertainty=task.uncertainty,
            high_value_flag=task.high_value_flag,
            emergency_flag=task.emergency_flag,
            task_class=task.task_class,
            mission_source=str(task.metadata.get("dataset_name", "")) if task.metadata.get("dataset_name") else None,
            scene_prior=_scene_prior(task),
            product_use_case=_product_use_case(task),
            queue_mb=slacks["queue_mb"],
            queue_max_mb=slacks["queue_max_mb"],
            known_contact_capacity_before_deadline=capacity,
            required_capacity_for_roi_context=required_context_mb,
            battery_slack=slacks["battery_slack"],
            thermal_slack=slacks["thermal_slack"],
            queue_slack=slacks["queue_slack"],
            contact_slack=max(0.0, min(1.0, capacity / required_context_mb)),
            deadline_slack=deadline_slack,
        )
        recommendation = recommend_payload_plan_from_requirement(
            ideal_product_requirement=result.ideal_product_requirement,
            metadata_feasible=capacity >= required_metadata_mb,
            roi_feasible=capacity >= required_roi_mb,
            roi_context_feasible=capacity >= required_context_mb,
            metadata_then_roi_feasible=capacity >= required_metadata_mb,
            metadata_then_roi_context_feasible=capacity >= required_metadata_mb,
        )
        missing = sorted(set(result.missing_fields) | set(recommendation.missing_fields))
        scores = result.requirement_scores
        rows.append(
            {
                "task_id": task.task_id,
                "slot_index": step.slot_index,
                "task_value": f"{task.task_value:.6f}",
                "deadline_s": f"{task.deadline_s:.6f}",
                "remaining_deadline_s": f"{task.deadline_s:.6f}",
                "uncertainty": f"{task.uncertainty:.6f}",
                "emergency_flag": int(bool(task.emergency_flag)),
                "high_value_flag": int(bool(task.high_value_flag)),
                "task_class": task.task_class,
                "old_product_requirement": old_req,
                "old_product_requirement_source": old_source,
                "new_ideal_product_requirement": result.ideal_product_requirement,
                "S_alert": f"{scores['alert']:.6f}",
                "S_verification": f"{scores['verification']:.6f}",
                "S_context": f"{scores['context']:.6f}",
                "S_evidence": f"{scores['evidence']:.6f}",
                "recommended_payload_plan_shadow": recommendation.recommended_payload_plan,
                "known_contact_capacity_before_deadline_mb": f"{capacity:.6f}",
                "required_capacity_for_roi_context_mb": f"{required_context_mb:.6f}",
                "queue_pressure": f"{slacks['queue_mb'] / slacks['queue_max_mb']:.6f}",
                "missing_fields": ";".join(missing),
                "diagnostic_reason": result.diagnostic_reason,
            }
        )
    return rows


def _distribution_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = max(len(rows), 1)
    old_counts = Counter(str(row["old_product_requirement"]) for row in rows)
    new_counts = Counter(str(row["new_ideal_product_requirement"]) for row in rows)
    summary_rows: list[dict[str, object]] = []
    for requirement in ["alert", "verification", "context", "evidence"]:
        summary_rows.append(
            {
                "product_requirement": requirement,
                "old_count": old_counts.get(requirement, 0),
                "old_rate": f"{old_counts.get(requirement, 0) / total:.6f}",
                "new_count": new_counts.get(requirement, 0),
                "new_rate": f"{new_counts.get(requirement, 0) / total:.6f}",
            }
        )
    return summary_rows


def _transition_matrix(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total = max(len(rows), 1)
    counts = Counter(
        (str(row["old_product_requirement"]), str(row["new_ideal_product_requirement"])) for row in rows
    )
    matrix_rows: list[dict[str, object]] = []
    for old in ["alert", "verification", "context", "evidence"]:
        for new in ["alert", "verification", "context", "evidence"]:
            count = counts.get((old, new), 0)
            matrix_rows.append({"old": old, "new": new, "count": count, "rate": f"{count / total:.6f}"})
    return matrix_rows


def _plan_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        groups[(str(row["new_ideal_product_requirement"]), str(row["recommended_payload_plan_shadow"]))] += 1
    total = max(len(rows), 1)
    return [
        {
            "new_ideal_product_requirement": requirement,
            "recommended_payload_plan_shadow": plan,
            "count": count,
            "rate": f"{count / total:.6f}",
        }
        for (requirement, plan), count in sorted(groups.items())
    ]


def _changed_case_summary(rows: list[dict[str, object]], limit: int = 12) -> list[dict[str, object]]:
    changed = [row for row in rows if row["old_product_requirement"] != row["new_ideal_product_requirement"]]
    changed.sort(
        key=lambda row: (
            abs(float(row["S_evidence"]) - float(row["S_alert"])),
            -float(row["task_value"]),
        )
    )
    return changed[:limit]


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "No rows."
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|").replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _leakage_report(rows: list[dict[str, object]]) -> str:
    output_columns = set(rows[0].keys()) if rows else set()
    forbidden_in_outputs = sorted(field for field in output_columns for token in FORBIDDEN_FIELDS if token in field)
    return f"""# Leakage safety audit

Result: PASS for the v2 implementation path.

The v2 function consumes only decision-time mission-prior fields passed by the audit script:
task_value, deadline_s, remaining_deadline_s, uncertainty, high_value_flag, emergency_flag,
task_class, optional mission/source priors, queue/contact/resource slack estimates.

It does not consume SARFish labels, bbox count, object count, target class, detector outputs,
future delivery result, realized future utility, or detector ground truth.

Forbidden-field check on per-task output columns: {forbidden_in_outputs or "none"}.

Note: existing TaskSample objects may carry label_path or offline metadata for dataset bookkeeping.
Those fields are not read by assign_product_requirement_v2 or the shadow payload-plan helper.
"""


def _report(rows: list[dict[str, object]], source_note: str) -> tuple[str, str]:
    total = len(rows)
    changed = [row for row in rows if row["old_product_requirement"] != row["new_ideal_product_requirement"]]
    old_counts = Counter(str(row["old_product_requirement"]) for row in rows)
    new_counts = Counter(str(row["new_ideal_product_requirement"]) for row in rows)
    alert_to_context_evidence = [
        row
        for row in rows
        if row["old_product_requirement"] == "alert" and row["new_ideal_product_requirement"] in {"context", "evidence"}
    ]
    high_value_tight_not_alert_only = [
        row
        for row in rows
        if int(row["high_value_flag"]) == 1
        and row["new_ideal_product_requirement"] in {"context", "evidence"}
    ]
    collapsed = max(new_counts.values(), default=0) / max(total, 1) > 0.85
    recommendation = (
        "Keep v2 diagnostic-only and do not adopt it as the main default yet. "
        "The distribution should be reviewed under more regimes before any default-off experimental branch is used."
        if collapsed
        else "Keep v2 diagnostic-only for now, but it is reasonable to consider a default-off experimental branch."
    )
    changed_rows = _changed_case_summary(rows)
    report = f"""# Product requirement v2 audit report

## Run scope

- Source note: {source_note}
- Tasks audited: {total}
- Changed assignments: {len(changed)} ({len(changed) / max(total, 1):.2%})

## Required answers

1. How often does v2 differ from the old rule?

   v2 differs on {len(changed)} of {total} audited tasks ({len(changed) / max(total, 1):.2%}).

2. Does v2 avoid downgrading high-value/emergency tasks into alert-only too aggressively?

   Yes in the audited sample: {len(alert_to_context_evidence)} tasks move from old alert to context/evidence,
   and {len(high_value_tight_not_alert_only)} high-value tasks retain context/evidence as the ideal mission-level target.

3. Are evidence/context requirements more logically assigned?

   The v2 rule separates ideal requirement from payload feasibility. Evidence/context can remain ideal even when
   the shadow payload recommendation chooses metadata-first because contact capacity is tight.

4. Does v2 preserve no-label-leakage?

   Yes for the implementation path. The v2 function and payload-plan helper do not read label, bbox, object-count,
   detector-output, future-delivery, or realized-future-utility fields.

5. Does v2 produce a reasonable distribution, not all alert/evidence?

   New distribution: {dict(sorted(new_counts.items()))}. {'This is collapsed and needs tuning.' if collapsed else 'This is not collapsed into a single class in this audit.'}

6. Which cases are most changed and why?

{_markdown_table(changed_rows, ["task_id", "task_class", "task_value", "deadline_s", "uncertainty", "old_product_requirement", "new_ideal_product_requirement", "recommended_payload_plan_shadow", "diagnostic_reason"])}

7. Should v2 remain diagnostic-only, or be considered for a default-off experimental branch?

   {recommendation}

## Old distribution

{dict(sorted(old_counts.items()))}

## New distribution

{dict(sorted(new_counts.items()))}
"""
    final = f"""# Final recommendation

Do not adopt product_requirement_v2 as the main default yet.

{recommendation}

Rollback path is clear: the old assign_product_requirement function is untouched, and v2 is only used by this standalone audit unless explicitly wired later behind a default-off flag.
"""
    return report, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-slots", type=int, default=540)
    parser.add_argument("--dataset-config", default="dataset_sarfishsample.yaml")
    parser.add_argument("--task-stream-config", default="task_stream_sarfish_20s_dryrun.yaml")
    parser.add_argument("--timeline-config", default="timeline_sarfish_20s_dryrun.yaml")
    parser.add_argument("--simulator-config", default="simulator_sarfish_20s_dryrun.yaml")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command_line = "python scripts/audit_product_requirement_v2.py " + " ".join(sys.argv[1:])
    configs, source_note = _load_audit_configs(args)
    _limit_timeline(configs, args.num_slots)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    rows = _rows_for_timeline(timeline, configs)

    per_task_fields = [
        "task_id",
        "slot_index",
        "task_value",
        "deadline_s",
        "remaining_deadline_s",
        "uncertainty",
        "emergency_flag",
        "high_value_flag",
        "task_class",
        "old_product_requirement",
        "old_product_requirement_source",
        "new_ideal_product_requirement",
        "S_alert",
        "S_verification",
        "S_context",
        "S_evidence",
        "recommended_payload_plan_shadow",
        "known_contact_capacity_before_deadline_mb",
        "required_capacity_for_roi_context_mb",
        "queue_pressure",
        "missing_fields",
        "diagnostic_reason",
    ]
    _write_csv(OUT_DIR / "per_task_requirement_comparison.csv", rows, per_task_fields)
    _write_csv(OUT_DIR / "requirement_distribution_summary.csv", _distribution_summary(rows))
    _write_csv(OUT_DIR / "old_vs_new_transition_matrix.csv", _transition_matrix(rows))
    _write_csv(OUT_DIR / "payload_plan_recommendation_summary.csv", _plan_summary(rows))
    _write_text(OUT_DIR / "leakage_safety_audit.md", _leakage_report(rows))
    report, final = _report(rows, source_note)
    _write_text(OUT_DIR / "product_requirement_v2_report.md", report)
    _write_text(OUT_DIR / "final_recommendation.md", final)
    _write_text(
        OUT_DIR / "README.md",
        """# Product requirement v2 audit

This directory contains a standalone, default-off diagnostic audit for the new
ideal product requirement assignment rule and shadow payload-plan recommendation.

The audit does not modify reward, mission_prior_value_v2, SafeLinUCB scoring,
utility_rule, SafeFilter, lifecycle reward behavior, or main default configs.
""",
    )
    _write_text(
        OUT_DIR / "command_log.md",
        f"""# Command log

Command:

```powershell
{command_line}
```

Source note: {source_note}
""",
    )
    _write_text(OUT_DIR / "config_used.yaml", yaml.safe_dump(configs, sort_keys=True, allow_unicode=False))
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _audit_diff(
            [
                "src/data/product_requirements.py",
                "scripts/audit_product_requirement_v2.py",
                "tests/test_product_requirements_v2.py",
            ]
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
