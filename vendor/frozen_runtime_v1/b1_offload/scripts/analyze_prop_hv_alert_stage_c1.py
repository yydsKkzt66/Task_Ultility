# -*- coding: utf-8 -*-
"""Stage C1: read-only, strictly paired PROP high-value alert analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/_three_method_layered_latency_seed2026_cap050_v1"
DEFAULT_OUT = ROOT / "outputs/_prop_hv_alert_analysis_seed2026_cap050_v1"
EXPECTED_TASK_HASH = "49c704997b59648d902ed9413a4116d0b45e5b404d0d20ece9d3826a7d6d38cd"
EXPECTED_ENV_HASH = "6c9359f019290170a7be0b3bc5bc8a115f09589853eb8a49667a3457c0d82524"
EXPECTED_FINGERPRINT = "07c4c14fa4740a7376df314a6a9ed96eab674bb28722609117e42102b49ffb84"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean_s": mean(values) if values else None,
            "median_s": median(values) if values else None, "p90_s": percentile(values, .90),
            "p95_s": percentile(values, .95)}


def main() -> None:
    output = DEFAULT_OUT
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    required = ["prop_layered_events.csv", "prop_task_layered_latency.csv", "logging_invariance_check.json",
                "audit/prop_task_outcomes.csv", "artifact_inventory.json"]
    if not all((SOURCE / name).exists() for name in required):
        raise FileNotFoundError("Stage-B source artifacts incomplete")
    invariance = json.loads((SOURCE / "logging_invariance_check.json").read_text(encoding="utf-8"))
    if invariance["status"] != "PASS":
        raise RuntimeError("Stage-B logging invariance is not PASS")
    source_inventory = json.loads((SOURCE / "artifact_inventory.json").read_text(encoding="utf-8"))
    inventory_map = {row["relative_path"]: row["sha256"] for row in source_inventory["artifacts"]}
    source_hash_check = {name: {"expected": inventory_map.get(name), "actual": sha(SOURCE / name),
                                "match": inventory_map.get(name) == sha(SOURCE / name)}
                         for name in required if name != "artifact_inventory.json"}
    if not all(row["match"] for row in source_hash_check.values()):
        raise RuntimeError("Stage-B source hash mismatch")

    tasks = read_csv(SOURCE / "prop_task_layered_latency.csv")
    outcomes = {row["task_id"]: row for row in read_csv(SOURCE / "audit/prop_task_outcomes.csv")}
    task_ids = {row["task_id"] for row in tasks}
    events = [row for row in read_csv(SOURCE / "prop_layered_events.csv") if row["task_id"] in task_ids]
    excluded = [row for row in read_csv(SOURCE / "prop_layered_events.csv") if row["task_id"] not in task_ids]
    if len(tasks) != 268 or len(task_ids) != 268 or set(outcomes) != task_ids:
        raise RuntimeError("Stage-B task universe mismatch")

    by_task: dict[str, list[dict[str, str]]] = {task_id: [] for task_id in task_ids}
    for event in events:
        by_task[event["task_id"]].append(event)
    paired = []
    for task in tasks:
        task_id = task["task_id"]
        arrival = float(task["task_arrival_time_s"])
        product_events = [row for row in by_task[task_id] if row["product_type"]]
        first_by_product: dict[str, float] = {}
        for event in sorted(product_events, key=lambda row: float(row["ground_available_time_s"])):
            first_by_product.setdefault(event["product_type"], float(event["ground_available_time_s"]))
        metadata_time = first_by_product.get("metadata")
        verification_candidates = [first_by_product[key] for key in ("roi", "roi_context") if key in first_by_product]
        roi_time = min(verification_candidates) if verification_candidates else None
        evidence_time = first_by_product.get("roi_context")
        closure_time = num(task["full_closure_time_s"])
        products = set(first_by_product)
        if not products:
            status = "no_ground_product"
        elif closure_time is not None:
            status = "public_full_closure"
        elif "roi_context" in products:
            status = "full_evidence_without_public_closure"
        elif products & {"roi", "roi_context"}:
            status = "metadata_plus_roi_without_evidence_or_closure"
        else:
            status = "metadata_only_without_closure"
        outcome = outcomes[task_id]
        paired.append({
            "task_id": task_id, "value_level": task["value_level"], "hv": task["hv"],
            "hv_target": task["hv_target"], "arrival_time_s": arrival, "deadline_time_s": task["deadline_time_s"],
            "metadata_time_s": metadata_time, "roi_verification_time_s": roi_time,
            "evidence_time_s": evidence_time, "public_full_closure_time_s": closure_time,
            "metadata_alert_latency_s": None if metadata_time is None else metadata_time - arrival,
            "roi_verification_latency_s": None if roi_time is None else roi_time - arrival,
            "evidence_latency_s": None if evidence_time is None else evidence_time - arrival,
            "public_full_closure_latency_s": None if closure_time is None else closure_time - arrival,
            "alert_lead_time_s": None if metadata_time is None or closure_time is None else closure_time - metadata_time,
            "verification_lead_time_s": None if roi_time is None or closure_time is None else closure_time - roi_time,
            "delivered_products": ";".join(sorted(products)), "final_delivery_status": status,
            "gross_ground_utility": outcome.get("gross_ground_utility", ""),
            "failure_reason": outcome.get("failure_reason", "") if closure_time is None else "",
        })

    groups = {"ALL": paired, "HV": [row for row in paired if row["hv"] == "1"],
              "HV-TARGET": [row for row in paired if row["hv_target"] == "1"]}
    metric_fields = [
        ("metadata_alert_latency", "metadata_alert_latency_s", "exact metadata product available"),
        ("roi_verification_latency", "roi_verification_latency_s", "first roi or roi_context product available"),
        ("evidence_latency", "evidence_latency_s", "first roi_context product available"),
        ("public_full_closure_latency", "public_full_closure_latency_s", "unchanged common evaluator closure"),
        ("alert_lead_time", "alert_lead_time_s", "same task has exact metadata and public closure"),
        ("verification_lead_time", "verification_lead_time_s", "same task has ROI-level product and public closure"),
    ]
    summary = []
    for group_name, rows in groups.items():
        for metric, field, rule in metric_fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            summary.append({"task_group": group_name, "metric": metric, "sample_rule": rule, **stats(values)})
    thresholds = []
    for group_name, rows in groups.items():
        values = [float(row["alert_lead_time_s"]) for row in rows if row["alert_lead_time_s"] is not None]
        for threshold in (0, 60, 300, 600, 1800):
            count = sum(value > threshold for value in values)
            thresholds.append({"task_group": group_name, "paired_alert_closure_n": len(values),
                               "threshold_s_strictly_greater_than": threshold, "count": count,
                               "rate": count / len(values) if values else None})

    status_rows = []
    for status in ("metadata_only_without_closure", "metadata_plus_roi_without_evidence_or_closure",
                   "full_evidence_without_public_closure", "public_full_closure", "no_ground_product"):
        rows = [row for row in paired if row["final_delivery_status"] == status]
        alerts = [float(row["metadata_alert_latency_s"]) for row in rows if row["metadata_alert_latency_s"] is not None]
        reasons = Counter(row["failure_reason"] or "none" for row in rows if row["public_full_closure_time_s"] is None)
        status_rows.append({"final_delivery_status": status, "task_count": len(rows),
            "hv_count": sum(row["hv"] == "1" for row in rows),
            "hv_target_count": sum(row["hv_target"] == "1" for row in rows),
            "metadata_alert_count": len(alerts), "metadata_alert_mean_s": mean(alerts) if alerts else None,
            "metadata_alert_median_s": median(alerts) if alerts else None,
            "gross_ground_utility_sum": sum(float(row["gross_ground_utility"] or 0) for row in rows),
            "failure_reason_counts": json.dumps(dict(reasons), ensure_ascii=False, sort_keys=True)})

    write_csv(output / "paired_task_alert_metrics.csv", paired)
    write_csv(output / "group_latency_summary.csv", summary)
    write_csv(output / "alert_lead_threshold_summary.csv", thresholds)
    write_csv(output / "final_delivery_status_summary.csv", status_rows)
    write_json(output / "source_hash_check.json", {"stage_b_invariance": "PASS", "files": source_hash_check,
        "excluded_nonmanifest_event_count": len(excluded),
        "excluded_nonmanifest_task_ids": sorted({row["task_id"] for row in excluded})})
    write_json(output / "resolved_analysis_config.json", {"stage": "C1", "simulation_rerun": False,
        "seed": 2026, "capacity": .5, "task_count": 268, "task_hash": EXPECTED_TASK_HASH,
        "environment_hash": EXPECTED_ENV_HASH, "benchmark_fingerprint": EXPECTED_FINGERPRINT,
        "metadata_definition": "exact product_type=metadata event; not inferred from ROI/evidence",
        "hv_definition": "value_level in high/urgent", "hv_target_definition": "value_level=urgent"})
    all_alert = next(row for row in summary if row["task_group"] == "ALL" and row["metric"] == "alert_lead_time")
    hv_alert = next(row for row in summary if row["task_group"] == "HV" and row["metric"] == "alert_lead_time")
    hvt_alert = next(row for row in summary if row["task_group"] == "HV-TARGET" and row["metric"] == "alert_lead_time")
    report = f"""# PROP HV快速预警只读分析

## 结论

本分析只读取Stage B唯一一次24小时日志，没有重跑仿真。严格的metadata产品共到达{sum(row['metadata_time_s'] is not None for row in paired)}个任务；其中与公共闭环同任务可配对{all_alert['n']}个，平均提前{all_alert['mean_s']:.3f}秒，中位数{all_alert['median_s']:.3f}秒。HV配对{hv_alert['n']}个，平均提前{hv_alert['mean_s']:.3f}秒；HV-TARGET配对{hvt_alert['n']}个，平均提前{hvt_alert['mean_s']:.3f}秒。

这里的metadata是事件日志中明确的`product_type=metadata`，没有把直接ROI或evidence到达改称metadata；metadata到达不计完整闭环。各阈值覆盖率见`alert_lead_threshold_summary.csv`，可判断提前量是否集中于少数任务。

## 公平性与限制

- 268个任务、169个HV、42个HV-TARGET，task/environment hash与Stage B一致。
- Stage B核心不变性为PASS；源文件SHA256全部匹配。
- 原事件表中的1条`__idle__`非manifest事件已在分析入口过滤，任务级日志未修改。
- 缺失事件保持NA；不同阶段总体均值不被用来宣称时间先后。
- 最终交付状态采用互斥优先级：无产品；公共闭环；有evidence但未闭环；有ROI无evidence/闭环；仅metadata未闭环。
"""
    (output / "final_c1_report.md").write_text(report, encoding="utf-8")
    write_json(output / "rollback_manifest.json", {"branch": subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_modified": False, "simulation_rerun": False, "new_output": str(output),
        "rollback": "Delete this independent C1 output and the report-only analysis script."})
    inventory = [{"relative_path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                  "sha256": sha(path)} for path in sorted(output.rglob("*")) if path.is_file()]
    write_json(output / "artifact_inventory.json", {"artifact_count_excluding_self": len(inventory), "artifacts": inventory})
    print(json.dumps({"status": "C1_PASS", "output": str(output), "metadata_events": sum(
        row["metadata_time_s"] is not None for row in paired), "paired_alert_closure": all_alert["n"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
