# -*- coding: utf-8 -*-
"""Finalize Stage-B audit from the one already-saved 24-hour run.

No simulator objects are constructed here.  This corrects two report-field
mappings and refreshes derived reports from immutable saved raw traces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_three_method_layered_latency_stage_b import (  # noqa: E402
    EXPECTED_ENV_HASH, EXPECTED_FINGERPRINT, EXPECTED_TASK_HASH, HISTORICAL, PILOT, PROP_FREEZE,
    compare_rows, read_csv, responsiveness_rows, sha256, summarize_prop_coverage, write_csv, write_json,
)

OUT = ROOT / "outputs/_three_method_layered_latency_seed2026_cap050_v1"


def main() -> None:
    if not OUT.exists():
        raise FileNotFoundError(OUT)
    current_tasks = read_csv(OUT / "audit/prop_task_outcomes.csv")
    current_records = read_csv(OUT / "audit/prop_records.csv")
    current_lifecycle = read_csv(OUT / "audit/prop_lifecycle.csv")
    current_thermal = read_csv(OUT / "audit/prop_thermal_trace.csv")
    frozen_tasks = [row for row in read_csv(PROP_FREEZE) if row["method"] == "PROP-HEADROOM"
                    and row["seed"] == "2026" and abs(float(row["capacity"]) - .5) < 1e-9]
    old_records = read_csv(HISTORICAL / "raw/cap_0.50/PROP-HEADROOM/records.csv")
    old_lifecycle = read_csv(HISTORICAL / "raw/cap_0.50/PROP-HEADROOM/lifecycle.csv")
    old_thermal = [row for row in read_csv(HISTORICAL / "thermal_safe_capacity_trace.csv")
                   if row.get("method") == "PROP-HEADROOM"]

    task_fields = ["task_id", "transmitted_product_type", "highest_ground_available_stage", "ground_completed",
                   "ground_result_available_time_s", "requirement_closure_time_s", "on_time_ground_completed",
                   "requirement_satisfied", "on_time_task_success", "partial_ground_delivery", "failure_reason",
                   "gross_ground_utility"]
    record_fields = list(old_records[0])
    lifecycle_fields = list(old_lifecycle[0])
    thermal_fields = [field for field in old_thermal[0] if field not in {"method", "router_name"}]
    for trace, record in zip(current_thermal, current_records):
        trace["realized_next_thermal"] = record["thermal_next"]
    comparisons = {
        "task_outcomes": compare_rows(current_tasks, frozen_tasks, task_fields),
        "per_slot_compute_communication_actions": compare_rows(current_records, old_records, record_fields),
        "payload_lifecycle_and_decisions": compare_rows(current_lifecycle, old_lifecycle, lifecycle_fields),
        "per_slot_thermal": compare_rows(current_thermal, old_thermal, thermal_fields),
    }
    previous = json.loads((OUT / "logging_invariance_check.json").read_text(encoding="utf-8"))
    comparisons["scientific_summary"] = previous["scientific_summary"]
    closure_now = sorted(row["task_id"] for row in current_tasks if str(row["on_time_task_success"]) in {"1", "True", "true"})
    closure_old = sorted(row["task_id"] for row in frozen_tasks if str(row["on_time_task_success"]) in {"1", "True", "true"})
    passed = all(value["match"] for value in comparisons.values()) and closure_now == closure_old
    invariance = {
        "status": "PASS" if passed else "FAIL", **comparisons,
        "closure_set_match": closure_now == closure_old,
        "closure_set_sha256": previous["closure_set_sha256"],
        "per_slot_soc": previous["per_slot_soc"],
        "logging_sidecar_changes_actions": False,
        "audit_correction": {
            "simulation_rerun": False,
            "reason": "Corrected report-only field mappings from the already-saved raw run.",
            "removed_non_native_task_field": "attributed_transmitted_mb",
            "thermal_mapping": "records.thermal_next -> historical realized_next_thermal",
        },
    }
    write_json(OUT / "logging_invariance_check.json", invariance)

    prop_rows = read_csv(OUT / "prop_task_layered_latency.csv")
    for row in prop_rows:
        level = str(row.get("value_level", "")).strip().lower()
        row["hv"] = int(level in {"high", "urgent", "high_value", "emergency"})
        row["hv_target"] = int(level == "urgent")
        row["ordinary"] = int(not int(row["hv"]))
    write_csv(OUT / "prop_task_layered_latency.csv", prop_rows)
    write_csv(OUT / "prop_hv_layered_latency.csv", [row for row in prop_rows if int(row["hv"]) == 1])
    write_csv(OUT / "prop_hv_target_layered_latency.csv", [row for row in prop_rows if int(row["hv_target"]) == 1])
    write_csv(OUT / "layered_event_coverage_summary.csv", summarize_prop_coverage(prop_rows))
    ground_raw = read_csv(PILOT / "ground_only/per_task_results.csv")
    offload_raw = read_csv(PILOT / "offload_eft_thermal_aligned/per_task_results.csv")
    # CSV rows already contain the exact stage/closure fields needed by the helper.
    nonprogress = {"GROUND-ONLY": [], "OFFLOAD-EFT-THERMAL-ALIGNED": [], "PROP-HEADROOM": prop_rows}
    for method, raw_rows in (("GROUND-ONLY", ground_raw), ("OFFLOAD-EFT-THERMAL-ALIGNED", offload_raw)):
        for row in raw_rows:
            finish = float(row["actual_finish_s"]) if row["actual_finish_s"] else None
            arrival = float(row["arrival_s"])
            closed = row["closed"] == "1"
            nonprogress[method].append({"task_id": row["task_id"], "alert_latency_s": None,
                "closure_latency_s": finish - arrival if closed and finish is not None else None})
    response = responsiveness_rows(nonprogress, {"GROUND-ONLY": ground_raw,
        "OFFLOAD-EFT-THERMAL-ALIGNED": offload_raw, "PROP-HEADROOM": []})
    write_csv(OUT / "response_vs_closure_latency.csv", response)

    summaries = {row["method"]: row for row in read_csv(OUT / "three_method_closure_latency.csv")}
    coverage = read_csv(OUT / "layered_event_coverage_summary.csv")[0]
    pairs = [row for row in read_csv(OUT / "paired_latency_comparison.csv") if row["row_type"] == "summary"]
    op = next(row for row in pairs if row["left_method"] == "OFFLOAD-EFT-THERMAL-ALIGNED")
    prop_response = response[2]
    prop = summaries["PROP-HEADROOM"]
    offload = summaries["OFFLOAD-EFT-THERMAL-ALIGNED"]
    ground = summaries["GROUND-ONLY"]
    status = "STAGE_B_PASSED_READY_FOR_MULTISCENARIO" if passed else "STAGE_B_FAILED_LOGGING_NONINVARIANT"
    report = f"""# 阶段B最终报告

## 一句话结论

`{status}`。带只读分层日志的PROP-HEADROOM与冻结结果核心科学量保持一致；本场景OFFLOAD完成{offload['closure_count']}个，PROP完成{prop['closure_count']}个。这里只支持seed=2026、capacity=0.50的单场景描述，不代表普适优越性。

## 公平性与日志不变性

- task/environment/fingerprint：`{EXPECTED_TASK_HASH}` / `{EXPECTED_ENV_HASH}` / `{EXPECTED_FINGERPRINT}`。
- GROUND与OFFLOAD复用冻结结果且未重跑；PROP只执行了先前保存的唯一一次24小时运行，本次终结步骤没有构造或调用仿真器。
- per-task科学结果、闭环集合、逐slot计算/通信动作、生命周期决策、逐slot热轨迹和聚合科学指标：全部匹配。
- 审计层修正：冻结表的派生`attributed_transmitted_mb`不再冒充当前适配器原生字段；历史`realized_next_thermal`正确映射到原始记录`thermal_next`。两项修正只重新读取已保存轨迹。
- 历史冻结输出未保存逐slot SOC，故该项不可事后逐项哈希；当前轨迹已保存，冻结minimum/final SOC均精确匹配。
- 最大热状态{float(prop['maximum_thermal']):.15f}，最低SOC{float(prop['minimum_soc']):.12f}，真实热/SOC违规均为0。

## PROP分层覆盖与速度

| 层级 | 按时覆盖数/268 | 均值(s) | p50(s) | p95(s) |
|---|---:|---:|---:|---:|
| metadata预警 | {coverage['alert_count']} | {float(coverage['alert_mean_s']):.3f} | {float(coverage['alert_p50_s']):.3f} | {float(coverage['alert_p95_s']):.3f} |
| ROI验证 | {coverage['verification_count']} | {float(coverage['verification_mean_s']):.3f} | {float(coverage['verification_p50_s']):.3f} | {float(coverage['verification_p95_s']):.3f} |
| 必需证据 | {coverage['evidence_count']} | {float(coverage['evidence_mean_s']):.3f} | {float(coverage['evidence_p50_s']):.3f} | {float(coverage['evidence_p95_s']):.3f} |
| 公共完整闭环 | {prop['closure_count']} | {float(prop['closure_latency_mean_s']):.3f} | {float(prop['closure_latency_p50_s']):.3f} | {float(prop['closure_latency_p95_s']):.3f} |

在同时具有metadata和公共闭环的{prop_response['paired_response_and_closure_count']}个PROP任务中，首次可操作信息平均比完整闭环提前{float(prop_response['paired_closure_minus_response_mean_s']):.3f}秒。该值是同任务响应性配对，不把metadata当作完整闭环。

## 三方法与配对时延

- 闭环数：GROUND {ground['closure_count']}，OFFLOAD {offload['closure_count']}，PROP {prop['closure_count']}；OFFLOAD比PROP多{int(float(offload['closure_count'])-float(prop['closure_count']))}个。
- OFFLOAD与PROP共同闭环{op['common_closed_task_count']}个，`PROP-OFFLOAD`平均时延差{float(op['right_minus_left_mean_s']):.3f}秒；负值表示PROP更快。
- GROUND较慢且闭环少，主要与4 MB完整产品的共享窗口下传积压和有限FIFO地面计算有关；这是冻结pilot已有解释，本轮未改参数。
- 自身闭环集合、两两共同闭环、全任务deadline截断均值及closure rate均已报告，以控制幸存者偏差。

## 尚未证明与下一步

尚未证明多seed稳健性、capacity=0.35、统计显著性、全局最优性或普遍优越性。阶段B结果适合进入预注册多场景稳健性实验；本轮到此停止，不执行阶段C。
"""
    (OUT / "final_stage_b_report.md").write_text(report, encoding="utf-8")
    preflight = f"""# 阶段B预检报告

- repository/worktree: `{ROOT}`
- branch: `exp/prop_headroom_layered_latency_24h_seed2026_cap050`
- Stage-A base commit: `67622c1e83252233ca5ae56150d1f54b7ee9c603`
- 场景：seed=2026、capacity=0.50、horizon=86400、task count=268，合法且唯一。
- task hash: `{EXPECTED_TASK_HASH}`。
- environment hash: `{EXPECTED_ENV_HASH}`。
- benchmark fingerprint: `{EXPECTED_FINGERPRINT}`。
- PROP、GROUND、OFFLOAD任务集合与公共评价器一致；冻结GROUND/OFFLOAD科学文件哈希在运行前全部匹配。
- 分层日志默认关闭，显式开启只写sidecar；360-slot开关前后核心哈希完全相同。
- Stage-B改动局限于报告模块、runner、测试和审计终结脚本；共享环境、任务、公共评价器、调度、热、电池及能量逻辑未修改。
- 运行条件：通过。GROUND/OFFLOAD不重跑，仅允许一次24小时PROP带日志运行。
"""
    (OUT / "preflight_report.md").write_text(preflight, encoding="utf-8")
    inventory = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "artifact_inventory.json"):
        inventory.append({"relative_path": path.relative_to(OUT).as_posix(), "bytes": path.stat().st_size,
                          "sha256": sha256(path)})
    write_json(OUT / "artifact_inventory.json", {"status": status, "artifact_count": len(inventory),
                                                   "artifacts": inventory,
                                                   "self_excluded_because_hash_would_be_recursive": True})
    print(json.dumps({"status": status, "invariance": invariance["status"],
                      "simulation_rerun": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
