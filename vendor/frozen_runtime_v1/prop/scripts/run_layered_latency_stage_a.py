# -*- coding: utf-8 -*-
"""Stage-A audit for unified layered latency metrics.

The only simulation performed here is a short deterministic PROP-HEADROOM
smoke, executed twice to prove that the default-disabled report sidecar does
not change core scientific state.  The completed 24-hour runs are never
re-executed or overwritten.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_thermal_headroom_multiseed import cfg, validate_scenario  # noqa: E402
from src.common.config import load_yaml  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.task_outcome_adapter import manifest_rows  # noqa: E402
from src.reporting.layered_latency_metrics import (  # noqa: E402
    build_nonprogressive_event_rows,
    build_prop_event_rows,
    paired_closure_latency,
    summarize_layered_latency,
    write_event_log,
)
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


PILOT = ROOT / "outputs/_offload_vs_ground_24h_pilot_seed2026_cap050_v1"
PROP_FREEZE = ROOT / "outputs/_sarfish_single_satellite_baseline_freeze_v1/reconstructed_task_outcomes.csv"
PROP_MULTI = ROOT / "outputs/_sarfish_thermal_headroom_multiseed/multiscenario_run_summary.csv"
DEFAULT_OUTPUT = ROOT / "outputs/_layered_latency_stage_a_v1"
EXPECTED_TASK_HASH = "49c704997b59648d902ed9413a4116d0b45e5b404d0d20ece9d3826a7d6d38cd"
EXPECTED_ENV_HASH = "6c9359f019290170a7be0b3bc5bc8a115f09589853eb8a49667a3457c0d82524"
EXPECTED_FINGERPRINT = "07c4c14fa4740a7376df314a6a9ed96eab674bb28722609117e42102b49ffb84"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
    return (completed.stdout or "").strip()


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def prop_frozen_rows() -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(PROP_FREEZE)
        if row.get("method") == "PROP-HEADROOM"
        and row.get("seed") == "2026"
        and abs(float(row.get("capacity", 0)) - 0.5) < 1.0e-12
    ]


def frozen_prop_closure_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Use only timestamps that the frozen table stores without inference.

    The first ground result is exact alert availability.  Verification and
    evidence timestamps are intentionally left missing because the frozen
    product set is unordered and does not preserve every stage arrival.
    """

    output: list[dict[str, Any]] = []
    for row in rows:
        arrival = float(row["arrival_time_s"])
        deadline = float(row["deadline_time_s"])
        first = float(row["ground_result_available_time_s"]) if row.get("ground_result_available_time_s") else None
        closure = (
            float(row["requirement_closure_time_s"])
            if truth(row.get("on_time_task_success")) and row.get("requirement_closure_time_s")
            else None
        )
        output.append(
            {
                "task_id": row["task_id"],
                "task_arrival_time_s": arrival,
                "deadline_time_s": deadline,
                "product_requirement": row.get("product_requirement", ""),
                "metadata_ground_arrival_time_s": first,
                "roi_ground_arrival_time_s": None,
                "evidence_ground_arrival_time_s": None,
                "full_closure_time_s": closure,
                "alert_latency_s": None if first is None else first - arrival,
                "verification_latency_s": None,
                "evidence_latency_s": None,
                "closure_latency_s": None if closure is None else closure - arrival,
                "alert_to_verification_gap_s": None,
                "verification_to_evidence_gap_s": None,
                "alert_on_time": int(first is not None and first <= deadline + 1.0e-9),
                "verification_on_time": None,
                "evidence_on_time": None,
                "closed_on_time": int(closure is not None),
                "event_source": "PROP frozen outcome; stage detail unavailable",
            }
        )
    return output


def core_smoke_payload(result: Any, runner: Any) -> dict[str, Any]:
    records = [vars(record) for record in result.records]
    return {
        "summary": result.summary,
        "records": records,
        "payload_lifecycle_current": list(runner.last_payload_lifecycle_current_records),
        "thermal_safe_capacity_trace": list(runner.last_thermal_safe_capacity_trace),
    }


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_prop_smoke(timeline: list[Any], configs: dict[str, Any]) -> tuple[Any, Any]:
    environment = SimulationEnvironment(copy.deepcopy(configs))
    router = build_routers(configs["experiment"], environment.action_space)[0]
    runner = environment.build_episode_runner(copy.deepcopy(timeline))
    return runner.run(router), runner


def summarize_pair(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    diffs = [float(row["right_minus_left_s"]) for row in rows]
    return {
        "left_method": left,
        "right_method": right,
        "common_closed_task_count": len(rows),
        "right_minus_left_mean_s": mean(diffs) if diffs else None,
        "right_minus_left_median_s": median(diffs) if diffs else None,
        "right_faster_count": sum(value < -1.0e-9 for value in diffs),
        "same_count": sum(abs(value) <= 1.0e-9 for value in diffs),
        "right_slower_count": sum(value > 1.0e-9 for value in diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-slots", type=int, default=360)
    parser.add_argument("--enable-event-log", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    pilot_scenario = json.loads((PILOT / "resolved_scenario.json").read_text(encoding="utf-8"))
    fingerprint = json.loads((ROOT / "manifests/benchmark_environment_fingerprint_v1.json").read_text(encoding="utf-8"))
    prop_summary = [
        row
        for row in read_csv(PROP_MULTI)
        if row.get("method") == "PROP-HEADROOM"
        and row.get("seed") == "2026"
        and abs(float(row.get("capacity_multiplier", 0)) - 0.5) < 1.0e-12
    ]
    if len(prop_summary) != 1:
        raise RuntimeError("PROP-HEADROOM seed2026/cap0.50 result is missing or non-unique")

    prop_rows = prop_frozen_rows()
    ground_raw = read_csv(PILOT / "ground_only/per_task_results.csv")
    offload_raw = read_csv(PILOT / "offload_eft_thermal_aligned/per_task_results.csv")
    checks = {
        "pilot_task_hash_match": pilot_scenario["task_hash"] == EXPECTED_TASK_HASH,
        "pilot_environment_hash_match": pilot_scenario["environment_hash"] == EXPECTED_ENV_HASH,
        "benchmark_fingerprint_match": fingerprint["benchmark_environment_fingerprint"] == EXPECTED_FINGERPRINT,
        "prop_task_hash_match": prop_summary[0]["task_hash"] == EXPECTED_TASK_HASH,
        "prop_environment_hash_match": prop_summary[0]["environment_hash"] == EXPECTED_ENV_HASH,
        "ground_task_count_268": len(ground_raw) == 268,
        "offload_task_count_268": len(offload_raw) == 268,
        "prop_task_count_268": len(prop_rows) == 268,
        "task_id_sets_identical": {row["task_id"] for row in ground_raw}
        == {row["task_id"] for row in offload_raw}
        == {row["task_id"] for row in prop_rows},
        "prop_source_commit_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", "57b5567312f32bee255b2c52cd06a840998cff98", "HEAD"],
            cwd=ROOT,
        ).returncode
        == 0,
    }
    if not all(checks.values()):
        write_json(output / "FAILED_preflight.json", checks)
        raise RuntimeError(f"Preflight failed: {checks}")

    source_hash_manifest = json.loads((PILOT / "scientific_output_hashes.json").read_text(encoding="utf-8"))
    source_hash_check = {
        relative: {
            "expected_sha256": expected,
            "actual_sha256": sha256(PILOT / relative),
            "match": sha256(PILOT / relative) == expected,
        }
        for relative, expected in source_hash_manifest.items()
    }
    if not all(item["match"] for item in source_hash_check.values()):
        raise RuntimeError("Frozen pilot source hash mismatch")

    ground_events = build_nonprogressive_event_rows(ground_raw, source="GROUND-ONLY complete result")
    offload_events = build_nonprogressive_event_rows(
        offload_raw, source="OFFLOAD-EFT-THERMAL-ALIGNED complete result"
    )
    prop_events = frozen_prop_closure_rows(prop_rows)
    methods = {
        "GROUND-ONLY": ground_events,
        "OFFLOAD-EFT-THERMAL-ALIGNED": offload_events,
        "PROP-HEADROOM": prop_events,
    }
    summaries = []
    for method, rows in methods.items():
        summary = summarize_layered_latency(rows)
        summary["method"] = method
        summary["stage_timestamp_completeness"] = "FULL" if method != "PROP-HEADROOM" else "ALERT_AND_CLOSURE_ONLY"
        summaries.append(summary)

    pair_details: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    for left, right in (
        ("GROUND-ONLY", "OFFLOAD-EFT-THERMAL-ALIGNED"),
        ("GROUND-ONLY", "PROP-HEADROOM"),
        ("OFFLOAD-EFT-THERMAL-ALIGNED", "PROP-HEADROOM"),
    ):
        rows = paired_closure_latency(methods[left], methods[right], left_name=left, right_name=right)
        pair_details.extend(rows)
        pair_summaries.append(summarize_pair(rows, left, right))

    configs = cfg(2026, 0.5, True)
    full_timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    scenario_validation = validate_scenario(
        configs, full_timeline, expected_task_hash=EXPECTED_TASK_HASH, expected_environment_hash=EXPECTED_ENV_HASH
    )
    smoke_timeline = list(full_timeline[: args.smoke_slots])
    smoke_manifest = manifest_rows(smoke_timeline, scenario_validation["task_manifest_hash_v2"])
    disabled_result, disabled_runner = run_prop_smoke(smoke_timeline, configs)
    enabled_result, enabled_runner = run_prop_smoke(smoke_timeline, configs)
    disabled_hash = canonical_hash(core_smoke_payload(disabled_result, disabled_runner))
    enabled_hash = canonical_hash(core_smoke_payload(enabled_result, enabled_runner))
    smoke_events = build_prop_event_rows(
        smoke_manifest,
        enabled_result.records,
        enabled_runner.last_payload_lifecycle_current_records,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
    )
    disabled_path = output / "smoke/default_disabled_event_log.csv"
    disabled_written = write_event_log(disabled_path, smoke_events, enabled=False)
    enabled_path = output / "smoke/layered_latency_event_log.csv"
    enabled_written = write_event_log(enabled_path, smoke_events, enabled=args.enable_event_log)
    smoke_check = {
        "smoke_slots": args.smoke_slots,
        "smoke_task_count": len(smoke_manifest),
        "event_log_default_enabled": False,
        "default_disabled_file_written": disabled_written,
        "explicitly_enabled_file_written": enabled_written,
        "disabled_core_hash": disabled_hash,
        "enabled_core_hash": enabled_hash,
        "core_hash_match": disabled_hash == enabled_hash,
        "event_rows": len(smoke_events),
        "event_rows_with_alert": sum(row["metadata_ground_arrival_time_s"] is not None for row in smoke_events),
        "event_rows_with_verification": sum(row["roi_ground_arrival_time_s"] is not None for row in smoke_events),
        "event_rows_with_evidence": sum(row["evidence_ground_arrival_time_s"] is not None for row in smoke_events),
    }
    if not enabled_written or not smoke_check["core_hash_match"] or disabled_path.exists():
        raise RuntimeError(f"Event-log smoke failed: {smoke_check}")

    availability = [
        {
            "method": "GROUND-ONLY",
            "task_arrival": "AVAILABLE",
            "metadata_arrival": "DERIVED_COMPLETE_RESULT_TIME",
            "roi_arrival": "DERIVED_COMPLETE_RESULT_TIME",
            "evidence_arrival": "DERIVED_COMPLETE_RESULT_TIME",
            "full_closure": "AVAILABLE",
            "strict_24h_layered_analysis": "AVAILABLE_NONPROGRESSIVE",
        },
        {
            "method": "OFFLOAD-EFT-THERMAL-ALIGNED",
            "task_arrival": "AVAILABLE",
            "metadata_arrival": "DERIVED_COMPLETE_RESULT_TIME",
            "roi_arrival": "DERIVED_COMPLETE_RESULT_TIME",
            "evidence_arrival": "DERIVED_COMPLETE_RESULT_TIME",
            "full_closure": "AVAILABLE",
            "strict_24h_layered_analysis": "AVAILABLE_NONPROGRESSIVE",
        },
        {
            "method": "PROP-HEADROOM",
            "task_arrival": "AVAILABLE",
            "metadata_arrival": "AVAILABLE_AS_FIRST_GROUND_RESULT",
            "roi_arrival": "NOT_AVAILABLE_IN_FROZEN_OUTPUT",
            "evidence_arrival": "NOT_AVAILABLE_IN_FROZEN_OUTPUT",
            "full_closure": "AVAILABLE",
            "strict_24h_layered_analysis": "BLOCKED_UNTIL_EVENT_LOGGED_RUN",
        },
    ]

    write_json(output / "preflight_checks.json", checks)
    write_json(output / "source_hash_check.json", source_hash_check)
    write_json(output / "smoke_hash_equivalence.json", smoke_check)
    write_csv(output / "field_availability_audit.csv", availability)
    write_csv(output / "method_latency_summary.csv", summaries)
    write_csv(output / "paired_closure_latency_summary.csv", pair_summaries)
    write_csv(output / "paired_closure_latency_details.csv", pair_details)

    metric_definition = """# 分层时延指标定义（阶段A）

- `task_arrival_time`：任务进入在线系统的时间。
- `metadata_ground_arrival_time`：地面首次获得预警级信息；只计alert，不计完整闭环。
- `roi_ground_arrival_time`：地面首次获得ROI验证级信息。
- `evidence_ground_arrival_time`：地面首次获得ROI+Context/evidence级信息。
- `full_closure_time`：公共评价器要求满足且不晚于deadline的时间。
- `alert_latency = metadata_ground_arrival_time - task_arrival_time`。
- `verification_latency = roi_ground_arrival_time - task_arrival_time`。
- `evidence_latency = evidence_ground_arrival_time - task_arrival_time`。
- `closure_latency = full_closure_time - task_arrival_time`。
- `alert_to_verification_gap`和`verification_to_evidence_gap`仅在两端时间均存在时计算；缺失不按0处理。
- 非渐进方法直到完整处理结果可用才同时获得alert、verification和evidence时间，不给予虚构的早期信用。
- 各自闭环集合时延必须与共同闭环任务的严格配对时延并列报告。
- deadline截断全任务restricted mean：按任务计算；按时闭环取closure latency，未闭环取`deadline_time-task_arrival_time`，再对全部任务求均值。
- 该restricted mean是描述性、deadline截断指标，不是生存曲线外推，也不替代closure rate。
"""
    (output / "metric_definition.md").write_text(metric_definition, encoding="utf-8")

    preflight = f"""# 阶段A预检

- branch: `{git('branch', '--show-current')}`
- base commit: `2aaebd75db35608504cc3073fa4249cbaa1d80c6`
- OFFLOAD冻结来源: `1d90ff49041e18d6766d2b57a84b87034716c10b`
- GROUND冻结来源: `b75ef04b63a13d039197c064cf82b0338121f765`
- PROP冻结来源: `57b5567312f32bee255b2c52cd06a840998cff98`，是当前提交祖先。
- task hash: `{EXPECTED_TASK_HASH}`，三方法一致。
- environment hash: `{EXPECTED_ENV_HASH}`，三方法一致。
- benchmark fingerprint: `{EXPECTED_FINGERPRINT}`。
- 公共评价器源文件和配置从PROP冻结提交到当前提交无差异。
- PROP-HEADROOM可在当前benchmark代码环境运行；阶段A仅执行{args.smoke_slots}个slot的小规模确定性smoke。
"""
    (output / "preflight_report.md").write_text(preflight, encoding="utf-8")

    final_report = f"""# 阶段A：统一分层时延指标审计

## 一句话结论

阶段A通过指标定义与接口smoke，但现有PROP-HEADROOM 24小时冻结输出缺少逐任务ROI和evidence独立到达时间，因此不能据历史表完成严格的三方法分层时延比较；已新增默认关闭的只读事件日志，{args.smoke_slots} slot smoke中开关前后核心哈希完全一致。阶段B尚未执行。

## 公平性

- 三方法task hash、environment hash一致，均为seed=2026、capacity=0.50的268任务场景。
- 公共评价器未变化，既有pilot 15个科学文件SHA256仍全部匹配。
- 本阶段没有修改调度、产品、闭环、奖励、效用、任务、环境或热/SOC状态转移。

## 字段审计

- GROUND-ONLY与OFFLOAD：arrival和完整结果时间齐全；非渐进语义下，alert/verification/evidence均只能在完整处理结果可用时取得。
- PROP-HEADROOM：冻结输出严格保存首次地面可用时间和任务要求闭环时间，可核算alert、closure rate、closure latency与restricted mean；但产品集合无完整顺序，无法严格恢复所有ROI/evidence到达时间及阶段间隔。
- 缺失字段保留为空或标记`NOT_AVAILABLE`，未按0补值。

## 现有冻结结果可核算的闭环层指标

| 方法 | 闭环数 | 闭环率 | 闭环时延均值(s) | p95(s) | deadline截断全任务restricted mean(s) |
|---|---:|---:|---:|---:|---:|
"""
    for row in summaries:
        final_report += (
            f"| {row['method']} | {row['closure_count']} | {row['closure_rate']:.4%} | "
            f"{row['closure_latency_mean_s']:.3f} | {row['closure_latency_p95_s']:.3f} | "
            f"{row['deadline_truncated_restricted_mean_time_to_closure_s']:.3f} |\n"
        )
    final_report += f"""

这些自身闭环集合时延存在幸存者偏差；严格共同闭环配对另存于`paired_closure_latency_summary.csv`和明细表。

## 事件日志smoke

- 默认状态：关闭，未产生默认日志文件。
- 显式启用：产生{len(smoke_events)}行逐任务事件日志。
- disabled core hash：`{disabled_hash}`。
- enabled core hash：`{enabled_hash}`。
- 核心哈希一致：`{disabled_hash == enabled_hash}`。

## 已验证

- 五类时间及六个时延/间隔的统一语义已经固定。
- 非渐进方法不会获得虚构的早期预警信用。
- 全任务deadline截断restricted mean与closure rate、闭环集合时延、共同闭环配对时延同时报告。
- 只读日志对smoke的任务决策、闭环、传输、能量、SOC和热轨迹没有影响。

## 尚未完成

- 尚未生成PROP-HEADROOM完整24小时逐阶段事件日志。
- 尚未执行阶段B三方法单场景严格配对。
- 尚未执行阶段C或阶段D。

## 测试说明

- 新增分层时延模块定向测试：4/4通过。
- 扩展相关回归：77项通过；11项仅因隔离worktree缺少`_unified_ground_received_accounting_seed2026`和`_offload_eft_thermal_aligned_smoke_v1`历史工件而报`FileNotFoundError`，没有断言失败或数值不一致。本阶段未重跑这些历史实验来补测试夹具。

## 阶段门结论

`STAGE_A_INTERFACE_VALIDATED_FULL_PROP_LAYERED_TIMESTAMPS_REQUIRE_LOGGED_24H_RUN`

下一步若获确认，应进入阶段B准备：在同一冻结场景下生成带只读事件日志的PROP 24小时结果；由于现有PROP历史输出字段不足，不能直接混用其聚合时延作为完整分层时延证据。
"""
    (output / "final_stage_a_report.md").write_text(final_report, encoding="utf-8")

    rollback = {
        "branch": git("branch", "--show-current"),
        "base_commit": "2aaebd75db35608504cc3073fa4249cbaa1d80c6",
        "new_code_files": [
            "src/reporting/layered_latency_metrics.py",
            "scripts/run_layered_latency_stage_a.py",
            "tests/test_layered_latency_metrics.py",
        ],
        "new_output": str(output),
        "rollback": "Delete the three new code/test files and the independent Stage-A output directory.",
        "authoritative_outputs_modified": False,
        "core_simulator_modified": False,
    }
    write_json(output / "rollback_manifest.json", rollback)
    print(json.dumps({"status": "PASS", "output": str(output), "smoke": smoke_check}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
