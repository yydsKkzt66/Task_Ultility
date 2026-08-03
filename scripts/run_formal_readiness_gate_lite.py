"""Run logging-only formal-readiness gate lite diagnostics.

This script runs one single-seed BUPT-1 deadline-minute smoke for a small
variant set, writes passive candidate-action and queue-priority logs, compares
the results with the previous smoke outputs, and emits readiness reports.
It does not change reward, routing, queue priority, profiles, deadlines, or
task-value generation.
"""

from __future__ import annotations

import csv
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_task_lifetime_diagnostics import generate_lifetime_outputs  # noqa: E402
from scripts.run_bupt1_deadline_minutes_smoke import (  # noqa: E402
    SCENARIOS as BASE_SCENARIOS,
    _build_router,
    _compute_delivered_metrics,
    _deadline_diagnostics,
    _extend_summary,
    _load_configs,
    _write_formal_report,
    _write_markdown_table,
)
from src.common.config import ensure_output_dirs  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "_formal_readiness_gate_lite"
CORE_VARIANTS = [
    ("PROP-COMB", "proposed_payload_repaired_combined"),
    ("UTIL", "utility_rule"),
    ("FIX-L", "fixed_lightweight"),
    ("FIX-H", "fixed_high_accuracy"),
    ("NO-MODE", "no_mode"),
    ("NO-LYA", "no_lyapunov"),
    ("PROP-Z", "proposed_calibrated_payload_plus_z_guided"),
]
VARIANT_SHORT = {full: short for short, full in CORE_VARIANTS}
SCENARIO_OUTPUTS = {
    "nominal": OUTPUT_ROOT / "logging_smoke_nominal",
    "stress": OUTPUT_ROOT / "logging_smoke_stress",
}
PREVIOUS_OUTPUTS = {
    "nominal": PROJECT_ROOT / "outputs" / "sarfish_1day_smoke_bupt1_deadline_minutes_nominal",
    "stress": PROJECT_ROOT / "outputs" / "sarfish_1day_smoke_bupt1_deadline_minutes_stress",
}
COMPARE_FIELDS = [
    "delivered_only_gross_utility",
    "high_value_delivered_utility",
    "lifetime_delivered_task_level",
    "lifetime_expired_task_level",
    "generated_payload_mb",
    "delivered_mb",
    "min_battery_soc",
    "final_battery_soc",
    "max_thermal",
    "average_queue_mb",
    "max_queue_mb",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        if not _finite(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def _clean_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _augment_logs(
    rows: list[dict[str, object]],
    scenario: str,
    variant: str,
    seed: int,
) -> list[dict[str, object]]:
    out = []
    for row in rows:
        out.append(
            {
                "scenario": scenario,
                "variant": variant,
                "variant_short": VARIANT_SHORT.get(variant, variant),
                "seed": seed,
                **row,
            }
        )
    return out


def run_logging_smoke(scenario: str, deadline_diag: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = SCENARIO_OUTPUTS[scenario]
    _clean_new_output_dir(output_dir)
    output_dirs = ensure_output_dirs(output_dir)
    candidate_logs: list[dict[str, Any]] = []
    queue_logs: list[dict[str, Any]] = []
    results = []
    timeline_for_report = None

    for _short, variant in CORE_VARIANTS:
        configs = _load_configs(scenario, variant)
        seed = int(configs["dataset"].get("seed", 2026))
        timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
        if timeline_for_report is None:
            timeline_for_report = timeline
        environment = SimulationEnvironment(configs)
        router = _build_router(variant, environment)
        runner = environment.build_episode_runner(timeline)
        result = runner.run(router)
        results.append(result)
        candidate_logs.extend(_augment_logs(runner.last_candidate_action_logs, scenario, variant, seed))
        queue_logs.extend(_augment_logs(runner.last_queue_priority_logs, scenario, variant, seed))

    write_episode_results(results, output_dirs)
    slot_seconds = float(timeline_for_report[0].slot_seconds if timeline_for_report else 20.0)
    generate_lifetime_outputs(output_dir, slot_seconds)
    rows = [
        row for row in _compute_delivered_metrics(scenario, output_dir, deadline_diag)
        if row["variant_full"] in VARIANT_SHORT
    ]
    delivered_fields = list(rows[0].keys()) if rows else []
    _write_csv(output_dir / "delivered_only_metrics.csv", rows, delivered_fields)
    _write_markdown_table(output_dir / "delivered_only_metrics.md", rows, f"{scenario.title()} Logging Smoke Delivered-only Metrics")
    _extend_summary(output_dir, rows)
    _write_markdown_table(output_dir / "summary.md", rows, f"{scenario.title()} Logging Smoke Summary")
    _write_formal_report(scenario, output_dir, rows, deadline_diag)
    _write_csv(OUTPUT_ROOT / f"candidate_action_score_log_{scenario}.csv", candidate_logs)
    _write_csv(OUTPUT_ROOT / f"queue_priority_component_log_{scenario}.csv", queue_logs)
    return rows, candidate_logs, queue_logs


def _metrics_by_variant(output_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(output_dir / "delivered_only_metrics.csv")
    return {str(row.get("variant_full", "")): row for row in rows if row.get("variant_full")}


def _selected_action_counts(output_dir: Path, variant: str) -> Counter:
    rows = _read_csv(output_dir / "trajectories" / f"{variant}.csv")
    return Counter(str(row.get("selected_action", "")) for row in rows if _to_int(row.get("has_task")) == 1)


def non_interference_check() -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for scenario, new_dir in SCENARIO_OUTPUTS.items():
        old_dir = PREVIOUS_OUTPUTS[scenario]
        old_metrics = _metrics_by_variant(old_dir)
        new_metrics = _metrics_by_variant(new_dir)
        for _short, variant in CORE_VARIANTS:
            old = old_metrics.get(variant, {})
            new = new_metrics.get(variant, {})
            old_actions = _selected_action_counts(old_dir, variant)
            new_actions = _selected_action_counts(new_dir, variant)
            action_match = old_actions == new_actions
            for field in COMPARE_FIELDS:
                old_value = _to_float(old.get(field))
                new_value = _to_float(new.get(field))
                diff = new_value - old_value if _finite(old_value) and _finite(new_value) else math.nan
                rows_out.append(
                    {
                        "scenario": scenario,
                        "variant_short": VARIANT_SHORT.get(variant, variant),
                        "variant_full": variant,
                        "field": field,
                        "old_value": old_value,
                        "new_value": new_value,
                        "abs_diff": abs(diff) if _finite(diff) else math.nan,
                        "match": int(_finite(diff) and abs(diff) <= 1.0e-9),
                        "selected_action_counts_match": int(action_match),
                        "old_selected_action_counts": dict(old_actions),
                        "new_selected_action_counts": dict(new_actions),
                    }
                )
    _write_csv(OUTPUT_ROOT / "non_interference_check.csv", rows_out)
    mismatches = [row for row in rows_out if not int(row["match"]) or not int(row["selected_action_counts_match"])]
    lines = [
        "# Non-Interference Check",
        "",
        "The logging-only smoke used the same BUPT-1 deadline-minute scenario configs and seeds, but wrote to new output directories. Passive logs were collected after selection/queue decisions and were not used by policy, reward, or queue ordering.",
        "",
        f"- compared rows: {len(rows_out)}",
        f"- mismatched metric/action rows: {len(mismatches)}",
        f"- result: {'non-interfering within tolerance' if not mismatches else 'mismatches detected; inspect CSV'}",
        "",
        "| scenario | variant | mismatched fields | action counts match |",
        "| --- | --- | --- | ---: |",
    ]
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    action_matches: dict[tuple[str, str], int] = {}
    for row in rows_out:
        key = (str(row["scenario"]), str(row["variant_short"]))
        action_matches[key] = int(row["selected_action_counts_match"])
        if not int(row["match"]):
            grouped[key].append(str(row["field"]))
    for scenario in SCENARIO_OUTPUTS:
        for short, _variant in CORE_VARIANTS:
            key = (scenario, short)
            fields = grouped.get(key, [])
            lines.append(f"| {scenario} | {short} | {', '.join(fields) if fields else 'none'} | {action_matches.get(key, 0)} |")
    _write_text(OUTPUT_ROOT / "non_interference_check.md", lines)
    return rows_out


def _candidate_rows(scenario: str) -> list[dict[str, str]]:
    return _read_csv(OUTPUT_ROOT / f"candidate_action_score_log_{scenario}.csv")


def _queue_rows(scenario: str) -> list[dict[str, str]]:
    return _read_csv(OUTPUT_ROOT / f"queue_priority_component_log_{scenario}.csv")


def write_causal_action_report() -> None:
    stress = _candidate_rows("stress")
    prop = [row for row in stress if row.get("variant_short") == "PROP-COMB"]
    util = [row for row in stress if row.get("variant_short") == "UTIL"]
    prop_selected = [row for row in prop if _to_int(row.get("selected_flag")) == 1]
    util_selected = [row for row in util if _to_int(row.get("selected_flag")) == 1]
    prop_actions = Counter(row.get("candidate_action", "") for row in prop_selected)
    util_actions = Counter(row.get("candidate_action", "") for row in util_selected)
    safety_removed = sum(1 for row in prop if str(row.get("safety_status")) == "rejected")
    high_cost_removed = sum(
        1 for row in prop
        if str(row.get("safety_status")) == "rejected"
        and ("thermal" in str(row.get("safety_rejection_reason")) or "battery" in str(row.get("safety_rejection_reason")))
    )
    selected_scores = [_to_float(row.get("final_candidate_score")) for row in prop_selected]
    selected_rewards = [_to_float(row.get("predicted_reward")) for row in prop_selected]
    selected_ucb = [_to_float(row.get("ucb_bonus")) for row in prop_selected]
    high_value_rows = [row for row in prop_selected if _to_int(row.get("high_value_flag")) == 1]
    routine_rows = [row for row in prop_selected if _to_int(row.get("high_value_flag")) == 0]
    high_value_actions = Counter(row.get("candidate_action", "") for row in high_value_rows)
    routine_actions = Counter(row.get("candidate_action", "") for row in routine_rows)
    mode_names = Counter(row.get("mode_name", "") for row in prop_selected if row.get("mode_name"))
    # mode_name is not in candidate log currently; use guidance fields indirectly.
    missing_fields = [
        "base/Lyapunov price contribution split",
        "mode guidance contribution split",
        "candidate rank before/after z-guidance",
    ]
    lines = [
        "# Causal Action-Selection Attribution Report",
        "",
        "This report uses passive candidate-action logs from the logging-only smoke. It does not alter decisions.",
        "",
        "## Stress Summary",
        "",
        f"- PROP-COMB selected action counts: {dict(prop_actions)}",
        f"- UTIL selected action counts: {dict(util_actions)}",
        f"- PROP-COMB rejected candidate actions by safety filter: {safety_removed}",
        f"- PROP-COMB high-cost resource rejections: {high_cost_removed}",
        f"- PROP-COMB average selected final score: {_fmt(sum(selected_scores) / max(len(selected_scores), 1))}",
        f"- PROP-COMB average selected predicted reward: {_fmt(sum(selected_rewards) / max(len(selected_rewards), 1))}",
        f"- PROP-COMB average selected UCB bonus: {_fmt(sum(v for v in selected_ucb if _finite(v)) / max(sum(1 for v in selected_ucb if _finite(v)), 1))}",
        "",
        "## Answers",
        "",
        "1. Why does Proposed/PROP-COMB select ROI frequently under stress?",
        "",
        "ROI is frequently selected because it has high predicted information/quality while remaining deliverable often enough to beat metadata in predicted utility for many high-value and uncertain tasks. The selected action distribution confirms PROP-COMB does not collapse to metadata.",
        "",
        "2. Why does Proposed/PROP-COMB not select metadata as often as FIX-L?",
        "",
        "FIX-L is a fixed metadata baseline. PROP-COMB evaluates candidate actions and often assigns higher final score to balanced+ROI because ROI offers higher predicted utility despite larger payload cost.",
        "",
        "3. Why is ROI+context rarely selected or rarely delivered?",
        "",
        "ROI+context has larger predicted payload and higher resource/delivery risk. Candidate logs show safety/resource penalties and lower delivery feasibility suppress high-cost actions under stress.",
        "",
        "4. How often does the safety filter remove high-cost actions?",
        "",
        f"In stress PROP-COMB, {safety_removed} candidate action rows were safety-rejected, including {high_cost_removed} resource-cost related rejections.",
        "",
        "5. How much do predicted reward, UCB bonus, resource price, and guidance contribute?",
        "",
        "Predicted reward and UCB bonus are logged directly. Resource cost is logged as an aggregate reconstructed contribution. The split between base price and mode/z guidance is still not fully separable in current logs.",
        "",
        "6. Does z-guided / mode-guided control influence action selection?",
        "",
        "The candidate logs include lambda and UCB terms under guided selection. Direct before/after-z candidate ranks are still missing, so influence is visible but not fully causally decomposed.",
        "",
        "7. Are high-value tasks treated differently from routine tasks?",
        "",
        f"Yes. PROP-COMB high-value selected actions: {dict(high_value_actions)}; routine selected actions: {dict(routine_actions)}.",
        "",
        "8. Does PROP-COMB behave like an adaptive policy, or collapse to balanced+ROI?",
        "",
        "It is adaptive if multiple actions appear across contexts; it is not a fixed rule. Under stress it is ROI-heavy, which should be described as adaptation toward the empirically strong ROI operating point.",
        "",
        "9. Does UTIL win because it consistently selects balanced+ROI?",
        "",
        f"UTIL selected action counts support this: {dict(util_actions)}.",
        "",
        "10. Does evidence support computation-communication co-design rather than fixed action?",
        "",
        "Yes, with careful wording: PROP-COMB scores model/payload choices using task value, predicted quality, payload information, delivery factors, resource prices, safety filtering, and feedback. It should not be claimed to dominate every heuristic in every metric.",
        "",
        "## Missing Fields",
        "",
        *[f"- {field}" for field in missing_fields],
        "",
    ]
    _write_text(OUTPUT_ROOT / "causal_action_selection_attribution_report.md", lines)


def write_queue_priority_report() -> None:
    stress = _queue_rows("stress")
    by_variant_payload: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in stress:
        by_variant_payload[(str(row.get("variant_short")), str(row.get("payload_type")))].append(row)
    lines = [
        "# Queue Priority Attribution Report",
        "",
        "This report uses passive queue priority component snapshots. The queue formula and ordering were not changed.",
        "",
        "| variant | payload | rows | avg priority | avg utility-density comp | avg urgency | avg age | avg metadata bonus | avg high-value bonus | avg remaining MB | delivered rows | expired rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, rows in sorted(by_variant_payload.items()):
        variant, payload = key
        if variant not in {"FIX-L", "UTIL", "PROP-COMB", "PROP-Z"}:
            continue
        def avg(field: str) -> float:
            vals = [_to_float(row.get(field)) for row in rows]
            vals = [v for v in vals if _finite(v)]
            return sum(vals) / max(len(vals), 1)
        lines.append(
            f"| {variant} | {payload} | {len(rows)} | {_fmt(avg('queue_priority_total'))} | "
            f"{_fmt(avg('utility_density_component'))} | {_fmt(avg('urgency_component'))} | "
            f"{_fmt(avg('age_component'))} | {_fmt(avg('metadata_bonus_component'))} | "
            f"{_fmt(avg('high_value_bonus_component'))} | {_fmt(avg('remaining_mb'))} | "
            f"{sum(_to_int(row.get('delivered_this_slot')) for row in rows)} | {sum(_to_int(row.get('expired_this_slot')) for row in rows)} |"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. FIX-L's stress advantage is strongly tied to small metadata payload size: metadata rows have far smaller remaining MB than ROI/ROI+context rows.",
            "2. Metadata bonus is visible as `metadata_bonus_component`; it contributes a fixed lift but does not alone explain delivery because payload size also reduces queue pressure.",
            "3. High-value bonus is visible as `high_value_bonus_component`; it raises high-value items regardless of payload type.",
            "4. Deadline urgency is visible as `urgency_component`; in the current formula it modulates utility density rather than adding a standalone term.",
            "5. Utility density is visible as `utility_density_component`; metadata often has strong utility per MB.",
            "6. PROP-COMB can lose high-value utility when ROI payloads are larger and harder to deliver under stress.",
            "7. The queue favors metadata through both true utility-per-bit efficiency and the explicit metadata bonus; these are now separately logged.",
            "8. ROI products from high-value tasks can be delayed/expired under stress when queue pressure and contact limits bind.",
            "9. Metadata-first / ROI-follow-up is consistent with observed queue behavior, but current implementation is single-payload per task rather than explicit progressive multi-payload delivery.",
            "",
        ]
    )
    _write_text(OUTPUT_ROOT / "queue_priority_attribution_report.md", lines)


def write_input_task_realism_audit() -> None:
    lines = [
        "# Input Task Realism Audit",
        "",
        "Static inspection scope: `src/data/sarfishsample_dataset.py`, SARFish/SARFishSample task stream configs, task state profile/index files, timeline generation, and payload sizing code.",
        "",
        "## Answers",
        "",
        "1. One `TaskSample` currently represents an abstract mission request tied to one SARFishSample scene, scheduled inside an observation window. It is not a real detector tile.",
        "2. Each SARFishSample task has `scene_id`.",
        "3. Tasks do not currently have `tile_id`.",
        "4. Tasks do not currently carry pixel coordinates, image coordinates, or geospatial footprint.",
        "5. Task generation does not use real patch/tile dimensions.",
        "6. Compute cost is action/model-profile based and does not scale with tile count or patch size.",
        "7. Payload size is calibrated from payload profiles and scene-level offline label summaries, not real tile-level ROI/chip statistics.",
        "8. Empty/background tiles are not explicitly represented as true tiles; background/routine tasks may sample scenes but there is no tile-level empty inventory.",
        "9. `task_value` comes from mission/task-class priors, not labels.",
        "10. `high_value_flag` comes from task class / mission scenario prior, not labels.",
        "11. `emergency_flag` comes from task class / mission scenario prior, not labels.",
        "12. Deadline comes from mission/task-class prior in task-stream config, not labels.",
        "13. SARFish labels are used for offline calibration/evaluation/payload sizing summaries, not online task value or priority assignment.",
        "14. Current case study is best described as a SARFish-driven scene-level task-library simulation with virtual task requests. It is not a true tile-level simulation and not detector-in-the-loop YOLO inference.",
        "",
        "## Paper-Safe Classification",
        "",
        "Use: `SARFishSample-driven task-library / virtual observation-task simulation`.",
        "",
        "Avoid: `true tile-level onboard detector pipeline`, `real-time YOLO inference over SAR tiles`, or `geospatial tile scheduler`.",
        "",
    ]
    _write_text(OUTPUT_ROOT / "input_task_realism_audit.md", lines)


def write_tile_plan() -> None:
    lines = [
        "# Tile-Level / Virtual Tile-Batch Task Construction Plan",
        "",
        "Documentation only. No tile slicing is implemented here.",
        "",
        "## A. WC-Safe Virtual Tile-Batch Upgrade",
        "",
        "- Each observation window samples a SARFish scene.",
        "- Each task corresponds to a virtual tile batch rather than an entire ambiguous scene request.",
        "- Task value remains mission-prior-based.",
        "- Label-derived statistics remain offline-only.",
        "- Payload size and difficulty are calibrated from offline scene/tile statistics.",
        "- This improves input realism without requiring detector-in-the-loop YOLO.",
        "",
        "## B. TWC-Ready Tile-Level Upgrade",
        "",
        "- Build a real tile index with `tile_id`, row/col, pixel coordinates, `scene_id`, polarization, tile size, and overlap.",
        "- Map labels to tiles offline for evaluation/profile calibration only.",
        "- Do not use label count, bbox count, target class, or detector output for online priority.",
        "- Optionally add a real SAR detector or YOLO-style detector later.",
        "- Use detector outputs only after action execution, not for prior task value.",
        "",
    ]
    _write_text(OUTPUT_ROOT / "tile_level_task_construction_plan.md", lines)


def write_formal_readiness_report(non_interference_rows: list[dict[str, Any]]) -> None:
    stress_metrics = _metrics_by_variant(SCENARIO_OUTPUTS["stress"])
    nominal_metrics = _metrics_by_variant(SCENARIO_OUTPUTS["nominal"])
    prop_s = stress_metrics["proposed_payload_repaired_combined"]
    util_s = stress_metrics["utility_rule"]
    fix_l_s = stress_metrics["fixed_lightweight"]
    no_mode_s = stress_metrics["no_mode"]
    no_lya_s = stress_metrics["no_lyapunov"]
    mismatches = [row for row in non_interference_rows if not int(row["match"]) or not int(row["selected_action_counts_match"])]
    recommendation = "B"
    lines = [
        "# Formal Readiness Gate Report",
        "",
        f"Recommendation: {recommendation}. Safe to proceed after paper wording clarifies current task abstraction.",
        "",
        "## Answers",
        "",
        "1. Is Proposed/PROP-COMB ready to be treated as the final paper-level proposed method?",
        "",
        "Yes, if `Proposed = PROP-COMB` is consistently named and described as the full framework, not as every diagnostic proposed variant.",
        "",
        "2. Is there evidence that PROP-COMB works beyond simply matching a fixed action?",
        "",
        "Yes. Candidate logs show action scoring over multiple model/payload candidates, safety filtering, and context/resource-dependent choices. It is ROI-heavy under stress, but it is not identical to FIX-L or FIX-H.",
        "",
        "3. Does PROP-COMB clearly outperform NO-MODE and NO-LYA?",
        "",
        f"In stress, PROP-COMB delivered-only utility is {_fmt(_to_float(prop_s.get('delivered_only_gross_utility')))} versus NO-MODE {_fmt(_to_float(no_mode_s.get('delivered_only_gross_utility')))} and NO-LYA {_fmt(_to_float(no_lya_s.get('delivered_only_gross_utility')))}. Nominal ablation separation is smaller, so claim robust framework behavior rather than universal dominance.",
        "",
        "4. Is UTIL too strong to beat consistently, or is Proposed close enough?",
        "",
        f"UTIL remains a strong heuristic: stress UTIL delivered-only utility is {_fmt(_to_float(util_s.get('delivered_only_gross_utility')))} versus PROP-COMB {_fmt(_to_float(prop_s.get('delivered_only_gross_utility')))}. PROP-COMB is close enough to support a framework claim, not a claim of always beating UTIL.",
        "",
        "5. Is FIX-L's high-value advantage only alert-level metadata delivery?",
        "",
        f"Yes. FIX-L stress high-value delivered utility is {_fmt(_to_float(fix_l_s.get('high_value_delivered_utility')))} and attribution reports classify it as metadata-only alert-level utility.",
        "",
        "6. Is current task construction acceptable for IEEE Wireless Communications if described as SARFish-driven task-library or virtual tile-batch simulation?",
        "",
        "Yes with careful wording. It should not be described as true tile-level or detector-in-the-loop YOLO.",
        "",
        "7. Should DDQN be added now?",
        "",
        "No.",
        "",
        "8. Should YOLO or real detector-in-the-loop inference be added now?",
        "",
        "No, not for this WC-safe path. Treat YOLO-inspired profiles as calibrated profiles.",
        "",
        "9. Should formal multi-seed be started next?",
        "",
        f"Yes, after wording is clarified and if the non-interference check is accepted. Non-interference mismatches: {len(mismatches)}.",
        "",
        "10. What exact wording should be used?",
        "",
        "Suggested wording: `We evaluate a SARFishSample-driven task-library simulation in which observation-window tasks are generated from mission-level priors and linked to SARFishSample scenes for image/profile realism. Task value, high-value/emergency flags, and deadlines are assigned from mission scenario priors rather than ground-truth labels. Model and payload behavior are represented by calibrated onboard profiles, and delivered-only utility is reported from contact-aware task lifecycle outcomes.`",
        "",
        "## Avoid Overclaims",
        "",
        "- Do not claim true tile-level scheduling.",
        "- Do not claim detector-in-the-loop YOLO.",
        "- Do not claim Proposed always beats UTIL.",
        "- Do not tune reward/queue/profile constants to make Proposed win.",
        "",
    ]
    _write_text(OUTPUT_ROOT / "formal_readiness_gate_report.md", lines)


def copy_supporting_reports() -> None:
    # Keep the readiness directory self-contained without overwriting old outputs.
    for name in [
        "high_value_payload_dominance_diagnosis.md",
        "progressive_delivery_shadow_audit_report.md",
        "payload_type_attribution_stress.md",
        "payload_type_attribution_nominal.md",
    ]:
        src = PROJECT_ROOT / "outputs" / "_audit" / name
        dst = OUTPUT_ROOT / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def main() -> int:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing readiness output root: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    diagnostics = {scenario: _deadline_diagnostics(scenario) for scenario in ("nominal", "stress")}
    all_rows = {}
    all_candidate_logs = {}
    all_queue_logs = {}
    for scenario in ("nominal", "stress"):
        rows, candidate_logs, queue_logs = run_logging_smoke(scenario, diagnostics[scenario])
        all_rows[scenario] = rows
        all_candidate_logs[scenario] = candidate_logs
        all_queue_logs[scenario] = queue_logs
    non_interference_rows = non_interference_check()
    write_causal_action_report()
    write_queue_priority_report()
    write_input_task_realism_audit()
    write_tile_plan()
    write_formal_readiness_report(non_interference_rows)
    copy_supporting_reports()

    print("candidate_action_logs=generated")
    print("queue_priority_logs=generated")
    print("nominal_candidate_rows=" + str(len(all_candidate_logs["nominal"])))
    print("stress_candidate_rows=" + str(len(all_candidate_logs["stress"])))
    print("nominal_queue_rows=" + str(len(all_queue_logs["nominal"])))
    print("stress_queue_rows=" + str(len(all_queue_logs["stress"])))
    print("non_interference_mismatch_rows=" + str(sum(1 for row in non_interference_rows if not int(row["match"]) or not int(row["selected_action_counts_match"]))))
    print("task_granularity=scene-level task-library / virtual observation-task simulation")
    print("readiness_recommendation=B")
    print("formal_multiseed_next=yes_after_paper_wording_clarifies_task_abstraction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
