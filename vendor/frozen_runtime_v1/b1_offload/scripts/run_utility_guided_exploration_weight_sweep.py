"""Default-off exploration-weight sweep for Utility-Guided SafeLinUCB.

This script is intentionally diagnostic-only. It does not modify project
configs on disk and does not touch reward, utility_rule, or SafeFilter logic.
"""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_utility_guided_safe_linucb_equivalence import run_check  # noqa: E402
from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "_utility_guided_exploration_weight_sweep"
ROUTER_NAME = "proposed_lyapunov_rule_mode_safe_bandit"
REGIME_NAME = "energy_shadow_stress_20s"


VARIANTS = [
    ("legacy_ref", False, 1.00),
    ("ug_wx100", True, 1.00),
    ("ug_wx050", True, 0.50),
    ("ug_wx035", True, 0.35),
    ("ug_wx020", True, 0.20),
]


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    text = completed.stdout or ""
    if completed.stderr:
        text += completed.stderr
    return text


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _pct_delta(value: float, reference: float) -> float:
    if abs(reference) <= 1.0e-12:
        return 0.0 if abs(value) <= 1.0e-12 else float("inf")
    return (value - reference) / abs(reference)


def _action_category(action_name: str) -> str:
    name = str(action_name).lower()
    if "metadata_then" in name:
        return "staged"
    if "delay" in name or "defer" in name:
        return "defer"
    if "roi_context" in name or "+roi" in name:
        return "direct_roi_or_roictx"
    if "metadata" in name:
        return "metadata_only"
    return "other"


def _compact_counter(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in sorted(counter.items()))


def _base_policy_repair(enable_utility_guided: bool, exploration_weight: float) -> dict[str, Any]:
    cfg = copy.deepcopy(load_yaml(PROJECT_ROOT / "configs" / "policy_repair_utility_guided_safe_linucb.yaml"))
    cfg["enable_utility_guided_scoring"] = bool(enable_utility_guided)
    scoring = dict(cfg.get("utility_guided_scoring", {}))
    scoring.update(
        {
            "utility_prior_weight": 1.0,
            "learned_correction_weight": 0.20,
            "exploration_weight": float(exploration_weight),
            "score_component_normalization": "zscore_by_candidate_set",
            "fallback_if_zero_variance": True,
            "preserve_legacy_score_logging": True,
        }
    )
    cfg["utility_guided_scoring"] = scoring
    cfg["enable_staged_feasibility_filter"] = False
    return cfg


def _load_run_configs(policy_repair: dict[str, Any], slots: int) -> dict[str, dict[str, Any]]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_stress_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["timeline"]["use_num_days"] = False
    configs["timeline"]["num_slots"] = int(slots)
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    configs["simulator"]["policy_repair"] = copy.deepcopy(policy_repair)
    configs["experiment"] = dict(configs["experiment"])
    configs["experiment"]["routers"] = [ROUTER_NAME]
    return configs


def _write_variant_config(path: Path, variant: str, configs: dict[str, dict[str, Any]], slots: int) -> None:
    payload = {
        "variant": variant,
        "regime": REGIME_NAME,
        "slots": int(slots),
        "source_config_files": [
            "configs/dataset_sarfish_grd_validation.yaml",
            "configs/task_stream_sarfish_energy_shadow_stress.yaml",
            "configs/timeline_sarfish_energy_shadow_stress_20s.yaml",
            "configs/simulator_sarfish_energy_shadow_stress_20s.yaml",
            "configs/resource_sarfish_energy_shadow_stress.yaml",
            "configs/contact_sarfish_physical_20s.yaml",
            "configs/reward.yaml",
            "configs/lyapunov_pricing.yaml",
            "configs/mode_controller.yaml",
            "configs/policy_repair_utility_guided_safe_linucb.yaml",
        ],
        "configs": configs,
    }
    _write_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _run_variant(variant: str, policy_repair: dict[str, Any], slots: int, output_dir: Path):
    configs = _load_run_configs(policy_repair, slots)
    _write_variant_config(output_dir / f"config_{variant}.yaml", variant, configs, slots)
    run_dir = output_dir / "runs" / variant
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], environment.action_space)[0]
    runner = environment.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    candidate_logs = [dict(row, variant=variant, regime=REGIME_NAME) for row in runner.last_candidate_action_logs]
    _write_csv(output_dir / "candidate_action_logs" / f"{variant}.csv", candidate_logs)
    return result, candidate_logs


def _component_values(row: dict[str, Any]) -> tuple[float, float, float]:
    utility = _to_float(row.get("utility_prior_weight")) * _to_float(row.get("utility_prior_norm"))
    learned = _to_float(row.get("learned_correction_weight")) * _to_float(row.get("learned_correction_norm"))
    exploration = _to_float(row.get("exploration_weight")) * _to_float(row.get("safe_exploration_norm"))
    return utility, learned, exploration


def _dominance_name(values: tuple[float, float, float]) -> str:
    labels = ("utility_prior", "learned_correction", "safe_exploration")
    magnitudes = [abs(value) for value in values]
    return labels[max(range(3), key=lambda index: magnitudes[index])]


def _selected_rows(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in logs
        if _to_int(row.get("selected_flag")) == 1 and _to_int(row.get("considered_by_router")) == 1
    ]


def _considered_rows(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in logs if _to_int(row.get("considered_by_router")) == 1]


def _build_scoring_summary(variant: str, logs: list[dict[str, Any]]) -> dict[str, Any]:
    selected = _selected_rows(logs)
    selected_count = len(selected)
    dominance = Counter(_dominance_name(_component_values(row)) for row in selected)
    contribution_rows = [_component_values(row) for row in selected]
    action_flip_values: list[int] = []
    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _considered_rows(logs):
        by_slot[_to_int(row.get("slot"))].append(row)
    for rows in by_slot.values():
        action_flip_values.append(max(_to_int(row.get("action_flip_vs_legacy_shadow")) for row in rows))
    return {
        "variant": variant,
        "selected_rows": selected_count,
        "action_flip_rate_vs_legacy_shadow": mean(action_flip_values) if action_flip_values else 0.0,
        "utility_prior_dominance_rate": dominance["utility_prior"] / max(selected_count, 1),
        "learned_correction_dominance_rate": dominance["learned_correction"] / max(selected_count, 1),
        "safe_exploration_dominance_rate": dominance["safe_exploration"] / max(selected_count, 1),
        "avg_utility_prior_contribution": mean([row[0] for row in contribution_rows]) if contribution_rows else 0.0,
        "avg_learned_correction_contribution": mean([row[1] for row in contribution_rows]) if contribution_rows else 0.0,
        "avg_safe_exploration_contribution": mean([row[2] for row in contribution_rows]) if contribution_rows else 0.0,
    }


def _build_action_summary(variant: str, result) -> dict[str, Any]:
    task_records = [record for record in result.records if record.has_task]
    categories = Counter(_action_category(record.raw_action) for record in task_records)
    model_distribution = Counter(record.selected_model for record in task_records)
    payload_distribution = Counter(record.selected_payload for record in task_records)
    total = max(len(task_records), 1)
    return {
        "variant": variant,
        "task_decisions": len(task_records),
        "staged_raw_action_rate": categories["staged"] / total,
        "direct_roi_or_roictx_rate": categories["direct_roi_or_roictx"] / total,
        "metadata_only_rate": categories["metadata_only"] / total,
        "defer_rate": categories["defer"] / total,
        "other_rate": categories["other"] / total,
        "model_complexity_distribution": _compact_counter(model_distribution),
        "payload_strategy_distribution": _compact_counter(payload_distribution),
    }


def _build_safety_summary(
    variant: str,
    result,
    legacy_result,
    disabled_equivalence_pass: bool,
) -> dict[str, Any]:
    summary = result.summary
    legacy = legacy_result.summary
    final_count = _to_float(summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count")))
    legacy_final = _to_float(legacy.get("final_resource_safety_violation_count", legacy.get("safety_violation_count")))
    raw_count = _to_float(summary.get("raw_safety_violation_count"))
    legacy_raw = _to_float(legacy.get("raw_safety_violation_count"))
    return {
        "variant": variant,
        "disabled_equivalence_pass": int(disabled_equivalence_pass),
        "raw_safety_violation_count": raw_count,
        "final_hard_safety_violation_count": final_count,
        "true_hard_failure_count": "",
        "hard_safety_violation_delta_vs_legacy": final_count - legacy_final,
        "true_hard_failure_delta_vs_legacy": "",
        "raw_safety_violation_delta_vs_legacy": raw_count - legacy_raw,
    }


def _build_outcome_summary(variant: str, result) -> dict[str, Any]:
    summary = result.summary
    return {
        "variant": variant,
        "delivered_items": summary.get("delivered_items", ""),
        "expired_items": summary.get("expired_items", ""),
        "delivered_utility": summary.get("delivered_only_incremental_utility", ""),
        "net_system_utility": summary.get("net_system_utility", ""),
        "effective_utility": summary.get("total_effective_utility", ""),
        "high_value_delivered_utility": summary.get("high_value_delivered_incremental_utility", ""),
        "critical_lifecycle_closure_count": summary.get("alert_to_verification_conversion_count", ""),
        "deadline_miss_count": summary.get("deadline_missed_count", ""),
    }


def _build_sweep_rows(
    variants: list[str],
    scoring_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_variant: dict[str, dict[str, Any]] = {variant: {"variant": variant} for variant in variants}
    for rows in (scoring_rows, action_rows, safety_rows, outcome_rows):
        for row in rows:
            by_variant[str(row["variant"])].update(row)
    return [by_variant[variant] for variant in variants]


def _build_flip_rows(variant: str, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _considered_rows(logs):
        by_slot[_to_int(row.get("slot"))].append(row)
    rows_out: list[dict[str, Any]] = []
    for slot, rows in sorted(by_slot.items()):
        legacy = next((row for row in rows if _to_int(row.get("selected_by_legacy_shadow"))), None)
        guided = next((row for row in rows if _to_int(row.get("selected_by_utility_guided_shadow"))), None)
        actual = next((row for row in rows if _to_int(row.get("selected_flag"))), None)
        if legacy is None or guided is None:
            continue
        legacy_action = str(legacy.get("candidate_action", ""))
        guided_action = str(guided.get("candidate_action", ""))
        if legacy_action == guided_action:
            continue
        rows_out.append(
            {
                "variant": variant,
                "slot": slot,
                "task_id": legacy.get("task_id", ""),
                "task_value": legacy.get("task_value", ""),
                "legacy_shadow_action": legacy_action,
                "utility_guided_shadow_action": guided_action,
                "actual_selected_action": "" if actual is None else actual.get("candidate_action", ""),
                "legacy_category": _action_category(legacy_action),
                "utility_guided_category": _action_category(guided_action),
                "legacy_score_shadow": legacy.get("legacy_score_shadow", ""),
                "utility_guided_score_shadow": guided.get("utility_guided_score_shadow", ""),
                "legacy_predicted_reward": legacy.get("predicted_reward", ""),
                "utility_guided_predicted_reward": guided.get("predicted_reward", ""),
                "legacy_risk_discount": legacy.get("risk_discount", ""),
                "utility_guided_risk_discount": guided.get("risk_discount", ""),
            }
        )
    return rows_out


def _write_top_flips(path: Path, flip_rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flip_rows:
        grouped[str(row["variant"])].append(row)
    lines = ["# Top Action Flips By Variant", ""]
    for variant in [name for name, _enabled, _wx in VARIANTS]:
        lines.extend([f"## {variant}", ""])
        rows = grouped.get(variant, [])
        if not rows:
            lines.extend(["No utility-guided shadow flips observed.", ""])
            continue
        lines.append("| slot | task | legacy shadow | utility-guided shadow | direction |")
        lines.append("|---:|---|---|---|---|")
        for row in rows[:20]:
            direction = f"{row['legacy_category']} -> {row['utility_guided_category']}"
            lines.append(
                f"| {row['slot']} | {row['task_id']} | {row['legacy_shadow_action']} | "
                f"{row['utility_guided_shadow_action']} | {direction} |"
            )
        lines.append("")
    _write_text(path, "\n".join(lines))


def _gate_value(rows: list[dict[str, Any]], variant: str, key: str) -> float:
    row = next(item for item in rows if item["variant"] == variant)
    return _to_float(row.get(key))


def _write_reports(
    output_dir: Path,
    disabled_equivalence_pass: bool,
    scoring_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    outcome_rows: list[dict[str, Any]],
    slots: int,
) -> None:
    wx035_safety_delta = _gate_value(safety_rows, "ug_wx035", "hard_safety_violation_delta_vs_legacy")
    wx035_utility_dom = _gate_value(scoring_rows, "ug_wx035", "utility_prior_dominance_rate")
    wx035_exploration_dom = _gate_value(scoring_rows, "ug_wx035", "safe_exploration_dominance_rate")
    wx035_direct = _gate_value(action_rows, "ug_wx035", "direct_roi_or_roictx_rate")
    wx100_direct = _gate_value(action_rows, "ug_wx100", "direct_roi_or_roictx_rate")
    wx035_defer = _gate_value(action_rows, "ug_wx035", "defer_rate")
    wx100_defer = _gate_value(action_rows, "ug_wx100", "defer_rate")
    wx035_staged = _gate_value(action_rows, "ug_wx035", "staged_raw_action_rate")
    wx100_staged = _gate_value(action_rows, "ug_wx100", "staged_raw_action_rate")
    wx035_net = _gate_value(outcome_rows, "ug_wx035", "net_system_utility")
    wx100_net = _gate_value(outcome_rows, "ug_wx100", "net_system_utility")
    wx035_delivered = _gate_value(outcome_rows, "ug_wx035", "delivered_utility")
    wx100_delivered = _gate_value(outcome_rows, "ug_wx100", "delivered_utility")

    hard_pass = disabled_equivalence_pass and wx035_safety_delta == 0.0
    behavior_pass = (
        wx035_utility_dom >= 0.60
        and wx035_exploration_dom <= 0.25
        and wx035_direct >= wx100_direct - 0.05
        and wx035_defer <= wx100_defer + 0.05
        and wx035_staged <= wx100_staged + 0.02
    )
    wx035_pass = hard_pass and behavior_pass
    net_warn = _pct_delta(wx035_net, wx100_net) < -0.05
    delivered_warn = _pct_delta(wx035_delivered, wx100_delivered) < -0.05

    lower_reduced_exploration = _gate_value(scoring_rows, "ug_wx035", "safe_exploration_dominance_rate") <= _gate_value(
        scoring_rows, "ug_wx100", "safe_exploration_dominance_rate"
    )
    utility_more_dominant = _gate_value(scoring_rows, "ug_wx035", "utility_prior_dominance_rate") >= _gate_value(
        scoring_rows, "ug_wx100", "utility_prior_dominance_rate"
    )
    direct_collapse = wx035_direct < wx100_direct - 0.05
    defer_abnormal = wx035_defer > wx100_defer + 0.05
    staged_reappeared = wx035_staged > wx100_staged + 0.02
    safety_increased = wx035_safety_delta > 0.0

    stable_candidates = []
    for variant in ("ug_wx050", "ug_wx035", "ug_wx020"):
        stable_candidates.append(
            (
                variant,
                _gate_value(scoring_rows, variant, "utility_prior_dominance_rate"),
                _gate_value(scoring_rows, variant, "safe_exploration_dominance_rate"),
                _gate_value(action_rows, variant, "direct_roi_or_roictx_rate"),
                _gate_value(action_rows, variant, "defer_rate"),
                _gate_value(action_rows, variant, "staged_raw_action_rate"),
                _gate_value(safety_rows, variant, "hard_safety_violation_delta_vs_legacy"),
            )
        )
    safest = min(
        stable_candidates,
        key=lambda row: (
            row[6] > 0.0,
            max(0.0, row[2] - 0.25),
            max(0.0, (wx100_direct - 0.05) - row[3]),
            abs(row[2] - 0.20),
        ),
    )[0]

    answer_lines = [
        "# Exploration Weight Sweep Report",
        "",
        f"- regime: {REGIME_NAME}",
        f"- slots: {slots}",
        "- patch type: audit/diagnostic only",
        "- reward core modified: no",
        "- mission_prior_value_v2 modified: no",
        "- lifecycle utility enabled: no",
        "- utility_rule modified: no",
        "- SafeFilter hard constraints modified: no",
        "",
        "## Phase 0 Router Implementation Audit",
        "- utility_prior_weight exists: yes",
        "- learned_correction_weight exists: yes",
        "- exploration_weight exists: yes",
        "- score_component_normalization supports zscore_by_candidate_set: yes",
        "- preserve_legacy_score_logging exists: yes",
        "- legacy_score_shadow is logged/exported in candidate action logs: yes",
        "- utility_guided_score_shadow is logged/exported in candidate action logs: yes",
        "- logging patch needed: no; existing fields were sufficient to compute contributions and dominance offline",
        "",
        "## Required Answers",
        f"1. Did disabled-equivalence remain intact? {'Yes' if disabled_equivalence_pass else 'No'}.",
        f"2. Did lowering exploration_weight reduce safe_exploration dominance? {'Yes' if lower_reduced_exploration else 'No'}.",
        f"3. Did utility prior become more dominant? {'Yes' if utility_more_dominant else 'No'}.",
        f"4. Did direct ROI / ROI+Ctx collapse? {'Yes' if direct_collapse else 'No'}.",
        f"5. Did defer become abnormally high? {'Yes' if defer_abnormal else 'No'}.",
        f"6. Did staged overuse reappear? {'Yes' if staged_reappeared else 'No'}.",
        f"7. Did safety violations increase? {'Yes' if safety_increased else 'No'}.",
        f"8. Which static exploration_weight is the safest recommended experimental variant? {safest}.",
        f"9. Is ug_wx035 acceptable as a utility-guided experimental default? {'Yes' if wx035_pass else 'No'}.",
        "10. Should adaptive exploration schedule be implemented now, or deferred? Deferred; this run only supports static, default-off diagnostics.",
        "",
        "## ug_wx035 Gates",
        f"- disabled-equivalence: {'PASS' if disabled_equivalence_pass else 'FAIL'}",
        f"- hard safety delta vs legacy: {wx035_safety_delta:.6g}",
        f"- utility prior dominance rate: {wx035_utility_dom:.3f} (gate >= 0.60)",
        f"- safe exploration dominance rate: {wx035_exploration_dom:.3f} (gate <= 0.25)",
        f"- direct ROI/ROI+Ctx rate: {wx035_direct:.3f}; ug_wx100={wx100_direct:.3f}",
        f"- defer rate: {wx035_defer:.3f}; ug_wx100={wx100_defer:.3f}",
        f"- staged raw action rate: {wx035_staged:.3f}; ug_wx100={wx100_staged:.3f}",
        f"- net system utility warning >5 percent degradation: {'YES' if net_warn else 'NO'}",
        f"- delivered utility warning >5 percent degradation: {'YES' if delivered_warn else 'NO'}",
        "",
        "## Conclusion",
    ]
    if wx035_pass:
        answer_lines.append(
            "ug_wx035 passes the configured hard and behavior gates. Recommend it only as a utility-guided experimental variant, not the main project default."
        )
    else:
        answer_lines.append(
            f"ug_wx035 fails at least one configured gate. Do not adopt it; keep legacy/main default and treat {safest} as a diagnostic comparator."
        )
    _write_text(output_dir / "exploration_weight_sweep_report.md", "\n".join(answer_lines) + "\n")

    final_lines = [
        "# Final Recommendation",
        "",
        f"- disabled-equivalence: {'PASS' if disabled_equivalence_pass else 'FAIL'}",
        f"- ug_wx035 gate result: {'PASS' if wx035_pass else 'FAIL'}",
        f"- recommended action: {'use ug_wx035 only as a default-off experimental variant' if wx035_pass else 'do not adopt ug_wx035; keep legacy/main default'}",
        f"- safest diagnostic comparator: {safest}",
        "- adaptive exploration schedule: deferred",
        "- performance superiority claim: not made",
    ]
    _write_text(output_dir / "final_recommendation.md", "\n".join(final_lines) + "\n")

    readme_lines = [
        "# Utility-Guided Exploration Weight Sweep",
        "",
        "Diagnostic, default-off sweep for Utility-Guided SafeLinUCB exploration_weight.",
        "",
        "Generated artifacts:",
        "- sweep_summary.csv",
        "- action_distribution_summary.csv",
        "- scoring_dominance_summary.csv",
        "- safety_guardrail_summary.csv",
        "- outcome_guardrail_summary.csv",
        "- top_action_flips_by_variant.md",
        "- exploration_weight_sweep_report.md",
        "- final_recommendation.md",
        "",
        "No main config defaults are modified by this script.",
    ]
    _write_text(output_dir / "README.md", "\n".join(readme_lines) + "\n")


def _write_checkpoint(output_dir: Path, command: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "git_status.txt", _run_text(["git", "status", "--short"]))
    _write_text(output_dir / "git_diff.patch", _run_text(["git", "diff"]))
    env_lines = [
        "# Environment Summary",
        "",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- python: {sys.version.split()[0]}",
        f"- executable: {sys.executable}",
        f"- pyyaml: {getattr(yaml, '__version__', 'unknown')}",
    ]
    try:
        import numpy as np

        env_lines.append(f"- numpy: {np.__version__}")
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        env_lines.append(f"- numpy: unavailable ({exc})")
    _write_text(output_dir / "environment_summary.md", "\n".join(env_lines) + "\n")
    command_lines = [
        "# Command Log",
        "",
        f"- {datetime.now().isoformat(timespec='seconds')}: {command}",
        f"- {datetime.now().isoformat(timespec='seconds')}: python scripts/check_utility_guided_safe_linucb_equivalence.py --output-dir {output_dir}",
    ]
    _write_text(output_dir / "command_log.md", "\n".join(command_lines) + "\n")


def _append_command(output_dir: Path, text: str) -> None:
    path = output_dir / "command_log.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Command Log\n\n"
    _write_text(path, existing.rstrip() + f"\n- {datetime.now().isoformat(timespec='seconds')}: {text}\n")


def run_sweep(output_dir: Path, slots: int, command: str) -> None:
    _write_checkpoint(output_dir, command)

    disabled_pass = run_check(output_dir)
    if not disabled_pass:
        _write_text(
            output_dir / "FAILED_disabled_equivalence.md",
            "# Disabled Equivalence Failed\n\nSweep stopped before exploration_weight diagnostics.\n",
        )
        raise SystemExit(1)

    results: dict[str, Any] = {}
    logs_by_variant: dict[str, list[dict[str, Any]]] = {}
    variant_names: list[str] = []
    for variant, enabled, wx in VARIANTS:
        variant_names.append(variant)
        _append_command(
            output_dir,
            f"run {variant}: enable_utility_guided_scoring={str(enabled).lower()}, exploration_weight={wx}",
        )
        policy_repair = _base_policy_repair(enabled, wx)
        result, logs = _run_variant(variant, policy_repair, slots, output_dir)
        results[variant] = result
        logs_by_variant[variant] = logs

    legacy_result = results["legacy_ref"]
    scoring_rows = [_build_scoring_summary(variant, logs_by_variant[variant]) for variant in variant_names]
    action_rows = [_build_action_summary(variant, results[variant]) for variant in variant_names]
    safety_rows = [
        _build_safety_summary(variant, results[variant], legacy_result, disabled_pass) for variant in variant_names
    ]
    outcome_rows = [_build_outcome_summary(variant, results[variant]) for variant in variant_names]
    sweep_rows = _build_sweep_rows(variant_names, scoring_rows, action_rows, safety_rows, outcome_rows)
    flip_rows: list[dict[str, Any]] = []
    for variant in variant_names:
        flip_rows.extend(_build_flip_rows(variant, logs_by_variant[variant]))

    _write_csv(output_dir / "sweep_summary.csv", sweep_rows)
    _write_csv(output_dir / "action_distribution_summary.csv", action_rows)
    _write_csv(output_dir / "scoring_dominance_summary.csv", scoring_rows)
    _write_csv(output_dir / "safety_guardrail_summary.csv", safety_rows)
    _write_csv(output_dir / "outcome_guardrail_summary.csv", outcome_rows)
    _write_csv(output_dir / "action_flips_by_variant.csv", flip_rows)
    _write_top_flips(output_dir / "top_action_flips_by_variant.md", flip_rows)
    _write_reports(output_dir, disabled_pass, scoring_rows, action_rows, safety_rows, outcome_rows, slots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Utility-Guided SafeLinUCB exploration_weight sweep.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--slots", type=int, default=120)
    args = parser.parse_args()
    command = "python scripts/run_utility_guided_exploration_weight_sweep.py"
    command += f" --output-dir {args.output_dir} --slots {args.slots}"
    run_sweep(args.output_dir, args.slots, command)


if __name__ == "__main__":
    main()
