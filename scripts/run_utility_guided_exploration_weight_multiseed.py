"""Multi-seed stability screen for Utility-Guided SafeLinUCB exploration weight.

Diagnostic-only runner:
- no reward changes
- no lifecycle metrics
- no tiebreaker or adaptive schedule
- no main config default changes
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
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


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "_utility_guided_exploration_weight_multiseed"
ROUTER_NAME = "proposed_lyapunov_rule_mode_safe_bandit"
REGIME_NAME = "energy_shadow_stress_20s"
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]

VARIANTS = [
    ("legacy_ref", False, 1.00),
    ("ug_wx100", True, 1.00),
    ("ug_wx050", True, 0.50),
    ("ug_wx035", True, 0.35),
]

PAIRWISE_COMPARISONS = [
    ("ug_wx035", "ug_wx100"),
    ("ug_wx035", "legacy_ref"),
    ("ug_wx050", "ug_wx100"),
    ("ug_wx050", "legacy_ref"),
]

IMPORTANT_DELTA_METRICS = [
    "net_system_utility",
    "delivered_utility",
    "effective_utility",
    "utility_prior_dominance_rate",
    "safe_exploration_dominance_rate",
    "direct_roi_or_roictx_rate",
    "defer_rate",
    "staged_raw_action_rate",
    "final_hard_safety_violation_count",
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


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _bootstrap_ci(values: list[float], iterations: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(2026)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _item in values]
        means.append(mean(sample))
    means.sort()
    low_index = max(0, int(0.025 * len(means)) - 1)
    high_index = min(len(means) - 1, int(0.975 * len(means)))
    return means[low_index], means[high_index]


def _pct_degradation(value: float, reference: float) -> float:
    if abs(reference) <= 1.0e-12:
        return 0.0 if abs(value) <= 1.0e-12 else math.inf
    return (reference - value) / abs(reference)


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


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _policy_repair(enable_utility_guided: bool, exploration_weight: float) -> dict[str, Any]:
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


def _load_run_configs(policy_repair: dict[str, Any], slots: int, seed: int) -> dict[str, dict[str, Any]]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_stress_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["dataset"]["seed"] = int(seed)
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["timeline"]["use_num_days"] = False
    configs["timeline"]["num_slots"] = int(slots)
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    configs["simulator"]["policy_repair"] = copy.deepcopy(policy_repair)
    configs["experiment"] = dict(configs["experiment"])
    configs["experiment"]["routers"] = [ROUTER_NAME]
    return configs


def _write_variant_config(output_dir: Path, variant: str, policy_repair: dict[str, Any], slots: int, seeds: list[int]) -> None:
    configs = _load_run_configs(policy_repair, slots, seeds[0])
    payload = {
        "variant": variant,
        "regime": REGIME_NAME,
        "paired_seeds": seeds,
        "slots": int(slots),
        "seed_control": "configs['dataset']['seed'] is replaced per run; all arms share the same seed per pair.",
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
        "policy_repair": policy_repair,
        "representative_first_seed_configs": configs,
    }
    _write_text(output_dir / f"config_{variant}.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _run_variant_seed(
    *,
    variant: str,
    policy_repair: dict[str, Any],
    slots: int,
    seed: int,
    output_dir: Path,
):
    configs = _load_run_configs(policy_repair, slots, seed)
    run_dir = output_dir / "runs" / f"seed_{seed}" / variant
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], environment.action_space)[0]
    runner = environment.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    logs = [dict(row, seed=seed, variant=variant, regime=REGIME_NAME) for row in runner.last_candidate_action_logs]
    _write_csv(output_dir / "candidate_action_logs" / f"seed_{seed}_{variant}.csv", logs)
    return result, logs


def _component_values(row: dict[str, Any]) -> tuple[float, float, float]:
    utility = _to_float(row.get("utility_prior_weight")) * _to_float(row.get("utility_prior_norm"))
    learned = _to_float(row.get("learned_correction_weight")) * _to_float(row.get("learned_correction_norm"))
    exploration = _to_float(row.get("exploration_weight")) * _to_float(row.get("safe_exploration_norm"))
    return utility, learned, exploration


def _dominance_name(values: tuple[float, float, float]) -> str:
    labels = ("utility_prior", "learned_correction", "safe_exploration")
    magnitudes = [abs(value) for value in values]
    return labels[max(range(3), key=lambda index: magnitudes[index])]


def _selected_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in logs
        if _to_int(row.get("selected_flag")) == 1 and _to_int(row.get("considered_by_router")) == 1
    ]


def _considered_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in logs if _to_int(row.get("considered_by_router")) == 1]


def _scoring_metrics(logs: list[dict[str, Any]]) -> dict[str, float]:
    selected = _selected_logs(logs)
    dominance = Counter(_dominance_name(_component_values(row)) for row in selected)
    contributions = [_component_values(row) for row in selected]
    by_slot: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _considered_logs(logs):
        by_slot[_to_int(row.get("slot"))].append(row)
    flip_values = [max(_to_int(row.get("action_flip_vs_legacy_shadow")) for row in rows) for rows in by_slot.values()]
    selected_count = max(len(selected), 1)
    return {
        "action_flip_rate_vs_legacy_shadow": mean(flip_values) if flip_values else 0.0,
        "utility_prior_dominance_rate": dominance["utility_prior"] / selected_count,
        "learned_correction_dominance_rate": dominance["learned_correction"] / selected_count,
        "safe_exploration_dominance_rate": dominance["safe_exploration"] / selected_count,
        "avg_utility_prior_contribution": mean([item[0] for item in contributions]) if contributions else 0.0,
        "avg_learned_correction_contribution": mean([item[1] for item in contributions]) if contributions else 0.0,
        "avg_safe_exploration_contribution": mean([item[2] for item in contributions]) if contributions else 0.0,
    }


def _action_metrics(result) -> dict[str, Any]:
    task_records = [record for record in result.records if record.has_task]
    categories = Counter(_action_category(record.raw_action) for record in task_records)
    models = Counter(record.selected_model for record in task_records)
    payloads = Counter(record.selected_payload for record in task_records)
    total = max(len(task_records), 1)
    return {
        "task_decisions": len(task_records),
        "staged_raw_action_rate": categories["staged"] / total,
        "direct_roi_or_roictx_rate": categories["direct_roi_or_roictx"] / total,
        "metadata_only_rate": categories["metadata_only"] / total,
        "defer_rate": categories["defer"] / total,
        "model_complexity_distribution": _compact_counter(models),
        "payload_strategy_distribution": _compact_counter(payloads),
    }


def _summary_metrics(result) -> dict[str, Any]:
    summary = result.summary
    return {
        "raw_safety_violation_count": _to_float(summary.get("raw_safety_violation_count")),
        "final_hard_safety_violation_count": _to_float(
            summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count"))
        ),
        "true_hard_failure_count": "",
        "deadline_miss_count": _to_float(summary.get("deadline_missed_count")),
        "queue_overflow_count": _to_float(summary.get("queue_overflow_count")),
        "net_system_utility": _to_float(summary.get("net_system_utility")),
        "delivered_utility": _to_float(summary.get("delivered_only_incremental_utility")),
        "effective_utility": _to_float(summary.get("total_effective_utility")),
        "high_value_delivered_utility": _to_float(summary.get("high_value_delivered_incremental_utility")),
        "delivered_items": _to_float(summary.get("delivered_items")),
        "expired_items": _to_float(summary.get("expired_items")),
        "completed_tasks": _to_float(summary.get("completed_tasks")),
        "deadline_satisfaction_ratio": _to_float(summary.get("deadline_satisfaction_ratio")),
    }


def _per_seed_row(seed: int, variant: str, result, logs: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"seed": seed, "variant": variant, "regime": REGIME_NAME}
    row.update(_summary_metrics(result))
    row.update(_scoring_metrics(logs))
    row.update(_action_metrics(result))
    return row


def _aggregate_rows(rows: list[dict[str, Any]], metrics: list[str], output_kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    for variant in [item[0] for item in VARIANTS]:
        group = by_variant.get(variant, [])
        aggregate: dict[str, Any] = {"variant": variant, "seed_count": len(group)}
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in group]
            aggregate[f"{metric}_mean"] = mean(values) if values else 0.0
            aggregate[f"{metric}_median"] = median(values) if values else 0.0
            aggregate[f"{metric}_std"] = _std(values)
        if output_kind == "action":
            aggregate["model_complexity_distribution_by_seed"] = " | ".join(
                str(row.get("model_complexity_distribution", "")) for row in group
            )
            aggregate["payload_strategy_distribution_by_seed"] = " | ".join(
                str(row.get("payload_strategy_distribution", "")) for row in group
            )
        out.append(aggregate)
    return out


def _build_paired_delta_rows(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(row["seed"]), str(row["variant"])): row for row in per_seed_rows}
    seeds = sorted({int(row["seed"]) for row in per_seed_rows})
    rows: list[dict[str, Any]] = []
    for left, right in PAIRWISE_COMPARISONS:
        for metric in IMPORTANT_DELTA_METRICS:
            deltas: list[float] = []
            left_values: list[float] = []
            right_values: list[float] = []
            for seed in seeds:
                left_value = _to_float(by_key[(seed, left)].get(metric))
                right_value = _to_float(by_key[(seed, right)].get(metric))
                deltas.append(left_value - right_value)
                left_values.append(left_value)
                right_values.append(right_value)
            ci_low, ci_high = _bootstrap_ci(deltas)
            rows.append(
                {
                    "comparison": f"{left}-{right}",
                    "metric": metric,
                    "seed_count": len(deltas),
                    "mean_delta": mean(deltas) if deltas else 0.0,
                    "median_delta": median(deltas) if deltas else 0.0,
                    "std_delta": _std(deltas),
                    "win_rate": sum(delta > 0 for delta in deltas) / max(len(deltas), 1),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "ci_crosses_zero": int(ci_low <= 0.0 <= ci_high),
                    "left_mean": mean(left_values) if left_values else 0.0,
                    "right_mean": mean(right_values) if right_values else 0.0,
                }
            )
    return rows


def _metric_mean(aggregate_rows: list[dict[str, Any]], variant: str, metric: str) -> float:
    row = next(item for item in aggregate_rows if item["variant"] == variant)
    return _to_float(row.get(f"{metric}_mean"))


def _delta_row(delta_rows: list[dict[str, Any]], comparison: str, metric: str) -> dict[str, Any]:
    return next(row for row in delta_rows if row["comparison"] == comparison and row["metric"] == metric)


def _write_manifest(output_dir: Path, seeds: list[int], slots: int, variants: list[str]) -> None:
    manifest = {
        "regime": REGIME_NAME,
        "slots": slots,
        "seeds": seeds,
        "variants": variants,
        "paired_design": True,
        "seed_field": "configs.dataset.seed",
        "constraints": [
            "reward core unchanged",
            "mission_prior_value_v2 unchanged",
            "utility_rule unchanged",
            "SafeFilter unchanged",
            "lifecycle utility not enabled",
            "adaptive exploration schedule not implemented",
            "score-side tiebreaker not implemented",
            "main configs not modified",
        ],
    }
    _write_text(output_dir / "run_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))


def _checkpoint(output_dir: Path, command: str, seeds: list[int], slots: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "git_status.txt", _run_text(["git", "status", "--short"]))
    _write_text(output_dir / "git_diff.patch", _run_text(["git", "diff"]))
    command_lines = [
        "# Command Log",
        "",
        f"- {datetime.now().isoformat(timespec='seconds')}: {command}",
        f"- seeds: {','.join(str(seed) for seed in seeds)}",
        f"- slots: {slots}",
        f"- {datetime.now().isoformat(timespec='seconds')}: python scripts/check_utility_guided_safe_linucb_equivalence.py --output-dir {output_dir}",
    ]
    _write_text(output_dir / "command_log.md", "\n".join(command_lines) + "\n")


def _append_command(output_dir: Path, text: str) -> None:
    path = output_dir / "command_log.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Command Log\n\n"
    _write_text(path, existing.rstrip() + f"\n- {datetime.now().isoformat(timespec='seconds')}: {text}\n")


def _warning_counts(per_seed_rows: list[dict[str, Any]], metric: str, candidate: str, reference: str) -> tuple[int, int]:
    by_key = {(int(row["seed"]), str(row["variant"])): row for row in per_seed_rows}
    seeds = sorted({int(row["seed"]) for row in per_seed_rows})
    warning_count = 0
    for seed in seeds:
        value = _to_float(by_key[(seed, candidate)].get(metric))
        ref = _to_float(by_key[(seed, reference)].get(metric))
        warning_count += int(_pct_degradation(value, ref) > 0.05)
    return warning_count, len(seeds)


def _write_reports(
    output_dir: Path,
    disabled_pass: bool,
    per_seed_rows: list[dict[str, Any]],
    paired_delta_rows: list[dict[str, Any]],
    scoring_summary: list[dict[str, Any]],
    action_summary: list[dict[str, Any]],
    safety_summary: list[dict[str, Any]],
    outcome_summary: list[dict[str, Any]],
    seeds: list[int],
    slots: int,
) -> None:
    ug035_safety = _metric_mean(safety_summary, "ug_wx035", "final_hard_safety_violation_count")
    legacy_safety = _metric_mean(safety_summary, "legacy_ref", "final_hard_safety_violation_count")
    hard_safety_delta = ug035_safety - legacy_safety
    utility_dom = _metric_mean(scoring_summary, "ug_wx035", "utility_prior_dominance_rate")
    exploration_dom = _metric_mean(scoring_summary, "ug_wx035", "safe_exploration_dominance_rate")
    direct_delta = _metric_mean(action_summary, "ug_wx035", "direct_roi_or_roictx_rate") - _metric_mean(
        action_summary, "ug_wx100", "direct_roi_or_roictx_rate"
    )
    defer_delta = _metric_mean(action_summary, "ug_wx035", "defer_rate") - _metric_mean(action_summary, "ug_wx100", "defer_rate")
    staged_delta = _metric_mean(action_summary, "ug_wx035", "staged_raw_action_rate") - _metric_mean(
        action_summary, "ug_wx100", "staged_raw_action_rate"
    )
    hard_pass = disabled_pass and hard_safety_delta == 0.0
    behavior_pass = (
        utility_dom >= 0.60
        and exploration_dom <= 0.25
        and direct_delta >= -0.05
        and defer_delta <= 0.05
        and staged_delta <= 0.02
    )
    net_warning_count, seed_count = _warning_counts(per_seed_rows, "net_system_utility", "ug_wx035", "ug_wx100")
    delivered_warning_count, _ = _warning_counts(per_seed_rows, "delivered_utility", "ug_wx035", "ug_wx100")
    net_warning_systematic = net_warning_count > seed_count / 2
    net_warning_label = "systematic tradeoff" if net_warning_systematic else "seed-sensitive behavior"

    net_delta = _delta_row(paired_delta_rows, "ug_wx035-ug_wx100", "net_system_utility")
    delivered_delta = _delta_row(paired_delta_rows, "ug_wx035-ug_wx100", "delivered_utility")
    utility_dom_delta = _delta_row(paired_delta_rows, "ug_wx035-ug_wx100", "utility_prior_dominance_rate")
    exploration_dom_delta = _delta_row(paired_delta_rows, "ug_wx035-ug_wx100", "safe_exploration_dominance_rate")
    direct_delta_row = _delta_row(paired_delta_rows, "ug_wx035-ug_wx100", "direct_roi_or_roictx_rate")
    wx050_net = _delta_row(paired_delta_rows, "ug_wx050-ug_wx100", "net_system_utility")
    wx050_utility_dom = _delta_row(paired_delta_rows, "ug_wx050-ug_wx100", "utility_prior_dominance_rate")

    wx035_stable = hard_pass and behavior_pass and not net_warning_systematic
    wx050_safer_compromise = _to_float(wx050_net["mean_delta"]) >= _to_float(net_delta["mean_delta"]) and _to_float(
        wx050_utility_dom["mean_delta"]
    ) < _to_float(utility_dom_delta["mean_delta"])

    if wx035_stable:
        recommendation = "Recommend ug_wx035 as the preferred default-off utility-guided experimental variant."
        later_variant = "ug_wx035"
    elif hard_pass and behavior_pass and net_warning_systematic:
        recommendation = "Keep ug_wx035 as a mechanism-aligned ablation variant only; do not adopt it as preferred."
        later_variant = "ug_wx050" if wx050_safer_compromise else "ug_wx035_ablation_only"
    elif wx050_safer_compromise:
        recommendation = "Recommend ug_wx050 as a conservative alternative; keep both variants default-off."
        later_variant = "ug_wx050"
    else:
        recommendation = "Do not adopt either static variant; keep legacy/main default and report diagnostics."
        later_variant = "legacy_ref"

    warning_lines = [
        "# Utility Warning Analysis",
        "",
        f"- seeds: {seed_count}",
        f"- ug_wx035 net_system_utility warnings vs ug_wx100: {net_warning_count}/{seed_count}",
        f"- ug_wx035 delivered_utility warnings vs ug_wx100: {delivered_warning_count}/{seed_count}",
        f"- net_system_utility warning classification: {net_warning_label}",
        f"- ug_wx035 net_system_utility mean delta vs ug_wx100: {_to_float(net_delta['mean_delta']):.6g}",
        f"- ug_wx035 net_system_utility CI crosses zero: {bool(_to_int(net_delta['ci_crosses_zero']))}",
        f"- ug_wx035 delivered_utility mean delta vs ug_wx100: {_to_float(delivered_delta['mean_delta']):.6g}",
        f"- ug_wx035 delivered_utility CI crosses zero: {bool(_to_int(delivered_delta['ci_crosses_zero']))}",
    ]
    _write_text(output_dir / "utility_warning_analysis.md", "\n".join(warning_lines) + "\n")

    report_lines = [
        "# Exploration Weight Multi-seed Report",
        "",
        f"- regime: {REGIME_NAME}",
        f"- seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- slots per seed: {slots}",
        "- patch type: diagnostic/report-only",
        "- reward core modified: no",
        "- mission_prior_value_v2 modified: no",
        "- utility_rule modified: no",
        "- SafeFilter modified: no",
        "- lifecycle utility enabled: no",
        "- adaptive schedule implemented: no",
        "- score-side tiebreaker implemented: no",
        "",
        "## Required Answers",
        f"1. Did disabled-equivalence remain intact? {'Yes' if disabled_pass else 'No'}.",
        f"2. Is ug_wx035 stable across paired seeds? {'Yes' if wx035_stable else 'No'}.",
        f"3. Was the previous net_system_utility warning seed noise or systematic? {net_warning_label}.",
        f"4. Does ug_wx035 consistently increase utility_prior_dominance? {'Yes' if _to_float(utility_dom_delta['mean_delta']) > 0 else 'No'}.",
        f"5. Does ug_wx035 consistently reduce safe_exploration_dominance? {'Yes' if _to_float(exploration_dom_delta['mean_delta']) < 0 else 'No'}.",
        f"6. Does ug_wx035 preserve direct ROI / ROI+Ctx behavior? {'Yes' if _to_float(direct_delta_row['mean_delta']) >= -0.05 else 'No'}.",
        f"7. Does ug_wx035 cause abnormal defer or staged overuse? {'Yes' if defer_delta > 0.05 or staged_delta > 0.02 else 'No'}.",
        f"8. Is ug_wx050 a safer compromise? {'Yes' if wx050_safer_compromise else 'No'}.",
        f"9. Which variant should be used for later lifecycle shadow evaluation? {later_variant}.",
        "10. Should adaptive exploration schedule still be deferred? Yes.",
        "",
        "## ug_wx035 Gates",
        f"- disabled-equivalence: {'PASS' if disabled_pass else 'FAIL'}",
        f"- hard safety violation delta vs legacy_ref: {hard_safety_delta:.6g}",
        f"- utility prior dominance mean: {utility_dom:.3f}",
        f"- safe exploration dominance mean: {exploration_dom:.3f}",
        f"- direct ROI/ROI+Ctx mean delta vs ug_wx100: {direct_delta:.3f}",
        f"- defer mean delta vs ug_wx100: {defer_delta:.3f}",
        f"- staged mean delta vs ug_wx100: {staged_delta:.3f}",
        "",
        "## Final Recommendation",
        recommendation,
        "No performance superiority claim is made unless supported by the paired metrics above.",
    ]
    _write_text(output_dir / "exploration_weight_multiseed_report.md", "\n".join(report_lines) + "\n")

    final_lines = [
        "# Final Recommendation",
        "",
        f"- disabled-equivalence: {'PASS' if disabled_pass else 'FAIL'}",
        f"- ug_wx035 safety gates: {'PASS' if hard_pass else 'FAIL'}",
        f"- ug_wx035 behavior gates: {'PASS' if behavior_pass else 'FAIL'}",
        f"- net_system_utility warning classification: {net_warning_label}",
        f"- recommended next variant for later lifecycle shadow evaluation: {later_variant}",
        f"- recommendation: {recommendation}",
        "- adaptive exploration schedule: deferred",
        "- score-side tiebreaker: not implemented",
        "- main default configs: unchanged",
    ]
    _write_text(output_dir / "final_recommendation.md", "\n".join(final_lines) + "\n")

    readme_lines = [
        "# Utility-Guided Exploration Weight Multi-seed Screen",
        "",
        "Paired multi-seed diagnostic for static exploration_weight variants.",
        "",
        "This block is report-only and default-off. It does not modify reward, SafeFilter, utility_rule, lifecycle utility, or main configs.",
        "",
        "Primary files:",
        "- per_seed_summary.csv",
        "- paired_delta_summary.csv",
        "- scoring_dominance_summary.csv",
        "- action_distribution_summary.csv",
        "- safety_guardrail_summary.csv",
        "- outcome_guardrail_summary.csv",
        "- utility_warning_analysis.md",
        "- exploration_weight_multiseed_report.md",
        "- final_recommendation.md",
    ]
    _write_text(output_dir / "README.md", "\n".join(readme_lines) + "\n")


def run_multiseed(output_dir: Path, seeds: list[int], slots: int, command: str) -> None:
    _checkpoint(output_dir, command, seeds, slots)
    variants = [name for name, _enabled, _wx in VARIANTS]
    _write_manifest(output_dir, seeds, slots, variants)

    disabled_pass = run_check(output_dir)
    if not disabled_pass:
        _write_text(
            output_dir / "FAILED_disabled_equivalence.md",
            "# Disabled Equivalence Failed\n\nMulti-seed sweep stopped before running variants.\n",
        )
        raise SystemExit(1)

    policy_by_variant = {name: _policy_repair(enabled, wx) for name, enabled, wx in VARIANTS}
    for variant, policy in policy_by_variant.items():
        _write_variant_config(output_dir, variant, policy, slots, seeds)

    per_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in variants:
            _append_command(output_dir, f"run seed={seed} variant={variant}")
            result, logs = _run_variant_seed(
                variant=variant,
                policy_repair=policy_by_variant[variant],
                slots=slots,
                seed=seed,
                output_dir=output_dir,
            )
            per_seed_rows.append(_per_seed_row(seed, variant, result, logs))

    by_seed_variant = {(int(row["seed"]), str(row["variant"])): row for row in per_seed_rows}
    for seed in seeds:
        legacy = by_seed_variant[(seed, "legacy_ref")]
        for variant in variants:
            row = by_seed_variant[(seed, variant)]
            row["raw_safety_delta_vs_legacy_ref"] = _to_float(row["raw_safety_violation_count"]) - _to_float(
                legacy["raw_safety_violation_count"]
            )
            row["hard_safety_delta_vs_legacy_ref"] = _to_float(row["final_hard_safety_violation_count"]) - _to_float(
                legacy["final_hard_safety_violation_count"]
            )

    paired_delta_rows = _build_paired_delta_rows(per_seed_rows)
    scoring_summary = _aggregate_rows(
        per_seed_rows,
        [
            "action_flip_rate_vs_legacy_shadow",
            "utility_prior_dominance_rate",
            "learned_correction_dominance_rate",
            "safe_exploration_dominance_rate",
            "avg_utility_prior_contribution",
            "avg_learned_correction_contribution",
            "avg_safe_exploration_contribution",
        ],
        "scoring",
    )
    action_summary = _aggregate_rows(
        per_seed_rows,
        ["staged_raw_action_rate", "direct_roi_or_roictx_rate", "metadata_only_rate", "defer_rate"],
        "action",
    )
    safety_summary = _aggregate_rows(
        per_seed_rows,
        [
            "raw_safety_violation_count",
            "final_hard_safety_violation_count",
            "deadline_miss_count",
            "queue_overflow_count",
            "raw_safety_delta_vs_legacy_ref",
            "hard_safety_delta_vs_legacy_ref",
        ],
        "safety",
    )
    outcome_summary = _aggregate_rows(
        per_seed_rows,
        [
            "net_system_utility",
            "delivered_utility",
            "effective_utility",
            "high_value_delivered_utility",
            "delivered_items",
            "expired_items",
            "completed_tasks",
            "deadline_satisfaction_ratio",
        ],
        "outcome",
    )

    _write_csv(output_dir / "per_seed_summary.csv", per_seed_rows)
    _write_csv(output_dir / "paired_delta_summary.csv", paired_delta_rows)
    _write_csv(output_dir / "scoring_dominance_summary.csv", scoring_summary)
    _write_csv(output_dir / "action_distribution_summary.csv", action_summary)
    _write_csv(output_dir / "safety_guardrail_summary.csv", safety_summary)
    _write_csv(output_dir / "outcome_guardrail_summary.csv", outcome_summary)
    _write_reports(
        output_dir,
        disabled_pass,
        per_seed_rows,
        paired_delta_rows,
        scoring_summary,
        action_summary,
        safety_summary,
        outcome_summary,
        seeds,
        slots,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed Utility-Guided SafeLinUCB exploration_weight screen.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--slots", type=int, default=120)
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = (
        "python scripts/run_utility_guided_exploration_weight_multiseed.py "
        f"--output-dir {args.output_dir} --seeds {','.join(str(seed) for seed in seeds)} --slots {args.slots}"
    )
    run_multiseed(args.output_dir, seeds, args.slots, command)


if __name__ == "__main__":
    main()
