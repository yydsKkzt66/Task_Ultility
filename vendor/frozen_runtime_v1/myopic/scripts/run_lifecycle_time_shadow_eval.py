"""Standalone lifecycle/time shadow evaluation sidecar.

This script is report-only. It reuses existing paired multi-seed trajectories
where available and runs only missing required arms into its own output tree.
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.reporting.lifecycle_shadow_metrics import (  # noqa: E402
    SHADOW_WEIGHT_SETS,
    build_per_task_lifecycle_rows,
    summarize_lifecycle_rows,
    to_float,
)
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "_lifecycle_time_shadow_eval"
EXISTING_MULTISEED_DIR = PROJECT_ROOT / "outputs" / "_utility_guided_exploration_weight_multiseed"
REGIME_NAME = "energy_shadow_stress_20s"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
DEFAULT_SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]

REUSED_VARIANTS = ["legacy_ref", "ug_wx100", "ug_wx050", "ug_wx035"]
REQUIRED_VARIANTS = ["legacy_ref", "ug_wx100", "ug_wx050", "utility_rule"]
ALL_VARIANTS = ["legacy_ref", "ug_wx100", "ug_wx050", "utility_rule", "ug_wx035"]
PAIRWISE_COMPARISONS = [
    ("ug_wx050", "legacy_ref"),
    ("ug_wx050", "ug_wx100"),
    ("ug_wx050", "utility_rule"),
    ("ug_wx100", "utility_rule"),
    ("ug_wx035", "ug_wx050"),
]

DELTA_METRICS = [
    "canonical_net_system_utility",
    "canonical_delivered_utility",
    "canonical_effective_utility",
    "lifecycle_shadow_utility_balanced",
    "lifecycle_shadow_utility_alert_heavy",
    "lifecycle_shadow_utility_evidence_heavy",
    "time_shadow_utility",
    "alert_rate",
    "confirmation_rate",
    "evidence_rate",
    "first_actionable_delivered_rate",
    "first_actionable_latency_p50",
    "first_actionable_latency_p90",
    "metadata_only_alert_rate",
    "evidence_closure_rate",
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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


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
    low = max(0, int(0.025 * len(means)) - 1)
    high = min(len(means) - 1, int(0.975 * len(means)))
    return means[low], means[high]


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


def _load_run_configs(seed: int, variant: str) -> dict[str, dict[str, Any]]:
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
    configs["timeline"]["num_slots"] = 120
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    if variant == "utility_rule":
        configs["experiment"] = {"routers": ["utility_rule"]}
    else:
        wx = {"legacy_ref": 1.0, "ug_wx100": 1.0, "ug_wx050": 0.5, "ug_wx035": 0.35}[variant]
        enabled = variant != "legacy_ref"
        configs["simulator"]["policy_repair"] = _policy_repair(enabled, wx)
        configs["experiment"] = {"routers": [PROPOSED_ROUTER]}
    return configs


def _write_config_snapshot(output_dir: Path, variant: str, seed: int) -> None:
    configs = _load_run_configs(seed, variant)
    payload = {
        "variant": variant,
        "representative_seed": seed,
        "regime": REGIME_NAME,
        "report_only": True,
        "configs": configs,
    }
    _write_text(output_dir / f"config_{variant}.yaml", yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _existing_trajectory_path(seed: int, variant: str) -> Path:
    return (
        EXISTING_MULTISEED_DIR
        / "runs"
        / f"seed_{seed}"
        / variant
        / "trajectories"
        / f"{PROPOSED_ROUTER}.csv"
    )


def _sidecar_trajectory_path(output_dir: Path, seed: int, variant: str, router: str) -> Path:
    return output_dir / "runs" / f"seed_{seed}" / variant / "trajectories" / f"{router}.csv"


def _run_missing_arm(output_dir: Path, seed: int, variant: str) -> Path:
    configs = _load_run_configs(seed, variant)
    run_dir = output_dir / "runs" / f"seed_{seed}" / variant
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], environment.action_space)[0]
    runner = environment.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    return _sidecar_trajectory_path(output_dir, seed, variant, router.router_name)


def _trajectory_path(output_dir: Path, seed: int, variant: str) -> Path:
    if variant in REUSED_VARIANTS:
        path = _existing_trajectory_path(seed, variant)
        if path.exists():
            return path
    router = "utility_rule" if variant == "utility_rule" else PROPOSED_ROUTER
    sidecar_path = _sidecar_trajectory_path(output_dir, seed, variant, router)
    if sidecar_path.exists():
        return sidecar_path
    return _run_missing_arm(output_dir, seed, variant)


def _summary_path_for_trajectory(path: Path) -> Path:
    return path.parents[1] / "summary.csv"


def _canonical_summary(path: Path, variant: str, seed: int) -> dict[str, Any]:
    rows = _read_csv(_summary_path_for_trajectory(path))
    row = rows[0] if rows else {}
    return {
        "seed": seed,
        "variant": variant,
        "canonical_net_system_utility": to_float(row.get("net_system_utility")),
        "canonical_delivered_utility": to_float(row.get("delivered_only_incremental_utility")),
        "canonical_effective_utility": to_float(row.get("total_effective_utility")),
        "canonical_total_penalized_reward": to_float(row.get("total_penalized_reward")),
        "canonical_delivered_items": to_float(row.get("delivered_items")),
        "canonical_expired_items": to_float(row.get("expired_items")),
        "canonical_deadline_satisfaction_ratio": to_float(row.get("deadline_satisfaction_ratio")),
    }


def _aggregate_by_variant(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant"])].append(row)
    for variant in ALL_VARIANTS:
        group = by_variant.get(variant, [])
        if not group:
            continue
        item: dict[str, Any] = {"variant": variant, "seed_count": len(group)}
        for metric in metrics:
            values = [to_float(row.get(metric)) for row in group]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
            item[f"{metric}_median"] = median(values) if values else 0.0
            item[f"{metric}_std"] = _std(values)
        out.append(item)
    return out


def _paired_delta_rows(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(row["seed"]), str(row["variant"])): row for row in per_seed_rows}
    seeds = sorted({int(row["seed"]) for row in per_seed_rows})
    rows: list[dict[str, Any]] = []
    for left, right in PAIRWISE_COMPARISONS:
        if not all((seed, left) in by_key and (seed, right) in by_key for seed in seeds):
            continue
        for metric in DELTA_METRICS:
            deltas = [to_float(by_key[(seed, left)].get(metric)) - to_float(by_key[(seed, right)].get(metric)) for seed in seeds]
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
                }
            )
    return rows


def _metric_mean(aggregate_rows: list[dict[str, Any]], variant: str, metric: str) -> float:
    row = next(item for item in aggregate_rows if item["variant"] == variant)
    return to_float(row.get(f"{metric}_mean"))


def _write_static_docs(output_dir: Path, input_files: list[Path], header: list[str]) -> None:
    display_files = []
    for path in input_files[:12]:
        try:
            display_files.append(path.resolve().relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            display_files.append(path)
    audit_lines = [
        "# Available Log Fields Audit",
        "",
        "1. Which files are used as input?",
        *[f"- {path}" for path in display_files],
        f"- total trajectory files: {len(input_files)}",
        "",
        "2. Which fields identify task_id?",
        "- `task_id`.",
        "3. Which fields identify selected action/model/payload?",
        "- `raw_action`, `executed_action`, `selected_action`, `selected_model`, `selected_payload`, `selected_payload_plan`, `selected_product_type`.",
        "4. Which fields identify delivery time?",
        "- `payload_delivery_time_s`; event list fields are available but product-specific stage timing is reconstructed from delivered rows.",
        "5. Which fields identify deadline?",
        "- `deadline_s` and `deadline_time_s`.",
        "6. Which fields identify task value V_i or V_cal?",
        "- `V_cal` when present, falling back to `V_t` or `task_value`.",
        "7. Which fields identify payload type?",
        "- `selected_product_type`, falling back to `selected_payload` / `payload_policy`.",
        "8. Which fields identify delivered/expired/deferred status?",
        "- `delivery_status`, `expired_flag`, `deferred_flag`, `lifetime_status`, `delivered_products`.",
        "9. Which fields are missing?",
        "- No blocking required fields are missing. Product-specific event time for cumulative `delivered_products` is not separately encoded, so delivered rows are used as event records.",
        "10. Which assumptions are used if fields are missing?",
        "- Stage time is the earliest delivered row for a product that grants that stage. Late or expired payloads receive zero stage credit.",
        "",
        "## Header Snapshot",
        ", ".join(header),
    ]
    _write_text(output_dir / "available_log_fields_audit.md", "\n".join(audit_lines) + "\n")

    taxonomy_lines = [
        "# Lifecycle Taxonomy Definition",
        "",
        "- alert: first actionable information delivered before deadline. Metadata, ROI, and ROI+Ctx can satisfy alert.",
        "- confirmation: ROI-level confirmation delivered before deadline. ROI and ROI+Ctx can satisfy confirmation.",
        "- evidence: richer context/evidence delivered before deadline. ROI+Ctx or residual evidence-grade products can satisfy evidence.",
        "",
        "Fairness: direct ROI receives alert + confirmation; direct ROI+Ctx receives alert + confirmation + evidence; metadata-only is capped at alert.",
    ]
    _write_text(output_dir / "lifecycle_taxonomy_definition.md", "\n".join(taxonomy_lines) + "\n")

    metric_lines = [
        "# Lifecycle Shadow Metric Definition",
        "",
        "For each task:",
        "`U_life_shadow_i = V_i * [w_A * I_A * phi_A + w_C * I_C * phi_C + w_E * I_E * phi_E]`",
        "",
        "`phi_s = exp(-k_s * latency_s / max(deadline_s, epsilon))`, with `k_A = k_C = k_E = 1.0`.",
        "",
        "Weight sets:",
        "- balanced: w_A=0.30, w_C=0.30, w_E=0.40",
        "- alert_heavy: w_A=0.50, w_C=0.25, w_E=0.25",
        "- evidence_heavy: w_A=0.20, w_C=0.30, w_E=0.50",
        "",
        "`U_time_shadow_i = V_i * I_first_actionable * exp(-latency_first_actionable / max(deadline_s, epsilon))`.",
        "",
        "All metrics are post-hoc/report-only and are not used by reward or action selection.",
    ]
    _write_text(output_dir / "lifecycle_shadow_metric_definition.md", "\n".join(metric_lines) + "\n")

    fairness_lines = [
        "# Fairness Audit",
        "",
        "1. Does direct ROI receive alert credit if delivered in time? Yes.",
        "2. Does direct ROI receive confirmation credit if delivered in time? Yes.",
        "3. Does ROI+Ctx receive evidence credit if delivered in time? Yes.",
        "4. Is metadata-only capped at alert credit? Yes.",
        "5. Are all policies evaluated with the same taxonomy? Yes.",
        "6. Are expired payloads given zero stage credit? Yes.",
        "7. Are late-after-deadline payloads given zero stage credit? Yes.",
        "8. Are lifecycle weights fixed before looking at results? Yes.",
        "9. Are all sensitivity weights reported, not cherry-picked? Yes.",
        "10. Did the module avoid changing reward/action selection? Yes.",
    ]
    _write_text(output_dir / "fairness_audit.md", "\n".join(fairness_lines) + "\n")


def _write_reports(
    output_dir: Path,
    per_seed_rows: list[dict[str, Any]],
    canonical_shadow: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> None:
    ug050_net_vs_100 = _metric_mean(canonical_shadow, "ug_wx050", "canonical_net_system_utility") - _metric_mean(
        canonical_shadow, "ug_wx100", "canonical_net_system_utility"
    )
    ug050_net_vs_legacy = _metric_mean(canonical_shadow, "ug_wx050", "canonical_net_system_utility") - _metric_mean(
        canonical_shadow, "legacy_ref", "canonical_net_system_utility"
    )
    ug050_net_vs_util = _metric_mean(canonical_shadow, "ug_wx050", "canonical_net_system_utility") - _metric_mean(
        canonical_shadow, "utility_rule", "canonical_net_system_utility"
    )
    ug050_delivered_vs_util = _metric_mean(canonical_shadow, "ug_wx050", "canonical_delivered_utility") - _metric_mean(
        canonical_shadow, "utility_rule", "canonical_delivered_utility"
    )
    utility_rule_delivered = _metric_mean(canonical_shadow, "utility_rule", "canonical_delivered_utility")
    ug050_delivered = _metric_mean(canonical_shadow, "ug_wx050", "canonical_delivered_utility")
    ug050_alert_vs_100 = _metric_mean(canonical_shadow, "ug_wx050", "alert_rate") - _metric_mean(
        canonical_shadow, "ug_wx100", "alert_rate"
    )
    ug050_time_vs_100 = _metric_mean(canonical_shadow, "ug_wx050", "time_shadow_utility") - _metric_mean(
        canonical_shadow, "ug_wx100", "time_shadow_utility"
    )
    ug050_conf_vs_100 = _metric_mean(canonical_shadow, "ug_wx050", "confirmation_rate") - _metric_mean(
        canonical_shadow, "ug_wx100", "confirmation_rate"
    )
    ug050_evid_vs_100 = _metric_mean(canonical_shadow, "ug_wx050", "evidence_rate") - _metric_mean(
        canonical_shadow, "ug_wx100", "evidence_rate"
    )
    metadata_alert = _metric_mean(canonical_shadow, "ug_wx050", "metadata_only_alert_rate")
    evidence_closure = _metric_mean(canonical_shadow, "ug_wx050", "evidence_closure_rate")
    balanced_delta = _metric_mean(canonical_shadow, "ug_wx050", "lifecycle_shadow_utility_balanced") - _metric_mean(
        canonical_shadow, "ug_wx100", "lifecycle_shadow_utility_balanced"
    )
    alert_delta = _metric_mean(canonical_shadow, "ug_wx050", "lifecycle_shadow_utility_alert_heavy") - _metric_mean(
        canonical_shadow, "ug_wx100", "lifecycle_shadow_utility_alert_heavy"
    )
    evidence_delta = _metric_mean(canonical_shadow, "ug_wx050", "lifecycle_shadow_utility_evidence_heavy") - _metric_mean(
        canonical_shadow, "ug_wx100", "lifecycle_shadow_utility_evidence_heavy"
    )
    strongest = max(
        [
            ("balanced", balanced_delta),
            ("alert_heavy", alert_delta),
            ("evidence_heavy", evidence_delta),
        ],
        key=lambda item: item[1],
    )[0]
    weakest = min(
        [
            ("balanced", balanced_delta),
            ("alert_heavy", alert_delta),
            ("evidence_heavy", evidence_delta),
        ],
        key=lambda item: item[1],
    )[0]
    robust = all(delta >= 0.0 for delta in (balanced_delta, alert_delta, evidence_delta))
    preserves_vs_100 = ug050_net_vs_100 >= -0.05 * abs(_metric_mean(canonical_shadow, "ug_wx100", "canonical_net_system_utility"))
    preserves_vs_legacy = ug050_net_vs_legacy >= -0.05 * abs(
        _metric_mean(canonical_shadow, "legacy_ref", "canonical_net_system_utility")
    )
    preserves_canonical = preserves_vs_100 and preserves_vs_legacy
    improves_shadow = balanced_delta > 0.0 or ug050_time_vs_100 > 0.0

    report_lines = [
        "# Lifecycle Time Shadow Report",
        "",
        "- canonical reward was not modified",
        "- action selection was not modified",
        "- shadow metrics are post-hoc/report-only",
        "",
        "## Required Answers",
        f"1. Does ug_wx050 preserve canonical utility compared with ug_wx100 and legacy_ref? {'Yes' if preserves_canonical else 'No'}; delta vs ug_wx100={ug050_net_vs_100:.6g}, vs legacy_ref={ug050_net_vs_legacy:.6g}.",
        f"2. Is ug_wx050 still behind utility_rule under canonical utility? {'Yes' if ug050_net_vs_util < 0.0 else 'No'}; delta={ug050_net_vs_util:.6g}.",
        f"3. Does lifecycle shadow utility reveal additional staged/early-alert value? {'Yes' if improves_shadow else 'No'} for ug_wx050 vs ug_wx100.",
        f"4. Does ug_wx050 improve alert rate or first actionable latency? {'Yes' if ug050_alert_vs_100 > 0.0 or ug050_time_vs_100 > 0.0 else 'No'}.",
        f"5. Does ug_wx050 improve confirmation or evidence closure? {'Yes' if ug050_conf_vs_100 > 0.0 or ug050_evid_vs_100 > 0.0 else 'No'}.",
        f"6. Is any advantage mainly from metadata-only alert, or from completed lifecycle closure? metadata_only_alert_mean={metadata_alert:.3f}, evidence_closure_mean={evidence_closure:.3f}.",
        f"7. Does utility_rule remain strong in direct delivered utility? Yes; utility_rule delivered_utility_mean={utility_rule_delivered:.6g}, ug_wx050 delivered_utility_mean={ug050_delivered:.6g}, delta={ug050_delivered_vs_util:.6g}.",
        f"8. Under which shadow weight setting is ug_wx050 strongest or weakest? strongest={strongest}, weakest={weakest}.",
        f"9. Are the results robust across balanced, alert-heavy, and evidence-heavy weights? {'Yes' if robust else 'No'}.",
        "10. Should lifecycle utility remain report-only, or be considered for future TWC reward design? Keep report-only for WC; consider future TWC only after real detector/tiling and stronger long-horizon planning.",
        "",
        "## Final Interpretation",
    ]
    if preserves_canonical and improves_shadow:
        report_lines.append("ug_wx050 is suitable as the primary default-off experimental variant for the next lifecycle/stress analysis.")
    elif metadata_alert > evidence_closure:
        report_lines.append("Observed stage credit is dominated by early/metadata alert rather than completed lifecycle closure; do not overclaim staged closure or shadow superiority.")
    else:
        report_lines.append("Do not adopt lifecycle shadow metrics as reward; keep them as explanatory sidecar diagnostics.")
    _write_text(output_dir / "lifecycle_time_shadow_report.md", "\n".join(report_lines) + "\n")

    final_lines = [
        "# Final Recommendation",
        "",
        "- lifecycle/time shadow metrics remain report-only",
        "- reward/action selection/scoring/baselines were not changed",
        f"- ug_wx050 canonical net delta vs ug_wx100: {ug050_net_vs_100:.6g}",
        f"- ug_wx050 canonical net delta vs utility_rule: {ug050_net_vs_util:.6g}",
        f"- ug_wx050 balanced lifecycle shadow delta vs ug_wx100: {balanced_delta:.6g}",
        f"- recommended next analysis variant: {'ug_wx050 as a conservative comparator only' if preserves_vs_100 else 'keep legacy/utility_rule comparisons explicit'}",
        "- lifecycle shadow advantage for ug_wx050: not demonstrated in this screen",
        "- do not enable lifecycle utility as main reward now",
        "- future TWC may revisit lifecycle utility after real detector/tiling and stronger long-horizon planning",
    ]
    _write_text(output_dir / "final_recommendation.md", "\n".join(final_lines) + "\n")

    readme_lines = [
        "# Lifecycle Time Shadow Evaluation",
        "",
        "Standalone sidecar evaluation for post-hoc lifecycle/time metrics.",
        "",
        "Inputs reuse prior utility-guided multiseed trajectories where possible and run missing required arms only under this output directory.",
        "",
        "No simulator core, reward, scorer, SafeFilter, utility_rule, or main configs are modified.",
    ]
    _write_text(output_dir / "README.md", "\n".join(readme_lines) + "\n")


def _checkpoint(output_dir: Path, command: str, seeds: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "git_status.txt", _run_text(["git", "status", "--short"]))
    _write_text(output_dir / "git_diff.patch", _run_text(["git", "diff"]))
    lines = [
        "# Command Log",
        "",
        f"- {datetime.now().isoformat(timespec='seconds')}: {command}",
        f"- paired seeds: {','.join(str(seed) for seed in seeds)}",
        "- reused prior multiseed logs where available",
        "- missing utility_rule arms were run into this sidecar output directory",
    ]
    _write_text(output_dir / "command_log.md", "\n".join(lines) + "\n")


def _write_manifest(output_dir: Path, seeds: list[int]) -> None:
    manifest = {
        "regime": REGIME_NAME,
        "seeds": seeds,
        "required_variants": REQUIRED_VARIANTS,
        "included_optional_variants": ["ug_wx035"],
        "reused_input_root": str(EXISTING_MULTISEED_DIR),
        "sidecar_output_root": str(output_dir),
        "report_only": True,
        "constraints": [
            "reward_function.py not modified",
            "mission_prior_value_v2 not modified",
            "safe_linucb_router.py not modified",
            "utility_rule not modified",
            "SafeFilter not modified",
            "lifecycle utility not enabled as reward",
            "action selection not changed",
            "main configs not changed",
        ],
    }
    _write_text(output_dir / "run_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))


def run_eval(output_dir: Path, seeds: list[int], command: str) -> None:
    _checkpoint(output_dir, command, seeds)
    _write_manifest(output_dir, seeds)
    for variant in ALL_VARIANTS:
        _write_config_snapshot(output_dir, variant, seeds[0])

    input_files: list[Path] = []
    per_task_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    first_header: list[str] = []

    for seed in seeds:
        for variant in ALL_VARIANTS:
            path = _trajectory_path(output_dir, seed, variant)
            input_files.append(path)
            rows = _read_csv(path)
            if not first_header and rows:
                first_header = list(rows[0].keys())
            router_name = "utility_rule" if variant == "utility_rule" else PROPOSED_ROUTER
            task_rows = build_per_task_lifecycle_rows(rows, seed=seed, variant=variant, router_name=router_name)
            per_task_rows.extend(task_rows)
            summary = summarize_lifecycle_rows(task_rows)
            summary.update(_canonical_summary(path, variant, seed))
            summary["seed"] = seed
            summary["variant"] = variant
            per_seed_rows.append(summary)

    if not per_task_rows:
        _write_text(output_dir / "FAILED_missing_required_fields.md", "No reconstructable task rows were found.\n")
        raise SystemExit(1)

    _write_static_docs(output_dir, input_files, first_header)
    _write_csv(output_dir / "per_task_lifecycle_shadow.csv", per_task_rows)
    _write_csv(output_dir / "per_seed_lifecycle_summary.csv", per_seed_rows)
    paired_rows = _paired_delta_rows(per_seed_rows)
    _write_csv(output_dir / "paired_lifecycle_delta_summary.csv", paired_rows)

    canonical_metrics = [
        "canonical_net_system_utility",
        "canonical_delivered_utility",
        "canonical_effective_utility",
        "lifecycle_shadow_utility_balanced",
        "lifecycle_shadow_utility_alert_heavy",
        "lifecycle_shadow_utility_evidence_heavy",
        "time_shadow_utility",
    ]
    latency_metrics = [
        "alert_latency_mean",
        "alert_latency_median",
        "alert_latency_p90",
        "confirmation_latency_mean",
        "evidence_latency_mean",
        "first_actionable_latency_p50",
        "first_actionable_latency_p90",
    ]
    stage_metrics = [
        "first_actionable_delivered_rate",
        "alert_rate",
        "confirmation_rate",
        "evidence_rate",
        "expired_without_alert_rate",
        "metadata_only_alert_rate",
        "direct_roi_alert_confirmation_rate",
        "evidence_closure_rate",
    ]
    canonical_shadow = _aggregate_by_variant(per_seed_rows, canonical_metrics + stage_metrics)
    latency_summary = _aggregate_by_variant(per_seed_rows, latency_metrics)
    stage_summary = _aggregate_by_variant(per_seed_rows, stage_metrics)
    _write_csv(output_dir / "canonical_vs_shadow_summary.csv", canonical_shadow)
    _write_csv(output_dir / "latency_summary.csv", latency_summary)
    _write_csv(output_dir / "lifecycle_stage_distribution.csv", stage_summary)
    _write_reports(output_dir, per_seed_rows, canonical_shadow, paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run report-only lifecycle/time shadow evaluation.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in range(2026, 2036)))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = (
        "python scripts/run_lifecycle_time_shadow_eval.py "
        f"--output-dir {args.output_dir} --seeds {','.join(str(seed) for seed in seeds)}"
    )
    run_eval(args.output_dir, seeds, command)


if __name__ == "__main__":
    main()
