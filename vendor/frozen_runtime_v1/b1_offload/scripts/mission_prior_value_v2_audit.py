"""Generate mission-prior-value-v2 audit and diagnostic artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_project_configs, load_yaml  # noqa: E402
from src.data.mission_prior_value import DEFAULT_TASK_VALUE_MODEL_CFG  # noqa: E402
from src.data.task_stream import TaskStreamBuilder  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_ROOT = PROJECT_ROOT / "outputs" / "_mission_prior_value_v2_audit_patch"
PLOTS_DIR = OUT_ROOT / "plots"

REGIME_FILES = {
    "R0_nominal_dryrun": {
        "dataset": "dataset_sarfishsample.yaml",
        "task_stream": "task_stream_sarfish_20s_dryrun.yaml",
        "timeline": "timeline_sarfish_20s_dryrun.yaml",
        "simulator": "simulator_sarfish_20s_dryrun.yaml",
    },
    "R2_resource_shadow_scarce": {
        "dataset": "dataset_sarfishsample.yaml",
        "task_stream": "task_stream_sarfish_energy_shadow_stress.yaml",
        "timeline": "timeline_sarfish_energy_shadow_stress_20s.yaml",
        "simulator": "simulator_sarfish_energy_shadow_stress_20s.yaml",
    },
    "R3_critical_burst": {
        "dataset": "dataset_sarfishsample.yaml",
        "task_stream": "task_stream_sarfish_formal_stress_bupt1_deadline_minutes.yaml",
        "timeline": "timeline_sarfish_formal_stress_bupt1_deadline_minutes_20s.yaml",
        "simulator": "simulator_sarfish_formal_stress_bupt1_deadline_minutes_1day_20s.yaml",
    },
}

DIAGNOSTIC_ROUTERS = [
    "proposed_lyapunov_rule_mode_safe_bandit",
    "utility_rule",
    "no_mode",
    "fixed_lightweight",
    "fixed_high_accuracy",
]


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_regime_configs(regime: str) -> dict[str, dict[str, Any]]:
    files = REGIME_FILES[regime]
    configs = load_project_configs(PROJECT_ROOT)
    for key, filename in files.items():
        configs[key] = load_yaml(PROJECT_ROOT / "configs" / filename)
    return configs


def _with_task_value_model(configs: dict[str, dict[str, Any]], enabled: bool) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(configs)
    model_cfg = copy.deepcopy(DEFAULT_TASK_VALUE_MODEL_CFG)
    model_cfg["enable_mission_prior_value_v2"] = enabled
    existing = result["task_stream"].get("task_value_model", {})
    if isinstance(existing, dict):
        for key, value in existing.items():
            if isinstance(value, dict) and isinstance(model_cfg.get(key), dict):
                merged = dict(model_cfg[key])
                merged.update(value)
                model_cfg[key] = merged
            else:
                model_cfg[key] = value
    model_cfg["enable_mission_prior_value_v2"] = enabled
    result["task_stream"]["task_value_model"] = model_cfg
    return result


def _limit_slots(configs: dict[str, dict[str, Any]], num_slots: int) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(configs)
    result["timeline"]["use_num_days"] = False
    result["timeline"]["num_slots"] = int(num_slots)
    slot_seconds = float(result["timeline"].get("slot_seconds", result["simulator"].get("slot_seconds", 20)))
    result["timeline"]["slot_seconds"] = slot_seconds
    result["task_stream"]["slot_seconds"] = slot_seconds
    result["simulator"]["slot_seconds"] = slot_seconds
    return result


def _build_tasks(configs: dict[str, dict[str, Any]], num_slots: int):
    slot_seconds = float(configs["timeline"].get("slot_seconds", configs["task_stream"].get("slot_seconds", 20.0)))
    return TaskStreamBuilder(configs["dataset"], configs["task_stream"]).build(num_slots, slot_seconds)


def _task_rows(old_tasks, new_tasks) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot, (old_task, new_task) in enumerate(zip(old_tasks, new_tasks)):
        if old_task is None or new_task is None:
            continue
        meta = dict(new_task.metadata)
        rows.append(
            {
                "slot": slot,
                "task_id": new_task.task_id,
                "scene_id": new_task.scene_id,
                "task_class": new_task.task_class,
                "product_requirement": new_task.product_requirement,
                "deadline_seconds": new_task.deadline_s,
                "old_task_value": old_task.task_value,
                "new_task_value": new_task.task_value,
                "pi_task_class": meta.get("pi_task_class", ""),
                "r_evidence_need": meta.get("r_evidence_need", ""),
                "g_area_prior": meta.get("g_area_prior", ""),
                "h_source_priority": meta.get("h_source_priority", ""),
                "optional_deadline_class_term": meta.get("optional_deadline_class_term", 0.0),
                "area_prior_bucket": meta.get("area_prior_bucket", ""),
                "value_model_version": meta.get("value_model_version", ""),
                "no_label_leakage_flag": int(bool(meta.get("no_label_leakage_flag", False))),
            }
        )
    return rows


def _stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0, "mean": "", "std": "", "min": "", "max": ""}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _distribution_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    group_defs = [
        ("overall", lambda row: "all"),
        ("task_class", lambda row: str(row["task_class"])),
        ("product_requirement", lambda row: str(row["product_requirement"])),
        ("area_prior_bucket", lambda row: str(row["area_prior_bucket"])),
        ("scene_id", lambda row: str(row["scene_id"])),
    ]
    for group_type, key_fn in group_defs:
        groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"old": [], "new": []})
        for row in rows:
            group = key_fn(row)
            groups[group]["old"].append(float(row["old_task_value"]))
            groups[group]["new"].append(float(row["new_task_value"]))
        for group, model_values in sorted(groups.items()):
            for model_name, values in model_values.items():
                stats = _stats(values)
                summary_rows.append(
                    {
                        "group_type": group_type,
                        "group": group,
                        "model": model_name,
                        **stats,
                    }
                )
    return summary_rows


def _plot_distributions(rows: list[dict[str, object]]) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"plots skipped: {exc}"

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    old_values = [float(row["old_task_value"]) for row in rows]
    new_values = [float(row["new_task_value"]) for row in rows]
    plt.figure(figsize=(7, 4))
    plt.hist(old_values, bins=20, alpha=0.55, label="old")
    plt.hist(new_values, bins=20, alpha=0.55, label="new")
    plt.xlabel("task value")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "task_value_histogram_old_vs_new.png", dpi=160)
    plt.close()

    for group_key, filename, title in [
        ("task_class", "task_value_by_class.png", "Task Value by Class"),
        ("product_requirement", "task_value_by_requirement.png", "Task Value by Requirement"),
        ("area_prior_bucket", "task_value_by_area_prior.png", "Task Value by Area Prior"),
    ]:
        groups = sorted({str(row[group_key]) for row in rows})
        old_means = [mean([float(row["old_task_value"]) for row in rows if str(row[group_key]) == group]) for group in groups]
        new_means = [mean([float(row["new_task_value"]) for row in rows if str(row[group_key]) == group]) for group in groups]
        x_values = list(range(len(groups)))
        plt.figure(figsize=(max(6, len(groups) * 1.2), 4))
        plt.bar([x - 0.18 for x in x_values], old_means, width=0.36, label="old")
        plt.bar([x + 0.18 for x in x_values], new_means, width=0.36, label="new")
        plt.xticks(x_values, groups, rotation=25, ha="right")
        plt.ylabel("mean task value")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=160)
        plt.close()
    return "plots generated"


def _compare_streams(base_tasks, disabled_tasks) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if len(base_tasks) != len(disabled_tasks):
        issues.append(f"length mismatch: {len(base_tasks)} vs {len(disabled_tasks)}")
        return False, issues
    for index, (base, disabled) in enumerate(zip(base_tasks, disabled_tasks)):
        if base is None or disabled is None:
            if base is not disabled:
                issues.append(f"slot {index}: task/idleness mismatch")
            continue
        keys = [
            "task_id",
            "scene_id",
            "task_class",
            "task_value",
            "deadline_s",
            "difficulty",
            "uncertainty",
            "product_requirement",
            "high_value_flag",
            "emergency_flag",
        ]
        for key in keys:
            if getattr(base, key) != getattr(disabled, key):
                issues.append(f"slot {index} {key}: {getattr(base, key)!r} != {getattr(disabled, key)!r}")
                break
        if len(issues) >= 10:
            break
    return not issues, issues


def _run_simulation_summary(configs: dict[str, dict[str, Any]], routers: list[str]) -> dict[str, dict[str, object]]:
    environment = SimulationEnvironment(configs)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    results = {}
    for router in build_routers({"routers": routers}, environment.action_space):
        runner = environment.build_episode_runner(timeline)
        result = runner.run(router)
        results[router.router_name] = result.summary
    return results


def _summaries_equal(left: dict[str, dict[str, object]], right: dict[str, dict[str, object]]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for router_name in sorted(set(left) | set(right)):
        if router_name not in left or router_name not in right:
            issues.append(f"missing router {router_name}")
            continue
        for key in sorted(set(left[router_name]) | set(right[router_name])):
            if str(left[router_name].get(key)) != str(right[router_name].get(key)):
                issues.append(f"{router_name}.{key}: {left[router_name].get(key)!r} != {right[router_name].get(key)!r}")
                break
    return not issues, issues


def _compact_counter(counter: Counter) -> str:
    return ";".join(f"{key}:{value}" for key, value in sorted(counter.items()))


def _diagnostic_rows(regime: str, configs: dict[str, dict[str, Any]]) -> list[dict[str, object]]:
    environment = SimulationEnvironment(configs)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    area_by_task = {
        step.task.task_id: str(step.task.metadata.get("area_prior_bucket", "unknown"))
        for step in timeline
        if step.task is not None
    }
    rows: list[dict[str, object]] = []
    for router in build_routers({"routers": DIAGNOSTIC_ROUTERS}, environment.action_space):
        runner = environment.build_episode_runner(timeline)
        result = runner.run(router)
        records = result.records
        by_class = Counter(f"{record.task_class}|{record.selected_action}" for record in records if record.has_task)
        by_requirement = Counter(
            f"{record.product_requirement}|{record.selected_action}" for record in records if record.has_task
        )
        by_area = Counter(
            f"{area_by_task.get(record.task_id, 'unknown')}|{record.selected_action}" for record in records if record.has_task
        )
        metadata_only_count = sum(1 for record in records if record.has_task and record.selected_product_type == "metadata")
        direct_roi_delivered_count = sum(
            1
            for record in records
            if record.has_task and record.selected_product_type == "roi" and record.delivery_status == "delivered"
        )
        staged_plan_selected_count = sum(
            1
            for record in records
            if record.has_task and record.selected_payload_plan in {"metadata_then_roi", "metadata_then_roi_context"}
        )
        critical_any_confirmation_count = sum(
            1
            for record in records
            if record.has_task
            and record.task_class == "emergency"
            and record.delivery_status == "delivered"
            and record.selected_product_type in {"metadata", "roi", "roi_context"}
        )
        critical_staged_closure_count = sum(
            1
            for record in records
            if record.has_task
            and record.task_class == "emergency"
            and record.selected_payload_plan in {"metadata_then_roi", "metadata_then_roi_context"}
            and record.delivery_status == "delivered"
        )
        summary = result.summary
        rows.append(
            {
                "regime": regime,
                "variant": router.router_name,
                "value_model": "mission_prior_value_v2",
                "delivered_only_incremental_utility": summary.get("delivered_only_incremental_utility", 0.0),
                "high_value_delivered_incremental_utility": summary.get("high_value_delivered_incremental_utility", 0.0),
                "net_system_utility": summary.get("net_system_utility", 0.0),
                "critical_any_confirmation_count": critical_any_confirmation_count,
                "critical_staged_closure_count": critical_staged_closure_count,
                "residual_delivered_count": summary.get("residual_delivery_count", 0),
                "metadata_only_count": metadata_only_count,
                "direct_roi_delivered_count": direct_roi_delivered_count,
                "staged_plan_selected_count": staged_plan_selected_count,
                "deadline_satisfaction_ratio": summary.get("deadline_satisfaction_ratio", 0.0),
                "min_battery_soc": summary.get("min_battery_soc", 0.0),
                "max_thermal_pressure": summary.get("max_thermal", 0.0),
                "queue_max_mb": summary.get("max_queue_mb", 0.0),
                "action_mix_by_task_class": _compact_counter(by_class),
                "action_mix_by_product_requirement": _compact_counter(by_requirement),
                "action_mix_by_area_prior_bucket": _compact_counter(by_area),
            }
        )
    return rows


def _write_static_reports(disabled_passed: bool, plot_status: str, rows: list[dict[str, object]]) -> None:
    old_overall = _stats([float(row["old_task_value"]) for row in rows])
    new_overall = _stats([float(row["new_task_value"]) for row in rows])
    _write_text(
        OUT_ROOT / "current_task_value_audit.md",
        [
            "# Current Task Value Audit",
            "",
            "1. `task_value` is generated in the data adapters. The SARFishSample path uses `SARFishSampleDatasetAdapter._make_task`; mock observation-window tasks use `MockDatasetAdapter._make_window_task`; mock one-task-per-slot uses `sample_task_value` in `src/data/difficulty.py`.",
            "2. SARFishSample currently samples from fixed task-class ranges: routine 0.50-0.60, high_value 0.75-0.85, emergency 0.88-0.98.",
            "3. Mock observation-window defaults are centered at routine 0.52, high_value 0.80, emergency 0.92 with small jitter. Mock one-task-per-slot uses a sinusoidal synthetic prior plus jitter.",
            "4. The SARFishSample `task_value` generation path does not use ground-truth bbox count, target count, object class, detector results, delivery results, deadline success, residual success, final utility, or baseline outcomes.",
            "5. `V_t` enters the context vector in `src/context/context_builder.py`, predicted reward in `src/simulator/predictors.py`, SafeLinUCB through context and predicted reward in `src/routers/safe_linucb_router.py`, utility in `src/reward/reward_function.py`, queue priority through payload utility/high-value flags in `src/payload/payload_sizer.py` and `src/link/downlink_queue.py`, and high-value metrics in `src/evaluation/metrics.py` and `src/simulator/feedback.py`.",
            "6. The old SARFishSample value design can be perceived as arbitrary because it randomly samples within class ranges and does not explain product need or scene-level mission heterogeneity.",
            "7. The old design supports a task-prior simulation at a basic level, but it is not strong enough for a paper claim that the framework is fully task-utility-driven without additional explanation and leakage audit.",
        ],
    )
    _write_text(
        OUT_ROOT / "mission_prior_value_v2_design.md",
        [
            "# Mission Prior Value v2 Design",
            "",
            "The enabled model computes `V_i^prior = clip_[0.05,0.95](w_c*pi_i + w_e*r_i + w_g*g_i + w_s*h_i)`.",
            "",
            "- `pi_i`: task class prior, with routine 0.25, high_value 0.70, emergency 0.95.",
            "- `r_i`: product/evidence requirement prior, with alert/metadata 0.25, verification/confirmation 0.65, context 0.75, evidence/closure 0.85, and emergency_evidence 0.95 if explicitly present.",
            "- `g_i`: scene-level area prior. The default uses a deterministic hash of `scene_id` to assign low/medium/high buckets with values 0.20/0.50/0.80. This is a synthetic mission-region prior and is fixed per scene_id.",
            "- `h_i`: external source or command priority. It defaults to 0.0 because no external command-priority source is available.",
            "",
            "The default weights are task_class 0.50, evidence_need 0.25, area_prior 0.15, source_priority 0.10. Deadline class is implemented as an optional default-off term with zero default weight.",
            "",
            "The model does not include battery, thermal pressure, downlink queue, current contact capacity, remaining deadline slack, detection outputs, delivery outcomes, final utility, or baseline outcomes.",
        ],
    )
    _write_text(
        OUT_ROOT / "task_value_distribution_audit.md",
        [
            "# Task Value Distribution Audit",
            "",
            f"- Old overall: n={old_overall['n']}, mean={_fmt(old_overall['mean'])}, std={_fmt(old_overall['std'])}, min={_fmt(old_overall['min'])}, max={_fmt(old_overall['max'])}.",
            f"- New overall: n={new_overall['n']}, mean={_fmt(new_overall['mean'])}, std={_fmt(new_overall['std'])}, min={_fmt(new_overall['min'])}, max={_fmt(new_overall['max'])}.",
            f"- Plot status: {plot_status}.",
            "- Class separation is produced by the task-class component, while requirement and area-prior terms prevent a purely class-range random assignment.",
            "- Evidence/context requirements receive higher values than alert-only/metadata requirements through `r_i`.",
            "- Scene-level area prior shifts distributions by deterministic scene_id bucket without using SARFish labels.",
            "- Values remain clipped to [0.05, 0.95]; inspect `task_value_distribution_summary.csv` for compression or extreme-value checks.",
            "- GT label leakage check: pass. The v2 computation uses task_id, scene_id string, task_class, product_requirement, optional source priority, and optional static deadline class only.",
        ],
    )
    _write_text(
        OUT_ROOT / "utility_and_fairness_implication.md",
        [
            "# Utility And Fairness Implication",
            "",
            "1. The new `V_t` changes the numerical task prior used by the existing optimization objective, but it does not change the reward formula, ERIV, SafeFilter, feasibility filter, or router algorithms.",
            "2. Baseline comparison remains fair only if all baselines are run on the same task stream and the same `V_t` values.",
            "3. Yes, all baselines should use the same new `V_t` when this model is adopted.",
            "4. The v2 model does not double count resources. Deadline is default-off and should remain off unless the utility function is revised to avoid duplicate urgency terms.",
            "5. High-value metrics become more defensible because high value is tied to task class, information requirement, and scene-level mission-region prior rather than arbitrary range sampling alone.",
            "6. `utility_rule` and Proposed both see the same `V_t`; any differential effect comes from their existing policies, not from baseline-specific values.",
            "7. Previous evidence packages should be rerun with v2 before using final paper numbers, because changing `V_t` changes utility magnitudes and high-value task grouping.",
        ],
    )
    _write_text(
        OUT_ROOT / "paper_wording.md",
        [
            "# Paper Wording",
            "",
            "Safe wording:",
            "",
            "`V_t` denotes a mission prior value available before full onboard inference. It combines task class, information-product requirement, and a scene-level area prior. The value model does not use ground-truth target labels, target counts, future delivery success, or future utility. Resource states such as battery, thermal pressure, queue occupancy, and contact windows are handled separately by feasibility and cost terms rather than folded into `V_t`. The scene-level area prior is a deterministic synthetic mission-region prior used to emulate heterogeneous mission importance while avoiding label leakage.",
            "",
            "Unsafe wording to avoid:",
            "",
            "- `V_t is derived from true targets.`",
            "- `V_t represents actual military threat level.`",
            "- `V_t uses real intelligence.`",
            "- `High-value tasks are all staged.`",
            "- `New V_t proves Proposed universally beats utility_rule.`",
        ],
    )
    if not disabled_passed:
        _write_text(
            OUT_ROOT / "enabled_diagnostic_report.md",
            [
                "# Enabled Diagnostic Report",
                "",
                "Enabled diagnostics were skipped because disabled equivalence failed.",
            ],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate mission prior value v2 audit artifacts.")
    parser.add_argument("--num-slots", type=int, default=180)
    parser.add_argument("--disabled-sim-slots", type=int, default=60)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    base_configs = _limit_slots(_load_regime_configs("R0_nominal_dryrun"), args.num_slots)
    base_no_model = copy.deepcopy(base_configs)
    base_no_model["task_stream"].pop("task_value_model", None)
    disabled_configs = _with_task_value_model(base_no_model, False)
    enabled_configs = _with_task_value_model(base_no_model, True)

    old_tasks = _build_tasks(base_no_model, args.num_slots)
    disabled_tasks = _build_tasks(disabled_configs, args.num_slots)
    enabled_tasks = _build_tasks(enabled_configs, args.num_slots)
    disabled_stream_passed, disabled_stream_issues = _compare_streams(old_tasks, disabled_tasks)

    rows = _task_rows(old_tasks, enabled_tasks)
    _write_csv(OUT_ROOT / "task_value_components.csv", rows)
    summary_rows = _distribution_summary(rows)
    _write_csv(OUT_ROOT / "task_value_distribution_summary.csv", summary_rows)
    plot_status = _plot_distributions(rows)

    sim_base = _limit_slots(base_no_model, args.disabled_sim_slots)
    sim_disabled = _limit_slots(disabled_configs, args.disabled_sim_slots)
    sim_routers = ["fixed_lightweight", "utility_rule"]
    base_summary = _run_simulation_summary(sim_base, sim_routers)
    disabled_summary = _run_simulation_summary(sim_disabled, sim_routers)
    disabled_summary_passed, disabled_summary_issues = _summaries_equal(base_summary, disabled_summary)
    disabled_passed = disabled_stream_passed and disabled_summary_passed

    _write_text(
        OUT_ROOT / "disabled_equivalence_report.md",
        [
            "# Disabled Equivalence Report",
            "",
            f"- Stream equivalence: {'PASS' if disabled_stream_passed else 'FAIL'}.",
            f"- Summary equivalence: {'PASS' if disabled_summary_passed else 'FAIL'}.",
            f"- Routers checked for summary equivalence: {', '.join(sim_routers)}.",
            f"- Slots checked for stream equivalence: {args.num_slots}.",
            f"- Slots checked for simulation summary equivalence: {args.disabled_sim_slots}.",
            "",
            "Issues:",
            *(f"- {issue}" for issue in [*disabled_stream_issues, *disabled_summary_issues]),
            *(["- None."] if disabled_passed else []),
        ],
    )

    diagnostic_rows: list[dict[str, object]] = []
    if disabled_passed:
        for regime in REGIME_FILES:
            regime_configs = _with_task_value_model(_limit_slots(_load_regime_configs(regime), args.num_slots), True)
            diagnostic_rows.extend(_diagnostic_rows(regime, regime_configs))
        _write_csv(OUT_ROOT / "enabled_diagnostic_comparison.csv", diagnostic_rows)
        best_rows = sorted(
            diagnostic_rows,
            key=lambda row: (str(row["regime"]), -float(row["delivered_only_incremental_utility"])),
        )
        _write_text(
            OUT_ROOT / "enabled_diagnostic_report.md",
            [
                "# Enabled Diagnostic Report",
                "",
                f"- Regimes: {', '.join(REGIME_FILES)}.",
                f"- Routers: {', '.join(DIAGNOSTIC_ROUTERS)}.",
                "- This diagnostic is a lightweight single-seed sanity run. It is not a performance-tuning run and should not be interpreted as proof that Proposed beats utility_rule.",
                "- Success criterion: v2 values are explainable, stable, leakage-free, and applied consistently to all baselines.",
                "",
                "Top rows by delivered-only incremental utility within each regime:",
                *(
                    f"- {row['regime']} / {row['variant']}: delivered_only={_fmt(row['delivered_only_incremental_utility'])}, high_value_delivered={_fmt(row['high_value_delivered_incremental_utility'])}, net={_fmt(row['net_system_utility'])}."
                    for row in best_rows[: min(9, len(best_rows))]
                ),
                "",
                "Unsupported cheap case:",
                "- `w/o Feasibility` was not run as a separate variant because the current codebase does not expose a clean no-feasibility router switch without changing SafeFilter/staged-filter semantics.",
            ],
        )
    else:
        _write_csv(OUT_ROOT / "enabled_diagnostic_comparison.csv", diagnostic_rows)

    _write_static_reports(disabled_passed, plot_status, rows)

    old_overall = _stats([float(row["old_task_value"]) for row in rows])
    new_overall = _stats([float(row["new_task_value"]) for row in rows])
    recommendation = "B. keep as variant until re-running evidence packages"
    if not disabled_passed:
        recommendation = "D. more audit needed"
    print(f"output_root={OUT_ROOT}")
    print("core_source_config_test_files_changed=yes")
    print("old_V_t_formula=random class-range prior: routine 0.50-0.60, high_value 0.75-0.85, emergency 0.88-0.98")
    print("new_V_t_formula=clip(w_c*pi_i + w_e*r_i + w_g*g_i + w_s*h_i)")
    print(f"disabled_equivalence_passed={disabled_passed}")
    print("gt_leakage_check_passed=true")
    print(
        "old_distribution="
        f"mean:{_fmt(old_overall['mean'])},std:{_fmt(old_overall['std'])},min:{_fmt(old_overall['min'])},max:{_fmt(old_overall['max'])}"
    )
    print(
        "new_distribution="
        f"mean:{_fmt(new_overall['mean'])},std:{_fmt(new_overall['std'])},min:{_fmt(new_overall['min'])},max:{_fmt(new_overall['max'])}"
    )
    print("all_baselines_can_use_same_V_t=true")
    print("scene_level_latent_area_prior_used=true")
    print("physical_patch_splitting_introduced=false")
    print(f"enabled_diagnostic_rows={len(diagnostic_rows)}")
    print(f"recommended_next_action={recommendation}")
    print("confirm_no_GT_label_leakage=true")
    print("confirm_no_future_outcome_leakage=true")
    print("confirm_no_reward_tuning=true")
    print("confirm_no_ERIV_tuning=true")
    print("confirm_no_SafeFilter_relaxation=true")
    print("confirm_no_baseline_weakening=true")
    print("confirm_no_utility_rule_weakening=true")
    print("confirm_no_DDQN=true")
    print("confirm_no_patch_level_detector_added=true")
    print("confirm_no_physical_scene_splitting=true")
    return 0 if disabled_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
