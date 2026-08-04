"""Run a small single-seed diagnostic for utility-guided SafeLinUCB scoring."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


PATCH_DIR = PROJECT_ROOT / "outputs" / "_utility_guided_safe_linucb_patch"
ROUTER_NAME = "proposed_lyapunov_rule_mode_safe_bandit"
REGIME_NAME = "energy_shadow_stress_20s"


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_fieldnames(rows) if rows else ["empty"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_stress_configs(policy_repair: dict[str, object], slots: int) -> dict[str, dict]:
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


def _run_policy(policy_name: str, policy_repair: dict[str, object], slots: int, output_dir: Path):
    configs = _load_stress_configs(policy_repair, slots)
    output_dirs = ensure_output_dirs(output_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    environment = SimulationEnvironment(configs)
    routers = build_routers(configs["experiment"], environment.action_space)
    runner = environment.build_episode_runner(timeline)
    result = runner.run(routers[0])
    write_episode_results([result], output_dirs)
    candidate_logs = [dict(row, policy=policy_name, regime=REGIME_NAME) for row in runner.last_candidate_action_logs]
    return result, candidate_logs


def _action_category(action_name: str) -> str:
    name = str(action_name).lower()
    if "metadata_then" in name:
        return "staged"
    if "delay" in name or "defer" in name:
        return "defer"
    if "roi_context" in name or "+roi" in name:
        return "direct_roi"
    if "metadata" in name:
        return "metadata"
    return "other"


def _build_flip_rows(score_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_slot: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in score_rows:
        if _to_int(row.get("considered_by_router")):
            by_slot[_to_int(row.get("slot"))].append(row)

    flip_rows: list[dict[str, object]] = []
    for slot, rows in sorted(by_slot.items()):
        legacy = next((row for row in rows if _to_int(row.get("selected_by_legacy_shadow"))), None)
        guided = next((row for row in rows if _to_int(row.get("selected_by_utility_guided_shadow"))), None)
        actual = next((row for row in rows if _to_int(row.get("selected_flag"))), None)
        if legacy is None or guided is None:
            continue
        legacy_action = str(legacy.get("candidate_action", ""))
        guided_action = str(guided.get("candidate_action", ""))
        flip_rows.append(
            {
                "regime": REGIME_NAME,
                "slot": slot,
                "task_id": legacy.get("task_id", ""),
                "task_value": legacy.get("task_value", ""),
                "legacy_shadow_action": legacy_action,
                "utility_guided_shadow_action": guided_action,
                "actual_selected_action": "" if actual is None else actual.get("candidate_action", ""),
                "legacy_category": _action_category(legacy_action),
                "utility_guided_category": _action_category(guided_action),
                "flip_vs_legacy": int(legacy_action != guided_action),
                "legacy_score_shadow": legacy.get("legacy_score_shadow", ""),
                "utility_guided_score_shadow": guided.get("utility_guided_score_shadow", ""),
                "legacy_predicted_reward": legacy.get("predicted_reward", ""),
                "utility_guided_predicted_reward": guided.get("predicted_reward", ""),
                "legacy_risk_discount": legacy.get("risk_discount", ""),
                "utility_guided_risk_discount": guided.get("risk_discount", ""),
                "legacy_payload_mb": legacy.get("predicted_payload_mb", ""),
                "utility_guided_payload_mb": guided.get("predicted_payload_mb", ""),
            }
        )
    return flip_rows


def _summarize_policy(policy_name: str, result) -> dict[str, object]:
    summary = dict(result.summary)
    records = result.records
    task_records = [record for record in records if record.has_task]
    staged_count = sum(1 for record in task_records if _action_category(record.raw_action) == "staged")
    summary_row: dict[str, object] = {
        "policy": policy_name,
        "regime": REGIME_NAME,
        "router": result.router_name,
        "task_decisions": len(task_records),
        "staged_raw_action_count": staged_count,
        "staged_raw_action_rate": staged_count / max(len(task_records), 1),
    }
    for key in (
        "total_effective_utility",
        "total_penalized_reward",
        "gross_task_utility",
        "net_system_utility",
        "safety_violation_count",
        "raw_safety_violation_count",
        "final_resource_safety_violation_count",
        "safety_correction_count",
        "task_deferred_count",
        "deadline_missed_count",
        "delivered_items",
        "expired_items",
        "dropped_items",
    ):
        summary_row[key] = summary.get(key, "")
    return summary_row


def _component_dominance(score_rows: list[dict[str, object]]) -> dict[str, object]:
    selected_rows = [row for row in score_rows if _to_int(row.get("selected_flag")) and _to_int(row.get("considered_by_router"))]
    if not selected_rows:
        return {
            "utility_prior_dominance_fraction": 0.0,
            "learned_nonzero_fraction": 0.0,
            "exploration_nonzero_fraction": 0.0,
        }
    utility_dominates = 0
    learned_nonzero = 0
    exploration_nonzero = 0
    for row in selected_rows:
        utility = abs(_to_float(row.get("utility_prior_weight")) * _to_float(row.get("utility_prior_norm")))
        learned = abs(_to_float(row.get("learned_correction_weight")) * _to_float(row.get("learned_correction_norm")))
        exploration = abs(_to_float(row.get("exploration_weight")) * _to_float(row.get("safe_exploration_norm")))
        utility_dominates += int(utility >= learned and utility >= exploration)
        learned_nonzero += int(learned > 1.0e-12)
        exploration_nonzero += int(exploration > 1.0e-12)
    return {
        "utility_prior_dominance_fraction": utility_dominates / len(selected_rows),
        "learned_nonzero_fraction": learned_nonzero / len(selected_rows),
        "exploration_nonzero_fraction": exploration_nonzero / len(selected_rows),
        "selected_rows": len(selected_rows),
    }


def _write_top_flips(path: Path, flip_rows: list[dict[str, object]]) -> None:
    flips = [row for row in flip_rows if _to_int(row.get("flip_vs_legacy"))]
    lines = ["# Top 20 Action Flips", ""]
    if not flips:
        lines.append("No utility-guided shadow action flips were observed.")
    else:
        lines.append("| slot | task | legacy shadow | utility-guided shadow | direction |")
        lines.append("|---:|---|---|---|---|")
        for row in flips[:20]:
            direction = f"{row['legacy_category']} -> {row['utility_guided_category']}"
            lines.append(
                f"| {row['slot']} | {row['task_id']} | {row['legacy_shadow_action']} | "
                f"{row['utility_guided_shadow_action']} | {direction} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    flip_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> str:
    task_decisions = len(flip_rows)
    flip_count = sum(_to_int(row.get("flip_vs_legacy")) for row in flip_rows)
    flip_rate = flip_count / max(task_decisions, 1)
    direction_counts = Counter(
        f"{row['legacy_category']} -> {row['utility_guided_category']}"
        for row in flip_rows
        if _to_int(row.get("flip_vs_legacy"))
    )
    legacy_staged = sum(1 for row in flip_rows if row.get("legacy_category") == "staged")
    guided_staged = sum(1 for row in flip_rows if row.get("utility_guided_category") == "staged")
    legacy_staged_rate = legacy_staged / max(task_decisions, 1)
    guided_staged_rate = guided_staged / max(task_decisions, 1)
    dominance = _component_dominance(score_rows)
    risk_suppressed = [
        row
        for row in score_rows
        if _to_int(row.get("considered_by_router"))
        and _to_float(row.get("predicted_payload_mb")) >= 2.0
        and _to_float(row.get("risk_discount"), 1.0) < 0.999
    ]
    legacy_summary = next(row for row in summary_rows if row["policy"] == "legacy")
    guided_summary = next(row for row in summary_rows if row["policy"] == "utility_guided")
    final_safety_delta = _to_float(guided_summary.get("final_resource_safety_violation_count")) - _to_float(
        legacy_summary.get("final_resource_safety_violation_count")
    )
    raw_safety_delta = _to_float(guided_summary.get("raw_safety_violation_count")) - _to_float(
        legacy_summary.get("raw_safety_violation_count")
    )

    if final_safety_delta > 0:
        outcome = "D: utility-guided scoring breaks legacy assumptions and should remain disabled."
    elif flip_rate <= 1.0e-9:
        outcome = "C: utility-guided scoring is inactive due to scaling/normalization issues."
    elif guided_staged_rate > legacy_staged_rate + 0.02:
        outcome = "B: utility-guided scoring is active but increases staged overuse risk."
    else:
        outcome = "A: utility-guided scoring is behaviorally active and aligned with task utility."

    utility_prior_uses_z = any(_to_int(row.get("utility_prior_uses_z_pricing")) for row in score_rows)
    learned_uses_z = any(_to_int(row.get("learned_correction_uses_z")) for row in score_rows)
    exploration_uses_z = any(_to_int(row.get("safe_exploration_uses_z")) for row in score_rows)
    lines = [
        "# Utility-Guided SafeLinUCB Patch Report",
        "",
        f"- regime: {REGIME_NAME}",
        f"- task decisions audited: {task_decisions}",
        f"- action flip rate vs legacy shadow: {flip_count}/{task_decisions} ({flip_rate:.3f})",
        f"- staged shadow rate: legacy {legacy_staged_rate:.3f}, utility-guided {guided_staged_rate:.3f}",
        f"- raw safety violation delta vs legacy full run: {raw_safety_delta:.0f}",
        f"- final hard safety violation delta vs legacy full run: {final_safety_delta:.0f}",
        f"- outcome classification: {outcome}",
        "",
        "## Flip Direction",
    ]
    if direction_counts:
        lines.extend(f"- {direction}: {count}" for direction, count in direction_counts.most_common())
    else:
        lines.append("- no flips observed")
    lines.extend(
        [
            "",
            "## Score Components",
            f"- utility prior dominance among selected actions: {dominance['utility_prior_dominance_fraction']:.3f}",
            f"- learned correction nonzero among selected actions: {dominance['learned_nonzero_fraction']:.3f}",
            f"- safe exploration nonzero among selected actions: {dominance['exploration_nonzero_fraction']:.3f}",
            f"- high-payload candidates with risk_discount < 1: {len(risk_suppressed)}",
            "",
            "## z_k Coupling Audit",
            f"- utility prior is z_k-priced: {str(utility_prior_uses_z).lower()}",
            "- utility prior path: StepRunner passes guidance_bundle to Predictor.predict_action; "
            "Predictor._compute_guided_predicted_reward reads lambda_E, lambda_L, lambda_Q, lambda_S and mode biases.",
            f"- learned correction uses z_k directly: {str(learned_uses_z).lower()}",
            "- learned correction path: theta_a^T x_t uses ContextVector only; current x_t has resource/task/link state, not explicit z_k or mode fields.",
            f"- safe exploration uses z_k/resource risk: {str(exploration_uses_z).lower()}",
            "- safe exploration path: alpha * exploration_scale * risk_discount * sigma; exploration_scale comes from mode guidance, "
            "risk_discount uses resource_context plus predicted action energy, latency, payload, thermal, queue, deadline, contact, and eclipse terms.",
            "",
            "## Safety",
            "- SafeFilter and hard guards were not modified.",
            f"- final hard safety violations increased: {str(final_safety_delta > 0).lower()}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcome


def run_diagnostic(output_dir: Path, slots: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    utility_cfg = load_yaml(PROJECT_ROOT / "configs" / "policy_repair_utility_guided_safe_linucb.yaml")
    legacy_cfg = copy.deepcopy(utility_cfg)
    legacy_cfg["enable_utility_guided_scoring"] = False
    legacy_cfg.setdefault("utility_guided_scoring", {})["preserve_legacy_score_logging"] = True

    config_used = {
        "regime": REGIME_NAME,
        "slots": int(slots),
        "legacy_policy_repair": legacy_cfg,
        "utility_guided_policy_repair": utility_cfg,
    }
    (output_dir / "utility_guided_config_used.yaml").write_text(
        yaml.safe_dump(config_used, sort_keys=False),
        encoding="utf-8",
    )

    legacy_result, legacy_logs = _run_policy("legacy", legacy_cfg, slots, output_dir / "diagnostic_runs" / "legacy")
    guided_result, guided_logs = _run_policy(
        "utility_guided",
        utility_cfg,
        slots,
        output_dir / "diagnostic_runs" / "utility_guided",
    )

    score_rows = [row for row in guided_logs if _to_int(row.get("considered_by_router"))]
    flip_rows = _build_flip_rows(score_rows)
    summary_rows = [_summarize_policy("legacy", legacy_result), _summarize_policy("utility_guided", guided_result)]
    flip_count = sum(_to_int(row.get("flip_vs_legacy")) for row in flip_rows)
    task_decisions = len(flip_rows)
    summary_by_regime = [
        {
            "regime": REGIME_NAME,
            "task_decisions": task_decisions,
            "action_flip_count": flip_count,
            "action_flip_rate": flip_count / max(task_decisions, 1),
            "legacy_staged_shadow_rate": mean([int(row.get("legacy_category") == "staged") for row in flip_rows]) if flip_rows else 0.0,
            "utility_guided_staged_shadow_rate": mean([int(row.get("utility_guided_category") == "staged") for row in flip_rows]) if flip_rows else 0.0,
            "legacy_final_resource_safety_violation_count": summary_rows[0].get("final_resource_safety_violation_count", ""),
            "utility_guided_final_resource_safety_violation_count": summary_rows[1].get("final_resource_safety_violation_count", ""),
        }
    ]

    _write_csv(output_dir / "score_component_audit.csv", score_rows)
    _write_csv(output_dir / "action_flip_vs_legacy.csv", flip_rows)
    _write_csv(output_dir / "summary_by_policy.csv", summary_rows)
    _write_csv(output_dir / "summary_by_regime.csv", summary_by_regime)
    _write_top_flips(output_dir / "top_20_action_flips.md", flip_rows)
    _write_report(output_dir / "utility_guided_patch_report.md", flip_rows, score_rows, summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run utility-guided SafeLinUCB diagnostic.")
    parser.add_argument("--output-dir", type=Path, default=PATCH_DIR)
    parser.add_argument("--slots", type=int, default=120)
    args = parser.parse_args()
    run_diagnostic(args.output_dir, args.slots)


if __name__ == "__main__":
    main()
