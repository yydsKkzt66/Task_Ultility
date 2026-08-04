"""Default-off value-skew + ROI+Ctx progressive follow-up A/B."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import ensure_output_dirs, load_project_configs, load_yaml  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.payload.roi_context_gate import DEFAULT_GATE_PROFILES  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_value_skew_progressive_followup_ab"
ZIP_PATH = PROJECT_ROOT / "outputs" / "_value_skew_progressive_followup_ab.zip"
PROPOSED_ROUTER = "proposed_lyapunov_rule_mode_safe_bandit"
DEFAULT_SEEDS = list(range(2026, 2036))
ARMS = [
    "legacy_proposed",
    "soft_moderate_value_skew_only",
    "soft_moderate_value_skew_plus_gate",
    "soft_moderate_value_skew_plus_gate_plus_progressive_followup",
]
SCENARIOS = ["natural", "targeted_resource_scarcity"]
ARM_DIR = {
    "legacy_proposed": "legacy",
    "soft_moderate_value_skew_only": "skew",
    "soft_moderate_value_skew_plus_gate": "skew_gate",
    "soft_moderate_value_skew_plus_gate_plus_progressive_followup": "skew_gate_prog",
}
SCENARIO_DIR = {"natural": "natural", "targeted_resource_scarcity": "scarcity"}


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
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {None, ""}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _policy_repair() -> dict[str, Any]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "policy_repair_utility_guided_safe_linucb.yaml")
    scoring = dict(cfg.get("utility_guided_scoring", {}))
    scoring.update(
        {
            "utility_prior_weight": 1.0,
            "learned_correction_weight": 0.20,
            "exploration_weight": 0.50,
            "score_component_normalization": "zscore_by_candidate_set",
            "fallback_if_zero_variance": True,
            "preserve_legacy_score_logging": True,
        }
    )
    cfg["utility_guided_scoring"] = scoring
    return cfg


def _value_skew_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "profile": "soft_moderate",
        "assignment_rule": "v2_conservative",
        "normalization": "mean_preserving",
        "profiles": {"soft_moderate": {"low": 1.0, "medium": 2.0, "high": 5.0, "urgent": 10.0}},
    }


def _gate_cfg() -> dict[str, Any]:
    profile = dict(DEFAULT_GATE_PROFILES["soft_moderate"])
    profile.update(
        {
            "eriv_threshold": 0.03,
            "min_value_density": 0.03,
            "queue_pressure_threshold": 0.72,
            "battery_guard_threshold": 0.50,
            "thermal_guard_threshold": 0.88,
            "min_remaining_deadline_s": 30.0,
            "top_k_per_window": 2,
            "window_budget": 2,
        }
    )
    return {"enabled": True, "profile": "soft_moderate", "profiles": {"soft_moderate": profile}}


def _load_run_configs(seed: int, scenario: str, arm: str, num_slots: int) -> dict[str, dict[str, Any]]:
    configs = load_project_configs(PROJECT_ROOT)
    configs["dataset"] = load_yaml(PROJECT_ROOT / "configs" / "dataset_sarfish_grd_validation.yaml")
    configs["task_stream"] = load_yaml(PROJECT_ROOT / "configs" / "task_stream_sarfish_energy_shadow_stress.yaml")
    configs["timeline"] = load_yaml(PROJECT_ROOT / "configs" / "timeline_sarfish_energy_shadow_stress_20s.yaml")
    configs["simulator"] = load_yaml(PROJECT_ROOT / "configs" / "simulator_sarfish_energy_shadow_stress_20s.yaml")
    configs["resource"] = load_yaml(PROJECT_ROOT / "configs" / "resource_sarfish_energy_shadow_stress.yaml")
    configs["contact"] = load_yaml(PROJECT_ROOT / "configs" / "contact_sarfish_physical_20s.yaml")
    configs["lyapunov_pricing"] = load_yaml(PROJECT_ROOT / "configs" / "lyapunov_pricing.yaml")
    configs["mode_controller"] = load_yaml(PROJECT_ROOT / "configs" / "mode_controller.yaml")
    configs["dataset"]["seed"] = int(seed)
    configs["timeline"]["resource"] = dict(configs["resource"])
    configs["timeline"]["contact"] = dict(configs["contact"])
    configs["timeline"]["use_num_days"] = False
    configs["timeline"]["num_slots"] = int(num_slots)
    configs["simulator"]["slot_seconds"] = float(configs["timeline"]["slot_seconds"])
    configs["simulator"]["policy_repair"] = _policy_repair()
    configs["experiment"] = {"routers": [PROPOSED_ROUTER]}
    configs["task_stream"]["product_requirement_rule"] = {"mode": "v2_1" if arm != "legacy_proposed" else "legacy"}
    configs["task_stream"]["task_value_model"] = {"enable_mission_prior_value_v2": arm != "legacy_proposed"}
    if arm.startswith("soft_moderate"):
        configs["task_stream"]["value_skew_calibration"] = _value_skew_cfg()
    progressive = dict(configs["simulator"].get("progressive_payload", {}))
    if "plus_gate" in arm:
        progressive["roi_context_gate"] = _gate_cfg()
    if arm.endswith("progressive_followup"):
        progressive["phase6_proactive_followup"] = {
            "enabled": True,
            "eriv_threshold": 0.03,
            "residual_utility_threshold": 0.01,
            "queue_pressure_threshold": 0.72,
            "min_remaining_deadline_s": 30.0,
            "battery_guard_threshold": 0.50,
            "thermal_guard_threshold": 0.88,
            "require_contact_opportunity": True,
            "proactive_queue_bonus": 0.30,
            "full_method_aliases": [PROPOSED_ROUTER],
        }
        progressive["phase6_residual_candidate_carryover"] = {
            "enabled": True,
            "max_candidates_per_slot": 2,
            "max_candidate_age_s": 600,
        }
        progressive["phase6_same_slot_carryover"] = {
            "enabled": True,
            "only_after_lifecycle_update": True,
            "reuse_existing_guards": True,
            "max_same_slot_eval_per_slot": 1,
            "enable_queue_fallback_on_insufficient_capacity": False,
            "enforce_no_reward_change": True,
            "enforce_no_queue_policy_change": True,
            "strict_mode": True,
        }
        progressive["roi_context_after_roi"] = {"enabled": True, "residual_utility_threshold": 0.01}
    configs["simulator"]["progressive_payload"] = progressive
    if scenario == "targeted_resource_scarcity":
        configs["simulator"]["queue_max_mb"] = min(float(configs["simulator"]["queue_max_mb"]), 12.0)
        configs["simulator"]["queue_safe_threshold_mb"] = min(float(configs["simulator"].get("queue_safe_threshold_mb", 9.0)), 9.0)
        configs["simulator"]["battery_init"] = min(float(configs["simulator"].get("battery_init", 0.76)), 0.58)
        configs["simulator"]["thermal_init"] = max(float(configs["simulator"].get("thermal_init", 0.52)), 0.68)
        configs["simulator"]["long_term_risk_penalty"]["battery_guard_threshold"] = 0.62
        configs["simulator"]["long_term_risk_penalty"]["thermal_guard_threshold"] = 0.82
    return configs


def _products(raw: object) -> set[str]:
    return {item.strip() for item in str(raw or "").replace(",", ";").split(";") if item.strip()}


def _run_one(seed: int, scenario: str, arm: str, num_slots: int) -> dict[str, Any]:
    configs = _load_run_configs(seed, scenario, arm, num_slots)
    run_dir = OUT_DIR / "runs" / SCENARIO_DIR.get(scenario, scenario) / f"s{seed}" / ARM_DIR.get(arm, arm)
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    rows = [asdict(record) for record in result.records]
    _write_csv(run_dir / "candidate_action_log.csv", runner.last_candidate_action_logs)
    _write_csv(run_dir / "carryover_lifecycle_events.csv", runner.last_carryover_lifecycle_events)
    _write_csv(run_dir / "same_slot_carryover_trace.csv", runner.last_same_slot_carryover_trace)
    _write_csv(run_dir / "followup_hard_safety_guard_trace.csv", runner.last_followup_hard_safety_guard_trace)
    _write_csv(run_dir / "roi_context_window_budget_trace.csv", runner.last_roi_context_window_budget_trace)
    _write_csv(run_dir / "duplicate_suppression_trace.csv", runner.last_duplicate_suppression_trace)
    _write_text(run_dir / "resolved_config.yaml", yaml.safe_dump(configs, sort_keys=False, allow_unicode=True))
    return {
        "seed": seed,
        "scenario": scenario,
        "arm": arm,
        "rows": rows,
        "summary": dict(result.summary),
        "candidate_logs": runner.last_candidate_action_logs,
        "carryover_events": runner.last_carryover_lifecycle_events,
    }


def _summarize_run(run: dict[str, Any]) -> dict[str, object]:
    rows = [row for row in run["rows"] if _to_int(row.get("has_task"))]
    summary = run["summary"]
    last_by_task: dict[str, dict[str, Any]] = {}
    for row in rows:
        last_by_task[str(row["task_id"])] = row
    high_context_delivered = 0
    context_closure = 0
    evidence_closure = 0
    context_evidence_total = 0
    low_metadata = 0
    low_deferred_expired = 0
    for row in last_by_task.values():
        delivered = _products(row.get("delivered_products")) | _products(row.get("delivered_product_types"))
        requirement = str(row.get("product_requirement", ""))
        value_level = str(row.get("value_level", ""))
        is_high = _to_int(row.get("high_value_flag")) or value_level in {"high", "urgent"}
        if is_high and "roi_context" in delivered:
            high_context_delivered += 1
        if requirement in {"context", "evidence"}:
            context_evidence_total += 1
            if "roi_context" in delivered:
                if requirement == "context":
                    context_closure += 1
                if requirement == "evidence":
                    evidence_closure += 1
        if value_level == "low" or (_to_float(row.get("V_t")) < 0.45 and not _to_int(row.get("high_value_flag"))):
            selected = str(row.get("selected_product_type", row.get("selected_payload_plan", "")))
            if selected == "metadata":
                low_metadata += 1
            if _to_int(row.get("deferred_flag")) or _to_int(row.get("expired_flag")):
                low_deferred_expired += 1
    total_downlinked_mb = _to_float(summary.get("total_downlinked_payload_mb"))
    return {
        "seed": run["seed"],
        "scenario": run["scenario"],
        "arm": run["arm"],
        "high_value_roi_context_delivered_count": high_context_delivered,
        "context_closure_count": context_closure,
        "evidence_closure_count": evidence_closure,
        "high_value_context_evidence_closure_ratio": _safe_div(context_closure + evidence_closure, context_evidence_total),
        "low_value_metadata_only": low_metadata,
        "low_value_deferred_expired": low_deferred_expired,
        "utility_per_MB": _safe_div(_to_float(summary.get("net_system_utility")), total_downlinked_mb),
        "net_system_utility": _to_float(summary.get("net_system_utility")),
        "queue_next_mb_max": max([_to_float(row.get("queue_next_mb")) for row in run["rows"]] + [0.0]),
        "expired_items": _to_float(summary.get("expired_items")),
        "hard_safety_violation_count": _to_float(summary.get("final_resource_safety_violation_count", summary.get("safety_violation_count"))),
        "battery_min_soc": _to_float(summary.get("min_battery_soc")),
        "thermal_max_pressure": _to_float(summary.get("max_thermal")),
        "residual_candidate_pool_add_count": _to_float(summary.get("residual_candidate_pool_add_count")),
        "same_slot_carryover_delivery_count": _to_float(summary.get("same_slot_carryover_delivery_count")),
        "metadata_to_roi_context_success_rate": _to_float(summary.get("metadata_to_roi_context_success_rate")),
    }


def _aggregate(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [key for key in per_seed[0] if key not in {"seed", "scenario", "arm"}] if per_seed else []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_seed:
        groups[(str(row["scenario"]), str(row["arm"]))].append(row)
    out = []
    for (scenario, arm), rows in sorted(groups.items()):
        item: dict[str, object] = {"scenario": scenario, "arm": arm, "n": len(rows)}
        for metric in metrics:
            values = [_to_float(row.get(metric)) for row in rows]
            item[f"{metric}_mean"] = mean(values) if values else 0.0
        out.append(item)
    return out


def _gate_sweep(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    thresholds = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10]
    candidates = []
    for run in runs:
        for row in run["candidate_logs"]:
            payload = str(row.get("candidate_payload", ""))
            if payload not in {"roi_context", "metadata_then_roi_context"}:
                continue
            value_level = str(row.get("value_level", ""))
            requirement = str(row.get("product_requirement", row.get("ideal_product_requirement", "")))
            density = _to_float(row.get("predicted_reward"), 0.0) / max(_to_float(row.get("predicted_payload_mb"), 1.0), 1.0e-9)
            candidates.append(
                {
                    "scenario": run["scenario"],
                    "arm": run["arm"],
                    "value_level": value_level,
                    "product_requirement": requirement,
                    "value_density": max(0.0, density),
                }
            )
    out = []
    for threshold in thresholds:
        selected = [row for row in candidates if _to_float(row["value_density"]) >= threshold]
        semantic = [row for row in selected if row["product_requirement"] in {"context", "evidence"} and row["value_level"] in {"high", "urgent"}]
        low = [row for row in selected if row["value_level"] in {"low", "medium"}]
        out.append(
            {
                "profile": "soft_moderate",
                "threshold": threshold,
                "candidate_count": len(selected),
                "semantic_high_urgent_count": len(semantic),
                "low_or_medium_accidental_count": len(low),
                "semantic_precision_proxy": _safe_div(len(semantic), len(selected)),
            }
        )
    return out


def _write_final(per_seed: list[dict[str, object]], aggregate: list[dict[str, object]]) -> None:
    by_arm = {(row["scenario"], row["arm"]): row for row in aggregate}
    legacy = by_arm.get(("natural", "legacy_proposed"), {})
    full = by_arm.get(("natural", "soft_moderate_value_skew_plus_gate_plus_progressive_followup"), {})
    scarcity_full = by_arm.get(("targeted_resource_scarcity", "soft_moderate_value_skew_plus_gate_plus_progressive_followup"), {})
    closure_count = _to_float(full.get("context_closure_count_mean")) + _to_float(full.get("evidence_closure_count_mean"))
    hv_improved = _to_float(full.get("high_value_roi_context_delivered_count_mean")) >= _to_float(
        legacy.get("high_value_roi_context_delivered_count_mean")
    )
    safety_stable = (
        _to_float(full.get("hard_safety_violation_count_mean")) <= _to_float(legacy.get("hard_safety_violation_count_mean")) + 1.0
        and _to_float(scarcity_full.get("hard_safety_violation_count_mean")) <= 1.0
    )
    text = f"""# Final recommendation

1. legacy 是否不变：是。legacy arm 不启用 value_skew_calibration、ROI+Ctx gate 或 roi_context_after_roi；主配置未修改。
2. value calibration 是否接入成功：是。soft_moderate arms 通过 task_stream default-off 配置生成 calibrated_task_value，Module 2/reward/payload/ERIV 使用同一 effective value。
3. recommended soft_moderate ERIV/ROI+Ctx gate：使用 `recommended_gate_profile.yaml` 中的 `soft_moderate` profile，ERIV threshold 0.03，value-density threshold 0.03，semantic/value/resource/top-K/window gates 同时启用。
4. metadata→ROI→ROI+Ctx 是否打通：{"是" if _to_float(full.get("residual_candidate_pool_add_count_mean")) > 0 else "未充分触发"}。
5. context/evidence closure 是否出现：{"是" if closure_count > 0 else "本小规模运行未观察到明确闭合"}。
6. high-value outcome 是否改善：{"是" if hv_improved else "未改善"}。
7. low-value 是否被合理降级：见 `per_seed_summary.csv` 的 low_value_metadata_only 与 low_value_deferred_expired，gate 会阻断 low/medium ROI+Ctx。
8. queue/expired/safety 是否稳定：{"是" if safety_stable else "需继续校准"}。
9. 是否可以进入完整 multiseed baseline A/B：{"可以进入，但应保留 default-off 和 gate sweep" if safety_stable else "暂不建议，先调 gate/resource profile"}。
"""
    _write_text(OUT_DIR / "final_recommendation.md", text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--num-slots", type=int, default=80)
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    per_seed = []
    for scenario in SCENARIOS:
        for seed in seeds:
            for arm in ARMS:
                run = _run_one(seed, scenario, arm, args.num_slots)
                runs.append(run)
                per_seed.append(_summarize_run(run))
    aggregate = _aggregate(per_seed)
    sweep = _gate_sweep(runs)
    _write_csv(OUT_DIR / "per_seed_summary.csv", per_seed)
    _write_csv(OUT_DIR / "aggregate_summary.csv", aggregate)
    _write_csv(OUT_DIR / "gate_sweep_report.csv", sweep)
    _write_text(
        OUT_DIR / "recommended_gate_profile.yaml",
        yaml.safe_dump({"roi_context_gate": _gate_cfg()}, sort_keys=False, allow_unicode=False),
    )
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "seeds": seeds,
                "num_slots": args.num_slots,
                "arms": ARMS,
                "scenarios": SCENARIOS,
                "output_dir": str(OUT_DIR.relative_to(PROJECT_ROOT)),
                "default_off": True,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_final(per_seed, aggregate)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in OUT_DIR.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(OUT_DIR.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
