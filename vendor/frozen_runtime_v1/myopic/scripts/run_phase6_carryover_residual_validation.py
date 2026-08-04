"""Default-off Phase 6/carryover residual validation for req-aware Module 2."""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_evidence_residual_closure_root_cause import (  # noqa: E402
    _build_per_task_trace,
    _candidate_pool_trace,
    _downlink_residual_trace,
    _funnel_summary,
    _selected_task_rows,
)
from scripts.run_product_requirement_aware_module2_ab import (  # noqa: E402
    DEFAULT_SEEDS,
    METRICS as BASE_METRICS,
    PROPOSED_ROUTER,
    _aggregate,
    _bootstrap_ci,
    _load_run_configs,
    _safe_div,
    _summarize_run,
    _to_float,
    _to_int,
)
from src.common.config import ensure_output_dirs  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.evaluation.report import write_episode_results  # noqa: E402
from src.payload.continue_request import phase5_continue_config  # noqa: E402
from src.payload.proactive_followup import (  # noqa: E402
    DEFAULT_FULL_METHOD_ALIASES,
    phase6_proactive_config,
    phase6_residual_candidate_carryover_config,
    phase6_same_slot_carryover_config,
)
from src.reporting.lifecycle_shadow_metrics import build_per_task_lifecycle_rows, summarize_lifecycle_rows  # noqa: E402
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_phase6_carryover_residual_validation"
ARM_A = "v21_reqaware_ug_wx050"
ARM_B = "v21_reqaware_phase6_carryover_ug_wx050"
ARMS = (ARM_A, ARM_B)
PAIRWISE = [(ARM_B, ARM_A)]
CORE_METRICS = [
    "residual_candidate_added_count",
    "followup_eligibility_evaluated_count",
    "followup_eligibility_passed_count",
    "followup_eligibility_blocked_count",
    "residual_payload_generated_count",
    "residual_payload_enqueued_count",
    "residual_payload_delivered_count",
    "evidence_closure_count",
    "evidence_closure_rate",
    "lifecycle_shadow_utility_balanced",
    "net_system_utility",
    "expired_items",
    "queue_overflow_count",
    "hard_safety_violation_count",
]
METRICS = list(dict.fromkeys([*BASE_METRICS, *CORE_METRICS]))


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


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout
    if completed.stderr:
        output += "\nSTDERR:\n" + completed.stderr
    return output.strip()


def _git_knows(path: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode == 0


def _synthetic_new_file_diff(path: str) -> str:
    file_path = PROJECT_ROOT / path
    if not file_path.exists():
        return ""
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = [
        f"diff --git a/{path} b/{path}",
        "new file mode 100644",
        "index 0000000..0000000",
        "--- /dev/null",
        f"+++ b/{path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    return "\n".join([*header, *[f"+{line}" for line in lines]])


def _audit_diff(paths: list[str]) -> str:
    parts: list[str] = []
    tracked = [path for path in paths if _git_knows(path)]
    if tracked:
        parts.append(_run_git(["diff", "--", *tracked]))
    for path in paths:
        if not _git_knows(path):
            parts.append(_synthetic_new_file_diff(path))
    return "\n\n".join(part for part in parts if part).strip()


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _enable_phase6_carryover(configs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    copied = copy.deepcopy(configs)
    simulator = copied.setdefault("simulator", {})
    progressive = simulator.setdefault("progressive_payload", {})
    progressive["phase6_proactive_followup"] = {
        "enabled": True,
        "eriv_threshold": 0.25,
        "residual_utility_threshold": 0.08,
        "max_proactive_residuals_per_slot": 1,
        "max_proactive_residuals_per_contact": 2,
        "queue_pressure_threshold": 0.65,
        "min_remaining_deadline_s": 60,
        "battery_guard_threshold": 0.60,
        "thermal_guard_threshold": 0.85,
        "require_contact_opportunity": True,
        "proactive_queue_bonus": 0.05,
    }
    progressive["phase6_residual_candidate_carryover"] = {
        "enabled": True,
        "re_evaluate_once_per_slot": True,
        "max_candidates_per_slot": 2,
        "max_candidate_age_s": 600,
        "require_metadata_delivered": True,
        "require_pending_residual": True,
        "require_positive_residual_utility": True,
        "remove_on_expired": True,
        "remove_on_closed": True,
        "remove_on_residual_delivered": True,
        "remove_on_duplicate_enqueued": False,
    }
    progressive["phase6_same_slot_carryover"] = {
        "enabled": False,
        "only_after_lifecycle_update": True,
        "reuse_existing_guards": True,
        "max_same_slot_eval_per_slot": 1,
        "enforce_no_reward_change": True,
        "enforce_no_queue_policy_change": True,
        "strict_mode": True,
    }
    return copied


def _load_validation_configs(seed: int, arm: str) -> dict[str, dict[str, Any]]:
    configs = _load_run_configs(seed, ARM_A)
    if arm == ARM_B:
        configs = _enable_phase6_carryover(configs)
    elif arm != ARM_A:
        raise ValueError(f"unknown arm: {arm}")
    return configs


def _gate_audit(configs: dict[str, dict[str, Any]], arm: str) -> dict[str, object]:
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    simulator_cfg = configs["simulator"]
    phase5_cfg = phase5_continue_config(simulator_cfg)
    phase6_cfg = phase6_proactive_config(simulator_cfg)
    carryover_cfg = phase6_residual_candidate_carryover_config(simulator_cfg)
    same_slot_cfg = phase6_same_slot_carryover_config(simulator_cfg)
    aliases = set(DEFAULT_FULL_METHOD_ALIASES)
    configured_aliases = phase6_cfg.get("full_method_aliases", ())
    if isinstance(configured_aliases, str):
        aliases.update(item.strip() for item in configured_aliases.split(",") if item.strip())
    else:
        try:
            aliases.update(str(item) for item in configured_aliases)
        except TypeError:
            pass
    router_name = router.router_name
    router_alias_allowed = router_name in aliases or (
        router_name.startswith("proposed_")
        and bool(getattr(router, "use_lyapunov_pricing", False))
        and bool(getattr(router, "use_mode_controller", False))
    )
    phase6_enabled = bool(phase6_cfg.get("enabled", False))
    carryover_cfg_enabled = bool(carryover_cfg.get("enabled", False))
    blockers = []
    if not phase6_enabled:
        blockers.append("phase6_proactive_followup.disabled")
    if not carryover_cfg_enabled:
        blockers.append("phase6_residual_candidate_carryover.disabled")
    if not router_alias_allowed:
        blockers.append("router_alias_not_allowed")
    return {
        "arm": arm,
        "router_name": router_name,
        "phase5_enabled": bool(phase5_cfg.get("enabled", False)),
        "phase6_enabled": phase6_enabled,
        "proactive_followup_enabled": phase6_enabled,
        "carryover_cfg_enabled": carryover_cfg_enabled,
        "same_slot_carryover_enabled": bool(same_slot_cfg.get("enabled", False)),
        "router_alias_allowed": router_alias_allowed,
        "carryover_enabled_for_current_branch": bool(phase6_enabled and carryover_cfg_enabled and router_alias_allowed),
        "primary_gate_blocker": ";".join(blockers) if blockers else "none",
        "source": "runtime copied experiment config",
    }


def _run_arm(seed: int, arm: str) -> dict[str, Any]:
    configs = _load_validation_configs(seed, arm)
    run_dir = OUT_DIR / "runs" / f"seed_{seed}" / arm
    output_dirs = ensure_output_dirs(run_dir)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    write_episode_results([result], output_dirs)
    records = [asdict(record) for record in result.records]
    for row in records:
        row["seed"] = seed
        row["arm"] = arm
        row["router"] = router.router_name
    lifecycle_rows = build_per_task_lifecycle_rows(records, seed=seed, variant=arm, router_name=router.router_name)
    return {
        "seed": seed,
        "arm": arm,
        "router": router.router_name,
        "records": records,
        "summary": dict(result.summary),
        "candidate_logs": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_candidate_action_logs],
        "queue_logs": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_queue_priority_logs],
        "carryover_events": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_carryover_lifecycle_events],
        "lifecycle_rows": lifecycle_rows,
        "lifecycle_summary": summarize_lifecycle_rows(lifecycle_rows),
    }


def _task_action_change(run: dict[str, Any], base: dict[str, Any] | None) -> float:
    if base is None:
        return 0.0
    rows = {row["task_id"]: row for row in run["records"] if _to_int(row.get("has_task"))}
    base_rows = {row["task_id"]: row for row in base["records"] if _to_int(row.get("has_task"))}
    total = changed = 0
    for task_id, row in rows.items():
        other = base_rows.get(task_id)
        if other is None:
            continue
        total += 1
        changed += int(row.get("selected_action") != other.get("selected_action"))
    return _safe_div(changed, total)


def _row_sum(records: list[dict[str, Any]], key: str) -> int:
    return sum(_to_int(row.get(key)) for row in records if _to_int(row.get("has_task")))


def _summarize_validation_run(run: dict[str, Any], action_change_rate: float) -> dict[str, object]:
    row = _summarize_run(run, action_change_rate)
    records = run["records"]
    row.update(
        {
            "residual_candidate_added_count": _row_sum(records, "residual_candidate_pool_add_count"),
            "followup_eligibility_evaluated_count": sum(
                int(
                    _to_int(record.get("proactive_followup_attempt_count"))
                    + _to_int(record.get("carryover_proactive_attempt_count"))
                    + _to_int(record.get("residual_candidate_re_evaluation_count"))
                    > 0
                )
                for record in records
                if _to_int(record.get("has_task"))
            ),
            "followup_eligibility_passed_count": sum(
                int(
                    _to_float(record.get("residual_utility_generated")) > 0.0
                    or _to_float(record.get("followup_payload_mb")) > 0.0
                )
                for record in records
                if _to_int(record.get("has_task"))
            ),
            "followup_eligibility_blocked_count": sum(
                int(
                    (
                        _to_int(record.get("proactive_followup_attempt_count"))
                        + _to_int(record.get("carryover_proactive_attempt_count"))
                        + _to_int(record.get("residual_candidate_re_evaluation_count"))
                    )
                    > 0
                    and _to_float(record.get("residual_utility_generated")) <= 0.0
                    and _to_float(record.get("followup_payload_mb")) <= 0.0
                )
                for record in records
                if _to_int(record.get("has_task"))
            ),
            "residual_payload_generated_count": sum(
                int(_to_float(record.get("residual_utility_generated")) > 0.0 or _to_float(record.get("followup_payload_mb")) > 0.0)
                for record in records
                if _to_int(record.get("has_task"))
            ),
            "residual_payload_enqueued_count": sum(
                int(_to_int(record.get("residual_enqueue_count")) > 0 or _to_int(record.get("proactive_followup_enqueue_count")) > 0)
                for record in records
                if _to_int(record.get("has_task"))
            ),
            "residual_payload_delivered_count": sum(
                int(_to_int(record.get("residual_delivery_count")) > 0 or _to_int(record.get("proactive_followup_delivery_count")) > 0)
                for record in records
                if _to_int(record.get("has_task"))
            ),
        }
    )
    return row


def _paired(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(int(row["seed"]), str(row["arm"])): row for row in per_seed}
    seeds = sorted({int(row["seed"]) for row in per_seed})
    out: list[dict[str, object]] = []
    for left, right in PAIRWISE:
        for metric in METRICS:
            deltas = [
                _to_float(by_key[(seed, left)].get(metric)) - _to_float(by_key[(seed, right)].get(metric))
                for seed in seeds
                if (seed, left) in by_key and (seed, right) in by_key
            ]
            low, high = _bootstrap_ci(deltas)
            out.append(
                {
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "mean_delta": mean(deltas) if deltas else 0.0,
                    "median_delta": median(deltas) if deltas else 0.0,
                    "std_delta": stdev(deltas) if len(deltas) >= 2 else 0.0,
                    "win_rate": _safe_div(sum(1 for value in deltas if value > 0.0), len(deltas)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "ci_crosses_zero": int(low <= 0.0 <= high),
                    "n": len(deltas),
                }
            )
    return out


def _first_zero(funnel_row: dict[str, object]) -> str:
    for metric in [
        "residual_candidate_added_count",
        "followup_eligibility_evaluated_count",
        "residual_payload_generated_count",
        "residual_payload_enqueued_count",
        "residual_payload_delivered_count",
        "evidence_closure_count",
    ]:
        if _to_float(funnel_row.get(metric)) <= 0.0:
            return metric
    return "none"


def _metric(rows: list[dict[str, object]], arm: str, metric: str) -> float:
    values = [_to_float(row.get(metric)) for row in rows if row.get("arm") == arm]
    return mean(values) if values else 0.0


def _delta(paired: list[dict[str, object]], metric: str) -> float:
    for row in paired:
        if row.get("comparison") == f"{ARM_B} - {ARM_A}" and row.get("metric") == metric:
            return _to_float(row.get("mean_delta"))
    return 0.0


def _dominant_blocker(blocker_rows: list[dict[str, object]]) -> str:
    filtered = [row for row in blocker_rows if _to_int(row.get("count")) > 0 and ":none_observed" not in str(row.get("blocker_reason"))]
    if not filtered:
        return "none_observed"
    return str(max(filtered, key=lambda row: _to_int(row.get("count"))).get("blocker_reason"))


def _split_reasons(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in text.replace(",", ";").split(";") if item.strip()]


def _followup_blocker_summary(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    category = {
        "metadata_not_delivered": "followup eligibility blocker",
        "no_pending_residual_product": "followup eligibility blocker",
        "multiple_pending_residual_products": "followup eligibility blocker",
        "no_useful_remaining_deadline": "followup feasibility blocker",
        "residual_utility_below_threshold": "followup eligibility blocker",
        "eriv_below_threshold": "followup feasibility blocker",
        "duplicate_residual_already_queued": "followup duplicate blocker",
        "queue_pressure_too_high": "followup feasibility blocker",
        "battery_unsafe": "followup safety/resource blocker",
        "thermal_unsafe": "followup safety/resource blocker",
        "no_contact_opportunity": "followup feasibility blocker",
        "z_k_guided_followup_permission_blocked": "followup z-guidance blocker",
        "phase6_disabled": "router/config gate",
        "variant_not_full_proposed": "router/config gate",
    }
    counter: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    evaluated = 0
    for row in records:
        if not _to_int(row.get("has_task")):
            continue
        attempts = (
            _to_int(row.get("proactive_followup_attempt_count"))
            + _to_int(row.get("carryover_proactive_attempt_count"))
            + _to_int(row.get("residual_candidate_re_evaluation_count"))
        )
        if attempts <= 0:
            continue
        evaluated += 1
        reasons = _split_reasons(row.get("proactive_followup_blockers"))
        if not reasons:
            reason = str(row.get("proactive_followup_reason_code", "")).strip()
            reasons = [reason] if reason and reason != "eligible" else ["eligible_no_payload_generated"]
        for reason in reasons:
            counter[reason] += 1
            task_id = str(row.get("task_id", ""))
            if task_id and len(examples[reason]) < 8:
                examples[reason].append(task_id)
    total = max(sum(counter.values()), 1)
    rows = [
        {
            "blocker_reason": reason,
            "count": count,
            "percentage": _safe_div(count, total),
            "affected_evaluations": count,
            "evaluated_task_rows": evaluated,
            "example_task_ids": ";".join(examples[reason]),
            "first_stage_where_detected": "followup_eligibility_evaluated",
            "blocker_category": category.get(reason, "followup eligibility blocker"),
        }
        for reason, count in counter.most_common()
    ]
    for required in [
        "eriv_below_threshold",
        "no_contact_opportunity",
        "no_useful_remaining_deadline",
        "queue_pressure_too_high",
        "battery_unsafe",
        "thermal_unsafe",
    ]:
        if required not in counter:
            rows.append(
                {
                    "blocker_reason": required,
                    "count": 0,
                    "percentage": 0.0,
                    "affected_evaluations": 0,
                    "evaluated_task_rows": evaluated,
                    "example_task_ids": "",
                    "first_stage_where_detected": "none_observed",
                    "blocker_category": category[required],
                }
            )
    return rows


def _markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def _write_config_snapshots(seeds: list[int]) -> None:
    for arm, filename in [
        (ARM_A, "config_v21_reqaware_ug_wx050.yaml"),
        (ARM_B, "config_v21_reqaware_phase6_carryover_ug_wx050.yaml"),
    ]:
        payload = {
            "arm": arm,
            "representative_seed": seeds[0],
            "default_off_experiment_only": True,
            "configs": _load_validation_configs(seeds[0], arm),
        }
        _write_text(OUT_DIR / filename, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def _write_audit_docs(
    command: str,
    seeds: list[int],
    gates: list[dict[str, object]],
    funnel: list[dict[str, object]],
    blocker_rows: list[dict[str, object]],
    per_seed: list[dict[str, object]],
    paired: list[dict[str, object]],
) -> None:
    arm_b_funnel = next((row for row in funnel if row.get("arm") == ARM_B), {})
    candidate_added = _to_float(arm_b_funnel.get("residual_candidate_added_count"))
    evaluated = _to_float(arm_b_funnel.get("followup_eligibility_evaluated_count"))
    generated = _to_float(arm_b_funnel.get("residual_payload_generated_count"))
    enqueued = _to_float(arm_b_funnel.get("residual_payload_enqueued_count"))
    delivered = _to_float(arm_b_funnel.get("residual_payload_delivered_count"))
    evidence = _to_float(arm_b_funnel.get("evidence_closure_count"))
    first_zero = _first_zero(arm_b_funnel)
    dominant = _dominant_blocker(blocker_rows)
    utility_delta = _delta(paired, "net_system_utility")
    lifecycle_delta = _delta(paired, "lifecycle_shadow_utility_balanced")
    safety_delta = _delta(paired, "hard_safety_violation_count")
    queue_delta = _delta(paired, "queue_overflow_count")
    if candidate_added > 0 and evaluated > 0 and enqueued <= 0:
        case = "Case A"
        recommendation = "Phase 6 gate is open; next blocker is follow-up eligibility or feasibility. Audit the dominant blocker, not Module 2."
    elif candidate_added > 0 and enqueued > 0 and delivered <= 0:
        case = "Case B"
        recommendation = "Residual generation/enqueue works; next blocker is downlink queue delivery, deadline, capacity, or priority."
    elif delivered > 0 and evidence <= 0:
        case = "Case C"
        recommendation = "Residual delivery works; next blocker is information sufficiency or closure accounting."
    elif evidence > 0:
        case = "Case D"
        recommendation = "Phase 6 carryover advances residual/evidence closure; keep default-off experimental only."
    else:
        case = "unclassified"
        recommendation = "Residual chain did not advance past the current first-zero stage; keep disabled and audit that stage."
    if utility_delta < -0.05 or safety_delta > 0:
        case = "Case E"
        recommendation = "Do not integrate. Keep disabled and inspect queue/deadline/resource pressure."
    utility_state = "improved" if utility_delta > 0.05 else "degraded" if utility_delta < -0.05 else "stable"
    safety_state = "worsened" if safety_delta > 0 else "normal"

    _write_text(
        OUT_DIR / "README.md",
        f"""# Phase 6 carryover residual validation

Default-off experiment comparing:

- A: `{ARM_A}`
- B: `{ARM_B}`

Only the B arm enables existing Phase 6 proactive follow-up and residual candidate carryover runtime config. Main defaults, reward, SafeFilter, utility_rule, lifecycle closure, and Module 2 scoring are not edited by this script.
""",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": seeds,
                "arms": list(ARMS),
                "output_dir": OUT_DIR.as_posix(),
                "default_off_experiment_only": True,
                "behavior_change_scope": "runtime copied config for B arm only",
                "gates": gates,
                "paired_comparison": f"{ARM_B} - {ARM_A}",
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(
        OUT_DIR / "carryover_config_audit.md",
        "# Carryover config audit\n\n"
        + _markdown_table(
            gates,
            [
                "arm",
                "phase5_enabled",
                "phase6_enabled",
                "proactive_followup_enabled",
                "carryover_cfg_enabled",
                "same_slot_carryover_enabled",
                "router_alias_allowed",
                "carryover_enabled_for_current_branch",
                "primary_gate_blocker",
            ],
        )
        + "\n",
    )
    _write_text(
        OUT_DIR / "phase6_carryover_validation_report.md",
        f"""# Phase 6 carryover validation report

## Core result

- residual_candidate_added_count: A={_metric(per_seed, ARM_A, 'residual_candidate_added_count'):.6f}, B={_metric(per_seed, ARM_B, 'residual_candidate_added_count'):.6f}, delta={_delta(paired, 'residual_candidate_added_count'):.6f}
- followup_eligibility_evaluated_count: A={_metric(per_seed, ARM_A, 'followup_eligibility_evaluated_count'):.6f}, B={_metric(per_seed, ARM_B, 'followup_eligibility_evaluated_count'):.6f}, delta={_delta(paired, 'followup_eligibility_evaluated_count'):.6f}
- residual_payload_generated_count: B={generated:.0f}
- residual_payload_enqueued_count: B={enqueued:.0f}
- residual_payload_delivered_count: B={delivered:.0f}
- evidence_closure_count: B={evidence:.0f}
- evidence_closure_rate delta: {_delta(paired, 'evidence_closure_rate'):.6f}
- lifecycle_shadow_utility_balanced delta: {lifecycle_delta:.6f}
- net_system_utility delta: {utility_delta:.6f}
- expired_items delta: {_delta(paired, 'expired_items'):.6f}
- queue_overflow_count delta: {queue_delta:.6f}
- hard_safety_violation_count delta: {safety_delta:.6f}

## Interpretation

- selected case: {case}
- new first-zero stage: {first_zero}
- dominant blocker: {dominant}
- canonical utility: {utility_state}
- safety: {safety_state}

## Required answers

1. candidate_added became >0: {'yes' if candidate_added > 0 else 'no'}
2. follow-up eligibility evaluated: {'yes' if evaluated > 0 else 'no'}
3. residual payload generated: {'yes' if generated > 0 else 'no'}
4. residual payload enqueued: {'yes' if enqueued > 0 else 'no'}
5. residual payload delivered: {'yes' if delivered > 0 else 'no'}
6. evidence closure nonzero: {'yes' if evidence > 0 else 'no'}
7. if blocked, new first-zero stage: {first_zero}
8. canonical utility: {utility_state}
9. safety normal: {'yes' if safety_state == 'normal' else 'no'}
10. default-off experimental branch only: yes
""",
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        f"""# Final recommendation

{recommendation}

Keep `{ARM_B}` as a default-off experimental branch only. Do not make it the main default.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    command = "python scripts/run_phase6_carryover_residual_validation.py --seeds " + ",".join(str(seed) for seed in seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_config_snapshots(seeds)

    gates = [_gate_audit(_load_validation_configs(seeds[0], arm), arm) for arm in ARMS]
    gate_by_arm = {str(row["arm"]): row for row in gates}
    run_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    all_candidate_logs: list[dict[str, object]] = []
    for seed in seeds:
        for arm in ARMS:
            run = _run_arm(seed, arm)
            run_by_key[(seed, arm)] = run
            all_records.extend(run["records"])
            all_candidate_logs.extend(run["candidate_logs"])

    runs = [run_by_key[key] for key in sorted(run_by_key)]
    task_rows = _selected_task_rows(runs)
    traces: list[dict[str, object]] = []
    for arm in ARMS:
        traces.extend(_build_per_task_trace([row for row in task_rows if row.get("arm") == arm], gate_by_arm[arm]))
    funnel = _funnel_summary(task_rows, traces)
    candidate_pool_trace = _candidate_pool_trace(runs, task_rows)
    downlink_residual_trace = _downlink_residual_trace(runs)
    arm_b_records = [record for run in runs if run["arm"] == ARM_B for record in run["records"]]
    blocker_rows = _followup_blocker_summary(arm_b_records)

    per_seed: list[dict[str, object]] = []
    for (seed, arm), run in sorted(run_by_key.items()):
        base = run_by_key.get((seed, ARM_A)) if arm == ARM_B else None
        per_seed.append(_summarize_validation_run(run, _task_action_change(run, base)))
    paired = _paired(per_seed)

    _write_csv(OUT_DIR / "residual_closure_funnel_summary.csv", funnel)
    _write_csv(OUT_DIR / "followup_eligibility_blocker_summary.csv", blocker_rows)
    _write_csv(OUT_DIR / "residual_candidate_pool_trace.csv", candidate_pool_trace)
    _write_csv(OUT_DIR / "downlink_queue_residual_trace.csv", downlink_residual_trace)
    _write_csv(OUT_DIR / "lifecycle_shadow_summary.csv", _aggregate(per_seed, ["alert_rate", "confirmation_rate", "evidence_rate", "first_actionable_latency_p50", "first_actionable_latency_p90", "lifecycle_shadow_utility_balanced", "lifecycle_shadow_utility_alert_heavy", "lifecycle_shadow_utility_evidence_heavy", "evidence_closure_count", "evidence_closure_rate"]))
    _write_csv(OUT_DIR / "staged_closure_summary.csv", _aggregate(per_seed, ["staged_plan_rate", "staged_metadata_success_rate", "staged_followup_success_rate", "metadata_to_roi_success_rate", "metadata_to_roi_context_success_rate", "residual_enqueue_count", "residual_delivery_count", "residual_delivery_rate"]))
    _write_csv(OUT_DIR / "canonical_metric_summary.csv", _aggregate(per_seed, ["net_system_utility", "delivered_utility", "effective_utility", "total_penalized_reward", "mission_utility", "completed_tasks", "delivered_items", "expired_items", "deadline_satisfaction_ratio", "utility_per_joule", "utility_per_downlink_bit"]))
    _write_csv(OUT_DIR / "safety_guardrail_summary.csv", _aggregate(per_seed, ["hard_safety_violation_count", "battery_min", "thermal_max", "queue_overflow_count", "deadline_miss_count", "expired_items"]))
    _write_csv(OUT_DIR / "paired_delta_summary.csv", paired)
    _write_csv(OUT_DIR / "per_seed_metric_summary.csv", per_seed)
    _write_csv(
        OUT_DIR / "candidate_action_log_summary.csv",
        [
            {"arm": arm, "candidate_log_rows": sum(1 for row in all_candidate_logs if row.get("arm") == arm)}
            for arm in ARMS
        ],
    )
    _write_audit_docs(command, seeds, gates, funnel, blocker_rows, per_seed, paired)
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(OUT_DIR / "git_diff.patch", _audit_diff(["scripts/run_phase6_carryover_residual_validation.py"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
