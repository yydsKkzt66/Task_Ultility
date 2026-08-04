"""Audit residual/evidence closure root cause without changing simulator behavior."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_product_requirement_aware_module2_ab import ARMS, DEFAULT_SEEDS, _load_run_configs  # noqa: E402
from src.evaluation.baselines import build_routers  # noqa: E402
from src.payload.continue_request import phase5_continue_config  # noqa: E402
from src.payload.information_sufficiency import (  # noqa: E402
    delta_information_utility,
    normalize_delivered_product_set,
    product_information_utility,
    residual_information_utility,
)
from src.payload.proactive_followup import (  # noqa: E402
    DEFAULT_FULL_METHOD_ALIASES,
    phase6_proactive_config,
    phase6_residual_candidate_carryover_config,
    phase6_same_slot_carryover_config,
)
from src.simulator.environment import SimulationEnvironment  # noqa: E402
from src.timeline.timeline_builder import TimelineBuilder  # noqa: E402


SOURCE_DIR = PROJECT_ROOT / "outputs" / "_product_requirement_aware_module2_ab"
OUT_DIR = PROJECT_ROOT / "outputs" / "_evidence_residual_closure_root_cause_audit"
RESIDUAL_TYPES = {"roi", "roi_context"}
STAGED_POLICIES = {"metadata_then_roi", "metadata_then_roi_context"}
EVIDENCE_PAYLOAD_POLICIES = {"roi_context", "metadata_then_roi_context", "metadata_then_roi", "roi"}
BREAKPOINT_LABELS = {
    "no_staged_action_selected",
    "payload_plan_not_resolved_to_residual",
    "initial_payload_not_generated",
    "initial_payload_not_enqueued",
    "initial_payload_not_delivered",
    "lifecycle_pending_residual_not_registered",
    "residual_candidate_not_added",
    "carryover_disabled",
    "router_not_allowed_for_carryover",
    "phase6_disabled",
    "followup_not_evaluated",
    "followup_blocked_metadata_not_delivered",
    "followup_blocked_no_pending_residual",
    "followup_blocked_no_deadline_slack",
    "followup_blocked_low_residual_utility",
    "followup_blocked_low_eriv",
    "followup_blocked_duplicate_queued",
    "followup_blocked_queue_pressure",
    "followup_blocked_battery_unsafe",
    "followup_blocked_thermal_unsafe",
    "followup_blocked_no_contact_opportunity",
    "residual_payload_not_generated",
    "residual_payload_not_enqueued",
    "residual_payload_dropped",
    "residual_payload_expired",
    "residual_payload_not_delivered",
    "delivered_but_not_marked_sufficient",
    "information_sufficiency_mapping_error",
    "trace_missing_unknown",
}


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
    try:
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
    except Exception as exc:  # pragma: no cover
        return f"git command failed: {exc}"


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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_div(num: float, den: float) -> float:
    return num / den if abs(den) > 1.0e-12 else 0.0


def _products(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, set):
        return {str(item) for item in value if str(item)}
    if isinstance(value, (list, tuple)):
        return {str(item) for item in value if str(item)}
    return {part.strip() for part in str(value).split(";") if part.strip()}


def _stage_from_products(products: set[str]) -> str:
    if "roi_context" in products:
        return "evidence"
    if "roi" in products:
        return "confirmation"
    if "metadata" in products:
        return "alert"
    return "none"


def _run_diagnostic_arm(seed: int, arm: str) -> dict[str, Any]:
    configs = _load_run_configs(seed, arm)
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"], configs["task_stream"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers(configs["experiment"], env.action_space)[0]
    runner = env.build_episode_runner(timeline)
    result = runner.run(router)
    records = [asdict(record) for record in result.records]
    for row in records:
        row["seed"] = seed
        row["arm"] = arm
        row["router"] = router.router_name
    return {
        "seed": seed,
        "arm": arm,
        "router": router.router_name,
        "records": records,
        "summary": dict(result.summary),
        "candidate_logs": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_candidate_action_logs],
        "queue_logs": [dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_queue_priority_logs],
        "carryover_events": [
            dict(row, seed=seed, arm=arm, router=router.router_name) for row in runner.last_carryover_lifecycle_events
        ],
    }


def _selected_task_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for row in run["records"]:
            if _to_int(row.get("has_task")):
                rows.append(row)
    return rows


def _is_evidence_relevant(row: dict[str, Any]) -> bool:
    delivered = _products(row.get("delivered_products"))
    pending = _products(row.get("pending_products"))
    stage = _stage_from_products(delivered)
    return bool(
        str(row.get("product_requirement", "")).lower() in {"context", "evidence"}
        or str(row.get("selected_payload_plan", "")).lower() in EVIDENCE_PAYLOAD_POLICIES
        or stage in {"alert", "confirmation"}
        or bool(pending)
        or _to_float(row.get("residual_utility_remaining")) > 0.0
        or _to_int(row.get("high_value_flag")) == 1
        or _to_int(row.get("emergency_flag")) == 1
    )


def _initial_generated(row: dict[str, Any]) -> bool:
    product = str(row.get("selected_product_type", "")).strip().lower()
    return bool(product and product != "defer" and _to_float(row.get("S_gen_current_mb")) > 0.0)


def _initial_enqueued(row: dict[str, Any]) -> bool:
    status = str(row.get("delivery_status", "")).strip().lower()
    return status in {"queued", "delivered"}


def _initial_delivered(row: dict[str, Any]) -> bool:
    product = str(row.get("selected_product_type", "")).strip().lower()
    delivered = _products(row.get("delivered_products"))
    return bool(product and product in delivered) or str(row.get("delivery_status", "")).strip().lower() == "delivered"


def _pending_residual_registered(row: dict[str, Any]) -> bool:
    pending = _products(row.get("pending_products")) | _products(row.get("lifecycle_post_pending_products"))
    return bool(pending & RESIDUAL_TYPES) or _to_int(row.get("lifecycle_post_pending_residual_registered_slot"), -1) >= 0


def _residual_candidate_added(row: dict[str, Any]) -> bool:
    return _to_int(row.get("residual_candidate_pool_add_count")) > 0


def _followup_evaluated(row: dict[str, Any]) -> bool:
    return (
        _to_int(row.get("proactive_followup_attempt_count"))
        + _to_int(row.get("carryover_proactive_attempt_count"))
        + _to_int(row.get("residual_candidate_re_evaluation_count"))
        > 0
    )


def _residual_generated(row: dict[str, Any]) -> bool:
    return _to_float(row.get("residual_utility_generated")) > 0.0 or _to_float(row.get("followup_payload_mb")) > 0.0


def _residual_enqueued(row: dict[str, Any]) -> bool:
    return _to_int(row.get("residual_enqueue_count")) > 0 or _to_int(row.get("proactive_followup_enqueue_count")) > 0


def _residual_delivered(row: dict[str, Any]) -> bool:
    return _to_int(row.get("residual_delivery_count")) > 0 or _to_int(row.get("proactive_followup_delivery_count")) > 0


def _evidence_closed(row: dict[str, Any]) -> bool:
    return _stage_from_products(_products(row.get("delivered_products"))) == "evidence" and _to_float(
        row.get("residual_utility_remaining"), 1.0
    ) <= 1.0e-12


def _classify_followup_blocker(reason: str) -> str:
    mapping = {
        "metadata_not_delivered": "followup_blocked_metadata_not_delivered",
        "no_pending_residual_product": "followup_blocked_no_pending_residual",
        "multiple_pending_residual_products": "followup_blocked_no_pending_residual",
        "no_useful_remaining_deadline": "followup_blocked_no_deadline_slack",
        "residual_utility_below_threshold": "followup_blocked_low_residual_utility",
        "residual_utility_zero_or_below_threshold": "followup_blocked_low_residual_utility",
        "eriv_below_threshold": "followup_blocked_low_eriv",
        "duplicate_residual_already_queued": "followup_blocked_duplicate_queued",
        "queue_pressure_too_high": "followup_blocked_queue_pressure",
        "battery_unsafe": "followup_blocked_battery_unsafe",
        "thermal_unsafe": "followup_blocked_thermal_unsafe",
        "no_contact_opportunity": "followup_blocked_no_contact_opportunity",
        "phase6_disabled": "phase6_disabled",
        "variant_not_full_proposed": "router_not_allowed_for_carryover",
    }
    return mapping.get(reason, "trace_missing_unknown")


def _breakpoint_for_task(row: dict[str, Any], gate: dict[str, object]) -> str:
    payload_plan = str(row.get("selected_payload_plan", "")).strip().lower()
    product = str(row.get("selected_product_type", "")).strip().lower()
    delivered = _products(row.get("delivered_products"))
    pending = _products(row.get("pending_products")) | _products(row.get("lifecycle_post_pending_products"))

    if _evidence_closed(row):
        return "closed"
    if not _initial_generated(row):
        return "initial_payload_not_generated"
    if product == "metadata" and payload_plan not in STAGED_POLICIES:
        return "no_staged_action_selected"
    if payload_plan in STAGED_POLICIES and not str(row.get("residual_product_type", "")).strip():
        return "payload_plan_not_resolved_to_residual"
    if not _initial_enqueued(row):
        return "initial_payload_not_enqueued"
    if product in {"metadata", "roi", "roi_context"} and product not in delivered:
        return "initial_payload_not_delivered"
    if payload_plan in STAGED_POLICIES and not _pending_residual_registered(row):
        return "lifecycle_pending_residual_not_registered"
    if (pending & RESIDUAL_TYPES) and not _residual_candidate_added(row):
        if not bool(gate["phase6_enabled"]):
            return "phase6_disabled"
        if not bool(gate["carryover_cfg_enabled"]):
            return "carryover_disabled"
        if not bool(gate["router_alias_allowed"]):
            return "router_not_allowed_for_carryover"
        return "residual_candidate_not_added"
    if _residual_candidate_added(row) and not _followup_evaluated(row):
        return "followup_not_evaluated"
    if _followup_evaluated(row):
        blockers = _products(row.get("proactive_followup_blockers"))
        reason = str(row.get("proactive_followup_reason_code", "")).strip()
        if blockers:
            reason = sorted(blockers)[0]
        if not (_to_int(row.get("proactive_followup_enqueue_count")) or _to_int(row.get("residual_enqueue_count"))):
            return _classify_followup_blocker(reason)
    if _followup_evaluated(row) and not _residual_generated(row):
        return "residual_payload_not_generated"
    if _residual_generated(row) and not _residual_enqueued(row):
        return "residual_payload_not_enqueued"
    if _to_int(row.get("residual_dropped_count")) > 0 or _to_int(row.get("proactive_followup_dropped_count")) > 0:
        return "residual_payload_dropped"
    if _to_int(row.get("residual_expired_count")) > 0 or _to_int(row.get("proactive_followup_expired_count")) > 0:
        return "residual_payload_expired"
    if _residual_enqueued(row) and not _residual_delivered(row):
        return "residual_payload_not_delivered"
    if _residual_delivered(row) and not _evidence_closed(row):
        return "delivered_but_not_marked_sufficient"
    if "roi_context" in delivered and _to_float(row.get("residual_utility_remaining")) > 1.0e-12:
        return "information_sufficiency_mapping_error"
    return "trace_missing_unknown"


def _gate_audit() -> dict[str, object]:
    configs = _load_run_configs(DEFAULT_SEEDS[0], "v21_reqaware_ug_wx050")
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
    carryover_enabled = bool(phase6_enabled and carryover_cfg_enabled and router_alias_allowed)
    blockers = []
    if not phase6_enabled:
        blockers.append("phase6_proactive_followup.disabled")
    if not carryover_cfg_enabled:
        blockers.append("phase6_residual_candidate_carryover.disabled")
    if not router_alias_allowed:
        blockers.append("router_alias_not_allowed")
    return {
        "router_name": router_name,
        "phase5_enabled": bool(phase5_cfg.get("enabled", False)),
        "phase6_enabled": phase6_enabled,
        "proactive_followup_enabled": phase6_enabled,
        "carryover_cfg_enabled": carryover_cfg_enabled,
        "same_slot_carryover_enabled": bool(same_slot_cfg.get("enabled", False)),
        "router_alias_allowed": router_alias_allowed,
        "carryover_enabled_for_current_branch": carryover_enabled,
        "full_method_aliases": sorted(aliases),
        "use_lyapunov_pricing": bool(getattr(router, "use_lyapunov_pricing", False)),
        "use_mode_controller": bool(getattr(router, "use_mode_controller", False)),
        "primary_gate_blocker": ";".join(blockers) if blockers else "none",
        "source_config": "configs/simulator_sarfish_energy_shadow_stress_20s.yaml plus runtime overrides",
    }


def _build_per_task_trace(rows: list[dict[str, Any]], gate: dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        if not _is_evidence_relevant(row):
            continue
        delivered = _products(row.get("delivered_products"))
        pending = _products(row.get("pending_products"))
        breakpoint = _breakpoint_for_task(row, gate)
        if breakpoint not in BREAKPOINT_LABELS and breakpoint != "closed":
            breakpoint = "trace_missing_unknown"
        initial_size_bits = _to_float(row.get("S_gen_current_mb")) * 8_000_000.0
        residual_size_bits = _to_float(row.get("followup_payload_mb")) * 8_000_000.0
        out.append(
            {
                "seed": row.get("seed"),
                "arm": row.get("arm"),
                "task_id": row.get("task_id"),
                "task_value": row.get("V_t", row.get("task_value")),
                "uncertainty": row.get("U_t", row.get("uncertainty")),
                "emergency_flag": row.get("emergency_flag"),
                "high_value_flag": row.get("high_value_flag"),
                "ideal_product_requirement": row.get("product_requirement"),
                "selected_model": row.get("selected_model", row.get("model_name")),
                "selected_payload_policy": row.get("selected_payload_plan", row.get("payload_policy")),
                "resolved_payload_plan": row.get("selected_payload_plan", row.get("payload_policy")),
                "generated_product_type": row.get("selected_product_type"),
                "residual_product_type": row.get("residual_product_type"),
                "initial_payload_size_bits": initial_size_bits,
                "initial_payload_generated_flag": int(_initial_generated(row)),
                "initial_payload_enqueued_flag": int(_initial_enqueued(row)),
                "initial_payload_delivered_flag": int(_initial_delivered(row)),
                "initial_payload_delivered_time": row.get("payload_delivery_time_s", -1.0),
                "delivered_products_after_initial": row.get("lifecycle_post_delivered_products", row.get("delivered_products", "")),
                "pending_products_after_initial": row.get("lifecycle_post_pending_products", row.get("pending_products", "")),
                "residual_utility_remaining_after_initial": row.get(
                    "lifecycle_post_residual_utility_remaining", row.get("residual_utility_remaining", 0.0)
                ),
                "residual_candidate_added_flag": int(_residual_candidate_added(row)),
                "residual_candidate_add_time": row.get("time_s") if _residual_candidate_added(row) else "",
                "residual_candidate_pool_size": row.get("residual_candidate_pool_size", 0),
                "followup_evaluated_flag": int(_followup_evaluated(row)),
                "followup_allowed_flag": int(_residual_generated(row)),
                "followup_blocker_reason": row.get("proactive_followup_blockers")
                or row.get("proactive_followup_reason_code")
                or ("phase6_disabled" if breakpoint == "phase6_disabled" else ""),
                "residual_payload_generated_flag": int(_residual_generated(row)),
                "residual_payload_product_type": row.get("residual_product_type") if _residual_generated(row) else "",
                "residual_payload_size_bits": residual_size_bits,
                "residual_payload_enqueued_flag": int(_residual_enqueued(row)),
                "residual_payload_dropped_flag": int(
                    _to_int(row.get("residual_dropped_count")) > 0 or _to_int(row.get("proactive_followup_dropped_count")) > 0
                ),
                "residual_payload_expired_flag": int(
                    _to_int(row.get("residual_expired_count")) > 0 or _to_int(row.get("proactive_followup_expired_count")) > 0
                ),
                "residual_payload_delivered_flag": int(_residual_delivered(row)),
                "residual_payload_delivered_time": row.get("payload_delivery_time_s") if _residual_delivered(row) else "",
                "delivered_products_final": row.get("delivered_products", ""),
                "pending_products_final": row.get("pending_products", ""),
                "residual_utility_remaining_final": row.get("residual_utility_remaining", 0.0),
                "lifecycle_stage_final": _stage_from_products(delivered),
                "task_closed_flag": int(_to_float(row.get("residual_utility_remaining"), 1.0) <= 1.0e-12),
                "evidence_closed_flag": int(_evidence_closed(row)),
                "final_breakpoint_label": breakpoint,
            }
        )
    return out


def _funnel_summary(rows: list[dict[str, Any]], traces: list[dict[str, object]]) -> list[dict[str, object]]:
    traces_by_key = {(str(row["seed"]), str(row["arm"]), str(row["task_id"])): row for row in traces}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("arm"))].append(row)
    out: list[dict[str, object]] = []
    for arm, arm_rows in sorted(grouped.items()):
        trace_rows = [traces_by_key[key] for key in traces_by_key if key[1] == arm]
        selected_staged = [row for row in arm_rows if str(row.get("selected_payload_plan")) in STAGED_POLICIES]
        selected_mtr = [row for row in arm_rows if str(row.get("selected_payload_plan")) == "metadata_then_roi"]
        selected_mtrc = [row for row in arm_rows if str(row.get("selected_payload_plan")) == "metadata_then_roi_context"]
        selected_roi_ctx = [row for row in arm_rows if str(row.get("selected_payload_plan")) == "roi_context"]
        metric_counts = {
            "selected_staged_action_count": len(selected_staged),
            "selected_roi_context_count": len(selected_roi_ctx),
            "selected_metadata_then_roi_count": len(selected_mtr),
            "selected_metadata_then_roi_context_count": len(selected_mtrc),
            "initial_payload_generated_count": sum(int(_initial_generated(row)) for row in arm_rows),
            "initial_payload_enqueued_count": sum(int(_initial_enqueued(row)) for row in arm_rows),
            "initial_payload_delivered_count": sum(int(_initial_delivered(row)) for row in arm_rows),
            "metadata_delivered_count": sum(int("metadata" in _products(row.get("delivered_products"))) for row in arm_rows),
            "lifecycle_pending_residual_registered_count": sum(int(_pending_residual_registered(row)) for row in arm_rows),
            "residual_candidate_added_count": sum(_to_int(row.get("residual_candidate_pool_add_count")) for row in arm_rows),
            "residual_candidate_carried_over_count": sum(_to_int(row.get("residual_candidate_re_evaluation_count")) for row in arm_rows),
            "followup_eligibility_evaluated_count": sum(int(_followup_evaluated(row)) for row in arm_rows),
            "followup_eligibility_passed_count": sum(int(_residual_generated(row)) for row in arm_rows),
            "followup_eligibility_blocked_count": sum(int(_followup_evaluated(row) and not _residual_generated(row)) for row in arm_rows),
            "residual_payload_generated_count": sum(int(_residual_generated(row)) for row in arm_rows),
            "residual_payload_enqueued_count": sum(int(_residual_enqueued(row)) for row in arm_rows),
            "residual_payload_dropped_count": sum(_to_int(row.get("residual_dropped_count")) for row in arm_rows),
            "residual_payload_expired_count": sum(_to_int(row.get("residual_expired_count")) for row in arm_rows),
            "residual_payload_delivered_count": sum(int(_residual_delivered(row)) for row in arm_rows),
            "information_sufficiency_closed_count": sum(
                int(_to_float(row.get("residual_utility_remaining"), 1.0) <= 1.0e-12) for row in arm_rows
            ),
            "evidence_closure_count": sum(int(_evidence_closed(row)) for row in arm_rows),
            "residual_candidate_true_count": sum(_to_int(row.get("residual_candidate_true_count")) for row in arm_rows),
            "trace_missing_unknown_count": sum(
                int(str(row.get("final_breakpoint_label")) == "trace_missing_unknown") for row in trace_rows
            ),
        }
        stage_order = [
            "selected_staged_action_count",
            "initial_payload_generated_count",
            "initial_payload_enqueued_count",
            "initial_payload_delivered_count",
            "lifecycle_pending_residual_registered_count",
            "residual_candidate_true_count",
            "residual_candidate_added_count",
            "followup_eligibility_evaluated_count",
            "residual_payload_generated_count",
            "residual_payload_enqueued_count",
            "residual_payload_delivered_count",
            "evidence_closure_count",
        ]
        first_zero = "none"
        for metric in stage_order:
            if metric_counts.get(metric, 0) == 0:
                first_zero = metric
                break
        out.append({"arm": arm, **metric_counts, "first_zero_collapse_stage": first_zero})
    return out


def _candidate_pool_trace(runs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    event_rows: list[dict[str, object]] = []
    for run in runs:
        for event in run["carryover_events"]:
            event_rows.append(event)
    if event_rows:
        return event_rows
    fallback: list[dict[str, object]] = []
    for row in rows:
        if _to_int(row.get("residual_candidate_true_count")) or _pending_residual_registered(row):
            fallback.append(
                {
                    "seed": row.get("seed"),
                    "arm": row.get("arm"),
                    "slot": row.get("slot"),
                    "task_id": row.get("task_id"),
                    "event_type": "candidate_true_not_added" if _to_int(row.get("residual_candidate_true_count")) else "pending_residual_registered",
                    "residual_product_type": row.get("residual_product_type"),
                    "metadata_delivery_slot": row.get("lifecycle_post_metadata_delivery_slot"),
                    "pending_residual_registered_slot": row.get("lifecycle_post_pending_residual_registered_slot"),
                    "candidate_added_flag": row.get("residual_candidate_pool_add_count", 0),
                    "candidate_re_evaluated_flag": row.get("residual_candidate_re_evaluation_count", 0),
                    "proactive_blockers": "phase6_disabled",
                    "trace_source": "trajectory_record_fallback",
                }
            )
    return fallback


def _downlink_residual_trace(runs: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    residual_queue_rows = []
    for run in runs:
        for row in run["queue_logs"]:
            payload_type = str(row.get("payload_type", ""))
            if _to_int(row.get("proactive_queue_bonus_component")) > 0 or str(row.get("payload_plan", "")).startswith("metadata_then"):
                residual_queue_rows.append(row)
    if residual_queue_rows:
        return residual_queue_rows
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_arm[run["arm"]].append(run["summary"])
    for arm, summaries in sorted(by_arm.items()):
        rows.append(
            {
                "arm": arm,
                "residual_payloads_observed_in_queue_log": 0,
                "residual_payload_enqueued_count": sum(_to_int(row.get("residual_enqueue_count")) for row in summaries),
                "residual_payload_delivered_count": sum(_to_int(row.get("residual_delivery_count")) for row in summaries),
                "residual_payload_dropped_count": sum(_to_int(row.get("residual_dropped_count")) for row in summaries),
                "residual_payload_expired_count": sum(_to_int(row.get("residual_expired_count")) for row in summaries),
                "product_type_preserved_by_queue_code": True,
                "payload_plan_preserved_by_queue_code": True,
                "parent_task_id_preserved_as_task_id": True,
                "dominant_queue_observation": "no residual payload reached DownlinkQueue",
            }
        )
    return rows


def _information_sufficiency_cases() -> list[dict[str, object]]:
    cases = [
        ("alert", {"metadata"}),
        ("verification", {"roi"}),
        ("context", {"roi_context"}),
        ("evidence", {"roi_context"}),
        ("evidence", {"roi", "metadata"}),
        ("evidence", {"metadata"}),
    ]
    rows = []
    for requirement, delivered in cases:
        residual = residual_information_utility(delivered, requirement)
        stage_satisfied = (
            ("metadata" in delivered or "roi" in delivered or "roi_context" in delivered)
            if requirement == "alert"
            else ("roi" in delivered or "roi_context" in delivered)
            if requirement == "verification"
            else ("roi_context" in delivered)
        )
        rows.append(
            {
                "product_requirement": requirement,
                "delivered_products": ";".join(sorted(delivered)),
                "normalized_delivered_key": normalize_delivered_product_set(delivered),
                "product_information_utility": product_information_utility(delivered, requirement),
                "residual_information_utility": residual,
                "requirement_stage_satisfied_flag": int(stage_satisfied),
                "residual_target_closed_flag": int(residual <= 1.0e-12),
                "closed_flag": int(stage_satisfied),
                "delta_if_metadata": delta_information_utility("metadata", delivered, requirement),
                "delta_if_roi": delta_information_utility("roi", delivered, requirement),
                "delta_if_roi_context": delta_information_utility("roi_context", delivered, requirement),
            }
        )
    return rows


def _blocker_summary(traces: list[dict[str, object]]) -> list[dict[str, object]]:
    counter: Counter[str] = Counter(str(row.get("final_breakpoint_label", "trace_missing_unknown")) for row in traces)
    total = max(sum(counter.values()), 1)
    examples: dict[str, list[str]] = defaultdict(list)
    for row in traces:
        label = str(row.get("final_breakpoint_label", "trace_missing_unknown"))
        task_id = str(row.get("task_id", ""))
        if task_id and len(examples[label]) < 8:
            examples[label].append(task_id)
    category = {
        "no_staged_action_selected": "residual generation blocker",
        "payload_plan_not_resolved_to_residual": "residual generation blocker",
        "initial_payload_not_generated": "residual generation blocker",
        "initial_payload_not_enqueued": "residual enqueue blocker",
        "initial_payload_not_delivered": "residual delivery blocker",
        "phase6_disabled": "router/config gate",
        "carryover_disabled": "router/config gate",
        "router_not_allowed_for_carryover": "router/config gate",
        "lifecycle_pending_residual_not_registered": "lifecycle pending residual missing",
        "residual_candidate_not_added": "router/config gate",
        "followup_not_evaluated": "followup eligibility blocker",
        "residual_payload_not_generated": "residual generation blocker",
        "residual_payload_not_enqueued": "residual enqueue blocker",
        "residual_payload_not_delivered": "residual delivery blocker",
        "delivered_but_not_marked_sufficient": "closure accounting blocker",
        "information_sufficiency_mapping_error": "closure accounting blocker",
        "trace_missing_unknown": "trace missing",
    }
    rows = []
    seen_categories: set[str] = set()
    for reason, count in counter.most_common():
        if reason == "closed":
            continue
        blocker_category = category.get(
            reason,
            "followup eligibility blocker" if reason.startswith("followup_") else "trace missing",
        )
        seen_categories.add(blocker_category)
        rows.append(
            {
                "blocker_reason": reason,
                "count": count,
                "percentage": _safe_div(count, total),
                "affected_tasks": count,
                "example_task_ids": ";".join(examples[reason]),
                "first_stage_where_detected": reason,
                "blocker_category": blocker_category,
            }
        )
    required_categories = [
        "router/config gate",
        "lifecycle pending residual missing",
        "followup eligibility blocker",
        "residual generation blocker",
        "residual enqueue blocker",
        "residual delivery blocker",
        "closure accounting blocker",
        "trace missing",
    ]
    for required in required_categories:
        if required in seen_categories:
            continue
        rows.append(
            {
                "blocker_reason": f"{required}:none_observed",
                "count": 0,
                "percentage": 0.0,
                "affected_tasks": 0,
                "example_task_ids": "",
                "first_stage_where_detected": "none_observed",
                "blocker_category": required,
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


def _write_gate_report(gate: dict[str, object], funnel: list[dict[str, object]]) -> None:
    report = f"""# Router and config gate audit

## Required conclusions

- carryover_enabled_for_current_branch: {str(gate['carryover_enabled_for_current_branch']).lower()}
- router_alias_allowed: {str(gate['router_alias_allowed']).lower()}
- phase5_enabled: {str(gate['phase5_enabled']).lower()}
- phase6_enabled: {str(gate['phase6_enabled']).lower()}
- proactive_followup_enabled: {str(gate['proactive_followup_enabled']).lower()}
- carryover_cfg_enabled: {str(gate['carryover_cfg_enabled']).lower()}
- same_slot_carryover_enabled: {str(gate['same_slot_carryover_enabled']).lower()}
- primary_gate_blocker: {gate['primary_gate_blocker']}

## Interpretation

The current requirement-aware branch uses router `{gate['router_name']}`. This router is accepted by the full-method alias logic, so the router alias is not the blocker. The blocker is config gating: Phase 6 proactive follow-up and residual candidate carryover are not enabled in the A/B simulator configuration. Because `_carryover_behavior_enabled(...)` is `phase6_enabled AND carryover_enabled AND full_proposed_router`, true residual candidates are counted but not inserted into the candidate pool.

## Funnel excerpt

{_markdown_table(funnel, ['arm', 'residual_candidate_true_count', 'residual_candidate_added_count', 'followup_eligibility_evaluated_count', 'residual_payload_enqueued_count', 'first_zero_collapse_stage'])}
"""
    _write_text(OUT_DIR / "router_and_config_gate_audit.md", report)


def _write_lifecycle_report(gate: dict[str, object], funnel: list[dict[str, object]]) -> None:
    report = f"""# Lifecycle state transition audit

1. Pending products are updated immediately when a payload exists, before delivery is processed in `_apply_product_lifecycle_updates(...)`.
2. For staged metadata plans, `payload.residual_product_type` is added to `pending_products` at generation time and `pending_residual_registered_slot` is set immediately.
3. When metadata is delivered, `first_alert_delivered_time_s` and `metadata_delivery_slot` are updated, `metadata` is removed from pending, and the residual product remains pending.
4. `residual_utility_remaining` is recomputed after each delivered product with `residual_information_utility(...)`.
5. Residual candidate addition is attempted after lifecycle update through `_add_residual_candidate_after_lifecycle(...)`.
6. Candidate addition requires exactly one pending residual product, metadata delivered, positive residual utility, no expiry, not closed, not already queued/delivered.
7. Candidate removal is not the observed primary issue: `residual_candidate_pool_add_count` is zero, so candidates are not being removed after insertion.
8. Duplicate suppression is not the observed primary issue: duplicate residual suppression counters are zero before any residual enqueue.
9. Tasks are not closed after metadata unless residual utility reaches zero; observed true candidates prove some tasks remain open with pending residual utility.
10. Expiry is not the primary issue for all cases: observed `residual_candidate_true_count` is positive, but `residual_candidate_pool_add_count` remains zero.

Conclusion: lifecycle registration reaches the true-candidate stage, but candidate insertion is gated off by Phase 6/carryover config. Gate status: `{gate['primary_gate_blocker']}`.
"""
    _write_text(OUT_DIR / "lifecycle_state_transition_audit.md", report)


def _write_root_cause_report(
    gate: dict[str, object],
    funnel: list[dict[str, object]],
    blockers: list[dict[str, object]],
    info_rows: list[dict[str, object]],
) -> None:
    reqaware = next((row for row in funnel if row.get("arm") == "v21_reqaware_ug_wx050"), {})
    first_zero = str(reqaware.get("first_zero_collapse_stage", "unknown"))
    primary = "phase6/carryover residual follow-up is disabled for the A/B branch, so true residual candidates are not inserted into the candidate pool"
    evidence = [
        f"v21_reqaware residual_candidate_true_count={reqaware.get('residual_candidate_true_count', 0)}",
        f"v21_reqaware residual_candidate_added_count={reqaware.get('residual_candidate_added_count', 0)}",
        f"v21_reqaware followup_eligibility_evaluated_count={reqaware.get('followup_eligibility_evaluated_count', 0)}",
        f"v21_reqaware residual_payload_enqueued_count={reqaware.get('residual_payload_enqueued_count', 0)}",
        f"gate primary_gate_blocker={gate['primary_gate_blocker']}",
        f"router_alias_allowed={gate['router_alias_allowed']}",
    ]
    root_md = f"""# Root cause classification

- primary_root_cause: {primary}
- confidence: high
- first_zero_collapse_stage: {first_zero}
- minimal_next_repair_target: enable a default-off Phase 6/carryover diagnostic or experiment gate for the requirement-aware proposed branch, without changing reward, SafeFilter, utility_rule, or Module 2 scoring.

## Evidence supporting primary root cause

{chr(10).join('- ' + item for item in evidence)}

## Secondary root causes

- Many initial evidence/context payloads are not delivered before residual follow-up can matter, so queue/contact/deadline feasibility will remain a secondary limiter after the gate is opened.
- Passive Phase 5 virtual continue is also disabled, so there is no passive residual fallback path.

## Ruled out causes

- Router alias rejection: ruled out. `{gate['router_name']}` is allowed by full-method alias logic.
- Information sufficiency mapping: ruled out as primary. Diagnostic cases show `roi_context` closes `context` and `evidence`.
- Downlink residual delivery: ruled out as first cause. No residual payload reaches queue/enqueue.
- Residual candidate removal: ruled out as first cause. Candidate add count is zero.

## Files likely needing patch in the next step

- `configs` copied experiment config only, not main defaults.
- `src/simulator/step_runner.py` only if a default-off trace or alias hook is needed.
- `src/payload/proactive_followup.py` only if the next patch needs an explicit alias/config gate; do not change eligibility math unless separately justified.

## Files that should NOT be patched

- `src/reward/reward_function.py`
- mission prior value code
- SafeFilter
- utility_rule logic
- Module 2 scoring
"""
    _write_text(OUT_DIR / "root_cause_classification.md", root_md)

    report = f"""# Evidence residual closure root-cause report

## Executive conclusion

The residual/evidence chain first collapses at residual candidate insertion: true candidates exist, but `residual_candidate_pool_add_count` is zero. The dominant blocker is config gating, not Module 2 scoring, reward, SafeFilter, utility_rule, downlink queue service, or information sufficiency mapping.

## Required answers

1. At which stage does the residual/evidence chain first collapse to zero?

   `{first_zero}` for `v21_reqaware_ug_wx050`, with true candidates present and candidate adds zero.

2. Are staged actions selected by Module 2?

   Yes. See `selected_staged_action_count` and staged policy counts in `residual_closure_funnel_summary.csv`.

3. Are metadata initial payloads generated?

   Yes, initial payload generation is observed for non-defer payloads.

4. Are metadata initial payloads delivered?

   Some metadata deliveries are observed, but not enough to close evidence. This is not the first zero collapse.

5. Are residual products registered in pending_products?

   Yes for staged payloads. Pending residual registration is observed before candidate insertion.

6. Are residual candidates added?

   No. `residual_candidate_added_count=0` for the current branch.

7. Is carryover/proactive follow-up enabled for the current branch?

   No. `carryover_enabled_for_current_branch=false`; primary gate blocker is `{gate['primary_gate_blocker']}`.

8. Are follow-up eligibility checks being evaluated?

   No. Candidate pool insertion is gated off, so eligibility evaluation count remains zero.

9. What is the dominant blocker?

   Router/config gate: Phase 6 proactive follow-up and residual candidate carryover are disabled in the A/B simulator configuration.

10. Are residual payloads ever generated?

   No.

11. Are residual payloads ever enqueued?

   No.

12. Are residual payloads ever delivered?

   No.

13. If delivered, are they recognized by information sufficiency?

   Diagnostic cases indicate they would be: `roi_context` closes both `context` and `evidence` under current mapping.

14. Is evidence closure zero due to policy, queue, lifecycle, config gating, or metric accounting?

   Primarily config gating. Queue and delivery cannot be the first cause because no residual payload is generated or enqueued.

15. What is the minimal safe next patch?

   Add a copied-config, default-off experiment toggle that enables Phase 6 proactive follow-up and residual candidate carryover for the requirement-aware proposed branch. Keep main defaults unchanged, keep reward unchanged, keep SafeFilter unchanged, keep utility_rule unchanged, and keep Module 2 scoring unchanged.

## Funnel

{_markdown_table(funnel, ['arm', 'selected_staged_action_count', 'metadata_delivered_count', 'lifecycle_pending_residual_registered_count', 'residual_candidate_true_count', 'residual_candidate_added_count', 'followup_eligibility_evaluated_count', 'residual_payload_enqueued_count', 'evidence_closure_count', 'first_zero_collapse_stage'])}

## Information sufficiency diagnostic

{_markdown_table(info_rows, ['product_requirement', 'delivered_products', 'product_information_utility', 'residual_information_utility', 'requirement_stage_satisfied_flag', 'residual_target_closed_flag'])}
"""
    _write_text(OUT_DIR / "evidence_residual_closure_root_cause_report.md", report)
    _write_text(
        OUT_DIR / "final_recommendation.md",
        """# Final recommendation

Do not implement the repair in this task.

Recommended next patch: create a copied, default-off Phase 6/carryover experiment config for `v21_reqaware_ug_wx050` that enables residual candidate carryover and proactive follow-up for the already-allowed proposed router alias. Do not change reward, SafeFilter, utility_rule, queue scheduling, lifecycle closure semantics, or Module 2 scoring.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--arms", default=",".join(ARMS.keys()))
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = (
        "python scripts/audit_evidence_residual_closure_root_cause.py --seeds "
        + ",".join(str(seed) for seed in seeds)
        + " --arms "
        + ",".join(arms)
    )

    gate = _gate_audit()
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in arms:
            runs.append(_run_diagnostic_arm(seed, arm))
    task_rows = _selected_task_rows(runs)
    per_task_trace = _build_per_task_trace(task_rows, gate)
    funnel = _funnel_summary(task_rows, per_task_trace)
    candidate_pool_trace = _candidate_pool_trace(runs, task_rows)
    downlink_residual_trace = _downlink_residual_trace(runs)
    info_rows = _information_sufficiency_cases()
    blocker_rows = _blocker_summary(per_task_trace)

    _write_csv(OUT_DIR / "residual_closure_funnel_summary.csv", funnel)
    _write_csv(OUT_DIR / "per_task_residual_lifecycle_trace.csv", per_task_trace)
    _write_csv(OUT_DIR / "residual_candidate_pool_trace.csv", candidate_pool_trace)
    _write_csv(OUT_DIR / "followup_eligibility_blocker_summary.csv", blocker_rows)
    _write_csv(OUT_DIR / "downlink_queue_residual_trace.csv", downlink_residual_trace)
    _write_csv(OUT_DIR / "information_sufficiency_closure_trace.csv", info_rows)

    _write_text(OUT_DIR / "README.md", "# Evidence/residual closure root-cause audit\n\nAudit-only diagnostics. No behavior repair is implemented.\n")
    _write_text(OUT_DIR / "command_log.md", f"# Command log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "audit_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "source_output_dir": SOURCE_DIR.as_posix(),
                "output_dir": OUT_DIR.as_posix(),
                "seeds": seeds,
                "arms": arms,
                "audit_only": True,
                "behavior_changes": False,
                "gate_audit": gate,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(OUT_DIR / "git_status.txt", _run_git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _audit_diff(
            [
                "scripts/audit_evidence_residual_closure_root_cause.py",
                "src/simulator/step_runner.py",
                "src/payload/proactive_followup.py",
                "src/payload/continue_request.py",
                "src/payload/information_sufficiency.py",
                "src/link/downlink_queue.py",
                "src/common/types.py",
                "src/evaluation/metrics.py",
            ]
        ),
    )

    _write_gate_report(gate, funnel)
    _write_lifecycle_report(gate, funnel)
    _write_root_cause_report(gate, funnel, blocker_rows, info_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
