"""Fail-fast labeling rules for the canonical full proposed method."""
from __future__ import annotations

from typing import Any


FULL_LABELS = {"PROP-FULL", "proposed_guarantee_aware_full"}


def resolved_full_feature_flags(configs: dict[str, Any]) -> list[dict[str, object]]:
    simulator = configs.get("simulator", {})
    progressive = simulator.get("progressive_payload", {})
    staged = progressive.get("value_aware_staged_delivery_queue", {})
    proactive = progressive.get("phase6_proactive_followup", {})
    carryover = progressive.get("phase6_residual_candidate_carryover", {})
    guarantee = simulator.get("guarantee", {})
    router_names = configs.get("experiment", {}).get("routers", [])
    router_ok = "proposed_lyapunov_rule_mode_safe_bandit" in router_names
    checks = [
        ("base_architecture", router_ok, str(router_names)),
        ("persistent_staged_delivery_queue", bool(staged.get("enabled")), str(staged.get("enabled"))),
        ("progressive_followup", bool(proactive.get("enabled")), str(proactive.get("enabled"))),
        ("cross_window_residual_recompetition", bool(carryover.get("enabled")), str(carryover.get("enabled"))),
        ("guarantee_aware_priority_debt", bool(guarantee.get("enabled")) and float(guarantee.get("lambda_H", 0)) > 0, f"enabled={guarantee.get('enabled')};lambda_H={guarantee.get('lambda_H')}"),
        ("hard_safety_guard", bool(guarantee.get("hard_safety_veto")), str(guarantee.get("hard_safety_veto"))),
        ("ordinary_direct_delivery", not bool(staged.get("force_all_tasks_staged", False)), f"force_all_tasks_staged={staged.get('force_all_tasks_staged', False)}"),
    ]
    return [{"feature": name, "required_for_prop_full": 1, "enabled": int(ok), "resolved_value": value} for name, ok, value in checks]


def validate_method_label(label: str, configs: dict[str, Any]) -> list[dict[str, object]]:
    flags = resolved_full_feature_flags(configs)
    if label in FULL_LABELS:
        disabled = [str(row["feature"]) for row in flags if not row["enabled"]]
        if disabled:
            raise ValueError(f"{label} is invalid because required features are disabled: {', '.join(disabled)}")
    return flags

