from __future__ import annotations

from scripts.run_roi_context_blocker_aware_gate_sweep import _select_pareto, _summarize_blockers


def test_blocker_summary_counts_required_reason_classes():
    runs = [
        {
            "scenario": "natural",
            "seed": 2026,
            "arm": "current",
            "config_id": "current",
            "budget_trace": [
                {
                    "stage": "module2_candidate_action_gate",
                    "allowed": 0,
                    "reason_code": "value_density_below_profile_threshold",
                    "blockers": "value_density_below_profile_threshold;no_contact_opportunity",
                }
            ],
            "carryover_events": [{"proactive_blockers": "eriv_below_profile_threshold"}],
            "followup_guard": [{"operation": "carryover_residual_enqueue", "blockers": "followup_hard_safety_blocked_queue"}],
            "duplicate_trace": [{"reason_code": "duplicate_residual_already_queued"}],
        }
    ]
    summary, by_scenario, by_stage = _summarize_blockers(runs)
    counts = {row["reason"]: row["count"] for row in summary}
    assert counts["value_density_below_profile_threshold"] == 1
    assert counts["no_contact_opportunity"] == 1
    assert counts["eriv_below_profile_threshold"] == 1
    assert counts["hard_safety_guard_blocked_queue"] == 1
    assert counts["duplicate_suppression"] == 1
    assert any(row["scenario"] == "natural" and row["reason"] == "no_contact_opportunity" for row in by_scenario)
    assert any(row["stage"] == "carryover_residual_enqueue" and row["reason"] == "hard_safety_guard_blocked_queue" for row in by_stage)


def test_pareto_selection_prioritizes_delivery_and_closure_before_net_utility():
    rows = [
        {
            "config_id": "unsafe_high_utility",
            "hard_safety_violation_count_mean": 0,
            "low_medium_accidental_roi_context_count_mean": 0,
            "queue_next_mb_max_max": 5,
            "metadata_to_roi_context_success_rate_mean": 0,
            "roi_context_delivered_count_mean": 0,
            "context_closure_count_mean": 0,
            "evidence_closure_count_mean": 0,
            "high_value_context_evidence_closure_ratio_mean": 0,
            "utility_per_MB_mean": 100,
            "net_system_utility_mean": 1000,
            "expired_items_mean": 0,
        },
        {
            "config_id": "delivers_with_closure",
            "hard_safety_violation_count_mean": 0,
            "low_medium_accidental_roi_context_count_mean": 0,
            "queue_next_mb_max_max": 5,
            "metadata_to_roi_context_success_rate_mean": 0.1,
            "roi_context_delivered_count_mean": 1,
            "context_closure_count_mean": 1,
            "evidence_closure_count_mean": 0,
            "high_value_context_evidence_closure_ratio_mean": 0.2,
            "utility_per_MB_mean": 1,
            "net_system_utility_mean": 10,
            "expired_items_mean": 0,
        },
    ]
    selected = _select_pareto(rows, queue_limit=10)
    assert selected[0]["config_id"] == "delivers_with_closure"
