from scripts.run_sarfish_slot_level_behavior_audit_seed2026 import (
    IDLE_REASONS,
    classify_idle_reason,
    is_avoidable_idle,
)


def test_idle_taxonomy_is_exact():
    assert IDLE_REASONS == {
        "NO_ACTIVE_TASK", "NO_PRODUCT_GENERATED", "PRODUCT_GENERATED_NOT_READY",
        "QUEUE_EMPTY", "READY_QUEUE_NONEMPTY_BUT_NO_ELIGIBLE_PRODUCT",
        "PRODUCT_DOES_NOT_FIT", "PARTIAL_TRANSMISSION_DISABLED",
        "NO_SECOND_CANDIDATE_TRIED", "DEADLINE_INFEASIBLE",
        "NON_POSITIVE_INCREMENTAL_UTILITY", "REQUIREMENT_ALREADY_SATISFIED",
        "ENERGY_BLOCKED", "THERMAL_BLOCKED", "OTHER_SAFETY_BLOCK",
        "SCHEDULER_RETURNED_NONE", "UNKNOWN",
    }


def test_avoidable_idle_requires_physically_sendable_positive_candidate():
    candidate = dict(generated=True, ready=True, expired=False, safety_allowed=True,
                     incremental_utility=0.2, can_send_one_byte=True)
    assert is_avoidable_idle(10, [candidate])
    candidate["incremental_utility"] = 0.0
    assert not is_avoidable_idle(10, [candidate])


def test_idle_reason_prefers_queue_empty_when_no_queued_product():
    assert classify_idle_reason({"active_task_count": 1, "generated_count": 1,
                                 "ready_count": 1, "queue_count": 0}) == "QUEUE_EMPTY"


def test_idle_reason_identifies_scheduler_none_with_eligible_candidate():
    assert classify_idle_reason({"active_task_count": 1, "generated_count": 1,
                                 "ready_count": 1, "queue_count": 1,
                                 "eligible_count": 1, "scheduler_returned_none": True}) == "SCHEDULER_RETURNED_NONE"
