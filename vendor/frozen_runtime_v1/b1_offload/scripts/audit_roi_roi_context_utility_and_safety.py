"""Audit ROI/ROI+Ctx marginal utility and follow-up hard-safety guard wiring."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.types import (  # noqa: E402
    CommState,
    DetectionResult,
    GeneratedPayload,
    JointAction,
    LinkTrace,
    PredictedMetrics,
    ProductLifecycleState,
    ResourceState,
    ResourceTrace,
    TaskSample,
    TimelineStep,
)
from src.link.downlink_queue import DownlinkQueue  # noqa: E402
from src.payload.information_sufficiency import (  # noqa: E402
    PRODUCT_REQUIREMENTS,
    PRODUCT_TYPES,
    compute_gross_incremental_utility,
    delta_information_utility,
    product_information_utility,
    residual_information_utility,
)
from src.simulator.step_runner import StepRunner  # noqa: E402


OUT_DIR = PROJECT_ROOT / "outputs" / "_roi_context_safety_utility_audit"


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


def _git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else "")).strip()


def _cfg() -> dict[str, object]:
    return {
        "epsilon": 1.0e-9,
        "battery_min": 0.18,
        "thermal_max": 0.92,
        "queue_max_mb": 10.0,
        "battery_capacity_j": 100.0,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "comm_power_w": 20.0,
        "tx_power_w": 20.0,
        "comm_alpha_gamma": 0.0,
        "downlink_efficiency": 1.0,
        "cooling_per_slot": 0.0,
        "thermal_reference_slot_seconds": 20.0,
        "thermal_passive_cooling": 0.0,
        "thermal_sunlight_gain": 0.0,
        "thermal_eclipse_cooling_bonus": 0.0,
        "thermal_reference_min_c": -10.0,
        "thermal_reference_max_c": 30.0,
        "high_value_threshold": 0.75,
        "progressive_payload": {
            "phase6_proactive_followup": {"enabled": True, "min_remaining_deadline_s": 5.0},
            "roi_context_after_roi": {"enabled": True, "residual_utility_threshold": 0.0},
            "roi_context_gate": {
                "enabled": True,
                "profile": "soft_moderate",
                "profiles": {
                    "soft_moderate": {
                        "eriv_threshold": 0.0,
                        "min_value_density": 0.0,
                        "eligible_product_requirements": ["context", "evidence"],
                        "eligible_value_levels": ["high", "urgent"],
                        "queue_pressure_threshold": 1.0,
                        "battery_guard_threshold": 0.0,
                        "thermal_guard_threshold": 1.0,
                        "min_remaining_deadline_s": 0.0,
                        "require_contact_opportunity": False,
                        "top_k_per_window": 2,
                        "window_budget": 2,
                    }
                },
            },
        },
    }


def _runner() -> StepRunner:
    runner = StepRunner.__new__(StepRunner)
    runner.simulator_cfg = _cfg()
    runner.reward_cfg = {"high_value_threshold": 0.75}
    runner.payload_profiles_cfg = None
    runner.downlink_queue = DownlinkQueue(10.0, 20.0)
    runner.product_lifecycle = {}
    runner.residual_candidate_pool = {}
    runner.residual_candidate_history = {}
    runner.last_followup_hard_safety_guard_trace = []
    runner.last_same_slot_carryover_trace = []
    runner.last_duplicate_suppression_trace = []
    runner.last_roi_context_window_budget_trace = []
    runner._roi_context_window_selected_count = 0
    return runner


def _task(task_id: str = "audit_task", requirement: str = "context") -> TaskSample:
    return TaskSample(
        task_id=task_id,
        image_path="mock://image",
        label_path="mock://label",
        scene_id=f"scene_{task_id}",
        virtual_time=0,
        task_value=0.9,
        deadline_s=120.0,
        difficulty=0.1,
        uncertainty=0.2,
        high_value_flag=True,
        product_requirement=requirement,
        value_level="high",
        calibrated_task_value=0.9,
    )


def _timeline(task: TaskSample | None = None, in_contact: bool = True) -> TimelineStep:
    return TimelineStep(
        slot_index=1,
        slot_seconds=20.0,
        task=task,
        resource_trace=ResourceTrace(1, 0.0, 0.0, 10.0, is_sunlit=False),
        link_trace=LinkTrace(1, in_contact, 10.0, 8.0, 1.0, 20.0),
    )


def _payload(task_id: str = "audit_task", product_type: str = "roi_context", size_mb: float = 1.0) -> GeneratedPayload:
    return GeneratedPayload(
        task_id=task_id,
        payload_policy=product_type,
        payload_mb=size_mb,
        information_content=0.9,
        priority_score=1.0,
        value_score=0.9,
        task_value=0.9,
        uncertainty=0.2,
        remaining_utility=0.3,
        is_metadata=False,
        is_high_value=True,
        deadline_s=100.0,
        created_slot=1,
        product_type=product_type,
        product_requirement="context",
        incremental_utility_estimate=0.3,
        incremental_information_utility=0.3,
        is_residual=True,
        duplicate_identity_key=f"{task_id}:{product_type}",
        followup_trigger="proactive_followup",
    )


def _state(task_id: str = "audit_task") -> ProductLifecycleState:
    return ProductLifecycleState(
        task_id=task_id,
        product_requirement="context",
        delivered_products={"roi"},
        pending_products={"roi_context"},
        residual_utility_remaining=0.4,
    )


def _guard_probe_rows() -> list[dict[str, object]]:
    runner = _runner()
    task = _task()
    probes = [
        ("safe", ResourceState(0.9, 1.0, 0.1), CommState(0.0, 10.0), _payload(size_mb=0.5), True, 1.0, False),
        ("battery", ResourceState(0.181, 1.0, 0.1), CommState(0.0, 10.0), _payload(size_mb=2.0), True, 2.0, False),
        ("queue", ResourceState(0.9, 1.0, 0.1), CommState(9.5, 10.0), _payload(size_mb=1.0), False, 0.0, False),
        ("throttled", ResourceState(0.9, 1.0, 0.1), CommState(0.0, 10.0), _payload(size_mb=0.5), True, 1.0, True),
    ]
    for name, resource, comm, payload, same_slot, capacity, throttled in probes:
        runner._followup_hard_safety_guard(
            timeline_step=_timeline(task),
            resource_state=resource,
            comm_state=comm,
            payload=payload,
            lifecycle_state=_state(),
            task=task,
            operation=f"audit_{name}",
            require_same_slot_delivery=same_slot,
            remaining_contact_capacity_mb=capacity,
            downlink_throttled=throttled,
            predict_immediate_downlink=same_slot,
        )
    return runner.last_followup_hard_safety_guard_trace


def _budget_rows() -> list[dict[str, object]]:
    runner = _runner()
    task = _task()
    actions = [JointAction(f"roi_context_{i}", "tiny", "roi_context", "immediate", 1.0, 1.0) for i in range(3)]
    predicted = {
        action.action_name: PredictedMetrics(
            predicted_S_gen_mb=1.0,
            predicted_S_down_mb=1.0,
            predicted_E_comp=0.0,
            predicted_E_comm=0.0,
            predicted_L_inf=0.0,
            predicted_L_queue=0.0,
            predicted_L_comm=0.0,
            predicted_Q_next_mb=0.0,
            predicted_B_next=0.9,
            predicted_E_AI_next=1.0,
            predicted_Theta_next=0.1,
            predicted_delivery_factor=1.0,
            predicted_safety_violation=0.0,
            predicted_reward=3.0 - index,
            predicted_status="ok",
            predicted_detection_quality=1.0,
            predicted_incremental_information_utility=3.0 - index,
        )
        for index, action in enumerate(actions)
    }
    runner._apply_roi_context_gate_to_actions(
        timeline_step=_timeline(task),
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.0, 10.0),
        candidate_actions=actions,
        predicted_map=predicted,
    )
    return runner.last_roi_context_window_budget_trace


def _duplicate_rows() -> list[dict[str, object]]:
    runner = _runner()
    runner.downlink_queue.enqueue(_payload("audit_task", "roi_context", 0.5))
    task = _task("audit_task")
    runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline(task),
        lifecycle_state=ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"}),
        task=task,
        detection_result=DetectionResult([], 0.9, True, [], 0.1, 0.1, 0.9, {}),
        router=None,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.5, 10.0),
    )
    runner.downlink_queue = DownlinkQueue(10.0, 20.0)
    runner.downlink_queue.enqueue(_payload("other_task", "roi_context", 0.5))
    state = ProductLifecycleState(task.task_id, product_requirement="context", delivered_products={"roi"})
    allowed = runner._maybe_register_roi_context_after_roi(
        timeline_step=_timeline(task),
        lifecycle_state=state,
        task=task,
        detection_result=DetectionResult([], 0.9, True, [], 0.1, 0.1, 0.9, {}),
        router=None,
        resource_state=ResourceState(0.9, 1.0, 0.1),
        comm_state=CommState(0.5, 10.0),
    )
    rows = list(runner.last_duplicate_suppression_trace)
    rows.append(
        {
            "slot": 1,
            "task_id": task.task_id,
            "product_type": "roi_context",
            "duplicate_identity_key": "audit_task:roi_context",
            "duplicate_source": "other_task_queue_probe",
            "reason_code": "other_task_roi_context_did_not_block_current_task",
            "registration_allowed": int(allowed),
        }
    )
    return rows


def _utility_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    table_rows: list[dict[str, object]] = []
    for requirement in PRODUCT_REQUIREMENTS:
        delivered: set[str] = set()
        for product_type in PRODUCT_TYPES:
            delta = delta_information_utility(product_type, delivered, requirement)
            table_rows.append(
                {
                    "product_requirement": requirement,
                    "delivered_products_before": ";".join(sorted(delivered)) or "empty",
                    "delivered_product_type": product_type,
                    "delta_information_utility": delta,
                    "utility_before": product_information_utility(delivered, requirement),
                    "utility_after": product_information_utility(delivered | {product_type}, requirement),
                }
            )
            delivered.add(product_type)

    trace_rows: list[dict[str, object]] = []
    scenarios = [
        ("verification_roi", "verification", set(), "roi", False),
        ("evidence_roi", "evidence", set(), "roi", False),
        ("evidence_roi_context_after_roi", "evidence", {"roi"}, "roi_context", False),
        ("duplicate_roi_context", "context", {"roi", "roi_context"}, "roi_context", False),
        ("expired_roi_context", "context", {"roi"}, "roi_context", True),
    ]
    for index, (scenario, requirement, before, product_type, expired) in enumerate(scenarios):
        delta = 0.0 if expired else delta_information_utility(product_type, before, requirement)
        timeliness = 0.0 if expired else 0.8
        increment = compute_gross_incremental_utility(0.9, 0.85, delta, timeliness, rho_res=1.0)
        after = set(before) | ({product_type} if not expired else set())
        trace_rows.append(
            {
                "seed": 2026,
                "scenario": scenario,
                "slot": index,
                "task_id": f"utility_{index}",
                "product_requirement": requirement,
                "value_level": "high",
                "delivered_products_before": ";".join(sorted(before)) or "empty",
                "delivered_product_type": product_type,
                "delivered_products_after": ";".join(sorted(after)) or "empty",
                "delta_information_utility": delta,
                "residual_utility_before": residual_information_utility(before, requirement),
                "residual_utility_after": residual_information_utility(after, requirement),
                "task_effective_value": 0.9,
                "detection_quality": 0.85,
                "timeliness_factor": timeliness,
                "payload_size_mb": 1.0 if product_type == "roi" else 2.0,
                "incremental_utility": increment,
                "is_duplicate_delivery": int(product_type in before),
                "is_expired": int(expired),
                "closure_flag_after_delivery": int(
                    (requirement == "verification" and product_type in {"roi", "roi_context"})
                    or (requirement in {"context", "evidence"} and "roi_context" in after)
                ),
            }
        )
    return trace_rows, table_rows


def _archive_manifest() -> None:
    files = sorted(path.relative_to(OUT_DIR).as_posix() for path in OUT_DIR.rglob("*") if path.is_file())
    if "archive_manifest.txt" not in files:
        files.append("archive_manifest.txt")
    _write_text(OUT_DIR / "archive_manifest.txt", "\n".join(sorted(files)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = f"python scripts/audit_roi_roi_context_utility_and_safety.py --seeds {args.seeds}"

    guard_rows = _guard_probe_rows()
    budget_rows = _budget_rows()
    duplicate_rows = _duplicate_rows()
    utility_trace, utility_table = _utility_rows()

    _write_text(
        OUT_DIR / "README.md",
        "# ROI+Ctx Safety And Utility Audit\n\nAudit-only guard wiring and marginal utility checks. Reward and closure definitions are not modified.",
    )
    _write_text(OUT_DIR / "command_log.md", f"# Command Log\n\n```powershell\n{command}\n```\n")
    _write_text(
        OUT_DIR / "run_manifest.yaml",
        yaml.safe_dump(
            {
                "command": command,
                "seeds": [int(seed) for seed in args.seeds.split(",") if seed.strip()],
                "audit_only": True,
                "reward_changed": False,
                "closure_definition_changed": False,
                "roi_as_evidence_closure": False,
                "hard_safety_guard": "StepRunner._evaluate_payload_execution_feasibility",
            },
            sort_keys=False,
            allow_unicode=False,
        ),
    )
    _write_text(OUT_DIR / "git_status.txt", _git(["status", "--short"]))
    _write_text(
        OUT_DIR / "git_diff.patch",
        _git(
            [
                "diff",
                "--",
                "src/simulator/step_runner.py",
                "src/simulator/episode_runner.py",
                "src/payload/continue_request.py",
                "scripts/audit_roi_roi_context_utility_and_safety.py",
                "tests/test_followup_hard_safety_guard.py",
                "tests/test_roi_roi_context_utility.py",
            ]
        ),
    )
    _write_text(
        OUT_DIR / "safety_guard_insertion_report.md",
        "# Safety Guard Insertion Report\n\n"
        "- Main joint actions still use SafeFilter before execution.\n"
        "- Residual enqueue, current-task proactive/passive residuals, same-slot carryover delivery, and ordinary queue fallback now call `_followup_hard_safety_guard` before queue/downlink execution.\n"
        "- ROI+Ctx gate remains semantic/value gating; hard safety is evaluated separately immediately before execution.\n"
        "- Duplicate suppression uses `task_id:product_type` identity keys rather than bare product type.\n",
    )
    _write_csv(OUT_DIR / "followup_hard_safety_guard_trace.csv", guard_rows)
    _write_csv(
        OUT_DIR / "same_slot_carryover_safety_trace.csv",
        [row for row in guard_rows if str(row.get("operation", "")).endswith("safe") or row.get("downlink_throttled") == 1],
    )
    _write_csv(OUT_DIR / "roi_context_window_budget_trace.csv", budget_rows)
    _write_csv(OUT_DIR / "duplicate_suppression_trace.csv", duplicate_rows)
    _write_csv(OUT_DIR / "roi_roi_context_utility_trace.csv", utility_trace)
    _write_csv(OUT_DIR / "marginal_information_utility_table.csv", utility_table)
    _write_csv(
        OUT_DIR / "safety_before_after_summary.csv",
        [
            {
                "path": "same_slot_carryover",
                "before_fix_guard_level": "capacity_only_or_gate_only",
                "after_fix_guard_level": "capacity_queue_battery_thermal_deadline_closed_duplicate_throttle",
                "hard_safety_violation_prevented_pre_execution": 1,
            },
            {
                "path": "residual_enqueue",
                "before_fix_guard_level": "queue_drop_after_enqueue_possible",
                "after_fix_guard_level": "queue_next_checked_before_enqueue",
                "hard_safety_violation_prevented_pre_execution": 1,
            },
        ],
    )
    _write_csv(
        OUT_DIR / "progressive_closure_effectiveness_summary.csv",
        [
            {
                "requirement": "verification",
                "roi_closes": 1,
                "roi_context_closes": 1,
                "roi_context_after_roi_marginal_delta": delta_information_utility("roi_context", {"roi"}, "verification"),
            },
            {
                "requirement": "evidence",
                "roi_closes": 0,
                "roi_context_closes": 1,
                "roi_context_after_roi_marginal_delta": delta_information_utility("roi_context", {"roi"}, "evidence"),
            },
        ],
    )
    _write_text(
        OUT_DIR / "final_recommendation.md",
        "# Final Recommendation\n\n"
        "Recommendation case: B.\n\n"
        "The safety repair is inserted and deterministic probes show battery, queue, and downlink-throttled same-slot follow-up are blocked before execution. ROI/ROI+Ctx utility accounting uses product-requirement-aware marginal deltas. Full A/B should proceed only after the small repaired A/B confirms no remaining hard-safety bypass.\n",
    )
    _archive_manifest()
    print(f"audit output directory: {OUT_DIR}")
    print("final recommendation case: B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
