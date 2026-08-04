"""Episode 执行器。"""

from __future__ import annotations

from src.common.types import EpisodeResult
from src.context.comm_state import build_initial_comm_state
from src.context.mission_state import build_initial_mission_state
from src.context.resource_state import build_initial_resource_state
from src.link.downlink_queue import DownlinkQueue
from src.payload.payload_lifecycle_ledger import PayloadLifecycleLedger
from src.routers.base_router import BaseRouter
from src.simulator.step_runner import StepRunner
from src.evaluation.metrics import summarize_episode


class EpisodeRunner:
    """运行一个 router 的完整 timeline。"""

    def __init__(
        self,
        timeline,
        step_runner_factory,
        simulator_cfg: dict[str, float],
    ) -> None:
        self.timeline = timeline
        self.step_runner_factory = step_runner_factory
        self.simulator_cfg = simulator_cfg
        self.last_candidate_action_logs: list[dict[str, object]] = []
        self.last_queue_priority_logs: list[dict[str, object]] = []
        self.last_carryover_lifecycle_events: list[dict[str, object]] = []
        self.last_same_slot_carryover_trace: list[dict[str, object]] = []
        self.last_staged_feasibility_filter_logs: list[dict[str, object]] = []
        self.last_followup_hard_safety_guard_trace: list[dict[str, object]] = []
        self.last_roi_context_window_budget_trace: list[dict[str, object]] = []
        self.last_roi_context_budget_trace: list[dict[str, object]] = []
        self.last_value_density_trace: list[dict[str, object]] = []
        self.last_same_slot_vs_queued_followup_trace: list[dict[str, object]] = []
        self.last_future_contact_feasibility_trace: list[dict[str, object]] = []
        self.last_duplicate_suppression_trace: list[dict[str, object]] = []
        self.last_staged_delivery_request_trace: list[dict[str, object]] = []
        self.last_staged_delivery_priority_trace: list[dict[str, object]] = []
        self.last_contact_window_scheduling_trace: list[dict[str, object]] = []
        self.last_contact_window_candidate_rank_trace: list[dict[str, object]] = []
        self.last_contact_window_delivery_order_trace: list[dict[str, object]] = []
        self.last_ordinary_queue_vs_staged_queue_capacity_trace: list[dict[str, object]] = []
        self.last_backlog_composition_trace: list[dict[str, object]] = []
        self.last_leftover_tail_value_trace: list[dict[str, object]] = []
        self.last_guard_admission_trace: list[dict[str, object]] = []
        self.last_protected_closing_task_trace: list[dict[str, object]] = []
        self.last_contact_capacity_reservation_trace: list[dict[str, object]] = []
        self.last_payload_lifecycle_ledger_trace: list[dict[str, object]] = []
        self.last_payload_lifecycle_current_records: list[dict[str, object]] = []
        self.last_payload_lifecycle_transition_summary: list[dict[str, object]] = []
        self.last_payload_lifecycle_orphan_check: list[dict[str, object]] = []
        self.last_next_window_recompetition_trace: list[dict[str, object]] = []
        self.last_priority_objective_component_trace: list[dict[str, object]] = []
        self.last_guarantee_debt_trace: list[dict[str, object]] = []
        self.last_guarantee_debt_window_update_trace: list[dict[str, object]] = []
        self.last_task_debt_trajectory_trace: list[dict[str, object]] = []
        self.last_candidate_priority_trace: list[dict[str, object]] = []
        self.last_guarantee_opportunity_trace: list[dict[str, object]] = []
        self.last_guarantee_progress_trace: list[dict[str, object]] = []
        self.last_guarantee_priority_component_trace: list[dict[str, object]] = []
        self.last_guarantee_budget_trace: list[dict[str, object]] = []
        self.last_guarantee_blocker_reason_trace: list[dict[str, object]] = []
        self.last_thermal_safe_capacity_trace: list[dict[str, object]] = []
        self.last_thermal_hysteresis_trace: list[dict[str, object]] = []

    def run(self, router: BaseRouter) -> EpisodeResult:
        resource_state = build_initial_resource_state(self.simulator_cfg)
        mission_state = build_initial_mission_state()
        comm_state = build_initial_comm_state(self.simulator_cfg)
        lifecycle_ledger = PayloadLifecycleLedger(slot_seconds=float(self.simulator_cfg["slot_seconds"]))
        queue = DownlinkQueue(
            queue_max_mb=float(self.simulator_cfg["queue_max_mb"]),
            slot_seconds=float(self.simulator_cfg["slot_seconds"]),
            downlink_efficiency=float(self.simulator_cfg.get("downlink_efficiency", 1.0)),
            queue_priority_tau_d=float(self.simulator_cfg.get("queue_priority_tau_d", 60.0)),
            queue_age_bonus=float(self.simulator_cfg.get("queue_age_bonus", 0.01)),
            queue_meta_first_bonus=float(self.simulator_cfg.get("queue_meta_first_bonus", 0.25)),
            queue_high_value_bonus=float(self.simulator_cfg.get("queue_high_value_bonus", 0.20)),
            continue_request_priority_bonus=float(
                self.simulator_cfg.get("progressive_payload", {})
                .get("phase5_virtual_continue", {})
                .get("continue_request_priority_bonus", 0.0)
            )
            if isinstance(self.simulator_cfg.get("progressive_payload", {}), dict)
            else 0.0,
            queue_pressure_priority_penalty=float(
                self.simulator_cfg.get("progressive_payload", {})
                .get("phase5_virtual_continue", {})
                .get("queue_pressure_priority_penalty", 0.0)
            )
            if isinstance(self.simulator_cfg.get("progressive_payload", {}), dict)
            else 0.0,
            use_uoi_density_priority=bool(self.simulator_cfg.get("tut_uoi_density_priority_enabled", False)),
            lifecycle_ledger=lifecycle_ledger,
        )
        step_runner: StepRunner = self.step_runner_factory(queue)
        self.last_candidate_action_logs = []
        self.last_queue_priority_logs = []
        self.last_carryover_lifecycle_events = []
        self.last_same_slot_carryover_trace = []
        self.last_staged_feasibility_filter_logs = []
        self.last_followup_hard_safety_guard_trace = []
        self.last_roi_context_window_budget_trace = []
        self.last_roi_context_budget_trace = []
        self.last_value_density_trace = []
        self.last_same_slot_vs_queued_followup_trace = []
        self.last_future_contact_feasibility_trace = []
        self.last_duplicate_suppression_trace = []
        self.last_staged_delivery_request_trace = []
        self.last_staged_delivery_priority_trace = []
        self.last_contact_window_scheduling_trace = []
        self.last_contact_window_candidate_rank_trace = []
        self.last_contact_window_delivery_order_trace = []
        self.last_ordinary_queue_vs_staged_queue_capacity_trace = []
        self.last_backlog_composition_trace = []
        self.last_leftover_tail_value_trace = []
        self.last_guard_admission_trace = []
        self.last_protected_closing_task_trace = []
        self.last_contact_capacity_reservation_trace = []
        self.last_payload_lifecycle_ledger_trace = []
        self.last_payload_lifecycle_current_records = []
        self.last_payload_lifecycle_transition_summary = []
        self.last_payload_lifecycle_orphan_check = []
        self.last_next_window_recompetition_trace = []
        self.last_priority_objective_component_trace = []
        self.last_guarantee_debt_trace = []
        self.last_guarantee_debt_window_update_trace = []
        self.last_task_debt_trajectory_trace = []
        self.last_candidate_priority_trace = []
        self.last_guarantee_opportunity_trace = []
        self.last_guarantee_progress_trace = []
        self.last_guarantee_priority_component_trace = []
        self.last_guarantee_budget_trace = []
        self.last_guarantee_blocker_reason_trace = []
        self.last_thermal_safe_capacity_trace = []
        self.last_thermal_hysteresis_trace = []
        records = []
        for timeline_step in self.timeline:
            record, _reward = step_runner.run_step(timeline_step, resource_state, mission_state, comm_state, router)
            records.append(record)
            self.last_candidate_action_logs.extend(
                dict(row, router_name=router.router_name) for row in step_runner.last_candidate_action_log
            )
            self.last_thermal_safe_capacity_trace.extend(
                dict(row, router_name=router.router_name) for row in getattr(step_runner, "last_thermal_safe_capacity_trace", [])
            )
            self.last_thermal_hysteresis_trace.extend(
                dict(row, router_name=router.router_name) for row in getattr(step_runner, "last_thermal_hysteresis_trace", [])
            )
            self.last_queue_priority_logs.extend(
                dict(row, router_name=router.router_name) for row in step_runner.last_queue_priority_log
            )
            self.last_carryover_lifecycle_events.extend(
                dict(row, router_name=router.router_name) for row in step_runner.last_carryover_lifecycle_events
            )
            self.last_same_slot_carryover_trace.extend(
                dict(row, router_name=router.router_name) for row in step_runner.last_same_slot_carryover_trace
            )
            self.last_staged_feasibility_filter_logs.extend(
                dict(row, router_name=router.router_name) for row in step_runner.last_staged_feasibility_filter_trace
            )
            self.last_followup_hard_safety_guard_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_followup_hard_safety_guard_trace", [])
            )
            self.last_roi_context_window_budget_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_roi_context_window_budget_trace", [])
            )
            self.last_roi_context_budget_trace = self.last_roi_context_window_budget_trace
            self.last_value_density_trace.extend(
                dict(row, router_name=router.router_name) for row in getattr(step_runner, "last_value_density_trace", [])
            )
            self.last_same_slot_vs_queued_followup_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_same_slot_vs_queued_followup_trace", [])
            )
            self.last_future_contact_feasibility_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_future_contact_feasibility_trace", [])
            )
            self.last_duplicate_suppression_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_duplicate_suppression_trace", [])
            )
            self.last_staged_delivery_request_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_staged_delivery_request_trace", [])
            )
            self.last_staged_delivery_priority_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_staged_delivery_priority_trace", [])
            )
            self.last_contact_window_scheduling_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_contact_window_scheduling_trace", [])
            )
            self.last_contact_window_candidate_rank_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_contact_window_candidate_rank_trace", [])
            )
            self.last_contact_window_delivery_order_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_contact_window_delivery_order_trace", [])
            )
            self.last_ordinary_queue_vs_staged_queue_capacity_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_ordinary_queue_vs_staged_queue_capacity_trace", [])
            )
            self.last_backlog_composition_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_backlog_composition_trace", [])
            )
            self.last_leftover_tail_value_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_leftover_tail_value_trace", [])
            )
            self.last_guard_admission_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guard_admission_trace", [])
            )
            self.last_protected_closing_task_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_protected_closing_task_trace", [])
            )
            self.last_contact_capacity_reservation_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_contact_capacity_reservation_trace", [])
            )
            self.last_guarantee_debt_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guarantee_debt_trace", [])
            )
            self.last_guarantee_debt_window_update_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guarantee_debt_window_update_trace", [])
            )
            self.last_task_debt_trajectory_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_task_debt_trajectory_trace", [])
            )
            self.last_candidate_priority_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_candidate_priority_trace", [])
            )
            self.last_guarantee_opportunity_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guarantee_opportunity_trace", [])
            )
            self.last_guarantee_progress_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guarantee_progress_trace", [])
            )
            self.last_guarantee_priority_component_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guarantee_priority_component_trace", [])
            )
            self.last_guarantee_budget_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guarantee_budget_trace", [])
            )
            self.last_guarantee_blocker_reason_trace.extend(
                dict(row, router_name=router.router_name)
                for row in getattr(step_runner, "last_guarantee_blocker_reason_trace", [])
            )

        self.last_payload_lifecycle_ledger_trace = [
            dict(row, router_name=router.router_name) for row in lifecycle_ledger.transition_trace
        ]
        self.last_payload_lifecycle_current_records = [
            dict(row, router_name=router.router_name) for row in lifecycle_ledger.current_rows()
        ]
        self.last_payload_lifecycle_transition_summary = [
            dict(row, router_name=router.router_name) for row in lifecycle_ledger.transition_summary_rows()
        ]
        self.last_payload_lifecycle_orphan_check = [
            dict(row, router_name=router.router_name)
            for row in lifecycle_ledger.orphan_check_rows(
                queue_items=queue.items,
                staged_requests=getattr(step_runner, "staged_delivery_requests", {}).values(),
            )
        ]
        self.last_next_window_recompetition_trace = [
            dict(row, router_name=router.router_name) for row in lifecycle_ledger.recompetition_trace
        ]
        self.last_priority_objective_component_trace = [
            dict(row, router_name=router.router_name) for row in lifecycle_ledger.priority_component_trace
        ]

        summary = summarize_episode(
            router_name=router.router_name,
            records=records,
            queue_max_mb=float(self.simulator_cfg["queue_max_mb"]),
            final_battery=resource_state.battery_soc,
            max_thermal=max([record.thermal_next for record in records] + [resource_state.thermal_pressure]),
            battery_min=float(self.simulator_cfg["battery_min"]),
            thermal_max=float(self.simulator_cfg["thermal_max"]),
            high_value_threshold=float(self.simulator_cfg.get("high_value_threshold", 0.75)),
            risk_cfg=self.simulator_cfg.get("long_term_risk_penalty", {}),
            final_thermal=resource_state.thermal_pressure,
            final_queue_mb=comm_state.queue_mb,
        )
        return EpisodeResult(router_name=router.router_name, records=records, summary=summary)
