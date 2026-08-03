"""Independent B1 baseline with OFFLOAD's shared finite ground server.

This module intentionally contains no ground-queue implementation.  The only
queue used at runtime is ``src.baselines.offload_eft.FiniteGroundComputeQueue``
from the frozen native B1/OFFLOAD source tree.  Everything up to and including
compressed-frame downlink remains delegated to ``B1EnhancedSimulator``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.baselines.b1_enhanced import B1EnhancedSimulator, GROUND_RESULT_STATE
from src.baselines.offload_eft import FiniteGroundComputeQueue, GroundJob


METHOD_ID = "B1-COMMON-GROUND"
GROUND_SERVERS = 1
GROUND_QUEUE_CAPACITY = 32
GROUND_QUEUE_POLICY = "fifo"


class B1CommonGroundSimulator(B1EnhancedSimulator):
    """B1 with the OFFLOAD shared single-server FIFO ground-compute system."""

    def __init__(
        self,
        config: Any,
        simulator_cfg: dict[str, Any],
        model_pool_cfg: dict[str, Any],
        *,
        ground_servers: int = GROUND_SERVERS,
        ground_queue_capacity: int = GROUND_QUEUE_CAPACITY,
    ) -> None:
        super().__init__(config, simulator_cfg, model_pool_cfg)
        self.ground_compute_queue = FiniteGroundComputeQueue(
            servers=ground_servers,
            max_queue_items=ground_queue_capacity,
        )
        self.ground_jobs_by_task: dict[str, GroundJob] = {}
        self.ground_queue_rejection_count = 0

    def _commit_new_ground_results(self) -> None:
        """Submit fully received frames to OFFLOAD's finite server exactly once."""
        for task_id, item in self.tasks.items():
            if task_id in self.ground_jobs_by_task:
                continue
            if item.state != GROUND_RESULT_STATE:
                continue
            received_s = item.ground_received_compressed_time_s
            if received_s is None:
                raise RuntimeError(f"ground result exists before full receive: {task_id}")
            duration_s = item.ground_decode_latency_s + item.ground_inference_latency_s
            try:
                job = self.ground_compute_queue.commit(task_id, received_s, duration_s)
            except OverflowError:
                # The compressed frame reached the ground, but the shared finite
                # service queue rejected it.  It is not a closed task.
                item.state = "ground_compute_queue_rejected"
                item.ground_decode_start_time_s = None
                item.ground_decode_end_time_s = None
                item.ground_inference_start_time_s = None
                item.ground_result_available_time_s = None
                item.capture_to_ground_latency_s = 0.0
                item.queue_wait_latency_s = 0.0
                item.late_delivery = True
                item.blocked_reason = "ground_compute_queue_full"
                self.ground_queue_rejection_count += 1
                continue

            self.ground_jobs_by_task[task_id] = job
            item.ground_decode_start_time_s = job.compute_start_time
            item.ground_decode_end_time_s = job.compute_start_time + item.ground_decode_latency_s
            item.ground_inference_start_time_s = item.ground_decode_end_time_s
            item.ground_result_available_time_s = job.compute_end_time
            item.capture_to_ground_latency_s = job.compute_end_time - item.capture_time_s
            item.queue_wait_latency_s = job.waiting_time
            item.late_delivery = job.compute_end_time > item.deadline_time_s + 1.0e-9

    def _process_downlink(self, step: Any, resource_state: Any, baseline_compute_energy_j: float = 0.0) -> tuple[float, float]:
        # The parent performs the unchanged B1 scheduler, safety checks,
        # transmission and complete-frame reception.  Only after that complete
        # receive event do we replace its zero-queue ground timing.
        sent_mb, comm_energy_j = super()._process_downlink(
            step,
            resource_state,
            baseline_compute_energy_j,
        )
        self._commit_new_ground_results()
        return sent_mb, comm_energy_j

    def _task_row(self, item: Any) -> dict[str, Any]:
        row = super()._task_row(item)
        job = self.ground_jobs_by_task.get(item.task.task_id)
        row.update(
            {
                "ground_queue_entry_time_s": job.queue_entry_time if job else "",
                "ground_compute_start_time_s": job.compute_start_time if job else "",
                "ground_compute_end_time_s": job.compute_end_time if job else "",
                "ground_queue_wait_time_s": job.waiting_time if job else "",
                "ground_server_id": job.server_id if job else "",
                "ground_queue_length_at_entry": job.queue_length_at_entry if job else "",
            }
        )
        return row

    def run(self, timeline: list[Any]) -> dict[str, Any]:
        result = super().run(timeline)
        jobs = [asdict(job) for job in self.ground_compute_queue.jobs]
        result["ground_compute_queue"] = jobs
        result["summary"].update(
            {
                "method_name": METHOD_ID,
                "ground_compute_servers": self.ground_compute_queue.servers,
                "ground_queue_policy": GROUND_QUEUE_POLICY,
                "ground_queue_capacity": self.ground_compute_queue.max_queue_items,
                "ground_queue_rejection_count": self.ground_queue_rejection_count,
                "ground_queue_peak_waiting_tasks": max(
                    (job.queue_length_at_entry for job in self.ground_compute_queue.jobs),
                    default=0,
                ),
                "ground_queue_mean_wait_time_s": (
                    sum(job.waiting_time for job in self.ground_compute_queue.jobs)
                    / max(len(self.ground_compute_queue.jobs), 1)
                ),
            }
        )
        return result
