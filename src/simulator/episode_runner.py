"""Episode 执行器。"""

from __future__ import annotations

from src.common.types import EpisodeResult
from src.context.comm_state import build_initial_comm_state
from src.context.mission_state import build_initial_mission_state
from src.context.resource_state import build_initial_resource_state
from src.link.downlink_queue import DownlinkQueue
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

    def run(self, router: BaseRouter) -> EpisodeResult:
        resource_state = build_initial_resource_state(self.simulator_cfg)
        mission_state = build_initial_mission_state()
        comm_state = build_initial_comm_state(self.simulator_cfg)
        queue = DownlinkQueue(
            queue_max_mb=float(self.simulator_cfg["queue_max_mb"]),
            slot_seconds=float(self.simulator_cfg["slot_seconds"]),
        )
        step_runner: StepRunner = self.step_runner_factory(queue)
        records = []
        for timeline_step in self.timeline:
            record, _reward = step_runner.run_step(timeline_step, resource_state, mission_state, comm_state, router)
            records.append(record)

        summary = summarize_episode(
            router_name=router.router_name,
            records=records,
            queue_max_mb=float(self.simulator_cfg["queue_max_mb"]),
            final_battery=resource_state.battery_soc,
            max_thermal=max([record.Theta_t for record in records] + [resource_state.thermal_pressure]),
        )
        return EpisodeResult(router_name=router.router_name, records=records, summary=summary)
