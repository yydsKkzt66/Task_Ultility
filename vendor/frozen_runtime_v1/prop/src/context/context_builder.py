"""9 维 context 构建器。"""

from __future__ import annotations

from src.common.types import CommState, ContextVector, ResourceState, TaskSample
from src.context.comm_state import compute_effective_downlink_opportunity
from src.data.value_skew_calibration import task_effective_value


class ContextBuilder:
    """负责把当前任务、资源状态和链路状态压缩为固定 9 维上下文。"""

    def __init__(self, queue_max_mb: float) -> None:
        self.queue_max_mb = queue_max_mb

    def build(
        self,
        task: TaskSample | None,
        resource_state: ResourceState,
        comm_state: CommState,
        link_trace,
    ) -> ContextVector:
        """构建当前时隙的紧凑上下文。

        D_t 这里采用“deadline urgency”的定义，即越接近截止期限越接近 1。
        """

        if task is None:
            deadline_urgency = 0.0
            task_value = 0.0
            difficulty = 0.0
            uncertainty = 0.0
        else:
            deadline_tolerance = min(task.deadline_s / 120.0, 1.0)
            deadline_urgency = 1.0 - deadline_tolerance
            task_value = task_effective_value(task)
            difficulty = task.difficulty
            uncertainty = task.uncertainty
        queue_pressure = min(comm_state.queue_mb / max(self.queue_max_mb, 1.0e-9), 1.0)
        downlink_opportunity = compute_effective_downlink_opportunity(link_trace)
        return ContextVector(
            B_t=resource_state.battery_soc,
            E_t_AI=resource_state.ai_energy_budget,
            Theta_t=resource_state.thermal_pressure,
            V_t=task_value,
            D_t=deadline_urgency,
            H_t=difficulty,
            U_t=uncertainty,
            C_t=downlink_opportunity,
            Q_t=queue_pressure,
        )
