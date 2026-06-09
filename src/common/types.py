"""项目核心 dataclass 定义。

本文件故意较为集中，是为了给后续的模型、通信、奖励、路由和仿真模块提供统一接口。
这样做可以显著降低不同模块之间的隐式耦合，也方便后续逐步替换 mock 组件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.constants import CONTEXT_NAMES


@dataclass(frozen=True)
class TaskSample:
    """统一的任务样本描述。

    当前版本中的样本可以是 synthetic task，也可以在未来替换为真实 SAR/EO tile。
    关键点在于：下游模块只认 TaskSample，不依赖具体数据集来源。
    """

    task_id: str
    image_path: str
    label_path: str
    scene_id: str
    virtual_time: int
    task_value: float
    deadline_s: float
    difficulty: float
    uncertainty: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceTrace:
    """资源侧外部 trace。

    它代表 timeline 提供的环境驱动量，而不是闭环状态本身。
    例如 battery 是闭环状态，solar/base power 则是 trace 输入。
    """

    slot_index: int
    solar_energy_j: float
    base_power_w: float
    temperature_reference_c: float


@dataclass(frozen=True)
class LinkTrace:
    """链路侧外部 trace。"""

    slot_index: int
    in_contact: bool
    snr_db: float
    throughput_mbps: float
    link_quality: float
    remaining_window_s: float


@dataclass(frozen=True)
class TimelineStep:
    """虚拟时间轴中的一个离散时隙。"""

    slot_index: int
    slot_seconds: float
    task: TaskSample
    resource_trace: ResourceTrace
    link_trace: LinkTrace


@dataclass
class ResourceState:
    """闭环递推得到的资源状态。"""

    battery_soc: float
    ai_energy_budget: float
    thermal_pressure: float


@dataclass
class MissionState:
    """任务执行侧的全局状态。"""

    slot_index: int = 0
    completed_tasks: int = 0
    safety_violations: int = 0


@dataclass
class CommState:
    """通信侧闭环状态。

    这里显式保留物理队列大小 queue_mb；ContextVector 中的 Q_t 只使用其归一化值。
    """

    queue_mb: float
    queue_max_mb: float
    queue_item_count: int = 0
    effective_opportunity: float = 0.0


@dataclass(frozen=True)
class ContextVector:
    """9 维 compact representative context。

    x_t = [B_t, E_t_AI, Theta_t, V_t, D_t, H_t, U_t, C_t, Q_t]

    注意：这是面向 magazine-style case study 的紧凑表示，而不是未来 TWC 扩展版的完整状态空间。
    """

    B_t: float
    E_t_AI: float
    Theta_t: float
    V_t: float
    D_t: float
    H_t: float
    U_t: float
    C_t: float
    Q_t: float

    def as_array(self) -> np.ndarray:
        """返回给 bandit/router 使用的定长向量。"""

        return np.array(
            [
                self.B_t,
                self.E_t_AI,
                self.Theta_t,
                self.V_t,
                self.D_t,
                self.H_t,
                self.U_t,
                self.C_t,
                self.Q_t,
            ],
            dtype=float,
        )

    @property
    def names(self) -> tuple[str, ...]:
        """提供字段名称，便于调试和日志输出。"""

        return CONTEXT_NAMES


@dataclass(frozen=True)
class JointAction:
    """联合动作。

    第一版主动作仅显式包含 model_name 与 payload_policy；
    transmission_mode 作为预留字段，支持后续扩展更复杂的下传策略。
    """

    action_name: str
    model_name: str
    payload_policy: str
    transmission_mode: str
    expected_payload_level: float
    expected_compute_level: float


@dataclass(frozen=True)
class ProfileMetrics:
    """模型 profiling 结果。

    当前版本是 profile-calibrated estimate；
    后续可以替换为真实本地 profiling 表或更复杂的 residual predictor 输入。
    """

    profile_name: str
    latency_s: float
    quality_base: float
    estimated_power_w: float
    robustness: float


@dataclass
class DetectionResult:
    """统一检测输出。

    MockDetector、YoloDetector 以及未来其他 detector 都必须输出同一种格式，
    这样 payload generation 才不会依赖具体模型后端。
    """

    bboxes: list[tuple[float, float, float, float]]
    confidence: float
    target_flag: bool
    roi_regions: list[tuple[float, float, float, float]]
    uncertainty: float
    inference_latency_s: float
    estimated_quality: float
    generated_metadata: dict[str, Any]
    miss_probability: float = 0.0
    false_alarm_probability: float = 0.0
    miss_flag: bool = False
    false_alarm_flag: bool = False
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class GeneratedPayload:
    """检测结果经过 payload policy 实例化后的实际负载。"""

    task_id: str
    payload_policy: str
    payload_mb: float
    information_content: float
    priority_score: float
    value_score: float
    deadline_s: float
    created_slot: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueItem:
    """下传队列中的内部对象。"""

    task_id: str
    payload_policy: str
    remaining_mb: float
    original_payload_mb: float
    priority_score: float
    value_score: float
    created_slot: int
    deadline_slot: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DownlinkResult:
    """一次 contact-aware downlink 处理后的观测值。"""

    S_down_total_mb: float
    S_down_current_task_mb: float
    generated_payload_current_mb: float
    dropped_current_task_mb: float
    queue_next_mb: float
    delta_queue_mb: float
    queue_latency_s: float
    queue_item_count: int
    current_task_delivery_status: str
    transmitted_task_ids: list[str] = field(default_factory=list)


@dataclass
class PredictedMetrics:
    """动作选择前的估计指标。"""

    predicted_S_gen_mb: float
    predicted_S_down_mb: float
    predicted_E_comp: float
    predicted_E_comm: float
    predicted_L_inf: float
    predicted_L_queue: float
    predicted_L_comm: float
    predicted_Q_next_mb: float
    predicted_B_next: float
    predicted_E_AI_next: float
    predicted_Theta_next: float
    predicted_delivery_factor: float
    predicted_safety_violation: float
    predicted_reward: float
    predicted_status: str
    safety_reasons: list[str] = field(default_factory=list)
    predicted_detection_quality: float = 0.0


@dataclass
class ObservedFeedback:
    """动作执行后的 simulation-observed feedback。"""

    S_gen_current_mb: float
    S_down_total_mb: float
    S_down_current_task_mb: float
    queue_next_mb: float
    delta_queue_mb: float
    E_comp: float
    E_comm: float
    L_inf: float
    L_queue: float
    L_comm: float
    delivery_status: str
    delivery_factor: float
    safety_violation: int
    total_latency_s: float
    dropped_current_task_mb: float
    queue_item_count: int
    miss_signal: float = 0.0
    false_alarm_signal: float = 0.0
    detection_quality: float = 0.0


@dataclass
class RewardBreakdown:
    """任务效用 reward 的可解释分解。"""

    detection_quality: float
    information_sufficiency: float
    timeliness: float
    delivery_factor: float
    dynamic_energy_weight: float
    energy_penalty: float
    latency_penalty: float
    queue_penalty: float
    safety_penalty: float
    mission_error_penalty: float
    reward: float


@dataclass
class TrajectoryRecord:
    """输出到轨迹 CSV 的单步记录。"""

    t: int
    task_id: str
    selected_action: str
    model_name: str
    payload_policy: str
    B_t: float
    E_AI_t: float
    Theta_t: float
    V_t: float
    D_t: float
    H_t: float
    U_t: float
    C_t: float
    Q_t: float
    predicted_reward: float
    observed_reward: float
    S_gen_current_mb: float
    S_down_total_mb: float
    S_down_current_task_mb: float
    queue_next_mb: float
    delta_queue_mb: float
    E_comp: float
    E_comm: float
    L_inf: float
    L_queue: float
    L_comm: float
    delivery_status: str
    safety_violation: int


@dataclass
class EpisodeResult:
    """单个 router 的完整回合结果。"""

    router_name: str
    records: list[TrajectoryRecord]
    summary: dict[str, float]
