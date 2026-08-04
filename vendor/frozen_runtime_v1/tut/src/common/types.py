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
    arrival_slot: int = 0
    observation_window_id: str = ""
    task_class: str = "routine"
    quality_requirement: float | None = None
    information_requirement: float | None = None
    high_value_flag: bool = False
    emergency_flag: bool = False
    product_requirement: str = "verification"
    product_requirement_source: str = "mission_prior_proxy"
    base_prior_score: float | None = None
    value_level: str = ""
    value_multiplier: float = 1.0
    value_skew_profile: str = "legacy"
    normalization_alpha: float = 1.0
    calibrated_task_value: float | None = None
    value_assignment_reason: str = ""


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
    is_sunlit: bool = True
    is_eclipse: bool = False
    orbit_index: int = 0
    day_index: int = 0
    orbit_phase: float = 0.0
    solar_power_w: float = 0.0
    wasted_solar_energy_j: float = 0.0
    battery_soc: float = 0.0
    dod: float = 0.0
    dod_guard_violation: int = 0
    compute_power_w: float = 0.0
    comm_power_w: float = 0.0


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
    task: TaskSample | None
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
class LongTermState:
    """涓婂眰闀挎椂鍩熻祫婧愯皟鍒舵墍闇€鐨勭揣鍑戠姸鎬併€?"""

    battery_soc: float
    battery_safe: float
    thermal_pressure: float
    thermal_max: float
    queue_mb: float
    queue_max_mb: float
    recent_deadline_urgency: float
    recent_high_value_ratio: float
    next_contact_s: float
    low_contact_capacity_factor: float
    queue_growth_rate: float = 0.0


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
    payload_plan: str = ""


@dataclass
class ProductLifecycleState:
    """Mutable per-task product delivery lifecycle."""

    task_id: str
    product_requirement: str = "verification"
    delivered_products: set[str] = field(default_factory=set)
    pending_products: set[str] = field(default_factory=set)
    first_alert_delivered_time_s: float | None = None
    metadata_delivery_slot: int | None = None
    pending_residual_registered_slot: int | None = None
    residual_utility_remaining: float = 0.0
    task_closed_flag: bool = False
    virtual_continue_request_flag: bool = False
    residual_requested_count: int = 0
    residual_enqueued_count: int = 0
    residual_delivered_count: int = 0
    residual_expired_count: int = 0
    residual_dropped_count: int = 0
    duplicate_residual_suppression_count: int = 0
    residual_utility_generated: float = 0.0
    residual_utility_delivered: float = 0.0
    residual_utility_lost: float = 0.0
    time_from_metadata_to_followup_s: float = 0.0
    residual_queue_wait_s: float = 0.0
    residual_parent_payload_plan: str = ""
    residual_request_time_s: float | None = None
    residual_enqueue_time_s: float | None = None
    proactive_followup_attempt_count: int = 0
    proactive_followup_enqueue_count: int = 0
    proactive_followup_delivery_count: int = 0
    proactive_followup_expired_count: int = 0
    proactive_followup_dropped_count: int = 0
    proactive_duplicate_suppression_count: int = 0
    proactive_residual_enqueue_time_s: float | None = None
    proactive_before_passive_request_count: int = 0
    passive_request_suppressed_by_prior_proactive_count: int = 0
    residual_utility_saved_by_proactive: float = 0.0
    carryover_proactive_enqueue_count: int = 0
    carryover_proactive_delivery_count: int = 0
    carryover_proactive_incremental_utility: float = 0.0


@dataclass
class ResidualCandidateState:
    """Carryover state for metadata-delivered tasks with pending residual value."""

    task_id: str
    scene_id: str
    product_requirement: str
    task_value: float
    high_value_flag: bool
    emergency_flag: bool
    payload_plan: str
    residual_product_type: str
    metadata_delivery_slot: int
    metadata_delivery_time_s: float | None
    pending_residual_registered_slot: int
    residual_utility_remaining: float
    residual_delta_u_I: float
    deadline_s: float
    created_slot: int
    last_evaluated_slot: int = -1
    evaluation_count: int = 0
    candidate_age_s: float = 0.0
    delivered_products_snapshot: set[str] = field(default_factory=set)
    pending_products_snapshot: set[str] = field(default_factory=set)
    duplicate_identity_key: str = ""
    source: str = "metadata_delivered_pending_residual"
    leakage_safe_flag: bool = True
    task_snapshot: Any | None = None
    detection_result_snapshot: Any | None = None
    value_level: str = ""


@dataclass(frozen=True)
class ProfileMetrics:
    """模型 profiling 结果。

    当前版本是 profile-calibrated estimate；
    后续可以替换为真实本地 profiling 表或更复杂的 residual predictor 输入。
    """

    profile_name: str
    latency_s: float
    quality_base: float
    energy_j: float
    robustness_to_difficulty: float
    thermal_gain_scale: float = 1.0
    enabled: bool = True

    @property
    def estimated_power_w(self) -> float:
        """鍏兼彁渚涘巻鍙蹭唬鐮佷粛鍦ㄤ娇鐢ㄧ殑鍔熺巼鍙ｃ€?"""

        if self.latency_s <= 1.0e-9:
            return 0.0
        return self.energy_j / self.latency_s

    @property
    def robustness(self) -> float:
        """鍏兼彁渚涘巻鍙蹭唬鐮佷粛鍦ㄤ娇鐢ㄧ殑 robustness 鍙ｃ€?"""

        return self.robustness_to_difficulty


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
    task_value: float
    uncertainty: float
    remaining_utility: float
    is_metadata: bool
    is_high_value: bool
    deadline_s: float
    created_slot: int
    metadata: dict[str, Any] = field(default_factory=dict)
    payload_plan: str = ""
    product_type: str = ""
    residual_product_type: str = ""
    staged_parent_task_id: str = ""
    product_requirement: str = "verification"
    incremental_utility_estimate: float = 0.0
    incremental_information_utility: float = 0.0
    incremental_delivered_utility: float = 0.0
    duplicate_product_attempt: bool = False
    zero_increment_product: bool = False
    is_residual: bool = False
    duplicate_identity_key: str = ""
    continue_request_priority_bonus: float = 0.0
    followup_trigger: str = ""

    @property
    def whether_metadata(self) -> bool:
        return self.is_metadata

    @property
    def whether_high_value(self) -> bool:
        return self.is_high_value


@dataclass
class QueueItem:
    """下传队列中的内部对象。"""

    task_id: str
    generated_time_s: float
    deadline_time_s: float
    payload_size_mb: float
    remaining_mb: float
    remaining_utility: float
    task_value: float
    uncertainty: float
    payload_type: str
    waiting_time_s: float
    is_metadata: bool
    is_high_value: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    payload_plan: str = ""
    product_type: str = ""
    product_requirement: str = "verification"
    incremental_utility: float = 0.0
    first_alert_product: bool = False
    duplicate_product_attempt: bool = False
    zero_increment_product: bool = False
    is_residual: bool = False
    duplicate_identity_key: str = ""
    continue_request_priority_bonus: float = 0.0
    followup_trigger: str = ""

    @property
    def age_s(self) -> float:
        return self.waiting_time_s

    @property
    def whether_metadata(self) -> bool:
        return self.is_metadata

    @property
    def whether_high_value(self) -> bool:
        return self.is_high_value


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
    delivered_item_count: int = 0
    expired_item_count: int = 0
    dropped_item_count: int = 0
    metadata_delivered_count: int = 0
    high_value_metadata_delivered_count: int = 0
    delivered_high_value_payload_count: int = 0
    delivered_waiting_time_s: float = 0.0
    transmitted_task_ids: list[str] = field(default_factory=list)
    delivered_task_ids: list[str] = field(default_factory=list)
    expired_task_ids: list[str] = field(default_factory=list)
    dropped_task_ids: list[str] = field(default_factory=list)
    delivered_task_event_times_s: list[float] = field(default_factory=list)
    expired_task_event_times_s: list[float] = field(default_factory=list)
    dropped_task_event_times_s: list[float] = field(default_factory=list)
    delivered_payload_ids: list[str] = field(default_factory=list)
    delivered_product_types: list[str] = field(default_factory=list)
    expired_payload_ids: list[str] = field(default_factory=list)
    expired_product_types: list[str] = field(default_factory=list)
    dropped_payload_ids: list[str] = field(default_factory=list)
    dropped_product_types: list[str] = field(default_factory=list)
    delivered_incremental_utilities: list[float] = field(default_factory=list)
    duplicate_product_attempt_count: int = 0
    zero_increment_product_count: int = 0
    residual_enqueue_count: int = 0
    residual_delivery_count: int = 0
    residual_expired_count: int = 0
    residual_dropped_count: int = 0
    duplicate_residual_suppression_count: int = 0
    followup_payload_mb: float = 0.0
    residual_delivered_incremental_utilities: list[float] = field(default_factory=list)
    residual_queue_wait_times_s: list[float] = field(default_factory=list)
    delivered_followup_triggers: list[str] = field(default_factory=list)
    proactive_followup_enqueue_count: int = 0
    proactive_followup_delivery_count: int = 0
    proactive_followup_expired_count: int = 0
    proactive_followup_dropped_count: int = 0
    proactive_duplicate_suppression_count: int = 0
    proactive_followup_payload_mb: float = 0.0
    proactive_delivered_incremental_utilities: list[float] = field(default_factory=list)
    delivered_payload_metadata: list[dict[str, Any]] = field(default_factory=list)
    expired_payload_metadata: list[dict[str, Any]] = field(default_factory=list)
    dropped_payload_metadata: list[dict[str, Any]] = field(default_factory=list)


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
    raw_info_gain: float = 0.0
    delivery_proxy_raw: float = 1.0
    delivery_success_proxy: float = 1.0
    effective_info_gain: float = 0.0
    payload_delivery_risk_reason: str = "disabled"
    predicted_payload_plan: str = ""
    predicted_product_type: str = ""
    predicted_residual_product_type: str = ""
    predicted_incremental_information_utility: float = 0.0


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
    delivered_items: int = 0
    expired_items: int = 0
    dropped_items: int = 0
    metadata_delivered_items: int = 0
    high_value_metadata_delivered_items: int = 0
    delivered_high_value_payloads: int = 0
    average_waiting_time_s: float = 0.0
    miss_signal: float = 0.0
    false_alarm_signal: float = 0.0
    detection_quality: float = 0.0
    resource_safety_reasons: list[str] = field(default_factory=list)
    downlink_throttled: int = 0
    downlink_throttle_reason: str = "none"
    planned_downlink_mb_before_throttle: float = 0.0
    actual_downlink_mb_after_throttle: float = 0.0
    planned_comm_energy_j_before_throttle: float = 0.0
    actual_comm_energy_j_after_throttle: float = 0.0
    battery_next_soc: float = 0.0
    thermal_next: float = 0.0
    thermal_guard_exceeded: int = 0
    battery_guard_exceeded: int = 0
    thermal_guard_penalty: float = 0.0
    battery_reserve_penalty: float = 0.0
    downlink_burden_penalty: float = 0.0
    queue_pressure_risk_penalty: float = 0.0
    long_term_risk_penalty: float = 0.0
    net_system_reward_slot: float = 0.0
    transmitted_task_ids: str = ""
    delivered_task_ids: str = ""
    expired_task_ids: str = ""
    dropped_task_ids: str = ""
    estimated_target_count: int = 0
    estimated_bbox_area: float = 0.0
    estimated_roi_size_kb: float = 0.0
    estimated_roi_context_size_kb: float = 0.0
    payload_size_source: str = "unknown"
    payload_encoding_latency_s: float = 0.0
    payload_encoding_energy_j: float = 0.0
    delivered_task_event_times_s: str = ""
    expired_task_event_times_s: str = ""
    dropped_task_event_times_s: str = ""
    product_requirement: str = "verification"
    selected_payload_plan: str = ""
    selected_product_type: str = ""
    residual_product_type: str = ""
    delivered_products: str = ""
    pending_products: str = ""
    first_alert_delivered_time_s: float = -1.0
    residual_utility_remaining: float = 0.0
    delivered_only_incremental_utility: float = 0.0
    high_value_delivered_incremental_utility: float = 0.0
    alert_delivered_utility: float = 0.0
    verification_delivered_utility: float = 0.0
    context_evidence_delivered_utility: float = 0.0
    duplicate_product_attempt_count: int = 0
    zero_increment_product_count: int = 0
    delivered_payload_ids: str = ""
    delivered_product_types: str = ""
    expired_payload_ids: str = ""
    expired_product_types: str = ""
    dropped_payload_ids: str = ""
    dropped_product_types: str = ""
    virtual_continue_request_count: int = 0
    residual_enqueue_count: int = 0
    residual_delivery_count: int = 0
    residual_expired_count: int = 0
    residual_dropped_count: int = 0
    duplicate_residual_suppression_count: int = 0
    passive_followup_incremental_utility: float = 0.0
    residual_utility_generated: float = 0.0
    residual_utility_delivered: float = 0.0
    residual_utility_lost: float = 0.0
    time_from_metadata_to_followup_s: float = 0.0
    residual_queue_wait_s: float = 0.0
    followup_payload_mb: float = 0.0
    metadata_to_roi_success_count: int = 0
    metadata_to_roi_context_success_count: int = 0
    alert_to_verification_conversion_count: int = 0
    context_evidence_followup_count: int = 0
    proactive_followup_attempt_count: int = 0
    proactive_followup_enqueue_count: int = 0
    proactive_followup_delivery_count: int = 0
    proactive_followup_expired_count: int = 0
    proactive_followup_dropped_count: int = 0
    proactive_followup_incremental_utility: float = 0.0
    proactive_followup_payload_mb: float = 0.0
    proactive_before_passive_request_count: int = 0
    passive_request_suppressed_by_prior_proactive_count: int = 0
    proactive_duplicate_suppression_count: int = 0
    ERIV: float = 0.0
    ERIV_selected: float = 0.0
    ERIV_threshold: float = 0.0
    z_k_followup_allowed_count: int = 0
    z_k_followup_blocked_count: int = 0
    proactive_queue_pressure_block_count: int = 0
    proactive_energy_block_count: int = 0
    proactive_thermal_block_count: int = 0
    proactive_deadline_block_count: int = 0
    proactive_contact_block_count: int = 0
    residual_utility_lost_after_no_proactive: float = 0.0
    residual_utility_saved_by_proactive: float = 0.0
    high_value_proactive_followup_count: int = 0
    high_value_proactive_followup_utility: float = 0.0
    proactive_followup_reason_code: str = ""
    proactive_followup_blockers: str = ""
    z_k_guided_followup_permission: int = 0
    lifecycle_pre_delivered_products: str = ""
    lifecycle_pre_pending_products: str = ""
    lifecycle_pre_residual_utility_remaining: float = 0.0
    lifecycle_pre_task_expired_flag: int = 0
    lifecycle_pre_task_closed_flag: int = 0
    lifecycle_pre_metadata_delivered_flag: int = 0
    lifecycle_pre_pending_residual_exists_flag: int = 0
    lifecycle_pre_true_candidate_flag: int = 0
    lifecycle_post_delivered_products: str = ""
    lifecycle_post_pending_products: str = ""
    lifecycle_post_residual_utility_remaining: float = 0.0
    lifecycle_post_metadata_delivery_slot: int = -1
    lifecycle_post_pending_residual_registered_slot: int = -1
    lifecycle_post_true_candidate_flag: int = 0
    residual_candidate_pool_size: int = 0
    candidate_added_count: int = 0
    candidate_removed_count: int = 0
    candidate_re_evaluated_count: int = 0
    candidate_expired_count: int = 0
    candidate_closed_count: int = 0
    candidate_duplicate_suppressed_count: int = 0
    residual_candidate_pool_add_count: int = 0
    residual_candidate_pool_size_max: int = 0
    residual_candidate_re_evaluation_count: int = 0
    residual_candidate_re_evaluation_after_metadata_delivery_count: int = 0
    residual_candidate_true_count: int = 0
    residual_candidate_expired_count: int = 0
    residual_candidate_removed_count: int = 0
    residual_candidate_duplicate_suppression_count: int = 0
    residual_candidate_age_s: float = 0.0
    carryover_proactive_attempt_count: int = 0
    carryover_proactive_enqueue_count: int = 0
    carryover_proactive_delivery_count: int = 0
    carryover_proactive_incremental_utility: float = 0.0
    carryover_before_passive_request_count: int = 0
    carryover_passive_request_suppressed_count: int = 0
    carryover_eriv: float = 0.0
    carryover_eriv_selected: float = 0.0
    carryover_contact_block_count: int = 0
    carryover_thermal_block_count: int = 0
    carryover_queue_block_count: int = 0
    carryover_deadline_block_count: int = 0
    carryover_energy_block_count: int = 0
    same_slot_candidate_detected_count: int = 0
    same_slot_candidate_evaluated_count: int = 0
    same_slot_carryover_enqueue_count: int = 0
    same_slot_carryover_delivery_count: int = 0
    same_slot_carryover_duplicate_block_count: int = 0
    same_slot_carryover_contact_block_count: int = 0
    same_slot_opportunity_saved_from_tplus1_count: int = 0
    same_slot_fallback_to_ordinary_queue_count: int = 0
    ordinary_queue_fallback_attempt_count: int = 0
    ordinary_queue_fallback_success_count: int = 0
    duplicate_residual_suppressed_after_fallback_count: int = 0
    zero_size_residual_payload_blocked_count: int = 0
    residual_fallback_blocked_by_expired_deadline_count: int = 0
    residual_fallback_dropped_by_queue_overflow_count: int = 0
    staged_feasibility_filter_enabled: int = 0
    staged_candidate_evaluated_count: int = 0
    staged_candidate_feasible_count: int = 0
    staged_candidate_infeasible_count: int = 0
    staged_candidate_blocked_by_contact_count: int = 0
    staged_candidate_blocked_by_deadline_count: int = 0
    staged_candidate_blocked_by_queue_count: int = 0
    staged_candidate_blocked_by_energy_count: int = 0
    staged_candidate_blocked_by_thermal_count: int = 0
    staged_candidate_demoted_count: int = 0
    staged_replaced_by_direct_roi_count: int = 0
    staged_replaced_by_metadata_only_count: int = 0
    staged_replaced_by_defer_count: int = 0
    metadata_only_after_filter_count: int = 0
    residual_completion_after_filter_count: int = 0
    passive_would_miss_but_proactive_meets_count: int = 0
    proactive_and_passive_both_miss_count: int = 0


@dataclass
class UtilityBreakdown:
    """Reserved-friendly value-cost utility decomposition."""

    value_raw: float
    value_calibrated: float
    mission_gain: float
    resource_cost: float
    requirement_penalty: float
    hard_penalty: float
    net_utility: float
    alert_utility_reserved: float | None = None
    confirmation_utility_reserved: float | None = None
    evidence_utility_reserved: float | None = None
    lifecycle_utility_reserved: float | None = None


@dataclass
class RewardBreakdown:
    """任务效用 reward 的可解释分解。"""

    effective_utility: float
    resource_penalty: float
    quality_requirement: float
    information_requirement: float
    quality_gap: float
    information_gap: float
    quality_gap_penalty: float
    information_gap_penalty: float
    requirement_penalty: float
    mission_utility: float
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
    gross_incremental_utility: float = 0.0
    net_incremental_utility: float = 0.0
    value_raw: float = 0.0
    value_calibrated: float = 0.0
    resource_cost: float = 0.0
    deadline_penalty: float = 0.0
    hard_penalty: float = 0.0
    net_utility: float = 0.0
    beta_resource_cost: float = 1.0
    beta_requirement_penalty: float = 1.0
    beta_hard_penalty: float = 1.0
    alert_utility_reserved: float | None = None
    confirmation_utility_reserved: float | None = None
    evidence_utility_reserved: float | None = None
    lifecycle_utility_reserved: float | None = None
    lifecycle_fields_active: bool = False


@dataclass(frozen=True)
class ShieldDecision:
    """缁熶竴瀹夊叏 shield 鍐崇瓥缁撴灉銆?"""

    raw_action: JointAction
    executed_action: JointAction
    safety_corrected: int
    safety_violation_before_correction: int
    safety_violation_after_correction: int
    reasons_before_correction: list[str] = field(default_factory=list)
    reasons_after_correction: list[str] = field(default_factory=list)


@dataclass
class TrajectoryRecord:
    """输出到轨迹 CSV 的单步记录。"""

    router_name: str
    slot: int
    t: int
    task_id: str
    has_task: int
    has_new_task: int
    scene_id: str
    observation_window_id: str
    task_class: str
    active_observation_window: int
    raw_action: str
    executed_action: str
    safety_corrected: int
    safety_violation_before_correction: int
    raw_safety_violation: int
    safety_violation_after_correction: int
    resource_safety_violation: int
    final_resource_safety_violation: int
    task_deferred: int
    deadline_missed: int
    service_failure_reason: str
    safety_violation_reason_before: str
    safety_violation_reason_after: str
    selected_action: str
    model_name: str
    payload_policy: str
    selected_model: str
    selected_payload: str
    orbit_index: int
    day_index: int
    in_sunlight: int
    is_sunlit: int
    is_eclipse: int
    solar_power_w: float
    solar_energy_j: float
    wasted_solar_energy_j: float
    idle_energy_j: float
    base_power_w: float
    dod: float
    dod_guard_violation: int
    temperature_reference_c: float
    battery_soc: float
    B_t: float
    E_AI_t: float
    Theta_t: float
    V_t: float
    deadline_s: float
    D_t: float
    H_t: float
    U_t: float
    C_t: float
    Q_t: float
    lambda_E_base: float
    lambda_L_base: float
    lambda_Q_base: float
    lambda_S_base: float
    mode_name: str
    lambda_E: float
    lambda_L: float
    lambda_Q: float
    lambda_S: float
    exploration_scale: float
    model_complexity_bias: float
    payload_size_bias: float
    high_value_bonus_scale: float
    safety_margin_scale: float
    guidance_bundle_summary: str
    selected_action_score: float
    predicted_reward: float
    effective_utility: float
    mission_utility: float
    resource_penalty: float
    penalized_reward: float
    observed_reward: float
    detection_quality: float
    payload_information_content: float
    quality_requirement: float
    information_requirement: float
    Q_req: float
    I_req: float
    predicted_quality: float
    information_factor: float
    quality_gap: float
    information_gap: float
    quality_gap_penalty: float
    information_gap_penalty: float
    requirement_penalty: float
    safety_penalty: float
    mission_error_penalty: float
    S_gen_current_mb: float
    S_down_total_mb: float
    S_down_current_task_mb: float
    reference_max_payload_mb: float
    queue_mb: float
    queue_next_mb: float
    delta_queue_mb: float
    compute_energy_j: float
    comm_energy_j: float
    E_comp: float
    E_comm: float
    L_inf: float
    L_queue: float
    L_comm: float
    queue_item_count: int
    delivered_items: int
    expired_items: int
    dropped_items: int
    metadata_delivered_items: int
    high_value_metadata_delivered_items: int
    delivered_high_value_payloads: int
    average_waiting_time_s: float
    delivery_status: str
    downlink_throttled: int
    downlink_throttle_reason: str
    planned_downlink_mb_before_throttle: float
    actual_downlink_mb_after_throttle: float
    planned_comm_energy_j_before_throttle: float
    actual_comm_energy_j_after_throttle: float
    battery_next_soc: float
    thermal_next: float
    thermal_guard_exceeded: int
    battery_guard_exceeded: int
    thermal_guard_penalty: float
    battery_reserve_penalty: float
    downlink_burden_penalty: float
    queue_pressure_risk_penalty: float
    long_term_risk_penalty: float
    net_system_reward_slot: float
    safety_violation: int
    raw_ucb_bonus: float = 0.0
    risk_discount: float = 1.0
    cost_aware_ucb_bonus: float = 0.0
    ucb_suppression_reason: str = "disabled"
    raw_info_gain: float = 0.0
    delivery_proxy_raw: float = 1.0
    delivery_proxy_after_floor: float = 1.0
    delivery_success_proxy: float = 1.0
    effective_info_gain: float = 0.0
    payload_delivery_risk_reason: str = "disabled"
    estimated_target_count: int = 0
    estimated_bbox_area: float = 0.0
    estimated_roi_size_kb: float = 0.0
    estimated_roi_context_size_kb: float = 0.0
    payload_size_source: str = "unknown"
    payload_encoding_latency_s: float = 0.0
    payload_encoding_energy_j: float = 0.0
    lambda_E_before_z: float = 0.0
    lambda_Q_before_z: float = 0.0
    lambda_S_before_z: float = 0.0
    lambda_L_before_z: float = 0.0
    lambda_E_after_z: float = 0.0
    lambda_Q_after_z: float = 0.0
    lambda_S_after_z: float = 0.0
    lambda_L_after_z: float = 0.0
    beta_before_z: float = 0.0
    beta_after_z: float = 0.0
    z_guidance_reason: str = "disabled"
    rank_before_z: int = 0
    rank_after_z: int = 0
    transmitted_task_ids: str = ""
    delivered_task_ids: str = ""
    expired_task_ids: str = ""
    dropped_task_ids: str = ""
    task_arrival_time_s: float = -1.0
    payload_generation_time_s: float = -1.0
    payload_delivery_time_s: float = -1.0
    deadline_time_s: float = -1.0
    lifetime_status: str = "current_step_only"
    high_value_flag: int = 0
    timeliness_at_decision: float = 0.0
    timeliness_at_delivery: float = 0.0
    effective_utility_provisional: float = 0.0
    delivered_only_utility: float = 0.0
    delivered_before_deadline_utility: float = 0.0
    queued_provisional_utility: float = 0.0
    delivered_late_flag: int = 0
    expired_flag: int = 0
    deferred_flag: int = 0
    delivered_task_event_times_s: str = ""
    expired_task_event_times_s: str = ""
    dropped_task_event_times_s: str = ""
    product_requirement: str = "verification"
    selected_payload_plan: str = ""
    selected_product_type: str = ""
    residual_product_type: str = ""
    delivered_products: str = ""
    pending_products: str = ""
    first_alert_delivered_time_s: float = -1.0
    residual_utility_remaining: float = 0.0
    delivered_only_incremental_utility: float = 0.0
    high_value_delivered_incremental_utility: float = 0.0
    alert_delivered_utility: float = 0.0
    verification_delivered_utility: float = 0.0
    context_evidence_delivered_utility: float = 0.0
    duplicate_product_attempt_count: int = 0
    zero_increment_product_count: int = 0
    virtual_continue_request_count: int = 0
    residual_enqueue_count: int = 0
    residual_delivery_count: int = 0
    residual_expired_count: int = 0
    residual_dropped_count: int = 0
    duplicate_residual_suppression_count: int = 0
    passive_followup_incremental_utility: float = 0.0
    residual_utility_generated: float = 0.0
    residual_utility_delivered: float = 0.0
    residual_utility_lost: float = 0.0
    time_from_metadata_to_followup_s: float = 0.0
    residual_queue_wait_s: float = 0.0
    followup_payload_mb: float = 0.0
    metadata_to_roi_success_count: int = 0
    metadata_to_roi_context_success_count: int = 0
    alert_to_verification_conversion_count: int = 0
    context_evidence_followup_count: int = 0
    proactive_followup_attempt_count: int = 0
    proactive_followup_enqueue_count: int = 0
    proactive_followup_delivery_count: int = 0
    proactive_followup_expired_count: int = 0
    proactive_followup_dropped_count: int = 0
    proactive_followup_incremental_utility: float = 0.0
    proactive_followup_payload_mb: float = 0.0
    proactive_before_passive_request_count: int = 0
    passive_request_suppressed_by_prior_proactive_count: int = 0
    proactive_duplicate_suppression_count: int = 0
    ERIV: float = 0.0
    ERIV_selected: float = 0.0
    ERIV_threshold: float = 0.0
    z_k_followup_allowed_count: int = 0
    z_k_followup_blocked_count: int = 0
    proactive_queue_pressure_block_count: int = 0
    proactive_energy_block_count: int = 0
    proactive_thermal_block_count: int = 0
    proactive_deadline_block_count: int = 0
    proactive_contact_block_count: int = 0
    residual_utility_lost_after_no_proactive: float = 0.0
    residual_utility_saved_by_proactive: float = 0.0
    high_value_proactive_followup_count: int = 0
    high_value_proactive_followup_utility: float = 0.0
    proactive_followup_reason_code: str = ""
    proactive_followup_blockers: str = ""
    z_k_guided_followup_permission: int = 0
    lifecycle_pre_delivered_products: str = ""
    lifecycle_pre_pending_products: str = ""
    lifecycle_pre_residual_utility_remaining: float = 0.0
    lifecycle_pre_task_expired_flag: int = 0
    lifecycle_pre_task_closed_flag: int = 0
    lifecycle_pre_metadata_delivered_flag: int = 0
    lifecycle_pre_pending_residual_exists_flag: int = 0
    lifecycle_pre_true_candidate_flag: int = 0
    lifecycle_post_delivered_products: str = ""
    lifecycle_post_pending_products: str = ""
    lifecycle_post_residual_utility_remaining: float = 0.0
    lifecycle_post_metadata_delivery_slot: int = -1
    lifecycle_post_pending_residual_registered_slot: int = -1
    lifecycle_post_true_candidate_flag: int = 0
    residual_candidate_pool_size: int = 0
    candidate_added_count: int = 0
    candidate_removed_count: int = 0
    candidate_re_evaluated_count: int = 0
    candidate_expired_count: int = 0
    candidate_closed_count: int = 0
    candidate_duplicate_suppressed_count: int = 0
    residual_candidate_pool_add_count: int = 0
    residual_candidate_pool_size_max: int = 0
    residual_candidate_re_evaluation_count: int = 0
    residual_candidate_re_evaluation_after_metadata_delivery_count: int = 0
    residual_candidate_true_count: int = 0
    residual_candidate_expired_count: int = 0
    residual_candidate_removed_count: int = 0
    residual_candidate_duplicate_suppression_count: int = 0
    residual_candidate_age_s: float = 0.0
    carryover_proactive_attempt_count: int = 0
    carryover_proactive_enqueue_count: int = 0
    carryover_proactive_delivery_count: int = 0
    carryover_proactive_incremental_utility: float = 0.0
    carryover_before_passive_request_count: int = 0
    carryover_passive_request_suppressed_count: int = 0
    carryover_eriv: float = 0.0
    carryover_eriv_selected: float = 0.0
    carryover_contact_block_count: int = 0
    carryover_thermal_block_count: int = 0
    carryover_queue_block_count: int = 0
    carryover_deadline_block_count: int = 0
    carryover_energy_block_count: int = 0
    same_slot_candidate_detected_count: int = 0
    same_slot_candidate_evaluated_count: int = 0
    same_slot_carryover_enqueue_count: int = 0
    same_slot_carryover_delivery_count: int = 0
    same_slot_carryover_duplicate_block_count: int = 0
    same_slot_carryover_contact_block_count: int = 0
    same_slot_opportunity_saved_from_tplus1_count: int = 0
    same_slot_fallback_to_ordinary_queue_count: int = 0
    ordinary_queue_fallback_attempt_count: int = 0
    ordinary_queue_fallback_success_count: int = 0
    duplicate_residual_suppressed_after_fallback_count: int = 0
    zero_size_residual_payload_blocked_count: int = 0
    residual_fallback_blocked_by_expired_deadline_count: int = 0
    residual_fallback_dropped_by_queue_overflow_count: int = 0
    staged_feasibility_filter_enabled: int = 0
    staged_candidate_evaluated_count: int = 0
    staged_candidate_feasible_count: int = 0
    staged_candidate_infeasible_count: int = 0
    staged_candidate_blocked_by_contact_count: int = 0
    staged_candidate_blocked_by_deadline_count: int = 0
    staged_candidate_blocked_by_queue_count: int = 0
    staged_candidate_blocked_by_energy_count: int = 0
    staged_candidate_blocked_by_thermal_count: int = 0
    staged_candidate_demoted_count: int = 0
    staged_replaced_by_direct_roi_count: int = 0
    staged_replaced_by_metadata_only_count: int = 0
    staged_replaced_by_defer_count: int = 0
    metadata_only_after_filter_count: int = 0
    residual_completion_after_filter_count: int = 0
    passive_would_miss_but_proactive_meets_count: int = 0
    proactive_and_passive_both_miss_count: int = 0
    product_requirement_distribution: str = ""
    delivered_product_set_distribution: str = ""
    V_raw: float = 0.0
    V_cal: float = 0.0
    base_prior_score: float = 0.0
    value_level: str = ""
    value_multiplier: float = 1.0
    value_skew_profile: str = "legacy"
    normalization_alpha: float = 1.0
    calibrated_task_value: float = 0.0
    value_assignment_reason: str = ""
    mission_gain: float = 0.0
    resource_cost: float = 0.0
    hard_penalty: float = 0.0
    net_utility: float = 0.0
    beta_resource_cost: float = 1.0
    beta_requirement_penalty: float = 1.0
    beta_hard_penalty: float = 1.0
    lifecycle_fields_active: bool = False
    alert_utility_reserved: float | None = None
    confirmation_utility_reserved: float | None = None
    evidence_utility_reserved: float | None = None
    lifecycle_utility_reserved: float | None = None


@dataclass
class EpisodeResult:
    """单个 router 的完整回合结果。"""

    router_name: str
    records: list[TrajectoryRecord]
    summary: dict[str, float | int | str]
