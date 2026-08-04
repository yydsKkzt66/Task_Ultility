"""Task stream builder entrypoint."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.common.types import TaskSample
from src.data.value_skew_calibration import apply_value_skew_calibration
from src.data.mock_dataset import MockDatasetAdapter
from src.data.product_requirements import assign_product_requirement_v2_1, recommend_payload_plan_from_requirement
from src.data.sarfish_adapter import SarfishAdapter
from src.data.sarfishsample_dataset import SARFishSampleDatasetAdapter


def _optional_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class TaskStreamBuilder:
    """Choose a dataset adapter and build a unified slot-aligned task stream."""

    def __init__(self, dataset_cfg: dict[str, object], task_stream_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg
        self.task_stream_cfg = task_stream_cfg

    def build(self, num_slots: int, slot_seconds: float):
        dataset_name = str(self.dataset_cfg.get("dataset_name", "mock")).lower()
        if dataset_name == "mock":
            stream = MockDatasetAdapter(self.dataset_cfg).build_task_stream(num_slots, slot_seconds, self.task_stream_cfg)
            return self._apply_value_skew(self._apply_product_requirement_rule(stream))
        if dataset_name in {"sarfishsample", "sarfish_sample"}:
            stream = SARFishSampleDatasetAdapter(self.dataset_cfg).build_task_stream(
                num_slots, slot_seconds, self.task_stream_cfg
            )
            return self._apply_value_skew(self._apply_product_requirement_rule(stream))
        if dataset_name == "sarfish":
            stream = SarfishAdapter(self.dataset_cfg).build_task_stream(num_slots, slot_seconds, self.task_stream_cfg)
            return self._apply_value_skew(self._apply_product_requirement_rule(stream))
        raise ValueError(f"Unsupported dataset type: {dataset_name}")

    def _apply_value_skew(self, stream: list[TaskSample | None]) -> list[TaskSample | None]:
        return apply_value_skew_calibration(stream, self.task_stream_cfg)

    def _apply_product_requirement_rule(self, stream: list[TaskSample | None]) -> list[TaskSample | None]:
        rule_cfg = self.task_stream_cfg.get("product_requirement_rule", {})
        if not isinstance(rule_cfg, dict):
            return stream
        mode = str(rule_cfg.get("mode", "legacy")).strip().lower()
        if mode in {"", "legacy", "v1"}:
            return stream
        if mode not in {"v2_1", "v21"}:
            raise ValueError(f"Unsupported product_requirement_rule.mode: {mode}")
        return [self._apply_v21_to_task(task) if task is not None else None for task in stream]

    def _apply_v21_to_task(self, task: TaskSample) -> TaskSample:
        metadata: dict[str, Any] = dict(task.metadata)
        scene_context_need = _optional_float(metadata.get("scene_context_need"))
        if scene_context_need is None:
            scene_context_need = _optional_float(metadata.get("g_area_prior"))
        product_use_case = metadata.get("product_use_case", metadata.get("mission_use_case"))
        timeliness_class = metadata.get("timeliness_class", metadata.get("deadline_policy_source"))
        result = assign_product_requirement_v2_1(
            task_value=task.task_value,
            uncertainty=task.uncertainty,
            emergency_flag=task.emergency_flag,
            high_value_flag=task.high_value_flag,
            task_class=task.task_class,
            mission_source=str(metadata.get("dataset_name", "")) if metadata.get("dataset_name") else None,
            scene_context_need=scene_context_need,
            product_use_case=str(product_use_case) if product_use_case else None,
            timeliness_class=str(timeliness_class) if timeliness_class else None,
        )
        recommendation = recommend_payload_plan_from_requirement(
            ideal_product_requirement=result.ideal_product_requirement,
            metadata_feasible=None,
            roi_feasible=None,
            roi_context_feasible=None,
            metadata_then_roi_feasible=None,
            metadata_then_roi_context_feasible=None,
        )
        metadata.update(
            {
                "product_requirement_rule_mode": "v2_1",
                "legacy_product_requirement": task.product_requirement,
                "v21_ideal_product_requirement": result.ideal_product_requirement,
                "v21_requirement_scores": result.requirement_scores,
                "v21_initial_alert_required": result.initial_alert_required,
                "v21_allowed_requirements": list(result.allowed_requirements),
                "v21_requirement_source": result.requirement_source,
                "v21_diagnostic_reason": result.diagnostic_reason,
                "v21_missing_fields": list(result.missing_fields),
                "recommended_payload_plan_shadow": recommendation.recommended_payload_plan,
                "recommended_payload_plan_shadow_source": recommendation.recommendation_source,
                "recommended_payload_plan_shadow_reason": recommendation.diagnostic_reason,
            }
        )
        return replace(
            task,
            metadata=metadata,
            product_requirement=result.ideal_product_requirement,
            product_requirement_source=result.requirement_source,
        )
