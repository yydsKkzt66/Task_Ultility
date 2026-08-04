"""Mock dataset adapter."""

from __future__ import annotations

import math
import random

from src.common.types import TaskSample
from src.data.base_dataset import DatasetAdapter
from src.data.difficulty import sample_deadline_s, sample_difficulty, sample_task_value, sample_uncertainty
from src.data.mission_prior_value import (
    compute_mission_prior_value_v2,
    mission_prior_value_v2_enabled,
    task_value_component_metadata,
)
from src.data.product_requirements import assign_product_requirement


class MockDatasetAdapter(DatasetAdapter):
    """Generate synthetic slot-aligned task streams for the closed-loop case study."""

    def __init__(self, dataset_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg
        self.seed = int(dataset_cfg.get("seed", 2026))

    def build_task_stream(
        self,
        num_slots: int,
        slot_seconds: float,
        task_stream_cfg: dict[str, object] | None = None,
    ) -> list[TaskSample | None]:
        task_stream_cfg = task_stream_cfg or {}
        mode = str(task_stream_cfg.get("mode", "one_task_per_slot")).lower()
        if mode == "one_task_per_slot":
            return self._build_one_task_per_slot(num_slots, task_stream_cfg)
        if mode == "observation_window":
            return self._build_observation_window_stream(num_slots, slot_seconds, task_stream_cfg)
        raise ValueError(f"Unsupported task stream mode: {mode}")

    def _build_one_task_per_slot(self, num_slots: int, task_stream_cfg: dict[str, object]) -> list[TaskSample]:
        rng = random.Random(self.seed)
        tasks: list[TaskSample] = []
        for slot_index in range(num_slots):
            task_id = f"task_{slot_index:04d}"
            task_value = sample_task_value(slot_index, rng)
            old_task_value = task_value
            deadline_s = sample_deadline_s(slot_index, rng)
            difficulty = sample_difficulty(slot_index, rng)
            uncertainty = sample_uncertainty(slot_index, rng)
            product_requirement, product_requirement_source = assign_product_requirement(
                task_value=task_value,
                deadline_s=deadline_s,
                uncertainty=uncertainty,
            )
            metadata = {
                "dataset_name": "mock",
                "synthetic_target_density": 1 + (slot_index % 3),
            }
            scene_id = f"scene_{slot_index // 12:03d}"
            if mission_prior_value_v2_enabled(task_stream_cfg):
                value_result = compute_mission_prior_value_v2(
                    task_id=task_id,
                    scene_id=scene_id,
                    task_class="routine",
                    product_requirement=product_requirement,
                    deadline_s=deadline_s,
                    container_cfg=task_stream_cfg,
                )
                task_value = value_result.task_value
                metadata.update(task_value_component_metadata(value_result, old_task_value=old_task_value))
            tasks.append(
                TaskSample(
                    task_id=task_id,
                    image_path=f"mock://{scene_id}/tile_{slot_index:04d}.png",
                    label_path=f"mock://{scene_id}/tile_{slot_index:04d}.txt",
                    scene_id=scene_id,
                    virtual_time=slot_index,
                    task_value=task_value,
                    deadline_s=deadline_s,
                    difficulty=difficulty,
                    uncertainty=uncertainty,
                    metadata=metadata,
                    arrival_slot=slot_index,
                    observation_window_id=f"window_{slot_index // 12:03d}",
                    task_class="routine",
                    quality_requirement=0.60,
                    information_requirement=0.55,
                    high_value_flag=False,
                    emergency_flag=False,
                    product_requirement=product_requirement,
                    product_requirement_source=product_requirement_source,
                )
            )
        return tasks

    def _sample_window_class(self, rng: random.Random, cfg: dict[str, object]) -> str:
        emergency_p = float(cfg.get("emergency_window_probability", 0.10))
        high_value_p = float(cfg.get("high_value_window_probability", 0.25))
        draw = rng.random()
        if draw < emergency_p:
            return "emergency"
        if draw < emergency_p + high_value_p:
            return "high_value"
        return "routine"

    def _sample_priority_window_class(self, rng: random.Random, cfg: dict[str, object]) -> str:
        emergency_p = float(cfg.get("emergency_window_probability", 0.10))
        high_value_p = float(cfg.get("high_value_window_probability", 0.25))
        total = max(emergency_p + high_value_p, 1.0e-9)
        return "emergency" if rng.random() < emergency_p / total else "high_value"

    @staticmethod
    def _class_defaults(task_class: str) -> dict[str, float | bool]:
        if task_class == "emergency":
            return {
                "task_value": 0.92,
                "deadline_s": 35.0,
                "difficulty": 0.72,
                "uncertainty": 0.58,
                "quality_requirement": 0.88,
                "information_requirement": 0.90,
                "high_value_flag": True,
                "emergency_flag": True,
            }
        if task_class == "high_value":
            return {
                "task_value": 0.80,
                "deadline_s": 70.0,
                "difficulty": 0.62,
                "uncertainty": 0.48,
                "quality_requirement": 0.78,
                "information_requirement": 0.76,
                "high_value_flag": True,
                "emergency_flag": False,
            }
        return {
            "task_value": 0.52,
            "deadline_s": 95.0,
            "difficulty": 0.45,
            "uncertainty": 0.30,
            "quality_requirement": 0.58,
            "information_requirement": 0.50,
            "high_value_flag": False,
            "emergency_flag": False,
        }

    def _make_window_task(
        self,
        slot_index: int,
        task_index: int,
        scene_id: str,
        window_id: str,
        task_class: str,
        defaults: dict[str, float | bool],
        rng: random.Random,
        task_stream_cfg: dict[str, object],
    ) -> TaskSample:
        task_value = max(0.10, min(1.0, float(defaults["task_value"]) + rng.uniform(-0.06, 0.06)))
        old_task_value = task_value
        deadline_s = max(10.0, float(defaults["deadline_s"]) + rng.uniform(-8.0, 8.0))
        difficulty = max(0.05, min(1.0, float(defaults["difficulty"]) + rng.uniform(-0.08, 0.08)))
        uncertainty = max(0.05, min(1.0, float(defaults["uncertainty"]) + rng.uniform(-0.08, 0.08)))
        quality_requirement = max(0.0, min(1.0, float(defaults["quality_requirement"]) + rng.uniform(-0.03, 0.03)))
        information_requirement = max(0.0, min(1.0, float(defaults["information_requirement"]) + rng.uniform(-0.03, 0.03)))
        product_requirement, product_requirement_source = assign_product_requirement(
            task_value=task_value,
            deadline_s=deadline_s,
            uncertainty=uncertainty,
            high_value_flag=bool(defaults["high_value_flag"]),
            emergency_flag=bool(defaults["emergency_flag"]),
            task_class=task_class,
        )
        metadata = {
            "dataset_name": "mock",
            "synthetic_target_density": 2 + (task_index % 4),
            "active_observation_window": True,
        }
        if mission_prior_value_v2_enabled(task_stream_cfg):
            value_result = compute_mission_prior_value_v2(
                task_id=f"task_{slot_index:04d}",
                scene_id=scene_id,
                task_class=task_class,
                product_requirement=product_requirement,
                deadline_s=deadline_s,
                container_cfg=task_stream_cfg,
            )
            task_value = value_result.task_value
            metadata.update(task_value_component_metadata(value_result, old_task_value=old_task_value))
        return TaskSample(
            task_id=f"task_{slot_index:04d}",
            image_path=f"mock://{scene_id}/tile_{task_index:04d}.png",
            label_path=f"mock://{scene_id}/tile_{task_index:04d}.txt",
            scene_id=scene_id,
            virtual_time=slot_index,
            task_value=task_value,
            deadline_s=deadline_s,
            difficulty=difficulty,
            uncertainty=uncertainty,
            metadata=metadata,
            arrival_slot=slot_index,
            observation_window_id=window_id,
            task_class=task_class,
            quality_requirement=quality_requirement,
            information_requirement=information_requirement,
            high_value_flag=bool(defaults["high_value_flag"]),
            emergency_flag=bool(defaults["emergency_flag"]),
            product_requirement=product_requirement,
            product_requirement_source=product_requirement_source,
        )

    def _make_background_task(self, slot_index: int, rng: random.Random, task_stream_cfg: dict[str, object]) -> TaskSample:
        base_scene = f"background_{slot_index // 24:03d}"
        task_value = max(0.10, min(0.45, 0.18 + rng.uniform(-0.05, 0.10)))
        old_task_value = task_value
        deadline_s = max(25.0, 110.0 + rng.uniform(-12.0, 12.0))
        difficulty = max(0.05, min(0.55, 0.22 + rng.uniform(-0.08, 0.08)))
        uncertainty = max(0.05, min(0.45, 0.18 + rng.uniform(-0.06, 0.06)))
        product_requirement, product_requirement_source = assign_product_requirement(
            task_value=task_value,
            deadline_s=deadline_s,
            uncertainty=uncertainty,
        )
        metadata = {
            "dataset_name": "mock",
            "synthetic_target_density": 1,
            "active_observation_window": False,
            "background_task": True,
        }
        if mission_prior_value_v2_enabled(task_stream_cfg):
            value_result = compute_mission_prior_value_v2(
                task_id=f"task_{slot_index:04d}",
                scene_id=base_scene,
                task_class="routine",
                product_requirement=product_requirement,
                deadline_s=deadline_s,
                container_cfg=task_stream_cfg,
            )
            task_value = value_result.task_value
            metadata.update(task_value_component_metadata(value_result, old_task_value=old_task_value))
        return TaskSample(
            task_id=f"task_{slot_index:04d}",
            image_path=f"mock://{base_scene}/tile_{slot_index:04d}.png",
            label_path=f"mock://{base_scene}/tile_{slot_index:04d}.txt",
            scene_id=base_scene,
            virtual_time=slot_index,
            task_value=task_value,
            deadline_s=deadline_s,
            difficulty=difficulty,
            uncertainty=uncertainty,
            metadata=metadata,
            arrival_slot=slot_index,
            observation_window_id="",
            task_class="routine",
            quality_requirement=0.48,
            information_requirement=0.35,
            high_value_flag=False,
            emergency_flag=False,
            product_requirement=product_requirement,
            product_requirement_source=product_requirement_source,
        )

    def _build_observation_window_stream(
        self,
        num_slots: int,
        slot_seconds: float,
        cfg: dict[str, object],
    ) -> list[TaskSample | None]:
        rng = random.Random(self.seed)
        stream: list[TaskSample | None] = [None for _ in range(num_slots)]
        orbit_period_slots = max(6, int(round(float(cfg.get("orbit_period_min", 90)) * 60.0 / max(slot_seconds, 1.0e-9))))
        window_count_range = cfg.get("observation_windows_per_orbit", [1, 3])
        duration_range = cfg.get("window_duration_s_range", [60, 300])
        interval_range = cfg.get("intra_window_task_interval_slots", [1, 3])
        background_task_probability = float(cfg.get("background_task_probability", 0.02))

        global_window_id = 0
        task_counter = 0
        num_orbits = max(1, math.ceil(num_slots / orbit_period_slots))

        for orbit_index in range(num_orbits):
            orbit_start = orbit_index * orbit_period_slots
            orbit_end = min(num_slots, orbit_start + orbit_period_slots)
            available_slots = orbit_end - orbit_start
            if available_slots <= 0:
                continue

            min_windows, max_windows = int(window_count_range[0]), int(window_count_range[1])
            window_count = rng.randint(min_windows, max(max_windows, min_windows))
            occupancy = [False] * available_slots

            for local_window_index in range(window_count):
                if available_slots <= 2:
                    break
                duration_slots = max(
                    1,
                    int(round(rng.uniform(float(duration_range[0]), float(duration_range[1])) / max(slot_seconds, 1.0e-9))),
                )
                duration_slots = min(duration_slots, max(1, available_slots - 2))
                interval_slots = rng.randint(int(interval_range[0]), int(interval_range[1]))
                latest_start = max(0, available_slots - duration_slots)
                candidate_starts = [
                    start
                    for start in range(latest_start + 1)
                    if not any(occupancy[idx] for idx in range(start, min(available_slots, start + duration_slots)))
                ]
                if not candidate_starts:
                    continue
                start = rng.choice(candidate_starts)
                end = min(available_slots, start + duration_slots)

                if local_window_index == 0 and (
                    float(cfg.get("emergency_window_probability", 0.10)) + float(cfg.get("high_value_window_probability", 0.25))
                ) > 0.0:
                    task_class = self._sample_priority_window_class(rng, cfg)
                else:
                    task_class = self._sample_window_class(rng, cfg)
                defaults = self._class_defaults(task_class)
                window_id = f"window_{global_window_id:04d}"
                scene_id = f"scene_orbit{orbit_index:03d}_{task_class}_{local_window_index:02d}"
                global_window_id += 1

                for local_slot in range(start, end, interval_slots):
                    absolute_slot = orbit_start + local_slot
                    if absolute_slot >= num_slots:
                        break
                    occupancy[local_slot] = True
                    stream[absolute_slot] = self._make_window_task(
                        slot_index=absolute_slot,
                        task_index=task_counter,
                        scene_id=scene_id,
                        window_id=window_id,
                        task_class=task_class,
                        defaults=defaults,
                        rng=rng,
                        task_stream_cfg=cfg,
                    )
                    task_counter += 1

            # sparse background tasks outside active observation windows
            for local_slot in range(available_slots):
                absolute_slot = orbit_start + local_slot
                if absolute_slot >= num_slots or stream[absolute_slot] is not None:
                    continue
                if rng.random() < background_task_probability:
                    stream[absolute_slot] = self._make_background_task(absolute_slot, rng, cfg)
                    task_counter += 1

        # If the horizon is very short and everything became active, keep at least one idle slot.
        if num_slots > 2 and all(item is not None for item in stream):
            stream[-1] = None

        return stream
