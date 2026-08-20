"""SARFishSample dataset adapter."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from src.common.types import TaskSample
from src.data.base_dataset import DatasetAdapter
from src.data.mission_prior_value import (
    compute_mission_prior_value_v2,
    mission_prior_value_v2_enabled,
    task_value_component_metadata,
)
from src.data.product_requirements import assign_product_requirement


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SARFishSampleDatasetAdapter(DatasetAdapter):
    """Build slot-aligned TaskSample streams from local SARFishSample GRD scenes."""

    TASK_CLASS_RANGES: dict[str, dict[str, tuple[float, float] | bool]] = {
        "routine": {
            "task_value": (0.50, 0.60),
            "deadline_s": (90.0, 120.0),
            "quality_requirement": (0.55, 0.65),
            "information_requirement": (0.45, 0.60),
            "high_value_flag": False,
            "emergency_flag": False,
        },
        "high_value": {
            "task_value": (0.75, 0.85),
            "deadline_s": (60.0, 90.0),
            "quality_requirement": (0.75, 0.82),
            "information_requirement": (0.72, 0.80),
            "high_value_flag": True,
            "emergency_flag": False,
        },
        "emergency": {
            "task_value": (0.88, 0.98),
            "deadline_s": (30.0, 60.0),
            "quality_requirement": (0.85, 0.92),
            "information_requirement": (0.85, 0.93),
            "high_value_flag": True,
            "emergency_flag": True,
        },
    }

    def __init__(self, dataset_cfg: dict[str, object]) -> None:
        self.dataset_cfg = dataset_cfg
        self.seed = int(dataset_cfg.get("seed", 2026))
        self.product_type = str(dataset_cfg.get("product_type", "GRD")).upper()
        self.split = str(dataset_cfg.get("split", "validation")).lower()
        configured_root = str(dataset_cfg.get("source_image_root", "") or "")
        env_name = str(dataset_cfg.get("source_image_root_env", "SARFISH_DATASET_ROOT") or "SARFISH_DATASET_ROOT")
        self.source_image_root = Path(os.environ.get(env_name, configured_root)) if (os.environ.get(env_name) or configured_root) else None
        self.require_source_files = bool(dataset_cfg.get("require_source_files", False))
        self.scene_index_path = self._resolve_path(
            str(dataset_cfg.get("scene_index_path", "data/indexes/sarfishsample_scene_index.csv"))
        )
        self.task_state_profile_path = self._resolve_path(
            str(dataset_cfg.get("task_state_profile_path", "data/indexes/sarfishsample_task_state_profile.json"))
        )
        self.task_state_profile = self._load_task_state_profile()
        self.scene_index = self._load_scene_index()

    def build_task_stream(
        self,
        num_slots: int,
        slot_seconds: float,
        task_stream_cfg: dict[str, object] | None = None,
    ) -> list[TaskSample | None]:
        task_stream_cfg = task_stream_cfg or {}
        mode = str(task_stream_cfg.get("mode", "observation_window")).lower()
        if mode == "observation_window":
            return self._build_observation_window_stream(num_slots, slot_seconds, task_stream_cfg)
        if mode == "one_task_per_slot":
            return self._build_one_task_per_slot(num_slots, task_stream_cfg)
        raise ValueError(f"Unsupported task stream mode: {mode}")

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def _load_task_state_profile(self) -> dict[str, Any]:
        if not self.task_state_profile_path.exists():
            raise FileNotFoundError(f"SARFishSample task state profile not found: {self.task_state_profile_path}")
        with self.task_state_profile_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _load_scene_index(self) -> list[dict[str, Any]]:
        if not self.scene_index_path.exists():
            raise FileNotFoundError(f"SARFishSample scene index not found: {self.scene_index_path}")

        scenes: list[dict[str, Any]] = []
        with self.scene_index_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if str(row.get("product_type", "")).upper() != self.product_type:
                    continue
                selected_image = self._select_image_path(str(row.get("image_path", "")))
                if selected_image is None:
                    continue
                image_path, logical_image_path = selected_image
                label_path = self._resolve_path(str(row.get("label_csv_path", "")))
                if self.require_source_files and not label_path.exists():
                    continue
                scenes.append(
                    {
                        "scene_id": str(row.get("scene_id", "")),
                        "product_type": str(row.get("product_type", "")),
                        "image_path": str(image_path),
                        "logical_image_path": logical_image_path,
                        "label_path": str(label_path),
                        "product_identifier": str(row.get("product_identifier", "")),
                        "target_count": self._to_int(row.get("target_count")),
                        "bbox_area_mean": self._to_float(row.get("bbox_area_mean")),
                        "bbox_area_max": self._to_float(row.get("bbox_area_max")),
                        "small_target_ratio": self._to_float(row.get("small_target_ratio")),
                        "vessel_length_mean": self._to_float(row.get("vessel_length_mean")),
                        "is_vessel_count": self._to_int(row.get("is_vessel_count")),
                        "is_fishing_count": self._to_int(row.get("is_fishing_count")),
                        "estimated_roi_size_mb": self._to_float(row.get("estimated_roi_size_mb")),
                        "recommended_difficulty": self._to_float(row.get("recommended_difficulty")),
                        "recommended_uncertainty": self._to_float(row.get("recommended_uncertainty")),
                    }
                )

        if not scenes:
            raise ValueError(
                "No usable SARFishSample GRD validation scenes were found in "
                f"{self.scene_index_path}"
            )
        return scenes

    @staticmethod
    def _logical_image_path(candidate: str) -> str:
        normalized = str(candidate).replace("\\", "/")
        if normalized.startswith("GRD/"):
            return normalized
        marker = "/GRD/"
        if marker in normalized:
            return "GRD/" + normalized.split(marker, 1)[1]
        return normalized.lstrip("/")

    def _select_image_path(self, raw_image_path: str) -> tuple[Path, str] | None:
        candidates = [item.strip() for item in raw_image_path.split(";") if item.strip()]
        ordered = [candidate for candidate in candidates if Path(candidate).suffix.lower() in {".tif", ".tiff"} and "measurement" in candidate.replace("\\", "/")]
        ordered.extend(candidate for candidate in candidates if Path(candidate).suffix.lower() in {".tif", ".tiff"} and candidate not in ordered)
        metadata_only_candidate: tuple[Path, str] | None = None
        for candidate in ordered:
            logical = self._logical_image_path(candidate)
            path = self.source_image_root / Path(logical) if self.source_image_root is not None else Path(logical)
            if self._path_exists(path):
                return self._storage_path(path), logical
            if metadata_only_candidate is None:
                metadata_only_candidate = (path, logical)
        return None if self.require_source_files else metadata_only_candidate

    @staticmethod
    def _extended_windows_path(path: Path) -> str:
        path_text = str(path)
        if os.name != "nt" or path_text.startswith("\\\\?\\"):
            return path_text
        if not path.is_absolute():
            path_text = str(path.resolve())
        return "\\\\?\\" + path_text

    def _path_exists(self, path: Path) -> bool:
        return path.exists() or os.path.exists(self._extended_windows_path(path))

    def _storage_path(self, path: Path) -> Path:
        if path.exists():
            return path
        extended_path = self._extended_windows_path(path)
        if os.path.exists(extended_path):
            return Path(extended_path)
        return path

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: object, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _sample_range(rng: random.Random, value_range: tuple[float, float]) -> float:
        return rng.uniform(float(value_range[0]), float(value_range[1]))

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

    def _sample_difficulty(self, scene: dict[str, Any], rng: random.Random) -> float:
        base = float(scene.get("recommended_difficulty", 0.0))
        if base <= 0.0:
            base = float(
                self.task_state_profile.get("recommended_difficulty_bins", {})
                .get("scene_distribution", {})
                .get("median", 0.65)
            )
        return self._clamp(base + rng.uniform(-0.04, 0.04), 0.05, 1.0)

    def _sample_uncertainty(self, scene: dict[str, Any], rng: random.Random) -> float:
        base = float(scene.get("recommended_uncertainty", 0.0))
        if base <= 0.0:
            base = float(
                self.task_state_profile.get("recommended_uncertainty_bins", {})
                .get("scene_distribution", {})
                .get("median", 0.55)
            )
        return self._clamp(base + rng.uniform(-0.04, 0.04), 0.05, 1.0)

    def _deadline_range_s(self, task_class: str, cfg: dict[str, object]) -> tuple[float, float, str]:
        class_prefix = "ordinary" if task_class == "routine" else task_class
        min_key = f"{class_prefix}_deadline_min_minutes"
        max_key = f"{class_prefix}_deadline_max_minutes"
        if min_key in cfg and max_key in cfg:
            lower = max(0.0, float(cfg[min_key]) * 60.0)
            upper = max(lower, float(cfg[max_key]) * 60.0)
            return lower, upper, "task_stream_minutes"

        ranges = self.TASK_CLASS_RANGES[task_class]
        lower, upper = ranges["deadline_s"]  # type: ignore[misc]
        return float(lower), float(upper), "adapter_default_seconds"

    def _make_task(
        self,
        slot_index: int,
        task_index: int,
        window_id: str,
        task_class: str,
        scene: dict[str, Any],
        rng: random.Random,
        task_stream_cfg: dict[str, object] | None = None,
        active_observation_window: bool = True,
    ) -> TaskSample:
        task_stream_cfg = task_stream_cfg or {}
        ranges = self.TASK_CLASS_RANGES[task_class]
        task_value = self._sample_range(rng, ranges["task_value"])  # type: ignore[arg-type]
        old_task_value = task_value
        deadline_range_s = self._deadline_range_s(task_class, task_stream_cfg)
        deadline_s = self._sample_range(rng, (deadline_range_s[0], deadline_range_s[1]))
        quality_requirement = self._sample_range(rng, ranges["quality_requirement"])  # type: ignore[arg-type]
        information_requirement = self._sample_range(rng, ranges["information_requirement"])  # type: ignore[arg-type]
        difficulty = self._sample_difficulty(scene, rng)
        uncertainty = self._sample_uncertainty(scene, rng)
        value_model_enabled = mission_prior_value_v2_enabled(task_stream_cfg)
        product_requirement, product_requirement_source = assign_product_requirement(
            task_value=task_value,
            deadline_s=deadline_s,
            uncertainty=uncertainty,
            high_value_flag=bool(ranges["high_value_flag"]),
            emergency_flag=bool(ranges["emergency_flag"]),
            task_class=task_class,
        )
        if value_model_enabled:
            product_requirement, product_requirement_source = assign_product_requirement(
                task_value=old_task_value,
                deadline_s=deadline_s,
                uncertainty=0.0,
                high_value_flag=bool(ranges["high_value_flag"]),
                emergency_flag=bool(ranges["emergency_flag"]),
                task_class=task_class,
            )
            product_requirement_source = "mission_prior_proxy_label_free_for_v2"
        metadata = {
            "dataset_name": "sarfishsample",
            "product_type": scene["product_type"],
            "split": self.split,
            "product_identifier": scene["product_identifier"],
            "logical_image_path": scene["logical_image_path"],
            "logical_label_path": "GRD/validation/GRD_validation.csv",
            "active_observation_window": active_observation_window,
            "offline_label_calibrated": True,
            "offline_label_summary": {
                "target_count": scene["target_count"],
                "bbox_area_mean": scene["bbox_area_mean"],
                "bbox_area_max": scene["bbox_area_max"],
                "small_target_ratio": scene["small_target_ratio"],
                "vessel_length_mean": scene["vessel_length_mean"],
                "is_vessel_count": scene["is_vessel_count"],
                "is_fishing_count": scene["is_fishing_count"],
                "estimated_roi_size_mb": scene["estimated_roi_size_mb"],
            },
            "ground_truth_usage": "offline_metadata_only_not_used_for_online_priority",
            "deadline_policy_source": deadline_range_s[2],
            "deadline_min_minutes": deadline_range_s[0] / 60.0,
            "deadline_max_minutes": deadline_range_s[1] / 60.0,
        }
        if value_model_enabled:
            value_result = compute_mission_prior_value_v2(
                task_id=f"sarfishsample_task_{task_index:04d}",
                scene_id=str(scene["scene_id"]),
                task_class=task_class,
                product_requirement=product_requirement,
                deadline_s=deadline_s,
                container_cfg=task_stream_cfg,
            )
            task_value = value_result.task_value
            metadata.update(task_value_component_metadata(value_result, old_task_value=old_task_value))
            metadata["value_model_product_requirement_source"] = product_requirement_source
            metadata["value_model_product_requirement_uncertainty_used"] = 0.0

        return TaskSample(
            task_id=f"sarfishsample_task_{task_index:04d}",
            image_path=str(scene["image_path"]),
            label_path=str(scene["label_path"]),
            scene_id=str(scene["scene_id"]),
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
            high_value_flag=bool(ranges["high_value_flag"]),
            emergency_flag=bool(ranges["emergency_flag"]),
            product_requirement=product_requirement,
            product_requirement_source=product_requirement_source,
            value_level={"routine": "medium", "high_value": "high", "emergency": "urgent"}[task_class],
        )

    def _build_one_task_per_slot(self, num_slots: int, cfg: dict[str, object] | None = None) -> list[TaskSample]:
        cfg = cfg or {}
        rng = random.Random(self.seed)
        tasks: list[TaskSample] = []
        for slot_index in range(num_slots):
            task_class = self._sample_window_class(rng, cfg)
            scene = rng.choice(self.scene_index)
            tasks.append(
                self._make_task(
                    slot_index=slot_index,
                    task_index=slot_index,
                    window_id=f"window_{slot_index // 12:04d}",
                    task_class=task_class,
                    scene=scene,
                    rng=rng,
                    task_stream_cfg=cfg,
                )
            )
        return tasks

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
        background_task_probability = float(cfg.get("background_task_probability", 0.0))

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
                window_id = f"sarfishsample_window_{global_window_id:04d}"
                global_window_id += 1

                for local_slot in range(start, end, interval_slots):
                    absolute_slot = orbit_start + local_slot
                    if absolute_slot >= num_slots:
                        break
                    occupancy[local_slot] = True
                    stream[absolute_slot] = self._make_task(
                        slot_index=absolute_slot,
                        task_index=task_counter,
                        window_id=window_id,
                        task_class=task_class,
                        scene=rng.choice(self.scene_index),
                        rng=rng,
                        task_stream_cfg=cfg,
                    )
                    task_counter += 1

            for local_slot in range(available_slots):
                absolute_slot = orbit_start + local_slot
                if absolute_slot >= num_slots or stream[absolute_slot] is not None:
                    continue
                if rng.random() < background_task_probability:
                    scene = rng.choice(self.scene_index)
                    stream[absolute_slot] = self._make_task(
                        slot_index=absolute_slot,
                        task_index=task_counter,
                        window_id="",
                        task_class="routine",
                        scene=scene,
                        rng=rng,
                        task_stream_cfg=cfg,
                        active_observation_window=False,
                    )
                    task_counter += 1

        forced_windows = cfg.get("forced_windows_s", [])
        if isinstance(forced_windows, list):
            for forced_index, window in enumerate(forced_windows):
                if not isinstance(window, dict):
                    continue
                start_slot = max(0, int(round(float(window.get("start_s", 0.0)) / max(slot_seconds, 1.0e-9))))
                end_slot = min(num_slots, int(round(float(window.get("end_s", 0.0)) / max(slot_seconds, 1.0e-9))))
                if end_slot <= start_slot:
                    continue
                interval_slots = int(window.get("interval_slots", 0))
                if interval_slots <= 0:
                    interval_slots = max(1, int(round(float(window.get("interval_s", slot_seconds)) / max(slot_seconds, 1.0e-9))))
                task_class = str(window.get("task_class", "high_value"))
                if task_class not in self.TASK_CLASS_RANGES:
                    task_class = "high_value"
                replace_existing = bool(window.get("replace_existing", False))
                window_id = str(window.get("window_id", f"sarfishsample_forced_{forced_index:04d}"))
                for slot_index in range(start_slot, end_slot, interval_slots):
                    if stream[slot_index] is not None and not replace_existing:
                        continue
                    scene = rng.choice(self.scene_index)
                    stream[slot_index] = self._make_task(
                        slot_index=slot_index,
                        task_index=task_counter,
                        window_id=window_id,
                        task_class=task_class,
                        scene=scene,
                        rng=rng,
                        task_stream_cfg=cfg,
                        active_observation_window=True,
                    )
                    task_counter += 1

        if num_slots > 2 and all(item is not None for item in stream):
            stream[-1] = None

        return stream
