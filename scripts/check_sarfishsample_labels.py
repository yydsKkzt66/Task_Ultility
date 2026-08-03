"""Read-only SARFish label CSV audit and TaskSample alignment reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "sarfishsample_audit"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

CSV_BASIC_REPORT = AUDIT_DIR / "label_csv_basic_report.md"
CSV_SCHEMA_REPORT = AUDIT_DIR / "label_schema_report.md"
BBOX_STATS_REPORT = AUDIT_DIR / "bbox_stats_report.md"
IMAGE_MATCH_REPORT = AUDIT_DIR / "label_image_match_report.md"
LEAKAGE_REPORT = AUDIT_DIR / "task_state_leakage_boundary.md"
TASKSAMPLE_MAPPING_REPORT = AUDIT_DIR / "tasksample_mapping_from_labels.md"
ENGINEERING_ALIGNMENT_REPORT = AUDIT_DIR / "engineering_alignment_report.md"
READINESS_SUMMARY_REPORT = AUDIT_DIR / "label_readiness_summary.md"

BBOX_STATS_JSON = INDEX_DIR / "sarfishsample_label_bbox_stats.json"
OBJECTS_PROBE_CSV = INDEX_DIR / "sarfishsample_label_objects_probe.csv"
TASK_STATE_PROFILE_JSON = INDEX_DIR / "sarfishsample_task_state_profile.json"
SCENE_INDEX_CSV = INDEX_DIR / "sarfishsample_scene_index.csv"

TARGET_CSV_NAMES = {
    "grd_train.csv",
    "grd_validation.csv",
    "slc_train.csv",
    "slc_validation.csv",
    "train.csv",
    "validation.csv",
    "public.csv",
    "esa_xview3_scenename_mapping.csv",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MISSING_VALUES = {"", "na", "n/a", "nan", "none", "null", "nil"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}

BBOX_FIELDS = ["top", "left", "bottom", "right", "detect_scene_row", "detect_scene_column", "detect_lat", "detect_lon"]
TARGET_FIELDS = ["is_vessel", "is_fishing", "vessel_length_m", "confidence", "source"]
SCENE_FIELDS = ["scene_id", "detect_id", "product_id", "product_identifier", "scene_name", "chip_id", "image_id"]


@dataclass
class CsvSummary:
    path: Path
    size_bytes: int
    row_count: int = 0
    column_count: int = 0
    columns: list[str] = field(default_factory=list)
    first_rows: list[dict[str, str]] = field(default_factory=list)
    missing_counts: dict[str, int] = field(default_factory=dict)
    dtype_inference: dict[str, str] = field(default_factory=dict)
    non_missing_counts: dict[str, int] = field(default_factory=dict)
    csv_type: str = "unknown"
    error: str = ""
    only_header: bool = False
    empty_file: bool = False
    field_matches: dict[str, dict[str, list[str]]] = field(default_factory=dict)


@dataclass
class ObjectRecord:
    csv_path: str
    row_index: int
    partition: str
    product_type: str
    scene_id: str
    product_identifier: str
    swath_index: str
    detect_id: str
    top: float
    left: float
    bottom: float
    right: float
    bbox_width: float
    bbox_height: float
    bbox_area: float
    detect_scene_row: float | None
    detect_scene_column: float | None
    detect_lat: float | None
    detect_lon: float | None
    vessel_length_m: float | None
    source: str
    is_vessel: str
    is_fishing: str
    confidence: str
    size_class: str = ""


@dataclass
class LocalImageRecord:
    product_identifier: str
    product_type: str
    image_path: str
    file_name: str
    polarization: str = ""
    swath_index: str = ""


def fs_path(path: Path) -> str:
    absolute = str(path.absolute())
    if os.name != "nt":
        return absolute
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def rel(path: Path, base: Path = PROJECT_ROOT) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def safe_size(path: Path) -> int:
    try:
        return os.stat(fs_path(path)).st_size
    except OSError:
        return 0


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in MISSING_VALUES


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def parse_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def normalize_bool(value: Any) -> str:
    if is_missing(value):
        return "missing"
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return "true"
    if text in {"false", "f", "0", "no", "n"}:
        return "false"
    return str(value).strip()


def percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = pct * (len(clean) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return clean[lower]
    weight = rank - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def describe(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "min": min(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p25": percentile(clean, 0.25),
        "p75": percentile(clean, 0.75),
        "p90": percentile(clean, 0.90),
        "p95": percentile(clean, 0.95),
        "max": max(clean),
    }


def compact(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: compact(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v, digits) for v in value]
    return value


def infer_value_type(value: str) -> str:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return "bool"
    try:
        int(text)
        return "int"
    except ValueError:
        pass
    try:
        float(text)
        return "float"
    except ValueError:
        return "string"


def combine_types(types: set[str]) -> str:
    if not types:
        return "empty"
    if types == {"bool"}:
        return "boolean"
    if types <= {"int"}:
        return "integer"
    if types <= {"int", "float"}:
        return "float"
    return "string"


def find_csv_files(root: Path, labels: Path | None) -> list[Path]:
    roots: list[Path] = [root]
    if labels is not None:
        roots.append(labels)
    found: dict[str, Path] = {}
    for base in roots:
        if not base.exists():
            continue
        if base.is_file() and base.suffix.lower() == ".csv":
            found[str(base.resolve()).lower()] = base.resolve()
            continue
        if base.is_dir():
            for directory, dir_names, file_names in os.walk(fs_path(base)):
                dir_names[:] = [name for name in dir_names if name not in SKIP_DIRS]
                directory_path = Path(directory.replace("\\\\?\\", ""))
                for file_name in file_names:
                    if not file_name.lower().endswith(".csv"):
                        continue
                    path = directory_path / file_name
                    lower_name = file_name.lower()
                    if lower_name in TARGET_CSV_NAMES or "grd" in lower_name or "slc" in lower_name or "validation" in lower_name or "train" in lower_name:
                        found[str(path.resolve()).lower()] = path.resolve()
    return sorted(found.values(), key=lambda p: str(p).lower())


def open_csv_reader(path: Path):
    file = open(fs_path(path), "r", encoding="utf-8-sig", errors="replace", newline="")
    return file, csv.DictReader(file)


def classify_csv(path: Path, columns: list[str]) -> str:
    name = path.name.lower()
    normalized = {normalize_col(col) for col in columns}
    if "esa_xview3_scenename_mapping" in name:
        return "scene name mapping"
    if "grd_product_identifier" in normalized or name.startswith("grd_"):
        return "SARFish GRD label"
    if "slc_product_identifier" in normalized or "swath_index" in normalized or name.startswith("slc_"):
        return "SARFish SLC label"
    if {"scene_id", "detect_scene_row", "detect_scene_column"}.issubset(normalized):
        return "xView3 original label"
    if {"scene_name", "product_identifier"}.intersection(normalized):
        return "auxiliary file"
    return "unknown"


def match_named_fields(columns: list[str], targets: list[str]) -> dict[str, list[str]]:
    normalized_to_original: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        normalized_to_original[normalize_col(col)].append(col)
    result: dict[str, list[str]] = {}
    for target in targets:
        target_norm = normalize_col(target)
        exact = normalized_to_original.get(target_norm, [])
        fuzzy = []
        if not exact:
            for normalized, originals in normalized_to_original.items():
                if target_norm in normalized or normalized in target_norm:
                    fuzzy.extend(originals)
        result[target] = exact or fuzzy
    return result


def summarize_csv(path: Path) -> CsvSummary:
    summary = CsvSummary(path=path, size_bytes=safe_size(path))
    if summary.size_bytes == 0:
        summary.empty_file = True
        return summary

    try:
        file, reader = open_csv_reader(path)
    except OSError as exc:
        summary.error = str(exc)
        return summary

    with file:
        columns = list(reader.fieldnames or [])
        summary.columns = columns
        summary.column_count = len(columns)
        missing_counts = {col: 0 for col in columns}
        non_missing_counts = {col: 0 for col in columns}
        type_sets: dict[str, set[str]] = {col: set() for col in columns}
        try:
            for row_index, row in enumerate(reader, start=1):
                summary.row_count += 1
                if row_index <= 5:
                    summary.first_rows.append({col: str(row.get(col, "")) for col in columns})
                for col in columns:
                    value = row.get(col, "")
                    if is_missing(value):
                        missing_counts[col] += 1
                    else:
                        non_missing_counts[col] += 1
                        if len(type_sets[col]) < 3:
                            type_sets[col].add(infer_value_type(str(value)))
        except csv.Error as exc:
            summary.error = f"CSV parse error after {summary.row_count} rows: {exc}"

    summary.missing_counts = missing_counts
    summary.non_missing_counts = non_missing_counts
    summary.dtype_inference = {col: combine_types(types) for col, types in type_sets.items()}
    summary.csv_type = classify_csv(path, columns)
    summary.only_header = bool(columns and summary.row_count == 0)
    summary.empty_file = not columns and summary.row_count == 0
    summary.field_matches = {
        "bbox_location": match_named_fields(columns, BBOX_FIELDS),
        "target_vessel": match_named_fields(columns, TARGET_FIELDS),
        "scene_product": match_named_fields(columns, SCENE_FIELDS),
    }
    return summary


def first_col(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_col(col): col for col in columns}
    for candidate in candidates:
        candidate_norm = normalize_col(candidate)
        if candidate_norm in normalized:
            return normalized[candidate_norm]
    for candidate in candidates:
        candidate_norm = normalize_col(candidate)
        for norm, original in normalized.items():
            if candidate_norm in norm or norm in candidate_norm:
                return original
    return None


def collect_objects(csv_paths: list[Path]) -> tuple[list[ObjectRecord], dict[str, Any]]:
    objects: list[ObjectRecord] = []
    per_csv: dict[str, Any] = {}
    invalid_examples: list[dict[str, Any]] = []

    for path in csv_paths:
        try:
            file, reader = open_csv_reader(path)
        except OSError as exc:
            per_csv[rel(path)] = {"error": str(exc)}
            continue

        with file:
            columns = list(reader.fieldnames or [])
            top_col = first_col(columns, ["top"])
            left_col = first_col(columns, ["left"])
            bottom_col = first_col(columns, ["bottom"])
            right_col = first_col(columns, ["right"])
            scene_col = first_col(columns, ["scene_id", "scene_name", "image_id", "chip_id"])
            detect_col = first_col(columns, ["detect_id"])
            partition_col = first_col(columns, ["partition"])
            product_type_col = first_col(columns, ["product_type"])
            product_col = first_col(columns, ["GRD_product_identifier", "SLC_product_identifier", "product_identifier", "product_id"])
            swath_col = first_col(columns, ["swath_index"])
            row_col = first_col(columns, ["detect_scene_row"])
            column_col = first_col(columns, ["detect_scene_column"])
            lat_col = first_col(columns, ["detect_lat"])
            lon_col = first_col(columns, ["detect_lon"])
            length_col = first_col(columns, ["vessel_length_m"])
            source_col = first_col(columns, ["source"])
            is_vessel_col = first_col(columns, ["is_vessel"])
            is_fishing_col = first_col(columns, ["is_fishing"])
            confidence_col = first_col(columns, ["confidence"])

            has_bbox = all([top_col, left_col, bottom_col, right_col])
            numeric_bbox_rows = 0
            valid_bbox_rows = 0
            bottom_gt_top_rows = 0
            right_gt_left_rows = 0
            reordered_y_rows = 0
            reordered_x_rows = 0
            invalid_bbox_rows = 0
            total_rows = 0
            for row_index, row in enumerate(reader, start=1):
                total_rows += 1
                if not has_bbox:
                    continue
                top = parse_float(row.get(top_col))
                left = parse_float(row.get(left_col))
                bottom = parse_float(row.get(bottom_col))
                right = parse_float(row.get(right_col))
                if top is None or left is None or bottom is None or right is None:
                    invalid_bbox_rows += 1
                    if len(invalid_examples) < 20:
                        invalid_examples.append({"csv": rel(path), "row": row_index, "reason": "non_numeric_bbox"})
                    continue
                numeric_bbox_rows += 1
                if bottom > top:
                    bottom_gt_top_rows += 1
                else:
                    reordered_y_rows += 1
                if right > left:
                    right_gt_left_rows += 1
                else:
                    reordered_x_rows += 1
                bbox_width = abs(right - left)
                bbox_height = abs(bottom - top)
                if bbox_width <= 0.0 or bbox_height <= 0.0:
                    invalid_bbox_rows += 1
                    if len(invalid_examples) < 20:
                        invalid_examples.append({"csv": rel(path), "row": row_index, "reason": "zero_extent_bbox"})
                    continue
                valid_bbox_rows += 1
                product_type = str(row.get(product_type_col, "") if product_type_col else "")
                if not product_type:
                    product_type = "GRD" if "grd" in path.name.lower() else "SLC" if "slc" in path.name.lower() else ""
                product_identifier = str(row.get(product_col, "") if product_col else "")
                objects.append(
                    ObjectRecord(
                        csv_path=rel(path),
                        row_index=row_index,
                        partition=str(row.get(partition_col, "") if partition_col else ""),
                        product_type=product_type,
                        scene_id=str(row.get(scene_col, "") if scene_col else ""),
                        product_identifier=product_identifier,
                        swath_index=str(row.get(swath_col, "") if swath_col else ""),
                        detect_id=str(row.get(detect_col, "") if detect_col else ""),
                        top=top,
                        left=left,
                        bottom=bottom,
                        right=right,
                        bbox_width=bbox_width,
                        bbox_height=bbox_height,
                        bbox_area=bbox_width * bbox_height,
                        detect_scene_row=parse_float(row.get(row_col)) if row_col else None,
                        detect_scene_column=parse_float(row.get(column_col)) if column_col else None,
                        detect_lat=parse_float(row.get(lat_col)) if lat_col else None,
                        detect_lon=parse_float(row.get(lon_col)) if lon_col else None,
                        vessel_length_m=parse_float(row.get(length_col)) if length_col else None,
                        source=str(row.get(source_col, "") if source_col else ""),
                        is_vessel=normalize_bool(row.get(is_vessel_col)) if is_vessel_col else "",
                        is_fishing=normalize_bool(row.get(is_fishing_col)) if is_fishing_col else "",
                        confidence=str(row.get(confidence_col, "") if confidence_col else ""),
                    )
                )

        per_csv[rel(path)] = {
            "has_bbox_columns": has_bbox,
            "total_rows": total_rows,
            "numeric_bbox_rows": numeric_bbox_rows,
            "valid_bbox_rows": valid_bbox_rows,
            "bottom_gt_top_rows": bottom_gt_top_rows,
            "right_gt_left_rows": right_gt_left_rows,
            "reordered_y_rows": reordered_y_rows,
            "reordered_x_rows": reordered_x_rows,
            "invalid_bbox_rows": invalid_bbox_rows,
        }

    area_values = [obj.bbox_area for obj in objects]
    q33 = percentile(area_values, 1 / 3)
    q66 = percentile(area_values, 2 / 3)
    for obj in objects:
        if obj.vessel_length_m is not None:
            if obj.vessel_length_m < 30.0:
                obj.size_class = "small"
            elif obj.vessel_length_m < 100.0:
                obj.size_class = "medium"
            else:
                obj.size_class = "large"
        elif q33 is not None and q66 is not None:
            if obj.bbox_area <= q33:
                obj.size_class = "small"
            elif obj.bbox_area <= q66:
                obj.size_class = "medium"
            else:
                obj.size_class = "large"
        else:
            obj.size_class = "unknown"

    return objects, {"per_csv": per_csv, "invalid_bbox_examples": invalid_examples}


def scan_local_images(root: Path) -> dict[str, Any]:
    files: list[Path] = []
    safe_dirs: set[str] = set()
    zip_products: set[str] = set()
    manifests: list[Path] = []
    xml_records: list[dict[str, str]] = []
    image_records: list[LocalImageRecord] = []

    for directory, dir_names, file_names in os.walk(fs_path(root)):
        dir_names[:] = [name for name in dir_names if name not in SKIP_DIRS]
        directory_path = Path(directory.replace("\\\\?\\", ""))
        for dir_name in dir_names:
            if dir_name.lower().endswith(".safe"):
                safe_dirs.add(dir_name[:-5])
        for file_name in file_names:
            path = directory_path / file_name
            files.append(path)
            lower = file_name.lower()
            if lower.endswith(".safe.zip"):
                zip_products.add(file_name[:-9])
            if lower == "manifest.safe":
                manifests.append(path)

    for path in files:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            product_id = nearest_safe_product(path)
            product_type = infer_product_type(path)
            image_records.append(
                LocalImageRecord(
                    product_identifier=product_id,
                    product_type=product_type,
                    image_path=rel(path),
                    file_name=path.name,
                    polarization=infer_polarization(path.name),
                    swath_index=infer_swath(path.name),
                )
            )
        elif suffix == ".xml" and "\\annotation\\" in str(path).lower():
            record = parse_annotation_header(path)
            if record:
                xml_records.append(record)

    possible_product_ids = sorted({item.product_identifier for item in image_records if item.product_identifier} | safe_dirs | zip_products)
    return {
        "safe_products": sorted(safe_dirs),
        "zip_products": sorted(zip_products),
        "manifests": [rel(path) for path in manifests],
        "annotation_xml_records": xml_records,
        "images": image_records,
        "possible_product_ids": possible_product_ids,
        "probe_csv_exists": (INDEX_DIR / "sarfishsample_probe.csv").exists(),
        "stats_json_exists": (INDEX_DIR / "sarfishsample_stats.json").exists(),
    }


def nearest_safe_product(path: Path) -> str:
    for parent in [path.parent, *path.parents]:
        name = parent.name
        if name.lower().endswith(".safe"):
            return name[:-5]
    return ""


def infer_product_type(path: Path) -> str:
    text = str(path).lower()
    if "\\grd\\" in text or "/grd/" in text or "-grd-" in path.name.lower():
        return "GRD"
    if "\\slc\\" in text or "/slc/" in text or "-slc-" in path.name.lower():
        return "SLC"
    return ""


def infer_polarization(name: str) -> str:
    lower = name.lower()
    if "-vv-" in lower or "_vv" in lower:
        return "VV"
    if "-vh-" in lower or "_vh" in lower:
        return "VH"
    return ""


def infer_swath(name: str) -> str:
    match = re.search(r"iw([123])", name.lower())
    return match.group(1) if match else ""


def xml_text(node: ET.Element, wanted: str) -> str:
    wanted = wanted.lower()
    for item in node.iter():
        tag = item.tag.split("}", 1)[-1].lower()
        if tag == wanted and item.text:
            return item.text.strip()
    return ""


def parse_annotation_header(path: Path) -> dict[str, str]:
    try:
        root = ET.parse(fs_path(path)).getroot()
    except Exception:
        return {}
    return {
        "xml_path": rel(path),
        "safe_product": nearest_safe_product(path),
        "product_type": xml_text(root, "productType"),
        "polarisation": xml_text(root, "polarisation"),
        "swath": xml_text(root, "swath"),
        "start_time": xml_text(root, "startTime"),
        "stop_time": xml_text(root, "stopTime"),
    }


def object_key(obj: ObjectRecord) -> str:
    product = obj.product_identifier or obj.scene_id
    return f"{obj.product_type}|{product}|{obj.scene_id}"


def build_group_stats(objects: list[ObjectRecord], local_scan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    groups: dict[str, list[ObjectRecord]] = defaultdict(list)
    for obj in objects:
        groups[object_key(obj)].append(obj)

    image_by_product: dict[str, list[LocalImageRecord]] = defaultdict(list)
    for item in local_scan["images"]:
        image_by_product[item.product_identifier].append(item)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        first = items[0]
        areas = [item.bbox_area for item in items]
        lengths = [item.vessel_length_m for item in items if item.vessel_length_m is not None]
        small_count = sum(1 for item in items if item.size_class == "small")
        vessel_count = sum(1 for item in items if item.is_vessel == "true")
        fishing_count = sum(1 for item in items if item.is_fishing == "true")
        target_count = len(items)
        area_mean = statistics.fmean(areas) if areas else 0.0
        area_max = max(areas) if areas else 0.0
        small_ratio = small_count / max(target_count, 1)
        length_mean = statistics.fmean(lengths) if lengths else None
        density_proxy = min(1.0, math.log1p(target_count) / math.log(200.0))
        area_inverse = 1.0 - min(area_mean / max(percentile([obj.bbox_area for obj in objects], 0.95) or area_mean or 1.0, 1.0), 1.0)
        difficulty = max(0.05, min(1.0, 0.20 + 0.30 * density_proxy + 0.30 * small_ratio + 0.20 * area_inverse))
        ambiguity = sum(1 for item in items if item.is_vessel == "missing" or item.is_fishing == "missing") / max(target_count, 1)
        low_conf = sum(1 for item in items if item.confidence.strip().lower() == "low") / max(target_count, 1)
        uncertainty = max(0.05, min(1.0, 0.15 + 0.30 * small_ratio + 0.25 * ambiguity + 0.20 * low_conf + 0.10 * density_proxy))
        matched_images = image_by_product.get(first.product_identifier, [])
        rows.append(
            {
                "scene_id": first.scene_id,
                "product_type": first.product_type,
                "product_identifier": first.product_identifier,
                "image_path": ";".join(item.image_path for item in matched_images),
                "label_csv_path": first.csv_path,
                "matched_label_count": target_count if matched_images else 0,
                "target_count": target_count,
                "bbox_area_mean": area_mean,
                "bbox_area_max": area_max,
                "small_target_ratio": small_ratio,
                "vessel_length_mean": length_mean,
                "is_vessel_count": vessel_count,
                "is_fishing_count": fishing_count,
                "estimated_roi_size_mb": 0.80 + 0.18 * target_count,
                "recommended_difficulty": difficulty,
                "recommended_uncertainty": uncertainty,
            }
        )

    local_products = set(local_scan["possible_product_ids"])
    label_products = {obj.product_identifier for obj in objects if obj.product_identifier}
    label_scenes = {obj.scene_id for obj in objects if obj.scene_id}
    matched_products = sorted(label_products & local_products)
    return rows, {
        "local_products": sorted(local_products),
        "label_products": sorted(label_products),
        "label_scenes": sorted(label_scenes),
        "matched_products": matched_products,
        "label_products_without_local_image": sorted(label_products - local_products),
        "local_products_without_label": sorted(local_products - label_products),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_objects_probe(objects: list[ObjectRecord]) -> None:
    rows = [obj.__dict__ for obj in objects]
    fieldnames = [
        "csv_path",
        "row_index",
        "partition",
        "product_type",
        "scene_id",
        "product_identifier",
        "swath_index",
        "detect_id",
        "top",
        "left",
        "bottom",
        "right",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "detect_scene_row",
        "detect_scene_column",
        "detect_lat",
        "detect_lon",
        "vessel_length_m",
        "source",
        "is_vessel",
        "is_fishing",
        "confidence",
        "size_class",
    ]
    write_csv(OBJECTS_PROBE_CSV, rows, fieldnames)


def write_scene_index(scene_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scene_id",
        "product_type",
        "image_path",
        "label_csv_path",
        "matched_label_count",
        "target_count",
        "bbox_area_mean",
        "bbox_area_max",
        "small_target_ratio",
        "vessel_length_mean",
        "is_vessel_count",
        "is_fishing_count",
        "estimated_roi_size_mb",
        "recommended_difficulty",
        "recommended_uncertainty",
        "product_identifier",
    ]
    write_csv(SCENE_INDEX_CSV, scene_rows, fieldnames)


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        escaped = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(escaped) + " |")
    return lines


def write_basic_report(summaries: list[CsvSummary], pandas_status: str) -> None:
    lines = ["# SARFish Label CSV Basic Report", "", f"- CSV parser path: {pandas_status}", ""]
    rows = []
    for item in summaries:
        rows.append([rel(item.path), item.size_bytes, item.row_count, item.column_count, item.csv_type, item.empty_file, item.only_header, item.error])
    lines.extend(md_table(["path", "size_bytes", "rows", "columns", "type", "empty", "only_header", "error"], rows))
    lines.append("")
    for item in summaries:
        lines.append(f"## {rel(item.path)}")
        lines.append("")
        lines.append(f"- Type: {item.csv_type}")
        lines.append(f"- Rows: {item.row_count}")
        lines.append(f"- Columns: {item.column_count}")
        lines.append("- Full columns:")
        for col in item.columns:
            lines.append(f"  - `{col}`")
        lines.append("")
        lines.append("### First 5 Rows")
        if item.first_rows:
            headers = item.columns
            lines.extend(md_table(headers, [[row.get(col, "") for col in headers] for row in item.first_rows]))
        else:
            lines.append("_No data rows._")
        lines.append("")
        lines.append("### Missing Values")
        lines.extend(md_table(["column", "missing_count", "non_missing_count", "dtype"], [[col, item.missing_counts.get(col, 0), item.non_missing_counts.get(col, 0), item.dtype_inference.get(col, "")] for col in item.columns]))
        lines.append("")
    CSV_BASIC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_schema_report(summaries: list[CsvSummary]) -> None:
    lines = ["# SARFish Label Schema Report", ""]
    for item in summaries:
        lines.append(f"## {rel(item.path)}")
        lines.append("")
        lines.append(f"- Classified as: **{item.csv_type}**")
        for group, matches in item.field_matches.items():
            lines.append(f"### {group}")
            rows = []
            for target, columns in matches.items():
                rows.append([target, ", ".join(f"`{col}`" for col in columns) if columns else "missing"])
            lines.extend(md_table(["required/equivalent field", "matched column(s)"], rows))
            lines.append("")
    lines.append("## Readiness")
    lines.append("")
    has_grd = any(item.csv_type == "SARFish GRD label" for item in summaries)
    has_slc = any(item.csv_type == "SARFish SLC label" for item in summaries)
    has_bbox = any(all(item.field_matches["bbox_location"].get(field) for field in ("top", "left", "bottom", "right")) for item in summaries)
    lines.append(f"- SARFish GRD labels present: {has_grd}")
    lines.append(f"- SARFish SLC labels present: {has_slc}")
    lines.append(f"- BBox columns present in at least one CSV: {has_bbox}")
    CSV_SCHEMA_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_bbox_stats(objects: list[ObjectRecord], parse_meta: dict[str, Any]) -> dict[str, Any]:
    by_scene = Counter(obj.scene_id for obj in objects)
    by_product = Counter(obj.product_identifier for obj in objects)
    stats = {
        "total_objects_with_valid_bbox": len(objects),
        "per_csv_parse": parse_meta["per_csv"],
        "invalid_bbox_examples": parse_meta["invalid_bbox_examples"],
        "bbox_width": describe([obj.bbox_width for obj in objects]),
        "bbox_height": describe([obj.bbox_height for obj in objects]),
        "bbox_area": describe([obj.bbox_area for obj in objects]),
        "target_count_per_scene_id": describe([float(count) for count in by_scene.values()]),
        "target_count_per_product_identifier": describe([float(count) for count in by_product.values()]),
        "unique_scene_id_count": len(by_scene),
        "unique_product_identifier_count": len(by_product),
        "is_vessel_distribution": dict(Counter(obj.is_vessel for obj in objects)),
        "is_fishing_distribution": dict(Counter(obj.is_fishing for obj in objects)),
        "source_distribution": dict(Counter(obj.source for obj in objects)),
        "confidence_distribution": dict(Counter(obj.confidence for obj in objects)),
        "vessel_length_m": describe([obj.vessel_length_m for obj in objects if obj.vessel_length_m is not None]),
        "size_class_distribution": dict(Counter(obj.size_class for obj in objects)),
    }
    return compact(stats)


def write_bbox_report(stats: dict[str, Any]) -> None:
    lines = ["# SARFish Label BBox Statistics Report", ""]
    lines.append(f"- Valid bbox objects: {stats['total_objects_with_valid_bbox']}")
    lines.append(f"- Unique `scene_id`: {stats['unique_scene_id_count']}")
    lines.append(f"- Unique product identifiers: {stats['unique_product_identifier_count']}")
    lines.append("")
    lines.append("## Per CSV BBox Validity")
    rows = []
    for path, item in stats["per_csv_parse"].items():
        rows.append(
            [
                path,
                item.get("has_bbox_columns"),
                item.get("total_rows"),
                item.get("numeric_bbox_rows"),
                item.get("bottom_gt_top_rows"),
                item.get("right_gt_left_rows"),
                item.get("valid_bbox_rows"),
                item.get("invalid_bbox_rows"),
            ]
        )
    lines.extend(
        md_table(
            [
                "csv",
                "has_bbox_columns",
                "rows",
                "numeric_bbox_rows",
                "bottom_gt_top_rows",
                "right_gt_left_rows",
                "valid_bbox_rows",
                "invalid_bbox_rows",
            ],
            rows,
        )
    )
    lines.append("")
    lines.append("Note: SARFish validation labels often have `top > bottom` in image-row coordinates. BBox size is therefore computed with `abs(bottom - top)` while preserving the original columns in the object probe.")
    lines.append("")
    for key in ["bbox_width", "bbox_height", "bbox_area", "target_count_per_scene_id", "target_count_per_product_identifier", "vessel_length_m"]:
        lines.append(f"## {key}")
        lines.append("```json")
        lines.append(json.dumps(stats[key], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    for key in ["is_vessel_distribution", "is_fishing_distribution", "confidence_distribution", "source_distribution", "size_class_distribution"]:
        lines.append(f"## {key}")
        lines.extend(md_table(["value", "count"], [[k, v] for k, v in sorted(stats[key].items())]))
        lines.append("")
    if stats["invalid_bbox_examples"]:
        lines.append("## Invalid BBox Examples")
        for item in stats["invalid_bbox_examples"][:20]:
            lines.append(f"- `{item['csv']}` row {item['row']}: {item['reason']}")
    BBOX_STATS_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_match_report(match_info: dict[str, list[str]], local_scan: dict[str, Any], objects: list[ObjectRecord]) -> None:
    label_scenes = match_info["label_scenes"]
    local_products = match_info["local_products"]
    matched_products = match_info["matched_products"]
    rows = []
    for product in matched_products[:50]:
        count = sum(1 for obj in objects if obj.product_identifier == product)
        image_paths = [item.image_path for item in local_scan["images"] if item.product_identifier == product]
        rows.append([product, count, "; ".join(image_paths[:4])])
    lines = ["# SARFish Label to Local Image Match Report", ""]
    lines.append(f"- Label unique `scene_id` count: {len(label_scenes)}")
    lines.append(f"- Local possible scene/product id count: {len(local_products)}")
    lines.append(f"- Matched product identifier count: {len(matched_products)}")
    lines.append(f"- Label product identifiers without local image product: {len(match_info['label_products_without_local_image'])}")
    lines.append(f"- Local image products without labels: {len(match_info['local_products_without_label'])}")
    lines.append(f"- Existing `sarfishsample_probe.csv` found: {local_scan['probe_csv_exists']}")
    lines.append(f"- Existing `sarfishsample_stats.json` found: {local_scan['stats_json_exists']}")
    lines.append("")
    lines.append("## Matched Products")
    lines.extend(md_table(["product_identifier", "label_object_count", "local_image_paths"], rows or [["none", 0, ""]]))
    lines.append("")
    lines.append("## Failed Label Product Samples")
    for product in match_info["label_products_without_local_image"][:20]:
        lines.append(f"- `{product}`")
    lines.append("")
    lines.append("## Failed Label Scene Samples")
    failed_scene_samples = []
    local_set = set(local_products)
    for obj in objects:
        if obj.product_identifier not in local_set and obj.scene_id:
            failed_scene_samples.append(obj.scene_id)
    for scene_id in sorted(set(failed_scene_samples))[:20]:
        lines.append(f"- `{scene_id}`")
    lines.append("")
    lines.append("## Local Products Without Labels")
    for product in match_info["local_products_without_label"][:20]:
        lines.append(f"- `{product}`")
    lines.append("")
    lines.append("## ESA_xView3_sceneName_mapping.csv Need")
    lines.append("")
    if matched_products:
        lines.append("The found SARFish CSVs include `GRD_product_identifier` or `SLC_product_identifier`, and these identifiers match the local SAFE product identifiers. For these CSVs, `ESA_xView3_sceneName_mapping.csv` is not required for product-level matching.")
    else:
        lines.append("No product identifier match was found. If only xView3 `scene_id` is available, `ESA_xView3_sceneName_mapping.csv` is likely required.")
    lines.append("")
    lines.append("Direct `scene_id` to SAFE product name matching is not assumed; SARFish labels should match local imagery by product identifier where possible.")
    IMAGE_MATCH_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_task_profile(objects: list[ObjectRecord], bbox_stats: dict[str, Any], scene_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scene_target_counts = [float(row["target_count"]) for row in scene_rows]
    difficulty_values = [float(row["recommended_difficulty"]) for row in scene_rows]
    uncertainty_values = [float(row["recommended_uncertainty"]) for row in scene_rows]
    size_counts = Counter(obj.size_class for obj in objects)
    total_size = max(sum(size_counts.values()), 1)
    target_p50 = percentile(scene_target_counts, 0.50) or 1.0
    target_p75 = percentile(scene_target_counts, 0.75) or target_p50
    profile = {
        "target_count_distribution": describe(scene_target_counts),
        "bbox_size_distribution": {
            "width": bbox_stats["bbox_width"],
            "height": bbox_stats["bbox_height"],
            "area": bbox_stats["bbox_area"],
        },
        "vessel_length_distribution": bbox_stats["vessel_length_m"],
        "small_medium_large_vessel_ratio": {key: count / total_size for key, count in sorted(size_counts.items())},
        "recommended_difficulty_bins": {
            "easy": {"range": [0.05, 0.35], "description": "low target density, larger targets, low ambiguity"},
            "medium": {"range": [0.35, 0.65], "description": "moderate density or mixed target sizes"},
            "hard": {"range": [0.65, 1.0], "description": "dense scenes, small targets, ambiguous labels, or low confidence"},
            "scene_distribution": describe(difficulty_values),
            "target_count_reference": {"median": target_p50, "p75": target_p75},
        },
        "recommended_uncertainty_bins": {
            "low": {"range": [0.05, 0.30]},
            "medium": {"range": [0.30, 0.60]},
            "high": {"range": [0.60, 1.0]},
            "scene_distribution": describe(uncertainty_values),
        },
        "recommended_task_value_ranges": {
            "routine": [0.45, 0.60],
            "high_value": [0.72, 0.85],
            "emergency": [0.88, 0.98],
            "note": "Use scenario-level prior, not direct ground-truth target presence.",
        },
        "recommended_deadline_s_ranges": {
            "routine": [90, 120],
            "high_value": [60, 80],
            "emergency": [25, 45],
        },
        "recommended_quality_requirement_ranges": {
            "routine": [0.55, 0.70],
            "high_value": [0.72, 0.86],
            "emergency": [0.82, 0.95],
        },
        "recommended_information_requirement_ranges": {
            "routine": [0.45, 0.62],
            "high_value": [0.70, 0.86],
            "emergency": [0.85, 0.98],
        },
        "recommended_payload_size_estimates_mb": {
            "metadata": "0.10 + 0.02 * roi_count",
            "roi": "0.80 + 0.18 * roi_count",
            "roi_context": "1.60 + 0.35 * roi_count",
            "scene_roi_distribution": describe([float(row["estimated_roi_size_mb"]) for row in scene_rows]),
        },
    }
    return compact(profile)


def write_leakage_report() -> None:
    lines = [
        "# Task State Leakage Boundary",
        "",
        "## Online-Available Prior Features",
        "",
        "- Product type (`GRD` or `SLC`) if known from scheduled observation source.",
        "- Local tile size, stride, polarization, and product metadata that would be known before inference.",
        "- Mission source, observation window id, orbit/pass timing, contact scenario, and assigned scenario class.",
        "- Scenario-level priority, deadline setting, and high-value/emergency prior.",
        "",
        "## Offline Calibration Features",
        "",
        "- `target_count` distribution.",
        "- BBox width/height/area distribution.",
        "- Small-target ratio and target density distribution.",
        "- `vessel_length_m` distribution.",
        "- `is_vessel` / `is_fishing` aggregate distribution.",
        "- ROI size distribution and payload-size calibration.",
        "- Label confidence distribution.",
        "",
        "These features may calibrate priors and bins, but should not be injected as exact per-tile online state unless the experiment is explicitly marked as oracle.",
        "",
        "## Evaluation-Only Ground Truth",
        "",
        "- Whether the current tile actually contains a vessel.",
        "- True bbox coordinates.",
        "- True `is_fishing` and `is_vessel` values for a specific object/tile.",
        "- True target count for the current tile.",
        "- Realized detector miss/false alarm labels derived from ground truth.",
        "",
        "## Hard Rule",
        "",
        "Do not set online `task_value`, `high_value_flag`, or `emergency_flag` directly because the current label row contains a ship or fishing vessel. These must come from scenario-level or mission-level priors.",
    ]
    LEAKAGE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tasksample_mapping_report() -> None:
    lines = [
        "# TaskSample Mapping from SARFish Labels",
        "",
        "The current project should keep `TaskSample` unchanged. `SARFishSampleDatasetAdapter` should emit `list[TaskSample | None]`.",
        "",
        "## Field Mapping",
        "",
        "| TaskSample field | Mapping plan |",
        "| --- | --- |",
        "| `task_id` | Stable id from product id, scene id, tile id, and arrival slot. |",
        "| `image_path` | Local GRD measurement TIFF path for the first version. If tiling is logical, store source TIFF path plus tile bounds in metadata. |",
        "| `label_path` | Path to the source CSV, usually `GRD_validation.csv` for local matched GRD validation product. |",
        "| `scene_id` | CSV `scene_id`; also keep product identifier in metadata. |",
        "| `virtual_time` | Arrival slot index. |",
        "| `task_value` | Scenario prior range: routine, high_value, emergency. Do not derive directly from label target presence. |",
        "| `deadline_s` | Scenario prior: routine 90-120 s, high_value 60-80 s, emergency 25-45 s. |",
        "| `difficulty` | Calibrated from offline target count, small target ratio, bbox area and density distributions. |",
        "| `uncertainty` | Calibrated from small target ratio, low confidence ratio, dense targets, and ambiguous vessel/fishing labels. |",
        "| `metadata` | Store `target_count`, `bbox_area_mean`, `bbox_area_max`, `small_target_ratio`, `vessel_length_stats`, `is_vessel_count`, `is_fishing_count`, `estimated_roi_size_mb`, product id, product type, polarization, and leakage flags. |",
        "| `arrival_slot` | Slot assigned by observation-window scheduler. |",
        "| `observation_window_id` | Window id generated by SARFish adapter. |",
        "| `task_class` | Scenario prior: routine, high_value, emergency. |",
        "| `quality_requirement` | Function of task class, difficulty, and uncertainty. |",
        "| `information_requirement` | Function of task class, difficulty, and uncertainty. |",
        "| `high_value_flag` | Scenario prior only. |",
        "| `emergency_flag` | Scenario prior only. |",
        "",
        "## Difficulty Calibration",
        "",
        "`difficulty = clamp(0.20 + 0.30*density + 0.30*small_target_ratio + 0.20*inverse_bbox_area, 0.05, 1.0)`",
        "",
        "## Uncertainty Calibration",
        "",
        "`uncertainty = clamp(0.15 + 0.30*small_target_ratio + 0.25*ambiguous_label_ratio + 0.20*low_confidence_ratio + 0.10*density, 0.05, 1.0)`",
        "",
        "## Requirement Mapping",
        "",
        "- routine: moderate quality and information requirement.",
        "- high_value: higher quality/information requirement.",
        "- emergency: highest requirement and tightest deadline.",
        "",
        "## Payload Size",
        "",
        "Use current policy-compatible estimates first: metadata `0.10 + 0.02*roi_count`, ROI `0.80 + 0.18*roi_count`, ROI+context `1.60 + 0.35*roi_count`. A later detector-backed adapter can replace ROI count with detector ROI output.",
        "",
        "## First Version",
        "",
        "Use GRD only. SLC is useful later, but it is complex-valued and not a minimal fit for the current profile/mock detector path.",
    ]
    TASKSAMPLE_MAPPING_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_engineering_reports(summaries: list[CsvSummary], bbox_stats: dict[str, Any], match_info: dict[str, list[str]]) -> None:
    can_map = bbox_stats["total_objects_with_valid_bbox"] > 0 and any(item.csv_type == "SARFish GRD label" for item in summaries)
    can_match = bool(match_info["matched_products"])
    lines = [
        "# Engineering Alignment Report",
        "",
        f"- SARFish labels can map to current `TaskSample`: {can_map}",
        f"- Labels can match local imagery by product identifier: {can_match}",
        "",
        "## TaskSample Fields Calibrated by SARFish Profile",
        "",
        "- `difficulty`",
        "- `uncertainty`",
        "- `metadata` audit summaries",
        "- `quality_requirement`",
        "- `information_requirement`",
        "",
        "## Fields That Must Remain Scenario Priors",
        "",
        "- `task_value`",
        "- `task_class`",
        "- `deadline_s`",
        "- `high_value_flag`",
        "- `emergency_flag`",
        "- `arrival_slot` and `observation_window_id`",
        "",
        "## MockDatasetAdapter Defaults That SARFish Profile Can Replace",
        "",
        "- Synthetic `difficulty` and `uncertainty` distributions.",
        "- `synthetic_target_density` metadata can be replaced by calibrated `target_count` or density bins.",
        "- Class default quality/information requirements can be calibrated from SARFish difficulty distributions.",
        "- Mock image/label paths can be replaced by real GRD paths and CSV references.",
        "",
        "## Can SARFishSampleDatasetAdapter Start?",
        "",
        f"- Start implementation: {can_map and can_match}",
        "- First target: GRD validation product, because local GRD TIFFs and GRD labels are present.",
        "",
        "## Minimal Files for First Adapter",
        "",
        "- Add `src/data/sarfishsample_dataset.py`.",
        "- Add `configs/dataset_sarfishsample.yaml`.",
        "- Add a `TaskStreamBuilder` branch for `dataset_name: sarfishsample` only after adapter tests exist.",
        "- Add tests for CSV reading, product matching, and stream construction.",
        "",
        "## Files Not to Touch",
        "",
        "- Routers.",
        "- Mode controller.",
        "- Reward weights.",
        "- DDQN/training code.",
        "- Existing mock dataset behavior.",
        "",
        "## Fallback",
        "",
        "Keep mock dataset as the default fallback for smoke tests and algorithm regression.",
    ]
    ENGINEERING_ALIGNMENT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_lines = [
        "# Label Readiness Summary",
        "",
        f"- Found CSV count: {len(summaries)}",
        f"- GRD label CSV present: {any(item.csv_type == 'SARFish GRD label' for item in summaries)}",
        f"- SLC label CSV present: {any(item.csv_type == 'SARFish SLC label' for item in summaries)}",
        f"- Valid bbox objects: {bbox_stats['total_objects_with_valid_bbox']}",
        f"- `scene_id` available: {any('scene_id' in [normalize_col(col) for col in item.columns] for item in summaries)}",
        f"- vessel fields available: {any(all(field in [normalize_col(col) for col in item.columns] for field in ['is_vessel', 'is_fishing', 'vessel_length_m']) for item in summaries)}",
        f"- Local product match available: {can_match}",
        f"- Adapter can start: {can_map and can_match}",
        "",
        "Recommended next step: implement a GRD-only SARFishSampleDatasetAdapter using product identifier matching and scenario-prior task classes.",
    ]
    READINESS_SUMMARY_REPORT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SARFishSample label CSV audit.")
    parser.add_argument("--root", type=Path, required=True, help="SARFishSample root directory.")
    parser.add_argument("--labels", type=Path, default=None, help="Optional label directory or specific CSV.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pandas_status = "pandas unavailable; using csv standard library"
    try:
        import pandas as pd  # noqa: F401

        pandas_status = "pandas import ok; using csv standard library for stable streaming"
    except Exception as exc:  # noqa: BLE001
        pandas_status = f"pandas unavailable ({type(exc).__name__}: {exc}); using csv standard library"

    csv_paths = find_csv_files(root, args.labels.resolve() if args.labels else None)
    if not csv_paths:
        raise SystemExit("No SARFish label CSV files found. Place GRD_train.csv/GRD_validation.csv/SLC_train.csv/SLC_validation.csv under datasets/SARFishSample or pass --labels.")

    summaries = [summarize_csv(path) for path in csv_paths]
    objects, parse_meta = collect_objects(csv_paths)
    local_scan = scan_local_images(root)
    scene_rows, match_info = build_group_stats(objects, local_scan)
    bbox_stats = build_bbox_stats(objects, parse_meta)
    task_profile = build_task_profile(objects, bbox_stats, scene_rows)

    write_basic_report(summaries, pandas_status)
    write_schema_report(summaries)
    write_bbox_report(bbox_stats)
    write_match_report(match_info, local_scan, objects)
    write_leakage_report()
    write_tasksample_mapping_report()
    write_engineering_reports(summaries, bbox_stats, match_info)
    write_objects_probe(objects)
    write_scene_index(scene_rows)
    BBOX_STATS_JSON.write_text(json.dumps(bbox_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    TASK_STATE_PROFILE_JSON.write_text(json.dumps(task_profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Found CSV files: {len(csv_paths)}")
    for path in csv_paths:
        print(f"- {path}")
    print(f"Valid bbox objects: {bbox_stats['total_objects_with_valid_bbox']}")
    print(f"Unique scene_id: {bbox_stats['unique_scene_id_count']}")
    print(f"Matched local products: {len(match_info['matched_products'])}")
    print(f"Wrote {CSV_BASIC_REPORT}")
    print(f"Wrote {CSV_SCHEMA_REPORT}")
    print(f"Wrote {BBOX_STATS_REPORT}")
    print(f"Wrote {IMAGE_MATCH_REPORT}")
    print(f"Wrote {BBOX_STATS_JSON}")
    print(f"Wrote {OBJECTS_PROBE_CSV}")
    print(f"Wrote {TASK_STATE_PROFILE_JSON}")
    print(f"Wrote {SCENE_INDEX_CSV}")


if __name__ == "__main__":
    main()
