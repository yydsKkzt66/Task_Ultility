"""Read-only SARFishSample dataset inspection utility."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
except ImportError:  # pragma: no cover - the script can still count files.
    Image = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "sarfishsample_audit" / "dataset_structure_report.md"
DEFAULT_PROBE_PATH = PROJECT_ROOT / "data" / "indexes" / "sarfishsample_probe.csv"
DEFAULT_STATS_PATH = PROJECT_ROOT / "data" / "indexes" / "sarfishsample_stats.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ANNOTATION_EXTS = {".txt", ".xml", ".json", ".csv", ".geojson", ".kml"}
OTHER_LABEL_LIKE_EXTS = {".npy", ".msk"}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}


def fs_path(path: Path) -> str:
    """Return a filesystem path string, using Windows long-path syntax when needed."""

    absolute = str(path.absolute())
    if os.name != "nt":
        return absolute
    if absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def safe_file_size(path: Path) -> int | None:
    try:
        return os.stat(fs_path(path)).st_size
    except OSError:
        return None


@dataclass
class BBoxRecord:
    label_path: str
    source_format: str
    class_name: str = "unknown"
    class_id: str = ""
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    image_key: str = ""
    image_width: int | None = None
    image_height: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def width(self) -> float | None:
        if self.x1 is None or self.x2 is None:
            return None
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float | None:
        if self.y1 is None or self.y2 is None:
            return None
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float | None:
        width = self.width()
        height = self.height()
        if width is None or height is None:
            return None
        return width * height

    def area_ratio(self) -> float | None:
        area = self.area()
        if area is None or not self.image_width or not self.image_height:
            return None
        image_area = float(self.image_width) * float(self.image_height)
        if image_area <= 0:
            return None
        return area / image_area

    def long_side_ratio(self) -> float | None:
        width = self.width()
        height = self.height()
        if width is None or height is None or not self.image_width or not self.image_height:
            return None
        return max(width / max(float(self.image_width), 1.0), height / max(float(self.image_height), 1.0))


@dataclass
class ParsedLabel:
    path: Path
    detected_format: str
    bboxes: list[BBoxRecord] = field(default_factory=list)
    error: str = ""
    preview: str = ""
    empty: bool = False
    yolo_normalized: bool | None = None


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if safe_is_dir(entry):
                if entry.name in SKIP_DIRS:
                    continue
                stack.append(entry)
            else:
                files.append(entry)
    return sorted(files, key=lambda path: str(path).lower())


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def short_text(path: Path, limit: int = 500) -> str:
    try:
        with open(fs_path(path), "rb") as file:
            data = file.read(limit)
    except OSError as exc:
        return f"<unreadable: {exc}>"
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace")
    return " ".join(text.split())


def get_image_size(path: Path) -> tuple[int | None, int | None, str]:
    if Image is None:
        return None, None, "Pillow not available"
    try:
        with Image.open(fs_path(path)) as image:
            width, height = image.size
        return int(width), int(height), ""
    except Exception as exc:  # noqa: BLE001 - reporting bad images is the point.
        return None, None, str(exc)


def first_numbers(values: list[str]) -> list[float] | None:
    numbers: list[float] = []
    for value in values:
        cleaned = value.strip().strip("[]()")
        if not cleaned:
            continue
        try:
            numbers.append(float(cleaned))
        except ValueError:
            return None
    return numbers


def parse_yolo_txt(path: Path, image_size: tuple[int | None, int | None] | None = None) -> ParsedLabel | None:
    preview = short_text(path)
    try:
        with open(fs_path(path), "r", encoding="utf-8", errors="replace") as file:
            lines = [line.strip() for line in file.read().splitlines()]
    except OSError as exc:
        return ParsedLabel(path=path, detected_format="txt", error=str(exc), preview=preview)

    nonempty = [line for line in lines if line and not line.startswith("#")]
    if not nonempty:
        return ParsedLabel(path=path, detected_format="yolo_txt", preview=preview, empty=True, yolo_normalized=True)

    width = image_size[0] if image_size else None
    height = image_size[1] if image_size else None
    bboxes: list[BBoxRecord] = []
    for line in nonempty:
        parts = line.replace(",", " ").split()
        if len(parts) < 5:
            return None
        class_token = parts[0]
        try:
            int(float(class_token))
        except ValueError:
            return None
        values = first_numbers(parts[1:5])
        if values is None:
            return None
        x_center, y_center, box_width, box_height = values
        if not all(0.0 <= value <= 1.0 for value in values):
            return None
        if width and height:
            x1 = (x_center - box_width / 2.0) * width
            y1 = (y_center - box_height / 2.0) * height
            x2 = (x_center + box_width / 2.0) * width
            y2 = (y_center + box_height / 2.0) * height
        else:
            x1 = x_center - box_width / 2.0
            y1 = y_center - box_height / 2.0
            x2 = x_center + box_width / 2.0
            y2 = y_center + box_height / 2.0
        bboxes.append(
            BBoxRecord(
                label_path=str(path),
                source_format="yolo_txt",
                class_id=class_token,
                class_name=class_token,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_width=width,
                image_height=height,
            )
        )
    return ParsedLabel(path=path, detected_format="yolo_txt", bboxes=bboxes, preview=preview, yolo_normalized=True)


def parse_custom_txt(path: Path) -> ParsedLabel:
    preview = short_text(path)
    try:
        with open(fs_path(path), "r", encoding="utf-8", errors="replace") as file:
            lines = [line.strip() for line in file.read().splitlines()]
    except OSError as exc:
        return ParsedLabel(path=path, detected_format="txt", error=str(exc), preview=preview)

    bboxes: list[BBoxRecord] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        numeric_positions: list[tuple[int, float]] = []
        for index, part in enumerate(parts):
            try:
                numeric_positions.append((index, float(part.strip("[]()"))))
            except ValueError:
                continue
        if len(numeric_positions) < 4:
            continue
        values = [item[1] for item in numeric_positions[:4]]
        x1, y1, x2, y2 = values
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        class_name = "unknown"
        if numeric_positions[0][0] > 0:
            class_name = parts[0]
        bboxes.append(
            BBoxRecord(
                label_path=str(path),
                source_format="custom_txt",
                class_name=class_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return ParsedLabel(path=path, detected_format="custom_txt" if bboxes else "unrecognized_txt", bboxes=bboxes, preview=preview, empty=not bboxes)


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1].lower()


def child_text(node: ET.Element, name: str) -> str | None:
    for child in list(node):
        if local_name(child.tag) == name.lower():
            return child.text
    return None


def parse_voc_xml(path: Path) -> ParsedLabel | None:
    preview = short_text(path)
    try:
        tree = ET.parse(fs_path(path))
    except Exception as exc:  # noqa: BLE001
        return ParsedLabel(path=path, detected_format="xml", error=str(exc), preview=preview)

    root = tree.getroot()
    objects = [node for node in root.iter() if local_name(node.tag) == "object"]
    if not objects:
        return None

    image_width = None
    image_height = None
    for size_node in root.iter():
        if local_name(size_node.tag) == "size":
            width_text = child_text(size_node, "width")
            height_text = child_text(size_node, "height")
            try:
                image_width = int(float(width_text)) if width_text else None
                image_height = int(float(height_text)) if height_text else None
            except ValueError:
                pass
            break

    bboxes: list[BBoxRecord] = []
    for object_node in objects:
        class_name = child_text(object_node, "name") or "unknown"
        bndbox = None
        for child in object_node.iter():
            if local_name(child.tag) == "bndbox":
                bndbox = child
                break
        if bndbox is None:
            continue
        values = {}
        for key in ("xmin", "ymin", "xmax", "ymax"):
            text = child_text(bndbox, key)
            if text is None:
                values = {}
                break
            try:
                values[key] = float(text)
            except ValueError:
                values = {}
                break
        if values:
            bboxes.append(
                BBoxRecord(
                    label_path=str(path),
                    source_format="voc_xml",
                    class_name=class_name,
                    x1=values["xmin"],
                    y1=values["ymin"],
                    x2=values["xmax"],
                    y2=values["ymax"],
                    image_width=image_width,
                    image_height=image_height,
                )
            )
    return ParsedLabel(path=path, detected_format="voc_xml", bboxes=bboxes, preview=preview, empty=not bboxes)


def parse_generic_xml(path: Path) -> ParsedLabel:
    preview = short_text(path)
    parsed = parse_voc_xml(path)
    if parsed is not None:
        return parsed
    try:
        tree = ET.parse(fs_path(path))
    except Exception as exc:  # noqa: BLE001
        return ParsedLabel(path=path, detected_format="xml", error=str(exc), preview=preview)

    root = tree.getroot()
    bboxes: list[BBoxRecord] = []
    for node in root.iter():
        values: dict[str, float] = {}
        for key in ("top", "left", "bottom", "right"):
            text = child_text(node, key)
            if text is None:
                values = {}
                break
            try:
                values[key] = float(text)
            except ValueError:
                values = {}
                break
        if values:
            bboxes.append(
                BBoxRecord(
                    label_path=str(path),
                    source_format="custom_xml",
                    class_name=child_text(node, "class") or child_text(node, "name") or "unknown",
                    x1=values["left"],
                    y1=values["top"],
                    x2=values["right"],
                    y2=values["bottom"],
                )
            )
    return ParsedLabel(path=path, detected_format="custom_xml" if bboxes else "unrecognized_xml", bboxes=bboxes, preview=preview, empty=not bboxes)


def parse_json_label(path: Path) -> ParsedLabel:
    preview = short_text(path)
    try:
        with open(fs_path(path), "r", encoding="utf-8", errors="replace") as file:
            data = json.loads(file.read())
    except Exception as exc:  # noqa: BLE001
        return ParsedLabel(path=path, detected_format="json", error=str(exc), preview=preview)

    if isinstance(data, dict) and {"images", "annotations"}.issubset(data):
        images_by_id: dict[Any, dict[str, Any]] = {}
        for image in data.get("images", []):
            images_by_id[image.get("id")] = image
        category_by_id = {
            category.get("id"): str(category.get("name", category.get("id")))
            for category in data.get("categories", [])
            if isinstance(category, dict)
        }
        bboxes: list[BBoxRecord] = []
        for annotation in data.get("annotations", []):
            if not isinstance(annotation, dict) or "bbox" not in annotation:
                continue
            bbox = annotation.get("bbox") or []
            if len(bbox) < 4:
                continue
            try:
                x, y, width, height = [float(value) for value in bbox[:4]]
            except (TypeError, ValueError):
                continue
            image = images_by_id.get(annotation.get("image_id"), {})
            image_key = str(image.get("file_name", annotation.get("image_id", "")))
            bboxes.append(
                BBoxRecord(
                    label_path=str(path),
                    source_format="coco_json",
                    class_id=str(annotation.get("category_id", "")),
                    class_name=category_by_id.get(annotation.get("category_id"), str(annotation.get("category_id", "unknown"))),
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                    image_key=image_key,
                    image_width=int(image["width"]) if image.get("width") is not None else None,
                    image_height=int(image["height"]) if image.get("height") is not None else None,
                )
            )
        return ParsedLabel(path=path, detected_format="coco_json", bboxes=bboxes, preview=preview, empty=not bboxes)

    bboxes = []
    records = data if isinstance(data, list) else data.get("annotations", []) if isinstance(data, dict) else []
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            coords = None
            if "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) >= 4:
                try:
                    x, y, width, height = [float(value) for value in item["bbox"][:4]]
                    coords = (x, y, x + width, y + height)
                except (TypeError, ValueError):
                    coords = None
            elif all(key in item for key in ("left", "top", "right", "bottom")):
                try:
                    coords = (float(item["left"]), float(item["top"]), float(item["right"]), float(item["bottom"]))
                except (TypeError, ValueError):
                    coords = None
            if coords is None:
                continue
            bboxes.append(
                BBoxRecord(
                    label_path=str(path),
                    source_format="custom_json",
                    class_name=str(item.get("class", item.get("category", item.get("category_id", "unknown")))),
                    x1=coords[0],
                    y1=coords[1],
                    x2=coords[2],
                    y2=coords[3],
                    image_key=str(item.get("file_name", item.get("image", ""))),
                )
            )
    return ParsedLabel(path=path, detected_format="custom_json" if bboxes else "unrecognized_json", bboxes=bboxes, preview=preview, empty=not bboxes)


def parse_csv_label(path: Path) -> ParsedLabel:
    preview = short_text(path)
    try:
        with open(fs_path(path), "r", encoding="utf-8-sig", errors="replace", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = [field.strip() for field in (reader.fieldnames or [])]
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        return ParsedLabel(path=path, detected_format="csv", error=str(exc), preview=preview)

    if not fieldnames:
        return ParsedLabel(path=path, detected_format="empty_csv", preview=preview, empty=True)

    lower_map = {field.lower(): field for field in fieldnames}
    bboxes: list[BBoxRecord] = []
    has_sarfish_bbox = all(key in lower_map for key in ("top", "left", "bottom", "right"))
    has_xywh = all(key in lower_map for key in ("x", "y", "width", "height"))
    class_field = next(
        (lower_map[key] for key in ("class", "class_id", "category", "category_id", "is_fishing", "is_vessel") if key in lower_map),
        None,
    )
    image_field = next(
        (lower_map[key] for key in ("file_name", "image", "image_path", "filename", "scene_id") if key in lower_map),
        None,
    )
    width_field = next((lower_map[key] for key in ("width", "image_width") if key in lower_map), None)
    height_field = next((lower_map[key] for key in ("height", "image_height") if key in lower_map), None)

    for row in rows:
        try:
            if has_sarfish_bbox:
                x1 = float(row[lower_map["left"]])
                y1 = float(row[lower_map["top"]])
                x2 = float(row[lower_map["right"]])
                y2 = float(row[lower_map["bottom"]])
                source_format = "sarfish_csv"
            elif has_xywh:
                x = float(row[lower_map["x"]])
                y = float(row[lower_map["y"]])
                width = float(row[lower_map["width"]])
                height = float(row[lower_map["height"]])
                x1, y1, x2, y2 = x, y, x + width, y + height
                source_format = "custom_csv"
            else:
                continue
        except (TypeError, ValueError, KeyError):
            continue
        class_name = str(row.get(class_field, "unknown")) if class_field else "unknown"
        image_key = str(row.get(image_field, "")) if image_field else ""
        image_width = None
        image_height = None
        try:
            image_width = int(float(row[width_field])) if width_field and row.get(width_field) else None
            image_height = int(float(row[height_field])) if height_field and row.get(height_field) else None
        except ValueError:
            pass
        bboxes.append(
            BBoxRecord(
                label_path=str(path),
                source_format=source_format,
                class_name=class_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                image_key=image_key,
                image_width=image_width,
                image_height=image_height,
                extra={field: row.get(field, "") for field in ("is_vessel", "is_fishing", "confidence", "vessel_length_m") if field in row},
            )
        )

    detected = "sarfish_csv" if has_sarfish_bbox and bboxes else "custom_csv" if bboxes else "unrecognized_csv"
    return ParsedLabel(path=path, detected_format=detected, bboxes=bboxes, preview=preview, empty=not bboxes)


def parse_label(path: Path, image_size: tuple[int | None, int | None] | None = None) -> ParsedLabel:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yolo = parse_yolo_txt(path, image_size=image_size)
        if yolo is not None:
            return yolo
        return parse_custom_txt(path)
    if suffix == ".xml":
        return parse_generic_xml(path)
    if suffix in {".json", ".geojson"}:
        return parse_json_label(path)
    if suffix == ".csv":
        return parse_csv_label(path)
    return ParsedLabel(path=path, detected_format=f"unsupported_{suffix.lstrip('.')}", preview=short_text(path), empty=True)


def build_dir_tree(root: Path, max_depth: int = 3) -> list[str]:
    lines: list[str] = [f"{root.name}/"]

    def visit(directory: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        try:
            entries = sorted([entry for entry in directory.iterdir() if entry.name not in SKIP_DIRS], key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError:
            return
        for index, entry in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            suffix = "/" if entry.is_dir() else ""
            if safe_is_dir(entry):
                try:
                    file_count = sum(1 for child in entry.iterdir() if not safe_is_dir(child))
                    dir_count = sum(1 for child in entry.iterdir() if safe_is_dir(child) and child.name not in SKIP_DIRS)
                    summary = f" ({dir_count} dirs, {file_count} files)"
                except OSError:
                    summary = ""
            else:
                size = safe_file_size(entry)
                summary = f" ({size} bytes)" if size is not None else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}{summary}")
            if safe_is_dir(entry):
                child_prefix = prefix + ("    " if index == len(entries) - 1 else "|   ")
                visit(entry, child_prefix, depth + 1)

    visit(root, "", 0)
    return lines


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "min": None, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": percentile(clean, 0.90),
        "max": max(clean),
    }


def classify_size(bbox: BBoxRecord) -> str:
    area_ratio = bbox.area_ratio()
    long_side_ratio = bbox.long_side_ratio()
    if area_ratio is None and long_side_ratio is None:
        return "unknown"
    area_ratio = area_ratio if area_ratio is not None else 0.0
    long_side_ratio = long_side_ratio if long_side_ratio is not None else 0.0
    if area_ratio < 0.001 or long_side_ratio < 0.03:
        return "small"
    if area_ratio < 0.01 or long_side_ratio < 0.10:
        return "medium"
    return "large"


def estimate_roi_crop(bbox: BBoxRecord, context_scale: float = 0.20) -> tuple[float, float, float] | None:
    width = bbox.width()
    height = bbox.height()
    if width is None or height is None:
        return None
    crop_width = width * (1.0 + 2.0 * context_scale)
    crop_height = height * (1.0 + 2.0 * context_scale)
    if bbox.image_width:
        crop_width = min(crop_width, float(bbox.image_width))
    if bbox.image_height:
        crop_height = min(crop_height, float(bbox.image_height))
    area = crop_width * crop_height
    return crop_width, crop_height, area


def match_label_for_image(image_path: Path, labels_by_stem: dict[str, list[Path]]) -> Path | None:
    candidates = labels_by_stem.get(image_path.stem, [])
    if candidates:
        preferred = sorted(candidates, key=lambda item: (item.suffix.lower() not in {".txt", ".xml", ".json", ".csv"}, len(str(item))))
        return preferred[0]
    return None


def write_probe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "random_sample",
        "image_path",
        "label_path",
        "image_width",
        "image_height",
        "image_error",
        "label_format",
        "parsed_bbox_count",
        "label_preview",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[tuple[Any, Any]], headers: tuple[str, str]) -> list[str]:
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | ---: |"]
    for key, value in rows:
        lines.append(f"| `{key}` | {value} |")
    return lines


def write_report(
    path: Path,
    root: Path,
    stats: dict[str, Any],
    tree_lines: list[str],
    sample_rows: list[dict[str, Any]],
    parsed_labels: list[ParsedLabel],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# SARFishSample Dataset Structure Report")
    lines.append("")
    lines.append(f"- Dataset root: `{root}`")
    lines.append(f"- Total scanned files: {stats['total_files']}")
    lines.append(f"- Image files: {stats['image_count']}")
    lines.append(f"- Annotation-like files: {stats['annotation_count']}")
    lines.append(f"- Label format verdict: {stats['label_format_verdict']}")
    lines.append("")

    lines.append("## Directory Tree Summary")
    lines.append("")
    lines.append("```text")
    lines.extend(tree_lines)
    lines.append("```")
    lines.append("")

    lines.append("## File Type Statistics")
    lines.append("")
    lines.append("### Image Extensions")
    lines.extend(markdown_table(sorted(stats["image_extension_counts"].items()), ("Extension", "Count")))
    lines.append("")
    lines.append("### Annotation Extensions")
    lines.extend(markdown_table(sorted(stats["annotation_extension_counts"].items()), ("Extension", "Count")))
    lines.append("")
    lines.append("### Other Label-like Extensions")
    lines.extend(markdown_table(sorted(stats["other_label_like_extension_counts"].items()), ("Extension", "Count")))
    lines.append("")

    lines.append("## Image and Label Matching")
    lines.append("")
    lines.append(f"- Images with same-stem annotation file: {stats['same_stem_image_matches']}")
    lines.append(f"- Images without same-stem annotation file: {stats['images_without_same_stem_label']}")
    lines.append(f"- Same-stem annotations without image: {stats['labels_without_same_stem_image']}")
    lines.append(f"- One-to-one image/label match: {stats['one_to_one_match']}")
    lines.append(f"- Bad/unreadable images: {stats['bad_image_count']}")
    lines.append(f"- Bad/unreadable labels: {stats['bad_label_count']}")
    lines.append(f"- Empty parsed annotation files: {stats['empty_label_count']}")
    lines.append("")

    lines.append("## Random Sample Probe")
    lines.append("")
    lines.append("| image_path | label_path | image_size | label_format | parsed_bbox_count | label_preview |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for row in sample_rows:
        size = f"{row.get('image_width') or ''}x{row.get('image_height') or ''}"
        preview = str(row.get("label_preview", "")).replace("|", "\\|")[:220]
        lines.append(
            f"| `{row['image_path']}` | `{row.get('label_path') or ''}` | {size} | "
            f"{row.get('label_format') or ''} | {row.get('parsed_bbox_count') or 0} | {preview} |"
        )
    lines.append("")

    lines.append("## Annotation Format Detection")
    lines.append("")
    lines.extend(markdown_table(sorted(stats["detected_label_format_counts"].items()), ("Detected format", "Count")))
    lines.append("")
    lines.append(f"- YOLO normalized coordinate check: {stats['yolo_normalized_verdict']}")
    lines.append("- Recognized SARFish CSV requires columns `top,left,bottom,right`; if absent, no SARFish bbox table was found.")
    lines.append("")

    lines.append("## Target and Bounding Box Statistics")
    lines.append("")
    lines.append(f"- Parsed bounding boxes: {stats['parsed_bbox_count']}")
    lines.append(f"- Positive images with parsed boxes: {stats['positive_image_count']}")
    lines.append(f"- Negative images with explicit empty same-stem labels: {stats['explicit_negative_image_count']}")
    lines.append(f"- Images with no explicit label: {stats['images_without_same_stem_label']}")
    lines.append("")
    lines.append("### Objects Per Image")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stats["objects_per_image"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Class Distribution")
    lines.extend(markdown_table(sorted(stats["class_distribution"].items()), ("Class", "Count")))
    lines.append("")
    lines.append("### BBox Width Distribution")
    lines.append("```json")
    lines.append(json.dumps(stats["bbox_width"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### BBox Height Distribution")
    lines.append("```json")
    lines.append(json.dumps(stats["bbox_height"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### BBox Area Distribution")
    lines.append("```json")
    lines.append(json.dumps(stats["bbox_area"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### BBox Area Ratio Distribution")
    lines.append("```json")
    lines.append(json.dumps(stats["bbox_area_ratio"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Small / Medium / Large Split")
    lines.extend(markdown_table(sorted(stats["size_class_distribution"].items()), ("Size class", "Count")))
    lines.append("")
    lines.append("### Estimated ROI Crop Size Distribution")
    lines.append("```json")
    lines.append(json.dumps(stats["roi_crop"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Data Quality Findings")
    lines.append("")
    lines.append(f"- Empty annotations: {stats['empty_label_count']}")
    lines.append(f"- Bad images: {stats['bad_image_count']}")
    lines.append(f"- Bad labels: {stats['bad_label_count']}")
    lines.append(f"- Unmatched images: {stats['images_without_same_stem_label']}")
    lines.append(f"- Labels without same-stem image: {stats['labels_without_same_stem_image']}")
    lines.append("")
    if stats["bad_image_examples"]:
        lines.append("### Bad Image Examples")
        for item in stats["bad_image_examples"][:10]:
            lines.append(f"- `{item['path']}`: {item['error']}")
        lines.append("")
    if stats["bad_label_examples"]:
        lines.append("### Bad Label Examples")
        for item in stats["bad_label_examples"][:10]:
            lines.append(f"- `{item['path']}`: {item['error']}")
        lines.append("")
    if stats["unrecognized_label_examples"]:
        lines.append("### Unrecognized Label Previews")
        for item in stats["unrecognized_label_examples"][:10]:
            lines.append(f"- `{item['path']}`: {item['preview'][:260]}")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- This script performs read-only inspection. It does not move, copy, or alter dataset files.")
    lines.append("- `.npy` and `.msk` are counted as label-like/mask assets, but they are not parsed as detection annotations by default.")
    lines.append("- For SARFishSample, product metadata XML files are usually not VOC object annotations; parsed bbox count can therefore be zero even when SAR products are present.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect(root: Path) -> dict[str, Any]:
    files = iter_files(root)
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTS]
    annotation_files = [path for path in files if path.suffix.lower() in ANNOTATION_EXTS]
    other_label_like_files = [path for path in files if path.suffix.lower() in OTHER_LABEL_LIKE_EXTS]

    image_extension_counts = Counter(path.suffix.lower() or "<none>" for path in image_files)
    annotation_extension_counts = Counter(path.suffix.lower() or "<none>" for path in annotation_files)
    other_label_like_extension_counts = Counter(path.suffix.lower() or "<none>" for path in other_label_like_files)

    image_sizes: dict[Path, tuple[int | None, int | None]] = {}
    bad_images: list[dict[str, str]] = []
    for image_path in image_files:
        width, height, error = get_image_size(image_path)
        image_sizes[image_path] = (width, height)
        if error:
            bad_images.append({"path": rel(image_path, root), "error": error})

    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    for label_path in annotation_files:
        labels_by_stem[label_path.stem].append(label_path)

    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in image_files:
        images_by_stem[image_path.stem].append(image_path)

    matched_label_for_image = {image_path: match_label_for_image(image_path, labels_by_stem) for image_path in image_files}
    same_stem_image_matches = sum(1 for value in matched_label_for_image.values() if value is not None)
    matched_label_paths = {value for value in matched_label_for_image.values() if value is not None}
    labels_without_same_stem_image = [path for path in annotation_files if path.stem not in images_by_stem]

    parsed_labels: list[ParsedLabel] = []
    for label_path in annotation_files:
        image_match = None
        if label_path in matched_label_paths:
            for image_path, candidate in matched_label_for_image.items():
                if candidate == label_path:
                    image_match = image_path
                    break
        parsed_labels.append(parse_label(label_path, image_size=image_sizes.get(image_match) if image_match else None))

    parsed_by_path = {item.path: item for item in parsed_labels}
    all_bboxes: list[BBoxRecord] = []
    for parsed in parsed_labels:
        for bbox in parsed.bboxes:
            if bbox.image_width is None or bbox.image_height is None:
                matching_image = None
                if bbox.image_key:
                    image_key = Path(bbox.image_key).name
                    for image_path in image_files:
                        if image_path.name == image_key or image_path.stem == Path(image_key).stem:
                            matching_image = image_path
                            break
                if matching_image is None and parsed.path in matched_label_paths:
                    for image_path, label_path in matched_label_for_image.items():
                        if label_path == parsed.path:
                            matching_image = image_path
                            break
                if matching_image is not None:
                    bbox.image_width, bbox.image_height = image_sizes.get(matching_image, (None, None))
                    bbox.image_key = rel(matching_image, root)
            all_bboxes.append(bbox)

    bboxes_by_image: dict[Path, list[BBoxRecord]] = defaultdict(list)
    for image_path, label_path in matched_label_for_image.items():
        if label_path is None:
            continue
        for bbox in parsed_by_path.get(label_path, ParsedLabel(label_path, "missing")).bboxes:
            bboxes_by_image[image_path].append(bbox)
    for bbox in all_bboxes:
        if not bbox.image_key:
            continue
        for image_path in image_files:
            if rel(image_path, root) == bbox.image_key or image_path.name == Path(bbox.image_key).name:
                bboxes_by_image[image_path].append(bbox)

    object_counts = [len(bboxes_by_image.get(image_path, [])) for image_path in image_files if image_path in matched_label_for_image]
    positive_images = [path for path, boxes in bboxes_by_image.items() if boxes]
    explicit_negative_images = [
        image_path
        for image_path, label_path in matched_label_for_image.items()
        if label_path is not None and parsed_by_path.get(label_path) is not None and parsed_by_path[label_path].empty
    ]

    class_distribution = Counter((bbox.class_name or bbox.class_id or "unknown") for bbox in all_bboxes)
    bbox_widths = [bbox.width() for bbox in all_bboxes if bbox.width() is not None]
    bbox_heights = [bbox.height() for bbox in all_bboxes if bbox.height() is not None]
    bbox_areas = [bbox.area() for bbox in all_bboxes if bbox.area() is not None]
    bbox_area_ratios = [bbox.area_ratio() for bbox in all_bboxes if bbox.area_ratio() is not None]
    size_class_distribution = Counter(classify_size(bbox) for bbox in all_bboxes)
    roi_crops = [estimate_roi_crop(bbox) for bbox in all_bboxes]
    roi_widths = [item[0] for item in roi_crops if item is not None]
    roi_heights = [item[1] for item in roi_crops if item is not None]
    roi_areas = [item[2] for item in roi_crops if item is not None]

    detected_label_format_counts = Counter(parsed.detected_format for parsed in parsed_labels)
    yolo_values = [parsed.yolo_normalized for parsed in parsed_labels if parsed.detected_format == "yolo_txt"]
    if not yolo_values:
        yolo_verdict = "No YOLO txt labels detected."
    elif all(value is True for value in yolo_values):
        yolo_verdict = "Detected YOLO labels use normalized class x_center y_center width height format."
    else:
        yolo_verdict = "YOLO-like txt labels were detected, but normalized coordinate format is not consistent."

    if any(fmt in detected_label_format_counts for fmt in ("sarfish_csv",)):
        label_format_verdict = "SARFish/xView3-style CSV bbox labels detected."
    elif detected_label_format_counts.get("voc_xml", 0):
        label_format_verdict = "VOC XML bbox labels detected."
    elif detected_label_format_counts.get("coco_json", 0):
        label_format_verdict = "COCO JSON bbox labels detected."
    elif detected_label_format_counts.get("yolo_txt", 0):
        label_format_verdict = "YOLO txt bbox labels detected."
    elif all_bboxes:
        label_format_verdict = "Custom bbox labels detected."
    else:
        label_format_verdict = "No parseable detection bbox labels found in scanned files."

    bad_labels = [{"path": rel(item.path, root), "error": item.error} for item in parsed_labels if item.error]
    unrecognized_label_examples = [
        {"path": rel(item.path, root), "preview": item.preview}
        for item in parsed_labels
        if item.detected_format.startswith("unrecognized")
    ]

    random_source = random.Random(2026)
    sample_images = random_source.sample(image_files, min(10, len(image_files))) if image_files else []
    sample_set = set(sample_images)
    probe_rows: list[dict[str, Any]] = []
    for image_path in image_files:
        label_path = matched_label_for_image.get(image_path)
        parsed_label = parsed_by_path.get(label_path) if label_path else None
        width, height = image_sizes.get(image_path, (None, None))
        image_error = next((item["error"] for item in bad_images if item["path"] == rel(image_path, root)), "")
        probe_rows.append(
            {
                "random_sample": int(image_path in sample_set),
                "image_path": rel(image_path, root),
                "label_path": rel(label_path, root) if label_path else "",
                "image_width": width or "",
                "image_height": height or "",
                "image_error": image_error,
                "label_format": parsed_label.detected_format if parsed_label else "",
                "parsed_bbox_count": len(parsed_label.bboxes) if parsed_label else 0,
                "label_preview": parsed_label.preview if parsed_label else "",
            }
        )

    sample_rows = [row for row in probe_rows if int(row["random_sample"]) == 1]
    sample_rows = sorted(sample_rows, key=lambda row: row["image_path"])

    stats: dict[str, Any] = {
        "dataset_root": str(root),
        "total_files": len(files),
        "image_count": len(image_files),
        "annotation_count": len(annotation_files),
        "other_label_like_count": len(other_label_like_files),
        "image_extension_counts": dict(image_extension_counts),
        "annotation_extension_counts": dict(annotation_extension_counts),
        "other_label_like_extension_counts": dict(other_label_like_extension_counts),
        "same_stem_image_matches": same_stem_image_matches,
        "images_without_same_stem_label": len(image_files) - same_stem_image_matches,
        "labels_without_same_stem_image": len(labels_without_same_stem_image),
        "one_to_one_match": same_stem_image_matches == len(image_files) == len(annotation_files),
        "bad_image_count": len(bad_images),
        "bad_image_examples": bad_images[:20],
        "bad_label_count": len(bad_labels),
        "bad_label_examples": bad_labels[:20],
        "empty_label_count": sum(1 for item in parsed_labels if item.empty),
        "detected_label_format_counts": dict(detected_label_format_counts),
        "label_format_verdict": label_format_verdict,
        "yolo_normalized_verdict": yolo_verdict,
        "parsed_bbox_count": len(all_bboxes),
        "positive_image_count": len(set(positive_images)),
        "explicit_negative_image_count": len(set(explicit_negative_images)),
        "objects_per_image": describe([float(value) for value in object_counts]),
        "class_distribution": dict(class_distribution),
        "bbox_width": describe([float(value) for value in bbox_widths]),
        "bbox_height": describe([float(value) for value in bbox_heights]),
        "bbox_area": describe([float(value) for value in bbox_areas]),
        "bbox_area_ratio": describe([float(value) for value in bbox_area_ratios]),
        "size_class_distribution": dict(size_class_distribution),
        "roi_crop": {
            "width": describe([float(value) for value in roi_widths]),
            "height": describe([float(value) for value in roi_heights]),
            "area": describe([float(value) for value in roi_areas]),
        },
        "unrecognized_label_examples": unrecognized_label_examples[:20],
        "sample_rows": sample_rows,
    }
    return {
        "stats": stats,
        "probe_rows": probe_rows,
        "sample_rows": sample_rows,
        "tree_lines": build_dir_tree(root, max_depth=3),
        "parsed_labels": parsed_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SARFishSample dataset audit.")
    parser.add_argument("--root", type=Path, required=True, help="Path to SARFishSample dataset root.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--probe-csv", type=Path, default=DEFAULT_PROBE_PATH)
    parser.add_argument("--stats-json", type=Path, default=DEFAULT_STATS_PATH)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"SARFishSample root does not exist or is not a directory: {root}")

    result = inspect(root)
    stats = result["stats"]
    write_probe_csv(args.probe_csv, result["probe_rows"])
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, root, stats, result["tree_lines"], result["sample_rows"], result["parsed_labels"])

    print(f"Wrote {args.report}")
    print(f"Wrote {args.probe_csv}")
    print(f"Wrote {args.stats_json}")
    print(f"Images: {stats['image_count']}; annotations: {stats['annotation_count']}; parsed bboxes: {stats['parsed_bbox_count']}")
    print(f"Verdict: {stats['label_format_verdict']}")


if __name__ == "__main__":
    main()
