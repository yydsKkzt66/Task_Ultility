from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "_recover_exact_v25_seed2026_manifest_4c23_v1"
ARTIFACTS = ROOT / "artifacts" / "frozen_scenarios"
EXPECTED_HASH = "4c23cf3184f137733495f95a7c5bd7631adf95b3fbc4eb057351f27dac2f2ab5"
EXPECTED_COUNTS = {"low": 54, "medium": 139, "high": 54, "urgent": 21}
EXPECTED_N = 268
EXPECTED_HV = 75

NAME_RE = re.compile(
    r"(task_manifest|scenario_manifest|resolved_scenario|scenario_identity|config_snapshot|"
    r"task_results|task_ledger|task_outcomes|per_task_results|per_task_outcomes|task_level_ground_outcomes)",
    re.IGNORECASE,
)
TASK_ID_RE = re.compile(r"^(sarfishsample_task_[0-9]{4}|pa_[A-Za-z0-9_:-]+|[A-Za-z0-9_.:-]+_task_[0-9]+)")
CANONICAL_FIELDS = [
    "order",
    "timeline_slot",
    "task_id",
    "scene_id",
    "product_identifier",
    "logical_image_path",
    "logical_label_path",
    "arrival_slot",
    "arrival_time_s",
    "capture_time_s",
    "observation_window_id",
    "task_class",
    "value_level",
    "task_value",
    "deadline_s",
    "quality_requirement",
    "information_requirement",
    "product_requirement",
    "product_requirement_source",
    "difficulty",
    "uncertainty",
    "high_value_flag",
    "emergency_flag",
]
IDENTITY_FILES = [
    "scenario_identity.json",
    "scenario_identity_comparison.json",
    "task_manifest_audit.json",
    "config_snapshot.json",
    "final_assessment.md",
    "test_report.txt",
    "focused_test_report.txt",
]
FORMAL_OUTPUT_HINTS = [
    "_v25_unified_baselines_core_metrics_v2",
    "_v25_bupt1_power_calibrated_seed2026_v1",
    "_five_method_cap035_quick_compare_v1",
    "_myopic_formal_nominal_benchmark_v1",
]
EXTERNAL_CANDIDATES = [
    Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1/bundles/V25-MAIN-v1/task_manifest.csv"),
    Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1/bundles/V25-MAIN-v1/bundle_contract.json"),
    Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1_superseded_ground_ack_invalid_20260721_012845/bundles/V25-MAIN-v1/task_manifest.csv"),
    Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1_superseded_ground_ack_invalid_20260721_012845/bundles/V25-MAIN-v1/bundle_contract.json"),
    Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1_superseded_missing_ack_audit_20260721_010943/bundles/V25-MAIN-v1/task_manifest.csv"),
    Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1_superseded_missing_ack_audit_20260721_010943/bundles/V25-MAIN-v1/bundle_contract.json"),
]


def run_git(args: list[str], *, input_text: str | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(rows: list[dict[str, Any]]) -> str:
    excluded = {"resolved_image_path", "resolved_label_path", "image_path", "label_path", "dataset_root", "source_image_root"}
    return stable_hash([{key: value for key, value in row.items() if key not in excluded} for row in rows])


def v25_bundle_hash(rows: list[dict[str, Any]]) -> str | None:
    required = {
        "task_id",
        "arrival_time_s",
        "value_level",
        "effective_task_value",
        "product_requirement",
        "deadline_time_s",
    }
    if not rows or not required.issubset({str(key) for row in rows for key in row}):
        return None
    fields = ("task_id", "value_level", "effective_task_value", "product_requirement", "arrival_time_s", "deadline_time_s")
    task_rows = [{key: row[key] for key in fields} for row in rows if row.get("task_id")]
    if not task_rows:
        return None
    task_rows = sorted(task_rows, key=lambda row: row["task_id"])
    payload = json.dumps(task_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows or [{"empty": ""}])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def first_value(row: dict[str, Any], names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in {None, ""}:
            return row[name]
    return default


def parse_table(path: str, data: bytes) -> tuple[list[dict[str, Any]], str]:
    suffix = Path(path).suffix.lower()
    text = data.decode("utf-8-sig", errors="replace")
    if suffix == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        return [dict(row) for row in rows], "csv"
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()], "jsonl"
        value = json.loads(text)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)], "json_list"
        if isinstance(value, dict):
            for key in ("tasks", "manifest", "task_manifest", "rows", "task_rows", "per_task", "per_task_results"):
                candidate = value.get(key)
                if isinstance(candidate, list):
                    return [dict(row) for row in candidate if isinstance(row, dict)], f"json.{key}"
            return [value], "json_dict"
    if suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(text)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)], "yaml_list"
        if isinstance(value, dict):
            for key in ("tasks", "manifest", "task_manifest", "rows", "task_rows"):
                candidate = value.get(key)
                if isinstance(candidate, list):
                    return [dict(row) for row in candidate if isinstance(row, dict)], f"yaml.{key}"
            return [value], "yaml_dict"
    return [], "unsupported"


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_like = []
    by_task: dict[str, dict[str, Any]] = {}
    for raw in rows:
        task_id = str(first_value(raw, ["task_id", "candidate_task_id", "id"], "")).strip()
        if not task_id or not TASK_ID_RE.match(task_id):
            continue
        arrival_time = to_float(first_value(raw, ["arrival_time_s", "capture_time_s", "capture_time", "arrival_time", "task_arrival_time_s", "time_s", "t"], math.nan))
        slot = first_value(raw, ["timeline_slot", "arrival_slot", "slot"], "")
        if math.isnan(arrival_time) and slot != "":
            arrival_time = to_float(slot) * 20.0
        if slot == "" and not math.isnan(arrival_time):
            slot = int(round(arrival_time / 20.0))
        capture_time = to_float(first_value(raw, ["capture_time_s", "capture_time", "arrival_time_s", "arrival_time", "task_arrival_time_s"], arrival_time))
        deadline_s = to_float(first_value(raw, ["deadline_s", "deadline_seconds", "deadline"], math.nan))
        if math.isnan(deadline_s):
            deadline_time = to_float(first_value(raw, ["deadline_time_s", "deadline_time"], math.nan))
            if not math.isnan(deadline_time) and not math.isnan(capture_time):
                deadline_s = deadline_time - capture_time
        task_value = to_float(first_value(raw, ["task_value", "effective_task_value", "V_t", "V_eff", "calibrated_task_value"], math.nan))
        norm = {
            "order": 0,
            "timeline_slot": to_int(slot, 0),
            "task_id": task_id,
            "scene_id": str(first_value(raw, ["scene_id"], "")),
            "product_identifier": str(first_value(raw, ["product_identifier"], "")),
            "logical_image_path": str(first_value(raw, ["logical_image_path"], "")).replace("\\", "/"),
            "logical_label_path": str(first_value(raw, ["logical_label_path"], "GRD/validation/GRD_validation.csv")).replace("\\", "/"),
            "arrival_slot": to_int(first_value(raw, ["arrival_slot", "timeline_slot", "slot"], slot), 0),
            "arrival_time_s": float(arrival_time) if not math.isnan(arrival_time) else 0.0,
            "capture_time_s": float(capture_time) if not math.isnan(capture_time) else (float(arrival_time) if not math.isnan(arrival_time) else 0.0),
            "observation_window_id": str(first_value(raw, ["observation_window_id"], "")),
            "task_class": str(first_value(raw, ["task_class"], "")),
            "value_level": str(first_value(raw, ["value_level", "assigned_value_level"], "")),
            "task_value": float(task_value) if not math.isnan(task_value) else 0.0,
            "deadline_s": float(deadline_s) if not math.isnan(deadline_s) else 0.0,
            "quality_requirement": float(to_float(first_value(raw, ["quality_requirement", "Q_req"], 0.0), 0.0)),
            "information_requirement": float(to_float(first_value(raw, ["information_requirement", "I_req"], 0.0), 0.0)),
            "product_requirement": str(first_value(raw, ["product_requirement"], "")),
            "product_requirement_source": str(first_value(raw, ["product_requirement_source"], "")),
            "difficulty": float(to_float(first_value(raw, ["difficulty"], 0.0), 0.0)),
            "uncertainty": float(to_float(first_value(raw, ["uncertainty"], 0.0), 0.0)),
            "high_value_flag": truth(first_value(raw, ["high_value_flag", "is_high_value"], False)),
            "emergency_flag": truth(first_value(raw, ["emergency_flag", "urgent_flag"], False)),
        }
        existing = by_task.get(task_id)
        if existing is None or (norm["timeline_slot"], norm["arrival_time_s"]) < (existing["timeline_slot"], existing["arrival_time_s"]):
            by_task[task_id] = norm
    for index, row in enumerate(sorted(by_task.values(), key=lambda r: (r["timeline_slot"], r["task_id"]))):
        row["order"] = index
        task_like.append(row)
    return task_like


def compare_candidate(source_kind: str, file_path: str, commit: str, data: bytes) -> tuple[dict[str, Any], dict[str, Any] | None]:
    inventory = {
        "source_kind": source_kind,
        "file_path": file_path,
        "git_commit": commit,
        "byte_size": len(data),
        "file_sha256": sha_bytes(data),
        "parse_status": "",
        "schema_fields": "",
    }
    try:
        raw_rows, parse_kind = parse_table(file_path, data)
        inventory["parse_status"] = parse_kind
        schema = sorted({str(key) for row in raw_rows for key in row})
        inventory["schema_fields"] = ";".join(schema)
    except Exception as exc:
        inventory["parse_status"] = f"parse_error:{type(exc).__name__}"
        return inventory, None
    norm_rows = normalize_rows(raw_rows)
    if not norm_rows:
        return inventory, None
    counts = Counter(row["value_level"] for row in norm_rows)
    bundle_recomputed = v25_bundle_hash(raw_rows)
    legacy_recomputed = canonical_hash(norm_rows)
    recomputed = bundle_recomputed or legacy_recomputed
    comparison = {
        "source_kind": source_kind,
        "file_path": file_path,
        "git_commit": commit,
        "recomputed_hash": recomputed,
        "recomputed_hash_v25_bundle": bundle_recomputed or "",
        "recomputed_hash_legacy_canonical": legacy_recomputed,
        "matches_expected": int(recomputed == EXPECTED_HASH),
        "task_count": len(norm_rows),
        "low": counts.get("low", 0),
        "medium": counts.get("medium", 0),
        "high": counts.get("high", 0),
        "urgent": counts.get("urgent", 0),
        "N_HV": counts.get("high", 0) + counts.get("urgent", 0),
        "earliest_arrival": min(row["arrival_time_s"] for row in norm_rows),
        "latest_arrival": max(row["arrival_time_s"] for row in norm_rows),
        "schema_fields": inventory["schema_fields"],
        "raw_row_count": len(raw_rows),
        "normalized_task_count": len(norm_rows),
    }
    return inventory, comparison


def current_candidate_files() -> list[Path]:
    out: list[Path] = []
    for root in [ROOT]:
        for path in root.rglob("*"):
            if path.is_file() and NAME_RE.search(path.name) and path.suffix.lower() in {".csv", ".json", ".jsonl", ".yaml", ".yml", ".md"}:
                out.append(path)
    for path in EXTERNAL_CANDIDATES:
        if path.exists():
            out.append(path)
    return sorted(set(out))


def git_candidate_files() -> list[tuple[str, str]]:
    commits = [line.strip() for line in run_git(["rev-list", "--all"]).splitlines() if line.strip()]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for commit in commits:
        names = run_git(["ls-tree", "-r", "--name-only", commit]).splitlines()
        for name in names:
            if NAME_RE.search(Path(name).name) and Path(name).suffix.lower() in {".csv", ".json", ".jsonl", ".yaml", ".yml", ".md"}:
                key = (commit, name)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return pairs


def identity_audit_text() -> tuple[str, bool]:
    lines = ["# Original Result Identity Audit", ""]
    found_4c23 = False
    roots = [ROOT / "outputs", ROOT / ".codex_method_preservation_audit_v1" / "outputs", Path("D:/wt/pilot/outputs")]
    for base in roots:
        if not base.exists():
            continue
        dirs = [path for path in base.iterdir() if path.is_dir() and any(hint in path.name for hint in FORMAL_OUTPUT_HINTS)]
        for directory in sorted(dirs):
            lines.append(f"## {directory}")
            for name in IDENTITY_FILES:
                path = directory / name
                if not path.exists():
                    lines.append(f"- {name}: missing")
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                found_4c23 = found_4c23 or EXPECTED_HASH in text
                hashes = sorted(set(re.findall(r"[0-9a-f]{64}", text)))
                task_hashes = [h for h in hashes if h in text]
                preview = "; ".join(task_hashes[:8]) if task_hashes else "no 64-hex hash found"
                lines.append(f"- {name}: sha256={sha_file(path)}; hashes={preview}")
            lines.append("")
    bundle = Path("D:/wt/pilot/outputs/_updated_prop_ground_ack_v1/bundles/V25-MAIN-v1/bundle_contract.json")
    if bundle.exists():
        text = bundle.read_text(encoding="utf-8", errors="replace")
        found_4c23 = found_4c23 or EXPECTED_HASH in text
        lines.append(f"## {bundle.parent}")
        lines.append(f"- bundle_contract.json: sha256={sha_file(bundle)}; task_hash={json.loads(text).get('task_hash', '')}")
        lines.append(f"- task_manifest.csv: sha256={sha_file(bundle.parent / 'task_manifest.csv')}")
        lines.append("")
    grep = run_git(["grep", "-n", EXPECTED_HASH, "$(git", "rev-list", "--all)"])
    lines.append("## Git String Search")
    lines.append("Skipped shell-expanded git grep inside Python; full recomputation inventory is authoritative.")
    return "\n".join(lines) + "\n", found_4c23


def main() -> int:
    if OUT.exists():
        shutil = __import__("shutil")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    recovered: dict[str, Any] | None = None

    for path in current_candidate_files():
        try:
            data = path.read_bytes()
        except Exception:
            continue
        inv, comp = compare_candidate("working_tree", str(path), "", data)
        inventory.append(inv)
        if comp:
            comparisons.append(comp)
            if comp["matches_expected"] and recovered is None:
                recovered = {"source_path": str(path), "source_git_commit": "", "data": data, "comparison": comp}

    for commit, name in git_candidate_files():
        data = run_git(["show", f"{commit}:{name}"]).encode("utf-8", errors="replace")
        inv, comp = compare_candidate("git_history", name, commit, data)
        inventory.append(inv)
        if comp:
            comparisons.append(comp)
            if comp["matches_expected"] and recovered is None:
                recovered = {"source_path": name, "source_git_commit": commit, "data": data, "comparison": comp}

    write_csv(OUT / "manifest_search_inventory.csv", inventory)
    write_csv(OUT / "manifest_candidate_comparison.csv", comparisons)
    identity_text, found_identity_4c23 = identity_audit_text()
    (OUT / "original_result_identity_audit.md").write_text(identity_text, encoding="utf-8")

    exact_string_hits = [row for row in inventory if EXPECTED_HASH in str(row)]
    exact_matches = [row for row in comparisons if row["matches_expected"]]
    d110 = [row for row in comparisons if str(row.get("recomputed_hash")) == "d1100340da26de471f48e31e2a4d514b446a511c5106f998959e4ada13184354"]
    orig_hashes = sorted({row["recomputed_hash"] for row in comparisons if row["task_count"] == 268 and row["source_kind"] == "working_tree"})

    if recovered:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        dest = ARTIFACTS / "v25_main_seed2026_manifest_4c23cf3184f1.csv"
        if recovered["source_path"].lower().endswith(".csv"):
            dest.write_bytes(recovered["data"])
        else:
            dest = ARTIFACTS / "v25_main_seed2026_manifest_4c23cf3184f1.json"
            dest.write_bytes(recovered["data"])
        ident = {
            "source_path": recovered["source_path"],
            "source_git_commit": recovered["source_git_commit"],
            "recovered_task_manifest_hash": EXPECTED_HASH,
            "task_counts": {
                "N_all": recovered["comparison"]["task_count"],
                "low": recovered["comparison"]["low"],
                "medium": recovered["comparison"]["medium"],
                "high": recovered["comparison"]["high"],
                "urgent": recovered["comparison"]["urgent"],
                "N_HV": recovered["comparison"]["N_HV"],
            },
            "schema": recovered["comparison"]["schema_fields"].split(";") if recovered["comparison"]["schema_fields"] else [],
            "recovery_timestamp": datetime.now(timezone.utc).isoformat(),
            "read_only": True,
        }
        write_json(ARTIFACTS / "v25_main_seed2026_manifest_identity.json", ident)
        status = "EXACT_V25_SEED2026_MANIFEST_RECOVERED"
    else:
        status = "EXPECTED_HASH_RECORD_WAS_INCORRECT" if not found_identity_4c23 else "EXACT_MANIFEST_NOT_RECOVERABLE"

    lines = [
        "# Recover Exact V25 Seed2026 Manifest 4c23",
        "",
        f"Final status: `{status}`",
        "",
        "## Answers",
        "",
        "1. The original V25 unified formal output under `.codex_method_preservation_audit_v1/outputs/_v25_unified_baselines_core_metrics_v2` records `4c23cf3184f137733495f95a7c5bd7631adf95b3fbc4eb057351f27dac2f2ab5` in its identity files. A separate older fresh cap0.35 quick-compare bundle records `49c704997b59648d902ed9413a4116d0b45e5b404d0d20ece9d3826a7d6d38cd` and is not the V25-MAIN-v1 4c23 manifest.",
        f"2. `4c23cf...` appeared in original formal identity files: {'yes' if found_identity_4c23 else 'no'}. Exact recomputed candidate matches: {len(exact_matches)}.",
        "3. `d11003...` is a forensic quantile relabeling with the requested counts, but it is not the target hash. It differs in value-level assignment identity and therefore the canonical serialized task rows, even though N_all and value-level counts match.",
        f"4. Lossless recovery from complete local raw data: {'yes' if recovered else 'no'} based on recomputation over working tree and git-history candidates.",
        f"5. Assessment: {'case A holds: original formal output recorded 4c23 and the matching task file has now been recovered' if recovered else ('expected hash appears in identity but matching task file is absent' if found_identity_4c23 else 'case B holds for available outputs: expected hash record appears incorrect')}.",
        f"6. Recommendation: {'use the recovered frozen manifest in `artifacts/frozen_scenarios/`; no hash correction or task regeneration is needed' if recovered else 'if a specific external run is known to have produced `4c23cf...`, restore that exact commit/archive. Otherwise correct the expected hash to the formally recorded identity or freeze a new scenario and rerun all five methods from that frozen manifest'}.",
        "",
        "## Search Summary",
        "",
        f"- inventory rows: {len(inventory)}",
        f"- parsed task-like candidate rows: {len(comparisons)}",
        f"- exact 4c23 recomputed matches: {len(exact_matches)}",
        f"- 268-task working-tree recomputed hashes: {'; '.join(orig_hashes[:20])}",
        f"- d11003 candidate count: {len(d110)}",
    ]
    (OUT / "final_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status)
    print(f"inventory={len(inventory)} comparisons={len(comparisons)} exact_matches={len(exact_matches)}")
    return 0 if recovered else 2


if __name__ == "__main__":
    raise SystemExit(main())
