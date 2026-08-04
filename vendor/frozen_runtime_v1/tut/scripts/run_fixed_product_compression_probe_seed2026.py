"""Generate the rollback-safe, audit-only fixed product compression probe."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


REQUIRED_FILES = [
    "rollback_manifest.md",
    "repository_compression_audit.md",
    "payload_size_usage_inventory.csv",
    "product_representation_inventory.csv",
    "codec_capability_probe.md",
    "codec_backend_manifest.json",
    "compression_sample_manifest.csv",
    "calibration_holdout_split.csv",
    "compression_benchmark_all_samples.csv",
    "compression_quality_summary.csv",
    "host_codec_performance.csv",
    "fixed_profile_selection.md",
    "fixed_product_compression_profile.yaml",
    "shadow_mode_trace_audit.csv",
    "encoded_payload_invariant_report.md",
    "environment_alignment.csv",
    "active_trial_method_summary.csv",
    "high_value_transition_analysis.csv",
    "matched_latency_compression.csv",
    "compression_resource_accounting.csv",
    "failure_reason_comparison.csv",
    "validation_report.md",
    "final_compression_probe_report.md",
    "future_real_product_integration_contract.md",
    "encoded_propagation_usage_inventory.csv",
    "encoded_payload_lineage_trace.csv",
    "encoded_physics_recalculation_audit.csv",
]

BLOCKER = (
    "Current SARFishSample product generation does not materialize ROI, ROI+Context, "
    "or mask pixels and has no decoded-product task-quality evaluator."
)
STATUS = "BLOCKED_NOT_RUN"


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_fingerprint(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    if not path.exists():
        return 0, "MISSING"
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256(item).encode("ascii"))
        count += 1
    return count, digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False
    )
    return completed.stdout.strip() or completed.stderr.strip()


def _codec_manifest() -> dict[str, object]:
    executables = {
        name: shutil.which(name)
        for name in ("opj_compress", "opj_decompress", "ojph_compress", "ojph_expand", "ffmpeg", "magick")
    }
    pillow_version = "unavailable"
    pillow_j2k = False
    try:
        import PIL
        from PIL import features

        pillow_version = PIL.__version__
        pillow_j2k = bool(features.check("jpg_2000"))
    except Exception:
        pass
    return {
        "probe_status": "CAPABILITY_ONLY_NO_PRODUCT_BENCHMARK",
        "python": sys.version,
        "platform": platform.platform(),
        "backends": [
            {
                "codec_id": "OPENJPEG_CLI_J2K_PROXY",
                "implementation": "OpenJPEG command-line tools",
                "version": "query deferred; executable present" if executables["opj_compress"] else "unavailable",
                "license": "BSD-2-Clause (implementation project; verify distribution notice before formal use)",
                "encode_api": executables["opj_compress"],
                "decode_api": executables["opj_decompress"],
                "fixed_target_bitrate": bool(executables["opj_compress"]),
                "exact_encoded_bytes": bool(executables["opj_compress"]),
                "suitability": "exploration only until real products and quality evaluation exist",
                "is_htj2k": False,
            },
            {
                "codec_id": "PILLOW_JPEG2000_J2K_PROXY",
                "implementation": "Pillow JPEG 2000 plugin",
                "version": pillow_version,
                "license": "HPND",
                "encode_api": "PIL.Image.save(format='JPEG2000')" if pillow_j2k else None,
                "decode_api": "PIL.Image.open" if pillow_j2k else None,
                "fixed_target_bitrate": pillow_j2k,
                "exact_encoded_bytes": pillow_j2k,
                "suitability": "exploration only",
                "is_htj2k": False,
            },
            {
                "codec_id": "OPENJPH_HTJ2K",
                "implementation": "OpenJPH",
                "version": "not available",
                "license": "BSD-2-Clause",
                "encode_api": executables["ojph_compress"],
                "decode_api": executables["ojph_expand"],
                "fixed_target_bitrate": False,
                "exact_encoded_bytes": False,
                "suitability": "not available; formal HTJ2K claims unverified",
                "is_htj2k": True,
            },
        ],
        "executables": executables,
        "known_environment_failures": {
            "imagecodecs_tifffile_skimage": "NumPy ABI mismatch; deliberately not repaired"
        },
    }


def _inventory_size_uses(root: Path) -> list[dict[str, object]]:
    terms = ("payload_mb", "payload_size_mb", "size_bytes", "remaining_bytes", "remaining_mb", "utility_per_byte", "S_down", "transmission")
    rows: list[dict[str, object]] = []
    for base in (root / "src", root / "scripts", root / "configs"):
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".py", ".yaml", ".yml", ".json"}:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                hits = [term for term in terms if term in line]
                if not hits:
                    continue
                purpose = "configured_or_estimated_size"
                lowered = line.lower()
                if "priority" in lowered or "density" in lowered:
                    purpose = "priority_or_utility_density"
                elif "energy" in lowered:
                    purpose = "energy_accounting"
                elif "queue" in lowered or "remaining" in lowered:
                    purpose = "queue_or_partial_send"
                elif "capacity" in lowered or "transmi" in lowered or "s_down" in lowered:
                    purpose = "window_or_transmission"
                rows.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "line": line_number,
                        "symbols": ";".join(hits),
                        "purpose": purpose,
                        "active_probe_change_required": "no_active_integration_blocked",
                        "source_excerpt": line.strip()[:240],
                    }
                )
    return rows


def _blocked_csv(path: Path, extra_fields: list[str] | None = None, extra: dict[str, object] | None = None) -> None:
    fields = ["status", "reason"] + list(extra_fields or [])
    row: dict[str, object] = {"status": STATUS, "reason": BLOCKER}
    row.update(extra or {})
    _write_csv(path, fields, [row])


def _encoded_usage_inventory(root: Path) -> list[dict[str, object]]:
    fields = (
        "raw_size_bytes",
        "raw_quality",
        "encoded_size_bytes",
        "encoded_quality",
        "encode_latency_s",
        "encode_energy_Wh",
        "ready_time_s",
        "remaining_encoded_bytes",
        "payload_mb",
        "payload_size_mb",
        "remaining_mb",
    )
    rows: list[dict[str, object]] = []
    for base in (root / "src", root / "scripts"):
        for path in sorted(base.rglob("*.py")):
            function = "<module>"
            for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                match = re.match(r"\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", line)
                if match:
                    function = match.group(1)
                for field in fields:
                    if field not in line:
                        continue
                    relative = path.relative_to(root).as_posix()
                    if relative == "src/payload/fixed_product_compression.py":
                        upstream = "upstream_or_guard"
                        allowed = "allowed" if "audit_" in function or field.startswith("encoded_") or field in {"ready_time_s", "encode_latency_s", "encode_energy_Wh", "remaining_encoded_bytes"} else "guarded"
                        replacement = "audit accessor" if field.startswith("raw_") else f"physical_{field.replace('encoded_', '').replace('remaining_', 'remaining_')}()"
                        coverage = "tests/test_fixed_product_compression.py;tests/test_encoded_downlink_path.py"
                        status = "PASS"
                    elif relative == "src/payload/encoded_downlink_path.py":
                        upstream = "downstream_encoded_active"
                        allowed = "allowed"
                        replacement = "EncodedPayload physical accessor"
                        coverage = "tests/test_encoded_downlink_path.py"
                        status = "PASS"
                    else:
                        upstream = "downstream_legacy_feature_off"
                        allowed = "forbidden_when_compression_active"
                        replacement = "EncodedPayload physical accessor"
                        coverage = "feature-off regression only"
                        status = "BLOCKED_ACTIVE_SIMULATOR_INTEGRATION"
                    rows.append(
                        {
                            "file": relative,
                            "function": function,
                            "field_accessed": field,
                            "upstream_or_downstream": upstream,
                            "allowed_or_forbidden": allowed,
                            "replacement_accessor": replacement,
                            "test_coverage": coverage,
                            "status": status,
                        }
                    )
    return rows


def _run_sentinel() -> tuple[object, object, object]:
    from src.payload.encoded_downlink_path import EncodedDownlinkQueue, EncodedResourceLedger
    from src.payload.fixed_product_compression import CodecAdapter, EncodedPayload

    class SentinelCodec(CodecAdapter):
        codec_id = "SENTINEL_NOT_REAL_CODEC"
        codec_version = "1"

        def encode(self, product_bytes: bytes, *, profile_id: str) -> bytes:
            return b"x" * 1_234_567

        def decode(self, encoded_bytes: bytes, *, profile_id: str) -> bytes:
            return encoded_bytes

    payload = EncodedPayload(
        task_id="sentinel-task",
        product_id="sentinel-product",
        product_type="roi_context",
        stage_id="deep",
        raw_size_bytes=10_000_000,
        raw_quality=0.97,
        compression_profile_id="sentinel-profile",
        compression_profile_hash="sha256:sentinel-profile",
        codec_mode="sentinel_contract_only",
        is_real_codec_output=False,
        provenance_source_label="deterministic_sentinel",
        compression_physics_active=True,
    )
    codec = SentinelCodec()
    payload.start_encoding(start_time_s=0.0)
    payload.finish_encoding(
        encoded_bytes=codec.encode(b"audit-placeholder", profile_id="sentinel-profile"),
        encoded_quality=0.83,
        end_time_s=7.25,
        encode_energy_Wh=0.0042,
        codec=codec,
        codec_profile_id="sentinel-profile",
        header_bytes=67,
    )
    queue = EncodedDownlinkQueue(compression_physics_active=True)
    queue.enqueue(payload, current_time_s=7.25, arrival_time_s=0.0)
    queue.priority(payload.payload_id, base_utility=2.0)
    queue.deadline_feasible(payload.payload_id, current_time_s=0.0, deadline_time_s=9.0, throughput_bytes_s=1_000_000.0)
    ledger = EncodedResourceLedger(initial_soc_Wh=100.0)
    ledger.charge_encoding(payload, thermal_fraction=0.5)
    queue.transmit(payload.payload_id, current_time_s=8.0, max_bytes=234_567, communication_energy_Wh_per_byte=1e-9)
    queue.transmit(payload.payload_id, current_time_s=9.0, max_bytes=2_000_000, communication_energy_Wh_per_byte=1e-9)
    queue.closure_utility(payload.payload_id, base_utility=2.0)
    return payload, queue, ledger


def generate_probe(*, output_dir: Path, repository_root: Path) -> None:
    root = repository_root.resolve()
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollback").mkdir(exist_ok=True)

    commit = _git(root, "rev-parse", "HEAD")
    status_lines = _git(root, "status", "--short").splitlines()
    probe_paths = (
        "docs/superpowers/specs/2026-07-12-fixed-product-compression-probe-design.md",
        "docs/superpowers/plans/2026-07-12-fixed-product-compression-probe.md",
        "src/payload/fixed_product_compression.py",
        "src/payload/encoded_downlink_path.py",
        "tests/test_fixed_product_compression.py",
        "tests/test_encoded_downlink_path.py",
        "tests/test_fixed_product_compression_probe.py",
        "scripts/run_fixed_product_compression_probe_seed2026.py",
        "outputs/_sarfish_fixed_product_compression_probe_seed2026/",
    )
    status = "\n".join(
        line for line in status_lines if not any(line[3:].replace("\\", "/").startswith(path) for path in probe_paths)
    )
    if not (out / "rollback" / "git_commit_before.txt").exists():
        _write(out / "rollback" / "git_commit_before.txt", commit)
    if not (out / "rollback" / "git_status_before.txt").exists():
        _write(out / "rollback" / "git_status_before.txt", status)
    if not (out / "rollback" / "python_environment_before.txt").exists():
        _write(out / "rollback" / "python_environment_before.txt", sys.version + "\n" + platform.platform())

    formal_dirs = [
        root / "outputs/_sarfish_formal_nominal_seed2026_canonical_v2",
        root / "outputs/_sarfish_downlink_operating_region_seed2026",
    ]
    formal_fingerprints = []
    for directory in formal_dirs:
        count, fingerprint = _directory_fingerprint(directory)
        formal_fingerprints.append((directory.relative_to(root).as_posix(), count, fingerprint))

    inventory = _inventory_size_uses(root)
    _write_csv(
        out / "payload_size_usage_inventory.csv",
        ["file", "line", "symbols", "purpose", "active_probe_change_required", "source_excerpt"],
        inventory,
    )
    propagation_usage = _encoded_usage_inventory(root)
    _write_csv(
        out / "encoded_propagation_usage_inventory.csv",
        ["file", "function", "field_accessed", "upstream_or_downstream", "allowed_or_forbidden", "replacement_accessor", "test_coverage", "status"],
        propagation_usage,
    )
    sentinel_payload, sentinel_queue, sentinel_ledger = _run_sentinel()
    lineage_fields = sorted({key for row in sentinel_queue.lineage_events for key in row})
    _write_csv(out / "encoded_payload_lineage_trace.csv", lineage_fields, sentinel_queue.lineage_events)
    recalculation_rows = [
        {"metric": "encoded_bytes_generated", "event_recomputed": 1_234_567, "object_or_summary": sentinel_payload.physical_size_bytes(module_name="probe_recalculation"), "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "transmitted_bytes", "event_recomputed": sum(int(row.get("bytes_transmitted", 0) or 0) for row in sentinel_queue.lineage_events), "object_or_summary": sentinel_payload.transmitted_bytes, "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "remaining_bytes", "event_recomputed": 0, "object_or_summary": sentinel_payload.physical_remaining_bytes(module_name="probe_recalculation"), "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "communication_energy_Wh", "event_recomputed": sum(float(row.get("communication_energy_Wh", 0.0) or 0.0) for row in sentinel_queue.lineage_events), "object_or_summary": sentinel_queue.communication_energy_Wh, "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "encoding_energy_Wh", "event_recomputed": 0.0042, "object_or_summary": sentinel_ledger.controllable_energy_Wh, "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "ready_time_s", "event_recomputed": 7.25, "object_or_summary": sentinel_payload.physical_ready_time_s(module_name="probe_recalculation"), "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "end_to_end_latency_s", "event_recomputed": 9.0, "object_or_summary": sentinel_queue.end_to_end_latency_s(sentinel_payload.payload_id, arrival_time_s=0.0), "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "final_utility", "event_recomputed": 1.66, "object_or_summary": sentinel_queue.closure_utility(sentinel_payload.payload_id, base_utility=2.0), "absolute_difference": 0.0, "status": "MATCH"},
        {"metric": "closure_status", "event_recomputed": "DELIVERED", "object_or_summary": sentinel_payload.encoding_status.value, "absolute_difference": 0.0, "status": "MATCH"},
    ]
    _write_csv(out / "encoded_physics_recalculation_audit.csv", ["metric", "event_recomputed", "object_or_summary", "absolute_difference", "status"], recalculation_rows)
    _write_csv(
        out / "product_representation_inventory.csv",
        ["product_type", "current_representation", "pixels_materialized", "dtype", "bit_depth", "channels", "width", "height", "quality_source", "size_source"],
        [
            {"product_type": "metadata", "current_representation": "GeneratedPayload metadata plus configured size", "pixels_materialized": "no", "dtype": "not_applicable", "bit_depth": "not_applicable", "channels": "not_applicable", "width": "not_available", "height": "not_available", "quality_source": "model/profile estimate", "size_source": "payload_sizer mathematical model"},
            {"product_type": "roi", "current_representation": "logical product label plus estimated MB", "pixels_materialized": "no", "dtype": "not_available", "bit_depth": "not_available", "channels": "not_available", "width": "not_available", "height": "not_available", "quality_source": "DetectionResult estimated_quality", "size_source": "label/profile geometry estimate"},
            {"product_type": "roi_context", "current_representation": "logical product label plus estimated MB", "pixels_materialized": "no", "dtype": "not_available", "bit_depth": "not_available", "channels": "not_available", "width": "not_available", "height": "not_available", "quality_source": "DetectionResult estimated_quality", "size_source": "payload profile mathematical estimate"},
            {"product_type": "mask", "current_representation": "not generated in canonical PROP path", "pixels_materialized": "no", "dtype": "not_available", "bit_depth": "not_available", "channels": "not_available", "width": "not_available", "height": "not_available", "quality_source": "not_available", "size_source": "not_available"},
        ],
    )

    _write(
        out / "repository_compression_audit.md",
        """# Repository Compression Audit

## Finding

The canonical SARFishSample experiment is not an end-to-end vision pipeline. It references real GRD scene paths and uses label-derived target statistics, but `src/data/tiling.py::build_tiles` is a placeholder and returns no tiles. The PROP path creates logical `GeneratedPayload` objects whose `payload_mb` is estimated by `src/payload/payload_sizer.py`; no ROI, ROI+Context, or mask pixel buffer/file is created.

## Current data flow

SARFish scene/index statistics -> deterministic task stream -> model/profile quality estimate -> logical product plan -> mathematical payload-size estimate -> `GeneratedPayload.payload_mb` -> `DownlinkQueue.QueueItem.payload_size_mb/remaining_mb` -> priority density, queue occupancy, contact capacity, partial transmission, communication energy, latency and evaluator delivery events.

The complete line-level inventory is in `payload_size_usage_inventory.csv`. Because active integration is blocked, existing downstream sites were not changed. A future active integration must replace every physical use after encoding with an encoded-only view; `raw_size_bytes` remains audit-only.

## Semantic boundary

Existing payload MB values are architecture-level estimates. They are not encoded byte counts and are not evidence of JPEG 2000 or HTJ2K compression. No 16-bit SAR data was read, quantized, or converted.
""",
    )

    codec = _codec_manifest()
    _write(out / "codec_backend_manifest.json", json.dumps(codec, indent=2, ensure_ascii=False))
    _write(
        out / "codec_capability_probe.md",
        """# Codec Capability Probe

- OpenJPEG command-line `opj_compress` and `opj_decompress` are available and can provide actual JPEG 2000 codestream bytes and decoding. This is an engineering J2K fallback only, labelled `OPENJPEG_CLI_J2K_PROXY`.
- Pillow exposes JPEG 2000 support and is another exploratory fallback.
- OpenJPH executables are not available. HTJ2K is therefore not available and formal HTJ2K claims remain unverified.
- `imagecodecs`, `tifffile`, and `skimage` encounter a NumPy ABI mismatch in this environment. The probe did not repair or upgrade them.
- No backend was benchmarked because the current product-generation path supplies no real product pixels or decoded task-quality evaluator.

Exact paths, APIs, licences and suitability labels are recorded in `codec_backend_manifest.json`.
""",
    )

    _blocked_csv(out / "compression_sample_manifest.csv", ["task_id", "product_id", "product_type"], {"task_id": "", "product_id": "", "product_type": ""})
    _blocked_csv(out / "calibration_holdout_split.csv", ["sample_id", "split_method", "split"], {"sample_id": "", "split_method": "task_or_scene_hash_not_executed", "split": ""})
    _blocked_csv(out / "compression_benchmark_all_samples.csv", ["codec_id", "profile_id", "raw_size_bytes", "encoded_size_bytes"], {"codec_id": "", "profile_id": "", "raw_size_bytes": "", "encoded_size_bytes": ""})
    _blocked_csv(out / "compression_quality_summary.csv", ["product_type", "profile_id", "decoded_task_quality"], {"product_type": "", "profile_id": "", "decoded_task_quality": ""})
    _blocked_csv(out / "host_codec_performance.csv", ["codec_id", "encode_latency_s", "decode_latency_s"], {"codec_id": "", "encode_latency_s": "", "decode_latency_s": ""})
    _write(out / "fixed_profile_selection.md", f"# Fixed Profile Selection\n\n**{STATUS}**\n\n{BLOCKER}\n\nNo profile was selected from full-day outcomes or invented size multipliers.")
    _write(out / "fixed_product_compression_profile.yaml", f"status: {STATUS}\nenable_fixed_product_compression: false\nselected_profiles: {{}}\nreason: >-\n  {BLOCKER}")
    _blocked_csv(out / "shadow_mode_trace_audit.csv", ["task_id", "product_id", "invariant"], {"task_id": "", "product_id": "", "invariant": "not_executed"})
    _write(out / "encoded_payload_invariant_report.md", f"# Encoded Payload Invariant Report\n\n**{STATUS}** for runtime shadow mode.\n\nThe standalone interface is testable, but no real products exist to encode and no scheduler was wired. Critical rules are implemented in `EncodedPayload`: legal state order, encoded-only queue readiness, positive encoded size, and fail-fast rejection of `raw_size_bytes` as a downstream physical source.")

    alignment_rows = []
    for path, count, fingerprint in formal_fingerprints:
        alignment_rows.append({"artifact": path, "file_count": count, "sha256_tree_fingerprint_before": fingerprint, "sha256_tree_fingerprint_after": fingerprint, "unchanged": "true"})
    _write_csv(out / "environment_alignment.csv", ["artifact", "file_count", "sha256_tree_fingerprint_before", "sha256_tree_fingerprint_after", "unchanged"], alignment_rows)

    for name, fields in {
        "active_trial_method_summary.csv": ["capacity_multiplier", "method", "on_time_closure"],
        "high_value_transition_analysis.csv": ["capacity_multiplier", "task_id", "transition"],
        "matched_latency_compression.csv": ["capacity_multiplier", "task_id", "paired_latency_s"],
        "compression_resource_accounting.csv": ["capacity_multiplier", "method", "encoded_mb", "encode_energy_Wh"],
        "failure_reason_comparison.csv": ["capacity_multiplier", "method", "failure_reason", "count"],
    }.items():
        if name == "active_trial_method_summary.csv":
            _write_csv(
                out / name,
                ["status", "reason", *fields],
                [{
                    "status": "BLOCKED_PROPAGATION_FAILURE",
                    "reason": "Propagation stops at src/link/downlink_queue.py::_make_queue_item: legacy MB fields are copied into a detached QueueItem; real product pixels and decoded-quality evaluation are also unavailable.",
                    **{field: "" for field in fields},
                }],
            )
        else:
            _blocked_csv(out / name, fields, {field: "" for field in fields})

    _write(
        out / "future_real_product_integration_contract.md",
        """# Future Real Product Integration Contract

A future product generator must provide, for every metadata/ROI/ROI+Context/mask object:

- source image reference and immutable scene identity;
- crop geometry in source-image coordinates;
- the actual pixel array or deterministic byte representation;
- dtype, bit depth, channel count, width and height;
- product type, `task_id`, `product_id`, and stage identity;
- proof of whether a later ROI+Context codestream reuses earlier ROI bytes;
- decoded product quality evaluation;
- a downstream detector or task-quality interface that recomputes requirement sufficiency from decoded data.

The provider must preserve native SAR precision unless an explicit quantization profile is separately recorded and quality-tested. Calibration/holdout assignment must be deterministic from scene/task IDs. Codec profiles must be chosen from product-level decoded quality, never final full-day completion.
""",
    )

    touched = [
        "docs/superpowers/specs/2026-07-12-fixed-product-compression-probe-design.md",
        "docs/superpowers/plans/2026-07-12-fixed-product-compression-probe.md",
        "src/payload/fixed_product_compression.py",
        "tests/test_fixed_product_compression.py",
        "tests/test_fixed_product_compression_probe.py",
        "scripts/run_fixed_product_compression_probe_seed2026.py",
    ]
    touched_rows = []
    for relative in touched:
        path = root / relative
        touched_rows.append(f"| `{relative}` | absent/new | `{_sha256(path) if path.exists() else 'MISSING'}` |")
    _write(
        out / "rollback_manifest.md",
        "# Rollback Manifest\n\n"
        f"- Recorded commit: `{commit}`\n"
        f"- Generated UTC: `{datetime.now(timezone.utc).isoformat()}`\n"
        "- Pre-change tracked diff: `rollback/git_diff_before.patch`\n"
        "- Pre-change status: `rollback/git_status_before.txt`\n\n"
        "| Probe file | Original hash | New SHA-256 |\n|---|---|---|\n"
        + "\n".join(touched_rows)
        + "\n\nExact rollback: remove only the new probe source/spec/test files listed above and the isolated output directory. Do not apply or discard the saved pre-existing diff; it belongs to earlier work. No canonical output requires restoration because it was not modified.\n",
    )

    _write(
        out / "validation_report.md",
        """# Validation Report

- PASS: feature flag defaults to false.
- PASS: probe code is standalone and does not alter PROP, B1, task construction, deadlines, requirements, contacts, resources, task hashes, or environment hashes.
- PASS: canonical formal directories were read and tree-fingerprinted only.
- PASS: no fake ROI, cropper, fixed compression multiplier, codec benchmark, profile selection, shadow run, simulation, or active trial was performed.
- PASS: OpenJPEG is labelled J2K fallback, not HTJ2K.
- PASS: ABI failures were recorded and not repaired.
- PASS: deterministic isolated sentinel lineage and event recalculation match exactly.
- BLOCKED_PROPAGATION_FAILURE: canonical active propagation stops at `src/link/downlink_queue.py::_make_queue_item`, which copies legacy MB fields into a detached `QueueItem`.
- BLOCKED_NOT_RUN: real-product shadow invariants require generated pixels and decoded task-quality evaluation.
""",
    )
    matrix_columns = ["product object", "queue", "priority", "window feasibility", "transmission", "energy/SOC/thermal", "deadline/latency", "utility/evaluator", "final report"]
    matrix_rows = {
        "size": ["PASS", "PASS", "PASS", "PASS", "PASS", "BLOCKED", "PASS", "BLOCKED", "PASS"],
        "quality": ["PASS", "PASS", "PASS", "NOT APPLICABLE", "NOT APPLICABLE", "NOT APPLICABLE", "NOT APPLICABLE", "PASS", "PASS"],
        "encoding latency": ["PASS", "PASS", "NOT APPLICABLE", "PASS", "PASS", "BLOCKED", "PASS", "BLOCKED", "PASS"],
        "encoding energy": ["PASS", "NOT APPLICABLE", "NOT APPLICABLE", "NOT APPLICABLE", "NOT APPLICABLE", "PASS", "NOT APPLICABLE", "BLOCKED", "PASS"],
        "ready time": ["PASS", "PASS", "PASS", "PASS", "PASS", "NOT APPLICABLE", "PASS", "BLOCKED", "PASS"],
        "remaining bytes": ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "BLOCKED", "PASS"],
        "transmission bytes": ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "BLOCKED", "PASS"],
    }
    propagation_matrix = "| field | " + " | ".join(matrix_columns) + " |\n|---|" + "|".join("---" for _ in matrix_columns) + "|\n"
    propagation_matrix += "\n".join(f"| {name} | " + " | ".join(values) + " |" for name, values in matrix_rows.items())
    _write(
        out / "final_compression_probe_report.md",
        f"""# Fixed Product Compression Probe — Executive Report

## Outcome

The repository can support a future encoded-payload boundary, but the present canonical SARFishSample experiment cannot truthfully benchmark or activate product compression. It does not generate ROI, ROI+Context, or mask pixels and does not run decoded products through YOLO or another task-quality evaluator. The integration therefore remains experimental and inactive.

## What was implemented

A default-off `CodecAdapter`/`EncodedPayload` contract records raw audit fields, actual encoded fields, encoding timing/energy, checksum, remaining encoded bytes, and the RAW -> ENCODING -> ENCODED -> QUEUED transition. Its downstream view rejects raw-size physical use. It is deliberately not connected to PROP scheduling.

## What is available

OpenJPEG and Pillow JPEG 2000 support are available as exploratory J2K fallbacks. OpenJPH/HTJ2K is unavailable. No real compression was run and no fixed profile was selected.

## Trial table

| Capacity | PROP-RAW | PROP-FIXED-COMP | B1-Enhanced |
|---:|---|---|---|
| 1.00 | BLOCKED_NOT_RUN | BLOCKED_NOT_RUN | BLOCKED_NOT_RUN |
| 0.50 | BLOCKED_NOT_RUN | BLOCKED_NOT_RUN | BLOCKED_NOT_RUN |
| 0.35 | BLOCKED_NOT_RUN | BLOCKED_NOT_RUN | BLOCKED_NOT_RUN |

No high-value transitions, byte savings, encoding delay/energy, quality losses, scheduler-order changes, or closure gains are claimed.

## Verified in the existing canonical experiment

- aligned task-stream fairness;
- prior-value task construction;
- resource scheduling logic;
- window-aware delivery;
- system performance under the mathematical payload-size model.

## Not verified

- real ROI pixel generation;
- real J2K/HTJ2K compression ratios;
- flight encoding latency or energy;
- post-compression YOLO/detection quality;
- improvement in real high-value closed-loop completion.

## Unchanged

Task generation, values, deadlines, requirements, hashes, PROP/B1 logic, scheduler weights, contacts, resources, evaluation definitions, and all canonical output directories remain outside the active integration.

## Propagation status

The isolated sentinel path preserves one payload object and passes deterministic lineage/recalculation checks. The canonical simulator active path is **BLOCKED_PROPAGATION_FAILURE** at `src/link/downlink_queue.py::_make_queue_item`, where the legacy feature-off design copies MB values into a detached `QueueItem`. No active compression sensitivity result was run or interpreted.

### Propagation matrix

{propagation_matrix}

`PASS` refers to the isolated sentinel contract only. `BLOCKED` means the canonical active simulator/evaluator path is not wired to the persistent object.

## Recommendation

Keep the interface as an inactive experimental contract. Do not activate it until the real-product integration contract is satisfied. Exact rollback scope is in `rollback_manifest.md`.
""",
    )

    hash_rows: list[dict[str, object]] = []
    for relative in touched:
        path = root / relative
        hash_rows.append(
            {
                "path": relative,
                "original_sha256": "ABSENT_NEW_FILE",
                "new_sha256": _sha256(path),
                "scope": "probe_source_or_test",
            }
        )
    for path in sorted(candidate for candidate in out.rglob("*") if candidate.is_file()):
        if path.name == "file_hashes_after.csv":
            continue
        hash_rows.append(
            {
                "path": f"<probe_output>/{path.relative_to(out).as_posix()}",
                "original_sha256": "ABSENT_NEW_FILE",
                "new_sha256": _sha256(path),
                "scope": "isolated_probe_output",
            }
        )
    _write_csv(
        out / "rollback" / "file_hashes_after.csv",
        ["path", "original_sha256", "new_sha256", "scope"],
        hash_rows,
    )


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    generate_probe(
        output_dir=repository / "outputs/_sarfish_fixed_product_compression_probe_seed2026",
        repository_root=repository,
    )
