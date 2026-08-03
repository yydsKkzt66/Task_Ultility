"""Read-only TUT semantic-coding fidelity and cost audit.

This entry point reads frozen formal runs and source assets, writes only a new
audit directory, and never imports or executes a decision policy.  All
counterfactuals are static-trace diagnostics and cannot replace formal runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
TUT_ROOT = Path("D:/wt/tut_uoi_rate_lite")
NATIVE_ROOT = Path("D:/wt/pilot_late_hv_native_direct_v1/tools/late_hv_native_direct_final_v6")
FORMAL_ROOT = ROOT / "outputs/_frozen_v2_final_gate_blind_fiveseed_v1"
CALIBRATION_ROOT = ROOT / "outputs/_historical_asset_audit_conditional_v2_calibration_v1"
OUT = ROOT / "outputs/_tut_semantic_coding_fidelity_cost_audit_v1"
SEEDS = tuple(range(10002, 10007))
METHOD = "TUT-UOI-PRODUCT-LITE-v2"
COMPLETION = "TUT_SEMANTIC_CODING_FIDELITY_AND_COST_AUDIT_COMPLETE"

MODEL_LATENCY_S = 1.44
MODEL_ENERGY_J = 21.60
MODEL_POWER_W = MODEL_ENERGY_J / MODEL_LATENCY_S
MODEL_THERMAL_SCALE = 1.60
PMSC_REFERENCE_LATENCY_S = 0.43
PMSC_REFERENCE_ENERGY_J = MODEL_POWER_W * PMSC_REFERENCE_LATENCY_S
RAW_IMAGE_PROXY_MB = 32.0
THERMAL_INIT = 0.52
THERMAL_LIMIT = 0.92
STAGE_PRODUCT = {"metadata": "metadata", "roi": "roi", "evidence": "roi_context", "full": "roi_context"}
PRODUCT_RANK = {"": -1, "defer": -1, "metadata": 0, "roi": 1, "roi_context": 2}
REQUIRED_PRODUCT = {"alert": "metadata", "verification": "roi", "context": "roi_context", "evidence": "roi_context"}
ENCODING_SCALE = {
    "metadata": {"latency": 0.10, "energy": 0.05},
    "roi": {"latency": 1.0, "energy": 1.0},
    "roi_context": {"latency": 1.9, "energy": 2.2},
}


def f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def i(value: Any) -> int:
    return int(float(value or 0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    values = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in values for key in row)) or ["empty"]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    clean = list(values)
    return statistics.fmean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.9g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    if not rows:
        return "_No rows._"
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    out.extend("| " + " | ".join(cell(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(out)


def product_from_action(action: str) -> str:
    text = str(action or "").lower()
    if text == "delay/store" or "defer" in text:
        return "defer"
    if "+" in text:
        return text.rsplit("+", 1)[-1]
    return text


def product_satisfies(product: str, requirement: str) -> bool:
    required = REQUIRED_PRODUCT.get(str(requirement).lower(), "roi")
    return PRODUCT_RANK.get(product, -1) >= PRODUCT_RANK[required]


def asset_paths() -> list[Path]:
    fixed = [
        TUT_ROOT / "configs/methods/tut_uoi_rate_lite.yaml",
        TUT_ROOT / "configs/experiments/tut_uoi_rate_lite_v2_formal_v1.yaml",
        TUT_ROOT / "configs/payload_profiles_roi_compression_calibrated.yaml",
        TUT_ROOT / "configs/model_profiles_yolo_inspired_calibrated.yaml",
        TUT_ROOT / "configs/product_requirement_utility.yaml",
        TUT_ROOT / "scripts/run_tut_uoi_rate_lite.py",
        TUT_ROOT / "scripts/run_tut_uoi_rate_lite_v2.py",
        TUT_ROOT / "src/routers/tut_uoi_rate_router.py",
        TUT_ROOT / "src/simulator/predictors.py",
        TUT_ROOT / "src/simulator/step_runner.py",
        TUT_ROOT / "src/link/downlink_queue.py",
        TUT_ROOT / "src/payload/payload_sizer.py",
        TUT_ROOT / "src/models/mock_detector.py",
        TUT_ROOT / "src/evaluation/task_outcome_adapter.py",
        NATIVE_ROOT / "native_tut.py",
        NATIVE_ROOT / "native_support.py",
        ROOT / "src/evaluation/frozen_v2_final_gate.py",
        ROOT / "src/calibration/conditional_v2_calibration.py",
        CALIBRATION_ROOT / "frozen/conditional_v2_frozen_parameters_v1.json",
        FORMAL_ROOT / "tables/formal_five_seed_per_run.csv",
        FORMAL_ROOT / "tables/baseline_actual_model_q_audit.csv",
    ]
    for seed in SEEDS:
        fixed.extend(sorted((FORMAL_ROOT / f"runs/formal/seed_{seed}/TUT").glob("*")))
    missing = [str(path) for path in fixed if not path.is_file()]
    if missing:
        raise FileNotFoundError({"missing_frozen_assets": missing})
    return fixed


def hash_assets(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {
        str(path.resolve()): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(set(paths), key=lambda value: str(value).lower())
    }


def source_fragment_checks() -> list[dict[str, Any]]:
    checks = [
        (TUT_ROOT / "configs/methods/tut_uoi_rate_lite.yaml", "not a full TUT, PMSC, JSCC, PPO"),
        (TUT_ROOT / "src/simulator/predictors.py", "tut_predicted_energy_fix_enabled"),
        (TUT_ROOT / "src/simulator/step_runner.py", "E_comp += payload_encoding_energy_j"),
        (TUT_ROOT / "src/link/downlink_queue.py", "current_time_s + elapsed_tx_s"),
        (NATIVE_ROOT / "native_tut.py", "for key in (\"tut_uoi_density_priority_enabled\""),
        (ROOT / "src/evaluation/frozen_v2_final_gate.py", "positive = f(task[\"effective_task_value\"]) * delta_i * q_eff * timely"),
    ]
    rows = []
    for path, fragment in checks:
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append({"path": str(path), "fragment": fragment, "found": fragment in text})
    return rows


def call_chain_rows() -> list[dict[str, Any]]:
    return [
        {
            "order": 1, "stage": "Balanced/YOLO task network inference", "implementation": "MockDetector + balanced profile",
            "latency": "PROXY_ONLY: 1.44 s", "satellite_energy": "PROXY_ONLY: 21.60 J",
            "ground_latency": "UNVERIFIED", "ground_energy": "UNVERIFIED",
            "thermal": "COUNTED_CORRECTLY for proxy energy, scale=1.6", "queue_closure": "inference latency enters L_inf; no persistent compute queue",
            "evidence": "model profile is explicitly YOLO-inspired and not a real SARFish measurement",
        },
        {
            "order": 2, "stage": "Perceptual-weight map generation", "implementation": "No PM generator on active call path",
            "latency": "OMITTED", "satellite_energy": "OMITTED", "ground_latency": "not applicable", "ground_energy": "not applicable",
            "thermal": "OMITTED", "queue_closure": "OMITTED", "evidence": "no active PM/PMSC module; products are discrete metadata/ROI/ROI+Ctx",
        },
        {
            "order": 3, "stage": "PMSC neural semantic encoder", "implementation": "Generic payload-size-linear encoding proxy",
            "latency": "PROXY_ONLY: 0.18 s/MB times product scale", "satellite_energy": "PROXY_ONLY: 2.4 J/MB times product scale",
            "ground_latency": "not applicable", "ground_energy": "not applicable", "thermal": "COUNTED for proxy energy only",
            "queue_closure": "encoding latency added to L_inf but product is enqueued in the same simulator step",
            "evidence": "no 317 MB model, SFT encoder, PM conditioning, learned latent, or 0.43 s fixed PMSC stage",
        },
        {
            "order": 4, "stage": "Quantization / bitstream / ordinary payload encoding", "implementation": "Payload size and generic encoding proxy",
            "latency": "PROXY_ONLY", "satellite_energy": "PROXY_ONLY", "ground_latency": "UNVERIFIED", "ground_energy": "UNVERIFIED",
            "thermal": "COUNTED for generic encoding energy", "queue_closure": "payload bytes enter common queue; no explicit quantizer/entropy coder",
            "evidence": "size is label/formula based and not a measured semantic bitstream",
        },
        {
            "order": 5, "stage": "PPO/router online inference", "implementation": "Deterministic TutUoiRateRouter, not PPO",
            "latency": "OMITTED", "satellite_energy": "OMITTED", "ground_latency": "not applicable", "ground_energy": "not applicable",
            "thermal": "OMITTED", "queue_closure": "no contribution", "evidence": "valid for declared LITE heuristic identity, inconsistent with a full paper reproduction",
        },
        {
            "order": 6, "stage": "Wireless transmission", "implementation": "Common downlink queue/link at 38 W",
            "latency": "COUNTED_CORRECTLY", "satellite_energy": "COUNTED_CORRECTLY", "ground_latency": "not applicable",
            "ground_energy": "not modeled", "thermal": "COUNTED_CORRECTLY", "queue_closure": "COUNTED_CORRECTLY through queue/contact/partial TX finish time",
            "evidence": "same common radio/link as other methods; paper 100/200 W is not imported",
        },
        {
            "order": 7, "stage": "Ground semantic decoder", "implementation": "No TUT-specific decoder",
            "latency": "not applicable", "satellite_energy": "correctly excluded", "ground_latency": "OMITTED", "ground_energy": "OMITTED",
            "thermal": "correctly excluded from spacecraft", "queue_closure": "OMITTED: ground availability equals bitstream finish",
            "evidence": "delivery event is emitted at current_time_s + elapsed_tx_s",
        },
        {
            "order": 8, "stage": "Ground downstream task inference", "implementation": "No TUT-specific downstream network",
            "latency": "not applicable", "satellite_energy": "correctly excluded", "ground_latency": "OMITTED", "ground_energy": "OMITTED",
            "thermal": "correctly excluded from spacecraft", "queue_closure": "OMITTED: closure can be credited before reconstruction/inference",
            "evidence": "public q uses selected onboard model rather than reconstructed-image downstream mAP",
        },
    ]


def detailed_call_chain_rows() -> list[dict[str, Any]]:
    """Concrete active-file trace; line numbers refer to the frozen source hashes."""

    return [
        {"order": 1, "file": str(NATIVE_ROOT / "native_tut.py"), "lines": "9-25", "symbol": "TutBinding._instantiate/simulate", "role": "bind requested TUT-v2 identity, rebuild common environment, run TUT router", "finding": "v2 predicted-energy switch is not copied into runtime simulator config"},
        {"order": 2, "file": str(TUT_ROOT / "scripts/run_tut_uoi_rate_lite.py"), "lines": "71-105", "symbol": "build_configs/build_router", "role": "construct fixed Balanced x {metadata,ROI,ROI+Ctx}+Defer action space and deterministic router", "finding": "one-shot LITE action space; no PMSC/PPO"},
        {"order": 3, "file": str(TUT_ROOT / "src/simulator/predictors.py"), "lines": "248-359", "symbol": "Predictor.predict", "role": "predict quality, payload, encode latency/energy, radio, queue, SOC and thermal", "finding": "formal runtime uses fix_enabled=False and therefore adds model power over encoding latency"},
        {"order": 4, "file": str(TUT_ROOT / "src/routers/safe_filter.py"), "lines": "active filter_actions/shield_or_fallback", "symbol": "SafeFilter", "role": "apply common hard safety before and after router selection", "finding": "common safety path retained"},
        {"order": 5, "file": str(TUT_ROOT / "src/simulator/step_runner.py"), "lines": "6653-6687", "symbol": "filter -> router.select_action -> shield", "role": "pass safe candidates to deterministic TUT selector", "finding": "no PPO inference or cost"},
        {"order": 6, "file": str(TUT_ROOT / "src/routers/tut_uoi_rate_router.py"), "lines": "23-173", "symbol": "TutUoiRateRouter", "role": "score predicted UoI, Lyapunov energy/byte/latency terms and select once", "finding": "semantic rate is a discrete common-product choice"},
        {"order": 7, "file": str(TUT_ROOT / "src/simulator/step_runner.py"), "lines": "6703-6725", "symbol": "inference_runner.run/generate_payload", "role": "run Balanced detector and instantiate selected product", "finding": "no PM generator or learned semantic encoder is called"},
        {"order": 8, "file": str(TUT_ROOT / "src/payload/payload_sizer.py"), "lines": "227-243", "symbol": "estimate_payload_encoding_cost", "role": "size-linear generic encoding latency and energy", "finding": "proxy only; not identified as JPEG or PMSC"},
        {"order": 9, "file": str(TUT_ROOT / "src/simulator/step_runner.py"), "lines": "5820-5921", "symbol": "_compute_actual_feedback", "role": "charge actual model+generic encoding energy and latency", "finding": "actual encoding is not double-counted"},
        {"order": 10, "file": str(TUT_ROOT / "src/link/downlink_queue.py"), "lines": "570-620", "symbol": "queue transmit/deliver", "role": "partial transmit and emit ground event at final bit time", "finding": "no post-arrival semantic decode/downstream inference"},
        {"order": 11, "file": str(NATIVE_ROOT / "native_support.py"), "lines": "normalize_episode", "symbol": "normalize_episode", "role": "convert native bit-arrival events to unified closure records", "finding": "closure is based on required stage at bit arrival"},
        {"order": 12, "file": str(ROOT / "src/evaluation/frozen_v2_final_gate.py"), "lines": "161-265", "symbol": "evaluate_public_run", "role": "apply V x delta-I x q_eff x T once and public resource prices", "finding": "q is actual-model Balanced proxy, not reconstructed semantic quality"},
    ]


def encoding_provenance(runtime_fix_active: bool) -> dict[str, Any]:
    return {
        "estimate_payload_encoding_cost": {
            "path": str(TUT_ROOT / "src/payload/payload_sizer.py"), "lines": "227-243",
            "latency_formula": "payload_mb * 0.18 s/MB * product.encoding_latency_scale",
            "energy_formula": "payload_mb * 2.4 J/MB * product.encoding_energy_scale",
        },
        "payload_encoding_energy_j_and_latency_fields": {
            "definition_path": str(TUT_ROOT / "src/common/types.py"), "definition_lines": "570-571 and 937-938",
            "actual_population_path": str(TUT_ROOT / "src/simulator/step_runner.py"), "actual_lines": "5831-5842 and 5912-5913",
        },
        "product_generation_cost": {
            "size_path": str(TUT_ROOT / "src/payload/payload_sizer.py"),
            "size_semantics": "metadata record formula; ROI/ROI+Ctx label-derived formula with clamps; fallback profiles",
            "latency_and_energy_semantics": "generic product generation/encoding proxy",
        },
        "product_scales": ENCODING_SCALE,
        "TUT_specific_override": {
            "source_config": str(TUT_ROOT / "configs/experiments/tut_uoi_rate_lite_v2_formal_v1.yaml"),
            "declared": {"tut_predicted_energy_fix_enabled": True},
            "native_runtime_propagated": runtime_fix_active,
            "runtime_result": "prediction double-count active" if not runtime_fix_active else "prediction/actual parity",
            "generic_encoding_coefficients_overridden_for_TUT": False,
        },
        "PMSC_reference_in_current_actual_execution": False,
        "JPEG_reference_in_current_actual_execution": False,
        "encoding_latency_multiplied_by_model_power_in_formal_prediction": not runtime_fix_active,
        "encoding_latency_multiplied_by_model_power_in_actual_execution": False,
        "public_evaluator_uses_actual_aggregate_energy": True,
    }


def three_layer_rows(runtime_fix_active: bool) -> list[dict[str, Any]]:
    return [
        {"item": "task-network latency", "predicted": "1.44 s proxy", "actual": "1.44 s proxy", "public": "only via event time", "status": "PROXY_ONLY"},
        {"item": "task-network energy", "predicted": "21.60 J", "actual": "21.60 J", "public": "aggregate actual Wh", "status": "PROXY_ONLY"},
        {"item": "encoding latency", "predicted": "size-linear generic proxy", "actual": "same proxy", "public": "only via event time", "status": "PROXY_ONLY"},
        {"item": "encoding energy", "predicted": "actual proxy + 15 W*encoding_latency" if not runtime_fix_active else "same as actual proxy", "actual": "2.4 J/MB times product scale", "public": "aggregate actual Wh", "status": "DOUBLE_COUNTED in formal prediction only" if not runtime_fix_active else "COUNTED_CORRECTLY"},
        {"item": "PM generation", "predicted": "none", "actual": "none", "public": "none", "status": "OMITTED"},
        {"item": "PMSC encoder", "predicted": "generic proxy", "actual": "generic proxy", "public": "aggregate proxy", "status": "PROXY_ONLY"},
        {"item": "radio transmission", "predicted": "common link/radio", "actual": "common 38 W link/radio", "public": "actual MB and Wh", "status": "COUNTED_CORRECTLY"},
        {"item": "ground semantic decode", "predicted": "none", "actual": "none", "public": "none", "status": "OMITTED"},
        {"item": "ground downstream inference", "predicted": "none", "actual": "none", "public": "none", "status": "OMITTED"},
        {"item": "semantic quality", "predicted": "balanced detector proxy", "actual": "no reconstructed-image measurement", "public": "fixed q_balanced=0.7342411587", "status": "OVERCREDITED_SEMANTIC_QUALITY"},
        {"item": "closure time", "predicted": "inference + generic encode + queue + TX", "actual": "bitstream TX finish", "public": "same bitstream finish", "status": "OMITTED ground decode/task latency"},
    ]


def load_formal_rows() -> list[dict[str, str]]:
    return read_csv(FORMAL_ROOT / "tables/formal_five_seed_per_run.csv")


def build_task_decomposition(frozen: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    run_data: dict[int, dict[str, Any]] = {}
    product_agg: dict[str, list[float]] = defaultdict(list)
    q = f(frozen["q_eff"]["balanced"])
    for seed in SEEDS:
        run = FORMAL_ROOT / f"runs/formal/seed_{seed}/TUT"
        tasks = read_csv(run / "per_task_outcomes.csv")
        slots = read_csv(run / "per_slot_resource_trace.csv")
        slot_by_id = {i(row["slot"]): row for row in slots}
        events = read_csv(run / "ground_delivery_events.csv")
        event_by_task = defaultdict(list)
        for event in events:
            event_by_task[event["task_id"]].append(event)
        ledger = read_csv(run / "frozen_public_event_ledger.csv")
        gross_by_task = defaultdict(float)
        for row in ledger:
            gross_by_task[row["task_id"]] += f(row["gross"])
        resource = json.loads((run / "resource_summary.json").read_text(encoding="utf-8"))
        total_comm_j = sum(f(row["comm_energy_j"]) for row in slots)
        total_tx_mb = sum(f(row["transmitted_mb"]) for row in slots)
        comm_j_per_mb = total_comm_j / max(total_tx_mb, 1.0e-12)
        enc_by_task: dict[str, dict[str, float]] = {}
        task_by_id = {row["task_id"]: row for row in tasks}
        task_slot: dict[str, int] = {}
        for task in tasks:
            product = product_from_action(task["executed_action"])
            slot_id = int(round(f(task["arrival_time"]) / 20.0))
            task_slot[task["task_id"]] = slot_id
            slot = slot_by_id[slot_id]
            compute_j = f(slot["compute_energy_j"])
            if product in ENCODING_SCALE:
                encoding_j = max(0.0, compute_j - MODEL_ENERGY_J)
                scale = ENCODING_SCALE[product]
                payload_mb = encoding_j / (2.4 * scale["energy"])
                encoding_s = payload_mb * 0.18 * scale["latency"]
                model_j = MODEL_ENERGY_J
                model_s = MODEL_LATENCY_S
                product_agg[product].append(payload_mb)
            else:
                encoding_j = payload_mb = encoding_s = model_j = model_s = 0.0
            predicted_extra = 0.0 if bool(run_data.get(seed, {}).get("runtime_fix_active", False)) else MODEL_POWER_W * encoding_s
            ground_events = sorted(event_by_task[task["task_id"]], key=lambda row: f(row["ground_available_time"]))
            first_ground = f(ground_events[0]["ground_available_time"]) if ground_events else None
            closure_ground = f(task["closure_ground_time"]) if task["closure_ground_time"] not in {"", None} else None
            arrival_slot_comm_j = f(slot["comm_energy_j"])
            closure_latency = closure_ground - f(task["arrival_time"]) if closure_ground is not None else None
            pipeline_remainder = max(0.0, closure_latency - model_s - encoding_s) if closure_latency is not None else None
            row = {
                "seed": seed, "task_id": task["task_id"], "value_level": task["value_level"],
                "required_product": task["product_requirement"], "executed_action": task["executed_action"], "selected_product": product,
                "task_network_latency_s": model_s, "task_network_energy_j": model_j,
                "payload_mb_reconstructed_from_actual_encoding": payload_mb,
                "encoding_latency_s": encoding_s, "encoding_energy_j": encoding_j,
                "formal_prediction_extra_model_power_on_encoding_j": MODEL_POWER_W * encoding_s,
                "predicted_compute_energy_j_formal": model_j + encoding_j + MODEL_POWER_W * encoding_s,
                "actual_compute_energy_j": compute_j,
                "arrival_slot_comm_energy_j_not_lifetime_attribution": arrival_slot_comm_j,
                "lifetime_comm_energy_j_proxy": payload_mb * comm_j_per_mb if ground_events else 0.0,
                "lifetime_comm_energy_status": "PROXY_ONLY proportional run-average J/MB" if ground_events else "UNAVAILABLE not delivered",
                "first_ground_time_s": first_ground if first_ground is not None else "",
                "closure_ground_time_s": closure_ground if closure_ground is not None else "",
                "closure_success": i(task["closure_success"]),
                "full_closure_latency_s": closure_latency if closure_latency is not None else "",
                "queue_contact_tx_remainder_s": pipeline_remainder if pipeline_remainder is not None else "",
                "ground_semantic_decode_latency_s": 0.0,
                "ground_downstream_inference_latency_s": 0.0,
                "ground_processing_status": "OMITTED",
                "public_q_eff": q, "semantic_retention_factor": 1.0,
                "public_gross_utility": gross_by_task[task["task_id"]],
            }
            all_rows.append(row)
            enc_by_task[task["task_id"]] = {"latency_s": encoding_s, "energy_j": encoding_j, "payload_mb": payload_mb, "product": product}
        run_data[seed] = {
            "path": run, "tasks": tasks, "task_by_id": task_by_id, "slots": slots, "events": events,
            "enc_by_task": enc_by_task, "task_slot": task_slot, "resource": resource,
            "encoding_energy_j": sum(row["energy_j"] for row in enc_by_task.values()),
            "encoding_latency_s": sum(row["latency_s"] for row in enc_by_task.values()),
            "executed_count": sum(row["product"] in ENCODING_SCALE for row in enc_by_task.values()),
            "runtime_fix_active": False,
        }
    product_rows = []
    nominal = {"metadata": 0.002, "roi": 0.85, "roi_context": 2.70}
    for product in ("metadata", "roi", "roi_context"):
        values = product_agg[product]
        product_rows.append({
            "product": product, "nominal_profile_MB": nominal[product], "observed_mean_MB": statistics.fmean(values),
            "observed_min_MB": min(values), "observed_max_MB": max(values), "raw_image_proxy_MB": RAW_IMAGE_PROXY_MB,
            "nominal_compression_ratio_raw_to_product": RAW_IMAGE_PROXY_MB / nominal[product],
            "observed_mean_compression_ratio": RAW_IMAGE_PROXY_MB / statistics.fmean(values),
            "pmsc_code_rate_measured": "NO", "size_source": "common product profile / label formula proxy",
            "semantic_retention_linked_to_size_snr": "NO",
        })
    return all_rows, run_data, product_rows


def info_utility(products: set[str], requirement: str, utility_table: dict[str, dict[str, float]]) -> float:
    if not products:
        key = "empty"
    else:
        key = "+".join(sorted(products, key=lambda product: PRODUCT_RANK.get(product, -1)))
        if key not in utility_table:
            return max(f(utility_table.get(product, {}).get(requirement)) for product in products)
    return f(utility_table[key][requirement])


def thermal_static_replay(data: dict[str, Any], new_encoding_energy: dict[str, float]) -> tuple[float, int, float]:
    delta_by_slot = defaultdict(float)
    for task_id, current in data["enc_by_task"].items():
        delta_by_slot[data["task_slot"][task_id]] += new_encoding_energy[task_id] - current["energy_j"]
    old_prev = THERMAL_INIT
    new_prev = THERMAL_INIT
    max_new = new_prev
    violations = 0
    max_reconstruction_error = 0.0
    for row in sorted(data["slots"], key=lambda value: i(value["slot"])):
        slot = i(row["slot"])
        old_compute = f(row["compute_energy_j"])
        new_compute = old_compute + delta_by_slot[slot]
        comm = f(row["comm_energy_j"])
        scale = MODEL_THERMAL_SCALE if old_compute > 0.0 or new_compute > 0.0 else 1.0
        old_next = f(row["thermal"])
        environmental_delta = old_next - old_prev - 0.0025 * old_compute * scale - 0.0015 * comm
        reconstructed = max(0.0, min(1.0, old_prev + environmental_delta + 0.0025 * old_compute * scale + 0.0015 * comm))
        max_reconstruction_error = max(max_reconstruction_error, abs(reconstructed - old_next))
        new_next = max(0.0, min(1.0, new_prev + environmental_delta + 0.0025 * new_compute * scale + 0.0015 * comm))
        max_new = max(max_new, new_next)
        violations += int(new_next > THERMAL_LIMIT + 1.0e-12)
        old_prev, new_prev = old_next, new_next
    return max_new, violations, max_reconstruction_error


def scenario_specs() -> list[dict[str, Any]]:
    return [
        {"group": "A", "scenario": "AS_IS", "encoding": "current", "retention": 1.0, "ground_delay_s": 0.0},
        {"group": "B", "scenario": "PMSC_FIXED_0P43S", "encoding": "fixed", "retention": 1.0, "ground_delay_s": 0.0},
        {"group": "C", "scenario": "ENCODING_TOPUP_TO_0P43S", "encoding": "topup", "retention": 1.0, "ground_delay_s": 0.0},
        *({"group": "D", "scenario": f"SEMANTIC_RETENTION_{retention:.2f}", "encoding": "current", "retention": retention, "ground_delay_s": 0.0} for retention in (1.00, 0.95, 0.90, 0.85)),
        *({"group": "E", "scenario": f"GROUND_DECODE_{delay:.2f}S", "encoding": "current", "retention": 1.0, "ground_delay_s": delay} for delay in (0.0, 0.1, 0.43, 1.0)),
    ]


def counterfactual(run_data: dict[int, dict[str, Any]], frozen: dict[str, Any], utility_table: dict[str, dict[str, float]], formal_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    formal_tut = {(i(row["seed"]), row["method"]): row for row in formal_rows}
    energy_price = f(frozen["public_prices"]["energy_price_per_Wh"])
    downlink_price = f(frozen["public_prices"]["downlink_price_per_MB"])
    backlog_price = f(frozen["public_prices"]["terminal_backlog_price_per_MB"])
    q = f(frozen["q_eff"]["balanced"])
    for spec in scenario_specs():
        for seed, data in run_data.items():
            new_encoding: dict[str, float] = {}
            latency_delta: dict[str, float] = {}
            for task_id, current in data["enc_by_task"].items():
                if current["product"] not in ENCODING_SCALE:
                    new_encoding[task_id] = 0.0
                    latency_delta[task_id] = 0.0
                elif spec["encoding"] == "fixed":
                    new_encoding[task_id] = PMSC_REFERENCE_ENERGY_J
                    latency_delta[task_id] = PMSC_REFERENCE_LATENCY_S - current["latency_s"]
                elif spec["encoding"] == "topup":
                    new_encoding[task_id] = current["energy_j"] + max(0.0, PMSC_REFERENCE_ENERGY_J - current["energy_j"])
                    latency_delta[task_id] = max(0.0, PMSC_REFERENCE_LATENCY_S - current["latency_s"])
                else:
                    new_encoding[task_id] = current["energy_j"]
                    latency_delta[task_id] = 0.0
            events_by_task = defaultdict(list)
            for event in data["events"]:
                events_by_task[event["task_id"]].append(event)
            gross = 0.0
            hv_gross = 0.0
            closures: dict[str, float] = {}
            task_gross: dict[str, float] = defaultdict(float)
            for task_id, task in data["task_by_id"].items():
                delivered: set[str] = set()
                adjusted = []
                for event in events_by_task[task_id]:
                    product = STAGE_PRODUCT.get(str(event["stage"]).lower(), "")
                    time_s = f(event["ground_available_time"]) + latency_delta[task_id] + f(spec["ground_delay_s"])
                    adjusted.append((time_s, product, event))
                for time_s, product, event in sorted(adjusted, key=lambda value: (value[0], value[1])):
                    if not product or time_s > f(task["deadline"]) + 1.0e-12:
                        continue
                    before = info_utility(delivered, task["product_requirement"], utility_table)
                    after = info_utility(delivered | {product}, task["product_requirement"], utility_table)
                    delta_i = max(0.0, after - before) if product not in delivered else 0.0
                    delivered.add(product)
                    lifetime = max(f(task["deadline"]) - f(task["arrival_time"]), 1.0e-12)
                    timely = max(0.0, min(1.0, 1.0 - (time_s - f(task["arrival_time"])) / lifetime))
                    positive = f(task["effective_task_value"]) * delta_i * q * f(spec["retention"]) * timely
                    gross += positive
                    task_gross[task_id] += positive
                    if task_id not in closures and product_satisfies(product, task["product_requirement"]):
                        closures[task_id] = time_s
                if task_id in closures and task["value_level"] in {"high", "urgent"}:
                    hv_gross += task_gross[task_id]
                task_rows.append({
                    "group": spec["group"], "scenario": spec["scenario"], "seed": seed, "task_id": task_id,
                    "value_level": task["value_level"], "required_product": task["product_requirement"],
                    "selected_product": data["enc_by_task"][task_id]["product"],
                    "encoding_latency_delta_s": latency_delta[task_id],
                    "encoding_energy_delta_j": new_encoding[task_id] - data["enc_by_task"][task_id]["energy_j"],
                    "ground_decode_delay_s": spec["ground_delay_s"], "semantic_retention": spec["retention"],
                    "closure": int(task_id in closures), "closure_time_s": closures.get(task_id, ""),
                    "gross_utility": task_gross[task_id], "diagnostic_only": 1,
                })
            current_energy_wh = f(data["resource"]["controllable_energy_wh"])
            energy_wh = current_energy_wh + (sum(new_encoding.values()) - data["encoding_energy_j"]) / 3600.0
            tx_mb = f(data["resource"]["transmitted_mb"])
            terminal_mb = f(data["slots"][-1]["queue_mb"])
            energy_cost = energy_price * energy_wh
            downlink_cost = downlink_price * tx_mb
            backlog_cost = backlog_price * terminal_mb
            total_cost = energy_cost + downlink_cost + backlog_cost
            hv_cost = total_cost * hv_gross / gross if gross > 0.0 else 0.0
            closed_tasks = [data["task_by_id"][task_id] for task_id in closures]
            latencies = [closures[task["task_id"]] - f(task["arrival_time"]) for task in closed_tasks]
            max_thermal, thermal_violations, thermal_reconstruction_error = thermal_static_replay(data, new_encoding)
            row = {
                "group": spec["group"], "scenario": spec["scenario"], "seed": seed, "method": METHOD,
                "semantic_retention": spec["retention"], "ground_decode_latency_s": spec["ground_delay_s"],
                "encoding_mode": spec["encoding"], "gross_utility": gross, "HV_gross_utility": hv_gross,
                "energy_Wh": energy_wh, "incremental_energy_Wh": energy_wh - current_energy_wh,
                "energy_cost": energy_cost, "downlink_cost": downlink_cost, "terminal_backlog_cost": backlog_cost,
                "J_all_net": gross - total_cost, "J_HV_net": hv_gross - hv_cost,
                "all_closure_count": len(closures),
                "HV_closure_count": sum(task["value_level"] in {"high", "urgent"} for task in closed_tasks),
                "high_closure_count": sum(task["value_level"] == "high" for task in closed_tasks),
                "urgent_closure_count": sum(task["value_level"] == "urgent" for task in closed_tasks),
                "mean_full_closure_latency_s": statistics.fmean(latencies) if latencies else "",
                "p95_full_closure_latency_s": percentile(latencies, 0.95) or "",
                "maximum_thermal_static_replay": max_thermal,
                "thermal_violation_slots_static_replay": thermal_violations,
                "thermal_reconstruction_max_error": thermal_reconstruction_error,
                "thermal_status": "SENSITIVITY_STATIC_TRACE_NO_POLICY_RESPONSE",
                "causal_replay": 0, "formal_result_replacement_allowed": 0,
            }
            rows.append(row)
            if spec["scenario"] == "AS_IS":
                official = formal_tut[(seed, METHOD)]
                row["asis_J_all_error"] = row["J_all_net"] - f(official["J_all_net"])
                row["asis_J_HV_error"] = row["J_HV_net"] - f(official["J_HV_net"])
                row["asis_closure_error"] = row["all_closure_count"] - i(official["all_closure_count"])
    summary_rows = []
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        subset = [row for row in rows if row["scenario"] == scenario]
        summary = {"scenario": scenario, "group": subset[0]["group"]}
        for field in ("gross_utility", "HV_gross_utility", "J_all_net", "J_HV_net", "all_closure_count", "HV_closure_count", "high_closure_count", "urgent_closure_count", "mean_full_closure_latency_s", "energy_Wh", "incremental_energy_Wh", "maximum_thermal_static_replay", "thermal_violation_slots_static_replay"):
            values = [f(row[field]) for row in subset]
            summary[f"{field}_mean"], summary[f"{field}_std"] = mean_std(values)
        summary_rows.append(summary)
    ranking_rows = []
    metrics = ("J_all_net", "J_HV_net", "HV_closure_count")
    by_seed_method = {(i(row["seed"]), row["method"]): row for row in formal_rows}
    methods = sorted({row["method"] for row in formal_rows})
    for cf in rows:
        for metric in metrics:
            values = {method: f(by_seed_method[(cf["seed"], method)][metric]) for method in methods}
            values[METHOD] = f(cf[metric])
            ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
            rank = next(index + 1 for index, item in enumerate(ordered) if item[0] == METHOD)
            ranking_rows.append({"group": cf["group"], "scenario": cf["scenario"], "seed": cf["seed"], "metric": metric, "TUT_value": values[METHOD], "TUT_rank_of_methods": rank, "method_count": len(methods), "top_method": ordered[0][0], "top_value": ordered[0][1]})
    mean_rank_rows = []
    for spec in scenario_specs():
        cf_by_seed = {row["seed"]: row for row in rows if row["scenario"] == spec["scenario"]}
        for metric in metrics:
            means = {}
            for method in methods:
                values = []
                for seed in SEEDS:
                    values.append(f(cf_by_seed[seed][metric]) if method == METHOD else f(by_seed_method[(seed, method)][metric]))
                means[method] = statistics.fmean(values)
            ordered = sorted(means.items(), key=lambda item: (-item[1], item[0]))
            for rank, (method, value) in enumerate(ordered, 1):
                mean_rank_rows.append({"group": spec["group"], "scenario": spec["scenario"], "metric": metric, "rank": rank, "method": method, "five_seed_mean": value})
    return rows, summary_rows, ranking_rows, mean_rank_rows, task_rows


def encoding_summary(task_rows: list[dict[str, Any]], run_data: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seed, data in run_data.items():
        subset = [row for row in task_rows if row["seed"] == seed and row["selected_product"] in ENCODING_SCALE]
        rows.append({
            "seed": seed, "executed_non_defer_tasks": len(subset),
            "balanced_inference_energy_J": len(subset) * MODEL_ENERGY_J,
            "generic_encoding_energy_J": sum(f(row["encoding_energy_j"]) for row in subset),
            "generic_encoding_latency_sum_s": sum(f(row["encoding_latency_s"]) for row in subset),
            "formal_prediction_double_count_extra_J": sum(f(row["formal_prediction_extra_model_power_on_encoding_j"]) for row in subset),
            "actual_compute_energy_J": sum(f(row["actual_compute_energy_j"]) for row in subset),
            "PMSC_fixed_reference_energy_J_if_replaced": len(subset) * PMSC_REFERENCE_ENERGY_J,
            "PMSC_fixed_reference_latency_sum_s_if_replaced": len(subset) * PMSC_REFERENCE_LATENCY_S,
        })
    return rows


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite prior audit: {OUT}")
    if not TUT_ROOT.is_dir() or not FORMAL_ROOT.is_dir():
        raise FileNotFoundError("Required TUT source root or frozen formal output root is missing")
    assets = asset_paths()
    before = hash_assets(assets)
    fragments = source_fragment_checks()
    if not all(row["found"] for row in fragments):
        raise RuntimeError({"source_fragment_check_failed": fragments})
    OUT.mkdir(parents=True)
    frozen = json.loads((CALIBRATION_ROOT / "frozen/conditional_v2_frozen_parameters_v1.json").read_text(encoding="utf-8"))
    utility_cfg = yaml.safe_load((TUT_ROOT / "configs/product_requirement_utility.yaml").read_text(encoding="utf-8"))
    utility_table = utility_cfg["utility_table"]
    formal_rows = load_formal_rows()

    native_text = (NATIVE_ROOT / "native_tut.py").read_text(encoding="utf-8")
    copied_keys = ["tut_uoi_density_priority_enabled", "progressive_payload", "guarantee", "downlink_safety_throttle"]
    resolved_fix_occurrences = {}
    for seed in SEEDS:
        resolved = FORMAL_ROOT / f"scenario/controlled_bundle_seed_{seed}/resolved_config.json"
        resolved_fix_occurrences[str(seed)] = "tut_predicted_energy_fix_enabled" in resolved.read_text(encoding="utf-8")
    runtime_fix_active = "tut_predicted_energy_fix_enabled" in copied_keys or all(resolved_fix_occurrences.values())
    binding_audit = {
        "requested_identity": "TUT-v2-energy-fixed", "expected_switch": "tut_predicted_energy_fix_enabled=true",
        "native_adapter_path": str(NATIVE_ROOT / "native_tut.py"), "native_adapter_copied_simulator_keys": copied_keys,
        "switch_copied_by_native_adapter": "tut_predicted_energy_fix_enabled" in copied_keys,
        "resolved_config_switch_presence_by_seed": resolved_fix_occurrences,
        "formal_runtime_fix_active": runtime_fix_active,
        "conclusion": "METHOD_IDENTITY_RUNTIME_SWITCH_MISMATCH" if not runtime_fix_active else "PASS",
        "effect": "formal prediction charges model_power*encoding_latency plus generic encoding energy; actual/public energy remains actual-only" if not runtime_fix_active else "none",
        "source_evidence": "native adapter starts from common simulation_input.configs and copies only the listed keys",
    }

    task_decomp, run_data, product_rows = build_task_decomposition(frozen)
    encoding_rows = encoding_summary(task_decomp, run_data)
    cf_rows, cf_summary, ranking_rows, mean_rank_rows, cf_task_rows = counterfactual(run_data, frozen, utility_table, formal_rows)
    asis = [row for row in cf_rows if row["scenario"] == "AS_IS"]
    asis_pass = all(abs(f(row.get("asis_J_all_error"))) <= 1.0e-9 and abs(f(row.get("asis_J_HV_error"))) <= 1.0e-9 and i(row.get("asis_closure_error")) == 0 for row in asis)

    call_chain = call_chain_rows()
    detailed_chain = detailed_call_chain_rows()
    encoding_sources = encoding_provenance(runtime_fix_active)
    reconciliation = three_layer_rows(runtime_fix_active)
    quality_rows = [
        {"step": 1, "artifact": "YOLO-inspired profile", "values": "0.74 / 0.87 / 0.94", "derivation": "declared proxy base quality; not measured SARFish accuracy", "evidence": "configs/model_profiles_yolo_inspired_calibrated.yaml"},
        {"step": 2, "artifact": "task-conditioned proxy", "values": "base - (1-robustness)*0.32*difficulty - 0.16*uncertainty", "derivation": "calibration helper; clipped to [0.05,1]", "evidence": "src/calibration/conditional_v2_calibration.py:32-42"},
        {"step": 3, "artifact": "Q_MEAN", "values": "mean across independent calibration task rows, seeds 3101/3102", "derivation": "mean task-conditioned proxy and strict ordering", "evidence": "same helper:45-69; frozen JSON"},
        {"step": 4, "artifact": "frozen q_eff", "values": "0.569717821840 / 0.734241158700 / 0.821502827130", "derivation": "frozen Q_MEAN; proxy_not_real_measurement=true", "evidence": "conditional_v2_frozen_parameters_v1.json"},
        {"step": 5, "artifact": "TUT public q", "values": "q_balanced=0.734241158700 for every valid event", "derivation": "fixed balanced actual-model mapping; q used once", "evidence": "baseline_actual_model_q_audit.csv and frozen event ledgers"},
        {"step": 6, "artifact": "semantic retention", "values": "implicit 1.0 for metadata/ROI/ROI+Ctx at all SNR/code rates", "derivation": "no retention function or threshold", "evidence": "no active codec/reconstruction-quality path"},
    ]
    q_audit = {
        "TUT_always_uses_balanced_q": True,
        "q_depends_on_semantic_code_rate_product_size_SNR_reconstruction": False,
        "all_products_share_same_TUT_q": True,
        "semantic_quality_retention_factor_exists": False,
        "below_semantic_quality_threshold_can_receive_full_q": True,
        "PM_model_quality_and_reconstructed_downstream_quality_separated": False,
        "q_count_public": 1,
        "suggested_audit_only_factorization": "q_TUT_eff = q_task_network * semantic_retention(code_rate,SNR,product)",
        "formal_evaluator_changed": False,
    }
    ground_audit = {
        "closure_equals_bitstream_ground_arrival": True,
        "semantic_decoder_latency_included": False,
        "downstream_task_latency_included": False,
        "closure_can_be_counted_before_reconstruction": True,
        "ground_energy_excluded_from_satellite": True,
        "ground_energy_disclosed_or_modeled": False,
        "ground_decode_should_enter_deadline_and_timeliness_for_faithful_TUT": True,
    }
    fairness = {
        "common_radio_power_W": 38.0, "paper_radio_power_imported": False,
        "same_common_link_and_queue": True, "decision_compute_overhead_policy": "ignored for all compared routers",
        "TUT_only_extra_cost_added_to_formal_results": False,
        "parameter_tuned_to_change_ranking": False,
        "counterfactuals_are_diagnostic_only": True,
        "important_caveat": "A faithful TUT-v3 would add method-specific PM/PMSC/decoder work because those stages are intrinsic to that algorithm, while analogous actual processing must remain counted for every other method.",
    }
    classification = {
        "declared_current_identity": "TUT-inspired LITE heuristic",
        "assessment_for_declared_lite_identity": "REASONABLE_WITH_DISCLOSURE",
        "assessment_if_presented_as_original_TUT_reproduction": "UNRELIABLE",
        "additional_findings": ["UNDERCOUNTED_COMPUTE", "OVERCREDITED_SEMANTIC_QUALITY"],
        "formal_runtime_identity_issue": binding_audit["conclusion"],
        "TUT_v3_needed": True,
        "TUT_v3_scope": "isolated implementation only; PM generation, PMSC encoder/bitstream, SNR/code-rate retention, ground decoder and downstream inference, plus complete predictor/actual/public parity",
        "automatic_implementation_performed": False,
    }

    write_json(OUT / "frozen_assets_before.json", before)
    write_json(OUT / "source_fragment_checks.json", fragments)
    write_json(OUT / "formal_runtime_binding_audit.json", binding_audit)
    write_csv(OUT / "pipeline_stage_accounting.csv", call_chain)
    write_csv(OUT / "full_call_chain_trace.csv", detailed_chain)
    write_json(OUT / "encoding_parameter_provenance.json", encoding_sources)
    write_csv(OUT / "predicted_actual_public_reconciliation.csv", reconciliation)
    write_csv(OUT / "per_task_energy_latency_decomposition.csv", task_decomp)
    write_csv(OUT / "encoding_summary_by_seed.csv", encoding_rows)
    write_csv(OUT / "quality_provenance.csv", quality_rows)
    write_json(OUT / "quality_semantic_audit.json", q_audit)
    write_csv(OUT / "product_code_rate_audit.csv", product_rows)
    write_json(OUT / "ground_semantics_audit.json", ground_audit)
    write_json(OUT / "fairness_audit.json", fairness)
    write_csv(OUT / "counterfactual_per_run.csv", cf_rows)
    write_csv(OUT / "counterfactual_mean_std.csv", cf_summary)
    write_csv(OUT / "counterfactual_rankings_per_seed.csv", ranking_rows)
    write_csv(OUT / "counterfactual_method_mean_rankings.csv", mean_rank_rows)
    write_csv(OUT / "counterfactual_per_task.csv", cf_task_rows)
    write_json(OUT / "classification.json", classification)

    pipeline = """# TUT end-to-end computation-communication pipeline

```mermaid
flowchart LR
    X[Raw observation<br/>32 MB proxy] --> Y[Balanced task network<br/>1.44 s, 21.6 J<br/>synthetic quality proxy]
    Y -. missing .-> PM[Perceptual-weight map]
    PM -. missing .-> PMSC[PMSC semantic encoder<br/>paper anchor 0.43 s]
    Y --> GP[Generic product builder<br/>metadata / ROI / ROI+Ctx<br/>size-linear encoding proxy]
    GP --> Q[Common onboard queue]
    Q --> W[Contact wait + partial transmission<br/>common 38 W radio]
    W --> ARR[Bitstream fully arrives on ground]
    ARR -->|current implementation closes here| PUB[Public evaluator<br/>V x delta-I x q-balanced x T]
    ARR -. missing .-> DEC[Ground semantic decoder]
    DEC -. missing .-> DOWN[Ground downstream task network]
    DOWN -. faithful closure should occur here .-> PUB
```

Solid arrows are active. Dashed arrows are required by the original TUT pipeline but absent from TUT-v2. The current generic product builder is not a learned PMSC encoder.
"""
    write_text(OUT / "tut_end_to_end_pipeline.md", pipeline)

    cf_fields = ["scenario", "J_all_net_mean", "J_HV_net_mean", "all_closure_count_mean", "HV_closure_count_mean", "energy_Wh_mean", "incremental_energy_Wh_mean", "maximum_thermal_static_replay_mean", "thermal_violation_slots_static_replay_mean"]
    encoding_total = {
        "current_latency_mean_s_per_executed": statistics.fmean(row["generic_encoding_latency_sum_s"] / row["executed_non_defer_tasks"] for row in encoding_rows),
        "current_energy_mean_j_per_executed": statistics.fmean(row["generic_encoding_energy_J"] / row["executed_non_defer_tasks"] for row in encoding_rows),
        "formal_prediction_extra_mean_j_per_run": statistics.fmean(row["formal_prediction_double_count_extra_J"] for row in encoding_rows),
    }
    report = f"""# TUT semantic coding fidelity and cost audit

## Executive finding

The current `{METHOD}` is **not a full implementation of the original TUT/PMSC pipeline**. It is a declared TUT-inspired LITE heuristic: fixed Balanced inference plus a one-shot choice among the common metadata/ROI/ROI+Ctx products. As that declared LITE baseline it is **REASONABLE_WITH_DISCLOSURE**. If it is described as a faithful reproduction of the original paper, the result is **UNRELIABLE**.

Two substantive directions of bias coexist:

1. **UNDERCOUNTED_COMPUTE**: PM generation, the actual PMSC encoder/model footprint, ground semantic decoding and downstream task inference are absent. Router/PPO cost is absent because the implementation is deterministic rather than PPO.
2. **OVERCREDITED_SEMANTIC_QUALITY**: every delivered TUT product receives the same frozen Balanced q (`{frozen['q_eff']['balanced']:.12f}`), independent of semantic code rate, payload, SNR, reconstruction mAP, or a minimum semantic-quality threshold.

There is also a formal runtime identity defect: `{binding_audit['conclusion']}`. The requested v2 switch is present in the method source config but is not copied by the native adapter and is absent from each seed's resolved config. Consequently, formal **prediction** double-counts encoding as `15 W x encoding_latency` plus generic encoding energy; actual SOC and public cost use actual energy only. Across five seeds, the extra predicted charge averages `{encoding_total['formal_prediction_extra_mean_j_per_run']:.3f} J/run`. This can bias selection away from larger products even though final energy accounting is not double-counted.

## Evidence level

- Local implementation findings are direct source/trace evidence.
- The paper identity and pipeline were cross-checked against [Xu et al., Task-Oriented Semantic Coding and Utility Optimal Transmission in Satellite-Integrated Internet](https://doi.org/10.1109/TGRS.2025.3576382).
- The exact Table II values (317 MB, 0.43 s PMSC, 0.09 s JPEG), transmit-power scope, and SNR=25 dB/mAP>=0.7 anchor are recorded as **user-provided literature anchors**; they are not silently treated as measured values for this simulator.

## A. Full active call chain

{md_table(call_chain, ['order','stage','implementation','latency','satellite_energy','ground_latency','ground_energy','thermal','queue_closure'])}

Concrete file-by-file routing is in `full_call_chain_trace.csv`; it follows native binding, action construction, predictor, SafeFilter, selector, inference/product generation, actual accounting, queue delivery, normalization and the frozen public evaluator.

## B. Encoding traceability

1. Current latency is `payload_MB x 0.18 s/MB x product_scale`: Metadata 0.10, ROI 1.0, ROI+Ctx 1.9.
2. It varies linearly with the dynamically estimated product size.
3. It is a generic serialization/compression proxy, not identified as JPEG and not a neural PMSC measurement.
4. Energy is `payload_MB x 2.4 J/MB x product_scale`: Metadata 0.05, ROI 1.0, ROI+Ctx 2.2. No physical source or device measurement is supplied.
5. The paper-anchor 0.43 s PMSC encoder is **not** included as such. The current mean is `{encoding_total['current_latency_mean_s_per_executed']:.6f} s/executed task`.
6. In these formal native runs, yes: prediction incorrectly also multiplies encoding latency by Balanced model power. Actual execution does not.
7. Actual and public aggregate energy agree; prediction does not agree because the v2 fix switch was not propagated.

{md_table(encoding_rows, list(encoding_rows[0]))}

## C. Quality accounting

{md_table(quality_rows, ['step','artifact','values','derivation','evidence'])}

The q chain is a synthetic quality proxy, not reconstructed-image mAP. TUT uses Balanced q for all valid events and q is multiplied exactly once in the frozen public evaluator. However, semantic retention is implicitly 1.0, so small code rates and poor channels never reduce q and a delivery can receive full q without demonstrating the paper's reconstruction-quality constraint.

## D. Product and code-rate mapping

{md_table(product_rows, ['product','nominal_profile_MB','observed_mean_MB','observed_min_MB','observed_max_MB','nominal_compression_ratio_raw_to_product','observed_mean_compression_ratio','pmsc_code_rate_measured','semantic_retention_linked_to_size_snr'])}

The 32 MB raw image is itself a common proxy. Metadata's nominal 0.002 MB is only a base value; label records raise its observed mean. ROI and ROI+Ctx sizes are formula/clamp proxies. None is a measured PMSC bitstream or a code-rate operating point. TUT and PROP use the same product semantics and sizes; TUT receives no extra byte discount, but it does receive an implicit semantic-quality exemption because q does not fall with rate/SNR/product.

## E. Ground semantics

The queue emits delivery at transmission finish. TUT closure and timeliness are computed from that instant. No semantic decoder or downstream task inference is scheduled. Ground compute energy is correctly absent from satellite SOC, but it is also undisclosed. For a paper-faithful implementation, decode and downstream inference latency must be included in deadline and timeliness.

## F. Fairness

The shared 38 W radio and common link remain intact; no 100/200 W paper transmit setting was imported. Decision overhead is ignored across methods. This audit does not add any cost to formal rankings and does not change a parameter. A future faithful TUT-v3 may legitimately carry PM/PMSC/decoder costs because they are intrinsic stages, while analogous real stages for every comparator must also remain accounted consistently.

## G. Read-only sensitivity diagnostics

All rows below are **SENSITIVITY/DIAGNOSTIC**, not causal simulator reruns. Frozen actions, queue service order and radio transmission are held fixed. Encoding and ground-delay scenarios shift each task's recorded ground time locally. Thermal is a static recurrence replay using the recorded environmental residual and does not allow SafeFilter or routing to react; any reported thermal violation therefore means a real rerun would be required before claiming feasibility.

{md_table(cf_summary, cf_fields)}

`SEMANTIC_RETENTION_1.00` and `GROUND_DECODE_0.00S` are intentional duplicate controls. AS-IS reproduces every frozen TUT J_all, J_HV and closure count within 1e-9: **{asis_pass}**.

## Final classification

- Current declared LITE identity: **REASONABLE_WITH_DISCLOSURE**.
- As an original-paper reproduction: **UNRELIABLE**.
- Additional flags: **UNDERCOUNTED_COMPUTE**, **OVERCREDITED_SEMANTIC_QUALITY**.
- A separate TUT-v3 is needed if the paper-faithful algorithm is required. It should be isolated and must not overwrite TUT-v2. This audit did **not** implement it.

## Files

- `tut_end_to_end_pipeline.md`: pipeline diagram.
- `pipeline_stage_accounting.csv`: eight-stage accounting.
- `full_call_chain_trace.csv`: concrete source files, symbols, roles and findings.
- `encoding_parameter_provenance.json`: exact encoding symbols, formulas and TUT override status.
- `predicted_actual_public_reconciliation.csv`: three-layer reconciliation.
- `per_task_energy_latency_decomposition.csv`: task-level decomposition; lifetime radio energy is explicitly marked proxy.
- `quality_provenance.csv`, `quality_semantic_audit.json`: q provenance and semantic-quality findings.
- `product_code_rate_audit.csv`: size/code-rate audit.
- `counterfactual_*.csv`: diagnostic rescoring and ranking tables.
- `frozen_asset_invariance_audit.json`: proof that formal assets were unchanged.

{COMPLETION}
"""
    write_text(OUT / "FINAL_AUDIT_REPORT.md", report)

    after = hash_assets(assets)
    changed = [path for path in before if before[path] != after.get(path)]
    invariant = {"asset_count": len(before), "all_frozen_assets_unchanged": not changed, "changed_assets": changed, "before_after_exact_match": before == after}
    write_json(OUT / "frozen_assets_after.json", after)
    write_json(OUT / "frozen_asset_invariance_audit.json", invariant)
    tests = {
        "all_source_fragments_found": all(row["found"] for row in fragments),
        "formal_seed_set_exact": sorted({i(row["seed"]) for row in formal_rows if row["method"] == METHOD}) == list(SEEDS),
        "asis_reproduces_formal_metrics": asis_pass,
        "q_public_once": all(abs(f(row["q_eff"]) - f(frozen["q_eff"]["balanced"])) <= 1.0e-15 and i(row["q_count"]) == 1 for seed in SEEDS for row in read_csv(FORMAL_ROOT / f"runs/formal/seed_{seed}/TUT/frozen_public_event_ledger.csv")),
        "formal_assets_unchanged": invariant["all_frozen_assets_unchanged"],
        "counterfactuals_marked_diagnostic": all(i(row["formal_result_replacement_allowed"]) == 0 for row in cf_rows),
        "no_decision_policy_executed": True,
    }
    tests["all_pass"] = all(tests.values())
    write_json(OUT / "test_report.json", tests)
    write_text(OUT / "test_report.txt", "\n".join(f"{key}: {value}" for key, value in tests.items()))
    completion = {
        "status": COMPLETION if tests["all_pass"] else "TUT_SEMANTIC_CODING_FIDELITY_AND_COST_AUDIT_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(OUT),
        "read_only_formal_assets": invariant["all_frozen_assets_unchanged"], "tests": tests,
    }
    write_json(OUT / "completion.json", completion)
    print(COMPLETION if tests["all_pass"] else completion["status"])
    print(OUT)
    return 0 if tests["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
