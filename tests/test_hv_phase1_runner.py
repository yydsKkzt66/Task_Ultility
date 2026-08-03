from pathlib import Path

from scripts.run_hv_phase1_seed2026 import PHASE1_REQUIRED_FILES, _b1_rows, verify_phase1_output
from src.common.config import load_yaml


def test_phase1_required_file_contract_is_bounded():
    assert "oracle_solver_results.csv" not in PHASE1_REQUIRED_FILES
    assert "deadline_sensitivity_seed2026.csv" not in PHASE1_REQUIRED_FILES
    assert "phase1_checkpoint_report.md" in PHASE1_REQUIRED_FILES


def test_verify_phase1_output_reports_missing_files(tmp_path: Path):
    result = verify_phase1_output(tmp_path)
    assert not result["passed"]
    assert set(result["missing_files"]) == set(PHASE1_REQUIRED_FILES)


def test_b1_export_contains_each_dominance_predicate():
    tasks = {"T1": {"task_id": "T1", "arrival_time_s": 0.0, "capture_time_s": 0.0,
                    "deadline_time_s": 100.0, "value_level": "high_value", "effective_task_value": 0.9,
                    "product_requirement": "evidence", "full_frame_information_capabilities": "evidence",
                    "task_manifest_hash": "fixture"}}
    trace = [{"task_id": "T1", "state": "ground_result_available", "ground_result_available": "1",
              "ground_result_available_time_s": "50", "ground_decode_latency_s": "1.5",
              "ground_inference_latency_s": "2.8", "ground_task_quality": "0.88",
              "reconstructed_image_quality_proxy": "0.96", "compressed_size_mb": "4",
              "capture_to_ground_latency_s": "50"}]
    rows, _ = _b1_rows(tasks, trace, "B1", "fixture", load_yaml(Path("configs/unified_ground_received_accounting.yaml")))
    assert {"full_frame_received", "ground_processing_success", "frame_contains_required_information",
            "retained_compression_quality", "quality_threshold", "quality_pass"} <= set(rows[0])
