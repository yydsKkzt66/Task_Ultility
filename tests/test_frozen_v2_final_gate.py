from pathlib import Path

from src.evaluation.frozen_v2_final_gate import (
    choose_unused_seeds,
    delivered_satisfies,
    model_normalize,
    product_normalize,
    product_satisfies,
    scan_used_seeds,
)


def test_product_aliases_and_order_are_centralized():
    assert product_normalize("Evidence") == "roi_context"
    assert product_normalize("ROI+Context") == "roi_context"
    assert product_satisfies("ROI+Context", "ROI")
    assert not product_satisfies("metadata", "evidence")
    assert delivered_satisfies("metadata+roi_context", "evidence")


def test_baseline_model_aliases_use_actual_architecture():
    assert model_normalize("ground_high_accuracy") == "high_accuracy"
    assert model_normalize("local") == "high_accuracy"


def test_seed_scan_and_deterministic_selection(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "outputs" / "seed_10001").mkdir(parents=True)
    (tmp_path / "configs" / "x.yaml").write_text("seeds: [10000, 2026]\n", encoding="utf-8")
    used, audit = scan_used_seeds(tmp_path)
    assert {10000, 10001, 2026} <= used
    assert choose_unused_seeds(used, 3, 10000) == [10002, 10003, 10004]
    assert audit["content_files_inspected"] == 1
