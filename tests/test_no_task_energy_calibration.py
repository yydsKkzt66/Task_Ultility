import csv

from scripts.run_no_task_energy_calibration import run_no_task_calibration


def test_no_task_24h_calibration_outputs_sawtooth_not_monotonic_drop(tmp_path):
    metrics = run_no_task_calibration(output_dir=tmp_path, num_slots=4320, slot_seconds=20.0)

    assert (tmp_path / "no_task_soc_trajectory.csv").exists()
    assert (tmp_path / "no_task_energy_balance_by_orbit.csv").exists()
    assert (tmp_path / "no_task_calibration_report.md").exists()
    assert metrics["monotonic_drop_detected"] == 0.0
    assert metrics["sawtooth_reversal_count"] > 0
    assert metrics["fraction_time_SOC_below_20"] == 0.0

    with (tmp_path / "no_task_soc_trajectory.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4320
    assert any(float(row["soc_delta"]) < 0.0 for row in rows)
    assert any(float(row["soc_delta"]) > 0.0 for row in rows)
