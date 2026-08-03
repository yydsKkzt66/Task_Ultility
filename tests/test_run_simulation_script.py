import subprocess
import sys
import csv
from pathlib import Path


def test_run_simulation_script_generates_outputs(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_simulation.py"),
        "--output-dir",
        str(output_dir),
        "--num-slots",
        "12",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert completed.returncode == 0
    summary_path = output_dir / "summary.csv"
    summary_raw_path = output_dir / "summary_raw.csv"
    summary_display_path = output_dir / "summary_display.csv"
    trajectory_paths = list((output_dir / "trajectories").glob("*.csv"))
    assert summary_path.exists()
    assert summary_raw_path.exists()
    assert summary_display_path.exists()
    assert trajectory_paths

    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        summary_rows = list(csv.DictReader(file))
    assert {
        "router_name",
        "total_effective_utility",
        "total_penalized_reward",
        "total_mission_utility",
        "high_value_task_utility",
        "emergency_task_utility",
        "total_resource_penalty",
        "total_quality_gap_penalty",
        "total_information_gap_penalty",
        "total_requirement_penalty",
        "total_safety_penalty",
        "safety_correction_count",
        "raw_safety_violation_count",
        "final_resource_safety_violation_count",
        "task_deferred_count",
        "deadline_missed_count",
        "service_failure_count",
        "feasible_episode_flag",
        "resource_feasible_flag",
        "min_battery_soc",
        "final_battery_soc",
        "average_battery_soc",
        "dominant_mode",
        "mode_switch_count",
        "rank_by_penalized_reward",
        "rank_by_effective_utility",
        "rank_by_high_value_utility",
        "deadline_met_tasks",
        "completed_tasks",
        "completed_deadline_satisfaction_ratio",
        "total_observation_windows",
        "total_generated_tasks",
        "active_observation_slot_ratio",
        "idle_slot_ratio",
        "routine_task_count",
        "high_value_task_count",
        "emergency_task_count",
        "average_tasks_per_window",
        "delivered_items",
        "expired_items",
        "dropped_items",
        "average_waiting_time",
    }.issubset(summary_rows[0].keys())

    with trajectory_paths[0].open("r", encoding="utf-8-sig", newline="") as file:
        trajectory_rows = list(csv.DictReader(file))
    assert {
        "router_name",
        "slot",
        "raw_action",
        "executed_action",
        "selected_model",
        "selected_payload",
        "safety_corrected",
        "resource_safety_violation",
        "task_deferred",
        "deadline_missed",
        "service_failure_reason",
        "safety_violation_reason_before",
        "safety_violation_reason_after",
        "has_new_task",
        "scene_id",
        "observation_window_id",
        "task_class",
        "active_observation_window",
        "delivered_items",
        "expired_items",
        "dropped_items",
        "quality_requirement",
        "information_requirement",
        "Q_req",
        "I_req",
        "predicted_quality",
        "information_factor",
        "quality_gap",
        "information_gap",
        "is_sunlit",
        "in_sunlight",
        "is_eclipse",
        "lambda_E_base",
        "lambda_L_base",
        "lambda_Q_base",
        "lambda_S_base",
        "mode_name",
        "lambda_E",
        "lambda_L",
        "lambda_Q",
        "lambda_S",
        "exploration_scale",
        "model_complexity_bias",
        "payload_size_bias",
        "high_value_bonus_scale",
        "guidance_bundle_summary",
        "selected_action_score",
        "resource_penalty",
        "requirement_penalty",
        "safety_penalty",
        "battery_soc",
        "idle_energy_j",
        "compute_energy_j",
        "comm_energy_j",
        "queue_mb",
    }.issubset(trajectory_rows[0].keys())

    for row in summary_rows:
        total_generated_tasks = float(row["total_generated_tasks"])
        expected_ratio = float(row["deadline_met_tasks"]) / max(total_generated_tasks, 1.0)
        assert abs(float(row["deadline_satisfaction_ratio"]) - expected_ratio) < 1.0e-9
