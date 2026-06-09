from pathlib import Path

from src.evaluation.sensitivity import ProfileScenario, RewardScenario, run_sensitivity_experiment


def test_sensitivity_experiment_generates_core_artifacts(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    paths = run_sensitivity_experiment(
        project_root=project_root,
        output_dir=tmp_path / "sensitivity",
        num_slots=6,
        reward_scenarios=[RewardScenario("baseline_reward", "baseline", {})],
        profile_scenarios=[ProfileScenario("baseline_profiles", "baseline", {})],
    )
    assert paths["manifest"].exists()
    assert paths["summary"].exists()
    assert paths["rankings"].exists()
    assert paths["report"].exists()
