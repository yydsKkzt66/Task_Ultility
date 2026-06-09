from pathlib import Path

from src.evaluation.high_value_error_penalty_experiment import (
    HighValuePenaltyScenario,
    run_high_value_error_penalty_experiment,
)


def test_high_value_error_penalty_experiment_generates_outputs(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    paths = run_high_value_error_penalty_experiment(
        project_root=project_root,
        output_dir=tmp_path / "high_value_error_penalty",
        num_slots=6,
        scenarios=[
            HighValuePenaltyScenario(
                name="high_value_error_penalty_5x",
                description="test scenario",
                multiplier=5.0,
                enabled=True,
            )
        ],
    )
    assert paths["manifest"].exists()
    assert paths["summary"].exists()
    assert paths["rankings"].exists()
    assert paths["report"].exists()
