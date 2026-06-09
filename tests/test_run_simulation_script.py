import subprocess
import sys
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
    assert (output_dir / "summary.csv").exists()
    assert list((output_dir / "trajectories").glob("*.csv"))
