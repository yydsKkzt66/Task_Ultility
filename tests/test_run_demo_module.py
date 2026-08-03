import subprocess
import sys
import csv
from pathlib import Path


def test_run_demo_module_generates_outputs(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "demo_outputs"
    command = [
        sys.executable,
        "-m",
        "src.run_demo",
        "--output-dir",
        str(output_dir),
        "--num-slots",
        "8",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, cwd=project_root)
    assert completed.returncode == 0
    summary_path = output_dir / "summary.csv"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    routers = {row["router"] for row in rows}
    assert "proposed_lyapunov_rule_mode_safe_bandit" in routers
    assert {"safe_bandit_only", "no_lyapunov", "no_mode"}.issubset(routers)
    proposed_row = next(row for row in rows if row["router"] == "proposed_lyapunov_rule_mode_safe_bandit")
    assert float(proposed_row["safety_violation_count"]) <= 1.0
