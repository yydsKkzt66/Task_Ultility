from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "frozen_runtime_v1" / "b1_offload"


def test_config_has_registered_shared_ground_contract() -> None:
    config = yaml.safe_load((ROOT / "configs" / "b1_common_ground_v1.yaml").read_text(encoding="utf-8"))
    ground = config["ground_processing"]
    assert config["method"]["method_id"] == "B1-COMMON-GROUND"
    assert ground["shared_implementation"] == "src.baselines.offload_eft.FiniteGroundComputeQueue"
    assert ground["servers"] == 1
    assert ground["queue_policy"] == "fifo"
    assert ground["queue_capacity_tasks"] == 32
    assert ground["decode_time_s"] == 1.5
    assert ground["inference_time_s"] == 2.8
    assert ground["submit_only_after_complete_downlink"] is True
    assert ground["streaming_compute_enabled"] is False


def test_adapter_uses_offload_queue_and_serializes_jobs() -> None:
    code = f"""
import importlib.util, sys
from pathlib import Path
sys.path.insert(0, {str(SOURCE)!r})
from src.baselines.offload_eft import FiniteGroundComputeQueue
spec = importlib.util.spec_from_file_location('b1cg_test', {str(ROOT / 'scripts' / 'b1_common_ground_runtime.py')!r})
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
assert mod.FiniteGroundComputeQueue is FiniteGroundComputeQueue
queue = FiniteGroundComputeQueue(servers=1, max_queue_items=32)
first = queue.commit('a', 100.0, 4.3)
second = queue.commit('b', 101.0, 4.3)
assert first.compute_start_time == 100.0
assert abs(first.compute_end_time - 104.3) < 1e-12
assert abs(second.compute_start_time - 104.3) < 1e-12
assert abs(second.waiting_time - 3.3) < 1e-12
assert first.server_id == second.server_id == 0
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


def test_adapter_does_not_copy_a_second_queue_implementation() -> None:
    source = (ROOT / "scripts" / "b1_common_ground_runtime.py").read_text(encoding="utf-8")
    assert "class FiniteGroundComputeQueue" not in source
    assert "from src.baselines.offload_eft import FiniteGroundComputeQueue" in source
    assert "super()._process_downlink" in source
