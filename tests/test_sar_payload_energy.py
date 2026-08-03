from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "energy" / "sar_payload_energy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sar_payload_energy_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def battery_update(**kwargs):
    load = kwargs["compute_energy_j"] + kwargs["comm_energy_j"]
    return max(0.0, kwargs["battery_soc"] - load / kwargs["discharge_efficiency"] / kwargs["battery_capacity_j"])


def test_multiple_tasks_from_one_acquisition_are_charged_once() -> None:
    module = load_module()
    tasks = [
        {"task_id": "a", "arrival_slot": 8, "metadata_json": '{"product_identifier":"p1"}'},
        {"task_id": "b", "arrival_slot": 3, "metadata_json": '{"product_identifier":"p1"}'},
        {"task_id": "c", "arrival_slot": 5, "metadata_json": '{"product_identifier":"p2"}'},
    ]
    cfg = module.SarPayloadConfig(True, 1600.0, 10.0, True)
    mapping, events = module.build_acquisition_schedule(tasks, cfg)
    assert len(mapping) == 3
    assert len(events) == 2
    assert events[0]["event_slot"] == 3
    assert events[0]["sar_energy_wh"] == 16000.0 / 3600.0
    assert sum(row["is_charge_trigger_task"] for row in mapping) == 2


def test_soc_debit_uses_common_discharge_efficiency_and_only_once() -> None:
    module = load_module()
    event = {"acquisition_id": "p1", "event_slot": 4, "sar_energy_j": 16000.0}
    controller = module.SarSocDebitController(
        [event], battery_capacity_j=230.0 * 3600.0, discharge_efficiency=0.95,
        battery_update=battery_update, method="m", profile="1600W",
    )
    expected = 0.85 - 16000.0 / 0.95 / (230.0 * 3600.0)
    assert abs(controller.debit_value(0.85, 4) - expected) < 1e-12
    assert controller.debit_value(expected, 4) == expected
    assert len(controller.trace) == 1


def test_disabled_profile_has_zero_energy() -> None:
    module = load_module()
    cfg = module.SarPayloadConfig(False, 3870.0, 10.0, True)
    assert cfg.energy_per_acquisition_j == 0.0
    assert cfg.energy_per_acquisition_wh == 0.0
