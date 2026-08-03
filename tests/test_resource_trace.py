from src.energy.battery_update import update_battery_soc
from src.energy.thermal_update import update_thermal_pressure
from src.timeline.resource_trace_loader import load_synthetic_resource_trace


def test_resource_trace_contains_sunlight_and_eclipse_states():
    traces = load_synthetic_resource_trace(
        120,
        {
            "slot_seconds": 10.0,
            "orbit_period_slots": 40,
            "sunlight_fraction": 0.6,
            "day_cycle_orbits": 8,
            "solar_charge_power_w": 30.0,
            "idle_power_w": 2.5,
            "resource_power_variation_w": 0.4,
            "sunlit_temperature_c": 18.0,
            "eclipse_temperature_c": 5.0,
            "temperature_cycle_variation_c": 2.0,
        },
    )
    assert any(trace.is_sunlit for trace in traces)
    assert any(trace.is_eclipse for trace in traces)
    assert all(trace.solar_energy_j == 0.0 for trace in traces if trace.is_eclipse)
    assert any(trace.solar_energy_j > 0.0 for trace in traces if trace.is_sunlit)


def test_battery_charge_discharge_model_can_charge_and_discharge():
    charged = update_battery_soc(
        battery_soc=0.5,
        solar_energy_j=80.0,
        base_power_w=2.0,
        slot_seconds=10.0,
        compute_energy_j=0.0,
        comm_energy_j=0.0,
        battery_capacity_j=1000.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.95,
    )
    discharged = update_battery_soc(
        battery_soc=0.5,
        solar_energy_j=0.0,
        base_power_w=2.0,
        slot_seconds=10.0,
        compute_energy_j=20.0,
        comm_energy_j=10.0,
        battery_capacity_j=1000.0,
        charge_efficiency=0.9,
        discharge_efficiency=0.95,
    )
    assert charged > 0.5
    assert discharged < 0.5


def test_thermal_cooling_is_stronger_in_eclipse_than_sunlight():
    sunlit = update_thermal_pressure(
        thermal_pressure=0.8,
        compute_energy_j=0.0,
        comm_energy_j=0.0,
        cooling=0.02,
        temperature_reference_c=18.0,
        is_sunlit=True,
        passive_cooling=0.12,
        sunlight_gain=0.015,
        eclipse_cooling_bonus=0.03,
        reference_min_c=-10.0,
        reference_max_c=30.0,
    )
    eclipse = update_thermal_pressure(
        thermal_pressure=0.8,
        compute_energy_j=0.0,
        comm_energy_j=0.0,
        cooling=0.02,
        temperature_reference_c=5.0,
        is_sunlit=False,
        passive_cooling=0.12,
        sunlight_gain=0.015,
        eclipse_cooling_bonus=0.03,
        reference_min_c=-10.0,
        reference_max_c=30.0,
    )
    assert eclipse < sunlit
