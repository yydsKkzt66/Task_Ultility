from pathlib import Path

from src.common.config import load_project_configs
from src.common.types import LinkTrace, ResourceTrace, TimelineStep
from src.context.comm_state import build_initial_comm_state
from src.context.mission_state import build_initial_mission_state
from src.context.resource_state import build_initial_resource_state
from src.evaluation.baselines import build_routers
from src.link.downlink_queue import DownlinkQueue
from src.simulator.environment import SimulationEnvironment


def test_idle_slot_skips_task_selection_but_keeps_simulation_running():
    project_root = Path(__file__).resolve().parents[1]
    configs = load_project_configs(project_root)
    env = SimulationEnvironment(configs)
    router = build_routers({"routers": ["safe_linucb"]}, env.action_space)[0]
    queue = DownlinkQueue(
        queue_max_mb=float(configs["simulator"]["queue_max_mb"]),
        slot_seconds=float(configs["simulator"]["slot_seconds"]),
        downlink_efficiency=float(configs["simulator"].get("downlink_efficiency", 1.0)),
        queue_priority_tau_d=float(configs["simulator"].get("queue_priority_tau_d", 60.0)),
        queue_age_bonus=float(configs["simulator"].get("queue_age_bonus", 0.01)),
        queue_meta_first_bonus=float(configs["simulator"].get("queue_meta_first_bonus", 0.25)),
        queue_high_value_bonus=float(configs["simulator"].get("queue_high_value_bonus", 0.20)),
    )
    step_runner = env.build_step_runner(queue)
    timeline_step = TimelineStep(
        slot_index=0,
        slot_seconds=float(configs["timeline"]["slot_seconds"]),
        task=None,
        resource_trace=ResourceTrace(
            slot_index=0,
            solar_energy_j=10.0,
            base_power_w=3.0,
            temperature_reference_c=12.0,
            is_sunlit=True,
            is_eclipse=False,
            orbit_index=0,
            day_index=0,
            orbit_phase=0.1,
        ),
        link_trace=LinkTrace(0, False, -3.0, 0.0, 0.0, 0.0),
    )
    resource_state = build_initial_resource_state(configs["simulator"])
    mission_state = build_initial_mission_state()
    comm_state = build_initial_comm_state(configs["simulator"])

    record, reward = step_runner.run_step(timeline_step, resource_state, mission_state, comm_state, router)

    assert record.has_task == 0
    assert record.raw_action == "no_task"
    assert record.executed_action == "no_task"
    assert reward.reward == 0.0
