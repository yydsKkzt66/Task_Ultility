from pathlib import Path

from src.common.config import load_project_configs
from src.common.types import CommState, LinkTrace, ResourceState, ResourceTrace, TaskSample
from src.simulator.environment import SimulationEnvironment


def test_predictor_comm_energy_accounts_for_total_transmitted_payload():
    configs = load_project_configs(Path(__file__).resolve().parents[1])
    env = SimulationEnvironment(configs)
    action = next(action for action in env.action_space.list_actions() if action.action_name == "lightweight+metadata")

    task = TaskSample("task_demo", "a", "b", "scene", 0, 0.8, 40.0, 0.4, 0.4)
    resource_state = ResourceState(0.8, 0.9, 0.2)
    comm_state = CommState(queue_mb=20.0, queue_max_mb=64.0)
    link_trace = LinkTrace(0, True, 10.0, 8.0, 0.8, 20.0)
    resource_trace = ResourceTrace(0, 20.0, 3.0, 10.0)
    context = env.context_builder.build(task, resource_state, comm_state, link_trace)

    predicted = env.predictor.predict_action(task, context, resource_trace, link_trace, comm_state, action)

    assert predicted.predicted_S_down_mb == 0.0
    assert predicted.predicted_E_comm > 0.0
