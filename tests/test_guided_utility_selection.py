from pathlib import Path

from src.common.config import load_project_configs
from src.common.types import CommState, LinkTrace, ResourceState, ResourceTrace, TaskSample
from src.simulator.environment import SimulationEnvironment


def test_guidance_bundle_changes_predicted_utility():
    configs = load_project_configs(Path(__file__).resolve().parents[1])
    env = SimulationEnvironment(configs)
    action = next(action for action in env.action_space.list_actions() if action.action_name == "balanced+roi")

    task = TaskSample("task_guided", "a", "b", "scene", 0, 0.8, 40.0, 0.5, 0.4)
    resource_state = ResourceState(0.35, 0.9, 0.4)
    comm_state = CommState(queue_mb=18.0, queue_max_mb=64.0)
    link_trace = LinkTrace(0, True, 10.0, 3.0, 0.5, 20.0)
    resource_trace = ResourceTrace(0, 8.0, 3.0, 12.0, True, False, 0, 0, 0.1)
    context = env.context_builder.build(task, resource_state, comm_state, link_trace)

    unguided = env.predictor.predict_action(task, context, resource_trace, link_trace, comm_state, action, guidance_bundle=None)
    guided = env.predictor.predict_action(
        task,
        context,
        resource_trace,
        link_trace,
        comm_state,
        action,
        guidance_bundle={
            "mode_name": "queue_draining",
            "lambda_E": 0.6,
            "lambda_L": 0.3,
            "lambda_Q": 0.9,
            "lambda_S": 0.8,
            "exploration_scale": 1.0,
            "model_complexity_bias": -0.1,
            "payload_size_bias": -0.2,
            "high_value_bonus_scale": 1.0,
            "safety_margin_scale": 1.1,
        },
    )

    assert guided.predicted_reward != unguided.predicted_reward
