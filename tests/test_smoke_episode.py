from src.common.config import load_project_configs
from src.evaluation.baselines import build_routers
from src.simulator.environment import SimulationEnvironment
from src.timeline.timeline_builder import TimelineBuilder


def test_smoke_episode_runs_complete_loop():
    configs = load_project_configs(__import__("pathlib").Path(__file__).resolve().parents[1])
    configs["timeline"]["num_slots"] = 8
    timeline = TimelineBuilder(configs["dataset"], configs["timeline"]).build()
    env = SimulationEnvironment(configs)
    router = build_routers({"routers": ["fixed_lightweight"]}, env.action_space)[0]
    result = env.build_episode_runner(timeline).run(router)
    assert len(result.records) == 8
