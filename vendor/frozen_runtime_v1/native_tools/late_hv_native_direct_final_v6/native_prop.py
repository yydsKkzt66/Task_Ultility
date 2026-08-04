from copy import deepcopy
from pathlib import Path
try:
    from .native_support import IdentityBinding, normalize_episode
except ImportError:
    from native_support import IdentityBinding, normalize_episode


class PropBinding(IdentityBinding):
    def __init__(self, source_root: Path): super().__init__("PROP-HEADROOM", source_root)
    def _instantiate(self):
        from src.actions.action_space import ActionSpace
        from src.evaluation.baselines import build_routers
        return build_routers({"routers": [self.spec["identifier"]]}, ActionSpace())[0]
    def simulate(self, simulation_input, slots: int):
        from src.evaluation.baselines import build_routers
        from src.simulator.environment import SimulationEnvironment
        configs = deepcopy(simulation_input.configs)
        configs["experiment"] = {"routers": [self.spec["identifier"]]}
        env = SimulationEnvironment(configs)
        runner = env.build_episode_runner(simulation_input.timeline[:slots])
        result = runner.run(build_routers(configs["experiment"], env.action_space)[0])
        normalized = normalize_episode(self.method, simulation_input, result, runner, slots)
        return {"normalized": normalized, "event_count": len(normalized["events"])}


def make_binding(source_root: Path): return PropBinding(source_root)
