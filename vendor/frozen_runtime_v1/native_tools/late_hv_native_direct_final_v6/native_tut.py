from copy import deepcopy
from pathlib import Path
try:
    from .native_support import IdentityBinding, normalize_episode
except ImportError:
    from native_support import IdentityBinding, normalize_episode


class TutBinding(IdentityBinding):
    def __init__(self, source_root: Path): super().__init__("TUT-UOI-PRODUCT-LITE-v2", source_root)
    def _instantiate(self):
        import scripts.run_tut_uoi_rate_lite as tut_v1
        import scripts.run_tut_uoi_rate_lite_v2 as tut_v2
        config = tut_v2.build_configs(self.source_root / "configs/experiments/tut_uoi_rate_lite_v2_formal_v1.yaml")
        return tut_v1.build_router(config)
    def simulate(self, simulation_input, slots: int):
        import scripts.run_tut_uoi_rate_lite as tut_v1
        import scripts.run_tut_uoi_rate_lite_v2 as tut_v2
        from src.simulator.environment import SimulationEnvironment
        method_configs = tut_v2.build_configs(self.source_root / "configs/experiments/tut_uoi_rate_lite_v2_formal_v1.yaml")
        configs = deepcopy(simulation_input.configs)
        configs["action_space"] = deepcopy(method_configs["action_space"])
        for key in ("tut_uoi_density_priority_enabled", "progressive_payload", "guarantee", "downlink_safety_throttle"):
            if key in method_configs["simulator"]:
                configs["simulator"][key] = deepcopy(method_configs["simulator"][key])
        configs["experiment"] = {"routers": ["tut_uoi_rate_lite"]}
        env = SimulationEnvironment(configs)
        runner = env.build_episode_runner(simulation_input.timeline[:slots])
        result = runner.run(tut_v1.build_router(configs))
        normalized = normalize_episode(self.method, simulation_input, result, runner, slots)
        return {"normalized": normalized, "event_count": len(normalized["events"])}


def make_binding(source_root: Path): return TutBinding(source_root)
