from copy import deepcopy
from pathlib import Path
try:
    from .native_support import IdentityBinding, normalize_episode
    from .value_adapter import adapt_myopic_scenario, validate_myopic_runtime_values
except ImportError:
    from native_support import IdentityBinding, normalize_episode
    from value_adapter import adapt_myopic_scenario, validate_myopic_runtime_values


class MyopicBinding(IdentityBinding):
    def __init__(self, source_root: Path): super().__init__("MYOPIC-E2E-UTILITY-GREEDY", source_root)
    def _instantiate(self):
        from src.actions.action_space import ActionSpace
        from src.evaluation.baselines import build_routers
        return build_routers({"routers": [self.spec["identifier"]]}, ActionSpace())[0]
    def build_scenario(self, bundle):
        canonical_scenario = super().build_scenario(bundle)
        adapted_scenario, self.myopic_value_contract_audit = adapt_myopic_scenario(canonical_scenario)
        return adapted_scenario
    def validate_scenario(self, scenario, bundle):
        base = super().validate_scenario(scenario, bundle)
        audit = validate_myopic_runtime_values(scenario, bundle)
        return {
            **base,
            "myopic_decision_value_contract_status": audit["status"],
            "myopic_mapping_sequence_sha256": audit["mapping_sequence_sha256"],
            "myopic_value_reference": audit["V_ref"],
            "myopic_normalization_formula": audit["normalization_formula"],
            "myopic_decision_value_minimum": audit["minimum_myopic_decision_value"],
            "myopic_decision_value_maximum": audit["maximum_myopic_decision_value"],
        }
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


def make_binding(source_root: Path): return MyopicBinding(source_root)
