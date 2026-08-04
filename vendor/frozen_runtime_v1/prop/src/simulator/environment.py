"""仿真环境装配器。"""

from __future__ import annotations

from src.actions.action_space import ActionSpace
from src.controllers.lyapunov_pricing import LyapunovPricing
from src.controllers.mode_controller import BaseModeController, RuleBasedModeController
from src.context.context_builder import ContextBuilder
from src.models.inference_runner import InferenceRunner
from src.models.model_registry import ModelRegistry
from src.routers.safe_filter import SafeFilter
from src.simulator.episode_runner import EpisodeRunner
from src.simulator.predictors import Predictor
from src.simulator.step_runner import StepRunner


class SimulationEnvironment:
    """负责装配一次实验所需的共享组件。"""

    def __init__(self, configs: dict[str, dict]) -> None:
        self.configs = configs
        self.dataset_cfg = configs["dataset"]
        self.model_pool_cfg = configs["model_pool"]
        self.payload_profiles_cfg = configs["payload_profiles"]
        self.action_space_cfg = configs["action_space"]
        self.timeline_cfg = configs["timeline"]
        self.simulator_cfg = configs["simulator"]
        if "battery_capacity_wh" in self.simulator_cfg:
            self.simulator_cfg["battery_capacity_j"] = float(self.simulator_cfg["battery_capacity_wh"]) * 3600.0
        self.reward_cfg = configs["reward"]
        self.lyapunov_pricing_cfg = configs.get("lyapunov_pricing", {})
        self.mode_controller_cfg = configs.get("mode_controller", {})

        self.action_space = ActionSpace(self.action_space_cfg)
        self.model_registry = ModelRegistry(self.model_pool_cfg)
        self.context_builder = ContextBuilder(queue_max_mb=float(self.simulator_cfg["queue_max_mb"]))
        self.predictor = Predictor(self.model_registry, self.reward_cfg, self.simulator_cfg, self.payload_profiles_cfg)
        self.safe_filter = SafeFilter(self.simulator_cfg)
        self.inference_runner = InferenceRunner(self.model_registry)
        self.lyapunov_pricing = self._build_lyapunov_pricing()
        self.mode_controller = self._build_mode_controller()

    def _build_lyapunov_pricing(self) -> LyapunovPricing | None:
        if not bool(self.lyapunov_pricing_cfg.get("enabled", False)):
            return None
        return LyapunovPricing(self.lyapunov_pricing_cfg)

    def _build_mode_controller(self) -> BaseModeController | None:
        if not bool(self.mode_controller_cfg.get("enabled", False)):
            return None
        controller_type = str(self.mode_controller_cfg.get("type", "rule_based")).lower()
        if controller_type == "rule_based":
            return RuleBasedModeController(self.mode_controller_cfg)
        raise ValueError(f"Unsupported mode controller type: {controller_type}")

    def build_step_runner(self, queue, timeline=None) -> StepRunner:
        """为指定队列构建 step runner。

        单独实验支线会直接复用这个入口，以便在不改默认 EpisodeRunner 的前提下，
        额外采集每一步的错判重罚统计量。
        """

        future_link_traces = [step.link_trace for step in timeline] if timeline is not None else None
        return StepRunner(
            action_space=self.action_space,
            context_builder=self.context_builder,
            predictor=self.predictor,
            safe_filter=self.safe_filter,
            inference_runner=self.inference_runner,
            downlink_queue=queue,
            model_registry=self.model_registry,
            reward_cfg=self.reward_cfg,
            simulator_cfg=self.simulator_cfg,
            payload_profiles_cfg=self.payload_profiles_cfg,
            lyapunov_pricing=self.lyapunov_pricing,
            mode_controller=self.mode_controller,
            future_link_traces=future_link_traces,
        )

    def build_episode_runner(self, timeline) -> EpisodeRunner:
        """为某条 timeline 创建 episode runner。"""

        def step_runner_factory(queue):
            return self.build_step_runner(queue, timeline)

        return EpisodeRunner(timeline=timeline, step_runner_factory=step_runner_factory, simulator_cfg=self.simulator_cfg)
