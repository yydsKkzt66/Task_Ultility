"""实验基线构造器。"""

from __future__ import annotations

from src.actions.action_space import ActionSpace
from src.routers.fixed_router import FixedRouter
from src.routers.rule_router import RuleBasedRouter
from src.routers.safe_linucb_router import SafeLinUCBRouter


def build_routers(experiment_cfg: dict[str, object], action_space: ActionSpace):
    """根据 experiment.yaml 构建 router 列表。"""

    actions = action_space.list_actions()
    action_names = [action.action_name for action in actions]
    routers = []
    for router_name in experiment_cfg.get("routers", []):
        if router_name == "fixed_lightweight":
            routers.append(FixedRouter(router_name, "lightweight+metadata"))
        elif router_name == "fixed_high_accuracy":
            routers.append(FixedRouter(router_name, "high_accuracy+roi_context"))
        elif router_name == "comm_conservative":
            routers.append(RuleBasedRouter(router_name, mode="comm_conservative"))
        elif router_name == "utility_rule":
            routers.append(RuleBasedRouter(router_name, mode="utility_rule"))
        elif router_name == "safe_linucb":
            routers.append(SafeLinUCBRouter(action_names))
        else:
            raise ValueError(f"未知 router: {router_name}")
    return routers
