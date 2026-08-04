"""Experiment baseline and router factory."""

from __future__ import annotations

from src.actions.action_space import ActionSpace
from src.routers.fixed_router import FixedRouter
from src.routers.b1_enhanced_router import B1EnhancedRouter
from src.routers.rule_router import RuleBasedRouter
from src.routers.safe_linucb_router import SafeLinUCBRouter


def build_routers(experiment_cfg: dict[str, object], action_space: ActionSpace):
    """Build routers from experiment config."""

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
            routers.append(
                SafeLinUCBRouter(
                    action_names,
                    router_name=router_name,
                    use_lyapunov_pricing=False,
                    use_mode_controller=False,
                )
            )
        elif router_name == "safe_bandit_only":
            routers.append(
                SafeLinUCBRouter(
                    action_names,
                    router_name=router_name,
                    use_lyapunov_pricing=False,
                    use_mode_controller=False,
                )
            )
        elif router_name == "no_lyapunov":
            routers.append(
                SafeLinUCBRouter(
                    action_names,
                    router_name=router_name,
                    use_lyapunov_pricing=False,
                    use_mode_controller=True,
                )
            )
        elif router_name == "no_mode":
            routers.append(
                SafeLinUCBRouter(
                    action_names,
                    router_name=router_name,
                    use_lyapunov_pricing=True,
                    use_mode_controller=False,
                )
            )
        elif router_name == "proposed_lyapunov_rule_mode_safe_bandit":
            routers.append(
                SafeLinUCBRouter(
                    action_names,
                    router_name=router_name,
                    use_lyapunov_pricing=True,
                    use_mode_controller=True,
                )
            )
        elif router_name in {"b1_enhanced", "b1_enhanced_contact_compressed"}:
            routers.append(B1EnhancedRouter(router_name="b1_enhanced_contact_compressed"))
        else:
            raise ValueError(f"Unknown router: {router_name}")
    return routers
