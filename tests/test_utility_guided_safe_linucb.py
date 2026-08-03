import math

import numpy as np

from src.actions.action_space import ActionSpace
from src.common.types import ContextVector, PredictedMetrics
from src.routers.safe_linucb_router import SafeLinUCBRouter


def _metrics(predicted_reward: float, payload_mb: float = 1.0, energy_j: float = 10.0) -> PredictedMetrics:
    return PredictedMetrics(
        predicted_S_gen_mb=payload_mb,
        predicted_S_down_mb=payload_mb,
        predicted_E_comp=energy_j,
        predicted_E_comm=0.5,
        predicted_L_inf=1.0,
        predicted_L_queue=0.0,
        predicted_L_comm=0.1,
        predicted_Q_next_mb=payload_mb,
        predicted_B_next=0.8,
        predicted_E_AI_next=0.9,
        predicted_Theta_next=0.3,
        predicted_delivery_factor=1.0,
        predicted_safety_violation=0.0,
        predicted_reward=predicted_reward,
        predicted_status="delivered",
    )


def _context() -> ContextVector:
    return ContextVector(0.8, 0.9, 0.2, 0.7, 0.3, 0.4, 0.2, 1.0, 0.1)


def test_utility_guided_disabled_preserves_legacy_formula():
    actions = ActionSpace().list_actions()[:3]
    router = SafeLinUCBRouter([action.action_name for action in actions], alpha=0.7, ridge=1.0)
    x = _context().as_array()
    for index, action in enumerate(actions):
        router.b[action.action_name] = np.ones(router.dimension) * (0.15 * (index + 1))
    predicted_map = {
        actions[0].action_name: _metrics(0.5),
        actions[1].action_name: _metrics(0.2),
        actions[2].action_name: _metrics(0.9),
    }
    guidance_bundle = {
        "exploration_scale": 1.0,
        "policy_repair": {
            "enable_utility_guided_scoring": False,
            "preserve_legacy_score_logging": True,
        },
    }

    selected = router.select_action(_context(), actions, predicted_map, guidance_bundle)

    expected_scores = {}
    for action in actions:
        exploit = float(router.b[action.action_name] @ x)
        explore = float(math.sqrt(x @ x))
        expected_scores[action.action_name] = exploit + 0.7 * explore + 0.20 * predicted_map[action.action_name].predicted_reward
        components = router.last_all_score_components[action.action_name]
        assert components["scoring_mode"] == "legacy"
        assert components["selected_by_legacy_shadow"] in {0, 1}
        assert math.isclose(components["final_action_score"], expected_scores[action.action_name], rel_tol=1.0e-12)

    assert selected.action_name == max(expected_scores, key=expected_scores.get)
    assert math.isclose(router.last_selected_action_score, expected_scores[selected.action_name], rel_tol=1.0e-12)


def test_utility_guided_enabled_can_make_utility_prior_primary_driver():
    actions = ActionSpace().list_actions()[:3]
    router = SafeLinUCBRouter([action.action_name for action in actions], alpha=0.7, ridge=1.0)
    router.b[actions[0].action_name] = np.ones(router.dimension) * 1.0
    predicted_map = {
        actions[0].action_name: _metrics(0.1),
        actions[1].action_name: _metrics(2.0),
        actions[2].action_name: _metrics(0.2),
    }
    guidance_bundle = {
        "exploration_scale": 1.0,
        "policy_repair": {
            "enable_utility_guided_scoring": True,
            "utility_guided_scoring": {
                "utility_prior_weight": 1.0,
                "learned_correction_weight": 0.0,
                "exploration_weight": 0.0,
                "score_component_normalization": "zscore_by_candidate_set",
                "fallback_if_zero_variance": True,
                "preserve_legacy_score_logging": True,
            },
        },
    }

    selected = router.select_action(_context(), actions, predicted_map, guidance_bundle)

    assert selected.action_name == actions[1].action_name
    assert router.last_score_components["scoring_mode"] == "utility_guided"
    assert router.last_score_components["selected_by_utility_guided_shadow"] == 1
    assert router.last_score_components["action_flip_vs_legacy_shadow"] == 1


def test_product_requirement_aware_disabled_does_not_change_utility_guided_selection():
    all_actions = ActionSpace().list_actions()
    actions = [
        next(action for action in all_actions if action.payload_policy == "metadata"),
        next(action for action in all_actions if action.payload_policy == "roi_context"),
    ]
    router = SafeLinUCBRouter([action.action_name for action in actions], alpha=0.7, ridge=1.0)
    predicted_map = {
        actions[0].action_name: _metrics(1.0),
        actions[1].action_name: _metrics(0.2),
    }
    base_guidance = {
        "exploration_scale": 1.0,
        "policy_repair": {
            "enable_utility_guided_scoring": True,
            "utility_guided_scoring": {
                "utility_prior_weight": 1.0,
                "learned_correction_weight": 0.0,
                "exploration_weight": 0.0,
                "score_component_normalization": "none",
            },
        },
    }
    disabled_guidance = {
        **base_guidance,
        "product_requirement_context": {
            "ideal_product_requirement": "context",
            "uncertainty": 0.8,
            "deadline_s": 100.0,
        },
        "module2_product_requirement_aware_scoring": {
            "enabled": False,
            "requirement_fit_weight": 10.0,
        },
    }

    base_selected = router.select_action(_context(), actions, predicted_map, base_guidance)
    disabled_selected = router.select_action(_context(), actions, predicted_map, disabled_guidance)

    assert disabled_selected.action_name == base_selected.action_name
    assert router.last_score_components["product_requirement_aware_enabled"] == 0


def test_product_requirement_aware_enabled_can_rank_requirement_fit_softly():
    all_actions = ActionSpace().list_actions()
    actions = [
        next(action for action in all_actions if action.payload_policy == "metadata"),
        next(action for action in all_actions if action.payload_policy == "roi_context"),
    ]
    router = SafeLinUCBRouter([action.action_name for action in actions], alpha=0.7, ridge=1.0)
    predicted_map = {
        actions[0].action_name: _metrics(0.5),
        actions[1].action_name: _metrics(0.5),
    }
    guidance = {
        "exploration_scale": 1.0,
        "policy_repair": {
            "enable_utility_guided_scoring": True,
            "utility_guided_scoring": {
                "utility_prior_weight": 1.0,
                "learned_correction_weight": 0.0,
                "exploration_weight": 0.0,
                "score_component_normalization": "none",
            },
        },
        "product_requirement_context": {
            "ideal_product_requirement": "context",
            "uncertainty": 0.8,
            "deadline_s": 100.0,
        },
        "resource_context": {"in_contact": True, "remaining_window_s": 10.0},
        "module2_product_requirement_aware_scoring": {
            "enabled": True,
            "requirement_fit_weight": 1.0,
            "model_fit_weight": 0.0,
            "deadline_feasibility_weight": 0.0,
            "staged_delivery_feasibility_weight": 0.0,
        },
    }

    selected = router.select_action(_context(), actions, predicted_map, guidance)

    assert selected.payload_policy == "roi_context"
    assert router.last_score_components["product_requirement_aware_enabled"] == 1
    assert router.last_score_components["requirement_fit_score"] == 1.0
