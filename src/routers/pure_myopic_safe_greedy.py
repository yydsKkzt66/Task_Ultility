"""Pure one-step greedy selector for an already SafeFiltered V2 action set.

The caller must neutralize long-horizon Lyapunov prices and resource-mode
guidance before constructing ``predicted_reward``.  This selector then performs
only deterministic current-step net-utility maximization.  It never learns,
explores, reads future tasks, or changes the shared action/execution pipeline.
"""

from __future__ import annotations

from copy import deepcopy
import math
import types
from typing import Any, Mapping


SCORE_TOLERANCE = 1.0e-12
GUIDANCE_KEYS = ("lambda_E", "lambda_L", "lambda_Q", "lambda_S")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _is_defer(action: Any) -> bool:
    return str(getattr(action, "payload_policy", "")) == "defer" or str(
        getattr(action, "model_name", "")
    ) == "delay_store"


def _tie_key(action: Any, predicted: Any) -> tuple[float, float, float, str]:
    latency = sum(
        _number(getattr(predicted, name, 0.0))
        for name in ("predicted_L_inf", "predicted_L_queue", "predicted_L_comm")
    )
    payload = _number(getattr(predicted, "predicted_S_gen_mb", 0.0))
    energy = _number(getattr(predicted, "predicted_E_comp", 0.0)) + _number(
        getattr(predicted, "predicted_E_comm", 0.0)
    )
    return latency, payload, energy, str(getattr(action, "action_name", ""))


def select_pure_myopic(
    candidate_actions: list[Any], predicted_map: Mapping[str, Any]
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Return max current-step predicted net, with Defer as the zero option."""

    if not candidate_actions:
        raise ValueError("candidate_actions must not be empty")
    rows: dict[str, dict[str, Any]] = {}
    winner = candidate_actions[0]
    winner_score = _number(predicted_map[winner.action_name].predicted_reward)
    winner_tie = _tie_key(winner, predicted_map[winner.action_name])
    for action in candidate_actions:
        predicted = predicted_map[action.action_name]
        score = _number(getattr(predicted, "predicted_reward", 0.0))
        tie = _tie_key(action, predicted)
        rows[str(action.action_name)] = {
            "predicted_reward": score,
            "pure_myopic_physical_net": score,
            "final_action_score": score,
            "ucb_bonus": 0.0,
            "raw_ucb_bonus": 0.0,
            "online_bandit_disabled": 1,
            "defer_outside_option": int(_is_defer(action)),
            "tie_latency_s": tie[0],
            "tie_payload_mb": tie[1],
            "tie_energy_j": tie[2],
        }
        if score > winner_score + SCORE_TOLERANCE or (
            abs(score - winner_score) <= SCORE_TOLERANCE and tie < winner_tie
        ):
            winner, winner_score, winner_tie = action, score, tie
    rows[str(winner.action_name)]["selected"] = 1
    return winner, rows


def install_pure_myopic_safe_greedy(delegate: Any, audit: dict[str, Any]) -> None:
    """Replace bandit ranking/update while preserving the router interface."""

    audit.update(
        {
            "enabled": True,
            "selector_contract": "SafeFilter -> zero long-term guidance -> q-once current-step net argmax; Defer=0",
            "greedy_select_call_count": 0,
            "blocked_online_update_count": 0,
            "nonzero_guidance_violation_count": 0,
            "defer_selection_count": 0,
        }
    )

    def select_action(
        self: Any,
        context: Any,
        candidate_actions: list[Any],
        predicted_map: Mapping[str, Any],
        guidance_bundle: Mapping[str, Any] | None = None,
    ) -> Any:
        del context
        audit["greedy_select_call_count"] += 1
        guide = dict(guidance_bundle or {})
        nonzero = any(abs(_number(guide.get(key, 0.0))) > SCORE_TOLERANCE for key in GUIDANCE_KEYS)
        nonzero = nonzero or abs(_number(guide.get("model_complexity_bias", 0.0))) > SCORE_TOLERANCE
        nonzero = nonzero or abs(_number(guide.get("payload_size_bias", 0.0))) > SCORE_TOLERANCE
        audit["nonzero_guidance_violation_count"] += int(nonzero)
        selected, rows = select_pure_myopic(candidate_actions, predicted_map)
        audit["defer_selection_count"] += int(_is_defer(selected))
        self.last_all_score_components = rows
        self.last_score_components = deepcopy(rows[str(selected.action_name)])
        self.last_selected_action_score = rows[str(selected.action_name)]["final_action_score"]
        return selected

    def update(self: Any, context: Any, action: Any, observed_reward: float) -> None:
        del self, context, action, observed_reward
        audit["blocked_online_update_count"] += 1

    delegate.select_action = types.MethodType(select_action, delegate)
    delegate.update = types.MethodType(update, delegate)
