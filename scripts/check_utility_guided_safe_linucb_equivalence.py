"""Check that disabled utility-guided scoring preserves legacy SafeLinUCB."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.actions.action_space import ActionSpace  # noqa: E402
from src.common.types import ContextVector, PredictedMetrics  # noqa: E402
from src.routers.safe_linucb_router import SafeLinUCBRouter  # noqa: E402


def _metrics(predicted_reward: float, payload_mb: float, energy_j: float) -> PredictedMetrics:
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


def run_check(output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    actions = ActionSpace().list_actions()[:4]
    context = ContextVector(0.82, 0.88, 0.24, 0.72, 0.31, 0.44, 0.22, 1.0, 0.12)
    x = context.as_array()
    router = SafeLinUCBRouter([action.action_name for action in actions], alpha=0.7, ridge=1.0)
    for index, action in enumerate(actions):
        router.b[action.action_name] = np.ones(router.dimension) * (0.07 * (index + 1))

    predicted_map = {
        action.action_name: _metrics(
            predicted_reward=0.25 + 0.17 * index,
            payload_mb=0.5 + 0.4 * index,
            energy_j=8.0 + 7.5 * index,
        )
        for index, action in enumerate(actions)
    }
    guidance_bundle = {
        "exploration_scale": 1.0,
        "policy_repair": {
            "enable_utility_guided_scoring": False,
            "preserve_legacy_score_logging": True,
        },
    }
    selected = router.select_action(context, actions, predicted_map, guidance_bundle)

    failures: list[str] = []
    expected_scores: dict[str, float] = {}
    for action in actions:
        exploit = float(router.b[action.action_name] @ x)
        explore = float(math.sqrt(x @ x))
        expected = exploit + 0.7 * explore + 0.20 * predicted_map[action.action_name].predicted_reward
        expected_scores[action.action_name] = expected
        actual = float(router.last_all_score_components[action.action_name]["final_action_score"])
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            failures.append(f"{action.action_name}: expected {expected:.12f}, got {actual:.12f}")

    expected_selected = max(expected_scores, key=expected_scores.get)
    if selected.action_name != expected_selected:
        failures.append(f"selected action changed: expected {expected_selected}, got {selected.action_name}")
    if router.last_score_components.get("scoring_mode") != "legacy":
        failures.append(f"scoring_mode should be legacy, got {router.last_score_components.get('scoring_mode')}")

    report = [
        "# Disabled Equivalence Check",
        "",
        f"- status: {'PASS' if not failures else 'FAIL'}",
        f"- selected_action: {selected.action_name}",
        f"- expected_selected_action: {expected_selected}",
        "- legacy_formula: exploit + cost_aware_ucb_bonus + 0.20 * predicted_reward",
        f"- checked_actions: {len(actions)}",
    ]
    if failures:
        report.append("")
        report.append("## Failures")
        report.extend(f"- {failure}" for failure in failures)
    (output_dir / "disabled_equivalence_check.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check disabled utility-guided SafeLinUCB equivalence.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "_utility_guided_safe_linucb_patch",
    )
    args = parser.parse_args()
    if not run_check(args.output_dir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
