from src.common.types import ContextVector
from src.payload.information_sufficiency import compute_information_sufficiency


def test_information_sufficiency_saturates_at_one():
    context = ContextVector(0.8, 0.9, 0.2, 0.3, 0.1, 0.1, 0.1, 0.5, 0.1)
    reward_cfg = {
        "info_base": 0.25,
        "info_value_weight": 0.35,
        "info_difficulty_weight": 0.20,
        "info_uncertainty_weight": 0.25,
        "info_urgency_weight": 0.20,
    }
    assert compute_information_sufficiency(context, 10.0, reward_cfg) == 1.0


def test_metadata_can_be_sufficient_for_easy_task():
    context = ContextVector(0.8, 0.9, 0.2, 0.1, 0.1, 0.05, 0.05, 0.5, 0.1)
    reward_cfg = {
        "info_base": 0.25,
        "info_value_weight": 0.35,
        "info_difficulty_weight": 0.20,
        "info_uncertainty_weight": 0.25,
        "info_urgency_weight": 0.20,
    }
    assert compute_information_sufficiency(context, 0.35, reward_cfg) > 0.8
