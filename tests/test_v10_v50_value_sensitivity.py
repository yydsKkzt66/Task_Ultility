from decimal import Decimal

from scripts.run_v10_v50_cap050_five_method_multiseed_v1 import (
    EXPECTED_COUNTS,
    VALUE_PROFILES,
    normalization_alpha,
    profile_weights,
)


def test_profiles_are_the_approved_ratios():
    assert profile_weights("V10") == {
        "low": Decimal("1"),
        "medium": Decimal("2"),
        "high": Decimal("5"),
        "urgent": Decimal("10"),
    }
    assert profile_weights("V50") == {
        "low": Decimal("1"),
        "medium": Decimal("5"),
        "high": Decimal("20"),
        "urgent": Decimal("50"),
    }


def test_mean_preserving_normalization_is_exact():
    for profile in VALUE_PROFILES:
        weights = profile_weights(profile)
        alpha = normalization_alpha(weights)
        total = sum(
            Decimal(EXPECTED_COUNTS[level]) * alpha * weights[level]
            for level in ("low", "medium", "high", "urgent")
        )
        assert total == Decimal("268")


def test_v10_and_v50_alphas_are_frozen_by_counts():
    assert normalization_alpha(profile_weights("V10")) == Decimal(268) / Decimal(812)
    assert normalization_alpha(profile_weights("V50")) == Decimal(268) / Decimal(2879)
