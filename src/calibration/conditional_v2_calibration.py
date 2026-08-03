"""Pure helpers for conditional q/price calibration of decoupled PROP-V2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import statistics
from typing import Any, Iterable, Mapping


MODEL_ORDER = ("lightweight", "balanced", "high_accuracy")
PRODUCT_ORDER = ("metadata", "roi", "roi_context")
PROFILE = {
    "lightweight": {"quality_base": 0.74, "robustness": 0.70},
    "balanced": {"quality_base": 0.87, "robustness": 0.84},
    "high_accuracy": {"quality_base": 0.94, "robustness": 0.91},
}


def _f(value: Any) -> float:
    return float(value)


def enforce_strict_quality_order(values: Mapping[str, float], epsilon: float = 1.0e-4) -> dict[str, float]:
    low = max(0.01, min(0.98, float(values["lightweight"])))
    balanced = max(low + epsilon, min(0.99 - epsilon, float(values["balanced"])))
    high = max(balanced + epsilon, min(0.99, float(values["high_accuracy"])))
    return {"lightweight": low, "balanced": balanced, "high_accuracy": high}


def task_conditioned_quality(model: str, difficulty: float, uncertainty: float) -> float:
    profile = PROFILE[model]
    return max(
        0.05,
        min(
            1.0,
            profile["quality_base"]
            - (1.0 - profile["robustness"]) * 0.32 * float(difficulty)
            - 0.16 * float(uncertainty),
        ),
    )


def derive_quality_candidates(task_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive q candidates from independent task covariates, never method outcomes."""

    tasks = list(task_rows)
    by_model = {
        model: [task_conditioned_quality(model, _f(task["difficulty"]), _f(task["uncertainty"])) for task in tasks]
        for model in MODEL_ORDER
    }
    mean_map = enforce_strict_quality_order({model: statistics.fmean(values) for model, values in by_model.items()})
    lower_map = enforce_strict_quality_order(
        {
            model: sorted(values)[max(0, int(math.floor(0.25 * (len(values) - 1))))]
            for model, values in by_model.items()
        }
    )
    shrunk_map = enforce_strict_quality_order(
        {
            model: 0.5 * mean_map[model] + 0.5 * PROFILE[model]["quality_base"]
            for model in MODEL_ORDER
        }
    )
    return [
        {"q_id": "Q_MEAN", "q_eff": mean_map, "derivation": "mean task-conditioned frozen proxy on calibration tasks"},
        {"q_id": "Q_LOWER_QUARTILE", "q_eff": lower_map, "derivation": "25th-percentile task-conditioned frozen proxy on calibration tasks"},
        {"q_id": "Q_SHRUNK_TO_PROFILE", "q_eff": shrunk_map, "derivation": "50% calibration mean + 50% profile base"},
    ]


def robust_median(values: Iterable[float], fallback: float) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)) and float(value) > 0.0)
    if not clean:
        return float(fallback)
    if len(clean) >= 8:
        lo, hi = int(0.10 * len(clean)), int(0.90 * len(clean))
        clean = clean[lo:max(lo + 1, hi)]
    return statistics.median(clean)


def derive_shadow_price_center(
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    energy_reference_j: float,
    queue_max_mb: float,
) -> dict[str, Any]:
    """Estimate fixed prices from observed marginal gross/resource slopes."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(str(row.get("seed", "")), str(row["task_id"]))].append(row)
    energy_slopes: list[float] = []
    downlink_slopes: list[float] = []
    for rows in grouped.values():
        table = {(str(row["model"]), str(row["product"])): row for row in rows}
        for product in PRODUCT_ORDER:
            for lower, upper in zip(MODEL_ORDER, MODEL_ORDER[1:]):
                if (lower, product) not in table or (upper, product) not in table:
                    continue
                low, high = table[(lower, product)], table[(upper, product)]
                delta_g = _f(high["v2_positive_benefit"]) - _f(low["v2_positive_benefit"])
                delta_e_wh = (
                    _f(high["v2_energy_term"]) - _f(low["v2_energy_term"])
                ) * float(energy_reference_j) / 3600.0
                if delta_g > 0.0 and delta_e_wh > 1.0e-12:
                    energy_slopes.append(delta_g / delta_e_wh)
        for model in MODEL_ORDER:
            for lower, upper in zip(PRODUCT_ORDER, PRODUCT_ORDER[1:]):
                if (model, lower) not in table or (model, upper) not in table:
                    continue
                low, high = table[(model, lower)], table[(model, upper)]
                delta_g = _f(high["v2_positive_benefit"]) - _f(low["v2_positive_benefit"])
                delta_mb = (
                    _f(high["v2_payload_term"]) - _f(low["v2_payload_term"])
                ) * float(queue_max_mb)
                if delta_g > 0.0 and delta_mb > 1.0e-12:
                    downlink_slopes.append(delta_g / delta_mb)
    energy = robust_median(energy_slopes, 1.0)
    downlink = robust_median(downlink_slopes, 0.01)
    return {
        "energy_price_per_Wh": energy,
        "downlink_price_per_MB": downlink,
        "terminal_backlog_price_per_MB": downlink,
        "energy_slope_count": len(energy_slopes),
        "downlink_slope_count": len(downlink_slopes),
        "estimator": "10%-trimmed median positive adjacent Pareto slope",
    }


def derive_budget_perturbation_prices(
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    reference_budgets_by_seed: Mapping[str, Mapping[str, float]],
    energy_reference_j: float,
    queue_max_mb: float,
    perturbation: float = 0.10,
) -> dict[str, Any]:
    """Estimate shadow prices from a method-neutral multiple-choice LP.

    Every task may receive at most one safe/deadline-feasible action.  The LP
    maximizes only q-once predicted gross under aggregate Wh and MB budgets.
    Central finite differences at 0.9/1.1 budgets provide the public prices.
    """

    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    by_seed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if str(row.get("product", "")) == "defer":
            continue
        if int(float(row.get("safe_filter_pass", 0) or 0)) != 1 or int(float(row.get("deadline_feasible", 0) or 0)) != 1:
            continue
        by_seed[str(row.get("seed", ""))].append(row)
    estimates = []
    for seed, rows in sorted(by_seed.items()):
        budgets = reference_budgets_by_seed[seed]
        task_ids = sorted({str(row["task_id"]) for row in rows})
        task_index = {task_id: idx for idx, task_id in enumerate(task_ids)}
        n = len(rows)
        matrix = lil_matrix((len(task_ids) + 2, n), dtype=float)
        gross = []
        for column, row in enumerate(rows):
            matrix[task_index[str(row["task_id"])], column] = 1.0
            matrix[len(task_ids), column] = float(row["v2_energy_term"]) * energy_reference_j / 3600.0
            matrix[len(task_ids) + 1, column] = float(row["v2_payload_term"]) * queue_max_mb
            gross.append(float(row["v2_positive_benefit"]))
        matrix = matrix.tocsr()

        def solve(energy_factor: float, downlink_factor: float) -> tuple[float, float, float]:
            e_budget = max(1.0e-9, float(budgets["energy_Wh"]) * energy_factor)
            b_budget = max(1.0e-9, float(budgets["downlink_MB"]) * downlink_factor)
            rhs = [1.0] * len(task_ids) + [e_budget, b_budget]
            result = linprog(c=[-value for value in gross], A_ub=matrix, b_ub=rhs, bounds=(0.0, 1.0), method="highs")
            if not result.success:
                raise RuntimeError({"budget_lp_failed": seed, "message": result.message})
            marginals = result.ineqlin.marginals
            return -float(result.fun), max(0.0, -float(marginals[-2])), max(0.0, -float(marginals[-1]))

        center, dual_e, dual_b = solve(1.0, 1.0)
        e_low, _, _ = solve(1.0 - perturbation, 1.0)
        e_high, _, _ = solve(1.0 + perturbation, 1.0)
        b_low, _, _ = solve(1.0, 1.0 - perturbation)
        b_high, _, _ = solve(1.0, 1.0 + perturbation)
        finite_e = max(0.0, (e_high - e_low) / (2.0 * perturbation * max(float(budgets["energy_Wh"]), 1.0e-9)))
        finite_b = max(0.0, (b_high - b_low) / (2.0 * perturbation * max(float(budgets["downlink_MB"]), 1.0e-9)))
        estimates.append(
            {
                "seed": seed,
                "reference_energy_Wh": budgets["energy_Wh"],
                "reference_downlink_MB": budgets["downlink_MB"],
                "center_optimal_gross": center,
                "energy_price_finite_difference": finite_e,
                "downlink_price_finite_difference": finite_b,
                "energy_price_lp_dual": dual_e,
                "downlink_price_lp_dual": dual_b,
                "gross_energy_0.9x": e_low,
                "gross_energy_1.1x": e_high,
                "gross_downlink_0.9x": b_low,
                "gross_downlink_1.1x": b_high,
            }
        )
    energy_values = [row["energy_price_finite_difference"] for row in estimates]
    downlink_values = [row["downlink_price_finite_difference"] for row in estimates]
    return {
        "energy_price_per_Wh": statistics.median(energy_values) if energy_values else 0.0,
        "downlink_price_per_MB": statistics.median(downlink_values) if downlink_values else 0.0,
        "terminal_backlog_price_per_MB": statistics.median(downlink_values) if downlink_values else 0.0,
        "perturbation": perturbation,
        "method": "multiple-choice LP central budget perturbation",
        "per_seed": estimates,
    }


def price_candidates(center: Mapping[str, float], factors: Iterable[float] = (0.75, 1.0, 1.25)) -> dict[str, list[dict[str, float | str]]]:
    return {
        "energy": [
            {"price_id": f"E_{factor:.2f}x", "factor": factor, "value": factor * float(center["energy_price_per_Wh"])}
            for factor in factors
        ],
        "downlink": [
            {"price_id": f"B_{factor:.2f}x", "factor": factor, "value": factor * float(center["downlink_price_per_MB"])}
            for factor in factors
        ],
    }


def online_lyapunov_baseline(
    *,
    energy_price_per_Wh: float,
    downlink_price_per_MB: float,
    terminal_backlog_price_per_MB: float,
    energy_reference_j: float,
    queue_max_mb: float,
    beta_resource_cost: float,
    base_lambda_L: float,
) -> dict[str, float]:
    beta = max(float(beta_resource_cost), 1.0e-12)
    return {
        "base_lambda_E": float(energy_price_per_Wh) * float(energy_reference_j) / (3600.0 * beta),
        "base_lambda_L": float(base_lambda_L),
        "base_lambda_Q": float(terminal_backlog_price_per_MB) * float(queue_max_mb) / beta,
        "base_lambda_S": float(downlink_price_per_MB) * float(queue_max_mb) / beta,
    }


@dataclass(frozen=True)
class UpgradeThreshold:
    task_id: str
    product: str
    lower_model: str
    upper_model: str
    energy_break_even_price_per_Wh: float | None
    downlink_price_per_MB: float
    positive_gain_delta: float
    energy_delta_Wh: float
    downlink_delta_MB: float
    gain_after_downlink_cost: float


def model_upgrade_thresholds(
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    energy_reference_j: float,
    queue_max_mb: float,
    downlink_price_per_mb: float,
) -> list[UpgradeThreshold]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[(str(row.get("seed", "")), str(row["task_id"]))].append(row)
    output: list[UpgradeThreshold] = []
    for rows in grouped.values():
        table = {(str(row["model"]), str(row["product"])): row for row in rows}
        task_id = str(rows[0]["task_id"])
        for product in PRODUCT_ORDER:
            for lower, upper in zip(MODEL_ORDER, MODEL_ORDER[1:]):
                if (lower, product) not in table or (upper, product) not in table:
                    continue
                low, high = table[(lower, product)], table[(upper, product)]
                delta_g = _f(high["v2_positive_benefit"]) - _f(low["v2_positive_benefit"])
                delta_e = (_f(high["v2_energy_term"]) - _f(low["v2_energy_term"])) * energy_reference_j / 3600.0
                delta_b = (_f(high["v2_payload_term"]) - _f(low["v2_payload_term"])) * queue_max_mb
                gain_after_downlink = delta_g - float(downlink_price_per_mb) * delta_b
                threshold = gain_after_downlink / delta_e if gain_after_downlink > 0.0 and delta_e > 1.0e-12 else None
                output.append(
                    UpgradeThreshold(
                        task_id,
                        product,
                        lower,
                        upper,
                        threshold,
                        float(downlink_price_per_mb),
                        delta_g,
                        delta_e,
                        delta_b,
                        gain_after_downlink,
                    )
                )
    return output
