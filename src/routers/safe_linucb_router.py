"""Safe LinUCB router銆?"""

from __future__ import annotations

import numpy as np

from src.common.types import ContextVector, JointAction, PredictedMetrics
from src.routers.base_router import BaseRouter


class SafeLinUCBRouter(BaseRouter):
    """浣跨敤 LinUCB 鍋?raw action 閫夋嫨锛屽悗缁敱缁熶竴 safety shield 鍐冲畾鎵ц鍔ㄤ綔銆?"""

    def __init__(
        self,
        action_names: list[str],
        alpha: float = 0.70,
        ridge: float = 1.0,
        router_name: str = "safe_linucb",
        use_lyapunov_pricing: bool = False,
        use_mode_controller: bool = False,
    ) -> None:
        super().__init__(router_name)
        self.alpha = alpha
        self.ridge = ridge
        self.action_names = action_names
        self.dimension = 9
        self.A = {name: np.eye(self.dimension) * ridge for name in action_names}
        self.b = {name: np.zeros(self.dimension) for name in action_names}
        self.use_lyapunov_pricing = use_lyapunov_pricing
        self.use_mode_controller = use_mode_controller
        self.use_guided_utility = use_lyapunov_pricing or use_mode_controller

    @staticmethod
    def _positive(value: float) -> float:
        return max(0.0, float(value))

    @staticmethod
    def _scoring_config(guidance_bundle: dict[str, object] | None) -> dict[str, object]:
        repair_cfg = guidance_bundle.get("policy_repair", {}) if guidance_bundle else {}
        if not isinstance(repair_cfg, dict):
            repair_cfg = {}
        section = repair_cfg.get("utility_guided_scoring", {})
        if not isinstance(section, dict):
            section = {}

        def _cfg_value(name: str, default: object) -> object:
            if name in section:
                return section[name]
            return repair_cfg.get(name, default)

        enabled = bool(repair_cfg.get("enable_utility_guided_scoring", False))
        preserve_logging = bool(_cfg_value("preserve_legacy_score_logging", enabled))
        return {
            "enabled": enabled,
            "utility_prior_weight": float(_cfg_value("utility_prior_weight", 1.0)),
            "learned_correction_weight": float(_cfg_value("learned_correction_weight", 0.20)),
            "exploration_weight": float(_cfg_value("exploration_weight", 1.0)),
            "score_component_normalization": str(
                _cfg_value("score_component_normalization", "zscore_by_candidate_set")
            ),
            "fallback_if_zero_variance": bool(_cfg_value("fallback_if_zero_variance", True)),
            "preserve_legacy_score_logging": preserve_logging,
        }

    @staticmethod
    def _product_requirement_scoring_config(guidance_bundle: dict[str, object] | None) -> dict[str, object]:
        cfg = guidance_bundle.get("module2_product_requirement_aware_scoring", {}) if guidance_bundle else {}
        if not isinstance(cfg, dict):
            cfg = {}
        return {
            "enabled": bool(cfg.get("enabled", False)),
            "requirement_fit_weight": float(cfg.get("requirement_fit_weight", 0.0)),
            "model_fit_weight": float(cfg.get("model_fit_weight", 0.0)),
            "deadline_feasibility_weight": float(cfg.get("deadline_feasibility_weight", 0.0)),
            "staged_delivery_feasibility_weight": float(cfg.get("staged_delivery_feasibility_weight", 0.0)),
        }

    @staticmethod
    def _product_requirement_context(guidance_bundle: dict[str, object] | None) -> dict[str, object]:
        context = guidance_bundle.get("product_requirement_context", {}) if guidance_bundle else {}
        return context if isinstance(context, dict) else {}

    @staticmethod
    def _resource_context(guidance_bundle: dict[str, object] | None) -> dict[str, object]:
        context = guidance_bundle.get("resource_context", {}) if guidance_bundle else {}
        return context if isinstance(context, dict) else {}

    @staticmethod
    def _payload_requirement_fit(requirement: str, payload_policy: str) -> float:
        requirement = str(requirement or "verification").strip().lower()
        payload_policy = str(payload_policy or "").strip().lower()
        matrix = {
            "alert": {
                "metadata": 1.0,
                "metadata_then_roi": 0.85,
                "metadata_then_roi_context": 0.85,
                "roi": 0.70,
                "roi_context": 0.60,
                "defer": 0.0,
            },
            "verification": {
                "metadata": 0.35,
                "metadata_then_roi": 0.90,
                "metadata_then_roi_context": 0.80,
                "roi": 1.0,
                "roi_context": 0.85,
                "defer": 0.0,
            },
            "context": {
                "metadata": 0.20,
                "metadata_then_roi": 0.55,
                "metadata_then_roi_context": 0.95,
                "roi": 0.60,
                "roi_context": 1.0,
                "defer": 0.0,
            },
            "evidence": {
                "metadata": 0.15,
                "metadata_then_roi": 0.45,
                "metadata_then_roi_context": 0.95,
                "roi": 0.55,
                "roi_context": 1.0,
                "defer": 0.0,
            },
        }
        return float(matrix.get(requirement, matrix["verification"]).get(payload_policy, 0.0))

    @staticmethod
    def _model_requirement_fit(requirement: str, model_name: str, uncertainty: float) -> float:
        requirement = str(requirement or "verification").strip().lower()
        model_name = str(model_name or "").strip().lower()
        complexity = 0.0
        if "high" in model_name:
            complexity = 1.0
        elif "balanced" in model_name or "medium" in model_name:
            complexity = 0.65
        elif "light" in model_name or "tiny" in model_name or "small" in model_name:
            complexity = 0.35
        elif "delay" in model_name or "defer" in model_name:
            complexity = 0.0
        uncertainty = max(0.0, min(1.0, float(uncertainty)))
        target = {
            "alert": 0.25,
            "verification": 0.35 + 0.25 * uncertainty,
            "context": 0.55 + 0.15 * uncertainty,
            "evidence": 0.65 + 0.20 * uncertainty,
        }.get(requirement, 0.45)
        return max(0.0, min(1.0, 1.0 - abs(complexity - target)))

    @staticmethod
    def _deadline_feasibility_score(
        predicted: PredictedMetrics,
        requirement_context: dict[str, object],
        resource_context: dict[str, object],
    ) -> float:
        deadline_s = max(float(requirement_context.get("deadline_s", 1.0)), 1.0e-9)
        total_latency = float(predicted.predicted_L_inf + predicted.predicted_L_queue + predicted.predicted_L_comm)
        slack_score = max(0.0, min(1.0, 1.0 - total_latency / deadline_s))
        remaining_window = max(0.0, float(resource_context.get("remaining_window_s", 0.0)))
        contact_bonus = 1.0 if bool(resource_context.get("in_contact", False)) and remaining_window >= predicted.predicted_L_comm else 0.4
        return max(0.0, min(1.0, 0.75 * slack_score + 0.25 * contact_bonus))

    @staticmethod
    def _staged_delivery_feasibility_score(
        action: JointAction,
        predicted: PredictedMetrics,
        requirement_context: dict[str, object],
        resource_context: dict[str, object],
    ) -> float:
        payload_policy = str(action.payload_policy or "").strip().lower()
        requirement = str(requirement_context.get("ideal_product_requirement", "verification")).strip().lower()
        deadline_score = SafeLinUCBRouter._deadline_feasibility_score(predicted, requirement_context, resource_context)
        if payload_policy not in {"metadata_then_roi", "metadata_then_roi_context"}:
            return 0.5 * deadline_score
        metadata_bonus = 0.35 if bool(requirement_context.get("initial_alert_required", False)) else 0.15
        residual_need = 0.45 if requirement in {"context", "evidence", "verification"} else 0.15
        queue_penalty = min(0.30, max(0.0, float(predicted.predicted_Q_next_mb)) / 100.0)
        return max(0.0, min(1.0, metadata_bonus + residual_need + 0.30 * deadline_score - queue_penalty))

    def _product_requirement_score_terms(
        self,
        action: JointAction,
        predicted: PredictedMetrics,
        guidance_bundle: dict[str, object] | None,
        cfg: dict[str, object],
    ) -> dict[str, object]:
        requirement_context = self._product_requirement_context(guidance_bundle)
        resource_context = self._resource_context(guidance_bundle)
        requirement = str(requirement_context.get("ideal_product_requirement", "verification"))
        uncertainty = float(requirement_context.get("uncertainty", 0.0))
        requirement_fit = self._payload_requirement_fit(requirement, action.payload_policy)
        model_fit = self._model_requirement_fit(requirement, action.model_name, uncertainty)
        deadline_fit = self._deadline_feasibility_score(predicted, requirement_context, resource_context)
        staged_fit = self._staged_delivery_feasibility_score(action, predicted, requirement_context, resource_context)
        bonus = (
            float(cfg["requirement_fit_weight"]) * requirement_fit
            + float(cfg["model_fit_weight"]) * model_fit
            + float(cfg["deadline_feasibility_weight"]) * deadline_fit
            + float(cfg["staged_delivery_feasibility_weight"]) * staged_fit
        )
        return {
            "ideal_product_requirement": requirement,
            "product_requirement_aware_enabled": int(bool(cfg["enabled"])),
            "requirement_fit_score": requirement_fit,
            "model_fit_score": model_fit,
            "deadline_feasibility_score": deadline_fit,
            "staged_delivery_feasibility_score": staged_fit,
            "product_requirement_aware_bonus": bonus,
        }

    @staticmethod
    def _zscore_by_candidate_set(values: list[float], fallback_if_zero_variance: bool) -> list[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=float)
        std = float(np.std(arr))
        if std <= 1.0e-12:
            if fallback_if_zero_variance:
                return [0.0 for _ in values]
            std = 1.0
        mean = float(np.mean(arr))
        return [float((value - mean) / std) for value in values]

    @staticmethod
    def _best_name(rows: list[dict[str, object]], score_key: str) -> str:
        best_name = ""
        best_score = -float("inf")
        for row in rows:
            score = float(row[score_key])
            if score > best_score:
                best_score = score
                best_name = str(row["action_name"])
        return best_name

    @staticmethod
    def _guidance_float(guidance_bundle: dict[str, object] | None, key: str, default: float = 0.0) -> float:
        if not guidance_bundle:
            return default
        try:
            return float(guidance_bundle.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _cost_aware_enabled(guidance_bundle: dict[str, object] | None) -> bool:
        repair_cfg = guidance_bundle.get("policy_repair", {}) if guidance_bundle else {}
        return isinstance(repair_cfg, dict) and bool(repair_cfg.get("enable_cost_aware_ucb", False))

    def _cost_aware_ucb(
        self,
        context: ContextVector,
        action: JointAction,
        predicted: PredictedMetrics,
        raw_ucb_bonus: float,
        guidance_bundle: dict[str, object] | None,
    ) -> tuple[float, float, str]:
        if not guidance_bundle:
            return 1.0, raw_ucb_bonus, "disabled"
        repair_cfg = guidance_bundle.get("policy_repair", {})
        if not isinstance(repair_cfg, dict) or not bool(repair_cfg.get("enable_cost_aware_ucb", False)):
            return 1.0, raw_ucb_bonus, "disabled"
        cfg = dict(repair_cfg.get("cost_aware_ucb", {}))
        resource_context = dict(guidance_bundle.get("resource_context", {}))
        queue_pressure = max(0.0, min(1.0, context.Q_t))
        dod = max(0.0, 1.0 - context.B_t)
        deadline_urgency = max(0.0, min(1.0, context.D_t))
        no_contact = 1.0 if not bool(resource_context.get("in_contact", False)) else 0.0
        eclipse = 1.0 if bool(resource_context.get("is_eclipse", False)) else 0.0
        energy_norm = predicted.predicted_E_comp / max(float(cfg.get("energy_ref_j", 60.0)), 1.0e-9)
        latency_norm = predicted.predicted_L_inf / max(float(cfg.get("latency_ref_s", 3.0)), 1.0e-9)
        payload_norm = predicted.predicted_S_gen_mb / max(float(cfg.get("payload_ref_mb", 1.6)), 1.0e-9)
        thermal_norm = max(0.0, predicted.predicted_Theta_next - context.Theta_t) / max(
            float(cfg.get("thermal_delta_ref", 0.10)),
            1.0e-9,
        )
        risk = (
            float(cfg.get("energy_weight", 0.25)) * energy_norm
            + float(cfg.get("latency_weight", 0.10)) * latency_norm
            + float(cfg.get("payload_weight", 0.20)) * payload_norm
            + float(cfg.get("thermal_weight", 0.20)) * thermal_norm
            + float(cfg.get("dod_weight", 0.35)) * dod
            + float(cfg.get("eclipse_weight", 0.20)) * eclipse
            + float(cfg.get("no_contact_weight", 0.25)) * no_contact
            + float(cfg.get("queue_weight", 0.20)) * queue_pressure
            + float(cfg.get("deadline_weight", 0.15)) * deadline_urgency
        )
        min_discount = float(cfg.get("min_discount", 0.15))
        risk_discount = max(min_discount, min(1.0, 1.0 / (1.0 + max(0.0, risk))))
        reasons = []
        if energy_norm > 0.8:
            reasons.append("high_energy")
        if payload_norm > 0.8:
            reasons.append("large_payload")
        if dod > float(cfg.get("dod_reason_threshold", 0.30)):
            reasons.append("high_dod")
        if eclipse:
            reasons.append("eclipse")
        if no_contact:
            reasons.append("no_contact")
        if queue_pressure > float(cfg.get("queue_reason_threshold", 0.20)):
            reasons.append("queue_pressure")
        if deadline_urgency > float(cfg.get("deadline_reason_threshold", 0.70)):
            reasons.append("deadline_tight")
        return risk_discount, raw_ucb_bonus * risk_discount, ";".join(reasons) if reasons else "low_risk"

    def select_action(
        self,
        context: ContextVector,
        candidate_actions: list[JointAction],
        predicted_map: dict[str, PredictedMetrics],
        guidance_bundle: dict[str, object] | None = None,
    ) -> JointAction:
        x = context.as_array()
        exploration_scale = 1.0 if guidance_bundle is None else float(guidance_bundle.get("exploration_scale", 1.0))
        scoring_cfg = self._scoring_config(guidance_bundle)
        product_req_cfg = self._product_requirement_scoring_config(guidance_bundle)
        use_utility_guided = bool(scoring_cfg["enabled"])
        use_product_requirement_aware = bool(product_req_cfg["enabled"]) and any(
            abs(float(product_req_cfg[key])) > 1.0e-12
            for key in (
                "requirement_fit_weight",
                "model_fit_weight",
                "deadline_feasibility_weight",
                "staged_delivery_feasibility_weight",
            )
        )
        normalization = str(scoring_cfg["score_component_normalization"])
        normalize_components = normalization == "zscore_by_candidate_set"
        log_utility_guided_fields = use_utility_guided or bool(scoring_cfg["preserve_legacy_score_logging"])
        self.last_all_score_components = {}

        rows: list[dict[str, object]] = []
        for action in candidate_actions:
            A_inv = np.linalg.inv(self.A[action.action_name])
            theta = A_inv @ self.b[action.action_name]
            exploit = float(theta @ x)
            explore = float(np.sqrt(x @ A_inv @ x))
            predicted = predicted_map[action.action_name]
            utility_prior_raw = float(predicted.predicted_reward)
            prior = 0.20 * utility_prior_raw
            raw_ucb_bonus = (self.alpha * exploration_scale) * explore
            risk_discount, cost_aware_ucb_bonus, suppression_reason = self._cost_aware_ucb(
                context,
                action,
                predicted,
                raw_ucb_bonus,
                guidance_bundle,
            )
            rows.append(
                {
                    "action": action,
                    "action_name": action.action_name,
                    "linucb_exploit": exploit,
                    "linucb_explore_norm": explore,
                    "raw_ucb_bonus": raw_ucb_bonus,
                    "risk_discount": risk_discount,
                    "cost_aware_ucb_bonus": cost_aware_ucb_bonus,
                    "ucb_suppression_reason": suppression_reason,
                    "predicted_reward_prior": prior,
                    "utility_prior_raw": utility_prior_raw,
                    "learned_correction_raw": exploit,
                    "safe_exploration_raw": cost_aware_ucb_bonus,
                    **self._product_requirement_score_terms(action, predicted, guidance_bundle, product_req_cfg),
                }
            )

        if normalize_components:
            utility_prior_norms = self._zscore_by_candidate_set(
                [float(row["utility_prior_raw"]) for row in rows],
                bool(scoring_cfg["fallback_if_zero_variance"]),
            )
            learned_correction_norms = self._zscore_by_candidate_set(
                [float(row["learned_correction_raw"]) for row in rows],
                bool(scoring_cfg["fallback_if_zero_variance"]),
            )
            safe_exploration_norms = self._zscore_by_candidate_set(
                [float(row["safe_exploration_raw"]) for row in rows],
                bool(scoring_cfg["fallback_if_zero_variance"]),
            )
        else:
            utility_prior_norms = [float(row["utility_prior_raw"]) for row in rows]
            learned_correction_norms = [float(row["learned_correction_raw"]) for row in rows]
            safe_exploration_norms = [float(row["safe_exploration_raw"]) for row in rows]

        utility_prior_weight = float(scoring_cfg["utility_prior_weight"])
        learned_correction_weight = float(scoring_cfg["learned_correction_weight"])
        exploration_weight = float(scoring_cfg["exploration_weight"])
        cost_aware_enabled = self._cost_aware_enabled(guidance_bundle)
        utility_prior_uses_z_pricing = bool(guidance_bundle is not None and self.use_lyapunov_pricing)
        learned_correction_uses_z = False
        safe_exploration_uses_z = bool(
            guidance_bundle is not None
            and (abs(exploration_scale - 1.0) > 1.0e-12 or cost_aware_enabled)
        )

        for index, row in enumerate(rows):
            row["utility_prior_norm"] = utility_prior_norms[index]
            row["learned_correction_norm"] = learned_correction_norms[index]
            row["safe_exploration_norm"] = safe_exploration_norms[index]
            legacy_score = (
                float(row["linucb_exploit"])
                + float(row["cost_aware_ucb_bonus"])
                + float(row["predicted_reward_prior"])
            )
            if normalize_components:
                utility_score_component = float(row["utility_prior_norm"])
                learned_score_component = float(row["learned_correction_norm"])
                exploration_score_component = float(row["safe_exploration_norm"])
            else:
                utility_score_component = float(row["utility_prior_raw"])
                learned_score_component = float(row["learned_correction_raw"])
                exploration_score_component = float(row["safe_exploration_raw"])
            utility_guided_score = (
                utility_prior_weight * utility_score_component
                + learned_correction_weight * learned_score_component
                + exploration_weight * exploration_score_component
            )
            product_requirement_aware_bonus = float(row["product_requirement_aware_bonus"]) if use_product_requirement_aware else 0.0
            row["legacy_score_shadow"] = legacy_score
            row["utility_guided_score_shadow"] = utility_guided_score
            row["product_requirement_aware_score_shadow"] = utility_guided_score + float(row["product_requirement_aware_bonus"])
            row["final_action_score"] = (
                utility_guided_score + product_requirement_aware_bonus if use_utility_guided else legacy_score
            )

        legacy_selected_name = self._best_name(rows, "legacy_score_shadow")
        utility_guided_selected_name = self._best_name(rows, "utility_guided_score_shadow")
        selected_name = self._best_name(rows, "final_action_score")
        best_row = next(row for row in rows if row["action_name"] == selected_name)

        for row in rows:
            action_name = str(row["action_name"])
            self.last_all_score_components[action_name] = {
                "linucb_exploit": float(row["linucb_exploit"]),
                "linucb_explore_norm": float(row["linucb_explore_norm"]),
                "raw_ucb_bonus": float(row["raw_ucb_bonus"]),
                "risk_discount": float(row["risk_discount"]),
                "cost_aware_ucb_bonus": float(row["cost_aware_ucb_bonus"]),
                "ucb_suppression_reason": str(row["ucb_suppression_reason"]),
                "predicted_reward_prior": float(row["predicted_reward_prior"]),
                "final_action_score": float(row["final_action_score"]),
                "beta_before": self.alpha,
                "beta_after": self.alpha * float(row["risk_discount"]),
                "scoring_mode": "utility_guided" if use_utility_guided else "legacy",
                "utility_prior_raw": float(row["utility_prior_raw"]),
                "utility_prior_norm": float(row["utility_prior_norm"]),
                "learned_correction_raw": float(row["learned_correction_raw"]),
                "learned_correction_norm": float(row["learned_correction_norm"]),
                "safe_exploration_raw": float(row["safe_exploration_raw"]),
                "safe_exploration_norm": float(row["safe_exploration_norm"]),
                "utility_prior_weight": utility_prior_weight,
                "learned_correction_weight": learned_correction_weight,
                "exploration_weight": exploration_weight,
                "score_component_normalization": normalization,
                "legacy_score_shadow": float(row["legacy_score_shadow"]),
                "utility_guided_score_shadow": float(row["utility_guided_score_shadow"]),
                "ideal_product_requirement": str(row["ideal_product_requirement"]),
                "product_requirement_aware_enabled": int(use_product_requirement_aware),
                "requirement_fit_score": float(row["requirement_fit_score"]),
                "model_fit_score": float(row["model_fit_score"]),
                "deadline_feasibility_score": float(row["deadline_feasibility_score"]),
                "staged_delivery_feasibility_score": float(row["staged_delivery_feasibility_score"]),
                "product_requirement_aware_bonus": float(row["product_requirement_aware_bonus"]) if use_product_requirement_aware else 0.0,
                "product_requirement_aware_score_shadow": float(row["product_requirement_aware_score_shadow"]),
                "requirement_fit_weight": float(product_req_cfg["requirement_fit_weight"]),
                "model_fit_weight": float(product_req_cfg["model_fit_weight"]),
                "deadline_feasibility_weight": float(product_req_cfg["deadline_feasibility_weight"]),
                "staged_delivery_feasibility_weight": float(product_req_cfg["staged_delivery_feasibility_weight"]),
                "selected_by_legacy_shadow": int(action_name == legacy_selected_name),
                "selected_by_utility_guided_shadow": int(action_name == utility_guided_selected_name),
                "selected_by_final_score": int(action_name == selected_name),
                "action_flip_vs_legacy_shadow": int(legacy_selected_name != utility_guided_selected_name),
                "log_utility_guided_score_fields": int(log_utility_guided_fields),
                "z_lambda_E": self._guidance_float(guidance_bundle, "lambda_E"),
                "z_lambda_L": self._guidance_float(guidance_bundle, "lambda_L"),
                "z_lambda_Q": self._guidance_float(guidance_bundle, "lambda_Q"),
                "z_lambda_S": self._guidance_float(guidance_bundle, "lambda_S"),
                "z_mode_name": str(guidance_bundle.get("mode_name", "disabled")) if guidance_bundle else "disabled",
                "z_exploration_scale": exploration_scale,
                "z_model_complexity_bias": self._guidance_float(guidance_bundle, "model_complexity_bias"),
                "z_payload_size_bias": self._guidance_float(guidance_bundle, "payload_size_bias"),
                "z_high_value_bonus_scale": self._guidance_float(guidance_bundle, "high_value_bonus_scale", 1.0),
                "z_safety_margin_scale": self._guidance_float(guidance_bundle, "safety_margin_scale", 1.0),
                "utility_prior_uses_z_pricing": int(utility_prior_uses_z_pricing),
                "learned_correction_uses_z": int(learned_correction_uses_z),
                "safe_exploration_uses_z": int(safe_exploration_uses_z),
            }
        self.last_selected_action_score = float(best_row["final_action_score"])
        best_action = best_row["action"]
        self.last_score_components = dict(self.last_all_score_components.get(best_action.action_name, {}))
        return best_action

    def update(self, context: ContextVector, action: JointAction, observed_reward: float) -> None:
        x = context.as_array()
        self.A[action.action_name] = self.A[action.action_name] + np.outer(x, x)
        self.b[action.action_name] = self.b[action.action_name] + observed_reward * x
