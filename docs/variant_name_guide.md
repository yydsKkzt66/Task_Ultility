# Variant Name Guide

后续读结果时优先看 `variant_short`，再结合 `experiment_stage`。同一个短名在不同 stage 中是 stage-local 的，不代表完全相同配置。

| short | full name | 中文含义 |
| --- | --- | --- |
| `PROP-BASE` | stage-local proposed baseline | 当前阶段的 proposed 基线；nominal/stress 是 `proposed_lyapunov_rule_mode_safe_bandit`，policy repair 是 `proposed_yolo_calibrated_baseline`，payload calibration 是 `proposed_yolo_profile_policy_repair_baseline`。 |
| `PROP-OLD` | `proposed_original_profile` | YOLO profile 校准前的 proposed。 |
| `PROP-YOLO` | `proposed_yolo_calibrated_profile` | 只接入 YOLO-inspired calibrated model profile 的 proposed。 |
| `PROP-NO-UCB` | `proposed_yolo_calibrated_no_ucb_bonus` | YOLO profile ablation 中关闭 UCB bonus 的诊断变体。 |
| `PROP-NO-MISS` | `proposed_yolo_calibrated_no_mission_priority_boost` | YOLO profile ablation 中关闭 mission priority boost 的诊断变体。 |
| `PROP-RES` | `proposed_yolo_calibrated_stronger_resource_price` | YOLO profile ablation 中加强 resource price 的诊断变体。 |
| `PROP-UCB` | stage-local cost-aware UCB variant | policy repair 中是 `proposed_cost_aware_ucb`；payload calibration 中是 `proposed_calibrated_payload_plus_cost_aware_ucb`。 |
| `PROP-DELIV` | `proposed_delivery_aware_info_gain` | policy repair 中只启用 delivery-aware information gain。 |
| `PROP-Z` | stage-local z-guided coupling variant | policy repair 中是 `proposed_z_guided_coupling`；payload calibration 中是 `proposed_calibrated_payload_plus_z_guided`。 |
| `PROP-U+D` | `proposed_ucb_plus_delivery` | policy repair 中同时启用 cost-aware UCB 和 delivery-aware info gain。 |
| `PROP-REPAIR` | `proposed_repaired_combined` | policy repair 中启用 UCB、delivery-aware info gain、z-guided coupling 的组合修复。 |
| `PROP-PAY` | `proposed_calibrated_payload_profile` | 只接入 calibrated payload profile。 |
| `PROP-PROXY` | `proposed_payload_specific_delivery_proxy` | calibrated payload profile 加 payload-specific delivery proxy。 |
| `PROP-COMB` | `proposed_payload_repaired_combined` | payload calibration 中启用 calibrated payload、payload-specific proxy、UCB、z-guided coupling 的组合变体。 |
| `UTIL` | `utility_rule` | 规则式 utility baseline；只能和同一 stage 内的其他 variant 比较。 |
| `NO-MODE` | `no_mode` | 关闭 mode controller 的 SafeLinUCB baseline；继承所在 stage 的 profile/payload 配置。 |
| `NO-LYA` | `no_lyapunov` | 关闭 Lyapunov pricing 的 SafeLinUCB baseline；继承所在 stage 的 profile/payload 配置。 |
| `SAFE-LINUCB` | `safe_linucb` | 较早阶段的 safe LinUCB baseline。 |
| `SAFE-BANDIT` | `safe_bandit_only` | 较早阶段的 safe-bandit-only baseline。 |
| `COMM-CONS` | `comm_conservative` | 通信保守规则 baseline。 |
| `FIX-L` | `fixed_lightweight` | 固定 lightweight+metadata。 |
| `FIX-H` | `fixed_high_accuracy` | 固定 high_accuracy+roi_context。 |

完整机器可读映射见 `configs/variant_aliases.yaml`。
