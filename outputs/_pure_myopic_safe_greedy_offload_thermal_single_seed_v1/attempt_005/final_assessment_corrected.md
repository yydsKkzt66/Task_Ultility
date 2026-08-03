# PURE-MYOPIC-SAFE-GREEDY-v2 and OFFLOAD thermal comparison

Scenario: ordinary-random V25, seed 2026, capacity 0.5, 24 h, 40 W power environment, common provisional two-node thermal model.

| method | J_all_net | J_HV_net | all/HV/high/urgent closure | energy Wh | transmitted MB |
|---|---:|---:|---:|---:|---:|
| PROP-V2-SAFE-METADATA-CURRENT-v1 | 46.740497432 | 39.192798090 | 99/54/39/15 | 2.212141 | 164.032539 |
| PURE-MYOPIC-SAFE-GREEDY-v2 | 48.496097387 | 34.378419270 | 185/54/39/15 | 3.169573 | 263.162348 |
| OFFLOAD-EFT-THERMAL-ALIGNED-2NODE-DIAGNOSTIC | 15.749553015 | 10.258982383 | 230/56/40/16 | 7.171152 | 573.500000 |

PURE disables all long-term Lyapunov prices and resource modes and blocks online bandit updates. It maximizes the current q-once marginal benefit minus the frozen public energy and downlink costs, with Defer equal to zero. No future task or queue information is used. It improves J_all and total closure over current PROP while preserving all three HV closure counts in this single seed, but J_HV falls because additional ordinary traffic raises shared resource costs.

OFFLOAD retains the frozen one-shot earliest-finish LOCAL/GROUND/DEFER policy. It executed 110 LOCAL, 120 GROUND, and 38 DEFER decisions. No net gate, requirement projection, or product/model upgrade was added. Its high closure behavior is retained, but under the common public q-once resource prices its large full-frame traffic makes net utility low. The older native J values are retained only as native-policy diagnostics and are not used in the comparison table.

All three methods use the same thermal parameter hash, all runs have zero SOC/chip/surface violations, and all frozen protected assets were unchanged. The initial test report contained one false negative because it incorrectly required a pre-replacement shadow dynamic cost to be zero. The replacement was active for all 6000 scoring calls; the corrected audit and report are additive and do not alter run outputs.

PURE_MYOPIC_SAFE_GREEDY_OFFLOAD_THERMAL_SINGLE_SEED_COMPLETE
