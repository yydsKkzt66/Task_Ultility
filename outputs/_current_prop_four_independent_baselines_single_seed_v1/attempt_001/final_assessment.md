# Current PROP and four independent baselines — single seed

Scenario: ordinary-random V25, seed=2026, capacity=0.5, 24 h, 40 W, task hash `c31d50bf0633e0d7395f78a785768c2ac492eb5f707c02a9e5be93b1226b37fa`.

| method | J_all | J_HV | all/HV/high/urgent closure | energy Wh | tx MB | mean/p95 latency s |
|---|---:|---:|---:|---:|---:|---:|
| PROP-V2-SAFE-METADATA-CURRENT-v1 | 46.740497 | 39.192798 | 99/54/39/15 | 2.212141 | 164.032539 | 2336.33/4442.00 |
| B1-COMMON-GROUND-THERMAL-CORRECTED | -12.493610 | -8.198499 | 226/56/40/16 | 5.914911 | 904.000000 | 2616.51/4645.94 |
| OFFLOAD-EFT-THERMAL-ALIGNED-2NODE-DIAGNOSTIC | 15.749553 | 10.258982 | 230/56/40/16 | 7.171152 | 573.500000 | 2453.65/4431.44 |
| TUT-PROXY-THERMAL-CORRECTED | 38.734473 | 26.486780 | 118/53/37/16 | 2.908935 | 227.082000 | 2425.61/4424.87 |
| MYOPIC-E2E-UTILITY-GREEDY-STANDALONE-v1 | 40.300654 | 2.510594 | 116/8/8/0 | 2.793588 | 165.716193 | 2535.70/4595.00 |

## Identity conclusion

Current PROP is V1 safe-set-consistency + V2 HV metadata/residual progressive delivery: **True**.
Its selector mode in this current thermal branch is: **safe-greedy replacement after prior SafeLinUCB diagnostic**.
The new MYOPIC is a standalone baseline, not an ablation behind PROP and not a direct public-evaluator oracle.
Execution mode: **four fresh parallel runs plus one hash-compatible prior OFFLOAD reuse**; OFFLOAD reuse identity gate: **True**.

CURRENT_PROP_FOUR_INDEPENDENT_BASELINES_SINGLE_SEED_COMPLETE
