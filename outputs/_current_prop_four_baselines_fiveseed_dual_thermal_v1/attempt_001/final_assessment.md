# Current PROP vs four baselines: five-seed dual-node-thermal comparison

V25 ordinary value-blind arrival order, capacity=0.5, corrected 40 W power environment, shared calibrated two-node thermal model.

| method | J_all mean+-std | J_HV mean+-std | all/HV/high/urgent closure mean | energy Wh | tx MB | mean/p95 latency s |
|---|---:|---:|---:|---:|---:|---:|
| PROP-V2-SAFE-METADATA-CURRENT-v1 | 44.908708+-5.465068 | 37.920654+-5.098861 | 103.20/56.20/40.20/16.00 | 2.338567 | 167.189596 | 2434.42/4604.80 |
| B1-COMMON-GROUND-THERMAL-CORRECTED | -15.142552+-4.835086 | -9.648439+-2.751062 | 231.40/59.00/42.20/16.80 | 6.032886 | 925.600000 | 2658.69/4653.43 |
| OFFLOAD-EFT-THERMAL-ALIGNED-2NODE-DIAGNOSTIC | 14.696683+-4.400577 | 9.463787+-3.232113 | 237.40/59.20/42.40/16.80 | 7.408378 | 592.390000 | 2489.49/4434.74 |
| TUT-PROXY-THERMAL-CORRECTED | 38.787327+-4.151583 | 26.378590+-3.850848 | 125.00/56.80/40.00/16.80 | 2.940159 | 234.916400 | 2443.93/4473.07 |
| MYOPIC-E2E-UTILITY-GREEDY-STANDALONE-v1 | 40.835838+-3.588434 | 3.240068+-0.717797 | 121.40/8.60/8.60/0.00 | 2.836582 | 172.278486 | 2572.09/4508.80 |

Audit status: **PASS**.

CURRENT_PROP_FOUR_BASELINES_FIVESEED_DUAL_THERMAL_COMPLETE
