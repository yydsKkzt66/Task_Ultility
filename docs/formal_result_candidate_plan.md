# Formal Result Candidate Plan

This is a recommendation only. No files are moved by this plan.

## Recommended Future Formal Layout

| proposed future directory | source to copy/rerun from | purpose |
| --- | --- | --- |
| `outputs/formal_candidate/energy_shadow_payload_calibrated/` | start from `outputs/sarfish_payload_profile_calibration/` | Current best payload-calibrated stress setting; strongest candidate for the main case-study table after consistency audit. |
| `outputs/formal_candidate/energy_shadow_policy_repair/` | start from `outputs/sarfish_policy_repair/` | Shows the policy-repair step before payload calibration. |
| `outputs/formal_candidate/yolo_profile_calibrated/` | start from `outputs/sarfish_yolo_profile_ablation/` | Supports the model-profile calibration narrative. |
| `outputs/formal_candidate/energy_shadow_baseline/` | start from `outputs/sarfish_energy_shadow_stress/` | Stress baseline before model/profile/policy/payload repair. |
| `outputs/formal_candidate/nominal_payload_calibrated/` | future rerun using nominal timeline plus calibrated payload profile | Needed if the paper wants a non-stress nominal payload-calibrated comparison. |
| `outputs/formal_candidate/multi_seed_1day/` | future 1-day multi-seed rerun | Needed for final claims about robustness and variance. |

## Keep As Formal Evidence Candidates

- `outputs/sarfish_payload_profile_calibration/`
- `outputs/sarfish_policy_repair/`
- `outputs/sarfish_yolo_profile_ablation/`
- `outputs/sarfish_energy_shadow_stress/`
- `outputs/sarfish_20s_dry_run/`
- `outputs/sarfish_grd_validation_audit/`

## Debug / Not Directly Paper-Ready

- `outputs/debug_*`: useful for engineering traceability, not final plots/tables.
- `outputs/phase*`: earlier demo/check/rerun directories; superseded by named SARFish stages.
- `outputs/demo/`, root-level `outputs/summary*.csv`, root-level `outputs/trajectories/`: legacy demonstration artifacts.
- `outputs/sarfishsample_short_sim/` and `outputs/sarfishsample_audit/`: integration/debug evidence, not final case-study metrics.
- `outputs/sarfish_profile_calibration/`: profile-audit precursor superseded by `outputs/sarfish_yolo_profile_ablation/`.

## Before Formal 1-Day Multi-Seed

1. Use `outputs/_index/leaderboard.csv` for stage-local winners, not a global winner claim.
2. Review `outputs/_index/metric_consistency_check.md` and resolve naming/config comparability issues.
3. Decide the canonical proposed variant for the formal run, likely payload-calibrated plus the least fragile repair combination.
4. Freeze configs under a future `configs/formal/` or equivalent manifest before rerunning.
