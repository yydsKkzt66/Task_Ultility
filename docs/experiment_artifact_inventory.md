# Experiment Artifact Inventory

This document inventories the current experiment artifacts without changing algorithms, configs, or numeric results.

## Formal / Recent Stages

| stage | configs | scripts | output directory | key summary/report files | result status |
| --- | --- | --- | --- | --- | --- |
| SARFish GRD validation audit | `configs/dataset_sarfish_grd_validation.yaml` | `scripts/check_sarfishsample_labels.py`, `scripts/inspect_sarfishsample.py` | `outputs/sarfish_grd_validation_audit/` | `adapter_validation_report.md`, `full_grd_validation_file_report.md`, `label_image_match_report.md` | dataset-readiness evidence; formal appendix candidate |
| SARFish 20s nominal dry run | `configs/simulator_sarfish_20s_dryrun.yaml`, `configs/timeline_sarfish_20s_dryrun.yaml`, `configs/task_stream_sarfish_20s_dryrun.yaml`, `configs/dataset_sarfish_grd_validation.yaml` | `scripts/run_sarfish_20s_dry_run.py`, `scripts/analyze_sarfish_20s_dryrun_decomposition.py` | `outputs/sarfish_20s_dry_run/` | `summary.csv`, `dry_run_sanity_report.md`, `action_payload_model_decomposition.md`, `router_cost_decomposition.md`, `proposed_vs_utility_rule_root_cause_report.md` | formal baseline candidate, but superseded for stress conclusions |
| Energy-shadow stress dry run | `configs/simulator_sarfish_energy_shadow_stress_20s.yaml`, `configs/timeline_sarfish_energy_shadow_stress_20s.yaml`, `configs/task_stream_sarfish_energy_shadow_stress.yaml`, `configs/resource_sarfish_energy_shadow_stress.yaml`, `configs/contact_sarfish_physical_20s.yaml` | `scripts/run_sarfish_energy_shadow_stress_dryrun.py`, `scripts/analyze_sarfish_energy_shadow_action_scores.py` | `outputs/sarfish_energy_shadow_stress/` | `summary.csv`, `energy_shadow_sanity_report.md`, `action_score_root_cause_report.md`, `resource_model_audit.md`, `proposed_vs_utility_rule_stress_report.md`, `action_score_counterfactual.csv` | formal stress baseline candidate |
| YOLO-inspired model profile ablation | `configs/model_profiles_yolo_inspired_calibrated.yaml`, `configs/simulator_sarfish_energy_shadow_yolo_profile_ablation_20s.yaml` | `scripts/run_sarfish_yolo_profile_ablation.py` | `outputs/sarfish_yolo_profile_ablation/` | `summary.csv`, `profile_ablation_report.md`, `calibrated_profile_sanity_report.md`, `calibrated_action_score_diagnosis.md`, `root_cause_after_profile_calibration.md` | formal model-profile calibration evidence |
| Policy repair ablation | `configs/policy_repair_ablation.yaml`, `configs/simulator_sarfish_energy_shadow_policy_repair_ablation_20s.yaml` | `scripts/run_sarfish_policy_repair_ablation.py` | `outputs/sarfish_policy_repair/` | `summary.csv`, `policy_repair_ablation_report.md`, `cost_aware_ucb_report.md`, `delivery_aware_info_gain_report.md`, `z_guided_coupling_report.md`, `root_cause_after_policy_repair.md` | formal policy-repair evidence, stage-local comparisons only |
| Payload profile calibration ablation | `configs/payload_profiles_roi_compression_calibrated.yaml`, `configs/payload_profile_ablation.yaml`, `configs/simulator_sarfish_energy_shadow_payload_profile_ablation_20s.yaml` | `scripts/run_sarfish_payload_profile_ablation.py` | `outputs/sarfish_payload_profile_calibration/` | `summary.csv`, `payload_profile_ablation_report.md`, `payload_size_diagnostics.csv`, `delivery_proxy_calibration_report.md`, `roi_context_overuse_report.md`, `root_cause_after_payload_calibration.md` | most recent formal candidate for payload/calibration claims |

## Supporting Source and Tests

| area | important files | role |
| --- | --- | --- |
| simulator wiring | `src/simulator/environment.py`, `src/simulator/step_runner.py`, `src/simulator/predictors.py`, `src/simulator/feedback.py` | experiment execution, prediction diagnostics, trajectory records |
| payload profile | `src/payload/payload_policy.py`, `src/payload/payload_sizer.py`, `src/payload/information_sufficiency.py` | payload config loading, size estimation, payload information |
| SARFish dataset | `src/data/sarfishsample_dataset.py`, `data/indexes/sarfishsample_scene_index.csv` | GRD task stream and offline label-derived metadata |
| controllers | `src/controllers/mode_controller.py`, `src/controllers/lyapunov_pricing.py` | mode and resource pricing; not modified in this cleanup pass |
| tests | `tests/test_sarfishsample_dataset.py`, `tests/test_resource_trace.py`, `tests/test_predictor.py`, `tests/test_guided_utility_selection.py`, `tests/test_controllers.py` | current focused regression tests for the recent stages |

## Formal Result Candidates

- `outputs/sarfish_payload_profile_calibration/`: best current payload-calibrated result set and diagnostics.
- `outputs/sarfish_policy_repair/`: policy-repair evidence before payload calibration.
- `outputs/sarfish_yolo_profile_ablation/`: model-profile calibration evidence.
- `outputs/sarfish_energy_shadow_stress/`: energy/contact/resource stress baseline.
- `outputs/sarfish_20s_dry_run/`: nominal 20s baseline and cost/action decomposition.
- `outputs/sarfish_grd_validation_audit/`: dataset integration evidence.

## Historical Debug / Archive Candidates

- `outputs/debug_*`: diagnosis snapshots; useful for traceability, not formal result candidates.
- `outputs/phase*`: earlier demo/check/rerun outputs; superseded by named SARFish outputs.
- `outputs/demo/`, `outputs/figures/`, root-level `outputs/summary*.csv`, root-level `outputs/trajectories/`: legacy/demo outputs; keep only if needed for reproducing early README figures.
- `outputs/sarfishsample_short_sim/`, `outputs/sarfishsample_audit/`: useful earlier integration checks, but not final case-study result directories.
- `outputs/sarfish_profile_calibration/`: profile audit precursor; superseded by `outputs/sarfish_yolo_profile_ablation/`.

## Cache / Low-Value Intermediates

- `__pycache__/`, `.pytest_cache/`, nested `__pycache__/` folders: cache candidates.
- `outputs/**/figures/*.png`: derived figures; keep for reports but regenerateable from trajectories in most runs.
- `summary_raw.csv` and `summary_display.csv`: useful duplicates around `summary.csv`; not canonical for indexing.
- Large `outputs/**/trajectories/*.csv`: not low value, but large; archive with the owning output directory if that directory is no longer active.
