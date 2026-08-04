# Final submission manifest v1

This manifest identifies the publication-facing implementation and experiment
assets in the repository. It is intentionally narrower than the complete local
experiment history.

## Canonical method and environment

- Proposed method: `PROP-V2-SAFE-METADATA-CURRENT-v1`.
- Main scenario: ordinary value-blind V25 arrivals, not late-HV-skew.
- Seeds: 2026, 2027, 2028, 2029, 2030.
- Main downlink-capacity multiplier: 0.50.
- Duration and slot: 86400 s and 20 s.
- Value-level counts: low/medium/high/urgent = 54/139/54/21.
- Power: corrected 40 W solar-peak environment and shared 38 W radio.
- Thermal environment: shared literature-anchored provisional two-node model.
- Public accounting: frozen q-once evaluator and frozen public resource prices.

The two-node parameters are literature-anchored provisional parameters, not a
claim of device-matched in-orbit thermal calibration.

## Canonical comparison methods

1. `PROP-V2-SAFE-METADATA-CURRENT-v1`
2. `B1-COMMON-GROUND-THERMAL-CORRECTED`
3. `OFFLOAD-EFT-THERMAL-ALIGNED-2NODE-DIAGNOSTIC`
4. `TUT-PROXY-THERMAL-CORRECTED`
5. `MYOPIC-E2E-UTILITY-GREEDY-STANDALONE-v1`

## Canonical entry points

- Main five-method, five-seed comparison:
  `scripts/run_current_prop_four_baselines_fiveseed_dual_thermal_v1.py`
- Single-seed five-method identity/reference run:
  `scripts/run_current_prop_four_independent_baselines_single_seed_v1.py`
- Five-method ablation:
  `scripts/run_current_prop_five_ablation_multiseed_dual_thermal_v1.py`
- Value/capacity sensitivity:
  `scripts/run_current_prop_value_capacity_sensitivity_multiseed_dual_thermal_v1.py`
- Two-node thermal calibration and reconciliation:
  `scripts/run_dual_node_thermal_calibration_and_four_method_rerun_v1.py`
- Main publication figure generator: `scripts/main metrics.py`
- Sensitivity/ablation figure generator: `scripts/sensitive.py`

## Canonical ablations

1. `FULL-V1-V2-DUAL-THERMAL`
2. `NO-LYAPUNOV-KEEP-MODE-DUAL-THERMAL`
3. `NO-MODE-KEEP-LYAPUNOV-DUAL-THERMAL`
4. `FIXED-BALANCED-V1-V2-DUAL-THERMAL`
5. `DIRECT-ONLY-NO-PROGRESSIVE-DUAL-THERMAL`

## Canonical sensitivity grid

- V10 at capacity 0.50.
- V50 at capacity 0.50.
- V25 at capacities 0.25, 0.35, 0.50, and 0.70.

Only the proposed method is used for the value/capacity sensitivity grid. The
V25/capacity-0.50 point is the main experiment reference.

## Submission policy

Commit:

- all source modules, configuration, tests, documentation, and Python research
  scripts;
- frozen scenario manifests and compact calibration parameters;
- the two reviewed figure-generator scripts and reviewed final PDF/PNG/CSV
  figure assets;
- compact final assessments, summary tables, completion markers, and identity
  audits required to substantiate the reported results.

Do not commit:

- raw `SARFishSample` imagery;
- full per-slot/per-task run directories and queues;
- aborted, smoke, preflight, or debug outputs;
- local rollback packages and agent scratch workspaces;
- intermediate figure iterations.

## Portable clean-clone reproduction

The former `D:/wt/...` runtime dependencies are now vendored under
`vendor/frozen_runtime_v1`.  `scripts/portable_runtime_paths_v1.py` prefers the
Git-tracked bundle and retains environment-variable and legacy-path fallbacks
only for audited overrides.  Every vendored file is protected by
`artifacts/reproducibility/frozen_runtime_v1_sha256.json`.

Use `python scripts/reproduce_on_new_computer_v1.py` to verify the bundle and
regenerate both final figure sets.  Use `--scope single` for a fresh seed-2026
five-method run or `--scope main` for all 25 main runs from fresh state.  Full
setup instructions are in `docs/reproduce_on_new_computer_v1.md`.

## Pre-commit validation

- Python syntax compilation: PASS (`python -m compileall -q src scripts tests`).
- Unit test suite: PASS (514 tests; 13 third-party Matplotlib/PyParsing
  deprecation warnings).
- Publication figure generators: PASS on Windows with project-relative output
  directories; both PDF and high-resolution PNG assets were generated.
- Secret-pattern scan of staged source: PASS; no credentials or private keys
  detected.
- GitHub large-file gate: PASS; the largest staged file is below 5 MB.
- Main comparison, ablation, and sensitivity completion markers: PASS.
- Portable B1, OFFLOAD, and TUT source-identity gates: PASS.
- Portable PROP configuration/calibration load: PASS.
- Clean-clone figure entry: PASS; both reviewed PDF/PNG sets regenerated.
