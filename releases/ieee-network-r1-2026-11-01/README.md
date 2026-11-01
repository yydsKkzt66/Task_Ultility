# IEEE Network — first-revision experiment archive

This archive, dated 1 November 2026, contains the accepted **6 h / 1,600-task** experiments for the submitted first revision. It is independent of the repository's earlier scaffold and the `comsoc-apb-2026-v1` competition release. Those earlier artifacts are unchanged.

| Component | Included results | Status |
|---|---|---|
| Main experiment | GROUND, OFFLOAD, causal-delay TUT-Adapt, PROP; seeds 4101–4105 | 20 accepted runs |
| Capacity / value sensitivity | Three paired seeds 4101–4103, all four methods | Complete |
| Service-outage resilience | Lost window 2; continuous loss of windows 2–3; same paired seeds | Complete core metrics |
| Sensitivity / resilience combined | 84 perturbation runs plus 12 shared normal references | 96 unique records |
| Ablation | Five-seed full-PROP reference and original design/configuration provenance | **Formal ablation results will be uploaded separately from the other computer** |

**The owner has assigned formal ablation publication to the other computer.** This local release covers the main experiment and sensitivity/resilience analysis. The four formal external ablation arms (w/o RM, w/o Prog, Fixed-H, Req-only) and the existing w/o Mode supplement are not included here or replaced by development results. `ablation/MISSING_EXTERNAL_RETURN.csv` is retained as an evidence checklist for the separate upload. Do not use the original V1 design/runtime as proof of the external run's final version.

## Read and reproduce the published metrics

```console
python reproduce.py verify
python reproduce.py tables --out reproduced_tables
python reproduce.py figures --out reproduced_figures
```

Verification and table recomputation use Python 3.11+ standard library only and run **no simulation**. Figure generation additionally uses NumPy, Matplotlib, Pillow and pypdf; see `requirements_plot.txt`. The original font is Times New Roman; the plotting scripts declare a serif fallback when it is unavailable. Use a new output directory. Relative paths suffice; historical Windows paths in provenance describe origins and are not required to recompute the tables.

- `main/analysis_input_per_seed.csv`: complete-precision main-experiment metrics, including latency and E/B/Q decomposition.
- `main/plot_input_per_seed.csv`, `main/main_exp.pdf`: data and accepted main figure.
- `main/SOURCE_INDEX.csv`, `main/accepted_evidence.zip`: accepted original audit/identity/config/accounting evidence available locally.
- `sensitivity_resilience/tables/ALL_96_PER_SEED_METRICS.csv`: all condition × method × seed rows.
- `sensitivity_resilience/tables/MEAN_SD_32_GROUPS_LONG.csv`: all reported means and sample SDs.
- `sensitivity_resilience/tables/MATCHED_84_SEEDS_CHANGES_RETENTION.csv`: changes relative to the **same method and same seed** normal run.
- `figures/sensitivity/sensitivity.pdf`: capacity, value and outage panels.
- `runtimes/INDEX.json`: exact original source/input manifests and scope of each frozen runtime ZIP.
- `RUN_SELECTION.csv`: the 104 unique main/perturbation jobs, their native identifiers, declared configuration hashes and result-binding evidence levels. Twelve normal sensitivity rows reuse main runs. For 24 unavailable remote PROP/TUT originals, the binding is explicitly the frozen plan plus prior acceptance, not a newly verified raw-run identity.

## Frozen code, configurations, inputs and tests

The five ZIPs under `runtimes/` include the original manifest-listed code, tests, profiles, task inputs and configurations. Every manifest-listed byte is checked against its original SHA-256. Separate archives preserve version identity; their implementations are not silently unified. Source archives may contain historical support/configuration files required by their original manifest. These are **not** additional scientific results or recommended experiment groups.

Example to inspect the formal PROP source and its tests:

```console
python reproduce.py extract-runtime --name prop_formal --out runtime_prop
cd runtime_prop
python -m pip install -r requirements.txt
python -I run.py selfcheck --out self_check
```

`selfcheck` executes bounded tests; it is optional and different from the no-simulation release verification above. Original full-run entry points remain in each extracted archive. Use the version/seed mapping in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), rather than launching every method in an older runtime's `RUN_MATRIX.json`. Long formal simulations have **not** been rerun for this archive. `runtimes/operational_history/` preserves original recovery helpers as historical evidence; some have original local paths and are not portable entry points.

## Evaluation contract and evidence limits

All methods use common fixed resource prices and ground-realized utility. Net utility equals gross utility minus controllable energy, actual downlink and terminal ready-backlog fees. HV means high + urgent tasks. Latencies cover each method's actually closed tasks; a seed's P95 is computed within that seed, then summarized across seeds. Product coverage counts each task once at its highest valid ground level, with fulfilled and partial delivery separated. Unclosed tasks with DDL beyond 21,600 s are right-censored.

All SDs use `ddof=1`. Main results use five seeds; sensitivity uses three paired seeds. Retention is the arithmetic mean of the three **per-seed ratios**, not a ratio of group means. Negative net utilities and ratios above 100% are retained. The small normal GROUND overall net makes its percentages sensitive to cancellation; inspect absolute values and changes first.

GROUND/OFFLOAD use independent ground processing: valid complete receipt + 4.29 s, with nonbinding receive resources. TUT-Adapt is the updated common causal-delay product-level proxy. PROP retains its accepted formal source. These are engineering model/profile and synthesized workload experiments, not real onboard hardware or real detector inference measurements.

GROUND main seed 4102 was accepted under the author's declared arrival-time numerical tolerance; the original strict-equality audit failure is preserved, not relabeled PASS. Its single arrival discrepancy was approximately 3.27e-11 s against the declared 1e-7 s tolerance. External returns provide validated metrics and provenance; unavailable raw physical logs are not represented as locally reaudited. B20 per-window backlog/recovery detail and 24 original PROP/TUT sensitivity directories were not included in the remote lightweight returns.

No manuscript, reviewer correspondence, credentials, datasets, model weights, checkpoints, or large candidate/physical logs are uploaded here. The original full local results remain archived. `FILE_SHA256.csv` covers every release file except itself; `PROVENANCE_FILES.csv` maps copied evidence to its original location and SHA. See [README_CN.md](README_CN.md) for a short Chinese index.
