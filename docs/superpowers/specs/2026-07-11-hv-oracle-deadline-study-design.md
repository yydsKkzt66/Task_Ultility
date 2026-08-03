# Seed-2026 High-Value Oracle and Deadline Study Design

## Purpose

Build the authoritative seed=2026 PROP-FULL versus PROP-NO-STAGE versus B1 study without changing policy behavior, tuning parameters, or overwriting historical outputs. The study uses the corrected requirement-aware evaluator, treats 72 closures as the authoritative PROP-NO-STAGE result, measures high-value and guarantee-target outcomes, constructs architecture-specific offline scheduling benchmarks, and runs fixed deadline multipliers.

## Immutable constraints

- Canonical methods are `proposed_guarantee_aware_full`, the corrected historical no-stage ablation, `B1-FIFO-1.4W`, and `B1-Enhanced-1.4W`. `B1-Enhanced-6.8W` is sensitivity-only and is not part of the deadline study.
- Existing PROP-FULL, PROP-NO-STAGE, and B1 scheduling, routing, reward, task value, payload, quality, contact, resource, and safety behavior remains unchanged.
- Historical output directories are read-only. All new artifacts go under `outputs/_hv_oracle_deadline_study_seed2026/`.
- The authoritative PROP-NO-STAGE closure count is 72. The stale 17-closure evaluator output is rejected.
- Deadline scaling is `arrival_time + k * (original_deadline - arrival_time)` for `k` in `{1.0, 1.25, 1.5, 2.0}`.
- Oracle primary objectives are maximum total on-time closures and maximum HV-TARGET on-time closures. Each has a 300-second solver time limit. Optional utility and lexicographic objectives run only if primary results and replay validation are complete and solver cost remains reasonable.

## Architecture

The implementation adds a focused `src/evaluation/hv_oracle_study/` package and one orchestration script. The package separates immutable input construction, B1 semantic dominance, metric aggregation, legal plan enumeration, mathematical optimization, independent replay, deadline scaling, and report rendering. Existing simulator and policy decision semantics are not edited. A narrow B1 observer/logging hook may record the candidate set passed to the scheduler and the feasible set returned by its existing rank function; regression tests must prove that selected task IDs and lifecycle outcomes are unchanged when logging is enabled.

The orchestration script creates a new output directory and fails if it already exists. It constructs the canonical timeline and authoritative task table once, imports corrected historical task outcomes, runs only the required new sensitivity simulations, solves the two primary architecture-specific oracle objectives, validates selected plans independently, executes all accounting checks, and writes the requested files.

## Authoritative task table

Each method is evaluated against the same 370 task IDs. The task table contains task ID, scene ID, arrival and capture time, original and scaled deadlines, value level, effective value, product requirement, quality threshold, source-frame information capabilities, contact/resource/solar trace hashes, and environment hash.

Alignment checks compare unique task IDs and hashes for arrivals, captures, original deadlines, values, requirements, contacts, resources, solar trace, seed, physical configuration, and horizon. A method row is rejected when its task-level fields disagree with the authoritative table. PROP-NO-STAGE rows come from `outputs/_direct_task_requirement_mapping_audit_seed2026/`, never from the stale unified evaluator output.

## B1 full-frame dominance semantics

A B1 frame closes a requirement only when all four predicates hold:

1. the complete compressed frame is received;
2. ground decode and inference finish successfully;
3. the captured full frame is capable of producing the required alert, verification, context, or evidence result;
4. retained downstream quality meets the task requirement threshold.

The dominance mapping is explicit and task-level. It maps each requirement to the required source-frame information capability and quality threshold. It does not infer that every full frame contains every required fact. The evaluator records `frame_contains_required_information`, `ground_processing_success`, `quality_threshold`, `quality_pass`, and `full_frame_dominates_requirement` independently.

Incomplete compression or downlink never receives closure or utility. A fully received frame with failed or unfinished ground processing does not close. A processed frame below threshold is assigned `quality_gap`. B1 context/evidence tasks do not require staged events; they require the full-frame dominance predicate. FIFO and Enhanced use the same adapter.

The B1 audit output includes required product, frame generation/compression status, bytes delivered, full-frame receipt, ground processing completion, completion time, retained quality, quality threshold, information capability, requirement satisfaction, on-time success, and concrete failure reason.

## Outcome taxonomy and metrics

Every arrived task belongs to exactly one of `on_time_requirement_success`, `late_requirement_satisfied`, `partial_only`, or `no_ground`. Required identities are checked per method. PROP-FULL closure is decomposed into mutually exclusive direct and staged closure IDs.

HV-ALL means value level in `{high, urgent}` after normalizing repository labels (`high_value` and `emergency`). HV-TARGET additionally requires product requirement in `{context, evidence}`.

Feasible-normalized success uses the offline oracle-feasible task set after oracle construction. No runtime policy-dependent feasibility flag is used as a denominator. Energy per HV success is `total_controllable_mission_energy / HV_on_time_success_count` and is labeled a system-level efficiency metric because reliable task-level attribution is unavailable.

Latency comparisons use exact task-ID intersections for ground products in both methods, requirement closures in both, HV closures in both, and HV-TARGET closures in both. The report uses “matched ground-received tasks” for the first set. Bootstrap confidence intervals use a deterministic seed and are omitted only when the matched set is empty.

## Legal completion plans

A candidate plan is an architecture-legal, task-specific completion path with a fixed sequence of actions and resource requirements. Each plan records task ID, architecture, direct/staged/B1 path, compute demand and eligible compute interval, product bytes, transmission segments, communication energy, compute/compression energy, retained quality, ground delay, completion time, deadline, contact requirements, and utility.

PROP plans enumerate legal direct model/product actions already available to PROP-FULL and legal staged metadata plus ROI/context residual paths. B1 plans enumerate complete compression followed by full-frame downlink and ground decode/inference. B1 has no staged plans. Plans failing information, quality, timing, or physical eligibility are retained in the enumeration audit with a rejection reason but are excluded from decision variables.

Plan generation uses full future trace information but never invokes or imitates online policy scores, virtual queues, bandit state, guarantee debt, or online dispatch choices.

## Oracle formulation

For each architecture, binary variable `x[p]` selects a legal candidate plan. At most one plan is selected per task. Sparse capacity constraints cover compute intervals, contact-window downlink capacity, queue occupancy, energy balance with solar charging, and any exactly encoded safety constraints. Ground processing delay and quality are embedded in plan eligibility and completion time.

The integer model is solved with SciPy HiGHS MILP. The same encoded candidate-plan model is solved as an LP relaxation by changing integrality and variable bounds. Terminology is enforced:

- an integer solution with a solver optimality certificate is a proven optimum under the encoded model;
- a time-limited integer incumbent is an offline feasible lower bound only;
- the LP relaxation objective is an optimistic upper bound only for the encoded candidate-plan model;
- unresolved optimum is reported as `[integer incumbent, valid solver/LP upper bound]`.

No feasible incumbent is reported as an upper bound. Thermal relaxation, if required, is limited to an explicitly labeled optimistic upper-bound model. Results never claim a physical or theoretical bound beyond the candidate-plan encoding.

Primary objectives are solved in this order for PROP and B1: total on-time closures, then HV-TARGET on-time closures. Optional common utility and lexicographic HV-TARGET/total/utility objectives run only after all primary solutions pass replay.

## Independent replay validation

The replay checker consumes only the authoritative trace, selected plan rows, and physical constraints; it does not consume solver constraint matrices. It reconstructs per-task sequencing, completion times, contact-byte use, compute occupancy, energy trajectory, queue occupancy, and quality/information predicates. It verifies one plan per task, deadlines, capacities, state transitions, and closure counts.

Any failed replay invalidates the corresponding oracle result. The pipeline stops before final recommendations if a primary oracle replay fails.

## Bottleneck bounds

Separate optimistic bounds are reported for deadline/contact eligibility only, aggregate downlink capacity only, aggregate compute capacity only, aggregate energy only, and all encoded constraints jointly. Each bound identifies which constraints were removed and is labeled as applying only to the encoded plan universe.

## Deadline sensitivity and B1 activation

For every multiplier, tasks, values, requirements, contact/resource/solar traces, physical model, and method parameters remain fixed. Only deadlines are replaced with the relative-arrival formula. Original `k=1.0` inputs remain unchanged.

B1 scheduling instrumentation records, at each dispatch decision, queued candidates before eligibility filtering, rejection counts by expired/full-fit/deadline/safety reason, feasible candidates after filtering, selected task, FIFO order, Enhanced order, and whether the methods would differ. Aggregate output distinguishes:

- no real queued competition before filtering;
- competition removed by deadline/full-fit feasibility filtering;
- early dispatch preventing candidate-set formation;
- genuine multi-candidate feasible competition.

If no tested multiplier activates B1-Enhanced, the reported activation multiplier is `none`; no easier B1-only scenario or additional multiplier is generated.

## Error handling and non-overwrite behavior

The main command fails before writing if the target output directory exists. It writes to a temporary sibling directory and atomically renames it only after required checks pass, leaving historical outputs untouched. On a consistency or replay failure, diagnostic files remain in the temporary directory and the final recommendation is not emitted.

Large unknown failure categories are investigated from lifecycle and queue traces. `unknown` is allowed only for rows lacking enough evidence, and its count and example task IDs are called out explicitly.

## Required outputs

The pipeline writes all requested artifacts:

- `authoritative_run_manifest.yaml`
- `environment_alignment.csv`
- `accounting_validation_report.md`
- `b1_ground_processing_semantics_audit.md`
- `b1_task_level_ground_outcomes.csv`
- `high_value_method_comparison.csv`
- `guarantee_target_comparison.csv`
- `high_value_failure_reasons.csv`
- `high_value_matched_latency.csv`
- `closure_set_decomposition.csv`
- `prop_oracle_model.md`
- `b1_oracle_model.md`
- `oracle_solver_results.csv`
- `oracle_constraint_replay_report.md`
- `resource_upper_bounds.csv`
- `deadline_sensitivity_seed2026.csv`
- `b1_scheduler_activation_by_deadline.csv`
- `scenario_classification.md`
- `final_recommendation.md`

Additional internal CSV files may be written for the authoritative task table, candidate plans, selected oracle plans, and deadline-run provenance.

## Tests and acceptance checks

Unit tests cover deadline scaling, B1 full-frame dominance, incomplete-frame rejection, quality thresholds, PROP direct/staged decomposition, candidate-plan legality, solver terminology, LP versus incumbent labeling, independent replay, candidate counts before/after filtering, and system-level energy efficiency.

Integration tests use small synthetic traces to prove that the oracle chooses legal plans rather than online-policy decisions and that replay catches contact, energy, deadline, and duplicate-plan violations.

The final pipeline checks:

1. identical task IDs and environment hashes;
2. no duplicate outcomes;
3. category totals equal arrivals;
4. closure subsets are valid;
5. on-time closure implies requirement satisfaction;
6. B1 processed full frames use the dominance mapping;
7. incomplete B1 frames never close;
8. PROP-FULL direct plus staged closures equal total closure;
9. matched latency uses the declared intersections;
10. utility and energy totals recompute from source rows;
11. PROP-NO-STAGE equals 72 and no stale 17-row source is used;
12. no NaN or infinity;
13. every accepted primary oracle schedule passes independent replay;
14. all sensitivity methods use identical relatively scaled deadlines.

Completion requires all mandatory checks, both primary objectives for both architectures, valid terminology, and all required output files. Optional objectives may be absent with an explicit solver-cost explanation.
