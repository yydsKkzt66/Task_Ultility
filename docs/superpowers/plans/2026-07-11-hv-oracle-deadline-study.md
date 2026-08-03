# Seed-2026 High-Value Oracle and Deadline Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the authoritative seed=2026 high-value, offline-oracle, and deadline-sensitivity study while preserving all existing policy behavior and historical outputs.

**Architecture:** Add a focused evaluation package that builds canonical task inputs, applies corrected B1 full-frame dominance semantics, aggregates high-value metrics, enumerates architecture-legal completion plans, solves sparse SciPy MILP/LP models, independently replays selected schedules, and renders the required artifacts. Add only passive candidate-set logging to the existing B1 scheduler path and prove selection equivalence with regression tests.

**Tech Stack:** Python 3, dataclasses, csv/json/yaml, NumPy, SciPy `optimize.milp`/`linprog`, pytest.

## Global Constraints

- Keep PROP-FULL, PROP-NO-STAGE, B1 FIFO, and B1 Enhanced policy behavior unchanged.
- Do not tune rewards, task values, payload sizes, quality, traces, deadlines, or policy parameters.
- Do not overwrite historical outputs; fail if `outputs/_hv_oracle_deadline_study_seed2026/` already exists.
- Use corrected evaluator rows and enforce authoritative PROP-NO-STAGE closure count 72.
- Scale deadlines only as `arrival + multiplier * (original_deadline - arrival)`.
- Distinguish integer incumbent, proven integer optimum, solver upper bound, and LP relaxation upper bound.
- A feasible incumbent is never labeled an upper bound.
- Validate every accepted primary oracle schedule through an independent replay checker.
- Solve total closures and HV-TARGET closures before optional objectives.

---

### Task 1: Deadline and B1 dominance primitives

**Files:**
- Create: `src/evaluation/hv_oracle_study/__init__.py`
- Create: `src/evaluation/hv_oracle_study/semantics.py`
- Test: `tests/test_hv_oracle_semantics.py`

**Interfaces:**
- Produces: `scale_deadline(arrival_s: float, original_deadline_s: float, multiplier: float) -> float`
- Produces: `FullFrameRequirement` and `evaluate_b1_full_frame(row, task) -> B1DominanceResult`

- [ ] **Step 1: Write failing semantic tests**

```python
def test_deadline_scaling_uses_relative_slack():
    assert scale_deadline(100.0, 160.0, 1.5) == 190.0

def test_processed_frame_requires_information_and_quality():
    result = evaluate_b1_full_frame(
        {"ground_result_available": 1, "downlinked_mb": 4.0, "compressed_size_mb": 4.0,
         "ground_task_quality": 0.79},
        {"product_requirement": "evidence", "quality_threshold": 0.8,
         "full_frame_information_capabilities": "alert;verification;evidence"},
    )
    assert not result.requirement_satisfied
    assert result.failure_reason == "quality_gap"

def test_incomplete_frame_never_closes():
    result = evaluate_b1_full_frame(
        {"ground_result_available": 0, "downlinked_mb": 3.9, "compressed_size_mb": 4.0},
        {"product_requirement": "alert", "quality_threshold": 0.0,
         "full_frame_information_capabilities": "alert"},
    )
    assert not result.requirement_satisfied
    assert result.failure_reason == "transmission_incomplete"
```

- [ ] **Step 2: Run tests and confirm missing-module failure**

Run: `python -m pytest tests/test_hv_oracle_semantics.py -q`

Expected: FAIL because `src.evaluation.hv_oracle_study.semantics` does not exist.

- [ ] **Step 3: Implement immutable semantic results and explicit dominance mapping**

```python
@dataclass(frozen=True)
class B1DominanceResult:
    full_frame_received: bool
    ground_processing_success: bool
    frame_contains_required_information: bool
    retained_quality: float
    quality_threshold: float
    quality_pass: bool
    requirement_satisfied: bool
    failure_reason: str

def scale_deadline(arrival_s, original_deadline_s, multiplier):
    if multiplier <= 0:
        raise ValueError("deadline multiplier must be positive")
    return float(arrival_s) + float(multiplier) * (float(original_deadline_s) - float(arrival_s))
```

Implement deterministic failure precedence: compression incomplete, transmission incomplete, ground processing late/unfinished, missing information, quality gap, success.

- [ ] **Step 4: Run semantic tests**

Run: `python -m pytest tests/test_hv_oracle_semantics.py -q`

Expected: PASS.

### Task 2: Passive B1 scheduler candidate instrumentation

**Files:**
- Modify: `src/baselines/b1_enhanced.py`
- Modify: `tests/test_b1_enhanced_baseline.py`

**Interfaces:**
- Produces: `B1EnhancedSimulator.scheduler_decision_trace: list[dict[str, Any]]`
- Preserves: `ContactAwareCompressedGroundScheduler.select(...)` selected task and decision rows.

- [ ] **Step 1: Write a failing equivalence and trace test**

```python
def test_candidate_trace_records_before_and_after_without_changing_selection():
    selected_plain, decisions_plain = scheduler.select(states, now, link, resource, sim_cfg, capacity)
    selected_logged, decisions_logged = scheduler.select(states, now, link, resource, sim_cfg, capacity)
    row = candidate_trace_row(states, decisions_logged, selected_logged, now)
    assert selected_plain.task.task_id == selected_logged.task.task_id
    assert [d.task_id for d in decisions_plain] == [d.task_id for d in decisions_logged]
    assert row["queued_candidate_count_before_filter"] == len(states)
    assert row["feasible_candidate_count_after_filter"] == sum(d.reason == "candidate" for d in decisions_logged)
```

- [ ] **Step 2: Run the focused test and confirm missing helper failure**

Run: `python -m pytest tests/test_b1_enhanced_baseline.py -q`

Expected: FAIL because `candidate_trace_row` is absent.

- [ ] **Step 3: Add observer-only trace construction**

Add a pure `candidate_trace_row` function that derives pre-filter count, post-filter count, rejection counts, order, and selected ID from existing inputs/outputs. Append it after each existing `select` call. Do not change rank, filter, sort, or selection expressions.

- [ ] **Step 4: Verify B1 regression suite**

Run: `python -m pytest tests/test_b1_enhanced_baseline.py tests/test_unified_ground_received_accounting.py -q`

Expected: PASS with unchanged selected task IDs in existing fixtures.

### Task 3: Authoritative inputs, corrected adapters, and metrics

**Files:**
- Create: `src/evaluation/hv_oracle_study/inputs.py`
- Create: `src/evaluation/hv_oracle_study/metrics.py`
- Test: `tests/test_hv_oracle_metrics.py`

**Interfaces:**
- Produces: `build_authoritative_tasks(timeline, config_hashes) -> dict[str, dict]`
- Produces: `adapt_b1_outcomes(tasks, trace, method) -> list[dict]`
- Produces: `validate_accounting(method_rows) -> list[CheckResult]`
- Produces: `summarize_group(rows, predicate, system_energy_wh) -> dict`

- [ ] **Step 1: Write failing accounting tests**

```python
def test_accounting_partition_and_system_energy_efficiency():
    summary = summarize_group(rows, lambda row: True, system_energy_wh=12.0)
    assert summary["arrived"] == 4
    assert summary["on_time_requirement_success"] + summary["late_requirement_satisfied"] + summary["partial_only"] + summary["no_ground"] == 4
    assert summary["energy_per_HV_success_system_level_wh"] == 6.0

def test_no_stage_source_must_equal_72():
    with pytest.raises(ValueError, match="authoritative PROP-NO-STAGE closure count is 72"):
        validate_no_stage_rows(stale_17_rows)
```

- [ ] **Step 2: Run tests and confirm missing functions**

Run: `python -m pytest tests/test_hv_oracle_metrics.py -q`

Expected: FAIL because the metrics module is absent.

- [ ] **Step 3: Implement canonical inputs and aggregation**

Implement normalized HV labels, HV-TARGET predicates, exact partitions, percentile/paired-latency helpers, deterministic bootstrap, concrete failure taxonomy, direct/staged decomposition, and oracle-feasible normalization that accepts only a post-solve feasible task-ID set.

- [ ] **Step 4: Run metric tests**

Run: `python -m pytest tests/test_hv_oracle_metrics.py -q`

Expected: PASS.

### Task 4: Legal plan enumeration, MILP/LP, and replay

**Files:**
- Create: `src/evaluation/hv_oracle_study/plans.py`
- Create: `src/evaluation/hv_oracle_study/oracle.py`
- Create: `src/evaluation/hv_oracle_study/replay.py`
- Test: `tests/test_hv_oracle_solver.py`

**Interfaces:**
- Produces: `CompletionPlan` dataclass with fixed task, timing, resource, quality, utility, and path fields.
- Produces: `enumerate_prop_plans(...)` and `enumerate_b1_plans(...)`.
- Produces: `solve_oracle(plans, capacities, objective, time_limit_s) -> OracleResult`.
- Produces: `solve_lp_upper_bound(...) -> OracleResult`.
- Produces: `replay_selected_plans(selected, trace) -> ReplayReport`.

- [ ] **Step 1: Write failing synthetic solver and replay tests**

```python
def test_oracle_selects_legal_nonconflicting_plans():
    result = solve_oracle(two_tasks_one_contact_plans, capacities, "total_closures", 5.0)
    assert result.integer_feasible
    assert result.incumbent_objective == 1.0
    assert len(result.selected_plan_ids) == 1

def test_lp_bound_is_not_incumbent_label():
    lp = solve_lp_upper_bound(fractional_fixture, capacities, "total_closures")
    assert lp.bound_kind == "lp_relaxation_upper_bound_for_encoded_candidate_plan_model"

def test_replay_rejects_contact_overuse():
    report = replay_selected_plans(over_capacity_selection, trace)
    assert not report.valid
    assert "contact_capacity" in report.failed_constraints
```

- [ ] **Step 2: Run tests and confirm missing solver failure**

Run: `python -m pytest tests/test_hv_oracle_solver.py -q`

Expected: FAIL because plan/oracle/replay modules are absent.

- [ ] **Step 3: Implement sparse candidate-plan optimization**

Use one binary variable per eligible plan, one-plan-per-task constraints, sparse resource rows, SciPy `milp`, and a separate continuous relaxation. Preserve raw HiGHS status, message, runtime, mip gap, dual bound, incumbent objective, selected IDs, and whether optimality was proven. Do not call the incumbent an upper bound.

- [ ] **Step 4: Implement matrix-independent replay**

Reconstruct selected plan sequences and capacities from plan fields and trace rows. Reject duplicate task selections, deadline misses, quality/information failures, contact-byte overuse, compute overlap, energy underflow, and queue overflow.

- [ ] **Step 5: Run solver tests**

Run: `python -m pytest tests/test_hv_oracle_solver.py -q`

Expected: PASS.

### Task 5: Study orchestration and required reports

**Files:**
- Create: `src/evaluation/hv_oracle_study/study.py`
- Create: `scripts/run_hv_oracle_deadline_study_seed2026.py`
- Test: `tests/test_hv_oracle_study_integration.py`

**Interfaces:**
- Produces: `run_study(output_dir: Path, solver_time_limit_s: float = 300.0) -> None`.
- Consumes: canonical seed=2026 configs and read-only authoritative historical outputs.

- [ ] **Step 1: Write a failing miniature end-to-end artifact test**

```python
def test_miniature_study_writes_required_artifacts(tmp_path):
    run_study(tmp_path / "study", fixture_bundle=mini_bundle(), solver_time_limit_s=5.0)
    required = {"authoritative_run_manifest.yaml", "accounting_validation_report.md",
                "oracle_solver_results.csv", "oracle_constraint_replay_report.md",
                "deadline_sensitivity_seed2026.csv", "final_recommendation.md"}
    assert required <= {path.name for path in (tmp_path / "study").iterdir()}
```

- [ ] **Step 2: Run and confirm orchestration failure**

Run: `python -m pytest tests/test_hv_oracle_study_integration.py -q`

Expected: FAIL because `run_study` is absent.

- [ ] **Step 3: Implement non-overwriting orchestration**

Create a temporary sibling output directory, construct/validate authoritative rows, run the four fixed deadline settings, solve and replay primary objectives before optional objectives, render all 19 requested files, and rename the temporary directory only when mandatory checks pass.

- [ ] **Step 4: Implement report language guards**

Render exact status labels for `proven_optimal_integer_solution`, `feasible_integer_incumbent_lower_bound`, and `lp_relaxation_upper_bound_for_encoded_candidate_plan_model`. Render `none` when B1-Enhanced never activates. State that energy per HV success is system-level and LP scope is limited to the encoded candidate plans.

- [ ] **Step 5: Run integration tests**

Run: `python -m pytest tests/test_hv_oracle_study_integration.py -q`

Expected: PASS.

### Task 6: Full seed=2026 execution and acceptance audit

**Files:**
- Create: `outputs/_hv_oracle_deadline_study_seed2026/` through the runner only.
- Modify: `docs/superpowers/plans/2026-07-11-hv-oracle-deadline-study.md` checkboxes only.

**Interfaces:**
- Produces: the complete authoritative result directory.

- [ ] **Step 1: Run focused study tests**

Run: `python -m pytest tests/test_hv_oracle_semantics.py tests/test_hv_oracle_metrics.py tests/test_hv_oracle_solver.py tests/test_hv_oracle_study_integration.py tests/test_b1_enhanced_baseline.py tests/test_unified_ground_received_accounting.py -q`

Expected: PASS with zero failures.

- [ ] **Step 2: Execute the full study once**

Run: `python scripts/run_hv_oracle_deadline_study_seed2026.py --solver-time-limit-s 300`

Expected: exit 0 and creation of `outputs/_hv_oracle_deadline_study_seed2026/` without modifying any historical output directory.

- [ ] **Step 3: Verify required files and invariants**

Run: `python scripts/run_hv_oracle_deadline_study_seed2026.py --verify-only outputs/_hv_oracle_deadline_study_seed2026`

Expected: exit 0; 370 aligned task IDs; PROP-NO-STAGE closure 72; all 14 accounting checks pass; all four primary oracle results replay-valid; no NaN/Inf; all required files present.

- [ ] **Step 4: Run full regression tests**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 5: Inspect repository scope**

Run: `git status --short` and `git diff --check`

Expected: only intended new study files, passive B1 logging changes, tests, and generated target output are attributable to this work; unrelated pre-existing changes remain untouched.
