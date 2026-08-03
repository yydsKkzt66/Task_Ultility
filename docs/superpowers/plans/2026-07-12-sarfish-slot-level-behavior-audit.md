# SARFish Slot-Level Behavior Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a rollback-safe, passive seed-2026 slot-level audit for PROP-FULL and B1-Enhanced-1.4W at capacity multipliers 1.00, 0.50, and 0.35 without changing either method.

**Architecture:** Add one independent audit runner which reuses the frozen canonical configuration and wraps existing simulator boundaries only to snapshot pre/post state. Keep tracing default-disabled and derive taxonomy, lineage, comparisons, validations, reports, and the compact ZIP offline from captured events.

**Tech Stack:** Python 3, standard library CSV/JSON/ZIP/hashlib, existing simulator/evaluator modules, pytest.

## Global Constraints

- Diagnosis only; do not change policies, scheduler ranking, staged delivery, guarantee/debt, utility, payload/task/deadline/trace/configuration, B1 behavior, evaluator semantics, or compression.
- Run only PROP-FULL and B1-Enhanced-1.4W, multipliers 1.00/0.50/0.35, seed 2026.
- Write only to `outputs/_sarfish_slot_level_behavior_audit_seed2026/` plus the new audit runner, tests, and this plan.
- Instrumentation is default-disabled and must not consume randomness or mutate decision inputs.

---

### Task 1: Baseline and isolation

**Files:** Create `outputs/_sarfish_slot_level_behavior_audit_seed2026/rollback/*` at run time.

- [ ] Record HEAD, branch, status, diff/stat and SHA-256 hashes before audit execution.
- [ ] Create/use branch `audit/sarfish-slot-level-seed2026` and verify canonical outputs remain untouched.

### Task 2: Pure audit contracts

**Files:** Create `tests/test_slot_level_behavior_audit.py`; create `scripts/run_sarfish_slot_level_behavior_audit_seed2026.py`.

- [ ] Write failing tests for the strict idle taxonomy, avoidable-idle predicate, finite-value validation, and required output schema.
- [ ] Run `python -m pytest tests/test_slot_level_behavior_audit.py -q` and confirm failure because the module does not exist.
- [ ] Implement the minimal pure helpers and rerun until green.

### Task 3: Passive trace capture

**Files:** Modify only `scripts/run_sarfish_slot_level_behavior_audit_seed2026.py`.

- [ ] Add `--enable-trace` (default false); when false, use the untouched simulator paths.
- [ ] Wrap PROP queue and B1 downlink boundaries to copy queue/candidate/resource state immediately before and after the original method call.
- [ ] Run paired trace-off/trace-on 1.00 executions and assert task outcomes, event order, per-slot bytes, summaries, and hashes are identical within `1e-9`.

### Task 4: Approved runs and raw artifacts

**Files:** Generate the requested CSVs under the isolated output directory.

- [ ] Execute exactly six approved method/capacity combinations using the same timeline, manifest, evaluator, and initial state.
- [ ] Write slot trace, lineage, partial causes, HV trace, compute/contact timing, product-size, matched-slot, and method summary CSVs.

### Task 5: Independent validation and reports

**Files:** Generate `validation_report.md`, `scheduler_work_conservation_audit.md`, `final_slot_level_behavior_audit.md`, and bundle README/ZIP.

- [ ] Reconcile 268 tasks, hashes, bytes, queue transitions, staged closure paths, finite values, idle reasons, and root-cause coverage.
- [ ] If equivalence fails, write `BLOCKED_INSTRUMENTATION_CHANGED_BEHAVIOR` and skip interpretation.
- [ ] Build `analysis_bundle/` and ZIP, then run the audit test and full targeted validation command fresh.
