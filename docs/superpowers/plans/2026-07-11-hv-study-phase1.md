# Seed-2026 High-Value Study Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete authoritative reconciliation, B1 ground semantics, HV/HV-TARGET comparisons, matched latency, and consistency validation only.

**Architecture:** Read the existing canonical seed=2026 timeline and only the authoritative PROP-FULL, corrected PROP-NO-STAGE, and B1 1.4 W traces. Build one passive evaluation module and a non-overwriting runner that emits Phase-1 artifacts plus an exact Phase-2/3 resume checkpoint.

**Tech Stack:** Python, csv/yaml, dataclasses, statistics, pytest.

## Global Constraints

- Do not modify policy behavior or historical outputs.
- Use PROP-NO-STAGE=72 and reject the stale 17-closure source.
- Do not build or solve an oracle.
- Do not run deadline sensitivity.
- Read only task-relevant files already identified in the full design.

### Task 1: Corrected passive evaluator

**Files:**
- Create: `src/evaluation/hv_phase1.py`
- Test: `tests/test_hv_phase1.py`

- [ ] Write tests for full-frame receipt, ground processing, information dominance, compression-quality thresholds, category partitions, HV/HV-TARGET predicates, and exact matched-ID latency.
- [ ] Run `python -m pytest tests/test_hv_phase1.py -q` and confirm failure because the module is absent.
- [ ] Implement the minimum pure functions needed by those tests.
- [ ] Run `python -m pytest tests/test_hv_phase1.py -q` and confirm PASS.

### Task 2: Non-overwriting Phase-1 runner

**Files:**
- Create: `scripts/run_hv_phase1_seed2026.py`
- Test: `tests/test_hv_phase1_runner.py`

- [ ] Write a miniature fixture test that requires the Phase-1 file set and rejects a stale 17-closure no-stage input.
- [ ] Run `python -m pytest tests/test_hv_phase1_runner.py -q` and confirm failure because the runner is absent.
- [ ] Implement authoritative source loading, alignment, comparisons, failure attribution, matched latency, checks, report rendering, and fail-if-output-exists behavior.
- [ ] Run `python -m pytest tests/test_hv_phase1_runner.py -q` and confirm PASS.

### Task 3: Full Phase-1 execution and verification

**Files:**
- Create through runner: `outputs/_hv_oracle_deadline_study_seed2026/`

- [ ] Run focused evaluator/runner and existing accounting tests.
- [ ] Run `python scripts/run_hv_phase1_seed2026.py` once.
- [ ] Run `python scripts/run_hv_phase1_seed2026.py --verify-only outputs/_hv_oracle_deadline_study_seed2026`.
- [ ] Confirm all Phase-1 checks pass, all rows are finite, PROP-NO-STAGE is 72, historical outputs are unchanged, and checkpoint lists exact Phase-2/3 resume inputs and code locations.
