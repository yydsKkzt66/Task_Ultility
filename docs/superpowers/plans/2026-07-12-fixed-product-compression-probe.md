# Fixed Product Compression Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a disabled-by-default, rollback-safe encoded-payload contract and document why real compression trials are blocked in the current architecture-level SARFishSample simulator.

**Architecture:** A standalone payload module defines codec and state-machine interfaces without wiring them into PROP. A focused report runner audits the repository/environment and emits explicit blocked artifacts without touching canonical results.

**Tech Stack:** Python 3.9, dataclasses, enum, abc, pytest, CSV/JSON/YAML/Markdown.

## Global Constraints

- Do not generate ROI pixels or use mathematical multipliers as codec results.
- Do not modify task generation, policy, reward, B1, hashes, deadlines, contacts, resources, or canonical outputs.
- `enable_fixed_product_compression` is false by default.
- Do not run active scheduling, Oracle, deadline sensitivity, or simulations.

---

### Task 1: Encoded payload contract

**Files:**
- Create: `tests/test_fixed_product_compression.py`
- Create: `src/payload/fixed_product_compression.py`

**Interfaces:**
- Produces: `CodecAdapter`, `EncodedPayload`, `EncodingState`, `FixedProductCompressionConfig`, `DownstreamEncodedPhysicalView`.

- [ ] Write focused tests for default-off configuration, state transitions, encoded readiness, encoded-only downstream fields, and raw-size leakage rejection.
- [ ] Run `python -m pytest tests/test_fixed_product_compression.py -q` and verify failure because the module is absent.
- [ ] Implement the minimal standalone contract.
- [ ] Rerun the focused tests and verify pass.

### Task 2: Probe report generator

**Files:**
- Create: `scripts/run_fixed_product_compression_probe_seed2026.py`
- Create: `tests/test_fixed_product_compression_probe.py`

**Interfaces:**
- Consumes: repository paths and `fixed_product_compression` public types.
- Produces: all required probe artifacts in the isolated output directory.

- [ ] Write tests for blocked status, required files, codec naming, and immutable-output hashing.
- [ ] Verify tests fail because the runner is absent.
- [ ] Implement deterministic repository scanning and report generation.
- [ ] Verify focused tests pass.

### Task 3: Evidence generation and validation

**Files:**
- Create/update only files under `outputs/_sarfish_fixed_product_compression_probe_seed2026/`.

- [ ] Run the probe generator.
- [ ] Run focused tests plus compile/import checks.
- [ ] Hash canonical formal directories before and after and compare.
- [ ] Inspect Git diff to verify no existing policy/simulator/B1/task files changed by this probe.
- [ ] Complete rollback manifest with original/new hashes and exact removal/restoration steps.

### Task 4: Persistent encoded propagation path

**Files:**
- Modify: `src/payload/fixed_product_compression.py`
- Create: `src/payload/encoded_downlink_path.py`
- Modify: `tests/test_fixed_product_compression.py`
- Create: `tests/test_encoded_downlink_path.py`

**Interfaces:**
- Consumes: one persistent `EncodedPayload` reference.
- Produces: encoded queue, priority, feasibility, partial transmission, energy, latency, utility and lineage events.

- [ ] Add failing sentinel and negative tests using distinctive raw/encoded values.
- [ ] Verify failures before implementation.
- [ ] Extend strict payload accessors and state transitions with contextual errors.
- [ ] Implement the isolated encoded-active queue path without changing legacy queue behavior.
- [ ] Verify sentinel values propagate exactly and all negative cases fail as intended.

### Task 5: Propagation audit artifacts

**Files:**
- Modify: `scripts/run_fixed_product_compression_probe_seed2026.py`
- Modify: `tests/test_fixed_product_compression_probe.py`

- [ ] Add failing output-contract tests for usage inventory, lineage, recalculation and matrix completeness.
- [ ] Generate `encoded_propagation_usage_inventory.csv`, `encoded_payload_lineage_trace.csv`, `encoded_physics_recalculation_audit.csv`, and the propagation matrix.
- [ ] Mark the real active trial `BLOCKED_NOT_RUN` while reporting sentinel propagation separately.
- [ ] Verify feature-off tests, canonical fingerprints, output hashes and all focused tests.
