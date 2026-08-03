# Fixed Product Compression Probe Design

## Scope

This probe preserves the canonical SARFishSample seed=2026 task and environment semantics. The current simulator is architecture-level: it references real GRD scenes and label-derived statistics, but it does not run YOLO or materialize ROI, ROI+Context, or mask pixels. Existing payload sizes remain mathematical estimates and must not be described as codec output.

The probe therefore implements only a disabled-by-default physical interface and repository audit. It does not activate compression, modify scheduling, create crops, benchmark invented products, select a codec profile, or rerun PROP/B1.

## Architecture

`src/payload/fixed_product_compression.py` defines a small boundary:

- `CodecAdapter` is the codec interface for a future real-product provider.
- `EncodedPayload` owns audit-only raw fields, encoded physical fields, and the RAW -> ENCODING -> ENCODED -> QUEUED transition.
- downstream access is exposed only through an encoded physical view. It fails fast before ENCODED, on non-positive encoded size, or when a caller declares `raw_size_bytes` as its physical source.
- `FixedProductCompressionConfig` defaults `enable_fixed_product_compression` to false.

No existing simulator object consumes this interface in this probe because real product bytes and decoded task-quality evaluation are absent. This prevents a nominal interface from silently changing the canonical scheduler.

## Evidence and outputs

The probe runner writes only to `outputs/_sarfish_fixed_product_compression_probe_seed2026/`. It inventories configured size use, product representation, codec availability, immutable canonical directory hashes, and the current Git/Python environment. Benchmark, profile-selection, shadow-active, and active-trial artifacts use an explicit `BLOCKED_NOT_RUN` status and state the missing prerequisites.

## Safety gates

- Existing canonical output files are read and hashed only.
- No task, task hash, environment hash, policy, reward, deadline, requirement, contact, resource, PROP, or B1 behavior changes.
- No fixed multiplier is accepted as real compression.
- OpenJPEG is labelled JPEG 2000/J2K fallback, never HTJ2K.
- ABI failures are reported and not repaired.
- Active scheduling remains blocked until real products and decoded task-quality evaluation are provided.

## Verification

Focused tests cover default-off behavior, legal and illegal state transitions, encoded-only downstream fields, raw-size leakage rejection, and codec adapter contract. The report generator checks required artifacts, finite numeric fields where applicable, immutable canonical hashes, and absence of simulator/policy edits.

## Encoded propagation extension

The approved extension uses a separate `compression_physics_active` execution path. It does not replace or mutate the legacy `GeneratedPayload -> QueueItem` path. The active encoded path retains one `EncodedPayload` object by reference through queueing, priority, readiness/deadline feasibility, partial transmission, energy accounting, delivery/expiry, and event reporting.

No detached size or quality copy is authoritative. Event rows are immutable observations derived from the persistent object and preserve `payload_id` plus `compression_profile_hash`. Distinctive sentinel values validate the contract without claiming a real codec result. The real SARFish compression trial remains blocked because real product pixels and decoded task-quality evaluation do not exist.
