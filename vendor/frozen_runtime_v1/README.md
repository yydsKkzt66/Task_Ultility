# Frozen runtime bundle v1

This directory makes the final V25 experiment entries independent of the
original workstation's `D:/wt/...` Git worktrees.  It contains only source,
configuration, small manifests, native adapters, and small calibration inputs.
Raw SAR imagery and historical run outputs are deliberately excluded.

The four runtime roots are:

- `prop`: updated-PROP/shared-prior runtime snapshot;
- `b1_offload`: frozen B1 and OFFLOAD runtime snapshot;
- `tut`: TUT-UOI-PRODUCT-LITE-v2 runtime snapshot;
- `myopic`: frozen historical MYOPIC runtime snapshot.

`native_tools/late_hv_native_direct_final_v6` contains the identity-checked
binding and normalization adapters.  `inputs` contains the original V25 bundle,
the shared-prior calibration, and the frozen one-orbit solar trace.

The copied scientific logic is byte-preserved.  Two path-only portability
adapters are intentionally applied:

1. the updated-PROP helper resolves its calibration and native tools relative
   to this directory;
2. the native identity contract resolves the same frozen source roots relative
   to this directory while retaining all original source-file SHA256 gates.

Every file is locked by
`artifacts/reproducibility/frozen_runtime_v1_sha256.json`.  Run
`python scripts/verify_portable_reproduction_v1.py` before an experiment.
