# Compute Environment Contract v1

The simulator uses abstract BUPT-1-like COTS compute profiles, not a named CPU, GPU, FPGA, or flight processor. Core count, memory capacity, and physical FLOP/s are unknown. One logical action is evaluated per task/slot path. Inference and encoding latency/energy are proxy-accounted; no persistent cross-slot compute queue or partial-compute execution is modeled.

Tiny/small/medium mock profiles are selected by method policy. Their calibrated relative latency, energy, quality, and thermal scales are in `compute_and_model_contract_v1.csv`. These values must not be presented as measured SARFish YOLO performance.
