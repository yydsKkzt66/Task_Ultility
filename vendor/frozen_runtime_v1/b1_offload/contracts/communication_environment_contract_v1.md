# Communication Environment Contract v1

Contacts are synthetic periodic windows: 5,700 s period, 600 s nominal duration, 420 s phase. Link quality drives 35--65 Mbit/s nominal throughput; experiment multipliers 1.00, 0.50, and 0.35 scale this throughput. Per-slot nominal MB is `throughput_mbps / 8 * 20`; contact capacity is its sum over contact slots.

Keep distinct: nominal capacity, multiplier-scaled contact capacity, thermal-safe effective capacity, and actual transmitted bytes. PROP queue items support partial transmission and resume across slots/contacts. Deadline-expired items are removed; equality at deadline remains eligible. B1's method policy requires a full 4 MB frame fit and is non-preemptive once started. No propagation delay or protocol overhead is represented (`UNKNOWN`). Communication energy uses the 35 W proxy and actual transmit time.

`PROP-HEADROOM` does not raise 0.92. It predicts next thermal state and binary-searches the maximum bytes that remain at or below `0.92 - 1e-10` (40 iterations); compute-only infeasibility or no safe bytes yields zero capacity.
