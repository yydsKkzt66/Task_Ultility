# Simulation Timeline Contract v1

The run spans `[0, 86400)` seconds in 4,320 slots of 20 seconds. A completion at `deadline + 1e-9` or earlier is on time.

Per-slot order implemented by `StepRunner.run_step`:

`state/context -> arrivals -> policy/safety/model action -> inference and product generation -> carryover request opportunity -> queue preview and thermal-safe capacity -> expire/enqueue/sort/transmit -> lifecycle/ground delivery -> energy accounting and battery update (solar/base/compute/communication) -> thermal update -> safety checks -> internal reward/controller update -> commit state`.

Arrivals are exposed before action selection. Generated payloads enter the downlink queue before transmission in that slot. Deadline eligibility is inclusive. Partially transmitted queue items retain remaining bytes across slots and contacts. Compute latency and generation latency are charged, but there is no persistent partial-compute state across slots.
