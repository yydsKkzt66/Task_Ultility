# Battery and Energy Contract v1

The active environment uses a 230 Wh proxy battery, initial SOC 0.90, charge efficiency 0.92, discharge efficiency 0.95, and clipping to [0,1]. Solar input and platform base load come from the synthetic orbit-aware resource trace; battery update uses solar minus base, compute, generation, and communication loads.

The active hard resource/SafeFilter floor is 0.55. The 0.70 and 0.50 values are policy regions, not physical battery limits; the general `simulator.yaml` 0.18 value is shadowed by the active formal config. At capacity 0.50, frozen PROP minimum SOC is approximately 0.889, so neither policy region activates. Controllable energy is method-driven compute/generation/communication; platform base energy is uncontrollable. Future matched-energy fields are reserved but inactive.
