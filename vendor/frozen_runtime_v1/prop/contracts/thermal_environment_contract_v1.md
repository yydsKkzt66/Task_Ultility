# Thermal Environment Contract v1

Thermal state is a dimensionless proxy pressure clipped to [0,1], initialized at 0.52. Guard is 0.85 and hard limit is 0.92. The update combines ambient relaxation/passive cooling, optional active cooling, sunlight/eclipse terms, compute heat (0.0025 normalized/J), and communication heat (0.0015 normalized/J), scaled to a 60 s reference.

`PROP-PAUSE` disables a hot slot under the old gate. `PROP-HEADROOM` instead computes the largest safe bytes for the current slot without changing the hard limit. `B1-COMMON-THERMAL` uses the common thermal accounting. A tested hysteresis variant showed no additional benefit and is not an active formal mechanism. Frozen reports record zero hard-limit violations.
