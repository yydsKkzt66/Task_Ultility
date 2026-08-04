# Configuration Conflicts v1

No blocking conflict was found.

| Severity | Active value | Shadowed definition | Resolution |
|---|---|---|---|
| SHADOWED_BUT_RESOLVED | formal slot 20 s, battery 230 Wh, initial SOC 0.90, hard floor 0.55, queue 5120 MB | generic `configs/simulator.yaml` defaults including 60 s, 35 Wh, 0.85, 0.18, 64 MB | formal runner explicitly loads the specialized config |
| SHADOWED_BUT_RESOLVED | B1 compression processing power 1.4 W | baseline YAML default 6.8 W | validated runtime `b1_config` override is authoritative |
| SHADOWED_BUT_RESOLVED | calibrated tiny/small/medium profiles | generic `model_pool.yaml` base values | calibrated profile overrides are loaded by formal PROP config |
| INFORMATIONAL | formula product sizes with fallbacks | fallback values may appear alone in reports | runtime formula/clamp is authoritative; all remain proxy |
