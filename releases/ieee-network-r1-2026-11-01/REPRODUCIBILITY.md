# Version-specific reproduction

The default release commands recompute tables and figures from archived results. They do not simulate. Frozen source bundles additionally permit inspection and new runs with their original native commands. Results of a new run are separate evidence and must be audited; do not replace an accepted source silently.

| Result | Runtime archive | Native entry point after extraction |
|---|---|---|
| PROP main, 4101–4105 | `prop_formal` | `python -I run.py worker --seed 4101 --arm PROP --out new_run` |
| TUT main, 4102 | `tut_causal_seed4102` | `python -I run.py worker --seed 4102 --arm TUT_CAUSAL_LATENCY --out new_run` |
| TUT main, 4101/4103/4104/4105 | `prop_tut_sensitivity` | `python -I sensitivity_run.py worker --job NORMAL_4101_TUT_ADAPTED --out new_run` |
| PROP/TUT capacity/value/outage | `prop_tut_sensitivity` | `python -I sensitivity_run.py worker --job A1_4101_PROP --out new_run` |
| GROUND/OFFLOAD main | `ground_parallel_main` | `python -I run.py worker --seed 4101 --arm GROUND --out new_run` |
| GROUND/OFFLOAD perturbations | `ground_parallel_sensitivity` | `python worker9h.py --job A1_GROUND_4101 --out new_run` |

Change only the method/seed/job identifier to another declared job in the applicable archive. Five-seed main GROUND/OFFLOAD has an arrival-tail recovery/acceptance history: strict-original execution can encounter the preserved floating-point arrival audit issue; do not change its audit to PASS. The sensitivity worker includes the separately frozen exact-arrival wrapper. Its operational source version is not silently substituted for the accepted main-run source. Original main recovery helpers are retained under `runtimes/operational_history/ground`, and the accepted evidence is under `main/accepted_evidence.zip`.

The original source archives retain complete manifests and their support files, including some old configurations. Do not launch the legacy GROUND/OFFLOAD or old TUT embedded in `prop_formal`, or treat its ablation design as the unretrieved external EXACT_ARRIVAL_V2 implementation. `runtimes/INDEX.json` is the authoritative mapping for this release.

Frozen scenario: 21,600 s; 1,600 arrivals; four service windows [3600,4200), [9300,9900), [15000,15600), [20700,21300). Normal absolute nominal capacity multiplier is 0.50; variants are 0.15/0.30/0.90, without double scaling. Value variants 1:2:5:10 and 1:5:20:50 preserve the normal total prior value by the frozen normalization. B1 service outage is [9300,9900); B2 is [9300,15600), including the gap. Controllers learn the interruption at 9300 s, before that instant's decision/service. Actual physics, transmitted bytes, energy, safety checks and original DDL remain in force.

Original runtime requirements record Python 3.11.15 and exact package versions. A version may need the corresponding future-dated package availability on the reproducing platform; the archive does not vendor dependency wheels. Install only from an environment you control. Runtime thread limits are one numerical thread per process. Parallel scheduler wall time and desktop RSS are not onboard latency/energy/memory measurements. Historical hardware binding remains unconfirmed; no current machine is silently claimed as the original host.

For release verification, every source/input byte in each ZIP is checked, all 20 main keys and 96 sensitivity keys are checked, accounting closes, normal seeds match the main data, and sensitivity tables are regenerated using only packaged inputs. This does not claim a fresh full-run reproduction. External original physical trajectories that were never returned are not fabricated.
