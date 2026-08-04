# Task Generation Contract v1

SARFishSample GRD validation scene metadata and annotations seed a deterministic proxy task stream. The active index has 50 source scenes. Tasks sample scenes with seeded `rng.choice`, so one scene may produce multiple simulation tasks; a task is a scheduled observation/task record, not a unique image or end-to-end pixel execution. Seeds control scene selection and synthetic arrival/deadline draws; existing task manifests are authoritative and must not be regenerated for comparison.

Task classes map `routine -> medium`, `high_value -> high`, and `emergency -> urgent`. HV means high-value or emergency under the runtime definition; HV-TARGET is the validated stricter target subgroup recorded by the existing manifests. Requirements are alert, verification, context, or evidence. Task IDs and duplicate handling are deterministic in the task builder/canonical environment.

Online policies do not read future arrivals or true evaluation labels. Target count and annotation summaries are stored as offline metadata for task construction/evaluation and explicitly excluded from online priority. B1 does not read and compress real pixels: its 4 MB frame is a fixed proxy. PROP ROI sizes are formula/fallback proxies, not measured codecs. Model quality and product quality are proxy formulas.
