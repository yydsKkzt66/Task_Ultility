# Current Metric Definition Audit

| metric_name | implementation_file | implementation_function | formula | unit | denominator | method-specific | cross-method comparable | known_problem |
|---|---|---|---|---|---|---|---|---|
| proposed gross_task_utility | `src/evaluation/metrics.py` | `summarize_episode` | sum of `record.effective_utility` | utility | all trajectory records | yes | no | online effective utility, not strictly ground-event marginal utility |
| proposed delivered_only_incremental_utility | `src/evaluation/metrics.py` | `summarize_episode` | sum of lifecycle delivery increments | utility | delivered product events | yes | partially | staged product semantics unavailable to B1 |
| proposed resource_penalty | `src/reward/reward_function.py` | `compute_reward` | calibrated energy + latency + queue penalty | utility | task decisions | yes | no | mixes costs already represented in common timeliness |
| proposed requirement_penalty | `src/reward/reward_function.py` | `compute_reward` | quality-gap + information-gap penalty | utility | task decisions | yes | no | common information utility must not subtract it again |
| proposed long_term_risk_penalty | `src/evaluation/metrics.py` | `summarize_episode` | thermal + battery reserve + downlink burden + queue risk | utility | slots | yes | no | internal control objective |
| proposed terminal_state_penalty | `src/evaluation/metrics.py` | `_compute_terminal_state_penalty` | normalized battery + thermal + terminal queue squares | utility | terminal state | yes | no | internal normalization differs from B1 legacy |
| B1 delivered_effective_utility | `src/baselines/b1_enhanced.py` | `_summary_metrics` | sum task value times ground quality over completed tasks | utility | ground completed | yes | no | no timeliness or requirement-aware marginal information |
| B1 delivered_task_net_utility_sum | `src/baselines/b1_enhanced.py` | `_summary_metrics` | equals delivered effective utility | utility | ground completed | yes | no | name says net but no cost is deducted |
| B1 net_system_utility | `src/baselines/b1_enhanced.py` | `_summary_metrics` | delivered utility minus 0.01 times total satellite Wh | utility | episode | yes | no | legacy energy coefficient and incomplete semantics |
| completed | both adapters | unified evaluator | finite requirement-closing ground result time | count | all arrivals | no | yes | legacy methods used different meanings |
| deadline satisfaction | unified evaluator | `summarize_method` | on-time completed / ground completed | ratio | ground completed | no | yes | legacy reports also used all-arrival denominator |
| energy | unified evaluator | method adapters | compute/compression/communication/idle/housekeeping converted from J to Wh | Wh | episode | no | yes | B1 legacy already used Wh; proposed legacy summaries use J |

No common term double-counts latency or information gaps: timeliness and product-requirement sufficiency enter common gross utility once. Housekeeping is reported separately and excluded from controllable mission energy.
