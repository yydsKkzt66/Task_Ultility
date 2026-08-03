# Corrected REQALIGN 与 MYOPIC 公平对比

唯一运行入口：

```powershell
python scripts/run_myopic_e2e_utility_greedy.py
```

该命令默认在完全相同的场景和任务流上运行两个方法：

- 已晋升的 `PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-v1`
- `MYOPIC-E2E-UTILITY-GREEDY-ON-CORRECTED-REQALIGN`

二者共享 corrected REQALIGN 的 requirement projection、defer re-entry、
payload lifecycle、队列调度、接触感知下传和到地核算；只替换 projection
之前的动作排序决策器。默认 seed 为
`2026,2027,2028,2029,2030`，默认输出目录为：

```text
outputs/myopic_vs_corrected_reqalign_multiseed/
```

显式等价命令：

```powershell
python scripts/run_myopic_e2e_utility_greedy.py `
  --seeds 2026,2027,2028,2029,2030 `
  --methods CORRECTED_REQALIGN,MYOPIC `
  --output-dir outputs/myopic_vs_corrected_reqalign_multiseed
```

已有同名输出时入口会拒绝覆盖。确认覆盖该精确目录时增加
`--overwrite`。只做 MYOPIC 快速检查时可使用
`--methods MYOPIC`，但正式公平比较应保留默认两方法。`REQALIGN`
可作为 `CORRECTED_REQALIGN` 的命令行别名；旧 PROP 已从本入口的方法
集合中移除。

## 输出目录

```text
outputs/myopic_vs_corrected_reqalign_multiseed/
├── run_manifest.json
├── scenario_manifest.csv
├── seed_metrics.csv
├── aggregate_metrics.csv
└── seeds/
    └── seed_<seed>/
        ├── scenario_identity.json
        ├── cross_method_fairness.csv
        ├── PROP-SHARED-CALIBRATED-PRIOR-REQALIGN-v1/
        └── MYOPIC-E2E-UTILITY-GREEDY-ON-CORRECTED-REQALIGN/
            ├── metrics.json
            ├── runtime_identity.json
            ├── task_outcomes.csv
            ├── ground_delivery_events.csv
            ├── task_full_closure_latency.csv
            ├── per_slot_resource_trace.csv
            ├── candidate_action_audit.csv
            └── payload_lifecycle.csv
```

`seed_metrics.csv` 是逐 seed 统一指标，`aggregate_metrics.csv`
是五个 seed 的汇总。两者均包含：

- `J_all_net`、`J_HV_net`
- `N_HV_closure`、`N_HV_metadata`、`N_HV_ROI`、`N_HV_evidence`
- `overall_closure_rate`
- `mean_closure_latency_s`、`p95_closure_latency_s`
- `energy_Wh`、`transmitted_bytes`
- `thermal_violations`、`SOC_violations`

`task_full_closure_latency.csv` 给出任务级闭环分解。闭环仅在任务所需
product 到地后成立；其中 `full_closure_latency_s` 从任务到达开始，
覆盖 inference、product generation、queue waiting、contact waiting
和 downlink transmission。`metadata` 只计入 metadata 阶段到达，不能
替代 verification、context 或 evidence 所要求的完整 product。

`runtime_identity.json` 和 `cross_method_fairness.csv` 额外记录 corrected
REQALIGN projection 配置哈希，验证两种方法确实共享同一个 projection、
re-entry 和下传后端。
