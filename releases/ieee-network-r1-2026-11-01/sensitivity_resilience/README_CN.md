# 完整数据入口

先读 **tables/00_REPORT_CN.md**，查看32组总体/HV净效用均值±样本标准差和来源限制。

- **tables/ALL_96_PER_SEED_METRICS.csv**：96条逐种子完整精度数据，含全部核心指标、部分补充字段和来源。
- **tables/MEAN_SD_32_GROUPS_WIDE.csv**：四方法×八条件，共32组的各项均值和样本标准差；LONG版适合脚本筛选。
- **tables/MATCHED_84_SEEDS_CHANGES_RETENTION.csv**：84条扰动结果相对同方法同种子的正常基准变化及保留率。
- **tables/MATCHED_28_GROUPS_CHANGES_RETENTION.csv**：28组配对差值与保留率均值±样本标准差。
- **tables/FIGURE_CAPACITY.csv、FIGURE_VALUE.csv、FIGURE_OUTAGE.csv**：三个敏感性子图数据。
- **tables/SOURCE_INDEX_96.csv、B20_IMPORT_VERIFICATION.csv、MERGE_VALIDATION.json**：来源、版本与匹配证据。
- **tables/FIELD_COMPLETENESS.csv**：所有核心指标齐全；部分附加物理/运行记录或HV分项费用缺失时留空，不填0。

全部组使用4101–4103且n=3、ddof=1。正常参照只在96条主表中保存一次，多个图可以共同引用。不要将这份三种子正常均值替代论文五种子主实验均值。

CSV浮点数保留程序输出精度，报告只调整显示小数位。总体/HV/普通净效用的符号全部保留。GROUND正常总体净效用基准较小，因此必须结合绝对变化阅读保留率。原有NORMAL GROUND4102数值容差接受状态与外机来源证据限制均保留。

输入证据在inputs，原目录另见inputs/COLLECTION.json；原项目和外机回传目录均未修改。原始大轨迹不在此包。无需访问原电脑即可复算：

```cmd
python merge_data.py --output recomputed_tables
```

输出目录须为空，不使用--collect。复算仅使用Python标准库。独立验证结果见INDEPENDENT_VALIDATION.json，包含错误种子/版本/配置、重复行、费用和产品计数错误的拒绝检查，以及异目录复算的逐文件一致性。
