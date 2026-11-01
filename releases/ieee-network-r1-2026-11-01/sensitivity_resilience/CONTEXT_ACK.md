# CONTEXT ACK

```text
CONTEXT_VERSION = 2026-09-24-AUTHOR-INTENT-BOUNDED-P0-CLOSED-v1；当前范围依用户后续6h正式实验及本轮合并授权
PROJECT_PURPOSE_IN_ONE_SENTENCE = 将实际到地任务效用和固定资源费用的已审计证据，按配对种子形成完整敏感性数据。
CURRENT_STAGE = A22完成，B20回传，冻结PROP/共同因果TUT48与新版地面正常6待统一汇总
TASK_CLASS = READ_ONLY_DIAGNOSTIC
TARGET_SCIENTIFIC_QUESTION = 四方法对容量、价值差异和地面服务中断的敏感性；不更改实验行为
AFFECTED_ARMS = GROUND、OFFLOAD、TUT-Adapt、PROP；不涉及历史D0/D0.5/D1.5修改
AFFECTED_TRUTH_LAYERS = 只读评价及来源证据；不改变决定、授权、预约、执行或计分
MODULE_OWNER = 本独立合并脚本；原Evaluator仍是实际到地与固定kappa的权威
NEXT_CONSUMER = 论文统计、敏感性子图、回复信
CAUSAL_CHAIN = 冻结任务与配置→原因果仿真→实际发送/到地→原终态计分和审计→只读逐种子合并→均值/样本标准差
TIME_SCALES = 微观因果事件时钟保持；慢控制20s；本次只读数据不推进仿真时间
HORIZON_S = 21600
MACRO_SLOT_S = 20
MACRO_SLOT_COUNT = 1080
TASK_COUNT_FOR_TARGET_RUN = 1600；HV448，普通1152
DETECTOR_BACKEND = MockDetector/profile工程代理，不是星载实测
DATA_FILES_ACTUALLY_USED = 原A22汇总包及轻量证据、用户B20_METRICS_RESULTS、冻结REUSE_PROP_TUT_48、NORMAL_NEW_GROUND_OFFLOAD_6及原正常参照
ENERGY_COMPOSITION = 可控推理/编码/TX成本与背景能量区分；沿用固定E/B/Q，不将历史24h平台占比套入本轮
CURRENT_P0_STATUS = 2026-09-24闭合仅适用历史诊断；当前正式运行另有用户授权，不扩大历史P0结论
FROZEN_ITEMS = 原科学源码/配置/结果、3种子4101–4103、1600/21600、产品和闭环语义、费用、价值归一化、窗口和通知时点
STOP_RULE_CONFLICT = 历史24h/268任务限制与本轮授权范围不同；只读合并，不启动仿真或调参
SUCCESS_CONDITION = 96个唯一条件×方法×种子键，32组每组3种子；A/B不交叠，B20身份匹配；成本/闭环/产品守恒；旧76条已有数值不改
LEGITIMATE_REJECTION_CONDITION = 哈希、配置、种子、费用或计数不一致则隔离，不用其他版本补位
FALSIFYING_EVIDENCE = B20不是冻结清单对应组，普通/HV或毛净效用无法闭合，重复或缺失，来源汇总数值被替换
UNKNOWN_ITEMS = B机原始轨迹未随轻量包回传；其原始审计依回传证明，不宣称本机重新执行；部分历史附加字段可能缺失
PRODUCTION_CHANGE_AUTHORIZED = NO；仅新建独立派生汇总和可复算小包
```

必读文档已按顺序加载，context_preflight.py PASS。原24h背景/268任务信息不用于本次统计。每组P95先在种子内按原口径计算，三种子均值与样本标准差ddof=1；不平均两机局部均值。完整结果保留负净效用和全部种子。
