## IEEE Network first-revision experiments (2026-11-01)

The revised 6 h / 1,600-task main experiments, capacity/value sensitivity and service-outage resilience data, frozen runtime sources and reproduction scripts are archived in [releases/ieee-network-r1-2026-11-01](releases/ieee-network-r1-2026-11-01/README.md). Main and sensitivity/resilience metrics are complete. Formal ablation results are managed on another computer and will be uploaded separately. The earlier frozen competition release [comsoc-apb-2026-v1](https://github.com/yydsKkzt66/Task_Ultility/tree/comsoc-apb-2026-v1) remains unchanged.

---

# LEO Task-Utility Case Study Scaffold

本项目服务于 **Task-Utility-Driven Computation-Communication Co-Design for Resource-Constrained LEO Satellites**。  
当前版本不是最终在轨实验实现，而是一个 **trace-driven closed-loop case study scaffold**：使用 mock dataset、mock detector、synthetic resource trace 和 synthetic link trace，在统一的 **virtual timeline** 上跑通完整闭环。

## 项目定位

这个项目强调的是“任务效用驱动的星上计算与星地通信协同设计”，不是普通遥感检测 demo，也不是单独的通信调度实验。

核心问题是：在电量、AI 预算、热压力、接触窗口、下传速率和队列都受限的前提下，系统如何联合决定：

1. 当前任务采用什么模型复杂度；
2. 生成什么粒度的下传载荷；
3. 是否立即下传、入队等待还是 defer/store。

当前版本固定 9 维 compact context：

`x_t = [B_t, E_t_AI, Theta_t, V_t, D_t, H_t, U_t, C_t, Q_t]`

这里：

- `C_t` 是 contact / SNR / throughput / remaining window 的紧凑通信机会表示，而不是把这些物理量全部展开进上下文；
- `Q_t` 是物理队列大小 `queue_mb` 的归一化值，真实队列单位始终保持为 MB；
- 这种 9D 设计是 magazine case study 阶段的 compact representative context，不是未来 TWC 扩展版的完整状态空间。

## 固定主流程

主流程严格采用：

`TaskSample -> ContextVector -> Candidate Joint Actions -> Predictor -> Safe Filter -> Router -> Model Execution -> DetectionResult -> Payload Generation -> Contact-Aware Downlink -> Observed Feedback -> Observed Reward -> State Update -> Router Update`

其中：

- 联合动作是 `model complexity x downlink payload policy`；
- 第一版候选动作固定为 5 个：
  - `lightweight+metadata`
  - `balanced+metadata`
  - `balanced+roi`
  - `high_accuracy+roi_context`
  - `delay/store`
- `delay/store` 会跳过重模型执行，只产生零或极小负载，用作安全 fallback action。

## 为什么 reward 不是单纯检测精度

奖励放在 `src/reward/`，显式区分 predicted reward 与 observed reward。  
动作选择前使用 **PredictedMetrics** 做估计；动作执行后使用 **ObservedFeedback** 重新计算 observed reward，并据此更新 router。

任务效用 reward 关注：

1. 任务价值；
2. 信息充分度，而不是 payload 越大越好；
3. 时效性；
4. 实际交付状态；
5. 计算与通信总能耗；
6. 队列压力；
7. 安全约束违背风险。

信息充分度使用饱和形式：

`S_info = min(1, I(p) / I_req(x_t))`

它表达“传到刚好够用”，而不是盲目传更多。

## 通信模块说明

`src/link/downlink_queue.py` 是一个 **contact-aware closed-loop downlink** 模块，而不是简单的 `send(payload)`：

- 非接触窗口时 `r_down = 0`，当前 payload 会进入队列；
- 接触窗口内按容量对当前任务和历史队列统一调度；
- 队列支持过期丢弃、优先级排序和当前任务交付状态返回；
- reward 中同时使用 `delivery_status`、`S_down_total_mb`、`queue_next_mb`、`E_comm`、`L_comm`。

## 当前默认组件

- 数据集：`MockDatasetAdapter`
- 检测器：`MockDetector`
- 模型层级：`lightweight / balanced / high_accuracy`
- 路由器：`FixedRouter / RuleBasedRouter / SafeLinUCBRouter`
- trace：synthetic resource trace + synthetic link trace

这些组件都可以替换，且接口已经为后续扩展保留：

1. 可替换为 SARFishSample-GRD 或其他 SAR/EO 数据集；
2. 可替换为 YOLO-family tiny/small/medium 模型池；
3. 可接入本地模型 profiling；
4. 可接入 COTS 卫星资源 trace；
5. 可接入 contact-window / SNR / throughput trace；
6. 可在未来扩展 residual predictor、dual energy budget controller 和更严格的 safe regret analysis。

## 目录结构

```text
configs/        YAML 配置
scripts/        实验与辅助脚本
src/common/     通用类型、常量、配置与日志
src/data/       数据集适配层
src/timeline/   virtual timeline 与 trace
src/models/     detector、profiling 与 registry
src/context/    9D context 构建
src/actions/    联合动作空间
src/payload/    payload 生成与信息充分度
src/link/       contact-aware downlink
src/energy/     计算/通信能耗与状态更新
src/reward/     reward 分解
src/routers/    fixed / rule / safe LinUCB
src/simulator/  predictor、step runner、episode runner
src/evaluation/ CSV 汇总与绘图
```

## 运行方式

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行完整仿真：

```bash
python scripts/run_simulation.py
```

运行测试：

```bash
pytest
```

## 输出内容

运行后会生成：

- `outputs/summary.csv`
- `outputs/trajectories/*.csv`
- `outputs/figures/mission_utility.png`
- `outputs/figures/queue_pressure.png`
- `outputs/figures/battery_state.png`
- `outputs/figures/action_distribution.png`
- `outputs/figures/utility_per_joule.png`

## 关于真实系统的边界

README 和代码注释都刻意使用以下表述：

- `trace-driven closed-loop simulation`
- `virtual timeline`
- `profile-calibrated estimate`
- `simulation-observed feedback`

这样做是为了明确：当前版本是严谨、可扩展、可替换、可调试的 case study scaffold，  
而不是宣称所有数据、模型、资源 trace 和链路 trace 来自同一颗卫星同一时刻的真实同步在轨闭环实验。
