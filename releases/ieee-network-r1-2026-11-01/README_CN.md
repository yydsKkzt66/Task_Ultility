# IEEE Network 第一次大修实验归档

本目录对应已提交的第一次大修，独立于旧竞赛发布 `comsoc-apb-2026-v1`。

已包含：新版主实验四方法×五种子20组；容量、价值敏感性与服务中断韧性分析三种子完整96条记录（84组扰动＋12条共享正常参照）。包含完整精度CSV、原版本清单、必要审计与账本、源代码/配置/输入/测试压缩包、复算与重绘入口。

**消融由另一台电脑另行上传，本次先发布本机已具备的主实验、敏感性与韧性材料。** 已放入五种子完整PROP参照与外机证据清单，供另一台电脑衔接。没有使用seed2026、DEV1600或旧268任务结果补齐；本目录不宣称已经包含四组正式消融及其EXACT_ARRIVAL_V2实际源码。

```console
python reproduce.py verify
python reproduce.py tables --out reproduced_tables
python reproduce.py figures --out reproduced_figures
```

前两条仅验证与复算已有数据，不启动实验。绘图依赖见 `requirements_plot.txt`。所有命令使用相对路径；原始来源清单中的D盘/C盘路径仅供追溯。

主实验用4101–4105五种子；敏感性/韧性用4101–4103配对参照。样本标准差ddof=1，保留率是三个逐种子比值的均值。保留负值和超过100%的比值；小正基准不能据比例夸大优势。时延仅统计实际闭环任务，P95先按种子计算。

GROUND/OFFLOAD是独立4.29s地面处理版；TUT为共同因果时延版；PROP为正式冻结版。GROUND/4102原严格浮点相等FAIL及作者数值容差接受说明均保留。外机大轨迹未回传部分沿用原接受证据，不声称本次重做物理审计。详见英文README的证据边界。

冻结源码分五个版本ZIP保存，不混合改写科学实现；其中旧支持文件只是清单依赖，不作为旧版结果混入表。`reproduce.py extract-runtime --name prop_formal --out runtime_prop`可解压查看源码和测试。原生完整运行命令与版本映射见 `REPRODUCIBILITY.md`，本次归档没有重跑正式实验。
