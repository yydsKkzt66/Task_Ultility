# 双节点热模型快速校准与四方法复核

最终判定：**LITERATURE_ANCHORED_PROVISIONAL_RERUN_COMPLETE**

校准证据：`LITERATURE_ANCHORED_PROVISIONAL_THERMAL`。未找到同设备、同工况原始轨迹；参数不能称为真实在轨精确校准。

热参数哈希：`17f98859448a41f2d0ab83036910b3505f9b37c3fc5a8e21dc62254f9e5bcfa9`
四进程PID：30400, 32400, 16076, 6640

物理拟合门控：PASS；三层记账一致性：`PASS`；公共热函数一致：True；受保护哈希不变：True。

| method | J_all | J_HV | all/HV/high/urgent closure | max chip/surface C |
|---|---:|---:|---:|---:|
| CURRENT-PROP-THERMAL-CORRECTED | 46.74049743165179 | 39.19279808986528 | 99/54/39/15 | 16.110/15.799 |
| FIXED-BALANCED-THERMAL-CORRECTED | 42.35041791282194 | 34.999000895937826 | 101/53/38/15 | 16.056/15.776 |
| TUT-PROXY-THERMAL-CORRECTED | 38.734473367881606 | 26.486780233061207 | 118/53/37/16 | 16.089/15.795 |
| B1-COMMON-GROUND-THERMAL-CORRECTED | -12.493610228079746 | -8.198499204932865 | 226/56/40/16 | 16.003/15.782 |

外部8.5W/540min与15min冷却锚点仅验证、不调参；因边界条件不同，偏差如实保留。四方法J值未参与参数选择。

新模型下所有High候选均未因热约束被拒，说明此前Oracle/模型自适应结论受旧热尺度影响，后续需要独立重诊断，但本轮不自动改方法。

建议仅将本轮用于机制诊断；在获得同设备chip/surface/environment/power原始轨迹前，不建议进入多seed正式论文实验。

DUAL_NODE_THERMAL_CALIBRATION_AND_FOUR_METHOD_PARALLEL_RERUN_COMPLETE
