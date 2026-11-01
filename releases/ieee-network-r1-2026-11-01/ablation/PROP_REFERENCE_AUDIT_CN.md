# 完整PROP正式主实验消融合并包

只读整理；5个最终审计通过attempt，1600任务/21600秒。未运行新实验。

|种子|最终attempt|毛效用|净效用|HV净效用|HV闭环|普通闭环|
|---:|---|---:|---:|---:|---:|---:|
|4101|attempt_06|536.8420857390238|400.4888197238886|322.13034284121477|321|333|
|4102|attempt_03|514.3971633370298|382.7168546109931|300.34503847142025|316|350|
|4103|attempt_05|522.8934467781245|390.4125640062176|306.7907251184923|319|360|
|4104|attempt_05|546.0412856618686|413.01627821411853|334.1210633633666|340|355|
|4105|attempt_03|529.0050306965127|389.49407064513935|311.4810285629538|332|347|

完整精度均值：总体395.22571744007143；HV314.97363967148954。
相对用户近似值差：总体-5.599285373136809e-07，HV-3.285104526185023e-07。逐种子先保留6位再求均值会得到总体395.2257176，解释末位显示差；未改任何结果。

到达时钟：8000/8000与冻结到达时刻二进制浮点精确相等；8000/8000有实际热积分端点。若EXACT_ARRIVAL_V2只增加所述注册前正向尾段积分，本参考轨迹新增分支触发0次。外机实际修复源码/清单未提供，因此不认证其全部变更与原版等价，原来源不改标为V2。

候选ID：原生ID未保存；导出的是明确标识的来源SHA/行号记录ID及原行SHA。预测热约束归因来自失败区间unsafe_certificate，不用失败前预测峰值替代。

物理记录：包含五种子全部实际积分区间及chip/RF/board/structure温度前后值、峰值和峰值时刻、热量账本、SOC和能量账本；4101单独绘图数据在DERIVED/PLOT_4101。

原始SOURCE_MANIFEST、INPUT_MANIFEST、恢复观察器manifest及五份AUDIT/IDENTITY/COMPLETED原字节保留。无其他方法结果、失败attempt、旧268/DEV任务流、图片缓存或pkl检查点。

导入先执行scripts/verify.py；复算可加--recompute。跨机合并须核对实际输入/配置/源文件差异，命令见README_CN.md。

ZIP: D:\北邮work\5.15多状态感知轻量化星上agent\leo_onboard_codesign_case_study_v2\handoff_out\NETWORK_PROP_ABLATION_MERGE_20261030_01\PROP_ABLATION_MERGE_PACKAGE.zip

SHA256: 78afb32e448ca95d819865deffe75a8b583f6bd92aea7b07f0e22cfbb8ab0f51

Independent fresh extraction: PASS; 60 derived files exactly reproduced. Raw source hashes verified. No simulation runs.
