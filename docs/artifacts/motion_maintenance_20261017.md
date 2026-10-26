# 动效审查与定向维护（2026-10-17）

结论：建议优化现有动效的中断、几何和生命周期，不增加整页过渡。主页面和OutputPanel已经采用直接切换，B-scan标签继续复用每组视图；本轮保留这些决策，不做布局重设计，不改算法、科学数据、异步身份守卫或处理草稿语义。

分支codex/maintenance-20261017，保留此前四轮未提交改动、TODO原差异和本地资产；本轮不提交、推送、打包或发布。证据目录为output/motion-maintenance-20261017，与前轮独立。

## 范围与修复

| 优先级 | 确证问题 | 最小修复 |
| --- | --- | --- |
| P1 | Windows动画开关用1字节c_bool接收Win32 BOOL | 使用wintypes.BOOL缓冲区并返回bool |
| P2 | 进度/徽章即时更新提前返回，旧动画随后可覆盖新状态 | 在判断即时路径前停止旧动画；主题重刷也经即时路径 |
| P2 | Fluent ProgressBar内置150ms补间与共享200ms补间叠加 | 共享helper作为唯一驱动，从实际绘制值续接；整数端点直连Qt控件slot |
| P2 | 徽章/资源栏每次新建parented animation积累；资源栏切页即时恢复未停旧动画 | 常驻单实例，所有即时路径中断旧动画；徽章用Qt动画virtual callback，资源栏用控件bound slot，避免捕获父控件的裸闭包 |
| P2 | SlimSegment快速再选时先跳到旧逻辑位置 | 从当前视觉x续接，最终选中/滑块一致；减少动画时即时落位 |
| P2 | Panel/Card动画中内容一直隐藏，Card从无限maximumHeight起播 | 保留220ms，改实际几何起点与可见裁切；Panel临时宽度/policy终态恢复；Card用独立外层viewport裁切自然高度内容，完成后才隐藏/释放高度上限；隐藏窗口也收敛 |
| P2 | Panel/Card与算法链胶囊未走已有减少动画总闸 | 沿用ui.motion偏好，保持原220/180ms及交互合同 |

产品改动限定ui/motion.py与job_widgets、dock_panel、collapsible_card、collapsible_panel、segment_tabs、chain_strip七模块。测试环境对齐真实启动，另将现有胶囊动画测试改为先显示控件再检查滑动。旧gallery/grid没有因搜索命中而顺带改写；BScanWorkspace只列观察，不因小矩阵局部计时改架构。

## 原生现状测量

由独立执行代理使用真实MyGPRMainWindow/backend/七页，UIA确认自建PID/HWND可见；动作回放为QTest鼠标键盘和Qt拖放/滚轮事件。当前Windows11、Qt/PyQt6.11.0、Fluent Widgets1.11.2，逻辑屏1536×864、DPR1.25；屏幕报告165.014Hz。四份512×768 float32合成预览不代表真实大数据。

有效基线native-geometry-baseline/report.json自然退出0：左右Panel分别96/95个中间maximumWidth采样，但content_visible均0；Card60个合理中间maximumHeight采样，但body_visible均0、实际height仅0/218，展开上限16777215。快速导航最终状态一致，BScan重排与分组资源集合完整。

8ms心跳间隔、QEvent.Paint/Resize计数、点击到目标页首次Paint延迟均为Qt事件代理指标，**不是屏幕合成帧率**。当前没有60fps/165fps结论，也没有100%/150%缩放、多屏、真实大数据或长时负载验收。

最终原生证据采用native-strict-before与native-strict-after：真实app_qt.main、相同探针SHA-256、窗口大小、小数据、DPR125%，修改前七模块由固定快照importhook加载，两轮串行。均main返回0、atexit执行、fatal扫描清净、failures为空；导航/滑块、快速Panel、Output、Ctrl+W/重开标签、滚轮缩放、settled标签往返range和drop资源集合断言通过。Card六个真实控件（Spin/Switch/Combo/Button）的中间帧与展开帧rect完全相同，实图确认自然布局不重叠、下方由viewport裁切。

| 同入口单轮心跳p95（ms） | before | after |
| --- | ---: | ---: |
| 快速导航 | 26.69 | 36.27 |
| 快速Output | 23.87 | 26.68 |
| 快速BScan标签 | 75.55 | 71.27 |

此前不同入口探索轮的BScan约44→84ms未在同入口重现；以上仍只是一对局部事件采样。导航/Output较差读数保留，不宣称整体提速或所有响应均无退化，不据此猜测并改写BScan渲染。真实屏幕帧时间、可重复性能分布与大数据响应仍属后续专项。

## 回归与生命周期证据

- 初始motion-red.log有9正式RED，panels-red.log有7正式RED，retarget-red.log有3正式RED；后续Card真实四行控件在card-clipping-red.log补1个正式RED，共20个故障反例。另补快速反向/重复目标/隐藏收敛、控件真实销毁与正常退出，总计27个新增参数化用例。
- Panel仅设置Ignored policy时，真实布局中途缩到38px的按钮宽；已以动画期间minimumWidth跟随maximumWidth并终态恢复解决，没有改变子控件固定尺寸。
- 中间版本native-after-final虽正常退出，但Card的mid-expand实图发现SpinBox与后续开关重叠、ComboBox压扁；仅body可见/height采样不足以验收。新增四行Spin/Check/Combo/Button测试正式失败后，局部viewport保持内部自然布局。card-clipping-green2.log的27项自然exit0；收/展中间帧内部行rect与展开态相同，无压缩或重叠。旧图保留作为失败证据，不作为最终通过图。
- 原关联首次出现motion lambda栈AV。父窗/子控件delete/close的12个独立Qt子进程均自然0；部分pytest组合才复现。禁GC仅用于诊断，未保留。进度条改为整数端点+Qt receiver slot，并为新测试正常派发DeferredDelete。
- 新生命周期子进程通过独立QObject observer确认六个在飞控件真实C++销毁、main返回和atexit；裸lambda销毁通知只有四次的初始断言不作为泄漏证明。未用isdeleted吞异常或os._exit。
- 另一处app.setStyleSheet AV由旧七模块隔离importhook在相同file_tree+mainwindow组合复现；不是after独有。Fusion单独仍失败，严格复刻app_qt的Fusion+空树阶段提前apply_theme后同组合88 passed、自然0。tests/conftest会话fixture现对齐这两步真实启动顺序，不跳断言、不降低门槛、不改产品主题源码。
- 初步修正启动后，既有262项完整关联在before七模块与after均计数全绿后退出AV（existing-before.log、existing-after.log），故不能归因本轮产品独有。全局测试fixture尝试正常close与DeferredDelete收尾，289关联末段在qframelesswindow.event删除路径Abort/AV（motion-final-cleanup.log）。该未奏效的全局清理已撤去；没有强退、sip硬删除、跳断言或禁GC。完整组合退出疑虑保留为测试运行顺序/生命周期限制，不声称已确定唯一根因。
- 初始pilot及中间关联日志包含退出阶段无Pythonframe AV，完整保留；不能凭“测试计数全绿”抹去非零退出。最终关联/原生后测结果待下文补充，不提前宣称完成。

## 最终验证

最终新增用例new-final.log为27 passed、3.18秒、自然exit0，包含在飞控件真正销毁子进程。静态ruff、编译624文件、架构、技术债164≤166通过，未调整预算或baseline。

因未解完整关联退出风险，追加普通全量pytest tests/（不是重新执行完整质量门禁），沿用既有正常测试顺序并记录faulthandler、pytest返回码与atexit。首轮full-final.log为1 failed、2009 passed、6 skipped，212.65秒，pytest返回1且atexit执行、无fatal。唯一失败为旧test_pill_slides_to_selected_chip从未show却要求动画端点；本轮隐藏控件即落位的行为符合合同。测试改为先显示、明确允许动画，仍验证动画终点并新增最终geometry断言；该文件与27新测试合计121 passed、9.14秒、自然exit0（contract-final.log）。没有为了旧测试恢复隐藏动画。

由这一明确失败修正触发第二次普通全量pytest：full-final-corrected.log为**2010 passed、6 skipped、206.84秒**，真实进程exit0、pytest返回0、atexit执行、fatal扫描清净。相比此前工业轮1983项增加27个新增用例。不覆盖首轮，也不把2009+1失败写成完整通过；全量顺序通过不抹去前述个别子集仍有既有退出AV且根因未定的限制。

| 验证项 | 最终状态 | 证据 |
| --- | --- | --- |
| 新增动效回归 | PASS：27项、exit0 | new-final.log |
| 修正的可见胶囊合同及新增回归 | PASS：121项、exit0 | contract-final.log |
| 修正后普通完整pytest | PASS：2010/6、exit0、正常atexit | full-final-corrected.log |
| 原生真实入口/视觉与最终状态 | PASS：严格before/after均exit0；Card正确裁切 | native-strict-before/、native-strict-after/、native-review.md |
| ruff、compile624、架构、技术债 | PASS；164≤166，预算未调整 | ruff-final.log、ruff-fixture-final.log、ruff-contract-final.log、compile-final.log、architecture-final.log、debt-final.log |
| 完整关联子集的退出生命周期 | PARTIAL；before/after既有262同样退出AV | existing-before.log、existing-after.log；失败清理尝试已撤去 |
| 可重复性能/屏幕帧率 | PARTIAL / NOT MEASURED | native-strict-comparison.json；单对照保留导航较差读数 |
| 收尾进程/源码 | PASS：自建测试进程0、654文件冻结前后哈希一致 | process-cleanup.json、workspace-final-recheck.log、verified-source-manifest.json |

最终源快照654文件：相较工业轮652文件仅七个产品模块、tests/conftest.py和现有test_processing_page_v2.py改变，新增两个动效测试文件，其余643文件哈希原样。HEAD与分支未变，TODO原patch未变；先前未提交成果全部保留。本轮没有commit/push/tag/build/release。

汇总机器可读证据为output/motion-maintenance-20261017/summary.json；源manifest SHA-256为0ef60e66772898fd1f09479082cfeae59b7e186acb130aec5686348549009a9d。保留主页面、OutputPanel和BScan即时切换；下一步如需进一步改善响应，应另做真实屏幕帧时间、不同DPI/大数据和可重复导航对照，而非继续添加装饰动画。
