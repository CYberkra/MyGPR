# 导航响应与 Qt 正常退出专项（2026-10-17）

本轮消除了导航时自动列宽调整引起的同步写盘，修复了菜单私有样式的生命周期，并加固真实入口的主窗关闭次序。最终完整测试与正常退出验证通过；**整体导航延迟尚未稳定改善，不宣称整体提速或屏幕帧率**。

分支 `codex/maintenance-20261017`，保留此前未提交成果和 TODO 原差异；没有提交、推送、打包或发布，不改整体视觉、科学处理、数据语义或异步身份守卫。

## 变更范围

| 产品文件 | 最终行为 |
| --- | --- |
| `ui/pages/project_page.py`、`ui/pages/project_preferences.py` | 保留原页面方法和header信号；自动Stretch/Fixed调整不保存，Interactive拖列即时保存；同值去重、失败重试，迁移成功后统一清理pending旧键 |
| `ui/theme_helpers.py` | 在现有兼容入口捕获RoundMenu实际factory Fusion base样式，仅把私有无owner样式归菜单所有；共享样式、构造异常finally与重复patch保持正确 |
| `app_qt.py` | 完整构造后、show前设置WA_DeleteOnClose；接受关闭时在QApplication仍存活时销毁主窗，拒绝关闭保留窗口，原脏数据确认/保存/后端收尾顺序不变 |

新增回归为 `tests/test_project_navigation_persistence.py` 的8项实际页面用例，以及 `tests/test_qt_shutdown_lifecycle.py` 的13项生命周期用例。保留原有测试，不用跳过断言、强退或禁GC替代正常退出。

## 导航证据与性能边界

三个独立before进程与三个after进程使用同一真实app_qt.main、探针SHA-256、1413×750窗口、125%DPI和小合成项目，串行测量。每轮21次常规点击（20次实际切页）、49次快速切页、7次重复当前页（零切页），并验证B-scan滚轮缩放后导航往返的视口。点击回调、目标页首次Qt Paint事件和8ms心跳是事件代理指标，均不是合成器FPS。

| 三轮percentile中位，快速导航（ms） | before p50 | after p50 | before p95 | after p95 |
| --- | ---: | ---: | ---: | ---: |
| 点击回调 | 5.93 | 12.84 | 22.52 | 30.65 |
| 首次Paint事件 | 9.11 | 20.29 | 30.18 | 37.14 |
| 心跳间隔 | 8.40 | 12.69 | 27.97 | 41.46 |

确证P2：49次快速切页触发14次自动列宽保存，合计43.62ms；文件树没有重建，导航门控刷新很小。单进程ABBA仅在隔离探针恢复旧自动保存，四块各21次同样导航：保存次数6→0→0→6，项目切换inclusive p50为19.39→4.34→4.34→17.99ms，确认修复消除了真实写盘成本。三轮总体p95仍不稳定，不能把局部因果贡献写成整体提速，亦未证实归因系统负载。

计时对应保留的导航阶段源码，早于迁移失败分支修正、职责提取和后续Qt修复；最终树没有重新测速。提取后两个信号处理器和自动布局提前返回逻辑保持同语义，并通过相同8项实际页面回归。只覆盖本机125%DPI、小合成数据，不代替其他DPI、大数据或科学验证。

## 两类 Qt 退出证据

菜单缺陷有精确工作流复现和所有权单变量证据：原262子集计数全绿后仍以0xC0000005退出，缩到3个菜单案例仍失败；给后期QSS代理加parent不能解决，捕获真实Fusion base并归菜单所有后原工作流正常退出。正式5项RED、最终菜单/入口13项回归覆盖共享/已有owner/None样式、异常finally、幂等与真实进程return/退出标记。原262子集连续3次、289关联均正常退出。

后续真实入口第三轮又出现main RETURN0后未到达我们注册退出标记的AV，未取得可明确归因的精确原生栈，不能称已证明唯一内部崩溃原因。结构性入口回归另有2项RED：全部页面就绪后关闭返回时C++主窗仍存活；早关原本通过。WA_DeleteOnClose闭合该寿命次序缺口，拒绝关闭/再接受、初始化线程收尾、main返回前C++销毁且QApp存活均受验证。最终3轮原生流程显示ComboBox与菜单、切深浅主题、确认菜单销毁并正常关闭，均实际exit0。动态windowsvista样式仅为未确证候选，未扩改。

## 最终验证

| 项目 | 结果与证据 |
| --- | --- |
| 完整pytest（含后端coverage） | **2031 passed、6 skipped、261.56秒**；实际child0、runner0、RETURN0、ATEXIT、无fatal；`full-final.log` |
| 提取后导航定向 | **19 passed、2.46秒**；直接子码0、RETURN0+ATEXIT；`navigation-extracted-green.log` |
| 新生命周期及启动关联 | 13项生命周期+10项既有启动，共23项通过；真实入口取消/接受关闭与线程收尾均检查 |
| 原生Windows最终退出 | 3/3实际exit0，菜单真实销毁、主窗C++已释放；`native-final-results.json` |
| 前置门禁 | 12项静态/接口/架构/编译/预算/smoke检查全部实际exit0，编译627文件；`quality-process-results.json` |
| 覆盖率与后置检查 | 后端覆盖率策略通过；后端HEAD→worktree含untracked变更行覆盖462/493=93.71%，并非Qt覆盖率；advisory、源码核对、排除原TODO差异的diffcheck均0；`post-quality-results.json` |
| 进程收尾 | 20个自启调试/测试PID全部退出，本轮Python残留0；`process-cleanup.json` |

初次debt门禁发现Page1024行导致modules>1000为3超预算2；提取为Page971行/helper76行后恢复2/2，宽异常164/limit166，预算未改，也未靠删空行/注释凑阈值。定向阶段曾只有PowerShell外层码，最终19直接子码及完整suite已补齐该退出证据，不覆盖原日志。

冻结657个源文件：此前motion654中651字节不变，仅app_qt/theme_helpers/project_page修改，新增helper和两个测试文件。旧 `tests/test_qt_shutdown.py` 已恢复此前字节，SHA-256为 `146efdf16c56cc46a8c2ca7ea1de4c1d6ab74df6b41fca9f6dea70d4308964bd`。最终源manifest SHA-256为 `50596ffdc778c582ac0b44bee38fb7ed836b07cdb8da77d393c3900aac11d34a`；TODO原差异保留。

详细复现、全部六轮读数、阶段失败与误诊控制、早期legacy设置隔离边界、源码/进程证据集中在 [导航审计](../../output/navigation-shutdown-20261017/navigation-audit.md)、[Qt审计](../../output/navigation-shutdown-20261017/qt-audit.md)及[独立交叉审查](../../output/navigation-shutdown-20261017/qt-cross-review.md)。早期导航明确隔离JSON设置但未预先重定向旧QSettings；最终入口与最终定向回归已使用临时INI，此前是否迁移旧键没有审计证明，不声称完整用户profile隔离。
