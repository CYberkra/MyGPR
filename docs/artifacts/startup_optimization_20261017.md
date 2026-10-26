# 启动性能有限优化实验（2026-10-17）

**状态：`NO_ACCEPTED_CHANGE`。本轮未达到新增提速目标，最终保留原基线。** 两项候选均未获得足够稳定的全链路收益证据；不将正确性通过写成性能验收通过。

候选1让真实 `darkdetect` 在纯 Python worker 导入，与主线程原本必需的 Qt/绘图导入重叠，并在 Fluent/窗口创建前 join。真实重叠及收尾成立，但不足以证明整体效率提高。完成两个原生 Windows 场景、每侧每场景5个有效样本，共20个正式进程；初3是分块探索，末2按 B-C-C-B 交替确认，不能称全部5对时间配对。双方另有排除的正确性/缓存预热 pilot，未清 OS 缓存。

| 中位数，秒 | 基线 | 候选1 |
|---|---:|---:|
| 空启动：首个 Paint | 4.700 | 4.885 |
| 空启动：全部页面接线完成 | 6.547 | 6.400 |
| 恢复项目：首个 Paint | 4.549 | 4.552 |
| 恢复项目：数据可用 | 6.134 | 6.653 |
| 首次处理：点击运行至成果可用 | 2.486 | 2.825 |
| 启动至首次处理成果可用 | 8.791 | 9.766 |

末2交替的首 Paint 比较均偏向候选，与初3趋势矛盾；导入、窗口构造、首次处理及事件循环也有明显波动。因此拒收候选1，不宣称已经证明某个内部环节必然回退。原始样本、范围和分阶段指标见本地 `output/startup-optimization-20261017/candidate-01-comparison.json` 与 `benchmark-ledger.md`。

首屏刚可交互即点击真实设置页签；恢复数据到达且处理入口可用即点击真实处理页签，探针不先调用 `ensure_pages_ready`。另测就绪后二次导航。首次处理通过真实运行按钮、coordinator/controller/backend/job 执行固定128×64 float32项目的 `dewow(window=11)` 与 `trace_savgol_filter(window_traces=7, polyorder=2)`；`scipy.signal` 在首次提交前未加载、之后实际加载。恢复场景10个正式进程的完整成果矩阵 SHA256、采样头/时间距离深度轴和算法 lineage 等价；隔离路径、成果ID和保存时间保留原值并单列。这是工程路径验证，不是科学或真实数据验收。

候选2仅做缺失字体族过滤的四次微诊断，计入 QFontDatabase 枚举成本：原方案842/378 ms，过滤457/457 ms，其中枚举146/107 ms。实际字体、中文/ASCII/单位文本度量及 glyph 几何等价，但末次原方案更快，未实施产品字体修改。构造诊断中的 Mica 启用/禁用仅约0.30/0.12 ms，也未修改。首次字体匹配的诊断成本不能靠移到另一次首次操作来计为收益。

最终在恢复后的产品上新增执行3个独立原生 Windows 进程，每个显示 ComboBox/上下文菜单、切深浅主题、确认菜单销毁并普通关闭。**3/3实际 exit0、RETURN0、双 ATEXIT、app-exit日志齐，无 fatal/超时；自启PID均退出。** 没有传诊断 `--delete-on-close`，使用产品原有 `WA_DeleteOnClose`；main返回后主窗C++已销毁，QApplication仍存活。菜单私有Fusion样式所有权及原退出修复保留。

原833项冻结文件、此前657项源配置、原28个未跟踪测试逐文件hash全部保留；候选1仅恢复自己的入口修改并将新增helper/test留在拒收输出。分支、HEAD、依赖及用户工作区原有改动未改写。最终源码与原基线相同，按主代理裁决未重复完整suite；**2031 passed / 6 skipped属于此前维护轮证据，并非本轮新跑。** 本轮候选阶段的9/38项定向检查也不表示接受了新性能功能。

测量进程均使用独立JSON/INI/runtime/log/TEMP目录，不改HOME/USERPROFILE。原始、排除计时的无GUI fixture准备子进程遗漏显式环境隔离；其自身日志记录仅创建指定输出项目，源码审查未见该路径使用用户设置持久化，但没有据此声称完成动态用户文件写审计。未查改或恢复用户registry/文件；未来准备环境已修正，固定模板未重建。细节见 `fixture-preparation-audit.json`。首三批探针的观测追加、最终同探针hash亦在台账披露。

本轮计时包含统一探针启动开销，未测安装包或冷OS启动。完整本地证据保留在 `output/startup-optimization-20261017/`；最终机器状态为 `summary.json`，退出证据为 `final-native-summary.json`，保全证据为 `implementation-preservation-final.json`。未提交、推送或外部发布。
