# MyGPR Windows 启动基线（2026-11-19）

本轮在设置页首窗构造阶段保留 B-scan 分类和更新服务所需控件。其余分类在首次点击或搜索定位时构造，之后复用。设置值与搜索索引在首窗前可读。相同主题无需重复应用设置页样式。

基线为 MyGPR `main` 的 `e73198accbf1fbe89f8aeaa5865f2f42eaffa29c`。同一台 Windows 机器、同一个 Python 3.12.0 环境，每个场景运行四对，顺序为 AB、BA、AB、BA。每次用独立配置、运行目录及日志启动真实窗口。恢复场景每次复制同一个合成项目模板。系统缓存未清空。以下时间包含探针自身启动开销，测量对象为源码启动。

| 场景 | 指标 | 基线中位数 | 候选中位数 | 候选更快的对数 |
| --- | --- | ---: | ---: | ---: |
| 空启动 | 设置页注册 | 237 ms | 120 ms | 3/4 |
| 空启动 | 进程启动至主窗首绘 | 6.408 s | 5.948 s | 3/4 |
| 空启动 | 第一次点击设置至设置页首绘 | 197 ms | 211 ms | 2/4 |
| 恢复项目 | 设置页注册 | 295 ms | 122 ms | 4/4 |
| 恢复项目 | 进程启动至主窗首绘 | 8.284 s | 6.166 s | 4/4 |
| 恢复项目 | 第一次点击设置至设置页首绘 | 251 ms | 206 ms | 3/4 |
| 恢复项目 | 进程启动至测线预览可用 | 11.760 s | 9.029 s | 4/4 |
| 恢复项目 | 点击运行至首份成果可读 | 5.052 s | 5.168 s | 1/4 |
| 恢复项目 | 进程启动至首份成果可读 | 17.048 s | 14.276 s | 2/4 |

设置页注册在八对中的七对缩短。空启动第四对注册差值为 +1 ms，主窗首绘差值为 +1.172 s。恢复场景的导入阶段也有明显波动，不能把主窗首绘的中位数差额全归因于设置页。首次点击设置和首次处理仍应与首绘一并记录。Windows 安装包性能需要用实际构建测量。

恢复场景经 GUI 运行 `dewow(window=11)` 和 `trace_savgol_filter(window_traces=7, polyorder=2)`。八次恢复都显示了处理成果，并重新读取了 128×64 的有限 float32 矩阵。SHA-256 均为 `c91538e9330dbf59d22ec03a1e0341e71808aff823c86356cd6ec774aecac6eb`。两场景全部 16 次主窗口接受普通关闭事件，进程退出码和应用退出日志均为 0。

额外的 Windows 实窗回归通过搜索定位、偏好保存及重启回读、B-scan 与 AGC 实时预览。深色启动探针确认默认 B-scan 已就绪，首次搜索「应用更新」能打开该分类并聚焦，后台绑定连接的是同一更新面板。开发环境的更新状态可见。

## 复跑

在 Windows 交互桌面上，用同一 Python 环境和两份源码运行。将 `$baseline` 指向已知基线源码目录，使用新的结果目录。

```powershell
python scripts/benchmark_windows_startup.py --baseline $baseline --candidate . --output output/startup-empty --scenario empty --pairs 4
python scripts/benchmark_windows_startup.py --baseline $baseline --candidate . --output output/startup-restore --scenario restore --pairs 4
```

比较两个场景各自 `summary.json` 的中位数与成对差值。对首绘、第一次进入设置、恢复后首份成果或正常退出有异常的样本，先查看该次运行的 `console.log` 和 `logs/`。界面改动涉及启动时，在合并前用相同基线复跑两种场景，再判断耗时是否从启动转移到了第一次操作。
