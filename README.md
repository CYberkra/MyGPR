# MyGPR 勘探定位工作台

当前版本：**v0.9.24 beta**

MyGPR 面向实际 GPR / UAV-GPR 勘探项目，核心流程是：项目建档、测线导入、数据检查、测线处理、目标标注、空间定位和成果报告输出。

默认主界面只保留现场工作需要的五个页面：

```text
项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
```

## v0.9.24 beta 重点

本版撤回了不够自然的“处理模板”方向，改为更贴近现场习惯的“连续手动处理”：

- 在测线处理页点击一次“执行当前步骤”，就按当前算法和参数，在当前结果上继续处理一次。
- 支持“撤回一步”“重置到原始”“前后对比”“保存当前结果”，避免为了多步处理再引入独立模板页面。
- 连续处理历史会显示在底部“处理历史”页签内，保存后会把完整链路写入处理 manifest。

本版在 v0.9.19 项目导航精简基础上，重点补齐模块联动闭环：

- 新增项目事件模型、依赖规则和 `metadata/project_state.json`，记录当前测线、dirty/stale 状态和最近事件。
- 处理“导入测线 / 导入 RTK / 运行质检 / 修正 B-scan / 保存处理结果 / 目标变化 / 空间刷新 / 报告生成”等操作时，自动更新空间成果与报告失效状态。
- 空间成果页显示“空间成果需刷新”提示，成果报告页显示“成果报告需重新生成”提示和具体原因。
- 项目树节点增强为导航入口：处理结果、目标标注、空间成果和报告节点可跳转到对应模块并同步当前测线。

既有能力包括：

- 正式项目创建、打开、最近项目、项目设置。
- 单条测线导入、后台批量导入、导入预检和坏文件诊断。
- 数据质检、B-scan 方向风险检测和方向修正。
- 经纬度到 CGCS2000 / 3-degree GK 工程坐标投影。
- 测线清单 CSV 导出和项目 ZIP 备份。
- 空间成果工具栏闭环：刷新、坐标导出、平面图、三维视图、图层控制。
- 三维成果窗口：三维轨迹、目标点、平面图、数据汇总、PNG 和三维点云 CSV 导出。
- 成果报告包：HTML、JSON、CSV 和 PDF 报告。
- 15.6 寸 1080P 笔记本适配：读取 Qt `availableGeometry()`，自动避开 Windows 任务栏并进入 compact mode。
- 自适应布局参数系统：统一计算主图高度、右侧栏宽度和底部区高度，减少不同 DPI / 任务栏环境下的页面压缩差异。
- 右侧辅助栏折叠：项目操作、处理参数、空间辅助和报告导出栏可收起，释放主工作区。
- 主图放大查看：B-scan、空间主图、轨迹/DEM 等图卡可弹出独立大图窗口查看。
- 原始来源文件追踪：导入时记录外部源文件路径、大小、mtime、hash 和项目内副本路径。
- 项目管理页新增源文件状态列、源文件检查、重新定位和来源清单导出。
- 删除项目增加预检摘要；删除后清理失效最近项目并回到未打开项目状态。

完整说明见：

```text
docs/user/manual_v0.9.13_beta.md
docs/developer/beta_boundary_v0.9.13.md
docs/audit/button_callback_audit_v0.9.13.md
docs/audit/report_export_closure_v0.9.13.md
```

## Windows 快速使用

首次使用建议按下面顺序操作：

1. 解压整个 ZIP，不要只拖出单个文件。
2. 双击 `安装MyGPR环境.bat`，创建本包专用 `.venv` 环境。
3. 双击 `启动MyGPR.bat` 启动软件。
4. 如果启动失败，双击 `检查MyGPR环境.bat`，再查看日志提示。

可用脚本：

```text
安装MyGPR环境.bat       安装本包运行环境
启动MyGPR.bat           启动主界面
启动MyGPR_调试日志.bat  启动并输出更详细日志
检查MyGPR环境.bat       只检查环境，不启动界面
```

对应英文脚本也保留：

```text
install_mygpr_environment.bat
start_mygpr.bat
start_mygpr_debug.bat
check_mygpr_environment.bat
```

## 主界面说明

### 1. 项目管理

用于新建项目、打开项目、导入测线、批量导入、导入 RTK / IMU、运行数据质检、修正 B-scan 方向、导出测线清单和项目备份。

### 2. 测线处理

用于查看 B-scan、选择处理方法、调整参数、运行处理流程、保存处理结果。

### 3. 目标定位

用于把测线中的疑似目标、异常范围、界面线等内容记录为结构化标注。当前自动识别是启发式辅助检测，不是深度学习模型。

### 4. 空间成果

用于查看测线轨迹、目标标注在空间中的位置关系，以及地形和飞行高度信息。工具栏已接入刷新、坐标导出、平面图、三维视图和图层控制。

### 5. 成果报告

用于交付前检查，并生成项目报告和成果文件。当前报告包包含 HTML、JSON、CSV 和 PDF；Excel 报告仍在下一轮 TODO 中。

## 数据与辅助文件

MyGPR 可读取常见 GPR / CSV 数据，也支持按项目匹配辅助文件：

```text
RTK              用于空间定位
IMU              用于姿态与运动补偿
高度计           用于飞行高度和地形关系
逐道时间戳       用于传感器同步检查
```

正式导入入口当前支持 CSV / TXT / NPY / NPZ / H5 / HDF5。DZT / RD3 / DT1 / OKO 等厂商格式必须继续按专项测试接入，不能只凭扩展名宣传完整解码。

## 1080P 适配验证

在 Windows 真机上运行：

```bat
python scripts\capture_field_workbench_windows_diagnostics.py --output windows_fit_check
```

输出目录会包含 6 张主页面截图和 `screen_diagnostics.json`。JSON 会记录完整屏幕、可用屏幕、DPI、捕获尺寸和 compact mode 状态。

## Python 运行环境

如果需要手动安装依赖：

```bash
python -m pip install -r requirements.txt
```

开发或测试时再安装：

```bash
python -m pip install -r requirements-dev.txt
```

主要运行依赖包括 PyQt6、qfluentwidgets、NumPy、Pandas、SciPy、Matplotlib、h5py、PyYAML、PyWavelets 和 pyproj。

## 命令行批处理

验证批处理配置：

```bash
python cli_batch.py validate --config config/cli_batch_mvp_example.json
```

运行批处理：

```bash
python cli_batch.py run --config config/cli_batch_mvp_example.json
```

重跑上次失败项：

```bash
python cli_batch.py resume --summary <summary.json>
```

## 开发人员说明

正常现场使用不需要打开研发工具。历史研发、仿真和验证工具默认不出现在主界面中；需要复现旧流程时，可在启动前设置：

```bat
set MYGPR_ENABLE_RESEARCH_UI=1
start_mygpr.bat
```

发布前建议运行：

```bash
python -m compileall . -q
python scripts/check_version_consistency.py --expected 0.9.24
python -m pytest tests/test_workbench_1080p_fit.py tests/test_capture_summary.py tests/test_field_project_store.py tests/test_field_project_operations.py tests/test_report_export_v098.py tests/test_version_consistency.py tests/test_field_workbench_boundaries.py tests/test_field_processing_bridge.py tests/test_project_status_metrics.py -q
```

## 版本

当前版本由根目录 `VERSION` 文件记录，并显示在软件标题、启动器和状态区域。
