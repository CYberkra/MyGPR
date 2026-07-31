# MyGPR v0.9.26 环境配置与验证报告

- 验证日期：2026-07-13
- 验证构建：MyGPR v0.9.26 field production refactor
- 验证平台：Linux x86_64，glibc 2.41
- Python：3.13.5
- 图形模式：Qt `offscreen` 无头模式
- 结论：当前源码、依赖环境、核心业务链和无头 GUI 启动均通过验证，可进入 Windows 实机与真实勘探数据验收阶段。

## 1. 已配置环境

已在项目目录建立隔离虚拟环境 `.venv`。主要版本：

| 组件 | 版本 |
|---|---:|
| NumPy | 2.5.1 |
| pandas | 2.3.3 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.0 |
| h5py | 3.16.0 |
| PyYAML | 6.0.3 |
| PyWavelets | 1.9.0 |
| PyQt6 | 6.11.0 |
| PyQt6-Fluent-Widgets | 1.11.2 |
| pyproj | 3.7.2 |
| rasterio | 1.5.0 |
| Fiona | 1.10.1 |
| openpyxl | 3.1.5 |
| pytest | 8.4.2 |
| PyVista | 0.48.4 |
| VTK | 9.6.2 |

验证结果：

- `scripts/check_env.py --strict`：通过；
- `pip check`：无依赖冲突；
- 全项目 `compileall`：通过；
- 必需工程目录、日志目录写入权限：通过。

## 2. 自动化验证结果

### 2.1 完整非慢速门禁

项目测试必须按模块隔离执行，因为 Qt、Matplotlib、VTK 和若干 C 扩展存在进程级全局状态。按项目正式隔离门禁执行后：

- 非慢速、非 integration 测试：**934/934 通过**；
- 断言失败：0；
- Python 测试错误：0。

四路并行运行时，`test_bscan_interaction_controller_gui.py` 曾因同机 Qt 原生资源争用以 `SIGABRT` 退出；该文件单独进程执行为 **3/3 通过**。其余第四分片测试为 **233/233 通过**。该现象不属于业务断言失败，也不影响项目既定的“一测试模块一进程”发布门禁。

### 2.2 业务闭环与 integration 验证

额外执行以下现场业务链：

- 正式项目创建、打开与测线导入；
- 雷达—RTK—IMU/飞高同步；
- sidecar 数据合并；
- GeoJSON、GeoTIFF/DEM 导入和事务回滚；
- GIS 正式平面图输出与取消保护；
- 报告、质检快照与证据 sidecar；
- 正式 PDF、HTML、Excel、审计与校验清单；
- 营山测线命名、坐标推断与批量导入；
- 运动补偿端到端输出。

共执行 28 项，其中 **14 项为 integration 标记测试**，全部通过。其余 14 项已包含于 934 项门禁中。

因此本次验证覆盖的唯一测试总数为：**948 项通过**。

### 2.3 v0.9.26 专项验证

针对本轮五类重构执行专项测试：

- Job Manager；
- 大文件轻量预检；
- 传感器同步；
- GIS 图层与地图输出；
- 标注视窗；
- 空间成果后台导出；
- 正式行业报告；
- 版本一致性。

结果：**23/23 通过**。

### 2.4 GUI 与布局验证

- `FieldWorkbenchWindow` 在 Qt offscreen 模式成功创建；
- 1600×900 窗口进入事件循环并正常退出，返回码 0；
- B-scan 交互控制器 GUI：**3/3 通过**；
- 紧凑 1080p 工作台布局：**4/4 通过**。

## 3. 验证期间发现并修复的问题

1. **紧凑 1080p 页面溢出**：标注页和空间页存在垂直滚动；已调整证据区和空间控制区尺寸策略。
2. **主窗口职责膨胀**：Job Manager UI 动作从主窗口拆分至 `JobActionsMixin`，主窗口重新满足模块行数预算。
3. **导入入口同步异常缺少保护**：增加统一错误捕获和操作错误呈现。
4. **旧批量导入测试仍绑定专用弹窗**：改为验证统一 Job Manager 契约、取消和批量任务键。
5. **Qt 测试 QApplication 生命周期不稳定**：测试保持 QApplication 强引用。
6. **GeoTIFF 单波段读取与 NumPy 2.5 兼容警告**：改为显式单波段数组读取。
7. **Linux 报告图件中文字体未自动注册**：新增无 Qt 依赖的核心 Matplotlib CJK 字体策略；GIS 与正式报告图件复测后不再出现中文缺字警告。

## 4. 本次验证构建相对原始 v0.9.26 的代码变更

修改：

- `core/field_report_export.py`
- `core/gis_layers.py`
- `core/gis_map_export.py`
- `tests/test_background_batch_import_v096.py`
- `tests/test_bscan_interaction_controller_gui.py`
- `ui/field_panels/interpretation_page.py`
- `ui/field_panels/project_page.py`
- `ui/field_panels/spatial_page.py`
- `ui/field_workbench_window.py`

新增：

- `core/plot_font_policy.py`
- `ui/field_panels/job_actions.py`
- `VERIFICATION_REPORT_2026-07-13.md`

## 5. 尚未覆盖的现场验收边界

以下项目不能由当前 Linux 无头容器替代：

- Windows 10/11 原生显示与 125%、150%、175% DPI 人工视觉验收；
- RTX GPU、真实磁盘吞吐和多 GB/数十 GB 测线压力测试；
- 真实厂商雷达文件的完整批次兼容性；
- 真实 RTK、IMU、飞高设备文件和触发/PPS 偏移标定；
- 外接硬盘断连、低磁盘空间、休眠、异常断电恢复；
- 现场打印机、PDF 阅读器和 Excel 交付兼容性；
- Windows 安装包、代码签名和升级/回滚。

此外，`PyQt6-Fluent-Widgets 1.11.2` 内部仍使用 SciPy 已弃用导入路径，会产生一条第三方 `DeprecationWarning`；当前不影响运行，属于上游依赖问题。

## 6. 放行判断

本构建达到：

> 依赖可复现、源码可编译、核心功能可执行、无头 GUI 可启动、主要业务闭环和自动化回归通过。

当前可放行至：

> **Windows 实机 + 真实雷达/RTK/IMU 数据 + 多 GB 压力测试的现场验收阶段。**

在完成上述实机验收前，不应将本次 Linux 无头验证等同于最终现场认证。
