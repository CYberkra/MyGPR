# MyGPR V0.8.70 截图自审与修复记录

## 发现的问题

1. 项目管理页测线预览不应出现整齐竖条。原因是该页面直接使用通用 CSV 矩阵读取器，遇到带经纬度/高程/高度/时间戳列的堆叠式 GPR CSV 时，会把辅助列当成二维图像显示。
2. 上一版大型演示数据的目标响应过于理想化，容易看起来像测试玩具而不是现场雷达图。
3. 成果报告截图只运行了成果检查，没有生成报告正文预览。
4. 截图目录存在重复编号文件，容易让用户误以为流程混乱。
5. 成果报告无问题时检查表为空白，应该给出明确通过状态。
6. 日志抽屉没有项目打开等基础事件，截图看起来像未运行流程。

## 修复

- `ui/workbench_window.py::open_line_document` 改为调用 `ProcessingSessionService.open_line(..., enforce_processing_gate=False)`，与测线处理页保持同一数据解析路径。
- `core/spatial_synthesis_service.py` 改为优先从 RTK/高度计辅助文件读取轨迹和高程，避免大型项目生成空间成果/成果报告时反复解析完整雷达主数据。
- 重新生成大型演示数据：保留 8 条测线与完整 RTK/IMU/高度/时间戳辅助文件，同时降低过饱和斜线、加入更自然的层状反射、双极性目标响应和局部噪声。
- 截图流程改为先生成成果报告，再截图报告正文与交付文件抽屉。
- 成果报告检查表在无问题时显示“成果检查通过”。
- 工作台日志增加项目打开、页面切换、测线预览、交付成果生成等基础事件，并显示 UTC 时间。
- 清理旧截图输出，只保留一套连续编号图片。

## 验证

- `python scripts/preflight_check.py`
- `python scripts/check_version_consistency.py --expected 0.8.70`
- `python -m compileall -q app_qt.py cli_batch.py core ui scripts tests PythonModule`
- 大型演示项目真实 GUI 截图重跑。
