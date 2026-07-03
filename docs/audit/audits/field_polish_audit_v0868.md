# MyGPR V0.8.68 现场产品化文案与可理解性审计

## 审计目标

本轮审计不是继续堆功能，而是检查软件是否像一个可以给勘探现场人员使用的工程软件：

- 默认界面是否围绕实际勘探流程，而不是科研验证流程。
- 按钮、标签、空状态、错误提示是否能让用户知道下一步该做什么。
- 是否还有明显的 AI/研发内测/论文实验式表述暴露在正式界面。
- 成果导出是否更像工程报告，而不是算法证据包。

## 审计范围

机器审计与人工复核覆盖：

- 根目录 README、启动说明、CHANGELOG。
- Workbench 主窗口、五个默认页面、底部抽屉和工程树。
- 旧版处理窗口、参数推荐页、目标标注页、空间成果页、成果报告页。
- 运行时错误提示、处理报告、导出报告、预检脚本。
- 默认正式模式与研发模式开关。

静态统计：

```text
可审计文本/源码文件：736 个
可审计文本/源码行数：168,039 行
Python 文件：390 个
Python 行数：123,664 行
Python AST 语法错误：0
```

## 本轮实际改动

### 1. 默认主流程固定为现场工程流程

正式模式默认只显示：

```text
项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
```

并用 offscreen Qt 脚本确认：

```text
field_ui_labels_ok: 项目管理 -> 测线处理 -> 目标定位 -> 空间成果 -> 成果报告
bottom: 任务 / 检查提示 / 交付文件 / 日志
```

默认界面不再显示“仿真验证”。历史 gprMax / 研究验证能力仍保留在研发开关之后。

### 2. 去掉正式界面中的研发/AI 风格表述

面向用户的文案统一改为工程现场语言：

| 原表述 | 新表述 |
|---|---|
| AutoTune / 自动选参 / 自动调参 | 参数推荐 / 自动推荐 |
| ROI | 关注范围 |
| Evidence / 证据 | 交付文件 / 处理记录 / 对比报告 |
| 解释对象 / 解释层 | 目标标注 |
| 成果包 | 交付成果 / 成果报告 |
| 工作空间 | 页面 |
| 上下文检查器 | 当前信息 |
| 经典处理 | 旧版处理窗口 |
| 真值标签 | 人工确认标注 |

### 3. 成果报告更像工程交付文件

旧版报告标题和内容偏向“Evidence Report”。本轮改成：

```text
MyGPR 项目报告
```

报告内容改为说明：

- 当前测线处理图像。
- 处理参数。
- 检查提示。
- 适用边界。
- 处理流程文件、运行日志、关注范围文件、图像清单等交付文件。

报告目录名从：

```text
MyGPR_Evidence_Report_*
```

调整为：

```text
MyGPR_Project_Report_*
```

### 4. README 改成现场用户指南

根目录 README 已重写为中文现场用户说明，重点是：

1. 解压 ZIP。
2. 安装本地环境。
3. 启动软件。
4. 新建/打开项目。
5. 导入测线。
6. 处理、标注、空间成果、报告导出。

历史审计文档和补丁说明已移入：

```text
docs/audits/history/
```

避免 ZIP 根目录看起来像研发临时目录。

### 5. 保留研发功能，但默认不打扰现场用户

以下功能没有硬删除：

- gprMax campaign / 仿真验证。
- 研究验证控制台。
- benchmark / 内部验证工具。
- 旧测试和兼容代码中的 AutoTune 命名。

它们仍可通过研发开关打开：

```bat
set MYGPR_ENABLE_RESEARCH_UI=1
start_mygpr.bat
```

或：

```bat
set MYGPR_PRODUCT_MODE=research
start_mygpr.bat
```

这样既不损失历史研发能力，也不会让正式用户一上来看到不明所以的科研功能。

## 验证结果

基础验证：

```text
python scripts/preflight_check.py                         通过
python scripts/check_version_consistency.py --expected 0.8.68  通过
python -m compileall -q app_qt.py cli_batch.py core ui scripts tests PythonModule  通过
```

重点 UI / 报告 / 交付回归：

```text
tests/test_version_consistency.py
tests/test_daily_processing_smoke.py
tests/test_workbench_ui.py
tests/test_processing_lab_ui.py
tests/test_delivery_page_ui.py
tests/test_delivery_service.py
tests/test_interpretation_workbench_ui.py
tests/test_spatial_synthesis_ui.py
tests/test_bscan_display_export.py

29 passed, 1 skipped
```

报告和参数推荐相关回归：

```text
tests/test_import_export_report.py
tests/test_autotune_recipe_ui.py
tests/test_auto_tune_result_dialog.py

14 passed
```

工程核心服务回归：

```text
tests/test_product_mode.py
tests/test_workbench_entry_and_bridge.py
tests/test_workbench_project_core.py
tests/test_processing_session_service.py
tests/test_delivery_service.py
tests/test_spatial_synthesis_service.py
tests/test_interpretation_service.py
tests/test_gpr_io_airborne_contract.py

48 passed
```

说明：跳过的 1 个测试是 Linux/offscreen 沙盒缺少 Windows CJK 字体 fallback，不是功能失败。

我没有声明全量测试一次性通过；这个环境下大型 Qt/Matplotlib 全量单进程测试仍可能不稳定，所以本轮按产品改动范围做了分组回归。

## 审计结论

V0.8.68 默认界面已经从“像研发/科研平台”进一步收敛为“实际勘探定位工作台”。用户打开软件后看到的是项目、测线、目标、空间和报告，而不是仿真、科研、证据、benchmark 或内部算法名。

研发能力仍保留，但被放到显式开关之后。正式用户默认不会被这些功能干扰。
