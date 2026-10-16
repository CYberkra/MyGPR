# MyGPR 工作台改版 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将已确认的项目融合、可吸附 B-scan 标签、结果续算/分支、右键规范与统一视觉规范落实到 Qt 软件。

**Architecture:** 保留 UI → controller/JobBridge → application 的调用链。项目工作区复用项目控制器；B-scan 分组放入独立 Qt widget；每步直接来源采用显式版本化 metadata，兼容旧 parent_artifact_id 语义。显示设置与算法数据隔离。

**Tech Stack:** Python 3.12、PyQt6、qfluentwidgets、pyqtgraph、pytest。

## Binding spec and integration constraints

用户已授权开始实施。设计以本轮 `output/ui-redesign-20261016/branch-lineage-v4.md`、`context-interaction-v3.md`、`compact-controls-v5.md`、`project-workspace-v6.md` 为依据；这些忽略目录文件将在实施结束后以精简现行合同纳入规格索引。

当前功能分支 `codex/maintenance-20261016` 存在先前维护改动，保留并集成，不 reset、不 stash、不自动提交/推送。开始前保存工作区基线到 `output/ui-redesign-20261016/implementation-baseline/`。不变更依赖、算法、质量门禁或采集/传输能力。长期任务仍走现有控制器；只保存真实成果才可续算。

## Task 1: 项目工作区与视觉规范

**Files:** `ui/main_window.py`, `ui/pages/project_page.py`, `ui/pages/home_page.py`, `ui/coordinator_project.py`, `ui/design_tokens.py`, `ui/theme_helpers.py`, `ui/constants.py`，及对应项目/主窗口测试。

- [x] 先增加行为测试：顶部只有项目入口；无项目显示打开/新建/最近；空项目只引导导入；有项目显示台账；旧 home 路由兼容导向项目。
- [x] 用实际 Qt 控件验证启动与工作状态互斥，检查打开/新建/关闭信号仍到原控制器。
- [x] 复用原导入、预检、传感器同步与成果删除能力；将低频表单收进明确工具入口，避免满页卡片与重复入口。
- [x] 统一中性底色、字号层级、间距、图标按钮与选中/禁用/错误状态；明暗主题均可读。
- [x] 运行项目、主页兼容、标题栏、启动恢复与控件接线测试。

## Task 2: 数据来源的直接依赖

**Files:** `mygpr/application/project/processing_service.py`、相应存储/删除依赖模块与项目处理测试。

- [x] 用实际临时项目测试从中间成果续算两步，确认旧成果不覆盖。
- [x] 保留旧 `parent_artifact_id` 的运行来源语义，新增 parameters 中 `lineage_version=2`、`run_input_artifact_id`、`step_input_artifact_id`、`run_step_index`（含最终输出）与方法/参数可追溯信息。
- [x] 第一输出直接依赖提交输入，后续输出直接依赖前一真实保存输出；块执行只持久化最终矩阵时不得生成不可读的假成果。
- [x] 删除依赖闭包优先识别版本化直接依赖，兼容旧记录；测试删除一条分支不误伤兄弟。
- [x] 运行项目处理/依赖/存储合同测试，不修改算法输出。

## Task 3: B-scan 标签分组 widget

**Files:** 新建 `ui/widgets/bscan_workspace.py`, `tests/test_bscan_workspace.py`。

- [x] 测试默认单组、原始输入可关闭、关闭全部保持空组、多标签不自动拆分。
- [x] 实现组内拖动排序、跨组移动、边缘拆分、可拖 QSplitter、Esc 取消、有效落点提示与点击菜单等价操作。
- [x] 明确接口 `set_sources(list[dict])`, `open_source(key, group_id=None)`, `close_source(key)`, `active_key()`, `views()`, `all_views()`, `primary_view()`, `save_layout()`, `restore_layout(state)`；数据源 dict 包含 key/title/bundle/artifact_id。
- [x] 信号 `active_changed(str)`, `source_closed(str)`, `preview_requested(str)`, `source_context_requested(str, object)`；preview_requested 参数为 source key。控件自己 set_bundle/clear，按可见组请求缺失 bundle；异步更新不激活目标。
- [x] 标签右键提供固定、关闭范围、拆分、移动、重新打开、来源详情；业务命令经 source_context_requested 委托处理页。
- [x] 新 BScanView 经 view_created(object) 通知宿主接入全局偏好；已有显示参数按组独立，数据矩阵不复制。
- [x] 运行 Qt widget 行为测试并检查空白、错误/未加载状态及主题。

## Task 4: 处理页、草稿、项目树与菜单集成

**Files:** `ui/pages/processing_page.py`, `ui/coordinator_processing.py`, `ui/file_tree.py`, `ui/widgets/file_tree_panel.py`, `ui/controllers/processing_controller.py`，对应处理/树/竞态测试。

- [x] RED：选中成果后 current_pipeline()['input_artifact_id'] 等于该成果；A/B 草稿切换恢复，原始标签关闭仍可续算。
- [x] 用新工作区替换结果网格、结果墙与独立输入选择；移除底部结果库和全部步骤开关，运行区清楚显示下次输入。
- [x] 左栏承担新增处理步骤与参数；历史结果只读。“修改此步参数并另建分支”定位父结果并载入单步参数；草稿替换具体确认并可恢复。
- [x] 提交冻结输入与草稿；运行输入草稿锁定，其他结果可配置；成功只清提交源草稿，取消/失败保留；切标签/页面后完成不抢焦点。
- [x] 项目树按真实依赖展示原始输入及后代，运行号为辅助信息；旧来源未确定的成果明确归类，不编造边。
- [x] 在现有工作区标题栏放项目资源/处理工具小图标，保持返回入口；右键不隐式切换处理输入；显示变化不产生数据分支。
- [x] 所有新长任务、删除、导出复用现有真实信号链；无可用实现的动作明确禁用及原因。
- [x] 执行处理页、预览 generation、树、控制器等回归。

## Task 5: 集成审查与验收

- [x] 检查所有设计条目是否存在落点与测试，保留此前维护测试，不靠改期待值掩盖行为缺陷。
- [x] 运行 `python scripts/check_architecture.py`、`python scripts/check_python_compile.py`、相关 pytest，再运行 `python -m pytest tests/ -q`；记录环境阻碍及任何失败。
- [x] Qt 离屏 smoke 截图检查项目/处理/菜单明暗主题；Windows 原生独立临时项目完成导入→选方法→运行→切结果续算→查看分支→关闭。
- [x] 按 skills 要求进行独立最终审查，修复实质缺陷；记录验证边界，不把截图当科学验证。
- [x] 精简当前规格、实施证据和残余项写入 docs/artifacts 与规格索引。未完成项继续推进，不能只完成子模块就宣称全任务结束。

## Assertions for integration

```python
assert page.current_pipeline()['input_artifact_id'] == selected_artifact_id
assert workspace.active_key() == 'artifact:' + selected_artifact_id
assert new_step2.parameters['step_input_artifact_id'] == new_step1.artifact_id
assert old_sibling.artifact_id not in descendants_of_new_branch
assert len(workspace.views()) == 1  # opening more tabs alone never splits
```

## Execution ledger

每个部分记录 RED/GREEN 命令、结果、偏离规则和完成状态到 `output/ui-redesign-20261016/implementation-progress.md`。代码所有权按上述独立文件分工；共享接口变化先通知集成人。任务最终以真实行为和测试收口，视觉微调不新增镜像实现测试。

## Final evidence

Completed in the feature-branch workspace. See docs/artifacts/workbench_redesign_20261016.md for final 1637 passed / 6 skipped results, native Windows cancellation/drag/exit checks, review fixes and retained validation limits. No commit, push or installer rebuild.
