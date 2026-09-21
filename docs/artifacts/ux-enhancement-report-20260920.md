# MyGPR UX 增强调研报告

日期：2026-09-20 ｜ 范围：v0.9.38 全部用户触点（GUI 八页 / 全局框架 / 控件层 / CLI）
方法：只读通读 `ui/` 全部页面与关键控件、四个 controllers、三个接线器、`cli_batch.py`；关键结论已逐条回到源码验证（含行号）。仅调研，未改任何代码。

---

## 一、现状基线：已经做对的事

增强前先明确哪些不该动——以下是当前体验的正面基线，改动时应保持同等水准：

- **错误文案体系**：`backend_controller.py:48-78` 把 `MyGPRError` 按错误码重组为「消息 — 建议」，HDF5 损坏、文件占用等场景给出可操作建议。
- **校验不打断输入**：`widgets/validators.py:91-103` 中间态放行、终态才标红；提交校验统一「红框 + InfoBar」。
- **空态有引导**：B-Scan「暂无数据 — 请先在项目页导入测线」、任务表/文件树空态占位、成果页无测线时隐藏列表改提示。
- **门控前置**：解释页会话未打开即禁用编辑与拾取（P1-6），处理页运行/取消按钮互斥（防重复提交），AutoTune 有在飞防护（P2-7）。
- **后端失败可见**：`main_window.py:544-561` 常驻错误横幅 + 重试按钮（P0-2）。
- **恢复有确认**：备份恢复覆盖项目前弹 MessageBox 明示「不可撤销」（`delivery_page.py:417-427`）。
- **CLI 兜底**：退出码分层（0/1/2）、校验聚合全部错误而非首错即停、`resume` 失败重跑、summary 带 traceback。
- **细节**：输出面板阅读位置保护、日志 5000 条上限、主题单源深浅双表、智能默认值（测线号 L01→L02 跳占用、结果名回退）。

---

## 二、问题清单（按主题分组，均已源码验证）

严重度定义：**高**=丢数据/丢结果/用户被误导；**中**=明显摩擦但可恢复；**低**=打磨项。

### A. 静默失败（最高优先级主题）

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| A1 | 高 | 「后端尚未就绪」只写日志不弹 UI | `controllers/project_controller.py:82`、`processing_controller.py:78`、`interpretation_controller.py:44`、`delivery_controller.py:37` | 四个控制器兜底路径只 `log_message.emit`。启动慢/失败时用户点任何操作都是"点了没反应"。建议：统一升级为 InfoBar.warning（主窗口已有 `_infobar` 通道，接线器可转发） |
| A2 | 高 | 成果页 busy 防护形同虚设 | `pages/delivery_page.py:278` `set_busy` **全仓库无人调用** | `DeliveryPage._busy` 永远为 False：空间成果/报告包/备份/恢复四个按钮可连点重复提交。页面已写好 busy 机制，只差接线（`delivery_controller` 无 busy 信号，`coordinator_project.py:106` 只连了 project 页）。补一条信号接线即可 |
| A3 | 中 | 任务取消失败静默，却提示"已请求取消" | `controllers/backend_controller.py:158-163` vs `coordinator_jobs.py:149` | cancel 捕获 KeyError/RuntimeError 仅写 WARNING 日志；接线器无条件弹"已请求取消任务"。用户以为取消成功，任务仍在跑。建议：取消失败回传 error 级提示 |
| A4 | 中 | 删除成果时后代查询为空 → 整个流程静默终止 | `coordinator_project.py:419-421` `if not descendants: return` | 点删除无确认框、无提示、无动作。建议：空后代也应弹简化确认框或提示"该成果无关联产物，确认删除？" |
| A5 | 中 | 未知异常兜底暴露英文异常串 | `controllers/backend_controller.py:78` `return message or type(exc).__name__` | 用户直接看到 Python 异常名。建议：映射为"发生未知错误，详情见日志（含打开日志目录动作）" |
| A6 | 中 | 备份目标目录为空时静默 no-op | `coordinator_project.py:532-534` | `if not dest: return` 无提示（页面侧已挡，但兜底链路静默） |
| A7 | 低 | 日志前缀泄漏 | `delivery_controller.py:190`、`backend_controller.py:361` | `log_message.emit("SUCCESS 已设为当前…")` 把内部级别前缀混入用户日志面板 |

### B. 长任务体验

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| B1 | 高 | 导入/传感器同步/空间成果/报告/备份/恢复六类任务无重复提交防护 | `project_controller.py:310-331`、`delivery_controller.py:43-67` | 对比处理页已有 `_running` 互斥。任务在飞时可反复点击重复提交同一任务。建议：统一 busy 门控（与 A2 同一修复面） |
| B2 | 高 | 关闭窗口不检查活动任务 | `main_window.py:797-819` `closeEvent` | 有作业在跑时直接退出 → 任务结果丢失，无任何确认。建议：closeEvent 检查 JobBridge 活动任务，弹「有 N 个任务在运行，退出将中断，确认？」 |
| B3 | 中 | 无 ETA、步骤内部无进度 | 全库无 ETA 计算；`mygpr/application/processing/service.py:80-83` 单方法仅 0→1 两拍 | 长步骤（SVD/大测线）期间进度条静止。建议：至少显示"已运行时长"；步骤内进度可后续按 chunk 回报 |
| B4 | 中 | 任务失败后无"重试"入口 | `coordinator_processing.py:176` 只弹 error InfoBar | 参数原样保留即可一键重提（CLI 反而有 resume）。建议：InfoBar 加"重试"按钮，复用上次 payload |
| B5 | 低 | 完成通知仅应用内 InfoBar | `coordinator_jobs.py` | 窗口最小化/失焦时易错过。建议：QSystemTrayIcon 消息（可开关） |

### C. 破坏性操作保护

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| C1 | 高 | 处理链步骤删除：三种路径均无确认、无撤销 | `widgets/pipeline_list.py:151,230,283-284,311-316` | 行内按钮/右键/Delete 键直接 `del self._steps[idx]`，误删即丢整套参数配置。建议：加轻量撤销栈（保留被删 step，Ctrl+Z 恢复）或删除前确认。注意与项目规范一致：项目页测线删除已有确认 + .trash 恢复，处理链应看齐 |
| C2 | 高 | 取消任务无二次确认 | `widgets/job_widgets.py:149,264,334` | 按钮与右键直接 emit cancel。误触即中断数小时处理。建议：对"运行中且已跑 >30s"的任务弹确认，刚提交的可直接取消 |
| C3 | 中 | 标注点「清空」无确认 | `pages/interpretation_page.py:409-414` | 清空全部拾取点无确认（撤销依赖会话 undo 是否覆盖，未验证）。建议：>5 个点时确认 |

### D. 表单与参数输入

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| D1 | 高 | 数值参数越界静默钳制，无单位标注 | `widgets/param_form.py:106-149` | schema 无 `unit` 字段，全表单无 m/ns/dB 单位；越界靠 SpinBox 静默钳制，`validators.py` 的标红机制未接入。GPR 参数单位是专业用户的核心信息。建议：schema 加 unit 渲染后缀；钳制发生时短暂高亮或状态栏提示 |
| D2 | 中 | 范围提示只在 label tooltip | `widgets/param_form.py:146-149` | 用户手在编辑器上时看不到范围。建议：编辑器本体也设 tooltip |
| D3 | 中 | 高级参数折叠无计数徽标 | `widgets/param_form.py:69-83` | >4 参数折叠且默认收起，用户不知道还有参数未配置。建议：折叠按钮显示"(n 项)" |
| D4 | 低 | 校验全部为提交后触发 | 各页面 | 输入过程无即时校验；文件路径存在性不前置检查（依赖手动点"预检"）。可在失焦时做存在性检查 |

### E. 视图空态/错误态

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| E1 | 中 | 深度切片全 NaN/无效矩阵 → 空白画布 | `widgets/depth_slice_view.py:78-83,113-120` | 静默 clear_grid，无"暂无数据/数据无效"占位。建议：对齐 BScanView 的空态标题做法 |
| E2 | 中 | 地图瓦片下载失败静默丢弃 | `widgets/map_view.py:201-204` | 离线时地图空白但轨迹照画，无"网络不可用，使用离线底图"提示 |
| E3 | 中 | B-Scan 初始标题是"B-Scan图像"而非空态 | `widgets/bscan_view.py:135` | 页面若从未调 clear()，首屏是误导性空图。建议：构造即设空态标题 |
| E4 | 中 | `set_matrix` 对非法矩阵直接 raise | `widgets/bscan_view.py:330-332` | 后端数据损坏时存在未捕获崩溃风险（取决于调用方兜底）。建议：视图层 try/except → 空态 + 错误标题 |
| E5 | 中 | A-Scan 空数据无提示 | `widgets/ascan_view.py:39-51` | 仅清曲线。低频使用场景，随 E1 一并处理 |
| E6 | 低 | 日志面板截断无提示 | `widgets/output_panel.py:86,156` | 5000 条静默丢弃，无"已截断"标记；且无级别过滤、无搜索框（`output_panel.py:104-123` 工具条仅自动滚动/清空/导出） |

### F. 发现性与帮助（含一处帮助文档失真）

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| F1 | 高 | **F1 帮助列出不存在的快捷键 Ctrl+L** | `main_window.py:744` | 全仓库无 Ctrl+L 注册，帮助文档与实现不符，直接伤害信任。建议：删掉该行或补实现 |
| F2 | 中 | 视图手势零引导 | `widgets/bscan_view.py` 全文；`map_view.py:744-752` | 滚轮缩放/左键平移/十字光标/右键菜单全靠探索。解释页已有优秀范例（"提示：在剖面图上左键点击拾取标注点"），建议把这类一行提示推广到处理页/主页预览 |
| F3 | 中 | 文件树双击行为不一致 | `widgets/file_tree_panel.py:442-455` | 测线双击=跳处理页，文件双击=系统打开，成果/报告叶子双击无动作。建议：统一"双击=该节点的主操作"，无主操作时用系统打开 |
| F4 | 中 | 文件树刷新入口弱 | `widgets/file_tree_panel.py:156-160,458-461` | F5 仅"文件"视图生效且无可见刷新按钮（只在右键空白菜单里）。建议：面板头部加刷新钮 |
| F5 | 低 | 拖拽能力不可见 | `widgets/pipeline_list.py:5-11`、`method_browser.py:9-10` | 拖拽重排/拖拽添加/双击添加多路径并存但无提示。F1 帮助已覆盖部分，可在控件空态文案里点一句 |

### G. 可访问性与一致性

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| G1 | 中 | disabled 状态徽章对比度 ~2.2:1 | `theme_helpers.py:176-178` + `constants.py:78,94` | 白字配 #9ca3af 灰底（排队/已取消徽章），12px 小字下远低于 WCAG AA 4.5:1。建议：深灰字或加深底色 |
| G2 | 低 | 字号 pt/px 混用、脱离三级档 | `constants.py:48-50` vs `bscan_view.py:733`、`theme_helpers.py:262`、`job_widgets.py:41` | 存在 11px/12px/13px 散值。建议：收敛到 constants 定义的档位 |

### H. CLI 触点

| # | 严重度 | 问题 | 位置 | 说明与建议 |
|---|---|---|---|---|
| H1 | 中 | 批处理无过程进度 | `cli_batch.py:484-502` | 每 job 仅结束打一行 `[OK]/[FAIL]`，长 job 零输出。建议：至少打印 job 开始行 + 耗时 |
| H2 | 中 | 无日志文件选项 | `cli_batch.py` | 仅 stdout + summary JSON，无人值守排障难。建议：加 `--log-file` |
| H3 | 低 | `--force` 帮助未说明风险 | `cli_batch.py:660-663` | 建议补一句"可能以非法参数产出废结果" |

---

## 三、建议实施路线

**第一批（P0，修复"丢数据/被误导"，预计都是小改动）**
1. A2 + B1：补 `delivery_controller` busy 信号并接线 `delivery.set_busy`（同一修复面，一次解决成果页连点 + 六类任务防重）。
2. F1：删除 F1 对话框中的 Ctrl+L 幽灵条目（一行）。
3. B2：closeEvent 活动任务检查 + 退出确认。
4. A1：四控制器"后端尚未就绪"升级为 InfoBar（接线器已有 InfoBar 通道可复用）。
5. C1/C2：处理链删除撤销栈 + 长任务取消确认。

**第二批（P1，明显摩擦）**
- D1 参数单位 + 越界反馈；B3 已运行时长显示；B4 失败重试按钮；A3 取消失败回传；E1/E3/E4 视图空态与防崩溃；G1 徽章对比度；H1/H2 CLI 进度与日志文件。

**第三批（P2，打磨）**
- F2 手势提示推广、F3/F4 文件树一致性、E6 日志过滤搜索、D3 折叠徽标、D4 失焦校验、B5 系统托盘通知、G2 字号收敛、H3/A5/A6/A7/C3 文案与小修。

**验证方式**（遵循仓库既有约定）：
- 每批跑 `QT_QPA_PLATFORM=offscreen python app_qt.py --smoke` 截图比对；
- 接线类改动（A2/A1）补 pytest 断言信号连接存在；
- `python scripts/check_architecture.py` + `check_python_compile.py` 守护分层纪律；
- GUI/DPI 项在 Windows 目标机人工验收。

## 四、风险与约束提醒

- 所有 UI→后端交互必须走 `desktop_backend_facade` / 接线器纪律，新增信号接线放 `coordinator_*`，不得在页面直调 controller。
- 长任务相关改动不得绕过 controller `run_worker` + JobBridge 机制。
- `ui/dialogs.py` 是接线器/controllers 许可的唯一对话框模块；C1/C2 的确认框应走该模块函数式 API。
- 大文件路径禁止整体读入 RAM 的约束同样适用于任何"预检查文件存在性"的新逻辑（D4 只查 stat，不读内容）。

## 五、修正说明

调研过程中两条自动化探查结论与源码不符，已人工复核剔除：① "无快捷键总览入口"——F1 快捷键清单对话框实际存在（`main_window.py:729-756`），真实缺口是其中列出了未实现的 Ctrl+L（见 F1 条目）；② "日志无级别过滤已有隐藏实现"——仅代码注释暗示，UI 未实现，按未实现计入 E6。
