# 页面按需构造 —— 实施方案（含实测基线）

> 状态：待评审。本文所有耗时均为本机实测（offscreen、热缓存），探针脚本见
> `output/page_cost_probe.py`、`output/importtime_report.py`。

---

## 0. 结论摘要

| 项 | 数值 |
|---|---|
| 当前启动总计 | **2867 ms** |
| 其中窗口构造 | ~1018 ms（探针另一次测得 1372 ms，视机器负载） |
| 其中**页面构造** | **539.1 ms** |
| 其中**页面模块 import** | **733.0 ms**（含在 `import ui.main_window` 的 1071 ms 内） |

**推荐路径：先做 A 档（定点惰性化，~360-400ms，改 1-2 个文件，零架构风险），
再评估 C 档（首屏后预热，~440ms）。B 档（真正的按需构造）必须连带重构跨页接线，
改造面大，建议单独立项。**

A+C 之后启动理论进入 **~2.0-2.1s** 区间。

---

## 1. 实测基线（逐页）

公共依赖（qfluentwidgets / core.observability / ui.constants）预加载后的增量耗时：

| 页面 | import (ms) | construct (ms) | 备注 |
|---|---:|---:|---|
| homeInterface | 206.9 | 97.0 | import 含 pyqtgraph 首装（BScanView） |
| projectInterface | 160.7 | 100.5 | import 主要是 qfluentwidgets 组件首装 |
| processingInterface | 1.1 | 69.1 | |
| interpretationInterface | 0.8 | 56.6 | |
| **spatialInterface** | **361.2** | **153.2** | **最大项** |
| deliveryInterface | 1.0 | 27.5 | |
| jobsInterface | 0.6 | 3.4 | |
| settingsInterface | 0.6 | 31.8 | |
| **合计** | **733.0** | **539.1** | |

### 归因（importtime 定位）

- `spatialInterface` 的 361 ms ≈ **`pyqtgraph.opengl` 363.6 ms**（+ shaders 335.8 ms），
  经 `ui/widgets/trajectory_3d_view.py:51-56` 的**模块级 try-import** 引入，
  而 `spatial_page.py:43` 顶层 import 该视图 → 启动期必然付费。
- `homeInterface` 的 207 ms：pyqtgraph（B-scan 预览必需）+ qfluentwidgets 组件首装，
  属共享/必需成本，剥离空间小。
- `projectInterface` 的 161 ms：主要是 qfluentwidgets 组件首装（共享），自有重依赖不明显。
- 附带发现：`darkdetect` 单次 **358 ms self**（由 `qfluentwidgets.common.config` 引入），
  冷启动一次性开销，值得单独立项核查（见第 7 节）。

---

## 2. 三个方案档位

| 档 | 做法 | 收益 | 改动面 | 是否动接线 | 是否动测试/冒烟 |
|---|---|---:|---|---|---|
| **A** | **定点惰性化页面重依赖**（opengl 等） | ~360-400 ms | 1-2 文件 | 否 | 否 |
| **B** | **真正的按需构造**：占位页注册路由，换页时实例化 | ~440 ms 构造 + 剩余 import | 大（占位/换页/接线全改） | **是** | **是** |
| **C** | **首屏后空闲预热**：首屏只造 home+settings，其余首帧后分片构造 | ~440 ms | 中（构造时序 + 接线延后） | 半（延后而非拆分） | 需补 `ensure_pages_ready()` |

### 为什么 B 档不能"只延迟构造"

`PageCoordinator.connect_all()` 在 `__init__` 内跑，三个域子接线器直接持有页面对象做信号连接，
**八个页面全部被接线引用**：

- `coordinator_project.py`：home / project / spatial / delivery（38-41、165-169、214-216）
- `coordinator_processing.py`：processing / interpretation / spatial（38-40、162、217 等 10 余处）
- `coordinator_jobs.py`：jobs / home / processing（33、41、53、97）

且 `page_coordinator.py:132-137` 明确注释"页面全部交付，显式属性访问；AttributeError 即真 bug"。
因此延迟构造必须**连带把接线拆成按页单元并在实例化时补做**，否则要么无收益（接线时又被强制实例化），
要么静默漏接线（真 bug）。这才是 B 档的真实工作量。

另有两个硬约束：

1. `app_qt.py:126-131` 冒烟断言 **"没有页面降级为 PlaceholderPage"** —— 若用 `PlaceholderPage`
   当占位体，冒烟直接失败；需另建语义明确的占位类型（如 `_DeferredPage`），并让 `window.pages`
   保持 8 个键齐全（`app_qt.py:120-123` 断言键列表严格相等）。
2. `_inject_page_settings`（`main_window.py:121-131`）在 `__init__` 内向 project / spatial 注入设置值，
   靠 `hasattr` 守卫 —— 延迟后需改到实例化时注入。

---

## 3. A 档详细步骤（✅ 已实施 2026-09-21，实测 spatial 页 import 361→56 ms）

**目标**：`pyqtgraph.opengl` 的 363 ms 从启动链上摘掉。

1. `ui/widgets/trajectory_3d_view.py`
   - 现状 51-56 行：模块级 `try: import pyqtgraph.opengl as _gl; from PyQt6.QtGui import QVector3D as _Vector / except: _gl = None; _Vector = None`。
   - 改为惰性访问器（语义不变：导入失败仍降级为 None）：

     ```python
     _GL_STATE: dict[str, Any] = {}

     def _gl_module():
         """惰性导入 pyqtgraph.opengl（启动期不付费；缺 PyOpenGL 时降级 None）。"""
         if "gl" not in _GL_STATE:
             try:
                 import pyqtgraph.opengl as _gl
                 from PyQt6.QtGui import QVector3D as _Vector
             except Exception:  # noqa: BLE001
                 _gl, _Vector = None, None
             _GL_STATE["gl"], _GL_STATE["vec"] = _gl, _Vector
         return _GL_STATE["gl"], _GL_STATE["vec"]
     ```

   - 8 处使用点改为经访问器取值：336 / 352（`if _gl is None` 降级分支）、355（`GLViewWidget`）、
     357（`GLGridItem`）、512（`_Vector`）、700、753（`GLLinePlotItem`）、827（`GLSurfacePlotItem`）。
   - 降级语义必须逐条保持：PyOpenGL 缺失时仍走 `_gl is None` 分支（`Trajectory3DView` 降级为 QLabel）。
2. 验证：`output/page_cost_probe.py` 复测，`spatialInterface` import 应从 361 ms 降到 ~10 ms 量级；
   `output/launch_profile.py` 复测总耗时。
3. 回归：pytest 全量（945 passed 不降）、smoke 9 图 OK；空间信息页 3D 视图功能人工确认
   （offscreen 无 GL 时应走既有降级分支，不报错）。
4. 可选延伸（收益待实测）：`home_page` 的 pyqtgraph（B-scan 预览）若也做惰性化，
   需确认主页 4×BScanView 的构造时机是否可推迟到首帧后——属于 C 档范畴。

**风险**：低。改动局限在一个 widgets 模块，不动构造时序、不动接线、不动测试。
**回滚**：单文件 revert。

---

## 4. C 档详细步骤（✅ 已实施 2026-09-21，构造段 1008→349 ms）

> 实施备注（超出原设计的两处）：① 降级提示 QLabel 也改为惰性创建
> （`_ensure_fallback_label`），否则 `__init__` 里的 `if _gl is None` 会让 A 档白做；
> ② 除 `_goto_page` 外，`_on_top_nav_changed`（页签点击）同样需要 `ensure_pages_ready()`
> 兜底——它直接查 `pages` 字典，预热未完成时会**静默不切换**；
> ③ `_finish_warmup` 必须幂等（`_pages_warmed` 标志），否则 ensure_pages_ready 与
> 预热队列竞态会导致 connect_all 重复连接；④ 冒烟断言从"列表相等"放宽为
> "集合相等"（延后页插入顺序 ≠ 页签顺序）。

**设计**：`_create_pages()` 首屏只构造 **home（初始页）+ settings（接线/状态必需）**，
其余 6 页在首帧渲染后由 `QTimer.singleShot(0)` 分片构造（每帧一个，避免一次性长阻塞）；
全部构造完成后再执行 `PageCoordinator.connect_all()` 与 `_inject_page_settings()`。

需要处理：

1. **接线延后而非拆分**：`connect_all()` 移到预热完成回调里，接线代码本身不改
   （这是 C 相对 B 的最大优势）。
2. **兜底路径**：用户在预热完成前切到未构造页面 → `switchTo` / `_goto_page` 需立即构造该页
   （按需强制实例化），避免空白页。
3. **测试与冒烟**：新增 `window.ensure_pages_ready()`（同步构造全部剩余页面并完成接线），
   供 3 个引用 `main_window` 的测试文件与冒烟脚本显式调用；或在访问页面前 `processEvents()`。
   冒烟的两条断言（8 键齐全、无 PlaceholderPage）保持不变即可通过。
4. **文件树 / 输出面板**：`_build_ui` 在 `_create_pages` 之后且不依赖具体页面，无需改。

**风险**：中低。主要风险是"预热未完成即切页"的竞态，由兜底路径覆盖；
次要风险是测试假设"构造完即所有页面可用"，由 `ensure_pages_ready()` 覆盖。
**收益**：~440 ms（6 个非首屏页面的构造），加上 A 档已省下的 import，合计 ~800 ms。

---

## 5. B 档（可选 / 长期）

真正的按需构造：占位页注册路由，换页时实例化并补接线。

- 需把 `coordinator_project/processing/jobs` 三者的接线拆成**按页单元**，
  在页面实例化时对该页执行接线；跨页链（project→processing→jobs）需保证后到页实例化时
  双向连接都能补上（coordinator 已有 `_pending_select_line_id` 之类的挂起状态机制可复用）。
- 冒烟需引入非 `PlaceholderPage` 的占位类型并放宽/调整断言语义。
- 建议顺序：**A → C →（若仍需再压）B**。B 的收益（剩余 import，A 之后已不多）
  与改造风险不成正比。

---

## 6. 验收标准（三档通用）

1. `output/launch_profile.py` 三轮取均值，与改动前对比（机器空闲状态测）。
2. `app_qt.py --smoke`：9 图 OK，8 页齐全，无 PlaceholderPage 降级。
3. pytest 全量：保持 **945 passed / 6 skipped**，不新增失败/跳过。
4. 功能冒烟：空间信息页（含 3D 分支/降级）、处理页方法库加载、项目页文件树、任务页刷新。
5. A 档额外：`output/page_cost_probe.py` 中 `spatialInterface` import 显著下降。

---

## 7. 附带发现：darkdetect —— 已结案，**不建议改动**

结论：**根因真实，但端到端净收益仅 ~58 ms（占启动 2%），风险/收益比不划算，不做。**

### 根因链（已逐层验证）

```
qfluentwidgets/common/config.py:7  模块级 import darkdetect
  └─ darkdetect/__init__.py:31 模块体  platform.release().isdigit()
       └─ platform.uname()            （Windows 无 os.uname）
            └─ platform.win32_ver()
                 └─ platform._win32_ver() → _wmi_query('OS', ...)
                      └─ 拉起 WMI 查询子进程（powershell/wmic）
```

即：**一个"查系统主题"的模块，为了判断 Windows 版本，在导入时同步拉起了一个 WMI 子进程。**

### 实测数据（子进程隔离，n=15 / n=7）

| 项 | 裸解释器 | 说明 |
|---|---|---|
| `platform.release()` | 中位 **197.1 ms**（191–253） | 分布很紧，是真实成本非噪声 |
| `platform.win32_ver()` | 202.1 ms | 同上 |
| `platform._get_machine_win32()` | 20.6 ms | |
| `import darkdetect`（importtime） | self 324 ms / cumulative 342 ms | 冷启动单次 |

### 关键反证：为什么不做

写了 `output/darkdetect_shim.py`（winreg 直读 `AppsUseLightTheme`，`listener()` 按需降级真实模块），
并做 A/B（`output/darkdetect_ab.py` / `output/darkdetect_cold_check.py`，n=7+9 交错）：

| 臂 | `import qfluentwidgets` | 事后 `platform.release()` | 合计 |
|---|---|---|---|
| base（真实 darkdetect） | 593.4 ms | **0.0 ms**（已被 darkdetect 付过） | 593.4 |
| shim（替身） | 535.7 ms | **35.1 ms**（残留，未被消除） | 570.8 |

**净值只有 22–58 ms**——替身确实让 `import darkdetect` 从 342 ms 归零，但 WMI 子进程的成本
在真实链路里只有约五分之一；把残留算进去后总账几乎持平。

### 否决理由

1. 收益 ~2%，低于本项目测量噪声（同机全启动 2.6–10.1 s 抖动）。
2. 手段是运行时替换 `sys.modules['darkdetect']`——猴子补丁第三方依赖，违背工业常规；
   上游升级 / 打包（PyInstaller 静态分析）都可能踩坑。
3. `listener()` 降级路径会引入"主题跟随系统"行为的分支差异，难测。
4. 正解在上游：darkdetect 在模块体做 OS 版本探测是设计缺陷，应在上游提 issue/PR，
   或在首次 `theme()` 调用时再探测。

> 附带结论：**importtime 的自耗时不能直接当端到端收益**。它把一次性 OS 级开销
> （子进程拉起、冷页错误）100% 记在首个触发模块名下，而该开销在完整启动链里往往会被
> 其它环节分摊或预热。本项目前两批优化都以端到端 `launch_profile` 为准，这个习惯要保留。

---

## 8. 工业级软件一般会怎么做

### 8.1 先把"快"变成"预算 + 回归"，而不是一次性优化

Chrome / VS Code / JetBrains / Office 都把冷启动当 **SLO** 管：

- 定义明确的指标——**TTI（time-to-interactive）/ 首屏可用**，而不是"进程起来"。
- 设预算并进 CI 做 **perf 回归**，超预算就红（JetBrains 有专职的 startup performance 测试套件）。
- **分段埋点常驻产品**（Chrome tracing、VS Code startup timers），出问题能立刻归因到段，
  不是临时写探针。

> 本项目差距：分段计时只在 `output/launch_profile.py`，是外部探针。建议做成
> `app_qt.py --startup-profile` 常驻开关 + 结构化日志，作为 A/B/C 三档的验收基础设施。

### 8.2 手段优先级：延迟 > 并行 > 加速 > 重构（最后才动架构）

| 优先级 | 手段 | 本轮对应 |
|---|---|---|
| 1 | **延迟**：首屏路径外的东西全部延后 | 第 1/2 批（GIS 懒导入、scipy 出链）、A/C 档 |
| 2 | **并行**：I/O 与 CPU 重叠、后台预热 | C 档（event loop 空闲分片） |
| 3 | **加速**：真减少工作量（缓存、去依赖、两级注册表） | 第 2 批两级注册表、色标查表 |
| 4 | **重构**：改构造模型 | B 档（拆接线，最后考虑） |

### 8.3 "按需构造"的工业标准形态

不是"延迟 import"，而是 **延迟 construct + 延迟 wire**：

- **首屏只构造用户第一眼看到的那一页**，其余在 Tab 首次激活时构造。
  Chrome DevTools 的 panel、VS Code 的 viewlet、JetBrains 的 tool window 全是这个模型。
- **前提是接线与构造解耦**：JetBrains 用 `ToolWindowFactory` + 声明式注册 + MessageBus 订阅
  （页面不在就不订阅）；VS Code 用 contribution point + 事件，构造时才注册。
- **注册表只加载描述符，实现首次调用才加载**——IDEA 插件模型，与本项目的两级注册表是同一招。
- **splash + 空闲分片**：主窗先出，剩余页面在 event loop idle 里分片构造，用户感知是"立刻能用"。

> 本项目卡点就在这里：`PageCoordinator.connect_all()` 直接持有 8 个页面对象做 signal-signal
> 连接，接线与构造是耦合的。所以 B 档不是"页面改造"，本质是**把接线改成订阅式**。

### 8.4 对本项目的路径建议

现状基线：**2743 ms**（分段合计中位数，n=5；原始 7122 ms，累计 **−61%**）
—— import main_window 924 / 构造 1027 / qfluentwidgets 500 / show 127。

| 档 | 预期收益 | 风险 | 建议 |
|---|---|---|---|
| **A** trajectory_3d_view 的 `pyqtgraph.opengl` 惰性化 | ~360 ms（import 段） | 极低，纯改 import | **先做** |
| **C** 首屏后空闲分片预热 | ~440 ms | 低 | **接着做**，与工业"splash + idle"同构 |
| **B** 真正按需构造（需先把接线改订阅式） | 500–1000 ms | 中高，架构改造 | 前两档做完仍不满足再上 |

这与工业实践的顺序一致：**先延迟、再并行，最后才动架构**。A+C 合计约 800 ms，
可把启动压到 ~1.9 s；B 档留作长期项，且必须先完成"接线解耦"这一前置重构。

> **实施结果（2026-09-21，A+C 两档同日落地）**：2743 → **1959 ms**（分段合计中位数，
> n=5；原始 7122 ms，累计 **−72%**）。构造段 1008→349 ms，show 段 82→227 ms
> （首帧后第 1 个延后页在此触发，属"挪出首帧"而非消失）。回归：pytest 951 项
> 0 失败 0 错误（6 skipped 与基线一致）、smoke 9 图 OK、`scripts/verify_gl_lazy.py`
> 通过（含无 PyOpenGL 降级路径）、`page_cost_probe` spatial 页 import 361→56 ms、
> 导入期 `sys.modules` 确认无 `pyqtgraph.opengl` / `OpenGL.*`。

> **收尾（2026-09-21 晚，已合并进 main）**：PR #17 合并，CI 5/5 全绿。本机复测两次
> 采样 1959 / 2144 ms（各分段同步上浮约 10%，属负载波动）——**启动耗时是分布不是
> 标量**，对外口径取"约 2.0 s（原 7.1 s，−70%~−72%）"。
>
> 合并前必须先修的既有 CI 红灯单独成 PR #18（fix/ci-green）。**修完 #18 之后 #17 才
> 第一次真正跑到 ruff 与 gui-linux**，随即暴露两处本 PR 引入的回归（已修）：
> ① `methods_registry.py` 的 `PROCESSING_METHODS` / `ALGORITHM_CATALOG` 因惰性化改走
> `__getattr__` 而在 `__all__` 里被判 F822 未定义——用 `TYPE_CHECKING` 类型声明修，
> 纯注解不产生运行期绑定；② 标题栏 z-order 测试的开屏竞态——开屏（自带 TitleBar 的
> 独立 frameless 窗口）600 ms 后才关，**构造提速把 `__init__` 压到 600 ms 以内**，
> `widgetAt` 就改命中开屏的 TitleBar，"越快越容易挂"，只在快机器/Linux 复现。
>
> **教训**：先修 CI 再合功能分支是对的——否则红灯掩盖红灯，本 PR 会带着两处真回归
> 进 main。反向教训是性能优化会让**时间依赖型测试**翻转，提速后需专门复查依赖墙钟
> 的断言。

### 8.5 不该做的事（工业界的负面清单）

- **不猴子补丁第三方库**（本节 darkdetect 就是现成反例）。要改就 fork/vendor 并写明，
  或推上游。
- **不为 2% 的收益引入行为分支**（替身的 `listener()` 降级路径即此类）。
- **不拿单次 importtime 当收益证据**——见 7 节反证。

---

## 9. B 档可行性研究（2026-09-21）

A/C 两档落地、启动从 7.1 s 压到约 2.0 s 之后，回头评估原计划的 B 档（真正的按需构造）。
结论：**完整 B 档不做；只建议做 B′ 最小版**。

### 9.1 结论摘要

| 项 | 完整 B 档（按需构造 + 接线订阅式改造） | B′ 最小版（仅页面模块 import 惰性化） |
|---|---|---|
| 首帧收益 | 0 ms（构造成本已被 C 档移入空闲帧） | **约 90 ms**（上界 ~175 ms） |
| 总工作量收益 | 跳过"从未访问"的页，最多 ~464 ms 空闲 CPU | 同上，但不需要为此改接线 |
| 新增代价 | 首次交互期 **数百毫秒同步卡顿** | 无（成本仍在空闲帧） |
| 新增风险面 | 状态回放、None 崩溃、竞态窗口扩至整个会话 | 基本无（构造时机不变） |
| 改动面 | 61 个页面访问点 + 三个子接线器重构 | `main_window._import_page_class` 一个函数 |

**判定：收益已被 C 档提前兑现，剩下的部分不足以抵消风险。**

### 9.2 接线现状：扇出是横切的，不是按页的

三个子接线器在 `connect_all()` 里静态取页（缺页即 `AttributeError`）：

| 子接线器 | connect_all 静态取页 |
|---|---|
| `ProjectChain` | home, project, spatial, delivery |
| `ProcessingChain` | processing, interpretation, spatial |
| `JobHub` | home, jobs |

但这只是静态要求。真正决定"按需"能否省钱的是**运行期扇出**——用 ast 提取三个文件里
每个方法触达的页面（`output/dep_graph.py`）：

| 处理器 | 触达页面 |
|---|---|
| `ProjectChain.on_lines_updated` | **project, delivery, spatial, processing**（4 页，全部无条件） |
| `ProjectChain.on_project_opened` | home, project, spatial +（经 `update_line_labels`→）interpretation, processing（**5 页**） |
| `ProjectChain.on_line_selected` | interpretation, processing, project |
| `ProjectChain.on_dataset_preview` | home, interpretation, processing, project |
| `ProcessingChain.on_*` | 集中在 processing / interpretation / spatial |
| `JobHub._views` | jobs + home（外加 `on_progress`→processing） |

**工作流闭包**：`打开项目 → 选中测线` 两个动作内，8 个页面全部被强制实例化。所谓
"按需"在真实工作流里等于"两个动作内全建完"，只是把成本从空闲帧挪到了用户点击的那一帧。

### 9.3 成本账：真正可省的只有约 90 ms

**构造成本**（延后 6 页，真实窗口父级，两次采样 n=2）：

| 页 | 采样 1 | 采样 2 |
|---|---|---|
| ProjectPage | 118.0 | 88.9 |
| ProcessingPage | 115.0 | 75.4 |
| InterpretationPage | 66.9 | 89.9 |
| SpatialPage | 124.3 | 141.8 |
| DeliveryPage | 34.1 | 24.4 |
| JobsPage | 6.0 | 3.4 |
| **合计** | **464.4 ms** | **423.8 ms** |

这笔钱 **C 档已经付过了**——它现在落在空闲帧里，不在首帧关键路径上。

**import 成本**：关键在于"延后某页能省多少"必须用**阻断该页后的模块差集**来算，不能用
"该页的 import 增量"——后者会把与别处共享的模块也算进去。这是本项目第 3 次踩到同一个
首因归因陷阱（前两次：data_context 冤案、darkdetect）。

实测（`output/it_runner.py` + `output/diff_it.py`，两轮 importtime 求差集）：

- 阻断 `spatial_page` → 23 个模块消失，其 self 耗时合计 **90.8 ms**；
  两轮 self 总计差 176.7 ms（含跨轮噪声），故可信区间 **约 90 ms，上界 ~175 ms**。
- 阻断全部 6 个延后页 → 端到端 A/B（n=3 交替）baseline 中位 1004 ms vs lazy 1066 ms，
  **差值落在噪声内**；`h5py` 在 lazy 档**仍然被导入**（来自 `core/*`，与页面无关）。
  即除 spatial 外，其余 5 页几乎没有独占重依赖。
- 单依赖复核：`import PIL(+Image)` 中位 42.0 ms（n=7），`defusedxml` ≈ 0 ms。

所以：**延后页面模块 import 的全部收益 ≈ 90 ms，且只来自 spatial 页的 PIL / map_tiles 栈。**

### 9.4 鲁棒性风险（四条硬伤）

**R1 — 缺页即崩溃，且与仓库既有纪律冲突**
`main_window._page()` 返回 `self.pages.get(name)`，缺页返回 `None`。三个子接线器共
**61 个 `co.page()` 访问点**（38/17/6），全部无条件解引用，没有一处判空。要么改成
"缺页即构造"（`require_page()`），要么到处加判空——但仓库在 `page_coordinator.connect_job_bridge`
里明确写着"显式属性访问：槽位被改名/移走时立刻 AttributeError 暴露，而不是 hasattr 探测
静默跳过（那次事故：load_methods 不执行 → 方法库为空）"。加判空正是被明令禁止的形态。

**R2 — 状态回放：最难、且无现成机制**（决定性风险）
全仓 grep 无 replay / catch-up 机制。以 `JobHub._upsert` 为例：

```python
def _upsert(self, job_id: str) -> None:
    if job_id in self.known_job_ids:
        return          # ← 一次性守卫
    self.known_job_ids.add(job_id)
    for view in self._views():
        view.upsert_job(job_id, title)
```

任务页若在任务跑过之后才构造，`known_job_ids` 已含这些 id → 直接 return → **新的 JobTable
永久缺行，且无任何自愈路径**。同理：project 页延后 → 测线表/成果表空；processing 页延后 →
方法库空——后者正是 R1 里那次历史事故的**同构形态**。每个延后页都要配一条补播路径，
即 6 页 × 多个状态持有者，这是 B 档真正的成本大头。

**R3 — 竞态窗口从 6 帧扩大到整个会话**
`_on_backend_ready` 里调 `connect_job_bridge(bridge)`，与页面构造/预热**没有任何同步关系**；
JobBridge 信号一到就走 `JobHub.on_status → _views() → co.page('jobsInterface').job_table()`。
C 档下这个窗口只有预热那 6 个空闲帧，且启动期无任务在跑，实际风险为 0；B 档下该窗口
**覆盖整个会话**，变成真实崩溃面。

**R4 — 测试契约**
`test_page_coordinator.py` / `test_coordinator_chains.py` / `test_ui_migration_regressions.py` /
`test_page_controls_wired.py` 约 800 行钉住现有接线契约；`app_qt --smoke` 断言 8 页齐全且无
PlaceholderPage。B 档需要引入新的占位类型并系统性调整这些断言。

### 9.5 B′ 最小版（建议做，收益约 90 ms，零接线改动）

只改 `main_window._import_page_class`：把 8 个页面模块的 import 从"模块导入期"挪到
"页面构造时"。因为 C 档已经把延后页的构造放在空闲帧，PIL / map_tiles 栈会随之在空闲帧
加载，首帧少付约 90 ms。

- 不改接线、不引入状态回放、不动测试契约；
- 风险点仅一个：`_import_page_class` 当前在导入失败时降级为占位页，惰性化后降级时机
  从"启动期"变成"构造时"，`--smoke` 的"无 PlaceholderPage"断言需在 `ensure_pages_ready()`
  之后判定（冒烟已调用）。

### 9.6 真正的下一棒不在页面

`import ui.main_window` 的 self 耗时合计 ~1887 ms，按顶层包：

| 顶层包 | self ms | 占比 |
|---|---|---|
| darkdetect | 405.3 | 19.9%（第 7 节已结案，端到端净收益仅 22–58 ms） |
| mygpr | 184.2 | 9.1% |
| qfluentwidgets | 177.7 | 8.7% |
| pyqtgraph | 174.3 | 8.6% |
| numpy | 163.4 | 8.0% |
| ui | 135.6 | 6.7% |
| PyQt6 | 122.1 | 6.0% |
| core | 115.1 | 5.7% |
| h5py | 79.2 | 3.9% |
| pyproj | 53.3 | 2.6% |

页面只占 `ui` 的一部分。两条具体线索：

- **pyqtgraph 174 ms 被钉死在关键路径**：`ui/theme_helpers.py:11` 顶层 `import pyqtgraph as pg`，
  而全文件只在 `apply_theme()` 里用 3 次 `pg.setConfigOption`。可改为"首次建图时才应用 pg 配色"，
  把 174 ms 随空闲帧挪走（注意：单纯挪到 `apply_theme` 内部只是改标签，不省钱——
  `apply_theme` 本身在启动早期就被调用）。
- **pyproj 53 ms**：`ui/widgets/proj_safe.py` 与 `core/*` 已多为函数级惰性，需确认剩余
  顶层入口。

### 9.7 方法论沉淀

- **首因归因陷阱（第 3 次）**：一次性开销被 100% 记在首个触发者名下。判定"延后 X 能省多少"
  必须用**阻断 X 后的差集**，不能用"X 的 import 增量"。本项目已三次栽在这个坑上
  （data_context、darkdetect、本次页面依赖）。
- **Git Bash heredoc 会污染反斜杠**：本次实测 `cat > f <<'EOF'` 里的 `\s` → `/s`、`\n` → `/n`，
  导致正则静默无匹配、注入代码 SyntaxError。**落脚本一律用 Write 工具**，不要走 heredoc。
- **本机噪声 ±200 ms**：小于 100 ms 的收益无法用端到端 A/B 证明，必须换更敏感的方法
  （importtime 差集 / 子进程隔离单依赖计时）。
