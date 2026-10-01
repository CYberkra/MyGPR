# MyGPR 研发技能工作流

2026-09-26 配置。此文件通过根 AGENTS.md 路由使用；不是运行时插件，不向软件注入算法或 UI 功能。

## 已有安装与选择规则

优先使用当前会话技能目录中同名入口，首次使用阅读完整 SKILL.md 及任务所需引用。当前机器位置如下，其他机器使用自身用户目录，不硬编码用户名到脚本。

| 技能 | 本机来源 | 用途 |
| --- | --- | --- |
| frontend-design-polish | `%USERPROFILE%/.codex/skills/frontend-design-polish/SKILL.md` | 科学桌面界面的布局、密度、常用操作和视觉检查 |
| mygpr-processing-validation | `%USERPROFILE%/.codex/skills/mygpr-processing-validation/SKILL.md` | 数据上下文、算法输出、处理链和 GUI/CLI 一致性 |
| windows-desktop-e2e | `%USERPROFILE%/.agents/skills/windows-desktop-e2e/SKILL.md` | pywinauto/UIA 桌面操作测试及失败截图 |
| python-testing | `%USERPROFILE%/.agents/skills/python-testing/SKILL.md` | pytest 夹具、参数化、异常与回归测试 |
| codebase-design | `%USERPROFILE%/.agents/skills/codebase-design/SKILL.md` | 接口职责、模块重构、可测试性 |
| superpowers:systematic-debugging | 会话技能目录中的 Superpowers 插件入口 | 先复现和查因，再修复和验证 |

Superpowers 缓存路径随插件版本变化，不在项目中固定缓存版本。多个发行源同时可见时，本项目优先当前会话 openai-curated-remote 版本，只读这一份；缺失时再使用已有另一发行源。找不到技能时明确报告，不能假装已加载。

这些入口已存在，因此本次不重复复制、不覆盖全局技能、不自动升级第三方包。`.codex/skills` 与 `.agents/skills` 保持各自目录，不合并。新增机器缺失时再安装指定技能，不安装整套无关技能。

## 实际工作方式

### 界面与交互

1. 用 frontend-design-polish 明确用户任务与关键操作路径，先解决操作步骤和空间分配。
2. 设计任务给出状态、空白/失败情况、小屏行为及验收条件；只做设计时不改功能代码。
3. 实施任务复用 PyQt6/qfluentwidgets 与项目控制器，不把网页框架示例直接搬进桌面应用。
4. 用现有 Qt 测试验证状态和信号；涉及实际点击、弹窗、焦点、滚动、DPI 时再使用 windows-desktop-e2e。
5. 桌面测试先检查控件 UIA 可访问性。自绘 B-scan/pyqtgraph 可能不能作为普通 UIA 控件操作，结合 Qt 级状态测试与截图/实机检查，不能仅凭技能中的通用 Qt 支持描述宣称可自动操作全部图形。

### 算法和处理链

1. 用 mygpr-processing-validation 确认实际数据类型、输入形状、轴单位、频带和传感器元数据；技能中的样例数据默认值不能覆盖当前文件元数据。
2. 记录输入、步骤顺序和参数，以小规模可复现数据建立基线。
3. 按问题验证 shape、有限值、幅度分布、目标区域及前后变化；独立自动色阶的图片不能证明振幅改善。
4. 用 python-testing 覆盖真实路径与关键异常；必要时固定数据对照 CLI 和 GUI。
5. 修改注册算法时同时核对方法注册表、参数 UI、CLI 与适用测试。运行昂贵仿真、批量处理或大规模性能试验仍须符合用户当前任务范围。

### 排错与重构

- 排错：选择一份 systematic-debugging，复现 → 收集状态/线程/数据证据 → 验证原因 → 最小修复 → 回归。不能仅通过改测试期待值消除失败。
- 重构：codebase-design 用于划分接口与职责；遵守 ui/controllers、facade、application/domain/infrastructure 的现有边界。不为一个小改动引入多层抽象。
- 修复期间需要测试时加载 python-testing；测试针对外部行为、错误路径与曾经失败的案例，不为了数值覆盖率添加镜像实现的测试。

## 验证入口（PowerShell）

按改动选择相关测试，不要求每个任务机械运行全部测试。

```powershell
# 处理页状态、成果预览异步失效、成果索引
$env:QT_QPA_PLATFORM = 'offscreen'
.venv\Scripts\python.exe -m pytest tests/test_processing_page.py tests/test_artifact_preview_generation.py tests/test_processing_artifact_index.py -q

# 跨层或较大改动
.venv\Scripts\python.exe scripts/check_architecture.py
.venv\Scripts\python.exe scripts/check_python_compile.py

# 人工/桌面 UIA 验收须使用真实桌面，不能沿用 offscreen
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
.venv\Scripts\python.exe app_qt.py
```

其他处理页测试（如 test_processing_page_v2.py）、算法、GUI/CLI 测试由实际改动定位后追加；以当前文件存在和测试内容为准。旧技能提到的 preflight_check.py 或 CLI 子命令先确认存在与用法，不存在时选择当前仓库等价入口并说明。

桌面 E2E 在独立临时项目和小样本上执行：导入 → 搜索/添加算法 → 设置参数 → 运行 → 查看输出 → 切换测线 → 关闭；不要接管用户正在处理真实项目的窗口。等待信号/状态而非固定睡眠；失败留截图与日志。pywinauto 可导入不等于该流程已经验证。

## 交付证据

每次报告：改了什么、为什么、实际运行了哪些验证及其结果、尚未验证什么。区分单元/集成测试通过、离屏渲染通过、Windows 实机操作通过、真实数据结果验证通过。

报告和截图写入 output/；仅小型可评审证据进入 docs/artifacts/。用户输入数据、模型和完整报告不提交。
