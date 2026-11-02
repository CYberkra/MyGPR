# MyGPR 研发技能工作流

2026-11-02 维护。此文件通过根 AGENTS.md 路由使用；不是运行时插件，不向软件注入算法或 UI 功能。

## 已有安装与选择规则

优先使用当前会话技能目录中同名入口，首次使用阅读完整 SKILL.md 及任务所需引用。当前机器位置如下，其他机器使用自身用户目录，不硬编码用户名到脚本。

| 技能 | 本机来源 | 用途 |
| --- | --- | --- |
| frontend-design-polish | `%USERPROFILE%/.codex/skills/frontend-design-polish/SKILL.md` | 科学桌面界面的布局、密度、常用操作和视觉检查 |
| code-context-finder | `%USERPROFILE%/.codex/skills/code-context-finder/SKILL.md` | 注册表、执行器、调用方、轴与元数据合同定位 |
| engineering-feedback-loop | `%USERPROFILE%/.codex/skills/engineering-feedback-loop/SKILL.md` | 缺陷假设、最小复现与反馈路径；默认把执行交给 CI |
| windows-desktop-e2e | `%USERPROFILE%/.agents/skills/windows-desktop-e2e/SKILL.md` | pywinauto/UIA 桌面操作测试及失败截图 |
| python-testing | `%USERPROFILE%/.agents/skills/python-testing/SKILL.md` | pytest 夹具、参数化、异常与回归测试 |
| codebase-design | `%USERPROFILE%/.agents/skills/codebase-design/SKILL.md` | 接口职责、模块重构、可测试性 |
| scientific-critical-thinking | `%USERPROFILE%/.codex/skills/scientific-critical-thinking/SKILL.md` | 需要解释科学效果时，区分物理证据、数值变化和显示变化 |
| scientific-visualization | `%USERPROFILE%/.codex/skills/scientific-visualization/SKILL.md` | 雷达对比图、统一色阶、单位、缺失区域与可追溯图件 |
| requesting-code-review | `%USERPROFILE%/.codex/skills/requesting-code-review/SKILL.md` | 按明确范围进行独立代码审查；子代理同样不默认运行本地测试 |

优先使用当前会话可见的技能。找不到时报告具体缺失，不自动恢复用户主动删除的技能，
也不把某个专用技能设为执行项目工作的硬前提。

不重复复制、不覆盖全局技能、不自动升级第三方包。`.codex/skills` 与 `.agents/skills`
保持各自目录，不合并。需要新技能时先审阅来源与适用范围，不安装整套无关技能。

## 实际工作方式

### 界面与交互

1. 用 frontend-design-polish 明确用户任务与关键操作路径，先解决操作步骤和空间分配。
2. 设计任务给出状态、空白/失败情况、小屏行为及验收条件；只做设计时不改功能代码。
3. 实施任务复用 PyQt6/qfluentwidgets 与项目控制器，不把网页框架示例直接搬进桌面应用。
4. 用现有 Qt 测试验证状态和信号；涉及实际点击、弹窗、焦点、滚动、DPI 时再使用 windows-desktop-e2e。
5. 桌面测试先检查控件 UIA 可访问性。自绘 B-scan/pyqtgraph 可能不能作为普通 UIA 控件操作，结合 Qt 级状态测试与截图/实机检查，不能仅凭技能中的通用 Qt 支持描述宣称可自动操作全部图形。

### 算法和处理链

1. 沿注册表 → 参数准备 → 执行器 → 成果保存/回读定位实际执行链，确认数据类型、输入形状、轴单位、频带和传感器元数据；依据项目合同与输入元数据，不能用技能示例默认值替代。
2. 记录输入、步骤顺序和参数，以小规模可复现数据建立基线。
3. 按问题验证 shape、有限值、幅度分布、目标区域及前后变化；独立自动色阶的图片不能证明振幅改善。
4. 用 python-testing 编写真实路径与关键异常的回归；必要时固定数据对照 CLI 和 GUI。默认在 GitHub CI 执行，未经用户当前任务明确要求不启动本地测试。
5. 修改注册算法时同时核对方法注册表、参数 UI、CLI 与适用测试。运行昂贵仿真、批量处理或大规模性能试验仍须符合用户当前任务范围。

### 排错与重构

- 排错：用 engineering-feedback-loop 建立 CI 可执行的复现 → 收集状态/线程/数据证据 → 验证原因 → 最小修复 → 远端回归。不能仅通过改测试期待值消除失败。
- 重构：codebase-design 用于划分接口与职责；遵守 ui/controllers、facade、application/domain/infrastructure 的现有边界。不为一个小改动引入多层抽象。
- 修复期间需要测试时加载 python-testing；测试针对外部行为、错误路径与曾经失败的案例，不为了数值覆盖率添加镜像实现的测试。

## 验证入口

自动化测试默认交给 GitHub Actions：PR / 主分支推送运行 `MyGPR CI` 完整门禁；
`MyGPR focused regressions` 可按 B-scan、算法、处理流程选择快速回归。
新定向工作流须先进入默认分支，随后可在 Actions 中选择分支执行；首次 PR 由原有完整门禁覆盖新用例。
具体命令与证据边界见 [测试策略](../developer/TESTING_STRATEGY.md)。

本机默认仅代码阅读、差异审查和静态检查。技能中的 TDD/测试运行建议不得覆盖用户对执行位置的约定。
新增用例未在远端运行时明确标为“待 CI”，不能用旧 run 或静态检查替代通过结论。

桌面 E2E 在独立临时项目和小样本上执行：导入 → 搜索/添加算法 → 设置参数 → 运行 → 查看输出 → 切换测线 → 关闭；不要接管用户正在处理真实项目的窗口。等待信号/状态而非固定睡眠；失败留截图与日志。pywinauto 可导入不等于该流程已经验证。

Windows 原生显示、真实设备和私有实测数据验收须单独安排；未得到当前任务的本机测试授权就保持待验收，
不为 CI 方便上传用户原始数据。托管 CI 的离屏 Qt 测试不能替代真实显卡/传感器验收。

## 其他候选技能

已调研 OpenAI 官方 [`gh-fix-ci`](https://github.com/openai/skills/tree/main/skills/.curated/gh-fix-ci)：
适合定位 PR 的 GitHub Actions 失败、读取日志、修复后核对同一提交的检查。
它是 CI 排障工具，不是雷达科学验证工具；当前只记录候选，未安装。
日常工作优先使用上表已有技能，科学数据合同留在项目内维护。

## 交付证据

每次报告：改了什么、为什么、实际运行了哪些验证及其结果、尚未验证什么。区分单元/集成测试通过、离屏渲染通过、Windows 实机操作通过、真实数据结果验证通过。

报告和截图写入 output/；仅小型可评审证据进入 docs/artifacts/。用户输入数据、模型和完整报告不提交。
