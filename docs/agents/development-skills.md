# MyGPR 研发技能工作流

2026-11-04 维护。此文件通过根 AGENTS.md 路由使用；不是运行时插件，不向软件注入算法或 UI 功能。

## 已有安装与选择规则

优先使用当前会话技能目录中同名入口，首次使用阅读完整 SKILL.md 及任务所需引用。当前机器位置如下，其他机器使用自身用户目录，不硬编码用户名到脚本。

| 技能 | 本机来源 | 用途 |
| --- | --- | --- |
| frontend-design-polish | `%USERPROFILE%/.codex/skills/frontend-design-polish/SKILL.md` | 科学桌面界面的布局、密度、常用操作和视觉检查 |
| interface-design | `%USERPROFILE%/.codex/skills/interface-design/SKILL.md` | 跨页信息层级、视觉规范、密度与组件一致性 |
| design-system | `%USERPROFILE%/.agents/skills/design-system/SKILL.md` | 提取或审计已有视觉规范；Web 示例需适配 QSS/Qt |
| ui-ux-pro-max | `%USERPROFILE%/.codex/skills/ui-ux-pro-max/SKILL.md` | 配色、字体、可访问性与交互规则的本地专项检索 |
| click-path-audit | `%USERPROFILE%/.agents/skills/click-path-audit/SKILL.md` | 追踪点击、信号、控制器与异步回调的最终状态 |
| gh-fix-ci | `%USERPROFILE%/.codex/skills/gh-fix-ci/SKILL.md` | PR 的 GitHub Actions 日志分析、失败定位与同提交复核 |
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

1. 日常 PyQt 布局与实现用 frontend-design-polish；跨页视觉规范与层级设计选 interface-design；提取或审计已有规范可选 design-system。按主任务选一个设计流程，先明确用户任务、关键操作路径和空间分配，不叠加完整流程。
2. 设计任务给出状态、空白/失败情况、小屏行为及验收条件；只做设计时不改功能代码。
3. 实施任务复用 PyQt6/qfluentwidgets 与项目控制器，不把网页框架示例直接搬进桌面应用。
4. 用现有 Qt 测试验证状态和信号；涉及实际点击、弹窗、焦点、滚动、DPI 时再使用 windows-desktop-e2e。
5. 桌面测试先检查控件 UIA 可访问性。自绘 B-scan/pyqtgraph 可能不能作为普通 UIA 控件操作，结合 Qt 级状态测试与截图/实机检查，不能仅凭技能中的通用 Qt 支持描述宣称可自动操作全部图形。

ui-ux-pro-max 仅在有具体设计疑问时加载，优先使用 `--domain ux/color/typography/chart`
中的一个域检索。当前版本没有 PyQt 栈，不指定虚构的 `--stack pyqt`，也不默认降级到
Tailwind/React。触控 44px、16px 正文、mobile-first 等是来源中的场景建议，不能覆盖
本项目的桌面布局合同；科学图件的轴、单位、振幅与色阶语义始终优先。

使用安装位置中的 `scripts/search.py`，按当前 SKILL.md 的父目录解析绝对路径。
本机安装已将 Claude 专用变量改为技能目录占位说明；无需设置 `CLAUDE_PLUGIN_ROOT`。
查询默认不加 `--persist`。需要保留设计结论时更新现行规格及其索引，避免自动生成
`.interface-design/system.md`、`DESIGN.md` 或 `design-system/*/MASTER.md` 成为重复合同；
已有这些文件时先读取并核对现行规格，不自动覆盖。

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
- 操作状态问题用 click-path-audit 追踪：用户操作 → Qt 信号 → 控制器/协调器 → 异步完成或取消 → 页面最终状态。检查后续回调是否覆盖先前状态、旧任务是否写回新测线；不引入示例里的 React 状态库。

### GitHub Actions 排障

1. 用 gh-fix-ci 时先确认 `gh auth status`，再读取指定 PR、失败检查和日志；不因安装技能自动触发工作流。
2. 记录 PR、head SHA、run URL、失败步骤与原因，区分产品缺陷、用例问题和 CI 环境问题。
3. 按当前会话已授权范围修复；技能里的确认提示不重复索取已有授权。提交、推送、合并仍按当前用户授权范围执行。
4. 回归执行遵守本项目的 GitHub Actions 约定。复核修复提交对应的检查；未运行、仍排队或旧 SHA 的成功都不能写成修复已通过。
5. 此技能负责 CI 排障，不替代真实数据科学验证和 Windows 原生桌面验收。

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

## 新增技能的安装来源

2026-11-04 经用户授权，用 Codex 自带 skill-installer 安装以下三个技能到用户级目录。
仅安装选定技能目录，不安装整套插件、hooks 或常驻服务；保留原有技能。
新技能在安装后的下一轮对话可用，也可用 `$interface-design`、`$gh-fix-ci`、
`$ui-ux-pro-max` 显式指定。下列为安装时固定的来源，不表示持续自动更新。

| 技能 | 上游仓库与固定提交 | 安装子目录 |
| --- | --- | --- |
| interface-design | [Dammyjay93/interface-design @ 2f9be320](https://github.com/Dammyjay93/interface-design/tree/2f9be3206855bcb2d1d0af262c8bae25cba6658d) | `.claude/skills/interface-design` |
| gh-fix-ci | [openai/skills @ 49f948fa](https://github.com/openai/skills/tree/49f948faa9258a0c61caceaf225e179651397431) | `skills/.curated/gh-fix-ci` |
| ui-ux-pro-max | [nextlevelbuilder/ui-ux-pro-max-skill @ 713eef51](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill/tree/713eef51b04efc98d6fa565323b495f7c9c89fd4) | `.claude/skills/ui-ux-pro-max` |

每个新增技能目录中的 `INSTALLATION.json` 记录来源、入口 SHA-256 和本地适配。
ui-ux-pro-max 仅适配命令示例的技能路径并补充 Codex 展示入口，检索脚本和数据库保持上游原样。
本次未安装 Impeccable；其他技能也不随这三个技能自动升级。

## 交付证据

每次报告：改了什么、为什么、实际运行了哪些验证及其结果、尚未验证什么。区分单元/集成测试通过、离屏渲染通过、Windows 实机操作通过、真实数据结果验证通过。

报告和截图写入 output/；仅小型可评审证据进入 docs/artifacts/。用户输入数据、模型和完整报告不提交。
