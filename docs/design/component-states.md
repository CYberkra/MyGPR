# MyGPR 组件状态矩阵（Component State Matrix）

> 依据 design-ui-designer 方法论：没有状态矩阵的组件是半成品。
> 色值一律引用 `ui/design_tokens.py` 语义令牌，本文档只记口径不记色号。

键盘焦点使用 `border_focus` 轮廓。禁用状态移除交互能力。时长由
`ui/design_tokens.py` 按用途定义。设置中的“减少动态效果”使正在进行的过渡直接落位。

## 1. B-scan 数据标签与拖放

| 状态 | 实现 | 来源 |
|---|---|---|
| selected | 剖面随来源直接切换，2 px 下划线从屏幕位置续接，160 ms 到达目标。恢复布局时直接对齐 | `BScanWorkspace` |
| focus-visible | 使用独立焦点样式，键盘可选中相邻来源与打开菜单 | `_SourceTabBar` |
| drop target | 文字与边框立即可见，底色首次淡入 100 ms。Esc、松手、关闭目标组立即清场 | `_DropHint` |
| loading / error / empty | 画布内给出当前来源的请求、失败原因和重试，或打开数据的入口 | `_PreviewStatus` |

## 2. 处理链 chip（`_Chip`）

| 状态 | 实现 | 来源 |
|---|---|---|
| default | 中性描边 0.22 + 淡底 0.16，胶囊圆角 | 已有（令牌化） |
| hover | 描边转 primary 0.45 + 淡底 0.08（可点性预示） | ✅ 本轮新增 |
| selected | 滑动胶囊（primary 0.20 淡底，QPropertyAnimation） | 已有（令牌化） |
| disabled | 虚线描边 0.30 + 淡底 0.08 + 次级文字；圆点置灰； | 已有 |
|            | 「真不可点」由信号宿主 `sig_step_toggled` 语义保证 | |
| focus-visible | 列表 `:focus` 描边环（上下键导航由 QListWidget 承担，当前行由胶囊指示） | ✅ 已实现 |
| loading | N/A（运行中反馈由运行按钮承担） | — |
| error/empty | N/A | — |

## 3. 开关（qfluentwidgets SwitchButton：统一色标 / 全部步骤）

| 状态 | 实现 |
|---|---|
| default / hover / checked / disabled | qfw 内置，样式随 `setThemeColor` |
| 文本 | 标签恒定（`setOnText(标签文本)`），无 On/Off 尾随 |
| 语义 | 勾选即生效并落盘（general_changed / bscan_view_changed） |

## 4. 主按钮（运行 / 取消 / AutoTune）

| 状态 | 实现 |
|---|---|
| default / hover / pressed / disabled | qfw PrimaryPushButton 内置 |
| loading | “运行中…”与“取消任务”显示在处理工具栏。已知总量的进度在 200 ms 内续接，未知总量显示阶段文字 |
| cancelling | 提交取消后，原位置和任务行显示“正在取消…”。作业事件确认后更新终态 |
| error | 失败使用可读徽章与任务详情。每项作业的失败提示由一个入口负责 |
| focus-visible | qfw 焦点样式（系统焦点环） |

## Backlog（后续做）

- 焦点顺序走查（Tab 顺序 = 视觉顺序的逐页验证）。
- 错误态卡片内联提示（当前错误只走 InfoBar，刷新后无残留痕迹）。
