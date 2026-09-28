# MyGPR 组件状态矩阵（Component State Matrix）

> 依据 design-ui-designer 方法论：没有状态矩阵的组件是半成品。
> 色值一律引用 `ui/design_tokens.py` 语义令牌，本文档只记口径不记色号。

通用基准：focus-visible = 2px 实线 `border_focus` + 2px 偏移；disabled =
不透明度 0.6 且**真不可点**（非仅变灰）；过渡时长 fast 150ms / normal 300ms。

## 1. 结果卡（`_ResultCard`，结果网格）

| 状态 | 实现 | 来源 |
|---|---|---|
| default | 2px 透明描边（占位防跳动） | 本轮令牌化 |
| hover | 淡 primary 描边 0.35 + 微底 0.03（QSS `:hover`） | ✅ 本轮新增 |
| active/pressed | 点击即选中（sig_clicked），无独立按压态 | 设计取舍：点选语义由选中态承担 |
| selected | primary 0.85 描边 + 0.06 淡底 | 已有（令牌化） |
| focus-visible | 2px border_focus 描边（QSS `:focus`）+ Tab 可达 + Enter/Space 选中 | ✅ 已实现 |
| disabled | N/A（结果卡无禁用语义） | — |
| loading | 骨架屏 `_Skeleton` + 交叉淡入 `_reveal`（预留尺寸防跳动） | 已有 |
| error | 预览失败由宿主 InfoBar 反馈；卡片保持骨架 | 已有 |
| empty | 网格空态浮层（「选择测线后显示…」） | 已有 |

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
| loading | 运行中 `运行中·` 旋转点动画 + 取消按钮点亮 |
| error | 运行失败经任务中心状态徽章（error 色 + 文案，非仅颜色） |
| focus-visible | qfw 焦点样式（系统焦点环） |

## Backlog（后续做）

- 焦点顺序走查（Tab 顺序 = 视觉顺序的逐页验证）。
- 错误态卡片内联提示（当前错误只走 InfoBar，刷新后无残留痕迹）。
