# MyGPR UI 样式规范（Design Tokens v2）

> 来源：2026-09-21 UI 设计评审（像素君）§3 固化。单源原则：**token 只定义在
> `ui/constants.py`（值）与 `ui/theme_helpers.py`（随主题取值/QSS 工厂）**，
> 页面代码零字面量。新页面 / 改版 PR 按文末 checklist 自查。

## 1. 字号（五档 pt，1pt = 4/3 px @96dpi）

| Token | 值 | 用途 |
|---|---|---|
| `FONT_SIZE_CAPTION` | 9 pt | 图表轴刻度（`CHART_TICK_FONT_SIZE` 即它）、最次要注释 |
| `FONT_SIZE_SECONDARY` | 9 pt | hint / 徽章 / 辅助行（QSS 用，随主题查色） |
| `FONT_SIZE_BODY` | 10 pt | 正文、表单标签、列表项（全局默认，见 `app_qt.py`） |
| `FONT_SIZE_SECTION` | 12 pt Bold | 卡片 / 分组标题 |
| `FONT_SIZE_TITLE` | 14 pt Bold | 页级大标题（预留） |

规则：QSS 一律引用 token（`pt` 单位），**禁止 px 字面量**；历史 px 值映射
11px→9pt、12px→9pt、13px→10pt。例外：dock_panel 竖排指示沿用 px 等值
（12px/10px），改动会放大字号，暂保持。

## 2. 字体回退链

```python
FONT_FAMILY_STACK = ('Microsoft YaHei UI', 'Microsoft YaHei',
                     'PingFang SC', 'Noto Sans SC')
FONT_FAMILY = FONT_FAMILY_STACK[0]   # 兼容别名
```

- **QFont** 一律走 `theme_helpers.ui_font(size_pt, weight=None)` 工厂
  （内部 `setFamilies` 逐族匹配），勿手写 `QFont(constants.FONT_FAMILY, ...)`。
- **QSS** `font-family` 一律引用 `theme_helpers.font_families_qss()`
  （加引号族名 + `sans-serif` 关键字兜底），勿写死单族。
- YaHei UI 的数字/拉丁字形更紧凑，与数值密集的 GPR 数据更搭。

## 3. 间距（4pt 网格）

| Token | 值 | 用途 |
|---|---|---|
| `SPACE_1/2/3/4/6` | 4/8/12/16/24 | 唯一刻度，布局取档位值，不自造奇数间距 |
| `PAGE_MARGINS` / `PAGE_SPACING` | 24 / 16 | 页面四边 / 页面纵向 |
| `CARD_MARGINS` / `CARD_SPACING` | 16 / 12 | 卡片内距 / 卡内行距 |
| `PANEL_MARGINS` / `PANEL_SPACING` | 8 / 8 | 折叠面板内距 / 行距 |
| `FORM_LABEL_MIN_WIDTH` | 112 | 表单行标签列统一最小宽（跨卡值列对齐） |

## 4. 语义色（随主题查表，禁用裸常量）

文字色一律走 `theme_helpers.status_color(key)`，徽章走 `badge_colors(key)`
/ `make_badge(text, key)`：

| 键 | 浅色 | 深色 | 语义 |
|---|---|---|---|
| `success/warning/error/info` | Tailwind 语义 | 同色相高亮变体 | 状态文字 |
| `secondary` | `#6b7280`（AA 4.84:1） | `#a8b0bd` | **hint 专用**；hint 一律 `make_hint()` / `hint_qss()` |
| `disabled` | `#9ca3af` 恒值 | 同 | **仅真禁用控件**，不得当 hint |
| `ACCENT` | `#009688` | `#2dd4bf` | 选中态文字/线条/链接（`accent_color()`）；实色块用 `ACCENT_SOLID` |

## 5. 组件范式

- **卡片**：`make_card(title)`；主操作控件传 `header_action=` 进卡头行右侧
  （「header 放动作」模式，参照 `make_segment_card`），勿在卡体孤悬一行。
  `make_segment_card` 即其 `header_action=segment` 特例。
- **空态**：`ui/widgets/empty_state.py` 的 `EmptyStateOverlay(host, icon=,
  title=, hint=)` 盖在画布/表格上，数据到达即隐藏。文案模式
  「〔什么数据〕+〔会出现在这里〕」，不写「暂无数据」死胡同文案。
- **hint**：`make_hint(text, parent=, key='secondary')`；运行期改色用
  `HintLabel.set_hint_key(key)`，勿手工 setStyleSheet。
- **徽章**：`make_badge(text, key)`；QSS 模板单源 `BADGE_QSS`（单占位，
  供 `motion.animate_badge_color` 正则回填）与 `BADGE_QSS_PAIR`（双占位）。
- **图内标题**：卡头即标题，pyqtgraph 图内不留 title；数据身份标题
  （如测线名）经 `BScanView._export_title` 机制保留并复用给导出。
- **顶栏**：药丸页签居中，右端固定为 文件树 / 输出面板 双开关
  （`main_window._create_top_nav_bar`），新增顶栏控件需在此模式内安置。

## 6. 主题接入

控件需随主题刷新时实现 `apply_theme(dark: bool)`（鸭子类型，主窗口
`findChildren` 遍历自动调用），内部用 `control_palette(dark)` /
`status_color()` 取值；勿在构造期写死颜色。

## 7. 新页面 / 改版 PR Checklist

- [ ] 字号全部来自五档 token，QSS 用 pt，无 px 字面量
      （`grep -nE "font-size:[^}]*px" ui/` 应仅剩 dock_panel 白名单）
- [ ] 字体经 `ui_font()` / `font_families_qss()`，无 `FONT_FAMILY` 直引
- [ ] 间距取 4pt 网格档位值，无自造数值
- [ ] 文字色走 `status_color()` / `hint_qss()` / `make_badge()`，无
      `setStyleSheet("color: #...")` 手写色
- [ ] 表单行用 `make_form_row()`（标签列 112px 自动对齐）
- [ ] 卡片用 `make_card()`；主操作在卡头 `header_action`，不孤悬
- [ ] 画布/表格空态有 `EmptyStateOverlay`，文案含前置条件说明
- [ ] 随主题变色的控件实现 `apply_theme(dark)`，颜色查表不自持
- [ ] 回归：`pytest tests/ -q` 全绿 +
      `QT_QPA_PLATFORM=offscreen python app_qt.py --smoke` 9 图无布局位移
      （offscreen CJK 渲染为方块属已知限制，只看布局/配色/间距）
