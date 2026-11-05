# 启动页实施计划

> 2026-11-04：用户已确认精简方向，并授权在深浅主题齐备后直接实施。

**Goal:** 将未打开项目的界面改为可搜索、可固定、有选中详情的项目启动台。
**Architecture:** HomePage 纯展示并沿用协调器打开链；目录探测独立 controller + run_command，代次隔离且合并刷新；明暗颜色来自 design_tokens。
**Tech stack:** PyQt6 / qfluentwidgets，现有 SettingsManager。
**Spec:** ../specs/2026-11-04-startup-page-design.md，首轮范围。

## 约束与评审重点

- 保留此前 settings_manager 及其回归修改，保持 startup / empty / ledger 和 homeInterface 路由。
- 单击、筛选、固定不打开项目；双击、Enter、按钮走同一打开入口；忙碌及后端门控一致。
- 不在 GUI 线程探测目录；忽略已移除记录和过期回包；自动恢复不抢占用户的新选择。
- 首轮不实现设计中的后续重新定位、拖放与启动偏好，也不展示无效按钮。
- 本机只做静态检查。按 AGENTS.md，回归执行及截图由 CI / 后续明确授权的 Windows 验收完成。

## Tasks

- [x] 1. 新增 `ui/controllers/recent_projects_controller.py`，后台检查、刷新合并、取消和过期保护；新增控制器回归。
- [x] 2. 新增 `ui/widgets/recent_project_list.py`，重构 `ui/pages/home_page.py`，实现双主题、响应布局、搜索/固定/详情/状态。
- [x] 3. 接入 `ui/coordinator_project.py` 的忙碌门控及 `ui/main_window.py` 的轻量导航、后台自动恢复。
- [x] 4. 更新 HomePage、项目工作区、恢复与导航回归；执行 Ruff、语法和架构等静态检查；整体独立代码评审。

## 验证记录

本机 Ruff、Python 语法、架构、复杂度与债务预算检查通过。独立评审发现菜单重启探测可能抢占自动恢复，以及 Fluent 标签自有样式覆盖父层颜色，两项已修复并增加回归；主窗口降级页调用增加 callable 守卫。固定/取消固定保持最近打开历史顺序。前一轮 settings_manager 修改保持独立归属。

测试已编写，未在本机执行；当前未提交/未推送，CI 待执行，无本轮 run URL。Windows 原生焦点、明暗截图及 DPI 验收待执行。未以静态检查或设计示例截图代替这些结果。

交互链复核：通过 click-path-audit 追踪新建/打开、选择/激活、搜索/固定、移除/复制、菜单和恢复回包。补齐了项目行右键及 Shift+F10 菜单取消自动恢复的遗漏（与顶部项目菜单共用行为），并扩大现有恢复竞态回归覆盖。新增回归仍待 CI。
