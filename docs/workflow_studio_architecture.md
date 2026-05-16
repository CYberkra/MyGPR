# Workflow Studio Architecture

本文档固定当前 MyGPR Workflow Studio 的职责边界，避免继续回到“旧表单页 + 新画布”混合状态。

## 用户界面分工

```text
Top Run Bar
  模板、运行、实时、安全、缩放、低频菜单

Left Rail / Panel
  项目数据、节点库、运行记录、调参、验证、导出

Workflow Canvas
  节点、端口、连线、Preview / Effect 节点、布局和 LOD

Inspector
  当前选中节点的完整参数、算法切换、输入输出、QC 和 warning

Bottom Runtime Drawer
  全局日志、Validation、QC/告警、Evidence、Export 结果
```

卡片只承担轻量编辑和状态摘要；完整诊断必须进 Inspector 或 Bottom Drawer。

## 当前执行模型

Workflow Studio 当前仍采用顺序执行：

```text
config.methods 按 order 排序运行
```

画布连线的职责是：

- 让用户看清处理意图。
- 支撑 Preview / Compare / QC / Evidence 节点。
- 在 Validate 中发现 graph/order mismatch。
- 为后续 DAG executor 保留数据结构。

不要在没有完整 DAG executor 测试前让连线直接决定处理顺序。

## 旧页面迁移策略

旧页面不要作为主入口长期存在，但也不要一次性删除：

- `BasicFlowPage`: 降级为导入和兼容后台能力。
- `AutoTunePage`: 作为 Tuning Lab 弹窗或面板进入。
- `AdvancedSettingsPage`: 迁入 B-scan Viewer / Preview Settings。
- `QualityLogPage`: 迁入 Bottom Drawer、Evidence、Export。

迁移完成的判断标准不是“旧类被删除”，而是用户在 Studio 里能找到同等功能入口。

## 代码拆分方向

当前大型文件仍需继续拆分：

- `app_qt.py`: 保留为应用 Shell，逐步抽出 project/runtime/preview/export controllers。
- `ui/gui_workflow_page.py`: 保留为 Studio Shell，逐步抽出 project panel、node palette、inspector、runtime drawer。
- `ui/workflow_canvas_cards.py`: 逐步拆分 node card、ports/edges、canvas view、context menus、preview nodes。

每次拆分必须保留旧 import 或 facade，直到测试迁移完成。
