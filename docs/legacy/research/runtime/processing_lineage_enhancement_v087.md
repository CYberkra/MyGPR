# V0.8.7 处理链路增强说明

V0.8.7 将主页面 B-scan 下方的处理链路从简单标签升级为可审计 stepper。

## 状态语义

- `Raw`：原始输入。
- `✓`：已成功应用的历史步骤。
- `当前`：正式当前结果。
- `查看中`：正在 display-only 临时查看历史步骤。
- `⚠`：该步骤存在 runtime warning。
- `裁剪`：完整数组已因历史内存策略裁剪，只保留 summary-only 审计信息。

## 交互

- 点击历史步骤：临时查看该步骤 B-scan，不修改正式当前结果。
- 点击当前步骤：返回正式当前结果。
- 对比：将选中历史步骤和当前结果送入已有滑动对比机制。
- 复制链路：复制报告友好的处理链路文本。

## Evidence 对齐

`ReportExportController` 现在优先调用 `ProcessingLineageController.build_export_steps()` 构建报告包中的 processing-chain 记录，因此 UI 中看到的步骤状态、参数、warning 和 memory state 会同步进入 Evidence sidecar。

## 非目标

本版本不做算法链路的拖拽重排、插入中间步骤、删除历史步骤或实时重算。这些属于后续 workflow editing 功能，应在 AutoTune / gprMax 评分闭环稳定后再设计。
