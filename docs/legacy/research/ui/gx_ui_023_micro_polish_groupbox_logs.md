# GX-UI-023 Micro Polish: GroupBox Titles and Runtime Logs

本轮针对实际截图中暴露出的细节做微调，不改变任何业务逻辑。

## 修复点

- QGroupBox 标题不再像白色小贴片一样压在边框上。
- 右侧页面中的“方法与常用参数”“查看顺序”等小节标题更自然地嵌入卡片。
- 全局日志 / 质量摘要文本区去掉过强的蓝色 focus 边框。
- QTextEdit / QPlainTextEdit 的背景、边框、viewport 统一。
- runtime drawer 和小型二级按钮的视觉更柔和。
- 保持浅色 / 深色主题变量一致。

## 未改动

- 不修改处理算法。
- 不修改 AutoTune 生产评分逻辑。
- 不运行 gprMax。
- 不修改模型和 Evidence。
- 不引入新依赖。
