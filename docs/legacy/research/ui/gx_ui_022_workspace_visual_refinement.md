# GX-UI-022 Workspace Visual Refinement

本轮继续做轻量 UI 视觉优化，不改变算法和业务逻辑。

## 改动

- 主 B-scan Matplotlib 画布增加主题感知 polish：
  - figure/axes 背景跟随浅色/深色主题
  - 坐标轴、标题、tick、spine 色彩统一
  - 子图边距收紧，让 B-scan 更充分利用卡片空间
- 主图运行信息抽屉高度从 220 降到 180，减少对 B-scan 的挤压。
- 全局主题补充 splitter、toolbar、runtime drawer、plot chip 等细节样式。
- AutoTune 页面微调 header 和推荐文本卡高度，减少右侧栏压迫感。

## 未改动

- 不修改处理算法。
- 不修改 AutoTune 生产评分逻辑。
- 不运行 gprMax。
- 不修改 GX-008 / GX-009 模型。
- 不修改 MyGPR-Evidence。
- 不引入 PyVista / PyVistaQt。
