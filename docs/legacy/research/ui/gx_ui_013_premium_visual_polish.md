# GX-UI-013 Premium Visual Polish

本轮只做高级视觉精修，不改变 MyGPR 的业务逻辑、数据处理算法、AutoTune 生产评分、gprMax 或 Evidence 链路。

## 视觉调整

- 强化浅色产品化主题：柔和背景、白色卡片、淡边框、蓝色主强调。
- 统一主工作区和 AutoTune 页的状态 chip 视觉，并增加中性/成功/警告/危险 tone。
- 优化主 B-scan 工作区卡片质感、空状态、输入框、表格、按钮和标签页样式。
- 优化 AutoTune 参数推荐控制台：
  - header 更柔和；
  - chip 更清晰；
  - 预览卡片更舒适；
  - 推荐、风险、结论边界区域更统一；
  - 英文标签进一步中文化。

## 未改变内容

- 不运行 gprMax。
- 不修改 GX-008/GX-009 模型。
- 不修改 MyGPR-Evidence。
- 不引入 PyVista/PyVistaQt。
- 不删除 legacy 页面。
- 不修改生产 AutoTune 评分逻辑。
- 不启用真正的 AutoTune 生产执行。

## 当前边界

AutoTune 页仍是 UI-local 推荐闭环 MVP：可以展示数据状态、ROI、候选空间、候选排名和风险提示，但推荐结果不代表生产评分结果，也不代表全局最优 workflow。
