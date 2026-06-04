# GX-UI-021 Visual Language Polish

本轮只做视觉语言统一与轻量美术精修，不改变处理算法、AutoTune 生产评分、gprMax、Evidence 或 3D viewer。

## 改动范围

- 主 B-scan 空状态改为更克制的产品化空状态：
  - 小型波形图标
  - 主/次动作按钮
  - 简洁三步引导
- AutoTune 候选空间改为 checklist row：
  - 方法名称
  - 方法说明
  - baseline / fast / robust / rank sweep / experimental 标签
- SVD rank 控件改为横向 compact rank panel。
- 全局主题补充候选行、方法标签、rank panel、主/次按钮样式。
- AutoTune 本地样式补充同样控件，确保浅色/深色模式一致。

## 未改动

- 不运行 AutoTune。
- 不修改 AutoTune 生产评分逻辑。
- 不运行 gprMax。
- 不修改 GX-008 / GX-009 模型。
- 不修改 MyGPR-Evidence。
- 不引入 PyVista / PyVistaQt。
- 不恢复旧工作台。
