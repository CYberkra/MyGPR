# GX-UI-031~035 Final UI Polish

本轮完成更深一层 UI 收口：

- AutoTune 页本地 QSS 完整迁入 `ui/theme.py`。
- 主 B-scan 工具栏与底部状态栏分离：
  - 顶部工具栏显示工具与显示状态 chip。
  - 底部状态栏显示处理链路和实时坐标。
- 日常处理页核心卡片增加 `cardStyle=modern`，弱化传统 QGroupBox 贴片感。
- 质量与导出页新增 Evidence 检查清单卡片。
- 保持旧工作台 retired shim，不恢复旧入口。

## 未改动

- 不修改处理算法。
- 不修改 AutoTune 生产评分逻辑。
- 不运行 gprMax。
- 不修改模型和 Evidence。
- 不引入 PyVista / PyVistaQt。
