# MyGPR 代码健康记录 v0.8.81

- 项目向导和导入校验已抽到 `core/field_import_preview.py` / `core/field_project_operations.py`，未继续把解析逻辑堆进 UI 回调。
- 最近项目入口已从仅记录升级为项目管理页可操作入口。
- 剩余风险：厂商格式仍需专项适配；项目元数据表单还未覆盖坐标系统、设备型号等完整工程字段。
