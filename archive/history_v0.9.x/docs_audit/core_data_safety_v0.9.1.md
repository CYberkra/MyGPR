# MyGPR v0.9.1 核心数据安全修复记录

## 背景

本轮来自 v0.9.0 Pass 2 核心数据链路逐行审计，优先修复 P1 问题。

## 修复项

1. CGCS2000 3-degree Gauss-Kruger Zone EPSG 映射修正：Zone 39 -> EPSG:4527，Zone 36 -> EPSG:4524。
2. 新增 `validate_line_id()`，限制测线编号只能使用安全 ASCII 标识符，阻止路径穿越和 Windows 保留设备名。
3. `import_line_data()` 改为事务式导入；失败时恢复 `project.json` 测线清单，并清理或恢复 `raw/<line_id>/`。
4. `load_trajectory()` 不再返回 demo 轨迹；缺失轨迹显式抛出 `FileNotFoundError`。
5. B-scan 转置修正 manifest 增加轴重建策略和工程交付警告。

## 影响范围

- 不改变 CSV 解析格式。
- 不改变算法入口。
- 不新增 UI 功能。
- 对正式项目更加严格；非法 line_id 会直接拒绝。

## 验证

```text
python -m compileall . -q
python scripts/check_version_consistency.py --expected 0.9.1
python -m pytest tests/test_core_data_safety_v091.py tests/test_beta_prep_v0900.py tests/test_data_quality_bscan_v098.py tests/test_coordinate_projection_v094.py tests/test_yingshan_csv_import_v091.py tests/test_mygpr_sidecar_csv_import.py -q
```

结果：22 passed。
