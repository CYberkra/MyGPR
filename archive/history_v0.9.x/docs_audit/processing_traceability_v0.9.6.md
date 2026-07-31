# MyGPR v0.9.6 算法入口与处理结果保存审计/修复

## 范围

- `core/field_processing_bridge.py`
- `core/field_artifact_store.py`
- `core/processing_artifact_index.py`
- `ui/field_panels/processing_page.py`
- `ui/field_panels/processing_panel.py`

## 已修复问题

1. 处理结果保存原先只有最新参数文件，历史 artifact 在索引时可能指向最新参数。v0.9.6 改为每个 artifact 使用时间戳参数 sidecar。
2. 保存 manifest 缺少输出数据哈希。v0.9.6 增加输入/输出 SHA256 和 manifest/params SHA256。
3. 保存处理结果前缺少 line_id/manifest 一致性防线。v0.9.6 在 UI 保存入口中禁止跨测线保存。
4. 项目内保存存在 synthetic fallback 的历史残留。v0.9.6 已移除该 fallback。
5. Artifact 索引按时间戳匹配 data/params/manifest，避免历史结果误绑定。

## 未改变

- 未改变任何算法数学含义。
- 未新增算法。
- 未改变 CSV 导入、坐标投影、数据质检和 B-scan 方向修正逻辑。
