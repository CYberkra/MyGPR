# MyGPR v0.8.88 CSV 导入恢复

本轮恢复旧 MyGPR 航空 GPR 主数据 CSV 解析规则。

## 支持格式

```text
Number of Samples = N
Time windows (ns) = T
Number of Traces = M
Trace interval (m) = dx
lon,lat,elevation,amplitude,height,timestamp
...
```

## 标准化结果

- `raw/<line_id>/<line_id>_gpr_dataset.npz`
- `raw/<line_id>/<line_id>_gpr_meta.json`
- `raw/<line_id>/<line_id>_trajectory.csv`
- `raw/<line_id>/import_manifest.json`

## 说明

该格式为旧 MyGPR 实测主 CSV。第 4 列 amplitude 会按 trace-major 顺序重塑为
`(Number of Samples, Number of Traces)` 的 B-scan 矩阵。
