# MyGPR v0.9.28 Spatial Closure V1.6

## 目标

将空间成果页从实时绘图与零散 CSV 导出，升级为正式、不可变、可追溯、可重复导出的 GIS 成果版本。

## 新增

- `mygpr.spatial_result.v1` 成果清单。
- `spatial/results/index.json` 当前版本索引。
- 空间成果预检与曲面覆盖门禁。
- 来源 SHA-256、文件大小、修改时间和快速过期检测。
- 空间成果版本树、来源属性与正式导出入口。
- ZIP、GeoJSON、CSV、KML、GeoPackage、Shapefile 导出。

## 兼容性

- 旧的逐测线空间 CSV 继续保留并可被读取。
- 新版本生成时会复制为独立快照，后续标注或处理变化不会修改旧成果。
- 主版本号仍为 v0.9.28，本文件描述 UI/业务增量 V1.6。
