# MyGPR v0.9.28 · V1.9.7 Basemap Visual Completion

- 空间成果页新增无需 Key 的 Esri World Imagery 在线影像预设。
- 在线底图设置保存后立即加载当前项目范围，不再需要二次点击。
- 修复带说明文字的工程坐标系（例如 `... (EPSG:4547)`）无法交给 GDAL/pyproj 的问题。
- GeoTIFF、MBTiles、在线瓦片、测线和界面曲线现在使用同一规范化 CRS。
- 新增真实栅格加载截图脚本及坐标系回归测试。
- 截图证据使用外部输入的真实航拍 GeoTIFF；影像不打包进源码和项目模板。
