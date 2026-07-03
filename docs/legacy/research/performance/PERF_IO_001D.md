# PERF-IO-001D：导入与内存 copy 审计

版本：0.8.58

## 范围

本轮只处理导入路径和内存审计：CSV / A-scan / common GPR loader 的数组 dtype、内存摘要、progress 刷新频率和导入日志。没有修改任何处理算法、AutoTune 评分、候选参数生成器、gprMax 数据链路或 Evidence schema。

## 已完成

1. 新增 `core/io_performance.py`：
   - `summarize_array_memory()`：记录 shape / dtype / nbytes / contiguous。
   - `choose_csv_read_dtype()`：矩阵型 CSV 使用 float32；带 header 或 sidecar 的航空 CSV 保持 pandas 推断，以保留经纬度/时间戳精度。
   - `csv_import_context()`：记录文件大小、是否有 header / sidecar、CSV 读取 dtype。
   - `sanitize_float32_matrix()`：集中完成 float32 转换、2D 规范化和 NaN/Inf 填充审计。

2. CSV 导入路径：
   - 矩阵型 CSV 在 chunked read 阶段直接使用 float32，减少临时 float64 内存压力。
   - header/sidecar 数据不强制 float32，避免损失经纬度与时间戳精度。
   - 读取完成后把 `import_context` 和 `import_memory_summary` 写入 `header_info`。

3. 加载对话框：
   - 对 progress signal 做约 80 ms 的节流。
   - 大文件分块读取时减少 QLabel/QProgressBar 过度刷新。

4. 数据加载完成日志：
   - 输出 dtype、数组大小 MB、C-contiguous 状态。

## 风险边界

- 矩阵 CSV 的算法输入仍是 float32，与既有处理数组 dtype 约定一致。
- Airborne stacked CSV 和带 sidecar 的 CSV 不在读取阶段强制 float32。
- 本轮不改变数据 shape、trace metadata 合并规则、header 解析或 sidecar 逻辑。

## 后续建议

- 继续审计 `read_ascans_folder()` 是否存在逐文件重复 copy。
- 对特别大的 CSV 加入 preflight 文件大小提示和取消响应。
- 将导入耗时写入统一 perf summary export。
