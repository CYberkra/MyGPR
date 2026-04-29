# GUI RTK/IMU Sidecar 验证数据包

这个目录用于验证“普通 CSV 导入不受影响 + 可选 RTK/IMU sidecar 选择入口可用”。

## 文件

- `gui_sidecar_all_data_main.csv`：GUI 主导入文件，空中堆叠格式：longitude, latitude, ground_elevation_m, amplitude, flight_height_m, trace_timestamp_s。第 6 列是显式 trace 时间戳。
- `gui_sidecar_all_data_rtk.csv`：RTK sidecar，包含 timestamp_s、经纬度、高程、飞行高度、fix、卫星数、HDOP。
- `gui_sidecar_all_data_imu.csv`：IMU sidecar，包含 timestamp_s、roll/pitch/yaw 和角速度。
- `gui_sidecar_all_data_trace_timestamps.csv`：每道 trace 的参考时间戳，用于人工核对。
- `gui_sidecar_all_data_combined_reference.csv`：合并参考表，包含每个 trace/sample 的主数据、RTK、IMU 字段，便于核对。

## GUI 验证步骤

1. 启动 `app_qt.py`。
2. 进入 `显示与对比` → `增强与性能` → `可选 RTK/IMU 辅助文件`。
3. 分别选择：
   - RTK：`sample_data/gui_sidecar_all_data_rtk.csv`
   - IMU：`sample_data/gui_sidecar_all_data_imu.csv`
4. 导入主 CSV：`sample_data/gui_sidecar_all_data_main.csv`。

这份主 CSV 已在第 6 列提供显式 `trace_timestamp_s`，所以不需要 GUI 推导时间戳；RTK/IMU sidecar 应能进入融合链路。

如果仍看到“缺少 trace_timestamps_s”，说明正在运行的不是当前已更新代码，或导入的不是这份 6 列主 CSV。
