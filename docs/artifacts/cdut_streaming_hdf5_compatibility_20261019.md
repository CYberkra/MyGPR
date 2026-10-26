# CDUT 1.1.10 HDF5 导入兼容验证

日期：2026-10-19。MyGPR 基线 `e379a5e`，当前工作分支 `codex/maintenance-20261017`。
本次只适配处理端文件读取，保留既有未提交维护修改；没有更改采集协议、仪器配置或用户输入文件。

## 修复范围

- `fdata` 原先被拒绝；现按原始显示采样读取，保持该标签，预检提示物理轴和幅值含义需要结合采集配置核对。
- 道×样点转为样点×道；保留原先 `real`、`magnitude` 及通用矩阵 HDF5 读取路径。
- 原先仅四个位置/高度字段进入处理接口；现还保留 HWT 姿态、加速度（g）、角速度（度/秒）、UTC 毫秒时间和 uint64 道号。
  提供处理端角度字段和秒单位道时间，不用接收时间补造源时间，不补零或丢掉缺测的雷达道。
- 存在的逐道字段检查形状和数值类型；提供 `committed_trace_count` 时必须与数据行数一致。
  此检查不提供实时跟随活跃文件的能力。
- 保留测次 UUID、模拟数据标记。已有正式传感器同步结果继续优先于采集快照。
- 显式布尔 `time_axis_endpoint` 属性控制时间轴端点；没有属性时沿用旧版 `window/N` 并在预检中提示。
  本次不反推生产文件未携带的频率轴、校准结果、变换状态或端点约定。

## 可复现验证

已有 Python 3.12.0 环境，未安装或更新依赖。

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cdut_streaming_hdf5_import.py tests/test_acquisition_hdf5_import.py tests/test_backend_acquisition_api.py tests/test_field_import_preflight_v0926.py tests/test_gpr_format_registry_and_readers.py tests/test_sensor_sync_v0926.py tests/test_sensor_sync_source_provenance.py -q
```

新增 12 项先在旧代码上失败，再验证修复；与上述既有测试共 68 项通过。
覆盖预检、直接和分块读取、缺测、uint64 精度、端点、时间戳推断、项目导入重开和原有 CLI/处理回归。

另直接调用已交付 CDUT 1.1.10 的原版 `StreamingH5Writer` 和 `Hdf5TraceWriter`，生成三份 5 道×501 点合成 H5
（远程采集端、地面端及本地 `real` 路径），通过实际 MyGPR 预检、读取、项目导入重开和 CLI 两倍幅度缩放。
波形与逐道时间逐值一致，缺失高度保留 NaN，导入前后源文件 SHA256 不变。
输入为合成波形及传感器值，原版写入器生成文件不等于真实硬件采集验收。

## 尚未验证

- 当前实测会话只有 CSV；尚未用两端更新后的真实野外 H5 验收。
- 当前生产 H5 未声明时间轴端点，物理时间轴与仪器显示含义仍须按实际采集路径核对。
- 本次没有执行 Windows 桌面点击验收，也没有重建或替换已安装的 MyGPR EXE。
