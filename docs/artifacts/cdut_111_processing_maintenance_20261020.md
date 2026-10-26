# CDUT 1.1.11 对应的 MyGPR 导入维护

日期：2026-10-20（北京时间）。MyGPR 基线 e379a5e，工作分支 codex/maintenance-20261017；保留此前全部本地维护修改。本轮更新源码，不重建安装包、不启动或控制仪器。

## 本轮修复

- 支持生产端 `mag`，保留原有 `real` / `magnitude` / `fdata`。预检区分幅值与实部，提示幅值不包含原相位和正负号。
- 保留 `input_domain` 与 `signal_domain`；频域输入、时域结果不再次 ICZT。显式非时域信号拒绝作为时域 A-Scan 导入。
- 有 `<文件名>.sfcw.json` 时，读取量限制为 64 KiB，核对非零 UUID、采样点、时窗、分量与合法 ICZT 参数。匹配原 Bluestein 配置后使用含末点时间轴：501 点、900 ns，间隔 1.8 ns。显式端点冲突、错测次或损坏配置均拒绝，不悄悄改轴。
- SFCW 配置缺失时保留主波形读取能力，沿用历史时间轴约定并明确提示不确定性；不能单凭 `input_domain=frequency` 推断原端点。
- H5 项目导入一并分块复制 SFCW 配置及可选原始 S21 日志，支持取消与失败回滚；替换同名测线时清理缺失的旧 companion，避免串入其他测次。
- 原传感器缺失值保留 NaN 和对应雷达道，沿用已有完整性提示。S21 日志仅检测存在并保留，不宣称逐条频谱校验或离线重算已经完成。
- 修复 Windows 跨盘 CLI：配置/输出在 C 盘、软件在 D 盘时，validate、run、resume 及步骤/最终结果路径均使用已有安全路径函数。保留同盘相对路径，跨盘保存绝对路径。

## 验证

使用已有 Python 3.12.0 虚拟环境，无依赖安装或更新。

- 新增 19 项定向用例：初始 14 项在修改前失败；复核补出的两项替换测线用例先失败后修复；另验证 companion 取消/写入失败回滚与真实 Windows 跨盘 CLI。
- 正式源码最终回归 **156 passed in 33.45s**，范围为 HDF5 导入、采集 API、预检、CSV 合同、传感器完整性与同步、CLI profiles/resume 完整性。
- 仓库内存语法检查：637 个 Python 文件通过；架构门禁通过。
- 直接调用已交付 1.1.11 的原版 SFCW 变换和 H5 写入器，分别生成 3 道×501 点 real/mag 合成生产文件，参数 20–170 MHz、900 ns。预检、矩阵/时间轴、缺失高度、原道号、项目导入重开、companion 原样保留及 CLI 两倍幅度结果均一致；主文件和伴随文件源哈希未改变。
- 实际命令行 `cli_batch.py validate/run` 跨盘复现：修复后退出码均为 0，real/mag 两个作业成功。
- 独立只读复核已修复旧 companion 残留及 CLI 换行副作用；无剩余阻断项。

可复现命令（仓库根目录）：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cdut_acquisition_111_import.py tests/test_acquisition_hdf5_import.py tests/test_cdut_streaming_hdf5_import.py tests/test_backend_acquisition_api.py tests/test_field_import_preflight_v0926.py tests/test_gpr_format_registry_and_readers.py tests/test_sensor_sync_v0926.py tests/test_sensor_sync_source_provenance.py tests/test_sensor_completeness_notices.py tests/test_sensor_report_model.py tests/test_gpr_io_airborne_contract.py tests/test_cli_batch_profiles.py tests/test_cli_resume_integrity.py -q
.venv\Scripts\python.exe scripts/check_python_compile.py
.venv\Scripts\python.exe scripts/check_architecture.py
```

## 使用与边界

源码版重启后使用本次维护。已归一化保存的旧项目不会自动迁移物理时间轴；需要采用原配置的测线，请保留主 H5 与同测次 `.sfcw.json` 后重新导入。没有原配置的地面副本仍需核对时间轴，不能凭 501 点猜测频率或仪器变换状态。

本轮主要对齐 **已完成的 H5/SFCW 文件导入**。CSV 原有合同通过回归，本轮未新增 CSV 原始日志接入、实时 TCP 处理或原始 S21 重算入口；没有更改处理算法、主题或页面布局。现有安装版 EXE 未重新打包。

验证输入为合成频谱与传感器快照；未进行真实野外 H5、UU 双机、硬件或 Windows 桌面点击验收。传感器字段可用率不代表 RTK 固定解、时序同步、运动补偿或工程精度已经验证。
