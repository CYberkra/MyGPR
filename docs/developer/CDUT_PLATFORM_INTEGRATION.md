# CDUT-UavGPR 平台对接说明

核查日期：2026-10-20。本文记录组件定位、已支持边界和后续对齐事项；表中待办尚未实现。
主仓库参考快照：[`8cf89c2`](https://github.com/MuseLinn/CDUT-UavGPR-Controller/commit/8cf89c2caa9cdd6551efd47e43a8aaa6986df5e0)。

## 组件定位

MyGPR 面向无人机探地雷达实测项目，组织项目与测线、数据质控和处理、成像解释、空间成果与报告。
主要数据来源是 CDUT-UavGPR 采集链路，同时保留本仓库已经支持的其他数据入口。

| 组件 | 维护位置 | 职责与当前边界 |
| --- | --- | --- |
| 采集控制与机载服务 | [CDUT-UavGPR-Controller](https://github.com/MuseLinn/CDUT-UavGPR-Controller) | VNA/SFCW 控制、RTK/姿态/高度传感器、本地存储、远程控制与 A-Scan 传输 |
| 处理工作台与无头后端 | [MyGPR](https://github.com/CYberkra/MyGPR) | 完成文件导入与预检、处理成像、解释和成果交付；当前没有对接父仓库 TCP 接收的实时处理入口 |
| 总平台启动与发行 | 主仓库 `launcher.py`、子模块及统一安装器 | 组织采集端、处理端和机载服务端的启动及交付；需分别验证集成后的版本和路径 |

主仓库以 `MyGPR/` Git 子模块固定处理端提交。主仓库具有的硬件控制或通信功能，须与 MyGPR 中
实际实现的入口分别核验；本仓库 `mygpr/application/acquisition` 服务的存在不代表已接通 VNA 现场采集。

## 数据对接与语义

生产端协议见 [主仓库数据格式与通信协议](https://github.com/MuseLinn/CDUT-UavGPR-Controller/blob/8cf89c2caa9cdd6551efd47e43a8aaa6986df5e0/docs/data_format_and_protocol.md)。
MyGPR 当前可读 schema 以 [数据格式参考](../user/reference/数据格式参考.md) 和
[`core/acquisition_hdf5.py`](../../core/acquisition_hdf5.py) 为准。

- 主仓库输出堆叠 CSV、采集 HDF5，以及部分采集路径的复数 S21 sidecar。
  MyGPR 采集 HDF5 范围现包括带 `real` / `mag` / `magnitude` / `fdata` 标记的已完成文件；
  不以扩展名相同推定全部生产端变体兼容。
- `/traces` 的道×采样点布局需要转为 MyGPR 的采样点×道布局；时窗、存储分量、
  逐道经纬度/海拔/高度及缺失值均属于数据语义。缺少测距或传感器值时保留未知状态。
- 逐道原始 HWT 姿态/加速度/角速度、`timestamp_ms` 和整数 `trace_id` 一并读取并映射到处理接口；
  模拟标记和测次 UUID 保留。已有正式同步结果仍优先，原始快照不等于完成外参标定或运动补偿。
- 时域实部、幅值包络、VNA 显示采样 `fdata` 和频域复数 S21 分别处理。
  S21 离线重算还需测次对应的频率轴与变换配置。1.1.11 SFCW 路径现另有 `.sfcw.json`（原 ICZT 参数和 UUID）及 `.s21.jsonl`（原频谱/快照）；MyGPR 本轮只读配置并保留日志，不提供原频谱重算入口。
- 生产端协议将机载文件作为成像事实源；地面文件供备份与实时显示。
  同一布局或逐道元数据推送不证明两端文件逐字节相同，处理前需核对完整性和测次来源。

## 待对齐事项

| 项目 | 当前事实 | 后续验证要求 |
| --- | --- | --- |
| `fdata` 和历史 HDF5 | MyGPR 可读取显式 `fdata` 并保留显示采样语义；新增 `mag` 幅值标签兼容，缺分量标记仍拒绝 | 已有 1.1.10 合成生产文件验证；1.1.11 维护记录见下文，真实野外文件及仪器显示轴仍需核对 |
| 时间轴端点 | 显式布尔属性 `time_axis_endpoint=true` 或经同测次验证的 1.1.11 SFCW 配置采用 `window/(N-1)`；两者冲突拒绝。无证据时仍沿用 `window/N` 并提示 | 地面备份可能没有原配置，应保留机载 companion；处理端不回写源文件，也不因 `input_domain=frequency` 对已变换矩阵再次 ICZT |
| 实时接收与处理 | 父仓库有 TCP A-Scan 推送、落盘和断点补传；MyGPR 当前导入已完成文件 | 如需联动，独立定义接收、完整性、元数据及处理刷新接口，保留父仓库原有本地存储和传输路径 |
| 原始 S21 与溯源 | 1.1.11 SFCW 自治路径保存原频谱和完整 ICZT 配置；H5 主矩阵仍是所选时域分量，旧路径能力不变 | MyGPR 保留两类 companion，配置按 UUID/轴/分量核对；S21 日志仅标为存在但未校验，不宣称原始频谱、标定或实机验证已经完成 |

这些是协议核对事项，不构成新的功能验收结论或已排期实施承诺。

## 发行与启动入口

本仓库发布的 `MyGPR-0.11.0-Windows-x64-Setup.exe` 安装处理端 `MyGPR.exe`。
源码入口仍为 `app_qt.py`；Python 包还提供 `mygpr` / `mygpr-batch` 命令。

主仓库参考快照中的 [`launcher.py`](https://github.com/MuseLinn/CDUT-UavGPR-Controller/blob/8cf89c2caa9cdd6551efd47e43a8aaa6986df5e0/launcher.py)
冻结启动路径为 `GPR_GUI/GPR_GUI.exe`，源码回退路径为 `MyGPR/app_qt.py`。
独立安装包与该冻结路径需要另行对齐；仅在 MyGPR 发布 Release 不会更新主仓库子模块或统一安装包。

后续总平台集成需完成：固定所需 MyGPR 子模块提交、核对两端 Python/原生依赖环境、对齐启动路径和安装布局，
再验证从统一启动器启动、采集文件导入与版本显示。两仓库的依赖声明分别维护，不直接互相覆盖。
主仓库快照 README 声明 Python 3.10；本仓库当前源码依赖 Python 3.12–3.13，统一环境不能据此默认兼容。
MyGPR 独立 Windows 包内置 Python，构建和验证方法见 [Windows 构建说明](../releases/windows-build.md)。
