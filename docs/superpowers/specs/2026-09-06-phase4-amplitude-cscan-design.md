# Phase 4.1 — 振幅/能量 C-scan（深度/时间切片）设计（2026-09-06 草案 v2，待用户拍板）

> 状态：**草案，未开工**。决策点（§4）需用户确认后才动代码。
> 来源：2026-09-06 文献调研（26+ 来源，关键文献见 §8）+ 既有代码事实核对。
> v2 修订（同日）：纠正 v1 的契约错误——现有 `DepthSliceView` payload 是
> **单张界面深度矩阵 + 其值域**，滑条只改等值线 level
> （`spatial_page.py::_on_depth_slider_changed` → `set_isoline`，矩阵静态不变），
> 无法表达「滑条每个位置 = 不同属性网格」的振幅 C-scan。§3/§5 改为 cube 架构。

## 1. 定位

路线图 Phase 4 第一项「能量属性切片（换 3.1 的数据源）」的正式设计。
**与 3.1 界面深度切片的关系**：复用网格化、GIS 图层链、视图组件骨架；
**不复用**单矩阵 payload 契约与滑条语义（见 §3 契约对比）。
**不再需要界面拾取作为前提**——只需要轨迹 + B-scan 数据。

## 2. 文献调研摘要（方法论依据）

经典振幅 C-scan 标准流程（GPR-SLICE 手册 [1]、AGU [2]、MDPI [3] 共识）：

```
B-scan 预处理（time-zero / dewow / 带通 / 去背景，增益谨慎）
  → 轨迹 (x,y) 定位
  → Hilbert 包络 Ae(t) = √(s² + Ĥ{s}²)     ← 避免正负相位对消 [2][3]
  → 窗口属性统计（中心 tc、厚度 Δt）
  → 网格化
  → 切片成图
```

关键结论：

1. **包络优先于原始幅值**：有符号幅值在厚窗内正负对消；薄窗（< 一个子波）落在
   过零点上会不稳，包络在反射峰附近恒正、稳健 [1][3]。
2. **窗口属性四选** [2]：最大包络（点目标探测首选，对子波形态稳健）/ 均值包络
   （连续层位）/ RMS（平均强度）/ 能量（宽异常，对窗厚敏感）。
3. **窗厚默认一个主频脉宽**（1/f0），再试窄/宽；过厚会糊掉相邻反射体 [1]。
4. **深度切片 = 表观深度**：z = vt/2，单常速不精确；速度来源用标定/钻孔/双曲线
   拟合约束（UAV 案例 [7] 即用钻孔定速度）。迁移/包络后的视觉深度与原双曲线
   深度可能不一致，解释时需回原始剖面对照 [5]。
5. **增益纪律**：跨测线幅值可比较的前提是同一预处理链；激进 AGC 会把晚期噪声
   放大成深部假异常 [6]。
6. **UAV 非规则测线**：文献做法是 GPS 轨迹清洗 + data binning（规则格网分桶），
   而非克里金 [8][9]；与我们 `grid_attribute` 的 cell-mean 分桶完全一致。
   Sensoft 也有"非网格测量 + GPS 定位出深度切片"的商业先例 [12]。

## 3. 数据链与契约设计（v2 核心修订）

### 3.1 契约对比（为什么不能复用单矩阵 payload）

| | 3.1 界面深度切片 | 4.1 振幅 C-scan |
|---|---|---|
| payload 核心 | 单张 (nrows,ncols) 界面深度矩阵 | **cube (nz,nrows,ncols)**：每个切片位置一张属性网格 |
| 滑条语义 | 改等值线 level（矩阵不变） | **切换 cube 层索引**（矩阵随滑条变） |
| 滑条范围 | 界面深度值域 ×100 | 切片轴 bin 索引 0..nz−1 |

### 3.2 架构决策：预计算 cube（否决按需逐层请求）

- **选定**：参数（数据源/属性/窗厚/速度/模式）变化时 worker 一次性算完
  cube（nz, nrows, ncols, float32），滑条拖动 = **纯本地层索引，零 worker 往返**。
- **否决项**：滑条每位置请求 center±window 的 2D grid——拖动需要防抖 + 缓存 +
  异步往返，交互必然卡顿；且缓存键随 center 连续变化，命中率低。
- **内存守卫**：`ncells × nz ≤ 24M`（float32 ≈ 96 MB）超限抛
  `GridAnalysisError`（与 `grid_attribute` 的 4M cell 上限同款风格），hint 引导
  增大 cell_size 或缩小深度范围。典型体量（200×200 cell × 60 bin）≈ 9.6 MB。
- **可选缓存**：cube 以参数元组为键驻留内存，数据源来回切换免重算（v1 可不做）。

### 3.3 切片轴与窗口

- bin 中心沿时间轴（ns）或深度轴（m，z=vt/2）均匀分布；
- **bin 步长默认 = 窗厚/2**（50% 重叠，属性分析惯例；可配）；
- 每 trace：包络 → 每个 bin 中心取窗 → 窗口属性 → 得到长度 nz 的 1D 序列；
- 每个 z-bin 对所有 trace 走 `grid_attribute` cell-mean 分桶 → cube 一层。

### 3.4 新 payload 契约

```python
{
  "kind": "amplitude_slice_cube",
  "attribute": "max_envelope" | "mean_envelope" | "rms_envelope" | "energy",
  "mode": "time" | "depth",
  "velocity_m_per_ns": float | None,     # mode='time' 时为 None
  "window_ns": float,                    # 窗厚（时间域定义，深度模式内部换算）
  "cell_size_m": float, "ncols": int, "nrows": int,
  "x_origin_m": float, "y_origin_m": float,
  "axis_values": np.ndarray,             # (nz,) bin 中心，ns 或 m
  "cube": np.ndarray,                    # (nz, nrows, ncols) float32，NaN 空洞
  "valid_count": int,
}
```

### 3.5 UI 契约变更

- `DepthSliceView` **新增** `set_cube(cube, axis_values, *, ...)` +
  `set_slice_index(k)`（本地切层）；`set_grid`/`set_isoline` 保留，3.1 界面深度
  路径零改动。
- `SpatialPage` 滑条双语义：数据源=界面深度 → 现有等值线语义；数据源=振幅 →
  range=0..nz−1，标签显示 `axis_values[k]`（按 mode 带 ns/m 单位）。
- 「存为图层」存**当前可见层** `cube[k]`：需给 `submit_grid_layer` 增加
  「直接提交矩阵 + 标题」入口（现签名是从 line_ids 重算界面深度，§5 列为适配点）。

## 4. 决策点（待用户拍板，附推荐）

| # | 决策点 | 选项 | 推荐 |
|---|---|---|---|
| 1 | 切片模式 | A. 只做深度切片（UI 统一）／B. 时间切片默认 + 深度切片需选速度 | **B**：速度不准时不伪装成精确深度，诚实；时间切片零新参数即可用 |
| 2 | B-scan 数据源默认 | A. 原始／B. 最近处理产物 | **A + 下拉可切产物 + 提示行「建议先 dewow/去背景」**；不绑死流程 |
| 3 | 速度来源 | A. 常数输入（默认 0.1 m/ns）／B. 常数 + 可选「用速度分析结果」 | **B**：Phase 2 双曲线拟合 service 已有 error_code/证据 schema，接上即可；UAV 单覆盖做不了 CMP，这是文献认可的替代 [7] |
| 4 | cube 架构 | A. 预计算 cube（本地切层）／B. 滑条按需逐层请求 | **A**（§3.2 已定，交互响应性决定；用户可推翻） |

## 5. 改动点（文件级）

| 层 | 文件 | 改动 |
|---|---|---|
| application | `mygpr/application/grid/service.py` | 新增 `amplitude_slice_preview(projects, project_id, line_ids, *, artifact_id=None, mode='time', velocity_m_per_ns=0.1, attribute='max_envelope', window_ns=None, cell_size_m=1.0)` → §3.4 cube payload；窗厚缺省 = 1/f0（header 中心频率，缺则 2×采样间隔）；cube 内存守卫 |
| interfaces | `mygpr/interfaces/backend.py` + `config/backend_api_v1.json` + `scripts/check_backend_api_contract.py` | 注册 `amplitude_slice_preview` 契约 |
| interfaces | `submit_grid_layer` 适配 | 增加「直接矩阵 + 标题」入口（存当前可见层），保持旧签名兼容 |
| ui 组件 | `ui/widgets/depth_slice_view.py` | 新增 `set_cube()` + `set_slice_index()`；`set_grid` 保留不动 |
| ui | `ui/pages/spatial_page.py` | 深度切片段加参数条：数据源（界面深度\|包络能量）、属性（max/mean/rms/energy）、模式（时间\|深度）、速度、窗厚；滑条双语义分支 |
| ui | `ui/controllers/project_controller.py` | 仿 `_DepthPreviewCommand` 加 `_AmplitudeSliceCommand`（run_worker + JobBridge，worker 线程不碰 Qt） |
| tests | `tests/test_amplitude_slice_api.py`（新） | ① 合成体积：已知 (x,y) 埋目标于已知深度 → cube 对应层此 cell 出峰、邻层不出峰；② payload 契约形状/NaN/axis 单调；③ 内存守卫超限报错；④ 无轨迹报错路径；⑤ 存图层 job 回读当前层 |

## 6. 非目标（第一版不做）

- **迁移后切片**：kirchhoff 默认参数 P0 bug 未修（weight≥0.4 全零）、stolt 刚及格，
  现在依赖会把 C-scan 绑在不稳地基上；待迁移算法修复后作为后续增量。
- **IDW/克里金插值**：cell-mean 分桶对 UAV 密集轨迹足够 [8][9]；空 cell 显 NaN
  空洞比假插值诚实。
- **3D 体渲染**（GLVolumeItem）：Phase 4 后半另行设计。
- **速度随深度/横向变化模型**：第一版单常速；界面标注明确「表观深度」。

## 7. 验收标准

1. `tests/test_amplitude_slice_api.py` 全绿（含合成体积定位断言 + 内存守卫）
2. 离屏探针：空间页切到振幅切片，滑条拖动切层**本地即时**（无 worker 往返）、
   四属性渲染互异、界面深度路径回归无损
3. 真机：真实项目勾选测线 → 振幅切片出图 → 拖滑条连续切层 → 存图层回读当前层
4. 定向测试套件（spatial/depth_slice/grid）无回归；全量 pytest 绿（C 盘空间允许时）
5. 报告口径：切片图注明属性类型/窗厚/速度/数据源 artifact id（文献 [1] 可复现性模板）

## 8. 关键参考文献（调研来源）

1. **GPR-SLICE User Manual**（行业切片标准软件手册）— 薄窗需用包络、窗厚选择、
   可复现性报告模板 https://s3-ap-southeast-2.amazonaws.com/gprslice/GPR-SLICE_User_Manual.pdf
2. **Svendsen et al. 2023, JGR Solid Earth**（跨孔 GPR 高损耗介质）— 解析信号包络
   公式、窗口属性稳健性对比 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JB025909
3. **MDPI Remote Sensing 12(10):1583**（多道相干分析）— 厚切片必须用包络避免相位
   对消 https://www.mdpi.com/2072-4292/12/10/1583
4. **MDPI Remote Sensing 17(20):3484**（大兴安岭活动层厚度）— 幅值质心属性 + 阈值
   https://www.mdpi.com/2072-4292/17/20/3484
5. **EPA Ground-Penetrating Radar**（方法权威页）— 深度转换依赖双程走时与速度；
   迁移需可靠速度 https://www.epa.gov/environmental-geophysics/ground-penetrating-radar-gpr
6. **PMC6480043**（GPR 定量属性分析综述）— 增益/衰减对幅值可比较性的影响；AGC
   制造假异常 https://pmc.ncbi.nlm.nih.gov/articles/PMC6480043/
7. **Famiglietti et al. 2026, MDPI Drones 10(5):331**（UAV-GPR + LiDAR 边坡变形，
   Melizzano 案例）— 钻孔标定速度做时深转换的 UAV 先例
   https://www.mdpi.com/2504-446X/10/5/331
8. **egusphere-2024-3074 preprint**（UAV-GPR 采集足迹）— 无人机速度变化导致的采集
   足迹、奇偶测线分桶降 footprint https://egusphere.copernicus.org/preprints/2024/egusphere-2024-3074/egusphere-2024-3074-ATC2.pdf
9. **iris.unil.ch 学位论文**（UAV-GPR 处理流程表）— GPS 清洗 + GPR data binning 标准
   流程 https://iris.unil.ch/bitstreams/31ed1984-bb2b-4e5d-b9c1-ef8ef816906e/download
10. **Sensors 24(10):3238**（空洞测绘组合时深转换）— 速度横向变化时的分段时深转换
    公式 https://www.mdpi.com/1424-8220/24/10/3238
11. **RGPR 官方教程**（开源 GPR 处理）— Hilbert 包络在开源工作流中的标准实现
    https://emanuelhuber.github.io/RGPR/02_RGPR_tutorial_basic-GPR-data-processing/
12. **Sensoft 技术博客**（EKKO_Project）— 无网格测量 + GPS 定位出深度切片的商业
    先例（对应 UAV 自由走测）https://www.sensoft.ca/blog/depth-slicing-without-grid/
