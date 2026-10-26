# 处理算法展示命名规则

适用：算法选择器、待执行处理链、步骤编辑器及目录返回的算法展示名。

## 规则

1. 使用准确、简短的中文作用名；有通用英文名称或缩写时采用“中文作用名（常见英文名或缩写）”，例如“二维小波去噪（Wavelet）”“自动增益控制（AGC）”。Stolt、Kirchhoff 等人名方法可保留原文。英文是否保留看领域辨识度，不要求逐项直译。
2. 依据当前实现命名，不按期望功能命名。区分增益与归一化、平滑结果与减背景、坐标更新与雷达振幅处理。SEC 旧节点当前仅实现时间幂函数曲线，因此显示为“时间幂函数增益”。
3. 数组约定为 `[时间采样, 雷达道]`。“道内／时间轴”指 axis=0；“道间”指 axis=1。“道向”不再用于新名称。二维、频率-波数等信息在有助于理解时保留。
4. 术语统一为“校正”“偏移成像”“振幅”。“反褶积”作为正式名称，“反卷积”保留为检索别名。不要把 Dewow 等同于仅去除整道常数均值。
5. 名称不混入实验、推荐、版本号等状态。既有 maturity、tags、visibility 不随改名调整；实验性方法在说明中明确标注。“V2”作为旧名称或检索词保留。
6. 同一方法的正式名称只维护在 `core/method_registry_metadata.py` 的 `display_name`。原生目录、UI facade、选择器和处理链通过算法 ID 取展示名，禁止另建平行名称表。
7. 每项有简短中文说明，交代作用、方向及必要限制；旧正式名称及常见英文缩写保留在说明的检索别名中。选择器搜索正式名、原名、ID、说明及分类，兼容全角／半角和大小写。
8. `method_id`、参数键、默认值、实现版本和处理行为不随展示改名调整。旧草稿／成果标签保留原始记录；处理链显示按当前目录解析，未知 ID 保留原标签。不得重写历史数据来更新名称。

## 本次核对表

以下按 36 项注册记录列出，包含原有隐藏项；可见性与数值实现未调整。

| 算法 ID | 原名称 | 规范名称 |
|---|---|---|
| `compensatingGain` | 增益补偿 | 线性分贝增益 |
| `dewow` | 低频漂移矫正（Dewow） | 低频漂移校正（Dewow） |
| `set_zero_time` | 零时矫正 | 零时校正（Time-zero） |
| `time_cut` | 时间窗裁剪 | 时间窗裁剪 |
| `trace_qc` | 坏道质量控制 | 坏道质量控制 |
| `agcGain` | 自动增益控制（AGC） | 自动增益控制（AGC） |
| `energy_decay_gain` | 能量衰减增益 | 能量衰减补偿增益 |
| `amplitude_scale` | 幅值缩放/归一化 | 振幅缩放与归一化 |
| `subtracting_average_2D` | 平均背景抑制 | 平均背景抑制 |
| `median_background_2D` | 中值背景抑制 | 中值背景抑制 |
| `running_average_2D` | 道向运行平均平滑 | 道间滑动平均平滑（Moving Average） |
| `trace_median_filter` | 道向中值滤波 | 道间中值滤波（Median Filter） |
| `trace_savgol_filter` | 道向 Savitzky-Golay 平滑 | 道间多项式平滑（Savitzky-Golay） |
| `svd_bg` | SVD 背景抑制 | 奇异值背景抑制（SVD） |
| `fk_filter` | F-K 频波数滤波 | 频率-波数滤波（F-K） |
| `frequency_filter_1d` | 一维频域滤波 | 时间轴频域滤波 |
| `hankel_svd` | Hankel-SVD 去噪 | 汉克尔矩阵去噪（Hankel-SVD） |
| `svd_subspace` | SVD 子空间去噪 | 奇异值子空间去噪（SVD） |
| `wavelet_2d` | Wavelet 2D 去噪 | 二维小波去噪（Wavelet） |
| `hilbert_envelope` | Hilbert 包络 | 包络提取（Hilbert） |
| `rpca_background` | RPCA 背景抑制 | 鲁棒主成分背景抑制（RPCA） |
| `ccbs` | 互相关背景抑制（CCBS） | 互相关背景抑制（CCBS） |
| `motion_compensation_height` | 飞行高度归一化 | 离地高度补偿 |
| `motion_compensation_speed` | 速度误差补偿 | 等距道重采样 |
| `trajectory_smoothing` | 轨迹平滑 | 轨迹平滑 |
| `motion_compensation_attitude` | 姿态/APC足迹修正 | 姿态与天线足迹校正 |
| `motion_compensation_vibration` | 周期条带伪影抑制（实验） | 周期条带伪影抑制 |
| `motion_compensation_v2` | UAV 运动补偿 V2 | 无人机综合运动补偿 |
| `stolt_migration` | Stolt 迁移 | Stolt 偏移成像 |
| `kirchhoff_migration` | Kirchhoff 迁移 | Kirchhoff 偏移成像 |
| `time_to_depth` | 时间-深度转换 | 时间-深度转换 |
| `sec_gain` | SEC 增益 | 时间幂函数增益 |
| `sliding_avg` | 滑动平均背景抑制 | 滑动平均背景抑制 |
| `rtm_migration` | RTM 反向时间迁移 | 逆时偏移成像（RTM） |
| `mixed_phase_deconvolution` | 混合相位反褶积 | 混合相位反褶积 |
| `inverse_q` | Inverse-Q 衰减补偿 | 逆 Q 衰减补偿（Inverse-Q） |

## 常见英文的取舍

- 保留已有的 Dewow、AGC、SVD、F-K、Hilbert、RTM 等方法标识，并补充 Moving Average、Median Filter、Savitzky-Golay、Wavelet、Time-zero、Inverse-Q。
- Hankel-SVD 保留为特定低秩方法标识，展示为“汉克尔矩阵去噪（Hankel-SVD）”；Stolt、Kirchhoff 继续采用人名加中文作用的形式，避免紧凑处理链把不同方法都缩成“偏移成像”。
- 能量衰减补偿、振幅缩放、等距重采样等名称保留清楚的中文；普通操作不强制附英文直译。时间幂函数增益不附含义易混淆的 Power Gain，更不附 SEC；SEC 只保留在历史检索说明中。
- 算法选择器展示完整中英名称；紧凑处理链保留中文主体，完整名称在悬停说明和步骤编辑器中可读。

## 实现依据与参考

- `mygpr/infrastructure/processing/algorithms/basic.py`：线性分贝增益、时间幂函数增益及道间滤波的具体曲线和轴。
- `algorithms/motion/speed.py`：等距离轴重采样；`height.py`：高度时延／振幅补偿；`attitude.py`：天线足迹元数据更新；`rtm.py`：二维标量零偏移距基线。
- [Sensors & Software GPR Glossary](https://www.sensoft.ca/resources/glossary-of-gpr-terms-2-2/)：SEC 为扩散与指数损耗的组合补偿，不能仅凭历史 ID 认定已有该模型。
- [EKKO_Project Processing Module User’s Guide](https://www.sensoft.ca/wp-content/uploads/2015/11/Processing-Module-Users-Guide.pdf)：区分时间滤波、空间滤波、背景相减、包络和偏移。
- [SciPy Savitzky–Golay 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)：滤波方向由 axis 指定；本项目道间 S-G 采用 axis=1、deriv=0。

展示名调整时允许逐项更新目录描述符基线的 `name`，保留其余契约字段；同时验证旧名搜索、双击返回原 ID，以及草稿显示刷新不改变参数／启停／脏状态。

- [SciPy Median Filter 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html)与 [PyWavelets 二维小波文档](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html)：核对相应英文方法名称。
