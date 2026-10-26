"""Display, maturity, category, and auto-tune metadata for processing methods."""
from __future__ import annotations

METHOD_METADATA = {
    "compensatingGain": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "线性分贝增益",
        "description": "增益的分贝值沿时间轴从起始值线性变化至终止值，再换算成振幅倍率。 旧名称：增益补偿。 检索别名：时间增益 指数时间增益 dB gain。",
    },
    "dewow": {
        "category": "drift_correction",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "低频漂移校正（Dewow）",
        "description": "沿每一道的时间轴减去滑动均值，去除低频基线漂移；与减去整道常数均值的去直流处理有区别。 旧名称：低频漂移矫正（Dewow）。 检索别名：去漂移 dewow。",
    },
    "set_zero_time": {
        "category": "time_correction",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "零时校正（Time-zero）",
        "description": "按指定零时位置平移每一道的时间采样，尾部补零。 旧名称：零时矫正。 检索别名：time zero time-zero。",
    },
    "time_cut": {
        "category": "time_correction",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "时间窗裁剪",
        "description": "按指定时间范围裁剪或屏蔽数据，具体行为由裁剪模式决定。 检索别名：time window crop。",
    },
    "trace_qc": {
        "category": "quality_control",
        "maturity": "stable",
        "visibility": "hidden",
        "display_name": "坏道质量控制",
        "description": "检测低能量道、能量异常道和手动指定道；按模式标记、置零或移除。 检索别名：坏道检测 trace QC。",
    },
    "agcGain": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "自动增益控制（AGC）",
        "description": "按每一道时间滑动窗口内的局部能量调整振幅。 检索别名：automatic gain control。",
    },
    "energy_decay_gain": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "能量衰减补偿增益",
        "description": "由各道绝对振幅的中位数估计随时间变化的衰减曲线，施加有限幅的补偿增益。 旧名称：能量衰减增益。 检索别名：energy decay gain。",
    },
    "amplitude_scale": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "振幅缩放与归一化",
        "description": "按所选模式执行固定倍率缩放、峰值归一化或均方根归一化。 旧名称：幅值缩放/归一化。 检索别名：幅值 幅度 amplitude normalization RMS。",
    },
    "subtracting_average_2D": {
        "category": "background_suppression",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "平均背景抑制",
        "description": "按道窗口估计平均背景并相减；窗口覆盖整条测线时使用全局平均背景。 检索别名：均值背景 去背景 mean background subtraction。",
    },
    "median_background_2D": {
        "category": "background_suppression",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "中值背景抑制",
        "description": "按道窗口估计中值背景并相减，可限定生效时间范围。 检索别名：median background subtraction。",
    },
    "running_average_2D": {
        "category": "denoising",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "道间滑动平均平滑（Moving Average）",
        "description": (
            "在相邻道之间对同一时间采样做滑动平均，直接保留平滑结果；与减去平均背景的处理不同。 旧名称：道向运行平均平滑。 检索别名：道向 running moving average spatial "
            "smoothing。"
        ),
    },
    "trace_median_filter": {
        "category": "denoising",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "道间中值滤波（Median Filter）",
        "description": "在相邻道之间对同一时间采样做中值滤波，不沿单道时间轴滤波。 旧名称：道向中值滤波。 检索别名：道向 spatial median filter。",
    },
    "trace_savgol_filter": {
        "category": "denoising",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "道间多项式平滑（Savitzky-Golay）",
        "description": (
            "在相邻道之间以局部多项式拟合平滑；采用 Savitzky-Golay 方法，不求导。 旧名称：道向 Savitzky-Golay 平滑。 检索别名：SG Savitzky-Golay Savitzky"
            " Golay 道向。 曾用显示名：道间多项式平滑（S-G）。"
        ),
    },
    "svd_bg": {
        "category": "background_suppression",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "奇异值背景抑制（SVD）",
        "description": "以奇异值分解估计前若干低秩分量作为背景，并从原数据中减去。 旧名称：SVD 背景抑制。 检索别名：singular value decomposition background。",
    },
    "fk_filter": {
        "category": "filtering",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "频率-波数滤波（F-K）",
        "description": "实验功能。在二维频率-波数域进行方向或倾角相关滤波，需正确的时间和空间采样信息。 旧名称：F-K 频波数滤波。 检索别名：FK f k frequency wavenumber。",
    },
    "frequency_filter_1d": {
        "category": "filtering",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "时间轴频域滤波",
        "description": (
            "逐道沿时间轴进行频域滤波；可选带通、低通、高通或陷波。 旧名称：一维频域滤波。 检索别名：一维频域滤波 频率滤波 bandpass lowpass highpass notch FFT。"
        ),
    },
    "hankel_svd": {
        "category": "denoising",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "汉克尔矩阵去噪（Hankel-SVD）",
        "description": "实验功能。将数据嵌入 Hankel 矩阵后，通过低秩奇异值分解重构去噪。 检索别名：汉克尔 MSSA。 曾用显示名：Hankel-SVD 去噪。",
    },
    "svd_subspace": {
        "category": "denoising",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "奇异值子空间去噪（SVD）",
        "description": "实验功能。保留指定序号范围内的奇异值分量重构数据；与减去低秩背景的 SVD 背景抑制不同。 旧名称：SVD 子空间去噪。 检索别名：subspace denoising。",
    },
    "wavelet_2d": {
        "category": "denoising",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "二维小波去噪（Wavelet）",
        "description": "实验功能。对 B-scan 进行二维小波分解、系数阈值处理和重构。 旧名称：Wavelet 2D 去噪。 检索别名：wavelet 2D wavelet denoising。",
    },
    "hilbert_envelope": {
        "category": "attribute_analysis",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "包络提取（Hilbert）",
        "description": "逐道取希尔伯特解析信号的模，得到振幅包络；输出不保留原始正负极性。 旧名称：Hilbert 包络。 检索别名：希尔伯特 Hilbert envelope。",
    },
    "rpca_background": {
        "category": "background_suppression",
        "maturity": "experimental",
        "visibility": "hidden",
        "display_name": "鲁棒主成分背景抑制（RPCA）",
        "description": "实验功能。将数据分解为低秩背景和稀疏分量，用于背景抑制。 旧名称：RPCA 背景抑制。 检索别名：robust principal component analysis。",
    },
    "ccbs": {
        "category": "background_suppression",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "互相关背景抑制（CCBS）",
        "description": "实验功能。依据各道与参考波的归一化互相关，调整参考背景与平均背景的混合权重后相减。 检索别名：cross-correlation background subtraction。",
    },
    "motion_compensation_height": {
        "category": "motion_compensation",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "离地高度补偿",
        "description": "实验功能。根据逐道离地高度，相对参考高度补偿空气传播时延和振幅；两项可分别开关。 旧名称：飞行高度归一化。 检索别名：高度归一化 height normalization AGL。",
    },
    "motion_compensation_speed": {
        "category": "motion_compensation",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "等距道重采样",
        "description": "实验功能。根据累计距离或局部平面坐标，将雷达道及道元数据线性重采样到等距离轴。 旧名称：速度误差补偿。 检索别名：速度补偿 equal-distance resampling。",
    },
    "trajectory_smoothing": {
        "category": "motion_compensation",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "轨迹平滑",
        "description": "实验功能。平滑定位轨迹并更新局部坐标和累计距离，不直接改变雷达振幅。 检索别名：GNSS GPS RTK trajectory smoothing。",
    },
    "motion_compensation_attitude": {
        "category": "motion_compensation",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "姿态与天线足迹校正",
        "description": (
            "实验功能。根据姿态、位置、高度和天线相位中心偏移，更新天线地面足迹坐标；不直接改变雷达振幅。 旧名称：姿态/APC足迹修正。 检索别名：APC antenna phase center "
            "footprint。"
        ),
    },
    "motion_compensation_vibration": {
        "category": "artifact_suppression",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "周期条带伪影抑制",
        "description": (
            "实验功能。沿道间方向抑制指定空间频带内的周期条带，并保护显著目标；不进行旋翼转速识别。 旧名称：周期条带伪影抑制（实验）。 检索别名：周期条带 振动 vibration artifact。"
        ),
    },
    "motion_compensation_v2": {
        "category": "motion_compensation",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "无人机综合运动补偿",
        "description": (
            "实验功能。综合逐道高度、姿态与位置进行时延、振幅和天线足迹补偿，并按配置执行等距重采样；输入不足时按现有规则跳过对应项并报告。逐道传感器值来自采集道元数据（trace_metadata），参数面板只设"
            "置补偿策略。 旧名称：UAV 运动补偿 V2。 检索别名：UAV motion compensation V2。"
        ),
    },
    "stolt_migration": {
        "category": "migration",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "Stolt 偏移成像",
        "description": "实验功能。采用 Stolt 频率-波数域映射实现偏移聚焦，需要适用的传播速度与道间距。 旧名称：Stolt 迁移。 检索别名：Stolt migration。",
    },
    "kirchhoff_migration": {
        "category": "migration",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "Kirchhoff 偏移成像",
        "description": (
            "实验功能。按传播走时进行 Kirchhoff 叠加偏移，需要适用的传播速度与道间距。 旧名称：Kirchhoff 迁移。 检索别名：基尔霍夫 Kirchhoff migration。"
        ),
    },
    "time_to_depth": {
        "category": "depth_conversion",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "时间-深度转换",
        "description": "按传播速度或介电常数将双程时间轴转换为估计深度轴，不等同于偏移成像。 检索别名：时深转换 time-to-depth。",
    },
    "sec_gain": {
        "category": "gain",
        "maturity": "stable",
        "visibility": "public",
        "display_name": "时间幂函数增益",
        "description": (
            "以归一化时间的幂函数在起始和终止增益之间构造振幅倍率。旧名 SEC 增益；当前实现未包含标准 SEC 的扩散与指数损耗联合补偿。 旧名称：SEC 增益。 检索别名：SEC power time "
            "gain。"
        ),
    },
    "sliding_avg": {
        "category": "background_suppression",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "滑动平均背景抑制",
        "description": "实验功能。减去指定轴上的局部滑动均值；默认沿道间方向，也可沿时间轴处理。 检索别名：sliding average background subtraction。",
    },
    "rtm_migration": {
        "category": "migration",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "逆时偏移成像（RTM）",
        "description": (
            "实验功能。二维标量、零偏移距爆炸反射面逆时延拓基线，需传播速度和道间距；不等同于完整多炮电磁逆时偏移。 旧名称：RTM 反向时间迁移。 检索别名：逆时偏移 reverse time "
            "migration。"
        ),
    },
    "mixed_phase_deconvolution": {
        "category": "denoising",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "混合相位反褶积",
        "description": "实验功能。以最小相位反褶积和全测线常相位旋转处理混合相位子波。 检索别名：反卷积 mixed-phase deconvolution。",
    },
    "inverse_q": {
        "category": "filtering",
        "maturity": "experimental",
        "visibility": "public",
        "display_name": "逆 Q 衰减补偿（Inverse-Q）",
        "description": (
            "实验功能。按常 Q 模型补偿振幅衰减和相位色散，并限制最大补偿增益；需要有效时间基准。 旧名称：Inverse-Q 衰减补偿。 检索别名：Inverse-Q inverse Q filtering。"
        ),
    },
}
PREFERRED_METHOD_ORDER = [
    "set_zero_time",
    "time_cut",
    "trace_qc",
    "dewow",
    "subtracting_average_2D",
    "median_background_2D",
    "svd_bg",
    "ccbs",
    "sliding_avg",
    "frequency_filter_1d",
    "fk_filter",
    "sec_gain",
    "energy_decay_gain",
    "amplitude_scale",
    "compensatingGain",
    "agcGain",
    "hankel_svd",
    "trace_median_filter",
    "trace_savgol_filter",
    "svd_subspace",
    "wavelet_2d",
    "hilbert_envelope",
    "running_average_2D",
    "motion_compensation_vibration",
    "motion_compensation_height",
    "motion_compensation_speed",
    "trajectory_smoothing",
    "motion_compensation_attitude",
    "motion_compensation_v2",
    "stolt_migration",
    "kirchhoff_migration",
    "inverse_q",
    "mixed_phase_deconvolution",
    "rtm_migration",
    "time_to_depth",
]

METHOD_TAGS = {
    "subtracting_average_2D": "推荐",
    "median_background_2D": "备选",
    "agcGain": "备选",
    "dewow": "推荐",
    "set_zero_time": "推荐",
    "time_cut": "备选",
    "trace_qc": "备选",
    "energy_decay_gain": "推荐",
    "amplitude_scale": "备选",
    "svd_bg": "备选",
    "fk_filter": "实验",
    "frequency_filter_1d": "推荐",
    "hankel_svd": "实验",
    "svd_subspace": "实验",
    "wavelet_2d": "实验",
    "hilbert_envelope": "推荐",
    "stolt_migration": "实验",
    "kirchhoff_migration": "实验",
    "rtm_migration": "实验",
    "sliding_avg": "实验",
    "running_average_2D": "备选",
    "trace_median_filter": "备选",
    "trace_savgol_filter": "备选",
    "motion_compensation_height": "实验",
    "motion_compensation_speed": "实验",
    "trajectory_smoothing": "实验",
    "motion_compensation_attitude": "实验",
    "motion_compensation_vibration": "实验",
    "motion_compensation_v2": "推荐",
    "mixed_phase_deconvolution": "实验",
    "inverse_q": "实验",
}

METHOD_CATEGORY_LABELS = {
    "time_correction": "时间校正",
    "drift_correction": "低频漂移校正",
    "background_suppression": "背景抑制",
    "clutter_suppression": "杂波抑制",
    "filtering": "频域滤波",
    "denoising": "去噪",
    "gain": "增益与振幅",
    "migration": "偏移成像",
    "depth_conversion": "时深转换",
    "motion_compensation": "运动补偿",
    "artifact_suppression": "伪影抑制",
    "quality_control": "质量控制",
    "attribute_analysis": "属性分析",
    "experimental": "实验功能",
}

AUTO_TUNE_STAGE_BY_METHOD = {
    "set_zero_time": "zero_time",
    "time_cut": "preprocess",
    "trace_qc": "preprocess",
    "dewow": "drift",
    "subtracting_average_2D": "background",
    "median_background_2D": "background",
    "svd_bg": "background",
    "fk_filter": "frequency",
    "frequency_filter_1d": "frequency",
    "ccbs": "background",
    "sec_gain": "gain",
    "energy_decay_gain": "gain",
    "amplitude_scale": "gain",
    "compensatingGain": "gain",
    "agcGain": "gain",
    "running_average_2D": "impulse",
    "hankel_svd": "denoise",
    "trace_median_filter": "denoise",
    "trace_savgol_filter": "denoise",
    "svd_subspace": "denoise",
    "wavelet_2d": "denoise",
    "motion_compensation_height": "motion_comp",
    "motion_compensation_speed": "motion_comp",
    "trajectory_smoothing": "motion_comp",
    "motion_compensation_attitude": "motion_comp",
    "motion_compensation_vibration": "artifact",
    "motion_compensation_v2": "motion_comp",
    "rtm_migration": "migration",
    "mixed_phase_deconvolution": "denoise",
    "inverse_q": "frequency",
}

METHOD_DISPLAY_NAMES = {key: value["display_name"] for key, value in METHOD_METADATA.items()}

__all__ = [
    "METHOD_METADATA",
    "METHOD_DISPLAY_NAMES",
    "PREFERRED_METHOD_ORDER",
    "METHOD_TAGS",
    "METHOD_CATEGORY_LABELS",
    "AUTO_TUNE_STAGE_BY_METHOD",
]
