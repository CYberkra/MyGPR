# 混合相位反褶积 + Inverse-Q 衰减补偿：实现设计稿（2026-09-06）

> 依据：`docs/artifacts/2026-09-06-deconv-inverseq-research.md`（commit 79f448a）。
> 状态：实现中。两个方法分开注册为独立 registry 条目。

## 0. 注册名与方法定位（定稿）

| 注册名 | 中文名 | native category | core category | block_axis | auto_tune family/stage |
|---|---|---|---|---|---|
| `mixed_phase_deconvolution` | 混合相位反褶积 | `denoise` | `denoising` | `loaded_global` | `denoise`/`denoise` |
| `inverse_q` | Inverse-Q 衰减补偿 | `filter` | `filtering` | `loaded_global` | `frequency`/`frequency` |

- 两者都是**全测线全局**方法（kurtosis/增益整形需要全样本统计），block_axis=`loaded_global`。
- 实现文件：`mygpr/infrastructure/processing/algorithms/extended/deconvolution.py`、
  `extended/inverse_q.py`；经 `extended/__init__.py` 包装 `native_*` 后注册进
  `NATIVE_ALGORITHMS`（单一事实源）。**无需 PythonModule wrapper、无需 bindings 条目**
  （`core/methods_registry.py` 对无 bindings 条目的方法自动 `_legacy_adapter` 包装）。
- 无 PythonModule 兼容门面（新方法，无历史 API 兼容义务）。

## 1. mixed_phase_deconvolution（Schmelzbach & Huber 2015）

### 1.1 数学（v1 蓝本 = S&H 2015 IEEE TGRS + RGPR 对标）

**步骤一：逐道最小相位 spike 反褶积（supertrace 稳定化）**

1. 对每个道位 c，取相邻 `supertrace_traces`（默认 11）道求平均 → supertrace s(t)。
2. 对 supertrace 自相关（lag 0..`operator_length`−1，默认 35）构造 Toeplitz 正态方程
   `R a = e`，加 prewhitening `prewhitening·max(ACF)`（默认 0.1，范围 0–0.1，EKKO/Geolitix 口径）
   对角加载后解 `a = m⁻¹`（最小相位逆算子）。超级道在边缘做复制填充。
3. 由 `a` 的自相关再解一次 Toeplitz 系统（Levinson/Durbin 递推）得到最小相位等效子波
   `m`（S&H 两步法：先估 m⁻¹ 再由其 ACF 估 m）。
4. 每道输出 = 该道与 `m⁻¹` 的卷积（`np.convolve(mode="same")` 等价 FFT 实现）。
   逐道复用同一算子集合（supertrace 网格：每 `supertrace_traces` 道一组算子，组内共享）。

**步骤二：全局相位旋转（kurtosis 最大 = 最稀疏输出）**

1. 输出 y(t) 的解析信号（Hilbert）做常相位旋转 `θ ∈ [0°, 180°)`，步进 `rotation_step_deg`
   （默认 5°；S&H 合成例最优 122°、实测 134°/92°）。
2. 对每个候选 θ 计算全测线（**全部样本**，kurtosis 收敛需要大样本量，不可短窗）的
   kurtosis（峰度）`K(θ) = E[y⁴]/E[y²]² − 3`，取 K 最大的 θ。
3. 最终输出 = 最优 θ 旋转后的实部。
4. metadata 记录：`optimal_rotation_deg`、`kurtosis`、`supertrace_traces`、`operator_length`、
   `prewhitening`。

### 1.2 参数 schema（registry 定稿）

```python
parameter_schema=_schema(
    operator_length={"type": "int", "default": 35, "min": 5},          # 反褶积算子长度(样本)
    supertrace_traces={"type": "int", "default": 11, "min": 1},        # 合成 supertrace 道数
    prewhitening={"type": "float", "default": 0.1, "min": 0.0, "max": 0.1},  # 白化系数
    window_start_ns={"type": "float", "default": 0.0},                 # 估计窗起点(0=顶)
    window_end_ns={"type": "float", "default": 0.0},                   # 估计窗终点(0=底)
    rotation_step_deg={"type": "float", "default": 5.0, "min": 0.5},   # 相位扫描步进
    apply_phase_rotation={"type": "bool", "default": True},            # 关闭则只做最小相位步
)
```

### 1.3 时间基准契约（严于 frequency_filter_1d）

- dt 解析顺序：`time_step_s` → `sample_interval_ns` 参数；再 `time_window_ns`/`total_time_ns`/ns 窗 ÷ 样本数。
- **全缺失 → 中文 ValueError**（不静默跳过）：`ValueError("缺少时间基准参数（time_step_s/sample_interval_ns/total_time_ns），无法解析采样间隔；请在参数或数据头中提供。")`
- 仅 `window_start_ns`/`window_end_ns` 窗口裁剪需要 dt；dt 缺失时若用户没有改窗口默认
  （0–0 = 全道），仍报 ValueError（契约一致性优先）。

### 1.4 数值保护

- supertrace 全零/能量低于 `1e-12·全局RMS` → 该组算子退化为恒等（不处理）+ warning
  `deconv_degenerate_supertrace`。
- 输出 `normalize_output` 收尾（float32、shape 不变、NaN 清洗）。
- 相对成本 `relative_cost="medium"`，memory_multiplier=4.0（全道解析信号 + kurtosis 扫描矩阵
  按候选角度分块算，峰值内存 ~2×数据×float64）。

## 2. inverse_q（Wang 2002 稳定化逆 Q 滤波）

### 2.1 数学

constant-Q 衰减正演（Turner 1994 / Bano 1996，GPR 带宽内单参数 Q* 成立）：

```
A(t, f) = exp(−π·t·f/Q)        # 振幅衰减
φ(t, f) = exp(−i·π·t·f/Q · (2/π)·ln(f/f_h))   # 速度色散相位（Wang 2002 式）
```

补偿（Wang 2002 稳定化逆 Q 滤波，(t,f) 域逐道）：

1. 逐道 FFT → 频谱 `X(f)`；对每个样本时刻 t 与频率 f：
   `G(f, t) = exp(π·t·f/Q) · exp(i·(2/π)·π·t·f/Q·ln(f/f_ref))`（振幅恢复 + 色散相位校正）。
2. **Wang 稳定化**：`G_stab = G / (1 + (β·G/max|G|)²)` 的等价实现——
   `A_comp(f,t) = A(f,t) / (A(f,t)² + σ²) · A(f,t)`，其中 `σ = 10^(−gain_limit_db/20)` 是
   由增益上限导出的稳定化阈值：任何频点的**总补偿增益**钳制在
   `10^(gain_limit_db/20)` 以内（硬钳制，非软衰减）。
3. 逐点增益 `g(t,f) = min(exp(π·t·f/Q), G_MAX)`，`G_MAX = 10^(gain_limit_db/20)`。
4. `Y(f) = X(f) · g(t,f) · exp(i·φ_disp(t,f))` → 逐道 IFFT。
5. 超限比例：`max_applied_gain_db` = 实际最大施加增益（dB）；若有频点被钳制，
   发 warning `inverse_q_gain_clamped`（附 clamped_fraction）。

**v1 简化决策**：色散相位项 `(2/π)·ln(f/f_ref)` 一并施加（Wang 2002 完整式），
f_ref 取 `f_max/2`。fixture 中验证相位项不改能量、只改波形对齐。

### 2.2 参数 schema（registry 定稿）

```python
parameter_schema=_schema(
    q_value={"type": "float", "default": 50.0, "min": 1.0},    # Q 值（必填语义：无有效值时报错）
    gain_limit_db={"type": "float", "default": 40.0, "min": 10.0, "max": 100.0},  # 硬钳制上限
    time_step_s={"type": "float", "default": 0.0},              # 采样间隔(s)，缺失走时间基准解析
    sample_interval_ns={"type": "float", "default": 0.0},       # 同上 ns 口径
)
```

- **时间基准契约同 §1.3**：四级解析，全缺失 → 中文 ValueError。
- `q_value <= 0` 或非有限 → ValueError（`q_value` 必填语义；UI default 50 是起始先验，
  语义上用户必须确认）。
- `gain_limit_db` 钳制在 [10, 100]（Wang 建议的稳定化范围；超范围静默钳制 + warning
  `inverse_q_gain_limit_adjusted`）。

### 2.3 数值保护

- 全零输入 → 恒等输出 + warning `inverse_q_silent_input`（避免 0×exp 爆 NaN）。
- 逐道计算在 FFT 域进行（rfft），输出 `normalize_output` 收尾。
- metadata 记录：`q_value`、`gain_limit_db`、`max_applied_gain_db`、`clamped_fraction`、
  `sample_rate_hz`、`f_ref_hz`。
- relative_cost `"medium"`，memory_multiplier=4.0。

## 3. 处理链位置

```
dewow → set_zero_time → [inverse_q] → 背景抑制 → 去噪 → [mixed_phase_deconvolution] → 增益 → 偏移
```

- inverse-Q 在零位后、deconv 前（平稳性假设需要衰减先补偿）。
- deconv 在背景抑制/去噪后、增益前。
- **不内置自动 bandpass**：方法 docstring + metadata `post_advice` 字段注明
  "建议后接 frequency_filter_1d 带通压制反褶积放大的带外噪声"（RGPR/REFLEXW 实操）。

## 4. 注册触点清单（全部文件）

1. `mygpr/infrastructure/processing/algorithms/extended/deconvolution.py`（新建）
2. `mygpr/infrastructure/processing/algorithms/extended/inverse_q.py`（新建）
3. `extended/__init__.py`：import 两个 `method_*` + 两个 `native_*` 包装（追加到 wavelet 之后）
4. `algorithms/methods.py`：import 两个 `native_*`；NATIVE_ALGORITHMS 末尾（rpca_background 后）追加两条
5. `core/method_registry_metadata.py`：
   - METHOD_METADATA 追加两键（rtm_migration 条目后）
   - PREFERRED_METHOD_ORDER：deconv 插 `wavelet_2d` 后；inverse_q 插 `frequency_filter_1d` 后
   - METHOD_TAGS：两键 "实验"
   - AUTO_TUNE_STAGE_BY_METHOD：deconv→`denoise`，inverse_q→`frequency`
6. `core/method_registry_groups/background_denoise.py`：deconv 条目（hilbert_envelope 后、wavelet_svd 区块前——最终 wavelet_svd 删除后自然居中）；inverse_q 条目（frequency_filter_1d 后）
   - 注意 group 文件 `"func"` 引用的是 bindings/legacy callable；新方法无 bindings →
     **`"func"` 直接给 `_legacy_adapter` 等价的可调用**：从
     `mygpr.infrastructure...extended` 导入 method 函数本身（`(data, **kwargs)` 签名），
     或直接引用 `core.methods_registry` 构建后的 PROCESSING_METHODS 里的包装函数。
     实现时采用**导入 method 函数**（`(data, params_dict)` 签名不兼容 legacy
     `(data, **kwargs)` 调用——但 group 条目经 methods_registry 重建后 func 已被
     `_legacy_callables` 优先；核对后若无冲突直接给 method 函数即可，实际调用都走
     native_func）。
     **定稿**：group 文件给 `func` 为 extended 的 `method_*`（**kwargs 签名）——两个新
     方法的 method 函数签名统一为 `(data, **kwargs)` 风格（与 extended 现有方法一致），
     由 `extended/__init__.py` 的 `_execute` 桥接 params dict → kwargs。
7. `core/field_processing_bridge.py`：METHOD_CATEGORY_OVERRIDES（deconv→去噪增强、
   inverse_q→频率滤波）+ METHOD_DISPLAY_NAMES 两键（fallback 会用 registry name，但显式
   中文更好）。
8. `tests/test_deconv_inverse_q.py`（新建）：见 §5
9. `tests/test_native_convergence_baseline.py`：EQUIVALENCE_EVIDENCE 两条
   （`determinism_contract`/`tests/test_deconv_inverse_q.py`）
10. `tests/fixtures/processing_convergence/descriptor_baseline.json`：+2 条
    （wavelet_svd 删除合并后一次到 37）
11. `core/benchmark_registry.py`：+2 fixture（见 §6）
12. `scripts/algorithm_fitness_benchmark.py`：+2 场景函数 + SCENES 注册 + TASKS +2 行 + 打分逻辑

## 5. 单元测试（tests/test_deconv_inverse_q.py）

1. **契约**：随机 2D 输入 → shape/dtype(float32)/finite 不变；metadata 含 method_id。
2. **缺时间基准 → ValueError**：两方法在无任何时间基准参数时 raise ValueError（中文消息）。
3. **deconv 合成验收**：已知带限反射率（稀疏脉冲列）⊛ 混合相位子波（最小相位子波 + 已知
   122° 旋转）+ 微噪 → deconv 后与带限反射率的互相关峰值高于 deconv 前。
4. **deconv determinism**：同 request 双跑 `assert_array_equal`；metadata
   `optimal_rotation_deg` 一致。
5. **inverse_q 谱恢复**：已知子波经 constant-Q 正演衰减（Q=50, dt=1.397ns, t 到 700ns）
   → inverse_q(Q=50) 后频谱能量比恢复到衰减前的合理邻域（>0.7 倍）；
   `max_applied_gain_db ≤ gain_limit_db`。
6. **inverse_q 增益钳制**：Q=10 + gain_limit_db=40 → metadata `max_applied_gain_db` ≈ 40
   （±0.5），`clamped_fraction > 0`，输出 finite。
7. **inverse_q determinism**：同 request 双跑 `assert_array_equal`。
8. **EQUIVALENCE_EVIDENCE** 扩展两条 + descriptor_baseline.json +2。

## 6. benchmark fixture 与打分

### 6.1 core/benchmark_registry.py +2 fixture

- `deconv_reference`：501×96、dt=1.397ns（对齐英山）。5 个稀疏反射脉冲列（深度递增、
  双程走时已知）⊛ 混合相位子波（Burgers/min-phase 长子波 + 90° 恒定旋转）+ 0.02 噪声。
  metadata：`ground_truth_reflectivity`（稀疏矩阵）、`wavelet_rotation_deg`、
  `header_info(total_time_ns=700)`。scenario="deconvolution"（BENCHMARK_SCENARIOS 加键）。
- `inverse_q_reference`：501×96、dt=1.397ns。顶层反射子波（带限 Ricker）经
  Q=40 正演 exp(−πtf/Q) 衰减 + 色散相位 + 0.01 噪声 → 存 `ground_truth_undamped`
  （未衰减原剖面）。scenario="inverse_q"（BENCHMARK_SCENARIOS 加键）。
- `default_methods`/`focus_metrics` 填新方法 id（注册校验循环会核对 PROCESSING_METHODS）。

### 6.2 scripts/algorithm_fitness_benchmark.py

- 场景函数 `scene_deconv()` / `scene_inverse_q()`：调 generate_benchmark_sample 取 fixture，
  meta 补 `clean_reference`（deconv→反射率列稀疏版；inverse_q→ground_truth_undamped）。
- TASKS 追加：
  ```python
  ("deconv", "mixed_phase_deconvolution", {}, "denoise"),
  ("inverse_q", "inverse_q", {"q_value": 40.0, "gain_limit_db": 40.0}, "filter"),
  ```
  family 复用 `denoise`/`filter`（现有打分分支走 denoise 的 SNR 路线需要
  clean_reference；filter 族对 inverse_q 需要专属打分）。
- `_score_task` 追加两分支：
  - `mixed_phase_deconvolution`：反射率恢复互相关（after vs gt_reflectivity 列脉冲窗内
    峰值对齐 acc）+ SNR 分量（走 denoise clean_reference 路线即可，meta 塞 clean_reference）。
  - `inverse_q`：谱恢复比 = 频带能量比(after/undamped) 相对 1 的接近度 +
    `max_applied_gain_db ≤ gain_limit_db` 的契约分 + 晚时能量提升比。

## 7. wavelet_svd 删除（终检清单）

按调研会话 18 触点 + 本会话复核（config/schema_catalog.json 无引用，grep=0）：

1. `cli_batch.py:160` — OPTIONAL_METHOD_DEPENDENCIES 移除条目
2. `core/field_processing_bridge.py:95,132` — 两 dict 移除
3. `core/method_registry_bindings.py` — `_method_wavelet_svd` 声明/import/missing 包装全删
   （wavelet_2d 保留）
4. `core/method_registry_groups/background_denoise.py:404-469` — wavelet_svd 条目全删
5. `core/method_registry_metadata.py:119,240,271,327` — 四处
6. `mygpr/application/autotune/candidate_planner.py:125-` — wavelet_svd 特例块删
7. `mygpr/application/autotune/refinement.py:217-221` — elif 分支删
8. `mygpr/application/processing/service.py:114` — 集合中移除
9. `mygpr/domain/autotune/constraints.py:133-138` — elif 条件收窄为 svd_subspace，
   内层 wavelet_svd 块删（levels/threshold clamp 只属 wavelet_svd）
10. `algorithms/methods.py:31,148-152` — import + 条目
11. `extended/__init__.py:16,64-65` — import + wrapper
12. `extended/wavelet.py:148-221` — method_wavelet_svd + __all__ 条目
13. `scripts/algorithm_fitness_benchmark.py:266` — 任务行
14. `tests/test_native_convergence_baseline.py:12,96,148-151` — docstring/evidence/param case
15. `tests/test_round2_processing_kernels.py:36,633-664,697-712,713-728` — import + 3 测试
16. `tests/fixtures/processing_convergence/descriptor_baseline.json:1255` — 条目
17. `PythonModule/wavelet_svd.py` — 删文件
18. wavelet_2d 与 wavelet_svd 共享的 `_threshold_details` 等帮助函数**保留**在 wavelet.py

终检：`grep -rn wavelet_svd core/ config/ mygpr/ tests/ PythonModule/ scripts/ cli_batch.py` → 0 命中（docs 例外）。

## 8. 执行顺序

1. 实现两方法 + 注册（§4.1-7）→ 快速冒烟（契约测试）
2. 单元测试全绿 → EQUIVALENCE_EVIDENCE +2（此时 baseline.json 是 36+2=38 中间态）
3. benchmark fixture/任务/打分 → `--run-all` 局部验证新任务得分非 0
4. 删 wavelet_svd（§7 清单）→ baseline.json 终态 37（36−1+2）
5. 全量 pytest（预期 816−3+新测试数）→ 定向 UI 子集
6. 逐文件 git add（避开 ui/main_window.py、ui/pages/spatial_page.py 用户 WIP、
   autoresearch.sh、phase4 spec）→ 中文提交 → push feat/phase3-depth-slice
