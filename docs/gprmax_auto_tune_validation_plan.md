#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPRMAX validation plan for MyGPR auto-tune research."""

# GPRMAX 正演验证自动选参计划

## 背景

组会决策：真实外业数据很难判断哪些双曲线、裂缝、层状结构或弱反射必须被保存，也容易受到未知地下结构、采集姿态、含水率和噪声影响。因此，MyGPR 自动选参需要一套 GPRMAX 正演数据作为可控 benchmark，用 ground truth 验证“自动选出的参数确实合适”，而不是只让图像看起来更干净。

## 优先级

当前优先级排序：

1. 完成自动选参科研对比导出闭环。
2. 建立 GPRMAX benchmark schema 和最小正演样例。
3. 用 GPRMAX ground truth 改进自动选参评分。
4. 用真实 UAV-GPR 外业数据做最终验收。

GPRMAX 应排在真实数据之前，因为它可以先回答“目标是否被保留”这个真实数据很难回答的问题。

## 对旧目录的判断

本机已有目录：`E:\gprMax\gprMax-v.3.1.7`。

已看到的可参考内容：

- `user_models/cylinder_Bscan_2D.in`：标准圆柱体 B-scan 示例，适合做最小双曲线 ground truth。
- `user_models/cylinder_Bscan_2D*.out`：已有按道输出，可用于测试 MyGPR 的 gprMax `.out` 读取。
- `crack_model_generator.py`：裂缝模型生成思路，可参考几何建模。
- `landslide_model_generator.py`：滑坡/复杂介质生成思路，可参考多材料场景。
- `gprmax_test/landslide_model.in`：已有滑坡模型输入文件。

不建议直接复用该目录作为 MyGPR benchmark：

- 混有原版 gprMax、venv、GUI 试验包和生成数据，不是干净数据集。
- 多处脚本存在硬编码输出路径，例如 `D:\ClawX-Data\sim\gprmax_outcsv`。
- 旧输出缺少 MyGPR 需要的 ground-truth manifest，无法严格评估目标保真。
- 不同脚本的坐标、深度、材料 ID 和输出命名不统一。

结论：旧目录用于参考，不作为稳定依赖。MyGPR 应重新建立自己的 `scenario -> gprMax input -> output -> MyGPR CSV/HDF5 -> ground truth manifest` 契约。

## 最小 benchmark 契约

每个 GPRMAX scenario 至少输出：

- `scenario.json`：场景参数、材料、天线、步进、时间窗、随机种子。
- `model.in`：可运行的 gprMax 输入。
- `raw_out/`：gprMax 原始 `.out` 或 merged `.out`。
- `mygpr_bscan.csv`：MyGPR 可直接读取的 B-scan。
- `ground_truth.json`：目标类型、目标位置、apex、层位、ROI、应保留结构。
- `preview.png`：几何模型或 B-scan 预览。

`ground_truth.json` 建议字段：

```json
{
  "schema": "mygpr_gprmax_ground_truth_v1",
  "scenario_id": "cylinder_single_v1",
  "targets": [
    {
      "target_id": "metal_cylinder_01",
      "type": "hyperbola",
      "apex_trace_idx": 60,
      "apex_time_ns": 8.5,
      "roi": {
        "time_start_idx": 80,
        "time_end_idx": 180,
        "dist_start_idx": 35,
        "dist_end_idx": 85
      },
      "must_preserve": true
    }
  ],
  "known_background": {
    "horizontal_layers": [],
    "air_ground_interface": null
  }
}
```

## 第一批场景

1. `cylinder_single_v1`
   - 单个金属圆柱体。
   - 目标：验证双曲线 apex、左右臂和目标能量保留。

2. `cylinder_double_depth_v1`
   - 两个不同埋深圆柱体。
   - 目标：验证浅部强目标不会让自动选参误删深部弱目标。

3. `layered_soil_interface_v1`
   - 水平层状介质。
   - 目标：验证背景抑制和去噪不破坏有效层位。

4. `crack_air_filled_v1`
   - 空气裂缝。
   - 目标：验证弱线性/倾斜结构保留。

5. `no_target_noise_v1`
   - 无目标，仅噪声与背景。
   - 目标：验证自动选参不会凭空制造假异常。

## 自动选参评分改进方向

GPRMAX 数据接入后，自动选参评分应增加 ground-truth aware 指标：

- `target_roi_energy_preservation`：目标 ROI 能量保留。
- `apex_saliency_preservation`：双曲线 apex 显著性保留。
- `hyperbola_arm_continuity`：双曲线臂连续性。
- `background_suppression_outside_roi`：目标外背景抑制。
- `false_positive_penalty`：无目标区域假异常惩罚。
- `over_smoothing_penalty`：边缘、apex、局部极值被抹平的惩罚。

自动选参结论不能只写“图像更清晰”。报告必须说明：

- 对哪些 ground-truth 目标更好。
- 哪些目标被削弱或存在风险。
- 评分改善来自目标保留、背景抑制还是显示增益。

## 已落地的最小样例

已新增 `cylinder_single_v1` 最小 benchmark 包：

- 生成脚本：`scripts/gprmax_benchmark/generate_cylinder_single_v1.py`
- 样例目录：`sample_data/gprmax_benchmarks/cylinder_single_v1/`
- 场景文件：`scenario.json`
- gprMax 输入：`model.in`
- ground truth：`ground_truth.json`
- MyGPR 输入：`mygpr_bscan.csv`
- 预览图：`preview.png`

可执行命令：

```powershell
python scripts\gprmax_benchmark\generate_cylinder_single_v1.py
python -m pytest tests\test_gprmax_benchmark_package.py -q
```

当前 bundled B-scan 是 deterministic fallback，作用是先固定 MyGPR 的 scenario、ground truth、CSV 读取、自动选参对比和证据导出契约。它不是完整电磁正演结果。后续若传入真实 gprMax `.out`：

```powershell
python scripts\gprmax_benchmark\generate_cylinder_single_v1.py --raw-out-path E:\gprMax\gprMax-v.3.1.7\user_models\cylinder_Bscan_2D1.out
```

脚本会通过 `core.gpr_io.read_gprmax_out` 将同目录的多道 `.out` 按数字顺序合并为 `mygpr_bscan.csv`，并保留同一套 `scenario.json` / `ground_truth.json` 契约。

`ground_truth.json` 当前固定了：

- 目标：`metal_cylinder_01`
- 类型：`hyperbola`
- apex：trace 40, sample 79, time 约 6.46 ns
- ROI：sample `[49, 149)`, trace `[16, 65)`
- 必须保留结构：apex、左臂、右臂

## 下一步实施建议

1. 等本机 gprMax 环境确认后，增加可选 smoke 脚本真正运行 `model.in` 正演。
2. 将真实 `.out` 转换结果与当前 deterministic fallback 放在同一 scenario 契约下比较。
3. 为 `cylinder_single_v1` 增加 ground-truth aware 指标：apex 显著性、双曲线臂连续性、目标 ROI 能量保留、目标外假异常惩罚。
4. 新增 `cylinder_double_depth_v1`，验证浅部强目标不会让自动选参误删深部弱目标。
5. 新增 `layered_soil_interface_v1` 和 `crack_air_filled_v1`，覆盖层位和裂缝结构。
