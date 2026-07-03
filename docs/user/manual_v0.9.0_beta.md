# MyGPR v0.9.0 beta 用户手册

## 版本定位

v0.9.0 beta 是真实项目试用版，适合用营山实测 CSV 做项目建档、测线导入、数据质检、B-scan 预览、坐标投影、空间成果检查和初步报告页浏览。

当前不承诺完整生产交付闭环；PDF/Excel 正式报告导出、厂商原生 DZT/RD3/DT1 解码和智能识别模型仍属于后续任务。

## 推荐工作流

1. 启动软件并新建项目。
2. 在项目设置中确认坐标系统，营山工程推荐使用 `CGCS2000 / 3-degree GK Zone 39` 或自动分带表达。
3. 导入单条测线或使用“批量导入测线”。
4. 批量导入结果表中检查成功/失败、矩阵尺寸、长度、耗时和诊断。
5. 运行数据质检。
6. 在“查看质检详情”中检查方向、NaN/Inf、轨迹点数和采样参数。
7. 仅当质检提示 `transpose_risk` 或 B-scan 明显方向错误时使用“修正B-scan方向”。
8. 切换到测线处理、空间成果和成果报告页检查联动结果。
9. 使用“导出清单”输出当前测线清单。
10. 使用“项目备份”生成当前项目 ZIP 备份。

## CSV 导入要求

优先支持旧 MyGPR / 营山实测主数据 CSV：

```text
Number of Samples = N
Time windows (ns) = T
Number of Traces = M
Trace interval (m) = dx
longitude, latitude, elevation, amplitude, height[, timestamp]
```

导入后会生成：

```text
raw/<line_id>/<line_id>_gpr_dataset.npz
raw/<line_id>/<line_id>_gpr_meta.json
raw/<line_id>/<line_id>_trajectory.csv
raw/<line_id>/import_manifest.json
raw/<line_id>/<line_id>_quality_report.json
```

## 批量导入

批量导入支持一次选择多个 CSV。文件名会尝试自动识别测线：

```text
Line9origin(30).csv -> L09_30 / 9号测线（30）
Line3origin.csv     -> L03 / 3号测线
L1origin.csv        -> L01 / L1号测线
X1origin.csv        -> X1 / X1号测线
```

失败文件不会中断后续导入。导入结果表可查看 raw 目录、manifest，或复制诊断信息。

## 数据质检与方向修正

质检检查：

- 采样点数；
- 道数；
- 时间窗；
- 测线长度；
- 振幅范围；
- NaN/Inf 比例；
- 轨迹点数；
- B-scan 方向风险。

方向修正不会自动执行。执行后会备份原数据并写入：

```text
raw/<line_id>/orientation_fixes/
raw/<line_id>/orientation_fix_manifest.json
```

## 当前 beta 限制

- 厂商原生 DZT/RD3/DT1 仍未完整解码。
- PDF/Excel 正式报告导出尚未闭环。
- 空间成果已使用工程坐标绘制轨迹，但目标点正式成果表仍需增强。
- 取消批量导入只能在文件之间生效，不能中断正在解析的单个百万行 CSV。
