---
name: run-benchmark
description: Run MyGPR processing pipeline benchmarks and compare with baseline to detect performance regressions
disable-model-invocation: true
---

# Run MyGPR Benchmarks

运行 GPR 处理管道性能基准测试，与基线对比检测性能回归。

## 使用方式

```
/run-benchmark              # 运行所有基准
/run-benchmark auto-tune    # 只跑 auto_tune 相关基准
/run-benchmark compare      # 与上次基线对比
```

## 运行命令

```bash
cd D:/Claude/MyGPR && python -m pytest -m slow --tb=short -q
```

## 基准模块

- `core/benchmark_registry.py` — 基准注册表
- `core/benchmark_runner.py` — 基准执行器

## 输出

基准结果包含：
- 处理管道各阶段耗时
- 与历史基线的差异百分比
- 内存使用峰值
- 推荐操作（通过/需调查/需优化）
