---
name: run-tests
description: Run MyGPR pytest suite by marker or by changed files. Automatically selects the right test marker based on what was modified.
---

# Run MyGPR Tests

根据改动范围自动选择 pytest 标记精准运行测试。

## 使用方式

```
/run-tests              # 自动判断：检查最近改动的文件，选择对应标记
/run-tests unit         # 只跑单元测试（核心算法）
/run-tests gui          # 只跑 GUI 测试
/run-tests all          # 全量测试（unit + integration）
/run-tests wavelet      # 只跑小波变换相关
```

## 自动判断逻辑

检查最近改动的 Python 文件路径：
- 只改了 `core/` → `python -m pytest -m unit --tb=short -q`
- 改了 `ui/` → `python -m pytest -m "unit or gui" --tb=short -q`
- 改了 `tests/` → `python -m pytest -m unit --tb=long -q`
- 全量验证 → `python -m pytest -m "unit or integration" --tb=short -q`
- 波数变换相关 → `python -m pytest -m wavelet --tb=short -q`

## 运行命令

```bash
cd D:/Claude/MyGPR && python -m pytest -m unit --tb=short -q
```

## 常用 pytest 选项

- `--tb=short` — 简短回溯
- `--tb=long` — 完整回溯（调试用）
- `-q` — 安静模式
- `--maxfail=3` — 3 个失败后停止
- `-x` — 首个失败后停止
- `--lf` — 只跑上次失败的测试
