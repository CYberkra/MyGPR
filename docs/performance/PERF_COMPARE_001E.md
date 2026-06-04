# PERF-COMPARE-001E 显示与对比页性能优化记录

版本：0.8.59

## 修改范围

本轮只修改显示与对比相关 UI 刷新路径，不修改算法数组、AutoTune 评分、candidate generator、gprMax 数据链路或 Evidence schema。

## 主要优化

1. **处理链路 stepper 签名缓存**
   - 主图刷新、滑动对比和显示参数切换会频繁调用链路条刷新。
   - 现在通过 `ProcessingLineageController._stepper_signature(...)` 判断可见链路状态是否变化。
   - 状态未变化时跳过 QPushButton / selector 全量重建，只刷新详情与对比篮状态。

2. **对比模式共享色阶计算优化**
   - 双图 / 滑动对比默认对称显示时，不再将两幅大图像 finite pixels 拼接成一个大数组再求 vmin/vmax。
   - 改为逐数组计算最大绝对值并合并，保持默认显示结果等价，同时减少临时内存拷贝。
   - 百分位/auto-contrast 模式仍走精确联合分布路径，避免改变显示语义。

3. **对比下拉框刷新跳过**
   - 对比快照标签未变化时，不再 clear/add 三个 QComboBox。
   - 减少显示与对比页切换、链路回看、plot refresh 触发的信号抖动。

4. **性能计数项扩展**
   - `display.lineage_stepper_rebuild_ms`
   - `display.lineage_stepper_skip_ms`
   - `display.compare_combo_refresh_ms`
   - `display.compare_combo_skip_ms`
   - `display.compare_shared_vmin_vmax_fast_ms`
   - `display.compare_shared_vmin_vmax_exact_ms`

## 边界

- 所有优化均为 display-only / UI-refresh 优化。
- 不改变正式处理结果。
- 不改变 trial table / manifest / claim boundary。
- 不改变 AutoTune 推荐逻辑。

## 验证

已执行：

```bash
python scripts/check_version_consistency.py
python -m py_compile app_qt.py core/*.py ui/*.py
QT_QPA_PLATFORM=offscreen MPLBACKEND=QtAgg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/test_version_consistency.py \
  tests/test_processing_lineage_controller_gui.py \
  tests/test_bscan_interaction_controller_gui.py \
  tests/test_bscan_display_export.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 60s \
  python -m pytest tests/test_autotune_v1_config.py \
  tests/test_autotune_candidate_generator.py \
  tests/test_autotune_scoring_v2.py \
  tests/test_autotune_recipe_ui.py -q
```

结果：

- version_check_ok: 0.8.59
- display/compare GUI tests: 9 passed
- AutoTune/config/scoring regression tests: 16 passed
- py_compile passed

## 截图

- `mygpr_v0859_perf_compare_empty.png`
- `mygpr_v0859_perf_compare_loaded.png`
- `mygpr_v0859_perf_compare_slider.png`
- `mygpr_v0859_perf_compare_grid.png`
