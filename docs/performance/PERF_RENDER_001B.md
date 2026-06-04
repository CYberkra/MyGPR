# PERF-RENDER-001B：B-scan 渲染与交互节流优化

版本：0.8.56
范围：显示/交互流畅度优化，不改变算法数组结果、AutoTune 评分、候选生成器或 Evidence schema。

## 修改内容

1. 新增主画布绘制合并器：`_request_main_canvas_draw(...)`
   - 高频交互不再直接多次调用 `canvas.draw_idle()`。
   - pan、ROI preview、slider compare、临时 overlay 清理统一走 bounded draw request。
   - 默认 60 FPS 上限；超出频率的请求合并到单次 QTimer flush。

2. B-scan 鼠标悬停坐标查找优化
   - 原路径使用 `np.argmin(abs(axis-value))` 扫描完整 axis。
   - 新路径使用 monotonic-axis `np.searchsorted` 寻找邻近索引。
   - 对 2000+ 道数据，鼠标移动时减少不必要 O(N) 扫描。

3. ROI / pan / slider compare 绘制路径统一节流
   - ROI 拖动只更新矩形 overlay。
   - 中键/右键 pan 只更新 axis limits，绘制请求按帧合并。
   - slider compare 继续复用轻量 clip/divider update，不重建整张 B-scan。

4. 性能监控字段扩展
   - 记录 `display.canvas_draw_request.*`。
   - 记录 `display.canvas_draw_flush.*`。
   - 用于后续判断高频事件是否被合并。

## 不改变的内容

- 不改处理算法。
- 不改 AutoTune scoring v2。
- 不改 candidate generator。
- 不改 gprMax / GPR Scene Studio 数据链路。
- 不改 Evidence manifest / trial schema。
- 不改变 B-scan 数组，只改变渲染请求调度。

## 验证

```text
python -m py_compile app_qt.py core/*.py ui/*.py
python scripts/check_version_consistency.py
QT_QPA_PLATFORM=offscreen MPLBACKEND=QtAgg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_perf_monitor.py tests/test_bscan_interaction_controller_gui.py tests/test_bscan_display_export.py tests/test_daily_processing_smoke.py tests/test_version_consistency.py -q
QT_QPA_PLATFORM=offscreen MPLBACKEND=QtAgg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_gui_presets.py::test_main_canvas_middle_drag_pans_like_grabbing_image -q
```

结果：

```text
version_check_ok: 0.8.56
12 passed, 1 warning
1 passed, 1 warning
```

warning 来自 qfluentwidgets 内部 scipy deprecated import，不是本轮修改导致。

## 截图与 smoke

使用 `501 × 2378` synthetic B-scan 做 offscreen smoke：

- 空状态截图
- 加载后截图
- 重复绘制缓存截图

性能监控显示 `display.prepare_view_cache_hit_ms` 和 `display.vmin_vmax_cache_hit_ms` 生效；`display.canvas_draw_request.plot_data` / `display.canvas_draw_flush.plot_data` 可用于后续判断绘制请求是否被合并。

## 后续建议

下一阶段进入 `PERF-AUTOTUNE-001C`：

- AutoTune progress 信号节流。
- trial table / 候选表批量刷新。
- 日志 flush 策略继续检查。
- 运行中 B-scan 只在正式步骤完成后刷新，候选 sweep 阶段不刷图。
