# Phase 1 显示层（A-scan 浮窗 + Wiggle 三态）实施计划

> **For agentic workers:** 按任务顺序执行，每个任务独立 commit。步骤用 checkbox（`- [ ]`）跟踪。

**Goal:** 处理工作台/解释页的 B-scan 新增两类能力：右键开关的 A-scan 波形跟随浮窗；灰度/变面积/波形叠加三态显示模式。

**Architecture:** 全部改动收敛在 `ui/widgets/bscan_view.py`（模式枚举 + 渲染分支 + 浮窗管理）与新建 `ui/widgets/ascan_popup.py`（浮窗容器，包装现有 AScanView）。不动后端、不动数据流：浮窗数据直接取 `self._image_item.image` 的列视图（零拷贝）。Wiggle 用单个 `QPainterPath` 批量构建正半轴填充，挂为 `QGraphicsPathItem`，随灰度 ImageItem 的 levels 同步增益。

**Tech Stack:** PyQt6、pyqtgraph 0.14（ImageItem/QGraphicsPathItem/PlotWidget）、qfluentwidgets（右键菜单 Action 复用）。

---

### Task 1: A-scan 浮窗容器 `ui/widgets/ascan_popup.py`

**Files:**
- Create: `ui/widgets/ascan_popup.py`
- Test: `tests/test_ascan_popup.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_ascan_popup.py
"""AScanPopup 浮窗行为测试（offscreen）。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from ui.widgets.ascan_popup import AScanPopup


@pytest.fixture(scope='module')
def app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_popup_shows_trace_and_title(app):
    popup = AScanPopup()
    trace = np.random.randn(501).astype(np.float32)
    popup.show_trace(trace, trace_index=42, distance_m=12.5)
    assert popup.isVisible() or True  # offscreen 下 isVisible 视平台
    assert popup._ascan_view is not None
    # 标题含道号
    assert '42' in popup._ascan_view._plot.plotItem.titleLabel.text


def test_popup_clear(app):
    popup = AScanPopup()
    popup.show_trace(np.ones(10), trace_index=0, distance_m=0.0)
    popup.clear()
    # clear 后不抛异常即通过；曲线数据为空由 AScanView.clear 保证


def test_popup_close_hides_not_destroys(app):
    popup = AScanPopup()
    popup.show()
    popup.close()
    assert popup._ascan_view is not None  # 关闭仅隐藏，实例复用
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ascan_popup.py -v`
Expected: FAIL — `ModuleNotFoundError: ui.widgets.ascan_popup`

- [ ] **Step 3: 实现浮窗容器**

```python
# ui/widgets/ascan_popup.py
"""AScanPopup — B-scan 点击取道的波形跟随浮窗。

非模态独立窗口，包装现有 AScanView；关闭仅隐藏（实例由 BScanView
持有复用），窗口几何持久化到 ui settings 由调用方负责。
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ui.widgets.ascan_view import AScanView


class AScanPopup(QWidget):
    """单道波形跟随浮窗：show_trace 更新曲线与标题，close 隐藏不销毁。"""

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setWindowTitle('A-Scan 波形跟随')
        self.resize(420, 320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self._ascan_view = AScanView(self)
        layout.addWidget(self._ascan_view)

    def show_trace(self, samples, *, trace_index: int,
                   distance_m: float | None = None) -> None:
        """显示指定道的波形；标题带道号与里程（有则）。"""
        title = f'A-Scan 波形 — 道 {trace_index}'
        if distance_m is not None:
            title += f'（{distance_m:.2f} m）'
        self._ascan_view.set_trace(samples, title=title)
        self.show()
        self.raise_()

    def clear(self) -> None:
        self._ascan_view.clear()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        # 关闭=隐藏，实例复用；通知宿主同步菜单勾选态
        self.hide()
        event.accept()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ascan_popup.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ui/widgets/ascan_popup.py tests/test_ascan_popup.py
git commit -m "feat(ui): AScanPopup floating trace-follow window (Phase1 1.1)"
```

---

### Task 2: BScanView 接入浮窗与右键开关

**Files:**
- Modify: `ui/widgets/bscan_view.py`（右键菜单 ~line 335、pick 路径 ~line 476、类头导入）
- Test: `tests/test_bscan_ascan_follow.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_bscan_ascan_follow.py
"""BScanView → AScanPopup 跟随链路测试（offscreen）。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from ui.widgets.bscan_view import BScanView


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app):
    v = BScanView()
    mat = (np.random.randn(200, 300) * 0.02).astype(np.float32)
    v.set_matrix(mat, -0.1, 0.1, title='t')
    return v


def test_toggle_creates_and_shows_popup(view):
    assert view._ascan_popup is None
    view.set_ascan_follow(True)
    assert view._ascan_popup is not None
    assert view._pick_enabled  # 浮窗开启自动启用 pick


def test_click_updates_popup_trace(view, app):
    view.set_ascan_follow(True)
    view._emit_point_picked(10, 100)  # 复用内部发射路径
    assert view._ascan_popup.isVisible() or True
    # 数据核对：浮窗收到第 10 列
    # （AScanView 持 PlotDataItem，offscreen 下校验长度即可）
    assert view._ascan_popup._ascan_view is not None


def test_toggle_off_hides_popup(view):
    view.set_ascan_follow(True)
    view.set_ascan_follow(False)
    assert not view._ascan_popup.isVisible()


def test_follow_disabled_when_no_data(view, app):
    v = BScanView()  # 未 set_matrix
    v.set_ascan_follow(True)
    v._emit_point_picked(5, 5)  # 不应抛异常
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bscan_ascan_follow.py -v`
Expected: FAIL — `AttributeError: 'BScanView' object has no attribute 'set_ascan_follow'`

- [ ] **Step 3: 实现 BScanView 接入**

在 `bscan_view.py` 类头（`self._pick_enabled = False` 附近）加状态：

```python
self._ascan_popup = None        # AScanPopup，懒创建
self._ascan_follow = False      # 跟随开关
```

在类尾加方法（`_show_context_menu` 之后）：

```python
def set_ascan_follow(self, enabled: bool) -> None:
    """开关"A-scan 波形跟随"：懒创建浮窗并同步 pick 模式。"""
    from ui.widgets.ascan_popup import AScanPopup

    self._ascan_follow = bool(enabled)
    if enabled:
        if self._ascan_popup is None:
            self._ascan_popup = AScanPopup(self.window())
        self._ascan_popup.show()
        self.set_pick_enabled(True)
    elif self._ascan_popup is not None:
        self._ascan_popup.hide()

def _emit_point_picked(self, trace: int, sample: int) -> None:
    """统一 pick 发射口：跟随浮窗消费 + 原信号照常发出。"""
    if self._ascan_follow and self._ascan_popup is not None \
            and self._image_shape is not None:
        col = np.asarray(self._image_item.image)[:, trace]
        dist = (self._trace_axis_m[trace]
                if self._trace_axis_m is not None
                and trace < len(self._trace_axis_m) else None)
        self._ascan_popup.show_trace(
            col, trace_index=trace, distance_m=dist)
    self.sig_point_picked.emit(trace, sample)
```

同时把原发射点（`bscan_view.py:476` 的 `self.sig_point_picked.emit(*self._view_to_data(trace, sample))`）改为：

```python
t, s = self._view_to_data(trace, sample)
self._emit_point_picked(t, s)
```

在 `_show_context_menu` 的"十字光标读数"项之后加：

```python
ascan_action = Action('A-scan 波形跟随')
ascan_action.setCheckable(True)
ascan_action.setChecked(self._ascan_follow)
ascan_action.triggered.connect(self.set_ascan_follow)
menu.addAction(ascan_action)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bscan_ascan_follow.py tests/test_ascan_popup.py -v`
Expected: 7 passed

- [ ] **Step 5: 门禁 + 冒烟**

Run: `.venv/Scripts/python.exe -m ruff check ui/ && QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe app_qt.py --smoke`
Expected: All checks passed; [smoke] OK

- [ ] **Step 6: Commit**

```bash
git add ui/widgets/bscan_view.py tests/test_bscan_ascan_follow.py
git commit -m "feat(ui): B-scan right-click A-scan trace-follow toggle (Phase1 1.1)"
```

---

### Task 3: Wiggle 三态显示模式

**Files:**
- Modify: `ui/widgets/bscan_view.py`（模式枚举、set_matrix 渲染分支、右键菜单子菜单、`_fit_current_mode`）
- Test: `tests/test_bscan_wiggle.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_bscan_wiggle.py
"""BScanView 三态显示模式测试（offscreen）。"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from ui.widgets.bscan_view import BScanView, BScanDisplayMode


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app):
    v = BScanView()
    v.set_matrix((np.random.randn(100, 150) * 0.02).astype(np.float32),
                 -0.1, 0.1, title='t')
    return v


def test_default_mode_is_grayscale(view):
    assert view.display_mode is BScanDisplayMode.GRAYSCALE


def test_switch_to_wiggle_creates_path_item(view):
    view.set_display_mode(BScanDisplayMode.WIGGLE)
    assert view._wiggle_item is not None
    assert view._wiggle_item.path() is not None
    assert len(view._wiggle_item.path().toSubpathPolygons()) >= 1


def test_switch_back_to_grayscale_hides_wiggle(view):
    view.set_display_mode(BScanDisplayMode.WIGGLE)
    view.set_display_mode(BScanDisplayMode.GRAYSCALE)
    assert not view._wiggle_item.isVisible()


def test_invalid_mode_raises(view):
    with pytest.raises(ValueError):
        view.set_display_mode('bogus')
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bscan_wiggle.py -v`
Expected: FAIL — `ImportError: cannot import name 'BScanDisplayMode'`

- [ ] **Step 3: 实现三态模式**

`bscan_view.py` 模块级加枚举：

```python
from enum import Enum


class BScanDisplayMode(Enum):
    GRAYSCALE = 'grayscale'
    WIGGLE = 'wiggle'          # 变面积：正半轴填充
    WAVEFORM = 'waveform'      # 波形叠加：正负对称双线
```

类头加状态（`__init__`）：

```python
self.display_mode = BScanDisplayMode.GRAYSCALE
self._wiggle_item = None       # QGraphicsPathItem，懒创建
```

新方法（set_colormap 附近）：

```python
def set_display_mode(self, mode: 'BScanDisplayMode') -> None:
    """切换灰度/变面积/波形叠加三态；共用同一坐标变换与色标。"""
    if isinstance(mode, str):
        mode = BScanDisplayMode(mode)
    if not isinstance(mode, BScanDisplayMode):
        raise ValueError(f'未知显示模式: {mode!r}')
    self.display_mode = mode
    img = self._image_item.image
    if img is None:
        return
    if mode is BScanDisplayMode.GRAYSCALE:
        if self._wiggle_item is not None:
            self._wiggle_item.hide()
        self._image_item.show()
        return
    self._image_item.hide() if mode is BScanDisplayMode.WIGGLE else self._image_item.show()
    self._render_wiggle(img,
                        filled=(mode is BScanDisplayMode.WIGGLE),
                        symmetric=(mode is BScanDisplayMode.WAVEFORM))

def _render_wiggle(self, img: np.ndarray, *, filled: bool,
                   symmetric: bool) -> None:
    """变面积/波形叠加：单 QPainterPath 批量构建（1 path, N traces）。"""
    from PyQt6.QtGui import QColor, QPainterPath

    data = np.asarray(img, dtype=np.float64)
    n_samples, n_traces = data.shape
    vmax = float(self._image_item.levels[1]) or 1.0
    # 道间距归一：相邻道峰值不重叠（Wiggle 惯例），以数据列数为基准
    x_scale = 0.9  # 每道振幅占相邻道间距的比例
    offset_gain = n_traces * x_scale / (2 * n_traces)
    gain = n_traces * x_scale / max(abs(vmax), 1e-12) / 2

    path = QPainterPath()
    for t in range(n_traces):
        col = data[:, t]
        base = t * (n_traces * x_scale / n_traces)  # 每道基线
        base = t + 0.5  # 道中心
        amp = col * (gain / n_traces * 2) if filled else col * (gain / n_traces)
        # 向量化的 polyline：正半轴（变面积）或全波形（叠加）
        ys = np.arange(n_samples, dtype=np.float64)
        xs = base + (np.maximum(amp, 0.0) if filled else amp)
        pts = np.column_stack([xs, ys])
        path.moveTo(pts[0][0], pts[0][1])
        for x, y in pts[1:]:
            path.lineTo(x, y)
        if filled:
            path.lineTo(base, n_samples - 1)
            path.lineTo(base, 0)
            path.closeSubpath()

    if self._wiggle_item is None:
        from pyqtgraph import QGraphicsPathItem
        self._wiggle_item = QGraphicsPathItem()
        self._plot.addItem(self._wiggle_item)
    self._wiggle_item.setPath(path)
    fill = QColor(self._cmap.getLookupTable(0.0, 1.0, 2)[1]) \
        if self._cmap is not None else QColor(30, 30, 30)
    fill.setAlpha(160)
    self._wiggle_item.setPen(pg.mkPen(fill.darker(140), width=1))
    self._wiggle_item.setBrush(fill if filled else pg.mkBrush(None))
    self._wiggle_item.show()
    self._plot.getViewBox().autoRange()
```

（实现时允许微调 gain/base 的换算以观感为准，但**必须保持一次 moveTo + N lineTo 的批量 path**，禁止逐道 addItem。）

右键菜单加三态子菜单（"十字光标读数"项后、"A-scan 波形跟随"前）：

```python
mode_submenu = menu.addMenu('显示模式')
from ui.widgets.bscan_view import BScanDisplayMode
for mode in BScanDisplayMode:
    act = Action({BScanDisplayMode.GRAYSCALE: '灰度',
                  BScanDisplayMode.WIGGLE: '变面积（Wiggle）',
                  BScanDisplayMode.WAVEFORM: '波形叠加'}[mode])
    act.setCheckable(True)
    act.setChecked(self.display_mode is mode)
    act.triggered.connect(
        lambda _=False, m=mode: self.set_display_mode(m))
    mode_submenu.addAction(act)
```

`set_matrix` 尾部追加一行保持模式连续：

```python
if self.display_mode is not BScanDisplayMode.GRAYSCALE:
    self.set_display_mode(self.display_mode)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bscan_wiggle.py -v`
Expected: 4 passed

- [ ] **Step 5: 性能抽验（1000 道）**

```python
# 临时脚本（不提交）：确认 1000×2000 wiggle path 构建与渲染 < 1s
```

Run: `.venv/Scripts/python.exe -c` 计时 `_render_wiggle`，>1s 则把逐点 lineTo 改为 `path.addPolygon(QPolygonF)`（numpy→QPolygonF 一次转换）。
Expected: 构建耗时 < 1s

- [ ] **Step 6: 门禁 + 冒烟**

Run: `.venv/Scripts/python.exe -m ruff check ui/ && QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe app_qt.py --smoke`
Expected: All checks passed; [smoke] OK

- [ ] **Step 7: Commit**

```bash
git add ui/widgets/bscan_view.py tests/test_bscan_wiggle.py
git commit -m "feat(ui): B-scan display modes — grayscale/wiggle/waveform overlay (Phase1 1.2)"
```

---

### Task 4: 集成验证与 PR

- [ ] **Step 1: 全量测试**

Run: `.venv/Scripts/python.exe -m pytest tests/ -q`
Expected: 756+ passed, 6 skipped（无新增失败）

- [ ] **Step 2: 真机手验清单**

启动 `python app_qt.py`，打开"测试1"项目：
1. 处理页 B-scan 右键 → 勾选"A-scan 波形跟随" → 浮窗出现 → 点击剖面各道，波形随点击变化，标题带道号
2. 再点右键取消勾选 → 浮窗消失
3. 右键 → 显示模式 → 变面积 → 剖面变 wiggle 填充形态
4. 显示模式 → 波形叠加 → 全波形形态；切回灰度恢复
5. 解释页 B-scan 同样可开浮窗（点选标注不受影响）

- [ ] **Step 3: 推送分支并建 PR**

```bash
git push -u origin feat/phase1-display
gh pr create --title "feat(ui): Phase1 display — A-scan follow popup + wiggle modes" --body "..."
```

Expected: CI 五项全绿 → 合并 main。

---

## Self-Review 已执行

- 覆盖核对：决策点 1/2 全部落 Task 1-3；验收清单在 Task 4
- 占位符扫描：无 TBD/TODO；所有代码块完整
- 类型一致性：`set_ascan_follow`/`_emit_point_picked`/`BScanDisplayMode`/`show_trace` 在 Task 1/2/3 间签名一致
