"""BScanView 数据 mixin——bundle 装载 / 海拔 warp / 色阶 / 色标 / 显示增益。

2026-09-24 组件化第二步（自 bscan_view.py 机械搬迁，方法体零改动）：
数据装载与显示域变换是最重的一节（~400 行）。状态仍在 ``self``，
对外 API 与信号零变化；模块级色标助手（_lookup_colormap 等）随迁，
bscan_view re-import 保持既有导入路径（tests 经 bscan_view 取
DEFAULT_P_LOW/HIGH 依然成立）。
"""

import math

import numpy as np
import pyqtgraph as pg

from ui.desktop_backend_facade import compute_display_levels
from ui.widgets._colormap_data import COLORMAP_DATA
from ui.widgets.bscan_axes import build_elevation_view
from ui.widgets.bscan_levels_dialog import LevelsDialog
from ui.widgets.bscan_view_interaction import BScanDisplayMode

# 色标缓存：名 → ColorMap（数据查表重建，进程内只构建一次）
_COLORMAP_CACHE = {}


# 显示动态范围默认值：与处理页 p_low/p_high 一致（2% / 98% 百分位）
DEFAULT_P_LOW, DEFAULT_P_HIGH = 2.0, 98.0


def _normalise_levels(levels) -> tuple[float, float] | None:
    """把外部传入的动态范围规整为合法 (low, high)；非法返回 None。

    「非法」= 不是两个数 / 非有限 / 不在 [0,100] / low >= high。跨会话恢复时
    设置文件可能被手改成任意内容，必须在入口收口，否则 0/0 会被
    compute_display_levels 带进退化分支（静默返回 ±1，图看起来"没坏"但色阶
    已经被改错）。返回 None 让调用方自己决定是「回落默认」还是「拒绝本次」。
    """
    try:
        low, high = float(levels[0]), float(levels[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    if not (0.0 <= low < high <= 100.0):
        return None
    return low, high


def _restored_levels(levels) -> tuple[float, float]:
    """跨会话恢复用：非法值静默回落默认百分位（不能让坏设置挡住出图）。"""
    return _normalise_levels(levels) or (DEFAULT_P_LOW, DEFAULT_P_HIGH)


def _lookup_colormap(name: str):
    """按名取 ColorMap：优先查预采样数据（零 matplotlib import）。

    getFromMatplotlib 首次调用会触发 matplotlib 全量 import（实测 ~0.9s），
    主页预览卡构造期即命中，是冷启动大头。九项 SPEC 色标由
    scripts/gen_colormap_data.py 离线采样为 _colormap_data.py，重建结果
    与 getFromMatplotlib 逐点一致（1024 点采样最大通道差 0，脚本自验）。
    数据缺失的色标（新增未重跑生成脚本）回落原路径，正确性不受影响。
    """
    cmap = _COLORMAP_CACHE.get(name)
    if cmap is not None:
        return cmap
    entry = COLORMAP_DATA.get(name)
    if entry is not None:
        cmap = pg.colormap.ColorMap(pos=entry[0], color=entry[1], name=name)
    else:
        cmap = pg.colormap.getFromMatplotlib(name)
    if cmap is not None:
        cmap.name = name
        _COLORMAP_CACHE[name] = cmap
    return cmap


class BScanViewDataMixin:
    # ------------------------------------------------------------------ 数据
    def set_bundle(self, bundle) -> None:
        """接收 PreviewBundle（鸭子类型，不 import core.gui_rendering）。"""
        self.set_matrix(
            bundle.matrix, bundle.vmin, bundle.vmax,
            title=str(getattr(bundle, 'title', '') or ''),
            x_label=getattr(bundle, 'x_label', '道数'),
            y_label=getattr(bundle, 'y_label', '采样点'),
        )
        # 十字光标读数用物理轴元数据（可选；与显示矩阵同降采样）
        self._trace_axis_m = getattr(bundle, 'trace_axis_m', None)
        self._sample_axis = getattr(bundle, 'sample_axis', None)
        self._sample_axis_label = str(
            getattr(bundle, 'sample_axis_label', '') or '')
        self._trace_count = int(getattr(bundle, 'trace_count', 0) or 0)
        self._sample_count = int(getattr(bundle, 'sample_count', 0) or 0)
        # 海拔纵轴的两份原料（均已降到显示网格）：逐道地面高程 + 米制深度
        self._ground_elevation_m = getattr(bundle, 'trace_elevation_m', None)
        self._depth_axis_m = getattr(bundle, 'depth_axis_m', None)
        # 数据换了 → 轴可用性可能变（偏好若是海拔，新的数据支持则自动恢复）
        self._apply_axis_modes()
        # 增益开启时 set_matrix 阶段只能用行号近似曲线（物理轴此刻才就位），
        # 按物理轴补一次整体重渲染；海拔路径已在 _refresh_elevation_image
        # 内通过 _resolve_display_target 拿到增益后目标，无需重复。
        if self._gain_mode != 'off' and not self._showing_elevation:
            self._refresh_gain_render()

    def set_matrix(self, matrix, vmin, vmax, *,
                   title="", x_label="道数", y_label="采样点") -> None:
        """matrix 为 (samples, traces)，零拷贝视图直接显示。

        row-major 语义下 array[r, c] → (x=c, y=r)，故 (samples, traces)
        矩阵本身就是显示布局：x=道数（列）、y=采样点（行），无需转置拷贝。
        （SPEC §5.1 字面要求"显示 .T 转置视图"系沿袭师兄缓冲 (traces,
        samples) 约定；本控件输入约定为 (samples, traces)，若再 .T 会
        使 x/y 轴互换，与 x_label=道数 / y_label=采样点 及
        sig_point_picked(trace, sample) 契约矛盾，故按契约语义实现。）

        本函数只送**原始（未变形）矩阵**的增益视图上屏（_matrix 本体保留
        raw，供十字读数/波形/海拔 warp 使用）：海拔 warp 的原料（逐道高程/
        深度轴）由 set_bundle 在本函数之后才赋值，故海拔模式的换图统一由
        _apply_axis_modes → _refresh_elevation_image 兜底（set_bundle 末尾
        必然调用），这里不重复做。
        """

        mat = np.asarray(matrix)
        if mat.ndim != 2:
            raise ValueError('B-Scan 矩阵必须是二维 (samples, traces)')
        view = mat  # 零拷贝；row-major 下 y=采样(行)、x=道(列)
        self._matrix = view
        # 直接 set_matrix 的调用方没有物理轴元数据，读数退回索引显示
        self._trace_axis_m = None
        self._sample_axis = None
        self._sample_axis_label = ''
        self._trace_count = 0
        self._sample_count = 0
        # 无 bundle 元数据 → 无物理轴：海拔/距离不可用（回落采样轴显示）
        self._ground_elevation_m = None
        self._depth_axis_m = None
        self._showing_elevation = False
        self._elev_axis = None
        new_shape = (view.shape[1], view.shape[0])   # (traces, samples)
        shape_changed = new_shape != self._image_shape
        self._image_shape = new_shape
        # 上屏的是增益后矩阵（_matrix 保留 raw 供读数/波形/海拔 warp）；
        # 此刻物理轴元数据尚未由 set_bundle 赋值，增益曲线先行退行号——
        # set_bundle 末尾在增益开启时会按物理轴整体重渲染一次。
        self._image_item.setImage(self._gain_applied(view), autoLevels=False,
                                  levels=(float(vmin), float(vmax)))
        if shape_changed:
            # 新数据尺寸变化时按当前比例策略铺满视野，避免换测线后图像跑出可视区
            self._fit_current_mode()
        # 色阶偏好（可能是跨会话恢复或用户在无数据时设的）应用到新数据上：
        # 优先于调用方给的 vmin/vmax，因为那只是数据的默认裁切。
        # 注意下面还会统一重渲染一次波形，故这里让它跳过重绘（见参数注释）。
        self._apply_levels_to_render(redraw_waveform=False)
        if self._colorbar is not None:
            self._colorbar.setLevels(self._image_item.levels)
        # 非灰度模式下按当前显示模式重渲染波形；新数据到达保持用户视野，
        # 不重置缩放（reset_view=False，模式切换才重置）
        if self.display_mode is not BScanDisplayMode.GRAYSCALE:
            self._apply_display_mode(self.display_mode, reset_view=False)
        # 图内标题只留数据身份（如"测线 L3"）；空标题时 pyqtgraph 自动隐藏。
        # 8pt 小字（空间瘦身）：身份保留、标题行高约减半，默认字号太占纵向。
        self._export_title = str(title or '')
        self._plot.setTitle(self._compact_title_html(self._export_title))
        self._plot.setLabel('bottom', x_label)
        self._plot.setLabel('left', y_label)
        # 数据换了 → 轴可用性与刻度都要重算（单位切换不动，标签由调用方给定）
        self._refresh_axis_state()
        self._empty_overlay.setVisible(False)

    def _resolve_display_target(self) -> tuple:
        """按纵轴偏好解析应显示的矩阵：海拔偏好且可 warp → (warped, axis)。

        增益是显示链第一级：先对 raw 矩阵逐行缩放（每个采样点拿到自己
        深度的精确增益），海拔 warp 的原料即增益后矩阵——warp 的逐列
        线性重采样不破坏行增益语义。增益关闭时 src 即 raw（零拷贝）。
        """
        src = self._gain_applied(self._matrix)
        if self._y_axis == 'elevation':
            warped, axis = self._warped_elevation(src)
            if warped is not None:
                return warped, axis
        return src, None

    def _warped_elevation(self, src) -> tuple:
        """海拔 warp（缓存：同一份数据/高程/深度只算一次，按对象身份判重）。

        缓存键用对象身份而不是 id()：id 在对象被回收后可能被新对象复用，
        身份比较（is）则绝对可靠。增益开关/参数变化会生成新的 src 对象
        （身份必变），warp 缓存随之自动失效重算。

        :param src: 增益后矩阵（见 _resolve_display_target）。
        """
        if src is None:
            return None, None
        cache = self._elev_cache
        if (cache is not None
                and cache[0] is src
                and cache[1] is self._ground_elevation_m
                and cache[2] is self._depth_axis_m):
            return cache[3], cache[4]
        warped, axis = build_elevation_view(
            src, self._ground_elevation_m, self._depth_axis_m)
        self._elev_cache = (src, self._ground_elevation_m,
                            self._depth_axis_m, warped, axis)
        return warped, axis

    def _refresh_elevation_image(self) -> bool:
        """按纵轴偏好换显示矩阵（海拔↔采样轴切换共用）。返回是否换图。"""
        if self._matrix is None:
            return False
        want = self._y_axis == 'elevation'
        target, elev_axis = None, None
        if want:
            target, elev_axis = self._resolve_display_target()
            if elev_axis is None:
                want = False          # 数据不支持，回落采样轴显示
        if want == self._showing_elevation:
            return False
        if not want:
            target, elev_axis = self._matrix, None
        levels = self._image_item.levels
        self._image_item.setImage(target, autoLevels=False, levels=levels)
        self._showing_elevation = want
        self._elev_axis = elev_axis
        self._image_shape = (target.shape[1], target.shape[0])
        return True

    def set_colormap(self, name: str) -> None:
        """按 matplotlib 名取 LUT（九项见 SPEC §1，默认 seismic）。"""
        self._cmap_name = str(name)
        self._cmap = _lookup_colormap(str(name))
        self._image_item.setColorMap(self._cmap)
        if self._colorbar is not None:
            self._colorbar.setColorMap(self._cmap)

    def display_levels(self) -> tuple[float, float]:
        """当前显示动态范围的百分位 ``(low, high)``。"""
        return self._p_low, self._p_high

    def set_display_levels(self, p_low, p_high, *, notify: bool = False) -> bool:
        """按百分位重算并应用显示色阶（只改显示，不动数据）。

        **偏好与渲染分离**：百分位本身是纯状态，永远记下来；只有「算 vmin/vmax
        并推给 ImageItem」这一步需要数据。跨会话恢复发生在**还没有数据**的
        时刻（主窗构造期），若此时整体放弃，用户存的 5/95 会被静默丢掉、
        直到他手动再设一次——`set_matrix` 会用 `_p_low/_p_high` 重算，
        所以这里记下来就够。

        :param notify: True 时发 sig_levels_changed（供宿主写回设置）；
            恢复持久化阶段传 False。
        :return: 是否真的重算了渲染（无数据时返回 False，但偏好已记下）
        """
        normalised = _normalise_levels((p_low, p_high))
        if normalised is None:
            return False
        low, high = normalised
        self._p_low, self._p_high = low, high
        self._apply_levels_to_render()
        if notify:
            self.sig_levels_changed.emit(low, high)
        return self._matrix is not None

    def _apply_levels_to_render(self, *, redraw_waveform: bool = True) -> None:
        """把当前 ``_p_low/_p_high`` 换算成 vmin/vmax 推到图像与色标（无数据则跳过）。

        :param redraw_waveform: 波形模式的振幅归一取自 levels，故色阶变了一般要
            重绘；但 ``set_matrix`` 会在随后统一重绘一次，传 False 避免同一帧
            画两遍（波形重绘是整幅遍历，不便宜）。
        """
        if self._matrix is None:
            return
        # 色阶在**增益后**矩阵上取百分位：增益与百分位正交（增益纵向拉平、
        # 百分位横向裁剪），顺序为先增益、再取百分位
        vmin, vmax = compute_display_levels(
            self._gain_applied(self._matrix),
            p_low=self._p_low, p_high=self._p_high)
        self._image_item.setLevels((float(vmin), float(vmax)))
        if self._colorbar is not None:
            self._colorbar.setLevels((float(vmin), float(vmax)))
        if (redraw_waveform
                and self.display_mode is not BScanDisplayMode.GRAYSCALE):
            # 波形模式的振幅归一取自 levels，色阶变了要按新 levels 重画
            self._apply_display_mode(self.display_mode, reset_view=False)

    def colorbar_visible(self) -> bool:
        """色标显隐偏好（与色标对象是否存在解耦）。"""
        return self._colorbar_visible

    # ------------------------------------------------------------------ 显示增益
    def gain_mode(self) -> str:
        """当前显示增益模式：'off' / 'sec' / 'tvg'。"""
        return self._gain_mode

    def set_gain(self, mode: str, *, alpha: float | None = None,
                 db: float | None = None, power: float | None = None,
                 notify: bool = False) -> None:
        """切换显示增益；显示域逐行缩放，raw 一个字节不动。

        :param mode: 'off' 关闭 / 'sec' SEC 补偿（扩散∝深度 + 指数吸收）/
            'tvg' TVG 补偿（用户滑条调参的幂次曲线）。
        :param alpha: SEC 衰减补偿系数（dB/采样轴单位：深度轴为 dB/m，
            时间轴为 dB/ns）；None 保持现值。
        :param db: TVG 深端总增益（dB，浅端恒为 0 增益）；None 保持现值。
        :param power: TVG 曲线弯度幂次（1 线性，>1 增益集中深部）；None 保持。
        :param notify: True 时发 sig_gain_changed（TVG 另发
            sig_gain_params_changed 供宿主持久化参数）；恢复持久化阶段
            传 False。
        """
        mode = str(mode)
        if mode not in ('off', 'sec', 'tvg'):
            raise ValueError(f'未知增益模式: {mode!r}')
        changed = (mode != self._gain_mode
                   or (alpha is not None and float(alpha) != self._gain_alpha)
                   or (db is not None and float(db) != self._gain_db)
                   or (power is not None
                       and float(power) != self._gain_power))
        self._gain_mode = mode
        if alpha is not None:
            self._gain_alpha = float(alpha)
        if db is not None:
            self._gain_db = float(db)
        if power is not None:
            self._gain_power = float(power)
        if changed:
            self._gain_cache = None
            self._refresh_gain_render()
        if notify:
            self.sig_gain_changed.emit(self._gain_mode)
            if mode == 'tvg':
                self.sig_gain_params_changed.emit(self._gain_db,
                                                  self._gain_power)

    def _refresh_gain_render(self) -> None:
        """增益变了：按当前解析目标整体换图 + 重算色阶/波形。

        与 _refresh_elevation_image 的差异：无论海拔状态是否变化都要换图
        （增益改的是像素值本身），色阶随增益后矩阵重算（见
        _apply_levels_to_render）。
        """
        if self._matrix is None:
            return
        target, elev_axis = self._resolve_display_target()
        levels = self._image_item.levels
        self._image_item.setImage(target, autoLevels=False, levels=levels)
        self._showing_elevation = elev_axis is not None
        self._elev_axis = elev_axis
        self._image_shape = (target.shape[1], target.shape[0])
        self._apply_levels_to_render()

    def _gain_applied(self, mat):
        """显示域增益变换：off 原样返回（零拷贝）；sec/tvg 返回逐行缩放结果。

        缓存按 (矩阵身份, 模式+参数) 判重——与海拔 warp 缓存同一纪律
        （身份比较，不用 id()）。增益关闭时必须原样返回以保持「零拷贝 +
        身份不变」语义（海拔 warp 缓存、回落判定都依赖它）。
        """
        if mat is None or self._gain_mode == 'off':
            return mat
        cache = self._gain_cache
        if (cache is not None and cache[0] is mat
                and cache[1] == self._gain_mode
                and cache[2] == self._gain_alpha
                and cache[3] == self._gain_db
                and cache[4] == self._gain_power):
            return cache[5]
        if self._gain_mode == 'sec':
            gain = self._sec_row_gain(mat.shape[0])
        else:
            gain = self._tvg_row_gain(mat.shape[0])
        out = np.asarray(mat) * gain[:, None]
        self._gain_cache = (mat, self._gain_mode, self._gain_alpha,
                            self._gain_db, self._gain_power, out)
        return out

    def _phys_depth_axis(self, n_rows: int):
        """物理采样轴（米/纳秒）；无轴/长度不符/含 NaN 时返回 None。

        返回原数组（不拷贝），浅端归零与归一化由各增益曲线自行处理。
        """
        axis = self._sample_axis
        if axis is None:
            return None
        arr = np.asarray(axis, dtype=np.float64)
        if (arr.ndim == 1 and arr.size == n_rows
                and np.isfinite(arr).all() and n_rows >= 2):
            return arr
        return None

    def _sec_row_gain(self, n_rows: int):
        """SEC 行增益曲线（长度 n_rows，浅→深单调放大）。

        g(d) = (d + d0)/d0 × 10^(α·d/20)：扩散项 ∝ 深度（1/r 幅值补偿）
        + 指数吸收补偿。d 取物理采样轴（米/纳秒，α 单位跟随轴）；无轴或
        轴长不符时退行号（set_matrix 阶段物理轴尚未就位的近似，set_bundle
        末尾会按物理轴重渲染）。浅端参考 d0 取轴量程 2% 防 1/r 发散；
        总增益限幅 1000×（60 dB）防深部噪声底被抬满。
        """
        arr = self._phys_depth_axis(n_rows)
        if arr is not None:
            d = np.abs(arr - arr[0])     # 浅端归零（兼容升/降序轴）
        else:
            d = np.arange(n_rows, dtype=np.float64)
        d0 = max(float(np.ptp(d)) * 0.02, 1e-6)
        gain = ((d + d0) / d0) * np.power(10.0, self._gain_alpha * d / 20.0)
        return np.clip(gain, 1.0, 1000.0)

    def _tvg_row_gain(self, n_rows: int):
        """TVG 行增益曲线：g(u) = 10^(db·u^power / 20)，u 为归一化深度。

        u∈[0,1]（浅端 0、深端 1，取物理轴，无轴退行号）；浅端恒 1×，
        深端恰为 10^(db/20)。power=1 线性递增，>1 增益向深部集中，
        <1 向浅部集中（滑条范围限 0.3~3.0）。限幅同 SEC（1000×）。
        """
        arr = self._phys_depth_axis(n_rows)
        if arr is not None:
            span = float(np.ptp(arr))
            u = (np.abs(arr - arr[0]) / span) if span > 0 else \
                np.linspace(0.0, 1.0, n_rows)
        else:
            u = (np.arange(n_rows, dtype=np.float64) / max(n_rows - 1, 1))
        power = min(max(self._gain_power, 0.1), 10.0)
        gain = np.power(10.0, self._gain_db * np.power(u, power) / 20.0)
        return np.clip(gain, 1.0, 1000.0)

    def _on_gain_menu(self, mode: str) -> None:
        """右键「显示增益」子菜单动作：切模式；选 TVG 时弹滑条面板。"""
        self.set_gain(mode, notify=True)
        if mode == 'tvg':
            self._open_tvg_dialog()

    def _open_tvg_dialog(self) -> None:
        """打开 TVG 滑条面板（非模态，随视图销毁；重复调用只唤起）。"""
        if self._tvg_dialog is not None:
            self._tvg_dialog.show()
            self._tvg_dialog.raise_()
            self._tvg_dialog.activateWindow()
            return
        from ui.widgets.bscan_tvg_dialog import TvgGainDialog
        self._tvg_dialog = TvgGainDialog(self)
        self._tvg_dialog.show()

    def set_colorbar_visible(self, visible: bool, *, notify: bool = False) -> None:
        """切换右侧色标条的显示；隐藏后其宽度完整归还画布。

        探针实证（``_probe_colorbar_toggle.py``）：ColorBarItem 插在 plot
        右列，纯 ``setVisible`` 即可——隐藏后 vb.width 644→716（72px 完整
        归还），再显示恢复 644，免动 GraphicsLayout。with_colorbar=False
        构造的视图没有色标对象，此调用只记偏好不动渲染。

        :param notify: True 时发 sig_colorbar_visible_changed（供宿主写回
            设置）；恢复持久化阶段传 False。
        """
        visible = bool(visible)
        self._colorbar_visible = visible
        if self._colorbar is not None:
            self._colorbar.setVisible(visible)
        if notify:
            self.sig_colorbar_visible_changed.emit(visible)

    def _edit_levels(self) -> None:
        """右键「色阶设置…」：弹窗取低/高百分位后应用。"""
        dialog = LevelsDialog(self.window(), self._p_low, self._p_high)
        if dialog.exec() != LevelsDialog.DialogCode.Accepted:
            return
        chosen = dialog.percentiles()
        if chosen is None:
            return
        self.set_display_levels(*chosen, notify=True)

    def _choose_colormap(self, name: str) -> None:
        """右键菜单选色标：应用到本视图并发信号让页面同步控件。"""
        self.set_colormap(name)
        self.sig_colormap_changed.emit(name)

