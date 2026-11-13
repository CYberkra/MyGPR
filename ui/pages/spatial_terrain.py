"""Terrain selection and background loading of local elevation grids."""
import os

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from ui import file_dialogs
from ui.widgets.local_dem import load_xyz_grid


# ------------------------------------------------------------ 本地 DEM 异步加载
# XYZ 格网解析（大文件整读 + numpy 格网还原）离开 GUI 线程，结果信号回主线程
# 再喂三维视图（参照 ui/widgets/trajectory_3d_view.py 的 QThreadPool worker 模式）。
class DemLoadSignals(QObject):
    """QRunnable 无法自带信号，用独立 QObject 回主线程。"""

    # generation, dem(dict|None), path, error（''=成功）
    finished = pyqtSignal(int, object, str, str)


class _DemLoadWorker(QRunnable):
    """后台线程解析 XYZ 格网；解析失败带回错误文本由主线程决定提示。"""

    def __init__(self, generation: int, path: str, signals: DemLoadSignals) -> None:
        super().__init__()
        self._generation = int(generation)
        self._path = path
        self._signals = signals

    def run(self) -> None:
        try:
            dem = load_xyz_grid(self._path)
        except (OSError, ValueError) as exc:
            self._signals.finished.emit(self._generation, None, self._path, str(exc))
        else:
            self._signals.finished.emit(self._generation, dem, self._path, '')


class SpatialTerrain:
    def _set_terrain_mode(self, mode: str, *, persist: bool = True) -> None:
        """设置三维地形来源（更新下拉 + 三维视图 + 可选持久化）。"""
        self._terrain_mode = mode
        self._restoring_terrain = True
        try:
            index = self._3d_terrain_combo.findData(mode)
            if index >= 0:
                self._3d_terrain_combo.setCurrentIndex(index)
        finally:
            self._restoring_terrain = False
        self._3d_view.set_terrain_source(mode)
        if persist:
            self._persist_setting('spatial_terrain_source', mode)

    def _auto_load_dem(self) -> None:
        """启动自动加载上次导入的本地 DEM（默认在线下载，无需任何操作）。

        文件已被移动/删除或解析失败时清除记录，静默回退在线下载。
        解析在工作线程执行（大 DEM 不冻结启动），结果回主线程应用。
        """
        sm = self._sm
        if sm is None:
            return
        path = str(sm.get('spatial_local_dem', '') or '')
        if not path:
            return
        if not os.path.isfile(path):
            self._persist_setting('spatial_local_dem', '')
            return
        self._start_dem_load(path, auto=True)

    def _start_dem_load(self, path: str, *, auto: bool) -> None:
        """把 XYZ 格网解析交给工作线程；auto=启动自动加载（失败静默清记录）。

        代次守卫：快速连续导入/自动加载时，旧任务的回包直接丢弃。
        """
        if self._closed:
            return
        self._dem_load_generation += 1
        self._pending_dem_auto = auto
        self._dem_pool.start(_DemLoadWorker(
            self._dem_load_generation, path, self._dem_load_signals))

    def _on_dem_loaded(self, generation: int, dem, path: str, error: str) -> None:
        """DEM 解析回包（GUI 线程）：应用格网或按来源处理失败。"""
        if self._closed or generation != self._dem_load_generation:
            return
        if dem is None:
            if self._pending_dem_auto:
                # 自动加载失败：清除记录，静默回退在线下载（原同步语义）
                self._persist_setting('spatial_local_dem', '')
            else:
                self._3d_dem_label.setText(f'导入失败：{error}')
            return
        self._apply_local_dem(dem, path)
        if not self._pending_dem_auto:
            self._set_terrain_mode('local_dem')
            self._persist_setting('spatial_local_dem', path)

    def _apply_local_dem(self, dem: dict, path: str) -> None:
        """应用本地 DEM 到三维视图并更新卡片显示。"""
        self._3d_view.set_local_dem(dem)
        rows, cols = dem['elev'].shape
        self._dem_base_text = f'{os.path.basename(path)}（{cols}×{rows}）'
        self._3d_dem_label.setText(self._dem_base_text)
        self._3d_dem_label.setToolTip(path)
        self._3d_dem_clear_btn.setEnabled(True)

    def _on_dem_notice(self, text: str) -> None:
        """三维视图的地形提示（DEM 覆盖 / 估算回退等）：拼到标签基础文本后。"""
        base = self._dem_base_text or '未导入本地 DEM'
        self._3d_dem_label.setText(f'{base}；{text}' if text else base)

    def _on_terrain_source_changed(self, index: int) -> None:
        """地形来源下拉：本地 DEM 尚未导入时先引导导入，取消则回退原选择。"""
        if self._restoring_terrain:
            return
        mode = str(self._3d_terrain_combo.itemData(index) or 'online')
        if mode == 'local_dem' and not self._3d_dem_clear_btn.isEnabled():
            self._on_import_dem_clicked()   # 成功路径内部会切到 local_dem
            if self._terrain_mode != 'local_dem':
                # 用户取消导入 → 下拉回退到原来源
                self._set_terrain_mode(self._terrain_mode, persist=False)
            return
        self._set_terrain_mode(mode)

    def _on_import_dem_clicked(self) -> None:
        """导入本地 DEM（XYZ 格网）：三维地形改用本地高程，免在线下载。

        路径记入设置，之后启动自动加载，无需重复导入。
        解析在工作线程执行，回包后切到 local_dem 并持久化路径。
        """
        path, _selected = file_dialogs.getOpenFileName(
            self, '选择本地 DEM 格网文件', '',
            'DEM 格网 (*.xyz *.csv *.txt);;所有文件 (*)')
        if not path:
            return
        self._start_dem_load(path, auto=False)

    def _on_clear_dem_clicked(self) -> None:
        """清除本地 DEM：三维地形回退在线高程瓦片，并删除设置记录。"""
        self._3d_view.set_local_dem(None)
        self._dem_base_text = ''
        self._3d_dem_label.setText('')
        self._3d_dem_label.setToolTip('')
        self._3d_dem_clear_btn.setEnabled(False)
        if self._terrain_mode == 'local_dem':
            self._set_terrain_mode('online')
        self._persist_setting('spatial_local_dem', '')
