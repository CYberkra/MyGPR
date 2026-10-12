# -*- coding: utf-8 -*-
"""ViewSyncMixin — 空间页四个视图的数据灌入与去重（从 spatial_page 拆出）。

职责（2026-10-08，随「每次点三维视图整窗消失数秒」修复引入）：

- ``_apply_tracks``：唯一实际重绘入口（view.set_tracks + 单视图异常隔离）；
- ``_apply_tracks_if_changed``：数据指纹去重——数据没变就不重复灌入。
  三维视图的 set_tracks 是全量 GL 重建（删光 item 重建 + numpy 变换 +
  4326→Mercator），数据一字未变时也白付 130–470 ms 主线程阻塞，叠加
  连续切段就是「整窗消失再出现」的直接来源；
- ``_tracks_fingerprint``：把 (tracks, colors) 压成可比较指纹；
- ``_flush_if_current``：延后一拍的 flush 守卫——定时器触发前用户已
  切走时跳过，维持「只重绘可见视图」的纪律。

页面侧（SpatialPage）保留编排层：_refresh_views / _flush_dirty_view /
_switch_view / showEvent。mixin 通过鸭子类型使用页面的 ``_views()`` /
``_flush_dirty_view()`` / ``_view_stack``。

拆出独立模块同时是为了空间页模块行数回到债务棘轮 1000 行以内。
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class ViewSyncMixin:
    """空间视图灌入 + 数据指纹去重（SpatialPage 专用 mixin）。"""

    def _init_view_sync(self) -> None:
        # 分段键 -> 上次成功灌入的数据指纹；主题切换时页面侧会 clear()
        self._views_applied: dict[str, tuple] = {}

    @staticmethod
    def _tracks_fingerprint(tracks, colors) -> tuple:
        """把 (tracks, colors) 压成可比较的指纹：line_id + 点数 + 高程范围。

        粒度取舍：不算逐点坐标——勾选集合不变时坐标不可能变（轨迹来自
        不可变的 SpatialTrack）；高程范围取 3 位小数，足以识别「海拔校正/
        地形偏移后数据确实变了」。NaN 高程跳过（NaN 任何比较恒 False，
        会让指纹永不命中、退化为每次都重建——保守但不出错）。
        """
        parts = []
        for track in tracks or ():
            pts = getattr(track, 'points', None) or ()
            elev_min = elev_max = None
            for pt in pts:
                e = getattr(pt, 'elevation_m', None)
                if e is None or e != e:  # None / NaN
                    continue
                if elev_min is None or e < elev_min:
                    elev_min = e
                if elev_max is None or e > elev_max:
                    elev_max = e
            parts.append((
                str(getattr(track, 'line_id', '') or ''),
                len(pts),
                round(elev_min, 3) if elev_min is not None else None,
                round(elev_max, 3) if elev_max is not None else None,
            ))
        return (tuple(parts), tuple(sorted((colors or {}).items())))

    def _apply_tracks_if_changed(self, key: str, view, tracks, colors,
                                 fingerprint: tuple | None = None) -> bool:
        """数据指纹没变就不灌（三维 GL 全量重建是整窗卡死的元凶之一）。

        仅在 set_tracks 成功后记录指纹：失败时下次重试（失败可能只是
        单个视图暂时性异常，不能让去重把它永久挡在门外）。
        重绘 ≥100 ms 打 info 日志，便于真机核对「切段还卡不卡」。
        """
        if fingerprint is None:
            fingerprint = self._tracks_fingerprint(tracks, colors)
        if self._views_applied.get(key) == fingerprint:
            return False
        started = time.perf_counter()
        applied = self._apply_tracks(view, tracks, colors)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if applied:
            self._views_applied[key] = fingerprint
            if elapsed_ms >= 100.0:
                logger.info('空间视图 %s 重绘 %.0f ms（%d 条测线）',
                            key, elapsed_ms, len(tracks or ()))
        return applied

    def _flush_if_current(self, key: str) -> None:
        """延后一拍的 flush：仅当该分段仍是当前视图时才补齐重绘。

        用户在定时器触发前又切走（快速连点分段）时跳过，维持「只重绘
        可见视图」的纪律；错过的重绘由下次 _refresh_views 重新标脏兜底。
        """
        view = self._views().get(key)
        if view is not None and self._view_stack.currentWidget() is view:
            self._flush_dirty_view(key)

    def _apply_tracks(self, view, tracks, colors) -> bool:
        """把轨迹数据灌给单个视图（唯一实际重绘入口）。成功返回 True。"""
        if view is None:
            return False
        try:
            view.set_tracks(tracks, colors)
        except Exception as exc:  # noqa: BLE001 — 单个视图失败不拖垮整页
            logger.debug('set_tracks 失败（%s）: %s', type(view).__name__, exc)
            return False
        return True
