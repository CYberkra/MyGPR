# -*- coding: utf-8 -*-
"""pyproj 线程安全封装。

背景：PROJ C 库在 QThreadPool 工作线程内创建 Transformer 会在 proj.dll
段错误（native-crash.log 实证：Windows 事件查看器 0xc0000005，
faulting module proj_9-*.dll），Python 层加锁也无法避免（锁内照样崩）。
因此 UI 侧所有 pyproj 调用限定在 GUI 线程（地图换算、地形预计算），
工作线程完全不碰 pyproj；本模块的进程级锁作为额外保险，防止未来
新增调用点时无意引入跨线程并发。
"""
from __future__ import annotations

import threading

_LOCK = threading.Lock()


class LockedTransformer:
    """带锁的 Transformer 封装：创建与 transform 均串行化。"""

    def __init__(self, src_epsg: int, dst_epsg: int) -> None:
        from pyproj import Transformer
        with _LOCK:
            self._transformer = Transformer.from_crs(
                f'EPSG:{int(src_epsg)}', f'EPSG:{int(dst_epsg)}',
                always_xy=True)

    def transform(self, xs, ys):
        """批量坐标转换（线程安全），返回 (xs', ys') 与原类型一致。"""
        with _LOCK:
            return self._transformer.transform(xs, ys)


def transform_coordinates(src_epsg: int, dst_epsg: int, xs, ys):
    """一次性 EPSG→EPSG 坐标转换（线程安全），返回 (xs', ys')。"""
    return LockedTransformer(src_epsg, dst_epsg).transform(xs, ys)
