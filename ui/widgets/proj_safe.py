# -*- coding: utf-8 -*-
"""pyproj 线程安全封装。

背景：PROJ C 库在 GUI 线程（地图坐标换算）与 QThreadPool 工作线程
（地形构建）并行创建 Transformer / 执行 transform 时，会在 proj.dll
内部段错误（Windows 事件查看器：python.exe 崩溃，faulting module
proj_9-*.dll，异常码 0xc0000005）并伴随堆损坏（0xc0000374）。

措施：进程级互斥锁串行化所有 Transformer 创建与坐标转换；入口
app_qt.py 同时设置 ``PYPROJ_GLOBAL_CONTEXT=ON``（须先于 pyproj 导入）
双保险。transform 是毫秒级纯计算，串行化对 UI 无感知影响。
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
