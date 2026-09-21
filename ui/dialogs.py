# -*- coding: utf-8 -*-
"""统一的对话框创建模块（架构纪律唯一许可点之一）。

qfluentwidgets 的 MessageBox / Dialog 只能在页面层或本模块创建；
ui/coordinator_* 接线器与 ui/controllers 均不得创建 QWidget，
需要阻塞式用户确认时调用本模块的函数式 API。
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget
from qfluentwidgets import MessageBox

__all__ = ['ask_artifact_delete', 'ask_cancel_job']


def ask_artifact_delete(parent: QWidget | None, names: list[str]) -> bool:
    """成果删除级联确认框：列出将删除的成果名（含级联派生成果）。

    数据会移入项目 .trash 回收站（可恢复）。返回 True = 用户确认删除；
    名单为空时直接返回 False（不弹框）。
    """
    names = [str(name) for name in (names or []) if str(name)]
    if not names:
        return False
    shown = '\n'.join(f'  • {name}' for name in names)
    box = MessageBox(
        '确认删除成果？',
        f'将删除 {len(names)} 个成果（含级联派生成果，数据会移入项目 '
        f'.trash 回收站，可恢复）：\n{shown}\n\n确认继续？',
        parent,
    )
    box.yesButton.setText('删除')
    box.cancelButton.setText('取消')
    return box.exec() == 1  # QDialog.DialogCode.Accepted


def _format_elapsed(seconds: float) -> str:
    """运行时长人性化：>1h 显示 x.y 小时，>60s 显示分钟，否则秒。"""
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f'{seconds / 3600:.1f} 小时'
    if seconds >= 60:
        return f'{seconds // 60} 分 {seconds % 60} 秒'
    return f'{seconds} 秒'


def ask_cancel_job(parent: QWidget | None, title: str, elapsed_s: float) -> bool:
    """取消长任务的确认框：已运行越久的任务误触代价越大。

    返回 True = 用户确认取消。title 为任务标题（无法取得时调用方传 job_id）。
    """
    box = MessageBox(
        '确认取消任务？',
        f'任务「{title}」已运行 {_format_elapsed(elapsed_s)}，\n'
        '取消后本次计算的中间结果会丢失。确认取消？',
        parent,
    )
    box.yesButton.setText('取消任务')
    box.cancelButton.setText('继续运行')
    return box.exec() == 1  # QDialog.DialogCode.Accepted
