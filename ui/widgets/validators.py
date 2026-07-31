"""validators — 输入校验函数与辅助控件（SPEC §5.8，style_spec §5.5）。

函数式接口统一返回 (ok, error_msg)，错误文案全中文：
- validate_non_empty(text, field_name='内容') → f'{field_name}不能为空'
- validate_host(text)：IPv4 或 RFC1123 主机名
- validate_port(value)：1–65535
- validate_directory(path)：存在且可写
- mark_invalid / clear_invalid：红框 + tooltip / 恢复空样式
- StrictComboBox(FluentComboBox)：仅枚举值，setCurrentText 遇表外值自动 addItem
"""

import os
import re

from PyQt6.QtGui import QValidator
from qfluentwidgets import ComboBox as FluentComboBox

_INVALID_STYLE = 'border: 1px solid #ef4444; border-radius: 5px;'

_HOSTNAME_RE = re.compile(
    r'^(?=.{1,253}$)([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*'
    r'[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$')


# ------------------------------------------------------------- 校验函数
def validate_non_empty(text, field_name='内容'):
    """非空校验。"""
    if not (text or '').strip():
        return False, '%s不能为空' % field_name
    return True, ''


def validate_host(text):
    """IPv4 或 RFC1123 主机名。"""
    text = (text or '').strip()
    if not text:
        return False, '服务器地址不能为空'
    parts = text.split('.')
    if all(p.isdigit() for p in parts):
        if len(parts) != 4:
            return False, '不是合法的 IP 地址或主机名'
        for p in parts:
            if not (0 <= int(p) <= 255):
                return False, 'IP 地址段超出 0-255'
        return True, ''
    if _HOSTNAME_RE.match(text):
        return True, ''
    return False, '不是合法的 IP 地址或主机名'


def validate_port(value):
    """端口 1–65535。"""
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return False, '端口必须是数字'
    if not (1 <= port <= 65535):
        return False, '端口范围必须在 1-65535 之间'
    return True, ''


def validate_directory(path):
    """目录非空 / 存在 / 可写。"""
    path = (path or '').strip()
    if not path:
        return False, '目录不能为空'
    if not os.path.isdir(path):
        return False, '目录不存在'
    if not os.access(path, os.W_OK):
        return False, '目录不可写'
    return True, ''


# ------------------------------------------------------------- 标红辅助
def mark_invalid(widget, msg):
    """设校验红框 + tooltip。"""
    widget.setStyleSheet(_INVALID_STYLE)
    widget.setToolTip(msg)


def clear_invalid(widget):
    """恢复空样式 + 清空 tooltip。"""
    widget.setStyleSheet('')
    widget.setToolTip('')


# ------------------------------------------------------------- 辅助类
class FunctionValidator(QValidator):
    """函数式校验器：中间态一律放行，终态非法返回 Intermediate
    （不打断输入，配合即时标红）。"""

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self._func = func

    def validate(self, text, pos):
        ok, _msg = self._func(text)
        if ok:
            return QValidator.State.Acceptable, text, pos
        return QValidator.State.Intermediate, text, pos


class StrictComboBox(FluentComboBox):
    """只允许枚举值；setCurrentText 遇表外值自动 addItem
    （保证恢复历史配置不丢值，同时阻止用户乱输）。"""

    def setCurrentText(self, text):
        text = str(text)
        index = self.findText(text)
        if index < 0:
            self.addItem(text)
            index = self.findText(text)
        self.setCurrentIndex(index)
