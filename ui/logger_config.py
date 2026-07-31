# -*- coding: utf-8 -*-
"""日志配置（复刻 style_spec §5.4）。

唯一工厂 ``setup_logger``：
- ``colorlog`` 彩色控制台 handler（级别 INFO）；
- ``RotatingFileHandler``（级别 DEBUG，10MB×5）；
- ``log_file`` 以 ``logs/`` 开头时重定向到 ``~/MyGPR/logs/<basename>``（避免权限问题）；
- ``logger.handlers`` 非空直接返回（防重复挂 handler）。
"""
import logging
import os
from logging.handlers import RotatingFileHandler

import colorlog

from ui.constants import DEFAULT_LOG_BACKUP_COUNT, DEFAULT_LOG_MAX_BYTES, LOG_DIR

_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}


def _redirect_log_file(log_file: str) -> str:
    """``logs/xxx.log`` → ``~/MyGPR/logs/xxx.log``（目录自动创建）。"""
    normalized = log_file.replace('\\', '/')
    if normalized.startswith('logs/'):
        log_file = os.path.join(LOG_DIR, os.path.basename(normalized))
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    return log_file


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO,
                 max_bytes: int = DEFAULT_LOG_MAX_BYTES,
                 backup_count: int = DEFAULT_LOG_BACKUP_COUNT) -> logging.Logger:
    """创建（或获取）logger：彩色控制台 + 轮转文件双 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:          # 防重复挂 handler
        return logger

    # 1) colorlog 彩色控制台（INFO）
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(colorlog.ColoredFormatter(
        f'%(log_color)s{_LOG_FORMAT}', log_colors=_LOG_COLORS))
    logger.addHandler(console_handler)

    # 2) RotatingFileHandler（DEBUG，10MB×5）
    if log_file:
        file_handler = RotatingFileHandler(
            _redirect_log_file(log_file), maxBytes=max_bytes,
            backupCount=backup_count, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger
