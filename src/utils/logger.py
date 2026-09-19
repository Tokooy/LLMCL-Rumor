# coding=utf-8
"""日志工具：统一的控制台 + 文件双写 logger。

约定：

* 控制台输出带颜色等级前缀，便于在长训练日志中定位；
* 同时写入 ``outputs/logs/<name>.log``，便于回溯；
* 重复调用 :func:`get_logger` 不会重复添加 handler（避免日志被打印多份）。
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

__all__ = ["get_logger", "setup_logging"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 已经配置过的 logger 名字，避免重复挂 handler
_CONFIGURED: set = set()

# 供 root logger 使用的一次性文件 handler 缓存
_ROOT_FILE_HANDLER: Optional[logging.FileHandler] = None


class _ColorFormatter(logging.Formatter):
    """仅在支持 ANSI 的终端上着色；重定向到文件时自动降级为纯文本。"""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str, use_color: bool):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return text
        color = self.COLORS.get(record.levelname, "")
        return f"{color}{text}{self.RESET}" if color else text


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    name: Optional[str] = None,
) -> logging.Logger:
    """配置并返回一个 logger。

    Args:
        log_file: 若非空，则额外把日志写入该文件（自动创建父目录）。
        level: 日志级别。
        name: logger 名称；``None`` 表示配置 root logger。

    Returns:
        配置完成的 logger。
    """
    global _ROOT_FILE_HANDLER

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # 让子 logger 的日志冒泡到 root，而不是各自重复输出
    logger.propagate = name is None

    key = name or "__root__"
    if key in _CONFIGURED:
        # 已配置过：如果这次带了（新的）文件路径，再补一个文件 handler
        if log_file:
            _attach_file_handler(logger, log_file)
        return logger
    _CONFIGURED.add(key)

    stream = sys.stdout
    console = logging.StreamHandler(stream)
    console.setFormatter(
        _ColorFormatter(_LOG_FORMAT, _DATE_FORMAT, _supports_color(stream))
    )
    logger.addHandler(console)

    # 文件 handler 在**首次**配置时就要挂上。
    # 曾经的写法是 `if log_file and name is None`，导致具名 logger（也就是所有脚本
    # 实际使用的那些）第一次调用不会创建日志文件，而每个脚本只调用一次
    # get_logger → outputs/logs/*.log 永远不会出现。
    if log_file:
        _attach_file_handler(logger, log_file)

    return logger


def _attach_file_handler(
    logger: logging.Logger, log_file: str, only_if_missing: bool = False
) -> Optional[logging.FileHandler]:
    """给 logger 挂一个文件 handler；目录不存在时自动创建。"""
    global _ROOT_FILE_HANDLER

    target = os.path.abspath(log_file)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if only_if_missing:
                return handler
            if os.path.abspath(handler.baseFilename) == target:
                return handler

    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    logger.addHandler(file_handler)

    if logger.name in ("", "root"):
        _ROOT_FILE_HANDLER = file_handler
    return file_handler


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO):
    """获取（并按需初始化）一个具名 logger。

    相比 :func:`setup_logging`，此函数保证 root logger 已配置，
    这样即便某个模块单独被导入也不会丢失日志输出。
    """
    setup_logging()
    return setup_logging(log_file=log_file, level=level, name=name)
