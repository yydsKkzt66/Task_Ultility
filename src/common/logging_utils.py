"""轻量日志工具。"""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """配置统一日志格式。

    输出尽量紧凑，方便用户在 terminal 中快速定位时隙、路由器和异常信息。
    """

    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""

    return logging.getLogger(name)
