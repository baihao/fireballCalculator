#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志配置：将控制台输出重定向到文件，超过 20MB 自动滚动。
"""

import sys
import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging():
    """
    初始化日志配置，stdout/stderr 重定向到滚动日志。
    """
    # exe 环境与源码环境统一处理
    base_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    log_dir = base_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    class _StreamToLogger:
        def __init__(self, level):
            self.level = level

        def write(self, message: str):
            msg = message.rstrip()
            if msg:
                logger.log(self.level, msg)

        def flush(self):
            pass

    sys.stdout = _StreamToLogger(logging.INFO)
    sys.stderr = _StreamToLogger(logging.ERROR)

    logger.info("日志初始化完成，输出目录: %s", os.fspath(log_file.parent))


__all__ = ["setup_logging"]

