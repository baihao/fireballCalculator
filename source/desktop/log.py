#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志配置：将控制台输出重定向到文件，超过 20MB 自动滚动。
"""

import sys
import os
import io
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

    original_stdout = sys.stdout if sys.stdout is not None else sys.__stdout__
    original_stderr = sys.stderr if sys.stderr is not None else sys.__stderr__
    if original_stdout is None:
        original_stdout = io.StringIO()
    if original_stderr is None:
        original_stderr = io.StringIO()

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

    safe_console_stream = original_stdout
    if safe_console_stream and hasattr(safe_console_stream, "buffer"):
        try:
            safe_console_stream = io.TextIOWrapper(
                safe_console_stream.buffer,
                encoding="utf-8",
                errors="replace",
                line_buffering=True,
            )
        except Exception:
            pass
    if safe_console_stream is None:
        safe_console_stream = io.StringIO()

    console_handler = logging.StreamHandler(safe_console_stream)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    class _StreamToLogger:
        def __init__(self, level, stream):
            self.level = level
            self.stream = stream
            self._in_write = False

        def write(self, message: str):
            if self._in_write:
                return
            self._in_write = True
            try:
                msg = message.rstrip("\n")
                if msg:
                    record = logger.makeRecord(
                        name="stdout" if self.level == logging.INFO else "stderr",
                        level=self.level,
                        fn="",
                        lno=0,
                        msg=msg,
                        args=(),
                        exc_info=None,
                    )
                    logger.handle(record)
                if self.stream:
                    try:
                        self.stream.write(message)
                    except Exception:
                        pass
            finally:
                self._in_write = False

        def flush(self):
            if self.stream:
                try:
                    self.stream.flush()
                except Exception:
                    pass

    sys.stdout = _StreamToLogger(logging.INFO, original_stdout)
    sys.stderr = _StreamToLogger(logging.ERROR, original_stderr)

    logger.info("日志初始化完成，输出目录: %s", os.fspath(log_file.parent))


__all__ = ["setup_logging"]

