#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将导出的 fireball_diameter_fit JSON 同步到程序安装目录上一级的 training_data。"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def app_executable_directory() -> Path:
    """可执行程序所在目录（打包为 exe 目录；开发时为 ``desktop``）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # …/source/desktop/extract_tab/utils/this_file.py → desktop
    return Path(__file__).resolve().parent.parent.parent


def resolve_training_data_mirror_dir() -> Path:
    """``(程序目录)/../training_data``。"""
    return app_executable_directory().parent / "training_data"


def mirror_json_to_training_data(exported_json: Path | str) -> Path | None:
    """
    复制 JSON 到 training_data；同名则覆盖。目录不存在则创建。

    Returns:
        目标路径；失败时返回 None。
    """
    src = Path(exported_json).expanduser().resolve()
    if not src.is_file():
        return None
    dest_dir = resolve_training_data_mirror_dir()
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        return dest
    except OSError:
        return None
