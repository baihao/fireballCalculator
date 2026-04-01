#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桌面 UI 简体宋体系字体：跨平台回退名列表。

Windows: SimSun / NSimSun；macOS: Songti SC / STSong；
Linux 常见: Noto Serif CJK SC、Source Han Serif SC 等。
"""

from __future__ import annotations

from typing import List

# 按优先级尝试；均为宋体或近宋衬线，用于中文界面
SONG_FAMILY_FALLBACK: List[str] = [
    "SimSun",
    "NSimSun",
    "宋体",
    "Songti SC",
    "STSong",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "AR PL UMing CN",
]


def song_family_qss() -> str:
    """供 Qt Style Sheet 使用的 font-family 回退串。"""
    parts = [f'"{name}"' for name in SONG_FAMILY_FALLBACK] + ["serif"]
    return ", ".join(parts)


def pick_system_song_font_family() -> str | None:
    """返回本机已安装的第一个候选字体族名，若无则 None。"""
    from PySide6.QtGui import QFontDatabase

    db = QFontDatabase()
    available = set(db.families())
    for name in SONG_FAMILY_FALLBACK:
        if name in available:
            return name
    return None


def apply_app_song_font(app) -> None:
    """将 QApplication 默认字体设为简体宋体系（若系统存在）。"""
    from PySide6.QtGui import QFont

    name = pick_system_song_font_family()
    if not name:
        return
    f = QFont(name)
    f.setPointSize(10)
    app.setFont(f)
