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


# Matplotlib 实际使用的族名（configure_matplotlib_cjk 成功后写入）
_matplotlib_font_family: str | None = None


def matplotlib_font_family() -> str:
    """图表轴标签/图例应使用的 fontfamily；未配置时回退 generic serif。"""
    return _matplotlib_font_family or "serif"


def configure_matplotlib_cjk() -> str | None:
    """
    在 QApplication 创建后调用：让 Matplotlib 与 Qt 使用同一套可显示中文的宋体族名。

    若仅设置 rcParams['font.serif'] 列表但本机字体未被 Matplotlib 缓存识别，
    中文轴标题/图例会回退到 DejaVu Serif，在 Windows 上常表现为空白或「丢失」。
    """
    global _matplotlib_font_family
    import matplotlib
    from matplotlib import font_manager

    chosen: str | None = pick_system_song_font_family()
    if chosen:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=chosen),
                fallback_to_default=False,
            )
        except Exception:
            chosen = None

    if not chosen:
        for name in SONG_FAMILY_FALLBACK:
            try:
                path = font_manager.findfont(
                    font_manager.FontProperties(family=name),
                    fallback_to_default=False,
                )
                if path and "dejavu" not in path.lower():
                    chosen = name
                    break
            except Exception:
                continue

    if not chosen:
        return None

    _matplotlib_font_family = chosen
    # 明确指定族名，避免仅写 'serif' 时回退链未命中 SimSun/宋体
    matplotlib.rcParams["font.family"] = chosen
    rest = [n for n in SONG_FAMILY_FALLBACK if n != chosen]
    matplotlib.rcParams["font.serif"] = [chosen, *rest, "DejaVu Serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen
