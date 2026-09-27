#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
桌面 UI 简体宋体系字体：跨平台回退名列表。

Windows: SimSun / NSimSun；macOS: Songti SC / STSong；
Linux 常见: Noto Serif CJK SC、Source Han Serif SC 等。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# 按优先级尝试；Qt 与 Matplotlib 共用（Windows 上 Matplotlib 常需显式 addfont）
SONG_FAMILY_FALLBACK: List[str] = [
    "SimSun",
    "NSimSun",
    "宋体",
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "Songti SC",
    "STSong",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Arial Unicode MS",
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
    if _matplotlib_font_family:
        return _matplotlib_font_family
    if sys.platform == "win32":
        return "Microsoft YaHei"
    return "serif"


def _register_windows_font_files() -> None:
    """将系统 Fonts 目录中的中文字体注册进 Matplotlib（Windows 上否则易回退 DejaVu）。"""
    if sys.platform != "win32":
        return
    from matplotlib import font_manager

    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if not fonts_dir.is_dir():
        return
    for fname in (
        "simsun.ttc",
        "simsunb.ttf",
        "msyh.ttc",
        "msyh.ttf",
        "msyhl.ttc",
        "simhei.ttf",
    ):
        path = fonts_dir / fname
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
        except Exception:
            continue


def _matplotlib_can_render_cjk(family: str) -> bool:
    from matplotlib import font_manager

    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except Exception:
        return False
    if not path or "dejavu" in path.lower():
        return False
    return True


def configure_matplotlib_cjk() -> Optional[str]:
    """
    在 QApplication 创建后调用：让 Matplotlib 与 Qt 使用同一套可显示中文的字体族名。

    若仅设置 rcParams['font.serif'] 列表但本机字体未被 Matplotlib 缓存识别，
    中文轴标题/图例会回退到 DejaVu Serif，在 Windows 上常表现为方框或「丢失」。
    """
    global _matplotlib_font_family
    import matplotlib
    from matplotlib import font_manager

    _register_windows_font_files()

    candidates: List[str] = []
    qt_pick = pick_system_song_font_family()
    if qt_pick:
        candidates.append(qt_pick)
    for name in SONG_FAMILY_FALLBACK:
        if name not in candidates:
            candidates.append(name)

    chosen: Optional[str] = None
    for name in candidates:
        if _matplotlib_can_render_cjk(name):
            chosen = name
            break

    if not chosen:
        return None

    _matplotlib_font_family = chosen
    matplotlib.rcParams["font.family"] = chosen
    rest = [n for n in SONG_FAMILY_FALLBACK if n != chosen]
    matplotlib.rcParams["font.serif"] = [chosen, *rest, "DejaVu Serif"]
    matplotlib.rcParams["font.sans-serif"] = [chosen, *rest, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen
